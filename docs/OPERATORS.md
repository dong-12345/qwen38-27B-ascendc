# 自研 AscendC 算子详解

本文件逐个说明我们为 **Qwen3.8-27B（Qwen3.5 混合 GDN 架构）** 手写的 AscendC 算子：
算法公式、数据布局、实现分段、310B 上的关键技巧与踩过的坑、以及验证结果。

> 运行环境：OrangePi AI Pro 20T（**Ascend 310B1**，单 AI core / 单向量核），CANN 8.0.0。
> UB 仅 **248 KB**、L1 1 MB、L2 4 MB，所有 kernel 都按"UB 放得下"来切分。
> 代码：`ascendc_ops/kernels/*.cpp`；host 包装与调度：`ascendc_ops/host/`。

---

## 0. 算子总览

### 0.1 解码主路径（引擎每个 token 都会调用）

| 算子 | 文件 | 作用 | 典型形状 | 实测 |
|---|---|---|---|---|
| `w4a16_gemv_gen` | `w4a16_gemv_gen.cpp` | W4A16 非对称 GEMV（group-32） | K≤17408, N≤248320 | **1.43 GB/s**（K=17408,N=5120, 31.2 ms） |
| `fp16_gemv` | `dense_ops.cpp` | fp16 GEMV（M=1） | K=5120, N≤10240 | — |
| `rmsnorm_op` | `dense_ops.cpp` | RMSNorm（任意 H） | H=5120 / 256 / 128 | — |
| `vec_axpy` | `dense_ops.cpp` | `out = a + b*s`（残差） | N=5120 / 6144 | — |
| `mul_silu` | `dense_ops.cpp` | `out = silu(gate)*up` | N=17408 | — |
| `conv1d_decode` | `dense_ops.cpp` | depthwise causal conv(k=4)+SiLU+状态 | C=10240 | — |
| `gdn_l2norm` | `gdn_l2norm.cpp` | q/k 逐头 L2 归一化（含 1/√d） | 16×128 / 16×128 | — |
| `gdn_rgdr` | `gdn_rgdr.cpp` | GDN 单 token 递推解码 | 48 heads × 128 | 0.72 ms/层 |
| `rmsnorm_gated` | `gdn_gated_rmsnorm.cpp` | RMSNormGated（norm·w·silu(z)） | 48×128 | — |
| `full_attn_decode` | `full_attn_decode.cpp` | GQA 解码 + partial RoPE + 输出门 + KV cache | 24/4 heads, D=256 | 0.716 ms/层 |

### 0.2 GDN chunked prefill 六阶段（长序列并行化用）

| 算子 | 文件 | 对应 FLA 阶段 | 公式 |
|---|---|---|---|
| `gdn_gating` | `gdn_gating.cpp` | gating | `g = -exp(A_log)·softplus(a+dt_bias)`，`β = sigmoid(b)` |
| `gdn_conv1d_silu` | `gdn_conv1d.cpp` | causal_conv1d | `y = silu(Σ_j w[j]·x[t-3+j])` |
| `gdn_cumsum` | `gdn_cumsum.cpp` | chunk_cumsum | chunk 内 `gc = cumsum(g)` |
| `gdn_kkt` | `gdn_kkt.cpp` | scaled_dot_kkt | `A[i,j] = β_i·(K_i·K_j)·exp(gc_i-gc_j)`，j≤i |
| `gdn_tril` | `gdn_tril.cpp` | solve_tril | `T = (I+A)^{-1}`（严格下三角前代） |
| `gdn_wy` | `gdn_wy.cpp` | wy_fast | `w = T·(k·β·exp(gc))`，`u = T·(v·β)` |
| `gdn_chunk_h` | `gdn_chunk_h.cpp` | chunk_h | `v_new = u - w@S`；`S ← exp(gc_last)·S + (k·exp(gc_last-gc))ᵀ@v_new` |
| `gdn_chunk_o` | `gdn_chunk_o.cpp` | chunk_o | `o = (q·exp(gc))@S + attn@v_new` |

> 引擎当前用的是**递推解码路径**（`gdn_rgdr`），chunked 六阶段用于 prefill 并行化探索，
> 每个算子都已单独用 CPU 参考对拍通过。

---

## 1. 通用约定

- **ABI**：`extern "C" __global__ __aicore__ void fn(GM_ADDR ...)`，host 侧用
  `aclrtLaunchKernel`（`xyz::xx` 包装见 `host/main.cpp`）。
- **UB 预算**：每个 kernel 启动时 `TBuf` 大小写死，全部控制在 **≤ 224 KB**（留 24 KB 余量）。
- **32B 对齐**：所有 UB 张量偏移、`DataCopy` 长度必须是 32B 的倍数（fp16 即 16 元素）。
  非 32B 对齐的偏移会直接触发 **507015 aicore exception**。
- **同步**：大量使用 `PipeBarrier<PIPE_V>()` + `SetFlag/WaitFlag`（V_S / S_V / V_MTE3），
  避免标量-向量跨流水线竞争；GM 读写与向量计算之间用 `PipeBarrier<PIPE_ALL>()`。
- **标量获取**：GM 上的单值（β、g 等）不能用 `DataCopy`（长度不合法），改为 host 传入
  kernel 标量参数，或用 `SetValue` 从 UB 侧写。

---

## 2. `w4a16_gemv_gen` —— W4A16 非对称 GEMV（核心算子）

### 2.1 公式与格式

权重按 **group-32 非对称 4bit** 量化为 `w ≈ s·(q - z)`，`q∈[0,15]`，`s`、`z` 为 fp16。
为减少一次乘法，转换时预存 `sz = s·z`，于是

```
out[n] = Σ_g s[n,g] · ( Σ_{k∈g} q_k · act_k )  −  Σ_g sz[n,g] · suma[g]
```

其中 `suma[g] = Σ_{k∈g} act_k`（每个 group 的激活和，行间共享）。

**相位交织存储**（关键设计）：`uint16` word 的 4 个半字节分别存
`q[j], q[j+KQ], q[j+2KQ], q[j+3KQ]`（`KQ = K/4`）。这样一条 16bit 向量指令能同时处理
4 个相位，`Gather` 展开 scale 后 4 个相位各自归约、最后一次相加，**每行只需 1 次 V_S 同步 + 8+8 次标量读**。

| 文件 | 形状 | 内容 |
|---|---|---|
| `*.w4.bin` | `uint16 [N, K/4]` | 相位交织的 4bit 权重 |
| `*.s.bin` | `fp16 [N, SBP]` | group scale，`SBP = ceil(K/32/16)*16` |
| `*.sz.bin` | `fp16 [N, SBP]` | `s·z`，用于标量偏置 |

### 2.2 实现分段

```
按行循环 n（每行 K 个权重）：
  ├ 每 16 行批量：DataCopy 16 行的 scale/sz 到 UB（scC/szC）
  ├ DataCopy 一行权重（KQ 个 uint16）
  ├ for p in 0..3（相位）:
  │    And/ShiftRight 抽出该相位的 4bit → Cast 到 fp32
  │    Gather(scaleSeg, scTab[p*GPP], idx)  ← 展开该相位的 group scale
  │    Mul scaleSeg *= act[p*KQ ..]（原地，省一块 UB）
  │    Mul phF *= scaleSeg；累加进 accSeg
  ├ 4 相位合并后一次折半归约（KQ→…→8）+ 8 元素尾循环
  ├ bias = Σ sz·suma（NBP=256 折半归约）
  └ acc.SetValue(n&15, dot - bias)；每 16 行 Cast 成 fp16 后一次 DataCopy 写出
```

### 2.3 310B 上的关键技巧

1. **act 分块加载（4096 元素）**：K=17408 时 act 全量 fp32 = 68 KB 放不下，
   改为按需 chunk 加载，且去掉中间 `actSeg` 缓冲（`Mul` 原地）。
2. **单次归约**：4 个相位先各自在 KQ 内累加，再合并归一，把标量读次数从 O(K/8) 降到 O(1)。
3. **标量偏置**：`suma` 用 `Duplicate(0)` + 标量 `SetValue` 构造（GM 上的零碎值不能 `DataCopy`）。
4. **`Gather` 的 base 参数**：用 `scTab[p*GPP]` 张量下标而不是 base 字节偏移，消除语义歧义。

### 2.4 踩过的坑

- **507015 异常**：测试缓冲按 group-128（SBP=48）分配而 kernel 用 group-32（SBP=160）→
  GM 越界，只有 `aclGetRecentErrMsg()` 才报 `The DDR address of the MTE instruction is out of range`。
- **进程内首次调用结果异常**：同一进程第 1 次调用结果错、第 2 次起正确（与输出缓冲无关）。
  引擎用 `ENG27_DOUBLE_GEMV`（先跑一次丢弃）规避，根因未完全定位。
- **K=17408 时 UB 溢出**：act 全量 + 权重行 + scale + 归约缓冲 = 268 KB > 248 KB → 分块。

### 2.5 验证

| 用例 | 结果 |
|---|---|
| 结构测试（K=5120, N=10240 真实 qkv 权重） | `max_abs = 1.95e-3` ✅ |
| `down_proj`（K=17408, N=5120 真实权重） | `max_abs = 0.0000`，**1.43 GB/s** ✅ |
| `gate_proj`（N=17408, K=5120 真实权重） | `max_abs = 2e-4`，0.80 GB/s ✅ |

---

## 3. `full_attn_decode` —— 全注意力层单 token 解码

24 q-heads / 4 kv-heads / D=256，partial RoPE（前 64 维），**输出门 `sigmoid(gate)`**。

### 3.1 关键设计

1. **KV cache 布局** `k/v: fp16 [4, 512, 256]`（token-major），按 `KC=64` 个 key 一块拷进 UB。
2. **RoPE 用标量旋转**：每 head 只转 64 维 × 32 对，用 `SetValue/GetValue` 完成
   `(x_i, x_{i+32})` 的 rotate_half 配对（与官方 `rotate_half` 一致）。
3. **softmax**：`score` 全量 fp32 放 UB（512×4B=2 KB），减最大值、exp、归一化。
4. **输出门**：`out *= sigmoid(gate)`（注意：**不是** SwiGLU 的 `silu`）。
5. **KV cache 写入必须用 `DataCopy`**（`Cast` 到 UB half 缓冲再写），
   **用 GM 标量 `SetValue` 会静默写错** —— 这是我们耗时最久定位的一个 bug。

### 3.2 验证

- 单层对拍（源层 3，pos=0）：`attnout max_abs=0.0156 / corr=1.0000`，post-attn `max_abs=0.0078`
- 双 token（pos=1，含 KV cache + RoPE）：`max_abs=0.0049 / corr=1.0000`
- 单元测试：`0.716 ms/层`（24 q-heads，pos=5）

---

## 4. `gdn_rgdr` —— GDN 递推解码

Gated DeltaNet 的**单 token 递推形式**（对应 fla 的 `torch_recurrent_gated_delta_rule`）：

```
g   = -exp(A_log) · softplus(a + dt_bias)          # 对数衰减
decay = exp(g)
S ← S · decay                                       # 状态衰减
bv = (v − Sᵀk) · β                                  # β = sigmoid(b)
S ← S + k ⊗ bv                                      # rank-1 更新
o = Sᵀ q · (1/√d)                                   # q 已 L2 归一化
```

- 单 head K=V=128，`S` 常驻 UB（**128×128 fp32 = 64 KB**），48 个 v-head 串行
- q/k **先 L2 归一化**（`gdn_l2norm`，q 额外乘 `1/√128`），与官方 `use_qk_l2norm_in_kernel=True` 一致
- `β`、`g` 用 kernel 标量参数传入（host 侧算好），避免 GM 零碎 `DataCopy`
- 状态在 **fp32**（`mamba_ssm_dtype=float32`），GM 上按 `[48, 128, 128]` 每层常驻

→ 单元测试 `0.72 ms/层`；单层对拍 `max_abs=0.0002 / corr=1.0000`（pos=0 与 pos=1 都验证过）。

---

## 5. `rmsnorm_gated` —— GDN 的输出门控归一化

```
y = x / sqrt(mean_d(x²)+eps) · w[d] · silu(z)
```

- `w` 是 **`RMSNormGated.weight`（直接相乘，不是 1-centered！）** —— 与 attention/MLP 的
  `Qwen3_5RMSNorm`（1-centered，`(1+w)`）不同，这里极易搞混
- 48 heads × 128 维，每头共享同一份 gamma

---

## 6. 基础 dense 算子（`dense_ops.cpp`）

| 算子 | 说明 | 坑 |
|---|---|---|
| `rmsnorm_op` | RMSNorm，H 任意 | H=5120 折半到 20 元素时 80B 不对齐 → 尾循环按 8 元素步长 |
| `fp16_gemv` | M=1 fp16 GEMV | 小 N（如 48）时 16 行批量写回要处理尾部 |
| `vec_axpy` | `out = a + b·s` | 残差连接 |
| `mul_silu` | `out = silu(gate)·up` | FF=17408 需 238 KB → 按 4096 分块 |
| `conv1d_decode` | depthwise causal conv(k=4)+SiLU，含 `state[3*C]` 移位更新 | 状态更新用标量循环 + `DataCopy` 写回 |

---

## 7. GDN chunked 六阶段（prefill 并行化）

| 算子 | 实现要点 |
|---|---|
| `gdn_gating` | a/b 转置为 `[HEADS, SEQ]` 连续访存；全程 fp32，输出 fp32 的 g/β |
| `gdn_conv1d_silu` | token-major `[SEQ, D]`，权重按 tap 转置成 `[4, D]`；prefill 版无状态 |
| `gdn_cumsum` | `[HEADS, SEQ]` fp32，chunk_size=64 内累加 |
| `gdn_kkt` | K 转置成 `Kt[D,T]`，对每个 i 用"秩-1 累加"求 S 的第 i 行，再乘 `β_i·exp(gc_i-gc_j)`，j>i 置 0；chunk 内先减 gc 最大值防 exp 溢出 |
| `gdn_tril` | `T=(I+A)^{-1}`，严格下三角逐行前代 `T[i,:] = e_i − Σ_{k<i} A[i,k]T[k,:]`，只需 Axpy |
| `gdn_wy` | `w = T·(k·β·exp(gc))`、`u = T·(v·β)`，逐行秩-1 累加 |
| `gdn_chunk_h` | 先逐行求 `v_new = u − w@S` 写 GM，再逐 token 做 rank-1 状态更新（S 在 UB 64 KB） |
| `gdn_chunk_o` | `o = (q·exp(gc))@S + attn@v_new`，v_new 一次性转 fp32 进 UB，内层不再访 GM |

六阶段之间用 GM 传递中间量（`T`、`w/u`、`v_new`），每阶段单独对拍通过。

---

## 8. 310B 踩坑清单（省你几天）

1. **UB 越界 → 507015**，且默认只报 `aicore exception`，必须调用
   `aclGetRecentErrMsg()` 才能看到 `DDR address ... out of range` 之类的根因。
2. **32B 对齐**：UB 偏移 / `DataCopy` 长度都要 32B 对齐（fp16 = 16 元素），
   折半归约长度必须 `%8==0`，否则要补尾循环。
3. **`DataCopy` 长度不能是 1 个元素**（fp16 才 2B）；单个标量用 `SetValue/GetValue`。
4. **GM 标量 `SetValue` 不可靠**：写 KV cache 时静默写错 → 一律改成 `Cast` 到 UB + `DataCopy`。
5. **同一进程首次调用异常**：`w4a16_gemv_gen` 第 1 次调用结果错，第 2 次起正确
   （先用一次丢弃 / warmup 前向规避）。
6. **TQue 深度 2 + 未同步**：`DeQue` 后立即 `FreeTensor` 再 `AllocTensor` 会踩踩内存，
   逐次 `EnQue/DeQue` 最稳。
7. **`Gather` 的索引是"字节偏移"**：fp32 表要 ×4；base 参数改为张量下标更安全。
8. **`ReduceSum` 的 work 语义**：需要额外的 work buffer，且对长度敏感 → 我们统一用手写折半归约。

---

## 9. 验证方法

每个算子都在 host 侧与 CPU（numpy）参考实现逐元素对拍：

```
python ascendc_ops/scripts/gen_data.py     # 生成输入 + CPU 参考输出
./qwen38_ops_test <operator_name>           # 跑 kernel 并比较 max_abs / rel_err
```

对拍标准：`max_abs ≤ 2e-3`（fp16 累加误差量级），全部 14 个算子均通过。
更完整的全模型验证见 [`verify/README.md`](../tools/verify/README.md)（独立 numpy 64 层参考、
权重转换独立校验、1-centered 修复脚本等）。
