# prefill：按 token 批处理的 W4A16 GEMV（`w4a16_gemv_tok`）设计说明

> 状态：**离线设计 + 源码已写，未编译、未上板**（2026-11-24，子代理 agent-2）
> kernel：`ascendc_ops/kernels/w4a16_gemv_tok.cpp`
> host 验证片段：`ascendc_ops/host/test_w4a16_gemv_tok.inc.cpp`（需粘进 `host/main.cpp`，**没有直接改 main.cpp**）
> 目标：把「prefill 逐 token 走 decode 路径（每 token 读一遍 13.5GB 权重 → 14.62 s/token）」
> 改成「一次处理 M 个 token，权重每 M 个 token 只读一遍」。

---

## 0. 一句话结论

- 预计把 prefill 从 **14.62 s/token → 0.85~0.95 s/token（≈15~17×）**，即 ~1.1 token/s。
- 但**向量路径有硬天花板**：每 token 至少要对每个权重元素做 ~2.9 次向量元素操作
  （1 次 Mul + ~1.9 次归约），`27G × 2.9 ÷ 134 Gelem/s ≈ 0.58 s/token（1.7 token/s）`，M 再放大也下不去。
- 对照：已实测的两段式 Cube 路径是 **187 ms/token（5.3 token/s）**（`docs/310B-性能上限与prefill优化.md` §6.1 基线），
  仍比本 kernel 快 ~5×。要上 10 token/s 以上必须走 §6.1 的 P0（融合反量化）/ P0.5（手写 matmul）。
- 所以本 kernel 的正确定位：**① 立刻止血的 prefill 路径（不改权重格式、不新增 fp16 权重副本）；② decode 侧 MTP
  （M=2~4 时每 token 1.5~2×）；③ 与 Cube 路径正交，可先用它把端到端链路跑通再替换 GEMM。**
- ⚠️ 集成层约束（最容易被忽略）：engine 现在 `runOne` 是逐 token 串整条链，GDN 递推/conv/KV cache 都是逐 token 演进，
  **只换 GEMV kernel 拿不到端到端收益**；需要把 `runOne` 改成 `runChunk(M)`（按层先批量算 M 个 token 的投影，再逐 token 跑递推）。见 §7 #9。

---

## 1. 数学与权重布局（完全不变）

```
out[m][n] = Σ_g s[n,g]·(Σ_{k∈g} q[n,k]·act[m][k]) − Σ_g sz[n,g]·suma[m,g]
suma[m,g] = Σ_{k∈g} act[m][k]
```

| 张量 | 布局 | 说明 |
|---|---|---|
| `wq` | uint16 `[N][KQ]`，`KQ=K/4` | **相位交织**：word `j` 的 4 半字节 = `q[j], q[j+KQ], q[j+2KQ], q[j+3KQ]` |
| `scales`/`szs` | fp16 `[N][SBP]`，`SBP=ceil(K/32/16)*16` | group-32；`sz=s·z`；相位 p 的 group g 在下标 `p·GPP+g`（`GPP=KQ/32`） |
| `act` | fp16 `[Mtok][K]` | 新增 token 批量维 |
| `out` | fp16 `[Mtok][N]` | 同 |

**权重格式一行不改** → 与 `w4a16_gemv_gen` / `w4a16_gemv_batch` 共用同一份 13.53GB 权重，可逐位对拍。

---

## 2. 核心设计（为什么能快）

### 2.1 权重读一次、反量化一次，M 个 token 共用

逐 token 路径的成本结构（旧 `w4a16_gemv_gen`，实测 55.6 ms / 44.6MB / 17408 行）是「每输出行 ~45 条向量指令 +
每行一次 V_S」，与带宽无关（0.80 GB/s）。本 kernel 把 per-row 固定成本压到 **一次反量化（≈4.8·K 元素操作）**，
再让 M 个 token 各自只做「乘加 + 归约」（≈2.9·K / token）。

### 2.2 scale 折进权重（关键简化）

反量化时直接算 `Wd = q·s`（`s` 用 Gather 从本行本相位的 fp16 scale 表展开），于是：

1. 不再需要 `16^-p`（掩码出来的 nibble 已是整数 `q`，16 的位权只出现在反量化阶段）；
2. **4 个相位可以先累加到同一个 `prod[m]`，整行只做一次归约**（旧做法是每相位归约一次 → 4×）；
3. 每个 token 每行的向量动作：`4×(Mul+Add) + 1×Add(偏移16) + ~6×块折叠 + 1×Gather + 4×Add` ≈ 11·KQ 元素操作。

### 2.3 归约**只用 ≥32B 对齐的偏移**（本仓库板端已验证 kernel 全部如此）

AscendC 向量指令的起始地址要求 32 字节对齐；本仓库所有跑通的 kernel（`w4a16_gemv_gen`、`gdn_*`、`dense_ops`）
用的偏移全部是 32B 的整数倍。所以本 kernel 的归约刻意避开 8/4/2 元素这类 <32B 偏移：

```
① 32 元素块 → 16 lane：Add(偏移 16 元素 = 32B)                      [1 次]
② GPP 个块 → 块 0（块内 16 lane）：按 2 的幂做块折叠
       Add(偏移 d·32 元素 = d·64B，d = 2^k)                          [≈6 次，共 ~1.25·KQ 元素]
       例 GPP=40：d=32,16,8,4,2,1
③ 16 lane → 1：Gather 重排成 [t(16)][Mp]（Mp = roundup(Mstep,16)），
       再 4 次 Add（偏移 8Mp/4Mp/2Mp/Mp 元素 = 256/128/64/32 B）      [1 Gather + 4 Add]
```

`Ga = roundup(GPP,16)`、`Mp = roundup(Mstep,16)` 都是为了「步长是 16 元素的倍数 → 偏移 32B 对齐」。
代价与原计划的「16/8 两步折叠」几乎一样（≈2.3·KQ vs 2·KQ），但**不会因为非对齐偏移在板端翻车**。

> 顺带：`w4a16_gemv_batch.cpp:199/201` 的 `Add(qf, qf, qf[16]/qf[8], ...)` 中 `qf[8]` 是 **16B 偏移**，
> 按这条规则可疑，建议 lead 顺手核对（可能正是 batch 板端对拍不过的原因之一）。

### 2.4 偏置项

`Σ_g sz[n,g]·suma[m,g]`：`suma` 与行无关，入口按 token 算一次（同样用「Add(偏移16) + 16 lane Gather + 4 Add」）；
每 R 行做 `M·SBP` 次 fp32 Mul + 折叠（约为主循环的 1%）。偏置与点积结果都在 fp32 做减法。

### 2.5 输出 staging

每 token 攒 512 行再一次性 `DataCopy`（≥1KB），避免历史踩过的 <32B 小拷贝（`docs/AscendC算子开发进度.md` §10.5）。

### 2.6 同步与事件

| 事件 | 用途 |
|---|---|
| `MTE2_V` EVENT_ID0/1 | 权重 tile（W + sc + sz）双缓冲预取，parity 0/1；一个 flag 覆盖三条 DataCopy |
| `V_MTE2` EVENT_ID2/3 | 本 tile 向量读结束后释放缓冲；每个 chunk 内 set/wait 严格配平（chunk 间 `PipeBarrier` + 1 次复位） |
| `V_S` / `S_V` EVENT_ID0 | 每 R 行一次的标量收尾（读 `resBuf` 写 `oStage`） |
| `PIPE_ALL` | 输出写回前后各一次（每 512 行一次） |

**同流水（V→V）之间不加 `PipeBarrier<PIPE_V>`**：AscendC 同一流水内指令按序执行、硬件处理 RAW。
旧 kernel/batch 版每条向量指令后都跟一个 `PipeBarrier<PIPE_V>`，等于反复冲刷向量流水。
若 bring-up 怀疑数据相关，把 kernel 里 `TK_VBAR_ENABLE` 置 1 即退回保守写法。

---

## 3. 二维批处理方案与 UB 预算

```
外层 token chunk（步长 Mstep）：Mtok > Mstep 时权重要重读 M/Mstep 遍
  中层 行 tile（R 行）：一次搬 R 行权重，双缓冲预取下一 tile
    内层 行 r → 相位 p(0..3) → token m(0..Mstep)
```

- **R 不影响元素操作数**（只影响指令效率、DataCopy 大小、同步次数）→ 取 UB 放得下的最大值（≤8）。
- **M 决定元素操作数**（每权重元素 `4.8/M + 2.9`）→ 先尽量把 `Mtok` 装进 UB。
- 选参：`Mstep` 从 `Mtok` 递减、`R` 从 8 递减，取第一组满足预算者。

### UB 占用表（K=5120, R=8, Mstep=8 → Mp=16, Ga=48, SBP=160, GPP=40, KQ=1280，预算 240KB）

| buffer | 类型/形状 | 字节 | 说明 |
|---|---|---|---|
| W0/W1 | uint16 R×KQ | 2×20480 | 权重 tile 双缓冲 |
| sc0/sc1 | fp16 R×SBP | 2×2560 | group scale tile 双缓冲 |
| sz0/sz1 | fp16 R×SBP | 2×2560 | sz tile 双缓冲 |
| act | fp16 M×K | 81920 | M 个 token 的激活常驻（否则每 tile 重读 act 比权重还贵） |
| suma | fp32 M×SBP | 5120 | Σ_{k∈g} act[m][k] |
| prod | fp16 M×KQ | 20480 | 每 token 的 4 相位累加器（之后原地块折叠） |
| tmp/qd/qi/scaleSeg/maskT | KQ ×5 | 12800 | 中间量 |
| idxPh | u32 KQ | 5120 | 逐元素 scale 展开的字节偏移 `(j/32)*2` |
| scHalf + idxCopy | fp16 GPP / u32 4×GPP | 80 / 640 | 相位 scale 段拷到偏移 0 |
| gs + idxG | fp16 16×Ga / u32 16×Ga | 1536 / 3072 | suma 的 16 lane 重排 |
| valsH + valsF | fp16/ fp32 16×Mp | 512 / 1024 | 点积收 t 维 |
| resBuf + bBias | fp32 R×Mp | 512 / 512 | 每 (行,token) 点积 / 偏置 |
| biasVal + idxB | fp32/u32 8×Mp | 512 / 512 | 偏置收 t 维 |
| idxT | u32 16×Mp | 1024 | 点积 16 lane 重排 |
| biasFold | fp32 M×SBP | 5120 | sz·suma 折叠缓冲 |
| szF | fp32 SBP | 640 | 当前行的 sz |
| oStage | fp16 M×512 | 8192 | 输出 staging（512 行一次写回） |
| **合计** | | **200,528 B ≈ 196 KB** | ≤ 240KB 预算 |

### K=17408 的退化

`act` 单 token 就要 34.8KB、`W` 双缓冲 `R×17408` 字节 → 自动选参落到 `R≈2~4, Mstep≈2~3`。
每 token 成本 ≈ `(4.8/3 + 2.9)·K ≈ 4.4·K`（vs 逐 token 的 7.7·K，仍 ~1.75×）。
要更好：① host 把 K 切段多次调用（每段只驻留 K/nK 的 act，重读按段数放大）；② 或走 Cube 融合路径。

---

## 4. 数值等价性与误差量级

- 数学逐项恒等（scale 折进权重 = 乘法结合律；`16^-p` 只是 nibble 位权）。
- 差异只在累加顺序/精度：旧 gen 全程 fp32；本 kernel **fp16 只做「4 相位累加 + 32 元素块内累加」**，
  组和→跨组→偏置→输出全 fp32。
- 误差估算：每个输出 = 160 个块和（块和量级 ~1~8）之和，输出量级 O(1~30)；
  fp16 块内误差 ≈ `eps(4.9e-4) × 块和量级 × ~4 层` ≈ 1e-3~3e-3/块，160 块随机符号相加 ≈ `√160 × 2e-3 ≈ 2.5e-2`。
  → **预期 `max_abs`（vs 旧 kernel）≈ 1e-2 ~ 5e-2，`mean_abs` ~1e-3**（与 batch 版同量级）。
- 风险：`lm_head` 的 argmax 可能偶发翻转 → 集成后要统计 token 一致率，不能只看 max_abs。

---

## 5. 验证方案（离线给片段，不直接改 main.cpp）

见 `ascendc_ops/host/test_w4a16_gemv_tok.inc.cpp`，三段粘贴：

1. `golden`：`w4a16_gemv_gen` 对 **M 个不同 act 行** 各跑一次（= 现在的 prefill 逐 token 路径）；
2. 新 kernel `w4a16_gemv_tok` 一次跑 M 个 token；
3. **bring-up 第一道闸**：新 kernel 用 `Mtok=1` 与 gen 对拍，数学上应逐位等价（`max_abs < 1e-3`）；
4. 逐 token 与总计 `max_abs / mean_abs / n(|d|>2e-2)`；
5. 计时：`old = M × gen 单次`（当前 prefill 真实成本）vs `new = tok(M)`，给 tok/s 与加速比；
6. 输出 dump 到 `output/gout_tok_{new,ref}.bin` 供 python 复核。

运行（新 kernel 后必须重跑 cmake，`CMakeLists` 用 `file(GLOB kernels/*.cpp)`）：
```bash
cd ~/code/qwen38_310B/ascendc_ops/build && cmake .. >/dev/null && make -j4
cd .. && cmake --install build >/dev/null && cp -f out/bin/qwen38_ops_test ./
export LD_LIBRARY_PATH=$PWD/out/lib:$PWD/out/lib64:/usr/local/Ascend/ascend-toolkit/latest/lib64:/usr/local/Ascend/ascend-toolkit/latest/runtime/lib64:$LD_LIBRARY_PATH
timeout -s KILL 300 ./qwen38_ops_test w4a16_gemv_tok        # GEMVT_K/N/M 可覆盖
```

---

## 6. 预期收益（Qwen3.8-27B，13.5GB 权重，10.2GB/s 单核带宽墙）

每权重元素向量元素操作数 ≈ `4.8/M + 2.9`（理论），按实测向量峰值 134 Gelem/s：

| M（每 pass token 数） | 元素操作/权重 | 理论 s/token | ×0.8 效率后 | 相对 14.62 s/token |
|---|---|---|---|---|
| 1 | 7.7 | 1.55 | 1.9 | 7.5× |
| 2 | 5.3 | 1.07 | 1.34 | 11× |
| 4 | 4.1 | 0.83 | 1.03 | 14× |
| 8 | 3.5 | 0.71 | **0.88** | **17×** |
| 16 | 3.2 | 0.65 | 0.81 | 18× |
| ∞（下限） | 2.9 | 0.58 | 0.73 | 20× |

- 每 pass 带宽：13.5GB / 5.6s ≈ **2.4 GB/s（10.2 的 24%）** → 纯计算受限（说明方向对，瓶颈从带宽换成了向量吞吐）。
- 512-token prompt：64 pass × ≈7s ≈ **7.5 分钟（0.9 s/token，1.1 token/s）**；当前是 512 × 14.62 s ≈ 2.1 小时。
- 对照 Cube：两段式已实测 187 ms/token（5.3 token/s）→ 仍是本 kernel 的 ~5×；融合反量化后能到 25~190 token/s。

---

## 7. 风险清单

| # | 风险 | 影响 | 缓解 |
|---|---|---|---|
| 1 | 同流水 V→V 不加 `PipeBarrier` 的假设 | 若版本要求显式同步 → 结果错 | `TK_VBAR_ENABLE=1` 一键回退；先跑 `M=1` 对拍闸 |
| 2 | `Gather` 偏移是**字节**偏移（`probe_gather` 实测） | 若某版本改成元素偏移 → scale 全错 | 三处表（idxPh×2 / idxG、idxT、idxB×2 / idxCopy×2）集中定义 |
| 3 | 我对 32B 对齐规则的判断（所有偏移已按 ≥16 元素设计） | 若其实允许 16B，本 kernel 只是略慢；若不允许而我用了 16B 会翻车 | **已全部 ≥32B**，无 8/4/2 偏移 |
| 4 | 大 count `Gather` 兼容性 | 可能 aicore exception | scale 展开已按 1024 分块；其余 count ≤ 16·Ga（K=17408 时 2304） |
| 5 | fp16 组内累加 | max_abs 1e-2~5e-2，argmax 偶发翻转 | 与 golden 对拍 + 统计 token 一致率 |
| 6 | UB 预算运行时随形状变化 | K=17408 时 Mstep→2~3；极端形状可能 InitBuffer 超限 | host 侧断言 ubBytes ≤ 240KB；对 K>5120 调小 Mtok |
| 7 | `Mtok > Mstep` 时权重重读 | 吞吐按 Mstep/Mtok 打折 | host 控制 Mtok ≤ Mstep（K=5120 时 8~10） |
| 8 | 输出 staging 依赖 `512 % R == 0` | R ∈ {1,2,4,8} 均满足 | 放开 R=16 也没问题 |
| 9 | **集成层（最大风险）**：`runOne` 逐 token 串整条链，GDN 递推/conv/KV cache 都是逐 token | 只换 GEMV 拿不到端到端收益 | 把 `runOne` 改 `runChunk(M)`：按层先批量算 M 个 token 的投影（1 次 GEMV / 矩阵），再逐 token 跑 GDN 递推/attention（M=8 时激活仅 ~0.2MB） |
| 10 | 进程内首次调用异常（历史已知） | 首次结果错 | 测试/引擎里 warmup 一次丢弃 |

---

## 8. 顺带发现（建议 lead 复核，均为**未经验证的静态观察**）

1. `w4a16_gemv_batch.cpp:147/154` 的 `Gather` 索引用的是**元素下标**，但 `probe_gather` 实测 offset 是**字节偏移**
   （fp32 表 `(k%16)*4` 才对；`gen` 里 `idx=(j/32)*4` 也是按字节写的，且 gen 与 PC 参考实测 `max_abs=0.0020` 通过，见 §10.39）。
   源 `qf`/`scC` 都是 **fp16**，字节偏移应是元素下标 **×2**。
   → 若 batch 板端对拍通过，说明 CANN 版本语义与 probe 结论不同，请以板端为准；我的 kernel 统一按字节偏移写。
2. `w4a16_gemv_batch.cpp:199/201` 的 `Add(qf, qf, qf[8], nQ-16)` 是 **16B 偏移**（fp16 8 元素），
   与本仓库所有已验证 kernel「只用 32B 倍数偏移」的惯例不一致，建议核对是否触发非对齐行为。
