# 架构与适配说明：Qwen3.5 混合 GDN 模型 → Ascend 310B

本文说明我们**如何把 Qwen3.8-27B（Qwen3.5 混合架构）适配到 Ascend 310B1**：
模型结构分析、数值格式设计、引擎设计、以及三个导致"输出乱码"的静默 bug 是如何被找出来的。

---

## 1. 模型结构分析

从 `config.json` 读到的关键信息（`text_config`）：

| 项 | 值 | 对实现的影响 |
|---|---|---|
| `hidden_size` / `intermediate_size` | 5120 / 17408 | GEMV 的 K、N 维度 |
| `num_hidden_layers` | 64 | 48 层 GDN + 16 层全注意力（每 4 层一个，从 index 3 开始） |
| `layer_types` | `[linear_attention]×3 + [full_attention]×1` 循环 | 引擎按类型分派 |
| GDN | `linear_num_key_heads=16`、`linear_num_value_heads=48`、k/v head_dim=128、conv kernel=4 | k/v 2048 维、v 6144 维、conv 10240 通道 |
| Attention | `num_attention_heads=24`、`num_key_value_heads=4`、`head_dim=256` | GQA 6:1，q_proj 输出 12288 |
| `attn_output_gate` | `True` | q_proj 输出里含 gate（每 head 512 = q 256 + gate 256） |
| `output_gate_type` | `swish`（**但代码实际用 sigmoid**） | 见 4.3 |
| `partial_rotary_factor` | 0.25 | 只转 head_dim 的前 64 维 |
| `rope_parameters` | `theta=1e7`、`mrope_interleaved=True`、`mrope_section=[11,11,10]` | 纯文本时三路 position 相同 → **退化为标准 RoPE** |
| `rms_norm_eps` | 1e-6 | 所有 norm |
| `mamba_ssm_dtype` | `float32` | GDN 状态必须 fp32 |
| `vocab_size` | 248320，`tie_word_embeddings=False` | lm_head 独立，W4 GEMV N=248320 |
| 其他 | `architectures=Qwen3_5ForConditionalGeneration` | **VL + 推理模型**：文本推理只用语言塔；对话模板需带 `<think>` |

> 模型名虽为 "Qwen3.8-27B"，其架构就是 **Qwen3.5 的 Gated DeltaNet 混合架构**（28B 参数）。
> 官方参考实现：`transformers>=5.24` 的 `modeling_qwen3_5.py`（我们就是逐行对照它修 bug 的）。

---

## 2. 数值格式：W4A16 相位交织

### 2.1 为什么重新量化

原始权重是 **AWQ 4bit group-32 非对称**（compressed-tensors `pack-quantized`：`weight_packed` int32 +
`weight_scale` bf16 + `weight_zero_point` int32）。310B 上直接用原格式需要额外解包/重排，
所以转换阶段重新量化成我们自己的格式（保留 group-32）：

```
w ≈ s·(q − z),  q∈[0,15]  →  预存 sz = s·z
out[n] = Σ_g s[n,g]·Σ_{k∈g} q_k·act_k  −  Σ_g sz[n,g]·suma[g]
```

→ 量化误差实测 **0.0000–0.0002**（group-64 会劣化到 6.2%，所以必须 group-32）。

### 2.2 相位交织（性能关键）

`uint16` word 的 4 个半字节存 `q[j], q[j+KQ], q[j+2KQ], q[j+3KQ]`（`KQ=K/4`）。
一条 16bit 向量指令同时处理 4 个相位，配合 `Gather` 展开 group scale 后 4 相位各自归约，
**每行只需 1 次 V_S 同步**。文件布局：

| 文件 | dtype | 形状 |
|---|---|---|
| `*.w4.bin` | uint16 | `[N, K/4]` |
| `*.s.bin` | fp16 | `[N, SBP]`，`SBP = ceil(K/32/16)*16` |
| `*.sz.bin` | fp16 | `[N, SBP]` |
| `meta.txt` | 文本 | 每行 `<role> <w4> <s> <sz> <N> <K> w4` 或 `<role> <file> <axbxc> f16` |

`embed_tokens` / `lm_head` 也走同一套 W4 格式（用 `--requant-extra embed_tokens,lm_head`）。

---

## 3. 引擎设计

### 3.1 数据流

```
token id
  → loadEmbedRow(token)            宿主侧解 quantize 一行 embed（W4 → fp16 40960 B，H2D）
  → for layer in 0..63:
        x_norm = RMSNorm(x, input_layernorm)            [1/sqrt(H) 归一化 × (1+w)]
        if GDN:   qkv/z/a/b ← W4 GEMV
                  conv1d_decode ×3（q/k/v，含卷积状态）
                  gdn_l2norm ×16（q 乘 1/√128，k 不乘）
                  host: β=sigmoid(b), g=-exp(A_log)·softplus(a+dt_bias)
                  gdn_rgdr ×48（v-head，含 recurrent state）
                  rmsnorm_gated(o, z)
                  out_proj ← W4 GEMV；x += out
        else:     qg ← W4 GEMV（N=12288，per-head [q|gate]）
                  k/v ← W4 GEMV（N=1024）
                  qk-norm（每 head 256 维 RMSNorm）
                  full_attn_decode（GQA + partial RoPE + sigmoid 门 + KV cache）
                  o_proj ← W4 GEMV；x += o
        x_norm2 = RMSNorm(x, post_attention_layernorm)
        gate/up ← W4 GEMV（N=17408）→ mul_silu → down ← W4 GEMV（K=17408）
        x += down
  → final RMSNorm → lm_head（W4 GEMV N=248320）→ argmax
```

### 3.2 显存预算（23 GB 板子）

| 项 | 大小 |
|---|---|
| 权重（16.9 GB 全驻留） | 16.89 GB |
| GDN 状态 `[48, 128, 128] fp32` × 48 层 | 151 MB |
| conv 状态 `3×C fp32` × 48 层 | 5.9 MB |
| KV cache `[4, 512, 256] fp16 ×2` × 16 层 | 32 MB |
| 激活/工作缓冲（复用 20+ 块） | ~2 MB |
| **合计** | **≈17.1 GB** ✓ |

### 3.3 权重流式加载

`meta.txt` 851 条记录逐条 `fread` → `aclrtMalloc` + H2D，**16.9 GB 加载耗时 ~84 s**
（受 SD 卡读带宽限制）。加载完一次性常驻，不做换出。

### 3.4 状态管理

- **GDN recurrent state**：每层 `[48,128,128]` fp32，初始 0，逐 token 原地更新
- **conv state**：每层 `[3, C]`，`conv1d_decode` 内部移位更新
- **KV cache**：每层 `[2, 4, 512, 256]` fp16，按 pos 写入
- 三者都是"引擎持有、kernel 原地读写"，避免回传 host

---

## 4. 关键实现细节与三个静默 bug

我们用**三重验证**（引擎 vs 独立 numpy 参考、权重转换 vs 原始 AWQ、逐算子对拍）
最终把"输出是片假名乱码"收敛到三处架构理解错误：

### 4.1 RMSNorm 是 1-centered（最致命）

```python
# transformers 官方 modeling_qwen3_5.py
class Qwen3_5RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        self.weight = nn.Parameter(torch.zeros(dim))      # 注意：初始化为 0！
    def forward(self, x):
        output = self._norm(x.float())
        return (output * (1.0 + self.weight.float())).type_as(x)   # 有效权重 = 1 + w
```

我们一开始直接用了 `w`。checkpoint 里的证据：`input_layernorm.weight` 均值只有 **0.17**
（min −0.23），`final_norm` 均值 0.94 —— 它们存的是**相对 1 的偏移**，不是增益本身。

→ **161 个 RMSNorm 权重全部改成 `1 + w`**（PC + 板端），转换器 `convert_qwen38.py` 已内置。
注意：GDN 的 `RMSNormGated` **不是** 1-centered（`weight=ones`，直接相乘）。

### 4.2 q_proj 输出是 per-head 交错

```python
query_states, gate = torch.chunk(
    self.q_proj(hidden_states).view(*input_shape, -1, self.head_dim * 2), 2, dim=-1)
```

即每个 head 的 512 维是 `[q_h(256) | gate_h(256)]`，**不是** `[q_all(6144) | gate_all(6144)]`。

### 4.3 注意力输出门是 sigmoid

```python
attn_output = attn_output * torch.sigmoid(gate)     # 不是 silu(gate)
```

配置里的 `output_gate_type: "swish"` 与实际代码不符，**以代码为准**。

### 4.4 另外两个非架构类坑

- **KV cache 写入**：`full_attn_decode` 原本用 GM 标量 `SetValue` 写 cache，结果静默出错
  → 改为 `Cast` 到 UB + `DataCopy` 向量写回。
- **首次调用异常**：`w4a16_gemv_gen` 在进程内第 1 次调用结果错、第 2 次起正确
  → 引擎 `ENG27_DOUBLE_GEMV`（先跑一次丢弃）或 warmup 前向规避。

修完这三处后：

| Prompt | 输出 |
|---|---|
| `The capital of France is` | **`Paris`** |
| `你好，请用一句话介绍我自己` | **`你好，我是通义千问，一个由阿里巴巴集团…`** |

---

## 5. 性能分析

### 5.1 实测

| 阶段 | 耗时 |
|---|---|
| 权重加载（16.9 GB） | ~84 s |
| 单 token 前向（64 层） | **14.3–14.6 s** |
| decode 吞吐 | **0.069 tok/s** |
| prefill | 与 decode 同速（逐 token 前向，无 batch） |

### 5.2 瓶颈在哪

单 token 前向 = 16.9 GB 权重读一次 + 27.8 G MAC。310B1 的 GM 读带宽实测 **10.2 GB/s**，
理论上限 ≈ 1.65 s/token；实测 14.5 s，说明**瓶颈不是带宽而是 W4 反量化 + 向量核 GEMV 的算力**：

- 我们测得的 W4A16 GEMV 最优 **1.43 GB/s**（K=17408 大矩阵），是 GM 带宽的 14%
- 单 AI core、单向量核；Cube 峰值 1.46 TFLOP/s 在本实现里**没用到**（GEMV M=1 不适合 Cube）
- 每层约 20 个 kernel 启动，48+16 layer → 每 token ~1300 次 launch，调度开销不可忽略

### 5.3 提速方向（按收益排序）

1. **Cube 矩阵乘**：把 M=1 的 GEMV 改成 M=16/32 的 GEMM（预填 batch），把算力从向量核转到 Cube
2. **prefill 批量化**：prompt 阶段一次算多 token，摊薄权重读取（当前每 token 都重读 16.9 GB！）
3. **减少 kernel 启动**：把 conv+l2norm+rgdr 融合成一个 GDN decode kernel
4. **W4 直接向量乘**：用 int8/uint8 向量指令做 4bit 解包 + 乘加，减少 Cast 开销

### 5.4 硬件上限参考

详见 [`310B-性能上限与prefill优化.md`](310B-性能上限与prefill优化.md)（实测 Cube 1.46 TF/s、
GM 10.2 GB/s、L2 27 GB/s、UB 248 KB 等，以及 prefill 优化推导）。

---

## 6. 验证方法（三重独立）

| 维度 | 方法 | 结果 |
|---|---|---|
| 算子 | host 侧 numpy 参考逐元素对拍 | 14 个算子全部 `max_abs ≤ 2e-3` |
| 全模型 | 独立 numpy 64 层流式参考（`tools/verify/ref_full_model.py`） | 单 token / 2-token logits **逐位一致** |
| 权重 | 从原始 AWQ safetensors 独立 GPTQ 反量化后对比（`tools/verify/check_conversion.py`） | 点积 **corr=0.99999957** |
| 语义 | 板端实际生成 | `Paris` / `你好，我是通义千问…` |

> 注意：引擎和我们的 numpy 参考来自同一份转换权重，所以"两者一致"只能证明**引擎实现正确**；
> 要证明**转换正确**必须回到原始 AWQ 权重做独立校验（已做）；要证明**架构理解正确**必须对照
> 官方 `modeling_qwen3_5.py`（已做，因此发现了 1-centered norm 等问题）。
