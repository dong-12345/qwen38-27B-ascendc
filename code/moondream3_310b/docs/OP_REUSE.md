# 旧算子复用与自研算子清单

> 基线：`ascendc_ops/`（Qwen3.8-27B 项目，14 个自研 AscendC 算子，310B1 实测可用）。
> Moondream 3 Preview 的文本塔是 **LayerNorm + GeGLU + MoE + 学习式 tau 温度缩放**，与 Qwen3.5 的 **RMSNorm + SwiGLU + GDN** 差异很大，因此只有一部分能直接复用。

## 1. 直接复用（无需改代码）

| 旧算子 | 在 Moondream 中的用途 | 说明 |
|---|---|---|
| `w4a16_gemv_gen` | **decode 主路径所有线性层**：qkv(2048→6144)、proj(2048→2048)、dense fc1/fc2、router(2048→64)、MoE 专家 fc1(2048→2048)/fc2(1024→2048)、lm_head(2048→51200) | 相位交织 W4A16 group32，与转换格式完全一致；专家权重按指针偏移调用 |
| `fp16_gemv` | tau 的 `wq/wv`（6144→32） | f16 权重 GEMV |
| `vec_axpy` | 残差 `x += proj`、MoE 8 专家加权累加 | `out = a + b*s` |

## 2. 需要改造 / 新增自研算子（10 个，全部已板端对拍通过）

| 新算子 | 替代/对应旧算子 | 原因 |
|---|---|---|
| `layernorm_f16` | 新写（旧的是 `rmsnorm_op`） | Moondream 用 **LayerNorm(weight+bias)**，且 LN 输出同时给 attention 与 MLP（每层仅 1 个 LN） |
| `gelu_f16` | 新写（旧的是 `mul_silu`） | 激活是 GELU(tanh approx)；310B 的 ACLNN Gelu kernel 缺失（实测 `aclnnGelu` 失败） |
| `geglu_f16` | 改造 `mul_silu` | MoE 专家是 `gelu(gate)*(up+1)`，不是 `silu(gate)*up` |
| `attn_decode_text` | 改造 `full_attn_decode` | 32 q / 32 kv（无 GQA）、D=64、partial RoPE 32、**无输出门控**、增加 **tau 温度缩放**；KV cache 布局 [32,4096,64] |
| `qkv_post_text` | 新写 | prefill 用：qkv bias + tau + partial RoPE + 拆 q/k/v 为 head-major |
| `dequant_w4_f16` | 新写 | prefill 用：W4 group32 → fp16 矩阵，喂 ACLNN MatMul |
| `add_bias_f16` | 新写 | ACLNN MatMul 无 bias，prefill 后补 bias |
| `heads_to_seq` | 新写 | prefill 注意力输出 [32,T,64] → [T,2048] |
| `add_f16` | 新写（`vec_axpy` 不分块，大 N UB 放不下） | prefill 残差 [T,2048] |
| `qkv_post_vision` | 新写 | Vision ViT prefill：qkv bias + 拆头 + q×(1/8)，无 RoPE/tau |

## 3. 完全用不上（与 Moondream 无关）

- GDN 系列：`gdn_l2norm`、`gdn_rgdr`、`gdn_gating`、`gdn_conv1d_silu`、`gdn_cumsum`、`gdn_kkt`、`gdn_tril`、`gdn_wy`、`gdn_chunk_h`、`gdn_chunk_o`、`rmsnorm_gated`、`conv1d_decode`；
- 这些是 Qwen3.5 Gated DeltaNet 专属，Moondream 没有线性注意力层。

## 4. 复用率

- decode 主路径：3 个旧算子承担 ~95% 的权重读取与向量计算（W4 GEMV 是绝对主力），**复用价值最大**；
- 新算子多数是"轻量 glue"（biass/reshape/激活），复杂度集中在 `attn_decode_text` 与 prefill 的 ACLNN 编排；
- Vision 编码器（27 层 ViT）不适合用 GEMV 逐个 token 跑，prefill 阶段统一走 **ACLNN MatMul（310B 实测 0.29–0.38 TFLOPS）**。
