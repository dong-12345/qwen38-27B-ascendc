# Moondream 3 Preview 架构拆解（从官方 modeling 代码提取）

> 依据：`ref/moondream3_src/{config,text,layers,vision,rope,image_crops,moondream}.py`（Moondream 官方 HF 实现）。
> 关键结论：**24 层中前 4 层 dense、后 20 层 MoE-64 选 8；每层只有 1 个 LayerNorm（attention 与 MLP 共享归一化输入）；attention 带学习式 tau 温度缩放；视觉塔为 SigLIP 式 ViT，输出 729 个 image token，加 BOS 共 730 个前缀 token。**

## 1. 文本塔（24 层）

| 参数 | 值 |
|---|---|
| hidden dim | 2048 |
| layers | 24（前 4 dense，后 20 MoE） |
| attention | 32 q heads / 32 kv heads（无 GQA），head_dim 64 |
| RoPE | **partial**：只转前 32 维（16 个频率），theta = 1,500,000 |
| attention 缩放 | 学习式温度缩放（见 §2），基础 scale = 1/8 |
| FFN（dense 层） | dim 2048 → 8192 → 2048，GELU(tanh) |
| FFN（MoE 层） | 64 experts，每 token 选 8；expert inner = 1024；GeGLU |
| 归一化 | **LayerNorm(weight+bias)**，eps=1e-5；每层仅 1 个 LN |
| vocab | 51200（SuperBPE tokenizer `moondream/starmie-v1`） |
| max_context | 4096（config 值；模型卡宣称支持到 32K） |
| lm_head | 51200×2048 + bias |

### 每层数据流

```
x ──LN──> xn
xn ──qkv(2048→6144)──> [q|k|v]  (每段 2048)
xn ──gelu──> feat ──wq/wv(6144→32)──> tanh ──> tok_q / tok_v  (每 head 一个标量)
q *= (tok_q + tau_pos(pos)) * 1/8 ;  k = k ;  v *= (tok_v + tau_pos(pos))
q,k 做 partial RoPE(前 32 维)
KV cache 写入
attn = softmax(q·kᵀ) · v         (32 heads，无门控)
attn ──proj(2048→2048)+bias──> 残差 x += attn
dense:  x += fc2(gelu(fc1(xn))) + bias
MoE:    router = xn·Wrᵀ + br (64)；top-8；softmax(top8)；
        对每个专家 e:  h = xn·W1ₑᵀ;  [u,g]=chunk(h);  y = gelu(u)*(g+1);  out += wₑ · (y·W2ₑᵀ)
        x += out
```

注意：
- `gelu` 是 **tanh 近似**：`0.5x(1+tanh(0.79788456(x+0.044715x³)))`；
- GeGLU 的第二路是 `(g + 1)`，不是 `g`；
- tau 的 `tau_pos = 0.5 + sigmoid(alpha_h · ln(pos+1))`（代码写作 `1 + sigmoid(...) - 0.5`）；
- MoE 路由 softmax 只对 **top-8 logits** 做（不是全量 64）。

## 2. 视觉塔（SigLIP 式 ViT）

| 参数 | 值 |
|---|---|
| 层数 | 27 |
| dim / FF | 1152 / 4304，16 heads，head_dim 72 |
| patch | 14×14，crop 378×378 → 27×27 = **729 patch** |
| 位置编码 | 可学习 pos_emb [729,1152] |
| 归一化 | 标准 LayerNorm（ln1 pre-attn，ln2 pre-MLP），GELU |
| crops | 1 个 global crop + 最多 12 个局部 overlap crops（最多 13 个 crop） |
| 输出投影 | 每 crop 输出 [729,1152]；局部 crop 反拼接成整图 → adaptive_avg_pool 到 27×27；与 global 拼接成 [729,2304] → MLP 2304→8192→2048（GELU） |

### 图像 → 文本前缀

```
13 crops ──ViT(独立 batch)──> [13,729,1152]
global = out[0]  (729×1152)
local  = out[1:] 反拼接 + adaptive_avg_pool(27×27)
feat   = cat([global, pooled], -1)   [729,2304]
img_emb= MLP(feat)                   [729,2048]
prefix = [BOS_emb(1×2048); img_emb]  [730,2048]   ← 这 730 个 token 的 attention 全连接
```

之后 question token 从 pos=730 开始 prefill，attention 掩码为 causal（可看到全部 730 个 image token）。

## 3. 端侧部署的算力/内存要点（310B1）

- 文本 decode 每 token 只激活 2B 参数：MoE 0.5B + attention/dense/lm_head ~0.3B；
- W4 权重常驻约 **4.8–5.4GB**，KV cache（4096 上下文、24 层、32 头 × 64 维 × fp16）约 **0.81GB**；
- Vision 编码器 prefill 计算量约 **8–9 TFLOP**，文本 730-token prefill 约 **1 TFLOP**，按 ACLNN 实测 0.3 TFLOPS 估算约 30–40 s/图（一次）；
- decode 受 GM 带宽限制：每 token 约读 0.9GB W4 权重，按自研 GEMV 实测 1.4GB/s 估算约 0.6–0.8 s/token。
