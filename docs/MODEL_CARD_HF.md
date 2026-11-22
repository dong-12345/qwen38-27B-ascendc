# Model Card — Qwen3.8-27B (Qwen3.5 hybrid GDN) W4A16-Phase-Interleaved for Ascend 310B

本仓库是 **Qwen3.8-27B（Qwen3.5 混合 Gated DeltaNet 架构，28B 参数）** 的**重排/重打包 4bit 权重**，
专供自研实验性推理引擎 [`qwen38-ascendc`](https://github.com/dong2542107715/qwen38-ascendc) 在
**Ascend 310B（OrangePi AI Pro 20T）** 上使用。

> ⚠️ 这不是通用格式（不是 AWQ/GPTQ/GGUF/bnb），只对我们自己的 AscendC 引擎有效。
> 需要通用格式请使用官方原始权重。

---

## 1. 基本信息

| 项 | 值 |
|---|---|
| 基础模型 | Qwen3.8-27B（Qwen3.5 架构：48 层 Gated DeltaNet + 16 层全注意力，hidden 5120，FF 17408，vocab 248320） |
| 上游权重格式 | AWQ 4bit，group_size 32（compressed-tensors `pack-quantized`） |
| 本仓库格式 | **W4A16 相位交织 + fp16 group scale / scale·zp**，group_size 32 |
| 参数量 | 27.78 B（4bit 后权重 ≈16.9 GB） |
| 文件数 | 1670 个（`model/` 下按层组织） + `meta.txt` |
| 语言 | 多语言（文本）；本打包不含视觉/音频塔 |
| 许可 | **Apache License 2.0**（上游 LICENSE 实测：Copyright 2026 Alibaba Cloud）；本仓库同许可分发并声明修改内容 |

---

## 2. 目录结构

```
meta.txt                                   # 引擎索引（扁平文本）
model/language_model/
  embed_tokens/weight.{w4,s,sz}.bin        # W4（重打包）
  layers/{0..63}/
    input_layernorm/weight.f16.bin         # 已做 1-centered 修正：有效权重 = 1 + w
    post_attention_layernorm/weight.f16.bin
    linear_attn/                           # GDN 层（48 层）
      in_proj_qkv.{w4,s,sz}.bin            # 10240 × 5120
      in_proj_z.{w4,s,sz}.bin              # 6144 × 5120
      in_proj_a/weight.f16.bin             # 48 × 5120
      in_proj_b/weight.f16.bin
      conv1d/weight.f16.bin                # [10240, 4]
      A_log.f16.bin  dt_bias.f16.bin
      norm/weight.f16.bin                  # RMSNormGated（**不是** 1-centered）
      out_proj/weight.f16.bin              # 少数层未量化
    self_attn/                             # 全注意力层（16 层）
      q_proj.{w4,s,sz}.bin                 # 12288 × 5120，per-head [q|gate] 交错
      k_proj.{w4,s,sz}.bin                 # 1024 × 5120
      v_proj.{w4,s,sz}.bin                 # 1024 × 5120
      o_proj.{w4,s,sz}.bin                 # 5120 × 6144
      q_norm/weight.f16.bin  k_norm/weight.f16.bin
    mlp/
      gate_proj.{w4,s,sz}.bin              # 17408 × 5120
      up_proj.{w4,s,sz}.bin                # 17408 × 5120
      down_proj.{w4,s,sz}.bin              # 5120 × 17408
  norm/weight.f16.bin                      # final norm（1-centered 已修正）
lm_head/weight.{w4,s,sz}.bin               # 248320 × 5120
```

### 2.1 W4 文件格式

对 `N × K` 的权重矩阵：

| 文件 | dtype | 形状 | 说明 |
|---|---|---|---|
| `*.w4.bin` | `uint16` | `[N, K/4]` | **相位交织**：word `j` 的 4 个半字节依次是 `q[j], q[j+K/4], q[j+2K/4], q[j+3K/4]` |
| `*.s.bin` | `fp16` | `[N, SBP]` | group scale，`SBP = ceil(K/32/16)*16` |
| `*.sz.bin` | `fp16` | `[N, SBP]` | `s × zero_point`（用于 `out = Σ s·Σq·act − Σ sz·suma`） |

反量化：`w[n,k] = s[n, k//32] · q[n,k] − sz[n, k//32]`

### 2.2 f16 文件

原始 fp16（bf16→fp16 无损范围内）权重直接存为 `.f16.bin`（little-endian）。
**注意**：所有 `Qwen3_5RMSNorm` 类的权重（`input_layernorm`、`post_attention_layernorm`、
`q_norm`、`k_norm`、`model/language_model/norm`）存的是 **有效权重 `1 + w`**，
因为官方实现是 1-centered 的；`linear_attn/norm`（RMSNormGated）不是 1-centered，保持原值。

---

## 3. 如何生成（从原始 AWQ 权重）

```bash
git clone https://github.com/dong2542107715/qwen38-ascendc
cd qwen38-ascendc

# 1) 转换（需要原始 AWQ 权重目录）
python tools/convert_qwen38.py \
  --src /path/to/Qwen3.8-27B-AWQ-INT4 \
  --out converted/qwen38_310b \
  --group 32 --requant-extra embed_tokens,lm_head

# 2) 生成引擎索引
python tools/export_engine_meta.py --dir converted/qwen38_310b
```

转换器内置 **1-centered RMSNorm 修正**；若你手动转换过旧版本，请用
`tools/verify/fix_norm_centering.py` 修补，否则模型输出会是乱码。

### 3.1 上传到 HuggingFace

```bash
pip install huggingface_hub
export HF_TOKEN=hf_xxx
python tools/push_to_hf.py \
  --folder converted/qwen38_310b \
  --repo dong2542107715/Qwen3.8-27B-AscendC-W4A16 \
  --private
```

脚本会把 `docs/MODEL_CARD_HF.md` 作为 `README.md` 一起上传，并支持断点续传（`upload_folder`）。

---

## 4. 使用（Ascend 310B 引擎）

```bash
# 板端：把本仓库权重放到 ~/code/qwen38_310B/models/qwen38_310b
cd ~/code/qwen38_310B && source env.sh
ENG27_GEN=8 ./src/out/bin/qwen38_ops_test engine_27b
```

实测（OrangePi AI Pro 20T，Ascend 310B1，CANN 8.0.0）：

| 指标 | 数值 |
|---|---|
| 权重加载 | ~84 s（16.9 GB） |
| decode | **0.069 tok/s（14.5 s/token）** |
| 语义正确性 | `The capital of France is` → `Paris`；`你好，请用一句话介绍你自己` → `你好，我是通义千问…` |

---

## 5. 限制

- **仅适配本项目引擎**：布局与我们的 kernel 强绑定，通用框架无法直接加载。
- **单 token 吞吐低**：310B 单核向量 GEMV，14.5 s/token；不适合交互式场景。
- **不含 MTP 模块、不含视觉/音频塔**。
- **fp16 激活**：与上游 bf16 训练有细微数值差异（实测 logits 与独立 fp32 参考一致到 1e-2）。
- `embed_tokens` / `lm_head` 也做了 4bit 重量化（误差 0.0000~0.0002，可忽略）。

## 6. 许可与合规（Apache-2.0）

上游 `Qwen3.8-27B` 的 LICENSE 实测为 **Apache License 2.0，Copyright 2026 Alibaba Cloud**（见权重目录 `LICENSE`）。
Apache-2.0 **允许再分发与修改**，但需履行以下义务（本仓库已逐条落实）：

| Apache-2.0 条款 | 本仓库的做法 |
|---|---|
| §4(a) 提供 License 副本 | 权重目录内含 `LICENSE`（原样复制） |
| §4(b) 声明修改 | 权重目录内含 `MODIFICATIONS.md`，逐项列出：重打包为 W4A16 相位交织格式、bf16→fp16、1-centered norm 加 1、只保留语言塔并移除视觉塔/MTP、embed/lm_head 重量化 |
| §4(c) 保留归属声明 | 本 model card 与 `MODIFICATIONS.md` 均保留 "Copyright 2026 Alibaba Cloud" 与上游模型名 |
| §6 商标 | 仅以描述性方式提及 "Qwen"（说明来源），不暗示官方背书；未使用 Qwen logo |

**使用须知**：本权重是**实验性格式**（只对自研 AscendC 引擎有效），不是通用模型格式；
需要通用格式请使用官方原始权重。商用/再分发前请自行复核上游许可条款与适用法律。

## 7. 引用 / 致谢

- 基础模型与架构：Qwen 团队（Qwen3.5）
- AscendC 算子与引擎：本项目
- 参考实现：`transformers` 的 `modeling_qwen3_5.py`（用于对齐 1-centered RMSNorm、per-head q/gate 布局、sigmoid 输出门）
