# tools/verify — 27B 引擎验证/修复脚本

这些脚本是从开发过程中的临时目录抢救出来的**验证与修复脚本**，用于独立校验
「AscendC 引擎 / 权重转换 / 参考实现」三者的正确性。注意：它们读取的中间 dump
文件（`tmp/eng_*.bin` 等）已在清理时删除，需要重新生成。

## 依赖的数据

- `converted/qwen38_310b/` — 转换后的 27B 权重（必须已应用 1-centered norm 修复）
- `models/Qwen3.8-27B-AWQ-INT4/` — 原始 AWQ 权重（仅 `check_conversion.py` 需要）
- `tmp/p1_eng_post.bin`、`tmp/p1m_eng_*.bin` 等中间 dump — 由板端引擎在
  `ENG27_DUMP*` 环境变量下产生，再用 `scp` 取回（见各脚本头部注释）

## 脚本说明

| 脚本 | 作用 |
|---|---|
| `check_conversion.py` | **权重转换独立校验**：从原始 AWQ safetensors 用 GPTQ 规则独立反量化，与我们转换产物逐元素比较（对拍层 0 in_proj_qkv；实测 corr=0.99999957） |
| `fix_norm_centering.py` | **1-centered RMSNorm 修复**：把 161 个 RMSNorm 权重改为 `1 + w`（Qwen3.5 `Qwen3_5RMSNorm` 语义）。GatedDeltaNet 的 RMSNormGated 不在其中 |
| `ref_full_model.py` | **全 64 层 numpy 独立参考**（单 token、pos=0）：流式读盘跑完 GDN+注意力+MLP+final norm+lm_head，输出 top-5 logits |
| `ref_full_2token.py` | 同上但 2 token，含 conv/GDN 状态演进与 KV cache（pos=1 logits） |
| `ref_gdn_layer.py` / `ref_gdn_2token.py` | GDN 层（源层 10）单/双 token 参考，含 conv 状态、RGDR 递推 |
| `ref_attn_layer.py` / `ref_attn_2token.py` | 全注意力层（源层 3）单/双 token 参考，含 qk-norm、partial rope、sigmoid 输出门、KV cache |
| `ref_pipeline0.py` | 0 层流水线参考（embed → final norm → lm_head） |
| `cmp_stages.py` | 把板端 dump 的中间量（xn/qkv/z/a/b/conv）逐项与参考比较 |
| `gen_down_test.py` / `gen_real_gemv_test.py` | 用**真实权重**生成 GEMV 单测数据：down_proj（K=17408）、gate_proj（N=17408）放入 `ascendc_ops/input/`、`ascendc_ops/input_golden/` |

## 关键结论（这些脚本验证过的）

- 引擎 logits 与 `ref_full_model.py` / `ref_full_2token.py` **逐位一致**
  （如 475(4.41) vs 475(4.408)；169571(5.00) vs 169571(4.98)）
- 权重转换与原始 AWQ 点积 **corr=0.99999957**
- 修好 3 个 bug 后模型输出语义正确："The capital of France is" → **"Paris"**、
  "你好，请用一句话介绍你自己" → **"你好，我是通义千问，一个由阿里巴巴集团…"**
