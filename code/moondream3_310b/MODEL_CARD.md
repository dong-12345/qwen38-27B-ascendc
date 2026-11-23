---
license: other
license_name: business-source-license-1.1
license_link: https://huggingface.co/moondream/moondream3-preview/blob/main/LICENSE
language:
  - en
tags:
  - moondream
  - vision-language
  - ascend
  - ascend-310b
  - orangepi
  - int4
  - deployment
pipeline_tag: image-text-to-text
---

# Moondream 3 Preview on Ascend 310B1 (OrangePi AI Pro 20T)

> 这不是 Moondream 官方模型仓库，而是 **Moondream 3 Preview 在昇腾 310B1 板卡上的部署套件与实验记录**。
> 本仓库中的权重是官方权重的 **量化/格式转换版本（Derivative）**，仍受 Moondream 官方许可约束。

## License（重要，请先读）

**权重与衍生版本：Business Source License 1.1（BSL 1.1，Licensor: M87 Labs, Inc.）**

- 官方 Licensed Work 为 “Moondream 3 (Preview)”，**明确包括 Model Weights and any Derivatives**；
  Derivatives 包括 fine-tunes、merges、**quantizations**、weight deltas 以及**其他 weight-level 的修改或转换**——
  本仓库的 W4A16 量化与 fp16 vision 权重正属于这一类，**因此仍然受 BSL 1.1 约束，并非 MIT/Apache**。
- **Additional Use Grant**：允许 **production use**，条件是不得把 Licensed Work 以 **hosted 或 embedded** 方式
  提供给第三方、且构成与 M87 Labs 付费版本竞争的 **付费产品（competitive offering）**。
  - 非付费提供的产品不属于 competitive offering；
  - 组织内部使用不算 competitive offering；
  - 若你的产品在首次公开发布时不属于 competitive offering，之后不因 M87 发布新版本而变成 competitive；
  - 如需商业/其他授权，请联系 contact@m87.ai。
- **Change Date**：该版本首次公开发布后 **两年**；到期后按 **Apache License 2.0** 授权。
- 按 BSL 1.1 要求，任何原始/修改副本**必须显眼展示该 License**：本仓库根目录附完整 `LICENSE` 文件。
- 本 License 不授予 M87 Labs 商标/Logo 的任何权利。

**本仓库中的脚本与文档（`tools/`、`docs/`）为 MIT License**（见 `LICENSE-MIT`），与权重许可相互独立。
部署代码仓库：<https://github.com/dong-12345/moondream3_310b>（代码 MIT）。

## 这是什么

本仓库提供在 **OrangePi AI Pro 20T（Ascend 310B1，23GB 统一内存，CANN 8.0.0）** 上端侧运行
Moondream 3 Preview（9B MoE VLM，2B 激活）所需的**转换后权重**：

- 文本：**W4A16 group32（相位交织）**，5.4 GB；
- vision：**fp16**（官方 int4 仓库中 vision 权重本就是 bf16 未量化），0.85 GB；
- 配套的 `meta.txt`、tokenizer（SuperBPE `moondream/starmie-v1`）。

部署代码（AscendC 算子 + C++ 混合引擎 + 转换/基准工具）在 GitHub：
<https://github.com/dong-12345/moondream3_310b>（MIT）。

## 实测性能（OrangePi AI Pro 20T / 310B1，单 AI Core）

| 场景 | 结果 | 耗时 |
|---|---|---|
| 文本 decode（24 层） | "Hello! How can I help you today?"，参考 corr 0.9996+ | 2.38 s/token |
| 文本 prefill（7 token，ACLNN 混合引擎） | 与 decode 输出一致 | 9.9 s |
| vision encoder（2 crops，27 层 ViT） | L0/attn 与 bf16 参考 corr≈1 | 40.4 s |
| 图像问答（730 token prefix + 16 token 生成） | 红圆 → "The circle is red." | 145 s |
| 投机解码（prompt-lookup，重复文本） | 接受 68.8%，decode 1.92x（端到端无净收益） | — |

## 用法（板端）

1. 把本仓库 `weights/` 内容放到板端 `~/code/moondream3_310b/models/moondream3_310b/`（保持 `meta.txt` + 权重文件 + `vision/` 结构）；
2. 按 GitHub 仓库 README 编译 `ops/` 并运行 `md3_engine`（`--dir` 指向上面的目录）。

## 已知限制

1. vision 全量编码与 bf16 参考存在 f16 累积偏差（问答语义正确）；
2. vision 的 attention 目前在 CPU（head_dim=72 与 310B1 的 32B/matmul 约束冲突）；
3. 图像裁剪为简化单 window（n_crops=2），大图会损失分辨率；
4. 仅验证 OrangePi AI Pro 20T（310B1）+ CANN 8.0.0。

## 致谢

Moondream team（模型与官方 modeling 代码）、华为昇腾 CANN/AscendC 生态。