---
license: other
license_name: bsl-1.1
license_link: https://huggingface.co/moondream/moondream3-preview
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
> 权重由官方 `moondream/moondream3-preview` 转换而来，仍受官方许可约束（BSL 1.1 + Additional Use Grant）。

## 这是什么

本仓库提供在 **OrangePi AI Pro 20T（Ascend 310B1，23GB 统一内存，CANN 8.0.0）** 上端侧运行
Moondream 3 Preview（9B MoE VLM，2B 激活）所需的：

- 转换后的板端权重：文本 **W4A16 group32（相位交织）** + vision **fp16**；
- AscendC 自研算子（W4 GEMV、LayerNorm/GELU 等 12 个）+ C++ 混合引擎（decode 用自研 GEMV，prefill 用 ACLNN matmul）；
- 转换/参考/基准工具（`tools/`）；
- 与官方 modeling 的 numpy 参考逐层对拍记录、基准与实验文档（`docs/`）。

## 实测性能（OrangePi AI Pro 20T / 310B1，单 AI Core）

| 场景 | 结果 | 耗时 |
|---|---|---|
| 文本 decode（24 层） | "Hello! How can I help you today?"，参考 corr 0.9996+ | 2.38 s/token |
| 文本 prefill（7 token，ACLNN 混合引擎） | 与 decode 输出一致 | 9.9 s |
| vision encoder（2 crops，27 层 ViT） | L0/attn 与 bf16 参考 corr≈1 | 40.4 s |
| 图像问答（730 token prefix + 16 token 生成） | 红圆 → "The circle is red." | 145 s |
| 投机解码（prompt-lookup，重复文本） | 接受 68.8%，decode 1.92x（端到端无净收益） | — |

## 快速开始（板端）

```bash
cd ~/code/moondream3_310b/ops && source /usr/local/Ascend/ascend-toolkit/set_env.sh
export LD_LIBRARY_PATH=$PWD/out/lib:$PWD/out/lib64:/usr/local/Ascend/ascend-toolkit/latest/lib64:$LD_LIBRARY_PATH
./md3_engine --dir ~/code/moondream3_310b/models/moondream3_310b \
  --ids 0,1,15381,2,8299,1158,3 --gen 8
```

PC 端一键图像问答：`python tools/run_qa.py --image circle.png --question "What color is the circle?"`

## 目录

```
weights/     板端权重（meta.txt + 文本 W4 文件 + vision/*.f16）
tokenizer/   SuperBPE tokenizer（moondream/starmie-v1）
tools/       权重转换、参考实现、图像预处理、QA/基准脚本
docs/        架构/权重格式/算子/部署/基准/投机解码文档
```

## 已知限制

1. vision 全量编码与 bf16 参考存在 f16 累积偏差（问答语义正确）；
2. vision 的 attention 目前在 CPU（head_dim=72 与 310B1 的 32B/matmul 约束冲突）；
3. 图像裁剪为简化单 window（n_crops=2），大图会损失分辨率；
4. 仅验证 OrangePi AI Pro 20T（310B1）+ CANN 8.0.0。

## 许可

- **模型权重**：Moondream 3 Preview 许可（**BSL 1.1 + Additional Use Grant**，发布两年后转 Apache-2.0）；
- **本仓库代码**：MIT（见 LICENSE）；`ref/` 下官方源码遵循其原许可。

## 致谢

Moondream team（模型与官方 modeling 代码）、华为昇腾 CANN/AscendC 生态。