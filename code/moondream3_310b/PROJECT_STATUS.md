# Moondream 3 Preview on OrangePi AI Pro 20T — 项目状态

更新：2026-11-23（文本+图像问答均跑通）

## 已完成

| 项 | 状态 | 证据 |
|---|---|---|
| 开源状态 / 许可证 | ✅ | \`docs/OPEN_SOURCE.md\`：BSL 1.1 + Additional Use Grant，非 OSI 开源，2 年后 Apache-2.0 |
| 架构拆解 | ✅ | \`docs/ARCHITECTURE.md\`：24 层 / MoE64-8 / 每层单 LayerNorm / tau / SigLIP 27 层 / 730 image prefix |
| 板端能力实测 | ✅ | \`docs/BOARD_CAPABILITY.md\`：ACLNN MatMul 0.29–0.38 TFLOPS、aclnnGelu 缺失、BatchMatMul 不支持、310B1 固定 1 AI Core |
| 算子复用 / 自研 | ✅ | \`docs/OP_REUSE.md\`：复用 3 个（w4a16_gemv_gen / fp16_gemv / vec_axpy），自研 12 个 |
| 新算子 + host 对拍 | ✅ | \`ops/kernels/md3_ops.cpp\`；\`./md3_engine test\` 全绿 |
| 权重转换工具 | ✅ | \`tools/convert_moondream3.py\`：int4 group64 → W4A16 group32 相位交织；完整 24 层 5.5GB 已上板 |
| C++ decode 引擎 | ✅ | 24 层文本 decode 与 numpy 参考 corr≈0.9996，生成 "Hello! How can I help you today?" |
| 文本 prefill（ACLNN 混合引擎） | ✅ | \`--prefill\` 与 decode 路径同 token 同输出；修掉首次 w4a16 调用异常、KV 拷贝同步等问题 |
| 视觉编码器（plate） | ✅ | \`encodeImage()\`：patch_emb(CPU 精确) + 27 层 ViT（f16 权重 matmul + CPU attention）+ post_ln + proj_mlp |
| 图像→730 token 多模态 prefill | ✅ | image prefix（730 token 全连接 mask）+ question prefill（prefix+causal mask）跑通，无崩溃 |
| 图像问答端到端 | ✅ | 红圆图 → "The circle is red."；vision 40.4 s + image prefix 30.1 s + 生成，端到端 145 s |
| 性能实测 | ✅ | 文本 decode 2.38 s/tok；prefill 7 token 9.9s；vision f16 40.4s；image prefix 30.1s（docs/BENCHMARK.md） |
| 投机解码实验 | ✅ | prompt-lookup：接受 68.8%，decode 1.92x，端到端无净收益（docs/SPECULATIVE.md） |
| AI Core 实验 | ✅ | 310B1 固定 1 AI Core（platform_config ai_core_cnt=1，npu-smi 无配置入口）；单核实测 0.29–0.38 TFLOPS |
| benchmark / 文档 | ✅ | docs/BENCHMARK.md、docs/SPECULATIVE.md、README/PROJECT_STATUS 更新 |

## 目录

\`\`\`
code/moondream3_310b/
  README.md  PROJECT_STATUS.md
  docs/      OPEN_SOURCE / ARCHITECTURE / WEIGHT_FORMAT / OP_REUSE / BOARD_CAPABILITY / DEPLOY
  ops/       板端 AscendC 算子 + C++ 引擎 + CMake
  tools/     convert_moondream3 / convert_vision_f16 / ref_moondream3 / run_demo / run_qa / compare_logits / prep_image
  ref/       官方 modeling 源码（对照）
\`\`\`

## 已知问题 / 后续优化

1. **vision 速度**：40.4 s 中 CPU attention（head_dim=72，ACLNN 不支持非 32B 对齐 stride）约 15–20 s。
   计划：qkv_post 输出 [H,T,80] + DataCopyPad，或物理转置 K 后用 ACLNN matmul，把 vision 降到 20 s 内。
2. **文本 decode 2.38 s/tok**：MoE 权重读取约 0.5 GB/token，受单核带宽限制；可尝试专家权重常驻/融合 kernel。
3. **image prefix 30 s**：730 token 的 MoE prefill 每专家 host 分组/同步；设备端累加优化已验证必要性，但首版实现有 bug，
   已回退（正确性优先），后续再重做。
4. `./md3_engine test` 中 attn_decode_text 单测在容差边缘（整模型 decode 与参考 corr 0.9996+，功能正确）。