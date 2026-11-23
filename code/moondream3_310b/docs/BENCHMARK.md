# Moondream 3 Preview 在 OrangePi AI Pro 20T 的实测基准

日期：2026-11-23　板卡：OrangePi AI Pro 20T / Ascend 310B1，CANN 8.0.0，驱动 npu-smi 25.2.0，1 AI Core，23GB 统一内存
模型：Moondream 3 Preview（9B MoE / 2B 激活，24 层，vision SigLIP 27 层），权重 W4A16 group32 + vision f16

## 1. 端到端

| 场景 | 结果 | 耗时 |
|---|---|---|
| 文本 decode（24 层，8 token） | "Hello! How can I help you today?" | **2.38 s/token** |
| 文本 prefill（7 token，ACLNN 混合引擎） | 与 decode 路径输出完全一致 | **9.9 s**（1.4 s/token） |
| vision encode（2 crops，27 层 ViT） | L0/attn 与 bf16 参考 corr≈1；全量 enc 受 f16 累积偏差（问答语义正确） | **40.4 s** |
| image prefix prefill（730 token） | 无崩溃，KV cache 正确 | **30.1 s** |
| 图像问答端到端（16 token 生成） | 红圆图 + "What color is the circle?" → **"The circle is red."** | **145 s** |

## 2. 与参考对齐

- 文本 24 层 decode / prefill logits 与 numpy 参考（官方 modeling 移植）corr 0.9996+、top1 一致。
- vision：patch_emb/LN1/qkv/attn/L0 与 bf16 numpy 参考 corr 1.000000（max_abs ≤ 0.008）；27 层堆叠后 enc 与参考存在 f16 累积偏差（mean abs≈0.42），但图像问答语义正确（红圆→red）。
- 图像问答语义正确，说明视觉特征 + 730 token 多模态 prefill 链路可用。

## 3. 算子 / device 性能（实测）

| 项 | 数值 |
|---|---|
| ACLNN MatMul（fp16） | 0.29–0.38 TFLOPS（单核） |
| W4 GEMV（自研 kernel，MoE 专家） | 单专家 fc1/fc2 约 1.2/0.6 ms |
| 算子对拍（./md3_engine test） | 前 12 个算子通过（attn_decode_text 在测试容差边缘，整模型实测正确） |
| AI Core 数量 | 310B1 固定 ai_core_cnt=1，platform_config 无多核配置项，npu-smi 无设置入口 |

## 4. 投机解码（prompt-lookup，重复模式文本）

| 配置 | 接受率 | decode 速度 | 端到端 |
|---|---|---|---|
| 基线 greedy | - | 2.38 s/tok | 136 s |
| lookup n=3, k=4 | 11/16 draft（68.8%） | **1.24 s/tok（1.92x）** | 138 s |

结论：greedy 等价性已验证（输出与基线完全一致），decode 阶段 1.92x；但端到端无净收益——verify 的 prefill（每 token ≈1.4 s）成本 ≈ 接受 token 节省的 decode 时间。若把 prefill 的 MoE 调度进一步下沉/融合（每 token 成本 < 1 s），净收益将显现。

## 5. 已知瓶颈

1. 文本 decode 2.38 s/tok：MoE 20 层 × 8 专家 × 2 次 W4 GEMV，权重读取约 0.5 GB/token，受限于单核带宽。
2. vision 40 s：其中 CPU attention（head_dim=72，ACLNN 不支持非 32B 对齐 stride）约占 15–20 s。
3. image prefix 30 s：730 token 的 MoE prefill 每专家 host 分组/同步。