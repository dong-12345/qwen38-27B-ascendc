# Qwen3.8-27B 部署可行性调研报告（2026-11）

> 结论先行：**OrangePi AI Pro 20T（昇腾 310B4）无法运行 Qwen3.8-27B，任何量化版本都不行。**
> 要跑它：昇腾 310P（Atlas 300I Duo / AI Studio Pro）约 9-21 tok/s，或 2×RTX 3090 约 67 tok/s。
> 本报告由子代理调研 + 实测数据汇总，含可直接执行的命令。

---

## 1. 为什么 310B 完全无解（四重死锁）

| 关卡 | 事实 | 来源 |
|---|---|---|
| **引擎** | vLLM-Ascend FAQ 明确 `Atlas 200I A2 (Ascend-cann-kernels-310b)` 未计划；SGLang-Ascend 仅 A2/A3；llama.cpp 的 ggml-cann 后端不支持 310B | vLLM-Ascend FAQ |
| **算子** | Qwen3.8 是 64 层混合架构（48× Gated DeltaNet + 16× Gated Attention）+ 视觉编码器 + MTP。GDN 的 `RecurrentGatedDeltaRule` 在 vLLM-Ascend 里只有 `_310p` 实现和 910B 路径；310B 的 aclnn 算子覆盖极窄（社区实测连 Cast/Pad 都缺） | vllm-ascend 源码 |
| **内存** | 310B NPU 内存 23.6GB（与 CPU 共享）。Qwen3.8-27B：AWQ INT4 ≈ 21GB、W8A8 ≈ 27GB+、GGUF Q4_K_M ≈ 17GB。除权重还要 KV/视觉塔/运行时 | 权重实测 |
| **带宽** | **310B DDR 实测读带宽 28.8 GB/s**（1 AI Core @1224MHz）。27B INT4 每 token 至少读 ~14GB → 理论 2 tok/s，实际 ~1 tok/s | 知乎实测 Gemma4 E2B |

> 补一个新数据点：有人在 310B 上用自研 C++/AscendC 引擎跑通 **Gemma4 E2B（2B 级，Q4_0）→ 11.8 tok/s**。
> 说明瓶颈就在"体积 ÷ 28.8GB/s"，不是"板子太弱"。

---

## 2. Qwen3.8 家族（没有小尺寸兄弟）

| 模型 | 规格 | 310B |
|---|---|---|
| **Qwen3.8-27B** | 27B dense，64 层（48 GDN + 16 全注意力），hidden 5120，4 KV 头，原生视觉 + MTP，262K→1M，Apache-2.0 | ❌ |
| Qwen3.8-2.4T-A95B | 2.4T 总 / 95B 激活 MoE | ❌ |
| Qwen3.8-Flash-Next | 125B 总 / 6B 激活 + 51B n-gram | ❌ |
| Qwen3.8-Max / Flash | API only | — |

**Qwen3.8 代际没有 0.8B/4B/8B 小模型**。小尺寸在 Qwen3.5 代（0.8B/2B/4B/9B），但同样全是 **GDN 混合架构** → 310B 依旧不行。

---

## 3. 各硬件部署路径

### 3.1 昇腾 310P（Atlas 300I Duo 96G / OrangePi AI Studio Pro 96/192G）— 官方验证

- 权重：`Eco-Tech/Qwen3.8-27B-w8a8-310p`（36.4GB / 9 分片，含视觉 + MTP 头）
- 镜像：`quay.io/ascend/vllm-ascend:v0.23.0-310p`（CANN 9.1.0 / torch_npu 2.10.0 / vllm 0.23.0）
- 硬件价格：300I Duo 96G 约 ¥1.1–1.36 万；AI Studio Pro 96G ¥13,606 / 192G ¥15,698（预售）

**生产稳定版（不开 MTP，实测 9.1 tok/s）**：

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1
vllm serve /root/.cache/models/Qwen3.8-27B-w8a8-310p \
  --host 0.0.0.0 --port 8080 \
  --tensor-parallel-size 2 \
  --served-model-name qwen3.8 \
  --max-num-seqs 16 --max-model-len 262144 \
  --trust-remote-code --quantization ascend \
  --dtype float16 --mamba-ssm-cache-dtype float16 \
  --gpu-memory-utilization 0.90 --enable-prefix-caching \
  --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [1,8]}' \
  --additional-config '{"ascend_compilation_config": {"enable_npugraph_ex": false}}'
```

**含 MTP 投机解码（官方文档版）**：

```bash
vllm serve Eco-Tech/Qwen3.8-27B-w8a8-310p \
  --host 0.0.0.0 --port 8000 \
  --tensor-parallel-size 4 --served-model-name qwen3.8 \
  --max-num-seqs 128 --max-model-len 16384 \
  --trust-remote-code --gpu-memory-utilization 0.90 \
  --mamba-ssm-cache-dtype float16 --dtype float16 \
  --speculative-config '{"method": "qwen3_5_mtp","num_speculative_tokens":1}' \
  --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [2,16]}' \
  --additional-config '{"ascend_compilation_config": {"enable_npugraph_ex": false}}'
```

⚠️ **310P 上 MTP 的真实表现（重要，别盲信文档）**：
- 社区生产实测（AI Studio Pro 192G，k=1）：acceptance 20%~57%、平均接受长度 1.2–1.57，但 **8.8 tok/s vs 不开 MTP 的 9.1 tok/s（净负收益）**，且**中文输出乱码**（310P rejection sampler 走 fallback 有验证数学缺陷），已回滚。
- 另一路实测（去掉 `--enforce-eager` + 图模式 + MTP3）：decode 9.4–10.9 → **18.3–21.3 tok/s**（未复现）。

**310P 关键约束**：只支持 TP=2/4；`--dtype float16` 必须；`--mamba-ssm-cache-dtype` 只支持 float16；`num_speculative_tokens` 建议 1；capture size 必须是 `n × (num_speculative_tokens+1)`；`enable_npugraph_ex` 必须 false。

### 3.2 昇腾 910B（Atlas 800I A2/A3）

- 权重：`Eco-Tech/Qwen3.8-27B-w8a8`（约 30GB）
- 镜像：`quay.io/ascend/vllm-ascend:qwen3.8-a2` / `qwen3.8-a3`
- 启动：TP=2、`--speculative-config '{"method":"qwen3_5_mtp","num_speculative_tokens":3}'`、`--dtype bfloat16`、`--max-model-len 133120`
- 官方口径 TP=2；同代 27B 在 A2 上通常 30–60 tok/s 量级

### 3.3 昇腾 950PR/950DT（预算允许时的最优昇腾路径）

- 镜像 `quay.io/ascend/vllm-ascend:qwen3.8-a5`，权重 `Eco-Tech/Qwen3.8-27B-w8a8c8-mxfp8`
- 单流 **64 tok/s（MTP 开启）**，TP=1 + DP

### 3.4 消费级 GPU（性价比最高）

| 硬件 | 可行性 | 权重 | 实测 |
|---|---|---|---|
| **2×RTX 3090/4090（48GB）** | ✅ 推荐 | `cyankiwi/Qwen3.8-27B-AWQ-INT4`（21GB） | vLLM TP=2：**67.0 tok/s @C1**，C4 236.8 tok/s（MTP off） |
| 1×RTX PRO 6000 96GB | ✅ | FP8 | **43→78 tok/s（MTP K=3，1.8×）**，TPOT 22→13ms |
| 1×RTX 5090 32GB | ⚠️ 勉强 | NVFP4 24.6GiB | 必须 `--enforce-eager`，`--max-model-len 32768` |
| 1×24GB（3090/4090） | ❌ AWQ 放不下；⚠️ GGUF 可跑 | GGUF Q4_K_M 17GB | llama.cpp 能吃，慢 |

> MTP 在 NVIDIA 路径上是**真实有效**的（RTX PRO 6000 上 1.8×，acceptance 0.75）。

---

## 4. 310B 上的正确选型（实测/官方配方）

| 模型 | 规模 | 310B 现状 | 速度 |
|---|---|---|---|
| **Qwen3-1.7B / Qwen3-4B** | 标准 GQA | ✅ **MindIE 官方 OrangePi 配方**（ModelZoo `MindIE/LLM/Qwen3/Qwen3-*-OrangePi`），需 CANN 8.1.RC1 + MindIE ATB-Models 2.0.RC1 | 4B 约 3–8 tok/s |
| DeepSeek-R1-Distill-Qwen-1.5B | 标准 | ✅ 官方主推（MindSpore / MindIE / openKylin 都适配） | ~7 tok/s |
| **Gemma4 E2B** | 2B 混合滑窗 | ✅ 社区自研引擎实测 | **11.8 tok/s（Q4_0）** |
| **MiniCPM-V 4.6/4.7-1B** | 1.5B 多模态（底座 Qwen3.5-0.8B GDN） | ✅ 社区从零 C++/AscendC 引擎（`lvyufeng/minicpm-v-4.6-orangepi`），支持多切片 OCR / server | ~5-10 tok/s |
| **Ling-mini-2.0** | 16B 总 / 1.4B 激活，纯 GQA + MoE | 🟡 GGUF 可 CPU 跑；NPU 需自写 MoE 路由（1–2 人月） | CPU 预计 3–8 tok/s |
| Qwen3-8B / 14B / 32B、Qwen3.8-27B、Qwen3.5 全系 | — | ❌（8B 是极限且 <1 tok/s；GDN 系列无算子） | — |

**310B 选型公式**：`总参数 ≤16B 且激活 ≤2B 的稀疏 MoE`，或 `≤4B 的纯 GQA dense`；**任何线性注意力（GDN/KDA/MLA/Lightning）都是死刑**。

**带宽换算表**（按 28.8GB/s 实测）：
- 权重体积 ÷ 28.8GB/s = 每 token 最短耗时
- 2GB Q4 → 理论 14 tok/s（实测 11.8 ✓）
- 4GB Q4 → 理论 7 tok/s
- 14GB（27B INT4）→ 理论 2 tok/s（实际 ~1）

---

## 5. 自研 AscendC 跑 Qwen3.8-27B 值不值？

**不值。** 需要补的不是几个算子，而是一条 GDN 6 阶段链路 + 状态机（chunk_cumsum / scaled_dot_kkt / solve_tril / wy_fast / chunk_h / chunk_o）+ RGDR 解码 + conv1d + 门控融合 + 状态缓存 + 投机解码状态回滚 + INT4 融合 GEMM，外加视觉塔/MTP。

- 工作量：L1（llama.cpp + 自写 matmul）1–2 人月 → 0.2–0.5 tok/s；L3（vLLM-Ascend `_310b` 后端）6–12 人月，且不会被上游 merge
- 硬件对比：一块 300I Duo（¥1.1–1.36 万）直接 9–21 tok/s；一个资深昇腾算子工程师 1 人月成本 ≥ 一块卡

---

## 6. 参考链接

- vLLM-Ascend Qwen3.8-27B 教程：https://docs.vllm.ai/projects/ascend/en/latest/tutorials/models/Qwen3.8-27B.html
- vLLM-Ascend 投机解码指南：https://docs.vllm.ai/projects/ascend/en/latest/user_guide/feature_guide/speculative_decoding.html
- vLLM-Ascend FAQ（310B 未计划）：https://docs.vllm.ai/projects/ascend/en/latest/faqs.html
- 310P 生产实测（9.1 tok/s、MTP 净负+乱码回滚）：https://bbs.huaweicloud.com/blogs/488684
- 华为云《300I-DUO 部署 Qwen3.8-27B》：https://developer.huawei.com/home/forum/ascend/thread-02201225443478597649-1-1.html
- 310P GDN 实现：https://github.com/vllm-project/vllm-ascend/blob/main/vllm_ascend/_310p/ops/fla/gdn_310.py
- 310B MindIE Qwen3-4B 配方：https://gitee.com/ascend/ModelZoo-PyTorch/blob/master/MindIE/LLM/Qwen3/Qwen3-4B-OrangePi/README.md
- 310B Gemma4 E2B 11.8 tok/s 实测：https://www.hiascend.com/dev/forum/thread-02200219947507257174-1-1.html
- 310B MiniCPM-V 4.6 自研引擎：https://github.com/lvyufeng/minicpm-v-4.6-orangepi
- 310B 上 llama.cpp 折腾实录（算子缺失）：https://www.hiascend.com/dev/forum/thread-02141208198249001080-1-1.html
- 2×3090 部署 Qwen3.8-27B AWQ：https://news.qiniu.com/archives/1787048758096
- vLLM Qwen3.8-27B recipe：https://recipes.vllm.ai/Qwen/Qwen3.8-27B
