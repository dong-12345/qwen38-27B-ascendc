# Qwen3.8-27B（AWQ/W8A8）+ MTP 投机采样 部署方案

> 结论：**OrangePi AI Pro 20T（昇腾 310B4）无法运行 Qwen3.8-27B**，任何量化版本都不行。
> 本文件给出可行硬件（310P / 910B）上的完整部署路径，以及 310B 上的实测能力对照。

---

## 1. 为什么 310B 跑不了 Qwen3.8-27B

| 维度 | Qwen3.8-27B 的要求 | OrangePi AI Pro 20T（310B4）的实际能力 |
|---|---|---|
| 参数/权重 | 27.8B dense；AWQ INT4 ≈ **21 GB**、W8A8 ≈ 28 GB、BF16 ≈ 55 GB | NPU 可用内存 **23.6 GB**（与系统共享 LPDDR4X） |
| 架构 | 64 层混合：48× Gated DeltaNet 线性注意力 + 16× Gated Attention，3:1 循环；原生视觉编码器；MTP 多 token 预测头 | CANN 8.0 无 GDN/视觉/MTP 算子；vLLM-Ascend 为 GDN 写的 `RecurrentGatedDeltaRule` 自定义算子只有 **310P** 版本 |
| 引擎 | vLLM-Ascend（nightly，需 GDN+MTP 支持）/ SGLang-Ascend / MindIE | 以上全部只支持 310P / 910B 系列；310B 只有 ATC-OM 与手写 AscendC 引擎（llm_simple） |
| KV Cache | 262K 原生上下文，8K 上下文也要数 GB | 权重已占满内存，无空间 |
| 算力 | decode 为访存瓶颈，27B 每 token 需读 ~21GB 权重 | 310B 的 LPDDR 带宽约 51.2 GB/s，即使能装下也只能 ~1-2 tok/s |

**结论：310B 上不存在 Qwen3.8/3.5（Gated DeltaNet 混合架构）的可行部署路径；27B 更是内存与算子双重不可能。**

---

## 2. 硬件选择

| 方案 | 硬件 | 说明 |
|---|---|---|
| **A（推荐，最贴合官方）** | 昇腾 **310P**：Atlas 300I Duo（96GB / 192GB LPDDR5）或 AI Studio Pro | 官方已有 Qwen3.8-27B **W8A8-310P** 权重与 vLLM-Ascend 部署教程 |
| **B（性能最好）** | 昇腾 **910B**（Atlas 800I A2 等） | vLLM-Ascend / SGLang 均支持 MTP 投机解码，吞吐更高 |
| C（非昇腾） | 32GB+ 显存 GPU（RTX 5090/4090 48G 改等） | AWQ INT4 21GB + KV Cache 需 24GB 以上余量 |

---

## 3. 方案 A：310P + vLLM-Ascend + W8A8 + MTP

### 3.1 环境

```bash
# 镜像：官方 vLLM-Ascend（310P 需带 310p 标签的镜像；Qwen3.5/3.8 的 GDN+MTP 依赖 nightly）
# 注意：Qwen3.5/3.8 混合架构在正式版 v0.18.0 上无法服务，需要
#       nightly-releases-v0.22.1rc 或更新的 tag（该镜像曾出现 nightly-main 损坏构建）
export IMAGE_TAG=v0.22.1rc-310p      # 以官方当期 tag 为准
docker run --rm -it --privileged \
  --device=/dev/davinci0 --device=/dev/davinci_manager \
  --device=/dev/devmm_svm --device=/dev/hisi_hdc \
  -v /usr/local/dcmi:/usr/local/dcmi \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
  -v /opt/models:/models \
  quay.io/ascend/vllm-ascend:${IMAGE_TAG} bash
```

### 3.2 权重

- 基座：`Qwen/Qwen3.8-27B`（Apache-2.0，HuggingFace / ModelScope 同步）
- 量化：**Qwen3.8-27B-w8a8-310p**（官方 310P W8A8 量化权重）或自行用 msmodelslim 生成

### 3.3 启动（含 MTP 投机解码）

```bash
# 参考配置来自官方 300I-DUO 部署实践；MTP 权重与主模型同仓，无需单独草稿模型
python3 -m vllm.entrypoints.openai.api_server \
  --model /models/Qwen3.8-27B-w8a8-310p \
  --served-model-name qwen3.8-27b \
  --quantization ascend \
  --tensor-parallel-size 4 \
  --max-model-len 24576 \
  --max-num-batched-tokens 8192 \
  --max-num-seqs 32 \
  --gpu-memory-utilization 0.85 \
  --speculative-config '{"method": "qwen3_5_mtp", "num_speculative_tokens": 2}' \
  --trust-remote-code \
  --port 8000
```

要点（官方实践总结）：
- `--gpu-memory-utilization 0.85`：开 MTP 后用 0.90 在高并发时 OOM。
- `--max-num-seqs 32`：必须的安全上限。
- `--speculative-config` 的 `qwen3_5_mtp` 方法直接复用模型自带 MTP 头，是官方推荐的 decode 提速手段；`num_speculative_tokens` 建议从 2 起调。
- 若显存紧张，降 `max-model-len` 或 `num_speculative_tokens`。
- 视觉输入（图/视频）需引擎支持 mmproj 的对应路径，仅文本推理可直接跑。

### 3.4 验证

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "qwen3.8-27b",
  "messages": [{"role":"user","content":"用一句话介绍你自己"}],
  "max_tokens": 128
}'
```

用 vLLM 日志中的 `Speculative metrics: Draft acceptance rate / Acceptance length` 观察 MTP 效果；官方实测基本接受长度 1.5-2.5 token 时 decode 提升约 1.5-2×（具体随任务与 `num_speculative_tokens` 变化）。

---

## 4. 方案 B：910B

同样使用 vLLM-Ascend（或 SGLang-Ascend），模型可用 BF16 或 W8A8：

```bash
python3 -m vllm.entrypoints.openai.api_server \
  --model /models/Qwen3.8-27B \
  --served-model-name qwen3.8-27b \
  --tensor-parallel-size 4 \
  --max-model-len 65536 \
  --max-num-seqs 64 \
  --speculative-config '{"method": "qwen3_5_mtp", "num_speculative_tokens": 3}' \
  --trust-remote-code --port 8000
```

> 910 上也可用 AWQ INT4 权重（21GB）单卡放下，但 decode 主要受 HBM 带宽限制，W8A8 + MTP 通常是更好的组合。

---

## 5. 310B 上的实测能力对照（供选型参考）

310B 上唯一"快且可用"的引擎是 **llm_simple**（手写 AscendC + AWQ 4bit，免 ATC 转换），同款 20T 24GB 实测：

| 模型 | decode | 吞吐 | 权重 |
|---|---|---|---|
| Qwen2.5-3B-Instruct-AWQ | 113 ms/token | **8.8 tok/s** | ~2 GB |
| Qwen2.5-7B-Instruct-AWQ | 206 ms/token | **4.85 tok/s** | ~4.7 GB |
| Qwen2.5-14B-Instruct-AWQ | 653 ms/token | 1.53 tok/s | ~9 GB |
| Qwen2.5-32B-Instruct-AWQ | ~1.3 s/token（推算） | ~0.7 tok/s | ~20 GB |
| Qwen3.8-27B（任何量化） | — | **不可用** | — |
| Qwen3.5/3.6/3.8 全系（GDN 混合） | — | **不可用** | — |

> 说明：310B 上没有 MTP 头可用，也没有现成的投机解码实现。若必须在 310B 上提速，
> 可行方向是自研 draft-model 投机解码（小模型草稿 + 大模型验证）或继续优化 AscendC kernel，
> 收益受 LPDDR 带宽限制，通常 1.5-2× 封顶。
