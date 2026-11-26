# OrangePi AI Pro 20T（Ascend 310B1）上的自研 AscendC 推理栈：Qwen3.8-27B + Moondream3

[![GitHub](https://img.shields.io/badge/GitHub-qwen38--27B--ascendc-black)](https://github.com/dong-12345/qwen38-27B-ascendc)
[![HuggingFace](https://img.shields.io/badge/&#129303;-Qwen3.8--27B--AscendC--W4A16-yellow)](https://huggingface.co/dong2542107715/Qwen3.8-27B-AscendC-W4A16)
[![HuggingFace](https://img.shields.io/badge/&#129303;-Moondream3--Ascend310B-orange)](https://huggingface.co/dong2542107715/moondream3-preview-ascend310b)

在 **OrangePi AI Pro 20T（Ascend 310B1，单 AI Core，CANN 8.0.0，23 GB 内存，~1.46 TFLOPS Cube）** 上，
用**纯 AscendC 手写算子 + 自研推理引擎**跑通两个 4bit（W4A16 group-32 非对称）模型，并做了系统性的性能优化：

| 模型 | 指标 | 优化前 | **优化后** | 加速 | 输出一致性 |
|---|---|---|---|---|---|
| **Qwen3.8-27B**（28B，48 层 GDN + 16 层全注意力） | **prefill**（17 token） | 212.8 s（12.52 s/token） | **37.1 s（2.18 s/token）** | **5.7x** | ✓ token 完全相同 |
| | **decode**（默认 gen2） | 14.28 s/token | 12.6 s/token | 1.1x | ✓ 逐位相同 |
| | **decode**（AX/Axpy-N） | 14.28 s/token | **3.60 s/token** | **4.0x** | ✓ token 完全相同 |
| **Moondream3**（VLM，W4A16 + MoE） | **decode** | 2.39 s/token | **0.21 s/token = 4.8 tok/s** | **11.4x** | ✓ token 完全相同 |

> 完整的前后对比、瓶颈定位方法、复现命令与踩坑记录：[docs/优化总结与复现.md](docs/优化总结与复现.md)

**两个关键结论**（都是板端实测推导出来的）：
1. **prefill 慢的根因是"逐 token 逐行 GEMV"** → 改成 layer-major 批量 + `dequant_w4`(AIV) + `aclnnMatmul`(Cube)
   两段式，单矩阵实测 **24.2x / 15.5x**，端到端 **5.7x**。
2. **decode 慢的根因不是带宽，而是"每输出行的向量指令条数"**（~48 条/行 × 65 ns ≈ 3.1 µs/行；10.2 GB/s 只用到 9%）
   → 权重离线重排成 **AX `[g][w8][n]`** + 沿 N 向量化的 **Axpy-N kernel**（无归约树），**4.0x / 11.4x**。
   物理地板：27B 每 token 15.73 GB ÷ 10.2 GB/s = 1.54 s/token（单核 0.65 tok/s）。

> 本项目**不依赖** MindIE / vLLM / Ascend C 的模型库，所有算子（含 Gated DeltaNet 全流程、部分 RoPE GQA 解码、
> 4bit 非对称权重 GEMV）都是自己用 AscendC 写的；推理引擎自己调度、自己管 NPU 显存与 KV/GDN 状态。

---

## 1. 结果速览（全部为板端实测）

| 项 | 结果 | 说明 |
|---|---|---|
| 1 层（调试子集） | 0.96 s/forward | 单层对拍用 |
| 41 层子集 | 0.110 tok/s（9.12 s/token） | 9.10 GB 权重，加载 51.5 s |
| 48 层子集 | 0.075 tok/s（13.31 s/token） | 10.48 GB 权重，加载 69 s |
| **64 层全量 decode（gen2，默认）** | **0.080 tok/s（12.6 s/token）** | 逐位一致基线 |
| **64 层全量 decode（AX/Axpy-N）** | **0.278 tok/s（3.60 s/token）** | token 与基线完全相同，**4.0x** |
| **64 层全量 prefill** | **0.46 tok/s（2.18 s/token）** | Cube 两段式批量，**5.7x** |
| **Cube 两段式单矩阵（M=32）** | 72.2 ms / 64.0 ms | 对向量 GEMV 24.2x / 15.5x |
| **Moondream3 decode（AX/Axpy-N）** | **0.21 s/token（4.8 tok/s）** | token 与基线完全相同，**11.4x** |
| AX/Axpy-N 单矩阵（K=5120,N=17408） | 8.4 ms（5.46x） | max_abs 0.0029，0/17408 超差 |


**语义正确性（同一引擎、同一权重）**

| Prompt | 生成（解码后） |
|---|---|
| `The capital of France is` | **`Paris`** |
| `你好，请用一句话介绍你自己` | **`你好，我是通义千问，一个由阿里巴巴集团…`** |

**三重独立验证**

1. 引擎 logits vs **独立 numpy 64 层参考**：逐位一致
   （单 token `475(4.41)` vs `475(4.408)`；2-token pos=1 `169571(5.00)` vs `169571(4.98)`）
2. 权重转换 vs **原始 AWQ safetensors**（独立 GPTQ 反量化）：点积 `corr = 0.99999957`
3. 14 个算子的 host 侧对拍：单算子 `max_abs ≤ 2e-3`；W4A16 GEMV 真实权重 **1.43 GB/s**

---

## 2. 硬件与模型

| 项 | 值 |
|---|---|
| SoC | Ascend 310B1（OrangePi AI Pro 20T），`ai_core_cnt=1 / vector_core_cnt=1` |
| CANN | 8.0.0（`ascendc_kernel_cmake` + `ascendc_library`） |
| 实测上限 | Cube 1.46 TFLOP/s；GM 读 10.2 / 写 10.5 GB/s；L2 27 GB/s；UB 248 KB / L1 1 MB / L2 4 MB |
| 内存 | 23 GB（可用 ~21 GB 分配给 NPU 侧权重） |

模型（`Qwen3_5ForConditionalGeneration` 的语言塔，文本推理）：

| 参数 | 值 |
|---|---|
| hidden / FF / layers | 5120 / 17408 / 64 |
| 层类型 | 48 层 `linear_attention`（Gated DeltaNet）+ 16 层 `full_attention`（每 4 层 1 个） |
| GDN | 16 k-heads × 48 v-heads，head_dim 128，conv kernel 4，`mamba_ssm_dtype=float32` |
| Attention | 24 q-heads / 4 kv-heads，head_dim 256，partial RoPE 0.25（64 维），**输出门 = sigmoid** |
| vocab | 248320，`tie_word_embeddings=false` |
| 量化 | AWQ 4bit，group_size 32（compressed-tensors `pack-quantized`） |
| 其他 | `rms_norm_eps=1e-6`，mrope sections `[11,11,10]`（纯文本退化为标准 RoPE），含 `<think>` 推理模板 |

---

## 3. 我们做了什么

### 3.1 自研 AscendC 算子（14 个，详见 [docs/OPERATORS.md](docs/OPERATORS.md)）

- **解码主路径**：`w4a16_gemv_gen`（W4A16 非对称 GEMV，K≤17408）、`fp16_gemv`、`rmsnorm_op`、
  `conv1d_decode`（depthwise causal conv + SiLU + 卷积状态）、`gdn_l2norm`、`gdn_rgdr`（recurrent gated delta rule）、
  `rmsnorm_gated`、`mul_silu`、`full_attn_decode`（GQA + partial RoPE + 输出门 + KV cache）、`vec_axpy`
- **GDN chunked prefill 六阶段**：`gdn_gating`、`gdn_conv1d_silu`、`gdn_cumsum`、`gdn_kkt`（scaled_dot_kkt）、
  `gdn_tril`（solve_tril）、`gdn_wy`（wy_fast）、`gdn_chunk_h`、`gdn_chunk_o`
- 全部通过 host 侧对拍（CPU numpy 参考），并对 UB 溢出、32B 对齐、`DataCopy` 长度约束、
  GM 标量读写等 310B 限制做了专门处理（踩坑记录见 [docs/OPERATORS.md](docs/OPERATORS.md)）

### 3.2 自研推理引擎

- 权重**流式加载**（16.9 GB）到 NPU，按层复用 20+ 个激活/工作缓冲（总 ~2 MB）
- 自己实现 embed 行解量化、final norm、lm_head（248320×5120 W4 GEMV）、argmax
- KV cache / GDN recurrent state / conv state 自己管理，支持 prefill→decode 连续多 token
- 单 stream 全异步提交，每层约 20 个 kernel 启动

### 3.3 权重转换工具链

`tools/convert_qwen38.py`：AWQ（compressed-tensors pack-quantized，group 32）
→ 本项目自定义 **W4A16 相位交织格式**（`uint16` word 存 4 个相位）+ fp16 `scale` / `scale*zero_point`，
并生成 `meta.txt`（扁平文本，引擎直接读）。转换误差：**group-32 重量化 0.0000–0.0002**。

---

## 4. 快速开始

### 4.1 板端编译

```bash
# 依赖：CANN 8.0.0（/usr/local/Ascend/ascend-toolkit/latest）
cd code/qwen38_310b/ops
bash run.sh build                     # 产物：out/lib、qwen38_ops_test
source /usr/local/Ascend/ascend-toolkit/latest/bin/setenv.bash
export LD_LIBRARY_PATH=$PWD/out/lib:$PWD/out/lib64:/usr/local/Ascend/ascend-toolkit/latest/lib64:$LD_LIBRARY_PATH
```

### 4.2 算子单测

```bash
./qwen38_ops_test w4a16_gemv_gen     # W4A16 GEMV（真实权重，N=10240）
./qwen38_ops_test dense_ops          # conv1d/rmsnorm/l2norm/mul_silu
./qwen38_ops_test gdn_rgdr           # GDN 递推解码
./qwen38_ops_test full_attn_decode   # GQA 解码（24/4 heads, D=256）
```

### 4.3 权重转换（PC 侧，需要原始 AWQ 权重）

```bash
python tools/convert_qwen38.py \
  --src /path/to/Qwen3.8-27B-AWQ-INT4 \
  --out converted/qwen38_310b \
  --group 32 --requant-extra embed_tokens,lm_head
python tools/export_engine_meta.py --dir converted/qwen38_310b
```

转换出的权重目录需整体拷到板端（本项目的板端布局：`~/code/qwen38_310B/models/qwen38_310b`，16.9 GB）。

### 4.4 跑模型

```bash
# 板端（目录布局见 docs/BOARD_LAYOUT.md）
cd ~/code/qwen38_310B && source env.sh
ENG27_GEN=6 ./src/out/bin/qwen38_ops_test engine_27b
# 指定 prompt（PC 侧，自动分词 + ssh + 解码；已内置 chat template + <think>）
python tools/run_27b_demo.py --prompt "The capital of France is" --gen 6
```

常用环境变量：`ENG27_IDS`（prompt token id，逗号分隔）、`ENG27_GEN`、`ENG27_NOWARMUP`、
`ENG27_TRACE`（打印每 token logits top-5）、`ENG27_DUMP*`（调试 dump）。

---

## 5. 核心技术点（为什么能跑起来）

1. **W4A16 相位交织格式**：`uint16` word 一次存 4 个相位（`q[j], q[j+KQ], q[j+2KQ], q[j+3KQ]`），
   让一条向量指令同时处理 4 个 group，配合 `Gather` 展开 group scale，达成 **1.43 GB/s**（占 GM 读带宽 14%
   的瓶颈下，已接近 310B 单核向量能力的实际上限）。
2. **GDN 用递推形式解码**：单 token 时 `S = S·exp(g)`；`bv = (v - S·k)·β`；`S += k⊗bv`；`o = q·S`，
   状态 128×128 fp32 常驻 UB（64 KB），避免 chunked 算法的多次 GM 往返。
3. **部分 RoPE + GQA 解码**：24 q-heads / 4 kv-heads、head_dim 256 只转前 64 维、softmax 按 `KC=64` 分块，
   KV cache 用 `DataCopy` 向量写回（**不能用 GM 标量 `SetValue`**，这是 310B 的坑）。
4. **1-centered RMSNorm**：Qwen3.5 的 `Qwen3_5RMSNorm` 是 `out = norm(x) * (1 + w)`（权重存偏移量），
   这是一个**极易踩中的静默错误** —— 直接用 `w` 会让整网归一化少一倍多增益，输出变成乱码。
5. **q_proj 输出是 per-head 交错** `[q_h(256) | gate_h(256)] × 24`，不是 `[q_all | gate_all]`。
6. **注意力输出门是 `sigmoid(gate)`**，不是 SwiGLU 风格的 `silu(gate)`。

---

## 6. 仓库结构

```
code/qwen38_310b/ops/
  kernels/          27 个 AscendC kernel 源文件（核心 14 个 + bench/probe 工具）
  host/             engine_27b.cpp（27B 引擎）、engine_mini.cpp、main.cpp（单测调度）、data_utils.h
  cmake/ scripts/   构建与数据生成
  CMakeLists.txt run.sh
tools/
  convert_qwen38.py     AWQ → 本项目 W4 格式（含 1-centered norm 修正）
  export_engine_meta.py 生成引擎 meta.txt
  run_27b_demo.py      端到端 demo（分词 + ssh + 解码）
  make_layer_subset.py / make_smoke_subset.py  调试用子集
  push_to_hf.py        转换权重上传 HuggingFace
  verify/              独立验证脚本（numpy 全模型参考、转换校验、1-centered 修复…）
docs/
  OPERATORS.md       ★ 自研算子详解（本文档的同级重点）
  ARCHITECTURE.md    ★ 我们怎么把 Qwen3.5 混合架构适配到 310B
  MODEL_CARD_HF.md   ★ 转换权重推 HF 用的 model card
  27B引擎与算子结果.md   实测结果汇总
  AscendC算子开发进度.md 开发日志（逐算子、逐 bug）
  310B-性能上限与prefill优化.md  硬件上限实测与优化方向
code/                其他支线（ToF U3V 直连、深度补全）
```

---

## 7. 已知限制 / 下一步

- **性能定位（2026-11-24 实测，详见 [docs/GEMV性能分析与优化路线.md](docs/GEMV性能分析与优化路线.md)）**：
  14.62 s/token 里 **≈100% 是自研 W4A16 GEMV**。结论是**指令条数受限，不是带宽**：
  单核 GM 读实测 10.2 GB/s，13.53 GB 权重的带宽地板是 **1.32 s/token**，当前只用掉 9%；
  旧 kernel 每输出行 ~48 条向量指令，实测 **3.2 µs/行 ≈ 48 × 67 ns**（每条指令固定开销 ~70 ns，
  用新增的 `kernels/probe_v2.cpp` 探针测得）。所以提速的关键是**减少每行的指令条数**。
- **已做的尝试**：行批处理实验内核 `kernels/w4a16_gemv_batch.cpp` 把指令数降到 ~10/行，
  实测 **55.6 → 34.5 ms（1.61x，0.80 → 1.29 GB/s）**；但数值尚未与 `w4a16_gemv_gen` 等价，
  因此**默认关闭**（`ENG27_GEMV_V2=1` 才启用），仅作性能研究基线。
- **真正的 5–10x 路线（未做，2–3 周）**：Cube 融合反量化 MMAD
  （GM int4 → UB 反量化 → L1 NZ → `Mmad` → `Fixpipe`），流量保持 0.5 B/param，
  预期 decode 2–3 s/token、prefill（M=64 批）~0.2 s/token；可行性依据见
  [docs/310B-性能上限与prefill优化.md](docs/310B-性能上限与prefill优化.md) §6.3。
- **prefill 与 decode 同速**（都是逐 token 走同一路径，14.66 s/token）；
  按 token 批处理的设计与内核草案见 `kernels/w4a16_gemv_tok.cpp` +
  [docs/prefill-token批量GEMV设计.md](docs/prefill-token批量GEMV设计.md)（未接线）。
- **首调异常**：`w4a16_gemv_gen` 在进程内首次调用结果异常，引擎用 `ENG27_DOUBLE_GEMV`（先跑一次丢弃）
  或 warmup 前向规避；根因未完全定位。
- **MTP**：`mtp_num_hidden_layers=1` 未接入（不影响基础生成）。
- **VL 输入**：视觉/音频塔未接入，只做文本推理。

---

## 8. License

本仓库代码（`code/qwen38_310b/ops/`、`tools/`）以 **Apache License 2.0** 发布。

上游模型 `Qwen3.8-27B` 的 LICENSE 实测为 **Apache License 2.0**
（Copyright 2026 Alibaba Cloud，见原始权重目录的 LICENSE）。Apache-2.0 **允许再分发与修改**，
转换后的权重同样以 Apache-2.0 分发，并履行 §4 义务：

| 义务 | 落实方式 |
|---|---|
| 提供 License 副本 | 权重目录内 `LICENSE` |
| 声明修改 | 权重目录内 `MODIFICATIONS.md`（重打包格式 / bf16→fp16 / 1-centered norm 修正 / 只保留语言塔） |
| 保留归属 | model card 与 `MODIFICATIONS.md` 保留 "Copyright 2026 Alibaba Cloud" |
| 商标 | 仅描述性提及 "Qwen"，不暗示官方背书 |

> ⚠️ 转换后的权重是**实验性格式**（仅自研引擎可用）；需要通用格式请使用官方原始权重。
> 商用或再分发前请自行复核上游许可条款与适用法律。

---

> 完整的前后对比数据、瓶颈分析与复现命令见 [docs/优化总结与复现.md](docs/优化总结与复现.md)。

> Moondream3 推荐用 AX 布局跑 decode（11.4x → 4.8 tok/s）：`MD3_LAYOUT=axpy ./md3_engine --dir .../moondream3_310b_ax --ids ... --gen 8`；Qwen 27B 的 AX 路径（4.0x）需 `ENG27_LAYOUT=axpy ENG27_DIR=.../qwen38_310b_ax`。
---

## Moondream3 优化（AX 布局，11.4x）

Moondream3（VLM，W4A16 + MoE，24 层）与 Qwen 共用同一套 W4A16 GEMV/AX 基础设施，收益更大：

| 场景 | 优化前 | 优化后 | 加速 | token |
|---|---|---|---|---|
| decode（默认 gen2） | 2.39 s/token | 2.00 s/token | 1.18x | ✓ |
| **decode（AX/Axpy-N，`MD3_LAYOUT=axpy`）** | 2.39 s/token | **0.21 s/token = 4.8 tok/s** | **11.4x** | ✓ 完全相同 |

```bash
# 一次性生成 AX 权重（约 1.5 分钟）
cd ~/code/qwen38_310B/src && ./out/bin/qwen38_ops_test relayout \
   ~/code/moondream3_310b/models/moondream3_310b ~/code/moondream3_310b/models/moondream3_310b_ax
# 跑 decode
cd ~/code/moondream3_310b/ops && MD3_LAYOUT=axpy ./md3_engine \
   --dir ~/code/moondream3_310b/models/moondream3_310b_ax --ids 0,1,15381,2,8299,1158,3 --gen 8
```

收益比 27B 大的原因：Moondream 的矩阵都很小（H=2048、专家 2048→2048），旧 GEMV 受"每行 ~48 条指令"的**下发开销**支配，
而 Axpy-N 的开销与字节数成正比。权重与模型卡：https://huggingface.co/dong2542107715/moondream3-preview-ascend310b