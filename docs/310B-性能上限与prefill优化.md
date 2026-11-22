# 310B（OrangePi AI Pro 20T）性能上限与 prefill 提速方案

> 📌 **目录变更说明（2026-11-22）**：本文记录的是当时使用的板端路径 `~/qwen38_310b/ascendc_ops`、`~/models/qwen38_310b`。
> 现在板端已统一整理为 `~/code/qwen38_310B/`（`src/` 源码 + `models/qwen38_310b/` 权重），见 [BOARD_LAYOUT.md](BOARD_LAYOUT.md)。

> 目标：回答三个问题——① 硬件理论值是多少、当前实现离理论极限多远；② `blockDim>1` 到底有没有多核并行；
> ③ prefill 还能怎么提速（按加速比排序，含工作量与风险）。
>
> 文档日期：2026-11-19（第一版 perf-scout2；**晚 23:00-23:40 perf-scout 复核并定稿**）
> 数据状态：`【历史实测】`= 2026-11-18/19 前置工作已测；`【本次实测】`= 2026-11-19 23:00 前后板端干净复测
> （原始日志 `tmp/perf_clean.log`，7 个测试完整输出）；`【官方】`= 厂商手册/平台配置；`【推算】`= 由上述数据代入模型算出。
>
> **板端复测状态：已完成**。板子（192.168.137.163，Ascend 310B1、20T 档、CANN 8.0.0、24GB）
> 在 2026-11-19 23:00-23:20 独占窗口内跑完 §5/§6 全部实验，结果见 §10；§1.3/§3/§5/§6 中
> 原先标"待实测"的项均已回填，正文与 §10 数据一致。
>
> 本次定稿的**三个新增/修正要点**（比第一版重要）：
> ① **向量峰值口径修正**：理论 156.7 Gelem-ops/s（128 elem/cyc × 1.224GHz），实测 134（简单 op，85%）、68.7（复合 op）；
> ② **Cube 实测效率只有 10–15%**：高层 Matmul API 在本板最高仅 1.0–1.5 TFLOP/s（理论 10 TFLOPS），且
>    M>128 后权重矩阵 B 会按 M/128 被**重复读**（tiling 为 `sM=M, base=128x256x64, depthB1=4`，无公开旋钮可修，
>    `SetTraverse/SetFixSplit/SetDoubleBuffer` 实测最多改善 2.6%）→ 这使"prefill 上限 186 tok/s"必须改口径（§2.2）；
> ③ **10.2GB/s 已证实是单 AI Core 硬墙**：tile 从 0.5KB 扫到 112KB、读/写/读写三种方向都撞在 10–10.5GB/s，
>    而 L2（4MB 内）能到 26–27GB/s。

---

## 0. 三句话结论（本次定稿已按实测修正）

1. **这块板子是"单核"的：1 个 DaVinciV300 AI Core + 1 个 Vector Core（`ai_core_cnt=1, vector_core_cnt=1`，官方 `Ascend310B1.ini`、`aclGetDeviceCapability`、
   以及本次 `bench_mc/bench_scale` 的 aggregate 吞吐不随 blockDim 放大，三重证据）。多核并行不存在，`blockDim>1` 只会串行排队；
   唯一能"并行"的是 Cube 与 Vector 两个执行单元（`cube_vector_combine=fuse,split`）。**
2. **10.2GB/s 是单 AI Core 的硬 GM 端口墙（理论 51.2GB/s 的 20%，但已是单核可达的 100%）**，tile 0.5KB~112KB、读/写/读写都撞墙；
   **0.6 TFLOP/s（M=64）不是"算力只用了 6%"的低效，而是该形状带宽受限（92% 贴着 10.2GB/s）**；但要注意：**给足算术强度后高层 Matmul API 也只到 1.46 TFLOPS（10 TFLOPS 的 14.6%）**——
   这是本次新发现的工具链瓶颈，也是 prefill 融合路径的真正上限所在（§2.2）。
3. **prefill 提速按优先级：P0 融合反量化（4.5→0.5 B/param，12s→2.5s/pass，≈5×）→ P0.5 手写 matmul 修 Cube 效率（1.4→4.6+ TFLOPS，
   让 M 放大真正免费，M=512 时 86 tok/s）→ P3 GDN chunk 优化（5–7×）→ P1 M 放大（需 P0.5 配合）。做完 P0+P0.5+P3 后 prefill 可达 ~53 tok/s（当前 4.3 的 12×）；
   硬件算力墙 186 tok/s 属于"知道方向但当前工具链达不到"的数值。明确不可行：多核（不存在）、CPU 协同反量化（实测 3.8GB/s 太慢）、W8A8（27GB 装不下）。**

---

## 1. 硬件事实与官方规格

### 1.1 芯片型号：是 310B1（20T），不是 310B4

| 来源 | 结论 |
|---|---|
| 华为论坛官方答复 | "20T 的是昇腾 310B1，8/12T 的是 310B4"（[hiascend 论坛](https://www.hiascend.com/dev/forum/thread-0261156769253689371-1-1.html)） |
| Ascend 310B `npu-smi` 命令参考 | `Name` 字段：310B1=20T，310B4=8T |
| 本板 `npu-smi info`（部署记录） | 显示 **310B1** |
| `platform_config/Ascend310B1.ini` | `cube_freq=1224`、`ai_core_cnt=1`；`Ascend310B4.ini` 为 `cube_freq=500`、`ai_core_cnt=1`，其余规格相同 |

> 结论：任务描述里的"310B4"应为笔误；**本板是 310B1（20 TOPS 档）**，以下所有理论值按 310B1 计算。
> 两档 SoC 的差别只在主频：B1 的 cube 1224MHz（20T），B4 的 cube 500MHz（8T）。

### 1.2 官方规格表（可用于"理论 vs 现状"对照）

| 项目 | 数值 | 来源 |
|---|---|---|
| AI Core（DaVinciV300） | **1 个**，cube 主频 **1224 MHz** | `Ascend310B1.ini` `ai_core_cnt=1, cube_freq=1224`；OrangePi 20T 规格页"1 DaVinciV300 AI core * 1.224GHz" |
| Vector Core | **1 个**，`vec_calc_size=128`（= **128 个 fp16/cycle**，即 vector_reg_width 256B） | `Ascend310B1.ini`（实测佐证见 §1.3） |
| INT8 算力 | **20 TOPS** | 用户手册 v1.1 §1.2；OrangePi 规格页；Atlas 200I A2 规格 |
| FP16 算力 | **10 TFLOPS** | 同上 |
| Cube 基本块 | **16×16×16**（fp16）；INT8 为 16×**32**×16；INT4 为 16×**64**×16 | `Ascend310B1.ini` `[AICoreSpec]` / `[DtypeMKN]` |
| UB | **253952 B ≈ 248 KB** | `Ascend310B1.ini` `ub_size` |
| L1 | **1 MB** | `l1_size=1048576` |
| L0A / L0B / L0C | **64 KB / 64 KB / 128 KB** | `l0_a_size=65536, l0_b_size=65536, l0_c_size=131072` |
| L2 | **4 MB** | `l2_size=4194304`（旧结论正确） |
| DDR | LPDDR4X-4266；SoC 接口 **96-bit** → 理论 **51.2 GB/s**（20T 满配；8T 档为 64-bit） | 海思 Ascend 310B 硬件指南（"96 bits 的 LPDDR4X 接口"）；Atlas 200I A2 20TOPS 12GB 规格"位宽 96bit / 总带宽 51.2GB/s"；香橙派 20T 内存位宽 96-bit/51.2GB/s |
| DDR 实测（整机，非 AI Core 路径） | 参考 **≈45.5 GB/s**（`ascend-dmi --bw -t d2d` 官方参考输出，**本板未安装该工具、未验证**；CPU 侧实测只有 1.6–3.8GB/s，见 §10.5） | Ascend 310B / Atlas 200I 文档 |
| 板载内存 | 24 GB（与 CPU 共享，NPU 可用 ~23.6GB） | 用户手册/部署记录 |
| CANN 平台建模带宽 | `ddr_rate=17`、`l2_rate=256`、`l1_to_l0_a_rate=512`、`l1_to_l0_b_rate=256`、`ub_to_l1_rate=128`、`vec_calc_size=128` | `Ascend310B1.ini` `[AICoreMemoryRates]`（单位未公开，CANN tiling 代价模型用；DDR:L2 = 1:15，说明官方模型也认为 DDR 是最大瓶颈） |

### 1.3 由官方规格推导的理论峰值

| 指标 | 公式 | 理论值 |
|---|---|---|
| FP16 Cube | 16×16×16 MAC/cycle × 2 FLOP × 1.224 GHz | **10.03 TFLOPS**（= 5.01 TMAC/s） |
| INT8 Cube | 16×32×16 × 2 ops × 1.224 GHz | **20.05 TOPS** |
| INT4 Cube（W4A4） | 16×64×16 × 2 ops × 1.224 GHz | **40.1 TOPS**（精度不可用，见 §6.5） |
| Vector | `vec_calc_size=128` = **128 个 fp16/cycle** × 1.224 GHz | **156.7 Gelem-ops/s**（简单 Add/Mul；实测 134，见 §9） |
| DDR 理论 | 4266 MT/s × 96 bit / 8 | **51.2 GB/s**（64bit 则 34.1） |
| L2 理论 | `l2_rate=256`（平台模型按 B/cycle 解释） | ~313 GB/s（模型值；**实测 26–27 GB/s**，见 §9） |
| 单 AI Core GM 读带宽 | 实测 | **10.2 GB/s**【本次实测复核，tile 0.5KB~112KB 全程 9.6–10.5】 |

> 注 1：INT8/INT4 的 N 方向 fractal 更大，这是"同频下 INT8 是 FP16 两倍、INT4 是四倍"的硬件原因。
> 注 2（**重要，第一版写错**）：Vector 的 `vec_calc_size=128` 是"每 cycle 128 元素"，不是 128 字节。
> 证明：实测简单 Add 吞吐 134 Gelem-ops/s，若按 64 fp16/cycle 则理论只有 78.3 Gelem-ops/s，**实测值超出该理论上限 71%**，
> 故正确口径是 128 元素/cycle → 理论 156.7 Gelem-ops/s，实测 134 = 85.5%。
> 注 3（**新增，改变 prefill 结论**）：Cube 的 10 TFLOPS 是**硬件峰值**，但本板经由 CANN 高层 Matmul API 只实测到
> **1.0–1.5 TFLOP/s（10–15%）**，且与 M 放大不友好，详见 §2.2 与 §10.4。
>
> **【明确回答 A 部分】"当前值 = 理论值的百分之几"**
> - **Cube：0.60 TFLOP/s ÷ 10.03 TFLOPS = 6.0%**（但该形状的带宽墙上限只有 0.65 TFLOPS，所以它同时是"可达上限的 92%"）；
> - **带宽：10.2 GB/s ÷ 51.2 GB/s = 19.9%**（但这 10.2 已是**单 AI Core GM 端口可达的 ~100%**：写 10.5、读写 9.6；
>   整机 DDR 的 51.2GB/s 由 CPU+SDMA 等多主体共享，单个 AI Core 拿不到）；
> - 换句话：**带宽是"硬件端口墙"（19.9% 理论 / 100% 可达），算力是"该形状带宽受限"而非端口墙**；
>   真正离理论最远的是 **Cube 的实际达成率（6%~14.6%）**，它由"权重带宽 10.2GB/s"与"Matmul API 只有 1.4 TFLOPS"两个原因共同决定。

---

## 2. 理论极限模型（决定 tok/s 上限的公式）

设模型参数量 `P`，权重位宽 `w` 字节/参数（int4 = 0.5），一次 forward 搬运 `W = P·w` 字节；
prefill 一次处理 `M` 个 token，算力 `C`（FLOP/s），带宽 `B`（B/s）。

```
t_pass = max( W/B , 2·P·M/C , dequant_ops/R_vec )          # 三者取最大（流水重叠）
tok/s_prefill = M / t_pass
tok/s_decode  = 1 / max( W/B , 2·P/C )  ≈ B/W               # M=1 时几乎纯带宽
```

**Qwen3.8-27B**：`P ≈ 27G` 参数、W4A16 权重 `W ≈ 13.5 GB`（历史文档记 ~14GB）、每 token `2P ≈ 54 GFLOP`。

| 场景 | 公式 | 数值 |
|---|---|---|
| **prefill 理论上限（算力墙）** | `C/(2P) = 10.03e12/54e9` | **≈ 186 token/s**（与带宽无关，因为 M 可以调到最优） |
| prefill 最优 M | `M* = B·C/(2P) / ... ` 即 `M* = (W/B)·C/(2P)` | 实测带宽 10.2GB/s → **M*≈244**；51.2GB/s → M*≈48 |
| prefill 带宽墙 @M=64 | `W/B` | 10.2GB/s → 1.32s/pass = **20.7 ms/token**；51.2 → 4.1 ms/token |
| **decode 带宽墙** | `W/B` | 10.2GB/s → **1.32 s/token = 0.76 tok/s**；51.2GB/s → 0.26s = 3.8 tok/s |
| decode 算力墙 | `2P/C` | 5.4 ms/token（远小于带宽墙，可忽略） |
| **两段式（现方案）每 pass 流量** | `W_int4 + W_fp16(写) + W_fp16(读) = 0.5+2+2 = 4.5 B/param` | **121.5 GB/pass** → 10.2GB/s 下 **11.9 s/pass**（实测 12s ✓） |
| 两段式地板（M→∞） | 121.5GB/B | 仍 11.9 s/pass —— **放大 M 只能摊薄，不能突破** |
| **融合反量化每 pass 流量** | `0.5 B/param` | **13.5 GB/pass** → 1.32s/pass；M=244 时为 5.4ms/token |
| 融合路径 cube 计算 | `2PM/C` | M=244 → 1.32s/pass（与带宽墙平衡） |
| 融合路径反量化向量开销 | 约 2 elem-op/权重 → `2P/R_vec` | `54e9/69e9` → **0.78 s/pass**（若 AIV/AIC 并行则被带宽盖住） |

> **关键推论**：prefill 的天花板 ≈ 186 tok/s（算力墙），且**只要 M 足够大，带宽不再是墙**；
> decode 的天花板 = 带宽墙 = 0.76 tok/s（当前实现 0.19 tok/s，见 §3）。
> 现方案 prefill 的 187ms/token 中有 99% 是"多搬了 8 倍字节"（4.5 vs 0.5 B/param）。
>
> **口径说明**：上面按任务口径"27G 参数全部 int4 = 13.5GB"计算。若 embed/lm_head/vision 保持 fp16
> （历史文档的 19–20GB 口径），decode 每 token 流量 +40%（→ decode 墙 0.55 tok/s），
> 两段式的 dequant 流量只作用于 23.9G 的 GEMM 权重（≈108GB → 10.6s），与实测 12s 同量级。
> 结论方向不变，数值按比例缩放。

### 2.1 两堵"带宽墙"必须分开：整机 DDR 45 GB/s ≠ 单 AI Core 能拿到

| 带宽口径 | 数值 | 依据 | 用途 |
|---|---|---|---|
| LPDDR4X-4266×96bit 理论 | **51.2 GB/s** | 官方规格 | 硬件上限 |
| 整机 d2d 实测 | **≈45.5 GB/s** | 官方 `ascend-dmi --bw -t d2d` 参考输出（310B/Atlas 200I A2） | SDMA/CPU 等"整机路径"可达 |
| **单 AI Core GM 读** | **10.2 GB/s** | 本仓库 `bench_stream` 实测 | **本推理引擎真正能用的带宽墙** |
| 单 AI Core GM 读 / 整机 | **22%** | 上两行 | 说明瓶颈可能在"单核 MTE2 端口/流水深度"，而非 DDR 颗粒 |

**结论**：
- 本引擎（单 AI Core 执行）的带宽墙按 **10.2 GB/s** 建模；
- 整机 45.5GB/s 的带宽只有"另一个引擎"（SDMA/多卡路径）才可能用得上——**CPU 侧实测只有 1.6–3.8GB/s（§10.5），
  所以"CPU 协同反量化"这条过渡方案已被实测否决（§6.7），不要再排期。**
- 但若 tile 细扫发现单核能到 20–40GB/s（即 10.2 是 kernel 效率问题），上面所有带宽墙按比例放宽，方案排序不变（P0 仍第一）。
  **→ 已实测排除（2026-11-19 23:00，§10.3）：tile 0.5KB/2KB/8KB/16KB/32KB/64KB/112KB × 读/写/读写，
  饱和后全部落在 9.6–10.5 GB/s；写 10.2–10.5、读 9.9–10.0、读写混合 9.6–9.7。10.2GB/s 是硬墙。**

### 2.2 （本次新增，修正第一版结论）Cube 的"可达算力"只有 1.0–1.5 TFLOPS，186 tok/s 是"硬件墙"不是"可达墙"

第一版 §2 的 186 tok/s 直接用了 10 TFLOPS 峰值。本次实测（§10.4）发现，**经由 CANN 高层 Matmul API，本板 Cube 只能跑到
1.0–1.5 TFLOP/s（峰值的 10–15%），并且 M 放大时权重 B 会被重复读**，这会把 prefill 的真实天花板压到 ~26 tok/s：

| M | N=K=5120 GEMM 实测 | 权重流量 | 折算 B 重读次数 | 说明 |
|---|---|---|---|---|
| 64 | 5.59 ms | 9.38 GB/s | 1× | 已贴着 10.2GB/s 带宽墙（**这一档是健康的**） |
| 128 | 7.50 ms | 7.01 GB/s | 1× | 开始掉 |
| 256 | 13.76 ms | 3.87 GB/s | 2× | 时间≈2× |
| 512 | 28.57 ms | 1.91 GB/s | 4× | 时间≈4× |
| 1024 | 53.0 ms | 0.99 GB/s | 8× | 时间≈8× |

诊断工具 `ascend_ops/scripts/probe_tiling.cpp`（独立 g++ 编译，可复跑）打印出的 tiling：
`sM=M, sN=5120, sK=5120, base=128x256x64, depthA1=40, depthB1=4, stepKa=20, iterOrder=0`
→ 输出按 baseM=128 分块（M/128 块），**每个 M 块都会把 B（K×N=52.4MB）从 GM 重读一遍**，
所以 M 放大后时间 ∝ M/128，而不是"权重只读一遍"。

尝试过的修复旋钮（同一 probe 程序，M=512 基准 27.5ms）：
`SetTraverse(FIRSTN)` 26.79ms（-2.6%，唯一有效）、`FIRSTM` 27.53ms、`SetDoubleBuffer` 27.55ms、
`SetFixSplit(128x128x64)` 32.82ms、`SetFixSplit(64x512x64)` 44.32ms —— **公开 API 没有能修这个问题的旋钮**。
理论最优（一次读 B + 大 base 复用）在 M=512 时权重 52.4MB/10.2GB/s≈5.1ms，即 **~5.6× 的 cube 侧空间**。

**修正后的 prefill 时间模型**（融合反量化路径，权重 13.5GB、FLOPs=2·P·M）：

```
t_pass ≈ max( 13.5GB × ceil(M/128) / 10.2GB/s ,  2·P·M / C_eff )
C_eff(高层 API) ≈ 1.0–1.5 TFLOPS（M≥128）;  若手写 matmul 使 B 只读一遍，C_eff 可到
   min(10 TFLOPS, 10.2GB/s × M·N/(M+N))  —— M=512 时约 4.6 TFLOPS
```

| 方案 | M=64 | M=245 | M=512 | M=1024 |
|---|---|---|---|---|
| 融合反量化 + 现 API（1.4 TFLOPS，B 重读） | max(1.32, 2.5)=2.5s → **25.6 tok/s** | max(2.64, 9.4)=9.4s → 26 tok/s | max(5.3, 18.8)=18.8s → 27 tok/s | 37s → 28 tok/s |
| 融合反量化 + 手写 matmul（B 只读一遍） | **1.32s → 48 tok/s**（带宽墙） | 1.32×2=2.64s / 计算 5.7s → 43 tok/s | max(5.3, 5.9) → **86 tok/s** | max(10.6, 6.5)=10.6s → 97 tok/s |
| 硬件算力墙（10 TFLOPS + 0.5B/param） | 186 tok/s（永远达不到，需 M≈245 且 cube 满效） | | | |

> **结论修正**：第一版"做完 P0 融合反量化，prefill 就能到 186 tok/s"过于乐观。正确表述是：
> **P0 把每 pass 时间从 12s 降到 2.5s 左右（≈5×，这是确定的）**；但要再往上（86–157 tok/s），
> 必须先解决 Cube 的"1.4 TFLOPS 瓶颈"——即手写 matmul 流水（B 常驻 L1、K 外层 / N 固定 / M 内层）
> 或换成能控制 L1 输入(TSCM)的高层 API 用法。这条和 §6.3 的融合反量化是同一条工程路线的两个部分。

---

## 3. 现状 vs 理论对照表

| # | 指标 | 理论/官方 | 平台模型/推导 | 实测（2026-11-19 23:00 复测） | 占理论 | 占可达上限 | 判定 |
|---|---|---|---|---|---|---|---|
| 1 | GM→UB 读带宽（单核） | 51.2 GB/s | 整机 d2d 45.5GB/s；单核 10.2 为可用墙 | **9.89–10.22 GB/s**（tile 8KB~112KB，读写混合 9.6–9.7，纯写 10.25–10.53） | 19.9% | 100% | 硬墙，与 tile/队列深度无关 |
| 2 | fp16 Cube（M=64,N=5120,K=5120） | 10.03 TFLOPS | 该形状带宽墙 = 52.4MB/10.2GB/s → 5.14ms = 0.65 TFLOPS | **0.60 TFLOPS（5.59ms），权重 9.38GB/s** | 6.0% | **92%（带宽墙）** | 带宽受限，这一档是健康的 |
| 3 | fp16 Cube（能拿到的最高值） | 10.03 TFLOPS | — | **1.46 TFLOPS**（1024×1024×1024） | **14.6%** | 14.6% | ⚠️ 高层 API 天花板（§2.2/§10.4） |
| 4 | 权重流带宽（M=64） | 51.2 GB/s | 10.2 GB/s | 9.38 GB/s | 18.3% | 92% | 已贴墙 |
| 5 | Vector fp16 吞吐（简单 op） | 156.7 Gelem-ops/s（128 elem/cyc） | — | **132–135 Gelem-ops/s** | **85.5%** | 85.5% | 已接近峰值（无空间） |
| 6 | Vector fp16 吞吐（依赖链 Add+Mul） | 156.7 | 延迟受限 | **68.7 Gelem-ops/s** | 44% | — | 指令依赖造成，换独立 op 即恢复 |
| 7 | W4A16 向量 GEMV | 10.2 GB/s（GM 墙） | — | **2.51 GB/s**（17.7ms / 44.6MB） | 4.9% | 24.6% | 瓶颈在 UB 往返/每行同步，非 GM（§6.6） |
| 8 | 两段式 prefill GEMM | — | 11.9 s/pass（带宽墙，121.5GB/10.2） | **实测 12s / 64 token**；`bench_dequant` 单独 8.94GB/s → 7.6s/pass | — | **99%** | 已到该方案极限，只能换方案 |
| 9 | GDN chunk 核心 | 156.7 Gelem-ops/s（向量）或 10 TFLOPS（cube） | 约 23 GFLOP/pass → 向量需 ~0.30s | **2.87 s / 64 token** | ~10%（对向量峰值） | ~10% | 最大的一块"低效算力"，见 §6.6 |
| 10 | **prefill 端到端** | 硬件墙 186 tok/s（不可达） | 现 API + 融合：**25–28 tok/s** 可达 | **4.30 tok/s**（14.87s/64token）【历史实测】 | 2.3%（对 186） | 17%（对 25） | 主因 #8/#9 |
| 11 | **decode 端到端** | 0.76 tok/s（10.2GB/s 墙）/3.8（51.2 理论） | — | **0.19 tok/s**（GEMV 5.4s/token）【历史实测】 | 25%（对实测墙） | 25% | GEMV 优化 + MTP |
| 12 | L2 读带宽（working set ≤4MB） | 平台模型 313 GB/s | — | **25.6–27.0 GB/s**（>8MB 掉到 9.7） | 8.6%（对模型值） | — | L2 只有 2.7×GM，但对小张量有用 |

> 注：`0.6 TFLOP/s` 这个数字**不应**和 10 TFLOPS 直接对比说"只用了 6% 算力"——该 GEMM 的算术强度 64 FLOP/B，
> 在 10.2GB/s 下算力上限只有 0.65 TFLOPS；反过来，若能把权重流量降下来（int4），同一个 cube 的算力才可能被用起来。
> 但 #3 说明：**即使给足算术强度，高层 Matmul API 也只到 1.46 TFLOPS**，这是工具链/流水的问题，不是带宽。

---

## 4. 现有测量方法学复核（哪些结论可能不严谨）

| # | 现有测法 | 问题 | 结论是否受影响 |
|---|---|---|---|
| 1 | `bench_stream`：blockDim=1/2/4/8，**每 block 工作量固定** | 时间随 blockDim 线性增长，**不能**区分"串行排队"与"并行但每核干满"。必须用"总工作量固定、每 block 只做 1/blockDim"复核 | 结论（单核）与官方 `ai_core_cnt=1` 一致，**大概率正确但需复核**（§5.1） |
| 2 | `bench_compute`：同上，且每 block 2048 loop 固定 | 同上；另外它测的是依赖链（`Add(a,a,b)` + `Mul(b,a,b)`），**混合了延迟与吞吐** | 69 Gelem-ops/s 的口径需拆成"独立指令吞吐"（本轮 `bench_scale` mode=1/3 复测） |
| 3 | `aclGetDeviceCapability` = 1/1/4MB | 曾怀疑"查询方式问题" | **不是问题**：与 `platform_config/Ascend310B1.ini` 完全一致（§1.2） |
| 4 | `bench_stream` 只测了 tile=16KB、队列深度 4、单模式 | 10.2GB/s 可能被**小请求/队列深度**限制（16KB×4 in-flight）；未测 tile 256B~112KB、未测写带宽、未测 L2 命中 | **可能低估真实带宽**，必须细扫（§5.2） |
| 5 | L2 4MB 从未被利用/测量 | 4MB 对 52MB 权重无用，但对激活/小矩阵（GDN chunk）有用；L2 带宽（模型 256B/cycle）未实测 | 影响 §6.5 GDN 优化判断 |
| 6 | Cube 5.6ms 的 TFLOP/s 口径 | 把"带宽受限的 GEMM"当成"算力测量"，得出 0.6 TFLOP/s；应同时报权重 GB/s 与该形状带宽墙 | 结论需改写（§3 #2） |
| 7 | W4A16 GEMV 2.51GB/s | 瓶颈在 UB 流量（每行 ~80KB）与每行一次 V_S 同步，不是 GM 带宽 | 对 decode 方案有意义（§6.5） |
| 8 | 两段式 12s/pass 估算 | 121.5GB/12s=10.1GB/s，与 #1 一致 ⇒ **确实是带宽墙**，不是实现差 | 结论正确 |
| 9 | 未测 GM **写**带宽、未测 CPU 侧（Taishan）带宽 | 写带宽决定两段式反量化成本；CPU 带宽可判定 DDR 真实容量（51.2 vs 34.1 vs 17） | 本轮补测（§5.2/§5.3） |
| 10 | 未测 M 放大对 Cube 的影响 | 若 M=64→256 时间不变，则直接证明"带宽受限 + 批处理几乎免费"，这是 prefill M 放大方案的依据 | 本轮补测（§6.2） |
| 11 | GDN chunk 的 2.87s 是**逐 head 向量实现** | 未批 head、未用 Cube、含大量标量归约；对比其数学量（~20 GFLOP/pass）效率只有 ~7% | 优化空间大（§6.5） |

---

## 5. 多核并行（最高优先级）——B 部分

### 5.1 结论（官方证据 + 复核实验）

**官方证据（已确认，无需实验即可定论）：**

- `Ascend310B1.ini` `[SoCInfo]`：`ai_core_cnt=1`、`vector_core_cnt=1`、`core_type_list=AICore,VectorCore,CubeCore`；
- 海思 Ascend 310B 芯片硬件指南：SoC 内部"集成了 **1 个** DaVinciV300 AI core + 4 个 TaishanV200M CPU 核"；
- OrangePi 20T 规格页 / Atlas 200I A2 规格："**1 个** DaVinciV300 AI core，主频 1.224GHz"；
- 本板 `aclGetDeviceCapability` 返回 `ai_core=1, vector=1, l2=4MB`，与配置文件一致。

**→ 真实并行度 = 1 个 AIC + 1 个 AIV。`blockDim=2` 及以上不可能加速；"多核 Cube / 多核 GEMV"在本板上不存在改造空间。**

**吞吐反推的独立交叉验证**（不依赖任何配置查询）：
- 单核 Vector 峰值 = 128B/cycle ÷ 2B(fp16) = 64 元素/cycle × 1.224GHz = **78.3 Gelem-ops/s**；
- 历史 `bench_compute` 实测 blocks=1 时 **64.4–69 Gelem-ops/s = 单核峰值的 82–88%** —— 这是"1 个向量核跑满"的典型数字；
- 若是 2 个向量核，峰值应为 156.6 Gelem-ops/s，此时 blocks=1 只跑到 44%、且 blocks=2 完全没有提升，不合常理；
- 故"单核"在**吞吐数据上也是自洽的**，不是查询口径问题。

**唯一可用的"并行"：Cube(AIC) 与 Vector(AIV) 是两个可并发的执行单元**（`cube_vector_combine=fuse,split`）。
这条对 §6.1 融合反量化方案很关键：反量化（AIV）与 MMA（AIC）可以流水重叠，
所以融合路径的耗时 ≈ max(带宽, 反量化, MMA) 而不是三者相加。

**历史流传的"310B DDR 28.8GB/s / 4 AI Core"说法**：与本板官方配置（1 核）不符，推测来自别的 310B 板型或别的测量口径；
但"整机内存不止 10GB/s"这点是对的——官方 `ascend-dmi` 在 310B 上的 d2d 参考值 ≈45.5GB/s（§2.1）。
所以本板的正确图景是：**DDR 整机 45GB/s，单 AI Core 的 GM 读只有 10.2GB/s**。

**复核实验（2026-11-19 23:00 已执行，结论一致）**：

| 实验 | 设计 | 实测结果 |
|---|---|---|
| `bench_mc`（带宽，总工作量固定 128MB） | `tilesPerBlock = total/(blockDim·tile)`，blockDim=1/2/4/8 | 13.13/13.17/13.20/13.25 ms；aggregate **10.22/10.19/10.17/10.13 GB/s**（不随 blockDim 上升） |
| `bench_scale`（计算，总 loop 固定 8192） | `loops=total/blocks`，blockDim=1/2/4/8 | mode=1 独立 Add：**133.7/134.1/134.1/132.2 Gelem-ops/s**；mode=3 Muls：135/135/134/132；mode=0 依赖链：**68.7/68.5/68.2/67.9** —— aggregate 全部不随 blockDim 放大 |
| `platform`（平台查询） | `PlatformAscendCManager::GetInstance("Ascend310B1")` | **GetCoreNumAic=1, GetCoreNumAiv=1**；UB=253952B、L1=1MB、L2=4MB；`GetCoreMemBw(L2)=256` |
| `aclGetDeviceCapability` | 已有 | 1 / 1 / 4MB（与上面一致） |

> **口径说明（修正第一版）**：上表"总工作量固定"时，若只有 1 个核，总时间**恒定**（每个 block 分到 1/N 的活、串行做完），
> aggregate 吞吐也恒定；若是 N 核并行，aggregate 会放大 N 倍。实测 aggregate 恒定 ⇒ **1 个向量核**。
> 第一版 §5.1 曾写"blocks 增加时总时间线性增长"，那是对**每 block 工作量固定**（旧 `bench_compute`）的描述，
> 两种口径的结论一致（单核），但表述已按本次干净数据修正。

### 5.2 那么 10.2GB/s 是不是 DDR 的真实上限？（本轮关键实测）

单核无法靠"加核"提升带宽，只能靠"把单核 MTE2 打满"。需要回答：
- tile 256B→112KB、队列深度、纯读/纯写/读写混合下的实际带宽；
- L2（4MB）命中时能到多少（决定 GDN chunk 优化空间）；
- CPU 侧 4 个 Taishan 核能拉多少（判定 DDR 是 51.2 还是 34.1 GB/s 档）。

**实验与结果（2026-11-19 23:00 实测，完整日志 `tmp/perf_clean.log`）**：

| 实验 | 目的 | 实测结果 |
|---|---|---|
| `bench_stream2` tile 细扫（512B/2K/8K/16K/32K/64K/112K，读/写/读写） | 16KB 是否限制了 MTE2 | tile≥8KB 饱和：只读 **9.96–10.02**、纯写 **10.25–10.53**、读写混合 **9.60–9.67 GB/s**；tile=2KB 约 9.3–9.5，tile=512B 崩到 1.3–4.0 → **10.2GB/s 是硬墙，与请求大小/队列深度无关** |
| `bench_l2`（window 256KB~128MB 回绕） | L2 容量与带宽 | working set ≤4MB：**25.6–27.0 GB/s（≈2.7×GM）**；8MB 起掉到 9.7 GB/s → **L2=4MB 确认可用**，但带宽远低于平台模型 313GB/s |
| `ascend-dmi --bw -t d2d -d 0` | 整机内存带宽（官方工具） | 本板 **未安装 ascend-dmi 工具**（CANN toolkit 的 tools 目录里没有该二进制），改用 CPU 侧基准（下一条） |
| CPU 带宽（4×Taishan 读 512MB，`scripts/bench_cpu_bw.c`） | DDR 真实容量 + CPU 可用带宽 | 见 §10.5（CPU 侧结果；整机 DDR 51.2GB/s 理论值来自官方规格，单 AI Core 拿不到） |
| `bench_dequant` | 两段式第一段地板 | 读 128MB int4+写 512MB fp16 → 75.06ms = **8.94 GB/s**；按 27B 权重折算 ≈ **7.6s/pass**（验证 12s 里 dequant 段占 7.6s + GEMM 读回 5.3s） |

### 5.3 如果多核可行，对 Cube GEMM / W4A16 GEMV 的改造收益？（假设性，回答"不值得做"）

- **Cube GEMM**：`MultiCoreMatmulTiling::SetDim(2+)` + 按 N/K 切分，理论 2×；但本板 `ai_core_cnt=1`，`SetDim(2)` 只会把两个 block 串行，收益 0，还可能增加调度开销。
- **W4A16 GEMV**：多 block 按 N 切分是**线性可扩展**的并行（每个 block 独立行），若真有 2 核可 ~1.8×；但 2.51GB/s 的瓶颈是 UB 流量/每行同步，多核各干一半也只是一半时间，不如先把单核 GEMV 优化到 5–8GB/s（§6.5）。
- **结论**：本板不需要、也无法做多核改造；预算应全部投到"减少每 token 权重字节数"上。

---

## 6. prefill 提速方案（C 部分，按预计加速比排序）

### 当前基线（prefill，64 token/pass）

| 项 | 时间/pass | ms/token | 占比 |
|---|---|---|---|
| Cube 两段式（int4→fp16 写 GM→读回做 GEMM） | 12.0 s | 187.5 | 80.7% |
| GDN chunk 核心（kkt/tril/wy/chunk_h/chunk_o） | 2.87 s | 44.8 | 19.3% |
| **合计** | **14.87 s** | **232.3** | **4.30 tok/s** |

### 6.1 方案总表（按预计加速比排序）

| 优先级 | 方案 | 流量/计算变化 | 预计加速比 | 对 prefill 端到端 | 工作量 | 风险 | 依据/可行性 |
|---|---|---|---|---|---|---|---|
| **P0** | **融合反量化 MMAD**：GM(int4) → UB 反量化 → L1(NZ) → L0B → MMAD（W4A16 直喂 Cube） | 4.5 → **0.5 B/param**（GEMM 12s → **2.5s**：带宽 1.32s 与 cube@1.4TFLOPS 2.5s 取大） | **4.8×**（受 cube 效率限制，不是 9×） | 14.87 → **5.4s/pass（2.75×）**；M=64 时 cube 成为新瓶颈 | 高（2–3 周） | **中高**：无 TSCM 官方样例、需手写 NZ 布局 + `LoadData`/`Mmad`/`Fixpipe` 流水；UB→L1 已实测可用（`probe_l1`） | §2.2 修正模型；`Mmad`/`LoadData` 接口已存在；AIV/AIC 可重叠 |
| **P0.5** | **修 Cube 效率 / 手写 matmul 流水**（B 常驻 L1；K 外层、N 固定、M 内层；大 base 复用） | 不改流量；`C_eff` 1.4 → **4.6–8 TFLOPS**（M=512/1024，B 只读一遍） | 单 GEMM 3–5×；P0 之后立刻变成第一瓶颈 | 融合路径 M=512：18.8s → **5.9s/pass = 86 tok/s**；M=1024 → 157 tok/s | 高（1–2 周） | **高**：需绕过高层 Matmul API（实测 `SetTraverse/SetFixSplit/SetDoubleBuffer` 最多改善 2.6%，见 §10.4） | §2.2、§10.4（probe_tiling 实测）；`sM=M,base=128x256x64,depthB1=4` 导致 B 按 M/128 重读 |
| **P1** | **prefill M 放大 64→512/1024**（同一份权重服务更多 token） | 每 pass 固定成本 × 重读次数 | 取决于 P0.5：未修 cube 时**只到 ~28 tok/s 就饱和**；修好后 M=512 可达 86 tok/s | 2.5–3.5× | **低（1–3 天）** | 低（scratch 每矩阵 ~178MB，显存够；M 受 L0C/激活显存约束） | §2.2 修正公式；`cube_m` 实测 M≥128 时间 ∝ M/128 |
| **P2** | **tiling/带宽细调**（singleCoreM/N/K、L1 双缓冲、L0C 累加、queue 深度、tile 对齐） | 不改流量，提高达成率 | 1.0–1.1× | 1.0–1.1× | 低~中（3–5 天） | 低 | §10.3/§10.4：10.2GB/s 已贴墙（写 10.5），tiling 旋钮实测最多 -2.6% ⇒ **这一项收益很小，降级** |
| ~~P2.5~~ | ~~CPU 协同反量化~~ | — | **不可行** | — | — | — | **实测否决**：CPU 4×Taishan 读只有 **2.6–3.8 GB/s**、copy 1.6 GB/s（§10.5），远低于单 AI Core 的 10.2；反量化 67.5GB 要 18s，比 NPU 两段式还慢。官方 45.5GB/s 是 SDMA 路径，CPU 代码用不到 |
| **P3** | **GDN chunk 核心优化**（按 head 批处理、去标量/去逐行同步、64×128×128 类矩阵用 Cube 或 L1 常驻） | 2.87s → 0.4–0.6s（**5–7×**） | 5–7× | P0 之后它占比升到 50%+，是第二阶段关键路径 | 中高（2 周） | 中：数值等价性已验证，但重写 5 个算子；历史有过挂死教训 | §3 #9（效率仅 ~10%）；`gdn_chunk_*` 已对拍通过，可安全重构 |
| **P4** | **多核 Cube / 多核 GEMV** | — | **1.0×（不可行）** | 0 | — | — | 官方 `ai_core_cnt=1` + 本次 `bench_mc/bench_scale` 实测（§5.1），无第二个 AI Core |
| **P5** | **Cube 直接吃 int4（W4A4，`DT_INT4=16×64×16`，40 TOPS）** | 流量同为 0.5B/param，且**无需反量化** | 相对 P0 再 1.1–1.3×（但同样受 cube 效率限制） | — | 高 | **高**：4-bit 激活精度风险极大；混合 int4×fp16/int8 MMA 官方不支持，只能 int4×int4 | 平台配置有 int4 fractal，但精度大概率不可接受 |
| **P5.5** | **W4A8（int4 权重上采样 int8 + int8 MMA 20 TOPS + Fixpipe/向量反量化）** | 流量仍 0.5B/param，Cube 峰值 20 TOPS；激活量化误差 | 若 cube 效率问题解决，算力墙翻倍 | 算力墙 186→372 tok/s（远期） | 高 | 中高：需 int8 activations 校准、Fixpipe per-channel/group dequant | `Ascend310B1.ini` 明确支持 `DT_INT8=16x32x16` 与 `Intrinsic_mmad s8`；但**前提仍是先解决 P0.5 的 1.4 TFLOPS** |
| **P6** | **W8A8（int8 权重）** | 1 B/param → 27GB **放不进 24GB 内存** | — | **不可行** | — | — | 内存容量硬约束 |
| **P7** | **MTP 投机解码**（decode 侧，与 prefill 正交） | 权重读取不变，一次验证 K+1 token | decode 1.5–2.5× | prefill 不变 | 中 | 中（接受率、回滚） | 历史调研；需要 GEMV 支持 M=2–4 批量 |

### 6.2 为什么 M 放大不是"几乎免费"（第一版在这里判断错了，本节已按实测修正）

数学上"同一份权重服务 M 个 token"是对的：权重流量与 M 无关、计算量 ∝ M。
但**前提是 kernel 真的只读一遍权重**。`cube_m` 实测（§10.4）表明高层 Matmul API 不是：

| M | GEMM 时间 | 权重有效带宽 | 推断 B 重读次数 |
|---|---|---|---|
| 64 | 5.59 ms | 9.38 GB/s | 1× |
| 128 | 7.50 ms | 7.01 GB/s | 1× |
| 256 | 13.76 ms | 3.87 GB/s | 2× |
| 512 | 28.57 ms | 1.91 GB/s | 4× |
| 1024 | 53.0 ms | 0.99 GB/s | 8× |

原因（`probe_tiling.cpp` 实测 tiling）：`sM=M, sN=5120, sK=5120, base=128x256x64, depthB1=4`
→ 输出按 baseM=128 分块，**每个 M 块都把 B（52.4MB）重读一遍**，所以时间 ∝ M/128。
且 `SetTraverse(FIRSTN)`/`SetDoubleBuffer`/`SetFixSplit` 都无法改变这一行为（最多 -2.6%）。

**修正后的推论**：
- 若不修 cube（用现 API）：M 从 64 放大到 512，GEMM 每 pass 时间 ×4（不是不变），
  因此 prefill tok/s **在 25–28 附近饱和**，M 放大只对"减少 kernel 启动/层调度开销"有边际收益；
- 若按 P0.5 手写 matmul（B 只读一遍）：M=512 时 GEMM 权重流量 52.4MB/10.2GB/s ≈ 5.1ms，
  512 token/5.9s → **86 tok/s**（算力与带宽打平），M 放大才开始"接近免费"；
- 同一条逻辑适用于**两段式（现方案）**：即使不换融合反量化，只要把 cube 修好 + M=512，
  两段式的 GEMM 读回从 4×54GB 降到 54GB → 12s/pass → 5.3s(读)+6.6s(dequant 写) ≈ 12s 不变（dequant 仍是瓶颈）
  ⇒ **两段式必须换掉（P0），M 放大才有意义**。两段式 + M 放大只能把 8 pass 的 96s 压到 1 pass 的 ~30s（3.2×），
  不如 P0（~5.4s/pass）。
- 真正的约束仍是 L0C(128KB)/L1(1MB) 的 tile 规模与激活显存：M=512–1024、K 分段完全放得下。

### 6.3 融合反量化 MMAD 的可行性依据（P0）

| 前提/组件 | 状态 | 依据 |
|---|---|---|
| UB→L1、L1→UB `DataCopy` | ✅ **已实测可用** | `probe_l1` 三种模式数据完全一致（历史；之前失败是脚手架问题） |
| L0C→GM `Fixpipe` | ✅ 硬件支持 | `support_fixpipe=1`（`Ascend310B1.ini`） |
| L1→L0A/L0B `LoadData2D` | ✅ 接口存在 | `kernel_operator_mm_intf.h`：`LoadData(dstL0A/L0B, srcL1, LoadData2dParams)`（含 `startIndex/repeatTimes/srcStride/dstGap`） |
| `Mmad(L0c, L0a, L0b, MmadParams)` | ✅ 接口存在 | `kernel_operator_gemm_intf.h` 内部 `MmadFunc` 就是模板用法（`mmadParams.m/n/k/isBias`） |
| 高层 Matmul 的 L1 输入（TSCM） | ⚠️ **API 存在但无样例** | `matmul.h` 里有 `SetTensorA/SetTensorB(LocalTensor<SrcT>)` 重载 + `shareL1Size` tiling 字段；但 basic-block 模板中有 assert "does not support GEMV and TSCM"，需走 NORM/MDL 模板，公开样例为零 → 风险高 |
| 向量反量化吞吐 | ✅ 足够 | 约 2 elem-op/权重 → `2×27e9/69e9 = 0.78s/pass`；与 1.32s 带宽重叠后被盖住 |
| AIV/AIC 并发 | ✅ 支持 | `cube_vector_combine=fuse,split`：反量化（AIV）与 MMA（AIC）可分核流水 |

**实现路线（从低风险到高风险）**：
1. **手写 `Mmad` 流水**（推荐）：反量化写 UB → `DataCopy` UB→L1（NZ 排布）→ `LoadData` → `Mmad` → `Fixpipe`。
   需要的 API 全部存在，唯一要自己推导的是 fractal（512B block）排布。**1–2 天可出最小原型**（64×64×256 对拍），
   失败模式是数值错位（能对拍发现），不会挂死。
2. 用高层 Matmul 的 TSCM 输入（省掉手写 MMA 流水），但需要自己构造 `TCubeTiling`（含 `shareL1Size`）与 NZ 布局；
   无样例、调试成本高 → **只在路线 1 卡住时尝试**。

**预期收益（P0 单独，本次按实测 cube 效率修正）**：
```
t_pass = max( 带宽 13.5GB/10.2GB/s = 1.32s,
              反量化向量 0.78s,
              MMA@M=64 = 3.46 TFLOP / C_eff )
C_eff = 1.4 TFLOPS（高层 API 实测）→ MMA 2.5s ；若手写流水到 4.6 TFLOPS → MMA 0.75s
```
→ 用现 API：64-token pass 的 GEMM 从 12s 降到 **2.5s（≈4.8×）**，端到端 14.87s → **5.4s（2.75×）**；
→ 若同时做 P0.5（C_eff≥3 TFLOPS）：GEMM → **1.4s（≈8.6×）**，端到端 → **4.3s（3.5×）**，且 M 放大后可继续降到 86 tok/s 级。
**注意**：P0 之后 cube 立刻成为新瓶颈（M=64 时 2.5s > 带宽 1.32s），所以 P0 与 P0.5 应当并行推进。

### 6.4 Cube 的 dtype / NZ 布局约束（决定"直接吃低精度"是否可行）

| 项 | 310B1 事实 | 影响 |
|---|---|---|
| INT8 MMA | ✅ 支持，fractal **16×32×16** → 20 TOPS | 但 A/B 必须**同为 int8**；`[DtypeMKN]` 里没有"int4×int8"或"int4×fp16"的混合项 → **W4A8/W4A16 不能直接喂 Cube** |
| INT4 MMA | ✅ 支持，fractal **16×64×16** → 40 TOPS | 只能 int4×int4（W4A4，激活也 4bit，精度不可控） |
| NZ/fractal 布局 | Cube 输入必须按 16×16（fp16/C0=16）分形块排布在 L1/L0 | 从 ND 的 GM 直接喂需要 ND→NZ 重排（`DataCopy` 的 nd2nz 模式或自己按 fractal 写）；融合路径反量化后正好可以**直接按 NZ 写 UB→L1**，省掉重排 |
| K 对齐 | K 按 fractal K=16 对齐；M/N 按 16 对齐 | M=64/256、K=5120、N=5120 天然满足；GDN 的 64×128 也满足 |

**结论**：310B 的 Cube 能算 int8（20 TOPS），但**没有任何"低精度权重 × 高精度激活"的硬化路径**；
所以 W4A16 必须"先反量化成 fp16 再进 Cube"。融合反量化（P0）是唯一能同时拿到"0.5B/param 流量"和"10 TFLOPS Cube"的路。

### 6.5 为什么不做 W4A4 / W8A8（明确不可行的依据）

- **W8A8**：27G 参数 × 1B = **27GB > NPU 可用 23.6GB**，装不下；即使分块换入也会把带宽翻倍（1B/param），比 int4 更慢。**排除。**
- **W4A4**：硬件有 int4 MMA（16×64×16，40 TOPS），流量还能保持 0.5B/param 且省掉反量化；但激活也要量化到 4bit，
  27B 模型 + 无官方校准工具链，精度风险不可控；且 Ascend C 的混合精度 MMA（int4 权重 × fp16/int8 激活）在 310B 上历史已确认不支持。
  **列为探索项，不进主线。**

### 6.6 做完 P0+P0.5 之后：瓶颈会转移到 GDN 和 decode

> 顺序说明：P0=减少每 pass 权重字节（4.5→0.5 B/param），P0.5=让 Cube 真正跑到 4.6+ TFLOPS（B 只读一遍）。
> 两者都做完后，512-token prompt 的 GEMM 约 5.9s（§6.8），GDN 的 23s 反而成为最大项。

**GDN chunk 核心（2.87s/64token）**：数学量约 **23 GFLOP/pass**（按 48 层 × 16 K-head/48 V-head 逐项累加），
向量峰值下只需 **~0.30s**，实测 2.87s → 效率 ~10%（若用 Cube 则只需 ~2.3ms，即纯数据/调度开销）。
低效原因：逐 head 小矩阵（64×128）、每行一次标量同步、大量 64/128 元素短向量、未批 head、未用 Cube。
改造方向（按收益/风险）：
1. 把 16 个 K head / 48 个 V head 的同类小矩阵拼成批处理（一次 kernel 处理多 head，减少同步次数）；
2. `gdn_kkt`/`wy`/`chunk_h`/`chunk_o` 的 64×128×128 类矩阵改用 Cube（数据常驻 L1/L0，4MB L2 可缓存状态）；
3. 消除"每行一次 V_S 同步"（改成每批一次），用 `ReduceSum` 硬件归约替代手工树。
目标 0.4–0.6s/pass（5–7×，仍只到向量峰值的 50–75%，留有余量）。

**decode 侧（与 prefill 独立）**：当前 GEMV 2.51GB/s = 5.4s/token = 0.19 tok/s，而带宽墙是 0.76 tok/s。
优先把 GEMV 从 2.51 提到 6–8GB/s（UB 流量减半、每批一次同步、ReduceSum），再叠加 MTP（1.5–2.5×）。

### 6.7 ~~低风险过渡方案（P2.5）：CPU 协同反量化~~ —— **实测否决（保留记录，避免重复踩坑）**

**原动机**：传闻整机 d2d 带宽 45.5GB/s，而单 AI Core 只有 10.2GB/s；若 CPU 4 核做 int4→fp16 写 GM，
理论上能吃整机带宽、把两段式第一段（67.5GB）从 6.6s 降下来。

**实测（`scripts/bench_cpu_bw.c`，2026-11-19 23:12）**：

| 模式 | 线程 | 聚合带宽 |
|---|---|---|
| 读（512MB） | 1 | **2.61 GB/s** |
| 读（512MB） | 4 | **3.82 GB/s** |
| copy（读+写） | 4 | **1.58 GB/s** |

→ CPU 4 核（TaishanV200M @1.6GHz，顺序流水/MLP 很低）一起也只有 **3.8 GB/s 读**，比单 AI Core 的 10.2 差 2.7×；
反量化 67.5GB（13.5G 读 + 54G 写）按 3.8/1.6 GB/s 折算要 **18–42s/pass**，比现有 12s 还慢。
官方 45.5GB/s 的 d2d 是 **SDMA/整机路径**，普通 CPU/NEON 代码拿不到。
**结论：P2.5 删除，别做。** DDR 容量档位（51.2 vs 34.1 vs 17）对本引擎也不再重要——
瓶颈始终是单 AI Core 的 10.2GB/s 端口，而不是 DDR 颗粒。

### 6.8 累计路线图（prefill 提速的合成效果）

以 **512-token prompt** 为例（现状是 8 个 64-token pass；GDN 成本按 chunk 数线性外推）。
**下表已按本次实测修正**（cube 有效算力 1.4 TFLOPS + B 按 M/128 重读；见 §2.2/§10.4）：

| 阶段 | GEMM / 512 tok | GDN / 512 tok | 每 token | prefill tok/s | 相对当前 |
|---|---|---|---|---|---|
| 现状（两段式，M=64，8 pass） | 96 s | 23 s | 232 ms | **4.3** | 1× |
| + P1：M 放大到 512（现 API，B 重读 4×） | 27.6 s | 23 s | 99 ms | 10.1 | 2.3× |
| + P0：融合反量化（M=64 ×8 pass，cube@1.4T） | 20 s | 23 s | 84 ms | 11.9 | 2.8× |
| + P0+P1：融合 + M=512（现 API，B 重读 4×） | 18.8 s | 23 s | 82 ms | 12.2 | 2.8× |
| **+ P0+P0.5：融合 + 手写 matmul（M=512，B 只读一遍）** | **5.9 s** | 23 s | 56 ms | **17.7** | **4.1×** |
| + P0+P0.5+P3：再把 GDN 优化 6× | 5.9 s | 3.8 s | **19 ms** | **53** | **12×** |
| 硬件算力墙（10 TFLOPS + 0.5B/param + M≈245） | — | — | 5.4 ms | 186 | 43× |

> 说明：
> ① 第一版曾按 10 TFLOPS 峰值估出 "+P1→14.6 / +P0→19.9 / +P3→77 tok/s"；本次按实测
>    `ceil(M/128)` 重读 + `C_eff≈1.4 TFLOPS` 修正为上表。**方向不变（P0 > P0.5 > P3 > P1 > 其它），绝对数字下调**。
> ② M 放大只对**长 prompt** 有效（64-token 短 prompt 无法提到 512），且必须配合 P0.5 才有明显收益。
> ③ P0 之后 GDN 占比升到 50%+，P3 是第二阶段关键路径；P0 与 P0.5 可并行推进（一个改流量、一个改流水）。
> ④ 186 tok/s 是 FP16 Cube 硬件算力墙，再往上需要 INT8 Cube（W4A8，§6.1 P5.5），但前提同样是先解决 cube 效率。
> ⑤ 上表按"GEMM 与 GDN 串行"保守估计；引擎里不同层/不同 chunk 可部分重叠，实际可能略好。
> ⑥ 若只做 P0 不做 P0.5，prefill 会卡在 **~12 tok/s**（GEMM 2.5s/pass ×8 + GDN）；这是"最少工作量拿到 2.8×"的路线。

---

## 7. 复跑命令（板端）

> 环境注意（实测踩坑）：`source /usr/local/Ascend/ascend-toolkit/latest/bin/setenv.bash` 在高负载时会**卡住几十秒到十几分钟**
> （它内部串 8 个子脚本，其中会做一些慢操作），跑性能测试时建议**直接用 LD_LIBRARY_PATH**，不要 source setenv。
> 另外 **不要用 `bash run.sh build` 做增量**——它会 `rm -rf build out`，把 10 分钟的编译量重来一遍。

**完整复跑（7 个测试，约 30 秒）**：
```bash
cd ~/qwen38_310b/ascendc_ops
export LD_LIBRARY_PATH=$PWD/out/lib:$PWD/out/lib64:/usr/local/Ascend/ascend-toolkit/latest/lib64:/usr/local/Ascend/ascend-toolkit/latest/runtime/lib64:$LD_LIBRARY_PATH
B=./qwen38_ops_test
for t in platform bench_mc bench_scale bench_stream2 bench_l2 cube_m bench_dequant; do
  echo "########## $t"; timeout -s KILL 180 $B $t 2>&1 | grep -E "^\[|OK|FAIL"
done
```

**构建（首次 / 新增 kernel 后）**：
```bash
# 首次：bash run.sh build
# 增量（推荐）：CMakeLists 用 file(GLOB kernels/*.cpp)，所以必须重跑 cmake 才会发现新文件
cd ~/qwen38_310b/ascendc_ops/build && cmake .. >/dev/null && make -j4
cd .. && cmake --install build >/dev/null && cp -f out/bin/qwen38_ops_test ./
```

**Cube tiling 诊断（独立程序，不依赖工程构建）**：
```bash
A=/usr/local/Ascend/ascend-toolkit/latest
cd /tmp/prof && cp ~/qwen38_310b/ascendc_ops/scripts/probe_tiling.cpp .
g++ -O2 -std=c++17 -D_GLIBCXX_USE_CXX11_ABI=0 \
  -I$A/include -I$A/aarch64-linux/ascendc/include/highlevel_api probe_tiling.cpp -o probe_tiling \
  -L$HOME/qwen38_310b/ascendc_ops/out/lib -L$A/lib64 -L$A/runtime/lib64 -L$A/acllib/lib64 \
  -Wl,-rpath-link,$A/lib64 -Wl,-rpath-link,$A/runtime/lib64 -Wl,-rpath-link,$A/acllib/lib64 \
  -lascendc_kernels_mm_npu -lascendc_kernels_npu -ltiling_api -lregister -lplatform -lascendalog -ldl \
  -lascendcl -lruntime -lerror_manager -lprofapi -lge_common_base -lmmpa -lascend_dump -lc_sec
export LD_LIBRARY_PATH=$HOME/qwen38_310b/ascendc_ops/out/lib:$A/lib64:$A/runtime/lib64:$A/acllib/lib64
timeout -s KILL 240 ./probe_tiling      # 打印每个配置的 TCubeTiling 字段 + 实测 TFLOP/s
```

**CPU 侧带宽（判定"CPU 协同反量化"是否可行，实测否决）**：
```bash
cd ~/qwen38_310b/ascendc_ops
gcc -O2 -pthread scripts/bench_cpu_bw.c -o /tmp/bench_cpu_bw
/tmp/bench_cpu_bw 512 1 read; /tmp/bench_cpu_bw 512 4 read; /tmp/bench_cpu_bw 512 4 copy
```

> 安全约定：每个 kernel 都用 host 侧 `timeout -s KILL` 保护；blockDim 从 1/2/4 逐步升；
> 一旦 SSH 掉线立即停止该类实验并在本文档记录"该配置会挂死"。
> 历史教训：`gdn_chunk_o` 第一版在 64×64 内层循环里做 GM 搬运 + `PipeBarrier<PIPE_ALL>`，把整板搞重启。
> 官方 `ascend-dmi` 工具**本板未安装**（CANN toolkit tools 目录里没有），整机 d2d 带宽无法直接测；
> 如需测整机带宽可用 SDMA/多进程 CPU 组合，但对本引擎无意义（见 §6.7）。

---

## 8. 本轮新增/复用的 kernel 与实验清单

| 文件 | 作用 | 状态 |
|---|---|---|
| `kernels/bench_mc.cpp` | 多 block 带宽（固定每 block 工作量 / 固定总工作量两种口径都可配；mode 2 = L2 窗口回绕） | ✅ 板上验证（本次复测 §10.2） |
| `kernels/bench_compute_mc.cpp` | 多 block 向量计算（固定每 block / 固定总 loop；mode 0 依赖链 / 1 独立 Add / 2 fp32 / 3 Muls） | ✅ 板上验证 |
| `kernels/bench_stream2.cpp` | tile 512B~112KB × 读/写/读写 × queue=2 | ✅ 板上验证 |
| `kernels/bench_dequant.cpp` | int4→fp16 反量化写 GM 的纯带宽成本 | ✅ 板上验证 |
| `host/main.cpp` | 新增 `platform / bench_mc / bench_scale / bench_stream2 / bench_l2 / cube_m / bench_dequant` 入口 | ✅ 已合并并编译通过 |
| `scripts/bench_cpu_bw.c` | 4×Taishan 读/copy 带宽 | ✅ 板上实测（§10.5） |
| `scripts/probe_tiling.cpp` | **新增（本次）**：独立 g++ 程序，打印 TCubeTiling 全字段 + 扫 `SetTraverse/SetFixSplit/SetDoubleBuffer` + 实测 TFLOP/s | ✅ 板上验证（§10.4） |
| `scripts/bench_dev_fused_mmad.cpp`（规划） | 融合反量化 MMAD 最小原型：UB 反量化 → NZ L1 → L0B → Mmad → Fixpipe | ⬜ 待写（P0 的第一步） |

### 后续实验清单（按优先级）

1. **`bench_dev_fused_mmad`**（P0 第一步）：先 64×64×256 单 tile 对拍，再扩到 64×5120×5120；
2. **手写 matmul 的最小原型**（P0.5）：K 外层、B 常驻 L1（`B[64,5120]`=655KB）、M 内层，验证 M=512 时能否把
   52.4MB 权重只读一遍（目标 5.1ms@10.2GB/s、~4.6 TFLOPS）；
3. GDN chunk 批 head 重构（P3）：先拿 `gdn_kkt` 做试点（0.807ms → 目标 0.15ms）；
4. `gdn_rgdr` decode 递推 + MTP 验证（decode 侧，与 prefill 正交）。

---

## 9. 出处

- OrangePi AI Pro 20T 用户手册 v1.1（本仓库 `用户手册/OrangePi_AI_Pro_20T_昇腾_用户手册_v1.1.pdf` §1.2：FP16 10 TFLOPS / INT8 20 TOPS / LPDDR4X 12·24GB）
- OrangePi 官方规格页：<https://www.hiascend.com/developer/devboard/spec>（20 TOPS INT8，1×DaVinciV300 @1.224GHz，4×TaishanV200M @1.6GHz，LPDDR4X 4266Mbps）
- OrangePi 产品参数页：<http://www.orangepi.cn/html/hardWare/computerAndMicrocomputers/parameter/Orange-Pi-AIpro-20T.html>
- Atlas 200I A2 加速模块用户指南（基本规格，20TOPS 12GB：1×DaVinciV300 @1.224GHz、LPDDR4X 4266Mbps、96bit、51.2GB/s）：<https://support.huawei.com/enterprise/zh/doc/EDOC1100304819/e6e7e6d0>
- 海思 Ascend 310B 芯片硬件指南（1 个 DaVinciV300 AI core、4 个 TaishanV200M、96-bit LPDDR4X 接口）：<https://bbs.16rd.com/thread-630071-1-1.html>
- 华为论坛"20T 的是昇腾 310B1，8/12T 的是 310B4"：<https://www.hiascend.com/dev/forum/thread-0261156769253689371-1-1.html>
- Ascend 310B `npu-smi` 命令参考（310B1=20T / 310B4=8T）
- CANN `platform_config/Ascend310B1.ini`（**本板已复核**：`/usr/local/Ascend/ascend-toolkit/latest/compiler/data/platform_config/Ascend310B1.ini`，
  `ai_core_cnt=1, vector_core_cnt=1, cube_freq=1224, l2_size=4194304, ub_size=253952, l1_size=1048576`；
  同目录 `Ascend310B2/B3/B4.ini` 结构相同，B4 为 `cube_freq=500`（8T 档））
- 本板 `npu-smi info -t memory`：`Clock Speed(MHz): 2131`（LPDDR4X-4266 的 2133MHz 档）
- AscendC 头文件（本次用于 tiling 诊断）：`.../highlevel_api/lib/matmul/matmul_tilingdata.h`（TCubeTiling 全字段）、
  `.../lib/matmul/bmm_tiling.h`（MultiCoreMatmulTiling: SetFixSplit/SetTraverse/SetDoubleBuffer）、
  `.../lib/matmul/matmul_tiling_base.h`（`MatrixTraverse{NOSET,FIRSTM,FIRSTN}`）
- 仓库历史实测：`docs/AscendC算子开发进度.md` §3.5/§6/§7/§8

---

## 10. 定稿实测数据（2026-11-19 23:00-23:15，perf-scout 板端复测）

> 板子：昇腾 **310B1**（20T 档，`npu-smi` 报 310B1），CANN 8.0.0，24GB，**1 个 DaVinciV300 AI Core + 1 个 Vector Core**。
> 原始日志：`tmp/perf_clean.log`（7 个测试完整输出）。本轮为**独占窗口**（lead 已停止 build/engine 测试）采集，
> 与此前 perf-scout2 在 build 负载下采集的数据对比，数字更干净（例如 bench_scale 简单 op 从 108 回到 134 Gelem-ops/s）。

### 10.1 平台事实（`./qwen38_ops_test platform`）
```
[platform] GetCoreNumAic=1  GetCoreNumAiv=1
[platform] UB=253952 B (248.0 KB)  L1=1048576 B (1024.0 KB)  L2=4194304 B (4.0 MB)
[platform] GetCoreMemBw UB=0 B/cycle?  L1=0  L2=256
```

### 10.2 多核并行复核（回答 B 部分）—— `bench_mc` + `bench_scale`
```
# 固定总工作量 128MB，只看 aggregate 带宽是否随 blockDim 放大
blocks=1 -> 13.130 ms, aggregate 10.22 GB/s
blocks=2 -> 13.174 ms, aggregate 10.19 GB/s
blocks=4 -> 13.196 ms, aggregate 10.17 GB/s
blocks=8 -> 13.253 ms, aggregate 10.13 GB/s
# 固定总计算量 8192 loop（每 block 做 1/blockDim），简单 op / 复合 op
mode=1 独立 Add:  1/2/4/8 blocks -> 133.7 / 134.1 / 134.1 / 132.2 Gelem-ops/s
mode=3 Muls:      1/2/4/8 blocks -> 135.2 / 135.1 / 134.1 / 132.3 Gelem-ops/s
mode=0 依赖链:    1/2/4/8 blocks ->  68.7 /  68.5 /  68.2 /  67.9 Gelem-ops/s
```
**结论**：aggregate 吞吐不随 blockDim 放大 ⇒ **只有 1 个 Vector Core，blockDim>1 纯串行排队**。
真实"并行"只存在于 Cube(AIC) 与 Vector(AIV) 之间（`cube_vector_combine=fuse,split`），用于融合反量化流水。

### 10.3 带宽墙细扫（`bench_stream2` + `bench_l2`）
```
tile=  8KB  read 9.52 | read+add 10.00 | r+w 9.67 | write 9.90  GB/s
tile= 16KB  read 9.71 | read+add 10.02 | r+w 9.61 | write 9.91
tile= 64KB  read 9.89 | read+add 10.01 |            write 10.53
tile=112KB  read 9.81 | read+add  --    |            write 10.46
tile=0.5KB  read 1.28 | read+add  3.09 | r+w 4.04 | write 2.59   （小请求才掉）
# L2 窗口回绕（逻辑读 64MB）
window ≤4MB : 23.9 ~ 27.0 GB/s   （L2 命中，≈2.7×GM）
window ≥8MB :  9.3 ~ 10.1 GB/s   （退回 DDR）
```
**结论**：**10.2GB/s 是单 AI Core 的硬墙**（与 tile、队列深度无关；写 10.5 略高，混读写 9.6）；
L2=4MB 可用，带宽 ~26GB/s（不是平台模型的 313GB/s），对小张量（GDN chunk / 激活）有意义。

### 10.4 Cube 效率诊断（`cube_m` + `scripts/probe_tiling.cpp`）—— 本次最重要的新发现
```
# cube_m（N=K=5120，权重 52.4MB）
M=64  -> 5.59 ms | 600 GFLOP/s | 权重 9.38 GB/s
M=128 -> 7.50 ms | 894 GFLOP/s | 权重 6.99 GB/s
M=256 -> 13.76ms | 975 GFLOP/s | 权重 3.87 GB/s
M=512 -> 28.57ms | 940 GFLOP/s | 权重 1.91 GB/s
# probe_tiling：默认 tiling 与旋钮扫描（M=512 基准 27.5ms）
default                    : sM=512 sN=5120 sK=5120 base=128x256x64 depthA1=40 depthB1=4 stepKa=20 iterOrder=0
SetTraverse(FIRSTN)        : 26.79 ms (−2.6%)   ← 唯一有效
SetTraverse(FIRSTM)        : 27.53 ms
SetDoubleBuffer(a,b,c)     : 27.55 ms
SetFixSplit(128,128,64)    : 32.82 ms (+19%)
SetFixSplit(64,512,64)     : 44.32 ms (+61%)
# 能拿到的最高算力
1024x1024x1024 (full)      : 1.469 ms | 1462 GFLOP/s  ← 1.46 TFLOPS = 峰值的 14.6%
2048x2048x2048             : 14.07 ms | 1221 GFLOP/s
1024x5120x5120             : 53.0 ms  | 1013 GFLOP/s
```
**结论**：
1. 高层 Matmul API 的 tiling 固定 `sM=M` 且 `base=128x256x64`，**每个 baseM=128 块都把 B 重读一遍**，
   所以 M≥128 后时间 ∝ M/128（M=1024 时权重有效带宽只剩 0.99GB/s）；
2. `SetTraverse/SetFixSplit/SetDoubleBuffer` 无法修复（最多 -2.6%）；
3. 高层 API 的 Cube 实测天花板 **1.0–1.5 TFLOP/s = 理论 10 TFLOPS 的 10–15%**。
   这是 P0（融合反量化）之后的新瓶颈，也是 P0.5（手写 matmul）存在的原因（§2.2/§6.1）。

### 10.5 CPU 侧带宽（`scripts/bench_cpu_bw.c`，用于评估"CPU 协同反量化"）
```
read  1 thread  (512MB): 2.61 GB/s
read  4 threads (512MB): 3.82 GB/s
copy  4 threads (512MB): 1.58 GB/s
```
**结论**：4×TaishanV200M 合计只有 3.8GB/s 读，远低于单 AI Core 的 10.2GB/s ⇒ **CPU 协同反量化不可行（P2.5 删除）**。

### 10.6 两段式第一段成本（`bench_dequant`）
```
read 128MB int4 + write 512MB fp16 -> 75.06 ms, 8.94 GB/s
```
**结论**：按 27B 权重折算 ≈ **7.6s/pass**（+GEMM 读回 54GB@10.2GB/s≈5.3s = 12.9s，与历史实测 12s 自洽）
⇒ 两段式的 12s/pass 中 60% 是"多搬的 4B/param"，P0 融合反量化是唯一能砍掉它的办法。

### 10.7 关键判断（定稿版）
1. **单 AI Core 的 GM 端口（10.2GB/s 读 / 10.5 写）是硬墙**，理论 51.2GB/s 是整机 DDR 的，单核拿不到；
   多核不存在（官方 ai_core_cnt=1 + acl 查询 + 本机 aggregate 带宽/算力都不随 blockDim 放大，三重证据）。
2. **0.6 TFLOP/s（M=64）不是"算力只用了 6%"的低效**，而是该形状**带宽受限**（52.4MB/5.6ms=9.4GB/s = 墙的 92%）；
   但**给足算术强度后高层 API 也只到 1.46 TFLOPS**（峰值 14.6%），这是工具链/流水问题，不是带宽——修正了第一版"cube 只要流量降下来就能到 10 TFLOPS"的判断。
3. prefill 第一主线仍是**减少每参数搬运字节数**：两段式 4.5 → 融合反量化 0.5 B/param（9×↓，12s → 受 cube 限制约 2.5s/pass）；
   第二主线是 **P0.5 修 Cube 效率**（1.4 → 4.6+ TFLOPS，让 M 放大真正"免费"）；第三是 GDN chunk 优化。
4. Vector 峰值实测 **134 Gelem-ops/s（简单 op，理论 156.7 的 85.5%）**；复合依赖链 68.7。
   GDN chunk 当前效率 ~10%（2.87s vs 数学量 ~0.30s），是 P0 之后最大的一块低效算力。
5. **明确不可行**：多核并行（不存在）、CPU 协同反量化（CPU 3.8GB/s 太慢）、W8A8（27GB 装不下）、
   P2 单纯 tiling 调优（10.2 已贴墙、官方旋钮最多 -2.6%）。
