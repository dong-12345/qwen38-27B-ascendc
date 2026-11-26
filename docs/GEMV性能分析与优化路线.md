# W4A16 GEMV 性能分析与提速路线（2026-11-24 实测）

> ## 📌 最新结论（2026-11-25）
>
> ### 已端到端验证
> | 项 | 现状 | 对比基线 | 加速 | 一致性 |
> |---|---|---|---|---|
> | **27B prefill（17 token）** | **36.7 s（2.16 s/token）** | 212.8 s（12.52 s/token） | **5.8x** | `TOKENS 760 6511 314 9338` ✓ 完全相同 |
> | 27B decode | **12.48 s/token** | 14.28（gen 旧内核） | **1.14x** | ✓ 逐位相同 |
> | Moondream3 decode | **2.02 s/token** | 2.39 s | **1.18x** | ✓ 逐位相同 |
> | 两段式 Cube（K=5120/N=17408, M=32） | **72.2 ms** | 1746 ms | **24.2x** | max_abs 0.0078 / 0 超差 |
> | 两段式 Cube（K=17408/N=5120, M=32） | **64.0 ms** | 989 ms | **15.5x** | max_abs 0.0020 / 0 超差 |
>
> ### 已实现并验证：AX 布局 + Axpy-N decode kernel（单矩阵 2.21x，数值正确）
> - 权重重排工具：`./qwen38_ops_test relayout <src> <dst>` —— 401 个 W4 张量 → **AX `[g][kp][n]`**（uint16，k-pair 打包），
>   scale/sz → `[NG][N]`；板上已生成 `~/code/qwen38_310B/models/qwen38_310b_ax`（29 GB，10m52s）。
> - 新 kernel `w4a16_gemv_axpy`（沿 N 向量化、无归约树）**单矩阵实测**：
>
>   | 形状 | gen2 | **axpy(AX)** | 加速 | 数值 |
>   |---|---|---|---|---|
>   | K=5120 / N=17408 | 46.10 ms | **20.83 ms（2.14 GB/s）** | **2.21x** | max_abs 0.0044，**0/17408 超差** |
>
> - 调试中踩到的真坑（已修）：**scale/sz 的 `DataCopy`(MTE2) 之后只加了 `PipeBarrier<PIPE_V>`，
>   不等 MTE2 就 `Cast` → 读到垃圾 scale**（表现为结果整体大 50 倍 / 饱和到 65504）。
>   改成 `PipeBarrier<PIPE_ALL>` 后数值立即正确。
> - 已知可再优化：AX 目前是 **1 B/权重**（1 个 uint16 只装 2 个半字节），
>   若改成 1 个 uint16 装 4 个半字节（k0..k3）→ 流量减半，按 DMA 占比估算还能再快 ~1.3–1.5x。
>
> ### 待完成
> - `ENG27_LAYOUT=axpy` 的**端到端** decode 验证（板子网络不稳，最后一步被中断）；
>   命令：`ENG27_LAYOUT=axpy ENG27_DIR=~/code/qwen38_310B/models/qwen38_310b_ax ENG27_IDS=248045,846 ENG27_GEN=8 ./out/bin/qwen38_ops_test engine_27b`
>   期望 tokens 与基线一致（`198 3710 369 248046 ...`），decode 从 12.5 s/token 降到 **~5.5–6 s/token**。
> - AX 模式下 prefill 目前强制逐 token（AX 版 dequant 未写）；补上后可两路共用同一份权重。

>
> ### 阻塞
> 2026-11-25 09:55 起开发板不可达（主机侧 `192.168.137.x` 网卡消失，ping 超时），
> 需恢复网络/给板子上电后继续验证。




> 目标：解释"为什么 27B 在这块板上 14.6 s/token"，给出**可执行的**提速路线与实测依据。
> 所有数字均为板端（OrangePi AI Pro 20T / Ascend 310B1 / CANN 8.0.0）实测。

---

## 1. 现状

| 项 | 实测 | 说明 |
|---|---|---|
| 27B decode（64 层全量） | **14.34 s/token**（0.070 tok/s） | `ENG27_IDS=<17 token> ENG27_GEN=4`；去栅栏优化后 |
| 27B decode（基线） | 14.58 s/token | 未去 `PipeBarrier<V>` 时 |
| 27B prefill | **14.34 s/token**（同速） | 逐 token 走同一路径，**prefill 与 decode 同速** |
| GEMV 单矩阵（K=5120, N=17408） | **54.5 ms = 0.82 GB/s** | 权重 44.6 MB（旧 55.6 ms） |
| Moondream3 decode | **2.38 s/token** | 24 层 MoE（20 层 × 8 专家 × 2 GEMV） |

**decode 时间 ≈ 100% 是自研 W4A16 GEMV**：每 token 要读 13.53 GB 权重，
按 0.8 GB/s 折算 16.9 s，实测约 14.3 s（小矩阵利用率略高）。

**本次唯一落地的提速**：删掉内核里 16 处同流水 `PipeBarrier<PIPE_V>()`
（向量指令同流水按序执行，不需要栅栏）→ **14.58 → 14.34 s/token（1.7%）**，
输出 token 与基线逐位相同（`TOKENS 760 6511 314 9338`）。

---

## 2. 带宽用满了吗？——没有，三样都没打满

| 口径 | 实测 | 当前用到 | 说明 |
|---|---|---|---|
| 单 AI Core GM 读 | **10.2 GB/s** | **9.2%**（13.53 GB / 14.34 s = 0.94 GB/s） | tile 0.5–112 KB 全扫都是 9.6–10.5，是单核端口硬墙 |
| 单 AI Core GM 写 | 10.5 GB/s | 未用 | 融合路径才有写 |
| L2（工作集 ≤4 MB） | 26–27 GB/s | 用不到 | 权重每 token 只读一遍，没有复用 |
| 整机 DDR 理论（96-bit LPDDR4X-4266） | 51.2 GB/s | 单核拿不到 | 官方 d2d 45.5 GB/s 是 SDMA 路径；CPU 侧实测仅 3.8 GB/s |
| Vector 吞吐 | 110 elem/cycle（fp16） | **~15%**（小向量实测 16 elem/cycle） | 每条指令固定开销 ~65 ns 吃掉 |
| **Cube** | 1.0–1.46 TFLOPS | **0%（完全空闲）** | 关键浪费点 |

**带宽地板**：13.53 GB / 10.2 GB/s = **1.33 s/token（0.75 tok/s）**；
现在 14.34 s/token，与地板差 **10.8 倍** —— 这 10.8 倍里没有一分是带宽天花板，
全是"每输出行 48 条向量指令 × 65 ns"的指令下发开销。

---

- 单核 GM 读带宽实测 **10.2 GB/s** → 13.53 GB 权重的**带宽地板 = 1.32 s/token**（0.76 tok/s）。
- 当前 14.34 s 只用掉带宽的 **9%**，所以不是带宽墙。
- 旧 kernel `w4a16_gemv_gen` 每输出行要 **~48 条向量指令**，且每行一次 `V_S` 同步：
  实测 **3.1 µs/行**（K=5120，54.5 ms / 17408 行），48 条 × 65 ns ≈ 3.1 µs，**完全吻合**。
- 用新增的探针 `kernels/probe_v2.cpp` 测单条向量原语的边际成本（两种长度相减、剔除 setup）：

| 原语（fp32） | 1280 元素 | 5120 元素 | 10240 元素 |
|---|---|---|---|
| Add（含 1 个 `PipeBarrier<V>`） | 19.5 ns* | 76 ns* | 289 ns* |
| 分组树形归约（32→8 lane） | 248 ns | — | 467 ns |
| Gather（count=len） | 134 ns | — | 464 ns |
| WholeReduceSum(mask=64) | 225 ns | — | 1153 ns |

\* 第 1 次探针测的"绝对成本（含 setup）"；关键结论是**固定开销 ≈ 60–90 ns/指令**，
小向量（1–5 KB）时几乎与元素数无关 —— 即"每条指令付一次 ~70 ns"。

> 推论：`3.1 µs/行 = 48 指令 × 65 ns`。**要提速就必须减少"每行的指令条数"，
> 而不是优化带宽**（带宽还有 11 倍余量）。

**栅栏本身几乎不花钱**（已实测）：把 v1 内核里 16 处 `PipeBarrier<PIPE_V>()` 全删掉，
只快了 1.7%（55.6 → 54.5 ms）——说明时间花在"每条向量指令的固定下发成本"上
（小向量时实测只有 ~16 元素/cycle，远低于峰值 110），而不是在流水冲刷上。

---

## 3. 已尝试：行批处理 kernel（`w4a16_gemv_batch.cpp`，实验版）

思路：一次算 R 行输出，把"每行一次"的指令开销摊到 R 行；4 个相位合并成大指令；
掩码值 `q·16^p` 直接把 `16^-p` 折进 act，省掉 ShiftRight；用 Gather 把每组的 16 个部分和
收成 `[t][r][g]` 布局再用 4 次 Add 合并 t 维。

**实测结果（K=5120, N=17408）**

| 版本 | 时间 | 有效带宽 | 加速 | 数值 |
|---|---|---|---|---|
| v1 `w4a16_gemv_gen` | 54.5 ms | 0.82 GB/s | 1.00x | 基准（已对拍正确） |
| v2 `w4a16_gemv_batch` | **35–36 ms** | **1.21–1.29 GB/s** | **1.5–1.6x** | ❌ 尚不等价（max_abs≈3.4–7.0） |

**为什么只有 1.6x**：合并后单条指令处理的元素量涨了 4 倍（`4*nQ = 41 KB`），
单指令成本从 ~70 ns 涨到 ~240 ns —— 固定开销摊薄了，但元素成本补回来。净收益 ≈ 1.6x。
另外 R 受 UB(248 KB) 限制：K=5120 只能 R=4；K=17408 只能 R=1–2。

**数值不等价的原因（已排除项 × 已确认项）**

| 假设 | 结论 |
|---|---|
| Gather 偏移是字节 vs 元素 | ✅ 已确认是**字节**（`stage=98` 试元素偏移误差更大） |
| 掩码 `0xF000` 但 int16 变负 | ✅ 已修（改成 `ShiftRight 12` 再 `And 0x000F`） |
| 偏置项误乘了相位因子 `fac[p]` | ✅ 已修（mean_abs 0.86 → 0.33） |
| 4 相位合并 vs 逐相位 | ✅ 已排除（`stage=97` 逐相位结果与合并版逐位相同） |
| 非 2 的幂长度的树形归约 | ⚠️ 仍在排查：lane i 与 lane i+8 内容重叠会重复累加；换成"只用 8 倍数偏移"后又出现新的偏差（说明还有第二个错） |
| 32B 对齐（`p*nRG`、`Cast` count） | ⚠️ K=17408 时 `nRG=136` 使 `p*nRG*2=272B` 不对齐 → 已确认会崩（aicore exception），引擎侧暂只对 K≤8192 开放 |

> ⚠️ 该版**默认关闭**（需 `ENG27_GEMV_V2=1`），因为数值尚未与 v1 等价。
> 保留在仓库里作为性能研究基线，**不要在生产路径启用**。


---

## 4. 已跑通：Cube 两段式（prefill 场景 **23x**，实测）

**这是本轮最重要的结果**：让 20T NPU 的 Cube 真正参与计算，而不是只用 Vector 核做 GEMV。

```
W4(int4) --dequant_w4(AIV)--> fp16 写 GM --aclnnMatmul(AIC/Cube)--> out[M][N]
```

实测（K=5120, N=17408, M=32 个 token 一次算，板端）：

| 路径 | 耗时 | 每 token | 加速 |
|---|---|---|---|
| 向量 GEMV × 32 | 1743.8 ms | 54.5 ms | 1.00x |
| **两段式**（dequant 49.8 + Cube matmul 25.1） | **75.0 ms** | **2.34 ms** | **23.26x** |

数值：`max_abs=0.0078`，`mean_abs=0.00001`，`mean|ref|=0.397`，`n(|d|>0.05)=0/557056`
→ 与向量 GEMV 的差异只有 fp16 舍入量级 ✓

**为什么这么快**：同样的权重流量，向量 GEMV 是"每 byte 0.5 次乘加 + 一堆指令开销"，
两段式是"dequant 一次（1.25 亿元素）+ Cube 一次矩阵乘（1.78 TFLOPS 级）"，
而且 dequant 的 4x 写放大被 M 个 token 摊薄：**M=32 时每 token 只需 0.14 B/param**（向量路径是 0.5）。

**代价与边界**
- 两段式流量 4.5 B/param（读 int4 + 写 fp16 + 读 fp16），M=1 时反而比向量 GEMV 慢 9 倍 → **只适合 prefill（M≥8）**；
- 需要一块 fp16 权重 scratch：最大矩阵 178 MB（K=5120、N=17408）✓ 显存够；
- 本原型是**单矩阵级**验证；端到端 prefill 需要引擎改成 layer-major 批量走（见 §5 #1）。

---

## 5. decode 的 5–10x 路线：Cube 融合反量化（P0，未做）

两段式对 decode（M=1）无效（4.5 B/param），decode 要用**融合**版本，让 fp16 展开只留在片上：

```
GM(int4) --MTE2--> UB --反量化(AIV)--> L1(NZ) --LoadData--> L0B --MMAD(AIC)--> L0C --Fixpipe--> GM
```

- GM 流量仍是 **0.5 B/param**（带宽地板 1.32 s/token，可望吃掉 70% 带宽）；
- 反量化向量开销 ≈ 0.78 s/token，与 MMA 在 AIC/AIV 上**并发重叠**（`cube_vector_combine=fuse,split`）；
- M=1 时把激活复制成 16 行喂 Cube，算力浪费 16 倍仍只需 ~86 ms/token → 不是瓶颈；
- 预期 decode **2–3 s/token（5–7x）**；prefill 用同一 kernel + M 批 → 更高；
- 代价 2–3 周：自推 NZ/fractal 排布 + 手写 `LoadData/Mmad/Fixpipe` 流水；风险是数值错位（可对拍发现）。

可行性依据：`probe_l1`（UB→L1 可用）、`cube_matmul`/`matmul_tiling`（Cube 链路通）、
`dequant_w4`（本项目 W4 格式的反量化已跑通）、`docs/310B-性能上限与prefill优化.md` §6.3。

---

## 6. 低风险次级优化（按性价比）

| # | 手段 | 预期 | 状态 |
|---|---|---|---|
| 1 | **prefill 端到端接两段式 Cube**（layer-major 批量：每层投影按 M 个 token 一起算） | prefill 0.3–0.7 s/token（20–40x） | **下一步**：kernel + 单矩阵级已验证（§4），还差引擎重构 |
| 2 | 合并指令版 GEMV（`w4a16_gemv_gen2`，与 gen 逐位相同） | **1.18x（已落地，默认启用）** | ✅ 已上线 |
| 3 | 去掉同流水 `PipeBarrier<V>` | 1.7%（已落地） | ✅ 已上线 |
| 4 | 行批处理版调试到数值等价（1.5–1.6x 已测到） | 1.5x | ⏸ 数值未等价，默认关闭 |
| 5 | MTP / 投机解码 | 1.5–2.5x | 未做 |
| 6 | 两段式与 AIV 计算**流水重叠**（dequant 矩阵 i+1 同时 MMA 矩阵 i） | +20–40% | 未做 |
| ~~7~~ | ~~多核并行~~ | **不可行**（`ai_core_cnt=1`） | — |

---

## 7. 本次新增的工具（可直接复用）

| 文件 | 用途 |
|---|---|
| `code/qwen38_310b/ops/kernels/probe_v2.cpp` | 向量原语开销探针（9 模式 × 4 长度，两次 iters 相减剔除 setup） |
| `code/qwen38_310b/ops/host/main.cpp` → `probe_v2` / `gemv_batch` | 板端跑法：`./out/bin/qwen38_ops_test probe_v2`、`gemv_batch`（新旧 GEMV 对拍+计时，`GEMVB_K/GEMVB_N` 可覆盖形状） |
| `code/qwen38_310b/ops/kernels/w4a16_gemv_batch.cpp` | 行批处理实验内核（1.61x，数值未等价，默认关闭） |
| `code/qwen38_310b/ops/kernels/w4a16_gemv_tok.cpp` | prefill 按 token 批处理内核草案（未接线） |
| `board/*.ps1`、`board/*.sh` | 板端推送（scp 二进制安全）/增量编译/清理/测速脚本 |

---

## 8. 复现命令

```powershell
# PC 侧（D:\项目4\orangepi_aipro）
& .\board\spush.ps1 -Map @("ascendc_ops\kernels\..=~/code/qwen38_310B/src/kernels/..")   # 推送
powershell -File board\brun.ps1 -Script board\build.sh                                  # 增量编译
powershell -File board\brun.ps1 -Script board\run_gemvb.sh                              # GEMV 对拍+计时
```

```bash
# 板端：GEMV 单矩阵
cd ~/code/qwen38_310B && source env.sh && cd src
./out/bin/qwen38_ops_test gemv_batch                    # 默认 K=5120 N=17408
GEMVB_K=17408 GEMVB_N=5120 ./out/bin/qwen38_ops_test gemv_batch
# 端到端（v1 基线 / v2 实验）
ENG27_NOWARMUP=1 ENG27_IDS=<ids> ENG27_GEN=4 ./out/bin/qwen38_ops_test engine_27b
ENG27_NOWARMUP=1 ENG27_GEMV_V2=1 ENG27_IDS=<ids> ENG27_GEN=4 ./out/bin/qwen38_ops_test engine_27b
```
