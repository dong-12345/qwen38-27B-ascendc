# LingBot-Depth 深度补全 · OrangePi AI Pro 20T（昇腾 310B1）调研 / 5 TOPS 预算 / 部署实测

> 平台：OrangePi AI Pro 20T（Ascend 310B1，CANN 8.0.0，npu-smi 25.2.0，Ubuntu 22.04 aarch64）
> 约束：**只允许用 5 TOPS（板子标称 20 TOPS 的 1/4）**；强实时目标 **30 fps（33.3 ms/帧）**
> 数据：海康 MV-DT01SDU U3V ToF（深度 640×480 C16@30fps + RGB 1536×1280 JPEG），真机抓帧见 `data/`

---

## 0. 一句话结论

**在只用 5 TOPS 的前提下，LingBot-Depth（ViT-L/14，321 M 参数）在这块板上完全不可行**：最小可配置 462 GFLOP/帧，板上实测单个 ViT block 就要 333 ms，整模型约 **8 秒/帧（0.1 fps）**，离 30 fps 差 **两个数量级**；能过 30 fps 强实时的只有「**经典多尺度填充（CPU 16.5 ms / 60 fps）**」和「**DAv2-Small 168×224（NPU 19.4 ms / 51.5 fps）**」这两档；而 RGB 引导的补全还受制于**本相机 RGB-D 未做像素级标定**（厂商 ImageAlign=0）——**真正的瓶颈是标定，不是 5 TOPS 算力**。

---

## 1. 调研结论：LingBot-Depth 是什么 / 有没有更小版本

从官方 checkpoint（`model.pt`，HF/ModelScope `robbyant/lingbot-depth-pretrain-vitl-14-v0.5`）实读的模型配置与结构：

| 项 | 值 |
|---|---|
| 骨干 | `dinov2_vitl14`（DINOv2 **ViT-L/14**，24 blocks，embed 1024，16 heads，patch 14） |
| RGB 输入 | `PatchEmbed`（3ch→1024），patch 位置编码 + modality embedding=1 |
| 深度输入 | 独立 `PatchEmbed`（1ch→1024，`depth_emb_mode=conv_1c`），modality embedding=2，深度先归一化到米再取 log，缺失置 0 |
| Token 序列 | `1 cls + N·RGB + N·depth`，`N = H·W/14²`，双模态 token **拼接**（`img_depth_fuse_mode=cat_token`） |
| 中间层 | `intermediate_layers=1`（**只取最后 1 层**）→ 1024ch 1×1 投影 |
| 解码器 | ConvStack neck `dim_res_blocks=[1024,256,128,64,32]` + `depth_head` + `mask_head`（`remap_depth_in=log`、`remap_output=exp`、`remap_depth_out=linear`） |
| 参数量 | **321.16 M**（backbone 305.62 M + decoder 15.54 M），与模型卡 "~300M" 一致 |
| 权重体积 | fp32 **1.28 GB** → fp16 **642 MB** → INT8 **321 MB**（**都超过板子 300 MB 推送预算**） |
| 分辨率 | 可调：`num_tokens_range=[1200,3600]`，`resolution_level 0–9`（默认 9 → 3600 tokens → ViT 输入 728×966） |

**重点问题的答案**：
1) **没有更小的骨干/配置**：代码里虽然有 `dinov2_vits14 / vitb14` 的构造器，但**官方只发布了 ViT-L 权重**（Model Zoo 三个模型全是 VitL-14）；**没有蒸馏、没有量化、没有 ONNX 导出脚本**（仓库只有内部 `onnx_compatible_mode` 开关，且官方推理路径依赖 xFormers 嵌套张量）。输入分辨率**可调**（见上），但 token 数、双模态输入是架构硬约束。
2) **算力需求**：见 §2 折线表 —— 321 M 参数、462 GFLOP ~ 10.2 TFLOP/帧。
3) **依赖**：必须双输入（RGB + 稀疏/原始深度）；深度以 **米→log→独立单通道 patch embed** 编码；**patch 内全缺失的 depth token 在推理时会被 mask 掉**（依赖 xFormers），因此固定形状导出时必须关闭该 masking（本工作 `enable_depth_mask=False`，与 ONNX/ATC 兼容；我们的 ToF 帧 67% 像素有效，被 mask 的 patch 很少）。

> 复现：`analysis/flops_budget.py`（参数/FLOPs/预算是从 checkpoint 实读配置计算，decoder 用 torch FLOP counter 实测后按像素数缩放）。

---

## 2. 5 TOPS 预算分析（口径说清楚）

**5 TOPS 的定义**：310B1 板卡标称 20 TOPS（INT8 理论峰值，且要跑满所有 AI core + Cube）；用户只给 **5 TOPS**。本板的前期实测（单 AI core、Cube 高层 API）：
- **fp16 峰值 1.46 TFLOP/s**（`ai_core_cnt=1`，无多核）
- GM 读带宽 10.2 GB/s、UB 248 KB、L2 4 MB

因此在"5 TOPS / 单 AI core"口径下，**30 fps 的每帧预算**：

| 口径 | 每帧可用算力 | 30 fps 每帧 FLOPs 预算 |
|---|---|---|
| 官方 5 TOPS（INT8 理论） | 5.0 TFLOP/s | **167 GFLOP** |
| 实测 fp16 峰值 | 1.46 TFLOP/s | **48.7 GFLOP** |
| 实测 ViT 类负载有效算力（本工作实测 0.25–0.60 TFLOPS） | 0.25–0.60 TFLOP/s | **8.3–20 GFLOP** |

**LingBot-Depth 的 FLOPs / 延迟表**（输入按 640×480 的 4:3 比例；`L=1+2N`）：

| num_tokens | patch grid | ViT 输入 | L | enc GFLOP | dec GFLOP | **总 GFLOP** | 理论@1.46TF | 实测/外推 | fps | 30fps? | 超"实测预算(40 GFLOP)"倍数 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 300 | 15×20 | 210×280 | 601 | 399.6 | 62.2 | **461.8** | 316 ms | **~600–730 ms**（外推） | 1.4–1.7 | ❌ | 11.5× |
| 600 | 21×28 | 294×392 | 1177 | 849.2 | 121.8 | **971.1** | 665 ms | ~2.0–2.3 s（外推） | ~0.5 | ❌ | 24× |
| 1200 | 30×40 | 420×560 | 2401 | 2021.3 | 248.6 | **2269.9** | 1.55 s | **~8.5–9.0 s**（实测 block ×24） | 0.11–0.12 | ❌ | 57× |
| 2400 | 42×57 | 588×798 | 4789 | 5155.9 | 496.0 | **5651.9** | 3.87 s | ~22–25 s（外推） | 0.04 | ❌ | 141× |
| **3600（默认）** | 52×69 | 728×966 | 7177 | 9411.6 | 743.4 | **10155** | 6.96 s | **~70 s**（外推，且 attention 张量 1.65 GB 已接近不可行） | ~0.014 | ❌ | 254× |

**板上实测（关键证据，单 block 实测后 ×24）**：

| 被测单元 | FLOPs | 板端实测 p50 | 实测有效算力 |
|---|---|---|---|
| 1× ViT-L block，L=1177（=600 tokens） | 35.29 GFLOP | **72.13 ms** | **0.489 TFLOPS** |
| 1× ViT-L block，L=2401（=1200 tokens） | 84.04 GFLOP | **334.72 ms** | **0.251 TFLOPS** |
| 24 blocks @L=2401 + decoder | 2270 GFLOP | 24×334.7 ms + ~0.5–1.0 s ≈ **8.5–9.0 s** | ≈0.25 TFLOPS |

> 测法：把官方 ViT-L block（LayerNorm+qkv+attention+proj+MLP+GELU+LayerScale，数值与官方一致，SDPA 换成显式 matmul 以导出 opset13）导出 ONNX → 板上 `atc --framework=5 --soc_version=Ascend310B1 --precision_mode=force_fp16` → pyACL 执行 30 次取 p50。OM 25.4 MB（可推送）。单 block 二次拟合 `t ≈ 5.5e-5·L² ms`（L=1177→72 ms、L=2401→335 ms 两点），默认 L=7177 外推 2.8 s/block。
>
> **注意**：ViT 类负载的有效算力只有 **0.25–0.49 TFLOPS**（远低于 1.46 TFLOP/s 峰值）——原因是 attention/LayerNorm 访存密集、算子是碎块。这进一步坐实"5 TOPS 是理论标签"。

**结论**：LingBot-Depth 最小配置（300 tokens，210×280 的 ViT 输入，输出上采到 640×480）仍要 **~0.6–0.7 s/帧**，只差 20 倍就到 30 fps；默认配置差 ~2000 倍。**降分辨率救不回来**（FLOPs 中 attention 项 ∝L²、MLP ∝L，即便降到 210×280 仍是 462 GFLOP）。

### 2.1 「600–730 ms」这个数字的来历与口径（重要）

**它不是按 20 TOPS 算的**，而是**板上实测单 block 延迟的外推**：

1. 板上实测（OM + pyACL，Ascend 310B1 单 AI core）：ViT-L block 在 `L=1177` → **72.13 ms**（35.29 GFLOP，0.489 TFLOPS）、`L=2401` → **334.72 ms**（84.04 GFLOP，0.251 TFLOPS）；二次拟合 `t≈5.5e-5·L² ms`。
2. 最小配置 `num_tokens=300 → L=601`：单 block ≈ 19.9 ms → 24 blocks ≈ 476.8 ms。
3. decoder 62.2 GFLOP，按实测量级 0.25–0.5 TFLOPS → 124–249 ms。
4. 合计 **601–726 ms**（1.4–1.7 fps）。**这是"实测单元 × 24 + decoder 估算"，不是端到端实测**——因为 321 M 参数 fp16 权重 642 MB / INT8 321 MB，都超过板上 300 MB 推送预算，整模型 OM 无法上板，只能按 block 测。

同一 461.8 GFLOP 在不同口径下的延迟（用来对照）：

| 口径 | 每帧 | fps | 说明 |
|---|---|---|---|
| **20 TOPS**（板卡标称，INT8 全核理论） | 23.1 ms | 43.3 | 理论天花板，不可达 |
| **5 TOPS**（用户预算；INT8 MAC 口径 5e12 MAC/s） | 46.2 ms（若按 FLOPs/5e12 保守口径 92.4 ms） | 21.7（保守 10.8） | **即使跑满 5 TOPS 也不满足 30fps**（30fps 需 ≥6.93 TOPS 有效） |
| 单 AI core fp16 峰值（实测 1.46 TFLOPS） | 316 ms | 3.2 | 只有大矩阵乘才接近 |
| 实测 ViT 有效算力 0.49 TFLOPS（L=1177 点） | 943 ms | 1.1 | |
| 实测 ViT 有效算力 0.25 TFLOPS（L=2401 点） | 1847 ms | 0.5 | |
| **报告采用：实测 block 外推（二次拟合）** | **601–726 ms** | **1.4–1.7** | 偏乐观端（短 L 时 attention 占比降为 9%，效率假设会改善） |

> 所以口径要说清楚：**按 20 TOPS 理论是 23 ms（43 fps）；按 5 TOPS 理论是 92 ms（10.8 fps）；按板上实测的 ViT 有效算力是 0.9–1.8 s；报告表里的 0.6–0.7 s 是"实测单 block 外推"的乐观值**。三种口径都远离 30 fps（33.3 ms），结论一致。默认配置（3600 tokens，10.2 TFLOP）按同法外推 ≈68–75 s/帧。

**量的边界（用户要求的口径）**：
- **ViT-L 640×480 原生**（patch grid 34×45=1530 tokens → L=3067）：**3351 GFLOP/帧**（encoder 2820 + decoder 531）→ 按实测 0.25 TFLOPS = **~13.4 s/帧（0.075 fps）**；按理论峰值 1.46 TFLOPS = 2.3 s（0.43 fps）。
- **ViT-L 320×240**（grid 17×23=391 tokens → L=783）：**614 GFLOP/帧** → 实测 ~2.5 s/帧（0.4 fps），理论峰值 0.42 s（2.4 fps）。
- 即使做 **INT8**（把 5 TOPS 全用上、算力 ×2）：320×240 也只有 **~0.8 fps**（实测口径）/4.8 fps（理论口径），640×480 ≈ 0.15 fps；**仍差 6–40 倍**，且 INT8 权重 321 MB 已超 300 MB 传输预算、ATC 量化还需校准集。
- 要实现 30 fps，LingBot-Depth 类 ViT-L 补全需要 **≈ 40–100 TFLOP/s 级有效算力**（本板实测有效算力的 70–200 倍）：不是"再省一点算力"能解决的，必须换架构（CNN/小 ViT）或等官方蒸馏小模型。

---

### 2.2 INT8 量化能救 LingBot-Depth 吗？**不能**（只能快 1.3–2×，救不了 30×的差距）

**先算硬账**：最小配置 461.8 GFLOP/帧 = **230.9 GMAC/帧**；30 fps 需要 **13.9 TFLOPS = 6.93 TOPS(MAC) 的有效算力**——已经**超过你给的 5 TOPS 预算 1.4 倍**。默认配置 10.2 TFLOP/帧，30 fps 需要 **152 TOPS(MAC)**，是整块板 20 TOPS 的 7.6 倍。

| 口径 | 最小配置（300 tok, 462 GFLOP） | 默认（3600 tok, 10.2 TFLOP） |
|---|---|---|
| 20 TOPS 满效率（INT8 全板标称，理论） | 11.5 ms → 87 fps | 254 ms → 3.9 fps |
| **5 TOPS 满效率（你的预算，理论）** | **46 ms → 21.7 fps（仍不到 30fps）** | 1.02 s → 1.0 fps |
| 保守口径（把 TOPS 当 FLOPs/s） | 92 ms → 10.8 fps | 2.0 s → 0.5 fps |
| 实测 fp16 ViT 效率 0.25–0.49 TFLOPS | 0.94–1.85 s → 0.5–1.1 fps | 21–41 s |
| **INT8 实测换算（fp16 效率 ×1.3–2.0）** | **0.47–1.42 s → 0.7–2.1 fps** | **10–31 s** |

**为什么 INT8 加速远小于 2×**：ViT-L 的时间大头是 LayerNorm/GELU/softmax/residual（元素级、访存受限）和超大 attention 中间张量读写（这些仍保持 fp16，不因权重 INT8 而减半）；只有 qkv/MLP 的 MatMul 能吃到 INT8 Cube 的 2× 峰值。实测本板 ViT 类负载 fp16 只有 0.25–0.49 TFLOPS（=20 TOPS 的 1.25–2.5%），INT8 后按 1.3–2× 折算也只有 0.33–0.98 TFLOPS。

**工程代价也不小**：
- 权重 INT8 = **321 MB**，加上量化参数/OM 头，仍**超过 300 MB 推送预算**；
- ATC **不能**直接把 fp32 ONNX 变成 INT8 OM，必须先用 **AMCT 离线量化**（QDQ），而且要 **100–300 帧真实 RGB-D 校准集**；ViT + log-depth 回归对量化敏感，LayerNorm/softmax 必须留 fp16，精度需重新评估；
- 即便做完，帧率也只从 0.6–0.7 s 降到 0.47–1.4 s，**离 33.3 ms 差 15–50 倍**。

**INT8 对当前可部署方案才有价值**：
- DAv2-S **238×308**：fp16 实测 49.9 ms → INT8 估 25–35 ms（30–40 fps）；
- DAv2-S **168×224**：fp16 实测 19.9 ms → INT8 估 10–15 ms（65–100 fps）；
- 经典填充（CPU）不受影响（16.5–23.5 ms）。
- 注意：DAv2 的"清晰度只有 27% 梯度"问题**不是量化能解决的**，INT8 只是让已有小模型更快。

**结论**：INT8 只是 1.3–2× 的"打补丁"，**要 30fps 必须把每帧 FLOPs 降 25–50 倍（换架构/蒸馏）**。参考量级：若官方出 ViT-S(12×384) 版 MDM，在 210×280 上约 32 GFLOP，INT8、5 TOPS 理论 3–4 ms，按 ViT 实测效率 15–25% 折算 ≈13–22 ms → **46–77 fps**——这才叫"能上 30fps"，但**目前没有这个权重**（只能等官方或自己蒸馏，属于另一个训练项目）。

---

## 3. 三档方案对比（**板上真机实测**）

| # | 方案 | 输入 | 算力需求 | 板上实测（p50/p95） | fps | 30fps 强实时 | 需要 RGB-D 标定? | 真 ToF 打洞精度（mm） |
|---|---|---|---|---|---|---|---|---|
| **A** | **经典多尺度填充**（cv2 push-pull，CPU 单核，无 NPU） | ToF 640×480 | ~0 NPU；CPU ≈17 ms 计算 | **16.49 / 16.79 ms** | **60.7** | ✅ | 不需要 | 随机 10% 洞 RMSE 288/305、MAE 103/128；45×45 块洞 RMSE 1794/2104、MAE 1338/1421 |
| **B** | **DAv2-Small 168×224**（11.71 GFLOP） | RGB 或 深度渲染图 | 11.7 GFLOP @ **0.603 TFLOPS** | **19.41 / 21.57 ms** | **51.5** | ✅（余量 1.7×） | 深度渲染输入不需要；RGB 输入需要 | 见 §4（标定好时大洞明显优于 A） |
| **B'** | DAv2-Small 238×308（23.0 GFLOP） | 同上 | 23.0 GFLOP @ **0.459 TFLOPS** | 50.09 / 51.23 ms | 20.0 | ❌（差 1.5×） | 同上 | 见 §4 |
| **C** | **LingBot-Depth ViT-L** | RGB+ToF | 462 GFLOP ~ 10.2 TFLOP | 单 block 334.7 ms ×24 ≈ 8.5–9 s（1200 tokens）；默认 ≈70 s | 0.014–0.12 | ❌❌ | 需要 | PC 参考跑通（见 §4），板上跑不动 |

**每帧预算视角（领任务要求）**：

| 模型 | FLOPs/帧 | 板上实测有效算力 | ms/帧（实测/外推） | fps | 是否 ≤33.3ms | 占 5 TOPS 理论预算(167 GF) | 占实测可用预算(49 GF) |
|---|---|---|---|---|---|---|---|
| 经典填充 A | ~0.02 G（访存型） | CPU 1 核 | **16.5** | **60.7** | ✅ | 0% | 0% |
| DAv2-S 168×224 | 11.71 G | 0.603 TF | **19.4** | **51.5** | ✅ | 7% | 24% |
| DAv2-S 238×308 | 23.0 G | 0.459 TF | **50.1** | 20.0 | ❌ | 14% | 47% |
| LingBot 300 tok | 462 G | 0.25–0.49 TF | ~601–726（实测block外推） | 1.4–1.7 | ❌ | 277% | 944% |
| LingBot 1200 tok | 2270 G | ~0.25 TF | ~8500–9000 | 0.11 | ❌ | 1360% | 4630% |
| LingBot 3600 tok | 10155 G | ~0.25 TF | ~70000 | 0.014 | ❌ | 6080% | 20700% |

> 实测环境说明：真机测试期间板子同时在跑用户 15 GB 权重传输（load average 17–30，系统 CPU 忙 44–75%），**CPU wall 时间被外部负载放大**（同一份填充从 17 ms 到 280 ms 都出现过）；表中 CPU 数取独立忙等基准（`deploy/bench_fill.py`，process CPU 17.57 ms、占单核 ~107%）。**NPU 推理时间不受影响**（多次复测偏差 <2%）。
> e2e（预处理+PIL resize+NPU+后处理）在外部负载下实测 0.74–0.94 s，不可信；按 NPU 19–50 ms + CPU 工作 ~25 ms 估计，**真实 e2e 约 45–80 ms（12–22 fps，取决于 DAv2 分辨率）**。

DAv2 OM 转换踩坑记录（`convert/`）：
- CANN 8.0 的 Onnx Resize 只支持 `align_corners / pytorch_half_pixel / asymmetric` 三种 `coordinate_transformation_mode`，官方 ONNX 的 `/Resize` 是 `half_pixel`+`cubic` → 用 `onnxsim` 固定静态形状（残差 Resize 全变 `align_corners`）+ 把 `coordinate_transformation_mode` 改写为 `asymmetric` 后转换成功（OM 52.3 MB，238×308 6 min19s / 168×224 约 6 min）。
- 预处理全部在 CPU（PIL 双线性 resize → ImageNet 归一化 → NCHW fp32），OM 只含纯模型；ATC 保留 ONNX 的 **fp32 I/O**（未加 `--input_fp16_nodes`），ACL 侧按 fp32 读写。
- 精度核对：fp16 OM 输出与 fp32 ONNXRuntime 参考的相关系数 **0.61、拟合后 RMSE 0.64**（DAv2 输出量纲），存在明显数值漂移（fp16 attention/LayerNorm 累计误差）；**延迟数据不受影响**，但先验质量按此打折。可改用 `--precision_mode=allow_mix_precision` 重转（本机转 15+ min 未跑完，留给后续）。

---

## 4. 精度评估（真 ToF 数据打洞回填，GT = 原始 ToF 深度）

协议（无外部真值，用 ToF 自身做 GT）：把 ToF 有效像素随机 10% 或 4 个 45×45 块状区域挖掉 → 用剩余深度（+可选 RGB 先验）补全 → 与挖掉前的真值比 RMSE/MAE（mm）。**挖洞和填充都在 640×480 原始分辨率上评估**。

### 4.1 真 ToF 帧（`real_tof` 67.4% 有效 / `tof_snap2` 77.4% 有效）

| 场景 | 洞型 | 经典填充 RMSE/MAE | DAv2 先验（RGB 输入） | DAv2 先验（**深度渲染图输入**） |
|---|---|---|---|---|
| real_tof | 随机 10% | **305.0 / 128.4** | 1452.3 / 852.9 | 1216.3 / 791.9 |
| real_tof | 45×45 块 | 2104.4 / 1420.6 | 1983.9 / 1128.9 | **1695.2 / 1118.5** |
| tof_snap2 | 随机 10% | **287.5 / 103.1** | 1425.3 / 751.9 | 1133.0 / 631.1 |
| tof_snap2 | 45×45 块 | 1794.5 / 1337.6 | 1799.2 / 1005.9 | **1227.3 / 753.4** |

→ 在**未标定**的 ToF 上：小洞经典填充最好；大洞"深度渲染图 + DAv2"略优（MAE 降 ~20%），RGB 先验无优势（§5）。

### 4.2 标定良好的公开 RGB-D 样例（`ex1`/`ex4`，RGB 与深度像素级对齐）

| 场景 | 洞型 | 经典填充 | DAv2 先验（RGB） | DAv2 先验（深度渲染） |
|---|---|---|---|---|
| ex1 | 随机 10% | **15.3 / 5.3** | 118.6 / 83.3 | 150.4 / 110.4 |
| ex1 | 45×45 块 | 494.5 / 376.6 | **188.5 / 115.0** | 173.2 / 130.7 |
| ex4 | 随机 10% | **4.3 / 3.0** | 100.8 / 62.2 | 146.0 / 123.9 |
| ex4 | 45×45 块 | 908.8 / 691.6 | **46.0 / 34.2** | 111.2 / 95.2 |

→ **RGB-D 标定好时，学习先验在大洞上碾压经典填充**（ex4 大洞 RMSE 46 mm vs 908 mm，**20×**；ex1 2.6×），小洞仍是经典填充更准。**这就是"RGB 引导补全值不值得"的分水岭**。

### 4.3 LingBot-Depth 参考质量（PC 端 CPU 跑通，板上不可行）

- 真机 ToF 帧 + RGB（resize 到 640×480）输入，`enable_depth_mask=False`：**600 tokens → 6.29 s/帧**，2400 tokens → 39.3 s/帧（16 核 CPU，6 线程）。
- 输出 vs 原始 ToF 的 hold-out 误差：600 tokens RMSE 956.8 mm / MAE 415.2 mm；2400 tokens RMSE 1348.6 / MAE 575.1（**注意这是"去噪/平滑 vs 粗糙 ToF"的差异，不是对 GT 的误差**；本像素对未标定 + 低 token 数，数值仅供参考）。
- 覆盖：refined 有效像素 99.7%（原始 67.4%），确实补满。
- 对比图：`results/pc_ref/real0001_t2400_comparison.png`（左 raw ToF / 右 LingBot 输出）。

PC 端另一组交叉验证（`analysis/eval_completion_pc.py`，ONNXRuntime 参考实现，与板端结果同量级）：随机 10% 洞 经典填充 503.7 mm 最优、DAv2 深度渲染 1186.0 mm；45×45 块 DAv2 深度渲染 1666.8 mm 优于经典填充 2144.5 mm。

### 4.4 「DAv2 先验-RGB」方法说明与失效机理（本相机为何不可用）

**定义（代码路径 `deploy/run_tof_pipeline.py`）**：`prior_rgb` 不是像素级 RGB-D 融合，而是：

1. 整幅 RGB（1536×1280）**不做任何配准**，直接 resize 到 308×238 送 DAv2-Small（.om, fp16, 23.0 GFLOP）→ 相对逆深度 prior；
2. prior 上采样到 640×480（与 ToF 同尺寸，但**只有"整幅图像"这一层对应关系，没有像素级空间对齐**）；
3. 在 ToF **有效像素**上用 IRLS 鲁棒拟合全局仿射 `1/z_tof ≈ a·prior + b`（8 次迭代、70% 内点裁剪、2 万点子采样）；
4. 打洞点的补全值 = `1/(a·prior + b)`；`prior_depth` 同流程，只是 DAv2 输入换成 **ToF 深度的 3 通道灰度渲染图**（洞=黑），因此天然像素对齐。

**失效机理（可量化）**：拟合出的斜率 `a` 是"先验有没有用"的直接指标——

| 场景 | RGB 先验 a / b | 深度渲染先验 a / b | 大洞 RMSE（RGB 先验 vs 经典填充） |
|---|---|---|---|
| real_tof（**未标定 ToF**） | **0.0109** / 0.466 | 0.1184 / 0.1998 | 1983.9 vs 2104.4 mm |
| tof_snap2（**未标定 ToF**） | **0.0073** / 0.4768 | 0.2013 / 0.0922 | 1799.2 vs 1794.5 mm |
| ex1（**标定良好**） | 0.2630 / 0.3744 | 0.2917 / 0.4764 | 188.5 vs 494.5 mm |
| ex4（**标定良好**） | 0.0806 / 0.3396 | 0.1969 / 0.1623 | **46.0** vs 908.8 mm |

- 未标定 ToF 上 a≈0.007–0.011 ⇒ 先验被压成**近乎常数深度场**（real_tof：prior∈[0.4,4.1] 映射后 z∈[1.96,2.13] m），对打洞点几乎不携带信息 ⇒ RMSE 1.2–1.45 m。这是"空间不相关 ⇒ 最小二乘最优解退化为均值"的必然结果。
- 标定良好的 ex1/ex4 上 a 放大 8–24 倍，先验真正携带结构 ⇒ 大洞 46–188 mm（比经典填充好 **2.6–20×**）。
- 附加误差源：fp16 OM 相对 fp32 参考有数值漂移（corr 0.61），但这不是主因（主因是 a≈0）。
- 为什么小洞（随机 10%）反而经典填充更准：随机洞的邻居几 mm 内就有真值，插值误差 ~mm 级，而单目先验本身有 cm 级全局残差 + 平滑偏置（ex4：4.3 mm vs 100.8 mm）。
- **结论**：(i) 本相机未标定前，**RGB 先验路线不可用**，应该改用「深度渲染输入」的先验（无对齐问题，大洞 MAE 改善约 20–40%）或纯经典填充；(ii) 标定后（`ImageAlign=1` 或标定板外参 warp RGB 到深度帧）再启 RGB 先验才有意义；(iii) 表中"先验-RGB"只是**对照实验**，不是最终交付方案——交付档位是 A（经典填充，CPU）与 B（DAv2-S 168×224，NPU，未标定时用深度渲染输入）。

---

## 5. 关键发现：当前 30fps 数据通路没有 RGB-D 对齐（**更正：原厂有标定能力**，见 §9）

- 相机输出 RGB **1536×1280** 与 ToF 深度 **640×480** 两路传感器。我们当前为了 30 fps 走的是 **U3V 直连**（绕过厂商 SDK、`ImageAlign=0`），拿到的是**未对齐**的原始 RGB + 深度。
- 对未对齐数据：RGB-D 边缘相关性粗配准最高只有 **0.16**（scale 0.55 + hflip + 平移，且宿舍场景左右对称导致翻转歧义）；用 DAv2 深度做"深度-深度"配准可到 **0.37**，但最优变换各向异性（sx=1.48, sy=0.82）→ 物理上不成立，**无法用全局相似变换对齐**。
- 结果：**RGB 引导的补全在当前数据上精度不可用**（§4.1 vs §4.2 对比明显）；同一模型在标定好的 RGB-D 上大洞误差好 5–20 倍。
- **重要更正**：这并不是"原厂没标定"。原厂（海康机器人）**每台出厂带 RGBD 标定文件**，SDK 提供 `MV3D_RGBD_INT_IMAGEALIGN` 对齐开关、`RGBDINNERPARAM`（深度→RGB 4×4 变换 + RGB 内参 + 畸变）；只是**我们为 30fps 绕过了 SDK，没用它的对齐输出**。落地路线见 §9。

---

## 6. 成本 / 精度 / 延迟权衡与建议

| 路线 | 延迟 | 质量 | 代价/风险 | 建议 |
|---|---|---|---|---|
| 纯深度 + 经典填充（CPU） | 16.5 ms（60 fps） | 小洞好、大洞一般 | 0 算力占用、无需标定 | **默认推荐**；NPU 完全留给别的任务 |
| 深度渲染 + DAv2-S 168×224（NPU） | 19.4 ms（51.5 fps）+ 填充 ~5–17 ms | 大洞比经典填充好 ~20% | 需 fp16 数值兜底、模型 52 MB OM | 大洞多（玻璃/反光）时叠加使用 |
| RGB + DAv2-S 168×224 | 19.4 ms | **标定后**大洞最好（5–20×） | **必须先做 RGB-D 标定** | 标定完成后推荐 |
| LingBot-Depth ViT-L | 8.5–70 s/帧 | 理论上限最高 | 321 M 参数、642 MB fp16、需 xFormers、5 TOPS 差 2 个数量级 | **本板不可行**；需 ≥10–40 TFLOP/s 级算力或官方蒸馏小模型 |

**一句话**：5 TOPS 上"强实时深度补全"完全可行，但可行的是 **十几 ms 级的小模型/经典算法**，不是 LingBot-Depth 这种 300 M 参数的 ViT 基础模型；**LingBot 类方案要落地，要么换算力（≥50×），要么等官方蒸馏/量化小版本，要么先用 5 TOPS 做"深度单模态补全 + 经典填充"，把 RGB 大模型留给别处**。

---

## 7. 最新真机帧自然空洞对照（15:29 抓帧，**无人工打洞**）

复抓一帧当前画面（`data/live_1529/`，`u3v_lowlat.py` 直连 U3V），用 **真实缺失像素**（不是合成洞）做补全对照：

- 采集：设备 **30.05 fps**、取流 CPU 6.3%、帧间隔 p50 32.4 ms（30 fps 稳定）
- 该帧：深度 640×480，**有效像素 86.8%（真实空洞 13.2% = 40480 px）**，量程 1.02–6.36 m，中位 2.08 m
- 结果图：`results/live/live1529_comparison_labeled.png`（a 当前 RGB / b 原始深度-黑洞 / c 经典填充 / d DAv2 深度渲染先验 / e DAv2 RGB 先验 / f 混合）

| 方法 | 计算位置 | 纯推理延迟 p50 / p95 | 完整调用 p50 / p95（含 resize/归一化/拷贝） | 实时性 |
|---|---|---|---|---|
| 经典多尺度填充（640×480） | CPU 1 核 | **23.5 / 26.4 ms**（min 22.9，42.6 fps） | — | ✅ 30fps（余量 1.4×） |
| DAv2-Small 168×224（11.71 GFLOP） | NPU | **19.9 / 21.5 ms**（50.2 fps） | 30.4 / 32.1 ms（32.9 fps） | ✅ 纯推理富余，整调用贴线 |
| DAv2-Small 238×308（23.0 GFLOP） | NPU | 49.9 / 51.7 ms（20.1 fps） | 66.5 / 67.6 ms（15.0 fps） | ❌ |

- 覆盖率：86.8% → **100%**（经典填充/混合都补满）。
- 自然空洞无真值，故此处只做视觉/覆盖对照；量化精度见 §4 打洞回填实验（同算法：随机洞 RMSE 140 mm、大洞经典填充 962 mm / 深度先验 280 mm）。
- 本帧观感：**经典填充（c）可用且最快**；DAv2 深度渲染先验（d）在本帧过于平滑（输入是黑洞渲染图，属 OOD），RGB 先验（e）因未标定退化为近常数；混合（f）只在大洞内部替换为先验，视觉平滑但精度提升有限、CPU 额外 ~2 ms。
- 延迟口径说明：纯推理=NPU `execute`+`sync`；完整调用额外含 CPU resize/归一化 + H2D + D2H。测试时板子有外部 15 GB 传输负载（load≈17.7），CPU 类数字比空闲时偏高（同一填充在早期独立基准为 16.5 ms）。

---

### 7.1 清晰度约束下的帧率边界（16:20 补充，回答"有足够清晰度时能到多少 fps"）

**清晰度怎么量化**：在当前最新真机帧（`live_1529`，640×480）上，只统计**原始 ToF 有效像素**（这些像素本身有真实细节），把各方法结果与原始深度比较：

| 方法 | Laplacian 强度比 | 梯度幅值比 | 与原始梯度相关系数 | 结论 |
|---|---|---|---|---|
| 原始 ToF | 1.00 | 1.00 | 1.00 | 基准 |
| 经典多尺度填充 | **1.00** | **1.00** | **1.00** | 有效像素逐像素不改动 ⇒ **清晰度 100% 保留** |
| DAv2 深度渲染先验 | 0.50 | **0.27** | **0.60** | 梯度只剩 27%、边界明显模糊（与肉眼观察一致） |
| 混合（填充+大洞先验） | **1.00** | **1.00** | **1.00** | 有效区细节保留，仅大洞内部用先验低频趋势 |

**帧率（板上实测/推算，640×480 输出）**：

| 方案（均为"只补洞、保留原生细节"） | 每帧耗时 | fps | 备注 |
|---|---|---|---|
| 纯 CPU 经典填充（不加模型） | **16.5–23.5 ms**（p95 17–26） | **42.6–60.7** | 板子空闲 16.5 ms/60.7 fps；有外部负载时 23.5 ms/42.6 fps |
| CPU 填充 ∥ NPU DAv2 168×224 先验（双分支并行，仅大洞低频用先验） | **≈25–30 ms（推算）** | **≈33–40** | 填充 23.5 ms 与 NPU 19.9 ms 并行，加 3–5 ms 融合；待板子恢复后用 `deploy/bench_parallel.py` 实测 |
| 纯 NPU DAv2 168×224（模型直接出补全） | 19.9 ms（纯 NPU）/ 30.4 ms（整调用） | 50.2 / 32.9 | 帧率够，但**清晰度只有 27% 梯度 ⇒ 不可用** |
| 纯 NPU DAv2 238×308 | 49.9 ms / 66.5 ms | 20.1 / 15.0 | 清晰度仍不足（0.5× 上采样前只有 238×308） |
| 自研 640×480 补全 CNN（**需训练权重，无现成预训练**，按实测 0.46–0.60 TFLOPS 估算） | 10 GFLOP ≈ 17–22 ms；20 GFLOP ≈ 33–43 ms；30 GFLOP ≈ 50–65 ms | 45–60 / 23–30 / 15–20 | 唯一能让"模型输出"本身清晰的路线，但要么自训要么等官方小模型 |
| LingBot-Depth ViT-L（最小配置） | 601–726 ms（实测 block 外推） | 1.4–1.7 | 不可行 |

**结论**：**"足够清晰"（保留 ToF 原生细节）的补全在这块板上可以跑到 40–60 fps（纯 CPU 填充），叠加 NPU 先验做"大洞低频修正"仍可保持约 33–40 fps**——都高于相机本身的 30 fps，所以**满足清晰度后真正的瓶颈是相机 30fps 与 USB 带宽，不是 NPU 算力**。反过来，如果让模型直接输出整幅清晰深度，当前可用的预训练小模型（DAv2-S）只有 27% 的细节，帧率再高也不满足清晰度要求。

---

## 8. 目录 / 复现

```
code/depth_completion/
├── README.md                       # 本文
├── data/                           # 真机 ToF 抓帧（real_tof, tof_snap2, live_1513, live_1529）+ 2 个公开 RGB-D 样例（ex1, ex4）
│   ├── real_tof/{depth16.png,color.jpg}
│   └── ...
├── analysis/
│   ├── flops_budget.py             # 从 checkpoint 实读配置 → 参数/FLOPs/预算表（§2 表）
│   ├── lb_infer_ref.py             # PC 端 LingBot-Depth 参考推理（600/2400 tokens）
│   ├── dav2_pc_ref.py              # PC 端 DAv2 ONNX 参考
│   ├── eval_completion_pc.py       # PC 端补全精度交叉验证
│   └── align_check.py / register_rgb_depth.py / register_prior_depth.py   # RGB-D 未标定证据（§5）
├── convert/
│   ├── export_lb_block_onnx.py     # ViT-L block ONNX 导出（用于板上单 block 实测）
│   ├── make_dav2_static.py         # DAv2 ONNX 静态化 + Resize 属性修补（CANN 8.0 兼容）
│   └── atc_convert.sh              # ATC 转换封装（IFMT/PMODE 可调）
├── deploy/
│   ├── acl_runner.py               # 通用 .om 推理 + p50/p95/p99 延迟测量（pyACL）
│   ├── run_live_compare.py          # 真实空洞（无人工打洞）单帧对照：填充/DAv2 双分辨率/混合 + 延迟
│   ├── bench_live_frame.py          # 单帧延迟基准：纯 NPU vs 完整调用 vs CPU 填充
│   ├── run_tof_pipeline.py         # 端到端：NPU DAv2 先验 + CPU 经典填充 + 融合 + 精度评估 + 可视化
│   └── bench_fill.py               # 经典填充 4 种实现板上基准（延迟/CPU 占用）
└── results/
    ├── flops_table.json            # §2 表原始数据（LingBot FLOPs/预算）
    ├── live/                        # 15:29 真机帧自然空洞对照（metrics/bench/结果图/深度 npy）
    ├── pc_ref/                     # PC 端 LingBot / DAv2 参考输出与对比图、metrics JSON
    └── board/                      # 板上实测：bench_*.txt/json、npu_smi.txt、pipeline/*.json+png
```

**板上复现命令**（`ssh HwHiAiUser@192.168.137.163`，`ssh -i tools/opi_key ...`）：
```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cd ~/code/depth_completion
# 1) OM 转换（示例：DAv2 238x308）
IFMT=NCHW bash convert/atc_convert.sh models/dav2_238x308_static_r.onnx dav2_238x308 "pixel_values:1,3,238,308"
# 2) 延迟基准
python3 deploy/acl_runner.py --model models/dav2_238x308.om --shape 1,3,238,308 --iters 50 --warmup 10
# 3) 端到端补全 + 精度 + 可视化
python3 deploy/run_tof_pipeline.py --om models/dav2_238x308.om --data data/real_tof --out results/pipeline --iters 30
# 4) 经典填充基准
python3 deploy/bench_fill.py --data data/real_tof --iters 20
# 5) ViT-L block 实测
python3 deploy/acl_runner.py --model models/lb_block_L2401.om --shape 1,2401,1024 --iters 20 --warmup 5
```

**PC 端复现**（Windows，torch 2.11 / onnx 1.18 / onnxruntime 1.22）：
```powershell
$env:XFORMERS_DISABLED="1"
$env:PYTHONPATH="...\code\depth_completion\lingbot-depth"
$env:LB_CKPT="...\model.pt"          # ModelScope/HF: robbyant/lingbot-depth-pretrain-vitl-14-v0.5 (1.28GB)
python analysis\flops_budget.py
python analysis\lb_infer_ref.py --tokens 2400 --tag real0001_t2400 --threads 6
```

---

## 9. RGB-D 标定：原厂能力与落地路线

### 9.1 原厂有什么（**有标定，且能直接出对齐图**）

- **设备出厂标定**：相机 GenICam XML 内有 `CalibrationControl`（`SensorCalibration`/`SystemCalibration` 模式、"reset the calibration files"、`TemperCalibCtrlEnable` 温补），即设备内带标定文件；深度图本身是标定后的公制深度（C16，mm）。
- **SDK 暴露 RGBD 内参**（海康机器人 VM3D 开发手册的 `RGBDINNERPARAM`）：**深度图→RGB 的 4×4 变换矩阵** + **RGB 相机 3×3 内参** + RGB 畸变 5 系数 + Depth 畸变 5 系数。SDK 也有对齐开关 `MV3D_RGBD_SetParam(h, MV3D_RGBD_INT_IMAGEALIGN, 1)`，打开后输出**深度已对齐到彩色（1536×1280）**的 RGB-D 帧。
- 本板实测证据（`tof_u3v_sdk/docs/history_vendor_sdk_analysis.md`，厂商 SDK 1.4.1.4 配置矩阵）：

| 配置（厂商 SDK） | 每帧 payload | 板上帧率 | 软触发 p50 |
|---|---|---|---|
| `ImageAlign=1`，RGB 原始 | 9.38 MiB（对齐深度 1536×1280 C16 3.93 MB + RGB8 5.9 MB） | 4.3 fps | 302.9 ms |
| `ImageAlign=1` + JPEG | 5.62 MiB/frame | 7.6 fps | 199.7 ms |
| `ImageAlign=0` + JPEG（我们当前 U3V 直连等价配置） | 2.58 MiB/frame | **30 fps** | 74–75 ms |

> 结论：**原厂标定和"深度对齐 RGB"的输出都是现成的**；但"对齐帧"数据量翻倍，而这块板子 USB 主机侧有效吞吐只有 ~40 MB/s（厂商分析文档 §8），所以 SDK 对齐模式在板上只有 4.3–7.6 fps。**30 fps 的 U3V 直连通路（align=0）天生不带对齐。**

### 9.2 推荐落地路线（保持 30 fps，一次性标定）

**方案 A（最省事）：从原厂 SDK 导出标定参数，自己在 30fps 通路上做 warp**

1. 装 Mv3dRgbd Linux SDK 1.4.1.4（板上曾装过且 `verify.sh` 19/19 PASS；当前 `/opt/Mv3dRgbdSDK` 已被移除、只剩 ldconfig 残留，需要重新下载安装；也可在 PC 的 Windows MVS 上读同一台相机）。
2. 20 行 C 程序：`MV3D_RGBD_Open` → `MV3D_RGBD_GetParam(..., MV3D_RGBD_*_INTRINSIC)`（或 HiViewer 导出）拿到 `RGBDINNERPARAM`：深度→RGB 4×4、K_rgb、畸变。
3. 离线生成映射：对 640×480 深度像素 (u,v)（z 反投影）→ 4×4 变换 → K_rgb 投影 → RGB 采样坐标；运行时按当前 z 生成 `cv2.remap` 的 map（向量化 ~2–3 ms）或只做深度→RGB 前向 z-buffer。
4. 部署：`cv2.remap`（1–2 ms）把 RGB warp 到 640×480 深度帧，然后送 DAv2/补全；总开销 <5 ms，30 fps 不受影响。
5. 验证：与 SDK `ImageAlign=1` 输出的对齐深度逐像素对比（<1 px 视为成功），或把 ToF 深度投影到 RGB 上叠加看边缘重合。

**方案 B（不装 SDK）：棋盘格自标定（OpenCV）**

1. 打印 9×6 内角点棋盘（方格 30 mm，贴刚性平板；ToF 对纯黑吸收差，建议白卡纸哑光打印）。
2. 相机固定，棋盘在 0.5–3 m、覆盖画面中心与四角，移动 **15–20 个姿态**；每个姿态同时存 RGB（1536×1280）+ 深度（640×480，`u3v_lowlat.py --snap`）。
3. RGB：`cv2.findChessboardCorners` + `cornerSubPix` + `cv2.calibrateCamera` → K_rgb + 5 畸变系数。
4. ToF：角点像素邻域中值深度 + 反投影得到 3D 角点（ToF 畸变一般可忽略；要精确可用棋盘平面在 ToF 点云拟合 + 已知方格尺寸联合优化）。
5. `cv2.solvePnP`（ToF 3D 角点 ↔ RGB 2D 角点）→ R,t（RGB←ToF）；重投影误差目标 **<1 px**。
6. 生成 warp LUT（同方案 A 第 3–4 步）。
7. 精度预期：2 m 处对齐误差 ~1–2 cm（原厂标定通常更好）。

**方案 C（不推荐）**：单应矩阵 H 近似——只对平面成立，3D 场景会随深度错位。

### 9.3 标定后的收益

- 对齐 RGB 后再跑 DAv2 RGB 先验：预期大洞精度从当前 RMSE **~1.2–1.45 m** 降到 **~50–200 mm**（参照 §4.2 标定良好样例 46–188 mm），"RGB 引导"这一档才真正可用。
- 纯深度路线（经典填充 / 深度渲染先验）**不受标定影响，可立即部署**。

---

## 10. 单目深度先验 + ToF 锚点融合：可行性实验（关键结论）

思路（用户提出、本工作验证）：**ToF 做"度量锚点"（准但噪/空洞），单目深度做"结构先验"（清晰、少噪、无玻璃空洞，但尺度漂移）**，二者融合。这是深度补全的标准低成本路线（类似 PromptDA / anchor-guided completion），比 LingBot-Depth 便宜 2–3 个数量级。

### 10.1 在"已标定 RGB-D"上的模拟结果（ex1/ex4，DAv2-S 518² 先验 + 90% ToF 锚点 + 合成洞）

| 场景 | 洞型 | 经典填充 RMSE/MAE (mm) | 先验·全局仿射 | 先验·**分块(tile)仿射** | **+ 锚点/引导滤波（推荐）** |
|---|---|---|---|---|---|
| ex1 | 随机 10% | **9.4 / 2.3** | 58.3 / 26.3 | 57.7 / 25.4 | 33.4 / 11.3 |
| ex1 | 45×45 块 | 296.2 / 201.0 | 99.6 / 41.8 | 100.1 / 39.1 | **89.1 / 35.6**（3.3× 好于填充） |
| ex4 | 随机 10% | **2.8 / 1.8** | 43.7 / 30.4 | 24.0 / 17.9 | 9.1 / 6.7 |
| ex4 | 45×45 块 | 165.5 / 108.8 | 47.8 / 31.7 | 21.2 / 17.6 | **17.9 / 14.5**（9.2× 好于填充） |

- **大洞**：融合比经典填充好 **2–9 倍**（18–89 mm vs 165–296 mm）；**小洞/随机洞**：经典填充仍最好（mm 级），融合也可到 cm 级 → 实践上按"洞的尺寸/到有效像素距离"切换（小洞填充、大洞先验）。
- **ToF 精度保留**：有效像素硬锚定（输出=ToF），单目只补洞；**引导滤波顺带压 ToF 噪声**。
- **清晰度**：在 ToF 有效像素上，DAv2 RGB 先验梯度能量是 ToF 的 **2.2–3.1 倍**（结构/边缘更丰富）。
- 实验文件：`results/live/fusion_calibrated_samples.json`（融合脚本见 `tmp` 实验记录，正式版待集成到 `deploy/`）。

### 10.2 未标定时的真实效果（live_1529，深度渲染输入先验，非 RGB）

| 洞型 | 经典填充 | 先验·全局仿射 | 先验·tile 仿射 | tile + 锚点/引导 |
|---|---|---|---|---|
| 随机 10% | **278 / 98** | 1167 / 721 | 893 / 561 | 558 / 297 |
| 45×45 块 | 1220 / 645 | 1161 / 719 | 1004 / 663 | **925 / 589** |

→ **没有 RGB-D 标定时，融合只能小幅改善大洞（RMSE −24%），远不如标定后的 2–9 倍**。所以：**融合路线成立，但前提是先做 RGB-D 标定（§9）**。

### 10.3 端到端帧率预算（33.3 ms/帧 = 30 fps）

| 组成 | 预算 |
|---|---|
| 相机取流/解码 | 2–4 ms |
| 单目模型（NPU）：DAv2-S 168×224 19.9 ms；yolo26n-depth @384–448 约 10–23 ms（估算） | 10–20 ms |
| 融合（分块仿射 + 锚点 + 引导滤波，cv2 实现） | 3–5 ms |
| 余量 | 5–10 ms |
| **合计** | **≈25–35 ms → 30–40 fps** |

---

## 11. 部署准备：yolo26n-depth + 原厂 RGB-D 标定 + 融合（2026-11-20 20:30 状态）

目标：**yolo26n-depth（单目，NPU）→ 原厂标定把 RGB 对齐到 ToF 帧 → 分块仿射 + ToF 硬锚点 + 引导滤波融合**，保留 ToF 度量精度、用单目补洞补结构。

### 11.1 已完成（PC 侧）

**模型导出与验证**（脚本 `tmp/export_yolo26n.py`，产物 `convert/yolo/`、`results/yolo/`）：

| 配置 | 输入/输出 | 参数量 | FLOPs | ONNX | 备注 |
|---|---|---|---|---|---|
| yolo26n-depth @448 | `images`[1,3,448,448] → `output0`[1,1,448,448] | 5.17 M | **15.47 G** | 20.8 MB, opset12 | PC CPU 150–320 ms |
| yolo26n-depth @768 | 同上 @768 | 5.17 M | **45.56 G**（官方 46.9 G @768 一致） | 20.8 MB | PC CPU 0.9–1.2 s |

- 算子对 ATC 友好：Conv×88（SiLU 分解为 Mul+Sigmoid）、Concat/Split/Reshape/Transpose/MatMul/Softmax、ConvTranspose、Exp/Clip；**Resize 只有 `nearest+asymmetric` 与 `linear+align_corners` 两种（CANN 8.0 都支持，不像 DAv2 踩到 `half_pixel`）**。
- 真机帧输出范围 0.6–6.9（相对逆深度/参考深度量纲），结构平滑无噪声（梯度 0.019–0.025 vs ToF 0.04–0.155，ToF 的梯度含噪声）。

**融合精度**（90% ToF 锚点 + 4×45×45 块洞，已标定 RGB-D 样例，RMSE/MAE mm；tile 仿射+锚点+引导）：

| 先验 | ex1 大洞 | ex4 大洞 | 经典填充（对照） |
|---|---|---|---|
| yolo26n @448 | 175.2 / 126.1 | 50.5 / 41.6 | 296.2 / 201.0（ex1）、165.5 / 108.8（ex4） |
| yolo26n @768 | **121.4 / 68.9** | **34.5 / 26.0** | 同上 |
| DAv2-S @518（参考） | 89.1 / 35.6 | 17.9 / 14.5 | 同上 |

→ 大洞比经典填充好 **1.5–4.8 倍**；DAv2-518 先验更强但 FLOPs 是 yolo768 的 2.2 倍。注意 ex5 出现 tile 仿射发散（先验离群值）→ 上线前需加保护：先验裁剪、a/b 限幅、拟合残差过大时回退经典填充。

**标定工具链**（头文件来自公开 SDK 仓库，API 已确认）：
- C 程序 `convert/get_rgbd_calib.c`：`MV3D_RGBD_GetCameraParam()` 读 `MV3D_RGBD_CAMERA_PARAM{stDepthCalibInfo, stRgbCalibInfo, stDepth2RgbExtrinsic}` → JSON（深度/RGB 内参 K、12 维畸变、深度→RGB 4×4 外参）；`convert/build_calib.sh` 编译。
- 对齐工具 `deploy/calib_to_warp.py`：深度→RGB 投影、**RGB→深度帧 warp**（z-buffer splat+inpaint）、`calib_overlay.png` 验证叠加图（已用合成标定跑通）。
- 备选 API：`MV3D_RGBD_GetCalibInfo(handle, CoordinateType_Depth/RGB, ...)`；`MV3D_RGBD_SetParam(handle,"ImageAlign",1)`（SDK 默认即 1）。

### 11.2 当前阻塞

1. **板子自 16:15 起 ping/SSH 不通**（PC 热点 192.168.137.1 正常）——ATC 转换、NPU 实测、SDK 标定全部依赖板子。
2. **Mv3dRgbd SDK 已从板子上被移除**（`/opt/Mv3dRgbdSDK` 不存在，只剩 `/etc/ld.so.conf.d/mv3drgbd.conf` 残留），需要重新安装 **Mv3dRgbd Linux aarch64 1.4.1.4（或更新版）** 才能读 `RGBDINNERPARAM`；没有 SDK 就只能退回棋盘格自标定。

### 11.3 解阻后一次跑完的清单

```bash
# 板子（把 SDK 装回 /opt/Mv3dRgbdSDK 后）
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cd ~/code/depth_completion
bash convert/build_calib.sh                      # 编译 get_rgbd_calib
./convert/get_rgbd_calib | tee results/rgbd_calib.json   # 导出原厂内参/外参
python3 deploy/calib_to_warp.py --calib results/rgbd_calib.json \
    --rgb data/live_1529/color.jpg --depth data/live_1529/depth16.png --outdir results  # 看 calib_overlay.png
# 转 OM（两个分辨率任选）
bash convert/atc_convert.sh yolo/yolo26n-depth_448.onnx models/yolo26n_448 "images:1,3,448,448"   # 先在 models/ 下准备 onnx
python3 deploy/bench_yolo.py --om models/yolo26n_448.om --size 448 --iters 50 --tag yolo448
python3 deploy/run_tof_pipeline.py --om models/yolo26n_448.om --data data/live_1529 \
    --out results/pipeline --iters 30 --tag live_yolo448 --npu-h 448 --npu-w 448 --normalize unit --prior-name yolo26n
```

预期（待实测确认）：yolo448 纯 NPU **15–25 ms（40–65 fps）**、yolo768 **45–90 ms（11–22 fps）**（按 CNN 有效算力 0.7–1.2 TFLOPS 估算）；融合 CPU 3–5 ms → yolo448 全链路 **30–50 fps** 可行。

---

## 12. RGB-D 标定实测 + yolo26n-depth 板端部署（2026-11-20 21:10）

### 12.1 原厂标定导出（已成功，真实参数）

在原厂 SDK `Mv3dRgbd Linux V1.4.1.4` 下用 `convert/get_rgbd_calib.py` 读 `MV3D_RGBD_GetCameraParam`：

**关键坑**：SDK 的 `ImageAlign` 参数会改变 `GetCameraParam` 的返回值——
- `ImageAlign=1`（SDK 默认）：深度已被 SDK 预对齐到彩色，depth/rgb 都报 1536×1280、外参为单位阵；
- **`ImageAlign=0`：才是 640×480 原始深度坐标系**，此时返回真实深度内参/畸变和深度→RGB 外参。
脚本已对两种模式都 dump（`results/rgbd_calib.json` 的 `align0`/`align1`）。

| 项 | 数值（ImageAlign=0，SN DA8687616） |
|---|---|
| 深度内参（640×480） | **fx=507.604, fy=507.702, cx=326.961, cy=243.043** |
| RGB 内参（1536×1280） | **fx=1005.958, fy=1006.076, cx=766.899, cy=634.596** |
| 畸变（两者相同） | k1=0.06203, k2=−0.08733, p1=−0.000145, p2=0.000210, k3=0 |
| 深度→RGB 外参 4×4 | R≈单位阵（≤0.2°），**t=[26.459, 0.228, 3.007] mm**（基线 ≈2.6 cm） |

**标定验证**（投影自检，`results/sdk_capture/`）：把 640×480 原始 ToF 深度按上式投影到 RGB 坐标系，与 SDK 自己 `ImageAlign=1` 输出的对齐深度比较（同一场景相邻帧）：
**中位误差 5.4 mm，83.5% 在 20 mm 内，94.7% 在 50 mm 内 → 原厂标定与我们的 warp 实现都验证通过。**
对齐工具：`deploy/calib_to_warp.py`（`calib_overlay.png` 叠加图、`rgb_aligned_to_depth.png` RGB→深度帧 warp）；SDK 抓帧工具 `convert/fetch_sdk_frames.py`。

### 12.2 yolo26n-depth 板端实测（OM + pyACL）

| 输入 | OM 大小 | **纯 NPU p50 / p95** | fps | 完整调用（含 resize/拷贝） | PC CPU（ONNX 参考） |
|---|---|---|---|---|---|
| **448×448** | 11.5 MB | **15.81 / 16.29 ms** | **63.3** | ≈+8–10 ms | 150–320 ms |
| 768×768 | 11.5 MB | 52.38 / 53.54 ms | 19.1 | ≈+10–15 ms | 0.9–1.2 s |

- 两个分辨率的 ATC 转换都成功（448 用 6m51s，768 约 6–7 min），注意力子图（Split/Transpose/MatMul/Softmax）在 CANN 8.0 上可转换；`.om` 均为 11.5 MB，传输成本极低。
- 端到端 pipeline（448，板子外部负载 load≈17 下）：p50 105 ms，其中预处理 18.8 ms、融合后处理 62.8 ms 被 CPU 负载放大（板子空闲时 CPU 部分预计 10–15 ms）→ **yolo448 + 融合的真实 e2e 预计 25–30 ms（33–40 fps）**。
- 板端补全结果（真实空洞）：`results/board/pipeline_yolo/`、`results/board/yolo/`（含 bench JSON、对比图、深度 npy）。

### 12.3 标定后的融合实验（真实标定 + SDK 对齐帧）

在 SDK 的「对齐 RGB-D 帧」（1536×1280，深度有效 41.4%）上做打洞回填（GT=对齐深度）：

| 洞型 | 经典填充 RMSE/MAE | yolo 先验（全局仿射） | yolo 先验（分块+锚点+引导） |
|---|---|---|---|
| 随机 10% | **47.7 / 7.1 mm** | 673.9 / 433.7 | 218.4 / 96.2 |
| 140×140 大洞 ×4 | 1065.7 / 715.9 | **765.7 / 537.7** | 853.5 / 501.1 |

- **大洞：先验比经典填充好 ~25–40%**；小洞/随机洞仍是经典填充最好（mm 级）→ 工程上用「按洞尺寸切换」的混合策略。
- yolo 输出语义经实测是**深度型**（与 ToF 深度相关 +0.47、与视差相关 −0.56）→ 融合拟合应改到**深度/对数深度空间**（当前 pipeline 用视差空间，仍可工作但非最优），并加保护（先验离群裁剪、a/b 限幅、拟合残差过大回退经典填充）。

### 12.4 下一步（可直接执行）

1. 把 `calib_to_warp` 的 warp 用 **cv2** 实现在板端（当前 numpy 版偏慢）→ RGB 对齐到 640×480 深度帧（3–5 ms）→ 送 yolo448。
2. 融合改到深度/对数深度空间 + 分块仿射 + 硬锚点 + 引导滤波 + 小洞/大洞切换。
3. 板子空闲时复测 e2e：目标 **30–40 fps**（yolo448 15.8 ms NPU + warp/融合 6–10 ms CPU）。
4. 若需要更高精度：yolo768（19 fps）或 DAv2-518（更慢），或对 yolo 先验做 temporal 复用（每 2–3 帧更新一次）。

---

## 13. 标定后的融合 v2：精度与延迟实测（2026-11-20 21:45）

新增 `deploy/fusion_v2.py` / `deploy/fusion_fast.py`：一条命令跑通「标定 warp → yolo OM → 混合融合 → 合成洞评估 → 延迟分解」，输出 `results/fusion_{v2,fast}/metrics_*.json`、`compare_*.png`、`fused_*.npy`。

**融合策略（v2）**：标定把 RGB warp 到 640×480 深度帧 → yolo 先验 → **对数深度空间分块鲁棒仿射**（`log z ≈ a·log prior + b`）→ **小洞用经典 push-pull 填充、大洞用先验** → ToF 硬锚点 + 先验引导滤波（2 轮）。`fusion_fast.py` 把融合降到半分辨率（320×240）+ 去掉 inpaint，进一步提速。

**实测**（板子外部 load≈17 但 CPU 空闲；NPU 数值多次重复稳定）：

| 配置 | NPU p50 | warp | fusion | e2e p50 | 大洞(45×45×4) fill → fused |
|---|---|---|---|---|---|
| yolo448 live_1529 | **15.84 ms** | 172 ms | 168 ms | 392 ms | 1632 → **930 mm**（1.75×） |
| yolo448 real_tof | 15.84 ms | 151 ms | 158 ms | 361 ms | 2104 → **1464 mm**（1.44×） |
| yolo768 live_1529 | 52.75 ms | 158 ms | 340 ms | 636 ms | 1632 → 1048 mm |

- 小洞（随机 10%）：经典填充最优（305–333 mm），融合 v2 在 399–656 mm——证明「小洞填充、大洞先验」切换策略正确；
- **瓶颈完全在 CPU 侧 Python/numpy**（warp ~150–170 ms、融合 ~160–340 ms），NPU 只占 15.8 ms（63 fps，余量 47 ms）；
- 板子当前状态异常：`loadavg` 17–18 却 CPU `us` 仅 1–14%、无 R/D 用户进程（只有 `dev0_sq_task` 内核线程），疑似 SD/USB I/O 内核线程造成负载假象；此状态下 CPU 计时不可信（早期空闲时 fill 16.5–23.5 ms、PIL 预处理 2–6 ms）。**要在真正空闲的板子上复测**。

**实时抓拍验证（2026-11-20 22:14，现场场景）**：用原厂 SDK 从板载相机抓了一帧新的 raw RGB-D（640×480 depth + 1536×1280 RGB，ToF 有效 93.3%），在该帧上跑 yolo448 + 标定融合：
- 大洞（45×45×4）RMSE：经典填充 **966 mm** → 单目先验 **296 mm** → **融合 270 mm（提升 3.58×）**
- 小洞（随机 10%）：经典填充最优（108.8 mm）→ 按洞尺寸切换策略不变
- 图：`results/live_now_compare_labeled.png`（实时 RGB + 5 深度面板）

**效果图**（PC `results/`，板端已同步同名文件）：
- `fusion_compare_with_rgb_labeled.png`：**补全效果 + 对应 RGB 六面板对照**（对应 RGB / 原始 ToF / 挖洞 / 经典填充 / 单目先验 / 融合）
- `rgb_tof_alignment_labeled.png`：**标定对齐验证图**（原图 RGB / 按标定 warp 到深度帧的 RGB / 原始 ToF 深度）
- `fusion_compare_labeled.png`：五面板定量对比（不含 RGB）
- `calib_overlay.png`：ToF 深度点按标定投影到 RGB 的叠加图

**结论**：yolo26n-depth@448 在 310B1 上 15.8 ms/帧（63 fps）已远超 30 fps 需求；「标定 + 先验 + 锚点」融合在大洞上比经典填充好 1.4–1.75 倍。下一步只需把 warp/融合改成 C++ 或 cv2 重写（目标 warp 3–5 ms、融合 3–5 ms），即可达到 **25–30 ms 全链路（33–40 fps）**；NPU 侧再无压力。

---

### 13.1 进一步优化：v3/v4（不再 warp RGB，改为「先验投影采样」）

v2 的 RGB→深度帧 warp 太慢（150–245 ms），改为：yolo 直接在**原 RGB** 上推理（1536×1280→448 拉伸），再把深度帧每个像素用标定投影到 RGB 坐标、对 448×448 先验做 cv2.remap 采样（先验低频，无需 warp/inpaint）。v4 进一步把投影降到 1/4 分辨率（160×120）再上采样、并把分块拟合改成向量化实现。

| 版本 | 关键改动 | e2e p50（448） | 大洞 fused RMSE（live_1529 / real_tof） |
|---|---|---|---|
| v2 | 全分辨率 RGB warp + 半分辨率融合 | 546 / 633 ms | 1083.8 / 1748.2 mm |

## 16. 当前方案：CPU 传统深度补全（2026-11-22 17:10，**最新**）

> 学习类补全模型（LingBot-Depth / DAv2 / PromptDA-Small-Transparent）的权重、OM、转换脚本、参考代码与结果**已全部删除**（PC 释放约 260 MB，板端项目 74 MB → 14 MB）。不再使用 NPU。

### 16.1 算法（deploy/pd_core.ClassicCompleter，纯 OpenCV/numpy）
1. **离散点删除**：3×3 中值 + 「偏离局部中值 > 180 mm 丢弃」（去椒盐/飞点）
2. **低信噪区判定**：8×8 块级时序不稳定统计（|Δd|>150 mm 比例 > 0.6）→ 整块用平滑面替换（中值救不了的面状噪声）
3. **平滑先验**：1/8 分辨率归一化卷积，空块用 1/32 兜底（否则出现黑洞）
4. **色彩引导补洞**：半分辨率导向滤波（自实现 boxFilter 版，比 ximgproc 快 6×），RGB 亮度当引导 → 补洞边缘跟随颜色
5. **时序 EMA**：α=0.45，跳变 >400 mm 立即重置（不闪、不拖影）

### 16.2 实测（板子空闲时）
| 项 | 数值 |
|---|---|
| 单帧处理（640×480 全流程） | **41–45 ms** |
| 端到端 | **11–19 fps**（上限由相机曝光决定，相机本身 11–19 fps） |
| 补洞前后有效率 | 99.9% → 100% |
| 点云 | 19,200 点/帧（stride 4），每 0.3 s |

### 16.3 运行
    cd ~/code/depth_completion
    python3 deploy/pd_live.py --port 8765 --depth-every 0.2 --cloud-every 0.3 --seconds 0
    # Foxglove 连 ws://<板子IP>:8765
    #   /camera/rgb (jpeg)  /depth/colored (jpeg, 补全后伪彩)
    #   /depth/depth16 (png, 补全后原始值)  /points (foxglove.PointCloud)

关键依赖：相机 U3V 直连（pd_capture.py）、Foxglove WebSocket 自实现（foxglove_ws.py，MessageData 帧头必须是 0x01 | subId(4) | ts(8) | payload）。

### 16.4 继续优化的方向
- 导向滤波降到 1/4 分辨率（省 ~4 ms）
- 平滑先验两帧共用一次（省 ~7 ms）
- 时序 EMA / np.where 改原地操作（省 ~5 ms）
- RGB 降采样（1536×1280→640×480，13 ms）已放在采集线程，不占主循环
