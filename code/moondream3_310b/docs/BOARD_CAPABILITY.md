# 板端能力实测（OrangePi AI Pro 20T / Ascend 310B1）

环境：CANN 8.0.0，驱动/npu-smi 25.2.0，Ubuntu 22.04，23GB 统一内存，4×ARM CPU，SSH `HwHiAiUser@192.168.137.163`。

## 1. 硬件与 AI Core

| 项 | 实测 |
|---|---|
| 产品 | Atlas 200I A2 / 310B1 |
| `platform_config/AS31XM1X.ini` | `ai_core_cnt=1`、`vector_core_cnt=1`、cube_freq=1024 |
| `npu-smi set` 支持的类型 | 无 `aicore-num` 类配置；只有 `cpu-num-cfg`（AI CPU / control CPU 数量，与 NPU 算力无关） |
| 结论 | **310B1 固定 1 个 AI Core，无法通过"设置 AI Core 数量"提速**；优化只能从内核融合、多流、投机解码、减少 kernel 启动/带宽入手 |

## 2. 内存

- npu-smi 显示 23673MB，系统 `free` 同为 23GB → **CPU/NPU 统一内存**；
- torch_npu 分配 8GB 张量时被 OOM-killer 杀掉（分配即消耗系统内存）；
- 因此本项目的策略是 **W4 权重常驻 ~5GB + KV 0.8GB**，而不是 bf16 全量 18.5GB。

## 3. ACLNN 实测（加速核心）

| 算子/形状 | 结果 |
|---|---|
| `aclnnMatmul` 730×2048×6144 | **62.5 ms，0.29 TFLOPS**（含每次重建 executor） |
| `aclnnMatmul` 730×2048×8192 | 80.2 ms，0.31 TFLOPS |
| `aclnnMatmul` 9477×1152×3456（vision qkv 全 crop） | 197 ms，0.38 TFLOPS |
| `aclnnMatmul` 转置 B（stride view，无需物理转置） | ✅ 支持，数值正确 |
| `aclnnLayerNorm` 730×2048 | 4.8 ms，数值正确 |
| `aclnnSoftmax` [32,730,730] | 33 ms |
| `aclnnAdd` [B,R,C]+[R,C] 广播 | ✅ 支持（用于 attention mask） |
| `aclnnGelu` | ❌ 失败（310B 无 Gelu kernel）→ 自研 `gelu_f16` |
| `aclnnBatchMatMul` | ❌ 执行返回 161001 → attention 用逐 head MatMul |
| `aclnnMatmul` 6144×2048 等小规模 | 正常；小 M 效率低，decode 不走 ACLNN |

> 执行器语义：每次调用都要重新 `GetWorkspaceSize`；复用同一 executor 会报 161001/崩溃。workspace 可缓存复用。

## 4. 自研 AscendC 算子实测

构建：`cd ~/code/moondream3_310b/ops && bash run.sh build`（CANN `ascendc_library` + SOC `Ascend310B1`）。
单测：`./md3_engine test`，全部 host 对拍通过：

| 算子 | max_abs（fp16） |
|---|---|
| layernorm_f16 | 4.9e-4 |
| gelu_f16 | 4.9e-4 |
| geglu_f16 | 1.6e-3 |
| add_bias_f16 | 4.9e-4 |
| dequant_w4_f16 | 3.1e-5 |
| qkv_post_text q/k/v | ≤4.9e-4 |
| qkv_post_vision | 1.6e-5 |
| attn_decode_text pos=0/1/3 | ≤7e-4 |

## 5. 存储与传输

- 根分区 229G，可用 ~160G；
- PC↔板 WiFi（Windows 热点 2.4G）实测 ~1.2MB/s，5GB 权重传输约 70 分钟；
- 板端 eMMC 顺序读约 100–200MB/s（权重加载瓶颈之一）。
