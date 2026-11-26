# Qwen3.8-27B / Ascend 310B1 推理工程

W4A16（int4 权重 + fp16 激活，group-32 非对称）的 Qwen3.8-27B 在
OrangePi AI Pro 20T（Ascend 310B1 / 单 AI Core，CANN 8.0.0）上的推理实现与性能优化。

目录结构与 `code/moondream3_310b/ops` 一致：

```
code/qwen38_310b/ops/
├── CMakeLists.txt          # 构建入口（cmake + CANN）
├── run.sh                  # 构建脚本（板上执行）
├── kernels/                # AscendC 算子
│   ├── w4a16_gemv_gen2.cpp     # decode 默认 GEMV（合并指令，与 gen 逐位等价）
│   ├── w4a16_gemv_axpy.cpp     # decode 新路径（AX 布局，沿 N 向量化）
│   ├── dequant_w4.cpp          # 两段式第一步：W4 → fp16
│   ├── mm/cube_matmul.cpp      # Cube Matmul（两段式第二步）
│   └── dense_ops.cpp           # rmsnorm / fp16_gemv / conv / silu 等
├── host/
│   ├── main.cpp                # 算子单测 + relayout 工具
│   ├── engine_27b.cpp          # 27B 引擎（prefill/decode、Cube 两段式、AX 开关）
│   └── matmul_tiling.cpp       # Cube tiling
└── scripts/                    # 板端辅助脚本
```

## 与开发板的对应关系

本地 `code/qwen38_310b/ops/<x>` ↔ 板上 `~/code/qwen38_310B/src/<x>`（用 `board/spush.ps1` 推送）：

```powershell
& .\board\spush.ps1 -Map @(
  "code\qwen38_310b\ops\host\engine_27b.cpp=~/code/qwen38_310B/src/host/engine_27b.cpp",
  "code\qwen38_310b\ops\kernels\w4a16_gemv_axpy.cpp=~/code/qwen38_310B/src/kernels/w4a16_gemv_axpy.cpp")
& .\board\brun.ps1 -Script board\build.sh      # 板上增量编译
```

## 关键开关

| 环境变量 | 作用 |
|---|---|
| `ENG27_DIR` | 权重目录（默认 `models/qwen38_310b`；AX 布局用 `models/qwen38_310b_ax`） |
| `ENG27_LAYOUT=axpy` | 用 AX 布局 + `w4a16_gemv_axpy`（decode 新路径） |
| `ENG27_GEMV_OLD=1` | 回退到最早的 gen GEMV |
| `ENG27_PREFILL_SEQ=1` | prefill 退回逐 token（用于 A/B） |
| `ENG27_IDS=... ENG27_GEN=N` | 指定 prompt / 生成 token 数 |

## 性能现状（板端实测）

| 项 | 现状 | 基线 | 加速 |
|---|---|---|---|
| prefill（17 token） | 36.7 s（2.16 s/token） | 212.8 s | **5.8x** |
| decode（gen2） | 12.48 s/token | 14.28 s/token | 1.14x |
| decode（AX/Axpy-N，单矩阵） | 19.9 ms（2.32x） | 46.1 ms | **2.32x** |

详见 `docs/GEMV性能分析与优化路线.md`。
