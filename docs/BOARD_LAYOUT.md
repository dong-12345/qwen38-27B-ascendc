# 板端目录说明（OrangePi AI Pro 20T / Ascend 310B1）

所有与 **Qwen3.8-27B 端侧推理**相关的内容都集中在：

```
~/code/qwen38_310B/
├── README.md                 ← 本文件
├── env.sh                    ← 环境变量（source 一次即可）
├── src/                      ← 端侧推理 + 自研 AscendC 算子（源码 + 构建）
│   ├── kernels/              27 个 AscendC kernel
│   │   ├── w4a16_gemv_gen.cpp    ★ W4A16 相位交织 GEMV（主力算子）
│   │   ├── full_attn_decode.cpp  ★ 全注意力解码（GQA + partial RoPE + 输出门 + KV cache）
│   │   ├── dense_ops.cpp         fp16_gemv / rmsnorm / axpy / mul_silu / conv1d_decode
│   │   ├── gdn_l2norm.cpp        GDN q/k L2 归一化
│   │   ├── gdn_rgdr.cpp          GDN 单 token 递推解码
│   │   ├── gdn_gated_rmsnorm.cpp GDN 门控归一化
│   │   └── gdn_{gating,conv1d,cumsum,kkt,tril,wy,chunk_h,chunk_o}.cpp  chunked prefill 六阶段
│   ├── host/
│   │   ├── engine_27b.cpp    ★ 27B 推理引擎（64 层调度/权重加载/状态/lm_head）
│   │   ├── main.cpp             可执行入口（qwen38_ops_test，含各算子单测）
│   │   ├── engine_mini.cpp      小模型验证引擎
│   │   └── data_utils.h         ACL 辅助
│   ├── cmake/  scripts/  CMakeLists.txt  run.sh
│   ├── input/ input_golden/     单测数据（53MB，可用 scripts/gen_data.py 重新生成）
│   ├── build/                   构建中间产物
│   └── out/                     产物：out/lib/libascendc_kernels_npu.so、out/bin/qwen38_ops_test
└── models/
    ├── qwen38_310b/         ★ 27B 转换后权重（16.9GB / 1670 文件 + meta.txt）
    └── debug_subsets/         调试用小子集（sub1/sub3/sub41/sub48，仅 meta.txt）
```

---

## 快速使用

```bash
cd ~/code/qwen38_310B
source env.sh                      # CANN 环境 + LD_LIBRARY_PATH + ENG27_DIR

# 1) 编译（约 2~4 分钟）
bash src/run.sh build

# 2) 算子单测（用真实权重）
cd src
./out/bin/qwen38_ops_test w4a16_gemv_gen      # W4 GEMV，K=5120 N=10240
./out/bin/qwen38_ops_test dense_ops           # conv1d/rmsnorm/l2norm/mul_silu
./out/bin/qwen38_ops_test gdn_rgdr            # GDN 递推解码
./out/bin/qwen38_ops_test full_attn_decode    # GQA 解码

# 3) 跑 27B（ENG27_IDS 是 prompt 的 token id，可用 PC 侧 tools/run_27b_demo.py 自动生成）
ENG27_GEN=6 ./out/bin/qwen38_ops_test engine_27b
```

## env.sh 里设置的关键变量

| 变量 | 含义 |
|---|---|
| `ENG27_DIR` | 权重目录，默认 `~/code/qwen38_310B/models/qwen38_310b` |
| `ENG27_IDS` | prompt token id（逗号分隔）；不设则用内置默认 |
| `ENG27_GEN` | 生成 token 数 |
| `ENG27_NOWARMUP=1` | 跳过 warmup 前向（省 ~1 次前向时间） |
| `ENG27_TRACE=1` | 打印每个 token 的 logits top-5（调试用） |

## 已知注意事项

- **跑之前清内存**：`sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'`，
  并确保没有其它吃内存的进程（残留的 `atc.bin` / python 曾在 27B 加载时导致 OOM 重启）。
- **权重巨大**：16.9GB 全部常驻 NPU 侧内存；加载约 84 秒。
- **性能**：decode ≈ 0.069 tok/s（14.5 s/token），prefill 同速（无 batch）。
- 每次改 kernel 后都要 `bash src/run.sh build` 重新编译（会 `rm -rf build out`）。
