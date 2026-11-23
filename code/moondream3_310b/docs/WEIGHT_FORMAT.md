# 权重格式与 meta.txt 规范

## 1. 目录

```
converted/moondream3_310b/
  meta.txt
  manifest.json
  text/
    wte.w4.bin  wte.s.bin  wte.sz.bin
    lm_head.w4.bin ...
    l00/ln1.w.f16.bin ...
  vision/
    patch_emb.w.f16.bin ...
  MODIFICATIONS.md  LICENSE  (合规材料)
```

## 2. meta.txt

逐行，空格分隔。三种行：

```
CFG <key> <value>
W4  <role> <file-prefix> <N> <K>
F16 <role> <relpath> <shape>          # shape 用 x 连接，如 32x6144
```

W4 张量三件套：

| 文件 | 类型 | 形状 | 含义 |
|---|---|---|---|
| `<prefix>.w4.bin` | uint16 | `[N, K/4]` | 相位交织 4bit 权重 |
| `<prefix>.s.bin`  | fp16   | `[N, SBP]` | group-32 scale，`SBP=ceil(ceil(K/32)/16)*16` |
| `<prefix>.sz.bin` | fp16   | `[N, SBP]` | `s*z`（非对称 zero-point 预乘） |

反量化：`w[n,k] = s[n,k/32] * q[n,k] - sz[n,k/32]`。

相位交织打包：`KQ=K/4`，`word[j] = q[j] | q[KQ+j]<<4 | q[2KQ+j]<<8 | q[3KQ+j]<<12`。
（与 `ascendc_ops/kernels/w4a16_gemv_gen.cpp` 的 ABI 一致。）

F16 文件为 little-endian fp16 原始字节，形状仅用于校验。

## 3. 角色表

### 文本（24 层，前 4 层 dense，后 20 层 MoE-64 选 8）

| role | 形状 | 说明 |
|---|---|---|
| `CFG hidden/n_layers/n_heads/n_kv_heads/head_dim/rot_dim/vocab/max_context` | | 2048/24/32/32/64/32/51200/4096 |
| `CFG moe_start/n_experts/experts_per_token/expert_inner/dense_ff` | | 4/64/8/1024/8192 |
| `CFG rope_theta/eps/prefix_tokens` | | 1500000/1e-5/730 |
| `T.WTE` | W4 51200x2048 | token embedding（host 侧行反量化） |
| `T.LM` / `T.LM.B` | W4 51200x2048 / f16 51200 | lm_head + bias |
| `T.FN.W` / `T.FN.B` | f16 2048 | final LayerNorm |
| `T.L{i}.LN1.W/B`, `LN2.W/B` | f16 2048 | LayerNorm（weight+bias） |
| `T.L{i}.QKV` / `.QKV.B` | W4 6144x2048 / f16 6144 | qkv 融合投影 |
| `T.L{i}.O` / `.O.B` | W4 2048x2048 / f16 2048 | attn proj |
| `T.L{i}.TAU.WQ` / `TAU.WV` / `TAU.A` | f16 32x6144 / 32x6144 / 32 | 学习式温度缩放 |
| `T.L{i}.FC1` / `.FC1.B` | W4 8192x2048 / f16 8192 | dense layer 0-3 |
| `T.L{i}.FC2` / `.FC2.B` | W4 2048x8192 / f16 2048 | dense layer 0-3 |
| `T.L{i}.ROUTER` / `.ROUTER.B` | W4 64x2048 / f16 64 | MoE 路由 |
| `T.L{i}.E{e}.FC1` | W4 2048x2048 | 专家 GeGLU gate+up（无 bias） |
| `T.L{i}.E{e}.FC2` | W4 2048x1024 | 专家 down（无 bias） |

### 视觉（SigLIP 式 ViT，27 层，27x27 patch，13 crop）

| role | 形状 | 说明 |
|---|---|---|
| `V.PATCH.W/B` | f16 1152x588 / 1152 | patch embedding（K=588 不量化） |
| `V.POS` | f16 729x1152 | 位置编码 |
| `V.L{i}.LN1.W/B`, `LN2.W/B` | f16 1152 | |
| `V.L{i}.QKV` / `.B` | W4 3456x1152 / f16 3456 | |
| `V.L{i}.O` / `.B` | W4 1152x1152 / f16 1152 | |
| `V.L{i}.FC1` / `.B` | W4 4304x1152 / f16 4304 | |
| `V.L{i}.FC2` / `.B` | W4 1152x4320 / f16 1152 | **K 由 4304 补零到 4320** |
| `V.FN.W/B` | f16 1152 | post_ln |
| `V.PROJ1` / `.B` | W4 8192x2304 / f16 8192 | 投影 MLP |
| `V.PROJ2` / `.B` | W4 2048x8192 / f16 2048 | |

> Region 坐标头（point/detect 用）本期不转换；query/caption 不需要。

## 4. 合规

上游权重为 **BSL 1.1 + Additional Use Grant**（非 OSI 开源），转换产物属于 “Derivatives”，同样受 BSL 约束；
`MODIFICATIONS.md` 记录：格式重打包 / bf16→fp16 / int4 group64→group32 重量化 / 只保留文本+视觉塔。

## Vision encoder 权重（f16 直存，不走 W4）

Moondream int4 仓库中 **vision encoder 权重是 bf16 未量化**（仅 text MoE experts 是 MLX affine int4）。
vision 只有 27 层 ViT、约 430M 参数，量化收益小、误差在 27 层里累积明显（实测 W4 版 enc 与 bf16 参考 corr≈0），
因此 vision 大矩阵权重改用 **fp16 直接存储**（\`tools/convert_vision_f16.py\` 从 int4 safetensors 的 bf16 转换）：

\`\`\`
models/moondream3_310b/vision/
  L{i}.qkv.f16   [3456,1152]   (行主序，配合 matmulTN)
  L{i}.o.f16     [1152,1152]
  L{i}.fc1.f16   [4304,1152]
  L{i}.fc2.f16   [1152,4352]   (K 方向补零 48 列到 4352)
  L{i}.qkv.b.f16 / o.b / fc1.b / fc2.b
  proj_fc1.f16   [8192,2304]
  proj_fc2.f16   [2048,8192]
\`\`\`

- 每层约 30.5 MB，27 层 + proj_mlp 约 0.85 GB；板上内存足够（文本 W4 5.4GB + vision 0.85GB）。
- patch_emb 权重（[1152,588] f16）在板端 **CPU 精确计算**（1.07 GMAC），避开 ACLNN 对 K=588 的非对齐 K。
- vision attention head_dim=72（1152/16），ACLNN matmul 对非 32B 对齐的 stride view 不支持，
  当前 **CPU 多线程实现**（softmax 无 mask，scale=1/sqrt(72)）。
- vision 结束后统一做 \`post_ln\`（LayerNorm），再进 \`proj_mlp\`（fc1→GELU→fc2）。
