# 部署与运行

## 1. 目录

PC 侧（工作区）：

```
code/moondream3_310b/
  ops/                 AscendC 算子 + C++ 引擎（板端编译）
  tools/convert_moondream3.py   权重转换
  tools/ref_moondream3.py       numpy 参考（对拍）
  tools/run_demo.py             PC 端 demo（tokenizer + ssh）
models/moondream3_int4/         官方 4bit 权重（源）
converted/moondream3_310b/      转换产物（meta.txt + w4/f16 文件）
```

板端：

```
~/code/moondream3_310b/
  ops/                          源码（与 PC 同步）
  ops/out/bin/md3_engine        编译产物
  models/moondream3_310b/       转换权重（5GB 级）
```

## 2. 权重转换（PC）

```bash
python tools/convert_moondream3.py \
  --src ../../models/moondream3_int4/model.safetensors \
  --out ../../converted/moondream3_310b
```

子集（调试用）：`--limit-layers 5 --skip-vision`（meta 中 n_layers=5）。

转换内容：MoE 专家 int4(group64)→fp32→W4A16(group32 相位交织)；其余 bf16 线性层→W4A16；
LayerNorm/bias/tau/pos/patch 等→fp16。输出大小约 5.3GB（含 vision）。

## 3. 上传板端

WiFi 约 1.2MB/s，5GB 约 70 分钟；建议 tar 单流：

```powershell
tar -czf $env:TEMP\md3_weights.tgz -C converted moondream3_310b
# 先设置板卡地址与本机私钥：BOARD=HwHiAiUser@<板卡IP>；KEY=~/.ssh/id_rsa（或你的板卡私钥）
scp -i $KEY $env:TEMP\md3_weights.tgz ${BOARD}:~/
# 板端
cd ~/code/moondream3_310b && tar -xzf ~/md3_weights.tgz -C models/
```

（压缩对 int4 基本无效，可不用 gzip；tar 打包只为减少小文件开销。）

## 4. 板端编译

```bash
cd ~/code/moondream3_310b/ops
bash run.sh build        # 2-4 分钟，产物 out/bin/md3_engine、md3_engine
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export LD_LIBRARY_PATH=$PWD/out/lib:$PWD/out/lib64:/usr/local/Ascend/ascend-toolkit/latest/lib64:$LD_LIBRARY_PATH
./md3_engine test        # 10 个自研算子 host 对拍
```

## 5. 运行

文本 decode（逐 token prefill，慢但稳）；`--prefill` 用 ACLNN 批量 prefill：

```bash
# 文本生成（greedy）
./md3_engine --dir ~/code/moondream3_310b/models/moondream3_310b \
             --ids 0,1,15381,2,8299,1158,3 --gen 8 --prefill
```

PC 端一键 demo（自动 tokenize / ssh / decode）：

```bash
python tools/run_demo.py --prompt "The capital of France is" --gen 8
```

输出：`[md3] ...` 日志 + `TOKENS <ids>`；`--dump logits.bin` 导出 fp32 logits 供对拍。

## 6. 对拍

```bash
# 板端导出 logits（limit 5 层子集）
./md3_engine --dir .../moondream3_sub5 --limit-layers 5 --ids ... --dump /tmp/board_logits.bin
# PC 参考
python tools/ref_moondream3.py --dir converted/moondream3_sub5 --limit-layers 5 \
       --ids ... --out /tmp/ref_logits.bin
python tools/compare_logits.py /tmp/ref_logits.bin /tmp/board_logits.bin
```

## 7. 注意事项

- 板端内存是 CPU/NPU 统一 23GB：加载 W4 权重约 5.4GB + KV 0.81GB，需保证没有其它大进程；
- ACLNN MatMul 0.29–0.38 TFLOPS，prefill 是延迟大头；decode 用自研 GEMV（1.4GB/s 级）；
- 310B1 只有 1 个 AI Core（`ai_core_cnt=1`），没有 AI Core 数量配置项；
- 首次运行前建议 `sync; sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'`（若 sudo 可用）。

## 图像问答（vision + 多模态 prefill）

1. PC 端预处理：\`python tools/prep_image.py --image a.png --out tmp/crops.bin\`
   （简化版：单 crop window，图像 resize 到 378x378，生成 global+local 两个相同 crop = n_crops 2）
2. PC 端把 \`tmp/crops.bin\` scp 到板端 \`/tmp/crops.bin\`。
3. PC 端 tokenize 问题：prompt = \`[1,15381,2] + encode(question) + [3]\`（有图时不加 BOS，BOS 已在 image prefix）。
4. 板端运行：

\`\`\`bash
cd ~/code/moondream3_310b/ops && source /usr/local/Ascend/ascend-toolkit/set_env.sh
export LD_LIBRARY_PATH=$PWD/out/lib:$PWD/out/lib64:/usr/local/Ascend/ascend-toolkit/latest/lib64:$LD_LIBRARY_PATH
./md3_engine --image /tmp/crops.bin --ncrops 2 --qids 1,15381,2,1677,1835,41216,5210,51,3 --gen 16 \
  --dir ~/code/moondream3_310b/models/moondream3_310b
\`\`\`

一键脚本：\`python tools/run_qa.py --image a.png --question "What color is the circle?" --gen 16\`
（内部完成预处理、上传、SSH 运行与解码）。

调试环境变量：\`MD3_VISION_ONLY=1\`（只编码 vision）、\`MD3_DUMP_IMGEMB=1\`（dump image embedding）、
\`MD3_DUMP_ENC=1\`（dump vision encoder 输出）、\`MD3_VISION_LIMIT=N\`（只跑前 N 层 vision）。
