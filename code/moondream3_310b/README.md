# Moondream 3 Preview 端侧部署（OrangePi AI Pro 20T / Ascend 310B1）

把 Moondream 3 Preview（9B MoE VLM，2B 激活）部署到昇腾 310B1 板卡：自研 W4A16 GEMV + ACLNN 混合引擎，
文本生成与图像问答均已跑通，并与官方 modeling 的 numpy 参考逐层对拍。

## 当前状态（2026-11-23）

| 能力 | 状态 | 实测 |
|---|---|---|
| 文本 decode（24 层） | ✅ | 2.38 s/token，与参考 corr 0.9996+ |
| 文本 prefill（ACLNN 混合引擎） | ✅ | 7 token 9.9 s，输出与 decode 一致 |
| vision encoder（27 层 ViT，f16 权重） | ✅ | 2 crops 40 s，逐层与 bf16 参考对齐 |
| image prefix（730 token）+ 图像问答 | ✅ | "What color is the circle?" → "The circle is red." |
| 投机解码（prompt-lookup） | ✅ 实验 | 接受 68.8%，decode 1.92x（端到端无净收益，见 docs/SPECULATIVE.md） |
| 算子复用/自研 | ✅ | 复用 3 + 自研 12（docs/OP_REUSE.md） |

## 快速开始

```bash
# 板端：文本生成
cd ~/code/moondream3_310b/ops && source /usr/local/Ascend/ascend-toolkit/set_env.sh
export LD_LIBRARY_PATH=$PWD/out/lib:$PWD/out/lib64:/usr/local/Ascend/ascend-toolkit/latest/lib64:$LD_LIBRARY_PATH
./md3_engine --dir ~/code/moondream3_310b/models/moondream3_310b \
  --ids 0,1,15381,2,8299,1158,3 --gen 8

# 板端：图像问答（PC 端先跑 tools/run_qa.py 一条龙，或手动 prep_image + scp crops.bin）
./md3_engine --image /tmp/crops.bin --ncrops 2 \
  --qids 1,15381,2,1677,1835,41216,5210,51,3 --gen 16 \
  --dir ~/code/moondream3_310b/models/moondream3_310b
```

PC 端一键图像问答：

```bash
python tools/run_qa.py --image circle.png --question "What color is the circle?" --gen 16
```

## 目录

```
code/moondream3_310b/
  README.md  PROJECT_STATUS.md
  docs/      OPEN_SOURCE / ARCHITECTURE / WEIGHT_FORMAT / OP_REUSE / BOARD_CAPABILITY / DEPLOY / BENCHMARK / SPECULATIVE
  ops/       AscendC 自研算子 + C++ 混合引擎（CMake，板端 bash run.sh build）
  tools/     convert_moondream3 / convert_vision_f16 / ref_moondream3 / prep_image / run_qa / run_demo / compare_logits / bench.sh
  ref/       官方 modeling 源码（对照，BSL 1.1）
```

## 关键数字

- 权重：文本 W4A16 5.4GB（int4 group64 → W4A16 group32）+ vision f16 0.85GB；
- 板端：23GB 统一内存，1 AI Core（310B1 固定），CANN 8.0.0；
- 文本 decode 2.38 s/tok、vision encode 40 s、image prefix 30 s、图像问答端到端 145 s；
- 详细基准见 docs/BENCHMARK.md。

## 许可提醒

Moondream 3 Preview 权重为 **BSL 1.1 + Additional Use Grant**（非 OSI 开源，2 年后转 Apache-2.0），
商用前请阅读 docs/OPEN_SOURCE.md 中的条款摘要。