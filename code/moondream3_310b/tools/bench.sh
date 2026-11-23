#!/bin/bash
# bench.sh — 板端基准测试（文本 decode / prefill / vision / 算子对拍）
set -u
cd ~/code/moondream3_310b/ops || exit 1
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export LD_LIBRARY_PATH=$PWD/out/lib:$PWD/out/lib64:/usr/local/Ascend/ascend-toolkit/latest/lib64:$LD_LIBRARY_PATH
DIR=~/code/moondream3_310b/models/moondream3_310b
IDS=0,1,15381,2,8299,1158,3
echo "== ops tests =="
./md3_engine test 2>&1 | tail -3
echo "== text decode 24L =="
./md3_engine --dir $DIR --ids $IDS --gen 8 2>&1 | grep -E "weights loaded|decode|TOKENS"
echo "== text prefill 24L =="
./md3_engine --prefill --dir $DIR --ids $IDS --gen 8 2>&1 | grep -E "prompt prefill|decode|TOKENS"
if [ -f /tmp/crops.bin ]; then
  echo "== vision + QA =="
  ./md3_engine --image /tmp/crops.bin --ncrops 2 --qids 1,15381,2,1677,1835,41216,5210,51,3 --gen 8 --dir $DIR 2>&1 | grep -E "vision encode|prefix prefill|decode|TOKENS"
else
  echo "== vision skipped (no /tmp/crops.bin) =="
fi
