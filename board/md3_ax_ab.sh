cd ~/code/moondream3_310b/ops
export ASCEND_HOME=/usr/local/Ascend/ascend-toolkit/latest
source $ASCEND_HOME/bin/setenv.bash >/dev/null 2>&1 || true
export LD_LIBRARY_PATH=$PWD/out/lib:$PWD/out/lib64:$ASCEND_HOME/lib64:$ASCEND_HOME/runtime/lib64:$LD_LIBRARY_PATH
ORIG=~/code/moondream3_310b/models/moondream3_310b
AX=~/code/moondream3_310b/models/moondream3_310b_ax
IDS=0,1,15381,2,8299,1158,3
echo "===== 基线 gen2 ====="
timeout -s KILL 900 ./md3_engine --dir $ORIG --ids $IDS --gen 8 2>&1 | tail -3
echo "===== AX 布局 + Axpy-N ====="
MD3_LAYOUT=axpy timeout -s KILL 900 ./md3_engine --dir $AX --ids $IDS --gen 8 2>&1 | tail -3
