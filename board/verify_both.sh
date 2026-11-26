cd ~/code/qwen38_310B && source env.sh >/dev/null 2>&1
cd src
echo "===== 默认路径（批量 prefill + gen2 decode）====="
IDS=248045,846,198,760,6511,314,9338,369,248046,198,248045,74455,198,248068,271,248069,271
ENG27_NOWARMUP=1 ENG27_IDS=$IDS ENG27_GEN=4 timeout -s KILL 1800 ./out/bin/qwen38_ops_test engine_27b 2>&1 | tail -3
echo "===== AX 路径（decode，2 token prompt）====="
ENG27_LAYOUT=axpy ENG27_NOWARMUP=1 ENG27_DIR=$HOME/code/qwen38_310B/models/qwen38_310b_ax ENG27_IDS=248045,846 ENG27_GEN=4 timeout -s KILL 1800 ./out/bin/qwen38_ops_test engine_27b 2>&1 | tail -3