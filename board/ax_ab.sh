cd ~/code/qwen38_310B && source env.sh >/dev/null 2>&1
cd src
IDS=248045,846
echo "########## 基线：原布局 + gen2 ##########"
ENG27_NOWARMUP=1 ENG27_IDS=$IDS ENG27_GEN=4 timeout -s KILL 1800 ./out/bin/qwen38_ops_test engine_27b 2>&1 | tail -3
echo "########## AX 布局 + Axpy-N kernel ##########"
ENG27_LAYOUT=axpy ENG27_NOWARMUP=1 ENG27_DIR=$HOME/code/qwen38_310B/models/qwen38_310b_ax ENG27_IDS=$IDS ENG27_GEN=4 timeout -s KILL 1800 ./out/bin/qwen38_ops_test engine_27b 2>&1 | tail -5
