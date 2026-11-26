cd ~/code/qwen38_310B && source env.sh >/dev/null 2>&1
cd src
IDS=248045,846,198,760,6511,314,9338,369,248046,198,248045,74455,198,248068,271,248069,271
echo "===== V1 (old w4a16_gemv_gen) ====="
ENG27_NOWARMUP=1 ENG27_GEMV_V2=0 ENG27_IDS=$IDS ENG27_GEN=4 timeout -s KILL 1200 ./out/bin/qwen38_ops_test engine_27b 2>&1 | tail -3
echo "===== V2 (w4a16_gemv_batch) ====="
ENG27_NOWARMUP=1 ENG27_IDS=$IDS ENG27_GEN=4 timeout -s KILL 1200 ./out/bin/qwen38_ops_test engine_27b 2>&1 | tail -3
