cd ~/code/qwen38_310B && source env.sh >/dev/null 2>&1
cd src
echo "=== v1 (no PipeBarrier<V>) GEMV 计时 ==="
timeout -s KILL 600 ./out/bin/qwen38_ops_test gemv_batch 2>&1 | grep -E "speedup|stage=99"
echo "=== 语义（应与基线 TOKENS 760 6511 314 9338 一致）==="
IDS=248045,846,198,760,6511,314,9338,369,248046,198,248045,74455,198,248068,271,248069,271
ENG27_NOWARMUP=1 ENG27_IDS=$IDS ENG27_GEN=4 timeout -s KILL 1200 ./out/bin/qwen38_ops_test engine_27b 2>&1 | tail -3
