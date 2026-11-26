cd ~/code/qwen38_310B && source env.sh >/dev/null 2>&1
cd src
echo "=== gemv_batch (K=5120 N=17408) ==="
timeout -s KILL 600 ./out/bin/qwen38_ops_test gemv_batch 2>&1 | tail -12
echo "=== gemv_batch (K=17408 N=5120) ==="
GEMVB_K=17408 GEMVB_N=5120 timeout -s KILL 600 ./out/bin/qwen38_ops_test gemv_batch 2>&1 | tail -6
