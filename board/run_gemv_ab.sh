cd ~/code/qwen38_310B && source env.sh >/dev/null 2>&1
cd src
echo "=== gen vs gen2 对拍+计时 (K=5120 N=17408) ==="
timeout -s KILL 600 ./out/bin/qwen38_ops_test gemv_ab 2>&1 | grep -E "gemv_ab"
echo "=== 同形状 K=6144 N=5120 ==="
GEMVB_K=6144 GEMVB_N=5120 timeout -s KILL 600 ./out/bin/qwen38_ops_test gemv_ab 2>&1 | grep -E "gemv_ab"
