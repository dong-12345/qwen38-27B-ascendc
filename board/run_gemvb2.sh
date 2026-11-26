cd ~/code/qwen38_310B && source env.sh >/dev/null 2>&1
cd src
echo "=== gemv_batch: stage99=4相位合并 / stage97=逐相位 / stage98=元素偏移 ==="
timeout -s KILL 900 ./out/bin/qwen38_ops_test gemv_batch 2>&1 | grep -E "stage=|speedup"
