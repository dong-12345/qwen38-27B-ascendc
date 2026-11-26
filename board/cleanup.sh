pkill -9 -f qwen38_ops_test; sleep 2
ps -o pid,stat,cmd -C qwen38_ops_test | head -5
echo "--- npu ---"
npu-smi info | sed -n '9,12p'
echo "--- mem ---"
free -m | head -2
