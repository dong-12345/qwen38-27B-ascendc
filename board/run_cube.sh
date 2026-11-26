cd ~/code/qwen38_310B && source env.sh >/dev/null 2>&1
cd src
echo "=== 两段式 Cube 路径原型（M 个 token 一次算）==="
CUBE_M=32 timeout -s KILL 900 ./out/bin/qwen38_ops_test cube_prefill 2>&1 | grep -E "cube_prefill"
