cd ~/code/qwen38_310B && source env.sh >/dev/null 2>&1
cd src
timeout -s KILL 300 ./out/bin/qwen38_ops_test probe_v2 2>&1 | tail -60
