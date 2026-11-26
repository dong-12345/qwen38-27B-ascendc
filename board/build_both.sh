cd ~/code/qwen38_310B && source env.sh >/dev/null 2>&1
cd src && bash run.sh build 2>&1 | tail -2
cd ~/code/moondream3_310b/ops && bash run.sh build 2>&1 | tail -2
echo BUILD_ALL_OK