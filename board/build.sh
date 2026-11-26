set -e
cd ~/code/qwen38_310B/src
export ASCEND_HOME=/usr/local/Ascend/ascend-toolkit/latest
export ASCEND_TOOLKIT_HOME=$ASCEND_HOME
export ASCEND_HOME_PATH=$ASCEND_HOME
source $ASCEND_HOME/bin/setenv.bash >/dev/null 2>&1 || source $ASCEND_HOME/set_env.sh >/dev/null 2>&1 || true
export LD_LIBRARY_PATH=$PWD/out/lib:$PWD/out/lib64:$ASCEND_HOME/lib64:$ASCEND_HOME/runtime/lib64:$LD_LIBRARY_PATH
cd build
cmake .. > /tmp/cmake.log 2>&1 || { tail -20 /tmp/cmake.log; exit 1; }
make -j8 > /tmp/make.log 2>&1 || { grep -E "error|Error" /tmp/make.log | head -20; tail -10 /tmp/make.log; exit 1; }
tail -2 /tmp/make.log
cmake --install . > /dev/null
cp -f ../out/bin/qwen38_ops_test ../
echo BUILD_OK
