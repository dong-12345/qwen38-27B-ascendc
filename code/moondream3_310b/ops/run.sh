#!/bin/bash
# Build the Moondream3 310B engine + ops.
set -e
CURRENT_DIR=$(cd $(dirname ${BASH_SOURCE:-$0}) && pwd)
cd "$CURRENT_DIR"
SOC_VERSION=${SOC_VERSION:-Ascend310B1}
ASCEND_HOME=${ASCEND_HOME:-/usr/local/Ascend/ascend-toolkit/latest}
export ASCEND_TOOLKIT_HOME=${ASCEND_HOME}
export ASCEND_HOME_PATH=${ASCEND_HOME}
source ${ASCEND_HOME}/bin/setenv.bash >/dev/null 2>&1 || source ${ASCEND_HOME}/set_env.sh >/dev/null 2>&1

build() {
    rm -rf build out
    mkdir -p build
    cmake -B build -DRUN_MODE=npu -DSOC_VERSION=${SOC_VERSION} -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=${CURRENT_DIR}/out -DASCEND_CANN_PACKAGE_PATH=${ASCEND_HOME} \
        > build/cmake.log 2>&1 || { tail -40 build/cmake.log; exit 1; }
    cmake --build build -j4 > build/build.log 2>&1 || {
        echo "[ERROR] build failed:"
        grep -E "error|Error|ERROR" build/build.log | head -40
        tail -30 build/build.log
        exit 1
    }
    cmake --install build > /dev/null
    cp -f ./out/bin/md3_engine ./md3_engine
    echo "[build] OK  soc=${SOC_VERSION}"
}
case "${1:-build}" in
    build) build ;;
    *) echo "usage: bash run.sh build"; exit 2 ;;
esac
