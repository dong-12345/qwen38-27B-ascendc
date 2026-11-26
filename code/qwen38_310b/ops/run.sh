#!/bin/bash
# Build + run + verify the self-developed AscendC ops for Ascend310B1.
# Usage: bash run.sh [build|test|all] [test_name ...]
set -e

CURRENT_DIR=$(cd $(dirname ${BASH_SOURCE:-$0}) && pwd)
cd "$CURRENT_DIR"

SOC_VERSION=${SOC_VERSION:-Ascend310B1}
ASCEND_HOME=${ASCEND_HOME:-/usr/local/Ascend/ascend-toolkit/latest}
PY=${PY:-/usr/local/miniconda3/bin/python}

export ASCEND_TOOLKIT_HOME=${ASCEND_HOME}
export ASCEND_HOME_PATH=${ASCEND_HOME}
source ${ASCEND_HOME}/bin/setenv.bash >/dev/null 2>&1 || source ${ASCEND_HOME}/set_env.sh >/dev/null 2>&1

MODE=${1:-all}
shift || true
TESTS=("$@")
if [ ${#TESTS[@]} -eq 0 ]; then
    TESTS=(add gdn_gating)
fi

build() {
    rm -rf build out
    mkdir -p build
    cmake -B build \
        -DRUN_MODE=npu \
        -DSOC_VERSION=${SOC_VERSION} \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=${CURRENT_DIR}/out \
        -DASCEND_CANN_PACKAGE_PATH=${ASCEND_HOME} > build/cmake.log 2>&1 || { tail -30 build/cmake.log; exit 1; }
    cmake --build build -j4 > build/build.log 2>&1 || {
        echo "[ERROR] build failed, tail:"
        grep -E "error|Error|ERROR" build/build.log | head -30
        tail -20 build/build.log
        exit 1
    }
    cmake --install build > /dev/null
    cp -f ./out/bin/qwen38_ops_test ./
    echo "[build] OK  soc=${SOC_VERSION}"
}

prepare() {
    mkdir -p input output input_golden
    $PY scripts/gen_data.py
}

run_tests() {
    export LD_LIBRARY_PATH=${CURRENT_DIR}/out/lib:${CURRENT_DIR}/out/lib64:${ASCEND_HOME}/lib64:$LD_LIBRARY_PATH
    for t in "${TESTS[@]}"; do
        echo "=== run $t ==="
        ./qwen38_ops_test "$t"
    done
    $PY scripts/verify.py
}

case "$MODE" in
    build) build ;;
    test)  prepare; run_tests ;;
    all)   build; prepare; run_tests ;;
    *) echo "usage: bash run.sh [build|test|all] [tests...]"; exit 2 ;;
esac
