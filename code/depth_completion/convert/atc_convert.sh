#!/bin/bash
# Generic ATC ONNX -> OM conversion helper (Ascend 310B1 / CANN 8.0)
# usage: atc_convert.sh <model.onnx> <output_base(no ext)> <input_name:shape> [extra atc args...]
# env: IFMT=input format (default NCHW), PMODE=precision mode (default force_fp16)
set -e
source /usr/local/Ascend/ascend-toolkit/set_env.sh
MODEL="$1"; OUT="$2"; SHAPE="$3"; shift 3 || true
IFMT="${IFMT:-NCHW}"
PMODE="${PMODE:-force_fp16}"
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/runtime/lib64:$LD_LIBRARY_PATH
echo "### ATC convert: $MODEL -> $OUT.om  shape=$SHAPE fmt=$IFMT pmode=$PMODE"
date
time atc --framework=5 \
  --model="$MODEL" \
  --input_format="$IFMT" \
  --input_shape="$SHAPE" \
  --soc_version=Ascend310B1 \
  --precision_mode="$PMODE" \
  --output="$OUT" \
  --log=error "$@" 2>&1 | tail -15
ls -la "$OUT.om"
date
