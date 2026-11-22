#!/bin/bash
# Build the Hikrobot MV-DT01SDU calibration exporter on the board.
# Requires the Mv3dRgbd SDK installed (default /opt/Mv3dRgbdSDK).
set -e
SDK=${SDK:-/opt/Mv3dRgbdSDK}
CC=${CC:-gcc}
OUT=${OUT:-/home/HwHiAiUser/code/depth_completion/deploy}
echo "SDK=$SDK"
ls "$SDK/include/Mv3dRgbdApi.h" >/dev/null
$CC -O2 -o "$OUT/get_rgbd_calib" "$(dirname "$0")/get_rgbd_calib.c" \
    -I"$SDK/include" \
    -L"$SDK/lib/aarch64" -lMv3dRgbd \
    -Wl,-rpath,"$SDK/lib/aarch64"
echo "built $OUT/get_rgbd_calib"
echo "run:  $OUT/get_rgbd_calib | tee ~/code/depth_completion/results/rgbd_calib.json"
echo "note: stop other camera readers first (u3v_lowlat / vendor SDK demos)."
