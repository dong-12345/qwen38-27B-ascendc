set -e
D=/usr/local/Ascend/ascend-toolkit/latest/aarch64-linux/tikcpp/tikcfw
mkdir -p ~/hdr
cp -r $D/interface ~/hdr/ 2>/dev/null || true
cp -r $D/impl/dav_m310 ~/hdr/ 2>/dev/null || true
cp $D/kernel_operator.h $D/kernel_tensor.h ~/hdr/ 2>/dev/null || true
du -sh ~/hdr
ls ~/hdr
