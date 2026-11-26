D=/usr/local/Ascend/ascend-toolkit/latest/aarch64-linux/tikcpp/tikcfw
rm -rf ~/hdr2 && mkdir -p ~/hdr2
cp -rL $D/interface ~/hdr2/iface
tar czhf ~/hdr2.tgz -C ~/hdr2 iface && ls -la ~/hdr2.tgz
