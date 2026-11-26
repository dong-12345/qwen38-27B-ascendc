D=/usr/local/Ascend/ascend-toolkit/latest/aarch64-linux/tikcpp/tikcfw
echo "=== Dot ===";       grep -rn "Dot("       $D/interface/ | head -5
echo "=== MulAddDst ==="; grep -rn "MulAddDst"   $D/interface/ | head -5
echo "=== Axpy ===";      grep -rn "Axpy"        $D/interface/ | head -5
echo "=== MulCast ===";   grep -rn "MulCast"     $D/interface/ | head -3
echo "=== Brcb ===";      grep -rn "Brcb"        $D/interface/ | head -3
echo "=== impl dir ===";  ls $D/impl | head
echo "=== dav_m310 ===";  ls $D/impl/dav_m310 2>/dev/null | head -40
