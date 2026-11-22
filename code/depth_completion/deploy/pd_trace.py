#!/usr/bin/env python3
"""Record the raw U3V transfer sequence (sizes + bytes) for offline framing analysis."""
import ctypes, os, struct, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pd_capture import U3V, EP_STREAM

secs = float(sys.argv[1]) if len(sys.argv) > 1 else 1.5
u = U3V(); u.open(); u.start()
buf = (ctypes.c_ubyte * (8 << 20))()
sizes = []
blob = bytearray()
t0 = time.time()
while time.time() - t0 < secs and len(blob) < 40_000_000:
    n = ctypes.c_int(0)
    rc = u.lib.libusb_bulk_transfer(u.h, EP_STREAM, buf, 8 << 20, ctypes.byref(n), 800)
    if rc != 0:
        sizes.append(-1)
        continue
    sz = n.value
    sizes.append(sz)
    blob += bytes(buf[:sz])
u.stop(); u.close()
open("/tmp/raw_stream.bin", "wb").write(bytes(blob))
open("/tmp/raw_sizes.txt", "w").write("\n".join(str(s) for s in sizes))
print("transfers=%d bytes=%d secs=%.2f" % (len(sizes), len(blob), secs))
from collections import Counter
print("size histogram:", Counter(sizes).most_common(8))
print("first 30:", sizes[:30])
