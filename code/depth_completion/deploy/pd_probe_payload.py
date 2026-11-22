#!/usr/bin/env python3
"""Dump one U3V payload and guess its layout."""
import os, sys, struct, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pd_capture import U3V, DEPTH_SHAPE

u = U3V()
u.open()
u.start()
t0 = time.time()
nfr = 0
sizes = {}
for _ in range(3):
    p, st = u.read_payload(3000)
    if p is None:
        print("no payload", st); break
    sizes[len(p)] = sizes.get(len(p), 0) + 1
    nfr += 1
    if nfr == 2:
        open("/tmp/pd_payload.bin", "wb").write(p)
        print("dumped", len(p), "bytes to /tmp/pd_payload.bin")
u.stop(); u.close()
print("sizes:", sizes, "elapsed %.1fs" % (time.time()-t0))

p = open("/tmp/pd_payload.bin", "rb").read()
print("payload", len(p))
h, w = DEPTH_SHAPE
for off in (0, 48, 64):
    if len(p) >= off + h*w*2:
        d = np.frombuffer(p[off:off+h*w*2], dtype=np.uint16).reshape(h, w)
        print("u16 depth @%d: valid%%=%.1f min=%d med=%d max=%d" % (off, (d > 0).mean()*100, d.min(), int(np.median(d)), d.max()))
for sig, name in ((b"\xff\xd8\xff", "JPEG"), (b"\x89PNG", "PNG")):
    idx = p.find(sig)
    print("%s at %s" % (name, idx))
rest = len(p) - 48 - h*w*2
print("remainder after 48+depth:", rest, " per-pixel bytes for 1536x1280:", rest / (1536*1280))
for off in (48 + h*w*2, 48, 614400, 0):
    n = 1536*1280*3
    if len(p) >= off + n:
        a = np.frombuffer(p[off:off+n], dtype=np.uint8).reshape(1280, 1536, 3)
        print("BGR @%d: mean=%s std=%s" % (off, np.round(a.mean(axis=(0,1)),1), np.round(a.std(axis=(0,1)),1)))
        break
