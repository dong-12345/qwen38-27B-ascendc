#!/usr/bin/env python3
"""Capture with a MINIMAL start sequence: only enable the data stream, leave the
camera's own (HiViewer) configuration untouched."""
import os, struct, sys, time
import numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pd_capture import U3V, SIRM, decode_payload
from pd_core import colorize

out = sys.argv[1] if len(sys.argv) > 1 else "/tmp/minimal"
u = U3V(); u.open()
# enable streaming only (no format/mode registers)
print("stream enable ->", u.write_mem(0x030804, struct.pack("<I", 1)))
print("sirm ctrl    ->", u.write_mem(SIRM + 0x04, struct.pack("<I", 1)))
time.sleep(1.2)
got = None
for i in range(8):
    p, st = u.read_payload(2000)
    if p is None:
        print("timeout", st); continue
    print("payload", len(p))
    d, rgb = decode_payload(p)
    if rgb is None:
        continue
    got = (len(p), d, rgb)
    break
u.write_mem(0x030804, struct.pack("<I", 0))
u.write_mem(SIRM + 0x04, struct.pack("<I", 0))
u.close()
if got is None:
    print("no usable frame"); sys.exit(1)
n, d, rgb = got
g = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
print("luma mean=%.1f p50=%.1f  depth p50=%d valid=%.3f" % (g.mean(), np.median(g), np.median(d[d > 0]), (d > 0).mean()))
cv2.imencode(".png", rgb)[1].tofile(out + "_rgb.png")
cv2.imencode(".png", colorize(d / 1000.0, 0.5, 8.0))[1].tofile(out + "_depth.png")
print("saved", out + "_rgb.png")
