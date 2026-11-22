#!/usr/bin/env python3
"""Grab one frame with the current camera settings (no configuration writes)."""
import os, sys, time
import numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pd_capture import U3V, decode_payload
from pd_core import colorize

out = sys.argv[1] if len(sys.argv) > 1 else "/tmp/shot"
u = U3V(); u.open()
u.start()
time.sleep(1.0)
best = None
for _ in range(6):
    p, st = u.read_payload(2500)
    if p is None:
        continue
    d, rgb = decode_payload(p)
    if rgb is None or d is None:
        continue
    best = (len(p), d, rgb)
    break
u.stop(); u.close()
if best is None:
    print("no frame")
    sys.exit(1)
n, d, rgb = best
g = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
print("payload=%d  luma mean=%.1f p50=%.1f sat=%.3f  depth valid=%.3f p50=%d" % (
    n, g.mean(), np.median(g), (g > 250).mean(), (d > 0).mean(), np.median(d[d > 0])))
cv2.imencode(".png", rgb)[1].tofile(out + "_rgb.png")
cv2.imencode(".png", colorize(d / 1000.0, 0.3, 8.0))[1].tofile(out + "_depth.png")
print("saved", out + "_rgb.png", out + "_depth.png")
