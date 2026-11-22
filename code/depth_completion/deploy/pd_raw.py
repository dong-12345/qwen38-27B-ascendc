#!/usr/bin/env python3
"""Dump raw RGB (uint8 BGR png) and raw depth (uint16 mm png). No processing."""
import os, sys, time
import numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pd_capture import U3V, decode_payload

out = sys.argv[1] if len(sys.argv) > 1 else "/tmp/raw"
u = U3V(); u.open(); u.start(); time.sleep(1.0)
got = None
for _ in range(8):
    p, st = u.read_payload(2500)
    if p is None:
        continue
    d, rgb = decode_payload(p)
    if rgb is None:
        continue
    got = (len(p), d, rgb)
    break
u.stop(); u.close()
if got is None:
    print("no frame"); sys.exit(1)
n, d, rgb = got
g = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
print("payload=%d rgb=%s depth=%s luma=%.1f depth_p50=%d valid=%.3f" % (
    n, rgb.shape, d.shape, g.mean(), np.median(d[d > 0]), (d > 0).mean()))
cv2.imencode(".png", rgb)[1].tofile(out + "_rgb.png")
cv2.imencode(".png", d)[1].tofile(out + "_depth16.png")
print("saved", out + "_rgb.png", out + "_depth16.png")
