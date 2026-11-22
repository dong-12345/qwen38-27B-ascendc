#!/usr/bin/env python3
"""Functional check of continuous auto-exposure: force a short exposure, then
let ExposureAuto=Continuous recover the brightness."""
import os, struct, sys, time
import numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pd_capture import U3V, decode_payload

REG_EXP_AUTO = 0x00030B08
REG_EXP_TIME = 0x00030B04

u = U3V(); u.open(); u.stop(); time.sleep(0.3)

def grab(n=2, tag=""):
    u.start(); time.sleep(0.5)
    out = []
    for _ in range(n):
        p, st = u.read_payload(2000)
        if p is None:
            continue
        d, rgb = decode_payload(p)
        if rgb is None:
            continue
        g = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
        out.append((float(g.mean()), int(np.median(d[d > 0])), rgb, d))
    u.stop(); time.sleep(0.2)
    if out:
        print("%-28s luma=%s depth_p50=%s" % (tag, [round(o[0], 1) for o in out], [o[1] for o in out]))
    return out

base = grab(2, "baseline")
print("set ExposureTime=300 ->", u.write_mem(REG_EXP_TIME, struct.pack("<I", 300)))
short = grab(2, "short exposure")
print("set ExposureAuto=2 ->", u.write_mem(REG_EXP_AUTO, struct.pack("<I", 2)))
time.sleep(1.0)
auto = grab(4, "ExposureAuto=Continuous")
print("set ExposureAuto=0 ->", u.write_mem(REG_EXP_AUTO, struct.pack("<I", 0)))
off = grab(2, "auto off again")
if auto:
    cv2.imwrite("/tmp/auto_rgb.png", cv2.cvtColor(auto[-1][2], cv2.COLOR_BGR2RGB))
u.close()
