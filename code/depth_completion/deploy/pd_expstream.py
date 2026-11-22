#!/usr/bin/env python3
"""Does writing the exposure register during streaming change the image?"""
import os, struct, sys, time
import numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pd_capture import U3V, decode_payload

REG_EXP_TIME = 0x00030B04
u = U3V(); u.open(); u.stop(); time.sleep(0.2)
u.start(); time.sleep(0.8)

def luma(tag):
    for _ in range(3):
        p, st = u.read_payload(1500)
        if p is None:
            continue
        d, rgb = decode_payload(p)
        if rgb is None:
            continue
        print("%-18s luma=%.1f" % (tag, cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY).mean()))
        return
    print("%-18s no frame" % tag)

luma("streaming")
for v in (200, 500, 2000, 5000):
    rc = u.write_mem(REG_EXP_TIME, struct.pack("<I", v))
    time.sleep(0.6)
    luma("expo=%d" % v)
u.stop(); u.close()
