#!/usr/bin/env python3
"""Measure per-frame transfer timing (USB vs camera)."""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pd_capture import U3V, decode_payload
u = U3V(); u.open(); u.start(); time.sleep(0.3)
ts = []
sizes = []
t0 = time.time()
while time.time() - t0 < 5.0:
    a = time.perf_counter()
    p, st = u.read_payload(2000)
    b = time.perf_counter()
    if p is None:
        continue
    ts.append((a, b))
    sizes.append(len(p))
u.stop(); u.close()
if len(ts) > 2:
    gaps = [(ts[i+1][0] - ts[i][0]) * 1000 for i in range(len(ts)-1)]
    durs = [(b-a)*1000 for a, b in ts]
    print("frames=%d over 5s  fps=%.2f" % (len(ts), len(ts)/5.0))
    print("frame gap p50=%.1fms (camera)   transfer p50=%.1fms (usb)" % (
        sorted(gaps)[len(gaps)//2], sorted(durs)[len(durs)//2]))
    print("sizes:", sorted(set(sizes)))
