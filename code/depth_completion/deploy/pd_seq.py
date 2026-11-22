#!/usr/bin/env python3
"""Dump N consecutive raw ToF frames (mm) for offline filter comparison."""
import os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pd_capture import U3V, decode_payload
n = int(sys.argv[1]) if len(sys.argv) > 1 else 8
u = U3V(); u.open(); u.start(); time.sleep(0.4)
frames = []
for _ in range(n * 3):
    p, st = u.read_payload(1500)
    if p is None:
        continue
    d, rgb = decode_payload(p)
    if d is None:
        continue
    frames.append(d)
    if len(frames) >= n:
        break
u.stop(); u.close()
np.save("/tmp/tof_seq.npy", np.stack(frames))
print("saved", len(frames), "frames", frames[0].shape)
