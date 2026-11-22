#!/usr/bin/env python3
"""pd_complete.py - depth completion in its own process.

Processes the frame the capture process just wrote (ping-pong buffers) and
publishes the result into a third buffer for the publisher process.  Keeping
this out of the capture process removes the 16 ms of per-frame waiting that
used to sit between "frame received" and "completion done".
"""
import os
import sys
import time
from multiprocessing import shared_memory

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
cv2.setNumThreads(1)

import pd_core as C

W, H = 640, 480


def run(rgb_names, raw_names, out_names, meta, temporal, regions, median_k, outlier_mm):
    rgbs, raws, outs = [], [], []
    for i in range(2):
        r = shared_memory.SharedMemory(name=rgb_names[i])
        a = shared_memory.SharedMemory(name=raw_names[i])
        o = shared_memory.SharedMemory(name=out_names[i])
        rgbs.append((r, np.ndarray((H, W, 3), dtype=np.uint8, buffer=r.buf)))
        raws.append((a, np.ndarray((H, W), dtype=np.uint16, buffer=a.buf)))
        outs.append((o, np.ndarray((H, W), dtype=np.uint16, buffer=o.buf)))

    comp = C.ClassicCompleter(temporal=temporal, regions=regions,
                              median_k=median_k, outlier_mm=outlier_mm)
    last = -1
    n = 0
    lat = []
    t_start = time.time()
    while meta[0] == 0:
        c = int(meta[2])
        if c == last:
            time.sleep(0.001)
            continue
        last = c
        slot = c % 2
        rgb = rgbs[slot][1].copy()
        dep = raws[slot][1].copy()
        t0 = time.time()
        out = comp(dep, rgb)
        dt = (time.time() - t0) * 1e3
        outs[slot][1][:] = out
        meta[6] = dt
        meta[3] = float(c)
        lat.append(dt)
        n += 1
        if n % 60 == 0:
            print("[cmp] n=%d %.1f fps  complete p50=%.1f ms  holes=%.2f%%"
                  % (n, n / (time.time() - t_start), float(np.median(lat[-60:])),
                     comp.hole_frac * 100), flush=True)

    for lst in (rgbs, raws, outs):
        for shm, _ in lst:
            shm.close()
    return n
