#!/usr/bin/env python3
"""Dump one raw payload (no configuration writes)."""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pd_capture import U3V
u = U3V(); u.open(); u.start(); time.sleep(1.0)
for _ in range(5):
    p, st = u.read_payload(2500)
    if p is None:
        continue
    open("/tmp/pd_dyn.bin", "wb").write(p)
    print("dumped", len(p))
    break
u.stop(); u.close()
