#!/usr/bin/env python3
"""Try to restore the colour sensor by rewriting its exposure register."""
import os, struct, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pd_capture import U3V

REG_COLOR_EXP = 0x01020000
u = U3V(); u.open()
for val in (int(a) for a in sys.argv[1:]) if len(sys.argv) > 1 else (2000, 1000, 500, 100, 0):
    u.stop(); time.sleep(0.2)
    rc = u.write_mem(REG_COLOR_EXP, struct.pack("<I", val))
    u.start(); time.sleep(0.8)
    got = None
    for _ in range(3):
        p, st = u.read_payload(1200)
        if p is not None:
            got = (len(p), p)
            break
    print("color_reg=%d write=%s payload=%s" % (val, rc, (got[0] if got else "none")))
    if got and got[0] != 6205488 and got[0] != 6512688:
        print("  odd size", got[0])
u.stop(); u.close()
