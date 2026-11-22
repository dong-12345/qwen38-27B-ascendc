#!/usr/bin/env python3
"""Probe the U3V register read command (0x0801) to verify camera settings."""
import ctypes, os, struct, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pd_capture import U3V, EP_OUT, EP_IN

def read_mem(u, addr, n=4, cmd=0x0801):
    u.pid = (u.pid + 1) & 0xFFFF or 1
    pkt = struct.pack("<I H H H H", 0x43563355, 0x4000, cmd, 8, u.pid) + struct.pack("<Q", addr)
    rc, _ = u._out(pkt)
    if rc:
        return "wr_err:" + u.err(rc)
    rc, d = u._in(256)
    if rc:
        return "rd_err:" + u.err(rc)
    return d

u = U3V()
u.open()
u.stop()
time.sleep(0.3)
for addr, name in ((0x00030B08, "ExposureAuto"), (0x00030B04, "ExposureTime"), (0x01020000, "ColorExposure")):
    for cmd in (0x0801, 0x0800):
        r = read_mem(u, addr, 4, cmd)
        if isinstance(r, bytes):
            print("%-14s cmd=0x%04x raw(%d)=%s" % (name, cmd, len(r), r[:24].hex()))
        else:
            print("%-14s cmd=0x%04x %s" % (name, cmd, r))
u.start()
time.sleep(0.3)
u.stop()
u.close()
