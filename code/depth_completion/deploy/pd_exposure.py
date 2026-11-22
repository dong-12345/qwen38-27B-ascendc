#!/usr/bin/env python3
"""Camera control over U3V (auto-exposure / exposure time) + frame check.

usage: pd_exposure.py [--auto 0|1|2] [--expo us] [--color 1] [--reps 3]
Decodes both payload layouts:
  A 6,512,688 B : 48 + 2x320x480 depth planes (column-interleaved) + 3x1536x1280 planar RGB
  B 6,205,488 B : 48 + 1x307200 depth block + 3x1536x1280 planar RGB
"""
import argparse, os, struct, sys, time
import numpy as np, cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pd_capture import U3V

REG_EXP_AUTO = 0x00030B08
REG_EXP_TIME = 0x00030B04
REG_COLOR_EXP = 0x01020000
RGB_BYTES = 1536 * 1280 * 3


def decode(p):
    body = p[48:]
    cblk = body[-RGB_BYTES:]
    dblk = body[:-RGB_BYTES]
    pl = np.frombuffer(cblk, dtype=np.uint8).reshape(3, 1280, 1536)
    rgb = np.stack([pl[0], pl[1], pl[2]], axis=2)
    if len(dblk) == 614400:
        dp = np.frombuffer(dblk, dtype=np.uint16).reshape(2, 480, 320)
        d = np.empty((480, 640), np.uint16); d[:, 0::2] = dp[0]; d[:, 1::2] = dp[1]
        kind = "640x480x2planes"
    elif len(dblk) == 307200:
        d = np.frombuffer(dblk, dtype=np.uint16).reshape(480, 320)
        kind = "320x480x1plane"
    else:
        d, kind = None, "unknown(%d)" % len(dblk)
    return rgb, d, kind


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--auto", type=int, default=-1)
    ap.add_argument("--expo", type=int, default=-1)
    ap.add_argument("--color", type=int, default=-1)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--save", default="")
    a = ap.parse_args()
    u = U3V(); u.open(); u.stop(); time.sleep(0.3)
    if a.expo >= 0:
        print("ExposureTime=%d ->" % a.expo, u.write_mem(REG_EXP_TIME, struct.pack("<I", a.expo)))
    if a.color >= 0:
        print("ColorReg=0x01020000 <- %d ->" % a.color, u.write_mem(REG_COLOR_EXP, struct.pack("<I", a.color)))
    if a.auto >= 0:
        print("ExposureAuto=%d ->" % a.auto, u.write_mem(REG_EXP_AUTO, struct.pack("<I", a.auto)))
    u.start()
    time.sleep(1.0)
    for i in range(a.reps):
        p, st = u.read_payload(2500)
        if p is None:
            print("no payload", st); continue
        rgb, d, kind = decode(p)
        g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        print("frame %d size=%d %s luma mean=%.1f p50=%.1f sat=%.3f depth=%s" % (
            i, len(p), kind, g.mean(), np.median(g), (g > 250).mean(),
            ("p50=%d valid=%.2f" % (np.median(d[d > 0]), (d > 0).mean())) if d is not None and (d > 0).any() else "n/a"))
        if a.save and i == 0:
            cv2.imwrite(a.save, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            print("saved", a.save)
    u.stop(); u.close()


if __name__ == "__main__":
    main()
