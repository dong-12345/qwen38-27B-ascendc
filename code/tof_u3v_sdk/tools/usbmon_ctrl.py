# -*- coding: utf-8 -*-
"""usbmon_ctrl.py - capture U3V control channel (EP0 + bulk 0x01/0x82) and summarize

Usage (root):  sudo python3 usbmon_ctrl.py <seconds> <busnum> <devnum>

Prints:
  - non-standard EP0 control requests (class/vendor)
  - bulk OUT 0x01 packets whose command != READ_MEMORY (0x0800): writes/stream control
  - histogram of command codes and of READ_MEMORY addresses
"""
import os
import select
import struct
import sys
import time

HDR = struct.Struct("<QBBBBHccqiiii8s")   # 48 bytes


def main():
    secs = float(sys.argv[1]) if len(sys.argv) > 1 else 3.0
    bus = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    dev = int(sys.argv[3]) if len(sys.argv) > 3 else -1
    fd = os.open("/dev/usbmon%d" % bus, os.O_RDONLY | os.O_NONBLOCK)
    t_end = time.time() + secs
    buf = b""
    evts = []
    while time.time() < t_end:
        r, _, _ = select.select([fd], [], [], 0.2)
        if not r:
            continue
        try:
            chunk = os.read(fd, 1 << 20)
        except BlockingIOError:
            continue
        buf += chunk
        while len(buf) >= HDR.size:
            (id_, typ, xfer, ep, devnum, busnum, fs, fdat, ts_sec, ts_usec,
             status, len_urb, len_cap, setup) = HDR.unpack_from(buf, 0)
            typ = chr(typ)
            total = HDR.size + (len_cap if 0 <= len_cap <= (1 << 20) else 0)
            if len(buf) < total:
                break
            data = buf[HDR.size:total]
            buf = buf[total:]
            if dev >= 0 and devnum != dev:
                continue
            if fs == b"s" and len(data) >= 8:
                setup = data[:8]
                data = data[8:] if len(data) > 8 else b""
            evts.append((ts_sec * 1e6 + ts_usec, typ, xfer, ep, status, len_urb,
                         len_cap, setup, data))
    os.close(fd)

    t0 = evts[0][0] if evts else 0
    print("capture %.1fs bus=%d dev=%s events=%d" % (secs, bus, dev, len(evts)))
    shown = 0
    stats = {}
    addr_stats = {}
    for (t, typ, xfer, ep, status, len_urb, len_cap, setup, data) in evts:
        rel = (t - t0) / 1000.0
        if xfer == 2 and typ in "Ss":
            bm, br, wv, wi, wl = struct.unpack("<BBHHH", setup[:8])
            if (bm & 0x60) in (0x20, 0x40):
                print("%8.2fms CTRL bm=0x%02x bReq=0x%02x wValue=0x%04x wIndex=0x%04x wLen=%d data=%s"
                      % (rel, bm, br, wv, wi, wl, data[:32].hex()))
                shown += 1
        elif xfer == 3 and ep == 0x01 and len(data) >= 12:
            magic, flags, cmd, size, pid = struct.unpack_from("<IHHHH", data, 0)
            key = (cmd, size)
            stats[key] = stats.get(key, 0) + 1
            if cmd == 0x0800 and len(data) >= 24:
                addr = struct.unpack_from("<Q", data, 12)[0]
                addr_stats[addr] = addr_stats.get(addr, 0) + 1
            if cmd != 0x0800:
                print("%8.2fms CMD cmd=0x%04x size=%d id=%d %s"
                      % (rel, cmd, size, pid, " ".join("%02x" % b for b in data)))
                shown += 1
        if shown > 80:
            print("...(truncated)")
            break
    print("-- bulk OUT 0x01 command histogram (cmd,size): count --")
    for k, v in sorted(stats.items(), key=lambda kv: -kv[1]):
        print("   cmd=0x%04x size=%-3d count=%d" % (k[0], k[1], v))
    print("-- READ_MEMORY address histogram: count --")
    for k, v in sorted(addr_stats.items(), key=lambda kv: -kv[1])[:15]:
        print("   addr=0x%06x count=%d" % (k, v))


if __name__ == "__main__":
    main()
