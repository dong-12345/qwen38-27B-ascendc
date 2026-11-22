# -*- coding: utf-8 -*-
"""usbmon_writes.py - print ALL U3V WRITE_MEMORY commands (addr, value) sent to the camera

Usage (root): sudo python3 usbmon_writes.py <seconds> <bus> <dev>
Run while starting vendor SDK demo; catches the whole initialization sequence.
"""
import os
import select
import struct
import sys
import time

HDR = struct.Struct("<QBBBBHccqiiii8s")


def main():
    secs = float(sys.argv[1]) if len(sys.argv) > 1 else 5.0
    bus = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    dev = int(sys.argv[3]) if len(sys.argv) > 3 else -1
    fd = os.open("/dev/usbmon%d" % bus, os.O_RDONLY | os.O_NONBLOCK)
    t_end = time.time() + secs
    buf = b""
    out = []
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
            if xfer == 3 and ep == 0x01 and typ in "Ss" and len(data) >= 20:
                # 只统计提交事件，避免重复
                magic, flags, cmd, size, pid = struct.unpack_from("<IHHHH", data, 0)
                if cmd == 0x0802:
                    addr = struct.unpack_from("<Q", data, 12)[0]
                    payload = data[20:20 + max(0, size - 8)]
                    out.append((ts_sec * 1e6 + ts_usec, addr, payload))
    os.close(fd)
    if not out:
        print("no WRITE_MEMORY captured")
        return
    t0 = out[0][0]
    print("WRITE_MEMORY commands (%d):" % len(out))
    for t, addr, payload in out:
        print("  +%7.1fms  addr=0x%06x  data=%s" % ((t - t0) / 1000.0, addr, payload.hex()))


if __name__ == "__main__":
    main()
