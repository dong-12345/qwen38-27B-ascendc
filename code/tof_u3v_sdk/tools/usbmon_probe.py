# -*- coding: utf-8 -*-
"""usbmon 二进制接口探针：确认 /dev/usbmonN 可用，能读到 URB 事件（S/C）。
板子上需要 root 运行。用法： sudo python3 usbmon_probe.py [秒] [bus]
"""
import os
import struct
import sys
import time
import select

HDR = struct.Struct("<QBBBBHccqiiii8s")   # mon_bin_hdr, 48 bytes
# 字段: id, type, xfer_type, epnum, devnum, busnum, flag_setup, flag_data,
#       ts_sec, ts_usec, status, len_urb, len_cap, setup
XFER = {0: "iso", 1: "intr", 2: "ctrl", 3: "bulk"}
TYPES = {"S": "submit", "C": "complete", "E": "error", "s": "submit",
         "c": "complete", "e": "error"}

secs = float(sys.argv[1]) if len(sys.argv) > 1 else 3.0
bus = int(sys.argv[2]) if len(sys.argv) > 2 else 4
path = "/dev/usbmon%d" % bus
fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
print("opened", path)
t0 = time.time()
n = 0
evts = []
buf = b""
while time.time() - t0 < secs:
    r, _, _ = select.select([fd], [], [], 0.2)
    if not r:
        continue
    try:
        chunk = os.read(fd, 65536)
    except BlockingIOError:
        continue
    buf += chunk
    while len(buf) >= HDR.size:
        vals = HDR.unpack_from(buf, 0)
        (id_, typ, xfer, ep, dev, busnum, fs, fd_, ts_sec, ts_usec, status,
         len_urb, len_cap, setup) = vals
        total = HDR.size + (len_cap if typ in "SC" else 0)
        if len(buf) < total:
            break
        evts.append((id_, typ, xfer, ep, dev, busnum, ts_sec, ts_usec, status,
                     len_urb, len_cap))
        buf = buf[total:]
        n += 1
os.close(fd)
print("events:", n)
for e in evts[:15]:
    (id_, typ, xfer, ep, dev, busnum, ts_sec, ts_usec, status, len_urb, len_cap) = e
    print("id=%x %-8s %-5s ep=0x%02x dev=%d bus=%d ts=%d.%06d status=%d urb=%d cap=%d"
          % (id_, TYPES.get(typ, typ), XFER.get(xfer, xfer), ep, dev, busnum,
             ts_sec, ts_usec, status, len_urb, len_cap))
