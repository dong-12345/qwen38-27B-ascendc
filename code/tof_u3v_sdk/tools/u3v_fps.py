# -*- coding: utf-8 -*-
"""u3v_fps.py — libusb 直连 U3V：按 leader 里的帧号算真实帧率，并统计数据吞吐/CPU

用法： python3 u3v_fps.py [seconds]
"""
import ctypes
import struct
import sys
import time

VID, PID = 0x2BDF, 0x0001
EP_OUT, EP_IN, EP_STREAM = 0x01, 0x82, 0x84
SIRM = 0x62000

lib = ctypes.CDLL("libusb-1.0.so.0")
lib.libusb_init.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
lib.libusb_open_device_with_vid_pid.argtypes = [ctypes.c_void_p, ctypes.c_uint16, ctypes.c_uint16]
lib.libusb_open_device_with_vid_pid.restype = ctypes.c_void_p
lib.libusb_claim_interface.argtypes = [ctypes.c_void_p, ctypes.c_int]
lib.libusb_release_interface.argtypes = [ctypes.c_void_p, ctypes.c_int]
lib.libusb_bulk_transfer.argtypes = [ctypes.c_void_p, ctypes.c_ubyte,
                                     ctypes.POINTER(ctypes.c_ubyte), ctypes.c_int,
                                     ctypes.POINTER(ctypes.c_int), ctypes.c_uint]
lib.libusb_close.argtypes = [ctypes.c_void_p]
lib.libusb_error_name.argtypes = [ctypes.c_int]
lib.libusb_error_name.restype = ctypes.c_char_p
_pid = [0]


def err(rc):
    return lib.libusb_error_name(rc).decode()


def _out(pkt):
    b = (ctypes.c_ubyte * len(pkt)).from_buffer_copy(pkt)
    n = ctypes.c_int(0)
    return lib.libusb_bulk_transfer(h, EP_OUT, b, len(pkt), ctypes.byref(n), 500), n.value


def _in(size=1024):
    b = (ctypes.c_ubyte * size)()
    n = ctypes.c_int(0)
    rc = lib.libusb_bulk_transfer(h, EP_IN, b, size, ctypes.byref(n), 500)
    return rc, bytes(bytearray(b[:n.value]))


def _nid():
    _pid[0] = (_pid[0] + 1) & 0xFFFF or 1
    return _pid[0]


def write_mem(addr, payload):
    pkt = struct.pack("<I H H H H", 0x43563355, 0x4000, 0x0802, 8 + len(payload), _nid()) + \
        struct.pack("<Q", addr) + payload
    rc, _ = _out(pkt)
    if rc:
        return "OUT-" + err(rc)
    rc, d = _in(64)
    if rc or len(d) < 12:
        return "IN-" + err(rc)
    magic, status, c, sz, rid = struct.unpack("<I H H H H", d[:12])
    return None if status == 0 else "status=0x%04x" % status


def cpu_times():
    with open("/proc/stat") as f:
        p = f.readline().split()
    v = [int(x) for x in p[1:]]
    return sum(v) - (v[3] + (v[4] if len(v) > 4 else 0)), sum(v)


ctx = ctypes.c_void_p()
lib.libusb_init(ctypes.byref(ctx))
h = lib.libusb_open_device_with_vid_pid(ctx, VID, PID)
if not h:
    sys.exit("open failed")
for itf in (0, 1, 2):
    lib.libusb_claim_interface(h, itf)
SEC = float(sys.argv[1]) if len(sys.argv) > 1 else 6.0
try:
    for addr, data in [
        (SIRM + 0x1C, struct.pack("<I", 0x100000)),
        (SIRM + 0x20, struct.pack("<I", 2)),
        (SIRM + 0x24, struct.pack("<I", 0x76030)),
        (SIRM + 0x28, struct.pack("<I", 0)),
        (SIRM + 0x18, struct.pack("<I", 0x400)),
        (SIRM + 0x2C, struct.pack("<I", 0x400)),
        (SIRM + 0x04, struct.pack("<I", 1)),
        (0x030804, struct.pack("<I", 1)),
    ]:
        e = write_mem(addr, data)
        if e:
            print("write 0x%06x failed: %s" % (addr, e))

    ids = []          # (host_us, frame_id, dev_ts_raw)
    total = 0
    c0 = cpu_times()
    t0 = time.perf_counter()
    buf = (ctypes.c_ubyte * (4 << 20))()
    while time.perf_counter() - t0 < SEC:
        n = ctypes.c_int(0)
        rc = lib.libusb_bulk_transfer(h, EP_STREAM, buf, 4 << 20, ctypes.byref(n), 800)
        if rc != 0:
            print("read err:", err(rc))
            break
        total += n.value
        data = bytes(bytearray(buf[:n.value]))
        t_us = time.perf_counter() * 1e6
        # 找 leader "U3VL"
        pos = 0
        while True:
            i = data.find(b"U3VL", pos)
            if i < 0 or i + 52 > len(data):
                break
            lid = struct.unpack_from("<I", data, i + 8)[0]
            # 时间戳候选：offset 20..28 (8字节)
            ts = struct.unpack_from("<q", data, i + 20)[0]
            ids.append((t_us, lid, ts))
            pos = i + 52
    wall = time.perf_counter() - t0
    c1 = cpu_times()
    print("RESULT bytes=%d wall=%.2fs -> %.2f MB/s | leaders=%d" % (total, wall, total / 1048576.0 / wall, len(ids)))
    if len(ids) > 2:
        print("  first id=%d last id=%d  (span %d frames) -> device fps %.2f"
              % (ids[0][1], ids[-1][1], ids[-1][1] - ids[0][1],
                 (ids[-1][1] - ids[0][1]) / ((ids[-1][0] - ids[0][0]) / 1e6)))
        dts = sorted([(b[0] - a[0]) / 1000.0 for a, b in zip(ids, ids[1:])])
        p = lambda q: dts[min(len(dts) - 1, int(q * (len(dts) - 1)))]
        print("  相邻 leader 主机时间间隔 ms: p50 %.2f p95 %.2f" % (p(.5), p(.95)))
        tss = [x[2] for x in ids]
        dts_dev = sorted([(b - a) / 1000.0 for a, b in zip(tss, tss[1:]) if b > a])
        if dts_dev:
            p2 = lambda q: dts_dev[min(len(dts_dev) - 1, int(q * (len(dts_dev) - 1)))]
            print("  设备时间戳差(原始单位/1000) p50 %.3f p95 %.3f" % (p2(.5), p2(.95)))
            print("  ts 样例:", [x[2] for x in ids[:5]])
    busy = (c1[0] - c0[0]) / max(1, c1[1] - c0[1]) * 100
    print("  system CPU busy %.1f%% (4核)" % busy)
finally:
    write_mem(0x030804, struct.pack("<I", 0))
    write_mem(SIRM + 0x04, struct.pack("<I", 0))
    for itf in (2, 1, 0):
        lib.libusb_release_interface(h, itf)
    lib.libusb_close(h)
