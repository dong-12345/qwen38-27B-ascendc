# -*- coding: utf-8 -*-
"""u3v_incremental_latency.py — 分级延迟测量：leader / 深度部分 / 完整 RGBD 各自何时可用

思路：触发后按需分块读：
  1) 读 leader(52B)             -> t_leader（≈相机固件流水线固定延迟）
  2) 读 depth 部分 614,448B     -> t_depth （深度图可用，SLAM/建图只需要它）
  3) 读剩余 JPEG 1,966,080B     -> t_full  （完整 RGBD 帧可用）
这样能区分"相机固件延迟"和"USB 传输时间"，并给出按需消费的最优延迟。

用法： python3 u3v_incremental_latency.py [n]
"""
import ctypes
import struct
import sys
import time

VID, PID = 0x2BDF, 0x0001
EP_OUT, EP_IN, EP_STREAM = 0x01, 0x82, 0x84
SIRM = 0x62000
DEPTH_BYTES = 614_448
JPEG_BYTES = 1_966_080

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
_h = [None]


def err(rc):
    return lib.libusb_error_name(rc).decode()


def _out(pkt, timeout=500):
    b = (ctypes.c_ubyte * len(pkt)).from_buffer_copy(pkt)
    n = ctypes.c_int(0)
    return lib.libusb_bulk_transfer(_h[0], EP_OUT, b, len(pkt), ctypes.byref(n), timeout), n.value


def _in(size=1024, timeout=500):
    b = (ctypes.c_ubyte * size)()
    n = ctypes.c_int(0)
    rc = lib.libusb_bulk_transfer(_h[0], EP_IN, b, size, ctypes.byref(n), timeout)
    return rc, bytes(bytearray(b[:n.value]))


def _nid():
    _pid[0] = (_pid[0] + 1) & 0xFFFF or 1
    return _pid[0]


def write_mem(addr, payload, timeout=500):
    pkt = struct.pack("<I H H H H", 0x43563355, 0x4000, 0x0802, 8 + len(payload), _nid()) + \
        struct.pack("<Q", addr) + payload
    rc, _ = _out(pkt, timeout)
    if rc:
        return "OUT-" + err(rc)
    rc, d = _in(64, timeout)
    if rc or len(d) < 12:
        return "IN-" + err(rc)
    magic, status, c, sz, rid = struct.unpack("<I H H H H", d[:12])
    return None if status == 0 else "status=0x%04x" % status


def bulk_read(size, timeout_ms=2000):
    b = (ctypes.c_ubyte * size)()
    n = ctypes.c_int(0)
    rc = lib.libusb_bulk_transfer(_h[0], EP_STREAM, b, size, ctypes.byref(n), timeout_ms)
    return rc, n.value


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round((len(xs) - 1) * q)))]


ctx = ctypes.c_void_p()
lib.libusb_init(ctypes.byref(ctx))
h = lib.libusb_open_device_with_vid_pid(ctx, VID, PID)
if not h:
    sys.exit("open failed")
_h[0] = h
for itf in (0, 1, 2):
    lib.libusb_claim_interface(h, itf)

N = int(sys.argv[1]) if len(sys.argv) > 1 else 15
try:
    for addr, data in [
        (0x0308d8, struct.pack("<I", 7)), (0x030840, struct.pack("<I", 0x40)),
        (0x030b1c, struct.pack("<I", 0)), (0x030b04, struct.pack("<I", 500)),
        (0x03081c, struct.pack("<I", 0x40007530)), (0x0500a8, struct.pack("<I", 1)),
        (SIRM + 0x1C, struct.pack("<I", 0x100000)), (SIRM + 0x20, struct.pack("<I", 2)),
        (SIRM + 0x24, struct.pack("<I", 0x76030)), (SIRM + 0x28, struct.pack("<I", 0)),
        (SIRM + 0x18, struct.pack("<I", 0x400)), (SIRM + 0x2C, struct.pack("<I", 0x400)),
        (SIRM + 0x04, struct.pack("<I", 1)), (0x030804, struct.pack("<I", 1)),
    ]:
        e = write_mem(addr, data)
        if e:
            print("WARN 0x%06x: %s" % (addr, e))

    tl, td, tf = [], [], []
    for i in range(N):
        # 清残留
        while True:
            rc, sz = bulk_read(4 << 20, 60)
            if rc != 0 or sz == 0:
                break
        time.sleep(0.05)
        t0 = time.perf_counter()
        e = write_mem(0x030880, struct.pack("<I", 6), timeout=1000)
        # 1) leader
        rc, sz = bulk_read(512, 3000)
        if rc != 0:
            print("  trig %d leader err %s" % (i, err(rc)))
            continue
        t_leader = (time.perf_counter() - t0) * 1000.0
        # 2) depth 部分
        got = 0
        while got < DEPTH_BYTES:
            rc, sz = bulk_read(DEPTH_BYTES - got, 3000)
            if rc != 0:
                break
            got += sz
        t_depth = (time.perf_counter() - t0) * 1000.0
        # 3) 剩余 JPEG
        got = 0
        while got < JPEG_BYTES:
            rc, sz = bulk_read(JPEG_BYTES - got, 3000)
            if rc != 0:
                break
            got += sz
        t_full = (time.perf_counter() - t0) * 1000.0
        tl.append(t_leader)
        td.append(t_depth)
        tf.append(t_full)
        print("  trig %2d: leader %.2f ms | depth可用 %.2f ms | RGBD完整 %.2f ms"
              % (i, t_leader, t_depth, t_full))
        time.sleep(0.15)

    print("\nRESULT n=%d" % len(tf))
    for name, xs in (("leader(首数据)", tl), ("depth 可用", td), ("完整 RGBD", tf)):
        print("  %-12s p50 %.2f  p95 %.2f  min %.2f  max %.2f ms"
              % (name, pct(xs, .5), pct(xs, .95), min(xs), max(xs)))
finally:
    print("\n== stop ==")
    write_mem(0x030804, struct.pack("<I", 0))
    write_mem(SIRM + 0x04, struct.pack("<I", 0))
    for itf in (2, 1, 0):
        lib.libusb_release_interface(h, itf)
    lib.libusb_close(h)
