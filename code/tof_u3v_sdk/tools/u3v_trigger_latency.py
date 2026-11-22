# -*- coding: utf-8 -*-
"""u3v_trigger_latency.py — U3V 直连 + 软触发，测端到端绝对延迟（纯主机时钟）

做法：
  1) 回放厂商 SDK 抓包得到的"触发模式配置"寄存器写；
  2) 开流；
  3) 每次：t0=perf_counter → 写触发命令 0x030880=6 → 读 leader+payload+trailer → t1；
     latency = t1 - t0（主机时钟，不需要和相机对时，与 PC 版测法一致）。

用法： python3 u3v_trigger_latency.py [n]
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


def stream_read(bufsize=4 << 20, timeout_ms=2000):
    b = (ctypes.c_ubyte * bufsize)()
    n = ctypes.c_int(0)
    t0 = time.perf_counter()
    rc = lib.libusb_bulk_transfer(_h[0], EP_STREAM, b, bufsize, ctypes.byref(n), timeout_ms)
    return rc, n.value, (time.perf_counter() - t0) * 1000.0, t0


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
    rc = lib.libusb_claim_interface(h, itf)
    print("claim iface %d: %s" % (itf, err(rc) if rc else "OK"))

N = int(sys.argv[1]) if len(sys.argv) > 1 else 20
try:
    print("== 回放触发模式配置（从厂商 SDK 抓包得到）=")
    cfg = [
        (0x0308d8, struct.pack("<I", 7)),        # TriggerSource=Software
        (0x030840, struct.pack("<I", 0x40)),     # Trigger 相关
        (0x030b1c, struct.pack("<I", 0)),
        (0x030b04, struct.pack("<I", 500)),      # 曝光 500us
        (0x03081c, struct.pack("<I", 0x40007530)),
        (0x0500a8, struct.pack("<I", 1)),
    ]
    for addr, data in cfg:
        print("  write 0x%06x <- %s : %s" % (addr, data.hex(), write_mem(addr, data) or "OK"))

    print("\n== 开流 ==")
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
            print("  WARN write 0x%06x: %s" % (addr, e))

    lat = []
    for i in range(N):
        # 清掉可能残留的数据
        while True:
            rc, sz, dt, _ = stream_read(4 << 20, 60)
            if rc != 0 or sz == 0:
                break
        time.sleep(0.05)
        t0 = time.perf_counter()
        e = write_mem(0x030880, struct.pack("<I", 6), timeout=1000)
        t_wr = time.perf_counter()
        got_payload = False
        t_end = None
        while time.perf_counter() - t0 < 3.0:
            rc, sz, dt, tt = stream_read(4 << 20, 2000)
            if rc != 0:
                print("  trig %d read err %s" % (i, err(rc)))
                break
            if sz == 2580528:
                got_payload = True
                t_end = time.perf_counter()
                break
        if got_payload:
            lat.append((t_end - t0) * 1000.0)
            print("  trig %2d: latency %.2f ms (write %.2f ms)" % (i, (t_end - t0) * 1000.0,
                                                                  (t_wr - t0) * 1000.0))
        else:
            print("  trig %2d: no frame" % i)
        time.sleep(0.15)

    print("\nRESULT n=%d" % len(lat))
    if lat:
        print("  绝对延迟 ms: min %.2f  p50 %.2f  p95 %.2f  max %.2f"
              % (min(lat), pct(lat, .5), pct(lat, .95), max(lat)))
finally:
    print("\n== stop ==")
    write_mem(0x030804, struct.pack("<I", 0))
    write_mem(SIRM + 0x04, struct.pack("<I", 0))
    for itf in (2, 1, 0):
        lib.libusb_release_interface(h, itf)
    lib.libusb_close(h)
