# -*- coding: utf-8 -*-
"""u3v_latency_opt.py — U3V 直连延迟优化实验

测三件事：
  A. 基线：软触发 + 等 ACK（和 u3v_trigger_latency.py 一样）
  B. 触发命令 **不等 ACK**（flags=0，省掉触发命令的 USB 往返）
  C. 不同曝光时间（写 0x030b04）对触发→出帧延迟的影响

用法： python3 u3v_latency_opt.py
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


def _out_raw(pkt, timeout=500):
    b = (ctypes.c_ubyte * len(pkt)).from_buffer_copy(pkt)
    n = ctypes.c_int(0)
    t0 = time.perf_counter()
    rc = lib.libusb_bulk_transfer(_h[0], EP_OUT, b, len(pkt), ctypes.byref(n), timeout)
    return rc, (time.perf_counter() - t0) * 1000.0


def _in(size=1024, timeout=500):
    b = (ctypes.c_ubyte * size)()
    n = ctypes.c_int(0)
    rc = lib.libusb_bulk_transfer(_h[0], EP_IN, b, size, ctypes.byref(n), timeout)
    return rc, bytes(bytearray(b[:n.value]))


def _nid():
    _pid[0] = (_pid[0] + 1) & 0xFFFF or 1
    return _pid[0]


def write_mem(addr, payload, ack=True, timeout=500):
    flags = 0x4000 if ack else 0x0000
    pkt = struct.pack("<I H H H H", 0x43563355, flags, 0x0802, 8 + len(payload), _nid()) + \
        struct.pack("<Q", addr) + payload
    rc, wms = _out_raw(pkt, timeout)
    if rc:
        return "OUT-" + err(rc), wms
    if not ack:
        return None, wms
    rc, d = _in(64, timeout)
    if rc or len(d) < 12:
        return "IN-" + err(rc), wms
    magic, status, c, sz, rid = struct.unpack("<I H H H H", d[:12])
    return (None if status == 0 else "status=0x%04x" % status), wms


def stream_read(timeout_ms=2000):
    b = (ctypes.c_ubyte * (4 << 20))()
    n = ctypes.c_int(0)
    rc = lib.libusb_bulk_transfer(_h[0], EP_STREAM, b, 4 << 20, ctypes.byref(n), timeout_ms)
    return rc, n.value


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round((len(xs) - 1) * q)))]


def trigger_once(wait_ack=True, timeout=3.0):
    # 清残留
    while True:
        rc, sz = stream_read(60)
        if rc != 0 or sz == 0:
            break
    time.sleep(0.05)
    t0 = time.perf_counter()
    e, wms = write_mem(0x030880, struct.pack("<I", 6), ack=wait_ack, timeout=1000)
    t_after_write = time.perf_counter()
    while time.perf_counter() - t0 < timeout:
        rc, sz = stream_read(2000)
        if rc != 0:
            return None, wms, e
        if sz == 2580528:
            return (time.perf_counter() - t0) * 1000.0, wms, e
    return None, wms, e


ctx = ctypes.c_void_p()
lib.libusb_init(ctypes.byref(ctx))
h = lib.libusb_open_device_with_vid_pid(ctx, VID, PID)
if not h:
    sys.exit("open failed")
_h[0] = h
for itf in (0, 1, 2):
    lib.libusb_claim_interface(h, itf)
try:
    print("== 触发模式配置 ==")
    for addr, data in [
        (0x0308d8, struct.pack("<I", 7)),
        (0x030840, struct.pack("<I", 0x40)),
        (0x030b1c, struct.pack("<I", 0)),
        (0x030b04, struct.pack("<I", 500)),
        (0x03081c, struct.pack("<I", 0x40007530)),
        (0x0500a8, struct.pack("<I", 1)),
    ]:
        e, w = write_mem(addr, data)
        print("  0x%06x <- %-10s : %s (%.2fms)" % (addr, data.hex(), e or "OK", w))
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
        e, w = write_mem(addr, data)
        if e:
            print("  WARN 0x%06x: %s" % (addr, e))

    # A. 基线（等 ACK）
    lat, wr = [], []
    for i in range(12):
        l, wms, e = trigger_once(wait_ack=True)
        if l:
            lat.append(l)
            wr.append(wms)
        time.sleep(0.12)
    print("\n[A] 基线（触发命令等 ACK）: n=%d p50 %.2f p95 %.2f ms | 触发写往返 p50 %.2f ms"
          % (len(lat), pct(lat, .5), pct(lat, .95), pct(wr, .5)))

    # B. 不等 ACK
    lat2, wr2 = [], []
    for i in range(12):
        l, wms, e = trigger_once(wait_ack=False)
        if l:
            lat2.append(l)
            wr2.append(wms)
        time.sleep(0.12)
    print("[B] 触发不等 ACK: n=%d p50 %.2f p95 %.2f ms | 触发写耗时 p50 %.2f ms%s"
          % (len(lat2), pct(lat2, .5), pct(lat2, .95), pct(wr2, .5),
             "" if not e else " (write rc=%s)" % e))

    # C. 不同曝光（写 0x030b04 后再触发）
    for exp_us in (1000, 500, 300, 200):
        e, w = write_mem(0x030b04, struct.pack("<I", exp_us))
        time.sleep(0.2)
        ls = []
        for i in range(10):
            l, wms, er = trigger_once(wait_ack=False)
            if l:
                ls.append(l)
            time.sleep(0.12)
        if ls:
            print("[C] 曝光 %4d us: 延迟 p50 %.2f p95 %.2f ms (n=%d)  %s"
                  % (exp_us, pct(ls, .5), pct(ls, .95), len(ls), "" if e is None else e))
    # 恢复 500us
    write_mem(0x030b04, struct.pack("<I", 500))
finally:
    print("\n== stop ==")
    write_mem(0x030804, struct.pack("<I", 0))
    write_mem(SIRM + 0x04, struct.pack("<I", 0))
    for itf in (2, 1, 0):
        lib.libusb_release_interface(h, itf)
    lib.libusb_close(h)
