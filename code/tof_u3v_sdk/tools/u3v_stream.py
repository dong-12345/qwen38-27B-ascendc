# -*- coding: utf-8 -*-
"""u3v_stream.py — 用 libusb 直接按 U3V 协议开流（绕过海康 Mv3dRgbd SDK 的取流路径）

已验证（2026-11-20 12:38）：
  - 直接 libusb 开流可跑满 **29.75 fps / 73 MB/s**（2.58MB/帧），而海康 SDK 只有 7~14fps
  - 寄存器/命令序列来自 usbmon 抓包，与厂商 SDK 完全一致：
      SIRM.payload_size=1MB, payload_count=2, transfer1_size=0x76030, transfer2=0,
      max_leader=0x400, max_trailer=0x400, SIRM.control=1, 然后 0x030804=1 (AcquisitionStart)
  - 每帧的 USB 读返回三段：leader(52B) + payload(2580528B) + trailer(36B)

用法： python3 u3v_stream.py [seconds]
"""
import ctypes
import struct
import sys
import time

VID, PID = 0x2BDF, 0x0001
EP_OUT, EP_IN = 0x01, 0x82
EP_STREAM = 0x84
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


def _out(pkt, timeout=500):
    b = (ctypes.c_ubyte * len(pkt)).from_buffer_copy(pkt)
    n = ctypes.c_int(0)
    return lib.libusb_bulk_transfer(h, EP_OUT, b, len(pkt), ctypes.byref(n), timeout), n.value


def _in(size=1024, timeout=500):
    b = (ctypes.c_ubyte * size)()
    n = ctypes.c_int(0)
    rc = lib.libusb_bulk_transfer(h, EP_IN, b, size, ctypes.byref(n), timeout)
    return rc, bytes(bytearray(b[:n.value]))


def _nid():
    _pid[0] = (_pid[0] + 1) & 0xFFFF or 1
    return _pid[0]


def read_mem(addr, size):
    pkt = struct.pack("<I H H H H", 0x43563355, 0x4000, 0x0800, 12, _nid()) + \
        struct.pack("<Q H H", addr, 0, size)
    rc, _ = _out(pkt)
    if rc:
        return None
    rc, d = _in(size + 64)
    if rc or len(d) < 12:
        return None
    magic, status, c, sz, rid = struct.unpack("<I H H H H", d[:12])
    return d[12:12 + sz] if status == 0 else None


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


def stream_read(bufsize, timeout_ms):
    b = (ctypes.c_ubyte * bufsize)()
    n = ctypes.c_int(0)
    t0 = time.perf_counter()
    rc = lib.libusb_bulk_transfer(h, EP_STREAM, b, bufsize, ctypes.byref(n), timeout_ms)
    dt = time.perf_counter() - t0
    return rc, bytes(bytearray(b[:n.value])), dt, t0 * 1e6


ctx = ctypes.c_void_p()
lib.libusb_init(ctypes.byref(ctx))
h = lib.libusb_open_device_with_vid_pid(ctx, VID, PID)
if not h:
    sys.exit("open failed")
for itf in (0, 1, 2):
    rc = lib.libusb_claim_interface(h, itf)
    print("claim iface %d: %s" % (itf, err(rc) if rc else "OK"))

secs = float(sys.argv[1]) if len(sys.argv) > 1 else 5.0
frames = []
leaders = []
trailers = []
try:
    print("\n== configure SIRM + AcquisitionStart ==")
    steps = [
        (SIRM + 0x1C, struct.pack("<I", 0x100000)),
        (SIRM + 0x20, struct.pack("<I", 2)),
        (SIRM + 0x24, struct.pack("<I", 0x76030)),
        (SIRM + 0x28, struct.pack("<I", 0)),
        (SIRM + 0x18, struct.pack("<I", 0x400)),
        (SIRM + 0x2C, struct.pack("<I", 0x400)),
        (SIRM + 0x04, struct.pack("<I", 1)),
        (0x030804, struct.pack("<I", 1)),
    ]
    for addr, data in steps:
        e = write_mem(addr, data)
        print("  write 0x%06x <- %s : %s" % (addr, data.hex(), e or "OK"))

    print("\n== streaming %.1fs ==" % secs)
    t_start = time.perf_counter()
    total = 0
    pend_leader = None
    while time.perf_counter() - t_start < secs:
        rc, data, dt, t_us = stream_read(4 << 20, 1000)
        if rc != 0:
            print("  stream read error: %s" % err(rc))
            break
        if not data:
            continue
        total += len(data)
        if len(leaders) < 3 and len(data) < 256:
            leaders.append((len(data), data, dt))
        if len(data) < 256:
            pend_leader = data
            continue
        if pend_leader is not None:
            frames.append((t_us, len(data), pend_leader))
            if len(trailers) < 3:
                trailers.append(data[-36:])
            pend_leader = None
    wall = time.perf_counter() - t_start

    print("\nRESULT bytes=%d wall=%.2fs -> %.2f MB/s, frames=%d -> %.2f fps"
          % (total, wall, total / 1048576.0 / wall, len(frames), len(frames) / wall))
    if len(frames) > 2:
        ts = [f[0] for f in frames]
        dts = sorted([(b - a) / 1000.0 for a, b in zip(ts, ts[1:])])
        p = lambda q: dts[min(len(dts) - 1, int(q * (len(dts) - 1)))]
        print("帧间隔 ms: p50 %.2f p95 %.2f max %.2f (目标 33.3ms=30fps)"
              % (p(.5), p(.95), dts[-1]))
        print("帧 payload 大小: %d bytes (%.2f MB)" % (frames[0][1], frames[0][1] / 1048576.0))
    print("\nleader 样例:")
    for ln, d, dt in leaders:
        print("  len=%d dt=%.2fms  %s" % (ln, dt * 1000, d.hex()))
    print("trailer 样例:")
    for d in trailers:
        print("  %s" % d.hex())
finally:
    print("\n== stop ==")
    e1 = write_mem(0x030804, struct.pack("<I", 0))
    e2 = write_mem(SIRM + 0x04, struct.pack("<I", 0))
    print("  0x030804<-0:", e1 or "OK", "| SIRM.control<-0:", e2 or "OK")
    for itf in (2, 1, 0):
        lib.libusb_release_interface(h, itf)
    lib.libusb_close(h)
