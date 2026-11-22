# -*- coding: utf-8 -*-
"""u3v_dump.py — 直连 libusb 开流后，打印前 N 次读的 (长度, 首字节) 和 leader/trailer 结构

用法： python3 u3v_dump.py [reads] [seconds]
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


ctx = ctypes.c_void_p()
lib.libusb_init(ctypes.byref(ctx))
h = lib.libusb_open_device_with_vid_pid(ctx, VID, PID)
if not h:
    sys.exit("open failed")
for itf in (0, 1, 2):
    lib.libusb_claim_interface(h, itf)

NR = int(sys.argv[1]) if len(sys.argv) > 1 else 12
SEC = float(sys.argv[2]) if len(sys.argv) > 2 else 2.0

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
    print("streaming...")
    t0 = time.perf_counter()
    n = 0
    while time.perf_counter() - t0 < SEC and n < NR:
        b = (ctypes.c_ubyte * (3 << 20))()
        cnt = ctypes.c_int(0)
        rc = lib.libusb_bulk_transfer(h, EP_STREAM, b, 3 << 20, ctypes.byref(cnt), 800)
        if rc != 0:
            print("read err:", err(rc))
            break
        n += 1
        d = bytes(bytearray(b[:cnt.value]))
        pre = d[:64].hex()
        suf = d[-40:].hex() if len(d) > 40 else ""
        print("%2d len=%-8d wall=%7.2fms head=%s" % (n, len(d), (time.perf_counter() - t0) * 1000, pre))
        if len(d) < 128:
            print("            tail=%s" % suf)
    write_mem(0x030804, struct.pack("<I", 0))
    write_mem(SIRM + 0x04, struct.pack("<I", 0))
finally:
    for itf in (2, 1, 0):
        lib.libusb_release_interface(h, itf)
    lib.libusb_close(h)
