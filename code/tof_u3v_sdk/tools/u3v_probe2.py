# -*- coding: utf-8 -*-
"""u3v_probe2.py — 摸清这台相机真实的 ABRM/SBRM/SIRM 布局，并验证 claim data 接口是否影响控制通道

用法： python3 u3v_probe2.py
"""
import ctypes
import struct
import sys
import time

VID, PID = 0x2BDF, 0x0001
EP_OUT, EP_IN = 0x01, 0x82

lib = ctypes.CDLL("libusb-1.0.so.0")
lib.libusb_init.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
lib.libusb_open_device_with_vid_pid.argtypes = [ctypes.c_void_p, ctypes.c_uint16, ctypes.c_uint16]
lib.libusb_open_device_with_vid_pid.restype = ctypes.c_void_p
lib.libusb_claim_interface.argtypes = [ctypes.c_void_p, ctypes.c_int]
lib.libusb_release_interface.argtypes = [ctypes.c_void_p, ctypes.c_int]
lib.libusb_bulk_transfer.argtypes = [ctypes.c_void_p, ctypes.c_ubyte,
                                     ctypes.POINTER(ctypes.c_ubyte), ctypes.c_int,
                                     ctypes.POINTER(ctypes.c_int), ctypes.c_uint]
lib.libusb_bulk_transfer.restype = ctypes.c_int
lib.libusb_close.argtypes = [ctypes.c_void_p]
lib.libusb_error_name.argtypes = [ctypes.c_int]
lib.libusb_error_name.restype = ctypes.c_char_p

_pid = [0]


def errname(rc):
    return lib.libusb_error_name(rc).decode()


def xfer_out(h, data, timeout=500):
    buf = (ctypes.c_ubyte * len(data)).from_buffer_copy(data)
    n = ctypes.c_int(0)
    rc = lib.libusb_bulk_transfer(h, EP_OUT, buf, len(data), ctypes.byref(n), timeout)
    return rc


def xfer_in(h, size=1024, timeout=500):
    buf = (ctypes.c_ubyte * size)()
    n = ctypes.c_int(0)
    rc = lib.libusb_bulk_transfer(h, EP_IN, buf, size, ctypes.byref(n), timeout)
    if rc != 0:
        return rc, b""
    return 0, bytes(bytearray(buf[:n.value]))


def read_mem(h, address, size, timeout=500):
    _pid[0] = (_pid[0] + 1) & 0xFFFF or 1
    pkt = struct.pack("<I H H H H", 0x43563355, 0x4000, 0x0800, 12, _pid[0]) + \
        struct.pack("<Q H H", address, 0, size)
    rc = xfer_out(h, pkt, timeout)
    if rc != 0:
        return None, "OUT-" + errname(rc)
    rc, data = xfer_in(h, max(size + 12, 512), timeout)
    if rc != 0:
        return None, "IN-" + errname(rc)
    if len(data) < 12:
        return None, "short %d" % len(data)
    magic, status, cmd, sz, rid = struct.unpack("<I H H H H", data[:12])
    if status != 0:
        return None, "status=0x%04x" % status
    return data[12:12 + sz], None


def write_mem(h, address, payload, timeout=500):
    _pid[0] = (_pid[0] + 1) & 0xFFFF or 1
    size = len(payload)
    pkt = struct.pack("<I H H H H", 0x43563355, 0x4000, 0x0802, 8 + size, _pid[0]) + \
        struct.pack("<Q", address) + payload
    rc = xfer_out(h, pkt, timeout)
    if rc != 0:
        return "OUT-" + errname(rc)
    rc, data = xfer_in(h, 64, timeout)
    if rc != 0:
        return "IN-" + errname(rc)
    magic, status, cmd, sz, rid = struct.unpack("<I H H H H", data[:12])
    return "status=0x%04x cmd=0x%04x" % (status, cmd)


def show(tag, address, size):
    d, err = read_mem(h, address, size)
    if err:
        print("  %-28s [0x%04x+%2d] ERR %s" % (tag, address, size, err))
        return None
    asc = "".join(chr(b) if 32 <= b < 127 else "." for b in d)
    print("  %-28s [0x%04x+%2d] %s | %s" % (tag, address, size, d.hex(), asc))
    return d


ctx = ctypes.c_void_p()
lib.libusb_init(ctypes.byref(ctx))
h = lib.libusb_open_device_with_vid_pid(ctx, VID, PID)
if not h:
    sys.exit("open failed")
print("claim 0:", errname(lib.libusb_claim_interface(h, 0)) or "OK")
try:
    print("\n== claim 0 only: read 0x1cc ==")
    show("max_response_time", 0x1CC, 4)

    print("\n== claim data interface 2 as well ==")
    rc = lib.libusb_claim_interface(h, 2)
    print("claim 2:", errname(rc) or "OK")
    show("max_response_time(again)", 0x1CC, 4)
    lib.libusb_release_interface(h, 2)
    show("max_response_time(released 2)", 0x1CC, 4)

    print("\n== ABRM layout scan ==")
    for tag, addr, size in [
        ("gencp_version", 0x0000, 4), ("manufacturer? 0x04", 0x0004, 64),
        ("model? 0x44", 0x0044, 64), ("family? 0x84", 0x0084, 64),
        ("device_version? 0xc4", 0x00C4, 64), ("manufacturer_info? 0x104", 0x0104, 64),
        ("serial? 0x144", 0x0144, 64), ("user_name? 0x184", 0x0184, 64),
        ("device_capability", 0x01C4, 8), ("max_dev_response_time", 0x01CC, 4),
        ("manifest_table_addr", 0x01D0, 8), ("sbrm_address", 0x01D8, 8),
        ("device_configuration", 0x01E0, 8), ("heartbeat_timeout", 0x01E8, 4),
        ("message_channel_id", 0x01EC, 4), ("timestamp", 0x01F0, 8),
        ("timestamp_latch", 0x01F8, 8), ("timestamp_increment", 0x01FC, 4),
        ("access_privilege", 0x0204, 4), ("protocol_endianness", 0x0208, 4),
        ("implementation_endianness", 0x020C, 4),
    ]:
        show(tag, addr, size)
finally:
    lib.libusb_release_interface(h, 0)
    lib.libusb_close(h)
