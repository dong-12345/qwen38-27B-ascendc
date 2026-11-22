# -*- coding: utf-8 -*-
"""u3v_probe3.py — 读 SBRM/SIRM 寄存器（U3V 流参数），为自写流式取图做准备"""
import ctypes
import struct
import sys

VID, PID = 0x2BDF, 0x0001
EP_OUT, EP_IN = 0x01, 0x82
SBRM = 0x52000   # 从 ABRM 0x01d8 读出

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


def cmd_out(h, pkt, timeout=500):
    b = (ctypes.c_ubyte * len(pkt)).from_buffer_copy(pkt)
    n = ctypes.c_int(0)
    return lib.libusb_bulk_transfer(h, EP_OUT, b, len(pkt), ctypes.byref(n), timeout)


def cmd_in(h, size=2048, timeout=500):
    b = (ctypes.c_ubyte * size)()
    n = ctypes.c_int(0)
    rc = lib.libusb_bulk_transfer(h, EP_IN, b, size, ctypes.byref(n), timeout)
    return (0, bytes(bytearray(b[:n.value]))) if rc == 0 else (rc, b"")


def _next_id():
    _pid[0] = (_pid[0] + 1) & 0xFFFF or 1
    return _pid[0]


def read_mem(h, addr, size):
    pkt = struct.pack("<I H H H H", 0x43563355, 0x4000, 0x0800, 12, _next_id()) + \
        struct.pack("<Q H H", addr, 0, size)
    rc = cmd_out(h, pkt)
    if rc:
        return None, "OUT-" + err(rc)
    rc, d = cmd_in(h)
    if rc:
        return None, "IN-" + err(rc)
    magic, status, cmd, sz, rid = struct.unpack("<I H H H H", d[:12])
    if status:
        return None, "status=0x%04x" % status
    return d[12:12 + sz], None


def hexdump(d):
    return " ".join("%02x" % b for b in d)


def show(h, tag, addr, size):
    d, e = read_mem(h, addr, size)
    if e:
        print("  %-26s [0x%05x+%d] ERR %s" % (tag, addr, size, e))
        return None
    print("  %-26s [0x%05x+%d] %s" % (tag, addr, size, hexdump(d)))
    return d


ctx = ctypes.c_void_p()
lib.libusb_init(ctypes.byref(ctx))
h = lib.libusb_open_device_with_vid_pid(ctx, VID, PID)
if not h:
    sys.exit("open failed")
lib.libusb_claim_interface(h, 0)
print("claim 0 OK")
sirm = None
try:
    print("\n== SBRM @0x%x ==" % SBRM)
    show(h, "u3v_version", SBRM + 0x00, 4)
    show(h, "u3vcp_capability", SBRM + 0x04, 8)
    show(h, "u3vcp_configuration", SBRM + 0x0c, 8)
    show(h, "max_cmd_transfer", SBRM + 0x14, 4)
    show(h, "max_ack_transfer", SBRM + 0x18, 4)
    show(h, "num_stream_channels", SBRM + 0x1c, 4)
    d = show(h, "sirm_address", SBRM + 0x20, 8)
    show(h, "sirm_length", SBRM + 0x28, 8)
    show(h, "eirm_address", SBRM + 0x2c, 8)
    show(h, "iidc2_address", SBRM + 0x38, 8)
    show(h, "current_speed", SBRM + 0x40, 4)
    if d:
        sirm = struct.unpack("<Q", d)[0]
        print("\n== SIRM @0x%x ==" % sirm)
        show(h, "info", sirm + 0x00, 4)
        show(h, "control", sirm + 0x04, 4)
        show(h, "req_payload_size", sirm + 0x08, 8)
        show(h, "req_leader_size", sirm + 0x10, 4)
        show(h, "req_trailer_size", sirm + 0x14, 4)
        show(h, "max_leader_size", sirm + 0x18, 4)
        show(h, "payload_size", sirm + 0x1c, 4)
        show(h, "payload_count", sirm + 0x20, 4)
        show(h, "transfer1_size", sirm + 0x24, 4)
        show(h, "transfer2_size", sirm + 0x28, 4)
        show(h, "max_trailer_size", sirm + 0x2c, 4)
finally:
    lib.libusb_release_interface(h, 0)
    lib.libusb_close(h)
