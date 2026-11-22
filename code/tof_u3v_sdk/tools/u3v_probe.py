# -*- coding: utf-8 -*-
"""u3v_probe.py — 用 libusb(ctypes) 直接对相机发 U3V 控制命令，验证 Packet ID 假设

背景：
  vendor SDK 的 U3V 包格式 = 标准 USB3 Vision（已从抓包解码确认）：
    magic 'U3VC' + flags(u16 LE=0x4000) + command(u16=0x0800 READ_MEMORY_CMD) + size(u16=12)
    + id(u16) + address(u64) + unknown(u16) + read_size(u16)
  vendor 第一条命令: id=0, address=0x1cc(ABRM_MAX_DEVICE_RESPONSE_TIME), size=4 -> 正常 ACK
  Aravis 第一条命令: id=0xff15, address=0x0004(MANUFACTURER_NAME), size=64 -> 相机完全不应答(EIO)
假设：这台相机只接受 Packet ID < 0x8000（或要求从 0 开始），Aravis/Cameleon 起始 id(65300) 被忽略。

用法： python3 u3v_probe.py [--root]
"""
import ctypes
import struct
import sys
import time

VID, PID = 0x2BDF, 0x0001
EP_CTRL_OUT = 0x01
EP_CTRL_IN = 0x82

lib = ctypes.CDLL("libusb-1.0.so.0")
lib.libusb_init.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
lib.libusb_init.restype = ctypes.c_int
lib.libusb_open_device_with_vid_pid.argtypes = [ctypes.c_void_p, ctypes.c_uint16, ctypes.c_uint16]
lib.libusb_open_device_with_vid_pid.restype = ctypes.c_void_p
lib.libusb_claim_interface.argtypes = [ctypes.c_void_p, ctypes.c_int]
lib.libusb_claim_interface.restype = ctypes.c_int
lib.libusb_release_interface.argtypes = [ctypes.c_void_p, ctypes.c_int]
lib.libusb_bulk_transfer.argtypes = [ctypes.c_void_p, ctypes.c_ubyte,
                                     ctypes.POINTER(ctypes.c_ubyte), ctypes.c_int,
                                     ctypes.POINTER(ctypes.c_int), ctypes.c_uint]
lib.libusb_bulk_transfer.restype = ctypes.c_int
lib.libusb_close.argtypes = [ctypes.c_void_p]
lib.libusb_error_name.argtypes = [ctypes.c_int]
lib.libusb_error_name.restype = ctypes.c_char_p


def make_read_mem_cmd(address, size, pkt_id):
    # 标准 U3V ReadMemory 命令（24 字节，全部小端）
    hdr = struct.pack("<I H H H H", 0x43563355, 0x4000, 0x0800, 12, pkt_id & 0xFFFF)
    body = struct.pack("<Q H H", address, 0, size)
    return hdr + body


def errname(rc):
    return lib.libusb_error_name(rc).decode()


def send_read(handle, address, size, pkt_id, timeout_ms=500):
    pkt = make_read_mem_cmd(address, size, pkt_id)
    out = (ctypes.c_ubyte * len(pkt)).from_buffer_copy(pkt)
    xfer = ctypes.c_int(0)
    rc = lib.libusb_bulk_transfer(handle, EP_CTRL_OUT, out, len(pkt), ctypes.byref(xfer), timeout_ms)
    if rc != 0:
        return "OUT-ERR(%s)" % errname(rc)
    buf = (ctypes.c_ubyte * 2048)()
    t0 = time.perf_counter()
    rc = lib.libusb_bulk_transfer(handle, EP_CTRL_IN, buf, 2048, ctypes.byref(xfer), timeout_ms)
    dt = (time.perf_counter() - t0) * 1000
    if rc != 0:
        return "IN-ERR(%s, %.1fms)" % (errname(rc), dt)
    data = bytes(bytearray(buf[:xfer.value]))
    if len(data) < 12:
        return "IN-SHORT(len=%d data=%s)" % (len(data), data.hex())
    magic, status, cmd, sz, rid = struct.unpack("<I H H H H", data[:12])
    payload = data[12:]
    return "ACK status=0x%04x cmd=0x%04x size=%d id=%d payload=%s (%.1fms)" % (
        status, cmd, sz, rid, payload[:24].hex(), dt)


def main():
    ctx = ctypes.c_void_p()
    rc = lib.libusb_init(ctypes.byref(ctx))
    if rc != 0:
        sys.exit("libusb_init failed: %s" % errname(rc))
    handle = lib.libusb_open_device_with_vid_pid(ctx, VID, PID)
    if not handle:
        sys.exit("open device 2bdf:0001 failed (被占用?)")
    print("device opened")
    rc = lib.libusb_claim_interface(handle, 0)
    print("claim interface 0:", rc, errname(rc) if rc else "OK")
    try:
        tests = [
            ("id=0x0000 addr=0x1cc size=4  (vendor 第一条)", 0x1CC, 4, 0x0000),
            ("id=0x0001 addr=0x0044 size=64 (MODEL_NAME)", 0x0044, 64, 0x0001),
            ("id=0x0002 addr=0x0004 size=64 (MANUFACTURER)", 0x0004, 64, 0x0002),
            ("id=0xff15 addr=0x0004 size=64 (Aravis 起始 id)", 0x0004, 64, 0xFF15),
            ("id=0x7fff addr=0x0004 size=64 (0x7fff)", 0x0004, 64, 0x7FFF),
            ("id=0x8000 addr=0x0004 size=64 (0x8000)", 0x0004, 64, 0x8000),
            ("id=0xfffe addr=0x0004 size=64 (0xfffe)", 0x0004, 64, 0xFFFE),
            ("id=0x0003 addr=0x01c4 size=8  (DEV_CAPABILITY)", 0x01C4, 8, 0x0003),
            ("id=0x0004 addr=0x01d8 size=8  (SBRM_ADDRESS)", 0x01D8, 8, 0x0004),
        ]
        for name, addr, size, pid in tests:
            print("%-52s -> %s" % (name, send_read(handle, addr, size, pid)))
            time.sleep(0.05)
    finally:
        lib.libusb_release_interface(handle, 0)
        lib.libusb_close(handle)


if __name__ == "__main__":
    main()
