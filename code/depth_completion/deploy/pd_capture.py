"""pd_capture.py - direct libusb U3V capture for Hikrobot MV-DT01SDU (no vendor SDK).

Frame = 52-byte leader packet, payload chunks, 36-byte trailer packet.
Payload layout is auto-detected (see decode_payload), covering the two layouts
observed on this camera:
  A) 48B header + 640x480 uint16 depth + 1536x1280 BGR888        (6,512,688 B)
  B) 48B header + 640x480 uint16 depth + 1536x1280 ??            (6,205,488 B)
"""
import ctypes
import struct
import time

import numpy as np
import cv2

VID, PID = 0x2BDF, 0x0001
EP_OUT, EP_IN, EP_STREAM = 0x01, 0x82, 0x84
SIRM = 0x62000
DEPTH_SHAPE = (480, 640)
RGB_SHAPE = (1280, 1536)


class U3V:
    def __init__(self):
        self.lib = ctypes.CDLL("libusb-1.0.so.0")
        self.lib.libusb_init.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        self.lib.libusb_open_device_with_vid_pid.argtypes = [ctypes.c_void_p, ctypes.c_uint16, ctypes.c_uint16]
        self.lib.libusb_open_device_with_vid_pid.restype = ctypes.c_void_p
        self.lib.libusb_claim_interface.argtypes = [ctypes.c_void_p, ctypes.c_int]
        self.lib.libusb_bulk_transfer.argtypes = [ctypes.c_void_p, ctypes.c_ubyte, ctypes.POINTER(ctypes.c_ubyte),
                                                  ctypes.c_int, ctypes.POINTER(ctypes.c_int), ctypes.c_uint]
        self.lib.libusb_close.argtypes = [ctypes.c_void_p]
        self.lib.libusb_error_name.argtypes = [ctypes.c_int]
        self.lib.libusb_error_name.restype = ctypes.c_char_p
        self.pid = 0
        self.h = None
        self.buf = (ctypes.c_ubyte * (8 << 20))()

    def err(self, rc):
        return self.lib.libusb_error_name(rc).decode()

    def _out(self, pkt, timeout=500):
        b = (ctypes.c_ubyte * len(pkt)).from_buffer_copy(pkt)
        n = ctypes.c_int(0)
        return self.lib.libusb_bulk_transfer(self.h, EP_OUT, b, len(pkt), ctypes.byref(n), timeout), n.value

    def _in(self, size=1024, timeout=500):
        b = (ctypes.c_ubyte * size)()
        n = ctypes.c_int(0)
        rc = self.lib.libusb_bulk_transfer(self.h, EP_IN, b, size, ctypes.byref(n), timeout)
        return rc, bytes(bytearray(b[:n.value]))

    def write_mem(self, addr, payload):
        self.pid = (self.pid + 1) & 0xFFFF or 1
        pkt = struct.pack("<I H H H H", 0x43563355, 0x4000, 0x0802, 8 + len(payload), self.pid) + \
            struct.pack("<Q", addr) + payload
        rc, _ = self._out(pkt)
        if rc:
            return self.err(rc)
        rc, d = self._in(64)
        if rc or len(d) < 12:
            return self.err(rc) if rc else "short"
        magic, status, c, sz, rid = struct.unpack("<I H H H H", d[:12])
        return None if status == 0 else "status=0x%04x" % status

    def open(self):
        ctx = ctypes.c_void_p()
        self.lib.libusb_init(ctypes.byref(ctx))
        self.h = self.lib.libusb_open_device_with_vid_pid(ctx, VID, PID)
        if not self.h:
            raise RuntimeError("U3V open failed (camera busy?)")
        for itf in (0, 1, 2):
            self.lib.libusb_claim_interface(self.h, itf)

    def start(self, width=640, height=480, fps=30):
        for addr, data in [
            (SIRM + 0x1C, struct.pack("<I", 0x100000)), (SIRM + 0x20, struct.pack("<I", 2)),
            (SIRM + 0x24, struct.pack("<I", 0x76030)), (SIRM + 0x28, struct.pack("<I", 0)),
            (SIRM + 0x18, struct.pack("<I", 0x400)), (SIRM + 0x2C, struct.pack("<I", 0x400)),
            (SIRM + 0x04, struct.pack("<I", 1)), (0x030804, struct.pack("<I", 1)),
        ]:
            self.write_mem(addr, data)

    def stop(self):
        self.write_mem(0x030804, struct.pack("<I", 0))
        self.write_mem(SIRM + 0x04, struct.pack("<I", 0))

    def close(self):
        if self.h:
            self.lib.libusb_close(self.h)
            self.h = None

    def read_payload(self, timeout_ms=1000):
        """Block until a full frame payload is assembled. Returns (bytes, stats).

        Copies through numpy (releases the GIL for the bulk memcpy); slicing the
        ctypes buffer directly holds the GIL long enough to stall the pipeline.
        """
        cur = None
        stats = {"chunks": 0, "resets": 0}
        while True:
            n = ctypes.c_int(0)
            rc = self.lib.libusb_bulk_transfer(self.h, EP_STREAM, self.buf, 8 << 20, ctypes.byref(n), timeout_ms)
            if rc != 0:
                return None, stats
            sz = n.value
            if sz == 52:                      # leader
                cur = None
                stats["resets"] += 1
                continue
            if sz == 36:                      # trailer
                if cur is not None and len(cur) > 100000:
                    return cur, stats
                continue
            chunk = np.frombuffer(self.buf, dtype=np.uint8, count=sz).tobytes()
            cur = chunk if cur is None else cur + chunk
            stats["chunks"] += 1

# ---------------------------------------------------------------- payload decode
HDR = 48
DEPTH_BYTES_2P = 640 * 480 * 2
RGB_BYTES = 1536 * 1280 * 3
FRAME_BYTES = HDR + DEPTH_BYTES_2P + RGB_BYTES          # 6,512,688 (HDR / dynamic exposure)
FRAME_BYTES_1P = HDR + 307200 + RGB_BYTES                # 6,205,488 (single depth plane)


def decode_payload(p, want_bgr=True, small_hw=None):
    """U3V payload -> (depth uint16 480x640, rgb uint8 1280x1536x3 or None).

    Layout (both variants share the planar colour block):
      48 B header | 2x(320x480 uint16) column-interleaved depth  | 3x(1536x1280 uint8) R,G,B
    The short variant carries only one depth plane (320x480, half resolution).
    """
    n = len(p)
    DEPTH_BYTES = 640 * 480 * 2
    # depth-only stream (colour disabled on the camera): 640x480 uint16 + 8 B tail
    if DEPTH_BYTES <= n <= DEPTH_BYTES + 4096:
        d = np.frombuffer(p, dtype=np.uint16, count=640 * 480).reshape(480, 640).copy()
        return d, None
    # depth + JPEG colour: the colour block is a JPEG stream somewhere after the
    # depth plane (sizes are variable, so locate it by its SOI/EOI markers)
    if n > DEPTH_BYTES + 1024:
        i = p.rfind(b"\xff\xd8\xff")
        j = p.rfind(b"\xff\xd9")
        if i > 0 and j > i:
            dblk = p[max(0, i - DEPTH_BYTES):i]
            if len(dblk) >= DEPTH_BYTES:
                d = np.frombuffer(dblk[-DEPTH_BYTES:], dtype=np.uint16).reshape(480, 640).copy()
                v = d[d > 0]
                if v.size > 0 and 100 < float(np.median(v)) < 20000:
                    img = cv2.imdecode(np.frombuffer(p[i:j + 2], dtype=np.uint8), cv2.IMREAD_COLOR)
                    if img is not None:
                        if small_hw is not None:
                            h2, w2 = small_hw
                            img = cv2.resize(img, (w2, h2), interpolation=cv2.INTER_AREA)
                        return d, img
    if not (FRAME_BYTES_1P - 2048 <= n <= FRAME_BYTES + 40960):
        return None, None
    cblk = np.frombuffer(p, dtype=np.uint8, count=RGB_BYTES, offset=n - RGB_BYTES)
    planes = cblk.reshape(3, 1280, 1536)
    dblk = p[HDR:n - RGB_BYTES]
    # Depth (millimetres).  Two on-wire variants:
    #   614,400 B  -> plain 640x480 uint16 depth          (full precision)
    #   307,200 B  -> 640x480 uint8 depth, 256 mm per LSB (quantised variant
    #                 the camera emits in the compressed/auto-exposure mode)
    if len(dblk) >= 640 * 480 * 2:
        d = np.frombuffer(dblk, dtype=np.uint16, count=640 * 480).reshape(480, 640).copy()
    elif len(dblk) >= 640 * 480:
        u8 = np.frombuffer(dblk, dtype=np.uint8, count=640 * 480).reshape(480, 640)
        d = (u8.astype(np.uint16) << 8)
    else:
        return None, None
    if not want_bgr:
        return d, planes
    if small_hw is not None:
        # resize each colour plane straight to the target size: avoids the
        # 1536x1280 stack + cvtColor + resize (~29 ms -> ~10 ms)
        h2, w2 = small_hw
        ps = [cv2.resize(planes[i], (w2, h2), interpolation=cv2.INTER_AREA) for i in range(3)]
        small = cv2.merge(ps)
        return d, cv2.cvtColor(small, cv2.COLOR_RGB2BGR)
    rgb = np.stack([planes[0], planes[1], planes[2]], axis=2)
    return d, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
