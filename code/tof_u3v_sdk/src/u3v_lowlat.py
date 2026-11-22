# -*- coding: utf-8 -*-
"""u3v_lowlat.py — 绕过海康 Mv3dRgbd SDK 的 U3V 直连低延迟取流（libusb/ctypes）

背景（2026-11-20 板子实测）：
  海康 Linux SDK 的 MV3D_RGBD_FetchFrame 单线程吃满 98.5% CPU、只有 7~14fps、170~500ms 延迟；
  本脚本用 libusb 直接按 USB3 Vision 协议取流（协议序列由 usbmon 抓包还原）：
     SIRM.payload_size=1MB, payload_count=2, transfer1_size=0x76030, transfer2=0,
     max_leader=0x400, max_trailer=0x400, SIRM.control=1, 0x030804=1(AcquisitionStart)
  实测：**30.1~30.5 fps / 74~75 MB/s**（2.58MB/帧），CPU 占用远低于厂商 SDK。

输出每帧：leader(52B, 含帧号) + payload(2,580,528B: depth 640x480 C16 + JPEG 彩图) + trailer(36B)

用法：
    python3 u3v_lowlat.py --seconds 8            # 跑 8 秒，打印 fps/延迟/CPU
    python3 u3v_lowlat.py --seconds 8 --snap     # 额外保存一帧 depth(16bit PNG)+color(jpg)
"""
import argparse
import ctypes
import os
import struct
import sys
import time

VID, PID = 0x2BDF, 0x0001
EP_OUT, EP_IN, EP_STREAM = 0x01, 0x82, 0x84
SIRM = 0x62000
HERE = os.path.dirname(os.path.abspath(__file__))

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


def png16_gray(path, w, h, buf):
    import array
    import zlib
    a = array.array("H")
    a.frombytes(buf[:w * h * 2])
    if sys.byteorder == "little":
        a.byteswap()
    be = a.tobytes()
    stride = w * 2
    raw = b"".join(b"\x00" + be[y * stride:(y + 1) * stride] for y in range(h))

    def chunk(t, d):
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xffffffff)
    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 16, 0, 0, 0, 0))
                + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))


def start_stream():
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


def stop_stream():
    write_mem(0x030804, struct.pack("<I", 0))
    write_mem(SIRM + 0x04, struct.pack("<I", 0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=8.0)
    ap.add_argument("--snap", action="store_true", help="保存一帧 depth16.png + color.jpg")
    ap.add_argument("--outdir", default="")
    a = ap.parse_args()

    ctx = ctypes.c_void_p()
    lib.libusb_init(ctypes.byref(ctx))
    h = lib.libusb_open_device_with_vid_pid(ctx, VID, PID)
    if not h:
        sys.exit("open 2bdf:0001 failed (camera in use?)")
    _h[0] = h
    for itf in (0, 1, 2):
        rc = lib.libusb_claim_interface(h, itf)
        print("claim iface %d: %s" % (itf, err(rc) if rc else "OK"))
    try:
        print("start stream ...")
        start_stream()
        buf = (ctypes.c_ubyte * (4 << 20))()
        buflen = 4 << 20
        frames = []          # (host_us, frame_id, transfer_ms)
        payload = None
        t_start = time.perf_counter()
        c0 = cpu_times()
        nbytes = 0
        reads = 0
        while time.perf_counter() - t_start < a.seconds:
            n = ctypes.c_int(0)
            t_a = time.perf_counter()
            rc = lib.libusb_bulk_transfer(h, EP_STREAM, buf, buflen, ctypes.byref(n), 800)
            t_b = time.perf_counter()
            if rc != 0:
                print("read error:", err(rc))
                break
            reads += 1
            nbytes += n.value
            sz = n.value
            if sz == 2_580_528:      # 一帧完整 payload（depth+color）
                frames.append((t_b * 1e6, None, (t_b - t_a) * 1000))
                if a.snap and payload is None:
                    payload = bytes(bytearray(buf[:sz]))
            elif sz == 52:           # leader：offset 8 是帧号
                fid = struct.unpack_from("<I", buf, 8)[0]
                if frames:
                    frames[-1] = (frames[-1][0], fid, frames[-1][2])
        wall = time.perf_counter() - t_start
        c1 = cpu_times()
        print("\nRESULT frames=%d wall=%.2fs -> %.2f fps | bytes=%d (%.2f MB/s) | reads=%d"
              % (len(frames), wall, len(frames) / wall, nbytes, nbytes / 1048576.0 / wall, reads))
        if len(frames) > 5:
            ts = [f[0] for f in frames]
            dts = sorted([(b - a_) / 1000.0 for a_, b in zip(ts, ts[1:])])
            p = lambda q: dts[min(len(dts) - 1, int(q * (len(dts) - 1)))]
            xfers = sorted([f[2] for f in frames])
            print("帧间隔(host) ms: p50 %.2f p95 %.2f (30fps=33.3)" % (p(.5), p(.95)))
            print("payload USB 传输耗时 ms: p50 %.2f (2.58MB -> %.0f MB/s 线上速率)"
                  % (xfers[len(xfers) // 2], 2.58 / (xfers[len(xfers) // 2] / 1000.0)))
            ids = [f[1] for f in frames if f[1] is not None]
            if len(ids) > 2:
                print("设备帧号 %d -> %d (span %d)，跨 %.2fs -> 设备帧率 %.2f fps"
                      % (ids[0], ids[-1], ids[-1] - ids[0], (ts[-1] - ts[0]) / 1e6,
                         (ids[-1] - ids[0]) / ((ts[-1] - ts[0]) / 1e6)))
        busy = (c1[0] - c0[0]) / max(1, c1[1] - c0[1]) * 100
        print("取流期间系统 CPU busy: %.1f%%（4 核口径；厂商 SDK 同配置约 75%%）" % busy)

        if a.snap and payload:
            outdir = a.outdir or os.path.join(os.path.dirname(HERE), "results", "u3v_snap")
            os.makedirs(outdir, exist_ok=True)
            # payload = depth(640x480 C16) + JPEG 彩图（中间可能有 24B U3V payload 头）
            jpg_at = payload.find(b"\xff\xd8")
            depth = payload[:jpg_at] if jpg_at > 0 else payload
            save_depth = None
            for hh, ww in ((480, 640), (1280, 1536)):
                if len(depth) >= ww * hh * 2:
                    save_depth = (ww, hh, depth[:ww * hh * 2])
                    break
            if save_depth:
                ww, hh, dbuf = save_depth
                png16_gray(os.path.join(outdir, "depth16.png"), ww, hh, dbuf)
                print("保存 depth %dx%d -> %s/depth16.png" % (ww, hh, outdir))
            if jpg_at > 0:
                open(os.path.join(outdir, "color.jpg"), "wb").write(payload[jpg_at:])
                print("保存 color jpeg %d bytes -> %s/color.jpg" % (len(payload) - jpg_at, outdir))
            open(os.path.join(outdir, "payload.bin"), "wb").write(payload[:64])
            print("payload 前 64B:", payload[:64].hex())
    finally:
        stop_stream()
        for itf in (2, 1, 0):
            lib.libusb_release_interface(h, itf)
        lib.libusb_close(h)


if __name__ == "__main__":
    main()
