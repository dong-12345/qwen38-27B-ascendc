#!/usr/bin/env python3
"""pd_live.py - U3V -> PromptDA (Ascend NPU) -> Foxglove WebSocket, 640x480 ToF + 1536x1280 RGB.

Board usage:
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
  python3 deploy/pd_live.py --om models/promptda_392x518.om --port 8765

Channels: /camera/rgb (jpeg), /depth/tof (png16), /depth/completed (png16),
          /depth/colored (jpeg)
"""
import argparse
import ctypes
import json
import os
import struct
import sys
import threading
import time

import cv2
import numpy as np

cv2.setNumThreads(1)          # avoid OpenCV's thread pool fighting the pipeline threads

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/usr/local/Ascend/ascend-toolkit/latest/pyACL/python/site-packages")
try:
    import acl  # noqa: E402
except ImportError:      # dry-run mode does not need the NPU
    acl = None

import foxglove_ws as FW
import pd_core as C
from pd_capture import U3V, DEPTH_SHAPE, RGB_SHAPE, decode_payload

ACL_MEM_MALLOC_NORMAL_ONLY = 0
ACL_MEMCPY_HOST_TO_DEVICE = 1
ACL_MEMCPY_DEVICE_TO_HOST = 2
DEPTH_BYTES = DEPTH_SHAPE[0] * DEPTH_SHAPE[1] * 2
RGB_BYTES = RGB_SHAPE[0] * RGB_SHAPE[1] * 3
HDR = 48
FRAME_BYTES = HDR + DEPTH_BYTES + RGB_BYTES


def check(ret, msg):
    if ret != 0:
        raise RuntimeError("%s failed ret=%d" % (msg, ret))


class Om:
    """Two-input Ascend model (fp32 in/out), buffers allocated once."""

    def __init__(self, path, hw=(None, None), n_in=2):
        self.hw = (C.MODEL_H, C.MODEL_W) if hw[0] is None else (int(hw[0]), int(hw[1]))
        if acl is None:
            raise RuntimeError("pyACL not importable - source /usr/local/Ascend/ascend-toolkit/set_env.sh")
        check(acl.init(), "acl.init")
        check(acl.rt.set_device(0), "set_device")
        self.ctx, ret = acl.rt.create_context(0)
        check(ret, "create_context")
        self.stream, ret = acl.rt.create_stream()
        check(ret, "create_stream")
        self.model_id, ret = acl.mdl.load_from_file(path)
        check(ret, "load_from_file")
        self.desc = acl.mdl.create_desc()
        check(acl.mdl.get_desc(self.desc, self.model_id), "get_desc")
        self.in_ds = acl.mdl.create_dataset()
        self.in_bufs = []
        for i in range(acl.mdl.get_num_inputs(self.desc)):
            size = acl.mdl.get_input_size_by_index(self.desc, i)
            ptr, ret = acl.rt.malloc(size, ACL_MEM_MALLOC_NORMAL_ONLY)
            check(ret, "malloc in")
            acl.mdl.add_dataset_buffer(self.in_ds, acl.create_data_buffer(ptr, size))
            self.in_bufs.append((ptr, size))
        self.out_ds = acl.mdl.create_dataset()
        self.out_bufs = []
        for i in range(acl.mdl.get_num_outputs(self.desc)):
            size = acl.mdl.get_output_size_by_index(self.desc, i)
            ptr, ret = acl.rt.malloc(size, ACL_MEM_MALLOC_NORMAL_ONLY)
            check(ret, "malloc out")
            acl.mdl.add_dataset_buffer(self.out_ds, acl.create_data_buffer(ptr, size))
            self.out_bufs.append((ptr, size))
        self.in_sizes = [s for _, s in self.in_bufs]
        self.out_size = self.out_bufs[0][1]
        self.out_host = np.zeros(self.out_size, dtype=np.uint8)
        self.out_ptr = acl.util.numpy_to_ptr(self.out_host)
        self.n_out_el = self.out_size // 4

    def infer(self, arrays):
        for (ptr, size), arr in zip(self.in_bufs, arrays):
            data = np.ascontiguousarray(arr, dtype=np.float32)
            check(acl.rt.memcpy(ptr, size, acl.util.numpy_to_ptr(data), data.nbytes,
                                ACL_MEMCPY_HOST_TO_DEVICE), "H2D")
        t0 = time.perf_counter()
        check(acl.mdl.execute_async(self.model_id, self.in_ds, self.out_ds, self.stream), "exec")
        check(acl.rt.synchronize_stream(self.stream), "sync")
        dt = (time.perf_counter() - t0) * 1e3
        check(acl.rt.memcpy(self.out_ptr, self.out_size, self.out_bufs[0][0], self.out_size,
                            ACL_MEMCPY_DEVICE_TO_HOST), "D2H")
        return self.out_host.view(np.float32)[: self.n_out_el].reshape(self.hw[0], self.hw[1]), dt

    def close(self):
        for fn in (lambda: acl.mdl.destroy_dataset(self.in_ds),
                   lambda: acl.mdl.destroy_dataset(self.out_ds),
                   lambda: acl.mdl.unload(self.model_id),
                   lambda: acl.mdl.destroy_desc(self.desc),
                   lambda: acl.rt.destroy_stream(self.stream),
                   lambda: acl.rt.destroy_context(self.ctx)):
            try:
                fn()
            except Exception:
                pass


class Grabber(threading.Thread):
    """Continuously grab U3V frames; keeps only the newest decoded frame."""

    def __init__(self):
        super().__init__(daemon=True)
        self.lock = threading.Lock()
        self.frame = None          # (ts, depth_u16, rgb_bgr)
        self.count = 0
        self.bad = 0
        self.running = False
        self.u = None

    def run(self):
        u = U3V()
        u.open()
        u.start()
        self.u = u
        self.running = True
        t_log = time.perf_counter()
        n_log = 0
        while self.running:
            p, st = u.read_payload(1500)
            if p is None:
                self.bad += 1
                continue
            dep, rgb = decode_payload(p)
            if dep is None or float((dep > 0).mean()) < 0.02:
                self.bad += 1
                continue
            ts = time.time()
            with self.lock:
                self.frame = (ts, dep, rgb)   # both are fresh arrays from the decoder
            self.count += 1
            n_log += 1
            if n_log >= 30:
                now = time.perf_counter()
                print("[cap] %.1f fps" % (n_log / (now - t_log)), flush=True)
                t_log, n_log = now, 0
        u.stop()
        u.close()

    def latest(self):
        with self.lock:
            return self.frame

    def stop(self):
        self.running = False

class Publisher(threading.Thread):
    """Encodes and pushes to Foxglove; never blocks the inference loop."""

    def __init__(self, srv, chans, depth_every=0.1, jpeg_q=80, cloud_every=0.2,
                 cloud_stride=4, intr=(507.6, 507.7, 326.96, 243.04)):
        super().__init__(daemon=True)
        self.srv = srv
        self.chans = chans
        self.depth_every = depth_every
        self.jpeg_q = jpeg_q
        self.cloud_every = cloud_every
        self.cloud_stride = cloud_stride
        self.fx, self.fy, self.cx, self.cy = intr
        self.lock = threading.Lock()
        self.item = None
        self.running = False
        self.n = 0
        self.n_depth = 0
        self.n_cloud = 0
        self.ms_encode = 0.0

    def _cloud(self, d_mm):
        s = max(1, int(self.cloud_stride))
        z = d_mm[::s, ::s].astype(np.float32) * 0.001
        ys, xs = np.mgrid[0:d_mm.shape[0]:s, 0:d_mm.shape[1]:s]
        X = (xs.astype(np.float32) - self.cx) * z / self.fx
        Y = (ys.astype(np.float32) - self.cy) * z / self.fy
        pts = np.stack([X, Y, z], axis=-1).reshape(-1, 3)
        ok = pts[:, 2] > 0.05
        return pts[ok]

    def submit(self, ts, rgb, dep_mm, fused_mm, depth_m, gain=1.0):
        with self.lock:
            self.item = (ts, rgb, dep_mm, fused_mm, depth_m, gain)

    def run(self):
        self.running = True
        last_depth = 0.0
        last_cloud = 0.0
        while self.running:
            with self.lock:
                it = self.item
                self.item = None
            if it is None:
                time.sleep(0.002)
                continue
            ts, rgb, dep_mm, fused_mm, depth_m, gain = it
            t0 = time.perf_counter()
            ns = int(ts * 1e9)
            small = cv2.resize(rgb, (640, 480), interpolation=cv2.INTER_LINEAR)
            if abs(gain - 1.0) > 0.02:
                small = cv2.convertScaleAbs(small, alpha=gain)
            small = C.grade(small)          # match HiViewer's look (gamma + saturation)
            ok, jb = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_q])
            if ok:
                self.srv.publish(self.chans["rgb"].id,
                                 FW.encode_compressed_image("rgb", "jpeg", jb.tobytes(), ns), ns)
            col = C.colorize(depth_m)
            ok, cb = cv2.imencode(".jpg", col, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if ok:
                self.srv.publish(self.chans["col"].id,
                                 FW.encode_compressed_image("depth", "jpeg", cb.tobytes(), ns), ns)
            now = time.perf_counter()
            if now - last_depth >= self.depth_every:
                last_depth = now
                ok, pb = cv2.imencode(".png", fused_mm, [cv2.IMWRITE_PNG_COMPRESSION, 1])
                if ok:
                    self.srv.publish(self.chans["dep"].id,
                                     FW.encode_compressed_image("depth", "16UC1; png", pb.tobytes(), ns), ns)
                    self.n_depth += 1
            if self.cloud_every > 0 and time.perf_counter() - last_cloud >= self.cloud_every:
                last_cloud = time.perf_counter()
                pts = self._cloud(fused_mm)
                if pts.size:
                    self.srv.publish(self.chans["pc"].id,
                                     FW.encode_point_cloud("depth", pts, ns), ns)
            self.ms_encode = (time.perf_counter() - t0) * 1e3
            self.n += 1

    def stop(self):
        self.running = False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--om", default="models/promptda_392x518.onnx".replace(".onnx", ".om"))
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--seconds", type=float, default=0, help="0 = run forever")
    ap.add_argument("--depth-every", type=float, default=0.1)
    ap.add_argument("--jpeg-q", type=int, default=80)
    ap.add_argument("--save-dir", default="")
    ap.add_argument("--no-publish", action="store_true")
    ap.add_argument("--frames", type=int, default=0, help="stop after N inferred frames")
    ap.add_argument("--dry-run", action="store_true", help="skip the NPU (capture/publish benchmark)")
    ap.add_argument("--model-size", default="", help="HxW of the OM input, e.g. 196x252")
    ap.add_argument("--cloud-every", type=float, default=0.2, help="seconds between point clouds (0=off)")
    ap.add_argument("--cloud-stride", type=int, default=4, help="point cloud pixel stride")
    ap.add_argument("--denoise", default="off", choices=["off", "temporal", "guided", "both"],
                    help="depth noise suppression before completion "
                         "(off = fastest; both = best quality, ~+10 ms)")
    args = ap.parse_args()

    srv = None
    chans = {}
    if not args.no_publish:
        srv = FW.FoxgloveServer(port=args.port)
        chans["rgb"] = srv.add_channel("/camera/rgb", "foxglove.CompressedImage", "")
        chans["col"] = srv.add_channel("/depth/colored", "foxglove.CompressedImage", "")
        chans["dep"] = srv.add_channel("/depth/completed", "foxglove.CompressedImage", "")
        chans["pc"] = srv.add_channel("/points", "foxglove.PointCloud", "")
        srv.start()
        print("[ws] listening on 0.0.0.0:%d" % args.port, flush=True)

    if args.model_size:
        h, w = [int(v) for v in args.model_size.lower().split("x")]
        C.set_model_size(h, w)
    om = None
    if not args.dry_run:
        om = Om(args.om)
        print("[npu] %s in=%s out=%d" % (args.om, om.in_sizes, om.out_size), flush=True)

    intr = (507.604, 507.702, 326.961, 243.043)
    calib_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "rgbd_calib.json")
    if os.path.exists(calib_path):
        try:
            import json as _json
            k = _json.load(open(calib_path))["align0"]["depth"]
            intr = (k["fx"], k["fy"], k["cx"], k["cy"])
        except Exception:
            pass
    pub = Publisher(srv, chans, args.depth_every, args.jpeg_q,
                    cloud_every=args.cloud_every, cloud_stride=args.cloud_stride, intr=intr) if srv else None
    if pub:
        pub.start()

    ae = C.AutoExposure(target=110.0)
    if args.denoise == "temporal":
        temporal = C.TemporalDepth(alpha=0.45)
    elif args.denoise == "both":
        temporal = C.DepthCleaner()
    else:
        temporal = None
    g = Grabber()
    g.start()
    while g.frame is None and g.running:
        time.sleep(0.05)

    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)

    lat = []
    npu_ms = []
    t_start = time.time()
    last_key = None
    n = 0
    saved = 0
    try:
        while True:
            if args.seconds and time.time() - t_start > args.seconds:
                break
            if args.frames and n >= args.frames:
                break
            fr = g.latest()
            if fr is None or fr[0] == last_key:
                time.sleep(0.001)
                continue
            last_key = fr[0]
            ts, dep_mm, rgb = fr
            t0 = time.perf_counter()
            _d0 = time.perf_counter()
            if temporal is not None:
                dep_mm = temporal(dep_mm)
            if args.denoise == "guided":
                guide = cv2.resize(rgb, (640, 480), interpolation=cv2.INTER_LINEAR)
                dep_mm = C.denoise_depth(dep_mm, guide)
            if om is None:
                depth_m = dep_mm.astype(np.float32) * 0.001
                t_npu = 0.0
                fused = dep_mm
                small_bgr = cv2.resize(rgb, (C.MODEL_W, C.MODEL_H), interpolation=cv2.INTER_AREA)
            else:
                prompt = C.fill_prompt(dep_mm)
                img, pr, gain, small_bgr, luma = C.preprocess_gain(rgb, prompt, gain=ae.gain)
                ae.update(luma)
                out, t_npu = om.infer([img, pr])
                depth_m = C.postprocess(out, dep_mm.shape)
                fused = C.fuse(dep_mm, depth_m)
            dt = (time.perf_counter() - t0) * 1e3
            lat.append(dt)
            npu_ms.append(t_npu)
            n += 1
            if pub:
                pub.submit(ts, rgb, dep_mm, fused, depth_m, ae.gain)
            if args.save_dir and saved < 6:
                np.save(os.path.join(args.save_dir, "f%03d_depth.npy" % saved), depth_m)
                cv2.imwrite(os.path.join(args.save_dir, "f%03d_rgb.png" % saved),
                            cv2.resize(rgb, (640, 480), interpolation=cv2.INTER_LINEAR))
                if saved == 0:
                    # full set for analysis/make_promptda_compare.py
                    np.save(os.path.join(args.save_dir, "tof.npy"), dep_mm)
                    np.save(os.path.join(args.save_dir, "prompt.npy"), C.fill_prompt(dep_mm))
                    np.save(os.path.join(args.save_dir, "pred.npy"), depth_m)
                    np.save(os.path.join(args.save_dir, "fused.npy"), fused)
                    cv2.imwrite(os.path.join(args.save_dir, "rgb.png"),
                                cv2.resize(rgb, (640, 480), interpolation=cv2.INTER_AREA))
                saved += 1
            if n % 30 == 0:
                el = time.time() - t_start
                print("[run] n=%d  %.1f fps  lat p50=%.1fms  npu p50=%.1fms  cap=%.1ffps bad=%d"
                      % (n, n / el, float(np.median(lat[-30:])), float(np.median(npu_ms[-30:])),
                         g.count / el, g.bad), flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        g.stop()
        if pub:
            pub.stop()
        if srv:
            srv.close()
        if om:
            om.close()
        el = time.time() - t_start
        res = {"frames": n, "seconds": round(el, 2), "fps": round(n / el, 2),
               "lat_p50_ms": round(float(np.median(lat)), 2) if lat else None,
               "lat_p95_ms": round(float(np.percentile(lat, 95)), 2) if lat else None,
               "npu_p50_ms": round(float(np.median(npu_ms)), 2) if npu_ms else None,
               "capture_fps": round(g.count / el, 2), "bad_frames": g.bad,
               "encode_ms": round(pub.ms_encode, 2) if pub else None,
               "published": pub.n if pub else 0, "sleep_ok": True}
        print("RESULT " + json.dumps(res), flush=True)
        sys.stdout.flush()
        os._exit(0)      # the ACL runtime aborts in its destructor; skip it


if __name__ == "__main__":
    main()
