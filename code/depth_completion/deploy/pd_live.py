#!/usr/bin/env python3
"""pd_live.py - U3V RGB-D capture -> CPU depth completion -> Foxglove WebSocket.

No NPU, no learned model: median outlier removal + colour-guided hole filling
(pd_core.ClassicCompleter), 640x480, published on ws://<board>:8765.

  source /usr/local/Ascend/ascend-toolkit/set_env.sh   # not needed any more
  python3 deploy/pd_live.py --port 8765 --seconds 0
"""
import argparse
import json
import os
import sys
import threading
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
cv2.setNumThreads(1)

import foxglove_ws as FW
import pd_core as C
from pd_capture import U3V, decode_payload

FRAME_ID = "rgb"


class Grabber(threading.Thread):
    """Reads U3V frames and keeps only the newest decoded one."""

    def __init__(self):
        super().__init__(daemon=True)
        self.lock = threading.Lock()
        self.frame = None
        self.count = 0
        self.bad = 0
        self.running = False

    def run(self):
        u = U3V()
        u.open()
        u.start()
        self.running = True
        t_log = time.perf_counter()
        n_log = 0
        while self.running:
            p, st = u.read_payload(1500)
            if p is None:
                self.bad += 1
                continue
            dep, rgb = decode_payload(p, small_hw=(480, 640))
            if dep is None or float((dep > 0).mean()) < 0.02:
                self.bad += 1
                continue
            rgb640 = rgb          # None when the camera streams depth only
            with self.lock:
                self.frame = (time.time(), dep, rgb640)   # colour already downscaled
            self.count += 1
            n_log += 1
            if n_log >= 60:
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
    """Encodes and pushes to Foxglove; never blocks the capture loop."""

    def __init__(self, srv, chans, depth_every=0.1, jpeg_q=80, cloud_every=0.3,
                 cloud_stride=4, intr=(507.604, 507.702, 326.961, 243.043)):
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
        self.lat = []

    def submit(self, ts, rgb, depth_mm, gain=1.0):
        with self.lock:
            self.item = (ts, rgb, depth_mm, gain)

    def _cloud(self, d_mm):
        s = max(1, int(self.cloud_stride))
        z = d_mm[::s, ::s].astype(np.float32) * 0.001
        ys, xs = np.mgrid[0:d_mm.shape[0]:s, 0:d_mm.shape[1]:s]
        X = (xs.astype(np.float32) - self.cx) * z / self.fx
        Y = (ys.astype(np.float32) - self.cy) * z / self.fy
        pts = np.stack([X, Y, z], axis=-1).reshape(-1, 3)
        return pts[pts[:, 2] > 0.05]

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
            ts, rgb, depth_mm, gain = it
            t0 = time.perf_counter()
            ns = int(ts * 1e9)
            small = rgb                      # already 640x480 from the grabber
            if small is not None:
                if abs(gain - 1.0) > 0.02:
                    small = cv2.convertScaleAbs(small, alpha=gain)
                small = C.grade(small)
                ok, jb = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_q])
                if ok:
                    self.srv.publish(self.chans["rgb"].id,
                                     FW.compressed_image_json(FRAME_ID, "jpeg", jb.tobytes(), ns), ns)
            ok, cb = cv2.imencode(".jpg", C.colorize(depth_mm), [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok:
                self.srv.publish(self.chans["col"].id,
                                 FW.compressed_image_json(FRAME_ID, "jpeg", cb.tobytes(), ns), ns)
            now = time.perf_counter()
            if now - last_depth >= self.depth_every:
                last_depth = now
                if "dep" in self.chans:
                    ok, pb = cv2.imencode(".png", depth_mm, [cv2.IMWRITE_PNG_COMPRESSION, 1])
                    if ok:
                        self.srv.publish(self.chans["dep"].id,
                                         FW.compressed_image_json(FRAME_ID, "png", pb.tobytes(), ns), ns)
                        self.n_depth += 1
            if self.cloud_every > 0 and now - last_cloud >= self.cloud_every:
                last_cloud = now
                pts = self._cloud(depth_mm)
                if pts.size:
                    self.srv.publish(self.chans["pc"].id,
                                     FW.encode_point_cloud(FRAME_ID, pts, ns), ns)
                    self.n_cloud += 1
            self.ms_encode = (time.perf_counter() - t0) * 1e3
            self.lat.append((time.time() - ts) * 1e3)
            self.n += 1

    def stop(self):
        self.running = False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--seconds", type=float, default=0)
    ap.add_argument("--depth-every", type=float, default=0.1)
    ap.add_argument("--cloud-every", type=float, default=0.3)
    ap.add_argument("--cloud-stride", type=int, default=4)
    ap.add_argument("--jpeg-q", type=int, default=80)
    ap.add_argument("--no-fill", action="store_true", help="publish the raw ToF (no completion)")
    ap.add_argument("--depth16", action="store_true", help="also publish /depth/depth16 (16-bit PNG)")
    ap.add_argument("--temporal", type=float, default=0.0, help="temporal EMA weight (0=off = sharpest)")
    ap.add_argument("--regions", action="store_true",
                    help="replace temporally unstable areas with a smooth surface (denoise, blurs)")
    ap.add_argument("--save-dir", default="")
    ap.add_argument("--seconds-only", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()

    srv = FW.FoxgloveServer(port=args.port)
    chans = {
        "rgb": srv.add_channel("/camera/rgb", "foxglove.CompressedImage", FW.COMPRESSED_IMAGE_SCHEMA,
                               encoding="json"),
        "col": srv.add_channel("/depth/colored", "foxglove.CompressedImage", FW.COMPRESSED_IMAGE_SCHEMA,
                               encoding="json"),
        "pc": srv.add_channel("/points", "foxglove.PointCloud", FW.POINTCLOUD_PROTO),
    }
    if args.depth16:
        chans["dep"] = srv.add_channel("/depth/depth16", "foxglove.CompressedImage",
                                       FW.COMPRESSED_IMAGE_SCHEMA, encoding="json")
    srv.start()
    print("[ws] ws://0.0.0.0:%d  topics: %s" % (args.port, ", ".join(c.topic for c in chans.values())), flush=True)

    intr = (507.604, 507.702, 326.961, 243.043)
    calib_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "rgbd_calib.json")
    if os.path.exists(calib_path):
        try:
            k = json.load(open(calib_path))["align0"]["depth"]
            intr = (k["fx"], k["fy"], k["cx"], k["cy"])
        except Exception:
            pass

    pub = Publisher(srv, chans, args.depth_every, args.jpeg_q,
                    args.cloud_every, args.cloud_stride, intr)
    pub.start()

    comp = None if args.no_fill else C.ClassicCompleter(temporal=args.temporal, regions=args.regions)
    ae = C.AutoExposure(target=110.0)
    g = Grabber()
    g.start()
    while g.frame is None and g.running:
        time.sleep(0.05)

    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)

    lat, t_start, last_key, n, saved = [], time.time(), None, 0, 0
    try:
        while True:
            if args.seconds and time.time() - t_start > args.seconds:
                break
            fr = g.latest()
            if fr is None or fr[0] == last_key:
                time.sleep(0.001)
                continue
            last_key = fr[0]
            ts, dep_mm, rgb = fr
            t0 = time.perf_counter()
            if comp is not None:
                dep_out = comp(dep_mm, rgb)
            else:
                dep_out = dep_mm
            dt = (time.perf_counter() - t0) * 1e3
            lat.append(dt)
            n += 1
            gain = ae.gain
            if rgb is not None:
                ae.update(float(cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY).mean()))
            pub.submit(ts, rgb, np.ascontiguousarray(dep_out), gain)
            if args.save_dir and saved < 4:
                np.save(os.path.join(args.save_dir, "tof.npy"), dep_mm)
                np.save(os.path.join(args.save_dir, "filled.npy"), dep_out)
                cv2.imwrite(os.path.join(args.save_dir, "rgb.png"), rgb)
                saved += 1
            if n % 60 == 0:
                el = time.time() - t_start
                print("[run] n=%d %.1f fps  proc p50=%.1f ms  e2e p50=%.0f ms  cap=%.1f fps bad=%d"
                      % (n, n / el, float(np.median(lat[-60:])),
                         float(np.median(pub.lat[-60:])) if pub.lat else -1,
                         g.count / el, g.bad), flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        g.stop()
        pub.stop()
        srv.close()
        el = time.time() - t_start
        print("RESULT " + json.dumps({
            "frames": n, "seconds": round(el, 2), "fps": round(n / el, 2),
            "proc_p50_ms": round(float(np.median(lat)), 2) if lat else None,
            "proc_p95_ms": round(float(np.percentile(lat, 95)), 2) if lat else None,
            "capture_fps": round(g.count / el, 2), "bad_frames": g.bad,
            "encode_ms": round(pub.ms_encode, 2), "published": pub.n,
            "e2e_lat_p50_ms": round(float(np.median(pub.lat[-120:])), 1) if pub.lat else None,
            "e2e_lat_p95_ms": round(float(np.percentile(pub.lat[-120:], 95)), 1) if pub.lat else None,
            "clouds": pub.n_cloud, "depth16": pub.n_depth,
        }), flush=True)
        sys.stdout.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
