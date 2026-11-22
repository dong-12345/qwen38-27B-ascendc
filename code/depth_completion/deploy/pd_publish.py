#!/usr/bin/env python3
"""pd_publish.py - Foxglove publisher running in its own process.

Reads the RGB frame + the completed depth written by pd_complete.py (ping-pong
buffers), encodes and pushes to Foxglove.  Separate process => the JPEG/colour
work never competes with capture or completion for the GIL.
"""
import os
import sys
import time
from multiprocessing import shared_memory

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
cv2.setNumThreads(1)

import foxglove_ws as FW
import pd_core as C

FRAME_ID = "rgb"
W, H = 640, 480


def run(rgb_names, out_names, meta, port, depth_every, cloud_every, cloud_stride,
        jpeg_q, intr):
    rgbs, outs = [], []
    for i in range(2):
        r = shared_memory.SharedMemory(name=rgb_names[i])
        o = shared_memory.SharedMemory(name=out_names[i])
        rgbs.append((r, np.ndarray((H, W, 3), dtype=np.uint8, buffer=r.buf)))
        outs.append((o, np.ndarray((H, W), dtype=np.uint16, buffer=o.buf)))

    srv = FW.FoxgloveServer(port=port)
    chans = {
        "rgb": srv.add_channel("/camera/rgb", "foxglove.CompressedImage", FW.COMPRESSED_IMAGE_SCHEMA, encoding="json"),
        "col": srv.add_channel("/depth/colored", "foxglove.CompressedImage", FW.COMPRESSED_IMAGE_SCHEMA, encoding="json"),
        "pc": srv.add_channel("/points", "foxglove.PointCloud", FW.POINTCLOUD_PROTO),
    }
    srv.start()
    print("[pub] ws://0.0.0.0:%d  %s" % (port, ", ".join(c.topic for c in chans.values())), flush=True)

    fx, fy, cx, cy = intr
    last = -1
    last_cloud = 0.0
    n = 0
    lat = []
    t_start = time.time()

    while meta[0] == 0:
        c = int(meta[3])
        if c == last:
            time.sleep(0.001)
            continue
        last = c
        slot = c % 2
        ts = float(meta[4 + slot])
        gain = float(meta[1])
        ns = int(ts * 1e9)
        rgb = rgbs[slot][1].copy()
        depth_mm = outs[slot][1].copy()
        if abs(gain - 1.0) > 0.02:
            rgb = cv2.convertScaleAbs(rgb, alpha=gain)
        img = C.grade(rgb)
        ok, jb = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, jpeg_q])
        if ok:
            srv.publish(chans["rgb"].id, FW.compressed_image_json(FRAME_ID, "jpeg", jb.tobytes(), ns), ns)
        ok, cb = cv2.imencode(".jpg", C.colorize(depth_mm), [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            srv.publish(chans["col"].id, FW.compressed_image_json(FRAME_ID, "jpeg", cb.tobytes(), ns), ns)
        now = time.time()
        if cloud_every > 0 and now - last_cloud >= cloud_every:
            last_cloud = now
            s = max(1, int(cloud_stride))
            z = depth_mm[::s, ::s].astype(np.float32) * 0.001
            ys, xs = np.mgrid[0:H:s, 0:W:s]
            X = (xs.astype(np.float32) - cx) * z / fx
            Y = (ys.astype(np.float32) - cy) * z / fy
            pts = np.stack([X, Y, z], axis=-1).reshape(-1, 3)
            pts = pts[pts[:, 2] > 0.05]
            if pts.size:
                srv.publish(chans["pc"].id, FW.encode_point_cloud(FRAME_ID, pts, ns), ns)
        lat.append((time.time() - ts) * 1e3)
        n += 1
        if n % 60 == 0:
            print("[pub] n=%d %.1f fps  capture->foxglove p50=%.0f ms"
                  % (n, n / (time.time() - t_start), float(np.median(lat[-60:]))), flush=True)

    srv.close()
    for lst in (rgbs, outs):
        for shm, _ in lst:
            shm.close()
    return n
