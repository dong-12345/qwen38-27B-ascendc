#!/usr/bin/env python3
"""How fast can we run 'detail-preserving completion' on this board?

Branches:
  * CPU: classical multi-scale fill on the native 640x480 ToF holes (keeps every
    valid ToF pixel untouched => full native detail), plus optional low-frequency
    fusion of the DAv2 prior in large-hole interiors.
  * NPU: DAv2-Small (168x224 or 238x308) depth-render prior.

Measures: each branch alone, then both branches in parallel (2 threads, one
barrier per frame), which is how a real pipeline would be scheduled.
"""
import argparse
import json
import statistics
import sys
import threading
import time

import numpy as np
from PIL import Image
from scipy import ndimage

sys.path.insert(0, "/usr/local/Ascend/ascend-toolkit/latest/pyACL/python/site-packages")
import acl  # noqa: E402
import cv2  # noqa: E402

ACL_MEM_MALLOC_NORMAL_ONLY = 0
ACL_MEMCPY_HOST_TO_DEVICE = 1
ACL_MEMCPY_DEVICE_TO_HOST = 2
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def check(ret, msg):
    if ret != 0:
        raise RuntimeError(f"{msg} failed ret={ret}")


class Om:
    def __init__(self, path, h, w):
        self.h, self.w = h, w
        self.model_id, ret = acl.mdl.load_from_file(path)
        check(ret, "load_from_file")
        self.desc = acl.mdl.create_desc()
        check(acl.mdl.get_desc(self.desc, self.model_id), "get_desc")
        self.stream, ret = acl.rt.create_stream()
        check(ret, "create_stream")
        self.in_ds = acl.mdl.create_dataset()
        self.in_bufs = []
        for i in range(acl.mdl.get_num_inputs(self.desc)):
            size = acl.mdl.get_input_size_by_index(self.desc, i)
            ptr, ret = acl.rt.malloc(size, ACL_MEM_MALLOC_NORMAL_ONLY)
            check(ret, "malloc in")
            buf = acl.create_data_buffer(ptr, size)
            acl.mdl.add_dataset_buffer(self.in_ds, buf)
            self.in_bufs.append((ptr, size))
        self.out_ds = acl.mdl.create_dataset()
        self.out_bufs = []
        for i in range(acl.mdl.get_num_outputs(self.desc)):
            size = acl.mdl.get_output_size_by_index(self.desc, i)
            ptr, ret = acl.rt.malloc(size, ACL_MEM_MALLOC_NORMAL_ONLY)
            check(ret, "malloc out")
            buf = acl.create_data_buffer(ptr, size)
            acl.mdl.add_dataset_buffer(self.out_ds, buf)
            self.out_bufs.append((ptr, size))
        self._data = {}

    def infer(self, img_u8):
        im = Image.fromarray(img_u8).resize((self.w, self.h), Image.BILINEAR)
        x = np.asarray(im, dtype=np.float32) / 255.0
        x = (x - MEAN) / STD
        data = self._data.get("in")
        if data is None:
            data = np.ascontiguousarray(x.transpose(2, 0, 1)[None].astype(np.float32))
            self._data["in"] = data
        else:
            data[...] = x.transpose(2, 0, 1)[None]
        host = acl.util.numpy_to_ptr(data)
        for (ptr, size) in self.in_bufs:
            check(acl.rt.memcpy(ptr, size, host, data.nbytes, ACL_MEMCPY_HOST_TO_DEVICE), "H2D")
        check(acl.mdl.execute_async(self.model_id, self.in_ds, self.out_ds, self.stream), "exec")
        check(acl.rt.synchronize_stream(self.stream), "sync")
        outs = []
        for (ptr, size) in self.out_bufs:
            buf = np.zeros(size, dtype=np.uint8)
            check(acl.rt.memcpy(acl.util.numpy_to_ptr(buf), size, ptr, size, ACL_MEMCPY_DEVICE_TO_HOST), "D2H")
            outs.append(np.frombuffer(buf.tobytes(), dtype=np.float32)[: size // 4].copy())
        return outs[0].reshape(self.h, self.w)

    def close(self):
        acl.mdl.destroy_dataset(self.in_ds)
        acl.mdl.destroy_dataset(self.out_ds)
        acl.mdl.unload(self.model_id)
        acl.mdl.destroy_desc(self.desc)
        acl.rt.destroy_stream(self.stream)


def fill_pyr(d, mask, levels=4):
    d = d.astype(np.float32).copy()
    v = (~mask).astype(np.float32)
    d[~np.isfinite(d)] = 0
    d *= v
    pd, pv = [d], [v]
    for _ in range(levels):
        h, w = pd[-1].shape
        pd.append(cv2.resize(pd[-1], (max(1, w // 2), max(1, h // 2)), interpolation=cv2.INTER_AREA) * 4)
        pv.append(cv2.resize(pv[-1], (max(1, w // 2), max(1, h // 2)), interpolation=cv2.INTER_AREA) * 4)
    est = np.where(pv[-1] > 1e-6, pd[-1] / np.maximum(pv[-1], 1e-6), 0).astype(np.float32)
    for lvl in range(len(pd) - 2, -1, -1):
        est = cv2.resize(est, (pd[lvl].shape[1], pd[lvl].shape[0]), interpolation=cv2.INTER_LINEAR)
        w = pv[lvl]
        est = np.where(w > 1e-6, (pd[lvl] / np.maximum(w, 1e-6)) * 0.5 + est * 0.5, est)
    pos = est[est > 0]
    if pos.size:
        est = np.where(est > 0, est, float(np.median(pos)))
    return est


def pct(lat):
    lat = sorted(lat)
    return dict(p50=round(statistics.median(lat), 2), p95=round(lat[int(0.95 * (len(lat) - 1))], 2),
                fps=round(1000 / statistics.median(lat), 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--iters", type=int, default=30)
    args = ap.parse_args()

    dep = np.asarray(Image.open(args.data + "/depth16.png"), dtype=np.float32) / 1000.0
    valid = dep > 0
    rgb = np.asarray(Image.open(args.data + "/color.jpg").convert("RGB"))
    H, W = dep.shape
    rgb_full = np.asarray(Image.fromarray(rgb).resize((W, H), Image.BILINEAR))
    g = np.clip(np.where(valid, dep, 1.0) / 7.0 * 255.0, 0, 255).astype(np.uint8)
    render = np.stack([g, g, g], -1)
    render[~valid] = 0

    check(acl.init(), "acl.init")
    check(acl.rt.set_device(0), "set_device")
    ctx, ret = acl.rt.create_context(0)
    check(ret, "ctx")

    out = {"data": args.data.split("/")[-1], "iters": args.iters,
           "raw_valid": round(float(valid.mean()), 4), "cpu_affinity": sorted(__import__("os").sched_getaffinity(0))}

    # ---- CPU branch alone (detail-preserving fill) ----
    lat_fill = []
    for _ in range(args.iters):
        t = time.perf_counter()
        fill_pyr(dep, ~valid)
        lat_fill.append((time.perf_counter() - t) * 1e3)
    out["cpu_fill_alone"] = pct(lat_fill)

    # ---- NPU branch alone, both resolutions ----
    for tag, om, h, w in [("168x224", "models/dav2_168x224.om", 168, 224),
                          ("238x308", "models/dav2_238x308.om", 238, 308)]:
        m = Om(om, h, w)
        for _ in range(5):
            m.infer(render)
        lat = []
        for _ in range(args.iters):
            t = time.perf_counter()
            m.infer(render)
            lat.append((time.perf_counter() - t) * 1e3)
        out[f"npu_{tag}_alone"] = pct(lat)

        # ---- parallel: CPU fill branch + NPU prior branch, 1 frame per barrier ----
        barrier = threading.Barrier(2)
        lat_par, lat_cpu, lat_npu = [], [], []
        stop = threading.Event()

        def cpu_worker():
            for _ in range(args.iters):
                barrier.wait()
                t = time.perf_counter()
                fill_pyr(dep, ~valid)
                lat_cpu.append((time.perf_counter() - t) * 1e3)
                barrier.wait()

        def npu_worker():
            for _ in range(args.iters):
                barrier.wait()
                t = time.perf_counter()
                m.infer(render)
                lat_npu.append((time.perf_counter() - t) * 1e3)
                barrier.wait()

        th1 = threading.Thread(target=cpu_worker)
        th2 = threading.Thread(target=npu_worker)
        th1.start(); th2.start()
        for _ in range(args.iters):
            t = time.perf_counter()
            barrier.wait()
            barrier.wait()
            lat_par.append((time.perf_counter() - t) * 1e3)
        th1.join(); th2.join()
        out[f"parallel_{tag}"] = dict(per_frame_wall=pct(lat_par), cpu_branch=pct(lat_cpu), npu_branch=pct(lat_npu))
        m.close()

    print("RESULT " + json.dumps(out))
    json.dump(out, open("results/live/bench_parallel.json", "w"), indent=1)
    acl.rt.destroy_context(ctx)
    acl.rt.reset_device(0)
    acl.finalize()


if __name__ == "__main__":
    main()
