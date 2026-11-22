#!/usr/bin/env python3
"""Generic Ascend ACL (pyACL) .om runner with latency measurement.

Usage:
  python3 acl_runner.py --model m.om --input in.bin --shape 1,3,238,308 --iters 50 --warmup 10
  python3 acl_runner.py --model m.om --input in.bin --shape 1,3,238,308 --dump out.bin --bin-out

Input/output are raw little-endian float16 (default) or float32 (--fp32).
Prints JSON with p50/p95/p99 latency (NPU execute+sync) and copies.
"""
import argparse
import json
import os
import statistics
import sys
import time

import numpy as np

sys.path.insert(0, "/usr/local/Ascend/ascend-toolkit/latest/pyACL/python/site-packages")
import acl  # noqa: E402

ACL_MEM_MALLOC_NORMAL_ONLY = 0
ACL_MEMCPY_HOST_TO_DEVICE = 1
ACL_MEMCPY_DEVICE_TO_HOST = 2


def check(ret, msg):
    if ret != 0:
        raise RuntimeError(f"{msg} failed ret={ret}")


class OmModel:
    def __init__(self, path):
        check(acl.init(), "acl.init")
        check(acl.rt.set_device(0), "set_device")
        self.context, ret = acl.rt.create_context(0)
        check(ret, "create_context")
        self.stream, ret = acl.rt.create_stream()
        check(ret, "create_stream")
        self.model_id, ret = acl.mdl.load_from_file(path)
        check(ret, "load_from_file")
        self.desc = acl.mdl.create_desc()
        check(acl.mdl.get_desc(self.desc, self.model_id), "get_desc")
        self.n_in = acl.mdl.get_num_inputs(self.desc)
        self.n_out = acl.mdl.get_num_outputs(self.desc)
        self.in_ds = acl.mdl.create_dataset()
        self.in_bufs = []
        for i in range(self.n_in):
            size = acl.mdl.get_input_size_by_index(self.desc, i)
            ptr, ret = acl.rt.malloc(size, ACL_MEM_MALLOC_NORMAL_ONLY)
            check(ret, "malloc in")
            buf = acl.create_data_buffer(ptr, size)
            _, ret = acl.mdl.add_dataset_buffer(self.in_ds, buf)
            check(ret, "add in buf")
            self.in_bufs.append((ptr, size, buf))
        self.out_ds = acl.mdl.create_dataset()
        self.out_bufs = []
        for i in range(self.n_out):
            size = acl.mdl.get_output_size_by_index(self.desc, i)
            ptr, ret = acl.rt.malloc(size, ACL_MEM_MALLOC_NORMAL_ONLY)
            check(ret, "malloc out")
            buf = acl.create_data_buffer(ptr, size)
            _, ret = acl.mdl.add_dataset_buffer(self.out_ds, buf)
            check(ret, "add out buf")
            self.out_bufs.append((ptr, size, buf))

    def copy_in(self, arrays):
        for (ptr, size, _), arr in zip(self.in_bufs, arrays):
            data = np.ascontiguousarray(arr)
            assert data.nbytes <= size, f"input {data.nbytes} > {size}"
            host = acl.util.numpy_to_ptr(data)
            check(acl.rt.memcpy(ptr, size, host, data.nbytes, ACL_MEMCPY_HOST_TO_DEVICE), "memcpy H2D")

    def execute(self):
        check(acl.mdl.execute_async(self.model_id, self.in_ds, self.out_ds, self.stream), "execute_async")
        check(acl.rt.synchronize_stream(self.stream), "sync_stream")

    def read_out(self):
        outs = []
        for ptr, size, _ in self.out_bufs:
            buf = np.zeros(size, dtype=np.uint8)
            check(acl.rt.memcpy(acl.util.numpy_to_ptr(buf), size, ptr, size, ACL_MEMCPY_DEVICE_TO_HOST), "memcpy D2H")
            outs.append(buf)
        return outs

    def close(self):
        acl.mdl.destroy_dataset(self.in_ds)
        acl.mdl.destroy_dataset(self.out_ds)
        acl.mdl.unload(self.model_id)
        acl.mdl.destroy_desc(self.desc)
        acl.rt.destroy_stream(self.stream)
        acl.rt.destroy_context(self.context)
        acl.rt.reset_device(0)
        acl.finalize()


def host_to_np(buf, dtype):
    return np.frombuffer(buf.tobytes(), dtype=dtype)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--shape", required=True, help="e.g. 1,3,238,308")
    ap.add_argument("--dtype", default="float32", choices=["float16", "float32"],
                    help="I/O dtype of the OM (ATC keeps ONNX fp32 I/O unless input_fp16_nodes is used)")
    ap.add_argument("--input", default=None, help="raw input bin; random if omitted")
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--dump", default=None, help="dump first output to this .npy")
    ap.add_argument("--out-shape", default=None, help="comma list for first output reshape")
    args = ap.parse_args()

    shape = [int(v) for v in args.shape.split(",")]
    dt = np.float16 if args.dtype == "float16" else np.float32
    n = int(np.prod(shape))
    if args.input:
        x = np.fromfile(args.input, dtype=dt)[:n].reshape(shape)
    else:
        x = np.random.randn(*shape).astype(dt)

    m = OmModel(args.model)
    m.copy_in([x])
    for _ in range(args.warmup):
        m.execute()
    lat = []
    for _ in range(args.iters):
        t0 = time.perf_counter()
        m.execute()
        lat.append((time.perf_counter() - t0) * 1000.0)
    lat_sorted = sorted(lat)
    res = dict(
        model=os.path.basename(args.model), shape=shape, iters=args.iters,
        p50_ms=round(statistics.median(lat), 3),
        p95_ms=round(lat_sorted[int(0.95 * (len(lat) - 1))], 3),
        p99_ms=round(lat_sorted[int(0.99 * (len(lat) - 1))], 3),
        mean_ms=round(statistics.mean(lat), 3),
        min_ms=round(min(lat), 3), max_ms=round(max(lat), 3),
        fps=round(1000.0 / statistics.median(lat), 2),
    )
    outs = m.read_out()
    if args.dump:
        buf = outs[0]
        size = int(buf.size)
        arr = host_to_np(buf, dt)
        if args.out_shape:
            arr = arr.reshape([int(v) for v in args.out_shape.split(",")])
        np.save(args.dump, arr)
        res["out0_bytes"] = size
        res["out0_stats"] = dict(min=float(arr.min()), max=float(arr.max()), mean=float(arr.mean()))
    print("RESULT " + json.dumps(res))
    m.close()


if __name__ == "__main__":
    main()
