"""check_conversion.py — 用原始 AWQ safetensors 独立反量化，和我们的转换产物对比。
若不一致 → 权重转换有 bug（这是「模型输出乱码」的最后一个可疑环节）。
"""
import os
import numpy as np
import torch
from safetensors import safe_open

AWQ = "models/Qwen3.8-27B-AWQ-INT4"
CONV = "converted/qwen38_310b"
LAYER = 0
TENSOR = "linear_attn.in_proj_qkv"
TENSOR_PATH = "linear_attn/in_proj_qkv"
N, K = 10240, 5120
GROUP = 32
NG = K // GROUP


def load_awq():
    idx_path = os.path.join(AWQ, "model.safetensors.index.json")
    import json
    idx = json.load(open(idx_path))
    pre = "model.language_model.layers.%d.%s." % (LAYER, TENSOR)
    key = pre + "weight_packed"
    shard = idx["weight_map"][key]
    out = {}
    with safe_open(os.path.join(AWQ, shard), framework="pt") as f:
        for name in ("weight_packed", "weight_scale", "weight_zero_point"):
            out[name] = f.get_tensor(pre + name)
    return out


def main():
    t = load_awq()
    wq = t["weight_packed"].to(torch.int64).numpy().astype(np.int64)
    s = t["weight_scale"].to(torch.float32).numpy()
    z = t["weight_zero_point"]
    print("packed", wq.shape, "| scale", s.shape, s.dtype, "| zp", tuple(z.shape), z.dtype)
    zp = z.to(torch.int64).numpy().astype(np.int64) if z.dtype in (torch.int32, torch.int64) else None
    if zp is None:
        # zp 可能是打包成 int32 的 4bit：用 GPTQ 的沿输出行打包假设
        zraw = z.to(torch.int32).numpy().astype(np.int64) if z.dtype == torch.float32 else None
        print("zp dtype unusual:", z.dtype)

    # 1) 反量化：q[:, i::8] = (packed >> 4i) & 0xF
    Q = np.empty((N, K), dtype=np.uint8)
    for i in range(8):
        Q[:, i::8] = ((wq >> (4 * i)) & 0xF).astype(np.uint8)
    if zp is not None and zp.shape[0] * 8 >= N and zp.shape[1] == NG:
        Z = np.empty((N, NG), dtype=np.uint8)
        for i in range(8):
            Z[i::8] = ((zp >> (4 * i)) & 0xF).astype(np.uint8)
        Z = Z[:N]
    else:
        Z = np.zeros((N, NG), dtype=np.uint8)
        print("WARN: zp shape not as expected, assume symmetric")
    w_awq = (Q.astype(np.float32) - np.repeat(Z.astype(np.float32), GROUP, axis=1)) * np.repeat(s.astype(np.float32), GROUP, axis=1)

    # 2) 我们的转换产物反量化
    kq = K // 4
    words = np.fromfile(os.path.join(CONV, "model/language_model/layers/%d/%s.w4.bin" % (LAYER, TENSOR_PATH)), dtype=np.uint16).reshape(N, kq)
    sc = np.fromfile(os.path.join(CONV, "model/language_model/layers/%d/%s.s.bin" % (LAYER, TENSOR_PATH)), dtype=np.float16).astype(np.float32)
    sz = np.fromfile(os.path.join(CONV, "model/language_model/layers/%d/%s.sz.bin" % (LAYER, TENSOR_PATH)), dtype=np.float16).astype(np.float32)
    sbp = sc.size // N
    sc = sc.reshape(N, sbp)
    sz = sz.reshape(N, sbp)
    Qc = np.empty((N, K), dtype=np.float32)
    for p in range(4):
        Qc[:, p * kq:(p + 1) * kq] = ((words >> (4 * p)) & 0xF).astype(np.float32)
    w_conv = Qc * np.repeat(sc[:, :NG], GROUP, axis=1) - np.repeat(sz[:, :NG], GROUP, axis=1)

    d = np.abs(w_awq - w_conv)
    print("weight diff: max=%.6f mean=%.8f (|w| mean=%.6f)" % (d.max(), d.mean(), np.abs(w_awq).mean()))
    # 3) 反量化一致时，对同一输入的点积也应一致
    rng = np.random.RandomState(0)
    x = rng.randn(K).astype(np.float32)
    a = w_awq.astype(np.float64) @ x.astype(np.float64)
    b = w_conv.astype(np.float64) @ x.astype(np.float64)
    print("dot: maxdiff=%.6f corr=%.8f" % (np.abs(a - b).max(), np.corrcoef(a, b)[0, 1]))
    print("awq[:4] =", np.round(a[:4], 4).tolist())
    print("conv[:4]=", np.round(b[:4], 4).tolist())


if __name__ == "__main__":
    main()
