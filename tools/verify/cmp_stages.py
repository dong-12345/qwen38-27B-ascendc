"""cmp_stages.py — 逐级对比板端 dump 与 PC numpy 参考（源层 10，token 846，pos 0）"""
import os
import numpy as np

BASE = "converted/qwen38_310b"
K, HID = 5120, 5120
KH, VH, HD = 16, 48, 128


def deq_mat(base_rel, n, k):
    kq, ng = k // 4, k // 32
    sbp = (ng + 15) // 16 * 16
    with open(os.path.join(BASE, base_rel + ".w4.bin"), "rb") as f:
        words = np.frombuffer(f.read(n * kq * 2), dtype=np.uint16).reshape(n, kq)
    with open(os.path.join(BASE, base_rel + ".s.bin"), "rb") as f:
        s = np.frombuffer(f.read(n * sbp * 2), dtype=np.float16).astype(np.float32).reshape(n, sbp)
    with open(os.path.join(BASE, base_rel + ".sz.bin"), "rb") as f:
        sz = np.frombuffer(f.read(n * sbp * 2), dtype=np.float16).astype(np.float32).reshape(n, sbp)
    q = np.empty((n, k), dtype=np.float32)
    for p in range(4):
        q[:, p * kq:(p + 1) * kq] = ((words >> (4 * p)) & 0xF).astype(np.float32)
    return q * np.repeat(s[:, :ng], 32, axis=1) - np.repeat(sz[:, :ng], 32, axis=1)



def deq_rows(base_rel, rows, k):
    kq, ng = k // 4, k // 32
    sbp = (ng + 15) // 16 * 16
    out = []
    fw = open(os.path.join(BASE, base_rel + ".w4.bin"), "rb")
    fs = open(os.path.join(BASE, base_rel + ".s.bin"), "rb")
    fz = open(os.path.join(BASE, base_rel + ".sz.bin"), "rb")
    for r in rows:
        fw.seek(r * kq * 2)
        words = np.frombuffer(fw.read(kq * 2), dtype=np.uint16)
        fs.seek(r * sbp * 2)
        s = np.frombuffer(fs.read(sbp * 2), dtype=np.float16).astype(np.float32)
        fz.seek(r * sbp * 2)
        sz = np.frombuffer(fz.read(sbp * 2), dtype=np.float16).astype(np.float32)
        q = np.empty(k, dtype=np.float32)
        for p2 in range(4):
            q[p2 * kq:(p2 + 1) * kq] = ((words >> (4 * p2)) & 0xF).astype(np.float32)
        out.append(q * np.repeat(s[:ng], 32) - np.repeat(sz[:ng], 32))
    fw.close(); fs.close(); fz.close()
    return np.array(out)

def f16(x):
    return np.asarray(x).astype(np.float16).astype(np.float32)


def rmsnorm(x, w, eps=1e-6):
    return f16(x / np.sqrt((x.astype(np.float64) ** 2).mean() + eps) * w)


def lf(rel, n=None):
    a = np.fromfile(os.path.join(BASE, rel), dtype=np.float16).astype(np.float32)
    return a.reshape(n, -1) if n else a


def cmp(name, eng, ref):
    eng = eng.astype(np.float64)
    ref = ref.astype(np.float64)
    d = np.abs(eng - ref)
    denom = np.abs(ref).mean() + 1e-12
    # 相关系数
    c = np.corrcoef(eng, ref)[0, 1] if eng.std() > 0 and ref.std() > 0 else float("nan")
    print("%-14s max=%.4f mean=%.5f  rel_mean=%.2f%%  corr=%.4f  eng[0:4]=%s ref[0:4]=%s"
          % (name, d.max(), d.mean(), 100 * d.mean() / denom, c,
             np.round(eng[:4], 4).tolist(), np.round(ref[:4], 4).tolist()))


def main():
    li = 10
    L = "model/language_model/layers/%d/" % li
    tok = 846
    x = f16(deq_rows("model/language_model/embed_tokens/weight", [846], K)[0])
    ln1 = lf(L + "input_layernorm/weight.f16.bin")
    xn = rmsnorm(x, ln1)
    qkv = f16(deq_mat(L + "linear_attn/in_proj_qkv", 10240, K) @ xn)
    z = f16(deq_mat(L + "linear_attn/in_proj_z", 6144, K) @ xn)
    a = f16(lf(L + "linear_attn/in_proj_a/weight.f16.bin", 48) @ xn)
    b = f16(lf(L + "linear_attn/in_proj_b/weight.f16.bin", 48) @ xn)

    eng_xn = np.fromfile("tmp/eng_xn.bin", dtype=np.float16).astype(np.float32)
    cmp("xn", eng_xn, xn)
    eng_qkv = np.fromfile("tmp/eng_qkv.bin", dtype=np.float16).astype(np.float32)
    cmp("qkv", eng_qkv, qkv)
    eng_z = np.fromfile("tmp/eng_z.bin", dtype=np.float16).astype(np.float32)
    cmp("z", eng_z, z)
    eng_ab = np.fromfile("tmp/eng_ab.bin", dtype=np.float16).astype(np.float32)
    cmp("a", eng_ab[:48], a)
    cmp("b", eng_ab[48:96], b)

    # conv (zero state) + silu
    cw = lf(L + "linear_attn/conv1d/weight.f16.bin").reshape(10240, 4)
    silu = lambda t: t / (1.0 + np.exp(-t))
    qc = silu(f16(qkv[:2048] * cw[:2048, 3]))
    kc = silu(f16(qkv[2048:4096] * cw[2048:4096, 3]))
    vc = silu(f16(qkv[4096:] * cw[4096:, 3]))
    cmp("convq", np.fromfile("tmp/eng_convq.bin", dtype=np.float16).astype(np.float32), qc)
    cmp("convk", np.fromfile("tmp/eng_convk.bin", dtype=np.float16).astype(np.float32), kc)
    cmp("convv", np.fromfile("tmp/eng_convv.bin", dtype=np.float16).astype(np.float32), vc)


if __name__ == "__main__":
    main()
