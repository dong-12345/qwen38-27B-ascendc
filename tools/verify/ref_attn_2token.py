"""ref_attn_2token.py — 参考：源层 3（全注意力）2 token（846→198）pos=1 的 post-attn 输出。
包含 rope（rotate_half 配对 (i, i+32)）、KV cache、softmax。
"""
import os
import numpy as np

BASE = "converted/qwen38_310b"
L = "model/language_model/layers/3/"
K = 5120
QH, KVH, DH = 24, 4, 256
RD = 64
theta = 1e7
kq, ng, sbp = K // 4, K // 32, 160


def f16(x):
    return np.asarray(x).astype(np.float16).astype(np.float32)


def lf(rel, n=None):
    a = np.fromfile(os.path.join(BASE, rel), dtype=np.float16).astype(np.float32)
    return a.reshape(n, -1) if n else a


def deq(base_rel, n, k):
    kq_, ng_ = k // 4, k // 32
    sbp_ = (ng_ + 15) // 16 * 16
    with open(os.path.join(BASE, base_rel + ".w4.bin"), "rb") as f:
        words = np.frombuffer(f.read(n * kq_ * 2), dtype=np.uint16).reshape(n, kq_)
    with open(os.path.join(BASE, base_rel + ".s.bin"), "rb") as f:
        s = np.frombuffer(f.read(n * sbp_ * 2), dtype=np.float16).astype(np.float32).reshape(n, sbp_)
    with open(os.path.join(BASE, base_rel + ".sz.bin"), "rb") as f:
        sz = np.frombuffer(f.read(n * sbp_ * 2), dtype=np.float16).astype(np.float32).reshape(n, sbp_)
    q = np.empty((n, k), dtype=np.float32)
    for p in range(4):
        q[:, p * kq_:(p + 1) * kq_] = ((words >> (4 * p)) & 0xF).astype(np.float32)
    return q * np.repeat(s[:, :ng_], 32, axis=1) - np.repeat(sz[:, :ng_], 32, axis=1)


def embed(tok):
    with open(os.path.join(BASE, "model/language_model/embed_tokens/weight.w4.bin"), "rb") as f:
        f.seek(tok * kq * 2)
        words = np.frombuffer(f.read(kq * 2), dtype=np.uint16)
    with open(os.path.join(BASE, "model/language_model/embed_tokens/weight.s.bin"), "rb") as f:
        f.seek(tok * sbp * 2)
        s = np.frombuffer(f.read(sbp * 2), dtype=np.float16).astype(np.float32)
    with open(os.path.join(BASE, "model/language_model/embed_tokens/weight.sz.bin"), "rb") as f:
        f.seek(tok * sbp * 2)
        sz = np.frombuffer(f.read(sbp * 2), dtype=np.float16).astype(np.float32)
    qq = np.empty(K, dtype=np.float32)
    for p in range(4):
        qq[p * kq:(p + 1) * kq] = ((words >> (4 * p)) & 0xF).astype(np.float32)
    return f16(qq * np.repeat(s[:ng], 32) - np.repeat(sz[:ng], 32))


def rope(vec, pos):
    """vec: [nhead, DH]，对前 RD 维做 rotate_half"""
    rh = RD // 2
    out = vec.copy()
    inv = np.array([1.0 / (theta ** (2 * i / RD)) for i in range(rh)], np.float64)
    ang = pos * inv
    c, s = np.cos(ang), np.sin(ang)
    x = vec.astype(np.float64)
    o = out.astype(np.float64)
    o[:, :rh] = x[:, :rh] * c - x[:, rh:RD] * s
    o[:, rh:RD] = x[:, rh:RD] * c + x[:, :rh] * s
    return o.astype(np.float32)


def main():
    toks = [846, 198]
    ln1 = lf(L + "input_layernorm/weight.f16.bin")
    Wq = deq(L + "self_attn/q_proj", 12288, K)
    Wk = deq(L + "self_attn/k_proj", 1024, K)
    Wv = deq(L + "self_attn/v_proj", 1024, K)
    Wout = deq(L + "self_attn/o_proj", 5120, 6144)
    qn_w = lf(L + "self_attn/q_norm/weight.f16.bin").reshape(-1)
    kn_w = lf(L + "self_attn/k_norm/weight.f16.bin").reshape(-1)
    silu = lambda t: t / (1.0 + np.exp(-t))

    kc = np.zeros((KVH, 16, DH), np.float32)
    vc = np.zeros((KVH, 16, DH), np.float32)
    cache = []
    for pos, tok in enumerate(toks):
        x = embed(tok)
        xn = f16(x / np.sqrt((x.astype(np.float64) ** 2).mean() + 1e-6) * ln1)
        qg = f16(Wq.astype(np.float64) @ xn.astype(np.float64))
        kk = f16(Wk.astype(np.float64) @ xn.astype(np.float64))
        vv = f16(Wv.astype(np.float64) @ xn.astype(np.float64))
        q = qg[:QH * DH].reshape(QH, DH)
        gate = qg[QH * DH:].reshape(QH, DH)
        kk = kk.reshape(KVH, DH)
        vv = vv.reshape(KVH, DH)
        # qk-norm
        for h in range(QH):
            s = q[h].astype(np.float64)
            q[h] = f16(s / np.sqrt((s ** 2).mean() + 1e-6) * qn_w)
        for h in range(KVH):
            s = kk[h].astype(np.float64)
            kk[h] = f16(s / np.sqrt((s ** 2).mean() + 1e-6) * kn_w)
        # rope
        q = rope(q, pos)
        kk = rope(kk, pos)
        kc[:, pos] = kk
        vc[:, pos] = vv
        cache.append((pos, x))
    # 最后一 token (pos=1) 的 attention 输出
    pos = 1
    x = cache[pos][1]
    xn = f16(x / np.sqrt((x.astype(np.float64) ** 2).mean() + 1e-6) * ln1)
    qg = f16(Wq.astype(np.float64) @ xn.astype(np.float64))
    kk = f16(Wk.astype(np.float64) @ xn.astype(np.float64))
    vv = f16(Wv.astype(np.float64) @ xn.astype(np.float64))
    q = qg[:QH * DH].reshape(QH, DH)
    gate = qg[QH * DH:].reshape(QH, DH)
    kk = kk.reshape(KVH, DH)
    vv = vv.reshape(KVH, DH)
    for h in range(QH):
        s = q[h].astype(np.float64)
        q[h] = f16(s / np.sqrt((s ** 2).mean() + 1e-6) * qn_w)
    for h in range(KVH):
        s = kk[h].astype(np.float64)
        kk[h] = f16(s / np.sqrt((s ** 2).mean() + 1e-6) * kn_w)
    q = rope(q, pos)
    kk = rope(kk, pos)
    kc[:, pos] = kk
    vc[:, pos] = vv

    out = np.zeros((QH, DH), np.float32)
    scale = 1.0 / 16.0
    for h in range(QH):
        hkv = h // (QH // KVH)
        nkeys = pos + 1
        ks = kc[hkv, :nkeys].astype(np.float64)
        scores = (ks @ q[h].astype(np.float64)) * scale
        m = scores.max()
        p = np.exp(scores - m)
        p /= p.sum()
        o = p @ vc[hkv, :nkeys].astype(np.float64)
        out[h] = f16(o * silu(gate[h].astype(np.float64)))
    oo = f16(Wout.astype(np.float64) @ out.reshape(-1).astype(np.float64))
    post = f16(x + oo)
    post.astype(np.float16).tofile("tmp/ref_attn3_pos1.bin")
    print("ref attn3 pos1[:6] =", np.round(post[:6], 4).tolist())
    print("ref stats: mean=%.5f std=%.5f" % (post.mean(), post.std()))


if __name__ == "__main__":
    main()
