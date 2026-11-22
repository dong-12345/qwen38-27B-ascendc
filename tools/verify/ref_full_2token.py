"""ref_full_2token.py — PC 全 64 层参考，2 token（846, 198），带状态演进。
与板端引擎 pos=1 的 logits top-5 对比。
"""
import os
import sys
import numpy as np

BASE = "converted/qwen38_310b"
H = 5120
KH, VH, HD = 16, 48, 128
QH, KVH, DH = 24, 4, 256
FF = 17408
RD = 64
theta = 1e7
NLAYER = 64
MAXT = 16
EPS = 1e-6
silu = lambda t: t / (1.0 + np.exp(-t))


def f16(x):
    return np.asarray(x).astype(np.float16).astype(np.float32)


def lf(rel, n=None):
    a = np.fromfile(os.path.join(BASE, rel), dtype=np.float16).astype(np.float32)
    return a.reshape(n, -1) if n else a


def deq(base_rel, n, k):
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
    w = q * np.repeat(s[:, :ng], 32, axis=1) - np.repeat(sz[:, :ng], 32, axis=1)
    del q
    return w


def gemv(w, x):
    return f16(w.astype(np.float64) @ x.astype(np.float64))


def rmsnorm(x, wgt):
    return f16(x / np.sqrt((x.astype(np.float64) ** 2).mean() + EPS) * wgt)


def embed(tok):
    kq, ng, sbp = H // 4, H // 32, 160
    with open(os.path.join(BASE, "model/language_model/embed_tokens/weight.w4.bin"), "rb") as f:
        f.seek(tok * kq * 2)
        words = np.frombuffer(f.read(kq * 2), dtype=np.uint16)
    with open(os.path.join(BASE, "model/language_model/embed_tokens/weight.s.bin"), "rb") as f:
        f.seek(tok * sbp * 2)
        s = np.frombuffer(f.read(sbp * 2), dtype=np.float16).astype(np.float32)
    with open(os.path.join(BASE, "model/language_model/embed_tokens/weight.sz.bin"), "rb") as f:
        f.seek(tok * sbp * 2)
        sz = np.frombuffer(f.read(sbp * 2), dtype=np.float16).astype(np.float32)
    q = np.empty(H, dtype=np.float32)
    for p in range(4):
        q[p * kq:(p + 1) * kq] = ((words >> (4 * p)) & 0xF).astype(np.float32)
    return f16(q * np.repeat(s[:ng], 32) - np.repeat(sz[:ng], 32))


def rope(vec, pos):
    rh = RD // 2
    inv = np.array([1.0 / (theta ** (2 * i / RD)) for i in range(rh)], np.float64)
    ang = pos * inv
    c, s = np.cos(ang), np.sin(ang)
    o = vec.astype(np.float64).copy()
    x = vec.astype(np.float64)
    o[:, :rh] = x[:, :rh] * c - x[:, rh:RD] * s
    o[:, rh:RD] = x[:, rh:RD] * c + x[:, :rh] * s
    return o.astype(np.float32)


def mlp(li, x):
    L = "model/language_model/layers/%d/" % li
    ln2 = lf(L + "post_attention_layernorm/weight.f16.bin")
    xn2 = rmsnorm(x, ln2)
    gate = gemv(deq(L + "mlp/gate_proj", FF, H), xn2)
    up = gemv(deq(L + "mlp/up_proj", FF, H), xn2)
    act = f16(f16(silu(gate)) * up)
    down = gemv(deq(L + "mlp/down_proj", H, FF), act)
    return f16(x + down)


def gdn_layer(li, x, st):
    L = "model/language_model/layers/%d/" % li
    ln1 = lf(L + "input_layernorm/weight.f16.bin")
    xn = rmsnorm(x, ln1)
    qkv = gemv(deq(L + "linear_attn/in_proj_qkv", 10240, H), xn)
    z = gemv(deq(L + "linear_attn/in_proj_z", 6144, H), xn)
    a = gemv(lf(L + "linear_attn/in_proj_a/weight.f16.bin", 48), xn)
    b = gemv(lf(L + "linear_attn/in_proj_b/weight.f16.bin", 48), xn)
    cw = lf(L + "linear_attn/conv1d/weight.f16.bin").reshape(10240, 4)
    acc = np.zeros(10240, np.float32)
    for j in range(3):
        acc += cw[:, j] * st["conv"][j]
    acc += cw[:, 3] * qkv
    conv = silu(f16(acc))
    st["conv"] = [st["conv"][1], st["conv"][2], qkv.copy()]
    qc, kc, vc = conv[:2048], conv[2048:4096], conv[4096:]
    qn = np.empty(2048, np.float32)
    kn = np.empty(2048, np.float32)
    for h in range(KH):
        s = qc[h * HD:(h + 1) * HD].astype(np.float64)
        qn[h * HD:(h + 1) * HD] = f16(s / np.sqrt((s ** 2).sum() + EPS) / np.sqrt(HD))
        s2 = kc[h * HD:(h + 1) * HD].astype(np.float64)
        kn[h * HD:(h + 1) * HD] = f16(s2 / np.sqrt((s2 ** 2).sum() + EPS))
    alog = lf(L + "linear_attn/A_log.f16.bin").reshape(-1)
    dt = lf(L + "linear_attn/dt_bias.f16.bin").reshape(-1)
    beta = 1 / (1 + np.exp(-b))
    sp = np.where((a + dt) > 20, (a + dt), np.log1p(np.exp(a + dt)))
    decay = np.exp(-np.exp(alog) * sp)
    gnorm = lf(L + "linear_attn/norm/weight.f16.bin").reshape(-1)[:HD]
    o = np.empty(VH * HD, np.float32)
    S = st["S"]
    for vh in range(VH):
        kh = vh // (VH // KH)
        kk = kn[kh * HD:(kh + 1) * HD].astype(np.float64)
        qq2 = qn[kh * HD:(kh + 1) * HD].astype(np.float64)
        vv = vc[vh * HD:(vh + 1) * HD].astype(np.float64)
        Sv = S[vh].astype(np.float64) * decay[vh]
        bv = (vv - kk @ Sv) * beta[vh]
        Sv = Sv + np.outer(kk, bv)
        S[vh] = Sv.astype(np.float32)
        o[vh * HD:(vh + 1) * HD] = f16(Sv.T @ qq2)
    st["S"] = S
    onorm = np.empty(VH * HD, np.float32)
    for vh in range(VH):
        s = o[vh * HD:(vh + 1) * HD].astype(np.float64)
        ms = (s ** 2).mean()
        onorm[vh * HD:(vh + 1) * HD] = f16(f16(s / np.sqrt(ms + EPS) * gnorm) * silu(z[vh * HD:(vh + 1) * HD]))
    ow = os.path.join(BASE, L + "linear_attn/out_proj/weight.f16.bin")
    Wout = lf(L + "linear_attn/out_proj/weight.f16.bin", 5120) if os.path.exists(ow) else deq(L + "linear_attn/out_proj", 5120, 6144)
    x = f16(x + gemv(Wout, onorm))
    return mlp(li, x)


def attn_layer(li, x, st, pos):
    L = "model/language_model/layers/%d/" % li
    ln1 = lf(L + "input_layernorm/weight.f16.bin")
    xn = rmsnorm(x, ln1)
    qg = gemv(deq(L + "self_attn/q_proj", 12288, H), xn)
    kk = gemv(deq(L + "self_attn/k_proj", 1024, H), xn).reshape(KVH, DH)
    vv = gemv(deq(L + "self_attn/v_proj", 1024, H), xn).reshape(KVH, DH)
    qn_w = lf(L + "self_attn/q_norm/weight.f16.bin").reshape(-1)
    kn_w = lf(L + "self_attn/k_norm/weight.f16.bin").reshape(-1)
    q = qg[:QH * DH].reshape(QH, DH)
    gate = qg[QH * DH:].reshape(QH, DH)
    for h in range(QH):
        s = q[h].astype(np.float64)
        q[h] = f16(s / np.sqrt((s ** 2).mean() + EPS) * qn_w)
    for h in range(KVH):
        s = kk[h].astype(np.float64)
        kk[h] = f16(s / np.sqrt((s ** 2).mean() + EPS) * kn_w)
    q = rope(q, pos)
    kk = rope(kk, pos)
    kc, vc = st["kc"], st["vc"]
    kc[:, pos] = kk
    vc[:, pos] = vv
    out = np.empty((QH, DH), np.float32)
    scale = 1.0 / 16.0
    nkeys = pos + 1
    for h in range(QH):
        hkv = h // (QH // KVH)
        scores = (kc[hkv, :nkeys].astype(np.float64) @ q[h].astype(np.float64)) * scale
        p = np.exp(scores - scores.max())
        p /= p.sum()
        o = p @ vc[hkv, :nkeys].astype(np.float64)
        out[h] = f16(o * silu(gate[h].astype(np.float64)))
    Wout = deq(L + "self_attn/o_proj", 5120, 6144)
    x = f16(x + gemv(Wout, out.reshape(-1)))
    return mlp(li, x)


def main():
    toks = [846, 198]
    types = None
    for ln in open(os.path.join(BASE, "meta.txt"), encoding="utf-8"):
        if ln.startswith("CFG layer_types"):
            types = [int(v) for v in ln.split()[2].split(",")]
            break
    st = {}
    for li in range(NLAYER):
        st[li] = {"conv": [np.zeros(10240, np.float32) for _ in range(3)],
                  "S": np.zeros((VH, HD, HD), np.float32),
                  "kc": np.zeros((KVH, MAXT, DH), np.float32),
                  "vc": np.zeros((KVH, MAXT, DH), np.float32)}
    x = None
    for pos, tok in enumerate(toks):
        x = embed(tok)
        for li in range(NLAYER):
            if types[li] == 1:
                x = gdn_layer(li, x, st[li])
            else:
                x = attn_layer(li, x, st[li], pos)
        print("  pos %d done, |x|=%.3f" % (pos, np.abs(x).mean()), flush=True)
    fn = lf("model/language_model/norm/weight.f16.bin")
    xn = rmsnorm(x, fn)
    kq, ng, sbp = H // 4, H // 32, 160
    n = 248320
    best = []
    chunk = 16384
    with open(os.path.join(BASE, "lm_head/weight.w4.bin"), "rb") as fw, \
         open(os.path.join(BASE, "lm_head/weight.s.bin"), "rb") as fs, \
         open(os.path.join(BASE, "lm_head/weight.sz.bin"), "rb") as fz:
        for r0 in range(0, n, chunk):
            r1 = min(n, r0 + chunk)
            words = np.frombuffer(fw.read((r1 - r0) * kq * 2), dtype=np.uint16).reshape(r1 - r0, kq)
            s = np.frombuffer(fs.read((r1 - r0) * sbp * 2), dtype=np.float16).astype(np.float32).reshape(r1 - r0, sbp)
            sz = np.frombuffer(fz.read((r1 - r0) * sbp * 2), dtype=np.float16).astype(np.float32).reshape(r1 - r0, sbp)
            q = np.empty((r1 - r0, H), dtype=np.float32)
            for p in range(4):
                q[:, p * kq:(p + 1) * kq] = ((words >> (4 * p)) & 0xF).astype(np.float32)
            lg = (q * np.repeat(s[:, :ng], 32, axis=1)) @ xn - (np.repeat(sz[:, :ng], 32, axis=1) @ xn)
            for i in np.argsort(-lg)[:5]:
                best.append((float(lg[i]), int(r0 + i)))
    best.sort(reverse=True)
    print("REF(pos1) top5:", [(t, round(v, 3)) for v, t in best[:5]])


if __name__ == "__main__":
    main()
