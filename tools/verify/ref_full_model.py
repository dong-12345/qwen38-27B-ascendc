"""ref_full_model.py — PC 全 64 层参考（单 token pos=0，numpy，流式读盘），
用于判断「模型+权重转换」是否正确（与板端引擎的 logits/top-5 对比）。
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
EPS = 1e-6


def f16(x):
    return np.asarray(x).astype(np.float16).astype(np.float32)


def lf(rel, n=None):
    a = np.fromfile(os.path.join(BASE, rel), dtype=np.float16).astype(np.float32)
    return a.reshape(n, -1) if n else a


_cache = {}


def deq(base_rel, n, k):
    """反量化 w4 矩阵（带缓存，避免重复读盘）"""
    key = (base_rel, n, k)
    if key in _cache:
        return _cache[key]
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


silu = lambda t: t / (1.0 + np.exp(-t))


def gdn_layer(li, x):
    L = "model/language_model/layers/%d/" % li
    ln1 = lf(L + "input_layernorm/weight.f16.bin")
    xn = rmsnorm(x, ln1)
    qkv = gemv(deq(L + "linear_attn/in_proj_qkv", 10240, H), xn)
    z = gemv(deq(L + "linear_attn/in_proj_z", 6144, H), xn)
    a = gemv(lf(L + "linear_attn/in_proj_a/weight.f16.bin", 48), xn)
    b = gemv(lf(L + "linear_attn/in_proj_b/weight.f16.bin", 48), xn)
    cw = lf(L + "linear_attn/conv1d/weight.f16.bin").reshape(10240, 4)
    conv = silu(f16(qkv * cw[:, 3]))
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
    for vh in range(VH):
        kh = vh // (VH // KH)
        kk = kn[kh * HD:(kh + 1) * HD].astype(np.float64)
        qq2 = qn[kh * HD:(kh + 1) * HD].astype(np.float64)
        vv = vc[vh * HD:(vh + 1) * HD].astype(np.float64)
        bv = beta[vh] * vv
        o[vh * HD:(vh + 1) * HD] = f16(np.outer(kk, bv).T @ qq2)
    onorm = np.empty(VH * HD, np.float32)
    for vh in range(VH):
        s = o[vh * HD:(vh + 1) * HD].astype(np.float64)
        ms = (s ** 2).mean()
        onorm[vh * HD:(vh + 1) * HD] = f16(f16(s / np.sqrt(ms + EPS) * gnorm) * silu(z[vh * HD:(vh + 1) * HD]))
    ow = os.path.join(BASE, L + "linear_attn/out_proj/weight.f16.bin")
    Wout = lf(L + "linear_attn/out_proj/weight.f16.bin", 5120) if os.path.exists(ow) else deq(L + "linear_attn/out_proj", 5120, 6144)
    x = f16(x + gemv(Wout, onorm))
    return mlp(li, x)


def attn_layer(li, x):
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
    # pos=0 → rope 恒等，attention out = v（单 key）
    out = np.empty((QH, DH), np.float32)
    for h in range(QH):
        kvh = h // (QH // KVH)
        out[h] = f16(vv[kvh].astype(np.float64) * silu(gate[h].astype(np.float64)))
    Wout = deq(L + "self_attn/o_proj", 5120, 6144)
    x = f16(x + gemv(Wout, out.reshape(-1)))
    return mlp(li, x)


def mlp(li, x):
    L = "model/language_model/layers/%d/" % li
    ln2 = lf(L + "post_attention_layernorm/weight.f16.bin")
    xn2 = rmsnorm(x, ln2)
    gate = gemv(deq(L + "mlp/gate_proj", FF, H), xn2)
    up = gemv(deq(L + "mlp/up_proj", FF, H), xn2)
    act = f16(f16(silu(gate)) * up)
    down = gemv(deq(L + "mlp/down_proj", H, FF), act)
    return f16(x + down)


def main():
    tok = int(sys.argv[1]) if len(sys.argv) > 1 else 846
    x = embed(tok)
    types = None
    for ln in open(os.path.join(BASE, "meta.txt"), encoding="utf-8"):
        if ln.startswith("CFG layer_types"):
            types = [int(v) for v in ln.split()[2].split(",")]
            break
    for li in range(NLAYER):
        if types[li] == 1:
            x = gdn_layer(li, x)
        else:
            x = attn_layer(li, x)
        if li % 8 == 0 or li == NLAYER - 1:
            print("  layer %d done, |x|=%.3f" % (li, np.abs(x).mean()), flush=True)
    fn = lf("model/language_model/norm/weight.f16.bin")
    xn = rmsnorm(x, fn)
    # lm_head
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
            idx = np.argsort(-lg)[:5]
            for i in idx:
                best.append((float(lg[i]), int(r0 + i)))
    best.sort(reverse=True)
    print("REF top5:", [(t, round(v, 3)) for v, t in best[:5]])
    print("REF stats: len(x)=%d mean=%.5f std=%.5f" % (len(x), x.mean(), x.std()))


if __name__ == "__main__":
    main()
