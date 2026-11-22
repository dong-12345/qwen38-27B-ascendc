"""ref_attn_layer.py — 参考：源层 3（全注意力）单 token (pos 0) 前向，与板端 dump 对比。
pos 0 时 attention 退化为 out_h = v_{kv(h)}，rope 为恒等。
"""
import os
import numpy as np

BASE = "converted/qwen38_310b"
L = "model/language_model/layers/3/"
K = 5120
HID = 5120
QH, KVH, DH = 24, 4, 256
kq, ng = K // 4, K // 32
sbp = 160


def deq(base_rel, n, k):
    kq = k // 4
    ng = k // 32
    sbp_l = (ng + 15) // 16 * 16
    with open(os.path.join(BASE, base_rel + ".w4.bin"), "rb") as f:
        words = np.frombuffer(f.read(n * kq * 2), dtype=np.uint16).reshape(n, kq)
    with open(os.path.join(BASE, base_rel + ".s.bin"), "rb") as f:
        s = np.frombuffer(f.read(n * sbp_l * 2), dtype=np.float16).astype(np.float32).reshape(n, sbp_l)
    with open(os.path.join(BASE, base_rel + ".sz.bin"), "rb") as f:
        sz = np.frombuffer(f.read(n * sbp_l * 2), dtype=np.float16).astype(np.float32).reshape(n, sbp_l)
    q = np.empty((n, k), dtype=np.float32)
    for p in range(4):
        q[:, p * kq:(p + 1) * kq] = ((words >> (4 * p)) & 0xF).astype(np.float32)
    return q * np.repeat(s[:, :ng], 32, axis=1) - np.repeat(sz[:, :ng], 32, axis=1)


def f16(x):
    return np.asarray(x).astype(np.float16).astype(np.float32)


def lf(rel, n=None):
    a = np.fromfile(os.path.join(BASE, rel), dtype=np.float16).astype(np.float32)
    return a.reshape(n, -1) if n else a


def rmsnorm(x, w, eps=1e-6, per=0):
    if per == 0:
        ms = (x.astype(np.float64) ** 2).mean()
        return f16(x / np.sqrt(ms + eps) * w)
    out = np.empty_like(x)
    for i in range(0, len(x), per):
        seg = x[i:i + per].astype(np.float64)
        ms = (seg ** 2).mean()
        out[i:i + per] = f16(seg / np.sqrt(ms + eps) * w)
    return out


def main():
    tok = 846
    # embed
    q = np.empty((1, K))
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
    x = f16(qq * np.repeat(s[:ng], 32) - np.repeat(sz[:ng], 32))

    ln1 = lf(L + "input_layernorm/weight.f16.bin")
    xn = rmsnorm(x, ln1)
    qg = f16(deq(L + "self_attn/q_proj", 12288, K).astype(np.float64) @ xn.astype(np.float64))
    kk = f16(deq(L + "self_attn/k_proj", 1024, K).astype(np.float64) @ xn.astype(np.float64))
    vv = f16(deq(L + "self_attn/v_proj", 1024, K).astype(np.float64) @ xn.astype(np.float64))
    qn_w = lf(L + "self_attn/q_norm/weight.f16.bin").reshape(-1)
    kn_w = lf(L + "self_attn/k_norm/weight.f16.bin").reshape(-1)

    eng = np.fromfile("tmp/eng_l0_postattn.bin", dtype=np.float16).astype(np.float32)
    Wout = deq(L + "self_attn/o_proj", HID, QH * DH) if not os.path.exists(os.path.join(BASE, L + "self_attn/o_proj/weight.f16.bin")) else lf(L + "self_attn/o_proj/weight.f16.bin", HID)

    def build(qgate_mode):
        if qgate_mode == "split":
            qpart = qg[:QH * DH]
            gpart = qg[QH * DH:]
        else:  # per-head interleaved: [q_h(256), gate_h(256)] * 24
            qpart = np.empty(QH * DH, np.float32)
            gpart = np.empty(QH * DH, np.float32)
            for h in range(QH):
                qpart[h * DH:(h + 1) * DH] = qg[h * 2 * DH:(h * 2 + 1) * DH]
                gpart[h * DH:(h + 1) * DH] = qg[(h * 2 + 1) * DH:(h * 2 + 2) * DH]
        # qk-norm per head
        qn = np.empty_like(qpart)
        for h in range(QH):
            qn[h * DH:(h + 1) * DH] = rmsnorm(qpart[h * DH:(h + 1) * DH], qn_w, per=DH)
        kn = np.empty_like(kk)
        for h in range(KVH):
            kn[h * DH:(h + 1) * DH] = rmsnorm(kk[h * DH:(h + 1) * DH], kn_w, per=DH)
        # pos 0: attention out = v (per q head -> its kv head), gated by swish(gate)
        silu = lambda t: t / (1.0 + np.exp(-t))
        out = np.empty(QH * DH, np.float32)
        for h in range(QH):
            kvh = h // (QH // KVH)
            out[h * DH:(h + 1) * DH] = f16(vv[kvh * DH:(kvh + 1) * DH] * silu(gpart[h * DH:(h + 1) * DH]))
        o = f16(Wout.astype(np.float64) @ out.astype(np.float64))
        return f16(x + o)

    for mode in ("split", "interleaved"):
        ref = build(mode)
        print("%-12s: max=%.4f corr=%.4f" % (mode, np.abs(ref - eng).max(), np.corrcoef(ref, eng)[0, 1]))


if __name__ == "__main__":
    main()
