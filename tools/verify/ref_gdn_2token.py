"""2-token GDN 层参考：源层 10，tokens [846, 198]，带 conv/rgdr 状态演进。
与板端 pos=1 的 post-attn 对比。
"""
import os
import numpy as np

BASE = "converted/qwen38_310b"
L = "model/language_model/layers/10/"
K = 5120
KH, VH, HD = 16, 48, 128
kq, ng = K // 4, K // 32
sbp = 160


def f16(x):
    return np.asarray(x).astype(np.float16).astype(np.float32)


def lf(rel, n=None):
    a = np.fromfile(os.path.join(BASE, rel), dtype=np.float16).astype(np.float32)
    return a.reshape(n, -1) if n else a


def deq(base_rel, n, k):
    kq_ = k // 4
    ng_ = k // 32
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


def main():
    toks = [846, 198]
    ln1 = lf(L + "input_layernorm/weight.f16.bin")
    Wqkv = deq(L + "linear_attn/in_proj_qkv", 10240, K)
    Wz = deq(L + "linear_attn/in_proj_z", 6144, K)
    Wa = lf(L + "linear_attn/in_proj_a/weight.f16.bin", 48)
    Wb = lf(L + "linear_attn/in_proj_b/weight.f16.bin", 48)
    cw = lf(L + "linear_attn/conv1d/weight.f16.bin").reshape(10240, 4)
    alog = lf(L + "linear_attn/A_log.f16.bin").reshape(-1)
    dt = lf(L + "linear_attn/dt_bias.f16.bin").reshape(-1)
    gnorm = lf(L + "linear_attn/norm/weight.f16.bin").reshape(-1)[:HD]
    Wout = deq(L + "linear_attn/out_proj", 5120, 6144)
    silu = lambda t: t / (1.0 + np.exp(-t))

    conv_state = [np.zeros(10240, np.float32) for _ in range(3)]   # st0(t-3), st1(t-2), st2(t-1)
    gdn_state = np.zeros((VH, HD, HD), np.float32)
    outs = []
    for tok in toks:
        x = embed(tok)
        xn = f16(x / np.sqrt((x.astype(np.float64) ** 2).mean() + 1e-6) * ln1)
        qkv = f16(Wqkv.astype(np.float64) @ xn.astype(np.float64))
        z = f16(Wz.astype(np.float64) @ xn.astype(np.float64))
        a = f16(Wa.astype(np.float64) @ xn.astype(np.float64))
        b = f16(Wb.astype(np.float64) @ xn.astype(np.float64))
        # conv: out = silu(w0*st0 + w1*st1 + w2*st2 + w3*x)
        convout = np.zeros(10240, np.float32)
        for j in range(3):
            convout += cw[:, j] * conv_state[j]
        convout += cw[:, 3] * qkv
        convout = silu(f16(convout))
        # 状态更新：st0<-st1, st1<-st2, st2<-x
        conv_state = [conv_state[1], conv_state[2], qkv.copy()]
        qc, kc, vc = convout[:2048], convout[2048:4096], convout[4096:]
        # l2norm + q scale
        qn = np.empty(2048, np.float32)
        kn = np.empty(2048, np.float32)
        for h in range(KH):
            s = qc[h * HD:(h + 1) * HD].astype(np.float64)
            qn[h * HD:(h + 1) * HD] = f16(s / np.sqrt((s ** 2).sum() + 1e-6) / np.sqrt(HD))
            s2 = kc[h * HD:(h + 1) * HD].astype(np.float64)
            kn[h * HD:(h + 1) * HD] = f16(s2 / np.sqrt((s2 ** 2).sum() + 1e-6))
        beta = 1 / (1 + np.exp(-b))
        sp = np.where((a + dt) > 20, (a + dt), np.log1p(np.exp(a + dt)))
        decay = np.exp(-np.exp(alog) * sp)
        o = np.empty(VH * HD, np.float32)
        for vh in range(VH):
            kh = vh // (VH // KH)
            kk = kn[kh * HD:(kh + 1) * HD].astype(np.float64)
            qq2 = qn[kh * HD:(kh + 1) * HD].astype(np.float64)
            vv = vc[vh * HD:(vh + 1) * HD].astype(np.float64)
            S = gdn_state[vh].astype(np.float64) * decay[vh]          # 先衰减
            bv = (vv - kk @ S) * beta[vh]                             # bv = (v - S^T k)*beta
            S = S + np.outer(kk, bv)
            gdn_state[vh] = S.astype(np.float32)
            o[vh * HD:(vh + 1) * HD] = f16(S.T @ qq2)
        onorm = np.empty(VH * HD, np.float32)
        for vh in range(VH):
            s = o[vh * HD:(vh + 1) * HD].astype(np.float64)
            ms = (s ** 2).mean()
            onorm[vh * HD:(vh + 1) * HD] = f16(f16(s / np.sqrt(ms + 1e-6) * gnorm) * silu(z[vh * HD:(vh + 1) * HD]))
        out = f16(Wout.astype(np.float64) @ onorm.astype(np.float64))
        outs.append(f16(x + out))
    np.asarray(outs[1]).astype(np.float16).tofile("tmp/ref_layer10_pos1.bin")
    print("ref pos1[:6] =", np.round(outs[1][:6], 4).tolist())
    print("ref pos1 stats: mean=%.5f std=%.5f" % (outs[1].mean(), outs[1].std()))


if __name__ == "__main__":
    main()
