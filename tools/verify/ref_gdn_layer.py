"""ref_gdn_layer.py — PC numpy 参考：源层 10（GDN）单 token decode，逐步与板端 dump 对比。

引擎子集 sub1 的 layer0 = 源层 10；输入 x = embed(846) 反量化。
"""
import os
import numpy as np

BASE = "converted/qwen38_310b"
K = 5120
HID = 5120
KH, VH, HD = 16, 48, 128
CONV = [2048, 2048, 6144]


def deq_mat(base_rel, n, k):
    kq = k // 4
    ng = k // 32
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
    s_full = np.repeat(s[:, :ng], 32, axis=1)
    sz_full = np.repeat(sz[:, :ng], 32, axis=1)
    return q * s_full - sz_full


def f16(x):
    return x.astype(np.float16).astype(np.float32)


def rmsnorm(x, w, eps=1e-6):
    ms = (x.astype(np.float64) ** 2).mean()
    return f16(x / np.sqrt(ms + eps) * w)


def load_f16(rel, n=None):
    a = np.fromfile(os.path.join(BASE, rel), dtype=np.float16).astype(np.float32)
    return a if n is None else a.reshape(n, -1)


def main():
    li = 10
    L = "model/language_model/layers/%d/" % li
    tok = 846

    # embed
    x = f16(deq_mat("model/language_model/embed_tokens/weight", 1, K)[0])

    # ln1 + qkv/z/a/b
    ln1 = load_f16(L + "input_layernorm/weight.f16.bin")
    xn = rmsnorm(x, ln1)
    qkv = f16(deq_mat(L + "linear_attn/in_proj_qkv", 10240, K) @ xn.astype(np.float64).astype(np.float32))
    z = f16(deq_mat(L + "linear_attn/in_proj_z", 6144, K) @ xn)
    a = f16(load_f16(L + "linear_attn/in_proj_a/weight.f16.bin", 48) @ xn)
    b = f16(load_f16(L + "linear_attn/in_proj_b/weight.f16.bin", 48) @ xn)

    # conv (zero state) + silu
    cw = load_f16(L + "linear_attn/conv1d/weight.f16.bin").reshape(10240, 4)  # [C,4]
    w3 = cw[:, 3]
    qc = f16(qkv[:2048] * w3[:2048])
    kc = f16(qkv[2048:4096] * w3[2048:4096])
    vc = f16(qkv[4096:] * w3[4096:])
    silu = lambda t: t / (1.0 + np.exp(-t))
    qc, kc, vc = silu(qc), silu(kc), silu(vc)

    # l2norm (+ q scale 1/sqrt(128))
    qn = np.empty(2048, np.float32)
    kn = np.empty(2048, np.float32)
    for h in range(KH):
        s = qc[h * HD:(h + 1) * HD]
        qn[h * HD:(h + 1) * HD] = s / np.sqrt((s.astype(np.float64) ** 2).sum() + 1e-6) / np.sqrt(HD)
        s2 = kc[h * HD:(h + 1) * HD]
        kn[h * HD:(h + 1) * HD] = s2 / np.sqrt((s2.astype(np.float64) ** 2).sum() + 1e-6)
    qn, kn = f16(qn), f16(kn)

    # beta / g
    alog = load_f16(L + "linear_attn/A_log.f16.bin", 48)
    dt = load_f16(L + "linear_attn/dt_bias.f16.bin", 48)
    beta = 1.0 / (1.0 + np.exp(-b))
    sp = np.where((a + dt) > 20, (a + dt), np.log1p(np.exp(a + dt)))
    decay = np.exp(-np.exp(alog) * sp)

    # rgdr with S=0: bv = beta*v ; S = k⊗bv ; o = q·S
    gnorm = load_f16(L + "linear_attn/norm/weight.f16.bin").reshape(-1)[:HD]
    o = np.empty(VH * HD, np.float32)
    for vh in range(VH):
        kh = vh // (VH // KH)
        kk = kn[kh * HD:(kh + 1) * HD].astype(np.float64)
        qq = qn[kh * HD:(kh + 1) * HD].astype(np.float64)
        vv = vc[vh * HD:(vh + 1) * HD].astype(np.float64)
        bv = beta[vh] * vv
        S = np.outer(kk, bv)                    # [HD,HD] (S[j,i]=k_j*bv_i)
        o[vh * HD:(vh + 1) * HD] = (S.T @ qq).astype(np.float32)
    o = f16(o)
    # gated norm per v head
    onorm = np.empty(VH * HD, np.float32)
    for vh in range(VH):
        s = o[vh * HD:(vh + 1) * HD]
        ms = (s.astype(np.float64) ** 2).mean()
        onorm[vh * HD:(vh + 1) * HD] = s / np.sqrt(ms + 1e-6) * gnorm
    onorm = f16(onorm)
    onorm = f16(onorm * silu(z))

    # out_proj: 检查是否 f16 还是 w4
    ow_w4 = os.path.exists(os.path.join(BASE, L + "linear_attn/out_proj.w4.bin"))
    if ow_w4:
        out = f16(deq_mat(L + "linear_attn/out_proj", HID, VH * HD) @ onorm)
    else:
        out = f16(load_f16(L + "linear_attn/out_proj/weight.f16.bin", HID) @ onorm)
    x_out = f16(x + out)

    # 与板端 dump 对比
    dump = "tmp/eng_l0_postattn.bin"
    if os.path.exists(dump):
        eng = np.fromfile(dump, dtype=np.float16).astype(np.float32)
        d = np.abs(eng - x_out)
        print("post-attn x: max_abs=%.4f  mean_abs=%.5f  (ref|max|=%.3f)" % (d.max(), d.mean(), np.abs(x_out).max()))
        print("  前 8 个: eng=%s" % np.round(eng[:8], 3).tolist())
        print("           ref=%s" % np.round(x_out[:8], 3).tolist())
        print("  eng 统计: mean=%.4f std=%.4f | ref mean=%.4f std=%.4f"
              % (eng.mean(), eng.std(), x_out.mean(), x_out.std()))
    else:
        print("(no dump yet) ref stats: mean=%.4f std=%.4f" % (x_out.mean(), x_out.std()))


if __name__ == "__main__":
    main()
