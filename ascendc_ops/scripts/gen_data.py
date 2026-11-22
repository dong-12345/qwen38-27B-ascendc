#!/usr/bin/env python3
"""Generate test vectors for the AscendC ops.  Run from the project root."""
import os

import numpy as np

SEQ = 64
HEADS = 48
D_CONV = 10240          # q 2048 + k 2048 + v 6144
KERNEL = 4
DQ_NORM = 2048          # q / k : 16 heads * 128
NH_NORM = 16


def fp16(x):
    return x.astype(np.float16)


def gen_add(outdir="input"):
    rng = np.random.default_rng(0)
    n = 8 * 2048
    x = fp16(rng.uniform(-1, 1, n))
    y = fp16(rng.uniform(-1, 1, n))
    x.tofile(os.path.join(outdir, "input_x.bin"))
    y.tofile(os.path.join(outdir, "input_y.bin"))
    fp16(x.astype(np.float32) + y.astype(np.float32)).tofile(
        os.path.join(outdir + "_golden", "golden.bin"))


def gen_gdn_gating(outdir="input"):
    rng = np.random.default_rng(1)
    # [HEADS, SEQ] 布局
    a = fp16(rng.normal(0.0, 2.0, (HEADS, SEQ)))
    b = fp16(rng.normal(0.0, 2.0, (HEADS, SEQ)))
    dt_bias = rng.uniform(-0.5, 0.5, HEADS).astype(np.float32)
    A_log = rng.uniform(-2.0, 1.0, HEADS).astype(np.float32)
    g_scale = (-np.exp(A_log)).astype(np.float32)

    a.tofile(os.path.join(outdir, "gdn_gating_a.bin"))
    b.tofile(os.path.join(outdir, "gdn_gating_b.bin"))
    dt_bias.tofile(os.path.join(outdir, "gdn_gating_dt_bias.bin"))
    g_scale.tofile(os.path.join(outdir, "gdn_gating_g_scale.bin"))

    # reference (float64)
    x = a.astype(np.float64) + dt_bias.astype(np.float64)[:, None]
    softplus = np.maximum(x, 0) + np.log1p(np.exp(-np.abs(x)))
    g_ref = g_scale.astype(np.float64)[:, None] * softplus
    beta_ref = 1.0 / (1.0 + np.exp(-b.astype(np.float64)))
    g_ref.astype(np.float32).tofile(os.path.join(outdir + "_golden", "gdn_gating_g.bin"))
    beta_ref.astype(np.float32).tofile(os.path.join(outdir + "_golden", "gdn_gating_beta.bin"))
    print("gdn_gating: g range [%.3f, %.3f], beta range [%.3f, %.3f]" %
          (g_ref.min(), g_ref.max(), beta_ref.min(), beta_ref.max()))


def gen_gdn_conv1d(outdir="input"):
    rng = np.random.default_rng(2)
    x = fp16(rng.normal(0.0, 1.0, (SEQ, D_CONV)))
    w = rng.normal(0.0, 0.25, (D_CONV, KERNEL)).astype(np.float32)
    # host 预处理： [D, KERNEL] -> [KERNEL, D]
    w_tap = np.ascontiguousarray(w.T)
    x.tofile(os.path.join(outdir, "gdn_conv1d_x.bin"))
    w_tap.tofile(os.path.join(outdir, "gdn_conv1d_wtap.bin"))

    x64 = x.astype(np.float64)
    acc = np.zeros((SEQ, D_CONV), dtype=np.float64)
    for j in range(KERNEL):
        shift = KERNEL - 1 - j          # j=0 用 x[s-3]
        src = np.zeros_like(x64)
        if shift == 0:
            src = x64
        else:
            src[shift:] = x64[:-shift]
        acc += src * w[:, j].astype(np.float64)[None, :]
    y = acc / (1.0 + np.exp(-acc))
    fp16(y).tofile(os.path.join(outdir + "_golden", "gdn_conv1d_y.bin"))
    print("gdn_conv1d: y range [%.3f, %.3f]" % (y.min(), y.max()))


def gen_gdn_l2norm(outdir="input"):
    rng = np.random.default_rng(3)
    x = fp16(rng.normal(0.0, 1.0, (SEQ, DQ_NORM)))
    x.tofile(os.path.join(outdir, "gdn_l2norm_x.bin"))
    x64 = x.astype(np.float64).reshape(SEQ, NH_NORM, 128)
    ss = (x64 ** 2).sum(axis=2, keepdims=True)
    scale = 1.0 / np.sqrt(128.0)
    y = (x64 * scale / np.sqrt(ss + 1e-6)).reshape(SEQ, DQ_NORM)
    fp16(y).tofile(os.path.join(outdir + "_golden", "gdn_l2norm_y.bin"))
    print("gdn_l2norm: y range [%.4f, %.4f]" % (y.min(), y.max()))


def gen_gdn_cumsum(outdir="input"):
    rng = np.random.default_rng(4)
    # g 通常是负的对数衰减，这里直接生成负值再累加
    g = (-rng.uniform(0.0, 0.5, (HEADS, SEQ))).astype(np.float32)
    g.tofile(os.path.join(outdir, "gdn_cumsum_g.bin"))
    g_cum = np.cumsum(g.astype(np.float64), axis=1).astype(np.float32)
    g_cum.tofile(os.path.join(outdir + "_golden", "gdn_cumsum_gcum.bin"))
    print("gdn_cumsum: g_cum range [%.3f, %.3f]" % (g_cum.min(), g_cum.max()))


def gen_gdn_gated_rmsnorm(outdir="input"):
    rng = np.random.default_rng(5)
    DV = 6144
    HD = 128
    x = fp16(rng.normal(0.0, 1.0, (SEQ, DV)))
    z = fp16(rng.normal(0.0, 1.0, (SEQ, DV)))
    w = rng.uniform(0.5, 1.5, HD).astype(np.float32)
    x.tofile(os.path.join(outdir, "gdn_gated_rmsnorm_x.bin"))
    z.tofile(os.path.join(outdir, "gdn_gated_rmsnorm_z.bin"))
    w.tofile(os.path.join(outdir, "gdn_gated_rmsnorm_w.bin"))

    x64 = x.astype(np.float64).reshape(SEQ, DV // HD, HD)
    z64 = z.astype(np.float64).reshape(SEQ, DV // HD, HD)
    ms = (x64 ** 2).mean(axis=2, keepdims=True)
    silu = z64 / (1.0 + np.exp(-z64))
    y = (x64 / np.sqrt(ms + 1e-6)) * w.astype(np.float64)[None, None, :] * silu
    fp16(y.reshape(SEQ, DV)).tofile(os.path.join(outdir + "_golden", "gdn_gated_rmsnorm_y.bin"))
    print("gdn_gated_rmsnorm: y range [%.3f, %.3f]" % (y.min(), y.max()))


def gen_w4a16_gemv(outdir="input"):
    rng = np.random.default_rng(6)
    K, N = 5120, 17408
    act = fp16(rng.normal(0.0, 1.0, K))
    q = rng.integers(0, 16, (N, K)).astype(np.uint8)
    scale = rng.uniform(0.01, 0.05, N).astype(np.float32)

    qw = q.reshape(N, 4, K // 4)              # qp[:, p, j] = q[:, p*KQ + j]
    words = (qw[:, 0, :].astype(np.uint16)
             | (qw[:, 1, :].astype(np.uint16) << 4)
             | (qw[:, 2, :].astype(np.uint16) << 8)
             | (qw[:, 3, :].astype(np.uint16) << 12))
    act.tofile(os.path.join(outdir, "w4a16_act.bin"))
    words.astype(np.uint16).tofile(os.path.join(outdir, "w4a16_wq.bin"))
    fp16(scale).tofile(os.path.join(outdir, "w4a16_scale.bin"))

    a64 = act.astype(np.float64)
    ref = ((q.astype(np.float64) - 8.0) * a64[None, :]).sum(axis=1)
    out = ref * scale.astype(np.float64)
    fp16(out).tofile(os.path.join(outdir + "_golden", "w4a16_out.bin"))
    print("w4a16_gemv: out range [%.3f, %.3f], weights %.1f MB" %
          (out.min(), out.max(), words.size * 2 / 1e6))


def gen_cube_matmul(outdir="input"):
    rng = np.random.default_rng(7)
    M, N, K = 64, 5120, 5120
    A = fp16(rng.normal(0.0, 1.0, (M, K)))
    B = fp16(rng.normal(0.0, 1.0, (K, N)))
    A.tofile(os.path.join(outdir, "cube_A.bin"))
    B.tofile(os.path.join(outdir, "cube_B.bin"))
    C = A.astype(np.float64) @ B.astype(np.float64)
    C.astype(np.float32).tofile(os.path.join(outdir + "_golden", "cube_C.bin"))
    print("cube_matmul: A%s B%s C range [%.2f, %.2f]" % (A.shape, B.shape, C.min(), C.max()))


def gen_gdn_kkt(outdir="input"):
    rng = np.random.default_rng(8)
    T, D = 64, 128
    K = fp16(rng.normal(0.0, 1.0, (T, D)))
    # 每个 token 的衰减 g 为负，gc 是 chunk 内 cumsum
    g = -rng.uniform(0.01, 0.3, T)
    gc = np.cumsum(g).astype(np.float32)
    beta = rng.uniform(0.05, 0.95, T).astype(np.float32)
    K.tofile(os.path.join(outdir, "gdn_kkt_k.bin"))
    gc.tofile(os.path.join(outdir, "gdn_kkt_gc.bin"))
    beta.tofile(os.path.join(outdir, "gdn_kkt_beta.bin"))

    k64 = K.astype(np.float64)
    s = k64 @ k64.T
    decay = np.exp(gc.astype(np.float64)[:, None] - gc.astype(np.float64)[None, :])
    A = beta.astype(np.float64)[:, None] * s * decay
    mask = np.tril(np.ones((T, T), dtype=bool))
    A = np.where(mask, A, 0.0)
    A.astype(np.float32).tofile(os.path.join(outdir + "_golden", "gdn_kkt_a.bin"))
    print("gdn_kkt: A range [%.3f, %.3f] (gc range [%.1f, %.1f])" %
          (A.min(), A.max(), gc.min(), gc.max()))


def gen_gdn_tril(outdir="input"):
    rng = np.random.default_rng(9)
    T = 64
    A = np.tril(rng.normal(0.0, 0.15, (T, T)), -1).astype(np.float32)
    A.tofile(os.path.join(outdir, "gdn_tril_a.bin"))
    M = np.eye(T, dtype=np.float64) + A.astype(np.float64)
    T_ref = np.linalg.inv(M)
    T_ref.astype(np.float32).tofile(os.path.join(outdir + "_golden", "gdn_tril_t.bin"))
    print("gdn_tril: T range [%.3f, %.3f]" % (T_ref.min(), T_ref.max()))


def gen_gdn_wy(outdir="input"):
    rng = np.random.default_rng(10)
    T, D = 64, 128
    A = np.tril(rng.normal(0.0, 0.15, (T, T)), -1).astype(np.float32)
    Tril = np.linalg.inv(np.eye(T, dtype=np.float64) + A.astype(np.float64)).astype(np.float32)
    k = fp16(rng.normal(0.0, 1.0, (T, D)))
    v = fp16(rng.normal(0.0, 1.0, (T, D)))
    g = -rng.uniform(0.01, 0.2, T)
    gc = np.cumsum(g).astype(np.float32)
    beta = rng.uniform(0.05, 0.95, T).astype(np.float32)

    Tril.tofile(os.path.join(outdir, "gdn_wy_t.bin"))
    k.tofile(os.path.join(outdir, "gdn_wy_k.bin"))
    v.tofile(os.path.join(outdir, "gdn_wy_v.bin"))
    beta.tofile(os.path.join(outdir, "gdn_wy_beta.bin"))
    gc.tofile(os.path.join(outdir, "gdn_wy_gc.bin"))

    kb = k.astype(np.float64) * beta.astype(np.float64)[:, None] * np.exp(gc.astype(np.float64))[:, None]
    vb = v.astype(np.float64) * beta.astype(np.float64)[:, None]
    w_ref = Tril.astype(np.float64) @ kb
    u_ref = Tril.astype(np.float64) @ vb
    fp16(w_ref).tofile(os.path.join(outdir + "_golden", "gdn_wy_w.bin"))
    fp16(u_ref).tofile(os.path.join(outdir + "_golden", "gdn_wy_u.bin"))
    print("gdn_wy: w range [%.2f, %.2f], u range [%.2f, %.2f]" %
          (w_ref.min(), w_ref.max(), u_ref.min(), u_ref.max()))


def gen_gdn_chunk_h(outdir="input"):
    """按 fla naive_chunk_gated_delta_rule 的公式生成单 chunk 的 w/u/S0 与参考输出。"""
    rng = np.random.default_rng(11)
    C, KD, VD = 64, 128, 128
    # 真实 GDN 里 k 是 L2 归一化后的单位向量；不归一化会让 (I+A)^{-1} 数值爆炸
    k_raw = rng.normal(0.0, 1.0, (C, KD))
    k = (k_raw / np.linalg.norm(k_raw, axis=1, keepdims=True)).astype(np.float16)
    v = fp16(rng.normal(0.0, 1.0, (C, VD)))
    g = -rng.uniform(0.01, 0.15, C)
    gc = np.cumsum(g).astype(np.float32)
    beta = rng.uniform(0.05, 0.95, C).astype(np.float32)
    S0 = (rng.normal(0.0, 0.1, (KD, VD))).astype(np.float32)

    k64 = k.astype(np.float64)
    v64 = v.astype(np.float64)
    b64 = beta.astype(np.float64)
    gc64 = gc.astype(np.float64)
    L = np.exp(gc64[:, None] - gc64[None, :])
    kkt = -(k64 * b64[:, None]) @ k64.T * np.tril(L, -1)
    Tril = np.linalg.inv(np.eye(C) + kkt)
    u_ref = Tril @ (v64 * b64[:, None])
    w_ref = Tril @ (k64 * b64[:, None] * np.exp(gc64)[:, None])

    S0_64 = S0.astype(np.float64)
    vnew = u_ref - w_ref @ S0_64
    kscaled = k64 * np.exp(gc64[-1] - gc64)[:, None]
    Sout = np.exp(gc64[-1]) * S0_64 + kscaled.T @ vnew

    fp16(w_ref).tofile(os.path.join(outdir, "gdn_ch_h_w.bin"))
    fp16(u_ref).tofile(os.path.join(outdir, "gdn_ch_h_u.bin"))
    k.tofile(os.path.join(outdir, "gdn_ch_h_k.bin"))
    gc.tofile(os.path.join(outdir, "gdn_ch_h_gc.bin"))
    S0.tofile(os.path.join(outdir, "gdn_ch_h_sin.bin"))
    fp16(vnew).tofile(os.path.join(outdir + "_golden", "gdn_ch_h_vnew.bin"))
    Sout.astype(np.float32).tofile(os.path.join(outdir + "_golden", "gdn_ch_h_sout.bin"))
    print("gdn_chunk_h: vnew range [%.2f, %.2f], Sout range [%.2f, %.2f]" %
          (vnew.min(), vnew.max(), Sout.min(), Sout.max()))


def gen_gdn_chunk_o(outdir="input"):
    rng = np.random.default_rng(12)
    C, KD, VD = 64, 128, 128
    q_raw = rng.normal(0.0, 1.0, (C, KD))
    q = (q_raw / np.linalg.norm(q_raw, axis=1, keepdims=True) * (KD ** -0.5)).astype(np.float16)
    k_raw = rng.normal(0.0, 1.0, (C, KD))
    k = (k_raw / np.linalg.norm(k_raw, axis=1, keepdims=True)).astype(np.float16)
    g = -rng.uniform(0.01, 0.15, C)
    gc = np.cumsum(g).astype(np.float32)
    S = rng.normal(0.0, 0.1, (KD, VD)).astype(np.float32)
    vnew = (rng.normal(0.0, 0.1, (C, VD))).astype(np.float16)

    q.tofile(os.path.join(outdir, "gdn_ch_o_q.bin"))
    k.tofile(os.path.join(outdir, "gdn_ch_o_k.bin"))
    gc.tofile(os.path.join(outdir, "gdn_ch_o_gc.bin"))
    S.tofile(os.path.join(outdir, "gdn_ch_o_s.bin"))
    vnew.tofile(os.path.join(outdir, "gdn_ch_o_vnew.bin"))

    q64 = q.astype(np.float64)
    k64 = k.astype(np.float64)
    gc64 = gc.astype(np.float64)
    L = np.exp(gc64[:, None] - gc64[None, :])
    attn = (q64 @ k64.T) * L
    attn = np.where(np.tril(np.ones((C, C), dtype=bool)), attn, 0.0)
    o_inter = (q64 * np.exp(gc64)[:, None]) @ S.astype(np.float64)
    o = o_inter + attn @ vnew.astype(np.float64)
    fp16(o).tofile(os.path.join(outdir + "_golden", "gdn_ch_o_o.bin"))
    print("gdn_chunk_o: o range [%.3f, %.3f]" % (o.min(), o.max()))


def gen_gdn_rgdr(outdir="input"):
    rng = np.random.default_rng(13)
    KD = 128
    k_raw = rng.normal(0.0, 1.0, KD)
    k = (k_raw / np.linalg.norm(k_raw)).astype(np.float16)
    q = fp16(rng.normal(0.0, 1.0, KD) / np.sqrt(KD))
    v = fp16(rng.normal(0.0, 1.0, KD))
    S0 = rng.normal(0.0, 0.1, (KD, KD)).astype(np.float32)
    beta, g = 0.5, -0.05

    q.tofile(os.path.join(outdir, "gdn_rg_q.bin"))
    k.tofile(os.path.join(outdir, "gdn_rg_k.bin"))
    v.tofile(os.path.join(outdir, "gdn_rg_v.bin"))
    S0.tofile(os.path.join(outdir, "gdn_rg_sin.bin"))
    np.array([beta, g], dtype=np.float32).tofile(os.path.join(outdir, "gdn_rg_bg.bin"))

    k64 = k.astype(np.float64)
    q64 = q.astype(np.float64)
    v64 = v.astype(np.float64)
    # 注意：S 是 [K,V]，k 是 [K]，naive 公式 (h * k[...,None]).sum(-2) = S^T @ k。
    # 因为 K==V==128，写成 S @ k 不会报错但含义完全错（曾因此误判 kernel 有 bug）。
    S = S0.astype(np.float64) * np.exp(g)
    bv = (v64 - S.T @ k64) * beta
    S = S + np.outer(k64, bv)
    o = q64 @ S
    fp16(o).tofile(os.path.join(outdir + "_golden", "gdn_rg_o.bin"))
    S.astype(np.float32).tofile(os.path.join(outdir + "_golden", "gdn_rg_sout.bin"))
    print("gdn_rgdr: o range [%.3f, %.3f], S_out range [%.3f, %.3f]" %
          (o.min(), o.max(), S.min(), S.max()))


def gen_full_attn_decode(outdir="input", pos=5):
    rng = np.random.default_rng(14)
    QH, KH, D, MD, RH = 24, 4, 256, 512, 32
    GK = QH // KH
    q = fp16(rng.normal(0.0, 1.0, QH * D))
    k = fp16(rng.normal(0.0, 1.0, KH * D))
    v = fp16(rng.normal(0.0, 1.0, KH * D))
    gate = fp16(rng.normal(0.0, 1.0, QH * D))
    # RoPE 频率：theta=1e7，取前 RH 个
    inv_freq = 1.0 / (1e7 ** (np.arange(0, RH * 2, 2) / (RH * 2)))
    cosv = np.cos(pos * inv_freq).astype(np.float32)
    sinv = np.sin(pos * inv_freq).astype(np.float32)
    kc = fp16(rng.normal(0.0, 0.2, KH * MD * D))
    vc = fp16(rng.normal(0.0, 0.2, KH * MD * D))

    q.tofile(os.path.join(outdir, "attn_q.bin"))
    k.tofile(os.path.join(outdir, "attn_k.bin"))
    v.tofile(os.path.join(outdir, "attn_v.bin"))
    cosv.tofile(os.path.join(outdir, "attn_cos.bin"))
    sinv.tofile(os.path.join(outdir, "attn_sin.bin"))
    gate.tofile(os.path.join(outdir, "attn_gate.bin"))
    kc.tofile(os.path.join(outdir, "attn_kcache.bin"))
    vc.tofile(os.path.join(outdir, "attn_vcache.bin"))

    def rope(row):
        out = row.copy()
        for i in range(RH):
            a = row[i]
            b = row[i + RH]
            out[i] = a * cosv[i] - b * sinv[i]
            out[i + RH] = b * cosv[i] + a * sinv[i]
        return out

    # 更新 cache（kernel 会做同样的事）
    kc_u = kc.astype(np.float64).reshape(KH, MD, D)
    vc_u = vc.astype(np.float64).reshape(KH, MD, D)
    k64 = k.astype(np.float64).reshape(KH, D)
    v64 = v.astype(np.float64).reshape(KH, D)
    for h in range(KH):
        kc_u[h, pos] = rope(k64[h])
        vc_u[h, pos] = v64[h]
    q64 = q.astype(np.float64).reshape(QH, D)
    gate64 = gate.astype(np.float64).reshape(QH, D)
    nkeys = pos + 1
    out = np.zeros((QH, D))
    for h in range(QH):
        hkv = h // GK
        qr = rope(q64[h])
        scores = kc_u[hkv, :nkeys] @ qr / 16.0
        scores = scores - scores.max()
        p = np.exp(scores)
        p /= p.sum()
        o = p @ vc_u[hkv, :nkeys]
        sw = gate64[h] / (1.0 + np.exp(-gate64[h]))
        out[h] = o * sw
    fp16(out.reshape(-1)).tofile(os.path.join(outdir + "_golden", "attn_out.bin"))
    print("full_attn_decode: pos=%d, out range [%.3f, %.3f]" % (pos, out.min(), out.max()))


def gen_dense_ops(outdir="input"):
    rng = np.random.default_rng(15)
    K, N, H, C = 512, 256, 512, 256
    act = fp16(rng.normal(0.0, 1.0, K))
    w = fp16(rng.normal(0.0, 0.1, (N, K)))
    act.tofile(os.path.join(outdir, "d_gemv_act.bin"))
    w.tofile(os.path.join(outdir, "d_gemv_w.bin"))
    fp16(w.astype(np.float64) @ act.astype(np.float64)).tofile(
        os.path.join(outdir + "_golden", "d_gemv_out.bin"))

    x = fp16(rng.normal(0.0, 1.0, H))
    ww = fp16(rng.normal(1.0, 0.1, H))
    x.tofile(os.path.join(outdir, "d_rms_x.bin"))
    ww.tofile(os.path.join(outdir, "d_rms_w.bin"))
    x64 = x.astype(np.float64)
    rms = np.sqrt((x64 ** 2).mean() + 1e-6)
    fp16(x64 / rms * ww.astype(np.float64)).tofile(
        os.path.join(outdir + "_golden", "d_rms_y.bin"))

    a = fp16(rng.normal(0.0, 1.0, H))
    b = fp16(rng.normal(0.0, 1.0, H))
    a.tofile(os.path.join(outdir, "d_ax_a.bin"))
    b.tofile(os.path.join(outdir, "d_ax_b.bin"))
    fp16(a.astype(np.float64) + 0.5 * b.astype(np.float64)).tofile(
        os.path.join(outdir + "_golden", "d_ax_out.bin"))

    g = fp16(rng.normal(0.0, 1.0, H))
    u = fp16(rng.normal(0.0, 1.0, H))
    g.tofile(os.path.join(outdir, "d_ms_g.bin"))
    u.tofile(os.path.join(outdir, "d_ms_u.bin"))
    g64 = g.astype(np.float64)
    fp16(g64 / (1.0 + np.exp(-g64)) * u.astype(np.float64)).tofile(
        os.path.join(outdir + "_golden", "d_ms_out.bin"))

    xv = fp16(rng.normal(0.0, 1.0, C))
    st = rng.normal(0.0, 1.0, (3, C)).astype(np.float32)
    wc = fp16(rng.normal(0.0, 0.5, (4, C)))
    xv.tofile(os.path.join(outdir, "d_cv_x.bin"))
    st.tofile(os.path.join(outdir, "d_cv_state.bin"))
    wc.tofile(os.path.join(outdir, "d_cv_w.bin"))
    hist = np.stack([st[0], st[1], st[2], xv.astype(np.float64)])
    conv = (wc.astype(np.float64) * hist).sum(axis=0)
    fp16(conv / (1.0 + np.exp(-conv))).tofile(os.path.join(outdir + "_golden", "d_cv_out.bin"))
    newst = np.stack([st[1], st[2], xv.astype(np.float32)])
    newst.tofile(os.path.join(outdir + "_golden", "d_cv_state_out.bin"))
    print("dense_ops: data ready")


def gen_w4a16_gemv_gen(outdir="input"):
    """通用 W4A16 GEMV：相位交织打包 + group-32 非对称（scale + sz=s*z）"""
    rng = np.random.default_rng(16)
    K, N = 5120, 5120
    GS = 32
    NG = K // GS
    SBP = (NG + 15) // 16 * 16
    act = fp16(rng.normal(0.0, 1.0, K))
    q = rng.integers(0, 16, (N, K)).astype(np.uint8)
    s = fp16(rng.uniform(0.002, 0.01, (N, NG)))
    z = rng.uniform(1.0, 14.0, (N, NG)).astype(np.float16)
    sz = fp16(s.astype(np.float32) * z.astype(np.float32))

    KQ = K // 4
    qq = q.reshape(N, 4, KQ).astype(np.uint16)
    words = (qq[:, 0, :] | (qq[:, 1, :] << 4) | (qq[:, 2, :] << 8) | (qq[:, 3, :] << 12)).astype(np.uint16)
    act.tofile(os.path.join(outdir, "gact.bin"))
    words.tofile(os.path.join(outdir, "gwq.bin"))
    sp = np.zeros((N, SBP), dtype=np.float16)
    szp = np.zeros((N, SBP), dtype=np.float16)
    sp[:, :NG] = s
    szp[:, :NG] = sz
    sp.tofile(os.path.join(outdir, "gsc.bin"))
    szp.tofile(os.path.join(outdir, "gsz.bin"))

    a64 = act.astype(np.float64)
    deq = (q.astype(np.float64) * np.repeat(s.astype(np.float64), GS, axis=1)
           - np.repeat(sz.astype(np.float64), GS, axis=1))
    out = deq @ a64
    fp16(out).tofile(os.path.join(outdir + "_golden", "gout.bin"))
    print("w4a16_gemv_gen: out range [%.2f, %.2f]" % (out.min(), out.max()))


if __name__ == "__main__":
    os.makedirs("input", exist_ok=True)
    os.makedirs("input_golden", exist_ok=True)
    gen_add()
    gen_gdn_gating()
    gen_gdn_conv1d()
    gen_gdn_l2norm()
    gen_gdn_cumsum()
    gen_gdn_gated_rmsnorm()
    gen_w4a16_gemv()
    gen_cube_matmul()
    gen_gdn_kkt()
    gen_gdn_tril()
    gen_gdn_wy()
    gen_gdn_chunk_h()
    gen_gdn_chunk_o()
    gen_gdn_rgdr()
    gen_full_attn_decode()
    gen_dense_ops()
    gen_w4a16_gemv_gen()
    print("inputs written")
