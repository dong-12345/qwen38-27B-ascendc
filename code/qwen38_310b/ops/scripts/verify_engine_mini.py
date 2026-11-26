#!/usr/bin/env python3
"""verify_engine_mini.py — 从 engine_mini dump 的权重重算参考，逐级对比中间结果。"""
import os
import numpy as np

OUT = "output"


def l16(name):
    return np.fromfile(os.path.join(OUT, name), dtype=np.float16).astype(np.float64)


def l32(name):
    return np.fromfile(os.path.join(OUT, name), dtype=np.float32).astype(np.float64)


def main():
    H, NKH, NVH, KD, VD, FF, VOCAB, GK = 512, 2, 6, 128, 128, 1024, 1024, 3
    EPS = 1e-6
    D = KD

    norm_w = l16("eng_norm_w.bin")
    Wq = l16("eng_Wq.bin").reshape(NKH * KD, H)
    Wk = l16("eng_Wk.bin").reshape(NKH * KD, H)
    Wv = l16("eng_Wv.bin").reshape(NVH * VD, H)
    cWq = l16("eng_convWq.bin").reshape(4, NKH * KD)
    cWk = l16("eng_convWk.bin").reshape(4, NKH * KD)
    cWv = l16("eng_convWv.bin").reshape(4, NVH * VD)
    Wb = l16("eng_Wb.bin").reshape(NVH, H)
    Wa = l16("eng_Wa.bin").reshape(NVH, H)
    A_log = l32("eng_Alog.bin")
    dt_bias = l32("eng_dtbias.bin")
    Wz = l16("eng_Wz.bin").reshape(NVH * VD, H)
    norm_out_w = l16("eng_norm_out_w.bin")
    Wo = l16("eng_Wo.bin").reshape(H, NVH * VD)
    mlp_norm_w = l16("eng_mlp_norm_w.bin")
    Wgate = l16("eng_Wgate.bin").reshape(FF, H)
    Wup = l16("eng_Wup.bin").reshape(FF, H)
    Wdown = l16("eng_Wdown.bin").reshape(H, FF)
    lm_head = l16("eng_lm_head.bin").reshape(VOCAB, H)
    x = l16("eng_x.bin")

    def rmsnorm(v, w):
        return v / np.sqrt((v ** 2).mean() + EPS) * w

    def l2norm(v):
        return v * (1.0 / np.sqrt(D)) / np.sqrt((v ** 2).sum() + EPS)

    def conv(v, w, C):
        # state 全 0（引擎首 token），hist = [0,0,0,v]
        return v * w[3] / (1.0 + np.exp(-(v * w[3])))

    xn = rmsnorm(x, norm_w)
    q = Wq @ xn
    k = Wk @ xn
    v = Wv @ xn
    qc = conv(q, cWq, NKH * KD)
    kc = conv(k, cWk, NKH * KD)
    vc = conv(v, cWv, NVH * VD)
    qn = np.concatenate([l2norm(qc[i * KD:(i + 1) * KD]) for i in range(NKH)])
    kn = np.concatenate([l2norm(kc[i * KD:(i + 1) * KD]) for i in range(NKH)])
    b = Wb @ xn
    a = Wa @ xn
    beta = 1.0 / (1.0 + np.exp(-b))
    g = -np.exp(A_log) * np.log1p(np.exp(a + dt_bias))
    z = Wz @ xn

    # rgdr（state=0 首 token）：S = outer(k, (v*beta))；o = q @ S
    o_list = []
    for vh in range(NVH):
        kh = vh // GK
        kv = kn[kh * KD:(kh + 1) * KD]
        vv = vc[vh * VD:(vh + 1) * VD]
        S = np.outer(kv, vv * beta[vh])
        o_list.append(qn[kh * KD:(kh + 1) * KD] @ S)
    o = np.concatenate(o_list)
    onorm = np.concatenate([
        rmsnorm(o[i * VD:(i + 1) * VD], norm_out_w) * (z[i * VD:(i + 1) * VD] /
                                                        (1.0 + np.exp(-z[i * VD:(i + 1) * VD])))
        for i in range(NVH)])
    y = Wo @ onorm
    h = x + y
    hn = rmsnorm(h, mlp_norm_w)
    actv = (Wgate @ hn) / (1.0 + np.exp(-(Wgate @ hn))) * (Wup @ hn)
    y2 = Wdown @ actv
    h2 = h + y2
    fn = rmsnorm(h2, l16("eng_final_norm_w.bin"))
    logits = lm_head @ fn

    def cmp(name, got, ref):
        got = got.astype(np.float64)
        ref = ref.astype(np.float64)
        ae = np.abs(got - ref)
        scale = max(np.abs(ref).max(), 1e-6)
        print("%-14s max_abs=%.3e  rel=%.2e  (ref max %.3f)" % (name, ae.max(), ae.max() / scale, scale))
        return ae.max() / scale

    print("=== engine_mini 逐级对比（相对误差）===")
    err = 0.0
    err = max(err, cmp("beta", l32("eng_out_beta.bin"), beta))
    err = max(err, cmp("g", l32("eng_out_g.bin"), g))
    err = max(err, cmp("qc", l16("eng_out_qc.bin"), qc))
    err = max(err, cmp("kc", l16("eng_out_kc.bin"), kc))
    err = max(err, cmp("vc", l16("eng_out_vc.bin"), vc))
    err = max(err, cmp("qn", l16("eng_out_qn.bin"), qn))
    err = max(err, cmp("o", l16("eng_out_o.bin"), o))
    err = max(err, cmp("onorm", l16("eng_out_onorm.bin"), onorm))
    err = max(err, cmp("logits", l16("eng_out_logits.bin"), logits))
    print("RESULT:", "PASS" if err < 2e-2 else "FAIL", "(worst rel err %.2e)" % err)


if __name__ == "__main__":
    main()
