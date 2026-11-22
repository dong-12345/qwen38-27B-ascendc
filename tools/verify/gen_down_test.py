"""gen_down_test.py — 用真实 down_proj 权重 (K=17408, N=5120) 构造 GEMV 测试数据。
act = 参考计算的 MLP 中间量（silu(gate)*up），来自已验证的 post-attn 输入。
"""
import os
import numpy as np

BASE = "converted/qwen38_310b"
L = "model/language_model/layers/10/"
K = 5120          # hidden
KF = 17408        # down_proj 的 K（FF）
N = 5120
kq, ng, sbp = K // 4, K // 32, 160
kqF, ngF = KF // 4, KF // 32
sbpF = (ngF + 15) // 16 * 16

OUT_IN = "ascendc_ops/input"
OUT_GOLD = "ascendc_ops/input_golden"
os.makedirs(OUT_IN, exist_ok=True)
os.makedirs(OUT_GOLD, exist_ok=True)


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


def main():
    # 1) 计算 act（MLP 中间量）
    xpost = np.fromfile("tmp/p1_eng_post.bin", dtype=np.float16).astype(np.float32)
    ln2 = lf(L + "post_attention_layernorm/weight.f16.bin")
    xn2 = f16(xpost / np.sqrt((xpost.astype(np.float64) ** 2).mean() + 1e-6) * ln2)
    Wg = deq(L + "mlp/gate_proj", KF, K)
    Wu = deq(L + "mlp/up_proj", KF, K)
    silu = lambda t: t / (1.0 + np.exp(-t))
    gate = f16(Wg.astype(np.float64) @ xn2.astype(np.float64))
    up = f16(Wu.astype(np.float64) @ xn2.astype(np.float64))
    act = f16(f16(silu(gate)) * up)          # [KF]
    act.astype(np.float16).tofile(os.path.join(OUT_IN, "gact.bin"))

    # 2) down_proj 权重（全量 N=5120 行，K=17408）
    with open(os.path.join(BASE, L + "mlp/down_proj.w4.bin"), "rb") as f:
        words = np.frombuffer(f.read(N * kqF * 2), dtype=np.uint16).reshape(N, kqF)
    with open(os.path.join(BASE, L + "mlp/down_proj.s.bin"), "rb") as f:
        sp = np.frombuffer(f.read(N * sbpF * 2), dtype=np.float16).reshape(N, sbpF)
    with open(os.path.join(BASE, L + "mlp/down_proj.sz.bin"), "rb") as f:
        szp = np.frombuffer(f.read(N * sbpF * 2), dtype=np.float16).reshape(N, sbpF)
    words.tofile(os.path.join(OUT_IN, "gwq.bin"))
    sp.tofile(os.path.join(OUT_IN, "gsc.bin"))
    szp.tofile(os.path.join(OUT_IN, "gsz.bin"))

    # 3) golden
    q = np.empty((N, KF), dtype=np.float32)
    for p in range(4):
        q[:, p * kqF:(p + 1) * kqF] = ((words >> (4 * p)) & 0xF).astype(np.float32)
    w = q * np.repeat(sp[:, :ngF].astype(np.float32), 32, axis=1) - np.repeat(szp[:, :ngF].astype(np.float32), 32, axis=1)
    gold = (w.astype(np.float64) @ act.astype(np.float64)).astype(np.float16)
    gold.tofile(os.path.join(OUT_GOLD, "gout.bin"))
    print("act %d, wq %d, sc %d, sz %d, gold %d" % (act.size, words.size, sp.size, szp.size, gold.size))
    print("gold[:6] =", gold[:6].tolist())


if __name__ == "__main__":
    main()
