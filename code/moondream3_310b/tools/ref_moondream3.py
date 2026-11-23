#!/usr/bin/env python3
"""ref_moondream3.py — 纯 numpy 参考实现（读取本项目 W4A16 转换格式），用于和板端引擎对拍。

只做文本 decode（逐 token，无 prefill），支持的 CFG 与 converter 输出一致。
"""
import argparse, json, os, struct
import numpy as np

DT = {"F16": np.float16, "F32": np.float32}


def load_meta(d):
    cfg, w4, f16 = {}, {}, {}
    for line in open(os.path.join(d, "meta.txt")):
        t = line.split()
        if not t: continue
        if t[0] == "CFG": cfg[t[1]] = t[2]
        elif t[0] == "W4": w4[t[1]] = (t[2], int(t[3]), int(t[4]))
        elif t[0] == "F16": f16[t[1]] = (t[2], t[3])
    return cfg, w4, f16


class Model:
    def __init__(self, d, n_layers=None):
        self.d = d
        cfg, self.w4, self.f16 = load_meta(d)
        self.H = int(cfg["hidden"]); self.L = int(cfg["n_layers"])
        if n_layers: self.L = min(self.L, n_layers)
        self.NH = int(cfg["n_heads"]); self.NKV = int(cfg["n_kv_heads"])
        self.D = int(cfg["head_dim"]); self.RD = int(cfg["rot_dim"]); self.V = int(cfg["vocab"])
        self.MAXT = int(cfg["max_context"]); self.MOE0 = int(cfg["moe_start"])
        self.NE = int(cfg["n_experts"]); self.TOPK = int(cfg["experts_per_token"])
        self.EI = int(cfg["expert_inner"]); self.FF = int(cfg["dense_ff"])
        self.theta = float(cfg["rope_theta"]); self.eps = float(cfg["eps"])
        self._w4cache = {}; self._f16cache = {}
        # 持久化非 MoE 权重（attention/dense/embed/lm/final norm）
        self.keep = {}
        for i in range(self.L):
            pre = "T.L%d" % i
            for role in [pre + ".LN.W", pre + ".LN.B", pre + ".QKV.B", pre + ".O.B",
                         pre + ".TAU.WQ", pre + ".TAU.WV", pre + ".TAU.A"]:
                self.keep[role] = self.get_f16(role)
            if i < self.MOE0:
                self.keep[pre + ".FC1.B"] = self.get_f16(pre + ".FC1.B")
                self.keep[pre + ".FC2.B"] = self.get_f16(pre + ".FC2.B")
            else:
                self.keep[pre + ".ROUTER.B"] = self.get_f16(pre + ".ROUTER.B")
        self.keep["T.LM.B"] = self.get_f16("T.LM.B")
        self.keep["T.FN.W"] = self.get_f16("T.FN.W")
        self.keep["T.FN.B"] = self.get_f16("T.FN.B")
        self.wte = self.get_w4("T.WTE")          # f16 [50000,2048]（~200MB）
        self.lm = self.get_w4("T.LM")            # f16
        print("[ref] kept weights loaded", flush=True)

    def get_f16(self, role):
        if role in self._f16cache: return self._f16cache[role]
        f, shape = self.f16[role]
        a = np.fromfile(os.path.join(self.d, f), dtype=np.float16)
        dims = [int(x) for x in shape.split("x")] if "x" in shape else [int(shape)]
        a = a.reshape(dims).astype(np.float32)
        self._f16cache[role] = a
        return a

    def get_w4(self, role, cache=True):
        if cache and role in self._w4cache: return self._w4cache[role]
        pre, N, K = self.w4[role]
        KQ = K // 4; NG = K // 32; SBP = (NG + 15) // 16 * 16
        words = np.fromfile(os.path.join(self.d, pre + ".w4.bin"), dtype=np.uint16).reshape(N, KQ)
        s = np.fromfile(os.path.join(self.d, pre + ".s.bin"), dtype=np.float16).reshape(N, SBP)[:, :NG]
        sz = np.fromfile(os.path.join(self.d, pre + ".sz.bin"), dtype=np.float16).reshape(N, SBP)[:, :NG]
        q = np.empty((N, K), dtype=np.uint8)
        qr = q.reshape(N, 4, KQ)
        for p in range(4):
            qr[:, p, :] = ((words >> (4 * p)) & 0xF).astype(np.uint8)
        q16 = q.reshape(N, NG, 32).astype(np.float32)
        w = q16 * s[:, :, None].astype(np.float32) - sz[:, :, None].astype(np.float32)
        w = w.reshape(N, K).astype(np.float16)
        if cache and not (role.startswith("T.WTE") or role.startswith("T.LM")):
            self._w4cache[role] = w
        return w

    def rope(self, x, pos):
        # x [heads, 64], 只转前 RD=32
        rd = self.RD; half = rd // 2
        i = np.arange(half, dtype=np.float32)
        freq = 1.0 / (self.theta ** (i / half))
        ang = pos * freq
        c = np.cos(ang).astype(np.float32); s = np.sin(ang).astype(np.float32)
        xr = x[:, :half].copy(); xi = x[:, half:rd].copy()
        x[:, :half] = xr * c - xi * s
        x[:, half:rd] = xr * s + xi * c
        return x

    def forward(self, tok, pos, kcs, vcs, dump_dir=None):
        H = self.H
        x = self.wte[tok].astype(np.float32)
        dump_pos = int(os.environ.get("MD3_DUMPPOS", "0"))
        dd = dump_dir if (dump_dir and pos == dump_pos) else None
        if dd: np.asarray(x, np.float16).tofile(os.path.join(dd, "ref_emb.bin"))
        for li in range(self.L):
            kc = kcs[li]; vc = vcs[li]
            pre = "T.L%d" % li
            lnw = self.get_f16(pre + ".LN.W"); lnb = self.get_f16(pre + ".LN.B")
            xn = (x - x.mean()) / np.sqrt(x.var() + self.eps) * lnw + lnb
            if dd: np.asarray(xn, np.float16).tofile(os.path.join(dd, "ref_L%d_xn.bin" % li))
            qkv = xn @ self.get_w4(pre + ".QKV").astype(np.float32).T + self.get_f16(pre + ".QKV.B")
            if dd: np.asarray(qkv, np.float16).tofile(os.path.join(dd, "ref_L%d_qkv.bin" % li))
            # tau
            def gelu(v):
                return 0.5 * v * (1 + np.tanh(0.7978845608028654 * (v + 0.044715 * v ** 3)))
            feat = gelu(qkv)
            wq = self.get_f16(pre + ".TAU.WQ"); wv = self.get_f16(pre + ".TAU.WV")
            a = self.get_f16(pre + ".TAU.A")
            tokq = np.tanh(feat @ wq.T)
            tokv = np.tanh(feat @ wv.T)
            tp = 0.5 + 1.0 / (1.0 + np.exp(-a * np.log(pos + 1.0)))
            tauq = (tokq + tp).astype(np.float32); tauv = (tokv + tp).astype(np.float32)
            if dd:
                np.asarray(tauq, np.float32).tofile(os.path.join(dd, "ref_L%d_tauq.bin" % li))
                np.asarray(tauv, np.float32).tofile(os.path.join(dd, "ref_L%d_tauv.bin" % li))
            if os.environ.get("MD3_DEBUG"):
                print("[ref dbg] alpha=%.4f %.4f %.4f tokq=%.4f tokv=%.4f tauq=%.4f tauv=%.4f" % (
                    a[0], a[1], a[2], tokq[0], tokv[0], tauq[0], tauv[0]))
            q = qkv[:2048].reshape(32, 64) * tauq[:, None] * 0.125
            k = qkv[2048:4096].reshape(32, 64)
            v = qkv[4096:].reshape(32, 64) * tauv[:, None]
            q = self.rope(q.astype(np.float32), pos).astype(np.float16)
            k = self.rope(k.astype(np.float32), pos).astype(np.float16)
            v = v.astype(np.float16)
            if dd: np.asarray(v.reshape(2048), np.float16).tofile(os.path.join(dd, "ref_L%d_v.bin" % li))
            kc[:, pos, :] = k; vc[:, pos, :] = v
            sc = np.einsum("hd,hkd->hk", q.astype(np.float32), kc[:, :pos + 1].astype(np.float32))
            sc = sc - sc.max(axis=-1, keepdims=True)
            pr = np.exp(sc); pr /= pr.sum(axis=-1, keepdims=True)
            o = np.einsum("hk,hkd->hd", pr.astype(np.float32), vc[:, :pos + 1].astype(np.float32))
            if dd: np.asarray(o.reshape(2048), np.float16).tofile(os.path.join(dd, "ref_L%d_attn.bin" % li))
            o = o.reshape(2048) @ self.get_w4(pre + ".O").astype(np.float32).T + self.get_f16(pre + ".O.B")
            if dd: np.asarray(o, np.float16).tofile(os.path.join(dd, "ref_L%d_proj.bin" % li))
            x = x + o
            if li < self.MOE0:
                h = xn @ self.get_w4(pre + ".FC1").astype(np.float32).T + self.get_f16(pre + ".FC1.B")
                if dd: np.asarray(h, np.float16).tofile(os.path.join(dd, "ref_L%d_fc1pre.bin" % li))
                h = gelu(h)
                if dd: np.asarray(h, np.float16).tofile(os.path.join(dd, "ref_L%d_h.bin" % li))
                m = h @ self.get_w4(pre + ".FC2").astype(np.float32).T + self.get_f16(pre + ".FC2.B")
                if dd: np.asarray(m, np.float16).tofile(os.path.join(dd, "ref_L%d_mlp.bin" % li))
            else:
                rl = xn @ self.get_w4(pre + ".ROUTER").astype(np.float32).T + self.get_f16(pre + ".ROUTER.B")
                idx = np.argsort(-rl)[: self.TOPK]
                top = rl[idx]
                e = np.exp(top - top.max()); w = (e / e.sum()).astype(np.float16)
                m = np.zeros(H, np.float32)
                for j, ei in enumerate(idx):
                    w1 = self.get_w4(pre + ".E%d.FC1" % ei, cache=False).astype(np.float32)
                    hh = xn @ w1.T
                    h1, h2 = hh[:self.EI], hh[self.EI:]
                    g = gelu(h1) * (h2 + 1.0)
                    w2 = self.get_w4(pre + ".E%d.FC2" % ei, cache=False).astype(np.float32)
                    m += float(w[j]) * (g @ w2.T)
                    del w1, w2
            x = x + m.astype(np.float32)
            if dd: np.asarray(x, np.float16).tofile(os.path.join(dd, "ref_L%d_out.bin" % li))
        fnw = self.get_f16("T.FN.W"); fnb = self.get_f16("T.FN.B")
        xn = (x - x.mean()) / np.sqrt(x.var() + self.eps) * fnw + fnb
        if dd: np.asarray(xn, np.float16).tofile(os.path.join(dd, "ref_finalxn.bin"))
        logits = xn @ self.lm.astype(np.float32).T + self.get_f16("T.LM.B")
        return logits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--ids", required=True)
    ap.add_argument("--limit-layers", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dump-dir", default="")
    args = ap.parse_args()
    m = Model(args.dir, n_layers=args.limit_layers)
    ids = [int(x) for x in args.ids.split(",")]
    kcs = [np.zeros((m.NH, m.MAXT, m.D), np.float16) for _ in range(m.L)]
    vcs = [np.zeros((m.NH, m.MAXT, m.D), np.float16) for _ in range(m.L)]
    for pos, tok in enumerate(ids):
        logits = m.forward(tok, pos, kcs, vcs, dump_dir=args.dump_dir or None)
        print("[ref] token %d pos %d top1=%d" % (tok, pos, int(np.argmax(logits))), flush=True)
    np.asarray(logits, dtype=np.float32).tofile(args.out)
    print("[ref] logits ->", args.out)


if __name__ == "__main__":
    main()
