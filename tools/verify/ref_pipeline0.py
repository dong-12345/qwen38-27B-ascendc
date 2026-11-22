import os
import numpy as np

BASE = "converted/qwen38_310b"
K = 5120
HIDDEN = 5120


def unpack_w4_row(path, row, k):
    """读 w4 文件的一行并反量化（相位交织 + group-32 s/sz）"""
    kq = k // 4
    with open(path, "rb") as f:
        f.seek(row * kq * 2)
        words = np.frombuffer(f.read(kq * 2), dtype=np.uint16)
    q = np.empty(k, dtype=np.uint8)
    for p in range(4):
        q[p * kq:(p + 1) * kq] = (words >> (4 * p)) & 0xF
    return q


def dequant_row(base_rel, row, k):
    kq = k // 4
    ng = k // 32
    sbp = (ng + 15) // 16 * 16
    q = unpack_w4_row(os.path.join(BASE, base_rel + ".w4.bin"), row, k)
    with open(os.path.join(BASE, base_rel + ".s.bin"), "rb") as f:
        f.seek(row * sbp * 2)
        s = np.frombuffer(f.read(sbp * 2), dtype=np.float16).astype(np.float32)
    with open(os.path.join(BASE, base_rel + ".sz.bin"), "rb") as f:
        f.seek(row * sbp * 2)
        sz = np.frombuffer(f.read(sbp * 2), dtype=np.float16).astype(np.float32)
    s_full = np.repeat(s, 32)[:k]
    sz_full = np.repeat(sz, 32)[:k]
    return q.astype(np.float32) * s_full - sz_full


def rmsnorm(x, w, eps=1e-6):
    ms = (x * x).mean()
    return x / np.sqrt(ms + eps) * w


def main():
    # final norm 权重
    fn = np.fromfile(os.path.join(BASE, "model/language_model/norm/weight.f16.bin"), dtype=np.float16).astype(np.float32)
    print("final norm w: shape", fn.shape, "mean", fn.mean())

    for tok in (248045, 846, 198):
        x = dequant_row("model/language_model/embed_tokens/weight", tok, K)
        xn = rmsnorm(x, fn)
        # lm_head 全量：逐行算太慢，抽 top 候选用分块
        lm_path = os.path.join(BASE, "lm_head/weight")
        n = 248320
        kq = K // 4
        ng = K // 32
        sbp = (ng + 15) // 16 * 16
        # 分块 (8192 行) 算 logits
        chunk = 8192
        best = []
        with open(lm_path + ".w4.bin", "rb") as fw, open(lm_path + ".s.bin", "rb") as fs, open(lm_path + ".sz.bin", "rb") as fz:
            for r0 in range(0, n, chunk):
                r1 = min(n, r0 + chunk)
                words = np.frombuffer(fw.read((r1 - r0) * kq * 2), dtype=np.uint16).reshape(r1 - r0, kq)
                s = np.frombuffer(fs.read((r1 - r0) * sbp * 2), dtype=np.float16).astype(np.float32).reshape(r1 - r0, sbp)
                sz = np.frombuffer(fz.read((r1 - r0) * sbp * 2), dtype=np.float16).astype(np.float32).reshape(r1 - r0, sbp)
                q = np.empty((r1 - r0, K), dtype=np.float32)
                for p in range(4):
                    q[:, p * kq:(p + 1) * kq] = ((words >> (4 * p)) & 0xF).astype(np.float32)
                sf = np.repeat(s[:, :ng], 32, axis=1)
                szf = np.repeat(sz[:, :ng], 32, axis=1)
                # logits = (q*s - sz) @ xn  =  (q*s)@xn - sz@xn
                logits = (q * sf) @ xn - szf @ xn
                idx = np.argsort(-logits)[:5]
                for i in idx:
                    best.append((float(logits[i]), int(r0 + i)))
        best.sort(reverse=True)
        print("token %d -> top5:" % tok, [(t, round(v, 3)) for v, t in best[:5]])


if __name__ == "__main__":
    main()
