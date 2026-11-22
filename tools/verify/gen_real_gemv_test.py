"""gen_real_gemv_test.py — 用真实 in_proj_qkv 权重（前 N 行）构造 w4a16_gemv_gen 测试数据，
用于判断「真实权重 (N=10240) 的 GEMV 是否算错」。"""
import os
import numpy as np

BASE = "converted/qwen38_310b"
L = "model/language_model/layers/10/"
K = 17408
N = 5120          # down_proj 行数
OUT_IN = "ascendc_ops/input"
OUT_GOLD = "ascendc_ops/input_golden"
os.makedirs(OUT_IN, exist_ok=True)
os.makedirs(OUT_GOLD, exist_ok=True)

# act = 真实 xn（layer10 的 rmsnorm 输出，token 846）
exec(open("tmp/cmp_stages.py", encoding="utf-8").read().split("def main()")[0])
x = np.fromfile("tmp/p1_eng_post.bin", dtype=np.float16).astype(np.float32)
ln1 = lf(L + "post_attention_layernorm/weight.f16.bin")
xn = rmsnorm(x, ln1).astype(np.float16)

kq = K // 4
ng = K // 32
sbp = (ng + 15) // 16 * 16
src = os.path.join(BASE, L + "mlp/down_proj")
with open(src + ".w4.bin", "rb") as f:
    words = np.frombuffer(f.read(N * kq * 2), dtype=np.uint16).reshape(N, kq)
with open(src + ".s.bin", "rb") as f:
    sp = np.frombuffer(f.read(N * sbp * 2), dtype=np.float16).reshape(N, sbp)
with open(src + ".sz.bin", "rb") as f:
    szp = np.frombuffer(f.read(N * sbp * 2), dtype=np.float16).reshape(N, sbp)

# 反量化 + golden（fp32 累加，输出 fp16）
q = np.empty((N, K), dtype=np.float32)
for p in range(4):
    q[:, p * kq:(p + 1) * kq] = ((words >> (4 * p)) & 0xF).astype(np.float32)
s_f = np.repeat(sp[:, :ng].astype(np.float32), 32, axis=1)
sz_f = np.repeat(szp[:, :ng].astype(np.float32), 32, axis=1)
w = q * s_f - sz_f
gold = (w.astype(np.float64) @ xn.astype(np.float64)).astype(np.float16)

xn.tofile(os.path.join(OUT_IN, "gact.bin"))
words.tofile(os.path.join(OUT_IN, "gwq.bin"))
sp.tofile(os.path.join(OUT_IN, "gsc.bin"))
szp.tofile(os.path.join(OUT_IN, "gsz.bin"))
gold.tofile(os.path.join(OUT_GOLD, "gout.bin"))

# 同时保存权重前 4 行的参考，便于对照
print("wrote: act %d, wq %d, sc %d, sz %d, gold %d"
      % (xn.size, words.size, sp.size, szp.size, gold.size))
print("gold[:6] =", gold[:6].tolist())
print("|w| mean = %.5f" % np.abs(w).mean())
