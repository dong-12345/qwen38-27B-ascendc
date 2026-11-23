#!/usr/bin/env python3
"""compare_logits.py — 比较板端 engine 与 numpy 参考的 logits。"""
import sys
import numpy as np

a = np.fromfile(sys.argv[1], dtype=np.float32)   # ref
b = np.fromfile(sys.argv[2], dtype=np.float32)   # board
n = min(len(a), len(b))
a, b = a[:n], b[:n]
d = np.abs(a - b)
print("len:", n)
print("max_abs: %.5f  mean_abs: %.5f" % (d.max(), d.mean()))
print("ref top1: %d (%.4f)   board top1: %d (%.4f)" % (int(np.argmax(a)), a.max(), int(np.argmax(b)), b.max()))
ta = np.argsort(-a)[:5]; tb = np.argsort(-b)[:5]
print("ref top5:  ", ta.tolist())
print("board top5:", tb.tolist())
print("top5 overlap:", len(set(ta.tolist()) & set(tb.tolist())))
print("cosine: %.6f" % (float(a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)))
