"""fix_norm_centering.py — Qwen3.5 的 Qwen3_5RMSNorm 是 1-centered（有效权重 = 1 + w），
把转换产物里所有 RMSNorm 权重改为 (1 + w)。
注意：linear_attn/norm（RMSNormGated）不是 1-centered，保持不变。
"""
import os
import sys
import numpy as np

roots = sys.argv[1:] or ["converted/qwen38_310b"]
patched = 0
for root in roots:
    targets = []
    lm = os.path.join(root, "model/language_model")
    fin = os.path.join(lm, "norm/weight.f16.bin")
    if os.path.exists(fin):
        targets.append(fin)
    for li in range(64):
        base = os.path.join(lm, "layers", str(li))
        for rel in ("input_layernorm/weight.f16.bin",
                    "post_attention_layernorm/weight.f16.bin",
                    "self_attn/q_norm/weight.f16.bin",
                    "self_attn/k_norm/weight.f16.bin"):
            p = os.path.join(base, rel)
            if os.path.exists(p):
                targets.append(p)
    for p in targets:
        a = np.fromfile(p, dtype=np.float16).astype(np.float32)
        b = (1.0 + a).astype(np.float16)
        b.tofile(p)
        patched += 1
    print("%s: patched %d norm files" % (root, len(targets)))
print("total patched:", patched)
