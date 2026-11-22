#!/usr/bin/env python3
"""Generate test vectors for the AscendC ops.  Run from the project root."""
import os

import numpy as np

SEQ = 64
HEADS = 48


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


if __name__ == "__main__":
    os.makedirs("input", exist_ok=True)
    os.makedirs("input_golden", exist_ok=True)
    gen_add()
    gen_gdn_gating()
    print("inputs written")
