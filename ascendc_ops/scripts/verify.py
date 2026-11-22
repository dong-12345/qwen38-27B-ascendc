#!/usr/bin/env python3
"""Compare NPU outputs against numpy reference. Run from project root."""
import sys

import numpy as np

SEQ = 64
HEADS = 48


def cmp(name, out_path, ref_path, dtype, rtol=2e-3, atol=2e-3):
    out = np.fromfile(out_path, dtype=dtype)
    ref = np.fromfile(ref_path, dtype=dtype)
    if out.size != ref.size:
        print("[FAIL] %-24s size mismatch out=%d ref=%d" % (name, out.size, ref.size))
        return False
    if out.size == 0:
        print("[FAIL] %-24s empty" % name)
        return False
    out64 = out.astype(np.float64)
    ref64 = ref.astype(np.float64)
    abs_err = np.abs(out64 - ref64)
    denom = np.maximum(np.abs(ref64), 1e-6)
    max_abs = float(abs_err.max())
    max_rel = float((abs_err / denom).max())
    mean_abs = float(abs_err.mean())
    # hardware fp32 vs numpy float64 reference: allow tiny numeric noise
    ok = max_abs <= atol + rtol * float(np.abs(ref64).max())
    flag = "PASS" if ok else "FAIL"
    print("[%s] %-24s n=%-7d max_abs=%.3e max_rel=%.3e mean_abs=%.3e  (ref max %.3f)"
          % (flag, name, out.size, max_abs, max_rel, mean_abs, float(np.abs(ref64).max())))
    return ok


def main():
    ok = True
    ok &= cmp("add.output_z", "output/output_z.bin", "input_golden/golden.bin", np.float16,
              rtol=2e-3, atol=2e-3)
    ok &= cmp("gdn_gating.g", "output/gdn_gating_g.bin", "input_golden/gdn_gating_g.bin",
              np.float32, rtol=2e-3, atol=2e-3)
    ok &= cmp("gdn_gating.beta", "output/gdn_gating_beta.bin",
              "input_golden/gdn_gating_beta.bin", np.float32, rtol=2e-3, atol=2e-3)
    print("ALL PASS" if ok else "SOME FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
