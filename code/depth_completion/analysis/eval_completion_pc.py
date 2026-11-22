"""Evaluate depth-only completion vs DAv2 prior on real ToF data (PC, onnxruntime).

Protocol (no external ground truth available):
  * take the real ToF frame; mask out valid pixels (random 10% or block holes)
  * run a completion method using only the remaining depth (+optional RGB prior)
  * RMSE/MAE (mm) against the held-out true ToF depth.

Methods:
  M0 raw-tof        : invalid pixels = 0 (coverage = raw valid on kept pixels)
  M1 multiscale-fill: classical push-pull style fill using kept depth only
  M2 dav2-grayscale : DAv2-small on a 3-channel rendering of the sparse depth
  M3 dav2-rgb       : DAv2-small on the RGB image (coarse aligned prior)
"""
import json
import time

import numpy as np
import onnxruntime as ort
from PIL import Image
from scipy import ndimage

ROOT = r"D:\项目4\orangepi_aipro\code\depth_completion"
D = ROOT + r"\data"
P = ROOT + r"\results\pc_ref"
ONNX = r"D:\项目4\orangepi_aipro\tmp\hfcache\models--onnx-community--depth-anything-v2-small\snapshots\4472b7362082ad9968fee890ca0f1e5aca36b93d\onnx\model.onnx"
sess = ort.InferenceSession(ONNX, providers=["CPUExecutionProvider"])
INAME = sess.get_inputs()[0].name


def dav2(img_u8, h=238, w=308):
    """img_u8: HxWx3 uint8 -> relative inverse depth at (h,w)."""
    im = Image.fromarray(img_u8).resize((w, h), Image.BILINEAR)
    x = np.asarray(im, dtype=np.float32) / 255.0
    x = (x - np.array([0.485, 0.456, 0.406], np.float32)) / np.array([0.229, 0.224, 0.225], np.float32)
    x = x.transpose(2, 0, 1)[None].astype(np.float32)
    out = sess.run(None, {INAME: x})[0][0]
    return out.astype(np.float32)


def multiscale_fill(d, mask, levels=5, iters=60):
    """Classical push-pull / diffusion fill of invalid pixels (mask=True = invalid)."""
    d = d.copy().astype(np.float32)
    v = (~mask).astype(np.float32)
    d[~np.isfinite(d)] = 0
    d = d * v
    pyr_d, pyr_v = [d], [v]
    cur_d, cur_v = d, v
    for _ in range(levels):
        cur_d = ndimage.uniform_filter(cur_d, 3) * 2
        cur_v = ndimage.uniform_filter(cur_v, 3) * 2
        pyr_d.append(cur_d)
        pyr_v.append(cur_v)
    est = np.where(pyr_v[-1] > 1e-6, pyr_d[-1] / np.maximum(pyr_v[-1], 1e-6), 0)
    for lvl in range(len(pyr_d) - 2, -1, -1):
        est = ndimage.zoom(est, (pyr_d[lvl].shape[0] / est.shape[0], pyr_d[lvl].shape[1] / est.shape[1]), order=1)
        w = pyr_v[lvl]
        est = np.where(w > 1e-6, (pyr_d[lvl] / np.maximum(w, 1e-6)) * 0.5 + est * 0.5, est)
    for _ in range(iters):
        est = ndimage.uniform_filter(est, 3)
    return est


def render_depth(d, valid):
    """Render sparse depth as a 3-channel grayscale image for DAv2."""
    z = np.where(valid, d, np.nan)
    lo, hi = np.nanpercentile(z, 2), np.nanpercentile(z, 98)
    t = np.clip((np.nan_to_num(z, nan=hi) - lo) / (hi - lo + 1e-9), 0, 1)
    g = (t * 255).astype(np.uint8)
    g[~valid] = 0
    return np.stack([g, g, g], -1)


def fit_affine(prior_full, disp_target, mask_valid):
    """fit disp_target ~ a*prior + b robustly (IRLS)."""
    x = prior_full[mask_valid].astype(np.float64)
    y = disp_target[mask_valid].astype(np.float64)
    if len(x) < 100:
        return 1.0, 0.0
    ok = np.ones_like(x, bool)
    a, b = 1.0, 0.0
    for _ in range(8):
        A = np.stack([x[ok], np.ones(int(ok.sum()))], 1)
        a, b = np.linalg.lstsq(A, y[ok], rcond=None)[0]
        r = np.abs(y - (a * x + b))
        ok = r < np.percentile(r, 70)
    return float(a), float(b)


def main():
    dep = np.asarray(Image.open(D + r"\depth16.png"), dtype=np.float32) / 1000.0
    H, W = dep.shape
    valid = dep > 0
    rgb = np.asarray(Image.open(D + r"\color.jpg").convert("RGB"))
    rng = np.random.default_rng(0)

    rand_mask = np.zeros_like(valid)
    idx = np.flatnonzero(valid.ravel())
    pick = rng.choice(idx, int(0.10 * len(idx)), replace=False)
    rand_mask.ravel()[pick] = True
    block_mask = np.zeros_like(valid)
    for cy, cx in [(120, 160), (150, 420), (330, 250), (260, 520), (380, 100), (60, 300)]:
        block_mask[cy - 23:cy + 22, cx - 23:cx + 22] = True
    masks = {"random10pct": rand_mask & valid, "blocks45": block_mask & valid}

    results = []
    for name, hole in masks.items():
        keep = valid & ~hole
        truth = dep[hole]
        row = dict(mask=name, n=int(hole.sum()), methods={})
        row["methods"]["M0_raw_tof"] = dict(rmse_mm=float("nan"), mae_mm=float("nan"),
                                            coverage=float((~hole).mean()))
        d_fill = multiscale_fill(np.where(keep, dep, 0), ~keep)
        pred = d_fill[hole]
        row["methods"]["M1_multiscale_fill"] = dict(
            rmse_mm=float(np.sqrt(np.mean((pred - truth) ** 2)) * 1000),
            mae_mm=float(np.mean(np.abs(pred - truth)) * 1000))
        t0 = time.time()
        render = render_depth(np.where(keep, dep, 0), keep)
        pr = dav2(render)
        pr_full = np.asarray(Image.fromarray(pr).resize((W, H), Image.BILINEAR))
        disp = 1.0 / np.maximum(dep, 0.05)
        a, b = fit_affine(pr_full, disp, keep)
        pred2 = 1.0 / np.maximum(a * pr_full + b, 1e-3)
        row["methods"]["M2_dav2_on_depthrender"] = dict(
            rmse_mm=float(np.sqrt(np.mean((pred2[hole] - truth) ** 2)) * 1000),
            mae_mm=float(np.mean(np.abs(pred2[hole] - truth)) * 1000), sec=round(time.time() - t0, 2))
        t0 = time.time()
        rgb_small = np.asarray(Image.fromarray(rgb).resize((W, H), Image.BILINEAR))
        pr3 = dav2(rgb_small)
        pr3_full = np.asarray(Image.fromarray(pr3).resize((W, H), Image.BILINEAR))
        a3, b3 = fit_affine(pr3_full, disp, keep)
        pred3 = 1.0 / np.maximum(a3 * pr3_full + b3, 1e-3)
        row["methods"]["M3_dav2_on_rgb_uncalib"] = dict(
            rmse_mm=float(np.sqrt(np.mean((pred3[hole] - truth) ** 2)) * 1000),
            mae_mm=float(np.mean(np.abs(pred3[hole] - truth)) * 1000), sec=round(time.time() - t0, 2))
        results.append(row)
        print(name, {k: (round(v["rmse_mm"], 1) if v["rmse_mm"] == v["rmse_mm"] else None,
                         round(v["mae_mm"], 1) if v["mae_mm"] == v["mae_mm"] else None)
                     for k, v in row["methods"].items()}, flush=True)

    open(P + r"\completion_eval_pc.json", "w").write(json.dumps(results, indent=1))
    print("saved", P + r"\completion_eval_pc.json")


if __name__ == "__main__":
    main()
