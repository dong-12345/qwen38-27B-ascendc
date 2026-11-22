#!/usr/bin/env python3
"""STANDARD comparison figure for all fusion results.

Every comparison from now on uses this 6-panel layout so the fusion can be
verified against the raw monocular depth estimate:

  (1) RGB
  (2) raw monocular depth estimate (model output, un-anchored, relative)
  (3) raw monocular depth sampled onto the depth grid (still un-anchored)
  (4) raw ToF depth (reference, holes black)
  (5) anchored prior (value-matched to metres, same scale as ToF)
  (6) fused completion result

Usage:
  python3 analysis/make_standard_compare.py \
      --rgb data/live_now/color.jpg --depth16 data/live_now/depth16.png \
      --raw-small results/raw_prior_small.npy \
      --raw-grid results/raw_prior_depthgrid.npy \
      --anchored results/anchored_prior.npy \
      --fused results/fused_live_now.npy \
      --out results/standard_compare_live_now.png
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def font(sz, bold=False):
    for fp in ((r"C:\Windows\Fonts\msyhbd.ttc" if bold else r"C:\Windows\Fonts\msyh.ttc"),
               r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\simhei.ttf"):
        try:
            return ImageFont.truetype(fp, sz)
        except Exception:
            continue
    return ImageFont.load_default()


def colorize(d, lo=None, hi=None, vmin_metric=None):
    d = np.asarray(d, np.float32)
    v = np.isfinite(d) & (d > 0)
    if lo is None:
        lo, hi = np.percentile(d[v], 1), np.percentile(d[v], 99)
    t = np.clip((np.nan_to_num(d, nan=lo) - lo) / (hi - lo + 1e-9), 0, 1)
    r = np.clip(34.61 + t * (1172.33 + t * (-10793.56 + t * (33300.12 + t * (-38394.49 + t * 14825.05)))), 0, 255)
    g = np.clip(23.31 + t * (557.33 + t * (1225.33 + t * (-3574.96 + t * (3202.75 + t * -895.6)))), 0, 255)
    b = np.clip(27.2 + t * (3211.1 + t * (-15327.97 + t * (27814.0 + t * (-22569.18 + t * 6838.66)))), 0, 255)
    out = np.stack([r, g, b], -1).astype(np.uint8)
    out[~v] = 0
    return out


def square_panel(a, PW, PH):
    """Put a possibly non-square image into a PW x PH black panel (keep aspect)."""
    im = Image.fromarray(a)
    s = min(PW / im.width, PH / im.height)
    im = im.resize((max(1, int(im.width * s)), max(1, int(im.height * s))), Image.BILINEAR)
    out = Image.new("RGB", (PW, PH), (0, 0, 0))
    out.paste(im, ((PW - im.width) // 2, (PH - im.height) // 2))
    return np.asarray(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rgb", required=True)
    ap.add_argument("--depth16", required=True)
    ap.add_argument("--raw-small", required=True)
    ap.add_argument("--raw-grid", required=True)
    ap.add_argument("--anchored", required=True)
    ap.add_argument("--fused", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="")
    args = ap.parse_args()

    rgb = np.asarray(Image.open(args.rgb).convert("RGB"))
    dep = np.asarray(Image.open(args.depth16)).astype(np.float32) / 1000.0
    raw_small = np.load(args.raw_small)
    raw_grid = np.load(args.raw_grid)
    anch = np.load(args.anchored)
    fused = np.load(args.fused)

    lo, hi = float(np.percentile(raw_small, 1)), float(np.percentile(raw_small, 99))
    rgb_p = square_panel(rgb, 640, 480)
    raw_small_p = square_panel(colorize(raw_small, lo, hi), 640, 480)
    raw_grid_c = colorize(raw_grid, lo, hi)
    raw_grid_c[raw_grid <= 0] = 0
    raw_grid_p = square_panel(raw_grid_c, 640, 480)
    dep_p = square_panel(colorize(np.where(dep > 0, dep, np.nan), 0.3, 5.0), 640, 480)
    anch_c = colorize(np.where(anch > 0, anch, np.nan), 0.3, 5.0)
    fused_c = colorize(fused, 0.3, 5.0)
    panels = [rgb_p, raw_small_p, raw_grid_p, dep_p, square_panel(anch_c, 640, 480), square_panel(fused_c, 640, 480)]
    labels = ["① 原始 RGB", "② 原始深度估计（模型直出，相对深度，未锚定）",
              "③ 原始深度估计（采样到深度帧，未锚定）", "④ 原始 ToF 深度（参照，空洞黑）",
              "⑤ 锚定后先验（值域锚定→米制）", "⑥ 融合补全结果"]

    PW, PH, LAB = 640, 480, 50
    W = PW * len(panels)
    H = 96 + LAB + PH + 16
    cv = Image.new("RGB", (W, H), (18, 18, 18))
    d = ImageDraw.Draw(cv)
    title = args.title or "深度估计 vs 锚定先验 vs 融合（标准对比图）"
    d.text((14, 12), title, font=font(26, True), fill=(240, 240, 240))
    d.text((14, 48), "②③ 为模型直接输出（相对深度，颜色尺度与米制无关）；④ 的黑区是 ToF 传感器失效区；⑤ 按“同相对深度值→ToF锚点深度中位数”映射到米制；⑥ 为最终融合。",
           font=font(15), fill=(175, 175, 175))
    y = 96
    for i, p in enumerate(panels):
        cv.paste(Image.fromarray(p), (i * PW, y + LAB))
        d.text((i * PW + 8, y + 6), labels[i], font=font(17), fill=(205, 205, 205))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    cv.save(args.out)
    print("saved", args.out, cv.size)


if __name__ == "__main__":
    main()
