"""PC-side reference inference for LingBot-Depth on real ToF data.

The official inference path needs xFormers (nested-tensor attention) for the
masked-depth-token mode.  We run with enable_depth_mask=False so the token
sequence is dense and fixed (required for ONNX/ATC anyway).

Usage (env):
  set XFORMERS_DISABLED=1
  set PYTHONPATH=D:\\项目4\\orangepi_aipro\\code\\depth_completion\\lingbot-depth
  set LB_CKPT=D:\\项目4\\orangepi_aipro\\tmp\\lbdl\\models\\Robbyant--lingbot-depth-pretrain-vitl-14-v0.5\\snapshots\\master\\model.pt
  python analysis/lb_infer_ref.py --tokens 600 --tag real0001
"""
import argparse, json, os, sys, time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]          # code/depth_completion
REPO = ROOT / "lingbot-depth"
CKPT = Path(os.environ.get("LB_CKPT", ""))
OUT = ROOT / "results" / "pc_ref"

sys.path.insert(0, str(REPO))
os.environ.setdefault("XFORMERS_DISABLED", "1")

from mdm.model.v2 import MDMModel  # noqa: E402


def load_rgb(path, size_wh):
    im = Image.open(path).convert("RGB").resize(size_wh, Image.BILINEAR)
    a = np.asarray(im, dtype=np.float32) / 255.0
    return torch.from_numpy(a).permute(2, 0, 1)[None]


def load_depth(path):
    a = np.asarray(Image.open(path), dtype=np.float32) / 1000.0  # mm -> m
    a[~np.isfinite(a)] = 0.0
    return torch.from_numpy(a)[None]


def colorize(d, vmin=None, vmax=None):
    valid = np.isfinite(d) & (d > 0)
    d2 = np.where(valid, d, np.nan)
    if vmin is None:
        vmin = np.nanpercentile(d2, 2) if valid.any() else 0.0
    if vmax is None:
        vmax = np.nanpercentile(d2, 98) if valid.any() else 1.0
    t = np.clip((d2 - vmin) / (vmax - vmin + 1e-9), 0, 1)
    t = np.nan_to_num(t, nan=0.0)
    r = np.clip(34.61 + t * (1172.33 + t * (-10793.56 + t * (33300.12 + t * (-38394.49 + t * 14825.05)))), 0, 255)
    g = np.clip(23.31 + t * (557.33 + t * (1225.33 + t * (-3574.96 + t * (3202.75 + t * -895.6)))), 0, 255)
    b = np.clip(27.2 + t * (3211.1 + t * (-15327.97 + t * (27814.0 + t * (-22569.18 + t * 6838.66)))), 0, 255)
    rgb = np.stack([r, g, b], -1).astype(np.uint8)
    rgb[~valid] = 0
    return rgb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=600, help="num_tokens (base patch grid H*W)")
    ap.add_argument("--tag", default="real0001")
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--rgb", default=str(ROOT / "data" / "color.jpg"))
    ap.add_argument("--depth", default=str(ROOT / "data" / "depth16.png"))
    ap.add_argument("--ckpt", default=str(CKPT))
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    OUT.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    model = MDMModel(**ck["model_config"])
    model.load_state_dict(ck["model"], strict=False)
    model.eval()
    del ck
    t_load = time.time() - t0
    nparam = sum(p.numel() for p in model.parameters())
    print(f"[load] {t_load:.1f}s params={nparam/1e6:.1f}M", flush=True)

    depth = load_depth(args.depth)
    h, w = depth.shape[-2:]
    img = load_rgb(args.rgb, (w, h))
    print(f"[data] rgb {tuple(img.shape)} depth {tuple(depth.shape)} valid={(depth>0).float().mean().item():.3f}", flush=True)

    t0 = time.time()
    with torch.inference_mode():
        out = model.infer(img, depth_in=depth, num_tokens=args.tokens, use_fp16=False,
                          intrinsics=None, enable_depth_mask=False)
    dt = time.time() - t0
    dref = out["depth"].squeeze().float().numpy()
    raw = depth.squeeze().numpy()
    print(f"[infer] {dt:.1f}s tokens={args.tokens} out {dref.shape} refined_valid={(np.isfinite(dref)&(dref>0)).mean():.3f}", flush=True)

    np.save(OUT / f"{args.tag}_depth_raw.npy", raw)
    np.save(OUT / f"{args.tag}_depth_refined.npy", dref)
    vmin = float(np.percentile(raw[raw > 0], 1))
    vmax = float(np.percentile(raw[raw > 0], 99))
    comp = np.concatenate([colorize(raw, vmin, vmax), colorize(dref, vmin, vmax)], 1)
    Image.fromarray(comp).save(OUT / f"{args.tag}_comparison.png")
    Image.fromarray(np.asarray(Image.open(args.rgb).convert("RGB").resize((w, h)))).save(OUT / f"{args.tag}_rgb.png")

    # held-out consistency: 10% of valid raw pixels, error of refined depth
    rng = np.random.default_rng(0)
    valid = ((raw > 0) & np.isfinite(dref)).ravel()
    idx = np.flatnonzero(valid)
    hold = rng.choice(idx, size=int(0.1 * len(idx)), replace=False)
    flat_ref = np.nan_to_num(dref.ravel(), nan=0.0, posinf=0.0, neginf=0.0)
    err = (flat_ref[hold] - raw.ravel()[hold]) * 1000.0
    err = err[np.isfinite(err)]
    metrics = dict(tag=args.tag, tokens=args.tokens, infer_s=round(dt, 2), load_s=round(t_load, 2),
                   params_m=round(nparam / 1e6, 1),
                   rmse_mm=float(np.sqrt((err ** 2).mean())), mae_mm=float(np.abs(err).mean()),
                   raw_valid=float((raw > 0).mean()),
                   refined_valid=float((np.isfinite(dref) & (dref > 0)).mean()))
    print("[metrics]", json.dumps(metrics), flush=True)
    mpath = OUT / "pc_ref_metrics.json"
    hist = json.loads(mpath.read_text()) if mpath.exists() else []
    hist.append(metrics)
    mpath.write_text(json.dumps(hist, indent=1))


if __name__ == "__main__":
    main()
