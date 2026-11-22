"""LingBot-Depth compute budget analysis (params / FLOPs / latency model).

Run on the PC:
  set XFORMERS_DISABLED=1
  set PYTHONPATH=...\\lingbot-depth
  set LB_CKPT=...\\model.pt
  python analysis/flops_budget.py

Outputs results/flops_table.json and prints a markdown table used in README.md.
"""
import json, os, sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lingbot-depth"))
os.environ.setdefault("XFORMERS_DISABLED", "1")

from mdm.model.v2 import MDMModel  # noqa: E402
from mdm.utils.geo import normalized_view_plane_uv  # noqa: E402
from torch.utils.flop_counter import FlopCounterMode  # noqa: E402

# measured on the board (single AI core, Cube unit, fp16) - from our perf study
PEAK_FP16_TFLOPS = 1.46
EFF_FP16_TFLOPS = 1.20       # realistic sustained (memory/launch overheads)
BUDGET_TOPS = 5.0
TARGET_FPS = 30.0

D = 1024          # ViT-L embed dim
NBLK = 24


def enc_flops(N, L):
    """ViT-L/14 RGB-D encoder FLOPs (macs*2) for N tokens per modality."""
    per_block = NBLK * (24 * L * D * D + 4 * L * L * D)
    patch = 2 * (3 + 1) * D * 14 * 14 * N          # two patch-embed convs (rgb=3ch, depth=1ch)
    proj = 2 * D * D * N                            # output_projection 1x1 conv
    return per_block + patch + proj


def main():
    ck = torch.load(os.environ["LB_CKPT"], map_location="cpu", weights_only=True)
    model = MDMModel(**ck["model_config"]).eval()
    nparam = sum(p.numel() for p in model.parameters())
    n_backbone = sum(p.numel() for p in model.encoder.parameters())
    print(f"params total {nparam/1e6:.2f}M  backbone {n_backbone/1e6:.2f}M  decoder {(nparam-n_backbone)/1e6:.2f}M")

    # decoder FLOPs measured with torch flop counter at a small base grid, then scaled
    ref_h, ref_w = 15, 20
    uv = [normalized_view_plane_uv(width=ref_w * 2 ** l, height=ref_h * 2 ** l, aspect_ratio=4 / 3)
          for l in range(5)]
    feats = [torch.zeros(1, 1026 if l == 0 else 2, ref_h * 2 ** l, ref_w * 2 ** l) for l in range(5)]
    with torch.no_grad(), FlopCounterMode(display=False) as fc:
        out = model.neck(feats)
        model.depth_head(out)
        model.mask_head(out)
    dec_flops_ref = fc.get_total_flops()
    print(f"decoder FLOPs at base {ref_h}x{ref_w}: {dec_flops_ref/1e9:.2f} GFLOP")

    rows = []
    for level, num_tokens in [(0, 300), (2, 600), (5, 1200), (8, 2100), (9, 2400), ("default", 3600)]:
        aspect = 640 / 480
        base_h = round((num_tokens / aspect) ** 0.5)
        base_w = round((num_tokens * aspect) ** 0.5)
        if abs(base_h * base_w - num_tokens) > 0.05 * num_tokens:  # keep 4:3-ish grid
            pass
        N = base_h * base_w
        L = 1 + 2 * N
        enc = enc_flops(N, L)
        dec = dec_flops_ref * (base_h * base_w) / (ref_h * ref_w)
        tot = enc + dec
        rows.append(dict(
            num_tokens=num_tokens, grid=f"{base_h}x{base_w}",
            vit_input=f"{base_h*14}x{base_w*14}", L=L,
            enc_gflop=round(enc / 1e9, 1), dec_gflop=round(dec / 1e9, 1), total_gflop=round(tot / 1e9, 1),
            t_peak_ms=round(tot / (PEAK_FP16_TFLOPS * 1e12) * 1e3, 1),
            t_eff_ms=round(tot / (EFF_FP16_TFLOPS * 1e12) * 1e3, 1),
            fps_eff=round(1e3 / (tot / (EFF_FP16_TFLOPS * 1e12) * 1e3), 1),
            x_over_budget=round(tot / 1e12 / (BUDGET_TOPS * 1e12 / TARGET_FPS), 1),
        ))
    print("\n| num_tokens | patch grid | ViT input | seq len L | enc GFLOP | dec GFLOP | total GFLOP | @1.46TF ms | @1.2TF ms | fps @1.2TF | x over 5TOPS/30fps budget |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['num_tokens']} | {r['grid']} | {r['vit_input']} | {r['L']} | {r['enc_gflop']} | {r['dec_gflop']} | "
              f"{r['total_gflop']} | {r['t_peak_ms']} | {r['t_eff_ms']} | {r['fps_eff']} | {r['x_over_budget']} |")

    budget = dict(
        budget_tops=BUDGET_TOPS, target_fps=TARGET_FPS,
        gflop_per_frame_at_budget=BUDGET_TOPS * 1e3 / TARGET_FPS,
        measured_peak_fp16_tflops=PEAK_FP16_TFLOPS,
        assumed_eff_fp16_tflops=EFF_FP16_TFLOPS,
        gflop_per_frame_at_eff=1.0 * EFF_FP16_TFLOPS * 1e3 / TARGET_FPS,
        params_total_m=round(nparam / 1e6, 2), params_backbone_m=round(n_backbone / 1e6, 2),
        weights_fp16_mb=round(nparam * 2 / 1e6, 1), weights_int8_mb=round(nparam * 1 / 1e6, 1),
    )
    out = ROOT / "results" / "flops_table.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(budget=budget, rows=rows), indent=1))
    print("\nbudget:", json.dumps(budget))
    print("saved", out)


if __name__ == "__main__":
    main()
