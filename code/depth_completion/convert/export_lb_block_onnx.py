"""Export a single LingBot-Depth ViT-L/14 transformer block to ONNX.

Purpose: the full 321M-param model (fp16 ~640 MB) cannot be pushed to the board
under the 300 MB transfer budget, so we measure the *unit* cost on the NPU
(one ViT-L block at the exact sequence length used by the full model) and
multiply by 24 blocks + measured/analytic patch-embed & decoder cost.

The block is the official DINOv2 block used by LingBot-Depth (LayerNorm, qkv,
attention, proj, MLP + GELU, LayerScale).  SDPA / xFormers attention is replaced
by explicit matmul-softmax so the graph is plain ONNX opset 13.

Usage:
  set PYTHONPATH=...\\lingbot-depth ; set LB_CKPT=...\\model.pt
  python convert/export_lb_block_onnx.py --L 2401 --out D:\\...\\lb_block_L2401.onnx
"""
import argparse, os, sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lingbot-depth"))
os.environ.setdefault("XFORMERS_DISABLED", "1")

from mdm.model.v2 import MDMModel  # noqa: E402


class AttnManual(nn.Module):
    """Plain ONNX-exportable multi-head self attention (same math as MemEffAttention)."""

    def __init__(self, attn):
        super().__init__()
        self.num_heads = attn.num_heads
        self.scale = attn.scale
        self.qkv = attn.qkv
        self.proj = attn.proj

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = torch.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(B, N, C)
        return self.proj(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=int, required=True, help="sequence length = 1 + 2*num_tokens (+registers)")
    ap.add_argument("--blk", type=int, default=0, help="which block index to export")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ck = torch.load(os.environ["LB_CKPT"], map_location="cpu", weights_only=True)
    model = MDMModel(**ck["model_config"])
    model.load_state_dict(ck["model"], strict=False)
    block = model.encoder.backbone.blocks[args.blk].eval()
    block.attn = AttnManual(block.attn).eval()
    for p in block.parameters():
        p.requires_grad_(False)

    d = model.encoder.backbone.embed_dim
    x = torch.randn(1, args.L, d)
    outp = args.out
    Path(outp).parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        block, (x,), outp, input_names=["x"], output_names=["y"],
        opset_version=13, do_constant_folding=True, dynamo=False,
    )
    import onnx
    m = onnx.load(outp)
    onnx.checker.check_model(m)
    print(f"exported {outp}  L={args.L} d={d} nodes={len(m.graph.node)} "
          f"size={os.path.getsize(outp)/1e6:.1f}MB")
    # analytic FLOPs of this block
    flops = (3 * 2 * args.L * d * d) + (2 * 2 * args.L * args.L * d) + (2 * args.L * d * d) + (16 * args.L * d * d)
    print(f"block GFLOP = {flops/1e9:.2f}   full-24-block GFLOP = {24*flops/1e9:.1f}")


if __name__ == "__main__":
    main()
