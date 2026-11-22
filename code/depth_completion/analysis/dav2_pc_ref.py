"""Run Depth-Anything-V2-Small ONNX on the PC (CPU) as reference + prior.

DAv2 outputs affine-invariant inverse depth (larger = closer).  At 518x434 the
model is at ~native training resolution but takes ~1 min/frame on this CPU.
"""
import argparse, time
import numpy as np
import onnxruntime as ort
from PIL import Image

ONNX = r"D:\项目4\orangepi_aipro\tmp\hfcache\models--onnx-community--depth-anything-v2-small\snapshots\4472b7362082ad9968fee890ca0f1e5aca36b93d\onnx\model.onnx"
D = r"D:\项目4\orangepi_aipro\code\depth_completion\data"
OUT = r"D:\项目4\orangepi_aipro\code\depth_completion\results\pc_ref"


def colorize(d):
    dv = d[np.isfinite(d)]
    vmin, vmax = np.percentile(dv, 1), np.percentile(dv, 99)
    t = np.clip((d - vmin) / (vmax - vmin + 1e-9), 0, 1)
    r = np.clip(34.61 + t * (1172.33 + t * (-10793.56 + t * (33300.12 + t * (-38394.49 + t * 14825.05)))), 0, 255)
    g = np.clip(23.31 + t * (557.33 + t * (1225.33 + t * (-3574.96 + t * (3202.75 + t * -895.6)))), 0, 255)
    b = np.clip(27.2 + t * (3211.1 + t * (-15327.97 + t * (27814.0 + t * (-22569.18 + t * 6838.66)))), 0, 255)
    return np.stack([r, g, b], -1).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h", type=int, default=518)
    ap.add_argument("--w", type=int, default=518)
    ap.add_argument("--tag", default="dav2_518")
    args = ap.parse_args()
    h, w = (args.h // 14) * 14, (args.w // 14) * 14
    im = Image.open(D + r"\color.jpg").convert("RGB").resize((w, h), Image.BILINEAR)
    x = np.asarray(im, dtype=np.float32) / 255.0
    x = (x - np.array([0.485, 0.456, 0.406], np.float32)) / np.array([0.229, 0.224, 0.225], np.float32)
    x = x.transpose(2, 0, 1)[None].astype(np.float32)
    sess = ort.InferenceSession(ONNX, providers=["CPUExecutionProvider"])
    iname = sess.get_inputs()[0].name
    t0 = time.time()
    out = sess.run(None, {iname: x})[0][0]
    dt = time.time() - t0
    print(f"dav2 {w}x{h} infer {dt:.1f}s out {out.shape} range {out.min():.3f}..{out.max():.3f}")
    np.save(f"{OUT}/{args.tag}.npy", out.astype(np.float32))
    Image.fromarray(colorize(out)).save(f"{OUT}/{args.tag}.png")
    Image.fromarray(np.asarray(im)).save(f"{OUT}/{args.tag}_input.png")


if __name__ == "__main__":
    main()
