"""make_promptda_compare.py - 6-panel figure for the PromptDA depth completion run."""
import os
import sys

import numpy as np
import cv2

DC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(DC, "deploy"))
from pd_core import colorize  # noqa: E402


def save(path, img):
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise RuntimeError("encode failed " + path)
    buf.tofile(path)


def panel(img, title, font=0.6):
    img = img.copy()
    cv2.rectangle(img, (0, 0), (img.shape[1], 30), (0, 0, 0), -1)
    cv2.putText(img, title, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, font, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def build(rgb, tof_mm, prompt_m, pred_m, fused_mm, metrics=None, vmax=8.0):
    tof_m = tof_mm.astype(np.float32) * 0.001
    fused_m = fused_mm.astype(np.float32) * 0.001
    err = np.abs(pred_m - tof_m)
    err[tof_mm == 0] = 0
    panels = [
        panel(cv2.resize(rgb, (640, 480), interpolation=cv2.INTER_AREA) if rgb.shape[:2] != (480, 640) else rgb.copy(),
              "1  RGB"),
        panel(colorize(tof_m, 0.3, vmax), "2  ToF raw (%d%% valid)" % int((tof_mm > 0).mean() * 100)),
        panel(colorize(prompt_m, 0.3, vmax), "3  depth prompt (ToF + fill)"),
        panel(colorize(pred_m, 0.3, vmax), "4  PromptDA dense"),
        panel(colorize(fused_m, 0.3, vmax), "5  fused output (ToF kept)"),
        panel(cv2.applyColorMap(np.clip(err / 1.0 * 255, 0, 255).astype(np.uint8), cv2.COLORMAP_MAGMA),
              "6  |PromptDA - ToF| (0-1 m)"),
    ]
    canvas = np.zeros((2 * 480, 3 * 640, 3), np.uint8)
    for i, p in enumerate(panels):
        r, c = divmod(i, 3)
        canvas[r * 480:(r + 1) * 480, c * 640:(c + 1) * 640] = p
    if metrics:
        txt = "  ".join("%s=%s" % (k, v) for k, v in metrics.items())
        cv2.rectangle(canvas, (0, 960 - 34), (1920, 960), (0, 0, 0), -1)
        cv2.putText(canvas, txt, (10, 960 - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1, cv2.LINE_AA)
    return canvas


if __name__ == "__main__":
    run = sys.argv[1] if len(sys.argv) > 1 else "results/live_run"
    rgb = cv2.imread(os.path.join(run, "rgb.png"))
    tof = np.load(os.path.join(run, "tof.npy"))
    prompt = np.load(os.path.join(run, "prompt.npy"))
    pred = np.load(os.path.join(run, "pred.npy"))
    fused = np.load(os.path.join(run, "fused.npy"))
    out = os.path.join(run, "promptda_compare.png")
    save(out, build(rgb, tof, prompt, pred, fused))
    print("saved", out)
