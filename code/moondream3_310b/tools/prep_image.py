#!/usr/bin/env python3
"""prep_image.py — 单 crop-window 简化版图像预处理（Moondream 官方 crop 逻辑的 n=2 特例）。

把图像 resize 到 378x378（LANCZOS），生成两个相同 crop（global + local），
归一化 (x/255-0.5)/0.5，输出 fp16 [2,3,378,378] 到 crops.bin。
"""
import argparse, sys
import numpy as np
from PIL import Image

CROP = 378


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", default="crops.bin")
    args = ap.parse_args()
    img = Image.open(args.image).convert("RGB").resize((CROP, CROP), Image.Resampling.LANCZOS)
    arr = np.asarray(img).astype(np.float32) / 255.0
    arr = (arr - 0.5) / 0.5
    chw = np.transpose(arr, (2, 0, 1))          # [3,378,378]
    crops = np.stack([chw, chw], axis=0).astype(np.float16)  # [2,3,378,378]
    crops.tofile(args.out)
    print("crops:", crops.shape, "->", args.out)


if __name__ == "__main__":
    main()
