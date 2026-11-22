"""Estimate RGB<->depth alignment (rotation/scale) for the U3V ToF snapshot.

Compares RGB edge magnitude against depth discontinuity edges for candidate
transforms, returning the best candidate by normalized correlation.
"""
import numpy as np
from PIL import Image
from scipy import ndimage

D = r"D:\项目4\orangepi_aipro\code\depth_completion\data"
rgb = np.asarray(Image.open(D + r"\color.jpg").convert("L").resize((640, 480), Image.BILINEAR), dtype=np.float32)
dep = np.asarray(Image.open(D + r"\depth16.png"), dtype=np.float32) / 1000.0
valid = dep > 0

# depth edges: gradient magnitude on filled depth (inpaint holes with nearest)
filled = dep.copy()
if (~valid).any():
    ind = ndimage.distance_transform_edt(~valid, return_distances=False, return_indices=True)
    filled = dep[tuple(ind)]
gx = ndimage.sobel(filled, 1); gy = ndimage.sobel(filled, 0)
de = np.hypot(gx, gy)
de[~valid] = 0
de = de / (de.std() + 1e-9)

rg = rgb - rgb.mean()
rg = np.abs(ndimage.sobel(rg, 1)) + np.abs(ndimage.sobel(rg, 0))
rg = rg / (rg.std() + 1e-9)


def corr(a, b, m):
    aa = a[m] - a[m].mean(); bb = b[m] - b[m].mean()
    return float((aa * bb).mean() / (aa.std() * bb.std() + 1e-9))


m = valid & (de > np.percentile(de[valid], 60))
print("mask pixels", m.sum())
for rot in (0, 90, 180, 270):
    for scale in (1.0, 0.8, 0.7, 0.6, 0.5):
        img = Image.fromarray(np.asarray(Image.open(D + r"\color.jpg").convert("L")))
        img = img.rotate(rot, expand=True)
        w, h = img.size
        # crop the central region with the given scale then resize to 640x480
        cw, ch = int(w * scale), int(h * scale)
        left, top = (w - cw) // 2, (h - ch) // 2
        img = img.crop((left, top, left + cw, top + ch)).resize((640, 480), Image.BILINEAR)
        a = np.asarray(img, dtype=np.float32)
        e = np.abs(ndimage.sobel(a - a.mean(), 1)) + np.abs(ndimage.sobel(a - a.mean(), 0))
        e = e / (e.std() + 1e-9)
        print(f"rot={rot:3d} scale={scale:.1f} corr={corr(e, de, m):+.3f}")
