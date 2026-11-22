"""Coarse RGB->depth registration for the ToF snapshot (no calibration available).

Searches similarity transform (scale, tx, ty, optional h/v flip) that maximizes
normalized cross-correlation between RGB gradient magnitude and ToF depth
gradient magnitude.
"""
import numpy as np
from PIL import Image
from scipy import ndimage

D = r"D:\项目4\orangepi_aipro\code\depth_completion\data"
W, H = 640, 480
rgb = np.asarray(Image.open(D + r"\color.jpg").convert("L"), dtype=np.float32)
dep = np.asarray(Image.open(D + r"\depth16.png"), dtype=np.float32) / 1000.0
valid = dep > 0
ind = ndimage.distance_transform_edt(~valid, return_distances=False, return_indices=True)
filled = dep[tuple(ind)]
gy = np.abs(ndimage.sobel(filled, 0)); gx = np.abs(ndimage.sobel(filled, 1))
dg = (gx + gy)
m = valid & (dg < np.percentile(dg[valid], 98))     # exclude depth holes/discontinuities
dg = dg / (dg[m].std() + 1e-9)
dg = dg - dg[m].mean()


def warp_score(s, tx, ty, flip, rot):
    h0, w0 = rgb.shape
    im = Image.fromarray(rgb.astype(np.uint8))
    if flip:
        im = im.transpose(Image.FLIP_LEFT_RIGHT)
    if rot:
        im = im.transpose(Image.ROTATE_180)
    # inverse map: dst(x',y') sampled from src at ((x'-cx-tx)/s + cx, ...)
    cx, cy = W / 2, H / 2
    sx, sy = (w0 / W) / s, (h0 / H) / s
    ax = cx - cx / s - tx / s
    ay = cy - cy / s - ty / s
    im2 = im.transform((W, H), Image.AFFINE, (sx, 0, ax * (w0 / W), 0, sy, ay * (h0 / H)),
                       resample=Image.BILINEAR)
    a = np.asarray(im2, dtype=np.float32)
    eg = np.abs(ndimage.sobel(a, 0)) + np.abs(ndimage.sobel(a, 1))
    e = eg[m]
    e = e - e.mean()
    den = e.std() + 1e-9
    return float((e * (dg[m] - 0)).mean() / den)


best = (-1, None)
for flip in (0, 1):
    for rot in (0, 1):
        for s in np.arange(0.5, 1.55, 0.1):
            for tx in range(-160, 161, 32):
                for ty in range(-160, 161, 32):
                    c = warp_score(s, tx, ty, flip, rot)
                    if c > best[0]:
                        best = (c, (round(float(s), 2), tx, ty, flip, rot))
print("coarse best", best)
# refine
c0, (s0, tx0, ty0, flip0, rot0) = best
for s in np.arange(s0 - 0.1, s0 + 0.11, 0.025):
    for tx in range(tx0 - 40, tx0 + 41, 10):
        for ty in range(ty0 - 40, ty0 + 41, 10):
            c = warp_score(s, tx, ty, flip0, rot0)
            if c > best[0]:
                best = (c, (round(float(s), 3), tx, ty, flip0, rot0))
print("refined best", best)

c0, (s, tx, ty, flip, rot) = best
im = Image.fromarray(rgb.astype(np.uint8))
if flip:
    im = im.transpose(Image.FLIP_LEFT_RIGHT)
if rot:
    im = im.transpose(Image.ROTATE_180)
w0, h0 = rgb.shape[1], rgb.shape[0]
cx, cy = W / 2, H / 2
sx, sy = (w0 / W) / s, (h0 / H) / s
ax = cx - cx / s - tx / s
ay = cy - cy / s - ty / s
im2 = im.transform((W, H), Image.AFFINE, (sx, 0, ax * (w0 / W), 0, sy, ay * (h0 / H)), resample=Image.BILINEAR)
im2.save(r"D:\项目4\orangepi_aipro\tmp\rgb_registered.png")
print("saved tmp/rgb_registered.png", best)
