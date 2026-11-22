"""Register DAv2 monocular depth (RGB frame) to ToF depth frame.

Both are depth maps of the same scene from different cameras -> a coarse-to-fine
search over (sx, sy, tx, ty, flip) maximizing normalized correlation in
disparity (1/z) space on ToF-valid pixels.
"""
import numpy as np
from PIL import Image
from scipy import ndimage

D = r"D:\项目4\orangepi_aipro\code\depth_completion\data"
P = r"D:\项目4\orangepi_aipro\code\depth_completion\results\pc_ref"
W, H = 640, 480
SW, SH = 160, 120          # search resolution

tof = np.asarray(Image.open(D + r"\depth16.png"), dtype=np.float32) / 1000.0
valid = tof > 0
# fill holes for gradient-free comparison
ind = ndimage.distance_transform_edt(~valid, return_distances=False, return_indices=True)
tof_f = tof[tuple(ind)]
tof_disp = np.where(tof_f > 0, 1.0 / np.maximum(tof_f, 0.05), 0.0)
m = ndimage.zoom(valid.astype(np.float32), (SH / H, SW / W), order=0) > 0.5
td = ndimage.zoom(tof_disp, (SH / H, SW / W), order=1)

prior = np.load(P + r"\dav2_434x518.npy").astype(np.float32)
ph, pw = prior.shape
pd = prior.copy()
pd = (pd - pd.min()) / (pd.max() - pd.min() + 1e-9)     # normalize disparity-like prior
td_n = (td - td[m].mean()) / (td[m].std() + 1e-9)


def warp(sx, sy, tx, ty, flip):
    """map depth-frame pixels -> prior-frame pixels (inverse affine for PIL)."""
    im = Image.fromarray((pd * 255).astype(np.uint8))
    if flip:
        im = im.transpose(Image.FLIP_LEFT_RIGHT)
    # dst (SW x SH) sampled from prior with scale sx,sy and translation tx,ty (in search px)
    cx, cy = SW / 2, SH / 2
    a = (pw / SW) / sx
    e = (ph / SH) / sy
    c = cx - a * (cx + tx)
    f = cy - e * (cy + ty)
    im2 = im.transform((SW, SH), Image.AFFINE, (a, 0, c, 0, e, f), resample=Image.BILINEAR)
    return np.asarray(im2, dtype=np.float32) / 255.0


def score(sx, sy, tx, ty, flip):
    p = warp(sx, sy, tx, ty, flip)
    pp = p - p[m].mean()
    den = pp[m].std() * td_n[m].std() + 1e-9
    return float((pp[m] * td_n[m]).mean() / den)


def search(flips=(0, 1)):
    best = (-2, None)
    for flip in flips:
        for sx in np.arange(0.4, 1.65, 0.15):
            for sy in np.arange(0.4, 1.65, 0.15):
                for tx in range(-40, 41, 10):
                    for ty in range(-30, 31, 10):
                        s = score(sx, sy, tx, ty, flip)
                        if s > best[0]:
                            best = (s, (round(float(sx), 2), round(float(sy), 2), tx, ty, flip))
    return best


best = search()
print("coarse", best)
s0, sy0, tx0, ty0, f0 = best[1]
for sx in np.arange(s0 - 0.12, s0 + 0.13, 0.04):
    for sy in np.arange(sy0 - 0.12, sy0 + 0.13, 0.04):
        for tx in range(tx0 - 12, tx0 + 13, 4):
            for ty in range(ty0 - 12, ty0 + 13, 4):
                s = score(sx, sy, tx, ty, f0)
                if s > best[0]:
                    best = (s, (round(float(sx), 3), round(float(sy), 3), tx, ty, f0))
print("refined", best)
np.save(P + r"\dav2_to_tof_transform.npy", np.array(best[1], dtype=np.float64))
