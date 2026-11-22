"""Make visual alignment candidates: RGB (various transforms) next to raw depth colormap."""
import numpy as np
from PIL import Image

D = r"D:\项目4\orangepi_aipro\code\depth_completion\data"
dep = np.asarray(Image.open(D + r"\depth16.png"), dtype=np.float32) / 1000.0
rgb = Image.open(D + r"\color.jpg").convert("RGB")

valid = dep > 0
d2 = np.where(valid, dep, np.nan)
vmin, vmax = np.nanpercentile(d2, 2), np.nanpercentile(d2, 98)
t = np.clip((np.nan_to_num(d2, nan=vmin) - vmin) / (vmax - vmin + 1e-9), 0, 1)
r = np.clip(34.61 + t * (1172.33 + t * (-10793.56 + t * (33300.12 + t * (-38394.49 + t * 14825.05)))), 0, 255)
g = np.clip(23.31 + t * (557.33 + t * (1225.33 + t * (-3574.96 + t * (3202.75 + t * -895.6)))), 0, 255)
b = np.clip(27.2 + t * (3211.1 + t * (-15327.97 + t * (27814.0 + t * (-22569.18 + t * 6838.66)))), 0, 255)
dcol = np.stack([r, g, b], -1).astype(np.uint8)
dcol[~valid] = 0

tiles = [np.asarray(Image.new("RGB", (640, 480), (30, 30, 30)))]
labels = ["blank"]
# candidate 0: full resize (aspect ignore)
tiles.append(np.asarray(rgb.resize((640, 480), Image.BILINEAR)))
labels.append("full resize")
# candidate 1: center 4:3 crop (1280x960) -> resize
w, h = rgb.size
cw, ch = 1280, 960
tiles.append(np.asarray(rgb.crop(((w - cw) // 2, (h - ch) // 2, (w + cw) // 2, (h + ch) // 2)).resize((640, 480))))
labels.append("center 4:3 crop")
# candidate 2: rot180 + full resize
tiles.append(np.asarray(rgb.transpose(Image.ROTATE_180).resize((640, 480))))
labels.append("rot180")
# candidate 3: hflip
tiles.append(np.asarray(rgb.transpose(Image.FLIP_LEFT_RIGHT).resize((640, 480))))
labels.append("hflip")
tiles.append(dcol)
labels.append("DEPTH")
out = np.concatenate(tiles, axis=1)
Image.fromarray(out).save(r"D:\项目4\orangepi_aipro\tmp\align_candidates.png")
print("saved", out.shape, labels)
