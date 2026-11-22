"""pd_core.py - PromptDA-Small-Transparent preprocessing / postprocessing (PC + board).

Contract of the exported ONNX/OM (see HANDOVER): two inputs
  pixel_values  1x3xHxW  float32  (ImageNet-normalised, RGB order)
  prompt_depth  1x1xHxW  float32  (metric depth in metres, MUST be complete:
                                   the graph normalises by global min/max, so 0
                                   means "0 m" and the head returns ~0 there)
one output
  predicted_depth 1xHxW float32 metres
"""
import numpy as np
import cv2

MODEL_H, MODEL_W = 392, 518          # multiples of 14 (patch 14), 4:3


def set_model_size(h, w):
    """Switch the network input size (must match the exported OM)."""
    global MODEL_H, MODEL_W
    MODEL_H, MODEL_W = int(h), int(w)
MEAN_BGR = np.array([0.406, 0.456, 0.485], np.float32)   # reversed ImageNet mean
STD_BGR = np.array([0.225, 0.224, 0.229], np.float32)


def fill_prompt(tof_mm, valid=None):
    """Complete a sparse ToF frame into a prompt the model can consume.

    Fast path: 1/4-resolution Telea inpaint of the holes, upsampled, then the
    original valid pixels are pasted back.  ~1.5 ms at 640x480.
    """
    if valid is None:
        valid = tof_mm > 0
    dm = tof_mm.astype(np.float32) * 0.001
    if valid.all():
        return dm
    h, w = dm.shape
    sw, sh = max(1, w // 4), max(1, h // 4)
    small = cv2.resize(dm, (sw, sh), interpolation=cv2.INTER_NEAREST)
    vs = cv2.resize(valid.astype(np.float32), (sw, sh), interpolation=cv2.INTER_AREA)
    # normalised convolution: smooth extrapolation from the valid pixels, which
    # beats Telea on large low-signal regions (no streaking / checkerboard)
    num = cv2.GaussianBlur(small * vs, (0, 0), 8.0)
    den = cv2.GaussianBlur(vs, (0, 0), 8.0)
    filled = num / np.maximum(den, 1e-3)
    up = cv2.resize(filled, (w, h), interpolation=cv2.INTER_LINEAR)
    return np.where(valid, dm, up)


def preprocess(rgb_bgr, prompt_m):
    """uint8 BGR + float32 depth (m) -> (1,3,H,W), (1,1,H,W) float32."""
    img = cv2.resize(rgb_bgr, (MODEL_W, MODEL_H), interpolation=cv2.INTER_AREA)
    blob = img.astype(np.float32)
    blob *= (1.0 / 255.0)
    blob -= MEAN_BGR
    blob /= STD_BGR
    blob = blob.transpose(2, 0, 1)[None]
    pr = cv2.resize(prompt_m, (MODEL_W, MODEL_H), interpolation=cv2.INTER_NEAREST)[None, None]
    return blob, pr


def preprocess_fast(rgb_bgr, prompt_m):
    """Same result, C++ path (no python transpose): ~40% faster."""
    blob = cv2.dnn.blobFromImage(rgb_bgr, 1.0 / 255.0, (MODEL_W, MODEL_H),
                                 tuple((MEAN_BGR * 255.0).tolist()), swapRB=True, crop=False,
                                 ddepth=cv2.CV_32F)
    blob[0, 0] *= 1.0 / STD_BGR[0]
    blob[0, 1] *= 1.0 / STD_BGR[1]
    blob[0, 2] *= 1.0 / STD_BGR[2]
    pr = cv2.resize(prompt_m, (MODEL_W, MODEL_H), interpolation=cv2.INTER_NEAREST)[None, None]
    return blob, pr


def postprocess(pred_m, size_hw):
    """model output (H,W) or (1,H,W) metres -> float32 (h,w) at target size."""
    d = np.squeeze(pred_m)
    return cv2.resize(d, (size_hw[1], size_hw[0]), interpolation=cv2.INTER_LINEAR)


def fuse(tof_mm, pred_m, valid=None):
    """Keep measured ToF, take the prediction only where ToF has no return.
    Returns uint16 millimetres."""
    if valid is None:
        valid = tof_mm > 0
    out = np.clip(pred_m * 1000.0, 0, 65535).astype(np.uint16)
    out[valid] = tof_mm[valid]
    return out


def to_u16(depth_m):
    return np.clip(depth_m * 1000.0, 0, 65535).astype(np.uint16)


def colorize(d, vmin=0.3, vmax=8.0):
    x = np.clip((d - vmin) / (vmax - vmin), 0, 1)
    return cv2.applyColorMap((x * 255).astype(np.uint8), cv2.COLORMAP_TURBO)

def preprocess_gain(rgb_bgr, prompt_m, gain=None, target_luma=110.0, max_gain=8.0):
    """Like preprocess_fast but applies an exposure gain (auto-exposure).

    Returns (blob, prompt, gain, small_bgr, mean_luma); pass the gain back in to
    keep the brightness continuous instead of recomputing it every frame.
    """
    small = cv2.resize(rgb_bgr, (MODEL_W, MODEL_H), interpolation=cv2.INTER_LINEAR)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    m = float(gray.mean())
    if gain is None:
        gain = 1.0 if m < 1.0 else min(max(target_luma / m, 1.0), max_gain)
    gain = float(np.clip(gain, 0.5, max_gain))
    blob = cv2.dnn.blobFromImage(small, gain / 255.0, (MODEL_W, MODEL_H),
                                 tuple((MEAN_BGR * 255.0 / gain).tolist()), swapRB=True,
                                 crop=False, ddepth=cv2.CV_32F)
    blob[0, 0] *= 1.0 / STD_BGR[0]
    blob[0, 1] *= 1.0 / STD_BGR[1]
    blob[0, 2] *= 1.0 / STD_BGR[2]
    pr = cv2.resize(prompt_m, (MODEL_W, MODEL_H), interpolation=cv2.INTER_NEAREST)[None, None]
    return blob, pr, gain, small, m

class AutoExposure:
    """Software auto-exposure: smooth per-frame gain towards a target luma.

    The camera's GenICam ExposureAuto node is not settable on this unit
    (E_PRECONDITION) and the vendor SDK only reaches ~5 fps, so we keep the
    sensor at a fixed exposure and regulate brightness in software.
    """

    def __init__(self, target=110.0, lo=1.0, hi=8.0, alpha=0.35):
        self.target = target
        self.lo = lo
        self.hi = hi
        self.alpha = alpha
        self.gain = 1.0

    def update(self, mean_luma):
        want = self.target / max(mean_luma, 1.0)
        want = min(max(want, self.lo), self.hi)
        self.gain += self.alpha * (want - self.gain)
        return self.gain

_GAMMA_LUT = None


def grade(bgr, gamma=0.75, sat=1.6):
    """Cheap display grade so the published image matches HiViewer's look.

    The camera's planar RGB comes out flat/desaturated (HiViewer applies its own
    ISP tone mapping); a gamma LUT + saturation mix gets us close for ~1 ms at
    640x480.
    """
    global _GAMMA_LUT
    if _GAMMA_LUT is None:
        x = np.arange(256, dtype=np.float32) / 255.0
        _GAMMA_LUT = np.clip((x ** gamma) * 255.0, 0, 255).astype(np.uint8)
    out = cv2.LUT(bgr, _GAMMA_LUT)
    if sat != 1.0:
        gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
        gray3 = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        out = cv2.addWeighted(out, sat, gray3, 1.0 - sat, 0)
    return out

class TemporalDepth:
    """Exponential moving average on the ToF depth with motion rejection.

    Static scenes lose most of the per-pixel measurement noise; moving objects
    (|d - prev| > reset_mm) jump straight to the new value, so no ghosting.
    """

    def __init__(self, alpha=0.45, reset_mm=300):
        self.alpha = alpha
        self.reset_mm = reset_mm
        self.prev = None

    def __call__(self, d_mm):
        d = d_mm.astype(np.float32)
        if self.prev is None or self.prev.shape != d.shape:
            self.prev = d.copy()
            return d_mm
        diff = np.abs(d - self.prev)
        a = np.where(diff > self.reset_mm, 1.0, self.alpha).astype(np.float32)
        self.prev = a * d + (1.0 - a) * self.prev
        return np.clip(self.prev, 0, 65535).astype(np.uint16)


def denoise_depth(d_mm, guide_bgr=None, radius=4, eps=600.0):
    """Edge-aware smoothing of the depth map (guided by RGB when available)."""
    f = d_mm.astype(np.float32)
    if guide_bgr is not None and hasattr(cv2, "ximgproc"):
        out = cv2.ximgproc.guidedFilter(guide_bgr, f, radius, eps)
    else:
        out = cv2.medianBlur(d_mm, 5).astype(np.float32)
    out[d_mm == 0] = 0.0
    return np.clip(out, 0, 65535).astype(np.uint16)

class DepthCleaner:
    """Suppress ToF speckle in low-signal regions before completion.

    1. temporal instability per 8x8 block (|d_t - d_{t-1}| > 150 mm) marks the
       blocks where the sensor is guessing (dark / low-reflectivity areas)
    2. inside those blocks only, keep pixels that agree with a robust local
       surface (1/4-res median, two passes) and drop the rest as holes
    3. temporal EMA on what survives (big jumps reset, so no ghosting)
    """

    def __init__(self, alpha=0.45, reset_mm=300, block_th=0.65, diff_mm=150,
                 inlier_mm=200.0):
        self.alpha = alpha
        self.reset_mm = reset_mm
        self.block_th = block_th
        self.diff_mm = diff_mm
        self.inlier_mm = inlier_mm
        self.prev = None
        self.unstable = None
        self.hole_frac = 0.0

    @staticmethod
    def _surface_small(small):
        med = cv2.medianBlur(small.astype(np.uint16), 5).astype(np.float32)
        return cv2.GaussianBlur(med, (0, 0), 1.0)

    def __call__(self, d_mm):
        d = d_mm.astype(np.float32)
        h, w = d.shape
        bh, bw = max(1, h // 8), max(1, w // 8)
        if self.prev is None or self.prev.shape != d.shape:
            self.prev = d.copy()
            self.unstable = np.zeros((bh, bw), np.float32)
            return d_mm
        d_s = cv2.resize(d, (bw, bh), interpolation=cv2.INTER_AREA)
        p_s = cv2.resize(self.prev, (bw, bh), interpolation=cv2.INTER_AREA)
        diff_s = cv2.absdiff(d_s, p_s)
        self.unstable = 0.7 * self.unstable + 0.3 * (diff_s > self.diff_mm).astype(np.float32)
        bad = (self.unstable > self.block_th).astype(np.uint8)
        if bad.any():
            bad = cv2.dilate(bad, np.ones((3, 3), np.uint8))
        if bad.any():
            # inlier test on the same 1/8 grid (cheap), then blown up to a mask
            surf_s = self._surface_small(d_s)
            d_out = (d_s - surf_s)
            keep_s = (np.abs(d_out) < self.inlier_mm) & (d_s > 0)
            bad = np.where(keep_s, 0, bad).astype(np.uint8)
            if bad.any():
                mask = cv2.resize(bad, (w, h), interpolation=cv2.INTER_NEAREST) > 0
                d = d.copy()
                d[mask] = 0.0
        self.hole_frac = float((d == 0).mean())
        prev_ok = self.prev > 0
        out = cv2.addWeighted(d, self.alpha, self.prev, 1.0 - self.alpha, 0)
        jump = cv2.absdiff(d, self.prev) > self.reset_mm
        out[jump] = d[jump]
        out[(d <= 0) | ~prev_ok] = d[(d <= 0) | ~prev_ok]
        out[d <= 0] = 0.0
        self.prev = np.where(d > 0, out, self.prev)
        return np.clip(out, 0, 65535).astype(np.uint16)
