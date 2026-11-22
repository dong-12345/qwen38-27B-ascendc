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


_COL_LUT = None
_COL_RANGE = (0.3, 8.0)


def colorize(d, vmin=0.3, vmax=8.0):
    """Depth (metres, float) -> TURBO colourmap.

    A 65536-entry uint16 LUT avoids the per-frame float normalisation
    (10.8 ms -> ~2 ms at 640x480).
    """
    global _COL_LUT, _COL_RANGE
    if _COL_LUT is None or _COL_RANGE != (vmin, vmax):
        _COL_RANGE = (vmin, vmax)
        mm = np.arange(65536, dtype=np.float32) / 1000.0
        _COL_LUT = np.clip((mm - vmin) / (vmax - vmin) * 255.0, 0, 255).astype(np.uint8)
    if d.dtype == np.float32 or d.dtype == np.float64:
        idx = _COL_LUT[np.clip(d * 1000.0, 0, 65535).astype(np.uint16)]
    else:
        idx = _COL_LUT[d]
    return cv2.applyColorMap(idx, cv2.COLORMAP_TURBO)

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
        cv2.addWeighted(out, sat, gray3, 1.0 - sat, 0, dst=out)   # in place
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
            # per-pixel: keep the readings that agree with the local surface,
            # drop only the outliers (block granularity decides *where* to look)
            surf_s = self._surface_small(d_s)
            surf = cv2.resize(surf_s, (w, h), interpolation=cv2.INTER_LINEAR)
            region = cv2.resize(bad, (w, h), interpolation=cv2.INTER_NEAREST) > 0
            keep = (np.abs(d - surf) < self.inlier_mm) & (d > 0)
            drop = region & ~keep
            if drop.any():
                d = d.copy()
                d[drop] = 0.0
        self.hole_frac = float((d == 0).mean())
        prev_ok = self.prev > 0
        out = cv2.addWeighted(d, self.alpha, self.prev, 1.0 - self.alpha, 0)
        jump = cv2.absdiff(d, self.prev) > self.reset_mm
        out[jump] = d[jump]
        out[(d <= 0) | ~prev_ok] = d[(d <= 0) | ~prev_ok]
        out[d <= 0] = 0.0
        self.prev = np.where(d > 0, out, self.prev)
        return np.clip(out, 0, 65535).astype(np.uint16)

def _box(img, r):
    return cv2.boxFilter(img, -1, (2 * r + 1, 2 * r + 1), borderType=cv2.BORDER_REFLECT)


def guided_filter_gray(guide, src, r=4, eps=400.0):
    """Single-channel guided filter (He et al.) with box filters."""
    I = guide.astype(np.float32)
    p = src.astype(np.float32)
    mI, mp = _box(I, r), _box(p, r)
    varI = _box(I * I, r) - mI * mI
    cov = _box(I * p, r) - mI * mp
    a = cov / (varI + eps)
    b = mp - a * mI
    return _box(a, r) * I + _box(b, r)


class ClassicCompleter:
    """Real-time CPU depth completion (no model, no NPU).

      1. 3x3 median + local-median test -> drop salt & pepper / flying pixels
      2. one smooth prior per frame (normalised convolution at 1/8, with a
         coarser fallback so it never collapses to zero)
      3. temporally unstable 8x8 blocks (low-signal areas) are replaced by that
         prior, which removes the dense speckle a median cannot reach
      4. remaining holes are filled with the prior refined by a guided filter
         (half resolution, luminance guide) so the fill follows RGB edges
      5. temporal EMA for flicker-free output
    """

    def __init__(self, median_k=3, outlier_mm=180.0, eps=60.0, radius=4, temporal=0.0,
                 div=4, unstable_th=0.6, diff_mm=150.0, regions=False, prior_div=12):
        self.regions = regions
        self.prior_div = prior_div
        self.median_k = median_k
        self.outlier_mm = outlier_mm
        self.eps = eps
        self.radius = radius
        self.temporal = temporal
        self.div = div
        self.unstable_th = unstable_th
        self.diff_mm = diff_mm
        self.prev = None
        self.unstable = None
        self.hole_frac = 0.0
        self.region_frac = 0.0

    def _prior(self, d, valid):
        h, w = d.shape
        pd = max(4, int(self.prior_div))
        sw, sh = max(1, w // pd), max(1, h // pd)
        src = np.where(valid, d, 0.0).astype(np.float32)
        small = cv2.resize(src, (sw, sh), interpolation=cv2.INTER_AREA)
        cnt = cv2.resize(valid.astype(np.float32), (sw, sh), interpolation=cv2.INTER_AREA)
        prior = small / np.maximum(cnt, 1e-3)
        prior = cv2.blur(prior, (3, 3))
        empty = cnt < 1e-3
        if empty.any():
            med = float(np.median(d[valid])) if valid.any() else 0.0
            cs, csh = max(1, sw // 4), max(1, sh // 4)
            cprior = cv2.resize(small, (cs, csh), interpolation=cv2.INTER_AREA) / \
                np.maximum(cv2.resize(cnt, (cs, csh), interpolation=cv2.INTER_AREA), 1e-3)
            cprior = cv2.blur(cprior, (3, 3))
            cprior = cv2.resize(cprior, (sw, sh), interpolation=cv2.INTER_LINEAR)
            cprior[cprior <= 0] = med
            prior = np.where(empty, cprior, prior)
        return cv2.resize(prior, (w, h), interpolation=cv2.INTER_LINEAR)

    def __call__(self, tof_mm, rgb640):
        # ---- fast path: median-based outlier removal in uint16 (no float pass)
        if self.median_k >= 3:
            med = cv2.medianBlur(tof_mm, self.median_k)
            bad = cv2.absdiff(tof_mm, med) > int(self.outlier_mm)
            out = tof_mm.copy()
            out[bad] = 0
        else:
            out = tof_mm.copy()
        valid = out > 0
        self.hole_frac = float((~valid).mean())
        if self.hole_frac >= 0.005 or self.regions:
            d = out.astype(np.float32)
            h, w = d.shape
            prior = self._prior(d, valid) if self.regions else None
            if self.regions and self.prev is not None and self.prev.shape == d.shape:
                bh, bw = max(1, h // 8), max(1, w // 8)
                d_s = cv2.resize(d, (bw, bh), interpolation=cv2.INTER_AREA)
                p_s = cv2.resize(self.prev, (bw, bh), interpolation=cv2.INTER_AREA)
                diff_s = cv2.absdiff(d_s, p_s)
                if self.unstable is None:
                    self.unstable = np.zeros((bh, bw), np.float32)
                self.unstable = 0.7 * self.unstable + 0.3 * (diff_s > self.diff_mm).astype(np.float32)
                lo = self.unstable_th * 0.6
                hi = max(self.unstable_th, lo + 1e-3)
                wgt = np.clip((self.unstable - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)
                self.region_frac = float((wgt > 0.5).mean())
                if wgt.max() > 0.05:
                    wf = cv2.resize(wgt, (w, h), interpolation=cv2.INTER_LINEAR)
                    wf = cv2.dilate(wf, np.ones((7, 7), np.float32))
                    d = d * (1.0 - wf) + prior * wf
                    valid = (d > 0) & (valid | (wf > 0.5))
            if not valid.all():
                # do the whole fill at 1/div resolution: one downscale, one
                # upscale (this is the cheapest correct way to fill thin holes)
                dv = max(2, int(self.div))
                sw, sh = max(1, w // dv), max(1, h // dv)
                d_s = cv2.resize(np.where(valid, d, 0.0), (sw, sh), interpolation=cv2.INTER_AREA)
                m_s = cv2.resize(valid.astype(np.float32), (sw, sh), interpolation=cv2.INTER_AREA)
                p_s = d_s / np.maximum(m_s, 1e-3)
                p_s = cv2.blur(p_s, (3, 3))
                empty = m_s < 1e-3
                if empty.any():
                    med = float(d[valid].mean()) if valid.any() else 0.0
                    p_s = np.where(empty, med if med > 0 else float(p_s.mean()), p_s)
                gray = cv2.cvtColor(rgb640, cv2.COLOR_BGR2GRAY)
                g_s = cv2.resize(gray, (sw, sh), interpolation=cv2.INTER_AREA).astype(np.float32)
                f_s = guided_filter_gray(g_s, p_s.astype(np.float32), self.radius, self.eps)
                filled = cv2.resize(f_s, (w, h), interpolation=cv2.INTER_LINEAR)
                out = np.where(valid, d, filled)
            else:
                out = d
        if self.temporal > 0:
            of = out.astype(np.float32) if out.dtype != np.float32 else out
            if self.prev is None or self.prev.shape != of.shape:
                self.prev = of.copy()
            else:
                a = self.temporal
                jump = np.abs(of - self.prev) > 600.0
                ema = of * (1.0 - a) + self.prev * a
                ema[jump] = of[jump]
                of = ema
                self.prev = of.copy()
            out = of
        if out.dtype != np.uint16:
            out = np.clip(out, 0, 65535).astype(np.uint16)
        out[out <= 0] = 0
        return out