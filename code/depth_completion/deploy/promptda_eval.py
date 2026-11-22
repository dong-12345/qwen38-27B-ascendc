"""PromptDA eval on a saved frame: full-prompt fidelity + synthetic-hole completion."""
import sys, os, json, time
import numpy as np, cv2
sys.path.insert(0, r"D:\项目4\orangepi_aipro\code\depth_completion\deploy")
from promptda_onnx import PromptDA, imread_u, imwrite_u

dc = r"D:\项目4\orangepi_aipro\code\depth_completion"

def load_frame(d):
    rgb = imread_u(os.path.join(d, "color.jpg"))
    dep = imread_u(os.path.join(d, "depth16.png"), cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
    return rgb, dep

def synth_keep(dep, erode=51, block=72, nblocks=6, seed=1):
    v = (dep > 0).astype(np.uint8)
    er = cv2.erode(v, np.ones((erode, erode), np.uint8))
    rng = np.random.default_rng(seed)
    h, w = dep.shape
    cut = np.zeros_like(v)
    for _ in range(nblocks):
        y = int(rng.integers(0, h - block)); x = int(rng.integers(0, w - block))
        cut[y:y+block, x:x+block] = 1
    return ((v == 1) & (er == 1) & (cut == 0)).astype(np.float32), (cut == 1) & (v == 1)

def colorize(d, vmin=0.3, vmax=8.0):
    x = np.clip((d - vmin) / (vmax - vmin), 0, 1)
    img = cv2.applyColorMap((x * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    img[d <= 0] = (0, 0, 0)
    return img

def err_stats(pred, gt, m):
    e = np.abs(pred[m] - gt[m]); gt_ = gt[m]
    return {"n": int(m.sum()), "mae_mm": round(float(e.mean()*1000),1), "p50_mm": round(float(np.median(e)*1000),1),
            "rmse_mm": round(float(np.sqrt((e**2).mean())*1000),1), "p90_mm": round(float(np.percentile(e,90)*1000),1),
            "absrel": round(float((e/gt_).mean()),4)}

def main(frame_dir, tag, th=392, tw=518):
    rgb, dep = load_frame(frame_dir)
    m = PromptDA()
    t0 = time.time(); out_full, size = m.run(rgb, dep, th, tw); t_full = (time.time()-t0)*1000
    keep, blockm = synth_keep(dep)
    t0 = time.time(); out_hole, _ = m.run(rgb, dep * keep, th, tw); t_hole = (time.time()-t0)*1000
    up = lambda a: cv2.resize(a, (dep.shape[1], dep.shape[0]), interpolation=cv2.INTER_LINEAR)
    f_full, f_hole = up(out_full), up(out_hole)
    gtv = dep > 0
    res = {"frame": os.path.basename(frame_dir), "model_size": list(size),
           "ms_cpu_full": round(t_full,1), "ms_cpu_holefill": round(t_hole,1)}
    a, b = f_full[gtv], dep[gtv]
    res["full_prompt"] = {"n": int(gtv.sum()), "median": round(float(np.median(a)),4), "median_tof": round(float(np.median(b)),4),
                          "scale_ratio": round(float(np.median(a/b)),4), "corr": round(float(np.corrcoef(a,b)[0,1]),4),
                          "mae_mm": round(float(np.mean(np.abs(a-b))*1000),1), "p90_mm": round(float(np.percentile(np.abs(a-b),90)*1000),1)}
    hv = blockm & gtv
    res["synthetic_holes"] = {"frac": round(float(hv.mean()),3), **err_stats(f_hole, dep, hv)}
    dv = (dep*1000).astype(np.uint16)
    inp = cv2.inpaint(dv, ((~gtv) | blockm).astype(np.uint8), 5, cv2.INPAINT_TELEA).astype(np.float32)/1000.0
    res["baseline_telea"] = err_stats(inp, dep, hv)
    res["baseline_zero"] = err_stats(np.zeros_like(dep), dep, hv)
    print(json.dumps(res, indent=1))
    outdir = os.path.join(dc, "results/pc_ref"); os.makedirs(outdir, exist_ok=True)
    np.save(os.path.join(outdir, "pd_full_%s.npy" % tag), f_full)
    np.save(os.path.join(outdir, "pd_hole_%s.npy" % tag), f_hole)
    json.dump(res, open(os.path.join(outdir, "pd_eval_%s.json" % tag), "w"), indent=1)
    h, w = dep.shape
    err = np.abs(f_hole - dep)
    panels = [("1 RGB", rgb), ("2 ToF raw", colorize(dep)),
              ("3 prompt kept (holes removed)", colorize(np.where(keep > 0, dep, 0))),
              ("4 PromptDA full prompt", colorize(f_full)),
              ("5 PromptDA holes filled", colorize(np.where(f_hole > 0, f_hole, 0))),
              ("6 |err| in holes (0-1m)", cv2.applyColorMap(np.clip(err/1.0*255, 0, 255).astype(np.uint8), cv2.COLORMAP_MAGMA))]
    canvas = np.zeros((2*h, 3*w, 3), np.uint8)
    for i, (t, im) in enumerate(panels):
        r, c = divmod(i, 3)
        canvas[r*h:(r+1)*h, c*w:(c+1)*w] = im
        cv2.putText(canvas, t, (c*w+8, r*h+22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1, cv2.LINE_AA)
    imwrite_u(os.path.join(outdir, "pd_eval_%s.png" % tag), canvas)
    return res

if __name__ == "__main__":
    main(os.path.join(dc, "data/live_1529_aligned"), "live1529")
