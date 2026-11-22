"""PromptDA (Prompt Depth Anything) ONNX inference helpers (PC reference + board).
Mirrors transformers DPTImageProcessor: ImageNet normalize, keep aspect, multiple of 14.
"""
import numpy as np, cv2, onnxruntime as ort

MODEL_PATH = r"D:\项目4\orangepi_aipro\tmp\promptda_hf\models--onnx-community--promptda_vits_transparent_hf\snapshots\b671c85c54f44caf16416213d221610c8f7e0379\onnx\model.onnx"
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)

def imread_u(path, flags=cv2.IMREAD_COLOR):
    buf = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(buf, flags) if buf.size else None

def imwrite_u(path, img):
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise RuntimeError("encode failed")
    buf.tofile(path)

def pick_size(h, w, target_h=392, target_w=518, multiple=14):
    s = min(target_h / h, target_w / w)
    nh = max(multiple, int(round(h * s / multiple)) * multiple)
    nw = max(multiple, int(round(w * s / multiple)) * multiple)
    return nh, nw

def preprocess(rgb_bgr, prompt_depth_m, target_h=392, target_w=518, multiple=14):
    h, w = rgb_bgr.shape[:2]
    nh, nw = pick_size(h, w, target_h, target_w, multiple)
    img = cv2.resize(rgb_bgr, (nw, nh), interpolation=cv2.INTER_CUBIC)
    img = img.astype(np.float32) / 255.0
    img = (img - MEAN) / STD
    img = np.transpose(img, (2, 0, 1))[None]
    pr = cv2.resize(prompt_depth_m.astype(np.float32), (nw, nh), interpolation=cv2.INTER_NEAREST)
    pr = np.where(np.isfinite(pr) & (pr > 0), pr, 0.0).astype(np.float32)[None, None]
    return img.astype(np.float32), pr, (nh, nw)

class PromptDA:
    def __init__(self, model_path=MODEL_PATH, threads=0):
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if threads:
            so.intra_op_num_threads = threads
        self.sess = ort.InferenceSession(model_path, so, providers=["CPUExecutionProvider"])
        self.in_names = [i.name for i in self.sess.get_inputs()]
    def __call__(self, img_chw, prompt_chw):
        return self.sess.run(None, {self.in_names[0]: img_chw, self.in_names[1]: prompt_chw})[0]
    def run(self, rgb_bgr, prompt_depth_m, target_h=392, target_w=518, multiple=14):
        img, pr, (nh, nw) = preprocess(rgb_bgr, prompt_depth_m, target_h, target_w, multiple)
        out = self(img, pr)
        return np.squeeze(out).astype(np.float32), (nh, nw)
