"""Freeze PromptDA ONNX to static input shapes for ATC."""
import sys, os
import numpy as np, onnx
import onnxruntime as ort
from onnx import helper, TensorProto

SRC = r"D:\项目4\orangepi_aipro\tmp\promptda_hf\models--onnx-community--promptda_vits_transparent_hf\snapshots\b671c85c54f44caf16416213d221610c8f7e0379\onnx\model.onnx"
H, W = int(sys.argv[1]) if len(sys.argv) > 1 else 392, int(sys.argv[2]) if len(sys.argv) > 2 else 518
DST = r"D:\项目4\orangepi_aipro\code\depth_completion\models\promptda_%dx%d.onnx" % (H, W)
os.makedirs(os.path.dirname(DST), exist_ok=True)

m = onnx.load(SRC, load_external_data=False)
shapes = {"pixel_values": [1, 3, H, W], "prompt_depth": [1, 1, H, W]}
for inp in m.graph.input:
    s = shapes[inp.name]
    tt = inp.type.tensor_type
    del tt.shape.dim[:]
    for d in s:
        tt.shape.dim.add().dim_value = d
onnx.checker.check_model(m)
onnx.save(m, DST)
print("saved", DST, os.path.getsize(DST))

# numeric equivalence: dynamic vs static
rng = np.random.default_rng(0)
img = rng.standard_normal((1, 3, H, W), dtype=np.float32)
pr = (rng.random((1, 1, H, W)) * 5).astype(np.float32)
a = ort.InferenceSession(SRC, providers=["CPUExecutionProvider"]).run(None, {"pixel_values": img, "prompt_depth": pr})[0]
b = ort.InferenceSession(DST, providers=["CPUExecutionProvider"]).run(None, {"pixel_values": img, "prompt_depth": pr})[0]
print("static out", b.shape, "max abs diff", float(np.abs(a - b).max()))
