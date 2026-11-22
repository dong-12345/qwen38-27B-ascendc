"""Build the ATC-ready ONNX: cubic->linear, half_pixel->asymmetric, Resize sizes folded."""
import os, sys, tempfile
import numpy as np, onnx
from onnx import numpy_helper, helper, TensorProto
import onnxruntime as ort

SRC = r"D:\项目4\orangepi_aipro\code\depth_completion\models\promptda_392x518.onnx"
DST = r"D:\项目4\orangepi_aipro\code\depth_completion\models\promptda_atc2.onnx"
H, W = 392, 518

m = onnx.load(SRC, load_external_data=False)
cnt = {"cubic": 0, "hp": 0}
for node in m.graph.node:
    if node.op_type != "Resize":
        continue
    for a in node.attribute:
        if a.name == "mode" and a.s == b"cubic":
            a.s = b"linear"
            cnt["cubic"] += 1
        elif a.name == "coordinate_transformation_mode" and a.s == b"half_pixel":
            a.s = b"asymmetric"
            cnt["hp"] += 1
print("patched", cnt)

# fold resize size inputs
inits = {i.name for i in m.graph.initializer}
targets = []
for node in m.graph.node:
    if node.op_type != "Resize":
        continue
    for idx in (2, 3):
        if len(node.input) > idx and node.input[idx] and node.input[idx] not in inits:
            targets.append((node, idx, node.input[idx]))
probe = onnx.load(SRC, load_external_data=False)
inf = onnx.shape_inference.infer_shapes(probe)
types = {v.name: v.type for v in list(inf.graph.value_info) + list(inf.graph.output)}
for _, _, n in targets:
    t = types.get(n)
    probe.graph.output.append(helper.make_tensor_value_info(n, t.tensor_type.elem_type if t else TensorProto.FLOAT, None))
tmp = os.path.join(tempfile.gettempdir(), "pd_probe2.onnx")
onnx.save(probe, tmp)
rng = np.random.default_rng(0)
img = rng.standard_normal((1, 3, H, W), dtype=np.float32)
pr = (rng.random((1, 1, H, W)) * 5).astype(np.float32)
sess = ort.InferenceSession(tmp, providers=["CPUExecutionProvider"])
outs = sess.run(None, {"pixel_values": img, "prompt_depth": pr})
vals = dict(zip([o.name for o in sess.get_outputs()], outs))
for node, idx, name in targets:
    v = np.asarray(vals[name])
    v = v.astype(np.int64) if (idx == 3 or v.dtype.kind in "iu") else v.astype(np.float32)
    cn = name + "_c"
    m.graph.initializer.append(numpy_helper.from_array(v, cn))
    node.input[idx] = cn
onnx.save(m, DST)
print("saved", DST, "folded", len(targets))
