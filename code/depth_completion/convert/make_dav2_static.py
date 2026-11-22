import numpy as np
import onnx
import onnxsim
from onnx import helper

SRC = r"D:\项目4\orangepi_aipro\tmp\hfcache\models--onnx-community--depth-anything-v2-small\snapshots\4472b7362082ad9968fee890ca0f1e5aca36b93d\onnx\model.onnx"
H, W = 168, 224
dst = rf"D:\项目4\orangepi_aipro\tmp\onnx\dav2_{H}x{W}_static_r.onnx"
m, _ = onnxsim.simplify(SRC, overwrite_input_shapes={"pixel_values": [1, 3, H, W]})
n = 0
for node in m.graph.node:
    if node.op_type == "Resize":
        n += 1
        attrs = {a.name: a for a in node.attribute}
        if "coordinate_transformation_mode" in attrs:
            attrs["coordinate_transformation_mode"].s = b"asymmetric"
        else:
            node.attribute.append(helper.make_attribute("coordinate_transformation_mode", "asymmetric"))
onnx.save(m, dst)
print("saved", dst, "resize patched", n, "size MB", round(__import__("os").path.getsize(dst) / 1e6, 1))

# FLOPs estimate
mi = onnx.shape_inference.infer_shapes(m, strict_mode=False)
inf = {t.name: t for t in mi.graph.value_info}
init = {t.name: t for t in m.graph.initializer}


def shp(nm):
    if nm in init:
        return list(init[nm].dims)
    if nm in inf:
        d = inf[nm].type.tensor_type.shape.dim
        return [x.dim_value if x.dim_value else None for x in d]
    return None


tot = 0.0
for node in m.graph.node:
    fl = 0.0
    if node.op_type == "Conv":
        w = shp(node.input[1]); y = shp(node.output[0])
        if w and y:
            fl = 2 * int(np.prod(w)) * int(np.prod([v for v in y[2:] if v]))
    elif node.op_type == "Gemm":
        w = shp(node.input[1]); x = shp(node.input[0])
        if w and x:
            fl = 2 * int(np.prod(w)) * int(np.prod([v for v in x[:-1] if v]))
    elif node.op_type == "MatMul":
        a = shp(node.input[0]); b = shp(node.input[1])
        if a and b and all(isinstance(v, int) and v for v in (a[-1], a[-2], b[-1])):
            fl = 2 * a[-2] * a[-1] * b[-1]
    elif node.op_type == "ConvTranspose":
        w = shp(node.input[1]); y = shp(node.output[0])
        if w and y:
            fl = 2 * int(np.prod(w)) * int(np.prod([v for v in y[2:] if v]))
    tot += fl
print(f"DAv2 GFLOP @ {H}x{W}: {tot/1e9:.2f}")
