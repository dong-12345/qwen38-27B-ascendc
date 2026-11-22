"""Rewrite Resize attrs in the static PromptDA ONNX so ATC/CANN can compile it.

CANN rejects coordinate_transformation_mode=half_pixel; switch only those nodes
to 'asymmetric' and keep align_corners / nearest untouched.
"""
import sys, onnx
from onnx import helper

SRC = r"D:\项目4\orangepi_aipro\code\depth_completion\models\promptda_392x518.onnx"
DST = r"D:\项目4\orangepi_aipro\code\depth_completion\models\promptda_392x518_atc.onnx"

m = onnx.load(SRC, load_external_data=False)
n_patch = 0
modes = {}
for node in m.graph.node:
    if node.op_type != "Resize":
        continue
    attrs = {a.name: a for a in node.attribute}
    mode = attrs["coordinate_transformation_mode"].s.decode() if "coordinate_transformation_mode" in attrs else "half_pixel"
    modes[mode] = modes.get(mode, 0) + 1
    if mode != "half_pixel":
        continue
    n_patch += 1
    for a in node.attribute:
        if a.name == "coordinate_transformation_mode":
            a.s = b"asymmetric"
print("resize modes:", modes, "patched:", n_patch)
onnx.checker.check_model(m)
onnx.save(m, DST)
print("saved", DST)
