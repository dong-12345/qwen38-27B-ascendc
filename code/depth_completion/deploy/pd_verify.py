#!/usr/bin/env python3
"""Run the PromptDA OM on fixed inputs and dump the output for comparison."""
import os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/home/HwHiAiUser/code/depth_completion/deploy")
sys.path.insert(0, "/usr/local/Ascend/ascend-toolkit/latest/pyACL/python/site-packages")
import acl
from pd_live import Om

om_path = sys.argv[1] if len(sys.argv) > 1 else "models/promptda_atc2.om"
img = np.load("/tmp/vry_img.npy")
pr = np.load("/tmp/vry_pr.npy")
print("inputs", img.shape, pr.shape)
import pd_core as C
C.set_model_size(img.shape[2], img.shape[3])
m = Om(om_path)
out, dt = m.infer([img, pr])
np.save("/tmp/om_out.npy", out.astype(np.float32))
print("om out", out.shape, "p50=%.3f" % np.median(out), "ms=%.1f" % dt)
lat = []
for _ in range(20):
    _, dt = m.infer([img, pr])
    lat.append(dt)
print("NPU latency p50=%.2f p95=%.2f ms  fps=%.1f" % (np.median(lat), np.percentile(lat, 95), 1000/np.median(lat)))
m.close()
