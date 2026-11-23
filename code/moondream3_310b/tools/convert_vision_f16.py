#!/usr/bin/env python3
"""convert_vision_f16.py — 把 int4 仓库里的 bf16 vision 权重转成 f16（[N,K] 行主序），
供 310B 板端 vision encoder 直接以 fp16 参与 matmul（避免 W4 量化误差）。"""
import os, json, struct
import numpy as np

ROOT = r"D:\项目4\orangepi_aipro"
CKPT = os.path.join(ROOT, "models", "moondream3_int4", "model.safetensors")
OUT = os.path.join(ROOT, "converted", "moondream3_310b", "vision")
os.makedirs(OUT, exist_ok=True)

class Reader:
    def __init__(self, path):
        self.f = open(path, "rb")
        n = struct.unpack("<Q", self.f.read(8))[0]
        self.h = json.loads(self.f.read(n).decode())
        self.base = 8 + n
    def get(self, name):
        m = self.h[name]
        self.f.seek(self.base + m["data_offsets"][0])
        raw = self.f.read(m["data_offsets"][1] - m["data_offsets"][0])
        if m["dtype"] == "BF16":
            u = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16
            return u.view(np.float32).reshape(m["shape"])
        if m["dtype"] == "F16":
            return np.frombuffer(raw, dtype=np.float16).reshape(m["shape"]).astype(np.float32)
        return np.frombuffer(raw, dtype=np.float32).reshape(m["shape"])

def main():
    r = Reader(CKPT)
    total = 0
    for li in range(27):
        pre = "vision.blocks.%d." % li
        items = {
            "L%d.qkv" % li: pre + "attn.qkv.weight",
            "L%d.qkv.b" % li: pre + "attn.qkv.bias",
            "L%d.o" % li: pre + "attn.proj.weight",
            "L%d.o.b" % li: pre + "attn.proj.bias",
            "L%d.fc1" % li: pre + "mlp.fc1.weight",
            "L%d.fc1.b" % li: pre + "mlp.fc1.bias",
            "L%d.fc2" % li: pre + "mlp.fc2.weight",
            "L%d.fc2.b" % li: pre + "mlp.fc2.bias",
        }
        for fn, key in items.items():
            a = r.get(key)
            if fn.endswith(".fc2"):
                a = np.concatenate([a, np.zeros((a.shape[0], 4352 - a.shape[1]), np.float32)], axis=1)
            a = a.astype(np.float16)
            p = os.path.join(OUT, fn + ".f16")
            a.tofile(p)
            total += a.nbytes
    for fn, key in {"proj_fc1": "vision.proj_fc1.weight", "proj_fc1.b": "vision.proj_fc1.bias",
                    "proj_fc2": "vision.proj_fc2.weight", "proj_fc2.b": "vision.proj_fc2.bias"}.items():
        a = r.get(key).astype(np.float16)
        a.tofile(os.path.join(OUT, fn + ".f16"))
        total += a.nbytes
    print("[cvf] wrote %.1f MB -> %s" % (total / 1048576.0, OUT))

if __name__ == "__main__":
    main()
