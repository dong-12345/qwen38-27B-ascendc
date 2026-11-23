#!/usr/bin/env python3
"""convert_moondream3.py - Moondream 3 Preview (md3p-int4) -> 310B W4A16 engine format.

Input : moondream/md3p-int4/model.safetensors
        - MoE expert fc1/fc2 stored as MLX affine int4 (group_size=64): <q uint32 [E,out,K/8]>,
          <scales bf16 [E,out,K/64]>, <biases bf16 [E,out,K/64]>,  w = q*scale + bias
        - all other weights bf16
Output: W4A16 group-32 phase-interleaved (same ABI as ascendc_ops/kernels/w4a16_gemv_gen.cpp)
        f16 for norms/biases/embeddings/tau/patch+pos embedding
        meta.txt + manifest.json + compliance files

Usage:
  python convert_moondream3.py --src models/moondream3_int4/model.safetensors \
      --out converted/moondream3_310b [--verify] [--limit-layers N] [--skip-vision]
"""
import argparse
import json
import os
import struct
import sys
import time

import numpy as np

BF16 = np.uint16


def bf16_to_f32(raw_u16):
    u32 = raw_u16.astype(np.uint32) << 16
    return u32.view(np.float32)


class SafeReader:
    def __init__(self, path):
        self.f = open(path, "rb")
        n = struct.unpack("<Q", self.f.read(8))[0]
        self.header = json.loads(self.f.read(n).decode("utf-8"))
        self.base = 8 + n
        self.names = [k for k in self.header if k != "__metadata__"]

    def read_raw(self, name):
        meta = self.header[name]
        offs = meta["data_offsets"]
        self.f.seek(self.base + offs[0])
        return self.f.read(offs[1] - offs[0])

    def read(self, name):
        meta = self.header[name]
        dt, shape = meta["dtype"], meta["shape"]
        raw = self.read_raw(name)
        if dt == "BF16":
            return bf16_to_f32(np.frombuffer(raw, dtype=np.uint16).reshape(shape))
        if dt == "F32":
            return np.frombuffer(raw, dtype=np.float32).reshape(shape)
        if dt == "F16":
            return np.frombuffer(raw, dtype=np.float16).reshape(shape).astype(np.float32)
        if dt == "U32":
            return np.frombuffer(raw, dtype=np.uint32).reshape(shape)
        if dt == "I32":
            return np.frombuffer(raw, dtype=np.int32).reshape(shape)
        raise ValueError("dtype %s for %s" % (dt, name))

    def has(self, name):
        return name in self.header

    def close(self):
        self.f.close()


def quant_group32(w):
    """w float32 [N,K] -> q uint8 [N,K], s f16 [N,NG], sz f16 [N,NG]  (w ~= s*(q-z))"""
    n, k = w.shape
    assert k % 32 == 0, (n, k)
    ng = k // 32
    x = w.reshape(n, ng, 32)
    amax = x.max(axis=2)
    amin = x.min(axis=2)
    s = np.maximum((amax - amin) / 15.0, 1e-8).astype(np.float32)
    z = np.rint(-amin / s).astype(np.float32)
    q = np.rint(x / s[:, :, None] + z[:, :, None])
    q = np.clip(q, 0, 15).astype(np.uint8).reshape(n, k)
    return q, s.astype(np.float16), (s * z).astype(np.float16)


def pack_phase(q):
    n, k = q.shape
    kq = k // 4
    qq = q.reshape(n, 4, kq).astype(np.uint16)
    return (qq[:, 0, :] | (qq[:, 1, :] << 4) | (qq[:, 2, :] << 8) | (qq[:, 3, :] << 12)).astype(np.uint16)


def sbp_of(k, group=32):
    ng = (k + group - 1) // group
    return (ng + 15) // 16 * 16


class Writer:
    def __init__(self, out, verify=False):
        self.out = out
        self.verify = verify
        self.meta = []
        self.manifest = {}
        self.total_out = 0
        self.total_in = 0

    def _write(self, rel, arr):
        p = os.path.join(self.out, rel)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        np.ascontiguousarray(arr).tofile(p)
        return p

    def cfg(self, key, val):
        self.meta.append("CFG %s %s" % (key, val))

    def w4(self, role, rel_prefix, w, group=32, pad_k=None):
        w = np.asarray(w, dtype=np.float32)
        n, k = w.shape
        if pad_k is not None and pad_k > k:
            wp = np.zeros((n, pad_k), dtype=np.float32)
            wp[:, :k] = w
            w = wp
            k = pad_k
        assert k % group == 0, (role, n, k)
        q, s, sz = quant_group32(w)
        words = pack_phase(q)
        sbp = sbp_of(k, group)
        sp = np.zeros((n, sbp), dtype=np.float16)
        szp = np.zeros((n, sbp), dtype=np.float16)
        ng = k // group
        sp[:, :ng] = s
        szp[:, :ng] = sz
        self._write(rel_prefix + ".w4.bin", words)
        self._write(rel_prefix + ".s.bin", sp)
        self._write(rel_prefix + ".sz.bin", szp)
        nb = words.nbytes + sp.nbytes + szp.nbytes
        self.total_out += nb
        self.manifest[role] = {"kind": "w4", "shape": [n, k],
                               "prefix": rel_prefix, "group": group, "bytes": int(nb)}
        self.meta.append("W4 %s %s %d %d" % (role, rel_prefix, n, k))
        if self.verify:
            r0 = min(4, n)
            s0 = s[:r0].astype(np.float32)
            z0 = sz[:r0].astype(np.float32)
            x = w[:r0].reshape(r0, ng, group)
            wb = ((q[:r0].reshape(r0, ng, group).astype(np.float32) * s0[:, :, None]
                   - z0[:, :, None])).reshape(r0, k)
            err = float(np.abs(wb - x.reshape(r0, k)).mean() / (np.abs(x).mean() + 1e-9))
            print("  [verify] %-28s N=%5d K=%5d requant_rel_err=%.4f" % (role, n, k, err), flush=True)
        return nb

    def f16(self, role, rel, arr):
        a = np.asarray(arr, dtype=np.float32)
        shape = "x".join(str(x) for x in a.shape) if a.ndim else "1"
        self._write(rel, a.astype(np.float16))
        self.total_out += a.size * 2
        self.manifest[role] = {"kind": "f16", "shape": list(a.shape), "file": rel, "bytes": int(a.size * 2)}
        self.meta.append("F16 %s %s %s" % (role, rel, shape))

    def finish(self):
        with open(os.path.join(self.out, "meta.txt"), "w") as f:
            f.write("\n".join(self.meta) + "\n")
        with open(os.path.join(self.out, "manifest.json"), "w") as f:
            json.dump({"tensors": self.manifest}, f, indent=1)


def dequant_mlx_expert(words, scales, biases):
    """words uint32 [out, K/8], scales/biases bf16(f32) [out, K/64] -> w f32 [out,K]"""
    out, kw = words.shape
    k = kw * 8
    q = np.empty((out, k), dtype=np.uint8)
    for i in range(8):
        q[:, i::8] = ((words >> (4 * i)) & 0xF).astype(np.uint8)
    assert k % 64 == 0
    s = np.repeat(scales.astype(np.float32), 64, axis=1)
    b = np.repeat(biases.astype(np.float32), 64, axis=1)
    return q.astype(np.float32) * s + b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--limit-layers", type=int, default=0, help="debug: only convert first N text layers")
    ap.add_argument("--skip-vision", action="store_true")
    ap.add_argument("--skip-moe", action="store_true", help="debug: skip expert weights")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    rd = SafeReader(args.src)
    wr = Writer(args.out, verify=args.verify)
    t0 = time.time()

    print("tensors in ckpt:", len(rd.names), flush=True)
    nlayers = args.limit_layers if args.limit_layers else 24
    wr.cfg("hidden", 2048); wr.cfg("n_layers", nlayers); wr.cfg("n_heads", 32); wr.cfg("n_kv_heads", 32)
    wr.cfg("head_dim", 64); wr.cfg("rot_dim", 32); wr.cfg("vocab", 51200); wr.cfg("max_context", 4096)
    wr.cfg("moe_start", 4); wr.cfg("n_experts", 64); wr.cfg("experts_per_token", 8)
    wr.cfg("expert_inner", 1024); wr.cfg("dense_ff", 8192); wr.cfg("eps", 1e-5)
    wr.cfg("rope_theta", 1500000); wr.cfg("prefix_tokens", 730)
    wr.cfg("vision_dim", 1152); wr.cfg("vision_layers", 27); wr.cfg("vision_heads", 16)
    wr.cfg("vision_ff", 4304); wr.cfg("vision_patch", 14); wr.cfg("vision_crop", 378)
    wr.cfg("vision_max_crops", 12); wr.cfg("vision_grid", 27); wr.cfg("vision_ff_pad", 4320)

    # ---------------- text ----------------
    print("[text] wte / lm_head / final norm", flush=True)
    wr.w4("T.WTE", "text/wte", rd.read("text.wte.weight"))
    wr.w4("T.LM", "text/lm_head", rd.read("text.lm_head.weight"))
    wr.f16("T.LM.B", "text/lm_head.bias.f16.bin", rd.read("text.lm_head.bias"))
    wr.f16("T.FN.W", "text/final_norm.weight.f16.bin", rd.read("text.post_ln.weight"))
    wr.f16("T.FN.B", "text/final_norm.bias.f16.bin", rd.read("text.post_ln.bias"))

    for i in range(nlayers):
        p = "text.blocks.%d." % i
        print("[text] layer %d" % i, flush=True)
        wr.f16("T.L%d.LN.W" % i, "text/l%02d/ln.w.f16.bin" % i, rd.read(p + "ln.weight"))
        wr.f16("T.L%d.LN.B" % i, "text/l%02d/ln.b.f16.bin" % i, rd.read(p + "ln.bias"))
        wr.w4("T.L%d.QKV" % i, "text/l%02d/qkv" % i, rd.read(p + "attn.qkv.weight"))
        wr.f16("T.L%d.QKV.B" % i, "text/l%02d/qkv.bias.f16.bin" % i, rd.read(p + "attn.qkv.bias"))
        wr.w4("T.L%d.O" % i, "text/l%02d/proj" % i, rd.read(p + "attn.proj.weight"))
        wr.f16("T.L%d.O.B" % i, "text/l%02d/proj.bias.f16.bin" % i, rd.read(p + "attn.proj.bias"))
        wr.f16("T.L%d.TAU.WQ" % i, "text/l%02d/tau_wq.f16.bin" % i, rd.read(p + "attn.tau_wq"))
        wr.f16("T.L%d.TAU.WV" % i, "text/l%02d/tau_wv.f16.bin" % i, rd.read(p + "attn.tau_wv"))
        wr.f16("T.L%d.TAU.A" % i, "text/l%02d/tau_alpha.f16.bin" % i, rd.read(p + "attn.tau_alpha"))

        if i < 4:
            wr.w4("T.L%d.FC1" % i, "text/l%02d/fc1" % i, rd.read(p + "mlp.fc1.weight"))
            wr.f16("T.L%d.FC1.B" % i, "text/l%02d/fc1.bias.f16.bin" % i, rd.read(p + "mlp.fc1.bias"))
            wr.w4("T.L%d.FC2" % i, "text/l%02d/fc2" % i, rd.read(p + "mlp.fc2.weight"))
            wr.f16("T.L%d.FC2.B" % i, "text/l%02d/fc2.bias.f16.bin" % i, rd.read(p + "mlp.fc2.bias"))
        else:
            wr.w4("T.L%d.ROUTER" % i, "text/l%02d/router" % i, rd.read(p + "mlp.router.weight"))
            wr.f16("T.L%d.ROUTER.B" % i, "text/l%02d/router.bias.f16.bin" % i, rd.read(p + "mlp.router.bias"))
            if args.skip_moe:
                continue
            f1q = rd.read(p + "mlp.fc1_q"); f1s = rd.read(p + "mlp.fc1_scales"); f1b = rd.read(p + "mlp.fc1_biases")
            f2q = rd.read(p + "mlp.fc2_q"); f2s = rd.read(p + "mlp.fc2_scales"); f2b = rd.read(p + "mlp.fc2_biases")
            for e in range(64):
                w = dequant_mlx_expert(f1q[e], f1s[e], f1b[e])
                wr.w4("T.L%d.E%d.FC1" % (i, e), "text/l%02d/e%02d_fc1" % (i, e), w)
                del w
                w = dequant_mlx_expert(f2q[e], f2s[e], f2b[e])
                wr.w4("T.L%d.E%d.FC2" % (i, e), "text/l%02d/e%02d_fc2" % (i, e), w)
                del w
                if e % 16 == 15:
                    print("    expert %d/64 done, %.1fs" % (e + 1, time.time() - t0), flush=True)
            del f1q, f1s, f1b, f2q, f2s, f2b

    # ---------------- vision ----------------
    if not args.skip_vision:
        print("[vision] patch/pos norm", flush=True)
        wr.f16("V.PATCH.W", "vision/patch_emb.w.f16.bin", rd.read("vision.patch_emb.weight"))
        wr.f16("V.PATCH.B", "vision/patch_emb.b.f16.bin", rd.read("vision.patch_emb.bias"))
        wr.f16("V.POS", "vision/pos_emb.f16.bin", rd.read("vision.pos_emb"))
        wr.f16("V.FN.W", "vision/post_ln.w.f16.bin", rd.read("vision.post_ln.weight"))
        wr.f16("V.FN.B", "vision/post_ln.b.f16.bin", rd.read("vision.post_ln.bias"))
        for i in range(27):
            p = "vision.blocks.%d." % i
            print("[vision] layer %d" % i, flush=True)
            wr.f16("V.L%d.LN1.W" % i, "vision/l%02d/ln1.w.f16.bin" % i, rd.read(p + "ln1.weight"))
            wr.f16("V.L%d.LN1.B" % i, "vision/l%02d/ln1.b.f16.bin" % i, rd.read(p + "ln1.bias"))
            wr.f16("V.L%d.LN2.W" % i, "vision/l%02d/ln2.w.f16.bin" % i, rd.read(p + "ln2.weight"))
            wr.f16("V.L%d.LN2.B" % i, "vision/l%02d/ln2.b.f16.bin" % i, rd.read(p + "ln2.bias"))
            wr.w4("V.L%d.QKV" % i, "vision/l%02d/qkv" % i, rd.read(p + "attn.qkv.weight"))
            wr.f16("V.L%d.QKV.B" % i, "vision/l%02d/qkv.bias.f16.bin" % i, rd.read(p + "attn.qkv.bias"))
            wr.w4("V.L%d.O" % i, "vision/l%02d/proj" % i, rd.read(p + "attn.proj.weight"))
            wr.f16("V.L%d.O.B" % i, "vision/l%02d/proj.bias.f16.bin" % i, rd.read(p + "attn.proj.bias"))
            wr.w4("V.L%d.FC1" % i, "vision/l%02d/fc1" % i, rd.read(p + "mlp.fc1.weight"))
            wr.f16("V.L%d.FC1.B" % i, "vision/l%02d/fc1.bias.f16.bin" % i, rd.read(p + "mlp.fc1.bias"))
            wr.w4("V.L%d.FC2" % i, "vision/l%02d/fc2" % i, rd.read(p + "mlp.fc2.weight"), pad_k=4320)
            wr.f16("V.L%d.FC2.B" % i, "vision/l%02d/fc2.bias.f16.bin" % i, rd.read(p + "mlp.fc2.bias"))
        wr.w4("V.PROJ1", "vision/proj_fc1", rd.read("vision.proj_fc1.weight"))
        wr.f16("V.PROJ1.B", "vision/proj_fc1.bias.f16.bin", rd.read("vision.proj_fc1.bias"))
        wr.w4("V.PROJ2", "vision/proj_fc2", rd.read("vision.proj_fc2.weight"))
        wr.f16("V.PROJ2.B", "vision/proj_fc2.bias.f16.bin", rd.read("vision.proj_fc2.bias"))

    wr.finish()
    # ---- 合规材料 ----
    refdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ref")
    try:
        import shutil
        shutil.copyfile(os.path.join(refdir, "LICENSE.md"), os.path.join(args.out, "LICENSE_UPSTREAM_BSL1.1.md"))
        shutil.copyfile(os.path.join(refdir, "INT4_README.md"), os.path.join(args.out, "INT4_README.md"))
    except Exception as e:
        print("compliance copy warn:", e)
    with open(os.path.join(args.out, "MODIFICATIONS.md"), "w") as f:
        f.write("""# Modifications

本目录是 **Moondream 3 (Preview)** 的转换权重（Derivative），来源：

- 上游模型卡：https://huggingface.co/moondream/moondream3-preview
- 上游 4bit 权重：https://huggingface.co/moondream/md3p-int4 （MoE experts MLX affine int4 group 64，其余 bf16）
- 上游许可证：Business Source License 1.1 + Additional Use Grant（见 LICENSE_UPSTREAM_BSL1.1.md）

修改内容：

1. **格式重打包**：W4A16 group-32 非对称量化、相位交织（uint16 word 存 4 个相位），配套 fp16 scale 与 s*z；
2. **MoE 专家重量化**：int4 group64（MLX affine，w=q*scale+bias）→ 反量化 fp32 → 重新 group-32 量化；
3. **数据类型**：bf16 → fp16（scale/bias/embedding/pos_emb/LayerNorm 等）；
4. **形状补零**：vision MLP fc2 的 K 由 4304 补零到 4352（32 的倍数，便于 W4 打包）；
5. **裁剪**：只保留文本塔 + 视觉塔；region 坐标头（point/detect 用）未转换。

版权归属：Copyright (c) 2025 M87 Labs, Inc.；转换脚本与本项目代码 Apache-2.0。
""")
    print("done: %.2f GB out, %.1f s" % (wr.total_out / 1e9, time.time() - t0), flush=True)
    print("meta:", len(wr.meta), "roles:", len(wr.manifest))
    return 0


if __name__ == "__main__":
    sys.exit(main())
