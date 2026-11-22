#!/usr/bin/env python3
"""convert_qwen38.py — 把 Qwen3.8-27B 的 compressed-tensors W4A16 权重转成自研 310B 引擎格式。

输入（llm-compressor "pack-quantized" 格式）：
  <name>.weight_packed      int32  [N, K/8]   —— 8 个 4bit 沿 K 维打包（低半字节是第 1 个值）
  <name>.weight_scale       bf16/fp16 [N, K/32]
  <name>.weight_zero_point  int8  [N, K/32]  （非对称，group_size=32）
  <name>.weight_shape       int32  [2]（部分仓库有）
  其余张量（embed/lm_head/vision/norm/GDN 的 in_proj_a/b/conv1d/A_log/dt_bias）保持 bf16/fp16。

输出（每个量化张量两个文件）：
  <base>.w4.bin   uint16 [N, K/4]   相位交织：word j = q[j] | q[KQ+j]<<4 | q[2KQ+j]<<8 | q[3KQ+j]<<12
  <base>.s.bin    fp16   [N]        行级对称 scale（zero=8），满足 w ≈ s*(q-8)
  非量化张量：
  <base>.f16.bin  fp16   原 shape
清单：<out>/manifest.json（名字 -> {kind, shape, files, bytes}）

用法：
  python tools/convert_qwen38.py --src models/Qwen3.8-27B-AWQ-INT4 --out converted/qwen38_310b [--only 关键词] [--verify]
"""
import argparse
import json
import os
import struct
import sys

import numpy as np

DTYPE_ITEMSIZE = {
    "F16": 2, "F32": 4, "BF16": 2, "I8": 1, "U8": 1, "I32": 4, "U32": 4,
    "I64": 8, "F64": 8, "BOOL": 1, "I16": 2, "U16": 2,
}
TORCH_TO_NP = {
    "F16": np.float16, "F32": np.float32, "I8": np.int8, "U8": np.uint8,
    "I32": np.int32, "U32": np.uint32, "I64": np.int64, "F64": np.float64,
    "BOOL": np.bool_, "I16": np.int16, "U16": np.uint16,
}


def read_safetensors(path, only=None):
    """手写解析 safetensors（支持 bf16）。only 可为字符串或字符串列表（子串匹配）"""
    if isinstance(only, str):
        only = [only]
    out = {}
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n).decode("utf-8"))
        base = 8 + n
        for name, meta in header.items():
            if name == "__metadata__":
                continue
            if only is not None and not any(o in name for o in only):
                continue
            dt, shape = meta["dtype"], meta["shape"]
            offs = meta["data_offsets"]
            f.seek(base + offs[0])
            raw = f.read(offs[1] - offs[0])
            if dt == "BF16":
                u32 = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32)
                u32 <<= 16
                arr = u32.view(np.float32).reshape(shape)
            else:
                arr = np.frombuffer(raw, dtype=TORCH_TO_NP[dt]).reshape(shape)
            out[name] = arr
    return out


def unpack_nibbles(packed, group_cols):
    """int32 [N, C] -> uint8 [N, C*8]，低半字节是第 1 个值。"""
    n, c = packed.shape
    q = np.empty((n, c * 8), dtype=np.uint8)
    for i in range(8):
        q[:, i::8] = ((packed >> (4 * i)) & 0xF).astype(np.uint8)
    return q


def unpack_zp(zp_packed, out_dim, ng):
    """weight_zero_point: int32 [out/8, ng] -> uint8 [out, ng]（沿输出行打包，低 nibble = 行 8j+0）"""
    c8, g = zp_packed.shape
    out = np.empty((c8 * 8, g), dtype=np.uint8)
    for i in range(8):
        out[i::8] = ((zp_packed >> (4 * i)) & 0xF).astype(np.uint8)
    return out[:out_dim]


def dequant(q, scale, zero, group=32):
    """q [N,K] uint8, scale/zp [N, Ng] -> fp32 [N,K]（非对称 group 量化）"""
    n, k = q.shape
    ng = k // group
    s = np.repeat(scale.astype(np.float32), group, axis=1)[:, :k]
    z = np.repeat(zero.astype(np.float32), group, axis=1)[:, :k]
    return (q.astype(np.float32) - z) * s


def requant_group(w, group=32):
    """fp32 [N,K] -> (q uint8 [N,K], s fp16 [N,NG], sz fp16 [N,NG])
    非对称 group 量化：w ≈ s*(q - z)，sz = s*z（供 kernel 的标量偏置用）"""
    n, k = w.shape
    ng = k // group
    x = w.reshape(n, ng, group)
    amax = x.max(axis=2)
    amin = x.min(axis=2)
    s = np.maximum((amax - amin) / 15.0, 1e-8).astype(np.float32)
    z = np.rint(-amin / s).astype(np.float32)
    q = np.rint(x / s[:, :, None] + z[:, :, None])
    q = np.clip(q, 0, 15).astype(np.uint8).reshape(n, k)
    return q, s.astype(np.float16), (s * z).astype(np.float16)


def pack_phase(q):
    """q [N,K] uint8 -> uint16 [N,K/4]，相位交织。"""
    n, k = q.shape
    kq = k // 4
    qq = q.reshape(n, 4, kq).astype(np.uint16)
    return (qq[:, 0, :] | (qq[:, 1, :] << 4) | (qq[:, 2, :] << 8) | (qq[:, 3, :] << 12)).astype(np.uint16)


def f16_save(path, arr, one_centered=False):
    """one_centered=True：Qwen3.5 的 Qwen3_5RMSNorm 有效权重 = 1 + w（权重存的是偏移量）。"""
    a = np.asarray(arr, dtype=np.float32)
    if one_centered:
        a = 1.0 + a
    np.ascontiguousarray(a.astype(np.float16)).tofile(path)


    np.ascontiguousarray(np.asarray(arr).astype(np.float16)).tofile(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", default=None, help="只转换名字含子串的张量，逗号分隔（调试用）")
    ap.add_argument("--requant-extra", default="", help="额外量化为 W4 的张量子串，逗号分隔（如 embed_tokens,lm_head）")
    ap.add_argument("--skip-vision", action="store_true", default=True)
    ap.add_argument("--verify", action="store_true", help="转换后抽样回读校验（pack/unpack 自洽）")
    ap.add_argument("--group", type=int, default=32, help="重新量化的 group 大小")
    args = ap.parse_args()

    if args.only:
        args.only = [x for x in args.only.split(",") if x]
    os.makedirs(args.out, exist_ok=True)
    index_path = os.path.join(args.src, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path) as f:
            weight_map = json.load(f)["weight_map"]
        shards = sorted(set(weight_map.values()))
    else:
        shards = sorted(x for x in os.listdir(args.src) if x.endswith(".safetensors"))
    if not shards:
        print("ERROR: no safetensors found in", args.src)
        return 2

    # 收集量化张量的三元组（manifest 合并，支持增量转换）
    quant_bases = {}
    manifest = {}
    man_path = os.path.join(args.out, "manifest.json")
    if os.path.exists(man_path):
        try:
            manifest = json.load(open(man_path, encoding="utf-8"))["tensors"]
            print("manifest loaded: %d tensors" % len(manifest), flush=True)
        except Exception as e:
            print("manifest load failed:", e)
    total_in = total_out = 0
    tmpdir = os.path.join(args.out, "_tmp")
    os.makedirs(tmpdir, exist_ok=True)

    print("shards:", shards)
    for shard in shards:
        path = os.path.join(args.src, shard)
        if shard.endswith(".incomplete"):
            continue
        print("  reading", shard, flush=True)
        tens = read_safetensors(path, only=args.only)
        # 第一遍：找 packed / scale / zero_point
        for name in list(tens.keys()):
            if name.endswith(".weight_packed"):
                base = name[: -len(".weight_packed")]
                quant_bases.setdefault(base, {})["packed"] = tens[name]
            elif name.endswith(".weight_scale"):
                base = name[: -len(".weight_scale")]
                quant_bases.setdefault(base, {})["scale"] = tens[name]
            elif name.endswith(".weight_zero_point"):
                base = name[: -len(".weight_zero_point")]
                quant_bases.setdefault(base, {})["zp"] = tens[name]
        # 非量化张量：先缓存（避免两个 shard 交叠；这里简单处理：当场转存）
        for name, arr in tens.items():
            if any(name.endswith(sfx) for sfx in (".weight_packed", ".weight_scale",
                                                  ".weight_zero_point", ".weight_shape")):
                continue
            if args.skip_vision and ("visual" in name or "vision" in name):
                continue
            rel = name.replace(".", "/")
            extras = [x for x in args.requant_extra.split(",") if x]
            if extras and any(e in name for e in extras):
                # 额外量化（分块，避免 5GB 级临时数组）
                w_all = arr.reshape(arr.shape[0], -1).astype(np.float32)
                n2, k2 = w_all.shape
                ng2 = k2 // args.group
                sbp2 = (ng2 + 15) // 16 * 16
                outp_w = os.path.join(args.out, rel + ".w4.bin")
                outp_s = os.path.join(args.out, rel + ".s.bin")
                outp_sz = os.path.join(args.out, rel + ".sz.bin")
                os.makedirs(os.path.dirname(outp_w), exist_ok=True)
                CH2 = 4096
                nw = 0
                with open(outp_w, "wb") as fw, open(outp_s, "wb") as fs, open(outp_sz, "wb") as fsz:
                    for r0 in range(0, n2, CH2):
                        r1 = min(n2, r0 + CH2)
                        q2, sgrp, szgrp = requant_group(w_all[r0:r1], args.group)
                        words = pack_phase(q2)
                        words.tofile(fw)
                        sp = np.zeros((r1 - r0, sbp2), dtype=np.float16)
                        szp = np.zeros((r1 - r0, sbp2), dtype=np.float16)
                        sp[:, :ng2] = sgrp
                        szp[:, :ng2] = szgrp
                        sp.tofile(fs)
                        szp.tofile(fsz)
                        nw += words.nbytes + sp.nbytes + szp.nbytes
                # 删除旧的 f16 文件（若存在）
                oldf16 = os.path.join(args.out, rel + ".f16.bin")
                if os.path.exists(oldf16):
                    os.remove(oldf16)
                total_in += arr.size * 2
                total_out += nw
                manifest[name] = {"kind": "w4", "shape": [n2, k2],
                                  "file": os.path.relpath(outp_w, args.out),
                                  "scale_file": os.path.relpath(outp_s, args.out),
                                  "sz_file": os.path.relpath(outp_sz, args.out),
                                  "group": args.group, "bytes": int(nw)}
                print("  %-60s N=%6d K=%5d -> W4 (extra)" % (name, n2, k2), flush=True)
                del w_all
                continue
            outp = os.path.join(args.out, rel + ".f16.bin")
            os.makedirs(os.path.dirname(outp), exist_ok=True)
            # Qwen3.5 的 Qwen3_5RMSNorm 是 1-centered（有效权重 = 1 + w）；
            # GatedDeltaNet 的 RMSNormGated 不是（其 weight 直接相乘）
            one_centered = (rel.endswith("input_layernorm.weight")
                            or rel.endswith("post_attention_layernorm.weight")
                            or rel.endswith("q_norm.weight")
                            or rel.endswith("k_norm.weight")
                            or rel.endswith("language_model.norm.weight"))
            f16_save(outp, arr, one_centered=one_centered)
            total_in += arr.size * 2
            total_out += arr.size * 2
            manifest[name] = {"kind": "f16", "shape": list(arr.shape),
                              "file": os.path.relpath(outp, args.out), "bytes": int(arr.size * 2)}
        del tens

    print("quantized tensors:", len(quant_bases), flush=True)
    for base, parts in quant_bases.items():
        if "packed" not in parts:
            continue
        if args.skip_vision and ("visual" in base or "vision" in base):
            continue
        packed = parts["packed"]                      # [N, K/8] int32
        scale = parts.get("scale")
        zp = parts.get("zp")
        if scale is None:
            print("  WARN no scale for", base)
            continue
        n, c = packed.shape
        k = c * 8
        q = unpack_nibbles(packed, c)
        if zp is not None:
            zp = unpack_zp(zp, n, k // 32)
            w = dequant(q, scale, zp)
        else:
            w = dequant(q, scale, np.zeros_like(scale))
        q2, sgrp, szgrp = requant_group(w, args.group)
        words = pack_phase(q2)
        rel = base.replace(".", "/")
        outp = os.path.join(args.out, rel + ".w4.bin")
        os.makedirs(os.path.dirname(outp), exist_ok=True)
        words.tofile(outp)
        ng = k // args.group
        sbp = (ng + 15) // 16 * 16
        sp = np.zeros((n, sbp), dtype=np.float16)
        szp = np.zeros((n, sbp), dtype=np.float16)
        sp[:, :ng] = sgrp
        szp[:, :ng] = szgrp
        s_path = os.path.join(args.out, rel + ".s.bin")
        sz_path = os.path.join(args.out, rel + ".sz.bin")
        sp.tofile(s_path)
        szp.tofile(sz_path)
        total_in += packed.nbytes + (scale.nbytes if scale is not None else 0)
        total_out += words.nbytes + sp.nbytes + szp.nbytes
        manifest[base] = {"kind": "w4", "shape": [n, k], "file": os.path.relpath(outp, args.out),
                          "scale_file": os.path.relpath(s_path, args.out),
                          "sz_file": os.path.relpath(sz_path, args.out),
                          "group": args.group, "bytes": int(words.nbytes + sp.nbytes + szp.nbytes)}
        if args.verify:
            back = unpack_nibbles_phase(words)
            assert np.array_equal(back, q2), "phase pack/unpack mismatch: " + base
            w_back = (q2.astype(np.float32)) * np.repeat(sgrp.astype(np.float32), args.group, axis=1)[:, :k] \
                     - np.repeat(szgrp.astype(np.float32), args.group, axis=1)[:, :k]
            err = np.abs(w_back - w).mean() / (np.abs(w).mean() + 1e-9)
            print("  %-60s N=%5d K=%5d  requant rel-err=%.4f" % (base, n, k, err), flush=True)
        del packed, scale, zp, w, q, q2, words

    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump({"tensors": manifest}, f, indent=1)
    print("done: %d tensors, in %.2f GB -> out %.2f GB" %
          (len(manifest), total_in / 1e9, total_out / 1e9))
    return 0


def unpack_nibbles_phase(words):
    """uint16 [N,K/4] -> uint8 [N,K]（pack_phase 的逆：第 p 个半字节对应该相位第 j 个元素）"""
    n, kq = words.shape
    q = np.empty((n, kq * 4), dtype=np.uint8)
    qr = q.reshape(n, 4, kq)
    for p in range(4):
        qr[:, p, :] = ((words >> (4 * p)) & 0xF).astype(np.uint8)
    return q


if __name__ == "__main__":
    sys.exit(main())
