#!/usr/bin/env python3
"""export_engine_meta.py — 从转换后的 manifest + 模型 config 生成引擎所需的扁平元数据。

输出 <converted>/meta.txt，每行：
  <role> <relative_path> <count_or_shape...> <kind>
其中 kind: w4 = uint16 nibble 打包（配 .s.bin/.sz.bin）、f16 = fp16 裸数据。
引擎按 role 顺序加载即可，无需 JSON 解析。
"""
import json
import os
import sys

SRC = "models/Qwen3.8-27B-AWQ-INT4"
CONV = sys.argv[1] if len(sys.argv) > 1 else "converted/qwen38_310b"


def main():
    cfg = json.load(open(os.path.join(SRC, "config.json"), encoding="utf-8"))
    tc = cfg.get("text_config", cfg)
    man_path = os.path.join(CONV, "manifest.json")
    if os.path.exists(man_path):
        man = json.load(open(man_path, encoding="utf-8"))["tensors"]
    else:
        # 从 safetensors header + weight_shape 张量读取期望条目（转换未完成时用）
        import struct
        man = {}
        # 第一遍：收集所有 shard 的 header + weight_shape 值
        shapes = {}
        allmeta = {}
        for f in sorted(os.listdir(SRC)):
            if not f.endswith(".safetensors"):
                continue
            path = os.path.join(SRC, f)
            with open(path, "rb") as fh:
                n = struct.unpack("<Q", fh.read(8))[0]
                h = json.loads(fh.read(n).decode("utf-8"))
                base_off = 8 + n
                for name, meta in h.items():
                    if name == "__metadata__":
                        continue
                    allmeta[name] = meta
                    if name.endswith(".weight_shape"):
                        offs = meta["data_offsets"]
                        fh.seek(base_off + offs[0])
                        out_dim, in_dim = struct.unpack("<qq", fh.read(16))
                        shapes[name] = (int(out_dim), int(in_dim))
        # 第二遍：构造条目
        for name, meta in allmeta.items():
            if name.endswith(".weight_packed"):
                b = name[: -len(".weight_packed")]
                if (b + ".weight_shape") in shapes:
                    man[b] = {"kind": "w4", "shape": list(shapes[b + ".weight_shape"])}
            elif not any(name.endswith(x) for x in (".weight_scale", ".weight_zero_point", ".weight_shape")):
                man[name] = {"kind": "f16", "shape": meta["shape"]}
    layer_types = tc["layer_types"]
    n_layers = tc["num_hidden_layers"]
    lines = []
    missing = []

    def add(role, base, kind_hint=None):
        if base not in man and (base + ".weight") in man:
            base = base + ".weight"   # 未量化张量在 safetensors 里带 .weight 后缀
        if base not in man:
            missing.append(base)
            return
        e = man[base]
        kind = e["kind"]
        rel = base.replace(".", "/")
        if kind == "w4":
            n, k = e["shape"]
            lines.append(f"{role} {rel}.w4.bin {rel}.s.bin {rel}.sz.bin {n} {k} w4")
        else:
            shp = "x".join(str(x) for x in e["shape"])
            lines.append(f"{role} {rel}.f16.bin {shp} f16")

    P = "model.language_model."
    add("EMBED", P + "embed_tokens")
    add("LM_HEAD", "lm_head.weight")
    add("LM_HEAD", P + "lm_head")
    add("FINAL_NORM", P + "norm")
    for i in range(n_layers):
        L = f"{P}layers.{i}."
        add(f"L{i}.LN1", L + "input_layernorm")
        add(f"L{i}.LN2", L + "post_attention_layernorm")
        add(f"L{i}.MGATE", L + "mlp.gate_proj")
        add(f"L{i}.MUP", L + "mlp.up_proj")
        add(f"L{i}.MDOWN", L + "mlp.down_proj")
        if layer_types[i] == "linear_attention":
            A = L + "linear_attn."
            add(f"L{i}.QKV", A + "in_proj_qkv")
            add(f"L{i}.Z", A + "in_proj_z")
            add(f"L{i}.APROJ", A + "in_proj_a")
            add(f"L{i}.BPROJ", A + "in_proj_b")
            add(f"L{i}.CONV", A + "conv1d")
            add(f"L{i}.ALOG", A + "A_log")
            add(f"L{i}.DTBIAS", A + "dt_bias")
            add(f"L{i}.GNORM", A + "norm")
            add(f"L{i}.OUT", A + "out_proj")
        else:
            S = L + "self_attn."
            add(f"L{i}.QPROJ", S + "q_proj")
            add(f"L{i}.KPROJ", S + "k_proj")
            add(f"L{i}.VPROJ", S + "v_proj")
            add(f"L{i}.OPROJ", S + "o_proj")
            add(f"L{i}.QNORM", S + "q_norm")
            add(f"L{i}.KNORM", S + "k_norm")

    meta = {
        "hidden": tc["hidden_size"], "n_layers": n_layers, "ff": tc["intermediate_size"],
        "vocab": tc["vocab_size"], "head_dim": tc["head_dim"],
        "n_q_heads": tc["num_attention_heads"], "n_kv_heads": tc["num_key_value_heads"],
        "rotary_dim": int(tc["head_dim"] * tc.get("partial_rotary_factor", 1.0)),
        "gdn_k_heads": tc["linear_num_key_heads"], "gdn_v_heads": tc["linear_num_value_heads"],
        "gdn_head_dim": tc["linear_key_head_dim"], "conv_k": tc["linear_conv_kernel_dim"],
        "eps": tc.get("rms_norm_eps", 1e-6), "layer_types": layer_types,
        "max_t": 512,
    }
    with open(os.path.join(CONV, "meta.txt"), "w", encoding="utf-8") as f:
        # 扁平 CFG（方便 C++ 直接解析，避免引入 JSON 库）
        for k in ("hidden", "n_layers", "ff", "vocab", "head_dim", "n_q_heads", "n_kv_heads",
                  "rotary_dim", "gdn_k_heads", "gdn_v_heads", "gdn_head_dim", "conv_k", "max_t"):
            f.write(f"CFG {k} {meta[k]}\n")
        f.write(f"CFG eps {meta['eps']}\n")
        f.write("CFG layer_types " + ",".join("1" if t == "linear_attention" else "0" for t in meta["layer_types"]) + "\n")
        for ln in lines:
            f.write(ln + "\n")
    print("wrote meta.txt: %d tensor entries, %d missing" % (len(lines), len(missing)))
    for m in missing[:20]:
        print("  MISSING:", m)
    # 统计
    tot = 0
    for ln in lines:
        parts = ln.split()
        path = os.path.join(CONV, parts[1])
        if os.path.exists(path):
            tot += os.path.getsize(path)
            if parts[-1] == "w4":
                tot += os.path.getsize(os.path.join(CONV, parts[2]))
                tot += os.path.getsize(os.path.join(CONV, parts[3]))
    print("total weight bytes: %.2f GB" % (tot / 1e9))


if __name__ == "__main__":
    main()
