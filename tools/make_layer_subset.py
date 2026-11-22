#!/usr/bin/env python3
"""make_layer_subset.py — 用已完整落地的部分层构造一个可运行的子集模型 meta。

用法: python tools/make_layer_subset.py --src converted/qwen38_310b --layers 10-50 --out converted/qwen38_sub41
生成的 <out>/meta.txt 中 L<i> 角色映射到源层 (start+i)，路径用相对 src 的 "../<srcname>/..."。
其余（EMBED/LM_HEAD/FINAL_NORM）直接引用原路径。
"""
import argparse
import json
import os
import re


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="converted/qwen38_310b")
    ap.add_argument("--layers", default="10-50")
    ap.add_argument("--groups", default=None, help="逗号分隔的层号列表，覆盖 --layers")
    ap.add_argument("--out", default="converted/qwen38_sub41")
    args = ap.parse_args()

    if args.groups:
        src_layers = [int(x) for x in args.groups.split(",")]
    else:
        a, b = args.layers.split("-")
        src_layers = list(range(int(a), int(b) + 1))

    base = os.path.abspath(args.src)
    outdir = os.path.abspath(args.out)
    os.makedirs(outdir, exist_ok=True)
    rel_prefix = os.path.relpath(base, outdir).replace("\\", "/") + "/"

    lines = open(os.path.join(base, "meta.txt"), encoding="utf-8").read().splitlines()
    cfg = {}
    body = []
    for ln in lines:
        if ln.startswith("CFG "):
            p = ln.split(" ")
            cfg[p[1]] = " ".join(p[2:])
        else:
            body.append(ln)

    types = cfg["layer_types"].split(",")
    out = []
    for k in ("hidden", "ff", "vocab", "head_dim", "n_q_heads", "n_kv_heads", "rotary_dim",
              "gdn_k_heads", "gdn_v_heads", "gdn_head_dim", "conv_k", "max_t", "eps"):
        if k in cfg:
            out.append("CFG %s %s" % (k, cfg[k]))
    out.append("CFG n_layers %d" % len(src_layers))
    out.append("CFG layer_types " + ",".join(types[i] for i in src_layers))

    for ln in body:
        p = ln.split(" ")
        role = p[0]
        m = re.match(r"^L(\d+)\.(.*)$", ln)
        if m:
            src_li = int(m.group(1))
            if src_li not in src_layers:
                continue
            dst_li = src_layers.index(src_li)
            rest = m.group(2)
            # 路径加 ../<src>/ 前缀
            parts = rest.split(" ")
            parts = [rel_prefix + x if (x.endswith(".bin")) else x for x in parts]
            out.append("L%d.%s" % (dst_li, " ".join(parts)))
        elif role in ("EMBED", "LM_HEAD", "FINAL_NORM"):
            parts = p[1:]
            parts = [rel_prefix + x if (x.endswith(".bin")) else x for x in parts]
            out.append(role + " " + " ".join(parts))

    with open(os.path.join(outdir, "meta.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")
    print("wrote %s/meta.txt: %d lines, layers %d (%d..%d)" % (outdir, len(out), len(src_layers),
                                                               src_layers[0], src_layers[-1]))
    # 校验引用文件存在
    missing = 0
    for ln in out:
        p = ln.split(" ")
        if p[0] == "CFG":
            continue
        for f in p[1:]:
            if f.endswith(".bin"):
                if not os.path.exists(os.path.join(outdir, f)):
                    missing += 1
                    if missing <= 3:
                        print("  MISSING:", f)
    print("missing refs:", missing)


if __name__ == "__main__":
    main()
