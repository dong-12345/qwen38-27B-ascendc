#!/usr/bin/env python3
"""make_smoke_subset.py — 从转换后的 27B 权重裁剪出小模型子集，用于引擎冒烟测试。

选取 2 个源层（一个 GDN、一个全注意力）+ 截断词表的 embed/lm_head + final norm，
写成 harness 目录（meta.txt 同步改写：n_layers=2, vocab=1024, layer_types=1,0，
EMBED/LM_HEAD 形状改为 1024x5120，L 行 role 与路径都重映射）。
体积约 540MB，便于快速推到板上验证引擎链路。
"""
import os
import re
import shutil

SRC = "converted/qwen38_310b"
DST = "converted/qwen38_smoke"
SRC_LAYERS = [0, 3]          # 0 = linear_attention(GDN), 3 = full_attention
SMALL_VOCAB = 1024
HIDDEN = 5120


def copy_layer(src_li, dst_li):
    s = os.path.join(SRC, "model/language_model/layers", str(src_li))
    d = os.path.join(DST, "model/language_model/layers", str(dst_li))
    if not os.path.isdir(s):
        print("  [skip] source layer missing:", s)
        return False
    os.makedirs(d, exist_ok=True)
    n = 0
    for root, _, files in os.walk(s):
        for f in files:
            sp = os.path.join(root, f)
            rel = os.path.relpath(sp, s)
            dp = os.path.join(d, rel)
            os.makedirs(os.path.dirname(dp), exist_ok=True)
            shutil.copyfile(sp, dp)
            n += 1
    print("  layer %d -> %d : %d files" % (src_li, dst_li, n))
    return True


def trunc_rows(src_path, dst_path, rows, row_bytes=None):
    if not os.path.exists(src_path):
        print("  [skip] missing", src_path)
        return False
    nbytes = rows * (row_bytes if row_bytes else HIDDEN * 2)
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    with open(src_path, "rb") as f:
        data = f.read(nbytes)
    with open(dst_path, "wb") as f:
        f.write(data)
    print("  %s -> %d rows (%.1f MB)" % (os.path.basename(src_path), rows, len(data) / 1e6))
    return True


def trunc_w4(src_base, dst_base, rows, k):
    """截断 W4 张量：w4 每行 k/4 个 uint16；s/sz 每行 SBP 个 fp16"""
    kq = k // 4
    ng = k // 32
    sbp = (ng + 15) // 16 * 16
    ok = True
    for suf, rowb in ((".w4.bin", kq * 2), (".s.bin", sbp * 2), (".sz.bin", sbp * 2)):
        ok = trunc_rows(src_base + suf, dst_base + suf, rows, row_bytes=rowb) and ok
    return ok


def main():
    if os.path.isdir(DST):
        shutil.rmtree(DST)
    os.makedirs(DST, exist_ok=True)
    for i, src_li in enumerate(SRC_LAYERS):
        copy_layer(src_li, i)
    # embed / lm_head：优先 W4（新格式），否则 f16
    for tag, sub in (("EMBED", "model/language_model/embed_tokens/weight"),
                     ("LM_HEAD", "lm_head/weight")):
        src_base = os.path.join(SRC, sub)
        dst_base = os.path.join(DST, sub)
        if os.path.exists(src_base + ".w4.bin"):
            trunc_w4(src_base, dst_base, SMALL_VOCAB, HIDDEN)
        else:
            trunc_rows(src_base + ".f16.bin", dst_base + ".f16.bin", SMALL_VOCAB)
    fn = os.path.join(SRC, "model/language_model/norm/weight.f16.bin")
    if os.path.exists(fn):
        os.makedirs(os.path.join(DST, "model/language_model/norm"), exist_ok=True)
        shutil.copyfile(fn, os.path.join(DST, "model/language_model/norm/weight.f16.bin"))

    lines = open(os.path.join(SRC, "meta.txt"), encoding="utf-8").read().splitlines()
    out = []
    for ln in lines:
        parts = ln.split(" ")
        if parts[0] == "CFG":
            key = parts[1]
            if key == "n_layers":
                out.append("CFG n_layers %d" % len(SRC_LAYERS))
            elif key == "vocab":
                out.append("CFG vocab %d" % SMALL_VOCAB)
            elif key == "layer_types":
                types = ["1" if li % 4 != 3 else "0" for li in SRC_LAYERS]
                out.append("CFG layer_types " + ",".join(types))
            else:
                out.append(ln)
            continue
        if parts[0] in ("EMBED", "LM_HEAD"):
            if parts[-1] == "w4":
                parts[4] = str(SMALL_VOCAB)
                parts[5] = str(HIDDEN)
            else:
                parts[2] = "%dx%d" % (SMALL_VOCAB, HIDDEN)
            out.append(" ".join(parts))
            continue
        m = re.match(r"^L(\d+)\.(.*)$", ln)
        if m:
            src_li = int(m.group(1))
            if src_li not in SRC_LAYERS:
                continue
            dst_li = SRC_LAYERS.index(src_li)
            rest = m.group(2).replace("layers/%d/" % src_li, "layers/%d/" % dst_li)
            out.append("L%d.%s" % (dst_li, rest))
            continue
        out.append(ln)
    with open(os.path.join(DST, "meta.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")
    total = sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(DST) for f in fs)
    print("smoke subset: %d layers, %d meta lines, %.1f MB" % (len(SRC_LAYERS), len(out), total / 1e6))


if __name__ == "__main__":
    main()
