#!/usr/bin/env python3
"""prepare_hf_upload.py — 构建 Hugging Face 上传目录。

用法：
  python tools/prepare_hf_upload.py --out D:\\hf_upload_moondream3
  python tools/prepare_hf_upload.py --out D:\\hf_upload_moondream3 --tar-weights

结构：
  README.md      <- MODEL_CARD.md
  LICENSE
  weights/       <- converted/moondream3_310b 原样（meta.txt + 文本 W4 + vision/*.f16）
  tokenizer/     <- starmie-v1 tokenizer 文件
  tools/ docs/   <- 项目脚本与文档
"""
import argparse, os, shutil, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WS = os.path.dirname(os.path.dirname(ROOT))


def link_or_copy(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def copy_tree(src, dst, ext=None):
    n = 0
    for dirpath, _dirnames, filenames in os.walk(src):
        for fn in filenames:
            if ext and not fn.lower().endswith(ext):
                continue
            s = os.path.join(dirpath, fn)
            rel = os.path.relpath(s, src)
            link_or_copy(s, os.path.join(dst, rel))
            n += 1
    return n


def dir_stats(path):
    total = 0
    cnt = 0
    for dirpath, _d, filenames in os.walk(path):
        for fn in filenames:
            total += os.path.getsize(os.path.join(dirpath, fn))
            cnt += 1
    return cnt, total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--converted', default=os.path.join(WS, 'converted', 'moondream3_310b'))
    ap.add_argument('--tokenizer', default=os.path.join(WS, 'models', 'starmie-v1'))
    ap.add_argument('--tar-weights', action='store_true', help='把 weights/ 打成单个 weights.tar（适合大文件上传）')
    args = ap.parse_args()

    out = os.path.abspath(args.out)
    if not os.path.isdir(args.converted):
        print('[hf] converted not found:', args.converted); return 1
    os.makedirs(out, exist_ok=True)

    shutil.copy2(os.path.join(ROOT, 'MODEL_CARD.md'), os.path.join(out, 'README.md'))
    shutil.copy2(os.path.join(ROOT, 'LICENSE'), os.path.join(out, 'LICENSE'))

    nw = copy_tree(args.converted, os.path.join(out, 'weights'))
    print('[hf] weights: %d files' % nw)

    if os.path.isdir(args.tokenizer):
        nt = copy_tree(args.tokenizer, os.path.join(out, 'tokenizer'))
        print('[hf] tokenizer: %d files' % nt)

    ntool = 0
    tdir = os.path.join(ROOT, 'tools')
    for fn in os.listdir(tdir):
        if fn.endswith(('.py', '.sh')):
            link_or_copy(os.path.join(tdir, fn), os.path.join(out, 'tools', fn)); ntool += 1
    print('[hf] tools: %d files' % ntool)

    nd = copy_tree(os.path.join(ROOT, 'docs'), os.path.join(out, 'docs'), ext='.md')
    print('[hf] docs: %d files' % nd)

    if args.tar_weights:
        import tarfile
        tp = os.path.join(out, 'weights.tar')
        print('[hf] packing weights.tar ...')
        with tarfile.open(tp, 'w') as tf:
            tf.add(os.path.join(out, 'weights'), arcname='weights')
        print('[hf] weights.tar: %.1f MB' % (os.path.getsize(tp) / 1048576.0))
        shutil.rmtree(os.path.join(out, 'weights'))

    cnt, size = dir_stats(out)
    print('[hf] done: %s  (%d files, %.1f MB)' % (out, cnt, size / 1048576.0))
    print('[hf] next: hf repo create <user>/moondream3-preview-ascend310b --repo-type model')
    print('       hf upload <user>/moondream3-preview-ascend310b %s . --repo-type model' % out)
    return 0


if __name__ == '__main__':
    sys.exit(main())
