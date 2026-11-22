#!/usr/bin/env python3
"""push_to_hf.py — 把转换好的 W4A16 权重目录上传到 HuggingFace 仓库。

用法：
    pip install huggingface_hub
    export HF_TOKEN=hf_xxx
    python tools/push_to_hf.py --folder converted/qwen38_310b \
        --repo your-name/Qwen3_8-27B-AscendC-W4A16 [--private] [--dry-run]

说明：
  - 权重目录约 16.9 GB / 1670 个文件，`upload_folder` 支持断点续传，中断后重跑即可。
  - 会自动把 docs/MODEL_CARD_HF.md 作为 README.md 一起上传（若仓库里还没有 README）。
  - 需要仓库写入权限（HF token，建议 `write` 权限）。
"""
import argparse
import os
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", required=True, help="本地权重目录（如 converted/qwen38_310b）")
    ap.add_argument("--repo", required=True, help="目标仓库 id，如 your-name/Qwen3_8-27B-AscendC-W4A16")
    ap.add_argument("--private", action="store_true", help="创建为私有仓库")
    ap.add_argument("--revision", default="main")
    ap.add_argument("--model-card", default="docs/MODEL_CARD_HF.md")
    ap.add_argument("--dry-run", action="store_true", help="只统计要上传的文件与大小")
    args = ap.parse_args()

    if not os.path.isdir(args.folder):
        sys.exit("目录不存在: %s" % args.folder)

    files = []
    total = 0
    for root, _dirs, names in os.walk(args.folder):
        for n in names:
            p = os.path.join(root, n)
            files.append(p)
            total += os.path.getsize(p)
    print("待上传: %d 个文件, %.2f GB" % (len(files), total / 1e9))
    print("目标仓库: %s (private=%s)" % (args.repo, args.private))

    if args.dry_run:
        for p in files[:10]:
            print("  " + os.path.relpath(p, args.folder))
        if len(files) > 10:
            print("  ... 其余 %d 个文件" % (len(files) - 10))
        return

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if not token:
        sys.exit("请先设置环境变量 HF_TOKEN（HuggingFace 写入 token）")

    from huggingface_hub import HfApi, create_repo, upload_folder

    api = HfApi(token=token)
    create_repo(args.repo, token=token, private=args.private, repo_type="model", exist_ok=True)
    print("仓库就绪: https://huggingface.co/%s" % args.repo)

    if os.path.exists(args.model_card):
        try:
            api.upload_file(
                path_or_fileobj=args.model_card,
                path_in_repo="README.md",
                repo_id=args.repo,
                repo_type="model",
                commit_message="Add model card",
            )
            print("已上传 model card -> README.md")
        except Exception as e:  # 已有 README 时不覆盖也可以
            print("[warn] model card 上传失败（可能已存在）: %s" % e)

    print("开始上传权重（大文件会自动分片，支持断点续传）...")
    upload_folder(
        folder_path=args.folder,
        repo_id=args.repo,
        repo_type="model",
        commit_message="Upload W4A16 phase-interleaved weights for Ascend 310B",
        ignore_patterns=["*.tmp", "*.log"],
    )
    print("完成: https://huggingface.co/%s" % args.repo)


if __name__ == "__main__":
    main()
