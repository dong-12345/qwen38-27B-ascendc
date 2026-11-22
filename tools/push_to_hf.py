"""push_to_hf.py — 把转换好的 W4A16 权重目录上传到 HuggingFace 仓库。

用法：
    pip install huggingface_hub
    huggingface-cli login          # 或者设置环境变量 HF_TOKEN
    python tools/push_to_hf.py --folder converted/qwen38_310b \
        --repo dong2542107715/Qwen3.8-27B-AscendC-W4A16 [--private] [--dry-run]

说明：
  - 权重目录约 16.5 GB / 1672 个文件。默认使用 `upload_large_folder`（分块/并行/可断点续传），
    中断后重跑同一条命令即可继续；加 `--simple` 可退回普通 `upload_folder`。
  - 会自动把 docs/MODEL_CARD_HF.md 作为 README.md 一起上传（若仓库里还没有 README）。
  - 需要仓库写入权限（HF token，write 权限）。
"""
import argparse
import os
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", required=True, help="本地权重目录（如 converted/qwen38_310b）")
    ap.add_argument("--repo", required=True, help="目标仓库 id，如 dong2542107715/Qwen3.8-27B-AscendC-W4A16")
    ap.add_argument("--private", action="store_true", help="创建为私有仓库")
    ap.add_argument("--revision", default="main")
    ap.add_argument("--model-card", default="docs/MODEL_CARD_HF.md")
    ap.add_argument("--simple", action="store_true", help="用普通 upload_folder（单次提交，不适合超大目录）")
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
        try:
            from huggingface_hub import HfFolder
            token = HfFolder.get_token()
        except Exception:
            token = None
    if not token:
        sys.exit("未找到 HF token：请先运行 `huggingface-cli login`，或设置环境变量 HF_TOKEN")

    from huggingface_hub import HfApi, create_repo

    api = HfApi(token=token)
    try:
        who = api.whoami()
        print("已登录账号: %s" % who.get("name"))
    except Exception as e:
        print("[warn] whoami 失败（token 可能无效）: %s" % e)

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
        except Exception as e:
            print("[warn] model card 上传失败（可能已存在）: %s" % e)

    if args.simple:
        from huggingface_hub import upload_folder
        upload_folder(
            folder_path=args.folder,
            repo_id=args.repo,
            repo_type="model",
            revision=args.revision,
            commit_message="Upload W4A16 phase-interleaved weights for Ascend 310B",
            ignore_patterns=["*.tmp", "*.log"],
        )
    else:
        from huggingface_hub import upload_large_folder
        print("使用 upload_large_folder（16.5GB，支持断点续传；中断后重跑本命令即可）...")
        upload_large_folder(
            folder_path=args.folder,
            repo_id=args.repo,
            repo_type="model",
            revision=args.revision,
            private=args.private,
            ignore_patterns=["*.tmp", "*.log"],
        )
    print("完成: https://huggingface.co/%s" % args.repo)


if __name__ == "__main__":
    main()
