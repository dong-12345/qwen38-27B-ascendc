#!/usr/bin/env python3
"""run_qa.py — 一条命令跑 Moondream3 图像问答（PC 预处理 + SSH 板端推理 + 解码）。

用法: python tools/run_qa.py --image xx.png --question "What color is the circle?" --gen 16
"""
import argparse, os, subprocess, sys
from tokenizers import Tokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ISO = os.path.join(ROOT, "tmp", "crops.bin")
TOK = os.path.join(ROOT, "models", "starmie-v1", "tokenizer.json")
KEY = os.environ.get("MD3_KEY", os.path.expanduser("~/.ssh/id_rsa"))
BOARD = os.environ.get("MD3_BOARD", "HwHiAiUser@<board-ip>")
ENV = ("cd ~/code/moondream3_310b/ops && source /usr/local/Ascend/ascend-toolkit/set_env.sh && "
       "export LD_LIBRARY_PATH=$PWD/out/lib:$PWD/out/lib64:/usr/local/Ascend/ascend-toolkit/latest/lib64:$LD_LIBRARY_PATH && ")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--question", required=True)
    ap.add_argument("--gen", type=int, default=16)
    ap.add_argument("--model-dir", default="~/code/moondream3_310b/models/moondream3_310b")
    ap.add_argument("--skip-upload", action="store_true")
    args = ap.parse_args()

    tk = Tokenizer.from_file(TOK)
    qids = [1, 15381, 2] + tk.encode(args.question).ids + [3]
    print("[qa] qids:", ",".join(map(str, qids)))
    if not args.skip_upload:
        subprocess.run([sys.executable, os.path.join(ROOT, "code", "moondream3_310b", "tools", "prep_image.py"),
                        "--image", args.image, "--out", ISO], check=True)
        subprocess.run(["scp", "-i", KEY, "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=NUL",
                        "-o", "LogLevel=ERROR", ISO, BOARD + ":/tmp/crops.bin"], check=True)
    cmd = (ENV + "./md3_engine --image /tmp/crops.bin --ncrops 2 --qids " + ",".join(map(str, qids)) +
           " --gen %d --dir %s" % (args.gen, args.model_dir))
    out = subprocess.run(["ssh", "-i", KEY, "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=NUL",
                          "-o", "LogLevel=ERROR", BOARD, cmd], capture_output=True, text=True)
    sys.stderr.write(out.stderr[-2000:] if out.stderr else "")
    toks = None
    for line in out.stdout.splitlines():
        if line.startswith("TOKENS"):
            toks = [int(x) for x in line.split()[1:]]
    if toks is None:
        print("[qa] failed, raw:", out.stdout[-2000:])
        return 1
    print("[qa] toks:", toks)
    print("[qa] answer:", repr(tk.decode(toks)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
