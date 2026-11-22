#!/usr/bin/env python3
"""run_27b_demo.py — 端到端跑板子上的 27B 引擎：本地编码 prompt → 板子生成 → 本地解码。

用法:
    python tools/run_27b_demo.py --prompt "你好，请用一句话介绍你自己。" --gen 24
    python tools/run_27b_demo.py --prompt "..." --gen 24 --dir ~/models/qwen38_310b
"""
import argparse
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KEY = os.path.join(ROOT, "tools", "opi_key")
BOARD = "HwHiAiUser@192.168.137.163"
TOK_JSON = os.path.join(ROOT, "models", "Qwen3.8-27B-AWQ-INT4", "tokenizer.json")
REMOTE_DIR = os.path.join(ROOT, "ascendc_ops")


def ssh_run(cmd, timeout=3600):
    return subprocess.run(
        ["ssh", "-i", KEY, "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=NUL",
         "-o", "ConnectTimeout=20", "-o", "ServerAliveInterval=30", "-o", "LogLevel=ERROR",
         BOARD, cmd],
        capture_output=True, text=True, timeout=timeout)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="你好，请用一句话介绍你自己。")
    ap.add_argument("--gen", type=int, default=24)
    ap.add_argument("--dir", default="~/models/qwen38_310b", help="板子上的权重目录")
    args = ap.parse_args()

    from tokenizers import Tokenizer

    tk = Tokenizer.from_file(TOK_JSON)
    text = ("<|im_start|>user\n" + args.prompt + "<|im_end|>\n<|im_start|>assistant\n")
    ids = tk.encode(text, add_special_tokens=False).ids
    print("prompt (%d tokens): %r" % (len(ids), args.prompt))

    ids_str = ",".join(str(i) for i in ids)
    remote = (
        "export LC_ALL=C; cd ~/qwen38_310b/ascendc_ops; "
        "source /usr/local/Ascend/ascend-toolkit/latest/bin/setenv.bash >/dev/null 2>&1; "
        "export LD_LIBRARY_PATH=$PWD/out/lib:$PWD/out/lib64:"
        "/usr/local/Ascend/ascend-toolkit/latest/lib64:$LD_LIBRARY_PATH; "
        "ENG27_IDS=%s ENG27_GEN=%d ENG27_DIR=%s timeout -s KILL 3000 ./qwen38_ops_test engine_27b 2>&1"
        % (ids_str, args.gen, args.dir)
    )
    print("running on board ...")
    r = ssh_run(remote, timeout=3600)
    out = r.stdout + r.stderr
    print(out.rstrip())

    m = re.search(r"^TOKENS((?: \d+)+)", out, re.M)
    if not m:
        print("\n[FAIL] no TOKENS line; see log above")
        return 1
    gen_ids = [int(x) for x in m.group(1).split()]
    gen_text = tk.decode(gen_ids, skip_special_tokens=False)
    print("\n================ 生成结果 ================")
    print(gen_text)
    print("=========================================")
    print("generated ids:", gen_ids)
    return 0


if __name__ == "__main__":
    sys.exit(main())
