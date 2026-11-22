#!/usr/bin/env python3
"""run_27b_demo.py — 端到端跑板子上的 27B 引擎：本地编码 prompt → 板子生成 → 本地解码。

板端目录（2026-11 重组后）：
    ~/code/qwen38_310B/src/out/bin/qwen38_ops_test   （二进制）
    ~/code/qwen38_310B/models/qwen38_310b            （16.9GB 权重）
    ~/code/qwen38_310B/env.sh                        （CANN 环境 + ENG27_DIR）

用法:
    python tools/run_27b_demo.py --prompt "你好，请用一句话介绍你自己。" --gen 24
    python tools/run_27b_demo.py --prompt "..." --gen 24 --dir ~/code/qwen38_310B/models/qwen38_310b
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
BOARD_ROOT = "~/code/qwen38_310B"
DEFAULT_DIR = BOARD_ROOT + "/models/qwen38_310b"


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
    ap.add_argument("--dir", default=DEFAULT_DIR, help="板子上的权重目录")
    args = ap.parse_args()

    from tokenizers import Tokenizer

    tk = Tokenizer.from_file(TOK_JSON)
    prefix = ("<|im_start|>user\n" + args.prompt + "<|im_end|>\n<|im_start|>assistant\n")
    ids = tk.encode(prefix, add_special_tokens=False).ids
    # 官方 chat template：thinking 关闭时 assistant 轮以 "<think>\n\n</think>\n\n" 开头
    ids = list(ids) + [248068, 271, 248069, 271]
    print("prompt (%d tokens): %r" % (len(ids), args.prompt))

    ids_str = ",".join(str(i) for i in ids)
    remote = (
        "export LC_ALL=C; source %s/env.sh >/dev/null 2>&1; "
        "cd %s/src; "
        "ENG27_IDS=%s ENG27_GEN=%d ENG27_DIR=%s timeout -s KILL 3000 "
        "./out/bin/qwen38_ops_test engine_27b 2>&1"
        % (BOARD_ROOT, BOARD_ROOT, ids_str, args.gen, args.dir)
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
