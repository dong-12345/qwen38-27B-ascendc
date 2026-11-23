#!/usr/bin/env python3
"""run_demo.py — PC 端：tokenizer 编码 + SSH 调板端引擎 + 解码输出。

用法:
  python tools/run_demo.py --prompt "The capital of France is" --gen 8
  python tools/run_demo.py --prompt "你好，请用一句话介绍你自己" --gen 24
"""
import argparse, os, re, subprocess, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KEY = os.environ.get("MD3_KEY", os.path.expanduser("~/.ssh/id_rsa"))
BOARD = os.environ.get("MD3_BOARD", "HwHiAiUser@<board-ip>")
BOARD_ROOT = "~/code/moondream3_310b"
TOK = os.path.join(ROOT, "..", "..", "models", "starmie-v1", "tokenizer.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--gen", type=int, default=8)
    ap.add_argument("--dir", default=BOARD_ROOT + "/models/moondream3_310b")
    ap.add_argument("--raw-ids", default="", help="直接给 token id 列表（调试）")
    args = ap.parse_args()

    from tokenizers import Tokenizer
    tk = Tokenizer.from_file(TOK)
    if args.raw_ids:
        ids = [int(x) for x in args.raw_ids.split(",")]
    else:
        # 文本 query 模板（reasoning=False）：[bos=0, 1, 15381, 2] + question + [3]
        q = tk.encode(args.prompt).ids
        ids = [0, 1, 15381, 2] + q + [3]
    print("prompt ids:", ids)

    idstr = ",".join(str(i) for i in ids)
    cmd = ("cd %s/ops && source /usr/local/Ascend/ascend-toolkit/set_env.sh && "
           "export LD_LIBRARY_PATH=$PWD/out/lib:$PWD/out/lib64:/usr/local/Ascend/ascend-toolkit/latest/lib64:$LD_LIBRARY_PATH && "
           "./md3_engine --dir %s --ids %s --gen %d 2>&1") % (BOARD_ROOT, args.dir, idstr, args.gen)
    r = subprocess.run(["ssh", "-i", KEY, "-o", "StrictHostKeyChecking=no",
                        "-o", "UserKnownHostsFile=NUL", "-o", "ConnectTimeout=20",
                        "-o", "ServerAliveInterval=30", "-o", "LogLevel=ERROR", BOARD, cmd],
                       capture_output=True, text=True, timeout=7200)
    out = r.stdout + r.stderr
    print(out)
    m = re.search(r"^TOKENS((?: \d+)+)", out, re.M)
    if not m:
        print("[FAIL] no TOKENS line")
        return 1
    gen_ids = [int(x) for x in m.group(1).split()]
    text = tk.decode(gen_ids, skip_special_tokens=False)
    print("\n================ 生成 ================")
    print(text)
    print("=====================================")
    return 0


if __name__ == "__main__":
    sys.exit(main())
