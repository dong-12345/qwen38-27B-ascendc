#!/usr/bin/env python3
"""Pack a converted weight directory into N equally sized transfer parts.

Usage: python pack_transfer.py <src_dir> <out_prefix> [n_parts]
Produces <out_prefix>.tar and <out_prefix>.partNN files, removing the tar afterwards.
"""
import os
import shutil
import subprocess
import sys


def main():
    src, prefix = sys.argv[1], sys.argv[2]
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    tar_path = prefix + ".tar"
    if os.path.exists(tar_path):
        os.remove(tar_path)
    r = subprocess.run(["tar.exE", "-cf", tar_path, "-C", src, "."])
    if r.returncode != 0:
        raise SystemExit("tar failed")
    total = os.path.getsize(tar_path)
    chunk = total // n + 1
    with open(tar_path, "rb") as f:
        for i in range(n):
            part = "%s.part%02d" % (prefix, i)
            with open(part, "wb") as out:
                remain = chunk
                while remain > 0:
                    b = f.read(min(1 << 24, remain))
                    if not b:
                        break
                    out.write(b)
                    remain -= len(b)
            if os.path.getsize(part) == 0:
                os.remove(part)
            else:
                print(part, os.path.getsize(part))
    os.remove(tar_path)
    print("total %d bytes in %d parts" % (total, n))


if __name__ == "__main__":
    main()
