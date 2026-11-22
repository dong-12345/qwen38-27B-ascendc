#!/usr/bin/env python3
"""Census of payload sizes over N frames."""
import os, sys, time
from collections import Counter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pd_capture import U3V, decode_payload
u = U3V(); u.open(); u.start(); time.sleep(0.5)
c = Counter(); t0 = time.time(); n = 0
for _ in range(40):
    p, st = u.read_payload(1500)
    if p is None:
        continue
    n += 1
    d, rgb = decode_payload(p)
    c[(len(p), None if d is None else int(__import__("numpy").median(d[d > 0])))] += 1
    if n >= 25 and time.time() - t0 > 4:
        break
u.stop(); u.close()
el = time.time() - t0
print("frames=%d  %.1f fps" % (n, n / el))
for k, v in c.most_common():
    print("  size=%d depth_p50=%s  x%d" % (k[0], k[1], v))
