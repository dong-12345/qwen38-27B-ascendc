# -*- coding: utf-8 -*-
"""usbmon_capture.py — 用 usbmon 二进制接口抓某一总线的 URB 事件并统计
用途：看相机取流时主机提交的 bulk URB 大小、完成间隔、同时在途数量，
      判断"USB3 链路正常但吞吐只有 ~25-55MB/s"是主机侧 URB/往返问题还是别的。

用法（root）：
    sudo python3 usbmon_capture.py <秒数> <busnum> <devnum>

例：相机在 Bus 004 Device 006 ->  sudo python3 usbmon_capture.py 3 4 6
注意：需要同时有另一个进程在取流，本脚本只旁路抓包。
"""
import os
import select
import struct
import sys
import time

HDR = struct.Struct("<QBBBBHccqiiii8s")   # mon_bin_hdr = 48 bytes
XFER = {0: "iso", 1: "intr", 2: "ctrl", 3: "bulk"}


def main():
    secs = float(sys.argv[1]) if len(sys.argv) > 1 else 3.0
    bus = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    dev = int(sys.argv[3]) if len(sys.argv) > 3 else -1

    path = "/dev/usbmon%d" % bus
    fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
    t_end = time.time() + secs
    buf = b""
    recs = []
    while time.time() < t_end:
        r, _, _ = select.select([fd], [], [], 0.2)
        if not r:
            continue
        try:
            chunk = os.read(fd, 1 << 20)
        except BlockingIOError:
            continue
        buf += chunk
        while len(buf) >= HDR.size:
            v = HDR.unpack_from(buf, 0)
            (id_, typ, xfer, ep, devnum, busnum, fs, fdat, ts_sec, ts_usec,
             status, len_urb, len_cap, _setup) = v
            typ = chr(typ)
            total = HDR.size + (len_cap if typ in "SC" else 0)
            if len(buf) < total:
                break
            buf = buf[total:]
            if dev >= 0 and devnum != dev:
                continue
            recs.append((id_, typ, xfer, ep, ts_sec * 1e6 + ts_usec, status,
                         len_urb, len_cap))
    os.close(fd)

    print("capture %.1fs bus=%d dev=%s  events=%d" % (secs, bus, dev, len(recs)))
    if not recs:
        print("没有事件：确认设备号正确、总线上确实有流量、并且 usbmon 已编译进内核")
        return

    # 按 (xfer, ep) 分组
    from collections import defaultdict
    groups = defaultdict(list)
    for r in recs:
        groups[(r[2], r[3])].append(r)

    print("\n%-6s %-4s %8s %10s %10s %10s" % ("xfer", "ep", "events", "S", "C", "bytes(C)"))
    for (xfer, ep), rs in sorted(groups.items()):
        ns = sum(1 for r in rs if r[1] in "Ss")
        nc = sum(1 for r in rs if r[1] in "CcEe")
        by = sum(r[6] for r in rs if r[1] in "CcEe")
        print("%-6s 0x%02x %8d %10d %10d %10d" % (XFER.get(xfer, xfer), ep, len(rs), ns, nc, by))

    # 详细：bulk 端点的时序（IN 看成帧数据，OUT 看 ACK/命令）
    for (xfer, ep), rs in sorted(groups.items()):
        if xfer != 3:
            continue
        rs = [r for r in rs if r[1] in "SsCcEe"]
        rs.sort(key=lambda r: r[4])
        sub = {}
        lat = []          # URB 提交->完成 延迟 us
        comps = []
        pend = set()
        max_inflight = 0
        sizes = {}
        for r in rs:
            if r[1] in "Ss":
                sub[r[0]] = r[4]
                pend.add(r[0])
                max_inflight = max(max_inflight, len(pend))
                sizes[r[6]] = sizes.get(r[6], 0) + 1
            else:
                pend.discard(r[0])
                comps.append((r[4], r[7], r[5]))
                if r[0] in sub:
                    lat.append((r[4] - sub[r[0]]) / 1000.0)
        dts = [(b[0] - a[0]) / 1000.0 for a, b in zip(comps, comps[1:])]
        dts.sort()
        lat.sort()
        st_hist = {}
        for _t, _b, st in comps:
            st_hist[st] = st_hist.get(st, 0) + 1
        actual = sum(b for _t, b, st in comps if st == 0)
        dur = (comps[-1][0] - comps[0][0]) / 1e6 if len(comps) > 1 else 1e-9
        print("\nbulk %s ep=0x%02x: S=%d C=%d 在途max=%d 请求大小=%s" %
              ("IN" if ep & 0x80 else "OUT", ep,
               sum(1 for r in rs if r[1] in "Ss"), len(comps), max_inflight,
               dict(sorted(sizes.items())[:6])))
        print("  状态分布: 0(成功)=%d 其他=%s" %
              (st_hist.get(0, 0), {k: v for k, v in st_hist.items() if k != 0}))
        if lat:
            p = lambda xs, q: xs[min(len(xs) - 1, int(q * (len(xs) - 1)))]
            print("  URB 提交->完成延迟 ms: p50 %.3f  p95 %.3f  max %.3f  (n=%d)"
                  % (p(lat, .5), p(lat, .95), lat[-1], len(lat)))
        if dts:
            p = lambda q: dts[min(len(dts) - 1, int(q * (len(dts) - 1)))]
            print("  完成间隔 ms: p50 %.3f  p95 %.3f  max %.3f" % (p(.5), p(.95), dts[-1]))
        print("  实际成功数据量 %d bytes / %.3fs = %.1f MB/s"
              % (actual, dur, actual / 1048576.0 / dur if dur > 0 else 0))
        # 每 200ms 桶
        if comps:
            t0 = comps[0][0]
            buckets = {}
            for t, b, st in comps:
                k = int((t - t0) // 200000)
                bb, cc = buckets.get(k, (0, 0))
                buckets[k] = (bb + (b if st == 0 else 0), cc + 1)
            print("  时间线(每200ms: 成功KB/完成数):",
                  "  ".join("%dms:%.1fKB/%d" % (k * 200, v[0] / 1024.0, v[1])
                            for k, v in sorted(buckets.items())[:12]))


if __name__ == "__main__":
    main()
