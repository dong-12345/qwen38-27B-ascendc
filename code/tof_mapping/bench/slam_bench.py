# -*- coding: utf-8 -*-
"""slam_bench.py — 板子上测 RGB-D SLAM/建图关键算子的耗时，估算可行帧率

用我们 U3V 抓到的真实彩色图（1536x1280）缩放成 640x480 / 320x240。
测：
  1. ORB 特征提取（ORB-SLAM3 / ORB-SLAM2 的核心算子）
  2. GFTT + ORB descriptor（RTAB-Map 默认 GFTT/BRIEF 路线的近似）
  3. 暴力 Hamming 匹配（1000x1000）
  4. solvePnPRansac（位姿求解）
  5. 点云降采样（pyrDown 近似）+ 估计 OctoMap/TSDF 前端开销

输出每帧各阶段 ms，并给出单线程/多线程下的估算 fps。
"""
import os
import time

import cv2
import numpy as np

cv2.setNumThreads(1)      # ORB-SLAM 的 tracking 是单线程的，按单线程测
IMG = os.path.expanduser("~/code/results/u3v_snap/color.jpg")
if not os.path.exists(IMG):
    IMG = os.path.expanduser("~/code/results/snap_20261120_113529/frame00_img1_color.jpg")
img = cv2.imread(IMG, cv2.IMREAD_GRAYSCALE)
print("image:", img.shape)


def bench(name, fn, n=20):
    # warmup
    for _ in range(2):
        fn()
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        r = fn()
        ts.append((time.perf_counter() - t) * 1000)
    ts.sort()
    print("  %-34s p50 %7.2f ms  p95 %7.2f ms" % (name, ts[len(ts) // 2], ts[int(len(ts) * 0.95) - 1]))
    return ts[len(ts) // 2], r


for size in ((640, 480), (320, 240)):
    frame = cv2.resize(img, size)
    print("\n=== %dx%d ===" % size)
    orb = cv2.ORB_create(nfeatures=1000)
    orb500 = cv2.ORB_create(nfeatures=500)
    t_orb, (kp, des) = bench("ORB detect+compute n=1000", lambda: orb.detectAndCompute(frame, None))
    t_orb500, _ = bench("ORB detect+compute n=500", lambda: orb500.detectAndCompute(frame, None))
    gftt = lambda: cv2.goodFeaturesToTrack(frame, maxCorners=500, qualityLevel=0.01, minDistance=7)
    t_gftt, corners = bench("GFTT detect (maxCorners=500)", gftt)
    # descriptor on GFTT corners via ORB.compute
    orb_c = cv2.ORB_create()
    kps = cv2.KeyPoint_convert(corners.reshape(-1, 2)) if corners is not None else []
    t_desc, _ = bench("ORB descriptor on GFTT corners",
                      lambda: orb_c.compute(frame, kps))
    # 匹配：两张相邻帧（平移 5px 模拟运动）
    frame2 = np.roll(frame, 5, axis=1)
    kp2, des2 = orb500.detectAndCompute(frame2, None)
    kp1, des1 = orb500.detectAndCompute(frame, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    t_match, _ = bench("BF hamming match (500x500)", lambda: bf.match(des1, des2))
    # PnP
    pts3d = np.random.rand(300, 3).astype(np.float32) * 3 + 0.3
    pts2d = np.random.rand(300, 2).astype(np.float32) * np.array(size[::-1])
    K = np.array([[size[0], 0, size[0] / 2], [0, size[0], size[1] / 2], [0, 0, 1]], np.float32)
    t_pnp, _ = bench("solvePnPRansac (300点)",
                     lambda: cv2.solvePnPRansac(pts3d, pts2d, K, None))
    # 降采样（frontend 前置）
    t_pyr, _ = bench("pyrDown (half)", lambda: cv2.pyrDown(frame))

    track_orb = t_orb + t_match + t_pnp
    track_gftt = t_gftt + t_desc + t_match + t_pnp
    print("  -> ORB 路线 tracking 估算: %.1f ms/frame = %.1f fps" % (track_orb, 1000.0 / track_orb))
    print("  -> GFTT 路线 tracking 估算: %.1f ms/frame = %.1f fps" % (track_gftt, 1000.0 / track_gftt))
    print("  -> 纯建图前端(降采样+点云准备)估算: %.1f ms/frame = %.1f fps"
          % (t_pyr, 1000.0 / t_pyr))
