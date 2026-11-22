# tof_mapping — 实时建图 / SLAM 项目

基于 ToF RGB-D 相机（MV-DT01SDU，depth 640×480@30fps + RGB 1536×1280）在
OrangePi AI Pro 20T（Ascend 310B1，A55×4）上做实时建图。

## 目录

```
tof_mapping/
├── bench/                     算子/前端基准
│   ├── slam_bench.py          ORB/GFTT/BF match/PnP 耗时（OpenCV）
│   ├── octomap_bench.cpp      OctoMap 插入耗时（raycast vs occupied-only）
│   └── mapping_frontend_bench.cpp  2D 栅格 / 3D OctoMap 前端对比
├── orbslam3/                  ORB-SLAM3 RGB-D 板端编译与运行
│   ├── build_orbslam3.sh      一键编译脚本（Pangolin + DBoW2 + g2o + ORB_SLAM3）
│   ├── rgbd_headless.cc       无窗口 RGB-D runner（板端跑分/验证）
│   └── results/               编译日志、运行结果
└── results/
    └── slam_mapping_bench.md  板端实测：算子/前端耗时与推荐方案
```

## 板端实测结论（详见 results/slam_mapping_bench.md）

| 方案 | 实测/估算 | 备注 |
|---|---|---|
| 2D/2.5D 占据栅格 | **222 fps** 前端 | 导航地图首选 |
| OctoMap occupied-only (4.8k 点/帧) | **145 fps** | 需自管自由空间清空 |
| OctoMap 标准 raycast | 1.5~4 fps | **不可用** |
| ORB-SLAM3 风格 tracking（640×480） | ~17 fps（算子估算） | 需降分辨率/特征 |
| ORB-SLAM3 风格 tracking（320×240） | ~27 fps（算子估算） | 20fps+ 可行 |
| RTAB-Map 风格（GFTT, 320×240） | ~32 fps（算子估算） | |

## ORB-SLAM3 RGB-D（板端编译）

源码放在 `~/orbslam3_ws/ORB_SLAM3`（第三方源码不放进本仓库；本目录只放我们的脚本/结果）。
Pangolin 0.6 编译安装到 `/usr/local`，ORB-SLAM3 编出 `lib/libORB_SLAM3.so` 与 `Examples/RGB-D/rgbd_tum`。

一键编译（板子）：
```bash
bash ~/code/tof_mapping/orbslam3/build_orbslam3.sh
```

无窗口跑 TUM RGB-D 序列（板端测真实 fps）：
```bash
# 编译 runner（在 ORB_SLAM3 源码目录）
g++ -O3 -std=c++11 ~/code/tof_mapping/orbslam3/rgbd_headless.cc \
    -I include -I include/CameraModels -I Thirdparty/Sophus -I /usr/include/eigen3 \
    -L lib -lORB_SLAM3 -lopencv_core -lopencv_imgproc -lopencv_imgcodecs -o rgbd_headless
# 运行
./rgbd_headless Vocabulary/ORBvoc.txt Examples/RGB-D/TUM1.yaml <TUM assoc.txt> 300
```

编译/运行结果见 `orbslam3/results/`。
