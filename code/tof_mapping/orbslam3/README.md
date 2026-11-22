# ORB-SLAM3 RGB-D 板端编译（OrangePi AI Pro 20T）

**状态：编译成功 ✅**（2026-11-20 13:44，板子 aarch64）

## 产物

| 产物 | 路径（板子） | 大小 |
|---|---|---|
| ORB-SLAM3 库 | `~/orbslam3_ws/ORB_SLAM3/lib/libORB_SLAM3.so` | 4.57 MB |
| RGB-D 官方示例 | `~/orbslam3_ws/ORB_SLAM3/Examples/RGB-D/rgbd_tum` | 42.9 KB |
| 无窗口 runner（本项目） | `~/orbslam3_ws/ORB_SLAM3/rgbd_headless` | 38.5 KB |
| ORB 词袋 | `~/orbslam3_ws/ORB_SLAM3/Vocabulary/ORBvoc.txt` | 145 MB（tar 解出） |

```bash
$ ./Examples/RGB-D/rgbd_tum
Usage: ./rgbd_tum path_to_vocabulary path_to_settings path_to_sequence path_to_association
```

## 环境（板端实测版本）

| 组件 | 版本 | 来源 |
|---|---|---|
| GCC / CMake | 11.4 / 3.22.1 | 系统 |
| OpenCV | **4.5.4** | apt `libopencv-dev` |
| Eigen | **3.4.0** | apt `libeigen3-dev` |
| Boost serialization | **1.74** | apt `libboost-serialization-dev`（编译时缺它报 `-lboost_serialization` 找不到） |
| Pangolin | **v0.6**（源码编译，装到 `/usr/local`） | GitHub 源码 |
| ORB-SLAM3 | master（UZ-SLAMLab） | GitHub |

## 复现步骤

```bash
# 0) 依赖
sudo apt install -y libopencv-dev libeigen3-dev libglew-dev libgl1-mesa-dev \
     libegl1-mesa-dev libxkbcommon-dev libwayland-dev libboost-serialization-dev unzip

# 1) 下载源码到 ~/orbslam3_ws/
#    - ORB_SLAM3 (master)：只需 src/ include/ Thirdparty/ Vocabulary/ Examples*(.cc/.yaml)
#    - Pangolin v0.6

# 2) 一键编译（Pangolin → DBoW2/g2o → ORB_SLAM3 库+RGB-D示例 → 无窗口runner）
bash ~/code/tof_mapping/orbslam3/build_orbslam3.sh
```

## 编译遇到的问题

1. **缺示例源码**：直接解压完整仓库太大（1.4GB，含 EuRoC/TUM-VI 数据集）；只解压需要部分后
   cmake 报 `No SOURCES given to target: *_old` → 补齐 `Examples*/**.cc`（53 个）即通过。
2. **链接缺 Boost**：`/usr/bin/ld: cannot find -lboost_serialization` → `apt install libboost-serialization-dev`。
3. **无窗口 runner 链接**：需要 `-lpangolin -lOpenGL -lGL -lEGL -lGLEW`（Pangolin 的 GL 依赖没有被
   自动带上，即使运行时 `bUseViewer=false`）。

## 无窗口跑 RGB-D（板端测真实 fps）

```bash
cd ~/orbslam3_ws/ORB_SLAM3
LD_LIBRARY_PATH=lib ./rgbd_headless Vocabulary/ORBvoc.txt Examples/RGB-D/TUM1.yaml <TUM assoc.txt> 300
```

## 下一步（未完成）

- **用真实 RGB-D 序列跑通并测帧率**：需要 TUM 数据集或对齐后的相机数据；
  - TUM 数据集下载（公网 ~170MB）非常慢，本次未完成；
  - 相机数据需先解决 **depth 640×480 与 RGB 1536×1280 的对齐**（相机 XML 里有标定参数，可做 host 端重投影对齐），
    然后按 TUM 格式组织：`rgb/`、`depth/`（16UC1，scale 5000）、`assoc.txt`。
- 板端预计 tracking（A55 单线程，320×240 特征）≈20~27fps，需实测确认。
