#!/usr/bin/env bash
# build_orbslam3.sh — 在板子上编译 ORB-SLAM3（RGB-D，含 Pangolin）
#
# 实测环境（OrangePi AI Pro 20T，Ubuntu 22.04 aarch64）：
#   GCC 11.4 / CMake 3.22.1 / OpenCV 4.5.4 / Eigen 3.4.0 / Boost 1.74 / Pangolin 0.6
#
# 前置（apt 依赖，一次性）：
#   sudo apt install -y libopencv-dev libeigen3-dev libglew-dev libgl1-mesa-dev \
#        libegl1-mesa-dev libxkbcommon-dev libwayland-dev libboost-serialization-dev unzip
#
# 源码位置（第三方源码不放仓库，按需下载）：
#   ~/orbslam3_ws/ORB_SLAM3          ORB-SLAM3 源码（UZ-SLAMLab/ORB_SLAM3 master）
#   ~/orbslam3_ws/Pangolin-0.6       Pangolin v0.6 源码
#
# 用法： bash build_orbslam3.sh
set -e
WS=${WS:-$HOME/orbslam3_ws}
ORB=$WS/ORB_SLAM3
PAN=$WS/Pangolin-0.6
HERE=$(cd "$(dirname "$0")" && pwd)

echo "== [1/4] Pangolin =="
if [ ! -f /usr/local/lib/libpangolin.so ]; then
  cd "$PAN"
  mkdir -p build && cd build
  cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_PANGOLIN_PYTHON=OFF \
           -DBUILD_EXAMPLES=OFF -DBUILD_TESTS=OFF -DBUILD_TOOLS=OFF
  make -j"$(nproc)"
  sudo make install && sudo ldconfig
else
  echo "Pangolin already installed: /usr/local/lib/libpangolin.so"
fi

echo "== [2/4] Thirdparty (DBoW2, g2o) =="
cd "$ORB/Thirdparty/DBoW2" && mkdir -p build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release && make -j"$(nproc)"
cd "$ORB/Thirdparty/g2o"  && mkdir -p build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release && make -j"$(nproc)"

echo "== [3/4] ORB-SLAM3 库 + RGB-D 示例（只编这两个目标，全编示例太慢）=="
cd "$ORB" && mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j"$(nproc)" ORB_SLAM3 rgbd_tum

echo "== [4/4] 我们的无窗口 RGB-D runner =="
cd "$ORB"
g++ -O3 -std=c++11 "$HERE/rgbd_headless.cc" \
  -I . -I include -I include/CameraModels -I Thirdparty/Sophus -I Thirdparty/DBoW2 \
  -I /usr/include/eigen3 $(pkg-config --cflags opencv4) \
  -L lib -L /usr/local/lib -lORB_SLAM3 -lpangolin -lOpenGL -lGL -lEGL -lGLEW \
  $(pkg-config --libs opencv4) -Wl,-rpath,/usr/local/lib -o rgbd_headless

[ -f Vocabulary/ORBvoc.txt ] || tar xzf Vocabulary/ORBvoc.txt.tar.gz -C Vocabulary

echo "== build OK =="
ls -l "$ORB/lib/libORB_SLAM3.so" "$ORB/Examples/RGB-D/rgbd_tum" "$ORB/rgbd_headless"
