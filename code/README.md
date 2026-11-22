# code/ — 项目目录

所有工程/项目按子目录存放，一个项目一个文件夹。临时脚本、旧版实验、厂商安装包不放在这里。

| 项目目录 | 说明 |
|---|---|
| [`tof_u3v_sdk/`](tof_u3v_sdk/README.md) | **自研 ToF 相机 U3V 直连 SDK**（libusb 实现，绕过海康 Linux SDK）：30fps 取流、软触发、延迟测量、协议抓包工具 |
| [`tof_mapping/`](tof_mapping/README.md) | **实时建图项目**：SLAM/建图算子基准、ORB-SLAM3 RGB-D 在板端的编译与运行 |

## 背景（一句话）

海康 Mv3dRgbd Linux SDK 的取流路径 CPU 绑定（7~14fps、170~500ms 延迟、单线程 98.5% CPU），
我们用 usbmon 抓包还原 USB3 Vision 协议后，用 libusb 自写 U3V 客户端直接取流：
**30fps / 74MB/s / 整机 CPU 3.6% / 软触发端到端延迟 p50 84.9ms**。

> 厂商 SDK 已不在此目录使用（本目录不再包含 vendored 厂商 SDK）。
> 系统层如仍装有 MVS 运行时，与这些项目无关，可自行卸载。
