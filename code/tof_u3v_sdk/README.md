# tof_u3v_sdk — 自研 ToF 相机 U3V 直连 SDK

用 **libusb + ctypes** 直接实现 USB3 Vision（U3V）协议，从海康 MV-DT01SDU ToF 相机取流，
不依赖海康 Mv3dRgbd / MVS 运行时。协议序列来自对厂商 SDK 的 usbmon 抓包还原。

## 实测结果（OrangePi AI Pro 20T，Ascend 310B1）

| 指标 | 海康 Linux SDK | **本 SDK（U3V 直连）** |
|---|---|---|
| 帧率 | 7~14 fps | **29.7~30.1 fps** |
| 吞吐 | 10~30 MB/s | **74~75 MB/s** |
| 整机 CPU | 单线程 98.5% | **3.6~4.5%** |
| 软触发端到端延迟 | 170~500 ms | **p50 84.9 ms / p95 88.4 ms** |
| 单帧 USB 传输 | ~90ms（CPU 拷贝） | 9.3 ms（278 MB/s 线上） |

帧内容：depth 640×480 C16 + 彩色 JPEG 1536×1280，每帧 2,580,528 B。

## 使用方法

```bash
# 30fps 基准 + CPU 统计
python3 src/u3v_lowlat.py --seconds 8

# 抓一帧存图（depth16.png + color.jpg）
python3 src/u3v_lowlat.py --seconds 5 --snap
```

依赖：`libusb-1.0.so.0`（系统自带）、Python 3。相机 VID:PID = `2bdf:0001`。

首次使用先装环境（udev 权限 + usbfs 内存上限，不需要海康任何 SDK）：

`ash
sudo bash install.sh
`

（厂商 SDK 已从本机删除：`/opt/MVS`、`/opt/Mv3dRgbdSDK`、厂商 udev 规则和 `profile.d` 均已清理，
已替换为本项目自己的 `udev/99-tof-u3v.rules`。）

## 协议要点（与厂商 SDK 一致，抓包还原）

1. 标准 U3V ReadMemory/WriteMemory（bulk OUT `0x01` / IN `0x82`），包头 `U3VC` + LE 字段；
2. SBRM @ `0x52000` → SIRM @ `0x62000`；
3. 开流：写 SIRM `payload_size=0x100000, payload_count=2, transfer1_size=0x76030,
   transfer2_size=0, max_leader_size=0x400, max_trailer_size=0x400` → `SIRM.control=1`
   → **`0x030804=1`（AcquisitionStart，缺了不出数据）**；
4. 数据：每帧 = leader(52B, `U3VL`, offset 8 = 帧号) + payload(2,580,528B) + trailer(36B, `U3VT`)；
5. 软触发：`0x030880=6`；触发模式配置：`0x0308d8=7`(TriggerSource=Software)、`0x030840=0x40`、
   `0x030b04`=曝光(µs)、`0x03081c=0x40007530`、`0x0500a8=1`；
6. 停止：`0x030804=0`、`SIRM.control=0`。

## 目录

```
tof_u3v_sdk/
├── src/
│   └── u3v_lowlat.py              ★ 取流主程序（--seconds / --snap）
├── tools/
│   ├── u3v_bench.py               吞吐基准（多轮）
│   ├── u3v_fps.py                 用 leader 帧号统计设备帧率
│   ├── u3v_dump.py                打印前 N 次读的长度/首字节（摸流结构）
│   ├── u3v_probe.py / 2 / 3       控制通道 / ABRM / SBRM / SIRM 探测
│   ├── u3v_stream.py              带 leader/trailer 解析的取流
│   ├── u3v_diag.py                单次读长度/间隔诊断
│   ├── u3v_latency_opt.py         延迟优化实验（no-ACK trigger、曝光矩阵）
│   ├── u3v_incremental_latency.py 分级延迟（leader / 深度 / 完整 RGBD）
│   ├── u3v_trigger_latency.py     软触发端到端绝对延迟测量
│   └── usbmon_capture.py / usbmon_ctrl.py / usbmon_probe.py / usbmon_writes.py   协议抓包工具
├── docs/
│   ├── u3v_direct_breakthrough.md 突破过程 + 协议 + 实测数据
│   ├── u3v_latency_optimization.md 延迟拆解与优化结论
│   ├── opensource_u3v_eval.md      Aravis / Cameleon 预研
│   ├── history_vendor_sdk_analysis.md  厂商 SDK 瓶颈的完整取证分析
│   └── camera_genie_xml.xml       相机 GenICam XML（寄存器/节点参考）
└── results/
    ├── u3v_trigger_latency.txt    软触发延迟原始数据（n=20）
    ├── u3v_snap/                  抓帧样例（depth16.png / color.jpg）
    └── viz/                       可视化（U3V 深度伪彩 / RGB）
```

## 延迟分解（软触发，主机时钟）

| 阶段 | 时间 |
|---|---|
| 触发命令写+ACK | 0.06~0.08 ms |
| 相机固件流水线（触发→首数据 leader） | **79.2 ms（硬下限）** |
| 深度部分可用 | 81.7 ms |
| 完整 RGBD | 84.9 ms |

- 曝光 500→200µs、触发不等 ACK、`OutputRgbd=0/DownSample=1` 等均无法再降；
- 进一步优化：深度优先消费（≈82ms）、硬触发（确定性/同步）；
- 想 <80ms 只能换更高帧率/更短流水线相机。
