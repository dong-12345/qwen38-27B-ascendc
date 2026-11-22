# 深度补全 / RGB-D 实时链路 — 项目现状

> 最后更新：2026-11-22 22:20

---

## 1. 硬件连接

```
┌──────────────┐   USB 3.0 (5 Gbps)   ┌──────────────────────────┐
│ 海康 MV-DT01SDU │ ───────────────────► │ OrangePi AI Pro 20T      │
│ ToF 640×480      │  RGBD 6.51 MB/帧     │ Ascend 310B1 / 3 核 ARM  │
│ RGB 1536×1280    │                      │ 192.168.137.163          │
└──────────────┘                      └───────────┬──────────────┘
                                                    │ USB 网络共享（RNDIS）
                                                    ▼
                                        ┌──────────────────────┐
                                        │ Windows PC           │
                                        │ 192.168.137.1        │
                                        │ Foxglove / HiViewer  │
                                        └──────────────────────┘
```

| 项 | 值 |
|---|---|
| 相机 | 海康 MV-DT01SDU（USB `2bdf:0001`，序列号 DA8687616）|
| 板子 | OrangePi AI Pro 20T，3 核 ARM，23 GB 内存 |
| 板子地址 | `192.168.137.163`（PC 侧 192.168.137.1）|
| 登录 | `HwHiAiUser` / `Mind@123`，密钥 `tools/opi_key` |
| 串口调试 | PC 的 `COM7`（CH343），115200 8N1 |
| Foxglove | `ws://192.168.137.163:8765`，无需鉴权 |

⚠️ 相机经 USB 供电，**长时间出流会过热掉线**（十几分钟一次），需拔插恢复；建议加散热或换有源 Hub。

---

## 2. 代码位置

| 位置 | 路径 |
|---|---|
| PC | `D:/项目4/orangepi_aipro/code/depth_completion/` |
| 板端 | `~/code/depth_completion/`（两边结构一致，用 scp 同步）|

```
deploy/
  pd_live.py        主程序：采集 → 写共享内存 → 启动两个子进程
  pd_complete.py    子进程：深度补全（中值去点 + 导向滤波填洞）
  pd_publish.py     子进程：调色 / JPEG / 点云 / Foxglove 发布
  pd_capture.py     U3V 直连（libusb）+ payload 解码（三种格式自适应）
  pd_core.py        算法库：ClassicCompleter / colorize / grade / AutoExposure
  foxglove_ws.py    Foxglove WebSocket 服务端（零依赖，手写 protobuf）
  camera_setup.py   用厂商 SDK 恢复相机 RGBD 配置（每次启动前跑）
  enable_rgbd.py    只设 OutputRgbd=1（调试用）
  pd_*.py           相机诊断小工具（抓帧/时序/寄存器探测）
convert/            标定与 SDK 抓帧工具
results/            rgbd_calib.json（内外参，别删）
run_live.sh         常驻启动脚本（板端）
HANDOVER.md         技术交接文档（协议坑、算法细节）
PROJECT_STATUS.md   本文
```

---

## 3. 运行

```bash
# 板端启动（已固化，重启不丢）
cd ~/code/depth_completion
./run_live.sh          # 内含 camera_setup.py + pd_live.py

# 停止
pkill -f pd_live

# 看日志
tail -f /tmp/pd_live.log
```

`run_live.sh` 内容：
```bash
export LD_LIBRARY_PATH=$HOME/Mv3dRgbdSDK/Libs/aarch64:$HOME/Mv3dRgbdSDK/MVCC/Libs/aarch64:$HOME/Mv3dRgbdSDK/MVCC/Libs/arm-none:$LD_LIBRARY_PATH
python3 deploy/camera_setup.py     # 设置 ImageMode=0x2 / ImageAlign=0 / OutputRgbd=0 / JPEG 关
exec python3 deploy/pd_live.py --port 8765 --depth-every 0.2 --cloud-every 0.3 --seconds 0
```

---

## 4. 数据流

```
相机 (6.51 MB/帧, 30 fps) 
   ↓ USB3 同步 bulk 读（约 33 ms）
pd_capture.decode_payload   拆出 深度 640×480 uint16 + 彩色 1536×1280×3
   ↓ 共享内存（双缓冲）
pd_complete  中值去飞点 → 1/4 分辨率归一化卷积先验 → 半分辨率导向滤波 → 补洞
   ↓ 共享内存
pd_publish   伪彩 + JPEG + 点云 → Foxglove WebSocket :8765
   ↓
Foxglove /camera/rgb  /depth/colored  /points
```

---

## 5. 当前性能（实测）

| 指标 | 数值 |
|---|---|
| USB 图像负载 | 6,512,688 B/帧（约 195.4 MB/s）|
| 帧率 | **USB 30.00 fps**（361 帧，0 次读失败）；完整实时流程 **29.58 fps** |
| 补全耗时 | 15 ms（独立进程 26 ms，竞争下）|
| 端到端延迟 | 帧收完 → 补全完成 66 ms；→ Foxglove 79 ms |
| 点云 | 19,200 点 / 0.3 s |
| NPU | 不使用（模型已全部删除）|

---

## 6. 已知问题

1. **相机过热掉线**：连续出流 10+ 分钟会从 USB 总线消失，需拔插；`lsusb` 里消失即确认。
2. 旧彩图模式的 USB payload 为 8,478,776 B。原先视为灰度图的区域包含可解码的 1536×1280 彩色 JPEG 和大量填充字节。
3. 默认配置改为深度模式（0x2），保留 640×480 uint16 深度和未压缩的 1536×1280 RGB，关闭相机对齐。
4. 6,512,688 B 格式先传深度再传彩图。采集解码支持两种格式，现有 Foxglove 预览将彩图缩为 640×480，未执行深度到 RGB 的几何配准。
5. **USB 帧可能被拆成多块传输**，必须按到达顺序用多缓冲区重组，否则两帧拼一起 → 花屏。

---

## 7. 快速自检

```bash
# 相机在不在
lsusb | grep 2bdf

# 服务在不在
pgrep -fa pd_live; ss -ltn | grep 8765

# 帧率 / 延迟
tail -5 /tmp/pd_live.log

# 从 PC 验证频道（PC 上跑）
python tmp/fgjson.py 192.168.137.163 8765
```

---

## 8. 后续可做

- 异步多 URB 读（libusb async），消除帧间空隙
- 相机加散热/有源 Hub，解决过热掉线
- 补全算法质量：低信噪区目前用平滑面替换，可换多尺度置信度加权