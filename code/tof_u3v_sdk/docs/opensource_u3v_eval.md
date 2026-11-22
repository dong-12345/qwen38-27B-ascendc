# 开源 U3V/相机软件栈评测（板子实测，2026-11-20 12:20-12:30）

目标：绕过海康 Mv3dRgbd Linux SDK（已证明取流路径 CPU 绑定：FetchFrame 主线程 98.5% user CPU），
用开源 USB3 Vision 栈直接取流，看能否恢复 30fps。

## 找到的开源项目

| 项目 | 语言/许可 | 性质 | 对本问题的价值 |
|---|---|---|---|
| [Aravis](https://github.com/AravisProject/aravis) | C / LGPL-2.1 | 成熟 GenICam/USB3 Vision + GigE Vision 库（libusb 直连），Ubuntu 22.04 自带 0.8.20 | **可绕过厂商 SDK**，已在板子安装测试 |
| [cameleon](https://github.com/cameleon-rs/cameleon) / [Pycameleon](https://github.com/Menchen/pycameleon) | Rust + Python / MIT | 纯 Rust U3V 实现，PyPI 有 **manylinux aarch64 wheel**（cp39-abi3，py3.10 可用） | **可绕过厂商 SDK**，已在板子安装测试 |
| [pyhikrobot](https://pypi.org/project/pyhikrobot/) | Python | MVS SDK 的 ctypes 封装（零拷贝 numpy 视图） | 只是封装厂商运行时 → **不快**（同一传输层） |
| [hikrobotcamlib](https://pypi.org/project/hikrobotcamlib/) | Python | 同上，ctypes 封装 MVS | 同上，无性能收益 |
| gitee Hikrobot/Mv3dRgbdSDK_ROS2 | C++/ROS2 | 官方 ROS2 驱动 | 还是用厂商 SDK |
| go-hikrobot-sdk / go-aravis | Go | 绑定 | 平台相关，无收益 |

## 板子实测结果

1. **Aravis 0.8.20**（Ubuntu jammy universe，pt install aravis-tools gir1.2-aravis-0.8）：
   - 能枚举到相机：Hikrobot-2bdfDA8687616-DA8687616 ✔
   - Camera.new() 失败：Failed to bootstrap USB device ... (3)，
     ARV_DEBUG 显示 [UvDevice::read_memory] Ack reception error: LIBUSB_ERROR_IO（反复重试 30 次）
   - 端点识别正常（control OUT=0x01 / IN=0x82，data=0x84 interface 2）
2. **Pycameleon 0.2.11**（pip install -i https://pypi.tuna.tsinghua.edu.cn/simple pycameleon，aarch64 wheel 直接可用）：
   - 枚举成功：CameraInfo { vendor_name: "Hikrobot", model_name: "MV-DT01SDU", serial_number: "DA8687616" } ✔
   - cam.open() 失败：RuntimeError: control error: input/output error: libusb error: input/output error
3. 交叉验证：
   - 用**厂商自带的 libusb**（LD_PRELOAD=/opt/MVS/lib/aarch64/libusb-1.0.so.0）重试 Pycameleon → 同样 EIO；
   - 在厂商 SDK 正常取流一次后立刻重试 → 同样 EIO；
   - USBDEVFS_RESET 后重试 → 同样 EIO。
   → **两个独立开源 U3V 栈都在 U3V control 通道 bootstrap/打开阶段拿到 LIBUSB_ERROR_IO**，
     说明这台 Hikrobot ToF 相机的 U3V 控制通道需要厂商私有初始化（或对标准 U3V 有偏离），
     不是 libusb 版本/权限/端点识别问题。
4. 厂商 SDK（MVS 4.5.0.3 自带 libusb）可以正常控制/取流 → 说明 USB 硬件与链路正常，差异在协议/初始化层。

## 结论

- 有开源替代（Aravis / Cameleon），**但开箱不能打开这台相机**（U3V bootstrap EIO）；
  要用开源栈必须逆向厂商初始化序列（可用 lowlat/usbmon_capture.py 抓厂商 SDK open 阶段的控制包），成本高、不保证成功。
- PyPI 上的 pyhikrobot/hikrobotcamlib 只是厂商 SDK 的包装，**不解决性能问题**。
- 现实建议：向海康提 Linux SDK 性能工单（附第 9 节证据），或尝试更新版 Linux SDK；
  短期生产用 RGBJpegEnable=1 + ImageAlign=0（170ms/10-14fps）。
