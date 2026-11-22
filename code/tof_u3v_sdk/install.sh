#!/usr/bin/env bash
# install.sh — tof_u3v_sdk 环境安装（不需要海康 Mv3dRgbd/MVS SDK）
#
# 做的事：
#   1) 安装 udev 规则，让普通用户可直接访问 2bdf:0001 相机
#   2) 提高 usbfs 内存上限（大 URB 连续取流更稳）
#   3) 检查 libusb 与 python3
set -e
HERE=$(cd "$(dirname "$0")" && pwd)

echo "== [1/2] udev rule =="
sudo cp "$HERE/udev/99-tof-u3v.rules" /etc/udev/rules.d/
sudo udevadm control --reload-rules
sudo udevadm trigger --subsystem-match=usb || true
echo "installed /etc/udev/rules.d/99-tof-u3v.rules （重新插拔相机后生效）"

echo "== [2/2] usbfs memory =="
echo "options usbcore usbfs_memory_mb=2000" | sudo tee /etc/modprobe.d/tof-usbfs.conf >/dev/null
echo "written /etc/modprobe.d/tof-usbfs.conf；当前生效值：$(cat /sys/module/usbcore/parameters/usbfs_memory_mb 2>/dev/null || echo unknown) MB"
echo "如需立即把内存上限调到 2000MB： echo 2000 | sudo tee /sys/module/usbcore/parameters/usbfs_memory_mb"

echo "== 检查 =="
lsusb | grep -i 2bdf || echo "未发现 2bdf:0001 相机（请检查连接）"
python3 -c "import ctypes; ctypes.CDLL('libusb-1.0.so.0'); print('libusb OK')"
echo "完成。运行示例： python3 $HERE/src/u3v_lowlat.py --seconds 8"
