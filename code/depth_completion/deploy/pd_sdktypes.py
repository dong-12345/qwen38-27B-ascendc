#!/usr/bin/env python3
"""After a clean U3V session, ask the SDK what image nodes the camera sends."""
import ctypes, os, sys, time
import numpy as np
sys.path.insert(0, "/home/HwHiAiUser/code/depth_completion/deploy")
from pd_capture import U3V

# 1) clean U3V session: enable stream, grab one frame, then stop
u = U3V(); u.open(); u.start(); time.sleep(1.0)
n = 0
for _ in range(4):
    p, st = u.read_payload(2000)
    if p is not None:
        n = len(p); break
u.stop(); time.sleep(0.5); u.close()
print("u3v payload:", n)

# 2) SDK
from Mv3dRgbdImport.Mv3dRgbdApi import Mv3dRgbd
from Mv3dRgbdImport.Mv3dRgbdDefine import (MV3D_RGBD_DEVICE_INFO_LIST, MV3D_RGBD_FRAME_DATA, DeviceType_USB)
Mv3dRgbd.MV3D_RGBD_Initialize()
cnt = ctypes.c_uint(0)
Mv3dRgbd.MV3D_RGBD_GetDeviceNumber(DeviceType_USB, ctypes.byref(cnt))
devs = MV3D_RGBD_DEVICE_INFO_LIST()
Mv3dRgbd.MV3D_RGBD_GetDeviceList(DeviceType_USB, ctypes.pointer(devs.DeviceInfo[0]), 20, ctypes.byref(cnt))
cam = Mv3dRgbd()
print("open", hex(cam.MV3D_RGBD_OpenDevice(ctypes.pointer(devs.DeviceInfo[0])) & 0xFFFFFFFF))
print("start", hex(cam.MV3D_RGBD_Start() & 0xFFFFFFFF))
fr = MV3D_RGBD_FRAME_DATA()
for i in range(6):
    r = cam.MV3D_RGBD_FetchFrame(ctypes.pointer(fr), 2000)
    if r != 0:
        print("fetch", hex(r & 0xFFFFFFFF)); continue
    print("frame %d: nImageCount=%d" % (i, fr.nImageCount))
    for j in range(fr.nImageCount):
        im = fr.stImageData[j]
        print("   type=%d w=%d h=%d len=%d" % (int(im.enImageType), int(im.nWidth), int(im.nHeight), int(im.nDataLen)))
    break
cam.MV3D_RGBD_Stop(); cam.MV3D_RGBD_CloseDevice(); Mv3dRgbd.MV3D_RGBD_Release()
