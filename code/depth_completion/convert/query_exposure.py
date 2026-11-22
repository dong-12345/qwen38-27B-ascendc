#!/usr/bin/env python3
"""Query Hikrobot MV-DT01SDU RGB camera exposure / gain settings and brightness behavior."""
import ctypes
import sys
import time

import numpy as np

from Mv3dRgbdImport.Mv3dRgbdApi import Mv3dRgbd
from Mv3dRgbdImport.Mv3dRgbdDefine import (
    MV3D_RGBD_DEVICE_INFO_LIST, MV3D_RGBD_PARAM, MV3D_RGBD_FRAME_DATA,
    DeviceType_USB, ParamType_Int, ParamType_Float, ParamType_Enum, ParamType_Bool,
    ImageType_Depth, ImageType_Rgbd, ImageType_Jpeg, ImageType_RGB8_Planar,
)

Mv3dRgbd.MV3D_RGBD_Initialize()
n = ctypes.c_uint(0)
Mv3dRgbd.MV3D_RGBD_GetDeviceNumber(DeviceType_USB, ctypes.byref(n))
devs = MV3D_RGBD_DEVICE_INFO_LIST()
Mv3dRgbd.MV3D_RGBD_GetDeviceList(DeviceType_USB, ctypes.pointer(devs.DeviceInfo[0]), 20, ctypes.byref(n))
cam = Mv3dRgbd()
ret = cam.MV3D_RGBD_OpenDevice(ctypes.pointer(devs.DeviceInfo[0]))
print("open_ret", hex(ret))
if ret != 0:
    sys.exit(1)

# print SDK version
try:
    from Mv3dRgbdImport.Mv3dRgbdDefine import MV3D_RGBD_VERSION_INFO
    v = MV3D_RGBD_VERSION_INFO()
    r = Mv3dRgbd.MV3D_RGBD_GetSDKVersion(ctypes.pointer(v))
    print("sdk_ret", hex(r), "version", getattr(v, "chVersion", ""), getattr(v, "nVersion", ""))
except Exception as e:
    print("sdk version probe failed:", e)

KEYS = ["ExposureTime", "Gain", "ExposureAuto", "ExposureMode", "AutoExposure",
        "AeEnable", "Brightness", "GainAuto", "AutoExposureTimeLowerLimit",
        "AutoExposureTimeUpperLimit", "ExposureTimeABS", "ExposureTimeMode"]
for key in KEYS:
    for ptype, name in ((ParamType_Float, "float"), (ParamType_Int, "int"), (ParamType_Enum, "enum"), (ParamType_Bool, "bool")):
        p = MV3D_RGBD_PARAM()
        p.enParamType = ptype
        try:
            r = cam.MV3D_RGBD_GetParam(key, ctypes.pointer(p))
        except Exception as e:
            print(f"  {key:32s} {name:5s} EXC {e}")
            continue
        if r == 0:
            val = None
            try:
                if ptype == ParamType_Float:
                    val = float(p.ParamInfo.stFloatParam.fCurValue)
                elif ptype == ParamType_Int:
                    val = int(p.ParamInfo.stIntParam.nCurValue)
                elif ptype == ParamType_Enum:
                    val = int(p.ParamInfo.stEnumParam.nCurValue)
                elif ptype == ParamType_Bool:
                    val = int(p.ParamInfo.stBoolParam.bCurValue)
            except Exception:
                val = "?"
            print(f"  {key:32s} {name:5s} OK value={val}")
            break
    else:
        print(f"  {key:32s} not supported")

# brightness over 6 frames (auto exposure check)
cam.MV3D_RGBD_Start()
frame = MV3D_RGBD_FRAME_DATA()
lums = []
t0 = time.time()
while len(lums) < 6 and time.time() - t0 < 10:
    r = cam.MV3D_RGBD_FetchFrame(ctypes.pointer(frame), 3000)
    if r != 0:
        continue
    for i in range(frame.nImageCount):
        im = frame.stImageData[i]
        et = int(im.enImageType)
        if et in (ImageType_Jpeg, ImageType_RGB8_Planar):
            raw = ctypes.string_at(im.pData, int(im.nDataLen))
            if et == ImageType_Jpeg:
                try:
                    import cv2
                    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_GRAYSCALE)
                except Exception:
                    img = None
            else:
                h, w = int(im.nHeight), int(im.nWidth)
                img = np.frombuffer(raw, np.uint8)[: w * h].reshape(h, w)
            if img is not None:
                lums.append(round(float(img.mean()), 2))
                break
cam.MV3D_RGBD_Stop()
cam.MV3D_RGBD_CloseDevice()
Mv3dRgbd.MV3D_RGBD_Release()
print("frame mean luma:", lums)
if len(lums) >= 2:
    print("luma variation (max-min):", round(max(lums) - min(lums), 2))
