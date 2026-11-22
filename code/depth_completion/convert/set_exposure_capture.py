#!/usr/bin/env python3
"""Set RGB exposure times via SDK, capture one frame each, report mean luma."""
import ctypes, os, sys, time
import numpy as np
import cv2
from Mv3dRgbdImport.Mv3dRgbdApi import Mv3dRgbd
from Mv3dRgbdImport.Mv3dRgbdDefine import (
    MV3D_RGBD_DEVICE_INFO_LIST, MV3D_RGBD_PARAM, MV3D_RGBD_FRAME_DATA,
    DeviceType_USB, ParamType_Int, ParamType_Float,
    MV3D_RGBD_INT_IMAGEALIGN, ImageType_Depth, ImageType_Jpeg, ImageType_RGB8_Planar,
)
OUT = "/home/HwHiAiUser/code/depth_completion/data/expo_test"
os.makedirs(OUT, exist_ok=True)
Mv3dRgbd.MV3D_RGBD_Initialize()
n = ctypes.c_uint(0)
Mv3dRgbd.MV3D_RGBD_GetDeviceNumber(DeviceType_USB, ctypes.byref(n))
devs = MV3D_RGBD_DEVICE_INFO_LIST()
Mv3dRgbd.MV3D_RGBD_GetDeviceList(DeviceType_USB, ctypes.pointer(devs.DeviceInfo[0]), 20, ctypes.byref(n))
cam = Mv3dRgbd()
print("open", hex(cam.MV3D_RGBD_OpenDevice(ctypes.pointer(devs.DeviceInfo[0]))))

def set_param(key, val):
    p = MV3D_RGBD_PARAM(); p.enParamType = ParamType_Float
    p.ParamInfo.stFloatParam.fCurValue = float(val)
    return cam.MV3D_RGBD_SetParam(key, ctypes.pointer(p))

def get_param(key):
    p = MV3D_RGBD_PARAM(); p.enParamType = ParamType_Float
    r = cam.MV3D_RGBD_GetParam(key, ctypes.pointer(p))
    return (r, float(p.ParamInfo.stFloatParam.fCurValue))

st = MV3D_RGBD_PARAM(); st.enParamType = ParamType_Int; st.ParamInfo.stIntParam.nCurValue = 0
cam.MV3D_RGBD_SetParam(MV3D_RGBD_INT_IMAGEALIGN, ctypes.pointer(st))
print("gain", get_param("Gain"))
cam.MV3D_RGBD_Start()
frame = MV3D_RGBD_FRAME_DATA()
for expo in (500, 2000, 5000, 10000):
    print("set ExposureTime", expo, "ret", hex(set_param("ExposureTime", expo)), "readback", get_param("ExposureTime"))
    time.sleep(0.6)
    for attempt in range(12):
        r = cam.MV3D_RGBD_FetchFrame(ctypes.pointer(frame), 3000)
        if r != 0:
            continue
        rgb = None; dep = None
        for i in range(frame.nImageCount):
            im = frame.stImageData[i]; et = int(im.enImageType)
            raw = ctypes.string_at(im.pData, int(im.nDataLen))
            if et == ImageType_Depth:
                dep = np.frombuffer(raw, np.uint16).reshape(int(im.nHeight), int(im.nWidth)).copy()
            elif et == ImageType_Jpeg:
                rgb = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
            elif et == ImageType_RGB8_Planar:
                h, w = int(im.nHeight), int(im.nWidth)
                rgb = np.frombuffer(raw, np.uint8).reshape(3, h, w).transpose(1, 2, 0).copy()
        if rgb is not None and dep is not None:
            g = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
            cv2.imwrite(os.path.join(OUT, f"color_{expo}.jpg"), rgb)
            cv2.imwrite(os.path.join(OUT, f"depth_{expo}.png"), dep)
            print(f"  expo={expo}: rgb {rgb.shape} mean_luma {g.mean():.1f} p50 {np.median(g):.1f} over255_frac {(g>250).mean():.3f}")
            break
cam.MV3D_RGBD_Stop()
cam.MV3D_RGBD_CloseDevice()
Mv3dRgbd.MV3D_RGBD_Release()
