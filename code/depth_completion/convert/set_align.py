#!/usr/bin/env python3
"""Set the Hikrobot camera ImageAlign mode (0 = raw depth frame, 1 = pre-aligned)."""
import ctypes
import sys

from Mv3dRgbdImport.Mv3dRgbdApi import Mv3dRgbd
from Mv3dRgbdImport.Mv3dRgbdDefine import (
    MV3D_RGBD_DEVICE_INFO_LIST, MV3D_RGBD_PARAM, DeviceType_USB,
    ParamType_Int, MV3D_RGBD_INT_IMAGEALIGN,
)

val = int(sys.argv[1]) if len(sys.argv) > 1 else 0
Mv3dRgbd.MV3D_RGBD_Initialize()
n = ctypes.c_uint(0)
Mv3dRgbd.MV3D_RGBD_GetDeviceNumber(DeviceType_USB, ctypes.byref(n))
devs = MV3D_RGBD_DEVICE_INFO_LIST()
Mv3dRgbd.MV3D_RGBD_GetDeviceList(DeviceType_USB, ctypes.pointer(devs.DeviceInfo[0]), 20, ctypes.byref(n))
cam = Mv3dRgbd()
ret = cam.MV3D_RGBD_OpenDevice(ctypes.pointer(devs.DeviceInfo[0]))
if ret != 0:
    print("open failed", hex(ret)); sys.exit(1)
st = MV3D_RGBD_PARAM()
st.enParamType = ParamType_Int
st.ParamInfo.stIntParam.nCurValue = val
ret = cam.MV3D_RGBD_SetParam(MV3D_RGBD_INT_IMAGEALIGN, ctypes.pointer(st))
print("set ImageAlign =", val, "ret =", hex(ret))
cam.MV3D_RGBD_CloseDevice()
Mv3dRgbd.MV3D_RGBD_Release()
