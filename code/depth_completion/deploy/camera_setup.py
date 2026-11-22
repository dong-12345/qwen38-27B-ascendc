#!/usr/bin/env python3
"""camera_setup.py - restore the RGBD stream configuration through the vendor SDK.

Two things are needed for the camera to stream depth + colour + IR:
  * ImageMode = 0x8   (colour mode, which flips PixelFormat to RGB8_Planar)
  * OutputRgbd = 1    (also emit the depth plane; this parameter is VOLATILE
                       and reverts to 0 whenever the SDK closes, so it has to be
                       re-applied before every capture session)

Run this once before starting pd_live.py:
    LD_LIBRARY_PATH must include the SDK libs (run_live.sh sets it).
"""
import ctypes
import sys

sys.path.insert(0, "/home/HwHiAiUser/Mv3dRgbdSDK/Res/demo/python")

from Mv3dRgbdImport.Mv3dRgbdApi import Mv3dRgbd
from Mv3dRgbdImport.Mv3dRgbdDefine import (
    MV3D_RGBD_DEVICE_INFO_LIST, MV3D_RGBD_PARAM, DeviceType_USB, ParamType_Enum, ParamType_Int,
)

IMAGE_MODE_COLOUR = 0x8


def main():
    Mv3dRgbd.MV3D_RGBD_Initialize()
    n = ctypes.c_uint(0)
    Mv3dRgbd.MV3D_RGBD_GetDeviceNumber(DeviceType_USB, ctypes.byref(n))
    if n.value == 0:
        print("[camera_setup] no device (camera busy or unplugged)")
        return 1
    devs = MV3D_RGBD_DEVICE_INFO_LIST()
    Mv3dRgbd.MV3D_RGBD_GetDeviceList(DeviceType_USB, ctypes.pointer(devs.DeviceInfo[0]), 20, ctypes.byref(n))
    cam = Mv3dRgbd()
    rc = cam.MV3D_RGBD_OpenDevice(ctypes.pointer(devs.DeviceInfo[0]))
    if rc != 0:
        print("[camera_setup] open failed 0x%08x" % rc)
        return 1
    try:
        p = MV3D_RGBD_PARAM(); p.enParamType = ParamType_Enum
        cam.MV3D_RGBD_GetParam("ImageMode", ctypes.pointer(p))
        cur = p.ParamInfo.stEnumParam.nCurValue
        if cur != IMAGE_MODE_COLOUR:
            p = MV3D_RGBD_PARAM(); p.enParamType = ParamType_Enum
            p.ParamInfo.stEnumParam.nCurValue = IMAGE_MODE_COLOUR
            print("[camera_setup] ImageMode -> 0x8 rc=0x%08x"
                  % cam.MV3D_RGBD_SetParam("ImageMode", ctypes.pointer(p)))
        p = MV3D_RGBD_PARAM(); p.enParamType = ParamType_Int
        p.ParamInfo.stIntParam.nCurValue = 1
        cam.MV3D_RGBD_SetParam("OutputRgbd", ctypes.pointer(p))
        p = MV3D_RGBD_PARAM(); p.enParamType = ParamType_Int
        cam.MV3D_RGBD_GetParam("OutputRgbd", ctypes.pointer(p))
        print("[camera_setup] OutputRgbd = %d" % p.ParamInfo.stIntParam.nCurValue)
    finally:
        cam.MV3D_RGBD_CloseDevice()
    return 0


if __name__ == "__main__":
    sys.exit(main())
