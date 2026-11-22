#!/usr/bin/env python3
"""Configure native 640x480 depth and full-resolution planar RGB over U3V."""
import ctypes
import sys

sys.path.insert(0, "/home/HwHiAiUser/Mv3dRgbdSDK/Res/demo/python")

from Mv3dRgbdImport.Mv3dRgbdApi import Mv3dRgbd
from Mv3dRgbdImport.Mv3dRgbdDefine import (
    MV3D_RGBD_DEVICE_INFO_LIST, MV3D_RGBD_PARAM, DeviceType_USB, ParamType_Enum, ParamType_Int, ParamType_Bool, ParamType_Float,
)

IMAGE_MODE_DEPTH = 0x2


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
        settings = (
            ("ImageMode", ParamType_Enum, "stEnumParam", "nCurValue", IMAGE_MODE_DEPTH),
            ("ImageAlign", ParamType_Int, "stIntParam", "nCurValue", 0),
            ("OutputRgbd", ParamType_Int, "stIntParam", "nCurValue", 0),
            ("RGBJpegEnable", ParamType_Bool, None, "bBoolParam", False),
            ("AcquisitionFrameRate", ParamType_Float, "stFloatParam", "fCurValue", 30),
        )
        for key, kind, member, field, value in settings:
            param = MV3D_RGBD_PARAM()
            rc = cam.MV3D_RGBD_GetParam(key, ctypes.byref(param))
            if not rc and key == "ImageAlign" and param.enParamType == ParamType_Enum:
                kind, member = ParamType_Enum, "stEnumParam"
            if rc or param.enParamType != kind:
                print(f"[camera_setup] cannot read {key}: {rc & 0xffffffff:#010x}")
                return 1
            holder = getattr(param.ParamInfo, member) if member else param.ParamInfo
            setattr(holder, field, value)
            rc = cam.MV3D_RGBD_SetParam(key, ctypes.byref(param))
            if rc:
                print(f"[camera_setup] cannot set {key}: {rc & 0xffffffff:#010x}")
                return 1
            print(f"[camera_setup] {key} = {value}")
    finally:
        cam.MV3D_RGBD_CloseDevice()
    return 0


if __name__ == "__main__":
    sys.exit(main())
