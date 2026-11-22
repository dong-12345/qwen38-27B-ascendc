import ctypes, sys
sys.path.insert(0, "/home/HwHiAiUser/Mv3dRgbdSDK/Res/demo/python")
from Mv3dRgbdImport.Mv3dRgbdApi import Mv3dRgbd
from Mv3dRgbdImport.Mv3dRgbdDefine import MV3D_RGBD_DEVICE_INFO_LIST, MV3D_RGBD_PARAM, DeviceType_USB, ParamType_Int
Mv3dRgbd.MV3D_RGBD_Initialize()
n = ctypes.c_uint(0)
Mv3dRgbd.MV3D_RGBD_GetDeviceNumber(DeviceType_USB, ctypes.byref(n))
devs = MV3D_RGBD_DEVICE_INFO_LIST()
Mv3dRgbd.MV3D_RGBD_GetDeviceList(DeviceType_USB, ctypes.pointer(devs.DeviceInfo[0]), 20, ctypes.byref(n))
cam = Mv3dRgbd()
rc = cam.MV3D_RGBD_OpenDevice(ctypes.pointer(devs.DeviceInfo[0]))
print("open", hex(rc))
def get_int(key):
    p = MV3D_RGBD_PARAM(); p.enParamType = ParamType_Int
    r = cam.MV3D_RGBD_GetParam(key, ctypes.pointer(p))
    return p.ParamInfo.stIntParam.nCurValue
def set_int(key, v):
    p = MV3D_RGBD_PARAM(); p.enParamType = ParamType_Int
    p.ParamInfo.stIntParam.nCurValue = int(v)
    return hex(cam.MV3D_RGBD_SetParam(key, ctypes.pointer(p)))
print("PixelFormat", hex(get_int("PixelFormat")))
print("set OutputRgbd=1 ->", set_int("OutputRgbd", 1), "readback", get_int("OutputRgbd"))
cam.MV3D_RGBD_CloseDevice()
print("done")
