#!/usr/bin/env python3
"""Dump Hikrobot MV-DT01SDU factory RGB-D calibration via Mv3dRgbd SDK.

IMPORTANT: the SDK's `ImageAlign` parameter changes what `GetCameraParam` reports:
  * ImageAlign=1 (SDK default): depth is pre-warped into the RGB frame; the reported
    depth/RGB calib are identical (1536x1280) and the extrinsic is identity.
  * ImageAlign=0: raw 640x480 depth frame; reported calib are the *true* depth
    intrinsics/distortion and the true depth->RGB 4x4 extrinsic.
We therefore dump BOTH modes (align0 is the one needed for our 30fps U8V path).
"""
import ctypes
import json
import sys

try:
    from Mv3dRgbdImport.Mv3dRgbdApi import Mv3dRgbd
    from Mv3dRgbdImport.Mv3dRgbdDefine import (
        MV3D_RGBD_DEVICE_INFO_LIST, MV3D_RGBD_CAMERA_PARAM, MV3D_RGBD_PARAM,
        DeviceType_USB, ParamType_Int, MV3D_RGBD_INT_IMAGEALIGN, MV3D_RGBD_OK,
    )
except Exception as e:  # pragma: no cover
    print(json.dumps({"error": f"import Mv3dRgbd SDK failed: {e}"}))
    sys.exit(2)


def charset(chs):
    return "".join(chr(c) for c in chs if c != 0)


def calib_dict(ci):
    return dict(
        width=int(ci.nWidth), height=int(ci.nHeight),
        K=[round(float(v), 8) for v in ci.stIntrinsic.fData],
        distortion_k1k2p1p2k3_k4k5k6s1s2s3s4=[round(float(v), 8) for v in ci.stDistortion.fData],
        fx=round(float(ci.stIntrinsic.fData[0]), 8), fy=round(float(ci.stIntrinsic.fData[4]), 8),
        cx=round(float(ci.stIntrinsic.fData[2]), 8), cy=round(float(ci.stIntrinsic.fData[5]), 8),
    )


def dump_camera(camera):
    cam = MV3D_RGBD_CAMERA_PARAM()
    ret = camera.MV3D_RGBD_GetCameraParam(ctypes.pointer(cam))
    if ret != 0:
        return {"get_camera_param_ret": hex(ret)}
    ext = cam.stDepth2RgbExtrinsic.fData
    return {
        "get_camera_param_ret": hex(ret),
        "depth": calib_dict(cam.stDepthCalibInfo),
        "rgb": calib_dict(cam.stRgbCalibInfo),
        "extrinsic_depth_to_rgb_4x4": [round(float(v), 8) for v in ext],
    }


def set_image_align(camera, value):
    st = MV3D_RGBD_PARAM()
    st.enParamType = ParamType_Int
    st.ParamInfo.stIntParam.nCurValue = value
    ret = camera.MV3D_RGBD_SetParam(MV3D_RGBD_INT_IMAGEALIGN, ctypes.pointer(st))
    return hex(ret)


def main():
    out = {}
    ret = Mv3dRgbd.MV3D_RGBD_Initialize()
    out["initialize_ret"] = hex(ret)
    if ret != 0:
        print(json.dumps(out)); return

    n = ctypes.c_uint(0)
    ret = Mv3dRgbd.MV3D_RGBD_GetDeviceNumber(DeviceType_USB, ctypes.byref(n))
    out["device_count"] = int(n.value)
    if ret != 0 or n.value == 0:
        Mv3dRgbd.MV3D_RGBD_Release(); print(json.dumps(out)); return

    devs = MV3D_RGBD_DEVICE_INFO_LIST()
    ret = Mv3dRgbd.MV3D_RGBD_GetDeviceList(DeviceType_USB, ctypes.pointer(devs.DeviceInfo[0]),
                                           20, ctypes.byref(n))
    if ret != 0:
        Mv3dRgbd.MV3D_RGBD_Release(); print(json.dumps(out)); return
    out["device"] = dict(model=charset(devs.DeviceInfo[0].chModelName),
                         serial=charset(devs.DeviceInfo[0].chSerialNumber))

    camera = Mv3dRgbd()
    ret = camera.MV3D_RGBD_OpenDevice(ctypes.pointer(devs.DeviceInfo[0]))
    out["open_ret"] = hex(ret)
    if ret != 0:
        Mv3dRgbd.MV3D_RGBD_Release(); print(json.dumps(out)); return

    out["set_align0_ret"] = set_image_align(camera, 0)
    out["align0"] = dump_camera(camera)
    out["set_align1_ret"] = set_image_align(camera, 1)
    out["align1"] = dump_camera(camera)

    # backward-compatible top-level keys (align0 = true depth-camera calibration)
    for k in ("depth", "rgb", "extrinsic_depth_to_rgb_4x4"):
        if k in out["align0"]:
            out[k] = out["align0"][k]

    camera.MV3D_RGBD_CloseDevice()
    Mv3dRgbd.MV3D_RGBD_Release()
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
