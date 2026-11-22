#!/usr/bin/env python3
"""Fetch one SDK-aligned RGB-D frame and one raw (unaligned) frame from MV-DT01SDU.

Uses the Mv3dRgbd SDK Python wrapper; output:
  results/sdk_capture/aligned_depth_1536x1280.png   (C16 mm)
  results/sdk_capture/aligned_rgb.png               (RGB)
  results/sdk_capture/raw_depth_640x480.png         (C16 mm, ImageAlign=0)
  results/sdk_capture/raw_rgb.jpg
  results/sdk_capture/meta.json
"""
import ctypes
import json
import os
import sys

import numpy as np
from PIL import Image

from Mv3dRgbdImport.Mv3dRgbdApi import Mv3dRgbd
from Mv3dRgbdImport.Mv3dRgbdDefine import (
    MV3D_RGBD_DEVICE_INFO_LIST, MV3D_RGBD_PARAM, MV3D_RGBD_FRAME_DATA,
    DeviceType_USB, ParamType_Int, MV3D_RGBD_INT_IMAGEALIGN, MV3D_RGBD_OK,
    ImageType_Depth, ImageType_RGB8_Planar, ImageType_Jpeg,
)

OUT = "/home/HwHiAiUser/code/depth_completion/results/sdk_capture"
os.makedirs(OUT, exist_ok=True)


def get_images(frame, camera, want_depth=True, want_rgb=True):
    depth = rgb = None
    for i in range(frame.nImageCount):
        im = frame.stImageData[i]
        w, h, ln = int(im.nWidth), int(im.nHeight), int(im.nDataLen)
        raw = ctypes.string_at(im.pData, ln)
        et = int(im.enImageType)
        if et == ImageType_Depth and want_depth:
            depth = np.frombuffer(raw, dtype=np.uint16).reshape(h, w).copy()
        elif et == ImageType_RGB8_Planar and want_rgb:
            rgb = np.frombuffer(raw, dtype=np.uint8).reshape(3, h, w).transpose(1, 2, 0).copy()
        elif et == ImageType_Jpeg and want_rgb:
            from io import BytesIO
            rgb = np.asarray(Image.open(BytesIO(raw)).convert("RGB"))
    return depth, rgb


def grab(depth_path, rgb_path, align):
    ret = Mv3dRgbd.MV3D_RGBD_Initialize()
    n = ctypes.c_uint(0)
    Mv3dRgbd.MV3D_RGBD_GetDeviceNumber(DeviceType_USB, ctypes.byref(n))
    devs = MV3D_RGBD_DEVICE_INFO_LIST()
    Mv3dRgbd.MV3D_RGBD_GetDeviceList(DeviceType_USB, ctypes.pointer(devs.DeviceInfo[0]), 20, ctypes.byref(n))
    camera = Mv3dRgbd()
    ret = camera.MV3D_RGBD_OpenDevice(ctypes.pointer(devs.DeviceInfo[0]))
    if ret != 0:
        return {"open_ret": hex(ret)}
    st = MV3D_RGBD_PARAM()
    st.enParamType = ParamType_Int
    st.ParamInfo.stIntParam.nCurValue = align
    set_ret = camera.MV3D_RGBD_SetParam(MV3D_RGBD_INT_IMAGEALIGN, ctypes.pointer(st))
    start_ret = camera.MV3D_RGBD_Start()
    info = dict(align=align, set_ret=hex(set_ret), start_ret=hex(start_ret))
    frame = MV3D_RGBD_FRAME_DATA()
    got = None
    for _ in range(10):
        r = camera.MV3D_RGBD_FetchFrame(ctypes.pointer(frame), 5000)
        info["fetch_ret"] = hex(r)
        if r == MV3D_RGBD_OK:
            depth, rgb = get_images(frame, camera)
            if depth is not None:
                got = (depth, rgb)
                break
    camera.MV3D_RGBD_Stop()
    camera.MV3D_RGBD_CloseDevice()
    Mv3dRgbd.MV3D_RGBD_Release()
    if got is None:
        return info
    depth, rgb = got
    if depth_path:
        Image.fromarray(depth).save(depth_path)
    if rgb_path and rgb is not None:
        Image.fromarray(rgb).save(rgb_path)
    info["depth_shape"] = list(depth.shape)
    info["rgb_shape"] = list(rgb.shape) if rgb is not None else None
    if depth.size:
        info["depth_min"] = int(depth.min()); info["depth_max"] = int(depth.max())
        info["depth_nonzero"] = float((depth > 0).mean())
    return info


if __name__ == "__main__":
    meta = {}
    meta["aligned"] = grab(os.path.join(OUT, "aligned_depth_1536x1280.png"),
                           os.path.join(OUT, "aligned_rgb.png"), 1)
    meta["raw"] = grab(os.path.join(OUT, "raw_depth_640x480.png"),
                       os.path.join(OUT, "raw_rgb.jpg"), 0)
    json.dump(meta, open(os.path.join(OUT, "meta.json"), "w"), indent=1)
    print(json.dumps(meta, indent=1))
