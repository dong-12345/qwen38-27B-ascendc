#!/usr/bin/env python3
"""Capture N fresh RGB-D frames from the camera (SDK, ImageAlign=0).

Writes frames to <outdir>/frame_{i:02d}_{rgb.jpg,depth16.png}, prints md5 + timestamp
for each frame (proves live capture), and copies the LAST frame to
<outdir>/{color.jpg, depth16.png} for the fusion pipeline.
"""
import ctypes
import hashlib
import json
import os
import sys
import time

import cv2
import numpy as np

from Mv3dRgbdImport.Mv3dRgbdApi import Mv3dRgbd
from Mv3dRgbdImport.Mv3dRgbdDefine import (
    MV3D_RGBD_DEVICE_INFO_LIST, MV3D_RGBD_PARAM, MV3D_RGBD_FRAME_DATA,
    DeviceType_USB, ParamType_Int, MV3D_RGBD_INT_IMAGEALIGN,
    ImageType_Depth, ImageType_Jpeg, ImageType_RGB8_Planar,
)

outdir = sys.argv[1] if len(sys.argv) > 1 else "data/live_new"
nframes = int(sys.argv[2]) if len(sys.argv) > 2 else 3
os.makedirs(outdir, exist_ok=True)
Mv3dRgbd.MV3D_RGBD_Initialize()
n = ctypes.c_uint(0)
Mv3dRgbd.MV3D_RGBD_GetDeviceNumber(DeviceType_USB, ctypes.byref(n))
devs = MV3D_RGBD_DEVICE_INFO_LIST()
Mv3dRgbd.MV3D_RGBD_GetDeviceList(DeviceType_USB, ctypes.pointer(devs.DeviceInfo[0]), 20, ctypes.byref(n))
cam = Mv3dRgbd()
print("open", hex(cam.MV3D_RGBD_OpenDevice(ctypes.pointer(devs.DeviceInfo[0]))))
st = MV3D_RGBD_PARAM(); st.enParamType = ParamType_Int; st.ParamInfo.stIntParam.nCurValue = 0
cam.MV3D_RGBD_SetParam(MV3D_RGBD_INT_IMAGEALIGN, ctypes.pointer(st))
cam.MV3D_RGBD_Start()
frame = MV3D_RGBD_FRAME_DATA()
saved = []
t0 = time.time()
while len(saved) < nframes and time.time() - t0 < 20:
    r = cam.MV3D_RGBD_FetchFrame(ctypes.pointer(frame), 3000)
    if r != 0:
        continue
    rgb = None; dep = None
    for i in range(frame.nImageCount):
        im = frame.stImageData[i]
        et = int(im.enImageType)
        raw = ctypes.string_at(im.pData, int(im.nDataLen))
        if et == ImageType_Depth:
            dep = np.frombuffer(raw, np.uint16).reshape(int(im.nHeight), int(im.nWidth)).copy()
        elif et == ImageType_Jpeg:
            rgb = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        elif et == ImageType_RGB8_Planar:
            h, w = int(im.nHeight), int(im.nWidth)
            rgb = np.frombuffer(raw, np.uint8).reshape(3, h, w).transpose(1, 2, 0).copy()
    if rgb is None or dep is None:
        continue
    idx = len(saved)
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    fn_rgb = os.path.join(outdir, f"frame_{idx:02d}_rgb.jpg")
    fn_dep = os.path.join(outdir, f"frame_{idx:02d}_depth16.png")
    cv2.imwrite(fn_rgb, rgb)
    cv2.imwrite(fn_dep, dep)
    h = hashlib.md5(rgb.tobytes()).hexdigest()[:12]
    saved.append(dict(idx=idx, time=ts, rgb_md5=h, rgb_path=fn_rgb, depth_path=fn_dep))
    print(f"frame {idx}: time={ts} rgb_md5={h} mean_luma={cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY).mean():.1f}")
cam.MV3D_RGBD_Stop()
cam.MV3D_RGBD_CloseDevice()
Mv3dRgbd.MV3D_RGBD_Release()
if saved:
    import shutil
    shutil.copy(saved[-1]["rgb_path"], os.path.join(outdir, "color.jpg"))
    shutil.copy(saved[-1]["depth_path"], os.path.join(outdir, "depth16.png"))
    json.dump(dict(frames=saved, captured_at=time.strftime("%Y-%m-%d %H:%M:%S")),
              open(os.path.join(outdir, "capture_meta.json"), "w"), indent=1, ensure_ascii=False)
print("saved", len(saved), "frames to", outdir)
