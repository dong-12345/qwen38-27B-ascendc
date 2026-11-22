#!/usr/bin/env python3
"""Recover camera, capture 2 fresh frames with SDK (known config), save depth+RGB.
Falls back to U3V parsing if SDK start fails.
"""
import ctypes, os, sys, time, hashlib, json, io
import numpy as np
from PIL import Image

OUT = "/home/HwHiAiUser/code/depth_completion/data/live_new"
os.makedirs(OUT, exist_ok=True)
from Mv3dRgbdImport.Mv3dRgbdApi import Mv3dRgbd
from Mv3dRgbdImport.Mv3dRgbdDefine import (
    MV3D_RGBD_DEVICE_INFO_LIST, MV3D_RGBD_PARAM, MV3D_RGBD_FRAME_DATA,
    DeviceType_USB, ParamType_Int, MV3D_RGBD_INT_IMAGEALIGN,
    ImageType_Depth, ImageType_Jpeg, ImageType_RGB8_Planar,
)

def save_frame(dep, rgb, tag=""):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    dep_path = os.path.join(OUT, "depth16.png")
    rgb_path = os.path.join(OUT, "color.jpg")
    Image.fromarray(dep).save(dep_path)
    Image.fromarray(rgb).save(rgb_path, quality=92)
    meta = dict(captured_at=ts,
                rgb_md5=hashlib.md5(rgb.tobytes()).hexdigest()[:12],
                depth_md5=hashlib.md5(dep.tobytes()).hexdigest()[:12],
                depth_valid=round(float((dep > 0).mean()), 4),
                rgb_mean_luma=round(float(rgb.mean()), 1),
                rgb_shape=list(rgb.shape), depth_shape=list(dep.shape), source=tag)
    json.dump(meta, open(os.path.join(OUT, "capture_meta.json"), "w"), indent=1)
    print("SAVED", json.dumps(meta))
    return True

def sdk_capture():
    Mv3dRgbd.MV3D_RGBD_Initialize()
    n = ctypes.c_uint(0)
    Mv3dRgbd.MV3D_RGBD_GetDeviceNumber(DeviceType_USB, ctypes.byref(n))
    if n.value == 0:
        print("no device"); return False
    devs = MV3D_RGBD_DEVICE_INFO_LIST()
    Mv3dRgbd.MV3D_RGBD_GetDeviceList(DeviceType_USB, ctypes.pointer(devs.DeviceInfo[0]), 20, ctypes.byref(n))
    cam = Mv3dRgbd()
    r = cam.MV3D_RGBD_OpenDevice(ctypes.pointer(devs.DeviceInfo[0]))
    print("sdk open", hex(r))
    if r != 0:
        return False
    st = MV3D_RGBD_PARAM(); st.enParamType = ParamType_Int; st.ParamInfo.stIntParam.nCurValue = 0
    print("set align0", hex(cam.MV3D_RGBD_SetParam(MV3D_RGBD_INT_IMAGEALIGN, ctypes.pointer(st))))
    ok = False
    for attempt in range(3):
        r = cam.MV3D_RGBD_Start()
        print(f"start attempt {attempt}: {hex(r)}")
        if r == 0:
            break
        cam.MV3D_RGBD_Stop(); time.sleep(1.0)
    if r != 0:
        cam.MV3D_RGBD_CloseDevice(); Mv3dRgbd.MV3D_RGBD_Release(); return False
    frame = MV3D_RGBD_FRAME_DATA()
    t0 = time.time(); got = 0
    while time.time() - t0 < 15 and got < 2:
        r = cam.MV3D_RGBD_FetchFrame(ctypes.pointer(frame), 3000)
        if r != 0:
            continue
        dep = rgb = None
        for i in range(frame.nImageCount):
            im = frame.stImageData[i]; et = int(im.enImageType)
            raw = ctypes.string_at(im.pData, int(im.nDataLen))
            if et == ImageType_Depth:
                dep = np.frombuffer(raw, np.uint16).reshape(int(im.nHeight), int(im.nWidth)).copy()
            elif et == ImageType_Jpeg:
                rgb = np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"))
            elif et == ImageType_RGB8_Planar:
                h, w = int(im.nHeight), int(im.nWidth)
                rgb = np.frombuffer(raw, np.uint8).reshape(3, h, w).transpose(1, 2, 0).copy()
        if dep is not None and rgb is not None:
            got += 1
            if got == 2:
                ok = save_frame(dep, rgb, "sdk")
    cam.MV3D_RGBD_Stop()
    cam.MV3D_RGBD_CloseDevice()
    Mv3dRgbd.MV3D_RGBD_Release()
    return ok

if __name__ == "__main__":
    if not sdk_capture():
        print("SDK capture failed")
        sys.exit(1)
