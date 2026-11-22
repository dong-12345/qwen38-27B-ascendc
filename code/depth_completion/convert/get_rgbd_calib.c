/*
 * Export Hikrobot MV-DT01SDU factory RGB-D calibration via the Mv3dRgbd SDK.
 *
 * Reads MV3D_RGBD_CAMERA_PARAM (depth intrinsics + distortion, RGB intrinsics +
 * distortion, depth->RGB 4x4 extrinsic) and prints it as JSON, then optionally
 * writes the intrinsics/extrinsics used by our fusion pipeline.
 *
 * Build (on the board, after installing the SDK):
 *   gcc -O2 get_rgbd_calib.c -o get_rgbd_calib \
 *       -I/opt/Mv3dRgbdSDK/include \
 *       -L/opt/Mv3dRgbdSDK/lib/aarch64 -lMv3dRgbd -Wl,-rpath,/opt/Mv3dRgbdSDK/lib/aarch64
 *
 * Run:
 *   ./get_rgbd_calib            # prints JSON
 *   ./get_rgbd_calib out.json   # also writes it
 *
 * NOTE: the SDK must not be in use by another process. Camera is on USB (2bdf:0001).
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include "Mv3dRgbdApi.h"
#include "Mv3dRgbdDefine.h"

static void print_calib(const char *name, const MV3D_RGBD_CALIB_INFO *c)
{
    printf("  \"%s\": {\n", name);
    printf("    \"width\": %u, \"height\": %u,\n", c->nWidth, c->nHeight);
    printf("    \"fx\": %.6f, \"fy\": %.6f, \"cx\": %.6f, \"cy\": %.6f,\n",
           c->stIntrinsic.fData[0], c->stIntrinsic.fData[4],
           c->stIntrinsic.fData[2], c->stIntrinsic.fData[5]);
    printf("    \"K\": [%.6f, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f],\n",
           c->stIntrinsic.fData[0], c->stIntrinsic.fData[1], c->stIntrinsic.fData[2],
           c->stIntrinsic.fData[3], c->stIntrinsic.fData[4], c->stIntrinsic.fData[5],
           c->stIntrinsic.fData[6], c->stIntrinsic.fData[7], c->stIntrinsic.fData[8]);
    printf("    \"distortion_k1k2p1p2k3_k4k5k6s1s2s3s4\": [%.6f, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f]\n",
           c->stDistortion.fData[0], c->stDistortion.fData[1], c->stDistortion.fData[2],
           c->stDistortion.fData[3], c->stDistortion.fData[4], c->stDistortion.fData[5],
           c->stDistortion.fData[6], c->stDistortion.fData[7], c->stDistortion.fData[8],
           c->stDistortion.fData[9], c->stDistortion.fData[10], c->stDistortion.fData[11]);
    printf("  }");
}

int main(int argc, char **argv)
{
    MV3D_RGBD_STATUS ret;
    HANDLE handle = NULL;
    uint32_t nDev = 0;
    MV3D_RGBD_DEVICE_INFO devs[8];

    memset(devs, 0, sizeof(devs));
    ret = MV3D_RGBD_Initialize();
    if (ret != MV3D_RGBD_OK) { fprintf(stderr, "Initialize failed: 0x%08x\n", ret); return 1; }

    ret = MV3D_RGBD_GetDeviceList(DeviceType_USB, devs, 8, &nDev);
    if (ret != MV3D_RGBD_OK || nDev == 0) {
        fprintf(stderr, "GetDeviceList failed: 0x%08x nDev=%u\n", ret, nDev);
        MV3D_RGBD_Release();
        return 1;
    }
    printf("{\n  \"device\": {\"serial\": \"%s\", \"model\": \"%s\", \"sdk_count\": %u},\n",
           devs[0].chSerialNumber, devs[0].chModelName, nDev);

    ret = MV3D_RGBD_OpenDevice(&handle, &devs[0]);
    if (ret != MV3D_RGBD_OK) {
        fprintf(stderr, "OpenDevice failed: 0x%08x\n", ret);
        MV3D_RGBD_Release();
        return 1;
    }

    MV3D_RGBD_CAMERA_PARAM cam;
    memset(&cam, 0, sizeof(cam));
    ret = MV3D_RGBD_GetCameraParam(handle, &cam);
    if (ret != MV3D_RGBD_OK) {
        fprintf(stderr, "GetCameraParam failed: 0x%08x\n", ret);
    } else {
        print_calib("depth", &cam.stDepthCalibInfo);
        printf(",\n");
        print_calib("rgb", &cam.stRgbCalibInfo);
        printf(",\n");
        printf("  \"extrinsic_depth_to_rgb_4x4\": [");
        for (int i = 0; i < 16; ++i) {
            printf("%.6f%s", cam.stDepth2RgbExtrinsic.fData[i], (i == 15) ? "" : ", ");
        }
        printf("],\n");
    }

    /* also try per-coordinate-type calibration info (some SDK versions differ) */
    MV3D_RGBD_CALIB_INFO ci;
    memset(&ci, 0, sizeof(ci));
    ret = MV3D_RGBD_GetCalibInfo(handle, CoordinateType_Depth, &ci);
    printf("  \"get_calib_info_depth_status\": %u,\n", ret);
    if (ret == MV3D_RGBD_OK) {
        printf("  \"calib_info_depth\": {\"w\": %u, \"h\": %u, \"fx\": %.6f, \"fy\": %.6f, \"cx\": %.6f, \"cy\": %.6f}\n",
               ci.nWidth, ci.nHeight, ci.stIntrinsic.fData[0], ci.stIntrinsic.fData[4],
               ci.stIntrinsic.fData[2], ci.stIntrinsic.fData[5]);
    } else {
        printf("  \"calib_info_depth\": null\n");
    }
    printf("}\n");

    MV3D_RGBD_CloseDevice(&handle);
    MV3D_RGBD_Release();
    return 0;
}
