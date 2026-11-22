/**
 * probe_tiling — 诊断 310B 上 CANN 高层 Matmul API 的 tiling 选择与实测算力
 * （perf-scout 新增，2026-11-19；独立于工程构建，g++ 直接编译）
 *
 * 目的：
 *   1) 打印 TCubeTiling 全字段，解释 "M>=128 后 TFLOP/s 不升反降"；
 *   2) 扫 SetTraverse / SetFixSplit / SetDoubleBuffer 等 tiling 旋钮，找 M 放大后
 *      让 B 矩阵只读一遍的配置（prefill 批处理的前提）。
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <cstdint>

#include "acl/acl.h"
#include "tiling/tiling_api.h"
#include "tiling/platform/platform_ascendc.h"

using namespace matmul_tiling;

static void Check(aclError r, const char *msg)
{
    if (r != ACL_ERROR_NONE) {
        printf("[FAIL] %s acl=%d msg=%s\n", msg, (int)r,
               aclGetRecentErrMsg() ? aclGetRecentErrMsg() : "(none)");
        std::exit(1);
    }
}

static uint8_t *AllocDev(size_t bytes)
{
    void *p = nullptr;
    Check(aclrtMalloc(&p, bytes, ACL_MEM_MALLOC_HUGE_FIRST), "Malloc");
    return (uint8_t *)p;
}

struct Cfg {
    int M, N, K;
    int baseM, baseN, baseK;   // >0 时 SetFixSplit
    int traverse;              // -1 不设；1=FIRSTM；2=FIRSTN
    int doubleBuf;             // 1 = SetDoubleBuffer(true,true,true,false)
    const char *tag;
};

// 注意：不要把 optiling::TCubeTiling 拷贝/赋值（内部带缓冲，会 double free）
static bool GenTiling(const Cfg &c, uint8_t *buf, size_t bufSize, size_t *outBytes, int *outCore,
                      int print)
{
    auto platform = platform_ascendc::PlatformAscendCManager::GetInstance("Ascend310B1");
    if (platform == nullptr) {
        printf("[tiling] platform null\n");
        return false;
    }
    optiling::TCubeTiling td;
    MultiCoreMatmulTiling api(*platform);
    int32_t coreNum = (int32_t)platform->GetCoreNumAic();
    if (coreNum <= 0) coreNum = 1;
    api.SetDim(coreNum);
    api.SetAType(TPosition::GM, CubeFormat::ND, DataType::DT_FLOAT16, false);
    api.SetBType(TPosition::GM, CubeFormat::ND, DataType::DT_FLOAT16, false);
    api.SetCType(TPosition::GM, CubeFormat::ND, DataType::DT_FLOAT);
    api.SetOrgShape(c.M, c.N, c.K);
    api.SetShape(c.M, c.N, c.K);
    if (c.baseM > 0) api.SetFixSplit(c.baseM, c.baseN, c.baseK);
    if (c.traverse == 1) api.SetTraverse(MatrixTraverse::FIRSTM);
    if (c.traverse == 2) api.SetTraverse(MatrixTraverse::FIRSTN);
    if (c.doubleBuf) api.SetDoubleBuffer(true, true, true, false);
    api.SetBias(false);
    api.SetBufferSpace(-1, -1, -1);

    if (api.GetTiling(td) == -1) {
        printf("[tiling] GetTiling FAIL %s\n", c.tag);
        return false;
    }
    uint32_t sz = td.GetDataSize();
    if ((size_t)sz + sizeof(uint64_t) > bufSize) return false;
    td.SaveToBuffer(buf, sz);
    uint64_t localMem = 0;
    platform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, localMem);
    *reinterpret_cast<uint64_t *>(buf + sz) = localMem;
    *outBytes = (size_t)sz + sizeof(uint64_t);
    *outCore = td.get_usedCoreNum() > 0 ? td.get_usedCoreNum() : 1;
    if (print) {
        printf("  %-28s | core=%d sM=%4d sN=%4d sK=%4d base=%dx%dx%d depthA1=%d depthB1=%d "
               "stepM=%d stepN=%d stepKa=%d stepKb=%d dbL0C=%d iterOrder=%d\n",
               c.tag, *outCore, (int)td.get_singleCoreM(), (int)td.get_singleCoreN(),
               (int)td.get_singleCoreK(), (int)td.get_baseM(), (int)td.get_baseN(),
               (int)td.get_baseK(), (int)td.get_depthA1(), (int)td.get_depthB1(),
               (int)td.get_stepM(), (int)td.get_stepN(), (int)td.get_stepKa(),
               (int)td.get_stepKb(), (int)td.get_dbL0C(), (int)td.get_iterateOrder());
    }
    return true;
}

void cube_matmul_do(uint32_t blockDim, void *stream, uint8_t *a, uint8_t *b, uint8_t *c,
                    uint8_t *workspace, uint8_t *tiling);

int main()
{
    aclrtStream stream = nullptr;
    Check(aclInit(nullptr), "aclInit");
    Check(aclrtSetDevice(0), "SetDevice");
    Check(aclrtCreateStream(&stream), "CreateStream");

    auto platform = platform_ascendc::PlatformAscendCManager::GetInstance("Ascend310B1");
    size_t wsSize = (size_t)platform->GetLibApiWorkSpaceSize() + (1 << 20);
    uint8_t *ws = AllocDev(wsSize);

    Cfg cfgs[] = {
        {64, 5120, 5120, 0, 0, 0, -1, 0, "M64 base(default)"},
        {128, 5120, 5120, 0, 0, 0, -1, 0, "M128 base"},
        {256, 5120, 5120, 0, 0, 0, -1, 0, "M256 base"},
        {512, 5120, 5120, 0, 0, 0, -1, 0, "M512 base"},
        {512, 5120, 5120, 0, 0, 0, 1, 0, "M512 FIRSTM"},
        {512, 5120, 5120, 0, 0, 0, 2, 0, "M512 FIRSTN"},
        {512, 5120, 5120, 0, 0, 0, -1, 1, "M512 doubleBuf"},
        {512, 5120, 5120, 128, 128, 64, 2, 0, "M512 fix128x128 FIRSTN"},
        {512, 5120, 5120, 64, 512, 64, 2, 0, "M512 fix64x512 FIRSTN"},
        {1024, 5120, 5120, 0, 0, 0, -1, 0, "M1024 base"},
        {1024, 5120, 5120, 0, 0, 0, 2, 0, "M1024 FIRSTN"},
        {1024, 5120, 5120, 128, 128, 64, 2, 0, "M1024 fix128x128 FIRSTN"},
        {2048, 2048, 2048, 0, 0, 0, 2, 0, "S2048 FIRSTN"},
    };
    const int nCfg = (int)(sizeof(cfgs) / sizeof(cfgs[0]));

    for (int i = 0; i < nCfg; ++i) {
        const Cfg &c = cfgs[i];
        size_t aBytes = (size_t)c.M * c.K * sizeof(uint16_t);
        size_t bBytes = (size_t)c.K * c.N * sizeof(uint16_t);
        size_t cBytes = (size_t)c.M * c.N * sizeof(float);
        uint8_t *a = AllocDev(aBytes);
        uint8_t *b = AllocDev(bBytes);
        uint8_t *cc = AllocDev(cBytes);
        Check(aclrtMemset(a, aBytes, 1, aBytes), "memset a");
        Check(aclrtMemset(b, bBytes, 1, bBytes), "memset b");

        uint8_t tilingHost[8192];
        std::memset(tilingHost, 0, sizeof(tilingHost));
        size_t tilingBytes = 0;
        int coreNum = 1;
        if (!GenTiling(c, tilingHost, sizeof(tilingHost), &tilingBytes, &coreNum, 1)) {
            Check(aclrtFree(a), "f"); Check(aclrtFree(b), "f"); Check(aclrtFree(cc), "f");
            continue;
        }
        uint8_t *tilingDev = AllocDev(tilingBytes);
        Check(aclrtMemcpy(tilingDev, tilingBytes, tilingHost, tilingBytes,
                          ACL_MEMCPY_HOST_TO_DEVICE), "copy tiling");
        double best = 1e9;
        for (int it = 0; it < 3; ++it) {
            auto t0 = std::chrono::high_resolution_clock::now();
            cube_matmul_do((uint32_t)coreNum, stream, a, b, cc, ws, tilingDev);
            Check(aclrtSynchronizeStream(stream), "sync");
            auto t1 = std::chrono::high_resolution_clock::now();
            double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            if (ms < best) best = ms;
        }
        double gflop = 2.0 * c.M * c.N * c.K / 1e9;
        printf("  %-28s -> %8.3f ms | %8.2f GFLOP/s | 权重 %6.2f GB/s\n", c.tag, best,
               gflop / (best / 1000.0), (double)bBytes / 1e9 / (best / 1000.0));
        Check(aclrtFree(a), "f"); Check(aclrtFree(b), "f"); Check(aclrtFree(cc), "f");
        Check(aclrtFree(tilingDev), "f");
    }
    Check(aclrtFree(ws), "f");
    Check(aclrtDestroyStream(stream), "f");
    Check(aclrtResetDevice(0), "f");
    Check(aclFinalize(), "f");
    return 0;
}
