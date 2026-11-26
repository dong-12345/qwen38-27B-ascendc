/**
 * matmul_tiling.cpp — host 端为 cube_matmul 生成 TCubeTiling（MultiCoreMatmulTiling）
 */
#include <cstdio>
#include <cstring>

#include "tiling/tiling_api.h"
#include "tiling/platform/platform_ascendc.h"

using namespace matmul_tiling;

// 返回需要写入 tiling GM 的字节数（TCubeTiling + uint64 localMemSize）
size_t GenerateMatmulTiling(int M, int N, int K, uint8_t *tilingBuf, size_t bufSize,
                            int *outCoreNum, size_t *outLocalMem)
{
    optiling::TCubeTiling tilingData;
    auto platform = platform_ascendc::PlatformAscendCManager::GetInstance("Ascend310B1");
    if (platform == nullptr) {
        printf("[tiling] PlatformAscendCManager null\n");
        return 0;
    }
    MultiCoreMatmulTiling tilingApi(*platform);
    int32_t coreNum = platform->GetCoreNumAic();
    if (coreNum <= 0) {
        coreNum = 1;
    }
    tilingApi.SetDim(coreNum);
    tilingApi.SetAType(TPosition::GM, CubeFormat::ND, DataType::DT_FLOAT16, false);
    tilingApi.SetBType(TPosition::GM, CubeFormat::ND, DataType::DT_FLOAT16, false);
    tilingApi.SetCType(TPosition::GM, CubeFormat::ND, DataType::DT_FLOAT);
    tilingApi.SetOrgShape(M, N, K);
    tilingApi.SetShape(M, N, K);
    tilingApi.SetBias(false);
    tilingApi.SetBufferSpace(-1, -1, -1);

    int64_t res = tilingApi.GetTiling(tilingData);
    if (res == -1) {
        printf("[tiling] GetTiling failed for M=%d N=%d K=%d\n", M, N, K);
        return 0;
    }
    uint32_t tcubeTilingSize = tilingData.GetDataSize();
    size_t need = tcubeTilingSize + sizeof(uint64_t);
    if (need > bufSize) {
        printf("[tiling] buffer too small need=%zu buf=%zu\n", need, bufSize);
        return 0;
    }
    tilingData.SaveToBuffer(tilingBuf, tcubeTilingSize);
    uint64_t localMemSize = 0;
    platform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, localMemSize);
    *reinterpret_cast<uint64_t *>(tilingBuf + tcubeTilingSize) = localMemSize;
    int usedCores = tilingData.get_usedCoreNum();
    if (outCoreNum != nullptr) {
        *outCoreNum = usedCores > 0 ? usedCores : 1;
    }
    if (outLocalMem != nullptr) {
        *outLocalMem = localMemSize;
    }
    return need;
}

size_t GetMatmulWorkspaceSize()
{
    auto platform = platform_ascendc::PlatformAscendCManager::GetInstance("Ascend310B1");
    if (platform == nullptr) {
        return 1 << 20;
    }
    return static_cast<size_t>(platform->GetLibApiWorkSpaceSize()) + (1 << 20);
}
