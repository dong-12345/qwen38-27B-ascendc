/**
 * cube_matmul — 用 CANN 高层 Matmul API 在 310B 的 Cube 上做 fp16 GEMM（C 为 fp32）
 *
 * A: [M, K] fp16 (ND), B: [K, N] fp16 (ND), C: [M, N] fp32 (ND)
 * tiling 由 host 端 MultiCoreMatmulTiling 生成后经 GM 传入。
 */
#include "kernel_operator.h"
#include "lib/matmul_intf.h"

using namespace matmul;

__aicore__ inline uint32_t CeilDiv(uint32_t a, uint32_t b) { return (a + b - 1) / b; }

__aicore__ inline void CopyTiling(TCubeTiling *tiling, uint64_t &localMemSize, GM_ADDR tilingGM)
{
    uint32_t *ptr = reinterpret_cast<uint32_t *>(tiling);
    auto tiling32 = reinterpret_cast<__gm__ uint32_t *>(tilingGM);
    for (uint32_t i = 0; i < sizeof(TCubeTiling) / sizeof(uint32_t); i++, ptr++) {
        *ptr = *(tiling32 + i);
    }
    localMemSize = *reinterpret_cast<__gm__ uint64_t *>(tilingGM + sizeof(TCubeTiling));
}

extern "C" __global__ __aicore__ void cube_matmul(GM_ADDR a, GM_ADDR b, GM_ADDR c,
                                                  GM_ADDR workspace, GM_ADDR tilingGm)
{
    using A_T = half;
    using B_T = half;
    using C_T = float;

    AscendC::TPipe pipe;
    TCubeTiling tiling;
    uint64_t localMemSize = 0;
    CopyTiling(&tiling, localMemSize, tilingGm);

    AscendC::GlobalTensor<A_T> aGlobal;
    AscendC::GlobalTensor<B_T> bGlobal;
    AscendC::GlobalTensor<C_T> cGlobal;
    aGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ A_T *>(a), tiling.M * tiling.Ka);
    bGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ B_T *>(b), tiling.Ka * tiling.N);
    cGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ C_T *>(c), tiling.M * tiling.N);

    Matmul<MatmulType<AscendC::TPosition::GM, CubeFormat::ND, A_T>,
           MatmulType<AscendC::TPosition::GM, CubeFormat::ND, B_T>,
           MatmulType<AscendC::TPosition::GM, CubeFormat::ND, C_T>>
        mm;
    REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), mm, &tiling);
    if (GetBlockIdx() >= tiling.usedCoreNum) {
        return;
    }
    AscendC::TBuf<> tmpMMFormatUb;
    AscendC::LocalTensor<uint8_t> mmFormatUb;
    pipe.InitBuffer(tmpMMFormatUb, localMemSize > 0 ? localMemSize : 1024);
    mmFormatUb = tmpMMFormatUb.Get<uint8_t>(localMemSize > 0 ? localMemSize : 1024);
    mm.SetLocalWorkspace(mmFormatUb);

    mm.SetOrgShape(tiling.M, tiling.N, tiling.Ka, tiling.Kb);
    mm.SetTensorA(aGlobal, false);
    mm.SetTensorB(bGlobal, false);
    mm.SetTail(tiling.M, tiling.N);
    mm.IterateAll(cGlobal);
    mm.End();
}

#ifndef ASCENDC_CPU_DEBUG
void cube_matmul_do(uint32_t blockDim, void *stream, uint8_t *a, uint8_t *b, uint8_t *c,
                    uint8_t *workspace, uint8_t *tiling)
{
    cube_matmul<<<blockDim, nullptr, stream>>>(a, b, c, workspace, tiling);
}
#endif
