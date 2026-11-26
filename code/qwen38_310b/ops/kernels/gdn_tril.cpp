/**
 * gdn_tril — GDN chunk 核心第二步：solve_tril
 *
 *   T = (I + A)^{-1}，其中 A 是 64x64 严格下三角矩阵（对角为 0）
 *
 * 行式前代：
 *   T[0,:] = e_0
 *   T[i,:] = e_i - Σ_{k<i} A[i,k] * T[k,:]
 * 由于 T[k,:] 只在列 ≤ k 处非零，逐行 Axpy 即可，无需显式三角掩码。
 */
#include "kernel_operator.h"

using namespace AscendC;

constexpr int32_t T = 64;
constexpr int32_t TT = T * T;

class KernelGdnTril {
public:
    __aicore__ inline KernelGdnTril() {}

    __aicore__ inline void Init(GM_ADDR a, GM_ADDR t)
    {
        aGm.SetGlobalBuffer((__gm__ float *)a, TT);
        tGm.SetGlobalBuffer((__gm__ float *)t, TT);
        pipe.InitBuffer(aBuf, TT * sizeof(float));
        pipe.InitBuffer(tBuf, TT * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        LocalTensor<float> aL = aBuf.Get<float>();
        LocalTensor<float> tl = tBuf.Get<float>();
        DataCopy(aL, aGm, TT);
        PipeBarrier<PIPE_ALL>();
        AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);

        // 初始化 T = 0
        Duplicate(tl, 0.0f, TT);
        PipeBarrier<PIPE_V>();

        for (int32_t i = 0; i < T; ++i) {
            // T[i,:] = e_i
            tl.SetValue(i * T + i, 1.0f);
            AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
            // T[i,:] -= Σ_{k<i} A[i,k] * T[k,:]
            for (int32_t k = 0; k < i; ++k) {
                float coeff = -aL.GetValue(i * T + k);
                Axpy(tl[i * T], tl[k * T], coeff, T);
            }
            PipeBarrier<PIPE_V>();
        }
        PipeBarrier<PIPE_ALL>();
        DataCopy(tGm, tl, TT);
        PipeBarrier<PIPE_ALL>();
    }

private:
    TPipe pipe;
    TBuf<TPosition::VECCALC> aBuf, tBuf;
    GlobalTensor<float> aGm, tGm;
};

extern "C" __global__ __aicore__ void gdn_tril(GM_ADDR a, GM_ADDR t)
{
    KernelGdnTril op;
    op.Init(a, t);
    op.Process();
}

#ifndef ASCENDC_CPU_DEBUG
void gdn_tril_do(uint32_t blockDim, void *stream, uint8_t *a, uint8_t *t)
{
    gdn_tril<<<blockDim, nullptr, stream>>>(a, t);
}
#endif
