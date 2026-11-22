/**
 * gdn_wy — GDN chunk 核心第三步：recompute_w_u_fwd
 *
 *   w = T · (k * beta * exp(gc))      [64,128]
 *   u = T · (v * beta)                [64,128]
 *
 *   T = (I + A)^{-1}（gdn_tril 的输出），A 严格下三角 → 只需 k <= i 的累加。
 *
 * 实现：行式秩-1 累加：w_i = Σ_{k<=i} T[i,k] * (k_k * cb_k)，u_i = Σ T[i,k]*beta_k*v_k。
 */
#include "kernel_operator.h"

using namespace AscendC;

constexpr int32_t T = 64;
constexpr int32_t D = 128;

class KernelGdnWy {
public:
    __aicore__ inline KernelGdnWy() {}

    __aicore__ inline void Init(GM_ADDR t, GM_ADDR k, GM_ADDR v, GM_ADDR beta, GM_ADDR gc,
                                GM_ADDR w, GM_ADDR u)
    {
        tGm.SetGlobalBuffer((__gm__ float *)t, T * T);
        kGm.SetGlobalBuffer((__gm__ half *)k, T * D);
        vGm.SetGlobalBuffer((__gm__ half *)v, T * D);
        betaGm.SetGlobalBuffer((__gm__ float *)beta, T);
        gcGm.SetGlobalBuffer((__gm__ float *)gc, T);
        wGm.SetGlobalBuffer((__gm__ half *)w, T * D);
        uGm.SetGlobalBuffer((__gm__ half *)u, T * D);

        pipe.InitBuffer(tBuf, T * T * sizeof(float));
        pipe.InitBuffer(kHalfBuf, T * D * sizeof(half));
        pipe.InitBuffer(vHalfBuf, T * D * sizeof(half));
        pipe.InitBuffer(kBuf, T * D * sizeof(float));
        pipe.InitBuffer(vBuf, T * D * sizeof(float));
        pipe.InitBuffer(betaBuf, T * sizeof(float));
        pipe.InitBuffer(gcBuf, T * sizeof(float));
        pipe.InitBuffer(cbBuf, T * sizeof(float));    // beta * exp(gc)
        pipe.InitBuffer(rowWBuf, D * sizeof(float));
        pipe.InitBuffer(rowUBuf, D * sizeof(float));
        pipe.InitBuffer(rowHBuf, D * sizeof(half));
    }

    __aicore__ inline void Process()
    {
        LocalTensor<float> tL = tBuf.Get<float>();
        LocalTensor<half> kH = kHalfBuf.Get<half>();
        LocalTensor<half> vH = vHalfBuf.Get<half>();
        LocalTensor<float> kF = kBuf.Get<float>();
        LocalTensor<float> vF = vBuf.Get<float>();
        LocalTensor<float> beta = betaBuf.Get<float>();
        LocalTensor<float> gc = gcBuf.Get<float>();
        LocalTensor<float> cb = cbBuf.Get<float>();
        LocalTensor<float> rowW = rowWBuf.Get<float>();
        LocalTensor<float> rowU = rowUBuf.Get<float>();
        LocalTensor<half> rowH = rowHBuf.Get<half>();

        DataCopy(tL, tGm, T * T);
        DataCopy(kH, kGm, T * D);
        DataCopy(vH, vGm, T * D);
        DataCopy(beta, betaGm, T);
        DataCopy(gc, gcGm, T);
        PipeBarrier<PIPE_ALL>();
        Cast(kF, kH, RoundMode::CAST_NONE, T * D);
        Cast(vF, vH, RoundMode::CAST_NONE, T * D);
        PipeBarrier<PIPE_V>();
        // cb[k] = beta[k] * exp(gc[k])
        Exp(cb, gc, T);
        PipeBarrier<PIPE_V>();
        Mul(cb, cb, beta, T);
        PipeBarrier<PIPE_V>();
        AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);

        for (int32_t i = 0; i < T; ++i) {
            Duplicate(rowW, 0.0f, D);
            Duplicate(rowU, 0.0f, D);
            PipeBarrier<PIPE_V>();
            for (int32_t k = 0; k <= i; ++k) {
                float tik = tL.GetValue(i * T + k);
                float bk = beta.GetValue(k);
                float cbk = cb.GetValue(k);
                Axpy(rowW, kF[k * D], tik * cbk, D);
                Axpy(rowU, vF[k * D], tik * bk, D);
            }
            PipeBarrier<PIPE_V>();
            Cast(rowH, rowW, RoundMode::CAST_NONE, D);
            PipeBarrier<PIPE_V>();
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            DataCopy(wGm[i * D], rowH, D);
            PipeBarrier<PIPE_ALL>();
            Cast(rowH, rowU, RoundMode::CAST_NONE, D);
            PipeBarrier<PIPE_V>();
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            DataCopy(uGm[i * D], rowH, D);
            PipeBarrier<PIPE_ALL>();
        }
        PipeBarrier<PIPE_ALL>();
    }

private:
    TPipe pipe;
    TBuf<TPosition::VECCALC> tBuf, kHalfBuf, vHalfBuf, kBuf, vBuf, betaBuf, gcBuf,
        cbBuf, rowWBuf, rowUBuf, rowHBuf;
    GlobalTensor<float> tGm, betaGm, gcGm;
    GlobalTensor<half> kGm, vGm, wGm, uGm;
};

extern "C" __global__ __aicore__ void gdn_wy(GM_ADDR t, GM_ADDR k, GM_ADDR v, GM_ADDR beta,
                                              GM_ADDR gc, GM_ADDR w, GM_ADDR u)
{
    KernelGdnWy op;
    op.Init(t, k, v, beta, gc, w, u);
    op.Process();
}

#ifndef ASCENDC_CPU_DEBUG
void gdn_wy_do(uint32_t blockDim, void *stream, uint8_t *t, uint8_t *k, uint8_t *v,
               uint8_t *beta, uint8_t *gc, uint8_t *w, uint8_t *u)
{
    gdn_wy<<<blockDim, nullptr, stream>>>(t, k, v, beta, gc, w, u);
}
#endif
