/**
 * gdn_rgdr — GDN decode 单 token 递推（recurrent gated delta rule）
 *
 *   S = S * exp(g)
 *   b_v = (v - S·k) * beta
 *   S = S + k ⊗ b_v
 *   o = q · S
 *
 * 单 head：K=V=128，S 常驻 UB（128×128 fp32 = 64KB）。
 * beta / g 用 kernel 标量参数传入（GM 上几个字节的 DataCopy 对齐不合法）。
 */
#include "kernel_operator.h"

using namespace AscendC;

constexpr int32_t KK = 128;
constexpr int32_t VV = 128;

class KernelGdnRgdr {
public:
    __aicore__ inline KernelGdnRgdr() {}

    __aicore__ inline void Init(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR sIn, GM_ADDR sOut,
                                GM_ADDR o)
    {
        qGm.SetGlobalBuffer((__gm__ half *)q, KK);
        kGm.SetGlobalBuffer((__gm__ half *)k, KK);
        vGm.SetGlobalBuffer((__gm__ half *)v, VV);
        sInGm.SetGlobalBuffer((__gm__ float *)sIn, KK * VV);
        sOutGm.SetGlobalBuffer((__gm__ float *)sOut, KK * VV);
        oGm.SetGlobalBuffer((__gm__ half *)o, VV);

        pipe.InitBuffer(qHBuf, KK * sizeof(half));
        pipe.InitBuffer(kHBuf, KK * sizeof(half));
        pipe.InitBuffer(vHBuf, VV * sizeof(half));
        pipe.InitBuffer(sBuf, KK * VV * sizeof(float));
        pipe.InitBuffer(qFBuf, KK * sizeof(float));
        pipe.InitBuffer(kFBuf, KK * sizeof(float));
        pipe.InitBuffer(bvBuf, VV * sizeof(float));
        pipe.InitBuffer(oBuf, VV * sizeof(float));
        pipe.InitBuffer(oHBuf, VV * sizeof(half));
        pipe.InitBuffer(gsBuf, 32);
    }

    __aicore__ inline void Process(float betaVal, float stateDecay)
    {
        LocalTensor<half> qH = qHBuf.Get<half>();
        LocalTensor<half> kH = kHBuf.Get<half>();
        LocalTensor<half> vH = vHBuf.Get<half>();
        LocalTensor<float> sL = sBuf.Get<float>();
        LocalTensor<float> qF = qFBuf.Get<float>();
        LocalTensor<float> kF = kFBuf.Get<float>();
        LocalTensor<float> bv = bvBuf.Get<float>();
        LocalTensor<float> oRow = oBuf.Get<float>();
        LocalTensor<half> oH = oHBuf.Get<half>();

        DataCopy(qH, qGm, KK);
        DataCopy(kH, kGm, KK);
        DataCopy(vH, vGm, VV);
        DataCopy(sL, sInGm, KK * VV);
        PipeBarrier<PIPE_ALL>();
        Cast(qF, qH, RoundMode::CAST_NONE, KK);
        Cast(kF, kH, RoundMode::CAST_NONE, KK);
        Cast(bv, vH, RoundMode::CAST_NONE, VV);
        PipeBarrier<PIPE_V>();
        AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);


        Muls(sL, sL, stateDecay, KK * VV);
        PipeBarrier<PIPE_V>();
        // b_v = v - S·k
        for (int32_t d = 0; d < KK; ++d) {
            Axpy(bv, sL[d * VV], -kF.GetValue(d), VV);
        }
        PipeBarrier<PIPE_V>();
        Muls(bv, bv, betaVal, VV);
        PipeBarrier<PIPE_V>();
        // S += k ⊗ b_v
        for (int32_t d = 0; d < KK; ++d) {
            Axpy(sL[d * VV], bv, kF.GetValue(d), VV);
        }
        PipeBarrier<PIPE_V>();
        // o = q · S
        Duplicate(oRow, 0.0f, VV);
        PipeBarrier<PIPE_V>();
        for (int32_t d = 0; d < KK; ++d) {
            Axpy(oRow, sL[d * VV], qF.GetValue(d), VV);
        }
        PipeBarrier<PIPE_V>();
        Cast(oH, oRow, RoundMode::CAST_NONE, VV);
        PipeBarrier<PIPE_V>();
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        DataCopy(oGm, oH, VV);
        PipeBarrier<PIPE_ALL>();
        DataCopy(sOutGm, sL, KK * VV);
        PipeBarrier<PIPE_ALL>();
    }

private:
    TPipe pipe;
    TBuf<TPosition::VECCALC> qHBuf, kHBuf, vHBuf, sBuf, qFBuf, kFBuf, bvBuf, oBuf,
        oHBuf, gsBuf;
    GlobalTensor<half> qGm, kGm, vGm, oGm;
    GlobalTensor<float> sInGm, sOutGm;
};

extern "C" __global__ __aicore__ void gdn_rgdr(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR sIn,
                                                GM_ADDR sOut, GM_ADDR o, float beta, float g)
{
    KernelGdnRgdr op;
    op.Init(q, k, v, sIn, sOut, o);
    op.Process(beta, g);
}

#ifndef ASCENDC_CPU_DEBUG
void gdn_rgdr_do(uint32_t blockDim, void *stream, uint8_t *q, uint8_t *k, uint8_t *v,
                 uint8_t *sIn, uint8_t *sOut, uint8_t *o, float beta, float g)
{
    gdn_rgdr<<<blockDim, nullptr, stream>>>(q, k, v, sIn, sOut, o, beta, g);
}
#endif
