/**
 * gdn_rgdr_dbg — 分阶段调试 gdn_rgdr
 * mode 0: 只做 load + 衰减，输出 S_dec
 * mode 1: 再算 bv = v - S·k（未乘 beta），输出 bv 到 sOut 前 128 个 float
 * mode 2: 完整（与 gdn_rgdr 相同），输出 S_out
 */
#include "kernel_operator.h"

using namespace AscendC;

constexpr int32_t KK = 128;
constexpr int32_t VV = 128;

extern "C" __global__ __aicore__ void gdn_rgdr_dbg(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR sIn,
                                                    GM_ADDR sOut, GM_ADDR o, float beta,
                                                    float stateDecay, int32_t mode)
{
    TPipe pipe;
    GlobalTensor<half> qGm, kGm, vGm, oGm;
    GlobalTensor<float> sInGm, sOutGm;
    qGm.SetGlobalBuffer((__gm__ half *)q, KK);
    kGm.SetGlobalBuffer((__gm__ half *)k, KK);
    vGm.SetGlobalBuffer((__gm__ half *)v, VV);
    oGm.SetGlobalBuffer((__gm__ half *)o, VV);
    sInGm.SetGlobalBuffer((__gm__ float *)sIn, KK * VV);
    sOutGm.SetGlobalBuffer((__gm__ float *)sOut, KK * VV);

    TBuf<TPosition::VECCALC> qHBuf, kHBuf, vHBuf, sBuf, kFBuf, bvBuf, oBuf, oHBuf;
    pipe.InitBuffer(qHBuf, KK * sizeof(half));
    pipe.InitBuffer(kHBuf, KK * sizeof(half));
    pipe.InitBuffer(vHBuf, VV * sizeof(half));
    pipe.InitBuffer(sBuf, KK * VV * sizeof(float));
    pipe.InitBuffer(kFBuf, KK * sizeof(float));
    pipe.InitBuffer(bvBuf, VV * sizeof(float));
    pipe.InitBuffer(oBuf, VV * sizeof(float));
    pipe.InitBuffer(oHBuf, VV * sizeof(half));

    LocalTensor<half> qH = qHBuf.Get<half>();
    LocalTensor<half> kH = kHBuf.Get<half>();
    LocalTensor<half> vH = vHBuf.Get<half>();
    LocalTensor<float> sL = sBuf.Get<float>();
    LocalTensor<float> kF = kFBuf.Get<float>();
    LocalTensor<float> bv = bvBuf.Get<float>();
    LocalTensor<float> oRow = oBuf.Get<float>();
    LocalTensor<half> oH = oHBuf.Get<half>();

    DataCopy(qH, qGm, KK);
    DataCopy(kH, kGm, KK);
    DataCopy(vH, vGm, VV);
    DataCopy(sL, sInGm, KK * VV);
    PipeBarrier<PIPE_ALL>();
    Cast(kF, kH, RoundMode::CAST_NONE, KK);
    Cast(bv, vH, RoundMode::CAST_NONE, VV);
    PipeBarrier<PIPE_V>();
    AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);

    // 衰减：分两块做，避免大 calCount
    Muls(sL, sL, stateDecay, 8192);
    Muls(sL[8192], sL[8192], stateDecay, 8192);
    PipeBarrier<PIPE_V>();

    if (mode == 0) {
        PipeBarrier<PIPE_ALL>();
        DataCopy(sOutGm, sL, KK * VV);
        PipeBarrier<PIPE_ALL>();
        return;
    }

    // bv = v - S·k
    for (int32_t d = 0; d < KK; ++d) {
        float kd = kF.GetValue(d);
        Axpy(bv, sL[d * VV], -kd, VV);
    }
    PipeBarrier<PIPE_V>();

    if (mode == 1) {
        PipeBarrier<PIPE_ALL>();
        LocalTensor<float> out = sL;
        Duplicate(out, 0.0f, 8192);
        PipeBarrier<PIPE_V>();
        for (int32_t j = 0; j < VV; ++j) {
            out.SetValue(j, bv.GetValue(j));
        }
        PipeBarrier<PIPE_ALL>();
        DataCopy(sOutGm, out, KK * VV);
        PipeBarrier<PIPE_ALL>();
        return;
    }

    // 完整
    Muls(bv, bv, beta, VV);
    PipeBarrier<PIPE_V>();
    for (int32_t d = 0; d < KK; ++d) {
        float kd = kF.GetValue(d);
        Axpy(sL[d * VV], bv, kd, VV);
    }
    PipeBarrier<PIPE_V>();
    Duplicate(oRow, 0.0f, VV);
    PipeBarrier<PIPE_V>();
    for (int32_t d = 0; d < KK; ++d) {
        float qd = qH.GetValue(d);   // 注意：q 是 half，直接读 half 再转
        Axpy(oRow, sL[d * VV], qd, VV);
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

#ifndef ASCENDC_CPU_DEBUG
void gdn_rgdr_dbg_do(uint32_t b, void *st, uint8_t *q, uint8_t *k, uint8_t *v, uint8_t *sIn,
                     uint8_t *sOut, uint8_t *o, float beta, float decay, int32_t mode)
{
    gdn_rgdr_dbg<<<b, nullptr, st>>>(q, k, v, sIn, sOut, o, beta, decay, mode);
}
#endif
