/**
 * gdn_chunk_o — GDN chunk 第五步：chunk 输出
 *
 *   o = (q * exp(gc)) @ S  +  attn @ v_new
 *   attn[i,j] = (q_i · k_j) * exp(gc_i - gc_j)   (j <= i，否则 0)
 *
 * 所有数据先在 UB 里准备好（v_new 一次性转 fp32），内层循环只用 UB，不做 GM 往返。
 */
#include "kernel_operator.h"

using namespace AscendC;

constexpr int32_t C = 64;
constexpr int32_t KD = 128;
constexpr int32_t VD = 128;

class KernelGdnChunkO {
public:
    __aicore__ inline KernelGdnChunkO() {}

    __aicore__ inline void Init(GM_ADDR q, GM_ADDR k, GM_ADDR gc, GM_ADDR s, GM_ADDR vnew,
                                GM_ADDR o)
    {
        qGm.SetGlobalBuffer((__gm__ half *)q, C * KD);
        kGm.SetGlobalBuffer((__gm__ half *)k, C * KD);
        gcGm.SetGlobalBuffer((__gm__ float *)gc, C);
        sGm.SetGlobalBuffer((__gm__ float *)s, KD * VD);
        vnGm.SetGlobalBuffer((__gm__ half *)vnew, C * VD);
        oGm.SetGlobalBuffer((__gm__ half *)o, C * VD);

        pipe.InitBuffer(qHalfBuf, C * KD * sizeof(half));
        pipe.InitBuffer(kHalfBuf, C * KD * sizeof(half));
        pipe.InitBuffer(ktBuf, KD * C * sizeof(float));
        pipe.InitBuffer(gcBuf, C * sizeof(float));
        pipe.InitBuffer(epBuf, C * sizeof(float));
        pipe.InitBuffer(ekBuf, C * sizeof(float));
        pipe.InitBuffer(epRawBuf, C * sizeof(float));
        pipe.InitBuffer(sBuf, KD * VD * sizeof(float));
        pipe.InitBuffer(vnHalfBuf, C * VD * sizeof(half));
        pipe.InitBuffer(vnFBuf, C * VD * sizeof(float));
        pipe.InitBuffer(oRowBuf, VD * sizeof(float));
        pipe.InitBuffer(aRowBuf, C * sizeof(float));
        pipe.InitBuffer(hRowBuf, VD * sizeof(half));
    }

    __aicore__ inline void Process()
    {
        LocalTensor<half> qH = qHalfBuf.Get<half>();
        LocalTensor<half> kH = kHalfBuf.Get<half>();
        LocalTensor<float> kt = ktBuf.Get<float>();
        LocalTensor<float> gc = gcBuf.Get<float>();
        LocalTensor<float> ep = epBuf.Get<float>();
        LocalTensor<float> epRaw = epRawBuf.Get<float>();
        LocalTensor<float> ek = ekBuf.Get<float>();
        LocalTensor<float> sL = sBuf.Get<float>();
        LocalTensor<half> vnH = vnHalfBuf.Get<half>();
        LocalTensor<float> vnF = vnFBuf.Get<float>();
        LocalTensor<float> oRow = oRowBuf.Get<float>();
        LocalTensor<float> aRow = aRowBuf.Get<float>();
        LocalTensor<half> hRow = hRowBuf.Get<half>();

        DataCopy(qH, qGm, C * KD);
        DataCopy(kH, kGm, C * KD);
        DataCopy(gc, gcGm, C);
        DataCopy(sL, sGm, KD * VD);
        DataCopy(vnH, vnGm, C * VD);
        PipeBarrier<PIPE_ALL>();
        Cast(vnF, vnH, RoundMode::CAST_NONE, C * VD);
        PipeBarrier<PIPE_V>();
        AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);

        // ep = exp(gc), ek = exp(-gc)；chunk 内先减最大值防溢出
        float gcMax = -1e30f;
        for (int32_t i = 0; i < C; ++i) {
            float v = gc.GetValue(i);
            if (v > gcMax) gcMax = v;
        }
        // o_inter 需要未归一化的 exp(gc)；attn 用归一化后的 exp(gc - gcMax)（差值不变）
        Exp(epRaw, gc, C);
        Adds(gc, gc, -gcMax, C);
        Exp(ep, gc, C);
        Muls(ek, gc, -1.0f, C);
        Exp(ek, ek, C);
        PipeBarrier<PIPE_V>();
        AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);

        // kt[d, i] = k[i, d]
        for (int32_t i = 0; i < C; ++i) {
            for (int32_t d = 0; d < KD; ++d) {
                kt.SetValue(d * C + i, (float)kH.GetValue(i * KD + d));
            }
        }
        PipeBarrier<PIPE_ALL>();

        for (int32_t i = 0; i < C; ++i) {
            float epi = ep.GetValue(i);
            float epiRaw = epRaw.GetValue(i);
            Duplicate(oRow, 0.0f, VD);
            Duplicate(aRow, 0.0f, C);
            PipeBarrier<PIPE_V>();
            for (int32_t d = 0; d < KD; ++d) {
                float qid = (float)qH.GetValue(i * KD + d);
                Axpy(oRow, sL[d * VD], qid * epiRaw, VD);
                Axpy(aRow, kt[d * C], qid, C);
            }
            PipeBarrier<PIPE_V>();
            Mul(aRow, aRow, ek, C);
            Muls(aRow, aRow, epi, C);
            PipeBarrier<PIPE_V>();
            AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
            for (int32_t j = i + 1; j < C; ++j) {
                aRow.SetValue(j, 0.0f);
            }
            AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
            for (int32_t j = 0; j <= i; ++j) {
                float aij = aRow.GetValue(j);
                Axpy(oRow, vnF[j * VD], aij, VD);
            }
            PipeBarrier<PIPE_V>();
            Cast(hRow, oRow, RoundMode::CAST_NONE, VD);
            PipeBarrier<PIPE_V>();
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            DataCopy(oGm[i * VD], hRow, VD);
            PipeBarrier<PIPE_ALL>();
        }
        PipeBarrier<PIPE_ALL>();
    }

private:
    TPipe pipe;
    TBuf<TPosition::VECCALC> qHalfBuf, kHalfBuf, ktBuf, gcBuf, epBuf, ekBuf, epRawBuf, sBuf,
        vnHalfBuf, vnFBuf, oRowBuf, aRowBuf, hRowBuf;
    GlobalTensor<half> qGm, kGm, vnGm, oGm;
    GlobalTensor<float> gcGm, sGm;
};

extern "C" __global__ __aicore__ void gdn_chunk_o(GM_ADDR q, GM_ADDR k, GM_ADDR gc, GM_ADDR s,
                                                   GM_ADDR vnew, GM_ADDR o)
{
    KernelGdnChunkO op;
    op.Init(q, k, gc, s, vnew, o);
    op.Process();
}

#ifndef ASCENDC_CPU_DEBUG
void gdn_chunk_o_do(uint32_t blockDim, void *stream, uint8_t *q, uint8_t *k, uint8_t *gc,
                    uint8_t *s, uint8_t *vnew, uint8_t *o)
{
    gdn_chunk_o<<<blockDim, nullptr, stream>>>(q, k, gc, s, vnew, o);
}
#endif
