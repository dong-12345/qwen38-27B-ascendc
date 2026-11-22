/**
 * w4a16_probe — 逐段定位 w4a16_gemv 的 aicore exception（retCode=0x26）
 * mode 0..3 基础指令；mode 4 完整单行；mode 5 完整 256 行循环。
 */
#include "kernel_operator.h"

using namespace AscendC;

constexpr int32_t N = 17408;
constexpr int32_t K = 5120;
constexpr int32_t KQ = K / 4;
constexpr int32_t KWORD = K / 4;
constexpr int32_t CW = 64;
constexpr int32_t NCHUNK = KWORD / CW;    // 20

extern "C" __global__ __aicore__ void probe_g(GM_ADDR act, GM_ADDR wq, GM_ADDR scale,
                                              GM_ADDR out, int32_t mode)
{
    TPipe pipe;
    GlobalTensor<half> actGm, sGm, outGm;
    GlobalTensor<uint16_t> wGm;
    actGm.SetGlobalBuffer((__gm__ half *)act, K);
    wGm.SetGlobalBuffer((__gm__ uint16_t *)wq, N * KWORD);
    sGm.SetGlobalBuffer((__gm__ half *)scale, N);
    outGm.SetGlobalBuffer((__gm__ half *)out, N);

    TBuf<TPosition::VECCALC> phBuf, phFBuf, maskBuf, obuf, actHalfBuf, actFBuf,
        scaleRowBuf, pxBuf, accBuf;
    TQue<TPosition::VECIN, 2> wQ;
    pipe.InitBuffer(phBuf, 4 * CW * sizeof(uint16_t));
    pipe.InitBuffer(phFBuf, 4 * CW * sizeof(float));
    pipe.InitBuffer(maskBuf, CW * sizeof(uint16_t));
    pipe.InitBuffer(obuf, 256 * sizeof(half));
    pipe.InitBuffer(actHalfBuf, K * sizeof(half));
    pipe.InitBuffer(actFBuf, K * sizeof(float));
    pipe.InitBuffer(scaleRowBuf, 256 * sizeof(half));
    pipe.InitBuffer(pxBuf, CW * sizeof(float));
    pipe.InitBuffer(accBuf, CW * sizeof(float));
    pipe.InitBuffer(wQ, 2, KWORD * sizeof(uint16_t));

    LocalTensor<uint16_t> ph = phBuf.Get<uint16_t>();
    LocalTensor<float> phF = phFBuf.Get<float>();
    LocalTensor<uint16_t> mask = maskBuf.Get<uint16_t>();
    LocalTensor<half> ob = obuf.Get<half>();
    LocalTensor<half> actH = actHalfBuf.Get<half>();
    LocalTensor<float> actF = actFBuf.Get<float>();
    LocalTensor<half> scaleRow = scaleRowBuf.Get<half>();
    LocalTensor<float> px = pxBuf.Get<float>();
    LocalTensor<float> acc = accBuf.Get<float>();

    if (mode <= 3) {
        LocalTensor<uint16_t> w = wQ.AllocTensor<uint16_t>();
        DataCopy(w, wGm, KWORD);
        wQ.EnQue(w);
        w = wQ.DeQue<uint16_t>();
        if (mode == 0) {
            DataCopy(outGm, w.ReinterpretCast<half>(), CW);
        } else if (mode == 1) {
            Duplicate(mask, (uint16_t)0x000F, CW);
            PipeBarrier<PIPE_V>();
            And(ph, w, mask, CW);
            PipeBarrier<PIPE_V>();
            DataCopy(ob, ph.ReinterpretCast<half>(), CW);
            DataCopy(outGm, ob, CW);
        } else if (mode == 2) {
            ShiftRight(ph, w, (uint16_t)4, CW);
            PipeBarrier<PIPE_V>();
            DataCopy(ob, ph.ReinterpretCast<half>(), CW);
            DataCopy(outGm, ob, CW);
        } else {
            And(ph, w, mask, CW);
            PipeBarrier<PIPE_V>();
            Cast(phF, ph.ReinterpretCast<int16_t>(), RoundMode::CAST_NONE, CW);
            PipeBarrier<PIPE_V>();
            Cast(ob, phF, RoundMode::CAST_NONE, CW);
            PipeBarrier<PIPE_V>();
            DataCopy(outGm, ob, CW);
        }
        wQ.FreeTensor(w);
        return;
    }

    // ---- mode >= 4：完整一行（先做 1 行，再做 256 行）----
    DataCopy(actH, actGm, K);
    PipeBarrier<PIPE_ALL>();
    Cast(actF, actH, RoundMode::CAST_NONE, K);
    PipeBarrier<PIPE_V>();
    SetFlag<HardEvent::V_S>(EVENT_ID0);
    WaitFlag<HardEvent::V_S>(EVENT_ID0);
    float sumAct = 0.0f;
    for (int32_t k = 0; k < K; ++k) {
        sumAct += actF.GetValue(k);
    }
    Duplicate(mask, (uint16_t)0x000F, CW);
    PipeBarrier<PIPE_V>();

    int32_t nrows = (mode == 4) ? 1 : 256;
    for (int32_t n = 0; n < nrows; ++n) {
        LocalTensor<uint16_t> w = wQ.AllocTensor<uint16_t>();
        DataCopy(w, wGm[n * KWORD], KWORD);
        wQ.EnQue(w);
        w = wQ.DeQue<uint16_t>();

        if ((n & 255) == 0) {
            DataCopy(scaleRow, sGm[n], 256);
            PipeBarrier<PIPE_ALL>();
            SetFlag<HardEvent::V_S>(EVENT_ID0);
            WaitFlag<HardEvent::V_S>(EVENT_ID0);
        }

        Duplicate(acc, 0.0f, CW);
        PipeBarrier<PIPE_V>();
        for (int32_t c = 0; c < NCHUNK; ++c) {
            LocalTensor<uint16_t> wc = w[c * CW];
            And(ph, wc, mask, CW);
            ShiftRight(ph[CW], wc, (uint16_t)4, CW);
            And(ph[CW], ph[CW], mask, CW);
            ShiftRight(ph[2 * CW], wc, (uint16_t)8, CW);
            And(ph[2 * CW], ph[2 * CW], mask, CW);
            ShiftRight(ph[3 * CW], wc, (uint16_t)12, CW);
            PipeBarrier<PIPE_V>();
            Cast(phF, ph.ReinterpretCast<int16_t>(), RoundMode::CAST_NONE, CW);
            Cast(phF[CW], ph[CW].ReinterpretCast<int16_t>(), RoundMode::CAST_NONE, CW);
            Cast(phF[2 * CW], ph[2 * CW].ReinterpretCast<int16_t>(), RoundMode::CAST_NONE, CW);
            Cast(phF[3 * CW], ph[3 * CW].ReinterpretCast<int16_t>(), RoundMode::CAST_NONE, CW);
            PipeBarrier<PIPE_V>();
            for (int32_t p = 0; p < 4; ++p) {
                Mul(px, phF[p * CW], actF[p * KQ + c * CW], CW);
                Add(acc, acc, px, CW);
            }
            PipeBarrier<PIPE_V>();
        }
        for (int32_t len = CW / 2; len >= 8; len >>= 1) {
            Add(acc, acc, acc[len], len);
            PipeBarrier<PIPE_V>();
        }
        SetFlag<HardEvent::V_S>(EVENT_ID0);
        WaitFlag<HardEvent::V_S>(EVENT_ID0);
        float sum = 0.0f;
        for (int32_t j = 0; j < 8; ++j) {
            sum += acc.GetValue(j);
        }
        float scale = (float)scaleRow.GetValue(n & 255);
        ob.SetValue(n & 255, (half)(scale * (sum - 8.0f * sumAct)));
        wQ.FreeTensor(w);
    }
    PipeBarrier<PIPE_ALL>();
    DataCopy(outGm, ob, (nrows < 256) ? nrows : 256);
}

#ifndef ASCENDC_CPU_DEBUG
void probe_g_do(uint32_t b, void *st, uint8_t *a, uint8_t *wq, uint8_t *s, uint8_t *o,
                int32_t mode)
{
    probe_g<<<b, nullptr, st>>>(a, wq, s, o, mode);
}
#endif
