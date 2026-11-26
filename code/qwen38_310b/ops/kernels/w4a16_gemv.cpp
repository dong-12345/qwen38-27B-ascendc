/**
 * w4a16_gemv — W4A16 权重 × fp16 激活的 GEMV（每行单 scale）
 *
 *   out[n] = scale[n] * ( Σ_k q_k * act_k  -  8 * Σ_k act_k )
 *
 * 权重布局（相位交织）：word j 的低→高 4 个半字节 = q[j], q[j+KQ], q[j+2KQ], q[j+3KQ]
 *
 * v5 优化：fp16 计算 + 混合精度归约
 *   - Cast int16 -> half（比 fp32 少一半 UB 流量）
 *   - Mul 在 fp16，归约先降到 320 个元素，再转 fp32 收尾（精度可控）
 */
#include "kernel_operator.h"

using namespace AscendC;

constexpr int32_t K = 5120;
constexpr int32_t N = 17408;
constexpr int32_t KQ = K / 4;                    // 1280
constexpr int32_t KWORD = K / 4;                 // 1280
constexpr int32_t CW = KWORD;
constexpr int32_t RB = 4;
constexpr int32_t RED_H = 320;                   // half 归约到 320 个

class KernelW4A16Gemv {
public:
    __aicore__ inline KernelW4A16Gemv() {}

    __aicore__ inline void Init(GM_ADDR act, GM_ADDR wq, GM_ADDR scale, GM_ADDR out)
    {
        actGm.SetGlobalBuffer((__gm__ half *)act, K);
        wGm.SetGlobalBuffer((__gm__ uint16_t *)wq, N * KWORD);
        sGm.SetGlobalBuffer((__gm__ half *)scale, N);
        outGm.SetGlobalBuffer((__gm__ half *)out, N);

        pipe.InitBuffer(actHalfBuf, K * sizeof(half));
        pipe.InitBuffer(wQueue, 2, RB * KWORD * sizeof(uint16_t));
        pipe.InitBuffer(scaleRowBuf, 256 * sizeof(half));
        pipe.InitBuffer(phBuf, K * sizeof(uint16_t));
        pipe.InitBuffer(accHalfBuf, K * sizeof(half));
        pipe.InitBuffer(accFBuf, RED_H * sizeof(float));
        pipe.InitBuffer(maskBuf, CW * sizeof(uint16_t));
        pipe.InitBuffer(outBuf, 256 * sizeof(half));
    }

    __aicore__ inline void Process(int32_t nrows)
    {
        LocalTensor<half> actH = actHalfBuf.Get<half>();
        DataCopy(actH, actGm, K);
        PipeBarrier<PIPE_ALL>();
        AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
        float sumAct = 0.0f;
        for (int32_t k = 0; k < K; ++k) {
            sumAct += (float)actH.GetValue(k);
        }

        LocalTensor<uint16_t> mask = maskBuf.Get<uint16_t>();
        Duplicate(mask, (uint16_t)0x000F, CW);
        PipeBarrier<PIPE_V>();

        LocalTensor<uint16_t> ph = phBuf.Get<uint16_t>();
        LocalTensor<half> accH = accHalfBuf.Get<half>();
        LocalTensor<float> accF = accFBuf.Get<float>();
        LocalTensor<half> oBuf = outBuf.Get<half>();
        LocalTensor<half> scaleRow = scaleRowBuf.Get<half>();

        int32_t n = 0;
        while (n < nrows) {
            int32_t batch = (nrows - n >= 256) ? 256 : (nrows - n);
            DataCopy(scaleRow, sGm[n], 256);
            PipeBarrier<PIPE_ALL>();
            AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
            int32_t batchBase = n;

            for (int32_t b = 0; b < batch; b += RB) {
                int32_t rows = (batch - b >= RB) ? RB : (batch - b);
                LocalTensor<uint16_t> wb = wQueue.AllocTensor<uint16_t>();
                DataCopy(wb, wGm[n * KWORD], rows * KWORD);
                wQueue.EnQue(wb);
                wb = wQueue.DeQue<uint16_t>();

                for (int32_t r = 0; r < rows; ++r) {
                    LocalTensor<uint16_t> w = wb[r * KWORD];

                    // 4 相位抽取到连续 K 个 int16
                    And(ph, w, mask, CW);
                    ShiftRight(ph[CW], w, (uint16_t)4, CW);
                    And(ph[CW], ph[CW], mask, CW);
                    ShiftRight(ph[2 * CW], w, (uint16_t)8, CW);
                    And(ph[2 * CW], ph[2 * CW], mask, CW);
                    ShiftRight(ph[3 * CW], w, (uint16_t)12, CW);
                    Cast(accH, ph.ReinterpretCast<int16_t>(), RoundMode::CAST_NONE, K);
                    Mul(accH, accH, actH, K);
                    // half 归约到 RED_H
                    for (int32_t len = K / 2; len >= RED_H; len >>= 1) {
                        Add(accH, accH, accH[len], len);
                    }
                    // 320 -> fp32（一次向量 Cast，避免标量循环）
                    Cast(accF, accH, RoundMode::CAST_NONE, RED_H);
                    for (int32_t len = RED_H / 2; len >= 40; len >>= 1) {
                        Add(accF, accF, accF[len], len);
                    }
                    for (int32_t off = 32; off >= 8; off -= 8) {
                        Add(accF, accF, accF[off], 8);
                    }
                    AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
                    AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
                    float sum = 0.0f;
                    for (int32_t j = 0; j < 8; ++j) {
                        sum += accF.GetValue(j);
                    }
                    float scale = (float)scaleRow.GetValue(b + r);
                    oBuf.SetValue(b + r, (half)(scale * (sum - 8.0f * sumAct)));
                }
                n += rows;
                wQueue.FreeTensor(wb);
            }
            PipeBarrier<PIPE_ALL>();
            DataCopy(outGm[batchBase], oBuf, batch);
            PipeBarrier<PIPE_ALL>();
        }
    }

private:
    TPipe pipe;
    TQue<TPosition::VECIN, 2> wQueue;
    TBuf<TPosition::VECCALC> actHalfBuf, scaleRowBuf, phBuf, accHalfBuf, accFBuf,
        maskBuf, outBuf;
    GlobalTensor<half> actGm, sGm, outGm;
    GlobalTensor<uint16_t> wGm;
};

extern "C" __global__ __aicore__ void w4a16_gemv(GM_ADDR act, GM_ADDR wq, GM_ADDR scale,
                                                  GM_ADDR out, int32_t nrows)
{
    KernelW4A16Gemv op;
    op.Init(act, wq, scale, out);
    op.Process(nrows);
}

#ifndef ASCENDC_CPU_DEBUG
void w4a16_gemv_do(uint32_t blockDim, void *stream, uint8_t *act, uint8_t *wq,
                   uint8_t *scale, uint8_t *out, int32_t nrows)
{
    w4a16_gemv<<<blockDim, nullptr, stream>>>(act, wq, scale, out, nrows);
}
#endif
