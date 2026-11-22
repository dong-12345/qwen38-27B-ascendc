/**
 * gdn_conv1d_silu — Qwen3.8 GDN 的短卷积（depthwise causal conv1d, kernel=4）+ SiLU
 *
 *   y[s, d] = silu( sum_{j=0..3} w[j, d] * x[s-3+j, d] ),   x[<0, :] = 0
 *
 * 布局（token-major，便于后续 l2norm / chunk 算子）：
 *   x      : [SEQ, D] fp16
 *   w_tap  : [KERNEL, D] fp32   （host 侧把 [D, KERNEL] 转置成按 tap 连续）
 *   y      : [SEQ, D] fp16
 *
 * 本版本只做 prefill（无卷积状态）；decode 版会复用同一 kernel 加 state 分支。
 */
#include "kernel_operator.h"

using namespace AscendC;

constexpr int32_t SEQ = 64;
constexpr int32_t D = 10240;                       // q 2048 + k 2048 + v 6144 (Qwen3.8-27B)
constexpr int32_t KERNEL = 4;
constexpr int32_t BLOCK_NUM = 8;
constexpr int32_t CH = D / BLOCK_NUM;              // 1280 channels per block

class KernelGdnConv1d {
public:
    __aicore__ inline KernelGdnConv1d() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR w_tap, GM_ADDR y)
    {
        chStart = static_cast<int32_t>(AscendC::GetBlockIdx()) * CH;
        xGm.SetGlobalBuffer((__gm__ half *)x, SEQ * D);
        yGm.SetGlobalBuffer((__gm__ half *)y, SEQ * D);
        wGm.SetGlobalBuffer((__gm__ float *)w_tap, KERNEL * D);

        pipe.InitBuffer(inQueue, 1, KERNEL * CH * sizeof(half));
        pipe.InitBuffer(rowF, KERNEL * CH * sizeof(float));
        pipe.InitBuffer(accBuf, CH * sizeof(float));
        pipe.InitBuffer(tmpBuf, CH * sizeof(float));
        pipe.InitBuffer(wBuf, KERNEL * CH * sizeof(float));
        pipe.InitBuffer(outQueue, 1, CH * sizeof(half));
    }

    __aicore__ inline void Process()
    {
        LocalTensor<float> wL = wBuf.Get<float>();
        for (int32_t j = 0; j < KERNEL; ++j) {
            DataCopy(wL[j * CH], wGm[j * D + chStart], CH);
        }
        PipeBarrier<PIPE_ALL>();

        LocalTensor<float> rowL = rowF.Get<float>();
        for (int32_t s = 0; s < SEQ; ++s) {
            LocalTensor<half> xL = inQueue.AllocTensor<half>();
            for (int32_t j = 0; j < KERNEL; ++j) {
                int32_t src = s - (KERNEL - 1) + j;
                if (src < 0) {
                    Duplicate(xL[j * CH], (half)0.0, CH);
                } else {
                    DataCopy(xL[j * CH], xGm[src * D + chStart], CH);
                }
            }
            inQueue.EnQue(xL);
            xL = inQueue.DeQue<half>();
            for (int32_t j = 0; j < KERNEL; ++j) {
                Cast(rowL[j * CH], xL[j * CH], RoundMode::CAST_NONE, CH);
            }
            inQueue.FreeTensor(xL);

            LocalTensor<float> acc = accBuf.Get<float>();
            LocalTensor<float> tmp = tmpBuf.Get<float>();
            Mul(acc, rowL, wL, CH);
            for (int32_t j = 1; j < KERNEL; ++j) {
                Mul(tmp, rowL[j * CH], wL[j * CH], CH);
                Add(acc, acc, tmp, CH);
            }
            // silu(acc) = acc * sigmoid(acc)
            Muls(tmp, acc, -1.0f, CH);
            Exp(tmp, tmp, CH);
            Adds(tmp, tmp, 1.0f, CH);
            Reciprocal(tmp, tmp, CH);
            Mul(acc, acc, tmp, CH);

            LocalTensor<half> yL = outQueue.AllocTensor<half>();
            Cast(yL, acc, RoundMode::CAST_NONE, CH);
            outQueue.EnQue(yL);
            yL = outQueue.DeQue<half>();
            DataCopy(yGm[s * D + chStart], yL, CH);
            outQueue.FreeTensor(yL);
        }
    }

private:
    int32_t chStart{0};
    TPipe pipe;
    TQue<TPosition::VECIN, 1> inQueue;
    TBuf<TPosition::VECCALC> rowF, accBuf, tmpBuf, wBuf;
    TQue<TPosition::VECOUT, 1> outQueue;
    GlobalTensor<half> xGm, yGm;
    GlobalTensor<float> wGm;
};

extern "C" __global__ __aicore__ void gdn_conv1d(GM_ADDR x, GM_ADDR w_tap, GM_ADDR y)
{
    KernelGdnConv1d op;
    op.Init(x, w_tap, y);
    op.Process();
}

#ifndef ASCENDC_CPU_DEBUG
void gdn_conv1d_do(uint32_t blockDim, void *stream, uint8_t *x, uint8_t *w_tap, uint8_t *y)
{
    gdn_conv1d<<<blockDim, nullptr, stream>>>(x, w_tap, y);
}
#endif
