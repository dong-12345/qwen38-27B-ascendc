/**
 * gdn_l2norm — Qwen3.8 GDN 对 q/k 的逐头 L2 归一化（含 1/sqrt(head_dim) 缩放）
 *
 *   y[s, h, d] = x[s, h, d] * scale / sqrt(sum_d x^2 + eps)
 *
 * 布局：
 *   x, y : [SEQ, DQ] fp16，DQ = num_k_heads * head_k_dim = 16 * 128 = 2048
 *   scale = 1/sqrt(head_k_dim)，eps = 1e-6（与 fla 的 l2norm 一致）
 *
 * 归约采用显式二叉 vector 归约（128 -> 64 -> ... -> 1），避免依赖 ReduceSum 的 work 语义。
 */
#include "kernel_operator.h"

using namespace AscendC;

constexpr int32_t SEQ = 64;
constexpr int32_t DQ = 2048;                       // q: 16 heads * 128
constexpr int32_t HEAD_DIM = 128;
constexpr int32_t BLOCK_NUM = 8;
constexpr int32_t CH = DQ / BLOCK_NUM;             // 256 channels = 2 heads per block
constexpr int32_t NHEADS_LOCAL = CH / HEAD_DIM;    // 2
constexpr float EPS = 1e-6f;

class KernelGdnL2Norm {
public:
    __aicore__ inline KernelGdnL2Norm() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y)
    {
        chStart = static_cast<int32_t>(AscendC::GetBlockIdx()) * CH;
        xGm.SetGlobalBuffer((__gm__ half *)x, SEQ * DQ);
        yGm.SetGlobalBuffer((__gm__ half *)y, SEQ * DQ);
        pipe.InitBuffer(inQueue, 1, HEAD_DIM * sizeof(half));
        pipe.InitBuffer(xfBuf, HEAD_DIM * sizeof(float));
        pipe.InitBuffer(sqBuf, HEAD_DIM * sizeof(float));
        pipe.InitBuffer(outQueue, 1, HEAD_DIM * sizeof(half));
    }

    __aicore__ inline void Process()
    {
        float scale = 1.0f / sqrt(static_cast<float>(HEAD_DIM));
        for (int32_t s = 0; s < SEQ; ++s) {
            for (int32_t h = 0; h < NHEADS_LOCAL; ++h) {
                int32_t off = s * DQ + chStart + h * HEAD_DIM;
                LocalTensor<float> xf = xfBuf.Get<float>();
                LocalTensor<half> xh = inQueue.AllocTensor<half>();
                DataCopy(xh, xGm[off], HEAD_DIM);
                inQueue.EnQue(xh);
                xh = inQueue.DeQue<half>();
                Cast(xf, xh, RoundMode::CAST_NONE, HEAD_DIM);
                inQueue.FreeTensor(xh);

                LocalTensor<float> sq = sqBuf.Get<float>();
                Mul(sq, xf, xf, HEAD_DIM);          // x^2
                PipeBarrier<PIPE_V>();
                // 二叉归约：128->64->32->16->8。局部张量偏移必须 32B 对齐，
                // 因此最后 8 个部分和用标量读回累加（避免 <32B 偏移导致的 UB 非对齐访问）。
                for (int32_t len = HEAD_DIM / 2; len >= 8; len >>= 1) {
                    Add(sq, sq, sq[len], len);
                    PipeBarrier<PIPE_V>();
                }
                AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
                float sum = 0.0f;
                for (int32_t k = 0; k < 8; ++k) {
                    sum += sq.GetValue(k);
                }
                float inv = scale / sqrt(sum + EPS);

                LocalTensor<half> yh = outQueue.AllocTensor<half>();
                LocalTensor<float> tmp = sqBuf.Get<float>();
                Muls(tmp, xf, inv, HEAD_DIM);
                Cast(yh, tmp, RoundMode::CAST_NONE, HEAD_DIM);
                outQueue.EnQue(yh);
                yh = outQueue.DeQue<half>();
                DataCopy(yGm[off], yh, HEAD_DIM);
                outQueue.FreeTensor(yh);
            }
        }
    }

private:
    int32_t chStart{0};
    TPipe pipe;
    TQue<TPosition::VECIN, 1> inQueue;
    TBuf<TPosition::VECCALC> xfBuf, sqBuf;
    TQue<TPosition::VECOUT, 1> outQueue;
    GlobalTensor<half> xGm, yGm;
};

extern "C" __global__ __aicore__ void gdn_l2norm(GM_ADDR x, GM_ADDR y)
{
    KernelGdnL2Norm op;
    op.Init(x, y);
    op.Process();
}

#ifndef ASCENDC_CPU_DEBUG
void gdn_l2norm_do(uint32_t blockDim, void *stream, uint8_t *x, uint8_t *y)
{
    gdn_l2norm<<<blockDim, nullptr, stream>>>(x, y);
}
#endif
