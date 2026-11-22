/**
 * gdn_gated_rmsnorm — Qwen3.8 GDN 的 RMSNormGated
 *
 *   y[s, h, d] = x[s, h, d] / sqrt(mean_d(x^2) + eps) * w[d] * silu(z[s, h, d])
 *
 * 布局：
 *   x, z, y : [SEQ, DV] fp16，DV = num_v_heads * head_v_dim = 48 * 128 = 6144
 *   w       : [head_v_dim] fp32（每头共享同一组 gamma）
 *   eps     : 1e-6
 *
 * 归约同 gdn_l2norm：显式二叉 vector 归约，尾部 8 个部分和标量累加（32B 对齐约束）。
 */
#include "kernel_operator.h"

using namespace AscendC;

constexpr int32_t SEQ = 64;
constexpr int32_t DV = 6144;                       // 48 heads * 128
constexpr int32_t HEAD_DIM = 128;
constexpr int32_t BLOCK_NUM = 8;
constexpr int32_t CH = DV / BLOCK_NUM;             // 768 = 6 heads
constexpr int32_t NHEADS_LOCAL = CH / HEAD_DIM;    // 6
constexpr float EPS = 1e-6f;

class KernelGdnGatedRmsNorm {
public:
    __aicore__ inline KernelGdnGatedRmsNorm() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR z, GM_ADDR w, GM_ADDR y)
    {
        chStart = static_cast<int32_t>(AscendC::GetBlockIdx()) * CH;
        xGm.SetGlobalBuffer((__gm__ half *)x, SEQ * DV);
        zGm.SetGlobalBuffer((__gm__ half *)z, SEQ * DV);
        yGm.SetGlobalBuffer((__gm__ half *)y, SEQ * DV);
        wGm.SetGlobalBuffer((__gm__ float *)w, HEAD_DIM);

        pipe.InitBuffer(inQueueX, 1, HEAD_DIM * sizeof(half));
        pipe.InitBuffer(inQueueZ, 1, HEAD_DIM * sizeof(half));
        pipe.InitBuffer(wBuf, HEAD_DIM * sizeof(float));
        pipe.InitBuffer(xfBuf, HEAD_DIM * sizeof(float));
        pipe.InitBuffer(zfBuf, HEAD_DIM * sizeof(float));
        pipe.InitBuffer(sqBuf, HEAD_DIM * sizeof(float));
        pipe.InitBuffer(tmpBuf, HEAD_DIM * sizeof(float));
        pipe.InitBuffer(outQueue, 1, HEAD_DIM * sizeof(half));
    }

    __aicore__ inline void Process()
    {
        LocalTensor<float> wL = wBuf.Get<float>();
        DataCopy(wL, wGm, HEAD_DIM);
        PipeBarrier<PIPE_ALL>();

        LocalTensor<float> xf = xfBuf.Get<float>();
        LocalTensor<float> zf = zfBuf.Get<float>();
        LocalTensor<float> sq = sqBuf.Get<float>();
        LocalTensor<float> tmp = tmpBuf.Get<float>();
        LocalTensor<float> wv = wBuf.Get<float>();

        for (int32_t s = 0; s < SEQ; ++s) {
            for (int32_t h = 0; h < NHEADS_LOCAL; ++h) {
                int32_t off = s * DV + chStart + h * HEAD_DIM;

                LocalTensor<half> xh = inQueueX.AllocTensor<half>();
                LocalTensor<half> zh = inQueueZ.AllocTensor<half>();
                DataCopy(xh, xGm[off], HEAD_DIM);
                DataCopy(zh, zGm[off], HEAD_DIM);
                inQueueX.EnQue(xh);
                inQueueZ.EnQue(zh);
                xh = inQueueX.DeQue<half>();
                zh = inQueueZ.DeQue<half>();
                Cast(xf, xh, RoundMode::CAST_NONE, HEAD_DIM);
                Cast(zf, zh, RoundMode::CAST_NONE, HEAD_DIM);
                inQueueX.FreeTensor(xh);
                inQueueZ.FreeTensor(zh);

                // rms
                Mul(sq, xf, xf, HEAD_DIM);
                PipeBarrier<PIPE_V>();
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
                float inv = 1.0f / sqrt(sum / static_cast<float>(HEAD_DIM) + EPS);

                // silu(z) = z * sigmoid(z)
                Muls(tmp, zf, -1.0f, HEAD_DIM);
                Exp(tmp, tmp, HEAD_DIM);
                Adds(tmp, tmp, 1.0f, HEAD_DIM);
                Reciprocal(tmp, tmp, HEAD_DIM);
                Mul(tmp, tmp, zf, HEAD_DIM);              // silu(z)

                // y = x * inv * w * silu(z)
                Muls(xf, xf, inv, HEAD_DIM);
                Mul(xf, xf, wv, HEAD_DIM);
                Mul(xf, xf, tmp, HEAD_DIM);

                LocalTensor<half> yh = outQueue.AllocTensor<half>();
                Cast(yh, xf, RoundMode::CAST_NONE, HEAD_DIM);
                outQueue.EnQue(yh);
                yh = outQueue.DeQue<half>();
                DataCopy(yGm[off], yh, HEAD_DIM);
                outQueue.FreeTensor(yh);
                PipeBarrier<PIPE_ALL>();
            }
        }
    }

private:
    int32_t chStart{0};
    TPipe pipe;
    TQue<TPosition::VECIN, 1> inQueueX, inQueueZ;
    TBuf<TPosition::VECCALC> wBuf, xfBuf, zfBuf, sqBuf, tmpBuf;
    TQue<TPosition::VECOUT, 1> outQueue;
    GlobalTensor<half> xGm, zGm, yGm;
    GlobalTensor<float> wGm;
};

extern "C" __global__ __aicore__ void gdn_gated_rmsnorm(GM_ADDR x, GM_ADDR z, GM_ADDR w, GM_ADDR y)
{
    KernelGdnGatedRmsNorm op;
    op.Init(x, z, w, y);
    op.Process();
}

#ifndef ASCENDC_CPU_DEBUG
void gdn_gated_rmsnorm_do(uint32_t blockDim, void *stream, uint8_t *x, uint8_t *z,
                          uint8_t *w, uint8_t *y)
{
    gdn_gated_rmsnorm<<<blockDim, nullptr, stream>>>(x, z, w, y);
}
#endif
