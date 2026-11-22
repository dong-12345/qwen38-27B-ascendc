/**
 * gdn_gating — Qwen3.8 (Qwen3.5) Gated DeltaNet 的门控融合算子
 *
 *   g    = -exp(A_log) * softplus(a + dt_bias)      (逐 token、逐 value-head 的对数衰减)
 *   beta = sigmoid(b)                               (delta rule 的写入门控)
 *
 * 布局（为逐 head 连续访存而转置）：
 *   a, b       : [HEADS, SEQ] fp16
 *   dt_bias    : [HEADS] fp32   (mamba_ssm_dtype = float32)
 *   g_scale    : [HEADS] fp32   = -exp(A_log)，host 侧预算
 *   g, beta    : [HEADS, SEQ] fp32
 *
 * 计算全部在 fp32 上做，最后按 fp32 写回（后续 chunk 算子需要 fp32 的 g）。
 */
#include "kernel_operator.h"

using namespace AscendC;

constexpr int32_t SEQ = 64;
constexpr int32_t HEADS = 48;
constexpr int32_t BLOCK_NUM = 8;
constexpr int32_t HEADS_PER_BLOCK = HEADS / BLOCK_NUM;   // 6
constexpr int32_t SMALL_BYTES = 256;

class KernelGdnGating {
public:
    __aicore__ inline KernelGdnGating() {}

    __aicore__ inline void Init(GM_ADDR a, GM_ADDR b, GM_ADDR dt_bias,
                                GM_ADDR g_scale, GM_ADDR g, GM_ADDR beta)
    {
        headStart = static_cast<int32_t>(AscendC::GetBlockIdx()) * HEADS_PER_BLOCK;
        aGm.SetGlobalBuffer((__gm__ half *)a + headStart * SEQ, HEADS_PER_BLOCK * SEQ);
        bGm.SetGlobalBuffer((__gm__ half *)b + headStart * SEQ, HEADS_PER_BLOCK * SEQ);
        gGm.SetGlobalBuffer((__gm__ float *)g + headStart * SEQ, HEADS_PER_BLOCK * SEQ);
        betaGm.SetGlobalBuffer((__gm__ float *)beta + headStart * SEQ, HEADS_PER_BLOCK * SEQ);
        dtGm.SetGlobalBuffer((__gm__ float *)dt_bias, HEADS);
        gsGm.SetGlobalBuffer((__gm__ float *)g_scale, HEADS);

        pipe.InitBuffer(inQueueA, 1, SEQ * sizeof(half));
        pipe.InitBuffer(inQueueB, 1, SEQ * sizeof(half));
        pipe.InitBuffer(smallDt, SMALL_BYTES);
        pipe.InitBuffer(smallGs, SMALL_BYTES);
        pipe.InitBuffer(bufA, SEQ * sizeof(float));
        pipe.InitBuffer(bufB, SEQ * sizeof(float));
        pipe.InitBuffer(bufT, SEQ * sizeof(float));
        pipe.InitBuffer(outQueueG, 1, SEQ * sizeof(float));
        pipe.InitBuffer(outQueueBeta, 1, SEQ * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        // dt_bias / g_scale 很小（48 float），整块搬到 UB 后按 head 取标量
        LocalTensor<float> dtL = smallDt.Get<float>();
        LocalTensor<float> gsL = smallGs.Get<float>();
        DataCopy(dtL, dtGm, HEADS);
        DataCopy(gsL, gsGm, HEADS);
        PipeBarrier<PIPE_ALL>();

        for (int32_t i = 0; i < HEADS_PER_BLOCK; ++i) {
            ComputeHead(i, dtL, gsL);
        }
    }

private:
    __aicore__ inline void ComputeHead(int32_t i, const LocalTensor<float> &dtL,
                                       const LocalTensor<float> &gsL)
    {
        LocalTensor<float> aF = bufA.Get<float>();
        LocalTensor<float> bF = bufB.Get<float>();
        LocalTensor<float> tF = bufT.Get<float>();

        // ---- a -> aF(fp32), b -> bF(fp32) ----
        LocalTensor<half> aH = inQueueA.AllocTensor<half>();
        LocalTensor<half> bH = inQueueB.AllocTensor<half>();
        DataCopy(aH, aGm[i * SEQ], SEQ);
        DataCopy(bH, bGm[i * SEQ], SEQ);
        inQueueA.EnQue(aH);
        inQueueB.EnQue(bH);
        aH = inQueueA.DeQue<half>();
        bH = inQueueB.DeQue<half>();
        Cast(aF, aH, RoundMode::CAST_NONE, SEQ);
        Cast(bF, bH, RoundMode::CAST_NONE, SEQ);
        inQueueA.FreeTensor(aH);
        inQueueB.FreeTensor(bH);

        float dt = dtL.GetValue(headStart + i);
        float gs = gsL.GetValue(headStart + i);

        // ---- g = gs * softplus(a + dt) ----
        Adds(aF, aF, dt, SEQ);              // x = a + dt_bias
        Abs(tF, aF, SEQ);                   // |x|
        Muls(tF, tF, -1.0f, SEQ);           // -|x|
        Exp(tF, tF, SEQ);                   // exp(-|x|)
        Adds(tF, tF, 1.0f, SEQ);            // 1 + exp(-|x|)
        Ln(tF, tF, SEQ);                    // log1p(exp(-|x|))
        Maxs(aF, aF, 0.0f, SEQ);            // max(x, 0)
        Add(tF, tF, aF, SEQ);               // softplus(x) = max(x,0) + log1p(exp(-|x|))
        LocalTensor<float> gO = outQueueG.AllocTensor<float>();
        Muls(gO, tF, gs, SEQ);              // g = -exp(A_log) * softplus
        outQueueG.EnQue(gO);
        gO = outQueueG.DeQue<float>();
        DataCopy(gGm[i * SEQ], gO, SEQ);
        outQueueG.FreeTensor(gO);

        // ---- beta = sigmoid(b) = 1 / (1 + exp(-b)) ----
        LocalTensor<float> betaO = outQueueBeta.AllocTensor<float>();
        Muls(betaO, bF, -1.0f, SEQ);
        Exp(betaO, betaO, SEQ);
        Adds(betaO, betaO, 1.0f, SEQ);
        Reciprocal(betaO, betaO, SEQ);
        outQueueBeta.EnQue(betaO);
        betaO = outQueueBeta.DeQue<float>();
        DataCopy(betaGm[i * SEQ], betaO, SEQ);
        outQueueBeta.FreeTensor(betaO);
    }

private:
    int32_t headStart{0};
    TPipe pipe;
    TQue<TPosition::VECIN, 1> inQueueA, inQueueB;
    TBuf<TPosition::VECCALC> smallDt, smallGs, bufA, bufB, bufT;
    TQue<TPosition::VECOUT, 1> outQueueG, outQueueBeta;
    GlobalTensor<half> aGm, bGm;
    GlobalTensor<float> gGm, betaGm, dtGm, gsGm;
};

extern "C" __global__ __aicore__ void gdn_gating(GM_ADDR a, GM_ADDR b,
                                                 GM_ADDR dt_bias, GM_ADDR g_scale,
                                                 GM_ADDR g, GM_ADDR beta)
{
    KernelGdnGating op;
    op.Init(a, b, dt_bias, g_scale, g, beta);
    op.Process();
}

#ifndef ASCENDC_CPU_DEBUG
void gdn_gating_do(uint32_t blockDim, void *stream, uint8_t *a, uint8_t *b,
                   uint8_t *dt_bias, uint8_t *g_scale, uint8_t *g, uint8_t *beta)
{
    gdn_gating<<<blockDim, nullptr, stream>>>(a, b, dt_bias, g_scale, g, beta);
}
#endif
