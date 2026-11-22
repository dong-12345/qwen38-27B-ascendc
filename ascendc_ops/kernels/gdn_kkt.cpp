/**
 * gdn_kkt — GDN chunk 核心第一步：scaled_dot_kkt
 *
 *   S[i,j] = K_i · K_j                       (chunk 内 64 个 token，head_dim=128)
 *   A[i,j] = beta_i * S[i,j] * exp(gc_i - gc_j)   (j <= i，否则 0)
 *
 * 实现（向量核，避免通用 GEMM）：
 *   先把 K 转置成 Kt[D, T]，然后对每个 i 用「秩-1 累加」求 S 的第 i 行：
 *     row = Σ_d K[i,d] * Kt[d, :]        -> Axpy(row, Kt[d], K[i,d], T)
 *   再乘以预计算的 exp(-gc_j) 与 beta_i*exp(gc_i)，最后把 j>i 置 0。
 *
 * 数值安全：chunk 内先把 gc 减去 chunk 最大值（只影响相对差值），避免 exp(-gc) 溢出/下溢。
 */
#include "kernel_operator.h"

using namespace AscendC;

constexpr int32_t T = 64;        // chunk size (tokens)
constexpr int32_t D = 128;       // head_dim
constexpr int32_t NHEAD = 1;     // 本版本单 head（后续循环多个 head）

class KernelGdnKkt {
public:
    __aicore__ inline KernelGdnKkt() {}

    __aicore__ inline void Init(GM_ADDR k, GM_ADDR gc, GM_ADDR beta, GM_ADDR a)
    {
        kGm.SetGlobalBuffer((__gm__ half *)k, NHEAD * T * D);
        gcGm.SetGlobalBuffer((__gm__ float *)gc, T);
        betaGm.SetGlobalBuffer((__gm__ float *)beta, T);
        aGm.SetGlobalBuffer((__gm__ float *)a, T * T);

        pipe.InitBuffer(kHalfBuf, T * D * sizeof(half));
        pipe.InitBuffer(kBuf, T * D * sizeof(float));       // K  [T, D] fp32
        pipe.InitBuffer(ktBuf, D * T * sizeof(float));      // Kt [D, T] fp32
        pipe.InitBuffer(gcBuf, T * sizeof(float));
        pipe.InitBuffer(epBuf, T * sizeof(float));          // exp(gc)
        pipe.InitBuffer(ekBuf, T * sizeof(float));          // exp(-gc)
        pipe.InitBuffer(betaBuf, T * sizeof(float));
        pipe.InitBuffer(rowBuf, T * sizeof(float));
        pipe.InitBuffer(tmpBuf, T * sizeof(float));
        pipe.InitBuffer(outBuf, T * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        LocalTensor<half> kH = kHalfBuf.Get<half>();
        LocalTensor<float> kF = kBuf.Get<float>();
        LocalTensor<float> kt = ktBuf.Get<float>();
        LocalTensor<float> gc = gcBuf.Get<float>();
        LocalTensor<float> ep = epBuf.Get<float>();
        LocalTensor<float> ek = ekBuf.Get<float>();
        LocalTensor<float> beta = betaBuf.Get<float>();
        LocalTensor<float> row = rowBuf.Get<float>();
        LocalTensor<float> tmp = tmpBuf.Get<float>();
        LocalTensor<float> out = outBuf.Get<float>();

        DataCopy(kH, kGm, T * D);
        DataCopy(gc, gcGm, T);
        DataCopy(beta, betaGm, T);
        PipeBarrier<PIPE_ALL>();
        Cast(kF, kH, RoundMode::CAST_NONE, T * D);
        PipeBarrier<PIPE_V>();
        AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);

        // chunk 内 gc 归一化（减去最大值），保证 exp 不溢出
        float gcMax = -1e30f;
        for (int32_t i = 0; i < T; ++i) {
            float v = gc.GetValue(i);
            if (v > gcMax) gcMax = v;
        }
        Muls(gc, gc, 1.0f, T);
        Adds(gc, gc, -gcMax, T);
        Exp(ep, gc, T);
        Muls(tmp, gc, -1.0f, T);
        Exp(ek, tmp, T);
        PipeBarrier<PIPE_V>();
        AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);

        // 转置 K -> Kt（标量，8192 次；此处只做正确性，后续可换向量 Transpose）
        for (int32_t i = 0; i < T; ++i) {
            for (int32_t d = 0; d < D; ++d) {
                kt.SetValue(d * T + i, kF.GetValue(i * D + d));
            }
        }
        PipeBarrier<PIPE_ALL>();

        for (int32_t i = 0; i < T; ++i) {
            Duplicate(row, 0.0f, T);
            PipeBarrier<PIPE_V>();
            for (int32_t d = 0; d < D; ++d) {
                float kd = kF.GetValue(i * D + d);
                Axpy(row, kt[d * T], kd, T);
            }
            // A[i,:] = beta_i*exp(gc_i) * (S[i,:] * exp(-gc_j))，并把 j>i 置 0
            Mul(row, row, ek, T);
            float bi = beta.GetValue(i);
            float epi = ep.GetValue(i);
            Muls(row, row, bi * epi, T);
            PipeBarrier<PIPE_V>();
            AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
            for (int32_t j = 0; j < T; ++j) {
                row.SetValue(j, (j <= i) ? row.GetValue(j) : 0.0f);
            }
            PipeBarrier<PIPE_ALL>();
            DataCopy(out, row, T);
            PipeBarrier<PIPE_ALL>();
            DataCopy(aGm[i * T], out, T);
            PipeBarrier<PIPE_ALL>();
        }
        PipeBarrier<PIPE_ALL>();
    }

private:
    TPipe pipe;
    TBuf<TPosition::VECCALC> kHalfBuf, kBuf, ktBuf, gcBuf, epBuf, ekBuf, betaBuf,
        rowBuf, tmpBuf, outBuf;
    GlobalTensor<half> kGm;
    GlobalTensor<float> gcGm, betaGm, aGm;
};

extern "C" __global__ __aicore__ void gdn_kkt(GM_ADDR k, GM_ADDR gc, GM_ADDR beta, GM_ADDR a)
{
    KernelGdnKkt op;
    op.Init(k, gc, beta, a);
    op.Process();
}

#ifndef ASCENDC_CPU_DEBUG
void gdn_kkt_do(uint32_t blockDim, void *stream, uint8_t *k, uint8_t *gc, uint8_t *beta, uint8_t *a)
{
    gdn_kkt<<<blockDim, nullptr, stream>>>(k, gc, beta, a);
}
#endif
