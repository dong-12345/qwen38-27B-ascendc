/**
 * gdn_chunk_h — GDN chunk 第四步：chunk 内状态更新
 *
 *   v_new = u - w @ S                     [C, V]
 *   S_new = exp(gc_last) * S + (k * exp(gc_last - gc))^T @ v_new   [K, V]
 *
 * C=64, K=V=128。S 放在 UB（128x128 fp32 = 64KB）。
 * 步 1：逐行求 v_new（每行 64 次 Axpy），先写 GM（后面 outer product 再读回）。
 * 步 2：S 乘衰减后，逐 token 做 rank-1 更新（每 token 128 次 Axpy）。
 */
#include "kernel_operator.h"

using namespace AscendC;

constexpr int32_t C = 64;       // chunk tokens
constexpr int32_t KD = 128;     // key head dim
constexpr int32_t VD = 128;     // value head dim

class KernelGdnChunkH {
public:
    __aicore__ inline KernelGdnChunkH() {}

    __aicore__ inline void Init(GM_ADDR w, GM_ADDR u, GM_ADDR k, GM_ADDR gc, GM_ADDR sIn,
                                GM_ADDR sOut, GM_ADDR vnew)
    {
        wGm.SetGlobalBuffer((__gm__ half *)w, C * KD);
        uGm.SetGlobalBuffer((__gm__ half *)u, C * VD);
        kGm.SetGlobalBuffer((__gm__ half *)k, C * KD);
        gcGm.SetGlobalBuffer((__gm__ float *)gc, C);
        sInGm.SetGlobalBuffer((__gm__ float *)sIn, KD * VD);
        sOutGm.SetGlobalBuffer((__gm__ float *)sOut, KD * VD);
        vnGm.SetGlobalBuffer((__gm__ half *)vnew, C * VD);

        pipe.InitBuffer(wHalfBuf, C * KD * sizeof(half));
        pipe.InitBuffer(uHalfBuf, C * VD * sizeof(half));
        pipe.InitBuffer(kHalfBuf, C * KD * sizeof(half));
        pipe.InitBuffer(gcBuf, C * sizeof(float));
        pipe.InitBuffer(sBuf, KD * VD * sizeof(float));
        pipe.InitBuffer(rowBuf, VD * sizeof(float));
        pipe.InitBuffer(rowHBuf, VD * sizeof(half));
        pipe.InitBuffer(decBuf, C * sizeof(float));    // exp(gc_last - gc_i)
    }

    __aicore__ inline void Process()
    {
        LocalTensor<half> wH = wHalfBuf.Get<half>();
        LocalTensor<half> uH = uHalfBuf.Get<half>();
        LocalTensor<half> kH = kHalfBuf.Get<half>();
        LocalTensor<float> gc = gcBuf.Get<float>();
        LocalTensor<float> sL = sBuf.Get<float>();
        LocalTensor<float> row = rowBuf.Get<float>();
        LocalTensor<half> rowH = rowHBuf.Get<half>();
        LocalTensor<float> dec = decBuf.Get<float>();

        DataCopy(wH, wGm, C * KD);
        DataCopy(uH, uGm, C * VD);
        DataCopy(kH, kGm, C * KD);
        DataCopy(gc, gcGm, C);
        DataCopy(sL, sInGm, KD * VD);
        PipeBarrier<PIPE_ALL>();
        AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);

        float gcLast = gc.GetValue(C - 1);

        // ---------- 步 1：v_new = u - w @ S ----------
        for (int32_t i = 0; i < C; ++i) {
            // row = u[i,:] (fp16 -> fp32)，用 Cast 到 row 缓冲
            Cast(row, uH[i * VD], RoundMode::CAST_NONE, VD);
            PipeBarrier<PIPE_V>();
            for (int32_t kk = 0; kk < KD; ++kk) {
                float coeff = -(float)wH.GetValue(i * KD + kk);
                Axpy(row, sL[kk * VD], coeff, VD);
            }
            PipeBarrier<PIPE_V>();
            Cast(rowH, row, RoundMode::CAST_NONE, VD);
            PipeBarrier<PIPE_V>();
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            DataCopy(vnGm[i * VD], rowH, VD);
            PipeBarrier<PIPE_ALL>();
        }

        // ---------- 步 2：S = exp(gc_last)*S + Σ_i kscaled_i ⊗ v_new_i ----------
        // 标量 exp(gc_last) 用 1 元素向量 Exp 计算（device 端无标量 expf）
        LocalTensor<float> tmp = rowBuf.Get<float>();
        tmp.SetValue(0, gcLast);
        AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
        Exp(tmp, tmp, 1);
        PipeBarrier<PIPE_V>();
        AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
        float stateDecay = tmp.GetValue(0);
        Muls(sL, sL, stateDecay, KD * VD);
        PipeBarrier<PIPE_V>();
        // dec[i] = exp(gc_last - gc_i)
        Muls(dec, gc, -1.0f, C);
        Adds(dec, dec, gcLast, C);
        Exp(dec, dec, C);
        PipeBarrier<PIPE_V>();
        AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);

        for (int32_t i = 0; i < C; ++i) {
            DataCopy(rowH, vnGm[i * VD], VD);
            PipeBarrier<PIPE_ALL>();
            Cast(row, rowH, RoundMode::CAST_NONE, VD);
            PipeBarrier<PIPE_V>();
            float ei = dec.GetValue(i);
            for (int32_t d = 0; d < KD; ++d) {
                float coeff = (float)kH.GetValue(i * KD + d) * ei;
                Axpy(sL[d * VD], row, coeff, VD);
            }
            PipeBarrier<PIPE_V>();
        }
        PipeBarrier<PIPE_ALL>();
        DataCopy(sOutGm, sL, KD * VD);
        PipeBarrier<PIPE_ALL>();
    }

private:
    TPipe pipe;
    TBuf<TPosition::VECCALC> wHalfBuf, uHalfBuf, kHalfBuf, gcBuf, sBuf, rowBuf,
        rowHBuf, decBuf;
    GlobalTensor<half> wGm, uGm, kGm, vnGm;
    GlobalTensor<float> gcGm;
    GlobalTensor<float> sInGm, sOutGm;
};

extern "C" __global__ __aicore__ void gdn_chunk_h(GM_ADDR w, GM_ADDR u, GM_ADDR k, GM_ADDR gc,
                                                   GM_ADDR sIn, GM_ADDR sOut, GM_ADDR vnew)
{
    KernelGdnChunkH op;
    op.Init(w, u, k, gc, sIn, sOut, vnew);
    op.Process();
}

#ifndef ASCENDC_CPU_DEBUG
void gdn_chunk_h_do(uint32_t blockDim, void *stream, uint8_t *w, uint8_t *u, uint8_t *k,
                    uint8_t *gc, uint8_t *sIn, uint8_t *sOut, uint8_t *vnew)
{
    gdn_chunk_h<<<blockDim, nullptr, stream>>>(w, u, k, gc, sIn, sOut, vnew);
}
#endif
