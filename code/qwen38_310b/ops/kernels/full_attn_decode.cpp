/**
 * full_attn_decode — Qwen3.8 全注意力层单 token 解码（GQA + 部分 RoPE + 输出门控）
 *
 * QH=24, KH=4, D=256, RD=64 (partial_rotary_factor 0.25), scale=1/16, attn_output_gate=true
 *
 * KV cache（fp16）：k [KH, MAXT, D]、v [KH, MAXT, D]（token-major，方便按 token 拷入 UB）
 * 本版本按 KC=64 个 key 一块处理，UB 占用 ≤ ~160KB。
 */
#include "kernel_operator.h"

using namespace AscendC;

constexpr int32_t QH = 24;
constexpr int32_t KH = 4;
constexpr int32_t D = 256;
constexpr int32_t RD = 64;
constexpr int32_t RH = RD / 2;      // 32
constexpr int32_t MAXT = 512;
constexpr int32_t GK = QH / KH;     // 6
constexpr int32_t KC = 64;          // key chunk

class KernelFullAttnDecode {
public:
    __aicore__ inline KernelFullAttnDecode() {}

    __aicore__ inline void Init(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR cosv, GM_ADDR sinv,
                                GM_ADDR gate, GM_ADDR kcache, GM_ADDR vcache, GM_ADDR out)
    {
        qGm.SetGlobalBuffer((__gm__ half *)q, QH * D);
        kGm.SetGlobalBuffer((__gm__ half *)k, KH * D);
        vGm.SetGlobalBuffer((__gm__ half *)v, KH * D);
        cosGm.SetGlobalBuffer((__gm__ float *)cosv, RH);
        sinGm.SetGlobalBuffer((__gm__ float *)sinv, RH);
        gateGm.SetGlobalBuffer((__gm__ half *)gate, QH * D);
        kcGm.SetGlobalBuffer((__gm__ half *)kcache, KH * MAXT * D);
        vcGm.SetGlobalBuffer((__gm__ half *)vcache, KH * MAXT * D);
        oGm.SetGlobalBuffer((__gm__ half *)out, QH * D);

        pipe.InitBuffer(qHalfBuf, QH * D * sizeof(half));
        pipe.InitBuffer(kHalfBuf, KH * D * sizeof(half));
        pipe.InitBuffer(vHalfBuf, KH * D * sizeof(half));
        pipe.InitBuffer(gateHalfBuf, QH * D * sizeof(half));
        pipe.InitBuffer(qBuf, QH * D * sizeof(float));
        pipe.InitBuffer(gateBuf, D * sizeof(float));
        pipe.InitBuffer(cosBuf, RH * sizeof(float));
        pipe.InitBuffer(sinBuf, RH * sizeof(float));
        pipe.InitBuffer(scoreBuf, MAXT * sizeof(float));
        pipe.InitBuffer(outBuf, D * sizeof(float));
        pipe.InitBuffer(tmpBuf, D * sizeof(float));
        pipe.InitBuffer(kChunkHBuf, KC * D * sizeof(half));
        pipe.InitBuffer(kChunkBuf, KC * D * sizeof(float));
        pipe.InitBuffer(vChunkHBuf, KC * D * sizeof(half));
        pipe.InitBuffer(outHalfBuf, D * sizeof(half));
        pipe.InitBuffer(kOutHBuf, D * sizeof(half));
    }

    __aicore__ inline void Process(int32_t pos)
    {
        LocalTensor<half> qH = qHalfBuf.Get<half>();
        LocalTensor<half> kH = kHalfBuf.Get<half>();
        LocalTensor<half> vH = vHalfBuf.Get<half>();
        LocalTensor<half> gateH = gateHalfBuf.Get<half>();
        LocalTensor<float> qF = qBuf.Get<float>();
        LocalTensor<float> gateF = gateBuf.Get<float>();
        LocalTensor<float> cosF = cosBuf.Get<float>();
        LocalTensor<float> sinF = sinBuf.Get<float>();
        LocalTensor<float> score = scoreBuf.Get<float>();
        LocalTensor<float> outF = outBuf.Get<float>();
        LocalTensor<float> tmp = tmpBuf.Get<float>();
        LocalTensor<half> kChunkH = kChunkHBuf.Get<half>();
        LocalTensor<half> kOutH = kOutHBuf.Get<half>();
        LocalTensor<float> kChunk = kChunkBuf.Get<float>();
        LocalTensor<half> vChunkH = vChunkHBuf.Get<half>();
        LocalTensor<half> outH = outHalfBuf.Get<half>();

        DataCopy(qH, qGm, QH * D);
        DataCopy(kH, kGm, KH * D);
        DataCopy(vH, vGm, KH * D);
        DataCopy(gateH, gateGm, QH * D);
        DataCopy(cosF, cosGm, RH);
        DataCopy(sinF, sinGm, RH);
        PipeBarrier<PIPE_ALL>();
        Cast(qF, qH, RoundMode::CAST_NONE, QH * D);
        PipeBarrier<PIPE_V>();
        AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);

        // ---- 部分 RoPE（标量，每 head 64 次）----
        for (int32_t h = 0; h < QH; ++h) {
            ropeRow(qF[h * D], cosF, sinF);
        }
        // k 的 RoPE 直接在 cache 写入前做（用 tmp 缓冲）
        for (int32_t h = 0; h < KH; ++h) {
            for (int32_t d = 0; d < D; ++d) {
                score.SetValue(d, (float)kH.GetValue(h * D + d));
            }
            PipeBarrier<PIPE_ALL>();
            AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
            ropeRow(score, cosF, sinF);
            AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
            // 向量写回 cache（GM 标量 SetValue 不可靠）
            AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
            Cast(kOutH, score, RoundMode::CAST_NONE, D);
            PipeBarrier<PIPE_V>();
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            DataCopy(kcGm[h * MAXT * D + pos * D], kOutH, D);
            DataCopy(vcGm[h * MAXT * D + pos * D], vH[h * D], D);
            PipeBarrier<PIPE_ALL>();
        }

        const int32_t nkeys = pos + 1;
        const float scale = 1.0f / 16.0f;

        for (int32_t h = 0; h < QH; ++h) {
            int32_t hkv = h / GK;
            // ---- scores = q_h · k_j ----
            for (int32_t base = 0; base < nkeys; base += KC) {
                int32_t cnt = (nkeys - base >= KC) ? KC : (nkeys - base);
                DataCopy(kChunkH, kcGm[hkv * MAXT * D + base * D], cnt * D);
                PipeBarrier<PIPE_ALL>();
                Cast(kChunk, kChunkH, RoundMode::CAST_NONE, cnt * D);
                PipeBarrier<PIPE_V>();
                AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
                for (int32_t j = 0; j < cnt; ++j) {
                    Mul(tmp, qF[h * D], kChunk[j * D], D);
                    PipeBarrier<PIPE_V>();
                    reduceToScalar(tmp, score, base + j);
                }
            }
            PipeBarrier<PIPE_V>();
            Muls(score, score, scale, nkeys);
            PipeBarrier<PIPE_V>();
            AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
            float mx = -1e30f;
            for (int32_t j = 0; j < nkeys; ++j) {
                float sv = score.GetValue(j);
                if (sv > mx) mx = sv;
            }
            Adds(score, score, -mx, nkeys);
            Exp(score, score, nkeys);
            PipeBarrier<PIPE_V>();
            AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
            float ssum = 0.0f;
            for (int32_t j = 0; j < nkeys; ++j) {
                ssum += score.GetValue(j);
            }
            Muls(score, score, 1.0f / ssum, nkeys);
            PipeBarrier<PIPE_V>();
            // ---- out = Σ_j p_j v_j ----
            Duplicate(outF, 0.0f, D);
            PipeBarrier<PIPE_V>();
            for (int32_t base = 0; base < nkeys; base += KC) {
                int32_t cnt = (nkeys - base >= KC) ? KC : (nkeys - base);
                DataCopy(vChunkH, vcGm[hkv * MAXT * D + base * D], cnt * D);
                PipeBarrier<PIPE_ALL>();
                Cast(tmp, vChunkH, RoundMode::CAST_NONE, D);   // 先转第 0 个 token 占位
                PipeBarrier<PIPE_V>();
                AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
                for (int32_t j = 0; j < cnt; ++j) {
                    DataCopy(vChunkH, vcGm[hkv * MAXT * D + (base + j) * D], D);
                    PipeBarrier<PIPE_ALL>();
                    Cast(tmp, vChunkH, RoundMode::CAST_NONE, D);
                    PipeBarrier<PIPE_V>();
                    float pj = score.GetValue(base + j);
                    Axpy(outF, tmp, pj, D);
                    PipeBarrier<PIPE_V>();
                }
            }
            // ---- 输出门控 swish ----
            Cast(gateF, gateH[h * D], RoundMode::CAST_NONE, D);
            PipeBarrier<PIPE_V>();
            Muls(tmp, gateF, -1.0f, D);
            Exp(tmp, tmp, D);
            Adds(tmp, tmp, 1.0f, D);
            PipeBarrier<PIPE_V>();
            Reciprocal(tmp, tmp, D);
            PipeBarrier<PIPE_V>();
            // 官方 Qwen3.5：attn_output *= sigmoid(gate)（不是 silu）
            Mul(outF, outF, tmp, D);
            PipeBarrier<PIPE_V>();
            Cast(outH, outF, RoundMode::CAST_NONE, D);
            PipeBarrier<PIPE_V>();
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            DataCopy(oGm[h * D], outH, D);
            PipeBarrier<PIPE_ALL>();
        }
        PipeBarrier<PIPE_ALL>();
    }

private:
    // D 长度向量的树形归约，结果写 score[idx]（标量）
    __aicore__ inline void reduceToScalar(LocalTensor<float> buf, LocalTensor<float> dst,
                                          int32_t idx)
    {
        for (int32_t len = D / 2; len >= 8; len >>= 1) {
            Add(buf, buf, buf[len], len);
            PipeBarrier<PIPE_V>();
        }
        AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
        float s = 0.0f;
        for (int32_t j = 0; j < 8; ++j) {
            s += buf.GetValue(j);
        }
        dst.SetValue(idx, s);
        AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
    }

    __aicore__ inline void ropeRow(LocalTensor<float> row, LocalTensor<float> cosF,
                                   LocalTensor<float> sinF)
    {
        for (int32_t i = 0; i < RH; ++i) {
            float a = row.GetValue(i);
            float b = row.GetValue(i + RH);
            float c = cosF.GetValue(i);
            float s = sinF.GetValue(i);
            row.SetValue(i, a * c - b * s);
            row.SetValue(i + RH, b * c + a * s);
        }
        PipeBarrier<PIPE_ALL>();
    }

    TPipe pipe;
    TBuf<TPosition::VECCALC> qHalfBuf, kHalfBuf, vHalfBuf, gateHalfBuf, qBuf, gateBuf,
        cosBuf, sinBuf, scoreBuf, outBuf, tmpBuf, kChunkHBuf, kChunkBuf, vChunkHBuf,
        outHalfBuf, kOutHBuf;
    GlobalTensor<half> qGm, kGm, vGm, gateGm, oGm, kcGm, vcGm;
    GlobalTensor<float> cosGm, sinGm;
};

extern "C" __global__ __aicore__ void full_attn_decode(GM_ADDR q, GM_ADDR k, GM_ADDR v,
                                                        GM_ADDR cosv, GM_ADDR sinv,
                                                        GM_ADDR gate, GM_ADDR kcache,
                                                        GM_ADDR vcache, GM_ADDR out,
                                                        int32_t pos)
{
    KernelFullAttnDecode op;
    op.Init(q, k, v, cosv, sinv, gate, kcache, vcache, out);
    op.Process(pos);
}

#ifndef ASCENDC_CPU_DEBUG
void full_attn_decode_do(uint32_t blockDim, void *stream, uint8_t *q, uint8_t *k, uint8_t *v,
                         uint8_t *cosv, uint8_t *sinv, uint8_t *gate, uint8_t *kcache,
                         uint8_t *vcache, uint8_t *out, int32_t pos)
{
    full_attn_decode<<<blockDim, nullptr, stream>>>(q, k, v, cosv, sinv, gate, kcache,
                                                     vcache, out, pos);
}
#endif
