/**
 * w4a16_gemv_gen — 通用形状 W4A16 GEMV（M=1，group-32 非对称，K 到 17408）
 *
 *   out[n] = Σ_g s[n,g] * Σ_{k∈g} q_k act_k  −  Σ_g sz[n,g] * suma[g]
 *
 * 权重布局（相位交织，KQ = K/4）：word j 的 4 半字节 = q[j], q[j+KQ], q[j+2KQ], q[j+3KQ]
 *
 * 核心结构：**按相位整段处理**（不再按 K 分块）
 *   phase p 覆盖 k = p*KQ + j (j ∈ [0,KQ))，act 段 = act[p*KQ ..] 连续 ✓
 *   group = p*(KQ/32) + j/32（每相位 KQ/32 组）
 *   每相位：抽取(KQ) → Cast → Gather 展开 scale(KQ) → Mul 两次 → 归约到 8 个部分和
 *   4 个相位先各自归约、再相加，每行只需 **1 次 V_S 同步 + 8+8 次标量读**。
 *
 * 约束：K % 128 == 0（KQ = K/4 是 32 的倍数）、K ≤ 17408
 */
#include "kernel_operator.h"

using namespace AscendC;

constexpr int32_t GS = 32;
constexpr int32_t RC = 16;      // scale 缓存 / 输出批量行数

static __aicore__ inline int32_t P2(int32_t x)
{
    int32_t p = 1;
    while (p < x) p <<= 1;
    return p;
}

extern "C" __global__ __aicore__ void w4a16_gemv_gen(GM_ADDR act, GM_ADDR wq, GM_ADDR scales,
                                                      GM_ADDR szs, GM_ADDR out, int32_t K,
                                                      int32_t N, int32_t stage)
{
    TPipe pipe;
    const int32_t KQ = K / 4;
    const int32_t NG = K / GS;
    const int32_t GPP = KQ / GS;                 // 每相位 group 数
    const int32_t SBP = (NG + 15) / 16 * 16;
    const int32_t NBP = P2(SBP);

    GlobalTensor<half> actGm, sGm, szGm, oGm;
    GlobalTensor<uint16_t> wGm;
    actGm.SetGlobalBuffer((__gm__ half *)act, K);
    wGm.SetGlobalBuffer((__gm__ uint16_t *)wq, (int64_t)N * KQ);
    sGm.SetGlobalBuffer((__gm__ half *)scales, (int64_t)N * SBP);
    szGm.SetGlobalBuffer((__gm__ half *)szs, (int64_t)N * SBP);
    oGm.SetGlobalBuffer((__gm__ half *)out, N);

    TBuf<TPosition::VECCALC> accSegBuf, actHBuf, actFBuf, idxBuf, scCacheBuf, szCacheBuf, scTabBuf,
        scaleSegBuf, phBuf, phFBuf, maskBuf, szFBuf, sumaBuf, prodBuf,
        sumRowBuf, accBuf, outHalfBuf, wRowBuf;
    pipe.InitBuffer(actHBuf, 4096 * sizeof(half));   // act 分块装载，避免 K=17408 时 UB 超限
    pipe.InitBuffer(actFBuf, K * sizeof(float));
    pipe.InitBuffer(idxBuf, KQ * sizeof(uint32_t));
    pipe.InitBuffer(scCacheBuf, RC * SBP * sizeof(half));
    pipe.InitBuffer(szCacheBuf, RC * SBP * sizeof(half));
    pipe.InitBuffer(scTabBuf, SBP * sizeof(float));
    pipe.InitBuffer(accSegBuf, KQ * sizeof(float));
    pipe.InitBuffer(scaleSegBuf, KQ * sizeof(float));
    pipe.InitBuffer(phBuf, KQ * sizeof(uint16_t));
    pipe.InitBuffer(phFBuf, KQ * sizeof(float));
    pipe.InitBuffer(maskBuf, KQ * sizeof(uint16_t));
    pipe.InitBuffer(szFBuf, NBP * sizeof(float));
    pipe.InitBuffer(sumaBuf, NBP * sizeof(float));
    pipe.InitBuffer(prodBuf, NBP * sizeof(float));
    pipe.InitBuffer(sumRowBuf, 16 * 4 * sizeof(float));
    pipe.InitBuffer(accBuf, RC * sizeof(float));
    pipe.InitBuffer(outHalfBuf, RC * sizeof(half));
    pipe.InitBuffer(wRowBuf, KQ * sizeof(uint16_t));

    LocalTensor<half> actH = actHBuf.Get<half>();
    LocalTensor<float> actF = actFBuf.Get<float>();
    LocalTensor<uint32_t> idx = idxBuf.Get<uint32_t>();
    LocalTensor<half> scC = scCacheBuf.Get<half>();
    LocalTensor<half> szC = szCacheBuf.Get<half>();
    LocalTensor<float> scTab = scTabBuf.Get<float>();
    LocalTensor<float> accSeg = accSegBuf.Get<float>();
    LocalTensor<float> scaleSeg = scaleSegBuf.Get<float>();
    LocalTensor<uint16_t> ph = phBuf.Get<uint16_t>();
    LocalTensor<float> phF = phFBuf.Get<float>();
    LocalTensor<uint16_t> mask = maskBuf.Get<uint16_t>();
    LocalTensor<float> szF = szFBuf.Get<float>();
    LocalTensor<float> suma = sumaBuf.Get<float>();
    LocalTensor<float> prod = prodBuf.Get<float>();
    LocalTensor<float> phaseSums = sumRowBuf.Get<float>();     // [16] 复用为 4 相位结果
    LocalTensor<float> acc = accBuf.Get<float>();
    LocalTensor<half> oH = outHalfBuf.Get<half>();
    LocalTensor<uint16_t> wRow = wRowBuf.Get<uint16_t>();

    for (int32_t k0 = 0; k0 < K; k0 += 4096) {
        int32_t len0 = (K - k0 >= 4096) ? 4096 : (K - k0);
        DataCopy(actH, actGm[k0], len0);
        PipeBarrier<PIPE_ALL>();
        Cast(actF[k0], actH, RoundMode::CAST_NONE, len0);
        PipeBarrier<PIPE_ALL>();
    }
    PipeBarrier<PIPE_ALL>();
    Duplicate(mask, (uint16_t)0x000F, KQ);
    Duplicate(suma, 0.0f, NBP);
    Duplicate(prod, 0.0f, NBP);
    Duplicate(phaseSums, 0.0f, 16);
    for (int32_t j = 0; j < KQ; ++j) {
        idx.SetValue(j, (uint32_t)((j / GS) * 4));
    }
    PipeBarrier<PIPE_ALL>();
    AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
    for (int32_t g = 0; g < NG; ++g) {
        float s = 0.0f;
        for (int32_t j = 0; j < GS; ++j) {
            s += actF.GetValue(g * GS + j);
        }
        suma.SetValue(g, s);
    }
    PipeBarrier<PIPE_ALL>();
    if (stage == 1) {
        return;
    }

    // 归约长度：先按 2 的幂折半（要求 len*4 是 32B 的倍数 → len%8==0），再按 8 元素步长收尾


    for (int32_t n = 0; n < N; ++n) {
        if ((n & (RC - 1)) == 0) {
            int32_t rows = (N - n >= RC) ? RC : (N - n);
            DataCopy(scC, sGm[n * SBP], rows * SBP);
            DataCopy(szC, szGm[n * SBP], rows * SBP);
            PipeBarrier<PIPE_ALL>();
            AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
        }
        if (stage == 2) {
            return;
        }
        int32_t off = (n & (RC - 1)) * SBP;
        DataCopy(wRow, wGm[n * KQ], KQ);
        PipeBarrier<PIPE_ALL>();
        Cast(scTab, scC[off], RoundMode::CAST_NONE, SBP);
        Duplicate(phaseSums, 0.0f, 16);
        for (int32_t p = 0; p < 4; ++p) {
            // 抽取相位 p（KQ 个 0..15）
            if (p == 0) {
                And(ph, wRow, mask, KQ);
            } else {
                ShiftRight(ph, wRow, (uint16_t)(4 * p), KQ);
                And(ph, ph, mask, KQ);
            }
            Cast(phF, ph.ReinterpretCast<int16_t>(), RoundMode::CAST_NONE, KQ);
            if (stage == 3) {
                return;
            }
            // Gather 展开本相位 KQ/32 个 group scale（base 偏移到该相位起始）
            Gather(scaleSeg, scTab[p * GPP], idx, 0, (uint32_t)KQ);
            Mul(scaleSeg, scaleSeg, actF[p * KQ], KQ);   // scaleSeg = s * act（原地）
            Mul(phF, phF, scaleSeg, KQ);
            // 折半归约（偏移 32B 对齐）
            if (p == 0) {
                Muls(accSeg, phF, 1.0f, KQ);        // 第一个相位直接拷贝（无 UB->UB copy）
            } else {
                Add(accSeg, accSeg, phF, KQ);
            }
            if (stage == 4) {
                return;
            }
        }
        // 4 个相位累加完，统一归约一次
        {
            int32_t lastLen = 8;
            for (int32_t len = KQ / 2; len >= 8 && (len % 8 == 0); len >>= 1) {
                Add(accSeg, accSeg, accSeg[len], len);
                lastLen = len;
            }
            for (int32_t st = 8; st + 8 <= lastLen; st += 8) {
                Add(accSeg, accSeg, accSeg[st], 8);
            }
            Muls(phaseSums, accSeg, 1.0f, 8);
        }
        // bias = Σ_g sz_g * suma_g
        Duplicate(prod, 0.0f, NBP);
        Cast(szF, szC[off], RoundMode::CAST_NONE, SBP);
        Mul(prod, szF, suma, SBP);
        for (int32_t len = NBP / 2; len >= 8; len >>= 1) {
            Add(prod, prod, prod[len], len);
        }
        AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
        float dot = 0.0f, bias = 0.0f;
        for (int32_t j = 0; j < 8; ++j) {
            dot += phaseSums.GetValue(j);
            bias += prod.GetValue(j);
        }
        acc.SetValue(n & (RC - 1), dot - bias);
        if (stage == 5) {
            return;
        }
        if ((n & (RC - 1)) == RC - 1 || n == N - 1) {
            int32_t cnt = ((n & (RC - 1)) == RC - 1) ? RC : ((n & (RC - 1)) + 1);
            PipeBarrier<PIPE_ALL>();
            AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
            Cast(oH, acc, RoundMode::CAST_NONE, RC);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            DataCopy(oGm[n - cnt + 1], oH, cnt);
            PipeBarrier<PIPE_ALL>();
        }
    }
}

#ifndef ASCENDC_CPU_DEBUG
void w4a16_gemv_gen_do(uint32_t b, void *st, uint8_t *act, uint8_t *wq, uint8_t *scales,
                       uint8_t *szs, uint8_t *out, int32_t K, int32_t N, int32_t stage)
{
    w4a16_gemv_gen<<<b, nullptr, st>>>(act, wq, scales, szs, out, K, N, stage);
}
#endif
