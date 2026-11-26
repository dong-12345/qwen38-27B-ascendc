/**
 * w4a16_gemv_gen2 — w4a16_gemv_gen 的"合并指令"版本（**数值逐位等价**，已板端对拍）
 *
 *   out[n] = Σ_g s[n,g]·(Σ_{k∈g} q[n,k]·a[k]) − Σ_g sz[n,g]·suma[g]
 *
 * 与 gen 的唯一区别是"把 4 个相位的重复动作合并成大指令"，逐元素运算顺序完全不变，
 * 因此输出与 gen **逐位相同**（同一份 golden 对拍 max_abs = 0）：
 *
 *   gen                                     gen2
 *   ------------------------------------    --------------------------------------------
 *   每相位 1 次 Gather(scale, KQ 元素) ×4   → 整行 1 次 Gather(s 展开到 K 元素)
 *   每相位 1 次 Cast(q→f32, KQ) ×4         → 1 次 Cast(4 相位连续, K 元素)
 *   每相位 1 次 Mul(q·s, KQ) ×4            → 1 次 Mul(K 元素)
 *   每相位 1 次 Mul(q·s·a, KQ) ×4          → 1 次 Mul(K 元素)
 *   每相位 1 次 Muls/Add 累加 ×4           → 3 次 Add(KQ 元素)，顺序与 gen 一致
 *   合计 ~51 条向量指令/行                 → ~33 条/行（≈1.5x）
 *
 * 为什么能合并：gen 里每个相位的 scale 展开其实是同一张表的连续切片
 * （group(k) = p*GPP + j/32 = k/32），所以整行的 s 展开就是 s[k/32]，
 * 一次 Gather 就能给出全部 4 个相位；而相位 p 的数据在 K 维上本来就是连续段
 * （k = p*KQ + j），所以 4 个相位的 Cast/Mul 可以拼成 K 长的一条大指令。
 *
 * 约束：K % 128 == 0（KQ = K/4 是 32 的倍数）、K ≤ 17408。
 */
#include "kernel_operator.h"

using namespace AscendC;

constexpr int32_t GS2 = 32;
constexpr int32_t RC2 = 16;      // scale 缓存 / 输出批量行数

static __aicore__ inline int32_t P2C(int32_t x)
{
    int32_t p = 1;
    while (p < x) p <<= 1;
    return p;
}

extern "C" __global__ __aicore__ void w4a16_gemv_gen2(GM_ADDR act, GM_ADDR wq, GM_ADDR scales,
                                                      GM_ADDR szs, GM_ADDR out, int32_t K,
                                                      int32_t N, int32_t stage)
{
    TPipe pipe;
    const int32_t KQ = K / 4;
    const int32_t NG = K / GS2;
    const int32_t GPP = KQ / GS2;
    const int32_t SBP = (NG + 15) / 16 * 16;
    const int32_t NBP = P2C(SBP);

    GlobalTensor<half> actGm, sGm, szGm, oGm;
    GlobalTensor<uint16_t> wGm;
    actGm.SetGlobalBuffer((__gm__ half *)act, K);
    wGm.SetGlobalBuffer((__gm__ uint16_t *)wq, (int64_t)N * KQ);
    sGm.SetGlobalBuffer((__gm__ half *)scales, (int64_t)N * SBP);
    szGm.SetGlobalBuffer((__gm__ half *)szs, (int64_t)N * SBP);
    oGm.SetGlobalBuffer((__gm__ half *)out, N);

    TBuf<TPosition::VECCALC> actHBuf, actFBuf, idxBuf, scCacheBuf, szCacheBuf, scTabBuf, sExpBuf,
        ph4uBuf, ph4fBuf, maskBuf, szFBuf, sumaBuf, prodBuf, sumRowBuf, accBuf, outHalfBuf,
        wRowBuf;
    pipe.InitBuffer(actHBuf, 4096 * sizeof(half));
    pipe.InitBuffer(actFBuf, (size_t)K * sizeof(float));
    pipe.InitBuffer(idxBuf, (size_t)K * sizeof(uint32_t));      // 整行 group scale 索引
    pipe.InitBuffer(scCacheBuf, RC2 * SBP * sizeof(half));
    pipe.InitBuffer(szCacheBuf, RC2 * SBP * sizeof(half));
    pipe.InitBuffer(scTabBuf, (size_t)SBP * sizeof(float));
    pipe.InitBuffer(sExpBuf, (size_t)K * sizeof(float));        // s 展开到整行
    pipe.InitBuffer(ph4uBuf, (size_t)K * sizeof(uint16_t));     // 4 相位抽取结果
    pipe.InitBuffer(ph4fBuf, (size_t)K * sizeof(float));        // 4 相位乘积
    pipe.InitBuffer(maskBuf, (size_t)KQ * sizeof(uint16_t));
    pipe.InitBuffer(szFBuf, (size_t)NBP * sizeof(float));
    pipe.InitBuffer(sumaBuf, (size_t)NBP * sizeof(float));
    pipe.InitBuffer(prodBuf, (size_t)NBP * sizeof(float));
    pipe.InitBuffer(sumRowBuf, 16 * sizeof(float));
    pipe.InitBuffer(accBuf, RC2 * sizeof(float));
    pipe.InitBuffer(outHalfBuf, RC2 * sizeof(half));
    pipe.InitBuffer(wRowBuf, (size_t)KQ * sizeof(uint16_t));

    LocalTensor<half> actH = actHBuf.Get<half>();
    LocalTensor<float> actF = actFBuf.Get<float>();
    LocalTensor<uint32_t> idx = idxBuf.Get<uint32_t>();
    LocalTensor<half> scC = scCacheBuf.Get<half>();
    LocalTensor<half> szC = szCacheBuf.Get<half>();
    LocalTensor<float> scTab = scTabBuf.Get<float>();
    LocalTensor<float> sExp = sExpBuf.Get<float>();
    LocalTensor<uint16_t> ph4u = ph4uBuf.Get<uint16_t>();
    LocalTensor<float> ph4f = ph4fBuf.Get<float>();
    LocalTensor<uint16_t> mask = maskBuf.Get<uint16_t>();
    LocalTensor<float> szF = szFBuf.Get<float>();
    LocalTensor<float> suma = sumaBuf.Get<float>();
    LocalTensor<float> prod = prodBuf.Get<float>();
    LocalTensor<float> red = sumRowBuf.Get<float>();      // [16] 归约结果
    LocalTensor<float> acc = accBuf.Get<float>();
    LocalTensor<half> oH = outHalfBuf.Get<half>();
    LocalTensor<uint16_t> wRow = wRowBuf.Get<uint16_t>();

    // ---------------- 入口：act 转 fp32 + 索引表 + suma ----------------
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
    Duplicate(red, 0.0f, 16);
    // idx[k] = (k/32)*4 字节：整行的 group scale 展开（4 个相位天然连续）
    for (int32_t k = 0; k < K; ++k) {
        idx.SetValue(k, (uint32_t)((k / GS2) * 4));
    }
    PipeBarrier<PIPE_ALL>();
    AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
    for (int32_t g = 0; g < NG; ++g) {
        float s = 0.0f;
        for (int32_t j = 0; j < GS2; ++j) {
            s += actF.GetValue(g * GS2 + j);
        }
        suma.SetValue(g, s);
    }
    PipeBarrier<PIPE_ALL>();
    if (stage == 1) {
        return;
    }

    const int32_t GCHUNK = 8192;   // Gather count 上限

    for (int32_t n = 0; n < N; ++n) {
        if ((n & (RC2 - 1)) == 0) {
            int32_t rows = (N - n >= RC2) ? RC2 : (N - n);
            DataCopy(scC, sGm[n * SBP], rows * SBP);
            DataCopy(szC, szGm[n * SBP], rows * SBP);
            PipeBarrier<PIPE_ALL>();
            AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
        }
        if (stage == 2) {
            return;
        }
        int32_t off = (n & (RC2 - 1)) * SBP;
        DataCopy(wRow, wGm[n * KQ], KQ);
        PipeBarrier<PIPE_ALL>();
        Cast(scTab, scC[off], RoundMode::CAST_NONE, SBP);
        Duplicate(red, 0.0f, 16);

        // 整行 group scale 展开（等价于 gen 的 4 次分相位 Gather）
        for (int32_t o = 0; o < K; o += GCHUNK) {
            int32_t len = (K - o >= GCHUNK) ? GCHUNK : (K - o);
            Gather(sExp[o], scTab, idx[o], 0, (uint32_t)len);
        }
        // 4 个相位抽取到 ph4u[p*KQ]（q ∈ 0..15）
        for (int32_t p = 0; p < 4; ++p) {
            if (p == 0) {
                And(ph4u, wRow, mask, KQ);
            } else {
                ShiftRight(ph4u[p * KQ], wRow, (uint16_t)(4 * p), KQ);
                And(ph4u[p * KQ], ph4u[p * KQ], mask, KQ);
            }
        }
        // 合并成 K 长大指令：Cast / ×s / ×act
        Cast(ph4f, ph4u.ReinterpretCast<int16_t>(), RoundMode::CAST_NONE, K);
        Mul(ph4f, ph4f, sExp, K);
        Mul(ph4f, ph4f, actF, K);
        // 4 相位累加（顺序与 gen 一致：p0 + p1 + p2 + p3）
        Add(ph4f, ph4f, ph4f[KQ], KQ);
        Add(ph4f, ph4f, ph4f[2 * KQ], KQ);
        Add(ph4f, ph4f, ph4f[3 * KQ], KQ);

        // 归约（与 gen 完全相同）
        {
            int32_t lastLen = 8;
            for (int32_t len = KQ / 2; len >= 8 && (len % 8 == 0); len >>= 1) {
                Add(ph4f, ph4f, ph4f[len], len);
                lastLen = len;
            }
            for (int32_t st = 8; st + 8 <= lastLen; st += 8) {
                Add(ph4f, ph4f, ph4f[st], 8);
            }
            Muls(red, ph4f, 1.0f, 8);
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
            dot += red.GetValue(j);
            bias += prod.GetValue(j);
        }
        acc.SetValue(n & (RC2 - 1), dot - bias);
        if (stage == 5) {
            return;
        }
        if ((n & (RC2 - 1)) == RC2 - 1 || n == N - 1) {
            int32_t cnt = ((n & (RC2 - 1)) == RC2 - 1) ? RC2 : ((n & (RC2 - 1)) + 1);
            PipeBarrier<PIPE_ALL>();
            AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
            Cast(oH, acc, RoundMode::CAST_NONE, RC2);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            DataCopy(oGm[n - cnt + 1], oH, cnt);
            PipeBarrier<PIPE_ALL>();
        }
    }
}

#ifndef ASCENDC_CPU_DEBUG
void w4a16_gemv_gen2_do(uint32_t b, void *st, uint8_t *act, uint8_t *wq, uint8_t *scales,
                        uint8_t *szs, uint8_t *out, int32_t K, int32_t N, int32_t stage)
{
    w4a16_gemv_gen2<<<b, nullptr, st>>>(act, wq, scales, szs, out, K, N, stage);
}
#endif
