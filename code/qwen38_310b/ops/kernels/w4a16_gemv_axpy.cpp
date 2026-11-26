/**
 * w4a16_gemv_axpy — decode 主路径的 Axpy-N 版本（沿 N 向量化，不需要归约）
 *
 *   out[n] = Σ_g s[n,g]·(Σ_{k∈g} q[n,k]·a[k]) − Σ_g sz[n,g]·suma[g]
 *
 * 权重用 **AX 布局**（离线重排成 [g][w8][n]，4 个半字节/word）：
 *   wq[(g*8 + w8)*N + n]  uint16，4 个半字节依次是 q[32g+4w8+0..3][n]
 *   sc[g*N + n] / sz[g*N + n]  half
 * 存储量与原始 W4 格式相同（0.5 B/权重）。
 *
 * 每个 group（32 个 k）：分成两趟，每趟 2 个半字节面（省 UB，NT 可到 1024）
 *   ⚠ 关键：u0/u1（uint16）与 h0/h1（half）必须用**独立缓冲**。早期版本让它们共用 TBuf
 *     （就地 Cast），在循环里被编译器重排 → 数值错（max_abs 2.0、token 全错），
 *     改成独立缓冲后 max_abs 0.0029、0/17408 超差。
 * 每个 group（32 个 k）：
 *   8 次 1D DataCopy + 2 次 scale/sz 搬运 → PIPE_ALL →
 *   4 次拆半字节 + 4 次 Cast → 32 次 Axpy（nt 宽）→ Mul(scale)/Add → bias
 * 无归约树。
 *
 * 实测：
 *   Qwen3.8-27B  K=5120/N=17408：19.89 ms（gen2 46.10 ms，2.32x）；端到端 decode 12.47 → 4.48 s/token
 *   Moondream3   整体：decode 2.00 → 0.27 s/token（7.4x，3.7 tok/s），token 与基线完全一致
 *
 * ⚠ 坑：scale/sz 的 DataCopy 是 MTE2，必须 PipeBarrier<PIPE_ALL>（PIPE_V 不等 MTE2），
 *      否则 Cast 读到垃圾 scale（表现为结果整体大 ~50 倍、饱和到 65504）。
 * ⚠ 已尝试但失败：软件流水（双缓冲让 DMA 与 V 重叠）。三种写法（SetFlag<MTE2_V>、
 *      barrier 版、两个独立 TBuf）都触发同一个 aicore exception（UB D-cache bus error,
 *      retCode=0x26）。当前每组 DMA 与计算串行，带宽只用到 ~37%。留待后续用
 *      Matmul/手写 LoadData 等其它手段解决。
 */
#include "kernel_operator.h"

using namespace AscendC;

extern "C" __global__ __aicore__ void w4a16_gemv_axpy(GM_ADDR act, GM_ADDR wq, GM_ADDR scales,
                                                      GM_ADDR szs, GM_ADDR out, int32_t K, int32_t N,
                                                      int32_t NTin)
{
    TPipe pipe;
    const int32_t NG = K / 32;
    const bool useF32 = (NTin < 0);
    const int32_t NT = useF32 ? (-NTin) : (NTin > 0 ? NTin : 1024);

    GlobalTensor<half> actGm, sGm, szGm, oGm;
    GlobalTensor<uint16_t> wGm;
    actGm.SetGlobalBuffer((__gm__ half *)act, K);
    wGm.SetGlobalBuffer((__gm__ uint16_t *)wq, (int64_t)NG * 8 * N);
    sGm.SetGlobalBuffer((__gm__ half *)scales, (int64_t)NG * N);
    szGm.SetGlobalBuffer((__gm__ half *)szs, (int64_t)NG * N);
    oGm.SetGlobalBuffer((__gm__ half *)out, N);

    TBuf<TPosition::VECCALC> aBuf, sumBuf, wpkBuf, tmpBuf, q0Buf, q1Buf, h0Buf, h1Buf, f0Buf, f1Buf, u1Buf, tmpFBuf, pHBuf,
        pFBuf, accBuf, biasBuf, scTBuf, scFBuf, szTBuf, szFBuf, oBuf, mBuf;
    pipe.InitBuffer(aBuf, (size_t)K * 4);
    pipe.InitBuffer(sumBuf, (size_t)NG * 4);
    pipe.InitBuffer(wpkBuf, (size_t)8 * NT * 2);
    pipe.InitBuffer(tmpBuf, (size_t)8 * NT * 2);
    pipe.InitBuffer(q0Buf, (size_t)8 * NT * 2);
    pipe.InitBuffer(q1Buf, (size_t)8 * NT * 2);
    pipe.InitBuffer(h0Buf, (size_t)8 * NT * 2);
    pipe.InitBuffer(h1Buf, (size_t)8 * NT * 2);
    pipe.InitBuffer(pHBuf, (size_t)(useF32 ? NT * 4 : NT * 2));
    pipe.InitBuffer(pFBuf, (size_t)NT * 4);
    pipe.InitBuffer(accBuf, (size_t)NT * 4);
    pipe.InitBuffer(biasBuf, (size_t)NT * 4);
    pipe.InitBuffer(scTBuf, (size_t)NT * 2);
    pipe.InitBuffer(scFBuf, (size_t)NT * 4);
    pipe.InitBuffer(szTBuf, (size_t)NT * 2);
    pipe.InitBuffer(szFBuf, (size_t)NT * 4);
    pipe.InitBuffer(oBuf, (size_t)NT * 2);
    pipe.InitBuffer(mBuf, (size_t)8 * NT * 2);
    if (useF32) {
        pipe.InitBuffer(f0Buf, (size_t)8 * NT * 4);
        pipe.InitBuffer(f1Buf, (size_t)8 * NT * 4);
        pipe.InitBuffer(tmpFBuf, (size_t)8 * NT * 4);
    }

    LocalTensor<float> aF = aBuf.Get<float>();
    LocalTensor<float> suma = sumBuf.Get<float>();
    LocalTensor<uint16_t> wpk = wpkBuf.Get<uint16_t>();
    LocalTensor<uint16_t> tmpU = tmpBuf.Get<uint16_t>();
    LocalTensor<uint16_t> u0 = q0Buf.Get<uint16_t>();
    LocalTensor<uint16_t> u1 = q1Buf.Get<uint16_t>();
    LocalTensor<half> h0 = h0Buf.Get<half>();
    LocalTensor<half> h1 = h1Buf.Get<half>();
    LocalTensor<half> pH = pHBuf.Get<half>();
    LocalTensor<float> pHf;
    LocalTensor<float> fTmp;
    LocalTensor<float> f0;
    LocalTensor<float> f1;
    if (useF32) {
        pHf = pHBuf.Get<float>();
        fTmp = tmpFBuf.Get<float>();
        f0 = f0Buf.Get<float>();
        f1 = f1Buf.Get<float>();
    }
    LocalTensor<float> pF = pFBuf.Get<float>();
    LocalTensor<float> acc = accBuf.Get<float>();
    LocalTensor<float> bias = biasBuf.Get<float>();
    LocalTensor<half> scT = scTBuf.Get<half>();
    LocalTensor<float> scF = scFBuf.Get<float>();
    LocalTensor<half> szT = szTBuf.Get<half>();
    LocalTensor<float> szF = szFBuf.Get<float>();
    LocalTensor<half> oH = oBuf.Get<half>();
    LocalTensor<uint16_t> maskT = mBuf.Get<uint16_t>();

    {
        TBuf<TPosition::VECCALC> aHBuf;
        pipe.InitBuffer(aHBuf, (size_t)K * 2);
        LocalTensor<half> aH = aHBuf.Get<half>();
        DataCopy(aH, actGm, K);
        PipeBarrier<PIPE_ALL>();
        Cast(aF, aH, RoundMode::CAST_NONE, K);
        PipeBarrier<PIPE_ALL>();
        AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
        for (int32_t g = 0; g < NG; ++g) {
            float s = 0.0f;
            for (int32_t j = 0; j < 32; ++j) s += aF.GetValue(g * 32 + j);
            suma.SetValue(g, s);
        }
        AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
    }
    Duplicate(maskT, (uint16_t)0x000F, 8 * NT);

    const int32_t nTiles = (N + NT - 1) / NT;
    for (int32_t t = 0; t < nTiles; ++t) {
        const int32_t n0 = t * NT;
        const int32_t nt = (N - n0 >= NT) ? NT : (N - n0);
        Duplicate(acc, 0.0f, NT);
        Duplicate(bias, 0.0f, NT);
        for (int32_t g = 0; g < NG; ++g) {
            for (int32_t w8 = 0; w8 < 8; ++w8) {
                DataCopy(wpk[(size_t)w8 * nt], wGm[(size_t)(g * 8 + w8) * N + n0], nt);
            }
            DataCopy(scT, sGm[(size_t)g * N + n0], nt);
            DataCopy(szT, szGm[(size_t)g * N + n0], nt);
            PipeBarrier<PIPE_ALL>();   // 等 MTE2 完成（PIPE_V 不等 MTE2）
            // 两个半字节面 × 两趟（省 UB，允许更大的 NT：NT=2048 时 Axpy 指令数减半）
            if (useF32) { Duplicate(pHf, 0.0f, NT); } else { Duplicate(pH, (half)0.0f, NT); }
            AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
            for (int32_t pass = 0; pass < 2; ++pass) {
                const uint16_t sh = (uint16_t)(pass == 0 ? 0 : 8);
                if (pass == 0) {
                    And(u0, wpk, maskT, 8 * nt);
                } else {
                    ShiftRight(tmpU, wpk, (uint16_t)8, 8 * nt);
                    And(u0, tmpU, maskT, 8 * nt);
                }
                Cast(h0, u0.ReinterpretCast<int16_t>(), RoundMode::CAST_NONE, 8 * nt);
                ShiftRight(tmpU, wpk, (uint16_t)(sh + 4), 8 * nt);
                And(u1, tmpU, maskT, 8 * nt);
                Cast(h1, u1.ReinterpretCast<int16_t>(), RoundMode::CAST_NONE, 8 * nt);
                PipeBarrier<PIPE_V>();
                if (useF32) {
                    Cast(f0, h0, RoundMode::CAST_NONE, 8 * nt);
                    Cast(f1, h1, RoundMode::CAST_NONE, 8 * nt);
                    PipeBarrier<PIPE_V>();
                }
                for (int32_t w8 = 0; w8 < 8; ++w8) {
                    float af0 = aF.GetValue(g * 32 + 4 * w8 + (pass == 0 ? 0 : 2));
                    float af1 = aF.GetValue(g * 32 + 4 * w8 + (pass == 0 ? 1 : 3));
                    if (useF32) {
                        Muls(fTmp, f0[(size_t)w8 * nt], af0, nt);
                        Add(pHf, pHf, fTmp, nt);
                        Muls(fTmp, f1[(size_t)w8 * nt], af1, nt);
                        Add(pHf, pHf, fTmp, nt);
                    } else {
                        Axpy(pH, h0[(size_t)w8 * nt], (half)af0, nt);
                        Axpy(pH, h1[(size_t)w8 * nt], (half)af1, nt);
                    }
                }
                PipeBarrier<PIPE_V>();
            }
            AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
            Cast(scF, scT, RoundMode::CAST_NONE, nt);
            Cast(szF, szT, RoundMode::CAST_NONE, nt);
            PipeBarrier<PIPE_V>();
            if (useF32) { Mul(pF, pHf, scF, nt); } else { Cast(pF, pH, RoundMode::CAST_NONE, nt); Mul(pF, pF, scF, nt); }
            Add(acc, acc, pF, nt);
            float sg = suma.GetValue(g);
            Muls(szF, szF, sg, nt);
            Add(bias, bias, szF, nt);
        }
        PipeBarrier<PIPE_V>();
        Sub(pF, acc, bias, nt);
        PipeBarrier<PIPE_V>();
        Cast(oH, pF, RoundMode::CAST_NONE, nt);
        PipeBarrier<PIPE_V>();
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        DataCopy(oGm[n0], oH, nt);
        PipeBarrier<PIPE_ALL>();
    }
}

#ifndef ASCENDC_CPU_DEBUG
void w4a16_gemv_axpy_do(uint32_t b, void *st, uint8_t *act, uint8_t *wq, uint8_t *scales,
                        uint8_t *szs, uint8_t *out, int32_t K, int32_t N, int32_t NT)
{
    w4a16_gemv_axpy<<<b, nullptr, st>>>(act, wq, scales, szs, out, K, N, NT);
}
#endif
