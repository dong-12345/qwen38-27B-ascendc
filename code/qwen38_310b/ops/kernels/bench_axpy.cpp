/**
 * bench_axpy — 验证"沿 N 向量化 + Axpy"结构的带宽上限（310B decode 提速关键实验）
 *
 * 目标形态（社区同板项目 llm_simple/minicpm 在同一块板子上验证过）：
 *   for n-tile (NT 宽):
 *     accF[NT] = 0
 *     for group g (32 个 k):
 *       权重按 [g][kp][n] 打包（1 字节 = k=2kp,2kp+1 两个半字节）→ 一次搬 16*NT 字节（NT=512: 8KB）
 *       拆半字节/转 half → 每行 NT 宽的 fp16
 *       for k in 0..31: Axpy(partH, wHalf[k], act[g*32+k], NT)   ← 沿 N 一次 NT 元素，无需归约
 *       partF = partH * scale[g][n0..]；accF += partF
 *     out[n0..n0+NT) = accF
 *
 * 本 bench 只测**结构效率**（数值不重要：半字节拆分用 1 次 Cast 近似），
 * 若 8–12 GB/s，则"重排权重 + Axpy-N kernel"这条路成立。
 */
#include "kernel_operator.h"

using namespace AscendC;

extern "C" __global__ __aicore__ void bench_axpy(GM_ADDR act, GM_ADDR wq, GM_ADDR scales, GM_ADDR out,
                                                 int32_t K, int32_t N, int32_t NT)
{
    TPipe pipe;
    const int32_t NG = K / 32;

    GlobalTensor<half> actGm, oGm;
    GlobalTensor<uint8_t> wGm;
    actGm.SetGlobalBuffer((__gm__ half *)act, K);
    wGm.SetGlobalBuffer((__gm__ uint8_t *)wq, (int64_t)N * (K / 4) * 2);
    oGm.SetGlobalBuffer((__gm__ half *)out, N);

    TBuf<TPosition::VECCALC> aBuf, wPackBuf, wHalfBuf, pHBuf, pFBuf, accBuf, oBuf;
    pipe.InitBuffer(aBuf, (size_t)K * 4);
    pipe.InitBuffer(wPackBuf, (size_t)16 * NT);
    pipe.InitBuffer(wHalfBuf, (size_t)16 * NT * 2);
    pipe.InitBuffer(pHBuf, (size_t)NT * 2);
    pipe.InitBuffer(pFBuf, (size_t)NT * 4);
    pipe.InitBuffer(accBuf, (size_t)NT * 4);
    pipe.InitBuffer(oBuf, (size_t)NT * 2);

    LocalTensor<float> aF = aBuf.Get<float>();
    LocalTensor<uint8_t> wp = wPackBuf.Get<uint8_t>();
    LocalTensor<half> wh = wHalfBuf.Get<half>();
    LocalTensor<half> pH = pHBuf.Get<half>();
    LocalTensor<float> pF = pFBuf.Get<float>();
    LocalTensor<float> acc = accBuf.Get<float>();
    LocalTensor<half> oH = oBuf.Get<half>();

    {
        TBuf<TPosition::VECCALC> aHBuf;
        pipe.InitBuffer(aHBuf, (size_t)K * 2);
        LocalTensor<half> aH = aHBuf.Get<half>();
        DataCopy(aH, actGm, K);
        PipeBarrier<PIPE_ALL>();
        Cast(aF, aH, RoundMode::CAST_NONE, K);
        PipeBarrier<PIPE_ALL>();
    }

    const int32_t nTiles = (N + NT - 1) / NT;
    const size_t groupBytes = (size_t)N * (K / 4) * 2 / (size_t)NG;
    for (int32_t t = 0; t < nTiles; ++t) {
        const int32_t n0 = t * NT;
        const int32_t nt = (N - n0 >= NT) ? NT : (N - n0);
        Duplicate(acc, 0.0f, NT);
        for (int32_t g = 0; g < NG; ++g) {
            size_t srcOff = (size_t)g * groupBytes + (size_t)n0 * 16;
            DataCopy(wp, wGm[srcOff], (uint16_t)(16 * nt));
            PipeBarrier<PIPE_ALL>();
            // 近似拆半字节：int8 → half（真实实现是 And/ShiftRight+Cast，多 2 条指令）
            Cast(wh, wp.ReinterpretCast<int8_t>(), RoundMode::CAST_NONE, 16 * nt);
            PipeBarrier<PIPE_V>();
            Duplicate(pH, (half)0.0f, NT);
            for (int32_t k = 0; k < 32; ++k) {
                AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
                half ak = (half)aF.GetValue(g * 32 + k);
                AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
                Axpy(pH, wh[(size_t)(k & 15) * NT], ak, NT);
            }
            PipeBarrier<PIPE_V>();
            Cast(pF, pH, RoundMode::CAST_NONE, NT);
            PipeBarrier<PIPE_V>();
            Add(acc, acc, pF, NT);
        }
        PipeBarrier<PIPE_V>();
        Cast(oH, acc, RoundMode::CAST_NONE, nt);
        PipeBarrier<PIPE_V>();
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        DataCopy(oGm[n0], oH, nt);
        PipeBarrier<PIPE_ALL>();
    }
}

#ifndef ASCENDC_CPU_DEBUG
void bench_axpy_do(uint32_t b, void *st, uint8_t *act, uint8_t *wq, uint8_t *scales, uint8_t *out,
                   int32_t K, int32_t N, int32_t NT)
{
    bench_axpy<<<b, nullptr, st>>>(act, wq, scales, out, K, N, NT);
}
#endif
