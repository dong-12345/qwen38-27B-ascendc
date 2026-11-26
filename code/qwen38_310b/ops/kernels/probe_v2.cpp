/**
 * probe_v2 — 测量 310B1 向量原语的"边际单次开销"（host 用两个 iters 相减，剔除 setup）
 *
 *   mode  0: fp32 Add(c,c,a,len) + 1 barrier
 *   mode  1: fp32 Add(c,c,a,len)，每 8 次才 1 个 barrier（测 barrier 代价）
 *   mode  2: fp16 Add(c,c,a,len) + 1 barrier
 *   mode  3: Cast int16 -> half (len)
 *   mode  4: Mul fp16 (len) + 1 barrier
 *   mode  5: 32 元素分组树形归约（fp32, 5 步 Add）
 *   mode  6: 32 元素分组树形归约（fp16, 5 步 Add）
 *   mode  7: Gather count=len
 *   mode  8: WholeReduceSum(mask=64, repeat=len/64)
 *   mode  9: DataCopy GM->UB (len float)
 *  mode 10: setup 之外什么都不做（基线）
 *  mode 11: fp16 Mul+Add 混链 4 次（模拟反量化+累加，len）
 */
#include "kernel_operator.h"

using namespace AscendC;

extern "C" __global__ __aicore__ void probe_v2(GM_ADDR gin, GM_ADDR gout, int32_t mode, int32_t len,
                                               int32_t iters)
{
    TPipe pipe;
    TBuf<TPosition::VECCALC> bA, bB, bC, bD, bE;
    pipe.InitBuffer(bA, 32768);   // fp32 x 8192
    pipe.InitBuffer(bB, 32768);
    pipe.InitBuffer(bC, 32768);
    pipe.InitBuffer(bD, 16384);   // fp16 x 8192
    pipe.InitBuffer(bE, 16384);

    LocalTensor<float> a = bA.Get<float>();
    LocalTensor<float> c = bB.Get<float>();
    LocalTensor<uint32_t> idx = bC.Get<uint32_t>();
    LocalTensor<half> h = bD.Get<half>();
    LocalTensor<half> h2 = bE.Get<half>();

    GlobalTensor<float> gi;
    gi.SetGlobalBuffer((__gm__ float *)gin, len);

    DataCopy(a, gi, len);
    Duplicate(c, 0.0f, len);
    Duplicate(h, (half)1.0f, len);
    PipeBarrier<PIPE_ALL>();

    if (mode == 7) {
        for (int32_t j = 0; j < len; ++j) idx.SetValue(j, (uint32_t)(j & 1023));
        AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
    }

    int32_t g32 = len / 32;
    if (mode == 0) {
        for (int32_t it = 0; it < iters; ++it) { Add(c, c, a, len); PipeBarrier<PIPE_V>(); }
    } else if (mode == 1) {
        for (int32_t it = 0; it < iters; it += 8) {
            Add(c, c, a, len); Add(c, c, a, len); Add(c, c, a, len); Add(c, c, a, len);
            Add(c, c, a, len); Add(c, c, a, len); Add(c, c, a, len); Add(c, c, a, len);
            PipeBarrier<PIPE_V>();
        }
    } else if (mode == 2) {
        for (int32_t it = 0; it < iters; ++it) { Add(h, h, h2, len); PipeBarrier<PIPE_V>(); }
    } else if (mode == 3) {
        LocalTensor<int16_t> i16 = bE.Get<int16_t>();
        LocalTensor<half> ho = bD.Get<half>();
        for (int32_t it = 0; it < iters; ++it) { Cast(ho, i16, RoundMode::CAST_NONE, len); PipeBarrier<PIPE_V>(); }
    } else if (mode == 4) {
        for (int32_t it = 0; it < iters; ++it) { Mul(h, h, h2, len); PipeBarrier<PIPE_V>(); }
    } else if (mode == 5) {
        for (int32_t it = 0; it < iters; ++it) {
            Add(c, c, c[16], len - 16);   Add(c, c, c[8], len - 8);
            Add(c, c, c[4], len - 4);     Add(c, c, c[2], len - 2);
            Add(c, c, c[1], len - 1);
            PipeBarrier<PIPE_V>();
        }
    } else if (mode == 6) {
        for (int32_t it = 0; it < iters; ++it) {
            Add(h, h, h[16], len - 16);   Add(h, h, h[8], len - 8);
            Add(h, h, h[4], len - 4);     Add(h, h, h[2], len - 2);
            Add(h, h, h[1], len - 1);
            PipeBarrier<PIPE_V>();
        }
    } else if (mode == 7) {
        for (int32_t it = 0; it < iters; ++it) { Gather(c, a, idx, 0, (uint32_t)len); PipeBarrier<PIPE_V>(); }
    } else if (mode == 8) {
        int32_t rep = len / 64;
        for (int32_t it = 0; it < iters; ++it) {
            WholeReduceSum<float>(c, a, 64, rep, 1, 1, 8);
            PipeBarrier<PIPE_V>();
        }
    } else if (mode == 9) {
        for (int32_t it = 0; it < iters; ++it) { DataCopy(a, gi, len); PipeBarrier<PIPE_ALL>(); }
    } else if (mode == 11) {
        for (int32_t it = 0; it < iters; ++it) {
            Mul(h, h, h2, len);
            Add(h, h, h2, len);
            Mul(h, h, h2, len);
            Add(h, h, h2, len);
            PipeBarrier<PIPE_V>();
        }
    } else {
        for (int32_t it = 0; it < iters; ++it) { PipeBarrier<PIPE_V>(); }
    }
    (void)g32;
    PipeBarrier<PIPE_ALL>();
}

#ifndef ASCENDC_CPU_DEBUG
void probe_v2_do(uint32_t b, void *st, uint8_t *gin, uint8_t *gout, int32_t mode, int32_t len,
                 int32_t iters)
{
    probe_v2<<<b, nullptr, st>>>(gin, gout, mode, len, iters);
}
#endif
