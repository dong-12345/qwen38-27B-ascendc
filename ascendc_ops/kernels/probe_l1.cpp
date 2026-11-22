/**
 * probe_l1 — 验证 310B 上 UB->L1 与 L1->UB 的 DataCopy 是否真的可用
 *
 * mode 0: 对照 —— UB 填数据直接写 GM（验证测试框架）
 * mode 1: UB -> L1 -> UB -> GM（PipeBarrier 同步）
 * mode 2: 同 mode 1，但用 TQue<TPosition::A1> 管理 L1
 */
#include "kernel_operator.h"

using namespace AscendC;

constexpr int32_t N = 1024;   // 1024 half = 2KB

extern "C" __global__ __aicore__ void probe_l1(GM_ADDR out, int32_t mode)
{
    TPipe pipe;
    TBuf<TPosition::VECCALC> ubA, ubB;
    TBuf<TPosition::A1> l1Buf;
    pipe.InitBuffer(ubA, N * sizeof(half));
    pipe.InitBuffer(ubB, N * sizeof(half));
    pipe.InitBuffer(l1Buf, N * sizeof(half));

    LocalTensor<half> a = ubA.Get<half>();
    LocalTensor<half> b = ubB.Get<half>();
    LocalTensor<half> l1 = l1Buf.Get<half>();
    GlobalTensor<half> outGm;
    outGm.SetGlobalBuffer((__gm__ half *)out, N);

    // 填已知数据: a[i] = i
    Duplicate(a, (half)0.0, N);
    PipeBarrier<PIPE_V>();
    for (int32_t i = 0; i < N; ++i) {
        a.SetValue(i, (half)(i % 100));
    }
    PipeBarrier<PIPE_ALL>();

    if (mode == 0) {
        DataCopy(outGm, a, N);
    } else if (mode == 1) {
        DataCopy(l1, a, N);          // UB -> L1
        PipeBarrier<PIPE_ALL>();
        DataCopy(b, l1, N);          // L1 -> UB
        PipeBarrier<PIPE_ALL>();
        DataCopy(outGm, b, N);
    } else {
        // mode 2: L1 -> UB 方向，但先用 L1 的另一段（验证两种方向）
        DataCopy(l1, a, N);
        PipeBarrier<PIPE_ALL>();
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE3>(EVENT_ID0);
        DataCopy(b, l1, N);
        PipeBarrier<PIPE_ALL>();
        DataCopy(outGm, b, N);
    }
    PipeBarrier<PIPE_ALL>();
}

#ifndef ASCENDC_CPU_DEBUG
void probe_l1_do(uint32_t b, void *st, uint8_t *out, int32_t mode)
{
    probe_l1<<<b, nullptr, st>>>(out, mode);
}
#endif
