/**
 * probe_ubl1 — 探测 310B 上是否存在 UB(VECCALC) -> L1(A1) 的 DataCopy 通路。
 *
 * 这条通路是「向量核反量化 -> L1 -> L0 -> Cube MMAD」的关键：
 * 若能编译并运行，W4A16 的 MAC 可以交给 Cube（算力提升 ~2 个数量级），
 * 若不能，只能全向量核实现。
 */
#include "kernel_operator.h"

using namespace AscendC;

constexpr int32_t N = 512;   // 512 half = 1KB

class KernelProbeUbL1 {
public:
    __aicore__ inline KernelProbeUbL1() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y)
    {
        xGm.SetGlobalBuffer((__gm__ half *)x, N);
        yGm.SetGlobalBuffer((__gm__ half *)y, N);
        pipe.InitBuffer(inQueue, 1, N * sizeof(half));
        pipe.InitBuffer(l1Buf, N * sizeof(half) * 2);
        pipe.InitBuffer(outQueue, 1, N * sizeof(half));
    }

    __aicore__ inline void Process()
    {
        LocalTensor<half> ub = inQueue.AllocTensor<half>();
        DataCopy(ub, xGm, N);
        inQueue.EnQue(ub);
        ub = inQueue.DeQue<half>();

        LocalTensor<half> l1 = l1Buf.Get<half>();
        // 关键探测：UB -> L1
        DataCopy(l1, ub, N);
        PipeBarrier<PIPE_ALL>();

        // L1 -> UB
        LocalTensor<half> out = outQueue.AllocTensor<half>();
        DataCopy(out, l1, N);
        outQueue.EnQue(out);
        out = outQueue.DeQue<half>();
        DataCopy(yGm, out, N);
        outQueue.FreeTensor(out);
        inQueue.FreeTensor(ub);
        PipeBarrier<PIPE_ALL>();
    }

private:
    TPipe pipe;
    TQue<TPosition::VECIN, 1> inQueue;
    TQue<TPosition::VECOUT, 1> outQueue;
    TBuf<TPosition::A1> l1Buf;
    GlobalTensor<half> xGm, yGm;
};

extern "C" __global__ __aicore__ void probe_ubl1(GM_ADDR x, GM_ADDR y)
{
    KernelProbeUbL1 op;
    op.Init(x, y);
    op.Process();
}

#ifndef ASCENDC_CPU_DEBUG
void probe_ubl1_do(uint32_t blockDim, void *stream, uint8_t *x, uint8_t *y)
{
    probe_ubl1<<<blockDim, nullptr, stream>>>(x, y);
}
#endif
