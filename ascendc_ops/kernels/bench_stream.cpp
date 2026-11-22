/**
 * bench_stream — 测 310B 的 GM -> UB 实际读带宽（决定 W4A16 推理的 tok/s 上限）
 *
 * mode = 0 : DataCopy + 向量累加（模拟真实负载）
 * mode = 1 : 只做 DataCopy（测纯拷贝上限）
 */
#include "kernel_operator.h"

using namespace AscendC;

constexpr int32_t QUEUE_DEPTH = 4;

class KernelBenchStream {
public:
    __aicore__ inline KernelBenchStream() {}

    __aicore__ inline void Init(GM_ADDR src, GM_ADDR out, int32_t tileHalf, int32_t tilesPerBlock)
    {
        blockIdx = static_cast<int32_t>(AscendC::GetBlockIdx());
        int32_t per = tilesPerBlock * tileHalf;
        srcGm.SetGlobalBuffer((__gm__ half *)src + blockIdx * per, per);
        outGm.SetGlobalBuffer((__gm__ half *)out + blockIdx * tileHalf, tileHalf);
        pipe.InitBuffer(inQueue, QUEUE_DEPTH, tileHalf * sizeof(half));
        pipe.InitBuffer(accBuf, tileHalf * sizeof(half));
    }

    __aicore__ inline void Process(int32_t tileHalf, int32_t tilesPerBlock, int32_t mode)
    {
        LocalTensor<half> acc = accBuf.Get<half>();
        Duplicate(acc, (half)0.0, tileHalf);
        PipeBarrier<PIPE_V>();

        for (int32_t i = 0; i < tilesPerBlock; ++i) {
            LocalTensor<half> x = inQueue.AllocTensor<half>();
            DataCopy(x, srcGm[i * tileHalf], tileHalf);
            inQueue.EnQue(x);
            x = inQueue.DeQue<half>();
            if (mode == 0) {
                Add(acc, acc, x, tileHalf);
            } else {
                // 触碰到数据，避免被优化掉；只做一次极便宜的标量写
                acc.SetValue(0, x.GetValue(0));
            }
            inQueue.FreeTensor(x);
        }
        PipeBarrier<PIPE_ALL>();
        DataCopy(outGm, acc, tileHalf);
        PipeBarrier<PIPE_ALL>();
    }

private:
    int32_t blockIdx{0};
    TPipe pipe;
    TQue<TPosition::VECIN, QUEUE_DEPTH> inQueue;
    TBuf<TPosition::VECCALC> accBuf;
    GlobalTensor<half> srcGm, outGm;
};

extern "C" __global__ __aicore__ void bench_stream(GM_ADDR src, GM_ADDR out,
                                                   int32_t tileHalf, int32_t tilesPerBlock,
                                                   int32_t mode)
{
    KernelBenchStream op;
    op.Init(src, out, tileHalf, tilesPerBlock);
    op.Process(tileHalf, tilesPerBlock, mode);
}

#ifndef ASCENDC_CPU_DEBUG
void bench_stream_do(uint32_t blockDim, void *stream, uint8_t *src, uint8_t *out,
                     int32_t tileHalf, int32_t tilesPerBlock, int32_t mode)
{
    bench_stream<<<blockDim, nullptr, stream>>>(src, out, tileHalf, tilesPerBlock, mode);
}
#endif
