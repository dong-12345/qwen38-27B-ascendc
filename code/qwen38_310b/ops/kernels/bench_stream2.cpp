/**
 * bench_stream2 — GM<->UB 带宽细扫（perf-scout2 新增，2026-11-19）
 *
 * 目的：复核「GM->UB 10.2GB/s 是 DDR 上限还是 16KB 小请求/队列深度导致的 MTE2 效率损失」。
 *       tile 大小运行时传入（256B ~ 112KB），支持只读 / 只写 / 读写三种方向。
 *
 * 参数：
 *   src, dst      GM 指针
 *   tileHalf      每次搬运的元素数（half，16 的倍数）
 *   tilesPerBlock 每 block 搬运次数
 *   mode          0=读+向量累加  1=只读(touch)  2=只写  3=读+写
 *   strideHalf    block 之间源地址间隔（元素）；=0 表示所有 block 读同一段
 *   windowHalf    >0 时源地址在 windowHalf 元素内回绕（测 L2 命中带宽）
 *
 * UB 预算（253952B）：按 mode 动态分配，见 host 侧过滤规则：
 *   mode 0: 3*tile <= 240KB      mode 1: 2*tile+256B <= 250KB
 *   mode 2: 2*tile <= 248KB      mode 3: 4*tile <= 248KB
 */
#include "kernel_operator.h"

using namespace AscendC;

constexpr int32_t QUEUE_DEPTH = 2;

class KernelBenchStream2 {
public:
    __aicore__ inline KernelBenchStream2() {}

    __aicore__ inline void Init(GM_ADDR src, GM_ADDR dst, int32_t tileHalf, int32_t strideHalf,
                                int32_t mode)
    {
        blockIdx = static_cast<int32_t>(AscendC::GetBlockIdx());
        int64_t base = (int64_t)blockIdx * (int64_t)strideHalf;
        srcGm.SetGlobalBuffer((__gm__ half *)src + base, (int64_t)tileHalf * 1024);
        dstGm.SetGlobalBuffer((__gm__ half *)dst + (int64_t)blockIdx * tileHalf + base, tileHalf);
        if (mode == 2 || mode == 3) {
            pipe.InitBuffer(outQueue, QUEUE_DEPTH, tileHalf * sizeof(half));
        }
        if (mode != 2) {
            pipe.InitBuffer(inQueue, QUEUE_DEPTH, tileHalf * sizeof(half));
        }
        if (mode == 0) {
            pipe.InitBuffer(accBuf, tileHalf * sizeof(half));
        } else {
            pipe.InitBuffer(touchBuf, 256);   // 只用来"消费"数据，避免被优化
        }
    }

    __aicore__ inline void Process(int32_t tileHalf, int32_t tilesPerBlock, int32_t mode,
                                   int32_t windowHalf)
    {
        if (mode == 2) {
            LocalTensor<half> x = outQueue.AllocTensor<half>();
            Duplicate(x, (half)1.0, tileHalf);
            for (int32_t i = 0; i < tilesPerBlock; ++i) {
                int64_t off = (int64_t)i * tileHalf;
                if (windowHalf > 0) {
                    off = off % windowHalf;
                }
                LocalTensor<half> y = outQueue.AllocTensor<half>();
                Adds(y, x, (half)0.0, tileHalf);
                outQueue.EnQue(y);
                y = outQueue.DeQue<half>();
                DataCopy(dstGm[off], y, tileHalf);
                outQueue.FreeTensor(y);
            }
            outQueue.FreeTensor(x);
            PipeBarrier<PIPE_ALL>();
            return;
        }

        LocalTensor<half> acc = (mode == 0) ? accBuf.Get<half>() : touchBuf.Get<half>();
        if (mode == 0) {
            Duplicate(acc, (half)0.0, tileHalf);
        }
        PipeBarrier<PIPE_V>();

        for (int32_t i = 0; i < tilesPerBlock; ++i) {
            int64_t off = (int64_t)i * tileHalf;
            if (windowHalf > 0) {
                off = off % windowHalf;
            }
            LocalTensor<half> x = inQueue.AllocTensor<half>();
            DataCopy(x, srcGm[off], tileHalf);
            inQueue.EnQue(x);
            x = inQueue.DeQue<half>();
            if (mode == 0) {
                Add(acc, acc, x, tileHalf);
            } else if (mode == 1) {
                acc.SetValue(0, x.GetValue(0));
            } else {
                LocalTensor<half> y = outQueue.AllocTensor<half>();
                Adds(y, x, (half)0.0, tileHalf);
                outQueue.EnQue(y);
                y = outQueue.DeQue<half>();
                DataCopy(dstGm[off], y, tileHalf);
                outQueue.FreeTensor(y);
            }
            inQueue.FreeTensor(x);
        }
        PipeBarrier<PIPE_ALL>();
        if (mode == 0 || mode == 1) {
            DataCopy(dstGm, acc, (mode == 0) ? tileHalf : 128);
        }
        PipeBarrier<PIPE_ALL>();
    }

private:
    int32_t blockIdx{0};
    TPipe pipe;
    TQue<TPosition::VECIN, QUEUE_DEPTH> inQueue;
    TQue<TPosition::VECOUT, QUEUE_DEPTH> outQueue;
    TBuf<TPosition::VECCALC> accBuf, touchBuf;
    GlobalTensor<half> srcGm, dstGm;
};

extern "C" __global__ __aicore__ void bench_stream2(GM_ADDR src, GM_ADDR dst, int32_t tileHalf,
                                                    int32_t tilesPerBlock, int32_t mode,
                                                    int32_t strideHalf, int32_t windowHalf)
{
    KernelBenchStream2 op;
    op.Init(src, dst, tileHalf, strideHalf, mode);
    op.Process(tileHalf, tilesPerBlock, mode, windowHalf);
}

#ifndef ASCENDC_CPU_DEBUG
void bench_stream2_do(uint32_t blockDim, void *stream, uint8_t *src, uint8_t *dst,
                      int32_t tileHalf, int32_t tilesPerBlock, int32_t mode,
                      int32_t strideHalf, int32_t windowHalf)
{
    bench_stream2<<<blockDim, nullptr, stream>>>(src, dst, tileHalf, tilesPerBlock, mode,
                                                strideHalf, windowHalf);
}
#endif
