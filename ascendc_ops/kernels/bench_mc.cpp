/**
 * bench_mc — 多 block 带宽/并行度测试（perf-scout 新增，2026-11-19）
 *
 * 目的：回答「blockDim>1 到底是真并行还是串行排队」以及「GM->UB 单核 10.2GB/s
 *       是 DDR 上限还是单核 MTE2 上限」。
 *
 * 关键设计：**每个 block 的工作量固定**（tilesPerBlock * tileHalf），
 *   - 若多个 block 真并行：总墙钟时间 ≈ 单 block 时间（带宽足够时）或
 *     aggregate 带宽随 blockDim 上升；
 *   - 若串行排队：总墙钟时间 ≈ blockDim × 单 block 时间，aggregate 带宽恒定。
 *
 * 参数：
 *   tileHalf        每个 tile 的元素数（half），必须 16 的倍数（32B 对齐）
 *   tilesPerBlock   每个 block 搬的 tile 数
 *   mode            0 = 搬运+向量累加；1 = 纯搬运（只 touch 一个元素）
 *   strideHalf      block 之间源地址的间隔（元素数）；=0 表示所有 block 读同一段
 *   windowHalf      >0 时，块内读地址在 windowHalf 范围内回绕（用于测 L2 命中带宽）
 *
 * out 布局：每个 block 写 tileHalf 个 half 到 out + blockIdx*tileHalf。
 */
#include "kernel_operator.h"

using namespace AscendC;

constexpr int32_t QUEUE_DEPTH = 4;

class KernelBenchMc {
public:
    __aicore__ inline KernelBenchMc() {}

    __aicore__ inline void Init(GM_ADDR src, GM_ADDR out, int32_t tileHalf, int32_t strideHalf)
    {
        blockIdx = static_cast<int32_t>(AscendC::GetBlockIdx());
        int64_t base = (int64_t)blockIdx * (int64_t)strideHalf;
        // 每个 block 的源起点
        srcGm.SetGlobalBuffer((__gm__ half *)src + base, tileHalf);
        outGm.SetGlobalBuffer((__gm__ half *)out + (int64_t)blockIdx * tileHalf, tileHalf);
        pipe.InitBuffer(inQueue, QUEUE_DEPTH, tileHalf * sizeof(half));
        pipe.InitBuffer(accBuf, tileHalf * sizeof(half));
    }

    __aicore__ inline void Process(int32_t tileHalf, int32_t tilesPerBlock, int32_t mode,
                                   int32_t windowHalf)
    {
        LocalTensor<half> acc = accBuf.Get<half>();
        Duplicate(acc, (half)0.0, tileHalf);
        PipeBarrier<PIPE_V>();

        for (int32_t i = 0; i < tilesPerBlock; ++i) {
            int32_t off = i * tileHalf;
            if (windowHalf > 0) {
                off = off % windowHalf;   // L2 命中模式：窗口内回绕
            }
            LocalTensor<half> x = inQueue.AllocTensor<half>();
            DataCopy(x, srcGm[off], tileHalf);
            inQueue.EnQue(x);
            x = inQueue.DeQue<half>();
            if (mode == 0) {
                Add(acc, acc, x, tileHalf);
            } else {
                acc.SetValue(0, x.GetValue(0));   // 保证数据被消费
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

extern "C" __global__ __aicore__ void bench_mc(GM_ADDR src, GM_ADDR out, int32_t tileHalf,
                                               int32_t tilesPerBlock, int32_t mode,
                                               int32_t strideHalf, int32_t windowHalf)
{
    KernelBenchMc op;
    op.Init(src, out, tileHalf, strideHalf);
    op.Process(tileHalf, tilesPerBlock, mode, windowHalf);
}

#ifndef ASCENDC_CPU_DEBUG
void bench_mc_do(uint32_t blockDim, void *stream, uint8_t *src, uint8_t *out, int32_t tileHalf,
                 int32_t tilesPerBlock, int32_t mode, int32_t strideHalf, int32_t windowHalf)
{
    bench_mc<<<blockDim, nullptr, stream>>>(src, out, tileHalf, tilesPerBlock, mode, strideHalf,
                                            windowHalf);
}
#endif
