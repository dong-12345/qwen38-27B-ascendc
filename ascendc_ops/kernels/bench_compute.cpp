/**
 * bench_compute — 纯向量计算（无 GM 访问），用于判断 310B 的 AIV 核数量/并行度
 *
 * 每个 block 独立做 loops 次 16KB 的向量 Add，用总耗时随 blockDim 的变化判断是否真并行：
 *   - 若 t(8 blocks) ≈ t(1 block)：8 个 block 并行（有 8 个可用核）
 *   - 若 t(8 blocks) ≈ 8 * t(1 block)：被串行化（实际只有 1 个核在跑）
 */
#include "kernel_operator.h"

using namespace AscendC;

constexpr int32_t TILE_HALF = 8192;    // 16KB
constexpr int32_t LOOPS = 2048;

class KernelBenchCompute {
public:
    __aicore__ inline KernelBenchCompute() {}

    __aicore__ inline void Init(GM_ADDR out)
    {
        blockIdx = static_cast<int32_t>(AscendC::GetBlockIdx());
        outGm.SetGlobalBuffer((__gm__ half *)out + blockIdx * TILE_HALF, TILE_HALF);
        pipe.InitBuffer(bufA, TILE_HALF * sizeof(half));
        pipe.InitBuffer(bufB, TILE_HALF * sizeof(half));
    }

    __aicore__ inline void Process()
    {
        LocalTensor<half> a = bufA.Get<half>();
        LocalTensor<half> b = bufB.Get<half>();
        Duplicate(a, (half)1.0, TILE_HALF);
        Duplicate(b, (half)0.5, TILE_HALF);
        PipeBarrier<PIPE_V>();
        for (int32_t i = 0; i < LOOPS; ++i) {
            // 依赖链，避免被优化；每轮 2 个 16KB 向量 op
            Add(a, a, b, TILE_HALF);
            Mul(b, a, b, TILE_HALF);
        }
        PipeBarrier<PIPE_ALL>();
        DataCopy(outGm, a, TILE_HALF);
        PipeBarrier<PIPE_ALL>();
    }

private:
    int32_t blockIdx{0};
    TPipe pipe;
    TBuf<TPosition::VECCALC> bufA, bufB;
    GlobalTensor<half> outGm;
};

extern "C" __global__ __aicore__ void bench_compute(GM_ADDR out)
{
    KernelBenchCompute op;
    op.Init(out);
    op.Process();
}

#ifndef ASCENDC_CPU_DEBUG
void bench_compute_do(uint32_t blockDim, void *stream, uint8_t *out)
{
    bench_compute<<<blockDim, nullptr, stream>>>(out);
}
#endif
