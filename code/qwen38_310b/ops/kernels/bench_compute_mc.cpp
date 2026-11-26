/**
 * bench_compute_mc — 纯向量计算并行度测试（perf-scout 新增，2026-11-19）
 *
 * **每个 block 的工作量固定**（loops * 2 条 16KB 向量指令），扫 blockDim：
 *   - t(blockDim) ≈ t(1)      -> 有 blockDim 个向量核真并行
 *   - t(blockDim) ≈ N*t(1)    -> 只有一个向量核，多 block 串行排队
 *
 * mode 0: 依赖链 Add+Mul（原 bench_compute 写法，测延迟/依赖）
 * mode 1: 独立 Add（a=b+c，无跨迭代依赖，测向量吞吐上限）
 * mode 2: fp32 独立 Add
 * mode 3: 半精度 Muls 标量乘（单操作数带宽）
 */
#include "kernel_operator.h"

using namespace AscendC;

constexpr int32_t TILE_HALF = 8192;    // 16KB fp16

class KernelBenchComputeMc {
public:
    __aicore__ inline KernelBenchComputeMc() {}

    __aicore__ inline void Init(GM_ADDR out)
    {
        blockIdx = static_cast<int32_t>(AscendC::GetBlockIdx());
        outGm.SetGlobalBuffer((__gm__ half *)out + (int64_t)blockIdx * TILE_HALF, TILE_HALF);
        pipe.InitBuffer(bufA, TILE_HALF * sizeof(half));
        pipe.InitBuffer(bufB, TILE_HALF * sizeof(half));
        pipe.InitBuffer(bufC, TILE_HALF * sizeof(half));
        pipe.InitBuffer(bufF, TILE_HALF * sizeof(float));
    }

    __aicore__ inline void Process(int32_t loops, int32_t mode)
    {
        LocalTensor<half> a = bufA.Get<half>();
        LocalTensor<half> b = bufB.Get<half>();
        LocalTensor<half> c = bufC.Get<half>();
        LocalTensor<float> f = bufF.Get<float>();
        Duplicate(a, (half)1.0, TILE_HALF);
        Duplicate(b, (half)0.5, TILE_HALF);
        Duplicate(c, (half)0.25, TILE_HALF);
        PipeBarrier<PIPE_V>();

        if (mode == 0) {
            for (int32_t i = 0; i < loops; ++i) {
                Add(a, a, b, TILE_HALF);
                Mul(b, a, b, TILE_HALF);
            }
        } else if (mode == 1) {
            for (int32_t i = 0; i < loops; ++i) {
                Add(a, b, c, TILE_HALF);
            }
        } else if (mode == 2) {
            for (int32_t i = 0; i < loops; ++i) {
                Add(f, f, f, TILE_HALF);   // fp32，就地累加
            }
        } else {
            for (int32_t i = 0; i < loops; ++i) {
                Muls(a, b, (half)1.5, TILE_HALF);
            }
        }
        PipeBarrier<PIPE_ALL>();
        DataCopy(outGm, a, TILE_HALF);
        PipeBarrier<PIPE_ALL>();
    }

private:
    int32_t blockIdx{0};
    TPipe pipe;
    TBuf<TPosition::VECCALC> bufA, bufB, bufC, bufF;
    GlobalTensor<half> outGm;
};

extern "C" __global__ __aicore__ void bench_compute_mc(GM_ADDR out, int32_t loops, int32_t mode)
{
    KernelBenchComputeMc op;
    op.Init(out);
    op.Process(loops, mode);
}

#ifndef ASCENDC_CPU_DEBUG
void bench_compute_mc_do(uint32_t blockDim, void *stream, uint8_t *out, int32_t loops,
                         int32_t mode)
{
    bench_compute_mc<<<blockDim, nullptr, stream>>>(out, loops, mode);
}
#endif
