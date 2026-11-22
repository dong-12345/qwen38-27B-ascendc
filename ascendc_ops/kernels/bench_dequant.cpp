/**
 * bench_dequant — int4 权重 -> fp16 反量化写回 GM 的纯带宽成本（perf-scout2 新增，2026-11-19）
 *
 * 目的：量化「Cube 两段式」第一段的真实成本：
 *        读 int4（0.5 B/param） + 写 fp16（2 B/param） = 2.5 B/param
 *       Qwen3.8-27B ≈ 27G 参数 → 读 13.5GB + 写 54GB = 67.5GB，这是两段式每 pass 的地板。
 *
 * 输入：src 每个 16bit 字装 4 个 int4（相位布局，与 w4a16_gemv 一致，但这里不校验数值）
 * 输出：dst 每个字展开成 4 个连续的 int16->fp16 值（低 4 位平面优先）
 *
 * 参数：wordsPerTile 每 tile 的 16bit 字数（建议 8192 = 16KB 读）
 *       tilesPerBlock 每 block tile 数
 */
#include "kernel_operator.h"

using namespace AscendC;

constexpr int32_t QUEUE_DEPTH = 2;

class KernelBenchDequant {
public:
    __aicore__ inline KernelBenchDequant() {}

    __aicore__ inline void Init(GM_ADDR src, GM_ADDR dst, int32_t wordsPerTile)
    {
        blockIdx = static_cast<int32_t>(AscendC::GetBlockIdx());
        srcGm.SetGlobalBuffer((__gm__ uint16_t *)src, (int64_t)wordsPerTile * 1024);
        dstGm.SetGlobalBuffer((__gm__ half *)dst, (int64_t)wordsPerTile * 4 * 1024);
        pipe.InitBuffer(inQueue, QUEUE_DEPTH, wordsPerTile * sizeof(uint16_t));
        pipe.InitBuffer(phBuf, 4 * wordsPerTile * sizeof(uint16_t));
        pipe.InitBuffer(outBuf, 4 * wordsPerTile * sizeof(half));
        pipe.InitBuffer(maskBuf, wordsPerTile * sizeof(uint16_t));
    }

    __aicore__ inline void Process(int32_t wordsPerTile, int32_t tilesPerBlock)
    {
        LocalTensor<uint16_t> mask = maskBuf.Get<uint16_t>();
        Duplicate(mask, (uint16_t)0x000F, wordsPerTile);
        LocalTensor<uint16_t> ph = phBuf.Get<uint16_t>();
        LocalTensor<half> out = outBuf.Get<half>();
        int32_t W = wordsPerTile;

        for (int32_t t = 0; t < tilesPerBlock; ++t) {
            LocalTensor<uint16_t> x = inQueue.AllocTensor<uint16_t>();
            DataCopy(x, srcGm[(int64_t)t * W], W);
            inQueue.EnQue(x);
            x = inQueue.DeQue<uint16_t>();

            And(ph[0], x, mask, W);
            ShiftRight(ph[W], x, (uint16_t)4, W);
            And(ph[W], ph[W], mask, W);
            ShiftRight(ph[2 * W], x, (uint16_t)8, W);
            And(ph[2 * W], ph[2 * W], mask, W);
            ShiftRight(ph[3 * W], x, (uint16_t)12, W);
            inQueue.FreeTensor(x);
            PipeBarrier<PIPE_V>();
            Cast(out, ph.ReinterpretCast<int16_t>(), RoundMode::CAST_NONE, 4 * W);
            SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
            WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
            // 分 4 次写（每次 W 个 half = 2W 字节），避免单条 DataCopy 过大
            for (int32_t j = 0; j < 4; ++j) {
                DataCopy(dstGm[(int64_t)t * 4 * W + (int64_t)j * W], out[j * W], W);
            }
            PipeBarrier<PIPE_MTE3>();
        }
        PipeBarrier<PIPE_ALL>();
    }

private:
    int32_t blockIdx{0};
    TPipe pipe;
    TQue<TPosition::VECIN, QUEUE_DEPTH> inQueue;
    TBuf<TPosition::VECCALC> phBuf, outBuf, maskBuf;
    GlobalTensor<uint16_t> srcGm;
    GlobalTensor<half> dstGm;
};

extern "C" __global__ __aicore__ void bench_dequant(GM_ADDR src, GM_ADDR dst, int32_t wordsPerTile,
                                                    int32_t tilesPerBlock)
{
    KernelBenchDequant op;
    op.Init(src, dst, wordsPerTile);
    op.Process(wordsPerTile, tilesPerBlock);
}

#ifndef ASCENDC_CPU_DEBUG
void bench_dequant_do(uint32_t blockDim, void *stream, uint8_t *src, uint8_t *dst,
                      int32_t wordsPerTile, int32_t tilesPerBlock)
{
    bench_dequant<<<blockDim, nullptr, stream>>>(src, dst, wordsPerTile, tilesPerBlock);
}
#endif
