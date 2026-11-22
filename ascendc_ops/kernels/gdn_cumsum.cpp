/**
 * gdn_cumsum — Qwen3.8 GDN 的 chunk 内累计和（chunk_local_cumsum）
 *
 *   g_cum[h, i] = sum_{j = chunk_start(i) .. i} g[h, j]      chunk_size = 64
 *
 * 布局：g, g_cum : [HEADS, SEQ] fp32，SEQ 为 chunk_size 的整数倍。
 * 本实现先覆盖「SEQ == 1 个 chunk」的 prefill 情况（SEQ=64），
 * 更大 SEQ 时按 64 逐 chunk 循环即可（下一步扩展）。
 */
#include "kernel_operator.h"

using namespace AscendC;

constexpr int32_t SEQ = 64;
constexpr int32_t HEADS = 48;
constexpr int32_t CHUNK = 64;
constexpr int32_t BLOCK_NUM = 8;
constexpr int32_t HEADS_PER_BLOCK = HEADS / BLOCK_NUM;   // 6

class KernelGdnCumsum {
public:
    __aicore__ inline KernelGdnCumsum() {}

    __aicore__ inline void Init(GM_ADDR g, GM_ADDR g_cum)
    {
        headStart = static_cast<int32_t>(AscendC::GetBlockIdx()) * HEADS_PER_BLOCK;
        gGm.SetGlobalBuffer((__gm__ float *)g, HEADS * SEQ);
        oGm.SetGlobalBuffer((__gm__ float *)g_cum, HEADS * SEQ);
        pipe.InitBuffer(buf, CHUNK * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        LocalTensor<float> b = buf.Get<float>();
        for (int32_t i = 0; i < HEADS_PER_BLOCK; ++i) {
            int32_t off = (headStart + i) * SEQ;
            DataCopy(b, gGm[off], CHUNK);
            PipeBarrier<PIPE_ALL>();

            float acc = 0.0f;
            for (int32_t k = 0; k < CHUNK; ++k) {
                acc += b.GetValue(k);
                b.SetValue(k, acc);
            }
            PipeBarrier<PIPE_ALL>();

            DataCopy(oGm[off], b, CHUNK);
            PipeBarrier<PIPE_ALL>();
        }
    }

private:
    int32_t headStart{0};
    TPipe pipe;
    TBuf<TPosition::VECCALC> buf;
    GlobalTensor<float> gGm, oGm;
};

extern "C" __global__ __aicore__ void gdn_cumsum(GM_ADDR g, GM_ADDR g_cum)
{
    KernelGdnCumsum op;
    op.Init(g, g_cum);
    op.Process();
}

#ifndef ASCENDC_CPU_DEBUG
void gdn_cumsum_do(uint32_t blockDim, void *stream, uint8_t *g, uint8_t *g_cum)
{
    gdn_cumsum<<<blockDim, nullptr, stream>>>(g, g_cum);
}
#endif
