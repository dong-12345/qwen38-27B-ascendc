/**
 * probe_gather — 验证 m200 上 Gather（按索引向量取数）是否可用、offset 单位是字节还是元素
 *
 * 输入：table [16] fp32（值 = i*100），idx 由 kernel 内部构造；
 * 输出 A：offset = (k%16)*4（假设字节偏移）→ 期望 table[k%16]
 * 输出 B：offset = (k%16)  （假设元素偏移）→ 期望 table[k%16]
 */
#include "kernel_operator.h"

using namespace AscendC;

constexpr int32_t N = 8192;

extern "C" __global__ __aicore__ void probe_gather(GM_ADDR outA, GM_ADDR outB, int32_t mode)
{
    TPipe pipe;
    GlobalTensor<float> oaGm, obGm;
    oaGm.SetGlobalBuffer((__gm__ float *)outA, N);
    obGm.SetGlobalBuffer((__gm__ float *)outB, N);
    TBuf<TPosition::VECCALC> tblBuf, idxBuf, aBuf, bBuf, outHBuf;
    pipe.InitBuffer(tblBuf, 16 * sizeof(float));
    pipe.InitBuffer(idxBuf, N * sizeof(uint32_t));
    pipe.InitBuffer(aBuf, N * sizeof(float));
    pipe.InitBuffer(bBuf, N * sizeof(float));

    LocalTensor<float> tbl = tblBuf.Get<float>();
    LocalTensor<uint32_t> idx = idxBuf.Get<uint32_t>();
    LocalTensor<float> a = aBuf.Get<float>();
    LocalTensor<float> b = bBuf.Get<float>();

    for (int32_t i = 0; i < 16; ++i) {
        tbl.SetValue(i, (float)(i * 100));
    }
    for (int32_t k = 0; k < N; ++k) {
        idx.SetValue(k, (uint32_t)((k % 16) * 4));      // 字节偏移假设
    }
    PipeBarrier<PIPE_ALL>();
    AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID0);

    Gather(a, tbl, idx, 0, (uint32_t)N);
    PipeBarrier<PIPE_V>();
    Duplicate(b, 0.0f, N);
    PipeBarrier<PIPE_V>();
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
    DataCopy(oaGm, a, N);
    PipeBarrier<PIPE_ALL>();

    // 元素偏移版本
    for (int32_t k = 0; k < N; ++k) {
        idx.SetValue(k, (uint32_t)(k % 16));
    }
    PipeBarrier<PIPE_ALL>();
    AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
    // 模拟 gemv：256 元素 fp32 表，idx 8192 个（钳位），count=8192
    for (int32_t i = 0; i < 256; ++i) { tbl.SetValue(i, (float)i); }
    for (int32_t k = 0; k < N; ++k) {
        int32_t g = k / 32; if (g > 255) g = 255;
        idx.SetValue(k, (uint32_t)(g * 4));
    }
    PipeBarrier<PIPE_ALL>();
    AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
    TBuf<TPosition::VECCALC> bigBuf;
    pipe.InitBuffer(bigBuf, N * sizeof(float));
    LocalTensor<float> big = bigBuf.Get<float>();
    Gather(big, tbl, idx, 0, (uint32_t)N);
    PipeBarrier<PIPE_V>();
    for (int32_t i = 0; i < 64; ++i) { a.SetValue(i, big.GetValue(i)); }
    PipeBarrier<PIPE_ALL>();
    AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
    Duplicate(b, 0.0f, N);
    PipeBarrier<PIPE_V>();
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
    DataCopy(obGm, b, N);
    PipeBarrier<PIPE_ALL>();
    (void)mode;
}

#ifndef ASCENDC_CPU_DEBUG
void probe_gather_do(uint32_t blk, void *st, uint8_t *outA, uint8_t *outB, int32_t mode)
{
    probe_gather<<<blk, nullptr, st>>>(outA, outB, mode);
}
#endif

