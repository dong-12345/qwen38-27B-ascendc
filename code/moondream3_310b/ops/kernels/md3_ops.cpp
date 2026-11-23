/**
 * md3_ops.cpp — Moondream 3 Preview 新增 AscendC 算子（Ascend 310B1）
 */
#include "kernel_operator.h"

using namespace AscendC;

static constexpr int32_t MD3_D = 64;
static constexpr int32_t MD3_RH = 16;
static constexpr int32_t MD3_H = 32;
static constexpr int32_t MD3_MAXT = 4096;

// ---------------- LayerNorm ----------------
extern "C" __global__ __aicore__ void layernorm_f16(GM_ADDR x, GM_ADDR w, GM_ADDR b,
                                                    GM_ADDR y, int32_t H, int32_t rows,
                                                    float eps)
{
    TPipe pipe;
    GlobalTensor<half> xGm, wGm, bGm, yGm;
    xGm.SetGlobalBuffer((__gm__ half *)x, (int64_t)rows * H);
    wGm.SetGlobalBuffer((__gm__ half *)w, H);
    bGm.SetGlobalBuffer((__gm__ half *)b, H);
    yGm.SetGlobalBuffer((__gm__ half *)y, (int64_t)rows * H);

    TBuf<TPosition::VECCALC> xHBuf, xFBuf, sqBuf, wFBuf, bFBuf, oHBuf, wHBuf, bHBuf;
    pipe.InitBuffer(xHBuf, H * sizeof(half));
    pipe.InitBuffer(xFBuf, H * sizeof(float));
    pipe.InitBuffer(sqBuf, H * sizeof(float));
    pipe.InitBuffer(wFBuf, H * sizeof(float));
    pipe.InitBuffer(bFBuf, H * sizeof(float));
    pipe.InitBuffer(oHBuf, H * sizeof(half));
    pipe.InitBuffer(wHBuf, H * sizeof(half));
    pipe.InitBuffer(bHBuf, H * sizeof(half));
    LocalTensor<half> xH = xHBuf.Get<half>();
    LocalTensor<float> xF = xFBuf.Get<float>();
    LocalTensor<float> sq = sqBuf.Get<float>();
    LocalTensor<float> wF = wFBuf.Get<float>();
    LocalTensor<float> bF = bFBuf.Get<float>();
    LocalTensor<half> oH = oHBuf.Get<half>();
    LocalTensor<half> wH = wHBuf.Get<half>();
    LocalTensor<half> bH = bHBuf.Get<half>();

    DataCopy(wH, wGm, H);
    DataCopy(bH, bGm, H);
    PipeBarrier<PIPE_ALL>();
    Cast(wF, wH, RoundMode::CAST_NONE, H);
    Cast(bF, bH, RoundMode::CAST_NONE, H);
    PipeBarrier<PIPE_V>();

    for (int32_t r = 0; r < rows; ++r) {
        DataCopy(xH, xGm[r * H], H);
        PipeBarrier<PIPE_ALL>();
        Cast(xF, xH, RoundMode::CAST_NONE, H);
        PipeBarrier<PIPE_V>();
        Muls(sq, xF, 1.0f, H);
        PipeBarrier<PIPE_V>();
        {
            int32_t lastLen = 8;
            for (int32_t len = H / 2; len >= 8 && (len % 8 == 0); len >>= 1) {
                Add(sq, sq, sq[len], len);
                PipeBarrier<PIPE_V>();
                lastLen = len;
            }
            for (int32_t st = 8; st + 8 <= lastLen; st += 8) {
                Add(sq, sq, sq[st], 8);
                PipeBarrier<PIPE_V>();
            }
        }
        AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
        float ssum = 0.0f;
        for (int32_t j = 0; j < 8; ++j) ssum += sq.GetValue(j);
        float mean = ssum / (float)H;
        Adds(xF, xF, -mean, H);
        PipeBarrier<PIPE_V>();
        Mul(sq, xF, xF, H);
        PipeBarrier<PIPE_V>();
        {
            int32_t lastLen = 8;
            for (int32_t len = H / 2; len >= 8 && (len % 8 == 0); len >>= 1) {
                Add(sq, sq, sq[len], len);
                PipeBarrier<PIPE_V>();
                lastLen = len;
            }
            for (int32_t st = 8; st + 8 <= lastLen; st += 8) {
                Add(sq, sq, sq[st], 8);
                PipeBarrier<PIPE_V>();
            }
        }
        AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
        float vsum = 0.0f;
        for (int32_t j = 0; j < 8; ++j) vsum += sq.GetValue(j);
        float var = vsum / (float)H;
        float inv = 1.0f / sqrt(var + eps);
        Muls(xF, xF, inv, H);
        PipeBarrier<PIPE_V>();
        Mul(xF, xF, wF, H);
        PipeBarrier<PIPE_V>();
        Add(xF, xF, bF, H);
        PipeBarrier<PIPE_V>();
        Cast(oH, xF, RoundMode::CAST_NONE, H);
        PipeBarrier<PIPE_V>();
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        DataCopy(yGm[r * H], oH, H);
        PipeBarrier<PIPE_ALL>();
    }
}

// ---------------- GELU (tanh approx) ----------------
static __aicore__ inline void gelu_chunk(LocalTensor<half> xH, LocalTensor<half> oH,
                                         LocalTensor<float> xF, LocalTensor<float> t0,
                                         LocalTensor<float> t1, LocalTensor<float> t2,
                                         int32_t n)
{
    Cast(xF, xH, RoundMode::CAST_NONE, n);
    PipeBarrier<PIPE_V>();
    Mul(t0, xF, xF, n);
    PipeBarrier<PIPE_V>();
    Mul(t1, t0, xF, n);
    PipeBarrier<PIPE_V>();
    Muls(t1, t1, 0.044715f, n);
    PipeBarrier<PIPE_V>();
    Add(t1, xF, t1, n);
    PipeBarrier<PIPE_V>();
    Muls(t1, t1, 2.0f * 0.7978845608028654f, n);
    PipeBarrier<PIPE_V>();
    Exp(t2, t1, n);
    PipeBarrier<PIPE_V>();
    Adds(t0, t2, -1.0f, n);
    Adds(t1, t2, 1.0f, n);
    PipeBarrier<PIPE_V>();
    Div(t0, t0, t1, n);
    PipeBarrier<PIPE_V>();
    Adds(t0, t0, 1.0f, n);
    PipeBarrier<PIPE_V>();
    Muls(t1, xF, 0.5f, n);
    PipeBarrier<PIPE_V>();
    Mul(t0, t0, t1, n);
    PipeBarrier<PIPE_V>();
    Cast(oH, t0, RoundMode::CAST_NONE, n);
    PipeBarrier<PIPE_V>();
}

extern "C" __global__ __aicore__ void gelu_f16(GM_ADDR x, GM_ADDR y, int32_t N)
{
    constexpr int32_t CH = 2048;
    TPipe pipe;
    GlobalTensor<half> xGm, yGm;
    xGm.SetGlobalBuffer((__gm__ half *)x, N);
    yGm.SetGlobalBuffer((__gm__ half *)y, N);
    TBuf<TPosition::VECCALC> xHBuf, oHBuf, aBuf, bBuf, cBuf, dBuf;
    pipe.InitBuffer(xHBuf, CH * sizeof(half));
    pipe.InitBuffer(oHBuf, CH * sizeof(half));
    pipe.InitBuffer(aBuf, CH * sizeof(float));
    pipe.InitBuffer(bBuf, CH * sizeof(float));
    pipe.InitBuffer(cBuf, CH * sizeof(float));
    pipe.InitBuffer(dBuf, CH * sizeof(float));
    LocalTensor<half> xH = xHBuf.Get<half>();
    LocalTensor<half> oH = oHBuf.Get<half>();
    LocalTensor<float> a = aBuf.Get<float>();
    LocalTensor<float> b = bBuf.Get<float>();
    LocalTensor<float> c = cBuf.Get<float>();
    LocalTensor<float> d = dBuf.Get<float>();
    for (int32_t k0 = 0; k0 < N; k0 += CH) {
        int32_t n = (N - k0 >= CH) ? CH : (N - k0);
        DataCopy(xH, xGm[k0], n);
        PipeBarrier<PIPE_ALL>();
        gelu_chunk(xH, oH, a, b, c, d, n);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        DataCopy(yGm[k0], oH, n);
        PipeBarrier<PIPE_ALL>();
    }
}

// ---------------- GeGLU ----------------
extern "C" __global__ __aicore__ void geglu_f16(GM_ADDR gate, GM_ADDR up, GM_ADDR out, int32_t N)
{
    TPipe pipe;
    GlobalTensor<half> gGm, uGm, oGm;
    gGm.SetGlobalBuffer((__gm__ half *)gate, N);
    uGm.SetGlobalBuffer((__gm__ half *)up, N);
    oGm.SetGlobalBuffer((__gm__ half *)out, N);
    TBuf<TPosition::VECCALC> gHBuf, uHBuf, oHBuf, aBuf, bBuf, cBuf, dBuf;
    pipe.InitBuffer(gHBuf, N * sizeof(half));
    pipe.InitBuffer(uHBuf, N * sizeof(half));
    pipe.InitBuffer(oHBuf, N * sizeof(half));
    pipe.InitBuffer(aBuf, N * sizeof(float));
    pipe.InitBuffer(bBuf, N * sizeof(float));
    pipe.InitBuffer(cBuf, N * sizeof(float));
    pipe.InitBuffer(dBuf, N * sizeof(float));
    LocalTensor<half> gH = gHBuf.Get<half>();
    LocalTensor<half> uH = uHBuf.Get<half>();
    LocalTensor<half> oH = oHBuf.Get<half>();
    LocalTensor<float> a = aBuf.Get<float>();
    LocalTensor<float> b = bBuf.Get<float>();
    LocalTensor<float> c = cBuf.Get<float>();
    LocalTensor<float> d = dBuf.Get<float>();
    DataCopy(gH, gGm, N);
    DataCopy(uH, uGm, N);
    PipeBarrier<PIPE_ALL>();
    gelu_chunk(gH, oH, a, b, c, d, N);
    Cast(b, uH, RoundMode::CAST_NONE, N);
    PipeBarrier<PIPE_V>();
    Adds(b, b, 1.0f, N);
    PipeBarrier<PIPE_V>();
    Cast(a, oH, RoundMode::CAST_NONE, N);
    PipeBarrier<PIPE_V>();
    Mul(a, a, b, N);
    PipeBarrier<PIPE_V>();
    Cast(oH, a, RoundMode::CAST_NONE, N);
    PipeBarrier<PIPE_V>();
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
    DataCopy(oGm, oH, N);
    PipeBarrier<PIPE_ALL>();
}

// ---------------- GeGLU（batch：输入 [rows, 2*EI]，输出 [rows, EI]） ----------------
extern "C" __global__ __aicore__ void geglu_batch_f16(GM_ADDR gateUp, GM_ADDR out, int32_t rows,
                                                      int32_t EI)
{
    TPipe pipe;
    GlobalTensor<half> gGm, oGm;
    gGm.SetGlobalBuffer((__gm__ half *)gateUp, (int64_t)rows * 2 * EI);
    oGm.SetGlobalBuffer((__gm__ half *)out, (int64_t)rows * EI);
    TBuf<TPosition::VECCALC> gHBuf, uHBuf, oHBuf, aBuf, bBuf, cBuf, dBuf;
    pipe.InitBuffer(gHBuf, EI * sizeof(half));
    pipe.InitBuffer(uHBuf, EI * sizeof(half));
    pipe.InitBuffer(oHBuf, EI * sizeof(half));
    pipe.InitBuffer(aBuf, EI * sizeof(float));
    pipe.InitBuffer(bBuf, EI * sizeof(float));
    pipe.InitBuffer(cBuf, EI * sizeof(float));
    pipe.InitBuffer(dBuf, EI * sizeof(float));
    LocalTensor<half> gH = gHBuf.Get<half>();
    LocalTensor<half> uH = uHBuf.Get<half>();
    LocalTensor<half> oH = oHBuf.Get<half>();
    LocalTensor<float> a = aBuf.Get<float>();
    LocalTensor<float> b = bBuf.Get<float>();
    LocalTensor<float> c = cBuf.Get<float>();
    LocalTensor<float> d = dBuf.Get<float>();
    for (int32_t r = 0; r < rows; ++r) {
        DataCopy(gH, gGm[(int64_t)r * 2 * EI], EI);
        DataCopy(uH, gGm[(int64_t)r * 2 * EI + EI], EI);
        PipeBarrier<PIPE_ALL>();
        gelu_chunk(gH, oH, a, b, c, d, EI);
        Cast(b, uH, RoundMode::CAST_NONE, EI);
        PipeBarrier<PIPE_V>();
        Adds(b, b, 1.0f, EI);
        PipeBarrier<PIPE_V>();
        Cast(a, oH, RoundMode::CAST_NONE, EI);
        PipeBarrier<PIPE_V>();
        Mul(a, a, b, EI);
        PipeBarrier<PIPE_V>();
        Cast(oH, a, RoundMode::CAST_NONE, EI);
        PipeBarrier<PIPE_V>();
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        DataCopy(oGm[(int64_t)r * EI], oH, EI);
        PipeBarrier<PIPE_ALL>();
    }
}

// ---------------- add bias ----------------
extern "C" __global__ __aicore__ void add_bias_f16(GM_ADDR y, GM_ADDR bias, int32_t M, int32_t N)
{
    TPipe pipe;
    GlobalTensor<half> yGm, bGm;
    yGm.SetGlobalBuffer((__gm__ half *)y, (int64_t)M * N);
    bGm.SetGlobalBuffer((__gm__ half *)bias, N);
    TBuf<TPosition::VECCALC> yHBuf, bFBuf, yFBuf, bHBuf;
    pipe.InitBuffer(yHBuf, N * sizeof(half));
    pipe.InitBuffer(bFBuf, N * sizeof(float));
    pipe.InitBuffer(yFBuf, N * sizeof(float));
    pipe.InitBuffer(bHBuf, N * sizeof(half));
    LocalTensor<half> yH = yHBuf.Get<half>();
    LocalTensor<half> bH = bHBuf.Get<half>();
    LocalTensor<float> bF = bFBuf.Get<float>();
    LocalTensor<float> yF = yFBuf.Get<float>();
    DataCopy(bH, bGm, N);
    PipeBarrier<PIPE_ALL>();
    Cast(bF, bH, RoundMode::CAST_NONE, N);
    PipeBarrier<PIPE_V>();
    for (int32_t m = 0; m < M; ++m) {
        DataCopy(yH, yGm[m * N], N);
        PipeBarrier<PIPE_ALL>();
        Cast(yF, yH, RoundMode::CAST_NONE, N);
        PipeBarrier<PIPE_V>();
        Add(yF, yF, bF, N);
        PipeBarrier<PIPE_V>();
        Cast(yH, yF, RoundMode::CAST_NONE, N);
        PipeBarrier<PIPE_V>();
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        DataCopy(yGm[m * N], yH, N);
        PipeBarrier<PIPE_ALL>();
    }
}

// ---------------- out = a + b（分块） ----------------
extern "C" __global__ __aicore__ void add_f16(GM_ADDR a, GM_ADDR b, GM_ADDR out, int32_t N)
{
    constexpr int32_t CH = 8192;
    TPipe pipe;
    GlobalTensor<half> aGm, bGm, oGm;
    aGm.SetGlobalBuffer((__gm__ half *)a, N);
    bGm.SetGlobalBuffer((__gm__ half *)b, N);
    oGm.SetGlobalBuffer((__gm__ half *)out, N);
    TBuf<TPosition::VECCALC> aHBuf, bHBuf, aFBuf, bFBuf, oHBuf;
    pipe.InitBuffer(aHBuf, CH * sizeof(half));
    pipe.InitBuffer(bHBuf, CH * sizeof(half));
    pipe.InitBuffer(aFBuf, CH * sizeof(float));
    pipe.InitBuffer(bFBuf, CH * sizeof(float));
    pipe.InitBuffer(oHBuf, CH * sizeof(half));
    LocalTensor<half> aH = aHBuf.Get<half>();
    LocalTensor<half> bH = bHBuf.Get<half>();
    LocalTensor<float> aF = aFBuf.Get<float>();
    LocalTensor<float> bF = bFBuf.Get<float>();
    LocalTensor<half> oH = oHBuf.Get<half>();
    for (int32_t k0 = 0; k0 < N; k0 += CH) {
        int32_t n = (N - k0 >= CH) ? CH : (N - k0);
        DataCopy(aH, aGm[k0], n);
        DataCopy(bH, bGm[k0], n);
        PipeBarrier<PIPE_ALL>();
        Cast(aF, aH, RoundMode::CAST_NONE, n);
        Cast(bF, bH, RoundMode::CAST_NONE, n);
        PipeBarrier<PIPE_V>();
        Add(aF, aF, bF, n);
        PipeBarrier<PIPE_V>();
        Cast(oH, aF, RoundMode::CAST_NONE, n);
        PipeBarrier<PIPE_V>();
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        DataCopy(oGm[k0], oH, n);
        PipeBarrier<PIPE_ALL>();
    }
}

// ---------------- GELU + 右侧补零（vision fc2 用） ----------------
extern "C" __global__ __aicore__ void gelu_pad_f16(GM_ADDR x, GM_ADDR y, int32_t rows, int32_t K,
                                                   int32_t Kpad)
{
    TPipe pipe;
    GlobalTensor<half> xGm, yGm;
    xGm.SetGlobalBuffer((__gm__ half *)x, (int64_t)rows * K);
    yGm.SetGlobalBuffer((__gm__ half *)y, (int64_t)rows * Kpad);
    TBuf<TPosition::VECCALC> xHBuf, oHBuf, aBuf, bBuf, cBuf, dBuf;
    pipe.InitBuffer(xHBuf, K * sizeof(half));
    pipe.InitBuffer(oHBuf, K * sizeof(half));
    pipe.InitBuffer(aBuf, K * sizeof(float));
    pipe.InitBuffer(bBuf, K * sizeof(float));
    pipe.InitBuffer(cBuf, K * sizeof(float));
    pipe.InitBuffer(dBuf, K * sizeof(float));
    LocalTensor<half> xH = xHBuf.Get<half>();
    LocalTensor<half> oH = oHBuf.Get<half>();
    LocalTensor<float> a = aBuf.Get<float>();
    LocalTensor<float> b = bBuf.Get<float>();
    LocalTensor<float> c = cBuf.Get<float>();
    LocalTensor<float> d = dBuf.Get<float>();
    for (int32_t r = 0; r < rows; ++r) {
        DataCopy(xH, xGm[(int64_t)r * K], K);
        PipeBarrier<PIPE_ALL>();
        gelu_chunk(xH, oH, a, b, c, d, K);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        DataCopy(yGm[(int64_t)r * Kpad], oH, K);
        PipeBarrier<PIPE_ALL>();
    }
}

// ---------------- W4A16 group32 -> fp16 [N,K] ----------------
extern "C" __global__ __aicore__ void dequant_w4_f16(GM_ADDR wq, GM_ADDR scales, GM_ADDR szs,
                                                     GM_ADDR out, int32_t K, int32_t N)
{
    TPipe pipe;
    const int32_t KQ = K / 4;
    const int32_t NG = K / 32;
    const int32_t GPP = KQ / 32;
    const int32_t GPPA = (GPP + 7) / 8 * 8;
    const int32_t SBP = (NG + 15) / 16 * 16;
    GlobalTensor<uint16_t> wGm;
    GlobalTensor<half> sGm, szGm, oGm;
    wGm.SetGlobalBuffer((__gm__ uint16_t *)wq, (int64_t)N * KQ);
    sGm.SetGlobalBuffer((__gm__ half *)scales, (int64_t)N * SBP);
    szGm.SetGlobalBuffer((__gm__ half *)szs, (int64_t)N * SBP);
    oGm.SetGlobalBuffer((__gm__ half *)out, (int64_t)N * K);

    TBuf<TPosition::VECCALC> wRowBuf, phBuf, phFBuf, scSegBuf, szSegBuf, scTabBuf, szTabBuf,
        idxBuf, maskBuf, oFBuf, oHBuf, sHBuf, szHBuf, scFBuf, szFBuf;
    pipe.InitBuffer(wRowBuf, KQ * sizeof(uint16_t));
    pipe.InitBuffer(phBuf, KQ * sizeof(uint16_t));
    pipe.InitBuffer(phFBuf, KQ * sizeof(float));
    pipe.InitBuffer(scSegBuf, KQ * sizeof(float));
    pipe.InitBuffer(szSegBuf, KQ * sizeof(float));
    pipe.InitBuffer(scTabBuf, 4 * GPPA * sizeof(float));
    pipe.InitBuffer(szTabBuf, 4 * GPPA * sizeof(float));
    pipe.InitBuffer(scFBuf, SBP * sizeof(float));
    pipe.InitBuffer(szFBuf, SBP * sizeof(float));
    pipe.InitBuffer(idxBuf, KQ * sizeof(uint32_t));
    pipe.InitBuffer(maskBuf, KQ * sizeof(uint16_t));
    pipe.InitBuffer(oFBuf, K * sizeof(float));
    pipe.InitBuffer(oHBuf, K * sizeof(half));
    pipe.InitBuffer(sHBuf, SBP * sizeof(half));
    pipe.InitBuffer(szHBuf, SBP * sizeof(half));

    LocalTensor<uint16_t> wRow = wRowBuf.Get<uint16_t>();
    LocalTensor<uint16_t> ph = phBuf.Get<uint16_t>();
    LocalTensor<float> phF = phFBuf.Get<float>();
    LocalTensor<float> scSeg = scSegBuf.Get<float>();
    LocalTensor<float> szSeg = szSegBuf.Get<float>();
    LocalTensor<float> scTab = scTabBuf.Get<float>();
    LocalTensor<float> szTab = szTabBuf.Get<float>();
    LocalTensor<uint32_t> idx = idxBuf.Get<uint32_t>();
    LocalTensor<uint16_t> mask = maskBuf.Get<uint16_t>();
    LocalTensor<float> oF = oFBuf.Get<float>();
    LocalTensor<half> oH = oHBuf.Get<half>();
    LocalTensor<half> sH = sHBuf.Get<half>();
    LocalTensor<half> szH = szHBuf.Get<half>();
    LocalTensor<float> scF = scFBuf.Get<float>();
    LocalTensor<float> szF = szFBuf.Get<float>();

    Duplicate(mask, (uint16_t)0x000F, KQ);
    for (int32_t j = 0; j < KQ; ++j) idx.SetValue(j, (uint32_t)((j / 32) * 4));
    PipeBarrier<PIPE_ALL>();
    AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID0);

    for (int32_t n = 0; n < N; ++n) {
        DataCopy(wRow, wGm[n * KQ], KQ);
        DataCopy(sH, sGm[n * SBP], SBP);
        DataCopy(szH, szGm[n * SBP], SBP);
        PipeBarrier<PIPE_ALL>();
        Cast(scF, sH, RoundMode::CAST_NONE, SBP);
        Cast(szF, szH, RoundMode::CAST_NONE, SBP);
        PipeBarrier<PIPE_V>();
        if ((GPP % 8) != 0) {
            for (int32_t p = 0; p < 4; ++p) {
                for (int32_t i = 0; i < GPP; ++i) {
                    scTab.SetValue(p * GPPA + i, scF.GetValue(p * GPP + i));
                    szTab.SetValue(p * GPPA + i, szF.GetValue(p * GPP + i));
                }
            }
            AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
        }
        for (int32_t p = 0; p < 4; ++p) {
            if (p == 0) {
                And(ph, wRow, mask, KQ);
            } else {
                ShiftRight(ph, wRow, (uint16_t)(4 * p), KQ);
                And(ph, ph, mask, KQ);
            }
            PipeBarrier<PIPE_V>();
            Cast(phF, ph.ReinterpretCast<int16_t>(), RoundMode::CAST_NONE, KQ);
            PipeBarrier<PIPE_V>();
            if ((GPP % 8) == 0) {
                Gather(scSeg, scF[p * GPP], idx, 0, (uint32_t)KQ);
                Gather(szSeg, szF[p * GPP], idx, 0, (uint32_t)KQ);
            } else {
                Gather(scSeg, scTab[p * GPPA], idx, 0, (uint32_t)KQ);
                Gather(szSeg, szTab[p * GPPA], idx, 0, (uint32_t)KQ);
            }
            PipeBarrier<PIPE_V>();
            Mul(phF, phF, scSeg, KQ);
            PipeBarrier<PIPE_V>();
            Sub(phF, phF, szSeg, KQ);
            PipeBarrier<PIPE_V>();
            Muls(oF[p * KQ], phF, 1.0f, KQ);
            PipeBarrier<PIPE_V>();
        }
        Cast(oH, oF, RoundMode::CAST_NONE, K);
        PipeBarrier<PIPE_V>();
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        DataCopy(oGm[n * K], oH, K);
        PipeBarrier<PIPE_ALL>();
    }
}

// ---------------- qkv post (text prefill) ----------------
extern "C" __global__ __aicore__ void qkv_post_text(GM_ADDR qkv, GM_ADDR tauq, GM_ADDR tauv,
                                                    GM_ADDR cosv, GM_ADDR sinv, GM_ADDR qo,
                                                    GM_ADDR ko, GM_ADDR vo, int32_t T)
{
    TPipe pipe;
    GlobalTensor<half> qkvGm, qoGm, koGm, voGm, h4;
    GlobalTensor<float> tqGm, tvGm, cosGm, sinGm;
    qkvGm.SetGlobalBuffer((__gm__ half *)qkv, (int64_t)T * 6144);
    tqGm.SetGlobalBuffer((__gm__ float *)tauq, (int64_t)T * MD3_H);
    tvGm.SetGlobalBuffer((__gm__ float *)tauv, (int64_t)T * MD3_H);
    cosGm.SetGlobalBuffer((__gm__ float *)cosv, (int64_t)T * MD3_RH);
    sinGm.SetGlobalBuffer((__gm__ float *)sinv, (int64_t)T * MD3_RH);
    qoGm.SetGlobalBuffer((__gm__ half *)qo, (int64_t)MD3_H * T * MD3_D);
    koGm.SetGlobalBuffer((__gm__ half *)ko, (int64_t)MD3_H * T * MD3_D);
    voGm.SetGlobalBuffer((__gm__ half *)vo, (int64_t)MD3_H * T * MD3_D);

    TBuf<TPosition::VECCALC> rowHBuf, rowFBuf, tqBuf, tvBuf, cosBuf, sinBuf, qBuf, kBuf, vBuf, oHBuf;
    pipe.InitBuffer(rowHBuf, 6144 * sizeof(half));
    pipe.InitBuffer(rowFBuf, 6144 * sizeof(float));
    pipe.InitBuffer(tqBuf, MD3_H * sizeof(float));
    pipe.InitBuffer(tvBuf, MD3_H * sizeof(float));
    pipe.InitBuffer(cosBuf, MD3_RH * sizeof(float));
    pipe.InitBuffer(sinBuf, MD3_RH * sizeof(float));
    pipe.InitBuffer(qBuf, MD3_D * sizeof(float));
    pipe.InitBuffer(kBuf, MD3_D * sizeof(float));
    pipe.InitBuffer(vBuf, MD3_D * sizeof(float));
    pipe.InitBuffer(oHBuf, MD3_D * sizeof(half));
    LocalTensor<half> rowH = rowHBuf.Get<half>();
    LocalTensor<float> rowF = rowFBuf.Get<float>();
    LocalTensor<float> tqF = tqBuf.Get<float>();
    LocalTensor<float> tvF = tvBuf.Get<float>();
    LocalTensor<float> cosF = cosBuf.Get<float>();
    LocalTensor<float> sinF = sinBuf.Get<float>();
    LocalTensor<float> qF = qBuf.Get<float>();
    LocalTensor<float> kF = kBuf.Get<float>();
    LocalTensor<float> vF = vBuf.Get<float>();
    LocalTensor<half> oH = oHBuf.Get<half>();

    for (int32_t t = 0; t < T; ++t) {
        DataCopy(rowH, qkvGm[t * 6144], 6144);
        DataCopy(tqF, tqGm[t * MD3_H], MD3_H);
        DataCopy(tvF, tvGm[t * MD3_H], MD3_H);
        DataCopy(cosF, cosGm[t * MD3_RH], MD3_RH);
        DataCopy(sinF, sinGm[t * MD3_RH], MD3_RH);
        PipeBarrier<PIPE_ALL>();
        Cast(rowF, rowH, RoundMode::CAST_NONE, 6144);
        PipeBarrier<PIPE_V>();
        AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
        for (int32_t h = 0; h < MD3_H; ++h) {
            float cq = tqF.GetValue(h);
            float cv = tvF.GetValue(h);
            for (int32_t d = 0; d < MD3_D; ++d) {
                qF.SetValue(d, rowF.GetValue(h * MD3_D + d) * cq * 0.125f);
                kF.SetValue(d, rowF.GetValue(2048 + h * MD3_D + d));
                vF.SetValue(d, rowF.GetValue(4096 + h * MD3_D + d) * cv);
            }
            for (int32_t i = 0; i < MD3_RH; ++i) {
                float c = cosF.GetValue(i);
                float s = sinF.GetValue(i);
                float a = qF.GetValue(i), b = qF.GetValue(i + MD3_RH);
                qF.SetValue(i, a * c - b * s);
                qF.SetValue(i + MD3_RH, b * c + a * s);
                a = kF.GetValue(i); b = kF.GetValue(i + MD3_RH);
                kF.SetValue(i, a * c - b * s);
                kF.SetValue(i + MD3_RH, b * c + a * s);
            }
            AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
            Cast(oH, qF, RoundMode::CAST_NONE, MD3_D);
            PipeBarrier<PIPE_V>();
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            DataCopy(qoGm[(int64_t)h * T * MD3_D + (int64_t)t * MD3_D], oH, MD3_D);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
            Cast(oH, kF, RoundMode::CAST_NONE, MD3_D);
            PipeBarrier<PIPE_V>();
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            DataCopy(koGm[(int64_t)h * T * MD3_D + (int64_t)t * MD3_D], oH, MD3_D);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
            Cast(oH, vF, RoundMode::CAST_NONE, MD3_D);
            PipeBarrier<PIPE_V>();
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            DataCopy(voGm[(int64_t)h * T * MD3_D + (int64_t)t * MD3_D], oH, MD3_D);
            PipeBarrier<PIPE_ALL>();
        }
    }
}

// ---------------- qkv post (vision) ----------------
extern "C" __global__ __aicore__ void qkv_post_vision(GM_ADDR qkv, GM_ADDR qo, GM_ADDR ko,
                                                      GM_ADDR vo, int32_t T, int32_t H, int32_t D,
                                                      int32_t Dpad, float qscale)
{
    TPipe pipe;
    GlobalTensor<half> qkvGm, qoGm, koGm, voGm;
    const int32_t W = 3 * D * H;
    qkvGm.SetGlobalBuffer((__gm__ half *)qkv, (int64_t)T * W);
    qoGm.SetGlobalBuffer((__gm__ half *)qo, (int64_t)H * T * Dpad);
    koGm.SetGlobalBuffer((__gm__ half *)ko, (int64_t)H * T * Dpad);
    voGm.SetGlobalBuffer((__gm__ half *)vo, (int64_t)H * T * Dpad);
    TBuf<TPosition::VECCALC> rowHBuf, rowFBuf, qBuf, oHBuf;
    pipe.InitBuffer(rowHBuf, W * sizeof(half));
    pipe.InitBuffer(rowFBuf, W * sizeof(float));
    pipe.InitBuffer(qBuf, D * sizeof(float));
    pipe.InitBuffer(oHBuf, Dpad * sizeof(half));
    LocalTensor<half> rowH = rowHBuf.Get<half>();
    LocalTensor<float> rowF = rowFBuf.Get<float>();
    LocalTensor<float> qF = qBuf.Get<float>();
    LocalTensor<half> oH = oHBuf.Get<half>();
    for (int32_t t = 0; t < T; ++t) {
        DataCopy(rowH, qkvGm[(int64_t)t * W], W);
        PipeBarrier<PIPE_ALL>();
        Cast(rowF, rowH, RoundMode::CAST_NONE, W);
        PipeBarrier<PIPE_V>();
        AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
        for (int32_t h = 0; h < H; ++h) {
            for (int32_t d = 0; d < D; ++d) qF.SetValue(d, rowF.GetValue(h * D + d) * qscale);
            AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
            Cast(oH, qF, RoundMode::CAST_NONE, D);
            PipeBarrier<PIPE_V>();
            for (int32_t d = D; d < Dpad; ++d) oH.SetValue(d, (half)0);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
            DataCopy(qoGm[(int64_t)h * T * Dpad + (int64_t)t * Dpad], oH, Dpad);
            for (int32_t d = 0; d < D; ++d) oH.SetValue(d, rowH.GetValue((int64_t)H * D + (int64_t)h * D + d));
            for (int32_t d = D; d < Dpad; ++d) oH.SetValue(d, (half)0);
            AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
            DataCopy(koGm[(int64_t)h * T * Dpad + (int64_t)t * Dpad], oH, Dpad);
            for (int32_t d = 0; d < D; ++d) oH.SetValue(d, rowH.GetValue((int64_t)2 * H * D + (int64_t)h * D + d));
            for (int32_t d = D; d < Dpad; ++d) oH.SetValue(d, (half)0);
            AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
            DataCopy(voGm[(int64_t)h * T * Dpad + (int64_t)t * Dpad], oH, Dpad);
            PipeBarrier<PIPE_ALL>();
        }
    }
}

// ---------------- heads-major -> seq-major ----------------
extern "C" __global__ __aicore__ void heads_to_seq(GM_ADDR src, GM_ADDR dst, int32_t T, int32_t H,
                                                   int32_t D)
{
    TPipe pipe;
    GlobalTensor<half> sGm, dGm;
    sGm.SetGlobalBuffer((__gm__ half *)src, (int64_t)H * T * D);
    dGm.SetGlobalBuffer((__gm__ half *)dst, (int64_t)T * H * D);
    TBuf<TPosition::VECCALC> hBuf;
    pipe.InitBuffer(hBuf, D * sizeof(half));
    LocalTensor<half> h = hBuf.Get<half>();
    for (int32_t hh = 0; hh < H; ++hh) {
        for (int32_t t = 0; t < T; ++t) {
            DataCopy(h, sGm[(int64_t)hh * T * D + (int64_t)t * D], D);
            PipeBarrier<PIPE_ALL>();
            DataCopy(dGm[(int64_t)t * H * D + (int64_t)hh * D], h, D);
            PipeBarrier<PIPE_ALL>();
        }
    }
}

// ---------------- 单 token 注意力解码 ----------------
extern "C" __global__ __aicore__ void attn_decode_text(GM_ADDR qkv, GM_ADDR tauq, GM_ADDR tauv,
                                                       GM_ADDR cosv, GM_ADDR sinv,
                                                       GM_ADDR kcache, GM_ADDR vcache,
                                                       GM_ADDR out, int32_t pos)
{
    TPipe pipe;
    GlobalTensor<half> qkvGm, kcGm, vcGm, oGm;
    GlobalTensor<float> tqGm, tvGm, cosGm, sinGm;
    qkvGm.SetGlobalBuffer((__gm__ half *)qkv, 6144);
    tqGm.SetGlobalBuffer((__gm__ float *)tauq, MD3_H);
    tvGm.SetGlobalBuffer((__gm__ float *)tauv, MD3_H);
    cosGm.SetGlobalBuffer((__gm__ float *)cosv, MD3_RH);
    sinGm.SetGlobalBuffer((__gm__ float *)sinv, MD3_RH);
    kcGm.SetGlobalBuffer((__gm__ half *)kcache, (int64_t)MD3_H * MD3_MAXT * MD3_D);
    vcGm.SetGlobalBuffer((__gm__ half *)vcache, (int64_t)MD3_H * MD3_MAXT * MD3_D);
    oGm.SetGlobalBuffer((__gm__ half *)out, MD3_H * MD3_D);

    TBuf<TPosition::VECCALC> qkvHBuf, qkvFBuf, qBuf, kBuf, vBuf, tmpBuf, cosBuf, sinBuf,
        scoreBuf, score2Buf, kChunkHBuf, kChunkFBuf, vChunkHBuf, vChunkFBuf, outBuf, oHBuf, kOHBuf, tqBuf, tvBuf;
    pipe.InitBuffer(qkvHBuf, 6144 * sizeof(half));
    pipe.InitBuffer(qkvFBuf, 6144 * sizeof(float));
    pipe.InitBuffer(qBuf, MD3_D * sizeof(float));
    pipe.InitBuffer(kBuf, MD3_D * sizeof(float));
    pipe.InitBuffer(vBuf, MD3_D * sizeof(float));
    pipe.InitBuffer(tmpBuf, MD3_D * sizeof(float));
    pipe.InitBuffer(cosBuf, MD3_RH * sizeof(float));
    pipe.InitBuffer(sinBuf, MD3_RH * sizeof(float));
    pipe.InitBuffer(scoreBuf, MD3_MAXT * sizeof(float));
    pipe.InitBuffer(score2Buf, MD3_MAXT * sizeof(float));
    pipe.InitBuffer(kChunkHBuf, 128 * MD3_D * sizeof(half));
    pipe.InitBuffer(kChunkFBuf, 128 * MD3_D * sizeof(float));
    pipe.InitBuffer(vChunkHBuf, 128 * MD3_D * sizeof(half));
    pipe.InitBuffer(vChunkFBuf, 128 * MD3_D * sizeof(float));
    pipe.InitBuffer(outBuf, MD3_D * sizeof(float));
    pipe.InitBuffer(oHBuf, MD3_D * sizeof(half));
    pipe.InitBuffer(kOHBuf, MD3_D * sizeof(half));
    pipe.InitBuffer(tqBuf, MD3_H * sizeof(float));
    pipe.InitBuffer(tvBuf, MD3_H * sizeof(float));

    LocalTensor<half> qkvH = qkvHBuf.Get<half>();
    LocalTensor<float> qkvF = qkvFBuf.Get<float>();
    LocalTensor<float> qF = qBuf.Get<float>();
    LocalTensor<float> kF = kBuf.Get<float>();
    LocalTensor<float> vF = vBuf.Get<float>();
    LocalTensor<float> tmp = tmpBuf.Get<float>();
    LocalTensor<float> cosF = cosBuf.Get<float>();
    LocalTensor<float> sinF = sinBuf.Get<float>();
    LocalTensor<float> score = scoreBuf.Get<float>();
    LocalTensor<float> score2 = score2Buf.Get<float>();
    LocalTensor<half> kChunkH = kChunkHBuf.Get<half>();
    LocalTensor<float> kChunk = kChunkFBuf.Get<float>();
    LocalTensor<half> vChunkH = vChunkHBuf.Get<half>();
    LocalTensor<float> vChunk = vChunkFBuf.Get<float>();
    LocalTensor<float> outF = outBuf.Get<float>();
    LocalTensor<half> oH = oHBuf.Get<half>();
    LocalTensor<half> kOH = kOHBuf.Get<half>();
    LocalTensor<float> tqF = tqBuf.Get<float>();
    LocalTensor<float> tvF = tvBuf.Get<float>();

    DataCopy(qkvH, qkvGm, 6144);
    DataCopy(tqF, tqGm, MD3_H);
    DataCopy(tvF, tvGm, MD3_H);
    DataCopy(cosF, cosGm, MD3_RH);
    DataCopy(sinF, sinGm, MD3_RH);
    PipeBarrier<PIPE_ALL>();
    Cast(qkvF, qkvH, RoundMode::CAST_NONE, 6144);
    PipeBarrier<PIPE_V>();
    AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);

    for (int32_t h = 0; h < MD3_H; ++h) {
        float cq = tqF.GetValue(h);
        float cv = tvF.GetValue(h);
        for (int32_t d = 0; d < MD3_D; ++d) {
            qF.SetValue(d, qkvF.GetValue(h * MD3_D + d) * cq * 0.125f);
            kF.SetValue(d, qkvF.GetValue(2048 + h * MD3_D + d));
            vF.SetValue(d, qkvF.GetValue(4096 + h * MD3_D + d) * cv);
        }
        for (int32_t i = 0; i < MD3_RH; ++i) {
            float c = cosF.GetValue(i), s = sinF.GetValue(i);
            float a = qF.GetValue(i), b = qF.GetValue(i + MD3_RH);
            qF.SetValue(i, a * c - b * s);
            qF.SetValue(i + MD3_RH, b * c + a * s);
            a = kF.GetValue(i); b = kF.GetValue(i + MD3_RH);
            kF.SetValue(i, a * c - b * s);
            kF.SetValue(i + MD3_RH, b * c + a * s);
        }
        for (int32_t d = 0; d < MD3_D; ++d) kOH.SetValue(d, (half)kF.GetValue(d));
        AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        DataCopy(kcGm[(int64_t)h * MD3_MAXT * MD3_D + (int64_t)pos * MD3_D], kOH, MD3_D);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        for (int32_t d = 0; d < MD3_D; ++d) kOH.SetValue(d, (half)vF.GetValue(d));
        AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        DataCopy(vcGm[(int64_t)h * MD3_MAXT * MD3_D + (int64_t)pos * MD3_D], kOH, MD3_D);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        PipeBarrier<PIPE_ALL>();
    }

    int32_t nkeys = pos + 1;
    for (int32_t h = 0; h < MD3_H; ++h) {
        // 重建本 head 的 q（含 tau 与 RoPE）
        float cq2 = tqF.GetValue(h);
        for (int32_t d = 0; d < MD3_D; ++d) qF.SetValue(d, qkvF.GetValue(h * MD3_D + d) * cq2 * 0.125f);
        for (int32_t i = 0; i < MD3_RH; ++i) {
            float c = cosF.GetValue(i), s = sinF.GetValue(i);
            float a = qF.GetValue(i), b = qF.GetValue(i + MD3_RH);
            qF.SetValue(i, a * c - b * s);
            qF.SetValue(i + MD3_RH, b * c + a * s);
        }
        AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
        Duplicate(score, -1e30f, MD3_MAXT);
        PipeBarrier<PIPE_V>();
        AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
        int32_t base = 0;
        for (; base < nkeys; base += 128) {
            int32_t cnt = (nkeys - base >= 128) ? 128 : (nkeys - base);
            DataCopy(kChunkH, kcGm[(int64_t)h * MD3_MAXT * MD3_D + (int64_t)base * MD3_D], cnt * MD3_D);
            PipeBarrier<PIPE_ALL>();
            Cast(kChunk, kChunkH, RoundMode::CAST_NONE, cnt * MD3_D);
            PipeBarrier<PIPE_V>();
            AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
            for (int32_t j = 0; j < cnt; ++j) {
                Mul(tmp, qF, kChunk[j * MD3_D], MD3_D);
                PipeBarrier<PIPE_V>();
                for (int32_t len = MD3_D / 2; len >= 8; len >>= 1) {
                    Add(tmp, tmp, tmp[len], len);
                    PipeBarrier<PIPE_V>();
                }
                AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
                float sv = 0.0f;
                for (int32_t q = 0; q < 8; ++q) sv += tmp.GetValue(q);
                score.SetValue(base + j, sv);
                AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
            }
        }
        AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
        int32_t pad = 8;
        while (pad < nkeys) pad <<= 1;
        AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
        Muls(score2, score, 1.0f, pad);
        PipeBarrier<PIPE_V>();
        for (int32_t len = pad / 2; len >= 8; len >>= 1) {
            Max(score2, score2, score2[len], len);
            PipeBarrier<PIPE_V>();
        }
        AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
        float mx = -1e30f;
        for (int32_t j = 0; j < 8; ++j) mx = (score2.GetValue(j) > mx) ? score2.GetValue(j) : mx;
        AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
        Adds(score, score, -mx, pad);
        PipeBarrier<PIPE_V>();
        Exp(score, score, pad);
        PipeBarrier<PIPE_V>();
        Muls(score2, score, 1.0f, pad);
        PipeBarrier<PIPE_V>();
        for (int32_t len = pad / 2; len >= 8; len >>= 1) {
            Add(score2, score2, score2[len], len);
            PipeBarrier<PIPE_V>();
        }
        AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
        float ssum = 0.0f;
        for (int32_t j = 0; j < 8; ++j) ssum += score2.GetValue(j);
        float inv = 1.0f / ssum;

        for (int32_t d = 0; d < MD3_D; ++d) outF.SetValue(d, 0.0f);
        for (base = 0; base < nkeys; base += 128) {
            int32_t cnt = (nkeys - base >= 128) ? 128 : (nkeys - base);
            DataCopy(vChunkH, vcGm[(int64_t)h * MD3_MAXT * MD3_D + (int64_t)base * MD3_D], cnt * MD3_D);
            PipeBarrier<PIPE_ALL>();
            Cast(vChunk, vChunkH, RoundMode::CAST_NONE, cnt * MD3_D);
            PipeBarrier<PIPE_V>();
            AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
            for (int32_t j = 0; j < cnt; ++j) {
                float p = score.GetValue(base + j) * inv;
                AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
                Axpy(outF, vChunk[j * MD3_D], p, MD3_D);
                PipeBarrier<PIPE_V>();
            }
        }
        AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
        Cast(oH, outF, RoundMode::CAST_NONE, MD3_D);
        PipeBarrier<PIPE_V>();
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        DataCopy(oGm[h * MD3_D], oH, MD3_D);
        PipeBarrier<PIPE_ALL>();
    }
}

#ifndef ASCENDC_CPU_DEBUG
void layernorm_f16_do(uint32_t b, void *st, uint8_t *x, uint8_t *w, uint8_t *bb, uint8_t *y,
                      int32_t H, int32_t rows, float eps)
{ layernorm_f16<<<b, nullptr, st>>>(x, w, bb, y, H, rows, eps); }
void gelu_f16_do(uint32_t b, void *st, uint8_t *x, uint8_t *y, int32_t N)
{ gelu_f16<<<b, nullptr, st>>>(x, y, N); }
void geglu_f16_do(uint32_t b, void *st, uint8_t *g, uint8_t *u, uint8_t *o, int32_t N)
{ geglu_f16<<<b, nullptr, st>>>(g, u, o, N); }
void add_bias_f16_do(uint32_t b, void *st, uint8_t *y, uint8_t *bias, int32_t M, int32_t N)
{ add_bias_f16<<<b, nullptr, st>>>(y, bias, M, N); }
void geglu_batch_f16_do(uint32_t b, void *st, uint8_t *gu, uint8_t *o, int32_t rows, int32_t EI)
{ geglu_batch_f16<<<b, nullptr, st>>>(gu, o, rows, EI); }
void dequant_w4_f16_do(uint32_t b, void *st, uint8_t *wq, uint8_t *s, uint8_t *sz, uint8_t *o,
                       int32_t K, int32_t N)
{ dequant_w4_f16<<<b, nullptr, st>>>(wq, s, sz, o, K, N); }
void qkv_post_text_do(uint32_t b, void *st, uint8_t *qkv, uint8_t *tq, uint8_t *tv, uint8_t *cos,
                      uint8_t *sin, uint8_t *q, uint8_t *k, uint8_t *v, int32_t T)
{ qkv_post_text<<<b, nullptr, st>>>(qkv, tq, tv, cos, sin, q, k, v, T); }
void qkv_post_vision_do(uint32_t b, void *st, uint8_t *qkv, uint8_t *q, uint8_t *k, uint8_t *v,
                        int32_t T, int32_t H, int32_t D, int32_t Dpad, float qscale)
{ qkv_post_vision<<<b, nullptr, st>>>(qkv, q, k, v, T, H, D, Dpad, qscale); }
void attn_decode_text_do(uint32_t b, void *st, uint8_t *qkv, uint8_t *tq, uint8_t *tv,
                         uint8_t *cos, uint8_t *sin, uint8_t *kc, uint8_t *vc, uint8_t *o, int32_t pos)
{ attn_decode_text<<<b, nullptr, st>>>(qkv, tq, tv, cos, sin, kc, vc, o, pos); }
void heads_to_seq_do(uint32_t b, void *st, uint8_t *src, uint8_t *dst, int32_t T, int32_t H, int32_t D)
{ heads_to_seq<<<b, nullptr, st>>>(src, dst, T, H, D); }
void add_f16_do(uint32_t b, void *st, uint8_t *a, uint8_t *b_, uint8_t *o, int32_t N)
{ add_f16<<<b, nullptr, st>>>(a, b_, o, N); }
void gelu_pad_f16_do(uint32_t b, void *st, uint8_t *x, uint8_t *y, int32_t rows, int32_t K, int32_t Kpad)
{ gelu_pad_f16<<<b, nullptr, st>>>(x, y, rows, K, Kpad); }
#endif
