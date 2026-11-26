/**
 * dense_ops — 引擎用基础算子（通用形状，host 传尺寸）
 *
 *  fp16_gemv(act [K], w [N,K], out [N], K, N)             —— 通用 fp16 GEMV（M=1）
 *  rmsnorm_op(x [H], w [H], y [H], H, eps)                —— RMSNorm
 *  vec_axpy(a [N], b [N], out [N], s, N)                  —— out = a + b*s
 *  mul_silu(gate [N], up [N], out [N], N)                 —— out = silu(gate)*up
 *  conv1d_decode(x, state, w, out, C)                     —— depthwise causal conv(k=4)+SiLU，w 为 [4,C]
 */
#include "kernel_operator.h"

using namespace AscendC;

// ---------------- 通用 fp16 GEMV ----------------
extern "C" __global__ __aicore__ void fp16_gemv(GM_ADDR act, GM_ADDR w, GM_ADDR out,
                                                int32_t K, int32_t N)
{
    TPipe pipe;
    GlobalTensor<half> actGm, wGm, oGm;
    actGm.SetGlobalBuffer((__gm__ half *)act, K);
    wGm.SetGlobalBuffer((__gm__ half *)w, (int64_t)N * K);
    oGm.SetGlobalBuffer((__gm__ half *)out, N);

    // K 补齐到 2 的幂：树形归约的偏移必须 32B 对齐（K=768 时最后一步会落在 48B 上）
    int32_t KP = 1;
    while (KP < K) KP <<= 1;

    TBuf<TPosition::VECCALC> actHBuf, actFBuf, rowHBuf, rowFBuf, accBuf, outHalfBuf;
    pipe.InitBuffer(actHBuf, KP * sizeof(half));
    pipe.InitBuffer(actFBuf, KP * sizeof(float));
    pipe.InitBuffer(rowHBuf, KP * sizeof(half));
    pipe.InitBuffer(rowFBuf, KP * sizeof(float));
    pipe.InitBuffer(accBuf, 16 * sizeof(float));
    pipe.InitBuffer(outHalfBuf, 16 * sizeof(half));

    LocalTensor<half> actH = actHBuf.Get<half>();
    LocalTensor<float> actF = actFBuf.Get<float>();
    LocalTensor<half> rowH = rowHBuf.Get<half>();
    LocalTensor<float> rowF = rowFBuf.Get<float>();
    LocalTensor<float> acc = accBuf.Get<float>();
    LocalTensor<half> oH = outHalfBuf.Get<half>();

    DataCopy(actH, actGm, K);
    PipeBarrier<PIPE_ALL>();
    Duplicate(actF, 0.0f, KP);
    Duplicate(rowF, 0.0f, KP);
    PipeBarrier<PIPE_V>();
    Cast(actF, actH, RoundMode::CAST_NONE, K);
    PipeBarrier<PIPE_V>();

    for (int32_t n = 0; n < N; ++n) {
        DataCopy(rowH, wGm[n * K], K);
        PipeBarrier<PIPE_ALL>();
        Cast(rowF, rowH, RoundMode::CAST_NONE, K);
        PipeBarrier<PIPE_V>();
        Mul(rowF, rowF, actF, KP);
        PipeBarrier<PIPE_V>();
        for (int32_t len = KP / 2; len >= 8; len >>= 1) {
            Add(rowF, rowF, rowF[len], len);
            PipeBarrier<PIPE_V>();
        }
        AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
        float s = 0.0f;
        for (int32_t j = 0; j < 8; ++j) {
            s += rowF.GetValue(j);
        }
        acc.SetValue(n & 15, s);
        if ((n & 15) == 15 || n == N - 1) {
            int32_t cnt = ((n & 15) == 15) ? 16 : ((n & 15) + 1);
            int32_t base = n - cnt + 1;
            PipeBarrier<PIPE_ALL>();
            Cast(oH, acc, RoundMode::CAST_NONE, 16);
            PipeBarrier<PIPE_V>();
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            DataCopy(oGm[base], oH, 16);
            PipeBarrier<PIPE_ALL>();
        }
    }
}

// ---------------- RMSNorm ----------------
extern "C" __global__ __aicore__ void rmsnorm_op(GM_ADDR x, GM_ADDR w, GM_ADDR y, int32_t H,
                                                 float eps)
{
    TPipe pipe;
    GlobalTensor<half> xGm, wGm, yGm;
    xGm.SetGlobalBuffer((__gm__ half *)x, H);
    wGm.SetGlobalBuffer((__gm__ half *)w, H);
    yGm.SetGlobalBuffer((__gm__ half *)y, H);

    TBuf<TPosition::VECCALC> xHBuf, xFBuf, wHBuf, wFBuf, sqBuf, yHBuf;
    pipe.InitBuffer(xHBuf, H * sizeof(half));
    pipe.InitBuffer(xFBuf, H * sizeof(float));
    pipe.InitBuffer(wHBuf, H * sizeof(half));
    pipe.InitBuffer(wFBuf, H * sizeof(float));
    pipe.InitBuffer(sqBuf, H * sizeof(float));
    pipe.InitBuffer(yHBuf, H * sizeof(half));

    LocalTensor<half> xH = xHBuf.Get<half>();
    LocalTensor<float> xF = xFBuf.Get<float>();
    LocalTensor<half> wH = wHBuf.Get<half>();
    LocalTensor<float> wF = wFBuf.Get<float>();
    LocalTensor<float> sq = sqBuf.Get<float>();
    LocalTensor<half> yH = yHBuf.Get<half>();

    DataCopy(xH, xGm, H);
    DataCopy(wH, wGm, H);
    PipeBarrier<PIPE_ALL>();
    Cast(xF, xH, RoundMode::CAST_NONE, H);
    Cast(wF, wH, RoundMode::CAST_NONE, H);
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
    float ss = 0.0f;
    for (int32_t j = 0; j < 8; ++j) {
        ss += sq.GetValue(j);
    }
    float rms = ss / (float)H + eps;
    float inv = 1.0f / sqrt(rms);
    Muls(xF, xF, inv, H);
    PipeBarrier<PIPE_V>();
    Mul(xF, xF, wF, H);
    PipeBarrier<PIPE_V>();
    Cast(yH, xF, RoundMode::CAST_NONE, H);
    PipeBarrier<PIPE_V>();
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
    DataCopy(yGm, yH, H);
    PipeBarrier<PIPE_ALL>();
}

// ---------------- out = a + b*s ----------------
extern "C" __global__ __aicore__ void vec_axpy(GM_ADDR a, GM_ADDR b, GM_ADDR out, float s,
                                               int32_t N)
{
    TPipe pipe;
    GlobalTensor<half> aGm, bGm, oGm;
    aGm.SetGlobalBuffer((__gm__ half *)a, N);
    bGm.SetGlobalBuffer((__gm__ half *)b, N);
    oGm.SetGlobalBuffer((__gm__ half *)out, N);
    TBuf<TPosition::VECCALC> aHBuf, bHBuf, aFBuf, bFBuf, oHBuf;
    pipe.InitBuffer(aHBuf, N * sizeof(half));
    pipe.InitBuffer(bHBuf, N * sizeof(half));
    pipe.InitBuffer(aFBuf, N * sizeof(float));
    pipe.InitBuffer(bFBuf, N * sizeof(float));
    pipe.InitBuffer(oHBuf, N * sizeof(half));
    LocalTensor<half> aH = aHBuf.Get<half>();
    LocalTensor<half> bH = bHBuf.Get<half>();
    LocalTensor<float> aF = aFBuf.Get<float>();
    LocalTensor<float> bF = bFBuf.Get<float>();
    LocalTensor<half> oH = oHBuf.Get<half>();
    DataCopy(aH, aGm, N);
    DataCopy(bH, bGm, N);
    PipeBarrier<PIPE_ALL>();
    Cast(aF, aH, RoundMode::CAST_NONE, N);
    Cast(bF, bH, RoundMode::CAST_NONE, N);
    PipeBarrier<PIPE_V>();
    Axpy(aF, bF, s, N);
    PipeBarrier<PIPE_V>();
    Cast(oH, aF, RoundMode::CAST_NONE, N);
    PipeBarrier<PIPE_V>();
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
    DataCopy(oGm, oH, N);
    PipeBarrier<PIPE_ALL>();
}

// ---------------- out = silu(gate) * up ----------------
extern "C" __global__ __aicore__ void mul_silu(GM_ADDR gate, GM_ADDR up, GM_ADDR out, int32_t N)
{
    // 分块处理（CH=4096），避免 FF=17408 时 UB 超限
    constexpr int32_t CH = 4096;
    TPipe pipe;
    GlobalTensor<half> gGm, uGm, oGm;
    gGm.SetGlobalBuffer((__gm__ half *)gate, N);
    uGm.SetGlobalBuffer((__gm__ half *)up, N);
    oGm.SetGlobalBuffer((__gm__ half *)out, N);
    TBuf<TPosition::VECCALC> gHBuf, uHBuf, gFBuf, uFBuf, sigBuf, oHBuf;
    pipe.InitBuffer(gHBuf, CH * sizeof(half));
    pipe.InitBuffer(uHBuf, CH * sizeof(half));
    pipe.InitBuffer(gFBuf, CH * sizeof(float));
    pipe.InitBuffer(uFBuf, CH * sizeof(float));
    pipe.InitBuffer(sigBuf, CH * sizeof(float));
    pipe.InitBuffer(oHBuf, CH * sizeof(half));
    LocalTensor<half> gH = gHBuf.Get<half>();
    LocalTensor<half> uH = uHBuf.Get<half>();
    LocalTensor<float> gF = gFBuf.Get<float>();
    LocalTensor<float> uF = uFBuf.Get<float>();
    LocalTensor<float> sg = sigBuf.Get<float>();
    LocalTensor<half> oH = oHBuf.Get<half>();

    for (int32_t k0 = 0; k0 < N; k0 += CH) {
        int32_t n = (N - k0 >= CH) ? CH : (N - k0);
        DataCopy(gH, gGm[k0], n);
        DataCopy(uH, uGm[k0], n);
        PipeBarrier<PIPE_ALL>();
        Cast(gF, gH, RoundMode::CAST_NONE, n);
        Cast(uF, uH, RoundMode::CAST_NONE, n);
        PipeBarrier<PIPE_V>();
        Muls(sg, gF, -1.0f, n);
        PipeBarrier<PIPE_V>();
        Exp(sg, sg, n);                                   // exp(-g)
        PipeBarrier<PIPE_V>();
        Adds(sg, sg, 1.0f, n);
        PipeBarrier<PIPE_V>();
        Div(gF, gF, sg, n);                              // g / (1+exp(-g)) = silu(g)
        PipeBarrier<PIPE_V>();
        Mul(gF, gF, uF, n);                              // * up
        PipeBarrier<PIPE_V>();
        Cast(oH, gF, RoundMode::CAST_NONE, n);
        PipeBarrier<PIPE_V>();
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        DataCopy(oGm[k0], oH, n);
        PipeBarrier<PIPE_ALL>();
    }
}

// ---------------- depthwise causal conv1d (k=4) + SiLU + state 更新 ----------------
// w 布局 [4, C]（w[j*C + c] 是通道 c 的第 j 个卷积权重）
// state 布局 [3, C] fp32：state[0] 最旧 … state[2] 最新
// out[c] = silu(w0*st0 + w1*st1 + w2*st2 + w3*x)
// 更新：st0<-st1, st1<-st2, st2<-x
extern "C" __global__ __aicore__ void conv1d_decode(GM_ADDR x, GM_ADDR state, GM_ADDR w,
                                                    GM_ADDR out, int32_t C)
{
    TPipe pipe;
    GlobalTensor<half> xGm, wGm, oGm;
    GlobalTensor<float> stGm;
    xGm.SetGlobalBuffer((__gm__ half *)x, C);
    wGm.SetGlobalBuffer((__gm__ half *)w, (int64_t)4 * C);
    oGm.SetGlobalBuffer((__gm__ half *)out, C);
    stGm.SetGlobalBuffer((__gm__ float *)state, 3 * C);

    TBuf<TPosition::VECCALC> xHBuf, xFBuf, stFBuf, wHBuf, wFBuf, tmpBuf, accBuf, oHBuf;
    pipe.InitBuffer(xHBuf, C * sizeof(half));
    pipe.InitBuffer(xFBuf, C * sizeof(float));
    pipe.InitBuffer(stFBuf, 3 * C * sizeof(float));
    pipe.InitBuffer(wHBuf, C * sizeof(half));
    pipe.InitBuffer(wFBuf, C * sizeof(float));
    pipe.InitBuffer(tmpBuf, C * sizeof(float));
    pipe.InitBuffer(accBuf, C * sizeof(float));
    pipe.InitBuffer(oHBuf, C * sizeof(half));

    LocalTensor<half> xH = xHBuf.Get<half>();
    LocalTensor<float> xF = xFBuf.Get<float>();
    LocalTensor<float> stF = stFBuf.Get<float>();
    LocalTensor<half> wH = wHBuf.Get<half>();
    LocalTensor<float> wF = wFBuf.Get<float>();
    LocalTensor<float> tmp = tmpBuf.Get<float>();
    LocalTensor<float> acc = accBuf.Get<float>();
    LocalTensor<half> oH = oHBuf.Get<half>();

    DataCopy(xH, xGm, C);
    DataCopy(stF, stGm, 3 * C);
    PipeBarrier<PIPE_ALL>();
    Cast(xF, xH, RoundMode::CAST_NONE, C);
    PipeBarrier<PIPE_V>();
    Duplicate(acc, 0.0f, C);
    PipeBarrier<PIPE_V>();

    for (int32_t j = 0; j < 4; ++j) {
        DataCopy(wH, wGm[j * C], C);
        PipeBarrier<PIPE_ALL>();
        Cast(wF, wH, RoundMode::CAST_NONE, C);
        PipeBarrier<PIPE_V>();
        if (j < 3) {
            Mul(tmp, wF, stF[j * C], C);
        } else {
            Mul(tmp, wF, xF, C);
        }
        PipeBarrier<PIPE_V>();
        Add(acc, acc, tmp, C);
        PipeBarrier<PIPE_V>();
    }
    Muls(tmp, acc, -1.0f, C);
    Exp(tmp, tmp, C);
    Adds(tmp, tmp, 1.0f, C);
    PipeBarrier<PIPE_V>();
    Reciprocal(tmp, tmp, C);
    PipeBarrier<PIPE_V>();
    Mul(acc, acc, tmp, C);
    PipeBarrier<PIPE_V>();
    Cast(oH, acc, RoundMode::CAST_NONE, C);
    PipeBarrier<PIPE_V>();
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
    DataCopy(oGm, oH, C);

    // state 更新：st0<-st1, st1<-st2, st2<-x（标量，C 通常 ≤ 6144）
    PipeBarrier<PIPE_ALL>();
    AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
    for (int32_t c = 0; c < C; ++c) {
        stF.SetValue(c, stF.GetValue(C + c));
        stF.SetValue(C + c, stF.GetValue(2 * C + c));
        stF.SetValue(2 * C + c, xF.GetValue(c));
    }
    PipeBarrier<PIPE_ALL>();
    DataCopy(stGm, stF, 3 * C);
    PipeBarrier<PIPE_ALL>();
}

// ---------------- L2Norm（带 1/sqrt(D) 缩放，GDN 的 q/k norm）----------------
// y = x * (1/sqrt(D)) / sqrt(Σx² + eps)
extern "C" __global__ __aicore__ void l2norm_op(GM_ADDR x, GM_ADDR y, int32_t D, float eps, float mul)
{
    TPipe pipe;
    GlobalTensor<half> xGm, yGm;
    xGm.SetGlobalBuffer((__gm__ half *)x, D);
    yGm.SetGlobalBuffer((__gm__ half *)y, D);
    TBuf<TPosition::VECCALC> xHBuf, xFBuf, sqBuf, yHBuf;
    pipe.InitBuffer(xHBuf, D * sizeof(half));
    pipe.InitBuffer(xFBuf, D * sizeof(float));
    pipe.InitBuffer(sqBuf, D * sizeof(float));
    pipe.InitBuffer(yHBuf, D * sizeof(half));
    LocalTensor<half> xH = xHBuf.Get<half>();
    LocalTensor<float> xF = xFBuf.Get<float>();
    LocalTensor<float> sq = sqBuf.Get<float>();
    LocalTensor<half> yH = yHBuf.Get<half>();
    DataCopy(xH, xGm, D);
    PipeBarrier<PIPE_ALL>();
    Cast(xF, xH, RoundMode::CAST_NONE, D);
    PipeBarrier<PIPE_V>();
    Mul(sq, xF, xF, D);
    PipeBarrier<PIPE_V>();
    {
        int32_t lastLen = 8;
        for (int32_t len = D / 2; len >= 8 && (len % 8 == 0); len >>= 1) {
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
    float ss = 0.0f;
    for (int32_t j = 0; j < 8; ++j) {
        ss += sq.GetValue(j);
    }
    float inv = mul / sqrt(ss + eps);
    Muls(xF, xF, inv, D);
    PipeBarrier<PIPE_V>();
    Cast(yH, xF, RoundMode::CAST_NONE, D);
    PipeBarrier<PIPE_V>();
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
    DataCopy(yGm, yH, D);
    PipeBarrier<PIPE_ALL>();
}

#ifndef ASCENDC_CPU_DEBUG
void l2norm_op_do(uint32_t b, void *st, uint8_t *x, uint8_t *y, int32_t D, float eps, float mul)
{
    l2norm_op<<<b, nullptr, st>>>(x, y, D, eps, mul);
}
void fp16_gemv_do(uint32_t b, void *st, uint8_t *act, uint8_t *w, uint8_t *out, int32_t K, int32_t N)
{
    fp16_gemv<<<b, nullptr, st>>>(act, w, out, K, N);
}
void rmsnorm_op_do(uint32_t b, void *st, uint8_t *x, uint8_t *w, uint8_t *y, int32_t H, float eps)
{
    rmsnorm_op<<<b, nullptr, st>>>(x, w, y, H, eps);
}
void vec_axpy_do(uint32_t b, void *st, uint8_t *a, uint8_t *b_, uint8_t *out, float s, int32_t N)
{
    vec_axpy<<<b, nullptr, st>>>(a, b_, out, s, N);
}
void mul_silu_do(uint32_t b, void *st, uint8_t *gate, uint8_t *up, uint8_t *out, int32_t N)
{
    mul_silu<<<b, nullptr, st>>>(gate, up, out, N);
}
void conv1d_decode_do(uint32_t b, void *st, uint8_t *x, uint8_t *state, uint8_t *w, uint8_t *out, int32_t C)
{
    conv1d_decode<<<b, nullptr, st>>>(x, state, w, out, C);
}
#endif
