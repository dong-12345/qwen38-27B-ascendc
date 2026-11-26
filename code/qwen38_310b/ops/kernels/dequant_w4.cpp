/**
 * dequant_w4 — 把本项目 W4 相位交织权重整体反量化成 fp16（[N][K]，行主序）
 *
 *   out[n][k] = q[n,k]·s[n,g(k)] − sz[n,g(k)]      （k = p·KQ + j, g = k/32）
 *
 * 用途：**两段式 Cube 路径**的第一步。反量化后的 fp16 权重直接喂 aclnnMatmul
 * （matmulTN：B 按 {K,N} + 步长 {1,K} 解释 = 我们这份 [N][K] 行主序），
 * 让 NPU 的 Cube 真正参与计算（M>=8 时相对向量 GEMV 实测 20-23x）。
 *
 * 两条路径：
 *   A) K ≤ 8192（绝大多数矩阵）：整 K 一次展开 s/sz（2 次 Gather），4 个相位合并成
 *      1 次 Cast / 1 次 Mul / 1 次 Sub / 1 次 Cast —— 每行约 15 条指令（旧版 29 条）。
 *   B) K > 8192（如 K=17408 的 mdown）：Gather count 上限 8192，退回逐相位版本。
 *
 * 数值与旧版逐元素一致（同一运算顺序 (q·s) − sz）。
 */
#include "kernel_operator.h"

using namespace AscendC;

constexpr int32_t DGS = 32;

extern "C" __global__ __aicore__ void dequant_w4(GM_ADDR wq, GM_ADDR scales, GM_ADDR szs, GM_ADDR out,
                                                 int32_t K, int32_t N)
{
    TPipe pipe;
    const int32_t KQ = K / 4;
    const int32_t NG = K / DGS;
    const int32_t GPP = KQ / DGS;
    const int32_t SBP = (NG + 15) / 16 * 16;
    const bool whole = (K <= 8192);

    GlobalTensor<uint16_t> wGm;
    GlobalTensor<half> sGm, szGm, oGm;
    wGm.SetGlobalBuffer((__gm__ uint16_t *)wq, (int64_t)N * KQ);
    sGm.SetGlobalBuffer((__gm__ half *)scales, (int64_t)N * SBP);
    szGm.SetGlobalBuffer((__gm__ half *)szs, (int64_t)N * SBP);
    oGm.SetGlobalBuffer((__gm__ half *)out, (int64_t)N * K);

    // 行缓冲
    TBuf<TPosition::VECCALC> wRowBuf, sRowBuf, szRowBuf, sTabBuf, szTabBuf, idxBuf, maskBuf, phBuf,
        phFBuf, sExpBuf, szExpBuf, oRowBuf;
    pipe.InitBuffer(wRowBuf, (size_t)KQ * 2);
    pipe.InitBuffer(sRowBuf, (size_t)SBP * 2);
    pipe.InitBuffer(szRowBuf, (size_t)SBP * 2);
    pipe.InitBuffer(sTabBuf, (size_t)SBP * 4);
    pipe.InitBuffer(szTabBuf, (size_t)SBP * 4);
    pipe.InitBuffer(idxBuf, (size_t)(whole ? K : KQ) * 4);
    pipe.InitBuffer(maskBuf, (size_t)KQ * 2);
    pipe.InitBuffer(phBuf, (size_t)(whole ? K : KQ) * 2);
    pipe.InitBuffer(phFBuf, (size_t)(whole ? K : KQ) * 4);
    pipe.InitBuffer(sExpBuf, (size_t)(whole ? K : KQ) * 4);
    pipe.InitBuffer(szExpBuf, (size_t)(whole ? K : KQ) * 4);
    pipe.InitBuffer(oRowBuf, (size_t)K * 2);

    LocalTensor<uint16_t> wRow = wRowBuf.Get<uint16_t>();
    LocalTensor<half> sRow = sRowBuf.Get<half>();
    LocalTensor<half> szRow = szRowBuf.Get<half>();
    LocalTensor<float> sTab = sTabBuf.Get<float>();
    LocalTensor<float> szTab = szTabBuf.Get<float>();
    LocalTensor<uint32_t> idx = idxBuf.Get<uint32_t>();
    LocalTensor<uint16_t> maskT = maskBuf.Get<uint16_t>();
    LocalTensor<uint16_t> ph = phBuf.Get<uint16_t>();
    LocalTensor<float> phF = phFBuf.Get<float>();
    LocalTensor<float> sExp = sExpBuf.Get<float>();
    LocalTensor<float> szExp = szExpBuf.Get<float>();
    LocalTensor<half> oRow = oRowBuf.Get<half>();

    Duplicate(maskT, (uint16_t)0x000F, KQ);
    const int32_t idxLen = whole ? K : KQ;
    for (int32_t j = 0; j < idxLen; ++j) {
        idx.SetValue(j, (uint32_t)((j / DGS) * 4));
    }
    AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID0);

    for (int32_t n = 0; n < N; ++n) {
        DataCopy(wRow, wGm[n * KQ], KQ);
        DataCopy(sRow, sGm[n * SBP], SBP);
        DataCopy(szRow, szGm[n * SBP], SBP);
        PipeBarrier<PIPE_ALL>();
        Cast(sTab, sRow, RoundMode::CAST_NONE, SBP);
        Cast(szTab, szRow, RoundMode::CAST_NONE, SBP);
        if (whole) {
            // ---- A) 整 K：2 次 Gather 展开 + 4 相位合并成 4 条大指令 ----
            Gather(sExp, sTab, idx, 0, (uint32_t)K);
            Gather(szExp, szTab, idx, 0, (uint32_t)K);
            PipeBarrier<PIPE_V>();
            for (int32_t p = 0; p < 4; ++p) {
                if (p == 0) {
                    And(ph, wRow, maskT, KQ);
                } else {
                    ShiftRight(ph[p * KQ], wRow, (uint16_t)(4 * p), KQ);
                    And(ph[p * KQ], ph[p * KQ], maskT, KQ);
                }
            }
            PipeBarrier<PIPE_V>();
            Cast(phF, ph.ReinterpretCast<int16_t>(), RoundMode::CAST_NONE, K);
            Mul(phF, phF, sExp, K);
            Sub(phF, phF, szExp, K);
            Cast(oRow, phF, RoundMode::CAST_NONE, K);
        } else {
            // ---- B) 逐相位（K 太大，Gather count 受限）----
            for (int32_t p = 0; p < 4; ++p) {
                if (p == 0) {
                    And(ph, wRow, maskT, KQ);
                } else {
                    ShiftRight(ph, wRow, (uint16_t)(4 * p), KQ);
                    And(ph, ph, maskT, KQ);
                }
                Cast(phF, ph.ReinterpretCast<int16_t>(), RoundMode::CAST_NONE, KQ);
                Gather(sExp, sTab[p * GPP], idx, 0, (uint32_t)KQ);
                Gather(szExp, szTab[p * GPP], idx, 0, (uint32_t)KQ);
                PipeBarrier<PIPE_V>();
                Mul(phF, phF, sExp, KQ);
                Sub(phF, phF, szExp, KQ);
                Cast(oRow[p * KQ], phF, RoundMode::CAST_NONE, KQ);
            }
        }
        PipeBarrier<PIPE_ALL>();
        DataCopy(oGm[n * K], oRow, K);
        PipeBarrier<PIPE_ALL>();
    }
}

#ifndef ASCENDC_CPU_DEBUG
void dequant_w4_do(uint32_t b, void *st, uint8_t *wq, uint8_t *scales, uint8_t *szs, uint8_t *out,
                   int32_t K, int32_t N)
{
    dequant_w4<<<b, nullptr, st>>>(wq, scales, szs, out, K, N);
}
#endif
