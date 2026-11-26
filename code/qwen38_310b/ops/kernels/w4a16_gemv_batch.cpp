/**
 * w4a16_gemv_batch — 行批处理 W4A16 GEMV（M=1 解码主路径提速版）
 *
 *   out[n] = Σ_g s[n,g]·(Σ_{k∈g} q[n,k]·a[k]) − Σ_g sz[n,g]·suma[g]
 *
 * 旧版 w4a16_gemv_gen 每行输出要 ~45 条向量指令（每条后面一个 PipeBarrier<V>），
 * 实测 3.2 µs/行 / 0.80 GB/s —— 瓶颈是"指令条数 × 固定开销"，不是带宽
 * （44.6 MB 权重理论上只需 4.4 ms，实测 55.6 ms）。
 *
 * 本版两个手段：
 *  A) **行批处理**：一次算 R 行（R 由 UB 预算与 N 的整除性选），每批只付一次指令开销。
 *  B) **4 个相位合并成大指令**：4 个掩码取出的相位分别写入 qf[p*nQ] 连续区，
 *     之后的 Cast / Mul(act) / 折叠 Add 都在 4*nQ 上一次完成。
 *
 * 相位抽取（掩码值本身是 q·16^p，fac[p]=16^-p 预先折进 actR，省掉 3 次 ShiftRight）：
 *   p=0: And 0x000F → 0..15      p=1: And 0x00F0 → 0..240
 *   p=2: And 0x0F00 → 0..3840    p=3: ShiftRight 12 → 0..15
 *   ⚠ 不能用 And 0xF000：值到 61440，int16 会变成负数（实测过，会整体算错）。
 *
 * 32 元素量化组的两级归约（向量指令偏移必须 32B 对齐，所以 32→1 不能一步做）：
 *   ① Add(偏移 16 元素 = 32B) → 每组 16 个部分和
 *   ② 1 次 Gather 收成 [t][r][g]（t 维 16 层在外），t 维用 4 次 Add 合并成组和
 *   ③ 组和压实 × 组 scale（小 Gather 展开），4 个相位累加
 *
 * 偏置 Σ_g sz·suma：suma 与输出行无关，入口算一次。
 * 输出攒在 16 行暂存区，满 16 行再 DataCopy（避免 <32B 的小拷贝）。
 * Gather 索引是**字节偏移**（probe_gather 实测），fp16 源要 ×2。
 *
 * 约束：K % 128 == 0，KQ % 32 == 0，K ≤ 17408；N % 16 == 0；R 取 16 的约数（无尾巴）。
 */
#include "kernel_operator.h"

using namespace AscendC;

constexpr int32_t BGS = 32;                 // 量化分组大小
constexpr int32_t NPT = 16;                 // 每组保留的部分和个数（32/2，32B 对齐折叠后）
constexpr size_t UB_BUDGET = 240 * 1024;    // 留出余量的 UB 预算

extern "C" __global__ __aicore__ void w4a16_gemv_batch(GM_ADDR act, GM_ADDR wq, GM_ADDR scales,
                                                       GM_ADDR szs, GM_ADDR out, int32_t K,
                                                       int32_t N, int32_t stage)
{
    (void)stage;
    TPipe pipe;
    const int32_t KQ = K / 4;
    const int32_t NG = K / BGS;
    const int32_t GPP = KQ / BGS;
    const int32_t SBP = (NG + 15) / 16 * 16;
    int32_t SPD = 8;
    while (SPD < SBP) SPD <<= 1;             // sz 行补齐到 2 的幂（补零后可按行安全对折）
    const int32_t OM = (stage == 98) ? 1 : 2;   // Gather 索引单位：1=元素 2=字节（用 stage 做实验开关）

    // ---- 选 R：① R | 16 ② R | N ③ UB 放得下 ④ Gather count ≤ 8192 ----
    auto ubBytes = [&](int32_t r) -> size_t {
        size_t nQ = (size_t)r * KQ, nG = (size_t)NPT * r * GPP, nRG = (size_t)r * GPP;
        return 2 * nQ * 2                     // W0 + W1
             + 4 * nQ * 2                     // actR
             + 4 * nQ * 2                     // qf（4 相位共用，uint16 视图）
             + 4 * nQ * 2                     // qg（Cast 目标，half）
             + nQ * 2                         // maskT
             + 4 * nG * 2                     // gsT（4 相位）
             + 4 * nRG * 2                    // scE（4 相位）
             + nRG * 2                        // accH → 复用 gsC 空间
             + nRG * 4                        // accF
             + nG * 4                         // idxG
             + nRG * 4 * 4                    // idxSp[4]
             + 4 * nRG * 4                    // idxC
             + (size_t)r * SBP * 2            // scC
             + ((size_t)r * SBP + 16) * 2      // szC（尾部留 16 个零，供补齐 Gather）
             + (size_t)r * SPD * 4 * 2        // szF / sumaP（SPD 口径）
             + (size_t)r * SPD * 2             // szPH
             + (size_t)r * SPD * 4             // idxZ
             + (size_t)K * 2 + 64;            // aH + out
    };
    int32_t R = 0;
    for (int32_t cand = 8; cand >= 1; --cand) {
        if ((16 % cand) != 0) continue;
        if ((N % cand) != 0) continue;
        if (ubBytes(cand) > UB_BUDGET) continue;
        if ((size_t)NPT * cand * GPP > 8192) continue;
        R = cand;
        break;
    }
    if (R == 0) R = 1;   // 兜底（实际形状不会走到；host 侧应保证前提）

    const int32_t nQ = R * KQ;
    const int32_t nG = NPT * R * GPP;
    const int32_t nRG = R * GPP;
    const int32_t nRGP = (nRG + 15) & ~15;   // Cast(half->float) 的 count 必须 32B 对齐

    GlobalTensor<half> actGm, sGm, szGm, oGm;
    GlobalTensor<uint16_t> wGm;
    actGm.SetGlobalBuffer((__gm__ half *)act, K);
    wGm.SetGlobalBuffer((__gm__ uint16_t *)wq, (int64_t)N * KQ);
    sGm.SetGlobalBuffer((__gm__ half *)scales, (int64_t)N * SBP);
    szGm.SetGlobalBuffer((__gm__ half *)szs, (int64_t)N * SBP);
    oGm.SetGlobalBuffer((__gm__ half *)out, N);

    TBuf<TPosition::VECCALC> bW0, bW1, bActR, bQf, bMask, bGs, bSc, bAccF, bQg, bIdxG, bIdxS, bIdxC,
        bAH, bScC, bSzC, bSzPH, bSzF, bSuma, bIdxZ, bOut;
    pipe.InitBuffer(bW0, nQ * 2);
    pipe.InitBuffer(bW1, nQ * 2);
    pipe.InitBuffer(bActR, (size_t)4 * nQ * 2);
    pipe.InitBuffer(bQf, (size_t)4 * nQ * 2);
    pipe.InitBuffer(bMask, nQ * 2);
    pipe.InitBuffer(bGs, (size_t)4 * nG * 2);
    pipe.InitBuffer(bSc, (size_t)4 * nRG * 2);
    pipe.InitBuffer(bAccF, (size_t)nRGP * 4);
    pipe.InitBuffer(bQg, (size_t)4 * nQ * 2);
    pipe.InitBuffer(bIdxG, (size_t)nG * 4);
    pipe.InitBuffer(bIdxS, (size_t)4 * nRG * 4);
    pipe.InitBuffer(bIdxC, (size_t)4 * nRG * 4);
    pipe.InitBuffer(bAH, (size_t)K * 2);
    pipe.InitBuffer(bScC, (size_t)R * SBP * 2);
    pipe.InitBuffer(bSzC, (size_t)R * SBP * 2 + 32);
    pipe.InitBuffer(bSzPH, (size_t)R * SPD * 2);
    pipe.InitBuffer(bSzF, (size_t)R * SPD * 4);
    pipe.InitBuffer(bSuma, (size_t)R * SPD * 4);
    pipe.InitBuffer(bIdxZ, (size_t)R * SPD * 4);
    pipe.InitBuffer(bOut, 16 * 2);

    LocalTensor<uint16_t> W0 = bW0.Get<uint16_t>();
    LocalTensor<uint16_t> W1 = bW1.Get<uint16_t>();
    LocalTensor<half> actR = bActR.Get<half>();
    LocalTensor<half> qf = bQf.Get<half>();
    LocalTensor<uint16_t> maskT = bMask.Get<uint16_t>();
    LocalTensor<half> gsT = bGs.Get<half>();
    LocalTensor<half> scE = bSc.Get<half>();
    LocalTensor<float> accF = bAccF.Get<float>();
    LocalTensor<half> qg = bQg.Get<half>();
    LocalTensor<uint32_t> idxG = bIdxG.Get<uint32_t>();
    LocalTensor<uint32_t> idxS = bIdxS.Get<uint32_t>();
    LocalTensor<uint32_t> idxC = bIdxC.Get<uint32_t>();
    LocalTensor<half> aH = bAH.Get<half>();
    LocalTensor<half> scC = bScC.Get<half>();
    LocalTensor<half> szC = bSzC.Get<half>();
    LocalTensor<half> szPH = bSzPH.Get<half>();
    LocalTensor<float> szF = bSzF.Get<float>();
    LocalTensor<float> sumaP = bSuma.Get<float>();
    LocalTensor<uint32_t> idxZ = bIdxZ.Get<uint32_t>();
    LocalTensor<half> oStage = bOut.Get<half>();

    // ---------------- 1) act 预处理（每 token / 每 GEMV 一次） ----------------
    DataCopy(aH, actGm, K);
    PipeBarrier<PIPE_ALL>();
    const float fac[4] = {1.0f, 1.0f / 16.0f, 1.0f / 256.0f, 1.0f};

    // suma[g] = Σ_{k∈g} act[k]：先借用 actR 前 KQ 个元素当 scratch
    // ⚠ 偏置项必须用**原始 act** 的组和，不能带 fac[p]（fac 只是补偿掩码的 16^p，
    //    只用于主路径 q·act；这里若乘了 fac，偏置会差 16/256 倍 —— 曾经导致整体算错）。
    Duplicate(sumaP, 0.0f, R * SPD);
    {
        LocalTensor<half> w2 = actR;
        for (int32_t p = 0; p < 4; ++p) {
            Muls(w2, aH[p * KQ], (half)1.0f, KQ);
            Add(w2, w2, w2[16], KQ - 16);
            AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
            for (int32_t g = 0; g < GPP; ++g) {
                float s = 0.0f;
                for (int32_t t = 0; t < NPT; ++t) s += (float)w2.GetValue(g * 32 + t);
                for (int32_t r = 0; r < R; ++r) sumaP.SetValue(r * SPD + p * GPP + g, s);
            }
            AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
        }
    }
    // actR[p][r][j] = act[p*KQ+j]·fac[p]
    for (int32_t p = 0; p < 4; ++p) {
        for (int32_t r = 0; r < R; ++r) {
            Muls(actR[p * nQ + r * KQ], aH[p * KQ], (half)fac[p], KQ);
        }
    }

    // ---------------- 2) Gather 索引表（每 token 一次，单位=字节） ----------------
    // idxG[i], i = t*(R*GPP) + r*GPP + g  →  源 fp16 元素 (r*KQ + g*32 + t) → 字节 ×2
    for (int32_t i = 0; i < nG; ++i) {
        int32_t t = i / nRG;
        int32_t rem = i - t * nRG;
        int32_t r = rem / GPP;
        int32_t g = rem - r * GPP;
        idxG.SetValue(i, (uint32_t)((r * KQ + g * 32 + t) * OM));
    }
    // idxS[p][i], i = r*GPP + g → 源 fp16 元素 (r*SBP + p*GPP + g) → 字节 ×2
    for (int32_t p = 0; p < 4; ++p) {
        for (int32_t i = 0; i < nRG; ++i) {
            int32_t r = i / GPP;
            int32_t g = i - r * GPP;
            idxS.SetValue(p * nRG + i, (uint32_t)((r * SBP + p * GPP + g) * OM));
        }
    }
    // idxC[p*nRG + i] → 压缩后的组和：源字节偏移 (p*nG + i)*2
    for (int32_t i = 0; i < 4 * nRG; ++i) {
        idxC.SetValue(i, (uint32_t)(((i / nRG) * nG + (i % nRG)) * OM));
    }
    // idxZ[r*SPD + g] → 源字节偏移；g<NG 取 (r*SBP+g)，其余指向 szC 尾部的零元素
    for (int32_t r = 0; r < R; ++r) {
        for (int32_t g = 0; g < SPD; ++g) {
            uint32_t e = (g < NG) ? (uint32_t)(r * SBP + g) : (uint32_t)(R * SBP);
            idxZ.SetValue(r * SPD + g, e * 2);
        }
    }
    // szC 尾部 16 个元素清零（补齐 lane 取这里，保证乘出来是 0）
    Duplicate(szC[R * SBP], (half)0.0f, 16);
    Duplicate(oStage, (half)0.0f, 16);
    AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID0);

    // ---------------- 3) 主循环（预取下一批，与计算重叠；不用事件标志，靠 PIPE_ALL 栅栏）----------------
    const int32_t nB = N / R;
    DataCopy(W0, wGm[0], nQ);
    PipeBarrier<PIPE_ALL>();

    const uint16_t masks[4] = {0x000Fu, 0x00F0u, 0x0F00u, 0x000Fu};
    int32_t stageFill = 0;

    for (int32_t b = 0; b < nB; ++b) {
        const int32_t n0 = b * R;
        const bool even = ((b & 1) == 0);
        LocalTensor<uint16_t> W = even ? W0 : W1;

        DataCopy(scC, sGm[n0 * SBP], R * SBP);
        DataCopy(szC, szGm[n0 * SBP], R * SBP);
        PipeBarrier<PIPE_ALL>();

        if (b + 1 < nB) {
            LocalTensor<uint16_t> Wn = even ? W1 : W0;
            DataCopy(Wn, wGm[(b + 1) * R * KQ], nQ);
        }

        // ---- 4 个相位抽取到 qf[p*nQ] ----
        LocalTensor<uint16_t> qU = qf.ReinterpretCast<uint16_t>();
        for (int32_t p = 0; p < 3; ++p) {
            Duplicate(maskT, masks[p], nQ);
            And(qU[p * nQ], W, maskT, nQ);
        }
        ShiftRight(qU[3 * nQ], W, (uint16_t)12, nQ);
        // ShiftRight 可能是算术右移（高位字 0x8xxx 会带符号扩展），再掩一次保证 0..15
        Duplicate(maskT, (uint16_t)0x000F, nQ);
        And(qU[3 * nQ], qU[3 * nQ], maskT, nQ);
        if (stage == 97) {
            // 逐相位版本（v1 风格）：用来二分"4 相位合并"是否引入了错误
            for (int32_t p = 0; p < 4; ++p) {
                Cast(qg[p * nQ], qU[p * nQ].ReinterpretCast<int16_t>(), RoundMode::CAST_NONE, nQ);
                Mul(qg[p * nQ], qg[p * nQ], actR[p * nQ], nQ);
                Add(qg[p * nQ], qg[p * nQ], qg[p * nQ + 16], nQ - 16);
            }
            for (int32_t p = 0; p < 4; ++p) {
                Gather(gsT[p * nG], qg[p * nQ], idxG, 0, (uint32_t)nG);
            }
        } else {
            // ---- 统一 Cast / Mul(act) / 折叠（作用在 4*nQ 上）----
            Cast(qg, qU.ReinterpretCast<int16_t>(), RoundMode::CAST_NONE, 4 * nQ);
            Mul(qg, qg, actR, 4 * nQ);
            Add(qg, qg, qg[16], 4 * nQ - 16);
            // ---- 每相位 Gather 出 [t][r][g] ----
            for (int32_t p = 0; p < 4; ++p) {
                Gather(gsT[p * nG], qg[p * nQ], idxG, 0, (uint32_t)nG);
            }
        }
        // ---- t 维合并（4 个相位块结构相同，偏移 < 块长，可一次做完）----
        Add(gsT, gsT, gsT[8 * nRG], 4 * nG - 8 * nRG);
        Add(gsT, gsT, gsT[4 * nRG], 4 * nG - 4 * nRG);
        Add(gsT, gsT, gsT[2 * nRG], 4 * nG - 2 * nRG);
        Add(gsT, gsT, gsT[nRG], 4 * nG - nRG);

        // ---- 4 相位组和压实到 scE 前半（复用 gsT 尾部空间不可，故用临时 tensor）----
        LocalTensor<half> gsC = scE;   // 先放这里，乘 scale 前不会冲突
        Gather(gsC, gsT, idxC, 0, (uint32_t)(4 * nRG));
        // scale 展开到另一块缓冲：借用 gsT 前 4*nRG（此时组和已取走）
        LocalTensor<half> scL = gsT;
        for (int32_t p = 0; p < 4; ++p) {
            Gather(scL[p * nRG], scC, idxS[p * nRG], 0, (uint32_t)nRG);
        }
        Mul(gsC, gsC, scL, 4 * nRG);
        Add(gsC, gsC, gsC[2 * nRG], 2 * nRG);
        Add(gsC, gsC, gsC[nRG], nRG);

        // ---- 逐行收缩成 8 个部分和 ----
        // 只用"8 的倍数偏移"做互不重叠的折叠：lane i (i<8) 累加 g_i, g_{i+8}, g_{i+16}, ...
        // （非 2 的幂长度不能用对折：lane i 与 lane i+8 的内容会重叠 → 重复累加，实测算错）
        Cast(accF, gsC, RoundMode::CAST_NONE, nRGP);
        for (int32_t off = 8; off < GPP; off += 8) {
            Add(accF, accF, accF[off], (int32_t)(nRG - off));
        }

        // ---- 偏置项 Σ_g sz·suma ----
        // sz 按行补齐到 SPD(=2 的幂) 且补零，之后可以安全对折（行间距是 2 的幂，不会串行）
        Gather(szPH, szC, idxZ, 0, (uint32_t)(R * SPD));
        Cast(szF, szPH, RoundMode::CAST_NONE, R * SPD);
        Mul(szF, szF, sumaP, R * SPD);
        for (int32_t off = SPD / 2; off >= 8; off >>= 1) {
            Add(szF, szF, szF[off], (int32_t)(R * SPD - off));
        }

        // ---- 收尾：每行 8 个部分和标量合并 ----
        AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
        for (int32_t r = 0; r < R; ++r) {
            float dot = 0.0f, bias = 0.0f;
            for (int32_t t = 0; t < 8; ++t) {
                dot += accF.GetValue(r * GPP + t);
                bias += szF.GetValue(r * SPD + t);
            }
            oStage.SetValue(stageFill++, (half)(dot - bias));
        }
        AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
        if (stageFill == 16) {
            DataCopy(oGm[n0 + R - 16], oStage, 16);
            stageFill = 0;
        }
        PipeBarrier<PIPE_ALL>();
    }
}

#ifndef ASCENDC_CPU_DEBUG
void w4a16_gemv_batch_do(uint32_t b, void *st, uint8_t *act, uint8_t *wq, uint8_t *scales,
                         uint8_t *szs, uint8_t *out, int32_t K, int32_t N, int32_t stage)
{
    w4a16_gemv_batch<<<b, nullptr, st>>>(act, wq, scales, szs, out, K, N, stage);
}
#endif
