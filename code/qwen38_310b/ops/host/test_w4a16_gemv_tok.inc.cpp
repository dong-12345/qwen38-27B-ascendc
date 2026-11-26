/**
 * test_w4a16_gemv_tok — w4a16_gemv_tok（按 token 批处理 GEMV）的 host 侧验证片段
 *
 * ⚠️ 本文件**不参与编译**（CMakeLists 只把 host/main.cpp 和 host/matmul_tiling.cpp 等显式列出），
 *    它是给 lead 直接粘贴到 ascendc_ops/host/main.cpp 用的三段代码：
 *      ① 文件头部的 kernel 声明（放在其它 xxx_do 声明旁边，约 main.cpp:82 之后）
 *      ② test_w4a16_gemv_tok() 函数体（放在 test_w4a16_gemv_batch() 之后，匿名 namespace 内）
 *      ③ main() 里的分发分支（约 main.cpp:1286 附近）
 *
 * 验证方法（离线设计，未上板）：
 *   - golden：用旧 kernel w4a16_gemv_gen 对 M 个**不同 act 行**各跑一次（等价于现在的 prefill 逐 token 路径）；
 *   - 新 kernel：w4a16_gemv_tok 一次跑 M 个 token；
 *   - 对拍 max_abs / mean_abs（预期 1e-2 ~ 5e-2 量级，见 kernel 文件头的误差分析）；
 *   - M=1 时两者数学上逐位等价（累加顺序也一致），max_abs 应 ≈ 0（<1e-3），这是 bring-up 的第一道闸；
 *   - 计时：old = M × gen 单次（= 现在 prefill 的真实成本），new = tok(M) 一次，给出 tok/s 与加速比。
 *
 * 数据要求：先跑 python3 scripts/gen_data.py（生成 input/gact.bin、input/gwq.bin、input/gsc.bin、input/gsz.bin），
 *          与 test_w4a16_gemv_gen / test_w4a16_gemv_batch 用的是同一份。
 * 环境变量：GEMVT_K(默认5120) GEMVT_N(默认17408) GEMVT_M(默认8)
 *
 * 新增 kernel 后必须重跑 cmake（CMakeLists 用 file(GLOB kernels/*.cpp)）：
 *   cd build && cmake .. >/dev/null && make -j4 && cd .. && cmake --install build >/dev/null
 *   cp -f out/bin/qwen38_ops_test ./
 * 运行：
 *   export LD_LIBRARY_PATH=$PWD/out/lib:$PWD/out/lib64:/usr/local/Ascend/ascend-toolkit/latest/lib64:/usr/local/Ascend/ascend-toolkit/latest/runtime/lib64:$LD_LIBRARY_PATH
 *   timeout -s KILL 300 ./qwen38_ops_test w4a16_gemv_tok
 */

// ==================== ① 声明（放 main.cpp 顶部其它 xxx_do 声明处） ====================
void w4a16_gemv_tok_do(uint32_t b, void *st, uint8_t *act, uint8_t *wq, uint8_t *scales,
                       uint8_t *szs, uint8_t *out, int32_t K, int32_t N, int32_t Mtok);

// ==================== ② 测试函数（放 main.cpp 匿名 namespace 内，test_w4a16_gemv_batch 之后） ====================
int test_w4a16_gemv_tok()
{
    const int32_t K = getenv("GEMVT_K") ? atoi(getenv("GEMVT_K")) : 5120;
    const int32_t N = getenv("GEMVT_N") ? atoi(getenv("GEMVT_N")) : 17408;
    const int32_t M = getenv("GEMVT_M") ? atoi(getenv("GEMVT_M")) : 8;
    const int32_t NG = K / 32;
    const int32_t SBP = (NG + 15) / 16 * 16;
    const size_t actBytes = (size_t)K * sizeof(uint16_t);                 // 单个 token 的 act
    const size_t wBytes = (size_t)N * (K / 4) * sizeof(uint16_t);         // 权重 int4（相位交织）
    const size_t sBytes = (size_t)N * SBP * sizeof(uint16_t);             // scale / sz
    const size_t oBytes = (size_t)N * sizeof(uint16_t);                   // 单个 token 的输出
    const size_t mActBytes = (size_t)M * actBytes;
    const size_t mOutBytes = (size_t)M * oBytes;

    // ---- 构造 M 行不同的 act：直接对 gact.bin 的 half 位模式做循环搬移（不做浮点运算，避免溢出/精度干扰）----
    std::vector<uint16_t> hActOne(K);
    size_t readSz = actBytes;     // ReadFile 的 fileSize 是引用参数，必须传左值
    ReadFile("./input/gact.bin", readSz, (uint8_t *)hActOne.data(), actBytes);
    std::vector<uint16_t> hActM((size_t)M * K);
    for (int32_t m = 0; m < M; ++m) {
        const int32_t sh = (m * 37) % K;      // 每个 token 不同的循环移位
        for (int32_t k = 0; k < K; ++k) {
            hActM[(size_t)m * K + k] = hActOne[(k + sh) % K];
        }
    }

    uint8_t *act = alloc_dev(mActBytes);
    uint8_t *wq = alloc_dev(wBytes);
    uint8_t *sc = alloc_dev(sBytes);
    uint8_t *sz = alloc_dev(sBytes);
    uint8_t *outNew = alloc_dev(mOutBytes);    // 新 kernel：一次 M 个 token
    uint8_t *outRef = alloc_dev(mOutBytes);    // 旧 kernel：M 次调用拼起来
    uint8_t *outOne = alloc_dev(oBytes);       // 新 kernel M=1（bring-up 对拍）
    copy_h2d(wq, "./input/gwq.bin", wBytes);
    copy_h2d(sc, "./input/gsc.bin", sBytes);
    copy_h2d(sz, "./input/gsz.bin", sBytes);
    CHECK_ACL_RET(aclrtMemcpy(act, mActBytes, hActM.data(), mActBytes, ACL_MEMCPY_HOST_TO_DEVICE));

    // ---- 预热：规避"进程内首次调用异常"（见 docs/AscendC算子开发进度.md §10.40）----
    w4a16_gemv_gen_do(1, g_stream, act, wq, sc, sz, outRef, K, N, 99);
    CHECK_ACL_RET(aclrtSynchronizeStream(g_stream));
    w4a16_gemv_tok_do(1, g_stream, act, wq, sc, sz, outNew, K, N, M);
    CHECK_ACL_RET(aclrtSynchronizeStream(g_stream));

    // ---- golden：旧 kernel 逐 token（等价于当前 prefill 路径）----
    w4a16_gemv_gen_do(1, g_stream, act, wq, sc, sz, outRef, K, N, 99);   // 再跑一次保证稳态
    CHECK_ACL_RET(aclrtSynchronizeStream(g_stream));
    for (int32_t m = 0; m < M; ++m) {
        w4a16_gemv_gen_do(1, g_stream, act + (size_t)m * actBytes, wq, sc, sz,
                          outRef + (size_t)m * oBytes, K, N, 99);
    }
    CHECK_ACL_RET(aclrtSynchronizeStream(g_stream));

    // ---- 新 kernel（M 个 token 一次）----
    w4a16_gemv_tok_do(1, g_stream, act, wq, sc, sz, outNew, K, N, M);
    CHECK_ACL_RET(aclrtSynchronizeStream(g_stream));

    // ---- M=1 的 bring-up 对拍（数学上应逐位等价，max_abs < 1e-3）----
    w4a16_gemv_tok_do(1, g_stream, act, wq, sc, sz, outOne, K, N, 1);
    CHECK_ACL_RET(aclrtSynchronizeStream(g_stream));

    std::vector<uint16_t> hNew(mOutBytes / 2), hRef(mOutBytes / 2), hOne(oBytes / 2);
    CHECK_ACL_RET(aclrtMemcpy(hNew.data(), mOutBytes, outNew, mOutBytes, ACL_MEMCPY_DEVICE_TO_HOST));
    CHECK_ACL_RET(aclrtMemcpy(hRef.data(), mOutBytes, outRef, mOutBytes, ACL_MEMCPY_DEVICE_TO_HOST));
    CHECK_ACL_RET(aclrtMemcpy(hOne.data(), oBytes, outOne, oBytes, ACL_MEMCPY_DEVICE_TO_HOST));

    auto cmp = [](const std::vector<uint16_t> &a, const std::vector<uint16_t> &b, int32_t n,
                  double &maxAbs, double &meanAbs, int &nBad) {
        maxAbs = 0.0; meanAbs = 0.0; nBad = 0;
        for (int32_t i = 0; i < n; ++i) {
            float fa = aclFloat16ToFloat(a[i]), fb = aclFloat16ToFloat(b[i]);
            double d = std::fabs((double)fa - (double)fb);
            if (d > maxAbs) maxAbs = d;
            meanAbs += d;
            if (d > 2e-2) ++nBad;
        }
        meanAbs /= (n > 0 ? n : 1);
    };
    {
        double mx, mean; int nb;
        cmp(hOne, hRef, N, mx, mean, nb);
        printf("[gemv_tok] BRING-UP M=1 vs gen : max_abs=%.5f mean_abs=%.6f n(|d|>2e-2)=%d / %d\n",
               mx, mean, nb, N);
        if (mx > 1e-3) printf("[gemv_tok] ⚠️ M=1 就对不上：先查反量化/相位/scale 索引，不要看 M>1\n");
    }
    for (int32_t m = 0; m < M; ++m) {
        double mx, mean; int nb;
        cmp(std::vector<uint16_t>(hNew.begin() + (size_t)m * N, hNew.begin() + (size_t)(m + 1) * N),
            std::vector<uint16_t>(hRef.begin() + (size_t)m * N, hRef.begin() + (size_t)(m + 1) * N),
            N, mx, mean, nb);
        printf("[gemv_tok] token %d vs gen: max_abs=%.5f mean_abs=%.6f n(|d|>2e-2)=%d\n", m, mx, mean, nb);
    }
    {
        double mx, mean; int nb;
        cmp(hNew, hRef, M * N, mx, mean, nb);
        printf("[gemv_tok] 全部 %d token   : max_abs=%.5f mean_abs=%.6f n(|d|>2e-2)=%d / %d\n",
               M, mx, mean, nb, M * N);
    }

    // ---- 计时：old = M × gen（一次 token 一遍权重），new = tok(M) ----
    double msOldOne = 1e9;
    for (int it = 0; it < 3; ++it) {
        auto t0 = std::chrono::high_resolution_clock::now();
        w4a16_gemv_gen_do(1, g_stream, act, wq, sc, sz, outRef, K, N, 99);
        CHECK_ACL_RET(aclrtSynchronizeStream(g_stream));
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        if (ms < msOldOne) msOldOne = ms;
    }
    double msOld = msOldOne * (double)M;

    double msNew = 1e9;
    for (int it = 0; it < 3; ++it) {
        auto t0 = std::chrono::high_resolution_clock::now();
        w4a16_gemv_tok_do(1, g_stream, act, wq, sc, sz, outNew, K, N, M);
        CHECK_ACL_RET(aclrtSynchronizeStream(g_stream));
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        if (ms < msNew) msNew = ms;
    }

    const double gb = (double)wBytes / 1e9;
    const double gbAll = (double)(wBytes + 2 * sBytes) / 1e9;
    printf("[gemv_tok] K=%d N=%d M=%d\n", K, N, M);
    printf("[gemv_tok] old(gen×%d) %9.2f ms  (%5.2f ms/token, 权重 %.2f GB/s, 含 scale %.2f GB/s)\n",
           M, msOld, msOld / M, gb / (msOldOne / 1000.0), gbAll / (msOldOne / 1000.0));
    printf("[gemv_tok] new(tok M=%d) %9.2f ms  (%5.2f ms/token, 权重 %.2f GB/s, 含 scale %.2f GB/s)\n",
           M, msNew, msNew / M, gb / (msNew / 1000.0), gbAll / (msNew / 1000.0));
    printf("[gemv_tok] speedup %.2fx   -> 折算 prefill %.3f token/s（单矩阵；整模型按 27G/13.5GB 等比放大）\n",
           msOld / msNew, (msNew > 0 ? (double)M / (msNew / 1000.0) : 0.0));

    copy_d2h("./output/gout_tok_new.bin", outNew, mOutBytes);
    copy_d2h("./output/gout_tok_ref.bin", outRef, mOutBytes);
    return 0;
}

// ==================== ③ main() 分发（粘贴到 test_w4a16_gemv_batch 分支之后） ====================
//     } else if (test == "w4a16_gemv_tok") {
//         rc = test_w4a16_gemv_tok();
