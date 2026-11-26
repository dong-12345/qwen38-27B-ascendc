/**
 * 自研 AscendC 算子测试驱动（310B kernel launch 模式）
 *
 * 用法: ./qwen38_ops_test <test_name>
 *   add          —— 官方 Add 样例（链路自检）
 *   gdn_gating   —— Qwen3.8 GDN 门控算子
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <chrono>

#include "acl/acl.h"
#include <aclnnop/aclnn_matmul.h>
#include "data_utils.h"
#include "tiling/platform/platform_ascendc.h"

#define CHECK_ACL_RET(x)                                                       \
    do {                                                                       \
        aclError __ret = (x);                                                   \
        if (__ret != ACL_ERROR_NONE) {                                          \
            const char *__msg = aclGetRecentErrMsg();                           \
            printf("[FAIL] acl error %d at %s:%d  msg=%s\n", (int)__ret,         \
                   __FILE__, __LINE__, __msg ? __msg : "(none)");               \
            std::exit(1);                                                       \
        }                                                                       \
    } while (0)

void add_custom_do(uint32_t blockDim, void *stream, uint8_t *x, uint8_t *y, uint8_t *z);
void gdn_gating_do(uint32_t blockDim, void *stream, uint8_t *a, uint8_t *b,
                   uint8_t *dt_bias, uint8_t *g_scale, uint8_t *g, uint8_t *beta);
void gdn_conv1d_do(uint32_t blockDim, void *stream, uint8_t *x, uint8_t *w_tap, uint8_t *y);
void gdn_l2norm_do(uint32_t blockDim, void *stream, uint8_t *x, uint8_t *y);
void gdn_cumsum_do(uint32_t blockDim, void *stream, uint8_t *g, uint8_t *g_cum);
void gdn_gated_rmsnorm_do(uint32_t blockDim, void *stream, uint8_t *x, uint8_t *z,
                          uint8_t *w, uint8_t *y);
void bench_stream_do(uint32_t blockDim, void *stream, uint8_t *src, uint8_t *out,
                     int32_t tileHalf, int32_t tilesPerBlock, int32_t mode);
void bench_compute_do(uint32_t blockDim, void *stream, uint8_t *out);
void w4a16_gemv_do(uint32_t blockDim, void *stream, uint8_t *act, uint8_t *wq,
                   uint8_t *scale, uint8_t *out, int32_t nrows);
void probe_g_do(uint32_t b, void *st, uint8_t *a, uint8_t *wq, uint8_t *s, uint8_t *o, int32_t mode);
void cube_matmul_do(uint32_t blockDim, void *stream, uint8_t *a, uint8_t *b, uint8_t *c,
                    uint8_t *workspace, uint8_t *tiling);
size_t GenerateMatmulTiling(int M, int N, int K, uint8_t *tilingBuf, size_t bufSize,
                            int *outCoreNum, size_t *outLocalMem);
size_t GetMatmulWorkspaceSize();
void probe_l1_do(uint32_t b, void *st, uint8_t *out, int32_t mode);
void gdn_kkt_do(uint32_t b, void *st, uint8_t *k, uint8_t *gc, uint8_t *beta, uint8_t *a);
void gdn_tril_do(uint32_t b, void *st, uint8_t *a, uint8_t *t);
void gdn_wy_do(uint32_t b, void *st, uint8_t *t, uint8_t *k, uint8_t *v, uint8_t *beta,
               uint8_t *gc, uint8_t *w, uint8_t *u);
void gdn_chunk_h_do(uint32_t b, void *st, uint8_t *w, uint8_t *u, uint8_t *k, uint8_t *gc,
                    uint8_t *sIn, uint8_t *sOut, uint8_t *vnew);
void gdn_chunk_o_do(uint32_t b, void *st, uint8_t *q, uint8_t *k, uint8_t *gc, uint8_t *s,
                    uint8_t *vnew, uint8_t *o);
void gdn_rgdr_do(uint32_t b, void *st, uint8_t *q, uint8_t *k, uint8_t *v,
                 uint8_t *sIn, uint8_t *sOut, uint8_t *o, float beta, float g);
void gdn_rgdr_dbg_do(uint32_t b, void *st, uint8_t *q, uint8_t *k, uint8_t *v, uint8_t *sIn,
                     uint8_t *sOut, uint8_t *o, float beta, float decay, int32_t mode);
void full_attn_decode_do(uint32_t b, void *st, uint8_t *q, uint8_t *k, uint8_t *v,
                         uint8_t *cosv, uint8_t *sinv, uint8_t *gate, uint8_t *kcache,
                         uint8_t *vcache, uint8_t *out, int32_t pos);
void fp16_gemv_do(uint32_t b, void *st, uint8_t *act, uint8_t *w, uint8_t *out, int32_t K, int32_t N);
void rmsnorm_op_do(uint32_t b, void *st, uint8_t *x, uint8_t *w, uint8_t *y, int32_t H, float eps);
void vec_axpy_do(uint32_t b, void *st, uint8_t *a, uint8_t *b_, uint8_t *out, float s, int32_t N);
void mul_silu_do(uint32_t b, void *st, uint8_t *gate, uint8_t *up, uint8_t *out, int32_t N);
void conv1d_decode_do(uint32_t b, void *st, uint8_t *x, uint8_t *state, uint8_t *w, uint8_t *out, int32_t C);
void w4a16_gemv_gen_do(uint32_t b, void *st, uint8_t *act, uint8_t *wq, uint8_t *scales,
                       uint8_t *szs, uint8_t *out, int32_t K, int32_t N, int32_t stage);
void probe_gather_do(uint32_t blk, void *st, uint8_t *outA, uint8_t *outB, int32_t mode);
int run_engine_mini(void *stream);
int run_engine_27b(void *stream, const char *dir);
// perf-scout2 新增（多核/带宽复核）
void bench_mc_do(uint32_t blockDim, void *stream, uint8_t *src, uint8_t *out, int32_t tileHalf,
                 int32_t tilesPerBlock, int32_t mode, int32_t strideHalf, int32_t windowHalf);
void bench_compute_mc_do(uint32_t blockDim, void *stream, uint8_t *out, int32_t loops, int32_t mode);
void bench_stream2_do(uint32_t blockDim, void *stream, uint8_t *src, uint8_t *dst, int32_t tileHalf,
                      int32_t tilesPerBlock, int32_t mode, int32_t strideHalf, int32_t windowHalf);
void bench_dequant_do(uint32_t blockDim, void *stream, uint8_t *src, uint8_t *dst,
                      int32_t wordsPerTile, int32_t tilesPerBlock);

namespace {

aclrtStream g_stream = nullptr;

uint8_t *alloc_dev(size_t bytes)
{
    uint8_t *p = nullptr;
    CHECK_ACL_RET(aclrtMalloc((void **)&p, bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    return p;
}

uint8_t *alloc_host(size_t bytes)
{
    uint8_t *p = nullptr;
    CHECK_ACL_RET(aclrtMallocHost((void **)&p, bytes));
    return p;
}

void copy_h2d(uint8_t *dev, const std::string &path, size_t bytes)
{
    uint8_t *host = alloc_host(bytes);
    ReadFile(path.c_str(), bytes, host, bytes);
    CHECK_ACL_RET(aclrtMemcpy(dev, bytes, host, bytes, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL_RET(aclrtFreeHost(host));
}

void copy_d2h(const std::string &path, uint8_t *dev, size_t bytes)
{
    uint8_t *host = alloc_host(bytes);
    CHECK_ACL_RET(aclrtMemcpy(host, bytes, dev, bytes, ACL_MEMCPY_DEVICE_TO_HOST));
    WriteFile(path.c_str(), host, bytes);
    CHECK_ACL_RET(aclrtFreeHost(host));
}

int test_add()
{
    const uint32_t blockDim = 8;
    const size_t n = 8 * 2048 * sizeof(uint16_t);
    uint8_t *x = alloc_dev(n), *y = alloc_dev(n), *z = alloc_dev(n);
    copy_h2d(x, "./input/input_x.bin", n);
    copy_h2d(y, "./input/input_y.bin", n);
    add_custom_do(blockDim, g_stream, x, y, z);
    CHECK_ACL_RET(aclrtSynchronizeStream(g_stream));
    copy_d2h("./output/output_z.bin", z, n);
    return 0;
}

// Qwen3.8 GDN 关键尺寸
constexpr int32_t SEQ = 64;
constexpr int32_t HEADS = 48;
constexpr uint32_t BLOCK_NUM = 8;

int test_gdn_gating()
{
    const size_t aBytes = SEQ * HEADS * sizeof(uint16_t);
    const size_t bBytes = aBytes;
    const size_t smallBytes = HEADS * sizeof(float);
    const size_t outBytes = SEQ * HEADS * sizeof(float);

    uint8_t *a = alloc_dev(aBytes), *b = alloc_dev(bBytes);
    uint8_t *dt = alloc_dev(smallBytes), *gs = alloc_dev(smallBytes);
    uint8_t *g = alloc_dev(outBytes), *beta = alloc_dev(outBytes);

    copy_h2d(a, "./input/gdn_gating_a.bin", aBytes);
    copy_h2d(b, "./input/gdn_gating_b.bin", bBytes);
    copy_h2d(dt, "./input/gdn_gating_dt_bias.bin", smallBytes);
    copy_h2d(gs, "./input/gdn_gating_g_scale.bin", smallBytes);

    gdn_gating_do(BLOCK_NUM, g_stream, a, b, dt, gs, g, beta);
    CHECK_ACL_RET(aclrtSynchronizeStream(g_stream));

    copy_d2h("./output/gdn_gating_g.bin", g, outBytes);
    copy_d2h("./output/gdn_gating_beta.bin", beta, outBytes);
    return 0;
}

int test_gdn_conv1d()
{
    constexpr int32_t SEQ_ = 64;
    constexpr int32_t D_ = 10240;
    constexpr int32_t KERNEL_ = 4;
    constexpr uint32_t BLOCK = 8;
    const size_t xBytes = SEQ_ * D_ * sizeof(uint16_t);
    const size_t wBytes = KERNEL_ * D_ * sizeof(float);

    uint8_t *x = alloc_dev(xBytes), *w = alloc_dev(wBytes), *y = alloc_dev(xBytes);
    copy_h2d(x, "./input/gdn_conv1d_x.bin", xBytes);
    copy_h2d(w, "./input/gdn_conv1d_wtap.bin", wBytes);

    gdn_conv1d_do(BLOCK, g_stream, x, w, y);
    CHECK_ACL_RET(aclrtSynchronizeStream(g_stream));
    copy_d2h("./output/gdn_conv1d_y.bin", y, xBytes);
    return 0;
}

int test_gdn_l2norm()
{
    constexpr int32_t SEQ_ = 64;
    constexpr int32_t DQ_ = 2048;
    constexpr uint32_t BLOCK = 8;
    const size_t n = SEQ_ * DQ_ * sizeof(uint16_t);

    uint8_t *x = alloc_dev(n), *y = alloc_dev(n);
    copy_h2d(x, "./input/gdn_l2norm_x.bin", n);
    gdn_l2norm_do(BLOCK, g_stream, x, y);
    CHECK_ACL_RET(aclrtSynchronizeStream(g_stream));
    copy_d2h("./output/gdn_l2norm_y.bin", y, n);
    return 0;
}

int test_gdn_cumsum()
{
    constexpr int32_t SEQ_ = 64;
    constexpr int32_t HEADS_ = 48;
    constexpr uint32_t BLOCK = 8;
    const size_t n = SEQ_ * HEADS_ * sizeof(float);

    uint8_t *g = alloc_dev(n), *gc = alloc_dev(n);
    copy_h2d(g, "./input/gdn_cumsum_g.bin", n);
    gdn_cumsum_do(BLOCK, g_stream, g, gc);
    CHECK_ACL_RET(aclrtSynchronizeStream(g_stream));
    copy_d2h("./output/gdn_cumsum_gcum.bin", gc, n);
    return 0;
}

int test_gdn_gated_rmsnorm()
{
    constexpr int32_t SEQ_ = 64;
    constexpr int32_t DV_ = 6144;
    constexpr int32_t HD_ = 128;
    constexpr uint32_t BLOCK = 8;
    const size_t n = SEQ_ * DV_ * sizeof(uint16_t);
    const size_t wn = HD_ * sizeof(float);

    uint8_t *x = alloc_dev(n), *z = alloc_dev(n), *y = alloc_dev(n), *w = alloc_dev(wn);
    copy_h2d(x, "./input/gdn_gated_rmsnorm_x.bin", n);
    copy_h2d(z, "./input/gdn_gated_rmsnorm_z.bin", n);
    copy_h2d(w, "./input/gdn_gated_rmsnorm_w.bin", wn);
    gdn_gated_rmsnorm_do(BLOCK, g_stream, x, z, w, y);
    CHECK_ACL_RET(aclrtSynchronizeStream(g_stream));
    copy_d2h("./output/gdn_gated_rmsnorm_y.bin", y, n);
    return 0;
}

int test_bench_stream()
{
    const size_t bytes = 256ull << 20;            // 256MB total (per run)
    const size_t totalHalf = bytes / 2;

    uint8_t *src = alloc_dev(bytes);
    uint8_t *out = alloc_dev((size_t)32768 * 2 * 32);
    uint8_t *host = alloc_host(bytes);
    std::memset(host, 0x3c, bytes);
    CHECK_ACL_RET(aclrtMemcpy(src, bytes, host, bytes, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL_RET(aclrtFreeHost(host));

    const int32_t tileHalf = 8192;                // 16KB tile
    const int32_t modes[] = {0, 1};
    const int32_t blocksList[] = {1, 2, 4, 8};
    for (int32_t mode : modes) {
        for (int32_t blocks : blocksList) {
            int32_t tiles = (int32_t)(totalHalf / ((size_t)blocks * tileHalf));
            size_t runBytes = (size_t)tiles * tileHalf * 2 * blocks;
            double best = 0.0;
            for (int it = 0; it < 3; ++it) {
                auto t0 = std::chrono::high_resolution_clock::now();
                bench_stream_do(blocks, g_stream, src, out, tileHalf, tiles, mode);
                CHECK_ACL_RET(aclrtSynchronizeStream(g_stream));
                auto t1 = std::chrono::high_resolution_clock::now();
                double sec = std::chrono::duration<double>(t1 - t0).count();
                double gbps = runBytes / 1e9 / sec;
                if (gbps > best) best = gbps;
            }
            printf("[bench_stream] mode=%d blocks=%2d tile=16KB -> %6.2f GB/s\n", mode, blocks, best);
        }
    }
    CHECK_ACL_RET(aclrtFree(src));
    CHECK_ACL_RET(aclrtFree(out));
    return 0;
}

int test_bench_compute()
{
    int64_t aiCore = -1, vecCore = -1, l2 = -1;
    aclError r1 = aclGetDeviceCapability(0, ACL_DEVICE_INFO_AI_CORE_NUM, &aiCore);
    aclError r2 = aclGetDeviceCapability(0, ACL_DEVICE_INFO_VECTOR_CORE_NUM, &vecCore);
    aclError r3 = aclGetDeviceCapability(0, ACL_DEVICE_INFO_L2_SIZE, &l2);
    printf("[device] ai_core=%lld (ret=%d)  vector_core=%lld (ret=%d)  l2_size=%lld (ret=%d)\n",
           (long long)aiCore, (int)r1, (long long)vecCore, (int)r2, (long long)l2, (int)r3);

    uint8_t *out = alloc_dev((size_t)8192 * 2 * 8);
    const int32_t blocksList[] = {1, 2, 4, 8};
    for (int32_t blocks : blocksList) {
        double best = 1e9;
        for (int it = 0; it < 3; ++it) {
            auto t0 = std::chrono::high_resolution_clock::now();
            bench_compute_do(blocks, g_stream, out);
            CHECK_ACL_RET(aclrtSynchronizeStream(g_stream));
            auto t1 = std::chrono::high_resolution_clock::now();
            double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            if (ms < best) best = ms;
        }
        // 每 block: 2048 loops * 2 ops * 8192 elems
        double elemOps = (double)blocks * 2048 * 2 * 8192;
        printf("[bench_compute] blocks=%d -> %7.2f ms, %.1f Gelem-ops/s\n", blocks, best,
               elemOps / (best / 1000.0) / 1e9);
    }
    CHECK_ACL_RET(aclrtFree(out));
    return 0;
}

int test_w4a16_gemv()
{
    constexpr int32_t K = 5120;
    constexpr int32_t N = 17408;
    const size_t actBytes = (size_t)K * sizeof(uint16_t);
    const size_t wBytes = (size_t)N * (K / 4) * sizeof(uint16_t);
    const size_t outBytes = (size_t)N * sizeof(uint16_t);
    const size_t sBytes = outBytes;

    uint8_t *act = alloc_dev(actBytes);
    uint8_t *wq = alloc_dev(wBytes);
    uint8_t *sc = alloc_dev(sBytes);
    uint8_t *out = alloc_dev(outBytes);
    copy_h2d(act, "./input/w4a16_act.bin", actBytes);
    copy_h2d(wq, "./input/w4a16_wq.bin", wBytes);
    copy_h2d(sc, "./input/w4a16_scale.bin", sBytes);

    // 1) 小规模正确性
    w4a16_gemv_do(1, g_stream, act, wq, sc, out, 256);
    CHECK_ACL_RET(aclrtSynchronizeStream(g_stream));
    printf("[w4a16_gemv] 256-row correctness run OK\n");
    copy_d2h("./output/w4a16_out.bin", out, outBytes);

    // 2) 全量性能
    double best = 1e9;
    for (int it = 0; it < 3; ++it) {
        auto t0 = std::chrono::high_resolution_clock::now();
        w4a16_gemv_do(1, g_stream, act, wq, sc, out, N);
        CHECK_ACL_RET(aclrtSynchronizeStream(g_stream));
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        if (ms < best) best = ms;
    }
    double gb = (double)wBytes / 1e9;
    printf("[w4a16_gemv] K=%d N=%d  %.2f ms  -> %.2f GB/s (weights only)\n", K, N, best,
           gb / (best / 1000.0));
    return 0;
}

int test_w4a16_probe()
{
    constexpr int32_t K = 5120;
    constexpr int32_t N = 17408;
    const size_t actBytes = (size_t)K * sizeof(uint16_t);
    const size_t wBytes = (size_t)N * (K / 4) * sizeof(uint16_t);
    const size_t outBytes = (size_t)N * sizeof(uint16_t);

    uint8_t *act = alloc_dev(actBytes);
    uint8_t *wq = alloc_dev(wBytes);
    uint8_t *sc = alloc_dev(outBytes);
    uint8_t *out = alloc_dev(outBytes);
    copy_h2d(act, "./input/w4a16_act.bin", actBytes);
    copy_h2d(wq, "./input/w4a16_wq.bin", wBytes);
    copy_h2d(sc, "./input/w4a16_scale.bin", outBytes);

    const char *names[] = {"g0_copy", "g1_and", "g2_shift", "g3_cast", "g4_one_row", "g5_256_rows"};
    for (int32_t mode = 0; mode < 6; ++mode) {
        probe_g_do(1, g_stream, act, wq, sc, out, mode);
        aclError rc = aclrtSynchronizeStream(g_stream);
        if (rc != ACL_ERROR_NONE) {
            printf("[PROBE FAIL] %-10s acl=%d msg=%s\n", names[mode], (int)rc,
                   aclGetRecentErrMsg() ? aclGetRecentErrMsg() : "(none)");
            return 1;
        }
        printf("[PROBE OK]   %s\n", names[mode]);
    }
    copy_d2h("./output/w4a16_probe_out.bin", out, 256 * sizeof(uint16_t));
    return 0;
}

int test_cube_matmul()
{
    constexpr int32_t M = 64;
    constexpr int32_t N = 5120;
    constexpr int32_t K = 5120;
    const size_t aBytes = (size_t)M * K * sizeof(uint16_t);
    const size_t bBytes = (size_t)K * N * sizeof(uint16_t);
    const size_t cBytes = (size_t)M * N * sizeof(float);

    uint8_t *a = alloc_dev(aBytes);
    uint8_t *b = alloc_dev(bBytes);
    uint8_t *c = alloc_dev(cBytes);
    copy_h2d(a, "./input/cube_A.bin", aBytes);
    copy_h2d(b, "./input/cube_B.bin", bBytes);

    uint8_t tilingHost[4096];
    std::memset(tilingHost, 0, sizeof(tilingHost));
    int coreNum = 1;
    size_t localMem = 0;
    size_t tilingBytes = GenerateMatmulTiling(M, N, K, tilingHost, sizeof(tilingHost),
                                             &coreNum, &localMem);
    if (tilingBytes == 0) {
        printf("[cube_matmul] tiling generation failed\n");
        return 1;
    }
    printf("[cube_matmul] tiling=%zu bytes, coreNum=%d, localMem=%zu (M=%d N=%d K=%d)\n",
           tilingBytes, coreNum, localMem, M, N, K);

    size_t wsSize = GetMatmulWorkspaceSize();
    uint8_t *ws = alloc_dev(wsSize);
    uint8_t *tilingDev = alloc_dev(tilingBytes);
    CHECK_ACL_RET(aclrtMemcpy(tilingDev, tilingBytes, tilingHost, tilingBytes,
                              ACL_MEMCPY_HOST_TO_DEVICE));

    double best = 1e9;
    for (int it = 0; it < 3; ++it) {
        auto t0 = std::chrono::high_resolution_clock::now();
        cube_matmul_do((uint32_t)coreNum, g_stream, a, b, c, ws, tilingDev);
        CHECK_ACL_RET(aclrtSynchronizeStream(g_stream));
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        if (ms < best) best = ms;
    }
    double gflop = 2.0 * M * N * K / 1e9;
    printf("[cube_matmul] %.2f ms -> %.2f GFLOP/s (%.3f TFLOP/s)\n", best,
           gflop / (best / 1000.0), gflop / (best / 1000.0) / 1000.0);
    copy_d2h("./output/cube_C.bin", c, cBytes);
    return 0;
}

int test_probe_l1()
{
    constexpr int32_t N = 1024;
    const size_t n = N * sizeof(uint16_t);
    uint8_t *out = alloc_dev(n);
    for (int32_t mode = 0; mode < 3; ++mode) {
        CHECK_ACL_RET(aclrtMemset(out, n, 0, n));
        probe_l1_do(1, g_stream, out, mode);
        aclError rc = aclrtSynchronizeStream(g_stream);
        if (rc != ACL_ERROR_NONE) {
            printf("[probe_l1] mode=%d acl=%d msg=%s\n", mode, (int)rc,
                   aclGetRecentErrMsg() ? aclGetRecentErrMsg() : "(none)");
            return 1;
        }
        char path[64];
        snprintf(path, sizeof(path), "./output/probe_l1_mode%d.bin", mode);
        copy_d2h(path, out, n);
        printf("[probe_l1] mode=%d done -> %s\n", mode, path);
    }
    return 0;
}

int test_gdn_kkt()
{
    constexpr int32_t T_ = 64;
    constexpr int32_t D_ = 128;
    const size_t kBytes = (size_t)T_ * D_ * sizeof(uint16_t);
    const size_t gBytes = (size_t)T_ * sizeof(float);
    const size_t aBytes = (size_t)T_ * T_ * sizeof(float);

    uint8_t *k = alloc_dev(kBytes);
    uint8_t *gc = alloc_dev(gBytes);
    uint8_t *beta = alloc_dev(gBytes);
    uint8_t *a = alloc_dev(aBytes);
    copy_h2d(k, "./input/gdn_kkt_k.bin", kBytes);
    copy_h2d(gc, "./input/gdn_kkt_gc.bin", gBytes);
    copy_h2d(beta, "./input/gdn_kkt_beta.bin", gBytes);

    auto t0 = std::chrono::high_resolution_clock::now();
    gdn_kkt_do(1, g_stream, k, gc, beta, a);
    CHECK_ACL_RET(aclrtSynchronizeStream(g_stream));
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    printf("[gdn_kkt] one chunk (T=64, D=128) -> %.3f ms\n", ms);
    copy_d2h("./output/gdn_kkt_a.bin", a, aBytes);
    return 0;
}

int test_gdn_tril()
{
    constexpr int32_t T_ = 64;
    const size_t n = (size_t)T_ * T_ * sizeof(float);
    uint8_t *a = alloc_dev(n);
    uint8_t *t = alloc_dev(n);
    copy_h2d(a, "./input/gdn_tril_a.bin", n);

    auto t0 = std::chrono::high_resolution_clock::now();
    gdn_tril_do(1, g_stream, a, t);
    CHECK_ACL_RET(aclrtSynchronizeStream(g_stream));
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    printf("[gdn_tril] 64x64 unit lower-tri inverse -> %.3f ms\n", ms);
    copy_d2h("./output/gdn_tril_t.bin", t, n);
    return 0;
}

int test_gdn_wy()
{
    constexpr int32_t T_ = 64;
    constexpr int32_t D_ = 128;
    const size_t tBytes = (size_t)T_ * T_ * sizeof(float);
    const size_t kvBytes = (size_t)T_ * D_ * sizeof(uint16_t);
    const size_t sBytes = (size_t)T_ * sizeof(float);

    uint8_t *t = alloc_dev(tBytes);
    uint8_t *k = alloc_dev(kvBytes);
    uint8_t *v = alloc_dev(kvBytes);
    uint8_t *beta = alloc_dev(sBytes);
    uint8_t *gc = alloc_dev(sBytes);
    uint8_t *w = alloc_dev(kvBytes);
    uint8_t *u = alloc_dev(kvBytes);
    copy_h2d(t, "./input/gdn_wy_t.bin", tBytes);
    copy_h2d(k, "./input/gdn_wy_k.bin", kvBytes);
    copy_h2d(v, "./input/gdn_wy_v.bin", kvBytes);
    copy_h2d(beta, "./input/gdn_wy_beta.bin", sBytes);
    copy_h2d(gc, "./input/gdn_wy_gc.bin", sBytes);

    auto t0 = std::chrono::high_resolution_clock::now();
    gdn_wy_do(1, g_stream, t, k, v, beta, gc, w, u);
    CHECK_ACL_RET(aclrtSynchronizeStream(g_stream));
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    printf("[gdn_wy] T@(k*beta*exp(g)) and T@(v*beta) -> %.3f ms\n", ms);
    copy_d2h("./output/gdn_wy_w.bin", w, kvBytes);
    copy_d2h("./output/gdn_wy_u.bin", u, kvBytes);
    return 0;
}

int test_gdn_chunk_h()
{
    constexpr int32_t C_ = 64;
    constexpr int32_t KD_ = 128;
    const size_t kvBytes = (size_t)C_ * KD_ * sizeof(uint16_t);
    const size_t gBytes = (size_t)C_ * sizeof(float);
    const size_t sBytes = (size_t)KD_ * KD_ * sizeof(float);

    uint8_t *w = alloc_dev(kvBytes);
    uint8_t *u = alloc_dev(kvBytes);
    uint8_t *k = alloc_dev(kvBytes);
    uint8_t *gc = alloc_dev(gBytes);
    uint8_t *sIn = alloc_dev(sBytes);
    uint8_t *sOut = alloc_dev(sBytes);
    uint8_t *vn = alloc_dev(kvBytes);
    copy_h2d(w, "./input/gdn_ch_h_w.bin", kvBytes);
    copy_h2d(u, "./input/gdn_ch_h_u.bin", kvBytes);
    copy_h2d(k, "./input/gdn_ch_h_k.bin", kvBytes);
    copy_h2d(gc, "./input/gdn_ch_h_gc.bin", gBytes);
    copy_h2d(sIn, "./input/gdn_ch_h_sin.bin", sBytes);

    auto t0 = std::chrono::high_resolution_clock::now();
    gdn_chunk_h_do(1, g_stream, w, u, k, gc, sIn, sOut, vn);
    CHECK_ACL_RET(aclrtSynchronizeStream(g_stream));
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    printf("[gdn_chunk_h] state update (C=64, K=V=128) -> %.3f ms\n", ms);
    copy_d2h("./output/gdn_ch_h_sout.bin", sOut, sBytes);
    copy_d2h("./output/gdn_ch_h_vnew.bin", vn, kvBytes);
    return 0;
}

int test_gdn_chunk_o()
{
    constexpr int32_t C_ = 64;
    constexpr int32_t KD_ = 128;
    const size_t kvBytes = (size_t)C_ * KD_ * sizeof(uint16_t);
    const size_t gBytes = (size_t)C_ * sizeof(float);
    const size_t sBytes = (size_t)KD_ * KD_ * sizeof(float);

    uint8_t *q = alloc_dev(kvBytes);
    uint8_t *k = alloc_dev(kvBytes);
    uint8_t *gc = alloc_dev(gBytes);
    uint8_t *s = alloc_dev(sBytes);
    uint8_t *vn = alloc_dev(kvBytes);
    uint8_t *o = alloc_dev(kvBytes);
    copy_h2d(q, "./input/gdn_ch_o_q.bin", kvBytes);
    copy_h2d(k, "./input/gdn_ch_o_k.bin", kvBytes);
    copy_h2d(gc, "./input/gdn_ch_o_gc.bin", gBytes);
    copy_h2d(s, "./input/gdn_ch_o_s.bin", sBytes);
    copy_h2d(vn, "./input/gdn_ch_o_vnew.bin", kvBytes);

    auto t0 = std::chrono::high_resolution_clock::now();
    gdn_chunk_o_do(1, g_stream, q, k, gc, s, vn, o);
    CHECK_ACL_RET(aclrtSynchronizeStream(g_stream));
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    printf("[gdn_chunk_o] output (C=64, K=V=128) -> %.3f ms\n", ms);
    copy_d2h("./output/gdn_ch_o_o.bin", o, kvBytes);
    return 0;
}

int test_gdn_rgdr()
{
    constexpr int32_t KD_ = 128;
    const size_t kBytes = (size_t)KD_ * sizeof(uint16_t);
    const size_t sBytes = (size_t)KD_ * KD_ * sizeof(float);

    uint8_t *q = alloc_dev(kBytes);
    uint8_t *k = alloc_dev(kBytes);
    uint8_t *v = alloc_dev(kBytes);
    uint8_t *sIn = alloc_dev(sBytes);
    uint8_t *sOut = alloc_dev(sBytes);
    uint8_t *o = alloc_dev(kBytes);
    copy_h2d(q, "./input/gdn_rg_q.bin", kBytes);
    copy_h2d(k, "./input/gdn_rg_k.bin", kBytes);
    copy_h2d(v, "./input/gdn_rg_v.bin", kBytes);
    copy_h2d(sIn, "./input/gdn_rg_sin.bin", sBytes);

    // beta/g 从文件读出后作为标量参数
    float bg[2] = {0.5f, -0.05f};
    {
        FILE *f = fopen("./input/gdn_rg_bg.bin", "rb");
        if (f != nullptr) {
            fread(bg, sizeof(float), 2, f);
            fclose(f);
        }
    }
    float decay = (float)std::exp(bg[1]);

    auto t0 = std::chrono::high_resolution_clock::now();
    gdn_rgdr_do(1, g_stream, q, k, v, sIn, sOut, o, bg[0], decay);
    CHECK_ACL_RET(aclrtSynchronizeStream(g_stream));
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    printf("[gdn_rgdr] one decode step (K=V=128, beta=%.3f, g=%.4f) -> %.3f ms\n",
           bg[0], bg[1], ms);
    copy_d2h("./output/gdn_rg_o.bin", o, kBytes);
    copy_d2h("./output/gdn_rg_sout.bin", sOut, sBytes);
    return 0;
}

// ============================ perf-scout2 新增 ============================

int test_memtest()
{
    // 逐块分配，测出设备可用内存上限（用于 27B 引擎可行性验证）
    const size_t CH = 256ull << 20;   // 256MB
    std::vector<void *> blocks;
    size_t total = 0;
    for (int i = 0; i < 100; ++i) {
        void *p = nullptr;
        aclError r = aclrtMalloc(&p, CH, ACL_MEM_MALLOC_HUGE_FIRST);
        if (r != ACL_ERROR_NONE) {
            printf("[memtest] alloc failed at %.2f GB (err %d)\n", total / 1e9, (int)r);
            break;
        }
        blocks.push_back(p);
        total += CH;
        if (total % (2ull << 30) == 0) printf("[memtest] %.0f GB ok\n", total / 1e9);
    }
    printf("[memtest] TOTAL allocatable: %.2f GB (blocks=%zu)\n", total / 1e9, blocks.size());
    for (auto p : blocks) aclrtFree(p);
    printf("[memtest] freed\n");
    return 0;
}

int test_platform()
{
    auto platform = platform_ascendc::PlatformAscendCManager::GetInstance("Ascend310B1");
    if (platform == nullptr) {
        printf("[platform] PlatformAscendCManager null\n");
        return 1;
    }
    uint32_t aic = platform->GetCoreNumAic();
    uint32_t aiv = platform->GetCoreNumAiv();
    printf("[platform] GetCoreNumAic=%u  GetCoreNumAiv=%u\n", aic, aiv);

    uint64_t ub = 0, l1 = 0, l2 = 0;
    platform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub);
    platform->GetCoreMemSize(platform_ascendc::CoreMemType::L1, l1);
    platform->GetCoreMemSize(platform_ascendc::CoreMemType::L2, l2);
    printf("[platform] UB=%llu B (%.1f KB)  L1=%llu B (%.1f KB)  L2=%llu B (%.1f MB)\n",
           (unsigned long long)ub, ub / 1024.0, (unsigned long long)l1, l1 / 1024.0,
           (unsigned long long)l2, l2 / 1048576.0);

    uint64_t bwUb = 0, bwL1 = 0, bwL2 = 0;
    platform->GetCoreMemBw(platform_ascendc::CoreMemType::UB, bwUb);
    platform->GetCoreMemBw(platform_ascendc::CoreMemType::L1, bwL1);
    platform->GetCoreMemBw(platform_ascendc::CoreMemType::L2, bwL2);
    printf("[platform] GetCoreMemBw UB=%llu B/cycle?  L1=%llu  L2=%llu\n",
           (unsigned long long)bwUb, (unsigned long long)bwL1, (unsigned long long)bwL2);
    return 0;
}

// 多核并行复核：总工作量固定，每 block 只做 1/blockDim
int test_bench_mc()
{
    const size_t totalBytes = 128ull << 20;              // 128MB total
    const size_t totalHalf = totalBytes / 2;
    uint8_t *src = alloc_dev(totalBytes);
    uint8_t *out = alloc_dev((size_t)8192 * 2 * 8);
    uint8_t *host = alloc_host(totalBytes);
    std::memset(host, 0x3c, totalBytes);
    CHECK_ACL_RET(aclrtMemcpy(src, totalBytes, host, totalBytes, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL_RET(aclrtFreeHost(host));

    const int32_t tileHalf = 8192;
    const int32_t blocksList[] = {1, 2, 4, 8};
    for (int32_t mode = 0; mode <= 1; ++mode) {
        for (int32_t blocks : blocksList) {
            int32_t tiles = (int32_t)(totalHalf / ((size_t)blocks * tileHalf));
            int32_t stride = tileHalf * tiles;               // block 各读自己的连续区间
            double best = 1e9;
            for (int it = 0; it < 3; ++it) {
                auto t0 = std::chrono::high_resolution_clock::now();
                bench_mc_do(blocks, g_stream, src, out, tileHalf, tiles, mode, stride, 0);
                CHECK_ACL_RET(aclrtSynchronizeStream(g_stream));
                auto t1 = std::chrono::high_resolution_clock::now();
                double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
                if (ms < best) best = ms;
            }
            double gb = (double)tiles * tileHalf * 2 * blocks / 1e9;
            printf("[mc_totalwork] mode=%d blocks=%d tiles/blk=%d -> %8.3f ms, aggregate %6.2f GB/s, "
                   "per-block %6.2f GB/s\n",
                   mode, blocks, tiles, best, gb / (best / 1000.0),
                   gb / (best / 1000.0) / blocks);
        }
    }
    CHECK_ACL_RET(aclrtFree(src));
    CHECK_ACL_RET(aclrtFree(out));
    return 0;
}

// 向量算力并行复核：总 loop 数固定，每 block 做 1/blockDim
int test_bench_scale()
{
    uint8_t *out = alloc_dev((size_t)8192 * 2 * 8);
    const int32_t totalLoops = 8192;
    const int32_t blocksList[] = {1, 2, 4, 8};
    const int32_t modes[] = {1, 3, 0};   // 1=独立Add 3=Muls 0=依赖链
    for (int32_t mode : modes) {
        for (int32_t blocks : blocksList) {
            int32_t loops = totalLoops / blocks;
            double best = 1e9;
            for (int it = 0; it < 3; ++it) {
                auto t0 = std::chrono::high_resolution_clock::now();
                bench_compute_mc_do(blocks, g_stream, out, loops, mode);
                CHECK_ACL_RET(aclrtSynchronizeStream(g_stream));
                auto t1 = std::chrono::high_resolution_clock::now();
                double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
                if (ms < best) best = ms;
            }
            double elems = (double)totalLoops * 2 * 8192;
            printf("[scale_compute] mode=%d blocks=%d loops/blk=%d -> %8.3f ms, aggregate %6.2f "
                   "Gelem-ops/s, per-block %6.2f\n",
                   mode, blocks, loops, best, elems / (best / 1000.0) / 1e9,
                   elems / (best / 1000.0) / 1e9 / blocks);
        }
    }
    CHECK_ACL_RET(aclrtFree(out));
    return 0;
}

// tile 大小细扫：确认 10.2GB/s 是硬件上限还是 16KB 小请求损失
int test_bench_stream2()
{
    const size_t totalBytes = 128ull << 20;
    const size_t totalHalf = totalBytes / 2;
    uint8_t *src = alloc_dev(totalBytes);
    uint8_t *dst = alloc_dev(totalBytes);
    uint8_t *host = alloc_host(totalBytes);
    std::memset(host, 0x3c, totalBytes);
    CHECK_ACL_RET(aclrtMemcpy(src, totalBytes, host, totalBytes, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL_RET(aclrtFreeHost(host));

    const int32_t tilesHalf[] = {256, 1024, 4096, 8192, 16384, 32768, 57344};  // 512B..112KB
    for (int32_t tileHalf : tilesHalf) {
        bool canAcc = (int64_t)tileHalf * 2 * 3 <= 240 * 1024;      // mode 0: 3*tile
        bool canRW = (int64_t)tileHalf * 2 * 4 <= 248 * 1024;       // mode 3: 4*tile
        int32_t modes[] = {1, 0, 3, 2};
        for (int32_t mode : modes) {
            if (mode == 0 && !canAcc) continue;
            if (mode == 3 && !canRW) continue;
            int32_t tiles = (int32_t)(totalHalf / tileHalf);
            double best = 1e9;
            for (int it = 0; it < 3; ++it) {
                auto t0 = std::chrono::high_resolution_clock::now();
                bench_stream2_do(1, g_stream, src, dst, tileHalf, tiles, mode, 0, 0);
                CHECK_ACL_RET(aclrtSynchronizeStream(g_stream));
                auto t1 = std::chrono::high_resolution_clock::now();
                double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
                if (ms < best) best = ms;
            }
            double bytes = (double)tiles * tileHalf * 2 * ((mode == 3) ? 2 : 1);
            printf("[stream2] tile=%6.1fKB mode=%d -> %8.3f ms, %6.2f GB/s\n", tileHalf / 512.0,
                   mode, best, bytes / 1e9 / (best / 1000.0));
        }
    }
    CHECK_ACL_RET(aclrtFree(src));
    CHECK_ACL_RET(aclrtFree(dst));
    return 0;
}

// L2 命中带宽 / 容量探测：逻辑读 64MB，地址在 window 内回绕
int test_bench_l2()
{
    const size_t totalBytes = 128ull << 20;
    uint8_t *src = alloc_dev(totalBytes);
    uint8_t *out = alloc_dev((size_t)8192 * 2 * 8);
    uint8_t *host = alloc_host(totalBytes);
    std::memset(host, 0x3c, totalBytes);
    CHECK_ACL_RET(aclrtMemcpy(src, totalBytes, host, totalBytes, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL_RET(aclrtFreeHost(host));

    const int32_t tileHalf = 4096;                       // 8KB
    const int32_t tiles = 8192;                          // 逻辑 64MB
    const int32_t windowsKB[] = {256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 0};
    for (int32_t wkb : windowsKB) {
        int32_t windowHalf = (wkb == 0) ? 0 : (int32_t)((int64_t)wkb * 1024 / 2);
        if (windowHalf > 0 && windowHalf < tileHalf) windowHalf = tileHalf;
        double best = 1e9;
        for (int it = 0; it < 3; ++it) {
            auto t0 = std::chrono::high_resolution_clock::now();
            bench_mc_do(1, g_stream, src, out, tileHalf, tiles, 1, 0, windowHalf);
            CHECK_ACL_RET(aclrtSynchronizeStream(g_stream));
            auto t1 = std::chrono::high_resolution_clock::now();
            double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            if (ms < best) best = ms;
        }
        double gb = (double)tiles * tileHalf * 2 / 1e9;
        printf("[bench_l2] window=%6dKB -> %8.3f ms, logical %6.2f GB/s\n", wkb, best,
               gb / (best / 1000.0));
    }
    CHECK_ACL_RET(aclrtFree(src));
    CHECK_ACL_RET(aclrtFree(out));
    return 0;
}

// Cube：固定权重（52MB），只改 M —— 若时间不随 M 涨，说明 100% 带宽受限
int test_cube_m()
{
    constexpr int32_t N = 5120;
    constexpr int32_t K = 5120;
    const int32_t Ms[] = {64, 128, 256, 512};
    const size_t bBytes = (size_t)K * N * sizeof(uint16_t);
    uint8_t *b = alloc_dev(bBytes);
    copy_h2d(b, "./input/cube_B.bin", bBytes);

    for (int32_t M : Ms) {
        const size_t aBytes = (size_t)M * K * sizeof(uint16_t);
        const size_t cBytes = (size_t)M * N * sizeof(float);
        uint8_t *a = alloc_dev(aBytes);
        uint8_t *c = alloc_dev(cBytes);
        copy_h2d(a, "./input/cube_A.bin", aBytes < (64 * K * 2) ? aBytes : (64 * K * 2));

        uint8_t tilingHost[4096];
        std::memset(tilingHost, 0, sizeof(tilingHost));
        int coreNum = 1;
        size_t localMem = 0;
        size_t tilingBytes = GenerateMatmulTiling(M, N, K, tilingHost, sizeof(tilingHost),
                                                 &coreNum, &localMem);
        if (tilingBytes == 0) {
            printf("[cube_m] M=%d tiling failed\n", M);
            continue;
        }
        size_t wsSize = GetMatmulWorkspaceSize();
        uint8_t *ws = alloc_dev(wsSize);
        uint8_t *tilingDev = alloc_dev(tilingBytes);
        CHECK_ACL_RET(aclrtMemcpy(tilingDev, tilingBytes, tilingHost, tilingBytes,
                                  ACL_MEMCPY_HOST_TO_DEVICE));
        double best = 1e9;
        for (int it = 0; it < 3; ++it) {
            auto t0 = std::chrono::high_resolution_clock::now();
            cube_matmul_do((uint32_t)coreNum, g_stream, a, b, c, ws, tilingDev);
            CHECK_ACL_RET(aclrtSynchronizeStream(g_stream));
            auto t1 = std::chrono::high_resolution_clock::now();
            double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            if (ms < best) best = ms;
        }
        double gflop = 2.0 * M * N * K / 1e9;
        double computeMs = gflop / 10000.0 * 1000.0;   // 10 TFLOPS 理论
        printf("[cube_m] M=%3d N=%d K=%d -> %7.2f ms | %7.2f GFLOP/s | weight %5.2f GB/s | "
               "compute@10TFLOPS %.2f ms\n",
               M, N, K, best, gflop / (best / 1000.0), (double)bBytes / 1e9 / (best / 1000.0),
               computeMs);
        CHECK_ACL_RET(aclrtFree(a));
        CHECK_ACL_RET(aclrtFree(c));
        CHECK_ACL_RET(aclrtFree(ws));
        CHECK_ACL_RET(aclrtFree(tilingDev));
    }
    CHECK_ACL_RET(aclrtFree(b));
    return 0;
}

// int4 -> fp16 反量化写 GM 的纯带宽成本（两段式第一段的地板）
int test_bench_dequant()
{
    const int32_t wordsPerTile = 8192;                    // 16KB int4 / tile
    const size_t srcBytes = 128ull << 20;                 // 128MB int4
    const size_t tiles = srcBytes / (wordsPerTile * 2);
    const size_t dstBytes = (size_t)tiles * wordsPerTile * 4 * 2;
    uint8_t *src = alloc_dev(srcBytes);
    uint8_t *dst = alloc_dev((size_t)srcBytes * 4);
    uint8_t *host = alloc_host(srcBytes);
    std::memset(host, 0x5a, srcBytes);
    CHECK_ACL_RET(aclrtMemcpy(src, srcBytes, host, srcBytes, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL_RET(aclrtFreeHost(host));

    double best = 1e9;
    for (int it = 0; it < 3; ++it) {
        auto t0 = std::chrono::high_resolution_clock::now();
        bench_dequant_do(1, g_stream, src, dst, wordsPerTile, (int32_t)tiles);
        CHECK_ACL_RET(aclrtSynchronizeStream(g_stream));
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        if (ms < best) best = ms;
    }
    double moved = (double)(srcBytes + dstBytes);
    printf("[bench_dequant] read %zu MB int4 + write %zu MB fp16 -> %.2f ms, %.2f GB/s "
           "(moved %.2f GB)\n",
           srcBytes >> 20, dstBytes >> 20, best, moved / 1e9 / (best / 1000.0), moved / 1e9);
    CHECK_ACL_RET(aclrtFree(src));
    CHECK_ACL_RET(aclrtFree(dst));
    return 0;
}

int test_gdn_rgdr_dbg()
{
    constexpr int32_t KD_ = 128;
    const size_t kBytes = (size_t)KD_ * sizeof(uint16_t);
    const size_t sBytes = (size_t)KD_ * KD_ * sizeof(float);

    uint8_t *q = alloc_dev(kBytes);
    uint8_t *k = alloc_dev(kBytes);
    uint8_t *v = alloc_dev(kBytes);
    uint8_t *sIn = alloc_dev(sBytes);
    uint8_t *sOut = alloc_dev(sBytes);
    uint8_t *o = alloc_dev(kBytes);
    copy_h2d(q, "./input/gdn_rg_q.bin", kBytes);
    copy_h2d(k, "./input/gdn_rg_k.bin", kBytes);
    copy_h2d(v, "./input/gdn_rg_v.bin", kBytes);
    copy_h2d(sIn, "./input/gdn_rg_sin.bin", sBytes);

    float beta = 0.5f;
    float decay = (float)std::exp(-0.05);
    for (int32_t mode = 0; mode < 3; ++mode) {
        gdn_rgdr_dbg_do(1, g_stream, q, k, v, sIn, sOut, o, beta, decay, mode);
        CHECK_ACL_RET(aclrtSynchronizeStream(g_stream));
        char p1[96], p2[96];
        snprintf(p1, sizeof(p1), "./output/gdn_rgdbg_s%d.bin", mode);
        snprintf(p2, sizeof(p2), "./output/gdn_rgdbg_o%d.bin", mode);
        copy_d2h(p1, sOut, sBytes);
        copy_d2h(p2, o, kBytes);
        printf("[gdn_rgdr_dbg] mode=%d done\n", mode);
    }
    return 0;
}

int test_full_attn_decode()
{
    constexpr int32_t QH_ = 24, KH_ = 4, D_ = 256, MD = 512, RH_ = 32;
    const int32_t pos = 5;
    const size_t qBytes = (size_t)QH_ * D_ * sizeof(uint16_t);
    const size_t kvBytes = (size_t)KH_ * D_ * sizeof(uint16_t);
    const size_t cBytes = (size_t)KH_ * MD * D_ * sizeof(uint16_t);
    const size_t rsBytes = (size_t)RH_ * sizeof(float);

    uint8_t *q = alloc_dev(qBytes);
    uint8_t *k = alloc_dev(kvBytes);
    uint8_t *v = alloc_dev(kvBytes);
    uint8_t *cs = alloc_dev(rsBytes);
    uint8_t *sn = alloc_dev(rsBytes);
    uint8_t *gt = alloc_dev(qBytes);
    uint8_t *kc = alloc_dev(cBytes);
    uint8_t *vc = alloc_dev(cBytes);
    uint8_t *out = alloc_dev(qBytes);
    copy_h2d(q, "./input/attn_q.bin", qBytes);
    copy_h2d(k, "./input/attn_k.bin", kvBytes);
    copy_h2d(v, "./input/attn_v.bin", kvBytes);
    copy_h2d(cs, "./input/attn_cos.bin", rsBytes);
    copy_h2d(sn, "./input/attn_sin.bin", rsBytes);
    copy_h2d(gt, "./input/attn_gate.bin", qBytes);
    copy_h2d(kc, "./input/attn_kcache.bin", cBytes);
    copy_h2d(vc, "./input/attn_vcache.bin", cBytes);

    auto t0 = std::chrono::high_resolution_clock::now();
    full_attn_decode_do(1, g_stream, q, k, v, cs, sn, gt, kc, vc, out, pos);
    CHECK_ACL_RET(aclrtSynchronizeStream(g_stream));
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    printf("[full_attn_decode] one decode step (24/4 heads, D=256, pos=%d) -> %.3f ms\n", pos, ms);
    copy_d2h("./output/attn_out.bin", out, qBytes);
    copy_d2h("./output/attn_kcache_out.bin", kc, cBytes);
    copy_d2h("./output/attn_vcache_out.bin", vc, cBytes);
    return 0;
}

int test_dense_ops()
{
    constexpr int32_t K_ = 512, N_ = 256, H_ = 512, C_ = 256;
    const size_t kb = (size_t)K_ * sizeof(uint16_t);
    const size_t nb = (size_t)N_ * sizeof(uint16_t);
    const size_t hb = (size_t)H_ * sizeof(uint16_t);
    const size_t wgemv = (size_t)N_ * K_ * sizeof(uint16_t);
    const size_t stb = (size_t)3 * C_ * sizeof(float);

    // 1) fp16_gemv
    {
        uint8_t *act = alloc_dev(kb), *w = alloc_dev(wgemv), *out = alloc_dev(nb);
        copy_h2d(act, "./input/d_gemv_act.bin", kb);
        copy_h2d(w, "./input/d_gemv_w.bin", wgemv);
        fp16_gemv_do(1, g_stream, act, w, out, K_, N_);
        CHECK_ACL_RET(aclrtSynchronizeStream(g_stream));
        copy_d2h("./output/d_gemv_out.bin", out, nb);
        printf("[dense_ops] fp16_gemv done\n");
    }
    // 2) rmsnorm
    {
        uint8_t *x = alloc_dev(hb), *w = alloc_dev(hb), *y = alloc_dev(hb);
        copy_h2d(x, "./input/d_rms_x.bin", hb);
        copy_h2d(w, "./input/d_rms_w.bin", hb);
        rmsnorm_op_do(1, g_stream, x, w, y, H_, 1e-6f);
        CHECK_ACL_RET(aclrtSynchronizeStream(g_stream));
        copy_d2h("./output/d_rms_y.bin", y, hb);
        printf("[dense_ops] rmsnorm done\n");
    }
    // 3) vec_axpy
    {
        uint8_t *a = alloc_dev(hb), *b = alloc_dev(hb), *o = alloc_dev(hb);
        copy_h2d(a, "./input/d_ax_a.bin", hb);
        copy_h2d(b, "./input/d_ax_b.bin", hb);
        vec_axpy_do(1, g_stream, a, b, o, 0.5f, H_);
        CHECK_ACL_RET(aclrtSynchronizeStream(g_stream));
        copy_d2h("./output/d_ax_out.bin", o, hb);
        printf("[dense_ops] vec_axpy done\n");
    }
    // 4) mul_silu
    {
        uint8_t *g = alloc_dev(hb), *u = alloc_dev(hb), *o = alloc_dev(hb);
        copy_h2d(g, "./input/d_ms_g.bin", hb);
        copy_h2d(u, "./input/d_ms_u.bin", hb);
        mul_silu_do(1, g_stream, g, u, o, H_);
        CHECK_ACL_RET(aclrtSynchronizeStream(g_stream));
        copy_d2h("./output/d_ms_out.bin", o, hb);
        printf("[dense_ops] mul_silu done\n");
    }
    // 5) conv1d_decode
    {
        size_t cb = (size_t)C_ * sizeof(uint16_t);
        size_t wb = (size_t)4 * C_ * sizeof(uint16_t);
        uint8_t *x = alloc_dev(cb), *st = alloc_dev(stb), *w = alloc_dev(wb), *o = alloc_dev(cb);
        copy_h2d(x, "./input/d_cv_x.bin", cb);
        copy_h2d(st, "./input/d_cv_state.bin", stb);
        copy_h2d(w, "./input/d_cv_w.bin", wb);
        conv1d_decode_do(1, g_stream, x, st, w, o, C_);
        CHECK_ACL_RET(aclrtSynchronizeStream(g_stream));
        copy_d2h("./output/d_cv_out.bin", o, cb);
        copy_d2h("./output/d_cv_state_out.bin", st, stb);
        printf("[dense_ops] conv1d_decode done\n");
    }
    return 0;
}

int test_w4a16_gemv_gen()
{
    constexpr int32_t K = 5120;
    constexpr int32_t N = 17408;
    constexpr int32_t NG = K / 32;          // 与 kernel 的 GS=32 保持一致
    constexpr int32_t SBP = (NG + 15) / 16 * 16;
    const size_t actBytes = (size_t)K * sizeof(uint16_t);
    const size_t wBytes = (size_t)N * (K / 4) * sizeof(uint16_t);
    const size_t sBytes = (size_t)N * SBP * sizeof(uint16_t);
    const size_t oBytes = (size_t)N * sizeof(uint16_t);

    uint8_t *act = alloc_dev(actBytes);
    uint8_t *wq = alloc_dev(wBytes);
    uint8_t *sc = alloc_dev(sBytes);
    uint8_t *sz = alloc_dev(sBytes);
    uint8_t *out = alloc_dev(oBytes);
    copy_h2d(act, "./input/gact.bin", actBytes);
    copy_h2d(wq, "./input/gwq.bin", wBytes);
    copy_h2d(sc, "./input/gsc.bin", sBytes);
    copy_h2d(sz, "./input/gsz.bin", sBytes);

    for (int32_t stage = 1; stage <= 5; ++stage) {
        w4a16_gemv_gen_do(1, g_stream, act, wq, sc, sz, out, K, N, stage);
        aclError r = aclrtSynchronizeStream(g_stream);
        if (r != ACL_ERROR_NONE) {
            printf("[w4a16_gemv_gen] STAGE %d FAILED acl=%d msg=%s\n", stage, (int)r,
                   aclGetRecentErrMsg() ? aclGetRecentErrMsg() : "(none)");
            return 1;
        }
        printf("[w4a16_gemv_gen] stage %d ok\n", stage);
    }
    double best = 1e9;
    for (int it = 0; it < 3; ++it) {
        auto t0 = std::chrono::high_resolution_clock::now();
        w4a16_gemv_gen_do(1, g_stream, act, wq, sc, sz, out, K, N, 99);
        CHECK_ACL_RET(aclrtSynchronizeStream(g_stream));
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        if (ms < best) best = ms;
    }
    double gb = (double)wBytes / 1e9;
    printf("[w4a16_gemv_gen] K=%d N=%d -> %.2f ms, %.2f GB/s\n", K, N, best, gb / (best / 1000.0));
    copy_d2h("./output/gout.bin", out, oBytes);
    return 0;
}

int test_probe_gather()
{
    constexpr int32_t N = 8192;
    const size_t nb = (size_t)N * sizeof(float);
    uint8_t *a = alloc_dev(nb);
    uint8_t *b = alloc_dev(nb);
    probe_gather_do(1, g_stream, a, b, 0);
    CHECK_ACL_RET(aclrtSynchronizeStream(g_stream));
    copy_d2h("./output/gather_a.bin", a, nb);
    copy_d2h("./output/gather_b.bin", b, nb);
    printf("[probe_gather] done (compare output/gather_{a,b}.bin)\n");
    return 0;
}

}  // namespace

void probe_v2_do(uint32_t b, void *st, uint8_t *gin, uint8_t *gout, int32_t mode, int32_t len,
                 int32_t iters);
void w4a16_gemv_batch_do(uint32_t b, void *st, uint8_t *act, uint8_t *wq, uint8_t *scales,
                         uint8_t *szs, uint8_t *out, int32_t K, int32_t N, int32_t stage);

// 行批处理版 GEMV：与旧版逐位对比 + 计时
int test_w4a16_gemv_batch()
{
    const int32_t K = getenv("GEMVB_K") ? atoi(getenv("GEMVB_K")) : 5120;
    const int32_t N = getenv("GEMVB_N") ? atoi(getenv("GEMVB_N")) : 17408;
    const int32_t NG = K / 32;
    const int32_t SBP = (NG + 15) / 16 * 16;
    const size_t actBytes = (size_t)K * sizeof(uint16_t);
    const size_t wBytes = (size_t)N * (K / 4) * sizeof(uint16_t);
    const size_t sBytes = (size_t)N * SBP * sizeof(uint16_t);
    const size_t oBytes = (size_t)N * sizeof(uint16_t);

    uint8_t *act = alloc_dev(actBytes), *wq = alloc_dev(wBytes), *sc = alloc_dev(sBytes);
    uint8_t *sz = alloc_dev(sBytes), *o1 = alloc_dev(oBytes), *o2 = alloc_dev(oBytes);
    copy_h2d(act, "./input/gact.bin", actBytes);
    copy_h2d(wq, "./input/gwq.bin", wBytes);
    copy_h2d(sc, "./input/gsc.bin", sBytes);
    copy_h2d(sz, "./input/gsz.bin", sBytes);

    // 先跑一遍预热（规避进程内首次调用异常）
    w4a16_gemv_gen_do(1, g_stream, act, wq, sc, sz, o1, K, N, 99);
    CHECK_ACL_RET(aclrtSynchronizeStream(g_stream));
    w4a16_gemv_batch_do(1, g_stream, act, wq, sc, sz, o2, K, N, 99);
    CHECK_ACL_RET(aclrtSynchronizeStream(g_stream));

    auto bench = [&](bool v2) {
        double best = 1e9;
        for (int it = 0; it < 3; ++it) {
            auto t0 = std::chrono::high_resolution_clock::now();
            if (v2) w4a16_gemv_batch_do(1, g_stream, act, wq, sc, sz, o2, K, N, 99);
            else w4a16_gemv_gen_do(1, g_stream, act, wq, sc, sz, o1, K, N, 99);
            CHECK_ACL_RET(aclrtSynchronizeStream(g_stream));
            auto t1 = std::chrono::high_resolution_clock::now();
            double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            if (ms < best) best = ms;
        }
        return best;
    };
    // 两个版本各算一次并保留输出，再对比；stage=99 用字节偏移假设，98 用元素偏移假设
    w4a16_gemv_gen_do(1, g_stream, act, wq, sc, sz, o1, K, N, 99);
    CHECK_ACL_RET(aclrtSynchronizeStream(g_stream));
    std::vector<uint16_t> h1(N), h2(N);
    CHECK_ACL_RET(aclrtMemcpy(h1.data(), oBytes, o1, oBytes, ACL_MEMCPY_DEVICE_TO_HOST));
    for (int stg = 99; stg >= 97; --stg) {
        w4a16_gemv_batch_do(1, g_stream, act, wq, sc, sz, o2, K, N, stg);
        CHECK_ACL_RET(aclrtSynchronizeStream(g_stream));
        CHECK_ACL_RET(aclrtMemcpy(h2.data(), oBytes, o2, oBytes, ACL_MEMCPY_DEVICE_TO_HOST));
        double maxAbs = 0.0, sumAbs = 0.0;
        int nBad = 0;
        for (int i = 0; i < N; ++i) {
            float a = aclFloat16ToFloat(h1[i]), b = aclFloat16ToFloat(h2[i]);
            double d = std::fabs((double)a - (double)b);
            if (d > maxAbs) maxAbs = d;
            sumAbs += d;
            if (d > 2e-2) ++nBad;
        }
        printf("[gemv_batch] stage=%d offsetUnit=%s  max_abs=%.5f mean_abs=%.6f  n(|d|>2e-2)=%d/%d\n",
               stg, stg == 98 ? "elem" : (stg == 97 ? "perPhase" : "byte"), maxAbs, sumAbs / N, nBad, N);
    }
    double ms1 = bench(false), ms2 = bench(true);
    double gb = (double)wBytes / 1e9;
    printf("[gemv_batch] K=%d N=%d  v1 %.2f ms (%.2f GB/s)  v2 %.2f ms (%.2f GB/s)  speedup %.2fx\n",
           K, N, ms1, gb / (ms1 / 1000.0), ms2, gb / (ms2 / 1000.0), ms1 / ms2);
    copy_d2h("./output/gout_batch.bin", o2, oBytes);
    return 0;
}

void w4a16_gemv_gen2_do(uint32_t b, void *st, uint8_t *act, uint8_t *wq, uint8_t *scales,
                        uint8_t *szs, uint8_t *out, int32_t K, int32_t N, int32_t stage);

// gen vs gen2：逐位对拍 + 计时（gen2 只是合并指令，期望 max_abs == 0）
int test_w4a16_gemv_ab()
{
    const int32_t K = getenv("GEMVB_K") ? atoi(getenv("GEMVB_K")) : 5120;
    const int32_t N = getenv("GEMVB_N") ? atoi(getenv("GEMVB_N")) : 17408;
    const int32_t NG = K / 32;
    const int32_t SBP = (NG + 15) / 16 * 16;
    const size_t actBytes = (size_t)K * sizeof(uint16_t);
    const size_t wBytes = (size_t)N * (K / 4) * sizeof(uint16_t);
    const size_t sBytes = (size_t)N * SBP * sizeof(uint16_t);
    const size_t oBytes = (size_t)N * sizeof(uint16_t);

    uint8_t *act = alloc_dev(actBytes), *wq = alloc_dev(wBytes), *sc = alloc_dev(sBytes);
    uint8_t *sz = alloc_dev(sBytes), *o1 = alloc_dev(oBytes), *o2 = alloc_dev(oBytes);
    copy_h2d(act, "./input/gact.bin", actBytes);
    copy_h2d(wq, "./input/gwq.bin", wBytes);
    copy_h2d(sc, "./input/gsc.bin", sBytes);
    copy_h2d(sz, "./input/gsz.bin", sBytes);

    w4a16_gemv_gen_do(1, g_stream, act, wq, sc, sz, o1, K, N, 99);
    CHECK_ACL_RET(aclrtSynchronizeStream(g_stream));
    w4a16_gemv_gen2_do(1, g_stream, act, wq, sc, sz, o2, K, N, 99);
    CHECK_ACL_RET(aclrtSynchronizeStream(g_stream));

    std::vector<uint16_t> h1(N), h2(N);
    CHECK_ACL_RET(aclrtMemcpy(h1.data(), oBytes, o1, oBytes, ACL_MEMCPY_DEVICE_TO_HOST));
    CHECK_ACL_RET(aclrtMemcpy(h2.data(), oBytes, o2, oBytes, ACL_MEMCPY_DEVICE_TO_HOST));
    int nDiff = 0;
    double maxAbs = 0.0;
    for (int i = 0; i < N; ++i) {
        if (h1[i] != h2[i]) {
            ++nDiff;
            double d = std::fabs((double)aclFloat16ToFloat(h1[i]) - (double)aclFloat16ToFloat(h2[i]));
            if (d > maxAbs) maxAbs = d;
        }
    }
    auto bench = [&](bool v2) {
        double best = 1e9;
        for (int it = 0; it < 3; ++it) {
            auto t0 = std::chrono::high_resolution_clock::now();
            if (v2) w4a16_gemv_gen2_do(1, g_stream, act, wq, sc, sz, o2, K, N, 99);
            else w4a16_gemv_gen_do(1, g_stream, act, wq, sc, sz, o1, K, N, 99);
            CHECK_ACL_RET(aclrtSynchronizeStream(g_stream));
            auto t1 = std::chrono::high_resolution_clock::now();
            double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            if (ms < best) best = ms;
        }
        return best;
    };
    double ms1 = bench(false), ms2 = bench(true);
    double gb = (double)wBytes / 1e9;
    printf("[gemv_ab] K=%d N=%d  gen %.2f ms (%.2f GB/s)  gen2 %.2f ms (%.2f GB/s)  speedup %.2fx\n",
           K, N, ms1, gb / (ms1 / 1000.0), ms2, gb / (ms2 / 1000.0), ms1 / ms2);
    printf("[gemv_ab] 逐位对拍: nDiff=%d / %d , max_abs=%.6f  %s\n", nDiff, N, maxAbs,
           nDiff == 0 ? "BIT-EXACT" : "DIFF!");
    return 0;
}

int test_probe_v2()
{
    const char *names[] = {"f32Add+bar", "f32Add/8bar", "f16Add+bar", "castI16toH", "f16Mul+bar",
                           "tree32_f32", "tree32_f16", "gather", "wholeRed", "dmaGM",
                           "noop", "f16MulAdd4"};
    const int32_t lens[] = {1280, 2560, 5120, 10240};
    const int32_t it0 = 20, it1 = 220;
    for (int32_t li = 0; li < 4; ++li) {
        const int32_t len = lens[li];
        uint8_t *in = alloc_dev((size_t)len * 4 + 64);
        uint8_t *out = alloc_dev(64);
        printf("---- len=%d ----\n", len);
        for (int32_t mode = 0; mode <= 11; ++mode) {
            volatile double t0ms = 0, t1ms = 0;
            for (int32_t t = 0; t < 2; ++t) {
                auto a = std::chrono::high_resolution_clock::now();
                probe_v2_do(1, g_stream, in, out, mode, len, it0);
                CHECK_ACL_RET(aclrtSynchronizeStream(g_stream));
                auto b = std::chrono::high_resolution_clock::now();
                t0ms = std::chrono::duration<double, std::milli>(b - a).count();
                auto c = std::chrono::high_resolution_clock::now();
                probe_v2_do(1, g_stream, in, out, mode, len, it1);
                CHECK_ACL_RET(aclrtSynchronizeStream(g_stream));
                auto d = std::chrono::high_resolution_clock::now();
                t1ms = std::chrono::duration<double, std::milli>(d - c).count();
                if (t1ms > t0ms) break;   // 取有效的一次
            }
            double ns = (t1ms - t0ms) * 1e6 / (it1 - it0);
            printf("  %-12s %8.1f ns  %7.1f cyc  %6.2f elem/cyc\n", names[mode], ns, ns * 1.224,
                   len / (ns * 1.224));
        }
        CHECK_ACL_RET(aclrtFree(in));
        CHECK_ACL_RET(aclrtFree(out));
    }
    return 0;
}

// ---- 两段式 Cube 路径原型：dequant_w4（W4→fp16 写 GM）+ aclnnMatmul ----
void dequant_w4_do(uint32_t b, void *st, uint8_t *wq, uint8_t *scales, uint8_t *szs, uint8_t *out,
                   int32_t K, int32_t N);

static aclTensor *mkAclTensor2(int64_t r, int64_t c, aclDataType dt, void *p)
{
    int64_t d[2] = {r, c};
    return aclCreateTensor(d, 2, dt, nullptr, 0, ACL_FORMAT_ND, d, 2, p);
}

// 对比：M 个 token 的向量 GEMV（M 次 gen） vs 一次两段式（dequant 一次 + Cube matmul 一次）
int test_cube_prefill()
{
    const int32_t K = getenv("CUBE_K") ? atoi(getenv("CUBE_K")) : 5120;
    const int32_t N = getenv("CUBE_N") ? atoi(getenv("CUBE_N")) : 17408;
    const int32_t M = getenv("CUBE_M") ? atoi(getenv("CUBE_M")) : 32;
    const int32_t NG = K / 32;
    const int32_t SBP = (NG + 15) / 16 * 16;
    const size_t actBytes = (size_t)K * sizeof(uint16_t);
    const size_t wBytes = (size_t)N * (K / 4) * sizeof(uint16_t);
    const size_t sBytes = (size_t)N * SBP * sizeof(uint16_t);
    const size_t wfBytes = (size_t)N * K * sizeof(uint16_t);

    uint8_t *wq = alloc_dev(wBytes), *sc = alloc_dev(sBytes), *sz = alloc_dev(sBytes);
    uint8_t *wf = alloc_dev(wfBytes);
    uint8_t *actM = alloc_dev((size_t)M * K * 2);
    uint8_t *refM = alloc_dev((size_t)M * N * 2);
    uint8_t *outM = alloc_dev((size_t)M * N * 2);
    copy_h2d(wq, "./input/gwq.bin", wBytes);
    copy_h2d(sc, "./input/gsc.bin", sBytes);
    copy_h2d(sz, "./input/gsz.bin", sBytes);

    {   // M 个 token 的激活：第 m 行 = golden act × (1+0.02m)
        std::vector<uint16_t> base(K);
        uint8_t *hb = alloc_host(actBytes);
        size_t rfSz = actBytes;
        ReadFile("./input/gact.bin", rfSz, hb, actBytes);
        memcpy(base.data(), hb, actBytes);
        CHECK_ACL_RET(aclrtFreeHost(hb));
        std::vector<uint16_t> rows((size_t)M * K);
        for (int32_t m = 0; m < M; ++m) {
            float f = 1.0f + 0.02f * m;
            for (int32_t k = 0; k < K; ++k) {
                rows[(size_t)m * K + k] = aclFloatToFloat16(aclFloat16ToFloat(base[k]) * f);
            }
        }
        CHECK_ACL_RET(aclrtMemcpy(actM, (size_t)M * K * 2, rows.data(), (size_t)M * K * 2,
                                  ACL_MEMCPY_HOST_TO_DEVICE));
    }

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int32_t m = 0; m < M; ++m) {
        w4a16_gemv_gen_do(1, g_stream, actM + (size_t)m * K * 2, wq, sc, sz,
                          refM + (size_t)m * N * 2, K, N, 99);
    }
    CHECK_ACL_RET(aclrtSynchronizeStream(g_stream));
    auto t1 = std::chrono::high_resolution_clock::now();
    double msGemv = std::chrono::duration<double, std::milli>(t1 - t0).count();

    auto t2 = std::chrono::high_resolution_clock::now();
    dequant_w4_do(1, g_stream, wq, sc, sz, wf, K, N);
    CHECK_ACL_RET(aclrtSynchronizeStream(g_stream));
    auto t3 = std::chrono::high_resolution_clock::now();
    double msDeq = std::chrono::duration<double, std::milli>(t3 - t2).count();

    int64_t bd[2] = {K, N}, bs[2] = {1, K};
    aclTensor *ta = mkAclTensor2(M, K, ACL_FLOAT16, actM);
    aclTensor *tb = aclCreateTensor(bd, 2, ACL_FLOAT16, bs, 0, ACL_FORMAT_ND, bd, 2, wf);
    aclTensor *tc = mkAclTensor2(M, N, ACL_FLOAT16, outM);
    uint64_t ws = 0;
    aclOpExecutor *ex = nullptr;
    auto st = aclnnMatmulGetWorkspaceSize(ta, tb, tc, 1, &ws, &ex);
    if (st != 0) {
        printf("[cube_prefill] aclnnMatmulGetWorkspaceSize fail %d\n", (int)st);
        return 1;
    }
    uint8_t *wsp = alloc_dev(ws ? ws : 1);
    auto t4 = std::chrono::high_resolution_clock::now();
    st = aclnnMatmul(wsp, ws, ex, g_stream);
    CHECK_ACL_RET(aclrtSynchronizeStream(g_stream));
    auto t5 = std::chrono::high_resolution_clock::now();
    double msMma = std::chrono::duration<double, std::milli>(t5 - t4).count();
    if (st != 0) {
        printf("[cube_prefill] aclnnMatmul exec fail %d\n", (int)st);
        return 1;
    }
    aclDestroyTensor(ta);
    aclDestroyTensor(tb);
    aclDestroyTensor(tc);

    std::vector<uint16_t> hRef((size_t)M * N), hOut((size_t)M * N);
    CHECK_ACL_RET(aclrtMemcpy(hRef.data(), (size_t)M * N * 2, refM, (size_t)M * N * 2,
                              ACL_MEMCPY_DEVICE_TO_HOST));
    CHECK_ACL_RET(aclrtMemcpy(hOut.data(), (size_t)M * N * 2, outM, (size_t)M * N * 2,
                              ACL_MEMCPY_DEVICE_TO_HOST));
    double maxAbs = 0.0, sumAbs = 0.0, sumRef = 0.0;
    int nBad = 0;
    for (size_t i = 0; i < (size_t)M * N; ++i) {
        double a = aclFloat16ToFloat(hRef[i]), b = aclFloat16ToFloat(hOut[i]);
        double d = std::fabs(a - b);
        if (d > maxAbs) maxAbs = d;
        sumAbs += d;
        sumRef += std::fabs(a);
        if (d > 0.05) ++nBad;
    }
    double msTwoStage = msDeq + msMma;
    printf("[cube_prefill] K=%d N=%d M=%d\n", K, N, M);
    printf("[cube_prefill] 向量GEMV x%d: %.1f ms (%.2f ms/token)\n", M, msGemv, msGemv / M);
    printf("[cube_prefill] 两段式: dequant %.1f + CubeMatmul %.1f = %.1f ms (%.2f ms/token)\n",
           msDeq, msMma, msTwoStage, msTwoStage / M);
    printf("[cube_prefill] 加速 %.2fx | max_abs=%.4f mean_abs=%.5f mean|ref|=%.3f n_bad=%d/%zu\n",
           msGemv / msTwoStage, maxAbs, sumAbs / ((double)M * N), sumRef / ((double)M * N), nBad,
           (size_t)M * N);
    return 0;
}


// ---- Axpy-N 结构带宽实验（决定"重排权重"这条路值不值得）----
void bench_axpy_do(uint32_t b, void *st, uint8_t *act, uint8_t *wq, uint8_t *scales, uint8_t *out,
                   int32_t K, int32_t N, int32_t NT);
int test_bench_axpy()
{
    const int32_t K = getenv("AXPY_K") ? atoi(getenv("AXPY_K")) : 5120;
    const int32_t N = getenv("AXPY_N") ? atoi(getenv("AXPY_N")) : 17408;
    const int32_t NG = K / 32;
    const int32_t SBP = (NG + 15) / 16 * 16;
    const size_t wBytes = (size_t)N * (K / 4) * sizeof(uint16_t);
    uint8_t *act = alloc_dev((size_t)K * 2), *wq = alloc_dev(wBytes);
    uint8_t *sc = alloc_dev((size_t)N * SBP * 2), *out = alloc_dev((size_t)N * 2);
    copy_h2d(act, "./input/gact.bin", (size_t)K * 2);
    copy_h2d(wq, "./input/gwq.bin", wBytes);
    copy_h2d(sc, "./input/gsc.bin", (size_t)N * SBP * 2);
    double gb = (double)wBytes / 1e9;
    for (int NT : {256, 512, 1024}) {
        double best = 1e9;
        for (int it = 0; it < 3; ++it) {
            auto t0 = std::chrono::high_resolution_clock::now();
            bench_axpy_do(1, g_stream, act, wq, sc, out, K, N, NT);
            CHECK_ACL_RET(aclrtSynchronizeStream(g_stream));
            auto t1 = std::chrono::high_resolution_clock::now();
            double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            if (ms < best) best = ms;
        }
        printf("[bench_axpy] K=%d N=%d NT=%4d -> %7.2f ms  %6.2f GB/s  (27B 折算 %5.2f s/token)\n", K, N,
               NT, best, gb / (best / 1000.0), best / 1000.0 * 13.53 / gb);
    }
    return 0;
}

// ---- W4 -> AX 布局（[g][kp][n]，k-pair 打包）；scale/sz -> [NG][N] ----
// 用法: ./qwen38_ops_test relayout <srcDir> <dstDir>
#include <sys/stat.h>
#include <sys/types.h>
static void rl_mkdirs(const std::string &path)
{
    std::string cur;
    for (size_t i = 0; i < path.size(); ++i) {
        cur += path[i];
        if (path[i] == '/' || i + 1 == path.size()) {
            if (cur != "/" && !cur.empty()) mkdir(cur.c_str(), 0755);
        }
    }
}
static int rl_copy(const std::string &src, const std::string &dst)
{
    FILE *fi = fopen(src.c_str(), "rb");
    if (!fi) return 1;
    size_t sl = dst.find_last_of('/');
    if (sl != std::string::npos) rl_mkdirs(dst.substr(0, sl));
    FILE *fo = fopen(dst.c_str(), "wb");
    if (!fo) { fclose(fi); return 1; }
    char buf[1 << 16];
    size_t n;
    while ((n = fread(buf, 1, sizeof(buf), fi)) > 0) fwrite(buf, 1, n, fo);
    fclose(fi);
    fclose(fo);
    return 0;
}

int run_relayout(const char *srcDir, const char *dstDir)
{
    std::string sd(srcDir), dd(dstDir);
    FILE *mf = fopen((sd + "/meta.txt").c_str(), "r");
    if (!mf) { printf("[relayout] no meta.txt in %s\n", srcDir); return 1; }
    std::vector<std::string> lines;
    {
        char buf[4096];
        while (fgets(buf, sizeof(buf), mf)) {
            std::string s(buf);
            while (!s.empty() && (s.back() == '\n' || s.back() == '\r')) s.pop_back();
            if (!s.empty()) lines.push_back(s);
        }
        fclose(mf);
    }
    rl_mkdirs(dd);
    auto readAll = [](const std::string &path, size_t expect) -> std::vector<uint8_t> {
        std::vector<uint8_t> v;
        FILE *f = fopen(path.c_str(), "rb");
        if (!f) { printf("[relayout] open fail %s\n", path.c_str()); return v; }
        v.resize(expect);
        size_t got = fread(v.data(), 1, expect, f);
        fclose(f);
        if (got != expect) { printf("[relayout] short read %s\n", path.c_str()); v.clear(); }
        return v;
    };
    auto writeAll = [](const std::string &path, const void *data, size_t bytes) -> int {
        size_t sl = path.find_last_of('/');
        if (sl != std::string::npos) rl_mkdirs(path.substr(0, sl));
        FILE *f = fopen(path.c_str(), "wb");
        if (!f) { printf("[relayout] write fail %s\n", path.c_str()); return 1; }
        fwrite(data, 1, bytes, f);
        fclose(f);
        return 0;
    };
    int nW4 = 0, nOther = 0;
    std::vector<std::string> outLines;
    for (auto &s : lines) {
        std::vector<std::string> t;
        size_t i = 0;
        while (i < s.size()) {
            size_t j = s.find(' ', i);
            if (j == std::string::npos) j = s.size();
            t.push_back(s.substr(i, j - i));
            i = j + 1;
        }
        if (t.empty() || t[0] == "CFG") { outLines.push_back(s); continue; }
        // 统一两种 meta 格式：
        //   qwen: <ROLE> <wq> <s> <sz> <N> <K> w4   /  <ROLE> <path> <N> f16
        //   md3 : W4 <ROLE> <prefix> <N> <K>        /  F16 <ROLE> <path> <N>
        const bool md3meta = (t[0] == "W4" || t[0] == "F16");
        std::vector<std::string> f = t;
        if (md3meta) {
            if (t[0] == "W4" && t.size() >= 5) {
                f.clear();
                f.push_back("W4");
                f.push_back(t[2] + ".w4.bin");
                f.push_back(t[2] + ".s.bin");
                f.push_back(t[2] + ".sz.bin");
                f.push_back(t[3]);
                f.push_back(t[4]);
                f.push_back("w4");
            } else if (t[0] == "F16" && t.size() >= 4) {
                f.clear();
                f.push_back("F16");
                f.push_back(t[2]);
                f.push_back(t[3]);
            } else {
                outLines.push_back(s);
                continue;
            }
        }
        const bool isEmbed = (f[0] == "EMBED") || (md3meta && t.size() >= 2 && t[1] == "T.WTE");
        if (isEmbed && f.back() == "w4" && f.size() >= 7) {
            // EMBED / T.WTE 走逐行 lookup（按原始 [N][KQ] 布局）→ 不转换，原样拷贝
            for (int q = 1; q <= 3; ++q) {
                if (rl_copy(sd + "/" + f[q], dd + "/" + f[q]) != 0) return 1;
            }
            outLines.push_back(s);
            ++nOther;
            continue;
        }
        if (f.back() == "w4" && f.size() >= 7) {
            const int N = atoi(f[4].c_str()), K = atoi(f[5].c_str());
            const int KQ = K / 4, NG = K / 32;
            const int SBP = (NG + 15) / 16 * 16;
            auto wsrc = readAll(sd + "/" + f[1], (size_t)N * KQ * 2);
            auto ssrc = readAll(sd + "/" + f[2], (size_t)N * SBP * 2);
            auto szsrc = readAll(sd + "/" + f[3], (size_t)N * SBP * 2);
            if (wsrc.empty() || ssrc.empty() || szsrc.empty()) return 1;
            const uint16_t *w = reinterpret_cast<const uint16_t *>(wsrc.data());
            const uint16_t *sc = reinterpret_cast<const uint16_t *>(ssrc.data());
            const uint16_t *sz = reinterpret_cast<const uint16_t *>(szsrc.data());
            // 4 半字节/word：word = q[k0..k0+3]，k0 = 32g + 4*w8（存储量与原始格式相同）
            std::vector<uint16_t> ax((size_t)N * KQ);
            const int NB = 256;
            for (int nb = 0; nb < N; nb += NB) {
                const int ne = (N - nb < NB) ? N : nb + NB;
                for (int g = 0; g < NG; ++g) {
                    for (int w8 = 0; w8 < 8; ++w8) {
                        const int k0 = 32 * g + 4 * w8;
                        const int p0 = k0 / KQ, j0 = k0 % KQ;
                        const int sh = 4 * p0;
                        uint16_t *dstRow = ax.data() + ((size_t)g * 8 + w8) * N + nb;
                        for (int n = nb; n < ne; ++n) {
                            const uint16_t *row = w + (size_t)n * KQ;
                            uint16_t v = (row[j0] >> sh) & 0xF;
                            v |= (uint16_t)(((row[j0 + 1] >> sh) & 0xF) << 4);
                            v |= (uint16_t)(((row[j0 + 2] >> sh) & 0xF) << 8);
                            v |= (uint16_t)(((row[j0 + 3] >> sh) & 0xF) << 12);
                            dstRow[n - nb] = v;
                        }
                    }
                }
            }
            if (writeAll(dd + "/" + f[1], ax.data(), ax.size() * 2)) return 1;
            for (int which = 0; which < 2; ++which) {
                const uint16_t *src = (which == 0) ? sc : sz;
                std::vector<uint16_t> tr((size_t)NG * N);
                for (int g = 0; g < NG; ++g) {
                    uint16_t *row = tr.data() + (size_t)g * N;
                    for (int n = 0; n < N; ++n) row[n] = src[(size_t)n * SBP + g];
                }
                const std::string &rel = (which == 0) ? f[2] : f[3];
                if (writeAll(dd + "/" + rel, tr.data(), tr.size() * 2)) return 1;
            }
            ++nW4;
            outLines.push_back(s);
        } else {
            std::string rel = f[1];
            if (rl_copy(sd + "/" + rel, dd + "/" + rel) != 0) {
                printf("[relayout] copy fail %s\n", rel.c_str());
                return 1;
            }
            ++nOther;
            outLines.push_back(s);
        }
    }
    {
        std::string joined;
        for (auto &l : outLines) { joined += l; joined += "\n"; }
        if (writeAll(dd + "/meta.txt", joined.data(), joined.size())) return 1;
    }
    printf("[relayout] done: w4=%d other=%d -> %s\n", nW4, nOther, dstDir);
    return 0;
}

// ---- AX 布局 Axpy-N kernel 单矩阵对拍 + 计时 ----
void w4a16_gemv_axpy_do(uint32_t b, void *st, uint8_t *act, uint8_t *wq, uint8_t *scales,
                        uint8_t *szs, uint8_t *out, int32_t K, int32_t N, int32_t NT);
int test_gemv_axpy()
{
    const int32_t K = getenv("AXK") ? atoi(getenv("AXK")) : 5120;
    const int32_t N = getenv("AXN") ? atoi(getenv("AXN")) : 17408;
    const int32_t NT = getenv("AXNT") ? atoi(getenv("AXNT")) : 512;
    const int32_t KQ = K / 4, NG = K / 32, SBP = (NG + 15) / 16 * 16;
    const size_t wBytes = (size_t)N * KQ * 2, sBytes = (size_t)N * SBP * 2;
    uint8_t *act = alloc_dev((size_t)K * 2), *wq = alloc_dev(wBytes), *sc = alloc_dev(sBytes),
            *sz = alloc_dev(sBytes), *o1 = alloc_dev((size_t)N * 2), *o2 = alloc_dev((size_t)N * 2);
    copy_h2d(act, "./input/gact.bin", (size_t)K * 2);
    copy_h2d(wq, "./input/gwq.bin", wBytes);
    copy_h2d(sc, "./input/gsc.bin", sBytes);
    copy_h2d(sz, "./input/gsz.bin", sBytes);

    // host 侧转 AX 布局
    std::vector<uint16_t> hw(wBytes / 2), hs(sBytes / 2), hsz(sBytes / 2);
    {
        uint8_t *buf = alloc_host(wBytes);
        size_t rsz = wBytes;
        ReadFile("./input/gwq.bin", rsz, buf, wBytes);
        memcpy(hw.data(), buf, wBytes);
        CHECK_ACL_RET(aclrtFreeHost(buf));
        uint8_t *b2 = alloc_host(sBytes);
        size_t rs2 = sBytes;
        ReadFile("./input/gsc.bin", rs2, b2, sBytes);
        memcpy(hs.data(), b2, sBytes);
        CHECK_ACL_RET(aclrtFreeHost(b2));
        uint8_t *b3 = alloc_host(sBytes);
        size_t rs3 = sBytes;
        ReadFile("./input/gsz.bin", rs3, b3, sBytes);
        memcpy(hsz.data(), b3, sBytes);
        CHECK_ACL_RET(aclrtFreeHost(b3));
    }
    std::vector<uint16_t> axw((size_t)N * KQ), axs((size_t)NG * N), axsz((size_t)NG * N);
    for (int g = 0; g < NG; ++g) {
        for (int w8 = 0; w8 < 8; ++w8) {
            const int k0 = 32 * g + 4 * w8;
            const int p0 = k0 / KQ, j0 = k0 % KQ, sh = 4 * p0;
            uint16_t *row = axw.data() + ((size_t)g * 8 + w8) * N;
            for (int n = 0; n < N; ++n) {
                const uint16_t *src = hw.data() + (size_t)n * KQ;
                uint16_t v = (src[j0] >> sh) & 0xF;
                v |= (uint16_t)(((src[j0 + 1] >> sh) & 0xF) << 4);
                v |= (uint16_t)(((src[j0 + 2] >> sh) & 0xF) << 8);
                v |= (uint16_t)(((src[j0 + 3] >> sh) & 0xF) << 12);
                row[n] = v;
            }
        }
        for (int n = 0; n < N; ++n) {
            axs[(size_t)g * N + n] = hs[(size_t)n * SBP + g];
            axsz[(size_t)g * N + n] = hsz[(size_t)n * SBP + g];
        }
    }
    uint8_t *dAxw = alloc_dev(axw.size() * 2), *dAxs = alloc_dev(axs.size() * 2),
            *dAxsz = alloc_dev(axsz.size() * 2);
    CHECK_ACL_RET(aclrtMemcpy(dAxw, axw.size() * 2, axw.data(), axw.size() * 2, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL_RET(aclrtMemcpy(dAxs, axs.size() * 2, axs.data(), axs.size() * 2, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL_RET(aclrtMemcpy(dAxsz, axsz.size() * 2, axsz.data(), axsz.size() * 2, ACL_MEMCPY_HOST_TO_DEVICE));

    w4a16_gemv_gen2_do(1, g_stream, act, wq, sc, sz, o1, K, N, 99);
    CHECK_ACL_RET(aclrtSynchronizeStream(g_stream));
    w4a16_gemv_axpy_do(1, g_stream, act, dAxw, dAxs, dAxsz, o2, K, N, NT);
    CHECK_ACL_RET(aclrtSynchronizeStream(g_stream));

    std::vector<uint16_t> h1(N), h2(N);
    CHECK_ACL_RET(aclrtMemcpy(h1.data(), (size_t)N * 2, o1, (size_t)N * 2, ACL_MEMCPY_DEVICE_TO_HOST));
    CHECK_ACL_RET(aclrtMemcpy(h2.data(), (size_t)N * 2, o2, (size_t)N * 2, ACL_MEMCPY_DEVICE_TO_HOST));
    double maxAbs = 0, sumAbs = 0, sumRef = 0;
    int nBad = 0;
    for (int i = 0; i < N; ++i) {
        double a = aclFloat16ToFloat(h1[i]), b = aclFloat16ToFloat(h2[i]);
        double d = std::fabs(a - b);
        if (d > maxAbs) maxAbs = d;
        sumAbs += d;
        sumRef += std::fabs(a);
        if (d > 0.02) ++nBad;
    }
    auto bench = [&](bool ax) {
        double best = 1e9;
        for (int it = 0; it < 3; ++it) {
            auto t0 = std::chrono::high_resolution_clock::now();
            if (ax) w4a16_gemv_axpy_do(1, g_stream, act, dAxw, dAxs, dAxsz, o2, K, N, NT);
            else w4a16_gemv_gen2_do(1, g_stream, act, wq, sc, sz, o1, K, N, 99);
            CHECK_ACL_RET(aclrtSynchronizeStream(g_stream));
            auto t1 = std::chrono::high_resolution_clock::now();
            double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            if (ms < best) best = ms;
        }
        return best;
    };
    double ms2 = bench(false), msA = bench(true);
    double gb = (double)wBytes / 1e9;
    printf("[axpy] K=%d N=%d NT=%d  gen2 %.2f ms (%.2f GB/s)  axpy %.2f ms (%.2f GB/s)  speedup %.2fx\n",
           K, N, NT, ms2, gb / (ms2 / 1000.0), msA, gb / (msA / 1000.0), ms2 / msA);
    printf("[axpy] 前8个: ref=");
    for (int i = 0; i < 8; ++i) printf(" %.4f", aclFloat16ToFloat(h1[i]));
    printf(" | axpy=");
    for (int i = 0; i < 8; ++i) printf(" %.4f", aclFloat16ToFloat(h2[i]));
    printf("\n");
    printf("[axpy] 数值: max_abs=%.4f mean_abs=%.5f mean|ref|=%.3f n(|d|>0.02)=%d/%d\n", maxAbs,
           sumAbs / N, sumRef / N, nBad, N);
    return 0;
}
int main(int argc, char *argv[])
{
    std::string test = (argc > 1) ? argv[1] : "add";
    if (test == "relayout") {
        if (argc < 4) { printf("usage: relayout <srcDir> <dstDir>\n"); return 2; }
        return run_relayout(argv[2], argv[3]);
    }
    CHECK_ACL_RET(aclInit(nullptr));
    CHECK_ACL_RET(aclrtSetDevice(0));
    CHECK_ACL_RET(aclrtCreateStream(&g_stream));

    int rc = 0;
    if (test == "add") {
        rc = test_add();
    } else if (test == "gdn_gating") {
        rc = test_gdn_gating();
    } else if (test == "gdn_conv1d") {
        rc = test_gdn_conv1d();
    } else if (test == "gdn_l2norm") {
        rc = test_gdn_l2norm();
    } else if (test == "gdn_cumsum") {
        rc = test_gdn_cumsum();
    } else if (test == "gdn_gated_rmsnorm") {
        rc = test_gdn_gated_rmsnorm();
    } else if (test == "bench_stream") {
        rc = test_bench_stream();
    } else if (test == "bench_compute") {
        rc = test_bench_compute();
    } else if (test == "w4a16_gemv") {
        rc = test_w4a16_gemv();
    } else if (test == "w4a16_probe") {
        rc = test_w4a16_probe();
    } else if (test == "cube_matmul") {
        rc = test_cube_matmul();
    } else if (test == "probe_l1") {
        rc = test_probe_l1();
    } else if (test == "gdn_kkt") {
        rc = test_gdn_kkt();
    } else if (test == "gdn_tril") {
        rc = test_gdn_tril();
    } else if (test == "gdn_wy") {
        rc = test_gdn_wy();
    } else if (test == "gdn_chunk_h") {
        rc = test_gdn_chunk_h();
    } else if (test == "gdn_chunk_o") {
        rc = test_gdn_chunk_o();
    } else if (test == "gdn_rgdr") {
        rc = test_gdn_rgdr();
    } else if (test == "gdn_rgdr_dbg") {
        rc = test_gdn_rgdr_dbg();
    } else if (test == "full_attn_decode") {
        rc = test_full_attn_decode();
    } else if (test == "dense_ops") {
        rc = test_dense_ops();
    } else if (test == "memtest") {
        rc = test_memtest();
    } else if (test == "engine_mini") {
        rc = run_engine_mini(g_stream);
    } else if (test == "engine_27b") {
        { const char *d = getenv("ENG27_DIR");
          rc = run_engine_27b(g_stream, d ? d : "/home/HwHiAiUser/code/qwen38_310B/models/qwen38_310b"); }
    } else if (test == "w4a16_gemv_gen") {
        rc = test_w4a16_gemv_gen();
    } else if (test == "probe_gather") {
        rc = test_probe_gather();
    } else if (test == "platform") {
        rc = test_platform();
    } else if (test == "bench_mc") {
        rc = test_bench_mc();
    } else if (test == "bench_scale") {
        rc = test_bench_scale();
    } else if (test == "bench_stream2") {
        rc = test_bench_stream2();
    } else if (test == "bench_l2") {
        rc = test_bench_l2();
    } else if (test == "cube_m") {
        rc = test_cube_m();
    } else if (test == "bench_dequant") {
        rc = test_bench_dequant();
    } else if (test == "probe_v2") {
        rc = test_probe_v2();
    } else if (test == "bench_axpy") {
        rc = test_bench_axpy();
    } else if (test == "cube_prefill") {
        rc = test_cube_prefill();
    } else if (test == "bench_axpy") {
        rc = test_bench_axpy();
    } else if (test == "cube_prefill") {
        rc = test_cube_prefill();
    } else if (test == "gemv_axpy") {
        rc = test_gemv_axpy();
    } else if (test == "gemv_ab") {
        rc = test_w4a16_gemv_ab();
    } else if (test == "gemv_batch") {
        rc = test_w4a16_gemv_batch();
    } else {
        printf("unknown test: %s\n", test.c_str());
        rc = 2;
    }

    CHECK_ACL_RET(aclrtDestroyStream(g_stream));
    CHECK_ACL_RET(aclrtResetDevice(0));
    CHECK_ACL_RET(aclFinalize());
    printf("[%s] %s\n", rc == 0 ? "OK" : "FAIL", test.c_str());
    return rc;
}
