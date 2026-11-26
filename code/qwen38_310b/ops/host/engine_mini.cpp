/**
 * engine_mini.cpp — 缩小版 GDN 混合模型 decode 引擎（host 调度 + 中间结果 dump）
 *
 * 目的：把已有 AscendC kernel 串成「一层 GDN 线性注意力 + MLP + lm_head」的完整前向，
 * 输出中间结果供 numpy 参考对拍，并测量单 token 端到端耗时。
 *
 * 全部权重随机生成并 dump 到 ./output/eng_*.bin，python 参考从同样的权重计算。
 */
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <chrono>
#include <random>
#include <vector>

#include "acl/acl.h"

// 复用 main.cpp 里注册的 kernel 启动函数
void fp16_gemv_do(uint32_t b, void *st, uint8_t *act, uint8_t *w, uint8_t *out, int32_t K, int32_t N);
void rmsnorm_op_do(uint32_t b, void *st, uint8_t *x, uint8_t *w, uint8_t *y, int32_t H, float eps);
void vec_axpy_do(uint32_t b, void *st, uint8_t *a, uint8_t *b_, uint8_t *out, float s, int32_t N);
void mul_silu_do(uint32_t b, void *st, uint8_t *gate, uint8_t *up, uint8_t *out, int32_t N);
void conv1d_decode_do(uint32_t b, void *st, uint8_t *x, uint8_t *state, uint8_t *w, uint8_t *out, int32_t C);
void l2norm_op_do(uint32_t b, void *st, uint8_t *x, uint8_t *y, int32_t D, float eps, float mul);
void gdn_rgdr_do(uint32_t b, void *st, uint8_t *q, uint8_t *k, uint8_t *v,
                 uint8_t *sIn, uint8_t *sOut, uint8_t *o, float beta, float stateDecay);

namespace {

constexpr int32_t H = 512;
constexpr int32_t NKH = 2;
constexpr int32_t NVH = 6;
constexpr int32_t KD = 128;
constexpr int32_t VD = 128;
constexpr int32_t FF = 1024;
constexpr int32_t VOCAB = 1024;
constexpr int32_t GK = NVH / NKH;      // 3
constexpr float EPS = 1e-6f;

#define CK(x) do { aclError r = (x); if (r != ACL_ERROR_NONE) { printf("[engine] acl err %d at line %d\n", (int)r, __LINE__); return 1; } } while (0)
#define SYNC(tag) do { aclError r = (aclrtSynchronizeStream(stream)); if (r != ACL_ERROR_NONE) { printf("[engine] sync FAIL at '%s' err %d (line %d)\n", tag, (int)r, __LINE__); return 1; } printf("[engine] ok: %s\n", tag); } while (0)

uint8_t *dev(size_t n) { void *p = nullptr; aclrtMalloc(&p, n, ACL_MEM_MALLOC_HUGE_FIRST); return (uint8_t *)p; }
void h2d(uint8_t *d, const void *h, size_t n) { aclrtMemcpy(d, n, h, n, ACL_MEMCPY_HOST_TO_DEVICE); }
void d2h(void *h, uint8_t *d, size_t n) { aclrtMemcpy(h, n, d, n, ACL_MEMCPY_DEVICE_TO_HOST); }

struct Weights {
    std::vector<float> norm_w, Wq, Wk, Wv, convWq, convWk, convWv, Wb, Wa, A_log, dt_bias,
        Wz, norm_out_w, Wo, mlp_norm_w, Wgate, Wup, Wdown, embed, lm_head, final_norm_w;
};

std::vector<float> rnd(size_t n, float scale, std::mt19937 &rng)
{
    std::normal_distribution<float> d(0.0f, scale);
    std::vector<float> v(n);
    for (auto &x : v) x = d(rng);
    return v;
}

void save_f16(const char *path, const std::vector<float> &v)
{
    std::vector<uint16_t> h(v.size());
    for (size_t i = 0; i < v.size(); ++i) {
        float x = v[i];
        // float -> half（简单 round-to-nearest 实现，够用）
        uint32_t f;
        memcpy(&f, &x, 4);
        uint32_t sign = (f >> 16) & 0x8000;
        int32_t exp = (int32_t)((f >> 23) & 0xFF) - 127 + 15;
        uint32_t mant = f & 0x7FFFFF;
        uint16_t hh;
        if (exp <= 0) hh = (uint16_t)sign;
        else if (exp >= 31) hh = (uint16_t)(sign | 0x7C00);
        else hh = (uint16_t)(sign | ((uint32_t)exp << 10) | (mant >> 13));
        h[i] = hh;
    }
    FILE *fp = fopen(path, "wb");
    if (fp) { fwrite(h.data(), 2, h.size(), fp); fclose(fp); }
}

void save_f32(const char *path, const std::vector<float> &v)
{
    FILE *fp = fopen(path, "wb");
    if (fp) { fwrite(v.data(), 4, v.size(), fp); fclose(fp); }
}

}  // namespace

int run_engine_mini(void *stream)
{
    std::mt19937 rng(12345);
    Weights W;
    W.norm_w = std::vector<float>(H, 1.0f);
    W.final_norm_w = std::vector<float>(H, 1.0f);
    W.mlp_norm_w = std::vector<float>(H, 1.0f);
    W.norm_out_w = std::vector<float>(VD, 1.0f);
    for (auto &x : W.norm_w) x += 0.05f * ((rng() % 200) / 100.0f - 1.0f);
    for (auto &x : W.final_norm_w) x += 0.05f * ((rng() % 200) / 100.0f - 1.0f);
    for (auto &x : W.mlp_norm_w) x += 0.05f * ((rng() % 200) / 100.0f - 1.0f);
    for (auto &x : W.norm_out_w) x += 0.05f * ((rng() % 200) / 100.0f - 1.0f);
    const float s = 0.05f;
    W.Wq = rnd((size_t)NKH * KD * H, s, rng);
    W.Wk = rnd((size_t)NKH * KD * H, s, rng);
    W.Wv = rnd((size_t)NVH * VD * H, s, rng);
    W.convWq = rnd((size_t)4 * NKH * KD, 0.5f, rng);
    W.convWk = rnd((size_t)4 * NKH * KD, 0.5f, rng);
    W.convWv = rnd((size_t)4 * NVH * VD, 0.5f, rng);
    W.Wb = rnd((size_t)NVH * H, s, rng);
    W.Wa = rnd((size_t)NVH * H, s, rng);
    W.A_log = rnd(NVH, 0.5f, rng);
    W.dt_bias = rnd(NVH, 0.5f, rng);
    W.Wz = rnd((size_t)NVH * VD * H, s, rng);
    W.Wo = rnd((size_t)H * NVH * VD, s, rng);
    W.Wgate = rnd((size_t)FF * H, s, rng);
    W.Wup = rnd((size_t)FF * H, s, rng);
    W.Wdown = rnd((size_t)H * FF, s, rng);
    W.embed = rnd((size_t)VOCAB * H, 1.0f, rng);
    W.lm_head = rnd((size_t)VOCAB * H, s, rng);

    // 输入 token 与初始状态
    const int32_t tok = 42;
    std::vector<float> xhost(W.embed.begin() + (size_t)tok * H, W.embed.begin() + (size_t)(tok + 1) * H);

    // dump 权重供 numpy 参考
    system("mkdir -p output");
    save_f16("./output/eng_norm_w.bin", W.norm_w);
    save_f16("./output/eng_Wq.bin", W.Wq);
    save_f16("./output/eng_Wk.bin", W.Wk);
    save_f16("./output/eng_Wv.bin", W.Wv);
    save_f16("./output/eng_convWq.bin", W.convWq);
    save_f16("./output/eng_convWk.bin", W.convWk);
    save_f16("./output/eng_convWv.bin", W.convWv);
    save_f16("./output/eng_Wb.bin", W.Wb);
    save_f16("./output/eng_Wa.bin", W.Wa);
    save_f32("./output/eng_Alog.bin", W.A_log);
    save_f32("./output/eng_dtbias.bin", W.dt_bias);
    save_f16("./output/eng_Wz.bin", W.Wz);
    save_f16("./output/eng_norm_out_w.bin", W.norm_out_w);
    save_f16("./output/eng_Wo.bin", W.Wo);
    save_f16("./output/eng_mlp_norm_w.bin", W.mlp_norm_w);
    save_f16("./output/eng_Wgate.bin", W.Wgate);
    save_f16("./output/eng_Wup.bin", W.Wup);
    save_f16("./output/eng_Wdown.bin", W.Wdown);
    save_f16("./output/eng_lm_head.bin", W.lm_head);
    save_f16("./output/eng_final_norm_w.bin", W.final_norm_w);
    save_f16("./output/eng_x.bin", xhost);

    // ---- 上传权重 ----
    auto up = [&](const std::vector<float> &v) -> uint8_t * {
        size_t n = v.size() * 2;
        uint8_t *d = dev(n);
        std::vector<uint16_t> h(v.size());
        for (size_t i = 0; i < v.size(); ++i) {
            float x = v[i]; uint32_t f; memcpy(&f, &x, 4);
            uint32_t sign = (f >> 16) & 0x8000;
            int32_t exp = (int32_t)((f >> 23) & 0xFF) - 127 + 15;
            uint32_t mant = f & 0x7FFFFF;
            uint16_t hh;
            if (exp <= 0) hh = (uint16_t)sign;
            else if (exp >= 31) hh = (uint16_t)(sign | 0x7C00);
            else hh = (uint16_t)(sign | ((uint32_t)exp << 10) | (mant >> 13));
            h[i] = hh;
        }
        h2d(d, h.data(), n);
        return d;
    };

    uint8_t *d_normW = up(W.norm_w), *d_Wq = up(W.Wq), *d_Wk = up(W.Wk), *d_Wv = up(W.Wv);
    uint8_t *d_cWq = up(W.convWq), *d_cWk = up(W.convWk), *d_cWv = up(W.convWv);
    uint8_t *d_Wb = up(W.Wb), *d_Wa = up(W.Wa), *d_Wz = up(W.Wz);
    uint8_t *d_normOutW = up(W.norm_out_w), *d_Wo = up(W.Wo);
    uint8_t *d_mlpNormW = up(W.mlp_norm_w), *d_Wgate = up(W.Wgate), *d_Wup = up(W.Wup);
    uint8_t *d_Wdown = up(W.Wdown), *d_lmHead = up(W.lm_head), *d_finalNormW = up(W.final_norm_w);
    (void)d_lmHead;

    const size_t hb = (size_t)H * 2;
    const size_t qb = (size_t)NKH * KD * 2;
    const size_t vb = (size_t)NVH * VD * 2;
    const size_t ffb = (size_t)FF * 2;
    const size_t vboc = (size_t)VOCAB * 2;

    uint8_t *d_x = dev(hb), *d_xn = dev(hb), *d_tmp = dev(hb), *d_y = dev(hb);
    uint8_t *d_q = dev(qb), *d_k = dev(qb), *d_v = dev(vb);
    uint8_t *d_qc = dev(qb), *d_kc = dev(qb), *d_vc = dev(vb);
    uint8_t *d_qn = dev(qb), *d_kn = dev(qb);
    uint8_t *d_o = dev(vb), *d_z = dev(vb), *d_onorm = dev(vb);
    uint8_t *d_b = dev(64), *d_a = dev(64);   // fp16_gemv 按 16 行一批写回，需 ≥32B 缓冲
    uint8_t *d_g1 = dev(ffb), *d_u = dev(ffb), *d_actv = dev(ffb), *d_y2 = dev(hb);
    uint8_t *d_logits = dev(vboc);
    uint8_t *st_q = dev((size_t)3 * NKH * KD * 4), *st_k = dev((size_t)3 * NKH * KD * 4),
             *st_v = dev((size_t)3 * NVH * VD * 4);
    uint8_t *st_gdn = dev((size_t)NVH * KD * VD * 4);
    // 初始化为 0
    {
        std::vector<uint8_t> zeros(3 * NKH * KD * 4, 0);
        CK(aclrtMemset(st_q, zeros.size(), 0, zeros.size()));
        CK(aclrtMemset(st_k, zeros.size(), 0, zeros.size()));
        std::vector<uint8_t> z2((size_t)3 * NVH * VD * 4, 0);
        CK(aclrtMemset(st_v, z2.size(), 0, z2.size()));
        std::vector<uint8_t> z3((size_t)NVH * KD * VD * 4, 0);
        CK(aclrtMemset(st_gdn, z3.size(), 0, z3.size()));
    }

    // 上传 x（fp16）
    {
        std::vector<uint16_t> h(H);
        for (int i = 0; i < H; ++i) {
            float f = xhost[i]; uint32_t b; memcpy(&b, &f, 4);
            uint32_t sign = (b >> 16) & 0x8000;
            int32_t exp = (int32_t)((b >> 23) & 0xFF) - 127 + 15;
            uint32_t mant = b & 0x7FFFFF;
            uint16_t hh;
            if (exp <= 0) hh = (uint16_t)sign;
            else if (exp >= 31) hh = (uint16_t)(sign | 0x7C00);
            else hh = (uint16_t)(sign | ((uint32_t)exp << 10) | (mant >> 13));
            h[i] = hh;
        }
        h2d(d_x, h.data(), hb);
    }

    // ---- 前向 ----
    rmsnorm_op_do(1, stream, d_x, d_normW, d_xn, H, EPS);
    fp16_gemv_do(1, stream, d_xn, d_Wq, d_q, H, NKH * KD);
    fp16_gemv_do(1, stream, d_xn, d_Wk, d_k, H, NKH * KD);
    fp16_gemv_do(1, stream, d_xn, d_Wv, d_v, H, NVH * VD);
    conv1d_decode_do(1, stream, d_q, st_q, d_cWq, d_qc, NKH * KD);
    conv1d_decode_do(1, stream, d_k, st_k, d_cWk, d_kc, NKH * KD);
    conv1d_decode_do(1, stream, d_v, st_v, d_cWv, d_vc, NVH * VD);
    for (int h = 0; h < NKH; ++h) {
        l2norm_op_do(1, stream, d_qc + h * KD * 2, d_qn + h * KD * 2, KD, EPS, 1.0f / sqrtf((float)KD));
        l2norm_op_do(1, stream, d_kc + h * KD * 2, d_kn + h * KD * 2, KD, EPS, 1.0f / sqrtf((float)KD));
    }
    fp16_gemv_do(1, stream, d_xn, d_Wb, d_b, H, NVH);
    fp16_gemv_do(1, stream, d_xn, d_Wa, d_a, H, NVH);
    fp16_gemv_do(1, stream, d_xn, d_Wz, d_z, H, NVH * VD);
    CK(aclrtSynchronizeStream(stream));

    // beta / g（host 计算）
    std::vector<uint16_t> hb_, ha_;
    hb_.resize(NVH); ha_.resize(NVH);
    d2h(hb_.data(), d_b, (size_t)NVH * 2);
    d2h(ha_.data(), d_a, (size_t)NVH * 2);
    auto h2f = [](uint16_t hv) -> float {
        uint32_t sign = (hv & 0x8000u) << 16;
        uint32_t exp = (hv >> 10) & 0x1F;
        uint32_t mant = hv & 0x3FF;
        uint32_t f;
        if (exp == 0) { f = sign; }
        else if (exp == 31) { f = sign | 0x7F800000u | (mant << 13); }
        else { f = sign | ((exp - 15 + 127) << 23) | (mant << 13); }
        float r; memcpy(&r, &f, 4); return r;
    };
    std::vector<float> beta(NVH), gt(NVH);
    for (int i = 0; i < NVH; ++i) {
        float bv = h2f(hb_[i]);
        float av = h2f(ha_[i]) + W.dt_bias[i];
        float sp = (av > 20.0f) ? av : logf(1.0f + expf(av));
        float gv = -expf(W.A_log[i]) * sp;
        beta[i] = 1.0f / (1.0f + expf(-bv));
        gt[i] = gv;
    }

    for (int vh = 0; vh < NVH; ++vh) {
        int kh = vh / GK;
        float decay = expf(gt[vh]);
        gdn_rgdr_do(1, stream, d_qn + kh * KD * 2, d_kn + kh * KD * 2, d_vc + vh * VD * 2,
                    st_gdn + (size_t)vh * KD * VD * 4, st_gdn + (size_t)vh * KD * VD * 4,
                    d_o + vh * VD * 2, beta[vh], decay);
    }
    for (int vh = 0; vh < NVH; ++vh) {
        rmsnorm_op_do(1, stream, d_o + vh * VD * 2, d_normOutW, d_tmp, VD, EPS);
        mul_silu_do(1, stream, d_z + vh * VD * 2, d_tmp, d_onorm + vh * VD * 2, VD);
    }
    fp16_gemv_do(1, stream, d_onorm, d_Wo, d_y, NVH * VD, H);
    SYNC("Wo gemv");
    vec_axpy_do(1, stream, d_x, d_y, d_tmp, 1.0f, H);      // h = x + y (d_tmp 复用为 h)
    SYNC("residual axpy");
    // MLP
    rmsnorm_op_do(1, stream, d_tmp, d_mlpNormW, d_xn, H, EPS);
    SYNC("mlp rmsnorm");
    fp16_gemv_do(1, stream, d_xn, d_Wgate, d_g1, H, FF);
    fp16_gemv_do(1, stream, d_xn, d_Wup, d_u, H, FF);
    SYNC("mlp gate/up gemv");
    mul_silu_do(1, stream, d_g1, d_u, d_actv, FF);
    SYNC("mlp mul_silu");
    fp16_gemv_do(1, stream, d_actv, d_Wdown, d_y2, FF, H);
    SYNC("mlp down gemv");
    vec_axpy_do(1, stream, d_tmp, d_y2, d_y, 1.0f, H);      // 层输出复用 d_y
    SYNC("mlp residual");
    // 最终 norm + lm_head
    rmsnorm_op_do(1, stream, d_y, d_finalNormW, d_xn, H, EPS);
    SYNC("final rmsnorm");
    fp16_gemv_do(1, stream, d_xn, d_lmHead, d_logits, H, VOCAB);
    SYNC("lm_head gemv");

    // dump 中间结果（必须在计时循环之前，否则会被计时用的陈旧前向覆盖）
    std::vector<uint16_t> tmp16;
    auto dump16 = [&](const char *path, uint8_t *d, size_t nElem) {
        tmp16.resize(nElem);
        d2h(tmp16.data(), d, nElem * 2);
        FILE *fp = fopen(path, "wb");
        if (fp) { fwrite(tmp16.data(), 2, nElem, fp); fclose(fp); }
    };
    dump16("./output/eng_out_xn.bin", d_xn, H);
    dump16("./output/eng_out_y.bin", d_y, H);
    dump16("./output/eng_out_logits.bin", d_logits, VOCAB);
    dump16("./output/eng_out_qc.bin", d_qc, NKH * KD);
    dump16("./output/eng_out_kc.bin", d_kc, NKH * KD);
    dump16("./output/eng_out_vc.bin", d_vc, NVH * VD);
    dump16("./output/eng_out_qn.bin", d_qn, NKH * KD);
    dump16("./output/eng_out_o.bin", d_o, NVH * VD);
    dump16("./output/eng_out_onorm.bin", d_onorm, NVH * VD);
    save_f32("./output/eng_out_beta.bin", beta);
    save_f32("./output/eng_out_g.bin", gt);

    // 计时
    auto t0 = std::chrono::high_resolution_clock::now();
    const int REP = 5;
    for (int r = 0; r < REP; ++r) {
        rmsnorm_op_do(1, stream, d_x, d_normW, d_xn, H, EPS);
        fp16_gemv_do(1, stream, d_xn, d_Wq, d_q, H, NKH * KD);
        fp16_gemv_do(1, stream, d_xn, d_Wk, d_k, H, NKH * KD);
        fp16_gemv_do(1, stream, d_xn, d_Wv, d_v, H, NVH * VD);
        fp16_gemv_do(1, stream, d_xn, d_Wb, d_b, H, NVH);
        fp16_gemv_do(1, stream, d_xn, d_Wa, d_a, H, NVH);
        fp16_gemv_do(1, stream, d_xn, d_Wz, d_z, H, NVH * VD);
        fp16_gemv_do(1, stream, d_onorm, d_Wo, d_y, NVH * VD, H);
        fp16_gemv_do(1, stream, d_xn, d_Wgate, d_g1, H, FF);
        fp16_gemv_do(1, stream, d_xn, d_Wup, d_u, H, FF);
        fp16_gemv_do(1, stream, d_actv, d_Wdown, d_y2, FF, H);
        fp16_gemv_do(1, stream, d_xn, d_lmHead, d_logits, H, VOCAB);
    }
    CK(aclrtSynchronizeStream(stream));
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / REP;
    // 参数量估算（含 lm_head/embed 分摊）
    double params = (double)(W.Wq.size() + W.Wk.size() + W.Wv.size() + W.Wb.size() + W.Wa.size() +
                             W.Wz.size() + W.Wo.size() + W.Wgate.size() + W.Wup.size() + W.Wdown.size() +
                             W.lm_head.size());
    printf("[engine_mini] GDN层+MLP+lm_head decode: %.2f ms/token, ~%.1f M params, 有效权重带宽 %.2f GB/s\n",
           ms, params / 1e6, params * 2.0 / 1e9 / (ms / 1000.0));
    printf("[engine_mini] done\n");
    return 0;
}
