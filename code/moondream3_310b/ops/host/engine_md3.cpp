// engine_md3.cpp — Moondream 3 Preview decode engine for Ascend 310B1
#include <acl/acl.h>
#include <aclnnop/aclnn_matmul.h>
#include <aclnnop/aclnn_add.h>
#include <aclnnop/aclnn_softmax.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>
#include <chrono>
#include <algorithm>
#include <random>

#define CK(x) do { aclError _r=(x); if(_r!=ACL_ERROR_NONE){ printf("[md3] acl err %d line %d\n",(int)_r,__LINE__); return 1; } } while(0)

// ---- kernels ----
void w4a16_gemv_gen_do(uint32_t b, void *st, uint8_t *act, uint8_t *wq, uint8_t *scales,
                       uint8_t *szs, uint8_t *out, int32_t K, int32_t N, int32_t stage);
void fp16_gemv_do(uint32_t b, void *st, uint8_t *act, uint8_t *w, uint8_t *out, int32_t K, int32_t N);
void vec_axpy_do(uint32_t b, void *st, uint8_t *a, uint8_t *b_, uint8_t *out, float s, int32_t N);
void layernorm_f16_do(uint32_t b, void *st, uint8_t *x, uint8_t *w, uint8_t *bb, uint8_t *y,
                      int32_t H, int32_t rows, float eps);
void gelu_f16_do(uint32_t b, void *st, uint8_t *x, uint8_t *y, int32_t N);
void geglu_f16_do(uint32_t b, void *st, uint8_t *g, uint8_t *u, uint8_t *o, int32_t N);
void add_bias_f16_do(uint32_t b, void *st, uint8_t *y, uint8_t *bias, int32_t M, int32_t N);
void dequant_w4_f16_do(uint32_t b, void *st, uint8_t *wq, uint8_t *s, uint8_t *sz, uint8_t *o,
                       int32_t K, int32_t N);
void qkv_post_text_do(uint32_t b, void *st, uint8_t *qkv, uint8_t *tq, uint8_t *tv, uint8_t *cos,
                      uint8_t *sin, uint8_t *q, uint8_t *k, uint8_t *v, int32_t T);
void attn_decode_text_do(uint32_t b, void *st, uint8_t *qkv, uint8_t *tq, uint8_t *tv,
                         uint8_t *cos, uint8_t *sin, uint8_t *kc, uint8_t *vc, uint8_t *o, int32_t pos);
void heads_to_seq_do(uint32_t b, void *st, uint8_t *src, uint8_t *dst, int32_t T, int32_t H, int32_t D);
void qkv_post_vision_do(uint32_t b, void *st, uint8_t *qkv, uint8_t *q, uint8_t *k, uint8_t *v,
                        int32_t T, int32_t H, int32_t D, int32_t Dpad, float qscale);
void gelu_pad_f16_do(uint32_t b, void *st, uint8_t *x, uint8_t *y, int32_t rows, int32_t K, int32_t Kpad);
void add_f16_do(uint32_t b, void *st, uint8_t *a, uint8_t *b_, uint8_t *o, int32_t N);
void geglu_batch_f16_do(uint32_t b, void *st, uint8_t *gu, uint8_t *o, int32_t rows, int32_t EI);
void gelu_pad_f16_do(uint32_t b, void *st, uint8_t *x, uint8_t *y, int32_t rows, int32_t K, int32_t Kpad);

namespace md3 {

static inline float h2f_bits(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    int32_t ex = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FFu;
    float r;
    if (ex == 0) {
        if (mant == 0) { r = 0.0f; }
        else { uint32_t u = sign | 0x3A000000u | (mant << 13); float f; memcpy(&f, &u, 4); r = f; }
    } else if (ex == 31) {
        uint32_t u = sign | 0x7F800000u | (mant << 13); float f; memcpy(&f, &u, 4); r = f;
    } else {
        uint32_t u = sign | ((uint32_t)(ex - 15 + 127) << 23) | (mant << 13); float f; memcpy(&f, &u, 4); r = f;
    }
    return r;
}
static float g_h2f_tab[65536];
static bool g_h2f_ready = false;
static inline float h2f_quick(uint16_t hv) {
    if (!g_h2f_ready) { for (int i = 0; i < 65536; ++i) g_h2f_tab[i] = h2f_bits((uint16_t)i); g_h2f_ready = true; }
    return g_h2f_tab[hv];
}
static aclrtStream g_stream = nullptr;
static void *g_ws = nullptr;
static size_t g_wsSize = 0;

static void *devAlloc(size_t n) {
    void *p = nullptr;
    aclError r = aclrtMalloc(&p, n ? n : 1, ACL_MEM_MALLOC_HUGE_FIRST);
    if (r != ACL_ERROR_NONE) { printf("[md3] malloc %zu failed %d\n", n, (int)r); return nullptr; }
    return p;
}
static int upload(void *d, const void *h, size_t n) {
    if (getenv("MD3_TRACE_MEM") && n > 1000000) { printf("[mem] up n=%zu dst=%p src=%p\n", n, d, h); fflush(stdout); }
    aclError r = aclrtMemcpy(d, n, h, n, ACL_MEMCPY_HOST_TO_DEVICE);
    if (r != ACL_ERROR_NONE) { printf("[md3] upload fail %zu err=%d msg=%s\n", n, (int)r, aclGetRecentErrMsg() ? aclGetRecentErrMsg() : ""); return 1; }
    return 0;
}
static int download(void *h, const void *d, size_t n) {
    if (getenv("MD3_TRACE_MEM") && n > 1000000) { printf("[mem] dn n=%zu dst=%p src=%p\n", n, h, d); fflush(stdout); }
    aclError r = aclrtMemcpy(h, n, d, n, ACL_MEMCPY_DEVICE_TO_HOST);
    if (r != ACL_ERROR_NONE) { printf("[md3] download fail %zu err=%d msg=%s\n", n, (int)r, aclGetRecentErrMsg() ? aclGetRecentErrMsg() : ""); return 1; }
    return 0;
}
static int sync() {
    aclError r = aclrtSynchronizeStream(g_stream);
    if (r != ACL_ERROR_NONE) { printf("[md3] sync fail err=%d msg=%s\n", (int)r, aclGetRecentErrMsg() ? aclGetRecentErrMsg() : ""); return 1; }
    return 0;
}
static int d2d(void *dst, const void *src, size_t n) {
    return aclrtMemcpy(dst, n, src, n, ACL_MEMCPY_DEVICE_TO_DEVICE) == ACL_ERROR_NONE ? 0 : 1;
}
static double now() {
    return std::chrono::duration<double>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();
}

static aclTensor *mk2(int64_t r, int64_t c, void *p, aclDataType dt = ACL_FLOAT16) {
    int64_t dims[2] = {r, c};
    int64_t strides[2] = {c, 1};
    return aclCreateTensor(dims, 2, dt, strides, 0, ACL_FORMAT_ND, dims, 2, p);
}
// C[M,N] = A[M,K] * B[K,N], B 以 [N,K] 行主序存储（转置 view）
static int matmulTN(int64_t M, int64_t K, int64_t N, void *A, void *Bnk, void *C) {
    aclTensor *ta = mk2(M, K, A);
    int64_t bd[2] = {K, N}, bs[2] = {1, K};
    aclTensor *tb = aclCreateTensor(bd, 2, ACL_FLOAT16, bs, 0, ACL_FORMAT_ND, bd, 2, Bnk);
    aclTensor *tc = mk2(M, N, C);
    uint64_t ws = 0; aclOpExecutor *ex = nullptr;
    auto st = aclnnMatmulGetWorkspaceSize(ta, tb, tc, 1, &ws, &ex);
    if (st != 0) { printf("[md3] matmulTN getws %ldx%ldx%ld fail %d\n", (long)M, (long)K, (long)N, (int)st); return 1; }
    if (ws > g_wsSize) {
        if (g_ws) { sync(); aclrtFree(g_ws); }
        g_ws = devAlloc(ws); g_wsSize = ws;
        if (!g_ws) return 1;
    }
    st = aclnnMatmul(g_ws, ws, ex, g_stream);
    aclDestroyTensor(ta); aclDestroyTensor(tb); aclDestroyTensor(tc);
    if (st != 0) { printf("[md3] matmulTN exec fail %d\n", (int)st); return 1; }
    return 0;
}
// C[M,N] = A[M,K] * B[K,N], B 连续 [K,N]
static int matmulNN(int64_t M, int64_t K, int64_t N, void *A, void *Bkn, void *C) {
    aclTensor *ta = mk2(M, K, A);
    aclTensor *tb = mk2(K, N, Bkn);
    aclTensor *tc = mk2(M, N, C);
    uint64_t ws = 0; aclOpExecutor *ex = nullptr;
    auto st = aclnnMatmulGetWorkspaceSize(ta, tb, tc, 1, &ws, &ex);
    if (st != 0) { printf("[md3] matmulNN getws fail %d\n", (int)st); return 1; }
    if (ws > g_wsSize) {
        if (g_ws) { sync(); aclrtFree(g_ws); }
        g_ws = devAlloc(ws); g_wsSize = ws;
        if (!g_ws) return 1;
    }
    st = aclnnMatmul(g_ws, ws, ex, g_stream);
    aclDestroyTensor(ta); aclDestroyTensor(tb); aclDestroyTensor(tc);
    if (st != 0) { printf("[md3] matmulNN exec fail %d\n", (int)st); return 1; }
    return 0;
}
static int addBroadcast(int64_t B, int64_t R, int64_t C, void *x, void *mask, void *y) {
    int64_t dims3[3] = {B, R, C};
    int64_t strides3[3] = {R * C, C, 1};
    aclTensor *tx = aclCreateTensor(dims3, 3, ACL_FLOAT16, strides3, 0, ACL_FORMAT_ND, dims3, 3, x);
    aclTensor *tm = mk2(R, C, mask);
    aclTensor *ty = aclCreateTensor(dims3, 3, ACL_FLOAT16, strides3, 0, ACL_FORMAT_ND, dims3, 3, y);
    float one = 1.0f; aclScalar *alpha = aclCreateScalar(&one, ACL_FLOAT);
    uint64_t ws = 0; aclOpExecutor *ex = nullptr;
    auto st = aclnnAddGetWorkspaceSize(tx, tm, alpha, ty, &ws, &ex);
    if (st == 0) {
        if (ws > g_wsSize) { if (g_ws) { sync(); aclrtFree(g_ws); } g_ws = devAlloc(ws); g_wsSize = ws; }
        st = aclnnAdd(g_ws, ws, ex, g_stream);
    }
    aclDestroyTensor(tx); aclDestroyTensor(tm); aclDestroyTensor(ty); aclDestroyScalar(alpha);
    return st == 0 ? 0 : 1;
}
static int softmax2(int64_t R, int64_t C, void *x, void *y) {
    aclTensor *tx = mk2(R, C, x);
    aclTensor *ty = mk2(R, C, y);
    uint64_t ws = 0; aclOpExecutor *ex = nullptr;
    auto st = aclnnSoftmaxGetWorkspaceSize(tx, 1, ty, &ws, &ex);
    if (st == 0) {
        if (ws > g_wsSize) { if (g_ws) { sync(); aclrtFree(g_ws); } g_ws = devAlloc(ws); g_wsSize = ws; }
        st = aclnnSoftmax(g_ws, ws, ex, g_stream);
    }
    aclDestroyTensor(tx); aclDestroyTensor(ty);
    return st == 0 ? 0 : 1;
}
static int softmax3(int64_t B, int64_t R, int64_t C, void *x, void *y) {
    int64_t dims3[3] = {B, R, C};
    int64_t strides3[3] = {R * C, C, 1};
    aclTensor *tx = aclCreateTensor(dims3, 3, ACL_FLOAT16, strides3, 0, ACL_FORMAT_ND, dims3, 3, x);
    aclTensor *ty = aclCreateTensor(dims3, 3, ACL_FLOAT16, strides3, 0, ACL_FORMAT_ND, dims3, 3, y);
    uint64_t ws = 0; aclOpExecutor *ex = nullptr;
    auto st = aclnnSoftmaxGetWorkspaceSize(tx, 2, ty, &ws, &ex);
    if (st == 0) {
        if (ws > g_wsSize) { if (g_ws) { sync(); aclrtFree(g_ws); } g_ws = devAlloc(ws); g_wsSize = ws; }
        st = aclnnSoftmax(g_ws, ws, ex, g_stream);
    }
    aclDestroyTensor(tx); aclDestroyTensor(ty);
    return st == 0 ? 0 : 1;
}

// ---------------- meta ----------------
struct Cfg {
    int hidden = 2048, n_layers = 24, n_heads = 32, n_kv = 32, head_dim = 64, rot_dim = 32;
    int vocab = 51200, max_context = 4096, moe_start = 4, n_experts = 64, topk = 8;
    int inner = 1024, dense_ff = 8192, prefix = 730;
    float eps = 1e-5f, theta = 1500000.0f;
};
struct W4 {
    int N = 0, K = 0;
    uint8_t *w = nullptr, *s = nullptr, *sz = nullptr;
    int sbp = 0;
};
struct F16 { int64_t n = 0; uint8_t *dev = nullptr; };
struct Layer {
    F16 lnw, lnb, qkvb, ob, tauwq, tauwv, taua, fc1b, fc2b, routerb;
    W4 qkv, o, fc1, fc2, router;
    std::vector<W4> efc1, efc2;
    bool moe = false;
    float alpha[32] = {0};
};

static bool readFile(const std::string &p, std::vector<uint8_t> &buf, size_t expect) {
    FILE *f = fopen(p.c_str(), "rb");
    if (!f) { printf("[md3] open fail %s\n", p.c_str()); return false; }
    if (buf.size() < expect) buf.resize(expect);
    size_t got = fread(buf.data(), 1, expect, f);
    fclose(f);
    if (got != expect) { printf("[md3] short read %s %zu/%zu\n", p.c_str(), got, expect); return false; }
    return true;
}
static int loadW4(const std::string &dir, const std::string &prefix, int N, int K, W4 &w,
                  std::vector<uint8_t> &scratch) {
    w.N = N; w.K = K;
    int ng = K / 32; w.sbp = (ng + 15) / 16 * 16;
    size_t wb = (size_t)N * (K / 4) * 2, sb = (size_t)N * w.sbp * 2;
    w.w = (uint8_t *)devAlloc(wb); w.s = (uint8_t *)devAlloc(sb); w.sz = (uint8_t *)devAlloc(sb);
    if (!w.w || !w.s || !w.sz) return 1;
    if (!readFile(dir + "/" + prefix + ".w4.bin", scratch, wb)) return 1;
    if (upload(w.w, scratch.data(), wb)) return 1;
    if (!readFile(dir + "/" + prefix + ".s.bin", scratch, sb)) return 1;
    if (upload(w.s, scratch.data(), sb)) return 1;
    if (!readFile(dir + "/" + prefix + ".sz.bin", scratch, sb)) return 1;
    if (upload(w.sz, scratch.data(), sb)) return 1;
    return 0;
}
static int loadF16(const std::string &dir, const std::string &file, int64_t n, F16 &t,
                   std::vector<uint8_t> &scratch) {
    t.n = n;
    t.dev = (uint8_t *)devAlloc((size_t)n * 2);
    if (!t.dev) return 1;
    if (!readFile(dir + "/" + file, scratch, (size_t)n * 2)) return 1;
    return upload(t.dev, scratch.data(), (size_t)n * 2);
}

}  // namespace md3


#include <map>
#include <set>
#include <sstream>
#include <utility>
#include <thread>

namespace md3 {

struct VLayer {
    F16 ln1w, ln1b, ln2w, ln2b, qkvb, ob, fc1b, fc2b;
    F16 qkvW, oW, fc1W, fc2W;   // fp16 权重（[N,K] 行主序）
};

struct Eng {
    Cfg cfg;
    bool withVision = false;
    std::vector<VLayer> vlayers;
    F16 vpatchb, vpos, vfnw, vfnb, vproj1b, vproj2b;
    F16 vproj1W, vproj2W;   // [8192,2304] / [2048,8192] fp16
    std::map<std::string, std::string> w4map, f16map;
    std::string dir;
    bool text_only = true;
    int limit_layers = 0;
    std::vector<Layer> layers;
    std::vector<uint8_t *> kc, vc;
    // host wte
    std::vector<uint8_t> hwte_w, hwte_s, hwte_sz;
    int wte_N = 0, wte_K = 0, wte_sbp = 0;
    // buffers
    uint8_t *d_x = nullptr, *d_xn = nullptr, *d_qkv = nullptr, *d_gelu = nullptr;
    uint8_t *d_taq = nullptr, *d_tav = nullptr, *d_tauq = nullptr, *d_tauv = nullptr;
    uint8_t *d_cos = nullptr, *d_sin = nullptr, *d_attn = nullptr, *d_proj = nullptr;
    uint8_t *d_fc1 = nullptr, *d_mlp = nullptr, *d_router = nullptr;
    uint8_t *d_e1 = nullptr, *d_e2 = nullptr, *d_zero = nullptr, *d_logits = nullptr;
    uint8_t *d_emb = nullptr;
    int n_layers = 0;

    int allocBufs() {
        const int H = cfg.hidden;
        d_x = (uint8_t *)devAlloc(H * 2);
        d_xn = (uint8_t *)devAlloc(H * 2);
        d_emb = (uint8_t *)devAlloc(H * 2);
        d_qkv = (uint8_t *)devAlloc(3 * H * 2);
        d_gelu = (uint8_t *)devAlloc(3 * H * 2);
        d_taq = (uint8_t *)devAlloc(64 * 2);
        d_tav = (uint8_t *)devAlloc(64 * 2);
        d_tauq = (uint8_t *)devAlloc(32 * 4);
        d_tauv = (uint8_t *)devAlloc(32 * 4);
        d_cos = (uint8_t *)devAlloc(16 * 4);
        d_sin = (uint8_t *)devAlloc(16 * 4);
        d_attn = (uint8_t *)devAlloc(H * 2);
        d_proj = (uint8_t *)devAlloc(H * 2);
        d_fc1 = (uint8_t *)devAlloc(cfg.dense_ff * 2);
        d_mlp = (uint8_t *)devAlloc(H * 2);
        d_router = (uint8_t *)devAlloc(64 * 2);
        d_e1 = (uint8_t *)devAlloc(2048 * 2);
        d_e2 = (uint8_t *)devAlloc(2048 * 2);
        d_zero = (uint8_t *)devAlloc(H * 2);
        d_logits = (uint8_t *)devAlloc(cfg.vocab * 2);
        if (!d_x || !d_xn || !d_qkv || !d_gelu || !d_attn || !d_proj || !d_fc1 ||
            !d_mlp || !d_router || !d_e1 || !d_e2 || !d_logits || !d_emb) return 1;
        CK(aclrtMemset(d_zero, H * 2, 0, H * 2));
        return 0;
    }

    int allocKv() {
        size_t one = (size_t)cfg.n_kv * cfg.max_context * cfg.head_dim * 2;
        for (int i = 0; i < n_layers; ++i) {
            uint8_t *k = (uint8_t *)devAlloc(one), *v = (uint8_t *)devAlloc(one);
            if (!k || !v) return 1;
            CK(aclrtMemset(k, one, 0, one));
            CK(aclrtMemset(v, one, 0, one));
            kc.push_back(k); vc.push_back(v);
        }
        return 0;
    }

    void wteRowHost(int tok, uint16_t *row) {
        const int K = wte_K, KQ = K / 4;
        const uint16_t *words = (const uint16_t *)(hwte_w.data() + (size_t)tok * KQ * 2);
        const uint16_t *sc = (const uint16_t *)(hwte_s.data() + (size_t)tok * wte_sbp * 2);
        const uint16_t *sz = (const uint16_t *)(hwte_sz.data() + (size_t)tok * wte_sbp * 2);
        for (int p = 0; p < 4; ++p) {
            for (int j = 0; j < KQ; ++j) {
                int kk = p * KQ + j, g = kk / 32;
                float sv = h2f_host(sc[g]), szv = h2f_host(sz[g]);
                int q = (words[j] >> (4 * p)) & 0xF;
                float w = q * sv - szv;
                uint32_t u; memcpy(&u, &w, 4);
                uint32_t sign = (u >> 16) & 0x8000u; int ex = (int)((u >> 23) & 0xFF) - 127 + 15;
                uint32_t mant = u & 0x7FFFFFu;
                row[kk] = (ex <= 0) ? (uint16_t)sign : (ex >= 31 ? (uint16_t)(sign | 0x7C00u)
                          : (uint16_t)(sign | (ex << 10) | (mant >> 13)));
            }
        }
    }

    int embedRow(int tok, uint8_t *dstDev) {
        std::vector<uint16_t> row(wte_K);
        wteRowHost(tok, row.data());
        return upload(dstDev, row.data(), wte_K * 2);
    }

    int loadHostWte(const std::string &prefix) {
        size_t wb = (size_t)wte_N * (wte_K / 4) * 2;
        size_t sb = (size_t)wte_N * wte_sbp * 2;
        hwte_w.resize(wb); hwte_s.resize(sb); hwte_sz.resize(sb);
        auto rd = [&](const std::string &f, std::vector<uint8_t> &b) -> bool {
            FILE *fp = fopen(f.c_str(), "rb");
            if (!fp) { printf("[md3] open fail %s\n", f.c_str()); return false; }
            size_t got = fread(b.data(), 1, b.size(), fp);
            fclose(fp);
            if (got != b.size()) { printf("[md3] short %s\n", f.c_str()); return false; }
            return true;
        };
        return (rd(dir + "/" + prefix + ".w4.bin", hwte_w) && rd(dir + "/" + prefix + ".s.bin", hwte_s)
                && rd(dir + "/" + prefix + ".sz.bin", hwte_sz)) ? 0 : 1;
    }

    int loadAll() {
        FILE *mf = fopen((dir + "/meta.txt").c_str(), "r");
        if (!mf) { printf("[md3] no meta.txt in %s\n", dir.c_str()); return 1; }
        std::vector<std::string> lines;
        char line[4096];
        while (fgets(line, sizeof(line), mf)) {
            std::string s(line);
            while (!s.empty() && (s.back() == '\n' || s.back() == '\r')) s.pop_back();
            if (!s.empty()) lines.push_back(s);
        }
        fclose(mf);
        for (auto &s : lines) {
            std::vector<std::string> t;
            size_t p = 0;
            while (p < s.size()) {
                size_t q = s.find(' ', p);
                if (q == std::string::npos) { t.push_back(s.substr(p)); break; }
                if (q > p) t.push_back(s.substr(p, q - p));
                p = q + 1;
            }
            if (t.empty()) continue;
            if (t[0] == "CFG" && t.size() >= 3) {
                int v = atoi(t[2].c_str());
                if (t[1] == "hidden") cfg.hidden = v;
                else if (t[1] == "n_layers") cfg.n_layers = v;
                else if (t[1] == "n_heads") cfg.n_heads = v;
                else if (t[1] == "n_kv_heads") cfg.n_kv = v;
                else if (t[1] == "head_dim") cfg.head_dim = v;
                else if (t[1] == "rot_dim") cfg.rot_dim = v;
                else if (t[1] == "vocab") cfg.vocab = v;
                else if (t[1] == "max_context") cfg.max_context = v;
                else if (t[1] == "moe_start") cfg.moe_start = v;
                else if (t[1] == "n_experts") cfg.n_experts = v;
                else if (t[1] == "experts_per_token") cfg.topk = v;
                else if (t[1] == "expert_inner") cfg.inner = v;
                else if (t[1] == "dense_ff") cfg.dense_ff = v;
                else if (t[1] == "prefix_tokens") cfg.prefix = v;
                else if (t[1] == "eps") cfg.eps = (float)atof(t[2].c_str());
                else if (t[1] == "rope_theta") cfg.theta = (float)atof(t[2].c_str());
            } else if (t[0] == "W4" && t.size() >= 5) {
                w4map[t[1]] = t[2] + " " + t[3] + " " + t[4];
            } else if (t[0] == "F16" && t.size() >= 4) {
                f16map[t[1]] = t[2] + " " + t[3];
            }
        }
        n_layers = limit_layers ? std::min(limit_layers, cfg.n_layers) : cfg.n_layers;
        printf("[md3] cfg hidden=%d layers=%d heads=%d/%d vocab=%d moe_start=%d exp=%d topk=%d\n",
               cfg.hidden, n_layers, cfg.n_heads, cfg.n_kv, cfg.vocab, cfg.moe_start, cfg.n_experts, cfg.topk);
        if (allocBufs() || allocKv()) return 1;

        std::vector<uint8_t> scratch;
        auto has = [&](const std::string &r) { return w4map.count(r) || f16map.count(r); };
        // WTE（host）
        {
            auto it = w4map.find("T.WTE");
            if (it == w4map.end()) { printf("[md3] no T.WTE\n"); return 1; }
            std::stringstream ss(it->second);
            std::string pre; int N, K; ss >> pre >> N >> K;
            wte_N = N; wte_K = K; wte_sbp = ((K / 32) + 15) / 16 * 16;
            if (loadHostWte(pre)) return 1;
            printf("[md3] wte host loaded %.1f MB\n", (hwte_w.size() + hwte_s.size() + hwte_sz.size()) / 1e6);
        }
        auto loadW4Role = [&](const std::string &role, W4 &w) -> int {
            auto it = w4map.find(role);
            if (it == w4map.end()) { printf("[md3] missing %s\n", role.c_str()); return 1; }
            std::stringstream ss(it->second);
            std::string pre; int N, K; ss >> pre >> N >> K;
            return loadW4(dir, pre, N, K, w, scratch);
        };
        auto loadF16Role = [&](const std::string &role, F16 &f, int64_t n) -> int {
            auto it = f16map.find(role);
            if (it == f16map.end()) { printf("[md3] missing %s\n", role.c_str()); return 1; }
            std::stringstream ss(it->second);
            std::string file; ss >> file;
            return loadF16(dir, file, n, f, scratch);
        };
        double t0 = now();
        layers.resize(n_layers);
        for (int i = 0; i < n_layers; ++i) {
            Layer &L = layers[i];
            std::string pre = "T.L" + std::to_string(i);
            if (loadF16Role(pre + ".LN.W", L.lnw, cfg.hidden)) return 1;
            if (loadF16Role(pre + ".LN.B", L.lnb, cfg.hidden)) return 1;
            if (loadW4Role(pre + ".QKV", L.qkv)) return 1;
            if (loadF16Role(pre + ".QKV.B", L.qkvb, 3 * cfg.hidden)) return 1;
            if (loadW4Role(pre + ".O", L.o)) return 1;
            if (loadF16Role(pre + ".O.B", L.ob, cfg.hidden)) return 1;
            if (loadF16Role(pre + ".TAU.WQ", L.tauwq, 32 * 3 * cfg.hidden)) return 1;
            if (loadF16Role(pre + ".TAU.WV", L.tauwv, 32 * 3 * cfg.hidden)) return 1;
            if (loadF16Role(pre + ".TAU.A", L.taua, 32)) return 1;
            {
                std::vector<uint16_t> ha(32);
                if (download(ha.data(), L.taua.dev, 64)) return 1;
                for (int h = 0; h < 32; ++h) {
                    uint16_t hv = ha[h]; unsigned sign = (unsigned)(hv & 0x8000u) << 16;
                    unsigned e = (hv >> 10) & 0x1F; unsigned m = hv & 0x3FF; unsigned f;
                    if (e == 0) f = sign; else if (e == 31) f = sign | 0x7F800000u | (m << 13);
                    else f = sign | ((e - 15 + 127) << 23) | (m << 13);
                    float r; memcpy(&r, &f, 4); L.alpha[h] = r;
                }
            }
            L.moe = (i >= cfg.moe_start);
            if (!L.moe) {
                if (loadW4Role(pre + ".FC1", L.fc1)) return 1;
                if (loadF16Role(pre + ".FC1.B", L.fc1b, cfg.dense_ff)) return 1;
                if (loadW4Role(pre + ".FC2", L.fc2)) return 1;
                if (loadF16Role(pre + ".FC2.B", L.fc2b, cfg.hidden)) return 1;
            } else {
                if (loadW4Role(pre + ".ROUTER", L.router)) return 1;
                if (loadF16Role(pre + ".ROUTER.B", L.routerb, cfg.n_experts)) return 1;
                L.efc1.resize(cfg.n_experts); L.efc2.resize(cfg.n_experts);
                for (int e = 0; e < cfg.n_experts; ++e) {
                    if (loadW4Role(pre + ".E" + std::to_string(e) + ".FC1", L.efc1[e])) return 1;
                    if (loadW4Role(pre + ".E" + std::to_string(e) + ".FC2", L.efc2[e])) return 1;
                }
            }
            if ((i % 4) == 3) {
                printf("[md3] load layer %d/%d  %.1fs\n", i + 1, n_layers, now() - t0);
                fflush(stdout);
            }
        }
        // LM head / final norm
        printf("[md3] load LM...\n"); fflush(stdout);
        Layer dummy;
        if (loadW4Role("T.LM", dummy.o)) return 1;
        printf("[md3] load LM.B...\n"); fflush(stdout);
        if (loadF16Role("T.LM.B", dummy.lnb, cfg.vocab)) return 1;
        printf("[md3] load FN...\n"); fflush(stdout);
        if (loadF16Role("T.FN.W", dummy.lnw, cfg.hidden)) return 1;
        if (loadF16Role("T.FN.B", dummy.fc1b, cfg.hidden)) return 1;
        printf("[md3] setup lm...\n"); fflush(stdout);
        lmW4 = dummy.o; lmB = dummy.lnb; fnW = dummy.lnw; fnB = dummy.fc1b;
        printf("[md3] weights loaded in %.1fs\n", now() - t0);
        // w4a16_gemv_gen 已知"进程内首次调用结果异常"，这里预热一次丢弃
        if (d_zero && lmW4.w) {
            w4a16_gemv_gen_do(1, g_stream, d_zero, lmW4.w, lmW4.s, lmW4.sz, d_logits, cfg.hidden, cfg.vocab, 99);
            fp16_gemv_do(1, g_stream, d_zero, layers[0].tauwq.dev, d_taq, 3 * cfg.hidden, 32);
            if (sync()) return 1;
        }
        if (sync()) return 1;
        return 0;
    }

    W4 lmW4; F16 lmB, fnW, fnB;

    int forwardOne(int tok, int pos, std::vector<float> &logits) {
        const int H = cfg.hidden;
        bool doDump = false;
        int dumpPos = 0;
        if (getenv("MD3_DUMP")) { doDump = true; dumpPos = getenv("MD3_DUMPPOS") ? atoi(getenv("MD3_DUMPPOS")) : 0; }
        auto dumpDev = [&](const char *name, uint8_t *ptr, size_t nbytes) {
            if (!doDump || pos != dumpPos) return;
            std::vector<uint8_t> b(nbytes);
            sync(); download(b.data(), ptr, nbytes);
            std::string path = std::string("/tmp/md3_") + name + ".bin";
            FILE *f = fopen(path.c_str(), "wb");
            if (f) { fwrite(b.data(), 1, nbytes, f); fclose(f); }
        };
        if (embedRow(tok, d_x)) return 1;
        dumpDev("emb", d_x, H * 2);
        for (int li = 0; li < n_layers; ++li) {
            Layer &L = layers[li];
            layernorm_f16_do(1, g_stream, d_x, L.lnw.dev, L.lnb.dev, d_xn, H, 1, cfg.eps);
            { char nm[32]; snprintf(nm, sizeof(nm), "L%d_xn", li); dumpDev(nm, d_xn, H * 2); }
            w4a16_gemv_gen_do(1, g_stream, d_xn, L.qkv.w, L.qkv.s, L.qkv.sz, d_qkv, H, 3 * H, 99);
            add_bias_f16_do(1, g_stream, d_qkv, L.qkvb.dev, 1, 3 * H);
            { char nm[32]; snprintf(nm, sizeof(nm), "L%d_qkv", li); dumpDev(nm, d_qkv, 3 * H * 2); }
            gelu_f16_do(1, g_stream, d_qkv, d_gelu, 3 * H);
            fp16_gemv_do(1, g_stream, d_gelu, L.tauwq.dev, d_taq, 3 * H, 32);
            fp16_gemv_do(1, g_stream, d_gelu, L.tauwv.dev, d_tav, 3 * H, 32);
            if (L.moe) w4a16_gemv_gen_do(1, g_stream, d_xn, L.router.w, L.router.s, L.router.sz,
                                         d_router, H, cfg.n_experts, 99);
            if (sync()) return 1;
            uint16_t hq[32], hv[32], hrt[64];
            if (download(hq, d_taq, 64)) return 1;
            if (download(hv, d_tav, 64)) return 1;
            float tauq[32], tauv[32];
            double lp = log((double)pos + 1.0);
            for (int h = 0; h < 32; ++h) {
                float aq = h2f_host(hq[h]), av = h2f_host(hv[h]);
                float tp = 0.5f + 1.0f / (1.0f + expf(-L.alpha[h] * (float)lp));
                tauq[h] = tanhf(aq) + tp;
                tauv[h] = tanhf(av) + tp;
            }
            float cosv[16], sinv[16];
            for (int i = 0; i < 16; ++i) {
                double freq = 1.0 / pow((double)cfg.theta, (double)i / 16.0);
                cosv[i] = (float)cos((double)pos * freq);
                sinv[i] = (float)sin((double)pos * freq);
            }
            if (getenv("MD3_DEBUG")) {
                printf("[dbg] alpha[0..2]=%.4f %.4f %.4f rawq=%.4f rawv=%.4f tauq=%.4f tauv=%.4f\n",
                       L.alpha[0], L.alpha[1], L.alpha[2], h2f_host(hq[0]), h2f_host(hv[0]), tauq[0], tauv[0]);
                fflush(stdout);
            }
            if (upload(d_tauq, tauq, 32 * 4)) return 1;
            if (upload(d_tauv, tauv, 32 * 4)) return 1;
            { char nm[32]; snprintf(nm, sizeof(nm), "L%d_tauq", li); dumpDev(nm, d_tauq, 128); }
            { char nm[32]; snprintf(nm, sizeof(nm), "L%d_tauv", li); dumpDev(nm, d_tauv, 128); }
            if (upload(d_cos, cosv, 16 * 4)) return 1;
            if (upload(d_sin, sinv, 16 * 4)) return 1;
            attn_decode_text_do(1, g_stream, d_qkv, d_tauq, d_tauv, d_cos, d_sin, kc[li], vc[li],
                                d_attn, pos);
            { char nm[32]; snprintf(nm, sizeof(nm), "L%d_attn", li); dumpDev(nm, d_attn, H * 2); }
            w4a16_gemv_gen_do(1, g_stream, d_attn, L.o.w, L.o.s, L.o.sz, d_proj, H, H, 99);
            add_bias_f16_do(1, g_stream, d_proj, L.ob.dev, 1, H);
            { char nm[32]; snprintf(nm, sizeof(nm), "L%d_proj", li); dumpDev(nm, d_proj, H * 2); }
            vec_axpy_do(1, g_stream, d_x, d_proj, d_x, 1.0f, H);
            if (!L.moe) {
                w4a16_gemv_gen_do(1, g_stream, d_xn, L.fc1.w, L.fc1.s, L.fc1.sz, d_fc1, H, cfg.dense_ff, 99);
                add_bias_f16_do(1, g_stream, d_fc1, L.fc1b.dev, 1, cfg.dense_ff);
                { char nm[32]; snprintf(nm, sizeof(nm), "L%d_fc1pre", li); dumpDev(nm, d_fc1, cfg.dense_ff * 2); }
                gelu_f16_do(1, g_stream, d_fc1, d_fc1, cfg.dense_ff);
                { char nm[32]; snprintf(nm, sizeof(nm), "L%d_h", li); dumpDev(nm, d_fc1, cfg.dense_ff * 2); }
                w4a16_gemv_gen_do(1, g_stream, d_fc1, L.fc2.w, L.fc2.s, L.fc2.sz, d_mlp,
                                  cfg.dense_ff, H, 99);
                add_bias_f16_do(1, g_stream, d_mlp, L.fc2b.dev, 1, H);
                { char nm[32]; snprintf(nm, sizeof(nm), "L%d_mlp", li); dumpDev(nm, d_mlp, H * 2); }
            } else {
                int idx[8]; float wt[8];
                if (download(hrt, d_router, cfg.n_experts * 2)) return 1;
                topk8(hrt, cfg.n_experts, idx, wt, cfg.topk);
                for (int k = 0; k < cfg.topk; ++k) {
                    W4 &e1 = L.efc1[idx[k]], &e2 = L.efc2[idx[k]];
                    w4a16_gemv_gen_do(1, g_stream, d_xn, e1.w, e1.s, e1.sz, d_e1, H, 2048, 99);
                    geglu_f16_do(1, g_stream, d_e1, d_e1 + 1024 * 2, d_e1, 1024);
                    w4a16_gemv_gen_do(1, g_stream, d_e1, e2.w, e2.s, e2.sz, d_e2, 1024, 2048, 99);
                    if (k == 0) vec_axpy_do(1, g_stream, d_zero, d_e2, d_mlp, wt[k], H);
                    else vec_axpy_do(1, g_stream, d_mlp, d_e2, d_mlp, wt[k], H);
                }
            }
            vec_axpy_do(1, g_stream, d_x, d_mlp, d_x, 1.0f, H);
            { char nm[32]; snprintf(nm, sizeof(nm), "L%d_out", li); dumpDev(nm, d_x, H * 2); }
        }
        layernorm_f16_do(1, g_stream, d_x, fnW.dev, fnB.dev, d_xn, H, 1, cfg.eps);
        w4a16_gemv_gen_do(1, g_stream, d_xn, lmW4.w, lmW4.s, lmW4.sz, d_logits, H, cfg.vocab, 99);
        if (sync()) return 1;
        std::vector<uint16_t> hl(cfg.vocab);
        if (download(hl.data(), d_logits, cfg.vocab * 2)) return 1;
        logits.resize(cfg.vocab);
        for (int i = 0; i < cfg.vocab; ++i) logits[i] = h2f_host(hl[i]);
        return 0;
    }

    static float h2f_host(uint16_t hv) {
        unsigned sign = (unsigned)(hv & 0x8000u) << 16;
        unsigned e = (hv >> 10) & 0x1F, m = hv & 0x3FF, f;
        if (e == 0) f = sign; else if (e == 31) f = sign | 0x7F800000u | (m << 13);
        else f = sign | ((e - 15 + 127) << 23) | (m << 13);
        float r; memcpy(&r, &f, 4); return r;
    }
    static void topk8(const uint16_t *logits, int n, int *idx, float *wt, int k) {
        std::vector<std::pair<float, int>> v(n);
        for (int i = 0; i < n; ++i) v[i] = {h2f_host(logits[i]), i};
        std::partial_sort(v.begin(), v.begin() + k, v.end(),
                          [](const std::pair<float, int> &a, const std::pair<float, int> &b) {
                              return a.first > b.first; });
        float mx = v[0].first, sum = 0;
        float e[64];
        for (int i = 0; i < k; ++i) { e[i] = expf(v[i].first - mx); sum += e[i]; }
        for (int i = 0; i < k; ++i) { idx[i] = v[i].second; wt[i] = e[i] / sum; }
    }


    // ---------------- 文本 prefill（ACLNN MatMul） ----------------

    // ---------------- 通用 prefill（可直接给 embeds，支持 pos_start / prefix mask） ----------------
    // prefix_len: 前缀 token 数（全连接块）；plain text 时 0（纯 causal）
    // want_logits=false 时只填 KV cache（用于 image prefix）
    int prefillEmbeds(const std::vector<uint16_t> &emb, int T, int pos_start, int prefix_len,
                      std::vector<float> *logitsOut, std::vector<std::vector<float>> *allLogits = nullptr) {
        const int H = cfg.hidden;
        if (T <= 0 || pos_start + T > cfg.max_context) { printf("[md3] bad prefill T=%d pos=%d\n", T, pos_start); return 1; }
        const int Tk = pos_start + T;
        std::vector<void *> tmp;
        auto TA = [&](size_t n) -> uint8_t * { void *p = devAlloc(n); if (p) tmp.push_back(p); return (uint8_t *)p; };
        size_t hb = (size_t)T * H * 2;
        uint8_t *p_x = TA(hb), *p_xn = TA(hb), *p_proj = TA(hb), *p_mlp = TA(hb);
        uint8_t *p_qkv = TA((size_t)T * 3 * H * 2), *p_gelu = TA((size_t)T * 3 * H * 2);
        uint8_t *p_tqraw = TA((size_t)T * 64), *p_tvraw = TA((size_t)T * 64);
        uint8_t *p_tauq = TA((size_t)T * 32 * 4), *p_tauv = TA((size_t)T * 32 * 4);
        uint8_t *p_cos = TA((size_t)T * 16 * 4), *p_sin = TA((size_t)T * 16 * 4);
        uint8_t *p_q = TA((size_t)32 * T * 64 * 2), *p_k = TA((size_t)32 * T * 64 * 2),
                *p_v = TA((size_t)32 * T * 64 * 2), *p_attn = TA((size_t)32 * T * 64 * 2);
        size_t scsz = (size_t)32 * T * Tk * 2;
        uint8_t *p_sc = TA(scsz), *p_sc2 = TA(scsz);
        uint8_t *p_mask = TA((size_t)T * Tk * 2);
        uint8_t *p_fc1 = TA((size_t)T * cfg.dense_ff * 2), *p_router = TA((size_t)T * 64 * 2);
        uint8_t *p_w = TA(40 * 1024 * 1024);
        uint8_t *p_gather = TA(hb), *p_ge = TA((size_t)T * cfg.inner * 2), *p_e2 = TA(hb);
        uint8_t *p_last = TA(H * 2);
        if (!p_x || !p_w || !p_qkv || !p_sc || !p_gather) { printf("[md3] prefill alloc fail\n"); return 1; }
        auto dumpP = [&](const char *nm, uint8_t *ptr, size_t nb) {
            if (!getenv("MD3_DUMP")) return;
            std::vector<uint8_t> b(nb); sync(); download(b.data(), ptr, nb);
            FILE *f = fopen((std::string("/tmp/") + nm + ".bin").c_str(), "wb");
            if (f) { fwrite(b.data(), 1, nb, f); fclose(f); }
        };
        if (upload(p_x, emb.data(), hb)) return 1;
        // mask
        std::vector<uint16_t> hmask((size_t)T * Tk);
        for (int i = 0; i < T; ++i)
            for (int j = 0; j < Tk; ++j)
                hmask[(size_t)i * Tk + j] = (j < prefix_len || j <= pos_start + i) ? 0 : 0xFBFF;
        if (upload(p_mask, hmask.data(), (size_t)T * Tk * 2)) return 1;
        std::vector<float> hcos((size_t)T * 16), hsin((size_t)T * 16);
        for (int t = 0; t < T; ++t) {
            int pos = pos_start + t;
            for (int i = 0; i < 16; ++i) {
                double freq = 1.0 / pow((double)cfg.theta, (double)i / 16.0);
                hcos[(size_t)t * 16 + i] = (float)cos((double)pos * freq);
                hsin[(size_t)t * 16 + i] = (float)sin((double)pos * freq);
            }
        }
        std::vector<uint16_t> hrawq((size_t)T * 32), hrawv((size_t)T * 32), hrouter((size_t)T * 64);
        std::vector<float> htauq((size_t)T * 32), htauv((size_t)T * 32);
        std::vector<uint16_t> hxn(hb);
        double t0 = now();
        for (int li = 0; li < n_layers; ++li) {
            Layer &L = layers[li];
            layernorm_f16_do(1, g_stream, p_x, L.lnw.dev, L.lnb.dev, p_xn, H, T, cfg.eps);
            dequant_w4_f16_do(1, g_stream, L.qkv.w, L.qkv.s, L.qkv.sz, p_w, L.qkv.K, L.qkv.N);
            matmulTN(T, H, 3 * H, p_xn, p_w, p_qkv);
            add_bias_f16_do(1, g_stream, p_qkv, L.qkvb.dev, T, 3 * H);
            gelu_f16_do(1, g_stream, p_qkv, p_gelu, T * 3 * H);
            matmulTN(T, 3 * H, 32, p_gelu, L.tauwq.dev, p_tqraw);
            matmulTN(T, 3 * H, 32, p_gelu, L.tauwv.dev, p_tvraw);
            if (L.moe) {
                dequant_w4_f16_do(1, g_stream, L.router.w, L.router.s, L.router.sz, p_w, L.router.K, L.router.N);
                matmulTN(T, H, cfg.n_experts, p_xn, p_w, p_router);
            }
            if (getenv("MD3_TRACE_PRE") && li < 3) { printf("[pre] L%d sync1\n", li); fflush(stdout); }
            if (sync()) return 1;
            if (download(hrawq.data(), p_tqraw, (size_t)T * 64)) return 1;
            if (download(hrawv.data(), p_tvraw, (size_t)T * 64)) return 1;
            if (L.moe && download(hrouter.data(), p_router, (size_t)T * cfg.n_experts * 2)) return 1;
            for (int t = 0; t < T; ++t) {
                int pos = pos_start + t;
                double lp = log((double)pos + 1.0);
                for (int h = 0; h < 32; ++h) {
                    float tp = 0.5f + 1.0f / (1.0f + expf(-L.alpha[h] * (float)lp));
                    htauq[(size_t)t * 32 + h] = tanhf(h2f_host(hrawq[(size_t)t * 32 + h])) + tp;
                    htauv[(size_t)t * 32 + h] = tanhf(h2f_host(hrawv[(size_t)t * 32 + h])) + tp;
                }
            }
            if (upload(p_tauq, htauq.data(), (size_t)T * 32 * 4)) return 1;
            if (upload(p_tauv, htauv.data(), (size_t)T * 32 * 4)) return 1;
            if (upload(p_cos, hcos.data(), (size_t)T * 16 * 4)) return 1;
            if (upload(p_sin, hsin.data(), (size_t)T * 16 * 4)) return 1;
            qkv_post_text_do(1, g_stream, p_qkv, p_tauq, p_tauv, p_cos, p_sin, p_q, p_k, p_v, T);
            if (getenv("MD3_TRACE_PRE") && li < 3) { printf("[pre] L%d sync2\n", li); fflush(stdout); }
            if (sync()) return 1;
            for (int h = 0; h < 32; ++h) {
                d2d(kc[li] + ((size_t)h * cfg.max_context + pos_start) * 64 * 2, p_k + (size_t)h * T * 64 * 2, (size_t)T * 64 * 2);
                d2d(vc[li] + ((size_t)h * cfg.max_context + pos_start) * 64 * 2, p_v + (size_t)h * T * 64 * 2, (size_t)T * 64 * 2);
            }
            if (T <= 64) {
                // 短 prefill（question）：attention 用 CPU（避免 ACLNN 小 M/大 mask 边界问题）
                std::vector<uint16_t> hq((size_t)32 * T * 64), hk((size_t)32 * T * 64), hv((size_t)32 * T * 64);
                std::vector<uint16_t> hkc((size_t)32 * Tk * 64), hvc((size_t)32 * Tk * 64);
                if (sync()) return 1;
                if (download(hq.data(), p_q, (size_t)32 * T * 64 * 2)) return 1;
                if (download(hk.data(), p_k, (size_t)32 * T * 64 * 2)) return 1;
                if (download(hv.data(), p_v, (size_t)32 * T * 64 * 2)) return 1;
                for (int h = 0; h < 32; ++h) {
                    if (download(&hkc[(size_t)h * Tk * 64], kc[li] + (size_t)h * cfg.max_context * 64 * 2, (size_t)Tk * 64 * 2)) return 1;
                    if (download(&hvc[(size_t)h * Tk * 64], vc[li] + (size_t)h * cfg.max_context * 64 * 2, (size_t)Tk * 64 * 2)) return 1;
                }
                std::vector<uint16_t> hout((size_t)32 * T * 64);
                std::vector<float> sc((size_t)Tk);
                for (int h = 0; h < 32; ++h) {
                    const uint16_t *q = &hq[(size_t)h * T * 64];
                    const uint16_t *k = &hkc[(size_t)h * Tk * 64];
                    const uint16_t *v = &hvc[(size_t)h * Tk * 64];
                    for (int ti = 0; ti < T; ++ti) {
                        const uint16_t *qr = &q[(size_t)ti * 64];
                        float mx = -1e30f;
                        for (int j = 0; j < Tk; ++j) {
                            const uint16_t *kr = &k[(size_t)j * 64];
                            float acc = 0.0f;
                            for (int d = 0; d < 64; ++d) acc += h2f_quick(qr[d]) * h2f_quick(kr[d]);
                            acc += (hmask[(size_t)ti * Tk + j] == 0) ? 0.0f : -65504.0f;
                            sc[j] = acc;
                            if (acc > mx) mx = acc;
                        }
                        float sum = 0.0f;
                        for (int j = 0; j < Tk; ++j) { float ex = expf(sc[j] - mx); sc[j] = ex; sum += ex; }
                        float inv = 1.0f / sum;
                        uint16_t *orow = &hout[((size_t)h * T + ti) * 64];
                        for (int d = 0; d < 64; ++d) {
                            float acc = 0.0f;
                            for (int j = 0; j < Tk; ++j) acc += sc[j] * inv * h2f_quick(v[(size_t)j * 64 + d]);
                            orow[d] = f2h_host(acc);
                        }
                    }
                }
                if (upload(p_attn, hout.data(), (size_t)32 * T * 64 * 2)) return 1;
            } else {
                for (int h = 0; h < 32; ++h) {
                    if (matmulTN(T, 64, Tk, p_q + (size_t)h * T * 64 * 2, kc[li] + (size_t)h * cfg.max_context * 64 * 2,
                                 p_sc + (size_t)h * T * Tk * 2)) return 1;
                }
                if (addBroadcast(32, T, Tk, p_sc, p_mask, p_sc2)) return 1;
                if (softmax3(32, T, Tk, p_sc2, p_sc)) return 1;
                for (int h = 0; h < 32; ++h) {
                    if (matmulNN(T, Tk, 64, p_sc + (size_t)h * T * Tk * 2, vc[li] + (size_t)h * cfg.max_context * 64 * 2,
                                 p_attn + (size_t)h * T * 64 * 2)) return 1;
                }
            }
            heads_to_seq_do(1, g_stream, p_attn, p_proj, T, 32, 64);
            dequant_w4_f16_do(1, g_stream, L.o.w, L.o.s, L.o.sz, p_w, L.o.K, L.o.N);
            matmulTN(T, H, H, p_proj, p_w, p_attn);
            add_bias_f16_do(1, g_stream, p_attn, L.ob.dev, T, H);
            add_f16_do(1, g_stream, p_x, p_attn, p_x, T * H);
            if (!L.moe) {
                dequant_w4_f16_do(1, g_stream, L.fc1.w, L.fc1.s, L.fc1.sz, p_w, L.fc1.K, L.fc1.N);
                matmulTN(T, H, cfg.dense_ff, p_xn, p_w, p_fc1);
                add_bias_f16_do(1, g_stream, p_fc1, L.fc1b.dev, T, cfg.dense_ff);
                gelu_f16_do(1, g_stream, p_fc1, p_fc1, T * cfg.dense_ff);
                dequant_w4_f16_do(1, g_stream, L.fc2.w, L.fc2.s, L.fc2.sz, p_w, L.fc2.K, L.fc2.N);
                matmulTN(T, cfg.dense_ff, H, p_fc1, p_w, p_attn);
                add_bias_f16_do(1, g_stream, p_attn, L.fc2b.dev, T, H);
                add_f16_do(1, g_stream, p_x, p_attn, p_x, T * H);
            } else {
                if (download(hxn.data(), p_xn, hb)) return 1;
                std::vector<std::vector<int>> groups(cfg.n_experts);
                for (int t = 0; t < T; ++t) {
                    int idx[8]; float wt[8];
                    topk8(&hrouter[(size_t)t * cfg.n_experts], cfg.n_experts, idx, wt, cfg.topk);
                    for (int k = 0; k < cfg.topk; ++k) groups[idx[k]].push_back(t);
                }
                std::vector<float> mout((size_t)T * H, 0.0f);
                for (int ei = 0; ei < cfg.n_experts; ++ei) {
                    if (groups[ei].empty()) continue;
                    int Te = (int)groups[ei].size();
                    std::vector<uint16_t> gath((size_t)Te * H);
                    for (int i = 0; i < Te; ++i) memcpy(&gath[(size_t)i * H], &hxn[(size_t)groups[ei][i] * H], (size_t)H * 2);
                    if (upload(p_gather, gath.data(), (size_t)Te * H * 2)) return 1;
                    W4 &w1 = L.efc1[ei], &w2 = L.efc2[ei];
                    dequant_w4_f16_do(1, g_stream, w1.w, w1.s, w1.sz, p_w, w1.K, w1.N);
                    matmulTN(Te, H, 2048, p_gather, p_w, p_fc1);
                    geglu_batch_f16_do(1, g_stream, p_fc1, p_ge, Te, cfg.inner);
                    dequant_w4_f16_do(1, g_stream, w2.w, w2.s, w2.sz, p_w, w2.K, w2.N);
                    matmulTN(Te, cfg.inner, H, p_ge, p_w, p_e2);
                    if (sync()) return 1;
                    std::vector<uint16_t> he2((size_t)Te * H);
                    if (download(he2.data(), p_e2, (size_t)Te * H * 2)) return 1;
                    for (int i = 0; i < Te; ++i) {
                        int t = groups[ei][i];
                        int idx2[8]; float wt2[8]; float w = 0.0f;
                        topk8(&hrouter[(size_t)t * cfg.n_experts], cfg.n_experts, idx2, wt2, cfg.topk);
                        for (int k = 0; k < cfg.topk; ++k) if (idx2[k] == ei) w = wt2[k];
                        for (int d = 0; d < H; ++d) mout[(size_t)t * H + d] += w * h2f_host(he2[(size_t)i * H + d]);
                    }
                }
                std::vector<uint16_t> hmlp((size_t)T * H);
                for (size_t i = 0; i < hmlp.size(); ++i) { float v = mout[i]; hmlp[i] = f2h_host(v); }
                if (upload(p_mlp, hmlp.data(), hb)) return 1;
                add_f16_do(1, g_stream, p_x, p_mlp, p_x, T * H);
            }
        }
        if (logitsOut || allLogits) {
            if (allLogits) allLogits->resize(T);
            int nOut = allLogits ? T : 1;
            for (int i = 0; i < nOut; ++i) {
                int row = allLogits ? i : (T - 1);
                if (d2d(p_last, p_x + (size_t)row * H * 2, H * 2)) return 1;
                layernorm_f16_do(1, g_stream, p_last, fnW.dev, fnB.dev, d_xn, H, 1, cfg.eps);
                w4a16_gemv_gen_do(1, g_stream, d_xn, lmW4.w, lmW4.s, lmW4.sz, d_logits, H, cfg.vocab, 99);
                if (sync()) return 1;
                std::vector<uint16_t> hl(cfg.vocab);
                if (download(hl.data(), d_logits, (size_t)cfg.vocab * 2)) return 1;
                std::vector<float> lg(cfg.vocab);
                for (int k = 0; k < cfg.vocab; ++k) lg[k] = h2f_host(hl[k]);
                if (allLogits) (*allLogits)[i] = std::move(lg); else *logitsOut = std::move(lg);
            }
        }
        if (sync()) return 1;   // 等所有异步 kernel 完成后再释放临时缓冲
        for (void *q : tmp) aclrtFree(q);
        if (getenv("MD3_TRACE_PREFILL")) printf("[md3] prefillEmbeds T=%d pos=%d done %.2fs\n", T, pos_start, now() - t0);
        return 0;
    }

    static uint16_t f2h_host(float f) {
        uint32_t u; memcpy(&u, &f, 4);
        uint32_t sign = (u >> 16) & 0x8000u; int ex = (int)((u >> 23) & 0xFF) - 127 + 15;
        uint32_t mant = u & 0x7FFFFFu;
        if (ex <= 0) return (uint16_t)sign;
        if (ex >= 31) return (uint16_t)(sign | 0x7C00u);
        return (uint16_t)(sign | (ex << 10) | (mant >> 13));
    }

    int prefillText(const std::vector<int> &ids, std::vector<float> &logits) {
        std::vector<uint16_t> emb((size_t)ids.size() * cfg.hidden);
        for (size_t t = 0; t < ids.size(); ++t) wteRowHost(ids[t], &emb[t * cfg.hidden]);
        return prefillEmbeds(emb, (int)ids.size(), 0, 0, &logits);
    }


    // ---------------- vision ----------------
    uint8_t *vpatchwPad = nullptr;
    std::vector<float> vpatchwHost;   // [1152,640] float

    int loadVision() {
        if (!vlayers.empty()) return 0;
        std::vector<uint8_t> scratch;
        auto R = [&](const std::string &role, W4 &w) -> int {
            auto it = w4map.find(role);
            if (it == w4map.end()) { printf("[md3] missing %s\n", role.c_str()); return 1; }
            std::stringstream ss(it->second); std::string pre; int N, K; ss >> pre >> N >> K;
            return loadW4(dir, pre, N, K, w, scratch);
        };
        auto F = [&](const std::string &role, F16 &f, int64_t n) -> int {
            auto it = f16map.find(role);
            if (it == f16map.end()) { printf("[md3] missing %s\n", role.c_str()); return 1; }
            std::stringstream ss(it->second); std::string file; ss >> file;
            return loadF16(dir, file, n, f, scratch);
        };
        const int VD = 1152, VFF = 4304, VT = 729, VFFPAD = 4352;
        auto FH = [&](const std::string &fname, F16 &f, int64_t n) -> int {
            f.dev = (uint8_t *)devAlloc((size_t)n * 2);
            if (!f.dev) { printf("[md3] vision f16 alloc fail %s\n", fname.c_str()); return 1; }
            std::vector<uint8_t> buf((size_t)n * 2);
            if (!readFile(dir + "/vision/" + fname, buf, (size_t)n * 2)) { printf("[md3] miss %s\n", fname.c_str()); return 1; }
            return upload(f.dev, buf.data(), (size_t)n * 2);
        };
        vlayers.resize(27);
        for (int i = 0; i < 27; ++i) {
            std::string pre = "V.L" + std::to_string(i);
            std::string fn = "L" + std::to_string(i);
            VLayer &L = vlayers[i];
            if (F(pre + ".LN1.W", L.ln1w, VD) || F(pre + ".LN1.B", L.ln1b, VD)) return 1;
            if (F(pre + ".LN2.W", L.ln2w, VD) || F(pre + ".LN2.B", L.ln2b, VD)) return 1;
            if (F(pre + ".QKV.B", L.qkvb, 3 * VD) || F(pre + ".O.B", L.ob, VD)) return 1;
            if (F(pre + ".FC1.B", L.fc1b, VFF) || F(pre + ".FC2.B", L.fc2b, VD)) return 1;
            if (FH(fn + ".qkv.f16", L.qkvW, (int64_t)3 * VD * VD)) return 1;
            if (FH(fn + ".o.f16", L.oW, (int64_t)VD * VD)) return 1;
            if (FH(fn + ".fc1.f16", L.fc1W, (int64_t)VFF * VD)) return 1;
            if (FH(fn + ".fc2.f16", L.fc2W, (int64_t)VD * VFFPAD)) return 1;
        }
        if (F("V.PATCH.B", vpatchb, VD) || F("V.POS", vpos, (int64_t)VT * VD)) return 1;
        if (F("V.FN.W", vfnw, VD) || F("V.FN.B", vfnb, VD)) return 1;
        if (F("V.PROJ1.B", vproj1b, 8192) || F("V.PROJ2.B", vproj2b, 2048)) return 1;
        if (FH("proj_fc1.f16", vproj1W, (int64_t)8192 * 2304)) return 1;
        if (FH("proj_fc2.f16", vproj2W, (int64_t)2048 * 8192)) return 1;
        auto it = f16map.find("V.PATCH.W");
        if (it == f16map.end()) { printf("[md3] missing V.PATCH.W\n"); return 1; }
        std::stringstream ss(it->second); std::string file; ss >> file;
        std::vector<uint8_t> raw((size_t)VD * 588 * 2);
        if (!readFile(dir + "/" + file, raw, raw.size())) return 1;
        std::vector<uint8_t> pad((size_t)VD * 640 * 2, 0);
        for (int r = 0; r < VD; ++r) memcpy(&pad[(size_t)r * 640 * 2], &raw[(size_t)r * 588 * 2], (size_t)588 * 2);
        vpatchwPad = (uint8_t *)devAlloc(pad.size());
        if (!vpatchwPad || upload(vpatchwPad, pad.data(), pad.size())) return 1;
        vpatchwHost.resize((size_t)VD * 640);
        for (int r = 0; r < VD; ++r)
            for (int c = 0; c < 640; ++c)
                vpatchwHost[(size_t)r * 640 + c] = h2f_host(((const uint16_t *)pad.data())[(size_t)r * 640 + c]);
        printf("[md3] vision weights loaded\n");
        return 0;
    }

    int encodeImage(const std::string &cropsPath, int nCrops, std::vector<uint16_t> &imgEmb) {
        const int VD = 1152, VH = 16, VHD = 72, VDP = 80, VFF = 4304, VFFPAD = 4352, VT = 729;
        if (nCrops != 2) { printf("[md3] only nCrops=2 supported now (got %d)\n", nCrops); return 1; }
        if (loadVision()) return 1;
        FILE *fp = fopen(cropsPath.c_str(), "rb");
        if (!fp) { printf("[md3] no crops %s\n", cropsPath.c_str()); return 1; }
        std::vector<uint16_t> crops((size_t)nCrops * 3 * 378 * 378);
        size_t got = fread(crops.data(), 2, crops.size(), fp);
        fclose(fp);
        if (got != crops.size()) { printf("[md3] crops short\n"); return 1; }
        const int np = nCrops * VT;
        const int npP = ((np + 15) / 16) * 16;
        std::vector<uint16_t> patches((size_t)npP * 640, 0);
        for (int c = 0; c < nCrops; ++c)
            for (int py = 0; py < 27; ++py)
                for (int px = 0; px < 27; ++px)
                    for (int ch = 0; ch < 3; ++ch)
                        for (int yy = 0; yy < 14; ++yy)
                            for (int xx = 0; xx < 14; ++xx) {
                                int pi = c * VT + py * 27 + px;
                                int dst = (ch * 14 + yy) * 14 + xx;
                                int src = ((c * 3 + ch) * 378 + py * 14 + yy) * 378 + px * 14 + xx;
                                patches[(size_t)pi * 640 + dst] = crops[src];
                            }
        std::vector<void *> tmp;
        auto TA = [&](size_t n) -> uint8_t * { void *q = devAlloc(n); if (q) tmp.push_back(q); return (uint8_t *)q; };
        uint8_t *p_patch = TA((size_t)npP * 640 * 2), *p_x = TA((size_t)npP * VD * 2), *p_xn = TA((size_t)npP * VD * 2);
        uint8_t *p_qkv = TA((size_t)npP * 3 * VD * 2), *p_attn = TA((size_t)npP * VD * 2), *p_proj = TA((size_t)npP * VD * 2);
        uint8_t *p_q = TA((size_t)nCrops * VH * VT * VDP * 2), *p_k = TA((size_t)nCrops * VH * VT * VDP * 2),
                *p_v = TA((size_t)nCrops * VH * VT * VDP * 2);
        uint8_t *p_seq = TA((size_t)npP * VD * 2), *p_fc1 = TA((size_t)npP * VFF * 2),
                *p_fc2in = TA((size_t)npP * VFFPAD * 2), *p_w = TA(40 * 1024 * 1024);
        uint8_t *p_img = TA((size_t)VT * 2304 * 2);
        if (!p_patch || !p_seq || !p_w || !p_img) { printf("[md3] vision alloc fail\n"); return 1; }
        // patch embedding 用 CPU 计算（1.07 GMAC，避免 ACLNN 首次调用异常）
        {
            std::vector<float> pf((size_t)npP * 640), wf = vpatchwHost;
            for (size_t i = 0; i < (size_t)npP * 640; ++i) pf[i] = h2f_host(patches[i]);
            std::vector<uint16_t> px((size_t)npP * VD);
            for (int r = 0; r < npP; ++r) {
                const float *pr = &pf[(size_t)r * 640];
                for (int n = 0; n < VD; ++n) {
                    const float *wr = &wf[(size_t)n * 640];
                    float acc = 0.0f;
                    for (int k = 0; k < 640; ++k) acc += pr[k] * wr[k];
                    px[(size_t)r * VD + n] = f2h_host(acc);
                }
            }
            // + patch bias + pos
            std::vector<uint16_t> hbias(VD), hpos((size_t)VT * VD);
            if (sync()) return 1;
            if (download(hbias.data(), vpatchb.dev, (size_t)VD * 2)) return 1;
            if (download(hpos.data(), vpos.dev, (size_t)VT * VD * 2)) return 1;
            for (int r = 0; r < npP; ++r) {
                int c = r / VT, rr = r % VT;
                if (c >= nCrops) continue;
                for (int n = 0; n < VD; ++n) {
                    float v = h2f_host(px[(size_t)r * VD + n]) + h2f_host(hbias[n]) + h2f_host(hpos[(size_t)rr * VD + n]);
                    px[(size_t)r * VD + n] = f2h_host(v);
                }
            }
            if (upload(p_x, px.data(), (size_t)npP * VD * 2)) return 1;
        }
        if (getenv("MD3_TRACE_V")) printf("[v] patch emb cpu ok\n");
        if (getenv("MD3_DUMP_PATCH")) {
            std::vector<uint16_t> hpe((size_t)npP * VD);
            if (sync()) return 1;
            if (download(hpe.data(), p_x, (size_t)npP * VD * 2)) return 1;
            FILE *f = fopen("/tmp/patch_emb.bin", "wb");
            if (f) { fwrite(hpe.data(), 2, hpe.size(), f); fclose(f); }
        }
        double t0 = now();
        const float qscale = 1.0f / sqrtf((float)VHD);
        for (int li = 0; li < 27; ++li) {
            VLayer &L = vlayers[li];
            layernorm_f16_do(1, g_stream, p_x, L.ln1w.dev, L.ln1b.dev, p_xn, VD, npP, 1e-5f);
            matmulTN(npP, VD, 3 * VD, p_xn, L.qkvW.dev, p_qkv);
            add_bias_f16_do(1, g_stream, p_qkv, L.qkvb.dev, npP, 3 * VD);
            if (li == 0 && getenv("MD3_DUMP_DETAIL")) {
                std::vector<uint16_t> t1((size_t)npP * VD), t2((size_t)npP * 3 * VD);
                if (sync()) return 1;
                if (download(t1.data(), p_xn, (size_t)npP * VD * 2)) return 1;
                FILE *f1 = fopen("/tmp/v_xn.bin", "wb"); if (f1) { fwrite(t1.data(), 2, t1.size(), f1); fclose(f1); }
                if (download(t2.data(), p_qkv, (size_t)npP * 3 * VD * 2)) return 1;
                FILE *f2 = fopen("/tmp/v_qkv.bin", "wb"); if (f2) { fwrite(t2.data(), 2, t2.size(), f2); fclose(f2); }
            }
            for (int c = 0; c < nCrops; ++c) {
                qkv_post_vision_do(1, g_stream, p_qkv + (size_t)c * VT * 3 * VD * 2,
                                   p_q + (size_t)c * VH * VT * VDP * 2, p_k + (size_t)c * VH * VT * VDP * 2,
                                   p_v + (size_t)c * VH * VT * VDP * 2, VT, VH, VHD, VDP, qscale);
            }
            {   // CPU attention（vision head_dim=72，ACLNN matmul 不支持非 32B 对齐）
                std::vector<uint16_t> hq((size_t)nCrops * VH * VT * VDP), hk((size_t)nCrops * VH * VT * VDP),
                    hv((size_t)nCrops * VH * VT * VDP);
                if (sync()) return 1;
                if (download(hq.data(), p_q, (size_t)nCrops * VH * VT * VDP * 2)) return 1;
                if (download(hk.data(), p_k, (size_t)nCrops * VH * VT * VDP * 2)) return 1;
                if (download(hv.data(), p_v, (size_t)nCrops * VH * VT * VDP * 2)) return 1;
                std::vector<uint16_t> hseq((size_t)nCrops * VT * VD);
                if (li == 0 && getenv("MD3_DUMP_QKV")) {
                    FILE *f = fopen("/tmp/hq.bin", "wb"); if (f) { fwrite(hq.data(), 2, hq.size(), f); fclose(f); }
                    f = fopen("/tmp/hk.bin", "wb"); if (f) { fwrite(hk.data(), 2, hk.size(), f); fclose(f); }
                    f = fopen("/tmp/hv.bin", "wb"); if (f) { fwrite(hv.data(), 2, hv.size(), f); fclose(f); }
                }
                std::vector<float> fq((size_t)nCrops * VH * VT * VDP), fk((size_t)nCrops * VH * VT * VDP),
                    fv((size_t)nCrops * VH * VT * VDP);
                for (size_t i = 0; i < fq.size(); ++i) { fq[i] = h2f_quick(hq[i]); fk[i] = h2f_quick(hk[i]); fv[i] = h2f_quick(hv[i]); }
                const int tasks = nCrops * VH;
                auto worker = [&](int tid, int nthreads) {
                    std::vector<float> sc(VT);
                    for (int task = tid; task < tasks; task += nthreads) {
                        int c = task / VH, h = task % VH;
                        const float *q = &fq[(size_t)task * VT * VDP];
                        const float *k = &fk[(size_t)task * VT * VDP];
                        const float *v = &fv[(size_t)task * VT * VDP];
                        for (int ti = 0; ti < VT; ++ti) {
                            const float *qr = &q[(size_t)ti * VDP];
                            float mx = -1e30f;
                            for (int j = 0; j < VT; ++j) {
                                const float *kr = &k[(size_t)j * VDP];
                                float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
                                for (int d = 0; d < VHD; d += 4) {
                                    acc0 += qr[d] * kr[d]; acc1 += qr[d+1] * kr[d+1];
                                    acc2 += qr[d+2] * kr[d+2]; acc3 += qr[d+3] * kr[d+3];
                                }
                                float acc = acc0 + acc1 + acc2 + acc3;
                                sc[j] = acc;
                                if (acc > mx) mx = acc;
                            }
                            float sum = 0.0f;
                            for (int j = 0; j < VT; ++j) { float e = expf(sc[j] - mx); sc[j] = e; sum += e; }
                            float inv = 1.0f / sum;
                            uint16_t *orow = &hseq[((size_t)c * VT + ti) * VD + (size_t)h * VHD];
                            for (int d = 0; d < VHD; ++d) {
                                float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
                                int j = 0;
                                for (; j + 3 < VT; j += 4) {
                                    acc0 += sc[j] * v[(size_t)j * VDP + d];
                                    acc1 += sc[j+1] * v[(size_t)(j+1) * VDP + d];
                                    acc2 += sc[j+2] * v[(size_t)(j+2) * VDP + d];
                                    acc3 += sc[j+3] * v[(size_t)(j+3) * VDP + d];
                                }
                                float acc = acc0 + acc1 + acc2 + acc3;
                                for (; j < VT; ++j) acc += sc[j] * v[(size_t)j * VDP + d];
                                orow[d] = f2h_host(acc * inv);
                            }
                        }
                    }
                };
                int nthreads = 4;
                std::vector<std::thread> ths;
                for (int i = 0; i < nthreads; ++i) ths.emplace_back(worker, i, nthreads);
                for (auto &th : ths) th.join();
                if (upload(p_seq, hseq.data(), (size_t)nCrops * VT * VD * 2)) return 1;
                if (li == 0 && getenv("MD3_DUMP_DETAIL")) {
                    FILE *f = fopen("/tmp/v_attn.bin", "wb");
                    if (f) { fwrite(hseq.data(), 2, hseq.size(), f); fclose(f); }
                }
                if (getenv("MD3_TRACE_V") && (li % 5) == 4) printf("[v] layer %d cpu-attn done %.1fs\n", li + 1, now() - t0);
            }
            matmulTN(npP, VD, VD, p_seq, L.oW.dev, p_proj);
            add_bias_f16_do(1, g_stream, p_proj, L.ob.dev, npP, VD);
            add_f16_do(1, g_stream, p_x, p_proj, p_x, npP * VD);
            layernorm_f16_do(1, g_stream, p_x, L.ln2w.dev, L.ln2b.dev, p_xn, VD, npP, 1e-5f);
            matmulTN(npP, VD, VFF, p_xn, L.fc1W.dev, p_fc1);
            add_bias_f16_do(1, g_stream, p_fc1, L.fc1b.dev, npP, VFF);
            gelu_pad_f16_do(1, g_stream, p_fc1, p_fc2in, npP, VFF, VFFPAD);
            matmulTN(npP, VFFPAD, VD, p_fc2in, L.fc2W.dev, p_proj);
            add_bias_f16_do(1, g_stream, p_proj, L.fc2b.dev, npP, VD);
            add_f16_do(1, g_stream, p_x, p_proj, p_x, npP * VD);
            if (li == 0 && getenv("MD3_DUMP_PATCH")) {
                std::vector<uint16_t> hx0((size_t)npP * VD);
                if (sync()) return 1;
                if (download(hx0.data(), p_x, (size_t)npP * VD * 2)) return 1;
                FILE *f = fopen("/tmp/vL0.bin", "wb");
                if (f) { fwrite(hx0.data(), 2, hx0.size(), f); fclose(f); }
            }
            if (getenv("MD3_VISION_LIMIT") && li + 1 >= atoi(getenv("MD3_VISION_LIMIT"))) break;
            if (getenv("MD3_TRACE") && (li % 5) == 4) printf("[md3] vision layer %d/27 %.1fs\n", li + 1, now() - t0);
        }
        layernorm_f16_do(1, g_stream, p_x, vfnw.dev, vfnb.dev, p_x, VD, npP, 1e-5f);
        if (getenv("MD3_DUMP_ENC")) {
            std::vector<uint16_t> henc((size_t)npP * VD);
            if (sync()) return 1;
            if (download(henc.data(), p_x, (size_t)npP * VD * 2)) return 1;
            FILE *fe = fopen("/tmp/enc.bin", "wb");
            if (fe) { fwrite(henc.data(), 2, henc.size(), fe); fclose(fe); printf("[md3] enc dumped\n"); }
        }
        std::vector<uint16_t> hx((size_t)npP * VD);
        if (sync()) return 1;
        if (download(hx.data(), p_x, (size_t)npP * VD * 2)) return 1;
        // reconstructed = local crop 特征（tiling=1x1，adaptive pool 27x27 恒等）
        std::vector<uint16_t> cat((size_t)VT * 2304);
        for (int r = 0; r < VT; ++r)
            for (int d = 0; d < VD; ++d) {
                cat[(size_t)r * 2304 + d] = hx[(size_t)r * VD + d];
                cat[(size_t)r * 2304 + VD + d] = hx[(size_t)(VT + r) * VD + d];
            }
        if (upload(p_img, cat.data(), (size_t)VT * 2304 * 2)) return 1;
        matmulTN(VT, 2304, 8192, p_img, vproj1W.dev, p_fc1);
        add_bias_f16_do(1, g_stream, p_fc1, vproj1b.dev, VT, 8192);
        gelu_f16_do(1, g_stream, p_fc1, p_fc1, VT * 8192);
        matmulTN(VT, 8192, 2048, p_fc1, vproj2W.dev, p_proj);
        add_bias_f16_do(1, g_stream, p_proj, vproj2b.dev, VT, 2048);
        imgEmb.resize((size_t)VT * 2048);
        if (sync()) return 1;
        if (download(imgEmb.data(), p_proj, (size_t)VT * 2048 * 2)) return 1;
        if (getenv("MD3_DUMP_IMGEMB")) {
            FILE *f = fopen("/tmp/imgemb.bin", "wb");
            if (f) { fwrite(imgEmb.data(), 2, imgEmb.size(), f); fclose(f); printf("[md3] imgemb dumped\n"); }
        }
        for (void *q : tmp) aclrtFree(q);
        printf("[md3] vision encode done %.2fs\n", now() - t0);
        if (getenv("MD3_VISION_ONLY")) return 0;
        return 0;
    }

    int runImageQA(const std::string &cropsPath, int nCrops, const std::vector<int> &qids, int gen,
                   std::vector<int> &out, std::vector<float> &logits) {
        std::vector<uint16_t> imgEmb;
        if (getenv("MD3_FAKE_IMG")) {
            imgEmb.assign((size_t)729 * cfg.hidden, 0);
            printf("[md3] FAKE image emb\n");
        } else if (encodeImage(cropsPath, nCrops, imgEmb)) return 1;
        // image prefix: BOS + 729 image emb, 全连接 mask
        std::vector<uint16_t> pref((size_t)730 * cfg.hidden);
        wteRowHost(0, &pref[0]);
        memcpy(&pref[(size_t)cfg.hidden], imgEmb.data(), (size_t)729 * cfg.hidden * 2);
        double t1 = now();
        if (!getenv("MD3_SKIP_PREFIX")) {
            if (prefillEmbeds(pref, 730, 0, 730, nullptr)) return 1;
            printf("[md3] image prefix prefill %.2fs\n", now() - t1);
        } else {
            printf("[md3] SKIP image prefix prefill\n");
        }
        // question
        std::vector<uint16_t> qemb((size_t)qids.size() * cfg.hidden);
        for (size_t t = 0; t < qids.size(); ++t) wteRowHost(qids[t], &qemb[t * cfg.hidden]);
        if (prefillEmbeds(qemb, (int)qids.size(), 730, 730, &logits)) return 1;
        int pos = 730 + (int)qids.size();
        for (int g = 0; g < gen; ++g) {
            int tok = sampleArgmax(logits);
            out.push_back(tok);
            printf("[md3] gen %d tok=%d\n", g, tok);
            fflush(stdout);
            if (g + 1 == gen) break;
            if (forwardOne(tok, pos, logits)) return 1;
            pos++;
        }
        return 0;
    }

    int sampleArgmax(const std::vector<float> &logits) {
        int best = 0; float bv = logits[0];
        for (int i = 1; i < (int)logits.size(); ++i) if (logits[i] > bv) { bv = logits[i]; best = i; }
        return best;
    }
};

int run_impl(int argc, char **argv) {
    std::string dir = "converted/moondream3_310b";
    std::vector<int> ids;
    int gen = 8, limit = 0, nCrops = 2, lookupN = 0, lookupK = 4;
    std::string imagePath; std::vector<int> qids;
    float temp = 0.0f;
    std::string dump;
    bool prefill = false;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--dir" && i + 1 < argc) dir = argv[++i];
        else if (a == "--ids" && i + 1 < argc) {
            std::string s = argv[++i]; size_t p = 0;
            while (p < s.size()) { size_t q = s.find(',', p); if (q == std::string::npos) q = s.size();
                ids.push_back(atoi(s.substr(p, q - p).c_str())); p = q + 1; }
        }
        else if (a == "--gen" && i + 1 < argc) gen = atoi(argv[++i]);
        else if (a == "--limit-layers" && i + 1 < argc) limit = atoi(argv[++i]);
        else if (a == "--temp" && i + 1 < argc) temp = (float)atof(argv[++i]);
        else if (a == "--dump" && i + 1 < argc) dump = argv[++i];
        else if (a == "--prefill") prefill = true;
        else if (a == "--lookup" && i + 1 < argc) lookupN = atoi(argv[++i]);
        else if (a == "--lookup-k" && i + 1 < argc) lookupK = atoi(argv[++i]);
        else if (a == "--image" && i + 1 < argc) imagePath = argv[++i];
        else if (a == "--ncrops" && i + 1 < argc) nCrops = atoi(argv[++i]);
        else if (a == "--qids" && i + 1 < argc) {
            std::string s = argv[++i]; size_t pp = 0;
            while (pp < s.size()) { size_t q = s.find(',', pp); if (q == std::string::npos) q = s.size();
                qids.push_back(atoi(s.substr(pp, q - pp).c_str())); pp = q + 1; }
        }
    }
    CK(aclInit(nullptr)); CK(aclrtSetDevice(0)); CK(aclrtCreateStream(&g_stream));
    printf("[md3] device init ok\n");
    Eng eng; eng.dir = dir; eng.limit_layers = limit;
    if (eng.loadAll()) return 1;
    if (!imagePath.empty()) {
        std::vector<int> out;
        std::vector<float> qlogits;
        if (eng.runImageQA(imagePath, nCrops, qids, gen, out, qlogits)) return 1;
        if (!dump.empty()) {
            FILE *f = fopen(dump.c_str(), "wb");
            if (f) { fwrite(qlogits.data(), 4, qlogits.size(), f); fclose(f); }
        }
        printf("TOKENS");
        for (int t : out) printf(" %d", t);
        printf("\n");
        CK(aclrtDestroyStream(g_stream)); CK(aclrtResetDevice(0)); CK(aclFinalize());
        return 0;
    }
    if (getenv("MD3_GEMVTEST")) {
        std::vector<uint16_t> hact(8192);
        FILE *fa = fopen("/tmp/h.bin", "rb");
        if (!fa) { printf("no /tmp/h.bin\n"); return 1; }
        fread(hact.data(), 2, 8192, fa); fclose(fa);
        uint8_t *dact = (uint8_t *)devAlloc(8192 * 2), *dout = (uint8_t *)devAlloc(2048 * 2);
        upload(dact, hact.data(), 8192 * 2);
        w4a16_gemv_gen_do(1, g_stream, dact, eng.layers[0].fc2.w, eng.layers[0].fc2.s,
                          eng.layers[0].fc2.sz, dout, 8192, 2048, 99);
        sync();
        std::vector<uint16_t> ho(2048);
        download(ho.data(), dout, 2048 * 2);
        FILE *fo = fopen("/tmp/gemv_out.bin", "wb");
        fwrite(ho.data(), 2, 2048, fo); fclose(fo);
        printf("[md3] gemvtest done\n");
        aclrtFree(dact); aclrtFree(dout);
        CK(aclrtDestroyStream(g_stream)); CK(aclrtResetDevice(0)); CK(aclFinalize());
        return 0;
    }
    if (ids.empty()) ids = {1, 2, 3};
    std::vector<float> logits;
    double t0 = now();
    int pos = (int)ids.size();
    if (prefill) {
        if (eng.prefillText(ids, logits)) return 1;
        printf("[md3] prompt prefill %zu tokens in %.2f s\n", ids.size(), now() - t0);
    } else {
        for (size_t i = 0; i < ids.size(); ++i) {
            if (eng.forwardOne(ids[i], (int)i, logits)) return 1;
        }
        printf("[md3] prompt %zu tokens in %.2f s (%.3f s/tok)\n", ids.size(), now() - t0,
               (now() - t0) / std::max<size_t>(1, ids.size()));
    }
    if (getenv("MD3_DUMP_KV")) {
        int ntok = std::min((int)ids.size(), 16);
        std::vector<uint16_t> bk((size_t)32 * ntok * 64), bv((size_t)32 * ntok * 64);
        for (int h = 0; h < 32; ++h) {
            download(&bk[(size_t)h * ntok * 64], eng.kc[0] + (size_t)h * eng.cfg.max_context * 64 * 2, (size_t)ntok * 64 * 2);
            download(&bv[(size_t)h * ntok * 64], eng.vc[0] + (size_t)h * eng.cfg.max_context * 64 * 2, (size_t)ntok * 64 * 2);
        }
        FILE *fk = fopen("/tmp/kv_k.bin", "wb"); if (fk) { fwrite(bk.data(), 2, bk.size(), fk); fclose(fk); }
        FILE *fv = fopen("/tmp/kv_v.bin", "wb"); if (fv) { fwrite(bv.data(), 2, bv.size(), fv); fclose(fv); }
        printf("[md3] kv dumped (%d tokens)\n", ntok);
    }
    if (!dump.empty()) {
        FILE *f = fopen(dump.c_str(), "wb");
        if (f) { fwrite(logits.data(), 4, logits.size(), f); fclose(f); }
        printf("[md3] logits dumped to %s\n", dump.c_str());
    }
    std::vector<int> out;
    double td = 0;
    std::vector<int> seq(ids.begin(), ids.end());
    int accTok = 0, draftTok = 0, rounds = 0, g = 0;
    while (g < gen) {
        std::vector<int> draft;
        if (lookupN > 0 && (int)seq.size() > lookupN) {
            int n = lookupN;
            for (int start = (int)seq.size() - n - 1; start >= 0; --start) {
                bool m = true;
                for (int i = 0; i < n; ++i)
                    if (seq[start + i] != seq[seq.size() - n + i]) { m = false; break; }
                if (!m) continue;
                for (int j = 0; j < lookupK; ++j) {
                    int idx = start + n + j;
                    if (idx >= (int)seq.size()) break;
                    draft.push_back(seq[idx]);
                }
                if (!draft.empty()) break;
            }
        }
        if (!draft.empty() && g + (int)draft.size() < gen) {
            rounds++;
            std::vector<uint16_t> demb((size_t)draft.size() * eng.cfg.hidden);
            for (size_t i = 0; i < draft.size(); ++i) eng.wteRowHost(draft[i], &demb[i * eng.cfg.hidden]);
            std::vector<std::vector<float>> allLog;
            double tv = now();
            if (eng.prefillEmbeds(demb, (int)draft.size(), pos, 0, nullptr, &allLog)) return 1;
            tv = now() - tv;
            int m = 0;
            if ((int)eng.sampleArgmax(logits) == draft[0]) {
                m = 1;
                for (size_t i = 1; i < draft.size(); ++i) {
                    if ((int)eng.sampleArgmax(allLog[i - 1]) == draft[i]) m++; else break;
                }
            }
            for (int i = 0; i < m; ++i) {
                out.push_back(draft[i]); seq.push_back(draft[i]);
                printf("[md3] gen %d tok=%d (draft)\n", g, draft[i]); fflush(stdout);
                g++;
            }
            int nextTok;
            if (m >= (int)draft.size()) nextTok = eng.sampleArgmax(allLog.back());
            else if (m == 0) nextTok = eng.sampleArgmax(logits);
            else nextTok = eng.sampleArgmax(allLog[m - 1]);
            out.push_back(nextTok); seq.push_back(nextTok);
            printf("[md3] gen %d tok=%d (verify %.2fs acc %d/%d)\n", g, nextTok, tv, m, (int)draft.size());
            fflush(stdout);
            g++; accTok += m; draftTok += (int)draft.size(); pos += m;
            if (g < gen) {
                double ts = now();
                if (eng.forwardOne(nextTok, pos, logits)) return 1;
                td += now() - ts;
                pos++;
            }
            continue;
        }
        int tok = eng.sampleArgmax(logits);
        out.push_back(tok); seq.push_back(tok);
        printf("[md3] gen %d tok=%d\n", g, tok);
        fflush(stdout);
        g++;
        if (g >= gen) break;
        double ts = now();
        if (eng.forwardOne(tok, pos, logits)) return 1;
        if (getenv("MD3_DUMP_STEP")) {
            char nm[64]; snprintf(nm, sizeof(nm), "/tmp/step_%d.bin", g);
            FILE *f = fopen(nm, "wb");
            if (f) { fwrite(logits.data(), 4, logits.size(), f); fclose(f); }
        }
        td += now() - ts;
        pos++;
    }
    if (draftTok > 0) printf("[md3] lookup rounds=%d accepted=%d/%d\n", rounds, accTok, draftTok);
    if (!out.empty()) {
        printf("TOKENS");
        for (int t : out) printf(" %d", t);
        printf("\n");
    }
    if (td > 0) printf("[md3] decode %.2f s/tok\n", td / std::max(1, (int)out.size() - 1));
    CK(aclrtDestroyStream(g_stream)); CK(aclrtResetDevice(0)); CK(aclFinalize());
    return 0;
}

}  // namespace md3

int run_engine(int argc, char **argv) { return md3::run_impl(argc, argv); }
