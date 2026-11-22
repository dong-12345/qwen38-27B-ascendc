/**
 * engine_27b.cpp — Qwen3.8-27B 自研推理引擎（host 调度，decode-only）
 *
 * 权重：converted/qwen38_310b（meta.txt 描述，group-32 W4A16 + f16 张量）
 * 流程：embed → 64 层（48 GDN 线性注意力 + 16 全注意力）→ final norm → lm_head → argmax
 *
 * 内存策略：权重按 meta 顺序**流式**读文件→host 小缓冲→aclrtMemcpy→device（避免 host 峰值 19GB）。
 */
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <chrono>
#include <string>
#include <vector>

#include "acl/acl.h"

static inline uint16_t f2h(float fv)
{
    uint32_t u;
    memcpy(&u, &fv, 4);
    uint32_t sign = (u >> 16) & 0x8000u;
    int32_t exp = (int32_t)((u >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = u & 0x7FFFFFu;
    if (exp <= 0) return (uint16_t)sign;
    if (exp >= 31) return (uint16_t)(sign | 0x7C00u);
    return (uint16_t)(sign | ((uint32_t)exp << 10) | (mant >> 13));
}

static bool g_notWarmup = false;
static int32_t g_dumpPos = 0;   // ENG27_DUMPPOS：dump 哪个位置

static int dumpDev(aclrtStream st, uint8_t *src, size_t nbytes, const char *path)
{
    aclError r = aclrtSynchronizeStream(st);
    if (r != ACL_ERROR_NONE) return 1;
    std::vector<uint8_t> buf(nbytes);
    r = aclrtMemcpy(buf.data(), nbytes, src, nbytes, ACL_MEMCPY_DEVICE_TO_HOST);
    if (r != ACL_ERROR_NONE) return 1;
    FILE *fp = fopen(path, "wb");
    if (!fp) return 1;
    fwrite(buf.data(), 1, nbytes, fp);
    fclose(fp);
    return 0;
}

static inline float h2f(uint16_t hv)
{
    uint32_t sign = (uint32_t)(hv & 0x8000u) << 16;
    uint32_t exp = (hv >> 10) & 0x1Fu;
    uint32_t mant = hv & 0x3FFu;
    uint32_t f;
    if (exp == 0) f = sign;
    else if (exp == 31) f = sign | 0x7F800000u | (mant << 13);
    else f = sign | ((exp - 15 + 127) << 23) | (mant << 13);
    float r; memcpy(&r, &f, 4); return r;
}

// ---- 已注册的 kernel 启动函数 ----
void w4a16_gemv_gen_do(uint32_t b, void *st, uint8_t *act, uint8_t *wq, uint8_t *scales,
                       uint8_t *szs, uint8_t *out, int32_t K, int32_t N, int32_t stage);
void fp16_gemv_do(uint32_t b, void *st, uint8_t *act, uint8_t *w, uint8_t *out, int32_t K, int32_t N);
void rmsnorm_op_do(uint32_t b, void *st, uint8_t *x, uint8_t *w, uint8_t *y, int32_t H, float eps);
void vec_axpy_do(uint32_t b, void *st, uint8_t *a, uint8_t *b_, uint8_t *out, float s, int32_t N);
void mul_silu_do(uint32_t b, void *st, uint8_t *gate, uint8_t *up, uint8_t *out, int32_t N);
void conv1d_decode_do(uint32_t b, void *st, uint8_t *x, uint8_t *state, uint8_t *w, uint8_t *out, int32_t C);
void l2norm_op_do(uint32_t b, void *st, uint8_t *x, uint8_t *y, int32_t D, float eps, float mul);
void gdn_rgdr_do(uint32_t b, void *st, uint8_t *q, uint8_t *k, uint8_t *v,
                 uint8_t *sIn, uint8_t *sOut, uint8_t *o, float beta, float decay);
void full_attn_decode_do(uint32_t b, void *st, uint8_t *q, uint8_t *k, uint8_t *v,
                         uint8_t *cosv, uint8_t *sinv, uint8_t *gate, uint8_t *kcache,
                         uint8_t *vcache, uint8_t *out, int32_t pos);

#define CK(x) do { aclError r_ = (x); if (r_ != ACL_ERROR_NONE) { printf("[engine27] acl err %d line %d\n", (int)r_, __LINE__); return 1; } } while (0)

namespace eng27 {

// ---------------- meta ----------------
struct MTensor {
    std::string role, path, sfile, szfile;
    int n = 0, k = 0;
    bool w4 = false;
    uint8_t *dev = nullptr, *devS = nullptr, *devSZ = nullptr;
};

struct Meta {
    int hidden = 5120, n_layers = 64, ff = 17408, vocab = 248320, head_dim = 256;
    int n_q_heads = 24, n_kv_heads = 4, rotary_dim = 64;
    int gdn_k_heads = 16, gdn_v_heads = 48, gdn_head_dim = 128, conv_k = 4, max_t = 512;
    float eps = 1e-6f;
    std::vector<int> layer_types;   // 1 = GDN, 0 = full attn
};

static std::vector<std::string> split(const std::string &s, char d)
{
    std::vector<std::string> out;
    size_t p = 0;
    while (p <= s.size()) {
        size_t q = s.find(d, p);
        if (q == std::string::npos) { out.push_back(s.substr(p)); break; }
        out.push_back(s.substr(p, q - p));
        p = q + 1;
    }
    return out;
}

static uint8_t *devAlloc(size_t n)
{
    void *p = nullptr;
    aclError r = aclrtMalloc(&p, n, ACL_MEM_MALLOC_HUGE_FIRST);
    if (r != ACL_ERROR_NONE) { printf("[engine27] aclrtMalloc(%zu) failed %d\n", n, (int)r); return nullptr; }
    return (uint8_t *)p;
}

static bool readFile(const std::string &path, std::vector<uint8_t> &buf, size_t expect)
{
    FILE *f = fopen(path.c_str(), "rb");
    if (!f) { printf("[engine27] open fail: %s\n", path.c_str()); return false; }
    if (buf.size() < expect) buf.resize(expect);
    size_t got = fread(buf.data(), 1, expect, f);
    fclose(f);
    if (got != expect) { printf("[engine27] short read %s: %zu/%zu\n", path.c_str(), got, expect); return false; }
    return true;
}

// 权重按需流式加载
static int loadTensor(MTensor &t, const std::string &dir, std::vector<uint8_t> &scratch)
{
    if (t.w4) {
        size_t wb = (size_t)t.n * (t.k / 4) * 2;
        size_t sb = (size_t)t.n * ((t.k / 32 + 15) / 16 * 16) * 2;
        t.dev = devAlloc(wb);
        t.devS = devAlloc(sb);
        t.devSZ = devAlloc(sb);
        if (!t.dev || !t.devS || !t.devSZ) return 1;
        if (!readFile(dir + "/" + t.path, scratch, wb)) return 1;
        CK(aclrtMemcpy(t.dev, wb, scratch.data(), wb, ACL_MEMCPY_HOST_TO_DEVICE));
        if (!readFile(dir + "/" + t.sfile, scratch, sb)) return 1;
        CK(aclrtMemcpy(t.devS, sb, scratch.data(), sb, ACL_MEMCPY_HOST_TO_DEVICE));
        if (!readFile(dir + "/" + t.szfile, scratch, sb)) return 1;
        CK(aclrtMemcpy(t.devSZ, sb, scratch.data(), sb, ACL_MEMCPY_HOST_TO_DEVICE));
    } else {
        // f16：meta 里 shape 用 "x" 分隔，这里只关心元素数
        size_t nb = (size_t)t.n * 2;
        t.dev = devAlloc(nb);
        if (!t.dev) return 1;
        if (!readFile(dir + "/" + t.path, scratch, nb)) return 1;
        CK(aclrtMemcpy(t.dev, nb, scratch.data(), nb, ACL_MEMCPY_HOST_TO_DEVICE));
    }
    return 0;
}

// f16 张量的 shape 解析（"a" / "axb" / "axbxc"），返回元素数
static int64_t shapeElems(const std::string &s)
{
    int64_t n = 1;
    for (auto &p : split(s, 'x')) n *= atoll(p.c_str());
    return n;
}

}  // namespace eng27

int run_engine_27b(void *stream, const char *dir)
{
    using namespace eng27;
    FILE *mf = fopen((std::string(dir) + "/meta.txt").c_str(), "r");
    if (!mf) { printf("[engine27] meta.txt not found in %s\n", dir); return 1; }
    Meta m;
    std::vector<std::string> lines;
    {
        char line[4096];
        while (fgets(line, sizeof(line), mf)) {
            std::string s(line);
            while (!s.empty() && (s.back() == '\n' || s.back() == '\r')) s.pop_back();
            if (!s.empty()) lines.push_back(s);
        }
        fclose(mf);
    }
    std::vector<MTensor> tensors;
    for (auto &s : lines) {
        auto t = split(s, ' ');
        if (t.empty()) continue;
        if (t[0] == "CFG" && t.size() >= 3) {
            if (t[1] == "hidden") m.hidden = atoi(t[2].c_str());
            else if (t[1] == "n_layers") m.n_layers = atoi(t[2].c_str());
            else if (t[1] == "ff") m.ff = atoi(t[2].c_str());
            else if (t[1] == "vocab") m.vocab = atoi(t[2].c_str());
            else if (t[1] == "head_dim") m.head_dim = atoi(t[2].c_str());
            else if (t[1] == "n_q_heads") m.n_q_heads = atoi(t[2].c_str());
            else if (t[1] == "n_kv_heads") m.n_kv_heads = atoi(t[2].c_str());
            else if (t[1] == "rotary_dim") m.rotary_dim = atoi(t[2].c_str());
            else if (t[1] == "gdn_k_heads") m.gdn_k_heads = atoi(t[2].c_str());
            else if (t[1] == "gdn_v_heads") m.gdn_v_heads = atoi(t[2].c_str());
            else if (t[1] == "gdn_head_dim") m.gdn_head_dim = atoi(t[2].c_str());
            else if (t[1] == "conv_k") m.conv_k = atoi(t[2].c_str());
            else if (t[1] == "max_t") m.max_t = atoi(t[2].c_str());
            else if (t[1] == "eps") m.eps = (float)atof(t[2].c_str());
            else if (t[1] == "layer_types") {
                for (auto &v : split(t[2], ',')) m.layer_types.push_back(atoi(v.c_str()));
            }
            continue;
        }
        MTensor mt;
        mt.role = t[0];
        if (t.back() == "w4") {
            if (t.size() < 7) continue;
            mt.path = t[1]; mt.sfile = t[2]; mt.szfile = t[3];
            mt.n = atoi(t[4].c_str()); mt.k = atoi(t[5].c_str());
            mt.w4 = true;
        } else {
            if (t.size() < 4) continue;
            mt.path = t[1];
            int64_t e = shapeElems(t[2]);
            mt.n = (int)e; mt.k = 0;
            mt.w4 = false;
        }
        tensors.push_back(mt);
    }
    printf("[engine27] meta: %zu tensors, layers=%d, GDN=%zu\n", tensors.size(), m.n_layers,
           (size_t)m.layer_types.size());
    if (getenv("ENG27_DUMPPOS")) g_dumpPos = atoi(getenv("ENG27_DUMPPOS"));

    // ---- 流式加载权重 ----
    auto t0 = std::chrono::high_resolution_clock::now();
    std::vector<uint8_t> scratch;
    size_t totalBytes = 0;
    int nLoaded = 0;
    for (auto &t : tensors) {
        if (loadTensor(t, dir, scratch) != 0) { printf("[engine27] load failed: %s\n", t.role.c_str()); return 1; }
        totalBytes += t.w4 ? (size_t)t.n * (t.k / 2) : (size_t)t.n * 2;
        ++nLoaded;
        if ((nLoaded % 100) == 0) {
            printf("[engine27] load progress %d/%zu (%.2f GB)\n", nLoaded, tensors.size(), totalBytes / 1e9);
            fflush(stdout);
        }
    }
    CK(aclrtSynchronizeStream(stream));
    auto t1 = std::chrono::high_resolution_clock::now();
    printf("[engine27] weights loaded: %.2f GB in %.1f s\n", totalBytes / 1e9,
           std::chrono::duration<double>(t1 - t0).count());

    // ---- role → 张量指针（按层索引） ----
    auto find = [&](const std::string &role) -> MTensor * {
        for (auto &t : tensors) if (t.role == role) return &t;
        return nullptr;
    };

    const int H = m.hidden, FF = m.ff, V = m.vocab;
    const int KH = m.gdn_k_heads, VH = m.gdn_v_heads, HD = m.gdn_head_dim;
    const int QH = m.n_q_heads, KVH = m.n_kv_heads, DH = m.head_dim;
    const float eps = m.eps;
    const float qmul = 1.0f / sqrtf((float)m.gdn_head_dim);
    const char *qgEnv = getenv("ENG27_QGATE");
    const bool qgateSplit = (qgEnv != nullptr) && (strcmp(qgEnv, "split") == 0);  // 默认 per-head 交错 [q_h|gate_h]（官方）  // q 归一化后再乘 1/sqrt(d)，k 不乘
    const int maxT = m.max_t;

    // ---- 激活/状态缓冲 ----
    const size_t hb = (size_t)H * 2;
    const size_t qkvN = (size_t)(2 * KH * HD + VH * HD);
    uint8_t *d_x = devAlloc(hb), *d_xn = devAlloc(hb), *d_tmp = devAlloc(hb), *d_proj = devAlloc(hb);
    uint8_t *d_qkv = devAlloc(qkvN * 2), *d_z = devAlloc((size_t)VH * HD * 2);
    uint8_t *d_qc = devAlloc((size_t)KH * HD * 2), *d_kc = devAlloc((size_t)KH * HD * 2);
    uint8_t *d_vc = devAlloc((size_t)VH * HD * 2);
    uint8_t *d_qn = devAlloc((size_t)KH * HD * 2), *d_kn = devAlloc((size_t)KH * HD * 2);
    uint8_t *d_o = devAlloc((size_t)VH * HD * 2), *d_onorm = devAlloc((size_t)VH * HD * 2);
    uint8_t *d_ab = devAlloc(2 * 64 * 2);          // a/b 各最多 64 个
    uint8_t *d_gate = devAlloc((size_t)FF * 2), *d_up = devAlloc((size_t)FF * 2);
    uint8_t *d_act = devAlloc((size_t)FF * 2), *d_mlp = devAlloc(hb);
    uint8_t *d_qg = devAlloc((size_t)(2 * QH * DH) * 2);   // q_proj 输出（q + gate）
    uint8_t *d_gate2 = devAlloc((size_t)(QH * DH) * 2);   // q/gate 需要拆分时的 gate 缓冲
    uint8_t *d_akv = devAlloc((size_t)(2 * KVH * DH) * 2);
    uint8_t *d_attnout = devAlloc((size_t)(QH * DH) * 2);
    uint8_t *d_cos = devAlloc(64 * 4), *d_sin = devAlloc(64 * 4);
    uint8_t *d_qkv2 = devAlloc(qkvN * 2);   // 调试：qkv 第二缓冲
    uint8_t *d_logits = devAlloc((size_t)V * 2);
    uint8_t *d_embrow = devAlloc(hb);
    if (!d_x || !d_qkv || !d_vc || !d_gate || !d_logits) { printf("[engine27] dev alloc failed\n"); return 1; }

    // conv 权重：converter 存 [C,1,4] row-major → kernel 需要 [4,C]，host 转录
    // 为省事：每层 conv 权重在用时转（缓存 3 个缓冲）
    std::vector<uint8_t *> convW(m.n_layers * 3, nullptr);

    // GDN 状态：每层 VH 个 [HD,HD] fp32；conv state：q/k/v 各 [3, C]
    const size_t gdnStateSz = (size_t)VH * HD * HD * 4;
    std::vector<uint8_t *> gdnState(m.n_layers, nullptr), convQ(m.n_layers, nullptr),
        convK(m.n_layers, nullptr), convV(m.n_layers, nullptr);
    for (int i = 0; i < m.n_layers; ++i) {
        gdnState[i] = devAlloc(gdnStateSz);
        convQ[i] = devAlloc((size_t)3 * KH * HD * 4);
        convK[i] = devAlloc((size_t)3 * KH * HD * 4);
        convV[i] = devAlloc((size_t)3 * VH * HD * 4);
        if (!gdnState[i] || !convQ[i] || !convK[i] || !convV[i]) return 1;
        CK(aclrtMemset(gdnState[i], gdnStateSz, 0, gdnStateSz));
        CK(aclrtMemset(convQ[i], (size_t)3 * KH * HD * 4, 0, (size_t)3 * KH * HD * 4));
        CK(aclrtMemset(convK[i], (size_t)3 * KH * HD * 4, 0, (size_t)3 * KH * HD * 4));
        CK(aclrtMemset(convV[i], (size_t)3 * VH * HD * 4, 0, (size_t)3 * VH * HD * 4));
    }
    // KV cache：每个 attn 层 k/v 各 [KVH, maxT, DH] fp16
    std::vector<uint8_t *> kc(m.n_layers, nullptr), vc(m.n_layers, nullptr);
    const size_t kvCacheSz = (size_t)KVH * maxT * DH * 2;
    for (int i = 0; i < m.n_layers; ++i) {
        kc[i] = devAlloc(kvCacheSz);
        vc[i] = devAlloc(kvCacheSz);
        if (!kc[i] || !vc[i]) return 1;
        CK(aclrtMemset(kc[i], kvCacheSz, 0, kvCacheSz));
        CK(aclrtMemset(vc[i], kvCacheSz, 0, kvCacheSz));
    }

    if (getenv("ENG27_PTRS")) {
        struct { const char *n; uint8_t *p; size_t sz; } bufs[] = {
            {"d_x", d_x, hb}, {"d_xn", d_xn, hb}, {"d_tmp", d_tmp, hb}, {"d_proj", d_proj, hb},
            {"d_qkv", d_qkv, qkvN * 2}, {"d_qkv2", d_qkv2, qkvN * 2}, {"d_z", d_z, (size_t)VH*HD*2},
            {"d_qc", d_qc, (size_t)KH*HD*2}, {"d_kc", d_kc, (size_t)KH*HD*2}, {"d_vc", d_vc, (size_t)VH*HD*2},
            {"d_gate", d_gate, (size_t)FF*2}, {"d_up", d_up, (size_t)FF*2}, {"d_act", d_act, (size_t)FF*2},
            {"d_mlp", d_mlp, hb}, {"d_logits", d_logits, (size_t)V*2},
        };
        printf("[ptrs] %-9s %-14s %8s\n", "name", "addr", "size");
        for (auto &b : bufs) printf("[ptrs] %-9s 0x%012llx %8zu\n", b.n, (unsigned long long)(uintptr_t)b.p, b.sz);
        // 重叠检测
        for (size_t i = 0; i < sizeof(bufs)/sizeof(bufs[0]); ++i)
            for (size_t j = i+1; j < sizeof(bufs)/sizeof(bufs[0]); ++j) {
                uintptr_t a0=(uintptr_t)bufs[i].p, a1=a0+bufs[i].sz;
                uintptr_t b0=(uintptr_t)bufs[j].p, b1=b0+bufs[j].sz;
                if (a0 < b1 && b0 < a1) printf("[ptrs] OVERLAP: %s <-> %s\n", bufs[i].n, bufs[j].n);
            }
        fflush(stdout);
    }
    auto T = find("EMBED");
    auto LM = find("LM_HEAD");
    auto FN = find("FINAL_NORM");
    if (!T || !LM || !FN) { printf("[engine27] missing EMBED/LM_HEAD/FINAL_NORM\n"); return 1; }

    // ---- 单 token 前向 ----
    int32_t tok = 42;   // 当前输入 token（每个位置由调用方设置）
    std::vector<uint8_t> bg(8);
    
    // conv 权重转置缓存（host）
    std::vector<uint16_t> convT;

    // 按权重类型分发：w4 -> w4a16_gemv_gen；f16 -> fp16_gemv
    auto gemvRun = [&](MTensor *w, uint8_t *act, uint8_t *out, int K, int N) {
        if (w->w4) {
            if (getenv("ENG27_DOUBLE_GEMV")) {
                // 诊断：先跑一次丢弃（规避首次调用异常），再跑真正的一次
                w4a16_gemv_gen_do(1, stream, act, w->dev, w->devS, w->devSZ, out, K, N, 99);
            }
            w4a16_gemv_gen_do(1, stream, act, w->dev, w->devS, w->devSZ, out, K, N, 99);
        } else {
            fp16_gemv_do(1, stream, act, w->dev, out, K, N);
        }
    };
    // embed 行反量化（支持 f16 / W4 group-32 非对称）
    std::vector<uint8_t> embRowBuf, embScaleBuf, embSzBuf;
    std::vector<uint16_t> embOutBuf;
    auto loadEmbedRow = [&](MTensor *emb, int32_t row, uint8_t *dstHalf) -> int {
        if (!emb->w4) {
            CK(aclrtMemcpy(dstHalf, hb, emb->dev + (size_t)row * H * 2, hb, ACL_MEMCPY_DEVICE_TO_DEVICE));
            return 0;
        }
        const int K = emb->k;
        const int KQ = K / 4;
        const int NG = K / 32;
        const int SBP = (NG + 15) / 16 * 16;
        embRowBuf.resize((size_t)KQ * 2);
        embScaleBuf.resize((size_t)SBP * 2);
        embSzBuf.resize((size_t)SBP * 2);
        embOutBuf.resize((size_t)K);
        CK(aclrtMemcpy(embRowBuf.data(), embRowBuf.size(), emb->dev + (size_t)row * KQ * 2,
                       embRowBuf.size(), ACL_MEMCPY_DEVICE_TO_HOST));
        CK(aclrtMemcpy(embScaleBuf.data(), embScaleBuf.size(), emb->devS + (size_t)row * SBP * 2,
                       embScaleBuf.size(), ACL_MEMCPY_DEVICE_TO_HOST));
        CK(aclrtMemcpy(embSzBuf.data(), embSzBuf.size(), emb->devSZ + (size_t)row * SBP * 2,
                       embSzBuf.size(), ACL_MEMCPY_DEVICE_TO_HOST));
        const uint16_t *words = (const uint16_t *)embRowBuf.data();
        const uint16_t *sc = (const uint16_t *)embScaleBuf.data();
        const uint16_t *sz = (const uint16_t *)embSzBuf.data();
        for (int p = 0; p < 4; ++p) {
            for (int j = 0; j < KQ; ++j) {
                int kk = p * KQ + j;
                int g = kk / 32;
                float sv = h2f(sc[g]);
                float szv = h2f(sz[g]);
                int q = (words[j] >> (4 * p)) & 0xF;
                embOutBuf[kk] = f2h(q * sv - szv);
            }
        }
        CK(aclrtMemcpy(dstHalf, (size_t)K * 2, embOutBuf.data(), (size_t)K * 2, ACL_MEMCPY_HOST_TO_DEVICE));
        return 0;
    };
    auto runOne = [&](int32_t pos, int32_t inTok) -> int {
        // 每 token 更新 RoPE 的 cos/sin（rotary_dim/2 个频率）
        {
            int rh = m.rotary_dim / 2;
            std::vector<float> cf(rh), sf(rh);
            for (int i = 0; i < rh; ++i) {
                double freq = 1.0 / pow(1e7, (double)(2 * i) / m.rotary_dim);
                cf[i] = (float)cos((double)pos * freq);
                sf[i] = (float)sin((double)pos * freq);
            }
            CK(aclrtMemcpy(d_cos, rh * 4, cf.data(), rh * 4, ACL_MEMCPY_HOST_TO_DEVICE));
            CK(aclrtMemcpy(d_sin, rh * 4, sf.data(), rh * 4, ACL_MEMCPY_HOST_TO_DEVICE));
        }
        // embed
        if (loadEmbedRow(T, inTok, d_x) != 0) return 1;
        if (g_notWarmup && getenv("ENG27_DUMPX") && pos == g_dumpPos) {
            if (dumpDev(stream, d_x, hb, "/tmp/eng_dx.bin")) return 1;
            printf("[dumpX] d_x saved at pos=%d tok=%d\n", pos, inTok);
        }
        for (int li = 0; li < m.n_layers; ++li) {
            auto ln1 = find("L" + std::to_string(li) + ".LN1");
            auto ln2 = find("L" + std::to_string(li) + ".LN2");
            auto mg = find("L" + std::to_string(li) + ".MGATE");
            auto mu = find("L" + std::to_string(li) + ".MUP");
            auto md = find("L" + std::to_string(li) + ".MDOWN");
            if (!ln1 || !ln2 || !mg || !mu || !md) { printf("[engine27] layer %d tensors missing\n", li); return 1; }
            if (getenv("ENG27_TRACE")) { printf("[trace] === layer %d/%d (%s) ===\n", li, m.n_layers, m.layer_types[li] == 1 ? "GDN" : "ATTN"); fflush(stdout); }
            rmsnorm_op_do(1, stream, d_x, ln1->dev, d_xn, H, eps);
            if (getenv("ENG27_TRACE")) { CK(aclrtSynchronizeStream(stream)); printf("[trace] layer %d ln1 ok\n", li); }
            if (m.layer_types[li] == 1) {
                auto qkvw = find("L" + std::to_string(li) + ".QKV");
                auto zw = find("L" + std::to_string(li) + ".Z");
                auto aw = find("L" + std::to_string(li) + ".APROJ");
                auto bw = find("L" + std::to_string(li) + ".BPROJ");
                auto cw = find("L" + std::to_string(li) + ".CONV");
                auto alog = find("L" + std::to_string(li) + ".ALOG");
                auto dtb = find("L" + std::to_string(li) + ".DTBIAS");
                auto gn = find("L" + std::to_string(li) + ".GNORM");
                auto ow = find("L" + std::to_string(li) + ".OUT");
                if (!qkvw || !zw || !aw || !bw || !cw || !alog || !dtb || !gn || !ow) {
                    printf("[engine27] gdn layer %d tensors missing\n", li);
                    return 1;
                }
                if (g_notWarmup && getenv("ENG27_DUMP2") && li == 0 && pos == g_dumpPos) {
                    printf("[gemv] qkv K=%d N=%d dev=%p s=%p sz=%p act=%p out=%p\n", H, (int)qkvN,
                           (void*)qkvw->dev, (void*)qkvw->devS, (void*)qkvw->devSZ, (void*)d_xn, (void*)d_qkv);
                }
                gemvRun(qkvw, d_xn, d_qkv, H, (int)qkvN);
                if (g_notWarmup && getenv("ENG27_DUMP2") && li == 0 && pos == g_dumpPos) {
                    if (dumpDev(stream, d_qkv, qkvN * 2, "/tmp/eng_qkv_early.bin")) return 1;
                    printf("[dump2] qkv call1 (d_qkv) saved\n");
                    gemvRun(qkvw, d_xn, d_qkv2, H, (int)qkvN);
                    if (dumpDev(stream, d_qkv2, qkvN * 2, "/tmp/eng_qkv_call2.bin")) return 1;
                    printf("[dump2] qkv call2 (d_qkv2) saved\n");
                }
                if (g_notWarmup && getenv("ENG27_DUMP3") && li == 0 && pos == g_dumpPos) {
                    // 交换顺序：先写 d_qkv2 再写 d_qkv
                    gemvRun(qkvw, d_xn, d_qkv2, H, (int)qkvN);
                    if (dumpDev(stream, d_qkv2, qkvN * 2, "/tmp/eng_qkv_A.bin")) return 1;
                    printf("[dump3] A -> d_qkv2 saved\n");
                    gemvRun(qkvw, d_xn, d_qkv, H, (int)qkvN);
                    if (dumpDev(stream, d_qkv, qkvN * 2, "/tmp/eng_qkv_B.bin")) return 1;
                    printf("[dump3] B -> d_qkv saved\n");
                }
                if (getenv("ENG27_TRACE")) { CK(aclrtSynchronizeStream(stream)); printf("[trace] qkv w4 gemv ok\n"); }
                gemvRun(zw, d_xn, d_z, H, VH * HD);
                if (getenv("ENG27_TRACE")) { CK(aclrtSynchronizeStream(stream)); printf("[trace] z w4 gemv ok\n"); }
                fp16_gemv_do(1, stream, d_xn, aw->dev, d_ab, H, VH);
                if (getenv("ENG27_TRACE")) { CK(aclrtSynchronizeStream(stream)); printf("[trace] aproj gemv ok\n"); }
                fp16_gemv_do(1, stream, d_xn, bw->dev, d_ab + VH * 2, H, VH);
                if (getenv("ENG27_TRACE")) { CK(aclrtSynchronizeStream(stream)); printf("[trace] bproj gemv ok\n"); }
                // conv（需要 [4,C] 转置权重；这里直接调用 kernel 假定已转置：临时在 host 构造）
                // 为简化首版：conv 权重转换在下面统一做（事先把所有层 conv 转好）
                if (g_notWarmup && getenv("ENG27_DUMP2") && li == 0 && pos == g_dumpPos) {
                    if (dumpDev(stream, d_xn, H * 2, "/tmp/eng_xn.bin")) return 1;
                    if (dumpDev(stream, d_qkv, qkvN * 2, "/tmp/eng_qkv.bin")) return 1;
                    if (dumpDev(stream, d_z, (size_t)VH * HD * 2, "/tmp/eng_z.bin")) return 1;
                    if (dumpDev(stream, d_ab, 2 * VH * 2, "/tmp/eng_ab.bin")) return 1;
                    if (dumpDev(stream, qkvw->dev, 64, "/tmp/eng_wqkv_head.bin")) return 1;
                    if (dumpDev(stream, qkvw->devS, 64, "/tmp/eng_sqkv_head.bin")) return 1;
                    if (dumpDev(stream, qkvw->devSZ, 64, "/tmp/eng_szqkv_head.bin")) return 1;
                    printf("[dump2] xn/qkv/z/ab saved\n");
                }
                conv1d_decode_do(1, stream, d_qkv, convQ[li], convW[li * 3 + 0], d_qc, KH * HD);
                conv1d_decode_do(1, stream, d_qkv + (size_t)KH * HD * 2, convK[li], convW[li * 3 + 1], d_kc, KH * HD);
                conv1d_decode_do(1, stream, d_qkv + (size_t)2 * KH * HD * 2, convV[li], convW[li * 3 + 2], d_vc, VH * HD);
                if (g_notWarmup && getenv("ENG27_DUMP2") && li == 0 && pos == g_dumpPos) {
                    if (dumpDev(stream, d_qc, (size_t)KH * HD * 2, "/tmp/eng_convq.bin")) return 1;
                    if (dumpDev(stream, d_kc, (size_t)KH * HD * 2, "/tmp/eng_convk.bin")) return 1;
                    if (dumpDev(stream, d_vc, (size_t)VH * HD * 2, "/tmp/eng_convv.bin")) return 1;
                    printf("[dump2] conv q/k/v saved\n");
                }
                for (int h = 0; h < KH; ++h) {
                    l2norm_op_do(1, stream, d_qc + (size_t)h * HD * 2, d_qn + (size_t)h * HD * 2, HD, eps, qmul);
                    l2norm_op_do(1, stream, d_kc + (size_t)h * HD * 2, d_kn + (size_t)h * HD * 2, HD, eps, 1.0f);
                }
                if (getenv("ENG27_TRACE")) { CK(aclrtSynchronizeStream(stream)); printf("[trace] l2norm x%d ok\n", 2 * KH); }
                CK(aclrtSynchronizeStream(stream));
                // beta / g（host）
                std::vector<uint16_t> ha(VH), hb2(VH), hlog(VH), hdt(VH);
                CK(aclrtMemcpy(ha.data(), VH * 2, d_ab, VH * 2, ACL_MEMCPY_DEVICE_TO_HOST));
                CK(aclrtMemcpy(hb2.data(), VH * 2, d_ab + VH * 2, VH * 2, ACL_MEMCPY_DEVICE_TO_HOST));
                CK(aclrtMemcpy(hlog.data(), VH * 2, alog->dev, VH * 2, ACL_MEMCPY_DEVICE_TO_HOST));
                CK(aclrtMemcpy(hdt.data(), VH * 2, dtb->dev, VH * 2, ACL_MEMCPY_DEVICE_TO_HOST));
                std::vector<float> beta(VH), decay(VH);
                for (int i = 0; i < VH; ++i) {
                    float bv = h2f(hb2[i]);
                    float av = h2f(ha[i]) + h2f(hdt[i]);
                    float sp = (av > 20.0f) ? av : logf(1.0f + expf(av));
                    float gv = -expf(h2f(hlog[i])) * sp;
                    beta[i] = 1.0f / (1.0f + expf(-bv));
                    decay[i] = expf(gv);
                }
                int gk = VH / KH;
                for (int vh = 0; vh < VH; ++vh) {
                    int kh = vh / gk;
                    gdn_rgdr_do(1, stream, d_qn + (size_t)kh * HD * 2, d_kn + (size_t)kh * HD * 2,
                                d_vc + (size_t)vh * HD * 2, gdnState[li] + (size_t)vh * HD * HD * 4,
                                gdnState[li] + (size_t)vh * HD * HD * 4,
                                d_o + (size_t)vh * HD * 2, beta[vh], decay[vh]);
                }
                if (getenv("ENG27_TRACE")) { CK(aclrtSynchronizeStream(stream)); printf("[trace] rgdr x%d ok\n", VH); }
                for (int vh = 0; vh < VH; ++vh) {
                    rmsnorm_op_do(1, stream, d_o + (size_t)vh * HD * 2, gn->dev, d_tmp, HD, eps);
                    mul_silu_do(1, stream, d_z + (size_t)vh * HD * 2, d_tmp,
                                d_onorm + (size_t)vh * HD * 2, HD);
                }
                if (getenv("ENG27_TRACE")) { CK(aclrtSynchronizeStream(stream)); printf("[trace] gatednorm x%d ok\n", VH); }
                gemvRun(ow, d_onorm, d_proj, VH * HD, H);
                if (getenv("ENG27_TRACE")) { CK(aclrtSynchronizeStream(stream)); printf("[trace] outproj gemv ok\n"); }
            } else {
                auto qp = find("L" + std::to_string(li) + ".QPROJ");
                auto kp = find("L" + std::to_string(li) + ".KPROJ");
                auto vp = find("L" + std::to_string(li) + ".VPROJ");
                auto op = find("L" + std::to_string(li) + ".OPROJ");
                auto qn = find("L" + std::to_string(li) + ".QNORM");
                auto kn = find("L" + std::to_string(li) + ".KNORM");
                if (!qp || !kp || !vp || !op || !qn || !kn) { printf("[engine27] attn layer %d missing\n", li); return 1; }
                gemvRun(qp, d_xn, d_qg, H, 2 * QH * DH);
                if (!qgateSplit) {
                    // per-head 交错：[q_h(256) | gate_h(256)] × 24 → 拆成两个连续数组
                    CK(aclrtSynchronizeStream(stream));
                    std::vector<uint16_t> raw((size_t)2 * QH * DH), qout((size_t)QH * DH), gout((size_t)QH * DH);
                    CK(aclrtMemcpy(raw.data(), raw.size() * 2, d_qg, raw.size() * 2, ACL_MEMCPY_DEVICE_TO_HOST));
                    for (int h = 0; h < QH; ++h) {
                        for (int d = 0; d < DH; ++d) {
                            qout[(size_t)h * DH + d] = raw[(size_t)h * 2 * DH + d];
                            gout[(size_t)h * DH + d] = raw[(size_t)h * 2 * DH + DH + d];
                        }
                    }
                    CK(aclrtMemcpy(d_qg, qout.size() * 2, qout.data(), qout.size() * 2, ACL_MEMCPY_HOST_TO_DEVICE));
                    CK(aclrtMemcpy(d_gate2, gout.size() * 2, gout.data(), gout.size() * 2, ACL_MEMCPY_HOST_TO_DEVICE));
                }
                gemvRun(kp, d_xn, d_akv, H, KVH * DH);
                gemvRun(vp, d_xn, d_akv + KVH * DH * 2, H, KVH * DH);
                // qk-norm（每 head 256 维）
                for (int h = 0; h < QH; ++h) {
                    rmsnorm_op_do(1, stream, d_qg + (size_t)h * DH * 2, qn->dev, d_qg + (size_t)h * DH * 2, DH, eps);
                }
                for (int h = 0; h < KVH; ++h) {
                    rmsnorm_op_do(1, stream, d_akv + (size_t)h * DH * 2, kn->dev, d_akv + (size_t)h * DH * 2, DH, eps);
                }
                if (g_notWarmup && getenv("ENG27_DUMPA") && li == 0 && pos == g_dumpPos) {
                    if (dumpDev(stream, d_qg, (size_t)2 * QH * DH * 2, "/tmp/eng_qg.bin")) return 1;
                    if (dumpDev(stream, d_akv, (size_t)2 * KVH * DH * 2, "/tmp/eng_akv.bin")) return 1;
                    printf("[dumpA] qg/akv saved\n");
                }
                full_attn_decode_do(1, stream, d_qg, d_akv, d_akv + KVH * DH * 2, d_cos, d_sin,
                                    qgateSplit ? (d_qg + (size_t)QH * DH * 2) : d_gate2,
                                    kc[li], vc[li], d_attnout, pos);
                if (g_notWarmup && getenv("ENG27_DUMPA") && li == 0 && pos == g_dumpPos) {
                    if (dumpDev(stream, d_attnout, (size_t)QH * DH * 2, "/tmp/eng_attnout.bin")) return 1;
                    printf("[dumpA] attnout saved\n");
                }
                gemvRun(op, d_attnout, d_proj, QH * DH, H);
            }
            vec_axpy_do(1, stream, d_x, d_proj, d_x, 1.0f, H);
            if (getenv("ENG27_TRACE")) { CK(aclrtSynchronizeStream(stream)); printf("[trace] axpy ok\n"); }
            if (g_notWarmup && getenv("ENG27_DUMP") && li == 0 && pos == g_dumpPos) {
                // 调试：dump 第一层 attn/gdn 输出与 MLP 输出，便于与 PC 参考对比
                CK(aclrtSynchronizeStream(stream));
                std::vector<uint16_t> dbuf(H);
                CK(aclrtMemcpy(dbuf.data(), H * 2, d_x, H * 2, ACL_MEMCPY_DEVICE_TO_HOST));
                FILE *fp = fopen("/tmp/eng_l0_postattn.bin", "wb");
                if (fp) { fwrite(dbuf.data(), 2, H, fp); fclose(fp); }
                printf("[dump] layer0 post-attn x saved\n");
            }
            // MLP
            rmsnorm_op_do(1, stream, d_x, ln2->dev, d_xn, H, eps);
            gemvRun(mg, d_xn, d_gate, H, FF);
            if (getenv("ENG27_TRACE")) { CK(aclrtSynchronizeStream(stream)); printf("[trace] mlp gate gemv ok\n"); }
            gemvRun(mu, d_xn, d_up, H, FF);
            if (getenv("ENG27_TRACE")) { CK(aclrtSynchronizeStream(stream)); printf("[trace] mlp up gemv ok\n"); }
            if (g_notWarmup && getenv("ENG27_DUMPM") && li == 0 && pos == g_dumpPos) {
                if (dumpDev(stream, d_gate, (size_t)FF * 2, "/tmp/eng_gate.bin")) return 1;
                if (dumpDev(stream, d_up, (size_t)FF * 2, "/tmp/eng_up.bin")) return 1;
                printf("[dumpM] gate/up saved\n");
            }
            mul_silu_do(1, stream, d_gate, d_up, d_act, FF);
            if (g_notWarmup && getenv("ENG27_DUMPM") && li == 0 && pos == g_dumpPos) {
                if (dumpDev(stream, d_act, (size_t)FF * 2, "/tmp/eng_act.bin")) return 1;
                printf("[dumpM] act saved\n");
            }
            if (getenv("ENG27_TRACE")) { CK(aclrtSynchronizeStream(stream)); printf("[trace] mul_silu ok\n"); }
            gemvRun(md, d_act, d_mlp, FF, H);
            if (g_notWarmup && getenv("ENG27_DUMP") && li == 0 && pos == g_dumpPos) {
                CK(aclrtSynchronizeStream(stream));
                std::vector<uint16_t> dbuf2(H);
                CK(aclrtMemcpy(dbuf2.data(), H * 2, d_mlp, H * 2, ACL_MEMCPY_DEVICE_TO_HOST));
                FILE *fp2 = fopen("/tmp/eng_l0_mlp.bin", "wb");
                if (fp2) { fwrite(dbuf2.data(), 2, H, fp2); fclose(fp2); }
                printf("[dump] layer0 mlp out saved\n");
            }
            if (getenv("ENG27_TRACE")) { CK(aclrtSynchronizeStream(stream)); printf("[trace] mlp down gemv ok\n"); }
            vec_axpy_do(1, stream, d_x, d_mlp, d_x, 1.0f, H);
        }
        rmsnorm_op_do(1, stream, d_x, FN->dev, d_xn, H, eps);
        gemvRun(LM, d_xn, d_logits, H, V);
        CK(aclrtSynchronizeStream(stream));
        // argmax
        static std::vector<uint16_t> hl;
        hl.resize(V);
        CK(aclrtMemcpy(hl.data(), (size_t)V * 2, d_logits, (size_t)V * 2, ACL_MEMCPY_DEVICE_TO_HOST));
        if (getenv("ENG27_TRACE")) {
            // 诊断：打印 top-5 与 logits 统计，便于判断输出是否发散
            int t5[5] = {-1,-1,-1,-1,-1};
            float v5[5] = {-1e30f,-1e30f,-1e30f,-1e30f,-1e30f};
            double sum = 0.0; float mx = -1e30f, mn = 1e30f;
            for (int i = 0; i < V; ++i) {
                float v = h2f(hl[i]);
                sum += v; if (v > mx) mx = v; if (v < mn) mn = v;
                for (int j = 0; j < 5; ++j) {
                    if (v > v5[j]) {
                        for (int k = 4; k > j; --k) { v5[k] = v5[k-1]; t5[k] = t5[k-1]; }
                        v5[j] = v; t5[j] = i; break;
                    }
                }
            }
            printf("[logits] mean=%.4f min=%.3f max=%.3f top5:", sum / V, mn, mx);
            for (int j = 0; j < 5; ++j) printf(" %d(%.2f)", t5[j], v5[j]);
            printf("\n");
        }
        int best = 0;
        float bv = h2f(hl[0]);
        for (int i = 1; i < V; ++i) {
            float v = h2f(hl[i]);
            if (v > bv) { bv = v; best = i; }
        }
        if (getenv("ENG27_TRACE")) { printf("[engine27] pos=%d in=%d -> token %d (logit %.3f)\n", pos, inTok, best, bv); }
        return best;
    };

    // ---- conv 权重转置（host → device），每层 3 个 ----
    {
        std::vector<uint16_t> src, dst;
        for (int li = 0; li < m.n_layers; ++li) {
            if (m.layer_types[li] != 1) continue;
            auto cw = find("L" + std::to_string(li) + ".CONV");
            if (!cw) continue;
            int C = 2 * KH * HD + VH * HD;   // 10240
            size_t elems = (size_t)C * 4;
            src.resize(elems);
            CK(aclrtMemcpy(src.data(), elems * 2, cw->dev, elems * 2, ACL_MEMCPY_DEVICE_TO_HOST));
            // src[c*4 + j] -> dst[j*C + c]
            int offs[3] = {0, KH * HD, 2 * KH * HD};
            int cs[3] = {KH * HD, KH * HD, VH * HD};
            for (int part = 0; part < 3; ++part) {
                dst.assign((size_t)cs[part] * 4, 0);
                for (int c = 0; c < cs[part]; ++c) {
                    for (int j = 0; j < 4; ++j) {
                        dst[(size_t)j * cs[part] + c] = src[(size_t)(offs[part] + c) * 4 + j];
                    }
                }
                convW[li * 3 + part] = devAlloc((size_t)cs[part] * 4 * 2);
                if (!convW[li * 3 + part]) return 1;
                CK(aclrtMemcpy(convW[li * 3 + part], (size_t)cs[part] * 4 * 2, dst.data(),
                               (size_t)cs[part] * 4 * 2, ACL_MEMCPY_HOST_TO_DEVICE));
            }
        }
        printf("[engine27] conv weights transposed\n");
    }

    // ---- RoPE cos/sin 表（rotary_dim/2 个频率）----
    {
        int rh = m.rotary_dim / 2;
        std::vector<float> cf(rh), sf(rh);
        for (int i = 0; i < rh; ++i) {
            double freq = 1.0 / pow(1e7, (double)(2 * i) / m.rotary_dim);
            cf[i] = (float)cos((double)0 * freq);
            sf[i] = (float)sin((double)0 * freq);
        }
        CK(aclrtMemcpy(d_cos, rh * 4, cf.data(), rh * 4, ACL_MEMCPY_HOST_TO_DEVICE));
        CK(aclrtMemcpy(d_sin, rh * 4, sf.data(), rh * 4, ACL_MEMCPY_HOST_TO_DEVICE));
    }

    // ---- prompt / 生成参数 ----
    std::vector<int32_t> ids;
    const char *idsEnv = getenv("ENG27_IDS");
    if (idsEnv) {
        const char *q = idsEnv;
        while (*q) {
            char *endp = nullptr;
            long v = strtol(q, &endp, 10);
            if (endp == q) break;
            ids.push_back((int32_t)v);
            q = endp;
            while (*q == ',' || *q == ' ') ++q;
        }
    }
    if (ids.empty()) { ids.push_back(151644); ids.push_back(9707); }   // 占位（真实 prompt 由 ENG27_IDS 传入）
    int nGen = 16;
    if (getenv("ENG27_GEN")) nGen = atoi(getenv("ENG27_GEN"));
    const int maxPos = m.max_t - 2;
    if ((int)ids.size() + nGen > maxPos) nGen = maxPos - (int)ids.size();
    if (nGen < 1) nGen = 1;
    printf("[engine27] prompt ids: %zu, gen: %d\n", ids.size(), nGen);

    // ---- warmup：规避 kernel 进程内首次调用异常（实测首次 GEMV 结果错误、后续正确）----
    if (!getenv("ENG27_NOWARMUP")) {
        printf("[engine27] warmup forward ...\n");
        fflush(stdout);
        runOne(0, ids[0]);
        // 重置所有推理状态
        for (int i = 0; i < m.n_layers; ++i) {
            CK(aclrtMemset(gdnState[i], gdnStateSz, 0, gdnStateSz));
            CK(aclrtMemset(convQ[i], (size_t)3 * KH * HD * 4, 0, (size_t)3 * KH * HD * 4));
            CK(aclrtMemset(convK[i], (size_t)3 * KH * HD * 4, 0, (size_t)3 * KH * HD * 4));
            CK(aclrtMemset(convV[i], (size_t)3 * VH * HD * 4, 0, (size_t)3 * VH * HD * 4));
            CK(aclrtMemset(kc[i], kvCacheSz, 0, kvCacheSz));
            CK(aclrtMemset(vc[i], kvCacheSz, 0, kvCacheSz));
        }
        CK(aclrtSynchronizeStream(stream));
        g_notWarmup = true;
        printf("[engine27] warmup done, states reset\n");
        fflush(stdout);
    }

    std::vector<int32_t> out;
    auto tAll0 = std::chrono::high_resolution_clock::now();
    int next = ids[0];
    double prefillSec = 0.0;
    // prefill（最后一个 token 也走一次前向拿到首个预测）
    auto tp0 = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < ids.size(); ++i) {
        next = runOne((int32_t)i, ids[i]);
    }
    auto tp1 = std::chrono::high_resolution_clock::now();
    prefillSec = std::chrono::duration<double>(tp1 - tp0).count();
    // decode
    auto td0 = std::chrono::high_resolution_clock::now();
    for (int g = 0; g < nGen; ++g) {
        out.push_back(next);
        int pos = (int)ids.size() + g;
        if (pos >= maxPos) break;
        next = runOne(pos, next);
    }
    auto td1 = std::chrono::high_resolution_clock::now();
    double decSec = std::chrono::duration<double>(td1 - td0).count();
    auto tAll1 = std::chrono::high_resolution_clock::now();
    (void)tAll0;
    (void)tAll1;
    (void)prefillSec;
    printf("TOKENS");
    for (auto v : out) printf(" %d", v);
    printf("\n");
    printf("[engine27] prefill %zu tok in %.2f s (%.2f tok/s); decode %d tok in %.2f s -> %.3f tok/s (%.2f s/token)\n",
           ids.size(), prefillSec, prefillSec > 0 ? ids.size() / prefillSec : 0.0,
           (int)out.size(), decSec, decSec > 0 ? out.size() / decSec : 0.0,
           out.size() > 0 ? decSec / out.size() : 0.0);
    return 0;
}
