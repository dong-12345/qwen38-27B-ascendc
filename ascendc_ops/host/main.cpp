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
#include <string>
#include <vector>

#include "acl/acl.h"
#include "data_utils.h"

#define CHECK_ACL_RET(x)                                                       \
    do {                                                                       \
        aclError __ret = (x);                                                   \
        if (__ret != ACL_ERROR_NONE) {                                          \
            printf("[FAIL] acl error %d at %s:%d\n", (int)__ret, __FILE__,      \
                   __LINE__);                                                   \
            std::exit(1);                                                       \
        }                                                                       \
    } while (0)

void add_custom_do(uint32_t blockDim, void *stream, uint8_t *x, uint8_t *y, uint8_t *z);
void gdn_gating_do(uint32_t blockDim, void *stream, uint8_t *a, uint8_t *b,
                   uint8_t *dt_bias, uint8_t *g_scale, uint8_t *g, uint8_t *beta);

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

}  // namespace

int main(int argc, char *argv[])
{
    std::string test = (argc > 1) ? argv[1] : "add";
    CHECK_ACL_RET(aclInit(nullptr));
    CHECK_ACL_RET(aclrtSetDevice(0));
    CHECK_ACL_RET(aclrtCreateStream(&g_stream));

    int rc = 0;
    if (test == "add") {
        rc = test_add();
    } else if (test == "gdn_gating") {
        rc = test_gdn_gating();
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
