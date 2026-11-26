/*
 * bench_cpu_bw.c — 4×TaishanV200M 读/拷贝带宽（判定板级 DDR 真实档位）
 *
 * 用法: ./bench_cpu_bw <总MB> <线程数> [read|copy]
 * 输出: 聚合 GB/s（每线程 GB/s 由聚合/线程数得到）
 *
 * 说明：310B 的 DDR 理论档位有三种可能（LPDDR4X-4266）：
 *   32bit = 17.1GB/s, 64bit = 34.1GB/s, 96bit = 51.2GB/s。
 *   若 CPU 多线程能拉到 ~40GB/s+，则"单 AI Core 10.2GB/s"是核内 MTE2 端口墙，不是 DDR 墙。
 */
#define _GNU_SOURCE
#include <pthread.h>
#include <sched.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static uint8_t *g_buf;
static size_t g_bytes_per;
static int g_mode;              /* 0=read, 1=copy */
static uint8_t *g_copy_dst;

static double now_sec(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static void *worker(void *arg)
{
    long tid = (long)arg;
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(tid % 4, &set);
    (void)sched_setaffinity(0, sizeof(set), &set);

    uint8_t *p = g_buf + (size_t)tid * g_bytes_per;
    if (g_mode == 1) {
        uint8_t *q = g_copy_dst + (size_t)tid * g_bytes_per;
        memcpy(q, p, g_bytes_per);
        return NULL;
    }
    uint64_t sum = 0;
    const uint64_t *w = (const uint64_t *)p;
    size_t n = g_bytes_per / 8;
    for (size_t i = 0; i + 7 < n; i += 8) {
        sum += w[i] + w[i + 1] + w[i + 2] + w[i + 3];
        sum += w[i + 4] + w[i + 5] + w[i + 6] + w[i + 7];
    }
    if (sum == 0xdeadbeefULL) {
        printf(" ");    /* 防优化屏障 */
    }
    return NULL;
}

int main(int argc, char **argv)
{
    size_t total_mb = (argc > 1) ? strtoul(argv[1], NULL, 0) : 512;
    int nthreads = (argc > 2) ? atoi(argv[2]) : 4;
    if (nthreads < 1) nthreads = 1;
    if (nthreads > 64) nthreads = 64;
    g_mode = (argc > 3 && strcmp(argv[3], "copy") == 0) ? 1 : 0;
    size_t total = total_mb * 1024UL * 1024UL;
    g_bytes_per = total / (size_t)nthreads;
    g_buf = (uint8_t *)malloc(total);
    if (!g_buf) {
        printf("malloc failed\n");
        return 1;
    }
    memset(g_buf, 0x5a, total);
    if (g_mode == 1) {
        g_copy_dst = (uint8_t *)malloc(total);
        if (!g_copy_dst) {
            printf("malloc failed\n");
            return 1;
        }
        memset(g_copy_dst, 0, total);
    }

    pthread_t th[64];
    double t0 = now_sec();
    for (long i = 0; i < nthreads; ++i) {
        if (pthread_create(&th[i], NULL, worker, (void *)i) != 0) {
            printf("pthread_create failed\n");
            return 1;
        }
    }
    for (int i = 0; i < nthreads; ++i) {
        pthread_join(th[i], NULL);
    }
    double t1 = now_sec() - t0;
    double gb = (double)total / 1e9;
    printf("[cpu_bw] mode=%s threads=%d total=%zuMB time=%.3fs -> %.2f GB/s aggregate "
           "(%.2f GB/s/thread)\n",
           g_mode ? "copy" : "read", nthreads, total_mb, t1, gb / t1, gb / t1 / nthreads);
    free(g_buf);
    free(g_copy_dst);
    return 0;
}
