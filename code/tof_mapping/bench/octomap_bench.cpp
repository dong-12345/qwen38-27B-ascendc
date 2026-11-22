// octomap_bench.cpp — 测 OctoMap 插入 640x480 深度点云的耗时（评估"实时建图"前端成本）
// 编译： g++ -O3 -std=c++17 octomap_bench.cpp -o /tmp/octomap_bench -loctomap -loctomath
#include <octomap/octomap.h>
#include <octomap/OcTree.h>
#include <chrono>
#include <cstdio>
#include <vector>
#include <cmath>
#include <random>

int main() {
    const int W = 640, H = 480;
    const float fx = 380.f, fy = 380.f, cx = W / 2.f, cy = H / 2.f;
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> noise(-0.01f, 0.01f);

    // 预生成 30 帧点云（640x480，隔行隔列降采样，模拟真实深度 1~5m）
    std::vector<octomap::Pointcloud> frames(30);
    for (int f = 0; f < 30; ++f) {
        frames[f].reserve(W * H / 4);
        for (int y = 0; y < H; y += 2) {
            for (int x = 0; x < W; x += 2) {
                float z = 1.0f + 4.0f * std::abs(std::sin((x + f * 3) * 0.01f) * std::cos(y * 0.013f));
                float px = (x - cx) * z / fx;
                float py = (y - cy) * z / fy;
                frames[f].push_back(octomap::point3d(px + noise(rng), py + noise(rng), z + noise(rng)));
            }
        }
    }

    octomap::OcTree tree(0.05);   // 5cm 体素
    double total_ms = 0;
    for (int f = 0; f < 30; ++f) {
        auto t0 = std::chrono::high_resolution_clock::now();
        tree.insertPointCloud(frames[f], octomap::point3d(0, 0, 0));
        tree.updateInnerOccupancy();
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        total_ms += ms;
        if (f < 5) printf("  frame %2d: %d points -> %.2f ms\n", f, (int)frames[f].size(), ms);
    }
    printf("OctoMap 640x480 隔行隔列（%d点/帧）x30: 平均 %.2f ms/帧 -> %.1f fps, 树节点=%zu\n",
           (int)frames[0].size(), total_ms / 30.0, 1000.0 / (total_ms / 30.0), tree.size());

    // 全分辨率
    octomap::Pointcloud full;
    full.reserve(W * H);
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) {
            float z = 1.0f + 4.0f * std::abs(std::sin(x * 0.01f) * std::cos(y * 0.013f));
            float px = (x - cx) * z / fx, py = (y - cy) * z / fy;
            full.push_back(octomap::point3d(px, py, z));
        }
    octomap::OcTree tree2(0.05);
    auto t0 = std::chrono::high_resolution_clock::now();
    tree2.insertPointCloud(full, octomap::point3d(0, 0, 0));
    tree2.updateInnerOccupancy();
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    printf("OctoMap 640x480 全分辨率 %d 点/帧: %.2f ms -> %.1f fps\n",
           (int)full.size(), ms, 1000.0 / ms);
    return 0;
}
