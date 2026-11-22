// mapping_frontend_bench.cpp — 板子上"实时建图"几种前端的耗时对比
//  1) OctoMap 3D 带 raycast（标准 insertPointCloud），不同降采样倍率
//  2) OctoMap 只标占据（无 raycast，updateNode）
//  3) 2D 占据栅格投影（模拟 nav_msgs/OccupancyGrid / grid_map）
// 编译： g++ -O3 -std=c++17 mapping_frontend_bench.cpp -o /tmp/mapping_bench -loctomap -loctomath
#include <octomap/octomap.h>
#include <octomap/OcTree.h>
#include <chrono>
#include <cstdio>
#include <vector>
#include <cmath>
#include <unordered_map>

using clk = std::chrono::high_resolution_clock;

static double ms_since(clk::time_point t0) {
    return std::chrono::duration<double, std::milli>(clk::now() - t0).count();
}

int main() {
    const int W = 640, H = 480;
    const float fx = 380.f, fy = 380.f, cx = W / 2.f, cy = H / 2.f;

    // 生成一帧深度图（1~5m）
    std::vector<float> depth(W * H);
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x)
            depth[y * W + x] = 1.0f + 4.0f * std::abs(std::sin(x * 0.011f) * std::cos(y * 0.013f));

    // ---------- 1) OctoMap 带 raycast，不同降采样 ----------
    for (int step : {2, 4, 8}) {
        octomap::Pointcloud pc;
        for (int y = 0; y < H; y += step)
            for (int x = 0; x < W; x += step) {
                float z = depth[y * W + x];
                pc.push_back(octomap::point3d((x - cx) * z / fx, (y - cy) * z / fy, z));
            }
        octomap::OcTree tree(0.05);
        auto t0 = clk::now();
        tree.insertPointCloud(pc, octomap::point3d(0, 0, 0));
        tree.updateInnerOccupancy();
        double ms = ms_since(t0);
        printf("OctoMap raycast  step=%d (%6d 点/帧): %8.2f ms -> %6.1f fps\n",
               step, (int)pc.size(), ms, 1000.0 / ms);
    }

    // ---------- 2) OctoMap 只标占据（无 raycast） ----------
    for (int step : {4, 8}) {
        octomap::OcTree tree(0.05);
        auto t0 = clk::now();
        for (int y = 0; y < H; y += step)
            for (int x = 0; x < W; x += step) {
                float z = depth[y * W + x];
                octomap::point3d p((x - cx) * z / fx, (y - cy) * z / fy, z);
                tree.updateNode(p, true);
            }
        tree.updateInnerOccupancy();
        double ms = ms_since(t0);
        printf("OctoMap occupied-only step=%d (%6d 点/帧): %8.2f ms -> %6.1f fps\n",
               step, (W / step) * (H / step), ms, 1000.0 / ms);
    }

    // ---------- 3) 2D 占据栅格投影（5cm 栅格，640x480 全量） ----------
    {
        const float res = 0.05f;
        std::unordered_map<long long, int> grid;
        grid.reserve(200000);
        auto t0 = clk::now();
        for (int y = 0; y < H; ++y)
            for (int x = 0; x < W; ++x) {
                float z = depth[y * W + x];
                float px = (x - cx) * z / fx, py = (y - cy) * z / fy;
                int gx = (int)std::floor(px / res), gy = (int)std::floor(py / res);
                long long key = ((long long)gx << 32) ^ (unsigned)gy;
                grid[key] = 1;
            }
        double ms = ms_since(t0);
        printf("2D 占据栅格 5cm 全量 %d 点/帧: %8.2f ms -> %6.1f fps (grid cells=%zu)\n",
               W * H, ms, 1000.0 / ms, grid.size());
    }
    return 0;
}
