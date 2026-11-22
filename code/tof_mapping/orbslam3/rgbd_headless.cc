// rgbd_headless.cc 鈥?ORB-SLAM3 RGB-D headless runner锛堟棤 Pangolin 绐楀彛锛屾澘绔窇鍒?楠岃瘉鐢級
//
// 鐢ㄦ硶锛?//   ./rgbd_headless <ORBvoc.txt> <TUM_RGBD.yaml> <assoc.txt> [max_frames]
//
// assoc.txt 姣忚锛?rgb_timestamp rgb_path depth_timestamp depth_path   锛圱UM rgbd_dataset 鏍囧噯鏍煎紡锛?// 杈撳嚭锛氭瘡甯ц窡韪姸鎬?鑰楁椂銆佸钩鍧囪窡韪抚鐜囷紝浠ュ強杞ㄨ抗淇濆瓨鍒?trajectory_rgbd.txt
//
// 缂栬瘧锛堝湪 ORB_SLAM3 婧愮爜鐩綍涓嬶紝鍏?make 鍑?lib/libORB_SLAM3.so锛夛細
//   g++ -O3 -std=c++11 rgbd_headless.cc -I include -I include/CameraModels -I Thirdparty/Sophus \
//       -I /usr/include/eigen3 -L lib -lORB_SLAM3 -lopencv_core -lopencv_imgproc -lopencv_imgcodecs \
//       -o rgbd_headless
#include <iostream>
#include <algorithm>
#include <fstream>
#include <chrono>
#include <vector>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <System.h>

using namespace std;

int main(int argc, char **argv) {
    if (argc < 4) {
        cerr << "usage: " << argv[0] << " <ORBvoc.txt> <settings.yaml> <assoc.txt> [max_frames]\n";
        return 1;
    }
    const string vocab = argv[1];
    const string settings = argv[2];
    const string assoc_file = argv[3];
    size_t max_frames = (argc > 4) ? (size_t)atoi(argv[4]) : 0;

    ifstream fin(assoc_file);
    if (!fin) { cerr << "cannot open assoc file: " << assoc_file << "\n"; return 1; }
    vector<string> rgb_files, depth_files;
    vector<double> rgb_ts, depth_ts;
    while (!fin.eof()) {
        string l;
        getline(fin, l);
        if (l.empty() || l[0] == '#') continue;
        istringstream iss(l);
        double rts, dts;
        string rp, dp;
        if (!(iss >> rts >> rp >> dts >> dp)) continue;
        rgb_ts.push_back(rts); rgb_files.push_back(rp);
        depth_ts.push_back(dts); depth_files.push_back(dp);
    }
    fin.close();
    if (max_frames > 0 && rgb_files.size() > max_frames) {
        rgb_files.resize(max_frames); depth_files.resize(max_frames);
        rgb_ts.resize(max_frames); depth_ts.resize(max_frames);
    }
    cout << "sequence frames: " << rgb_files.size() << endl;

    // 鏃?viewer锛氱涓変釜鍙傛暟 false
    ORB_SLAM3::System SLAM(vocab, settings, ORB_SLAM3::System::RGBD, false);
    float total_ms = 0;
    size_t tracked = 0, lost = 0;
    auto t_all0 = chrono::steady_clock::now();
    for (size_t i = 0; i < rgb_files.size(); i++) {
        cv::Mat imRGB = cv::imread(rgb_files[i], cv::IMREAD_UNCHANGED);
        cv::Mat imD = cv::imread(depth_files[i], cv::IMREAD_UNCHANGED);
        if (imRGB.empty() || imD.empty()) {
            cerr << "skip frame " << i << " (read fail)" << endl;
            continue;
        }
        if (imRGB.channels() == 3 && imRGB.type() != CV_8UC3) {
            imRGB.convertTo(imRGB, CV_8UC3);
        }
        auto t0 = chrono::steady_clock::now();
        Sophus::SE3f Tcw = SLAM.TrackRGBD(imRGB, imD, rgb_ts[i]);
        auto t1 = chrono::steady_clock::now();
        float ms = chrono::duration_cast<chrono::microseconds>(t1 - t0).count() / 1000.0f;
        total_ms += ms;
        auto state = SLAM.GetTrackingState();
        if (state == 2 /*OK*/ || state == 3 /*RECENTLY_LOST*/) tracked++; else lost++;
        if (i < 10 || i % 20 == 0)
            cout << "frame " << i << " ts=" << rgb_ts[i] << " track " << ms
                 << " ms state=" << (int)state << endl;
    }
    double wall = chrono::duration_cast<chrono::milliseconds>(
                      chrono::steady_clock::now() - t_all0).count() / 1000.0;
    cout << "\nRESULT frames=" << rgb_files.size() << " tracked_state=" << tracked
         << " lost=" << lost << " avg_track=" << (total_ms / max<size_t>(1, rgb_files.size()))
         << " ms -> " << (1000.0 * rgb_files.size() / std::max(1.0f, total_ms)) << " fps(track-only)"
         << " | wall=" << wall << " s -> " << (rgb_files.size() / std::max(0.001, wall)) << " fps(end-to-end)"
         << endl;
    SLAM.SaveTrajectoryTUM("trajectory_rgbd.txt");
    SLAM.SaveKeyFrameTrajectoryTUM("keyframes_rgbd.txt");
    SLAM.Shutdown();
    return 0;
}
