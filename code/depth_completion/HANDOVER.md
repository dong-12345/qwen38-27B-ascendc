# HANDOVER — 深度补全（PromptDA 方向）交接文档

> 生成时间：2026-11-22 12:xx（会话上下文已过长，交接给新子代理继续）
> 当前机器：PC (Windows) 工作目录 `D:\项目4\orangepi_aipro`
> 板子：OrangePi AI Pro 20T (Ascend 310B1)，IP `192.168.137.163`，用户 `HwHiAiUser`，
> SSH key `D:\项目4\orangepi_aipro\tools\opi_key`，sudo 密码 `Mind@123`（`echo Mind@123 | sudo -S ...`）

## 0. 最新任务（下一步要做的事）
1. **已删除所有 yolo 相关产物**（板端/PC 端 OM、ONNX、results、脚本；`find . -iname '*yolo*'` 为 0）。
2. 改用 **PromptDA-Small-Transparent，输入 640×480** 做深度补全（RGB + ToF 稀疏深度 prompt → 稠密米制深度）。
3. **不用 ROS**；在板端用纯 Python 实现 **Foxglove WebSocket** 发布（RGB / ToF 原始深度 / PromptDA 补全深度，可加点云），PC Foxglove 连 `ws://192.168.137.163:8765`。
4. U3V 高速读取（直连 libusb，30fps）→ PromptDA(NPU) → 补全 → Foxglove。

## 1. PromptDA 资产（已下载，PC）
- 模型仓库：`onnx-community/promptda_vits_transparent_hf`（HF mirror 可用：`HF_ENDPOINT=https://hf-mirror.com`）
- 已下载缓存：`D:\项目4\orangepi_aipro\tmp\promptda_hf\models--onnx-community--promptda_vits_transparent_hf\snapshots\b671c85c54f44caf16416213d221610c8f7e0379\`
  - `config.json`、`preprocessor_config.json`、`onnx/model.onnx`（100,440,808 B，fp32）
  - 同仓库还有 `model_fp16.onnx / model_int8.onnx / model_uint8.onnx` 等量化变体
- **下一步第一件事**：用 onnxruntime 打开 `model.onnx`，打印 inputs/outputs 名称与 shape（注意：`promptda` pip 包安装失败；直接用 onnxruntime 或 transformers 的 onnx 后端）。
- 权重/配置来源：`https://hf-mirror.com/onnx-community/promptda_vits_transparent_hf`（下载必须用 `huggingface_hub.hf_hub_download` + HF_ENDPOINT 镜像，`urlretrieve` 直链会 403）。

## 2. 板端现状（`~/code/depth_completion`）
- **相机**：Hikrobot MV-DT01SDU（USB `2bdf:0001`），ToF 640×480 C16 + RGB 1536×1280。
- **标定（已验证）**：`results/rgbd_calib.json`（ImageAlign=0 模式导出）：
  - 深度 K: fx=507.604 fy=507.702 cx=326.961 cy=243.043（640×480）
  - RGB K: fx=1005.958 fy=1006.076 cx=766.899 cy=634.596（1536×1280）
  - 畸变: k1=0.06203 k2=-0.08733 p1=-1.45e-4 p2=2.10e-4
  - 外参 depth→RGB: R≈I，t=[26.459, 0.228, 3.007] mm
  - 验证：投影 vs SDK 对齐深度 中位 5.4mm、83.5%<20mm
- **U3V 直连**：`/tmp/u3v_dump.py`（可能被 /tmp 清理，源码在 PC `tmp/u3v_dump.py`）可每秒收到约 28.7 帧 payload；libusb 需要 `sudo sh -c 'echo 2000 > /sys/module/usbcore/parameters/usbfs_memory_mb'`。
  - 正常 payload：6,512,688 B = 48B 头 + 640×480 C16(614,400) + 1536×1280 RGB 平面(5,898,240)
  - **当前异常**：用户改动态曝光后 payload 变成 **6,205,488 B**（连续稳定），旧布局对不上、也找不到可解码 JPEG → 需要自适应解析，或用 SDK 把参数设回已知格式（脚本：`convert/set_exposure_capture.py`、`convert/recover_capture.py`）
  - SDK 抓帧曾报 `start_ret=0x80060005 / fetch_ret=0x80060003`，需 USB 复位（authorized 0→1）后重试
- **SDK 安装**：`~/Mv3dRgbdSDK`（原厂 1.4.1.4），环境：
  ```
  export MVSDK=/home/HwHiAiUser/Mv3dRgbdSDK
  export LD_LIBRARY_PATH=$MVSDK/Libs/aarch64:$MVSDK/MVCC/Libs/aarch64:$LD_LIBRARY_PATH
  export PYTHONPATH=$MVSDK/Res/demo/python:$PYTHONPATH
  ```

## 3. 现有可用代码（PC + 板端同步）
- `deploy/fusion_live.py`：cv2 优化的补全融合（投影 1/4 分辨率、先验采样、值域锚定 LUT、接缝羽化、时序 EMA）；已支持矩形先验 (H,W)。
- `deploy/dump_raw_prior.py`：跑一帧「OM 推理 + 投影/采样 + 锚定 + 融合 + 合成洞 RMSE」，输出 npy + `hole_metrics_latest.json`。
- `deploy/live_u3v.py`：U3V 采集 + NPU + 融合 + 视频（yolo 参数名还在，可复用逻辑）。
- `convert/`：`get_rgbd_calib.py`、`fetch_sdk_frames.py`、`set_align.py`、`capture_seq.py`、`recover_capture.py`、`atc_convert.sh`。
- `analysis/make_standard_compare.py`：标准 6 面板对比图（必须含 ②原始深度估计 ③采样到深度帧 ④原始ToF ⑤锚定后先验 ⑥融合）。
- `results/standard_compare_now.png`：最近一次对比图（PromptDA 之前版本）。
- `README.md`：§12–§15 有标定/融合/并发/模型对比的完整历史数据（yolo 数字仅作历史参考，产物已删）。

## 4. 关键实测数据（历史参考）
- 310B1 NPU 实测：yolo26n 448=15.8ms/63fps、640×480=23.8ms/42fps、640=34.7ms/28.8fps、768=52.3ms/19fps；2 路并发总吞吐 +25~30%。
- 融合精度（值域锚定版）：大洞 966→220~322mm，小洞填充最优。
- 单模型有效算力约 0.9–1.0 TFLOPS（实测 cube 峰值 1.46 TF/s）。
- ATC 命令模板：`bash /tmp/atc_convert.sh <onnx> <out_name> "images:1,3,480,640"`（`/tmp/atc_convert.sh` 常被清理，源码 `convert/atc_convert.sh`）。

## 5. 下一步建议顺序（新代理）
1. 检查板子可达性 + 相机状态（先 USB 复位，再决定 SDK 还是 U3V 抓帧；把 payload 格式确定下来）。
2. 打开 PromptDA onnx，确认输入（图像 + 深度 prompt）名称/shape/归一化，用我们保存的帧（PC `code/depth_completion/data/live_1529/`、`data/live_new/`）跑通参考推理，输出米制深度并与 ToF 对比。
3. 固定 640×480 导出/整理 ONNX（若模型原生输入尺寸不同则按 preprocessor_config 处理），ATC 转 OM（fp16）并跑分。
4. 板端实时链路：U3V → PromptDA(NPU) → 补全 → Foxglove WebSocket 发布。
5. Foxglove：Python 实现（websockets 库或手写 WS）；发布 `foxglove.CompressedImage`（RGB / 深度 16bit PNG）与/或点云；端口 8765。
6. 出实测 fps/延迟 + 对比图（含原始 ToF 与 PromptDA 输出）。
---

## 2026-11-22 14:xx 新会话更新（PromptDA 方向，**以本段为准**）

### A. 相机 / U3V（踩过的坑，勿重复）
1. **不要再往相机写任何配置寄存器**。用户用 PC 端 HiViewer 设置「动态曝光」，写寄存器会覆盖它。
   代码里只保留两处写的必要动作：`0x030804=1` + `SIRM+0x04=1`（使能出流），停止时置 0。
   `pd_capture.U3V.start()` 里那段 `SIRM+0x18/0x20/0x24/0x28=0x76030...` 会改变相机模式，**已废弃**，按 `pd_minimal.py` 的做法只使能出流。
2. 写 `ExposureAuto(0x30B08)` 返回 `E_PRECONDITION`，`ExposureTime(0x30B04)` 在出流后直写无效 → **自动曝光只能靠 HiViewer 设置**，代码侧不要再碰。
3. 相机写挂后的救活办法：SDK `MV3D_RGBD_Execute("DeviceReset")`（恢复出厂，会丢 HiViewer 设置，慎用）；或拔插 USB。
4. **SSH 断连/重启原因**：ATC 编译占满 3 核 + 无 swap → OOM → `kbox` 看门狗重启。跑 ATC 前先 `swapon /swapfile`（16G 已建好），并 `nice -n 10 taskset -c 0,1` + `TE_PARALLEL_COMPILER=1` 留一核，否则 SSH 会 `Exceeded MaxStartups` 或直接重启。

### B. U3V payload（两种都在出现，解码器要都能吃）
| 总长 | 48B 头之后 | 说明 |
|---|---|---|
| 6,512,688 | 614,400（640×480 uint16）+ 5,898,240（3×1536×1280 平面 RGB） | 深度块是**整张 640×480 u16**（相邻像素差 |dx|≈63，可直接用）；**不要**再做偶奇列交织（|dx|≈494，是错的） |
| 6,205,488 | 307,200（320×480 u16）+ 5,898,240 | 半宽深度，横向 2× 拉伸到 640×480 |

RGB 是 **planar**（R / G / B 三个 1536×1280 平面），不是交错；拼好再 `cv2.cvtColor(RGB2BGR)`。

### C. PromptDA ONNX 的真实语义（重要）
- 图里 `ReduceMin/Max → Sub → Div`：**prompt 会按全场 min/max 归一化**，输出再 `× (max-min) + min` 还原。
- 因此 **prompt 必须完整**：prompt=0 的像素输出恒为 0（网络把 0 当 0 米）。稀疏点云直接喂不行，必须先把洞填掉（`pd_core.fill_prompt`，1/4 分辨率 Telea）。
- 输出是**米制**，尺度跟着 prompt 的 (max-min) 走（prompt 整体 ×0.5 → 输出 ×0.5）。

### D. ATC 转换的三个坑（已解决，脚本在 convert/）
1. `Resize` 的 `coordinate_transformation_mode=half_pixel` **CANN 不支持**（与尺寸是否静态无关）→ 改 `asymmetric`。
2. `/backbone/embeddings/Resize` 是 `mode=cubic` → CANN 也不支持 → 改 `linear`（位置编码插值，影响很小）。
3. 图里有 67 个 `Shape` + 118 个 `Gather` 等**动态形状算子**，ATC 会把它们留在 host 侧，NPU 只有 3.7 fps → 用 **onnxsim 固定形状**折叠（1156 → 749 节点，数值差 < 0.02 mm）。
   一键：`convert/promptda_for_atc.py <static.onnx> <out.onnx>`，再 `bash convert/atc_convert.sh ... 'pixel_values:1,3,H,W;prompt_depth:1,1,H,W'`。

### E. 已实测
- PC(CPU/onnxruntime)：392×518 单帧 0.6–0.9 s（仅参考）；精度 p50 ≈ 47–63 mm、MAE ≈ 190–250 mm（对 ToF 有效区）。
- 板端 `promptda_atc2.om`（未折叠版，54.5 MB）：**272 ms/帧（3.7 fps）**，输出与 PC 一致（p50 = 2.234）。
- 合成洞回填（72×72 块 ×6，GT=原 ToF）：Telea 填充 MAE 828 / p50 579 mm → PromptDA 精修 **657 / 339 mm**（-21% / -41%）。
- Foxglove WS（手写，零依赖）已验证：PC 连 `ws://192.168.137.163:8765`，收到 /camera/rgb (jpeg)、/depth/colored (jpeg)、/depth/completed (16UC1;png)。

### F. 下一步
1. onnxsim 版重转 OM，实测（目标 < 60 ms）。
2. 板端端到端跑 `deploy/pd_live.py`，出 fps/延迟。
3. 与 SDK `ImageType_Depth` 对照确认 raw 深度噪声是相机原始输出还是解码问题。
### E2. 最终实测（2026-11-22 15:40，已写入 README §16）
- NPU：196×252 OM = **34.5 ms**（28 fps 上限）；392×518 OM = 297 ms（3.4 fps）。
- 端到端：**16.2 fps**，延迟 p50 **56.6 ms**，相机出帧 18.8 fps，退出码 0。
- 精度：p50 69 mm(392) / 109 mm(196)；合成洞回填 Telea 828/579 → PromptDA 657/339 mm。
- Foxglove：/camera/rgb、/depth/colored、/depth/completed（16UC1;png）三频道验证通过。
- **性能关键**：U3V 读包用 `bytes(buf[:sz])` 会持 GIL 6.5MB 拷贝 → 全链路慢 10 倍；改 `np.frombuffer().tobytes()` 后 1.6 fps → 16.2 fps。预处理的 INTER_AREA 改 INTER_LINEAR：25 ms → 4 ms。
