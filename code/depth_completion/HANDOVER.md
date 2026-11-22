# HANDOVER — 深度补全 / RGB-D 实时链路（2026-11-22 17:10）

> 交接给下一个人/代理。**以本文为准**，旧的 PromptDA / LingBot / DAv2 相关内容已全部删除。
> 工作目录：PC `D:/项目4/orangepi_aipro`，板端 `~/code/depth_completion`

---

## 0. 一句话

相机 U3V 直连 → **纯 CPU 传统深度补全**（中值去离散点 + 时序不稳定区替换 + 归一化卷积先验 + 导向滤波 + 时序 EMA）→ **手写 Foxglove WebSocket** 发布 RGB / 深度 / 点云。**不用 NPU、不用任何学习模型。**

---

## 1. 硬件与环境

| 项 | 值 |
|---|---|
| 板子 | OrangePi AI Pro 20T（Ascend 310B1，3 核 ARM，23 GB 内存） |
| 地址 | `192.168.137.163`（USB 网络共享，PC 侧 192.168.137.1） |
| 用户 / 密码 | `HwHiAiUser` / `Mind@123`（sudo 用 `echo Mind@123 | sudo -S ...`） |
| SSH key | `D:/项目4/orangepi_aipro/tools/opi_key` |
| 相机 | Hikrobot MV-DT01SDU（USB `2bdf:0001`），ToF 640×480 + RGB 1536×1280 |
| 相机 SDK | `~/Mv3dRgbdSDK`（1.4.1.4，只在标定/改参数时用；实时链路走直连 U3V） |

踩过的环境坑：
- 板子 3 核，跑 ATC/编译会把 SSH 拖死（`Exceeded MaxStartups`）甚至 OOM 重启。benchmark 前先 `uptime` 确认 load。
- 板端 `bisheng`（Ascend 编译器）可能被别的会话拉起占满 3 核，实测 fps 直接腰斩。

---

## 2. 文件地图（`code/depth_completion/`）

```
deploy/
  pd_live.py       主程序：采集线程 + 补全主循环 + 发布线程
  pd_core.py       算法：ClassicCompleter / AutoExposure / grade / colorize / fuse
  pd_capture.py    U3V 直连（libusb）+ payload 解码
  foxglove_ws.py   Foxglove WebSocket 服务端（零依赖）+ protobuf 编码
  pd_*.py          相机诊断小工具（抓包/时序/曝光/寄存器探测），按需用
convert/           相机标定与 SDK 抓帧工具（get_rgbd_calib / fetch_sdk_frames / set_align ...）
data/              相机抓的原始帧（color.jpg / depth16.png）
results/           rgbd_calib.json（标定，别删）
README.md          §16 = 当前方案
HANDOVER.md        本文
```

板端同名目录 + `/tmp/run_live.sh`（常驻启动脚本，见 §7）。

---

## 3. 相机 / 数据格式（最容易踩）

### 3.1 不要写相机配置寄存器
用户在 **PC 端 HiViewer** 里设「动态曝光」，我们直连 U3V 时**只允许**写两个寄存器：
`0x030804=1`（使能出流）、`SIRM+0x04=1`；停止时置 0。别的都不要写，否则会覆盖 HiViewer 的设置（曾经把相机写挂过）。
写挂了怎么救：SDK `MV3D_RGBD_Execute("DeviceReset")`（恢复出厂，会丢 HiViewer 设置）；或拔插 USB。

### 3.2 payload 两种格式（都要能吃）
| 总长 | 结构 |
|---|---|
| `6,512,688` | 48B 头 + **640×480 uint16 深度** + **3×1536×1280 平面 RGB** |
| `6,205,488` | 48B 头 + **640×480 uint8 深度（256 mm/LSB）** + 3×1536×1280 平面 RGB |

- RGB 是 **planar**（R/G/B 各一个 1536×1280 平面），不是交错；拼好再 `cvtColor(RGB2BGR)`。
- 深度解析见 `pd_capture.decode_payload()`：块 ≥ 614,400 → uint16；否则当 uint8 `*256`。
- 解码正确性判据：相邻像素差 |dx| 应该是几十 mm；几百就说明解错了。

### 3.3 U3V 读取的两个坑
1. **不要用 ctypes 切片**：`bytes(buf[:sz])` 拷 6.5 MB 会一直持 GIL，整条流水线慢 10 倍。用 `np.frombuffer(buf, np.uint8, count=sz).tobytes()`。
2. USB3 Vision 一帧 = 52B leader + payload + 36B trailer；一个 payload 通常一次 bulk 传完（约 30 ms）。

---

## 4. 算法（`pd_core.ClassicCompleter`）

```
median 3×3  ->  与中值差 >180 mm 的像素丢弃（去椒盐/飞点）
            ->  8×8 块级时序不稳定度（|Δd|>150 mm 的比例）
            ->  软权重 w = clip((unstable-0.36)/0.24, 0, 1)，d = d*(1-w) + prior*w
            ->  prior = 1/8 归一化卷积（空块用 1/32 兜底，否则出黑洞）
            ->  半分辨率导向滤波（RGB 亮度当引导，boxFilter 自实现）
            ->  时序 EMA α=0.45，|Δd|>400 mm 立即重置
```

要点：
- 低信噪区（窗帘后空腔那种）**中值救不了**，必须靠时序不稳定度判定 + 平滑面替换；用硬阈值会每帧 on/off 跳变（曾导致点云/颜色闪烁），所以现在是软权重。
- 导向滤波用 OpenCV 的 `ximgproc.guidedFilter` 在板端要 70 ms；自己用 `boxFilter` 写（半分辨率）只要 7 ms。
- 相机 RGB 1536×1280 → 640×480 的 `INTER_AREA` 要 13 ms，**放在采集线程里做**，不占主循环。

可调参数（`ClassicCompleter.__init__`）：`median_k` / `outlier_mm` / `eps` / `radius` / `temporal` / `div` / `unstable_th` / `diff_mm`。

---

## 5. Foxglove（`foxglove_ws.py`，零依赖手写）

### 5.1 协议必知
- 连 `ws://<board>:8765`，子协议 `foxglove.websocket.v1`，**没有鉴权**。
- 控制面是 JSON 文本帧（`serverInfo` / `advertise` / `subscribe` / `unsubscribe`）。
- **数据面二进制帧必须是**：`0x01`(MessageData) + `subId(4B LE)` + `timestamp(8B LE)` + payload。
  **少这 1 个字节就会报 `invalid wire type 4 at offset 1`**（整条消息错位 1 字节）——血泪教训。
- `advertise` 里 `encoding=protobuf` 时，`schema` 必须是 **base64 的 FileDescriptorSet**，不是 .proto 文本。现在发的就是 FDS（含 `builtin_interfaces/Time.proto` 依赖，和官方结构一致）。

### 5.2 频道
| Topic | 类型 | 说明 |
|---|---|---|
| `/camera/rgb` | foxglove.CompressedImage | 640×480 JPEG（已调色对齐 HiViewer） |
| `/depth/colored` | foxglove.CompressedImage | **补全后**深度伪彩 |
| `/depth/depth16` | foxglove.CompressedImage | 补全后 16bit PNG（原始值） |
| `/points` | foxglove.PointCloud | 补全后点云，`frame_id=rgb`，stride 4（19,200 点） |

所有消息 `frame_id` 统一 `rgb`，**不依赖 TF**。3D 面板 Frame 填 `rgb`。

### 5.3 面板颜色抖动
3D 面板默认「最小值/最大值 = auto」，每帧按当前 z 范围重映射彩虹色 → 看起来颜色乱跳。把 min/max 固定成 `1` / `5` 即可。

---

## 6. 实测数字（板子空闲时）

| 项 | 数值 |
|---|---|
| 单帧处理（640×480 全流程） | 41–45 ms |
| 端到端 | 11–19 fps（**上限由相机曝光决定**） |
| 补洞前后有效率 | 99.9% → 100% |
| 点云 | 19,200 点/帧，每 0.3 s，约 225 KB |
| 16bit PNG | 每 0.2 s 一帧，约 250–300 KB |

> 相机是动态曝光，暗环境下曝光时间自动拉长 → 出帧率掉到 5–12 fps。想上 25 fps 需在 HiViewer 把曝光上限压到 ~35 ms（画面亮度靠 `pd_core.AutoExposure` 软件增益补）。

---

## 7. 常用命令

```bash
# 常驻启动（板端）
pkill -f "deploy/pd_live.py"; sleep 1
setsid nohup /tmp/run_live.sh >/dev/null 2>&1 </dev/null &
tail -f /tmp/pd_live.log

# /tmp/run_live.sh 内容
cd /home/HwHiAiUser/code/depth_completion
exec python3 deploy/pd_live.py --port 8765 --depth-every 0.2 --cloud-every 0.3 --seconds 0

# 抓一帧原始 RGB + 深度（不碰相机配置）
python3 deploy/pd_raw.py /tmp/raw        # 输出 raw_rgb.png / raw_depth16.png

# 看帧率与传输耗时
python3 deploy/pd_timing.py

# 回 PC（在 PC 端执行）
scp -i D:/项目4/orangepi_aipro/tools/opi_key HwHiAiUser@192.168.137.163:/tmp/raw_rgb.png .
```

参数：`--seconds 0` 常驻、`--no-fill` 发原始 ToF、`--temporal 0` 关时序平滑、`--cloud-stride 2` 点云加密、`--no-fill` 用于对比。

---

## 8. 已知问题 / 下一步

1. **帧率上限在相机**：动态曝光下 5–19 fps 波动；要稳定 25+ fps 得压曝光上限（HiViewer 侧）。
2. **补全质量**：低信噪区目前用平滑面替换（细节丢失），可改成多尺度 + 置信度加权再试。
3. **性能还有 ~15 ms 余量**：导向滤波降到 1/4 分辨率（-4 ms）、先验两帧共用（-7 ms）、EMA/where 原地化（-5 ms）。
4. 彩色发布前的 `grade()`（gamma 0.75 + 饱和度 1.6）是为了对齐 HiViewer 观感；相机 ISP 能开的话可以去掉。

---

## 9. 本日删除记录（别再找这些了）

- 学习类补全模型全部删除：LingBot-Depth（31.5 MB）、PromptDA 权重缓存（95.8 MB）、DAv2 ONNX（48.3 MB）、板端所有 OM（约 600 MB）。
- 旧的融合脚本（fusion_live/v2/v3/v4、run_live_compare、live_u3v ...）、ATC 转换脚本、各类 npy 中间结果。
- ⚠️ `ascendc_ops/` 是 **Qwen3.8-27B 的自研 AscendC 算子**（另一个项目，32 个源文件），一度被误删，已从板端 `~/ascendc_ops.tar` 恢复源码；构建产物（约 53 MB）需在其目录下重新 `./run.sh` 生成。

---

## 10. 变更史（简）

| 时间 | 事件 |
|---|---|
| 2026-11-21 | 接手时是 yolo/DAv2 先验 + cv2 融合那一套 |
| 2026-11-22 上午 | 尝试 PromptDA-Small-Transparent（ONNX→OM→NPU）；发现 prompt 必须完整、ATC 需改 Resize 属性 + onnxsim 静态化 |
| 2026-11-22 15:30 | PromptDA 板端跑通：196×252 → 35 ms/帧，端到端 16.2 fps |
| 2026-11-22 16:40 | Foxglove 手写服务端打通（修掉 MessageData 漏 1 字节的 bug） |
| 2026-11-22 17:00 | 按用户要求**放弃所有学习模型**，改纯 CPU 传统算法；模型权重/代码/结果全部删除 |
| 2026-11-22 17:10 | 软权重替换 + 固定色标建议；本文档成稿 |
