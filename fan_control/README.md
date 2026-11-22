# OrangePi AI Pro 20T（Ascend 310B4）风扇降噪方案（v2，40% 起步 + 线性升速）

把"风扇常转 100%"改为**温度自适应调速**：45–55°C 固定 **40%**，55→80°C **线性升到 100%**，
>80°C 强制 100% 并告警。当前板卡实测：**40% 占空比，NPU 46–47°C**。

## 1. 背景：这块板子的风扇是怎么控制的

实测结论（2026-11-19，板卡 Linux 5.10.0+ aarch64，npu-smi 25.2.0，固件 7.3.7.0.b507）：

| 检查项 | 结果 |
|---|---|
| `/sys/class/hwmon/*/` | 只有 `bq25700_charger`、`tcpm_source_psy`，**没有风扇 pwm/fan 节点** |
| `/sys/class/thermal/` | **空目录**，没有 thermal zone |
| `/sys/devices/**/pwm*`、`*fan*` | 无 |
| 风扇 PWM 驱动 | 内核模块 `pwm`，由 Ascend 驱动内部使用 **channel 2**（dmesg: `[pwm] channel: 2 ratio: 100`） |
| 唯一控制通道 | `npu-smi set -t pwm-mode -d 0` + `npu-smi set -t pwm-duty-ratio -d N` |
| 温度来源 | `npu-smi info -t temp -i 0`（读温度不需要 root；读/写 PWM 需要 root） |
| 转速反馈 | **无 tach 读数**，`npu-smi info -t sensors` 返回"不支持" |

即：**hwmon 方案不可行**（没有 pwm 节点），`npu-smi` 是唯一可行的控制/读取通道。
`/opt/opi_test/fan/` 里的官方样例只有 `max.sh`（mode 0 + duty 100）和 `tesh.sh`（100%↔0% 来回切），
正是"默认满载很吵"的来源。

## 2. 转速曲线（v2）

| NPU 温度 | 占空比 |
|---|---|
| < 42 °C | 30%（超低温档） |
| 42 – 45 °C | 滞回保持（不主动升档；若当前 >40% 则回落到 40%） |
| **45 – 55 °C** | **40%（固定档）** |
| **55 – 80 °C** | **线性递增：40% → 100%**（公式 `40 + (T-55)×2.4`，约每 2°C 升 5%） |
| ≥ 80 °C | 100% |
| > 80 °C | 100% + `ALERT` 日志 |

线性区实际取值（5% 步进去抖）：

| 温度 | 56 | 58 | 60 | 65 | 70 | 75 | 79 | 80 |
|---|---|---|---|---|---|---|---|---|
| 占空比 | 40* | 47 | 52 | 64 | 76 | 88 | 97 | 100 |

\* 56°C 理论值 42%，因 5% 步进（抗抖动）保持 40%，57°C 起跟随曲线。

- **去抖**：线性区只在新目标与当前值相差 ≥5% 时才写，避免温度微抖导致风扇频繁变速；
- **滞回**：42–45°C 为保持区，45% 档不会在边界反复跳 30/40；
- **安全底线**：
  - 任何情况下占空比不低于 **30%**，脚本不会写 0%；
  - NPU 温度 **> 80°C 立即强制 100%** 并写 `ALERT`（持续高温每 60s 最多一条）；
  - 温度连续 3 次读取失败时**保持上次占空比**并告警，绝不下调到 0；
  - 每次写占空比后**回读校验**，不一致重试 2 次，并自动确保 `pwm-mode=manual`。

## 3. 文件说明

| 文件 | 用途 |
|---|---|
| `fan_control.sh` | 调速器主脚本（`--once` / `--loop [秒]` / `--status`） |
| `opi-fan.service` | systemd unit（开机自启、崩溃自动重启） |
| `README.md` | 本文件 |
| `REPORT.md` | 实测报告 |

## 4. 安装（在板子上执行）

```bash
# 1) 拷贝文件（本机 Windows 执行）
scp -i "D:\项目4\orangepi_aipro\tools\opi_key" \
    fan_control.sh opi-fan.service HwHiAiUser@192.168.137.163:/home/HwHiAiUser/fan_control/

# 2) 板子上安装（以下在板子 shell 中）
chmod +x /home/HwHiAiUser/fan_control/fan_control.sh
sudo cp /home/HwHiAiUser/fan_control/opi-fan.service /etc/systemd/system/opi-fan.service
sudo systemctl daemon-reload
sudo systemctl enable --now opi-fan
sudo systemctl status opi-fan --no-pager
```

## 5. 常用命令

```bash
# 查看状态（温度 / 占空比 / 模式 / 服务 / 最近日志）—— 一行版
sudo /home/HwHiAiUser/fan_control/fan_control.sh --status

# 服务管理
sudo systemctl status opi-fan --no-pager     # 状态
sudo systemctl restart opi-fan               # 重启（改脚本后必须执行）
sudo systemctl stop opi-fan                  # 停止（停止后风扇保持最后占空比）
sudo systemctl disable --now opi-fan         # 停用 + 取消开机自启

# 日志
tail -f /var/log/opi_fan.log                 # 实时日志（推荐，本板实测唯一可靠来源）
grep ALERT /var/log/opi_fan.log              # 只看高温告警
journalctl -u opi-fan -f                     # systemd 日志（注意：本板 journald 异常，见下）

# 手动调节（先停服务，否则 5 秒内会被接管）
sudo systemctl stop opi-fan
echo Mind@123 | sudo -S npu-smi set -t pwm-mode -d 0
echo Mind@123 | sudo -S npu-smi set -t pwm-duty-ratio -d 60   # 手动 60%
sudo systemctl start opi-fan                 # 恢复自动

# 单次按当前温度调整（调试用）
sudo /home/HwHiAiUser/fan_control/fan_control.sh --once
```

> **注意**：本板 `systemd-journald` 虽为 active，但实测从 11 月 16 日起不再写入新记录，
> `journalctl -u opi-fan` 会显示 `-- No entries --`。请以 `/var/log/opi_fan.log` 为准。
> 日志只在"占空比变化"时写入 + 每 5 分钟一条心跳，正常每天几十行，无需 logrotate；
> 如需清理：`sudo truncate -s 0 /var/log/opi_fan.log`。

## 6. 卸载 / 回滚

```bash
# 方案 A：卸载服务，恢复官方"常满转"
sudo systemctl disable --now opi-fan
sudo rm -f /etc/systemd/system/opi-fan.service
sudo systemctl daemon-reload
sudo npu-smi set -t pwm-mode -d 0
sudo npu-smi set -t pwm-duty-ratio -d 100

# 方案 B：完全恢复到出厂（自动模式，由固件控速）
sudo systemctl disable --now opi-fan
sudo rm -f /etc/systemd/system/opi-fan.service
sudo systemctl daemon-reload
sudo npu-smi set -t pwm-mode -d 1        # 1 = auto（固件自动）
sudo npu-smi info -t pwm-mode
```

> 注意：卸载服务后风扇**停在最后一次的占空比**，不会自动回到 100%。需要"安静但安全"就手动
> `-d 100` 或 `-d 60`；需要交还固件控制就切 auto 模式。

## 7. 参数覆盖

脚本支持环境变量覆盖（改 `/etc/systemd/system/opi-fan.service` 的 `Environment=` 或直接改脚本）：

| 变量 | 默认 | 说明 |
|---|---|---|
| `FAN_INTERVAL` | 5 | `--loop` 采样秒数 |
| `FAN_MIN_DUTY` | 30 | 最低占空比（安全下限，不建议 <30） |
| `FAN_LOW_TEMP` | 42 | 低于此温度用 `FAN_MIN_DUTY` |
| `FAN_HOLD_TEMP` | 45 | 42–45 为滞回保持区，45–55 用 `FAN_HOLD_DUTY` |
| `FAN_HOLD_DUTY` | 40 | 45–55°C 固定档 |
| `FAN_RAMP_START_TEMP` / `FAN_RAMP_END_TEMP` | 55 / 80 | 线性区起止温度 |
| `FAN_RAMP_START_DUTY` / `FAN_RAMP_END_DUTY` | 40 / 100 | 线性区起止占空比 |
| `FAN_RAMP_STEP` | 5 | 线性区最小变化步进（去抖） |
| `FAN_SAFETY_TEMP` | 80 | 超过即强制 100% 并告警 |
| `FAN_LOG` | /var/log/opi_fan.log | 日志路径 |

## 8. 调试 / 验证（可选）

想在不加热板子的情况下验证曲线，可用 `FAN_TEST_TEMP` 注入温度（**仅调试用，正常服务不设置该变量**）：

```bash
sudo systemctl stop opi-fan      # 先停服务，避免 5 秒后被真实温度覆盖
for t in 41 44 46 50 56 58 60 65 70 75 79 80 84 70; do
  echo Mind@123 | sudo -S env FAN_TEST_TEMP=$t \
       /home/HwHiAiUser/fan_control/fan_control.sh --once >/dev/null
  echo "注入 ${t}C -> $(echo Mind@123 | sudo -S npu-smi info -t pwm-duty-ratio | grep -o '[0-9]*' | tail -1)%"
done
sudo systemctl start opi-fan
```

实测预期（2026-11-19 已在板上验证）：41→30%，44→30%，46→40%，50→40%，56→40%，
58→47%，60→52%，65→64%，70→76%，75→88%，79→97%，80→100%，84→100% 且日志出现
`[ALERT] NPU 温度 84C > 80C ...`，70→76%（降温回落）。

## 9. 风险提示

1. **310B 长期高温影响寿命与稳定性**：SoC 长期 >75°C 会加速老化、可能触发降频/告警。
   按 v2 曲线，75°C 时占空比为 88%，80°C 才到 100%；80–85°C 区间仍有充足散热，但若长期处于
   75°C 以上，建议把线性区终点提前（如 75°C 就 100%）——改 `FAN_RAMP_END_TEMP=75` 即可。
2. **最低 30% / 45–55°C 固定 40%**：实测空闲 46–47°C 时 40% 完全压得住（与之前 45% 时温度相同）。
   不要为了更安静把最低档改到 0–20%，本方案不支持。
3. **温度只有一个来源**（`npu-smi` NPU 芯片温度）：板载无 thermal zone，无法读取主板/环境温度。
   如果 `npu-smi` 服务异常，脚本会保持上次占空比并告警，不会失控降速。
4. **无转速反馈**：降噪效果用占空比数值佐证，无法读取 RPM；风扇是否真的转起来需要目视/听感确认。
5. **`pwm-mode` 掉电可能复位**：脚本每次调节前都会检查并重新置 manual；机器重启后若 systemd 未启动，
   风扇会回到固件默认行为（通常满转），属于安全侧失败。
6. **Health=Alarm**：本板 `npu-smi info` 的 Health 字段实测已是 Alarm（温度 48°C、功耗 0W 时即如此），
   与风扇方案无关，属板卡既有状态，建议另行排查。
