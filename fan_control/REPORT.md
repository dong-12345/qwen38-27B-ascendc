# OrangePi AI Pro 20T 风扇降噪 —— 实测报告（v2 曲线）

- 板卡：OrangePi AI Pro 20T（Ascend 310B4，Ubuntu 22.04，内核 5.10.0+）
- 软件：npu-smi 25.2.0，固件 7.3.7.0.b507，CANN 8.0.0
- 时间：2026-11-19（v1 初版 45% 档；v2 按需求改为 40% 起步 + 55→80°C 线性升速）
- 目标：把"风扇常满转"降下来，同时保证温度安全

## 一、结论摘要

| 项目 | 结果 |
|---|---|
| 当前曲线 | 45–55°C 固定 **40%**；55→80°C **线性 40%→100%**；>80°C 强制 100%+告警 |
| 当前运行状态 | systemd 服务 `opi-fan` enabled+active，**40% 占空比**，NPU **46–47°C**（空闲） |
| 降噪幅度（占空比口径） | **100% → 40%**（PWM 占空比降低 60%），空闲温度与满转时（45–48°C）基本相同 |
| 负载表现 | 3×CPU 满载 90s：温度 48→52°C，40% 档压得住（未到 55°C 升速线），kill 后热惯性峰值 53°C |
| 控制通道 | 板载**无 hwmon/thermal/sysfs pwm 节点**，`npu-smi` 是唯一可行通道 |
| 安全底线 | >80°C 立即 100% + ALERT 已实测；最低 30%；线性区 5% 步进去抖 |
| 遗留问题 | 板载 journald 异常（用文件日志）；无转速反馈；Health=Alarm 为板卡既有状态 |

## 二、改造前现状

```
npu-smi info -t pwm-mode         : manual
npu-smi info -t pwm-duty-ratio   : 100      <- 风扇常满转，噪音源
npu-smi info -t temp -i 0        : 45–48°C（空闲，AICore 0%，功耗 0W）
npu-smi info 的 Health           : Alarm（改造前后一致，属板卡既有状态）
```

`/opt/opi_test/fan/` 官方样例只有 `max.sh`（duty=100）和 `tesh.sh`（100%↔0% 反复切），
即"出厂默认满转"的来源，没有任何温控逻辑。

## 三、控制通道调研（为什么不用 hwmon）

| 检查 | 结果 |
|---|---|
| `/sys/class/hwmon/*/` | 仅 `bq25700_charger`、`tcpm_source_psy`，**无风扇 pwm/fan 节点** |
| `/sys/class/thermal/` | **空目录**（没有 thermal zone） |
| `/sys/devices/**/pwm*`、`*fan*` | 无 |
| 风扇驱动 | 内核模块 `pwm` + Ascend 驱动 ioctl，dmesg 显示 **PWM channel 2** |
| 转速反馈 | 无 tach；`npu-smi info -t sensors` 返回"不支持" |
| 温度来源 | `npu-smi info -t temp -i 0`（读温度不需 root，读写 PWM 需 root） |

结论：**hwmon 方案不可行**（不存在 pwm 节点），只能走 `npu-smi set -t pwm-duty-ratio`。

## 四、`npu-smi set` 生效性实测

| 设置 | 回读 | 结果 |
|---|---|---|
| duty 30% | 30 | OK（`Set pwm duty ratio 30 success`） |
| duty 50% | 50 | OK |
| duty 80% | 80 | OK |
| duty 100% | 100 | OK |
| 30% 持续 60s | 每 10s 回读均 30，mode 恒 manual | **未被固件覆盖** |

dmesg 佐证（PWM 确实作用于通道 2）：

```
[ascend] [pwm] [pwm_ioctl_get_duty_ratio 185] channel: 2 ratio: 30
[ascend] [pwm] [pwn_ioctl_get_duty_ratio_adjust_mode 270] Get duty ratio adjust mode success. (mode=0)
```

## 五、v2 转速曲线

| NPU 温度 | 占空比 | 说明 |
|---|---|---|
| < 42°C | 30% | 超低温档（实测空闲 45–47°C，基本不触发） |
| 42–45°C | 滞回保持 | 不主动升档；当前 >40% 则回落到 40% |
| **45–55°C** | **40%** | 固定档 |
| **55–80°C** | **线性 40%→100%** | `40 + (T−55)×2.4`，约每 2°C +5% |
| ≥ 80°C | 100% | — |
| > 80°C | 100% + ALERT | 持续高温每 60s 最多一条 |

线性区 5% 步进去抖（温度微抖不频繁改速）；42–45°C 滞回避免 30/40 边界抖动。
**安全底线**：最低 30%、永不写 0%；>80°C 强制 100% 并告警；温度读不到保持上次占空比；
每次写入回读校验、失败重试 2 次；自动确保 `pwm-mode=manual`；systemd `Restart=always`。

## 六、实测数据（v2）

### 6.1 空闲稳态（真实温度）

```
+15s 46C 40% | +30s 46C 40% | +45s 46C 40% | +60s 47C 40% | +75s 46C 40% | +90s 47C 40%
```

40% 占空比稳定，温度 46–47°C（v1 用 45% 时同为 45–47°C，即再降 5% 占空比温度不变）。

### 6.2 加负载测试（3× `yes` 跑满 3 核，90 秒）

```
负载中：+10s 48C | +20s 49C | +30s 50C | +40s 51C | +50s 51C | +60s 51C | +70s 51C | +80s 52C | +90s 52C   占空比全程 40%
kill 后：52→53°C（热惯性峰值）→ 回落，占空比仍 40%
```

**结论**：CPU 满载下 40% 档把温度压在 ≤53°C，未触发 55°C 线性升速；说明 40% 起步档足够安静且安全。
（该板 NPU 功耗字段恒为 0.0W，传感器不支持，无法用功耗佐证负载强度。）

### 6.3 曲线逐点验证（注入温度，服务停止，`--once`，已在板上实测）

| 注入温度 | 实际占空比 | 说明 |
|---|---|---|
| 41°C | 30% | 超低温档 |
| 44°C | 30% | 滞回保持（当前 30% 不升；若当前 40% 则保持 40%） |
| 46°C | 40% | 45–55 固定档 |
| 50°C | 40% | 固定档 |
| 56°C | 40% | 理论 42%，5% 步进保持 40% |
| 58°C | 47% | 线性区 |
| 60°C | 52% | 线性区 |
| 65°C | 64% | 线性区 |
| 70°C | 76% | 线性区 |
| 75°C | 88% | 线性区 |
| 79°C | 97% | 线性区 |
| 80°C | 100% | 线性区终点 |
| 84°C | 100% + `[ALERT] NPU 温度 84C > 80C` | 安全强制 |
| 70°C（从 100%） | 76% | 降温回落 ✅ |

### 6.4 异常处理

- 温度连续 3 次读取失败 → 保持 `LAST_DUTY`/状态文件值，返回码 4，不写 0%；
- 写入回读不一致 → 重试 2 次，仍失败保持原值并写 ERROR；
- 服务重启 → 自动按真实温度接管（实测 76%→40%）。

## 七、噪音主观描述

- 改造前：占空比 100%，风扇持续满转，是主要持续噪音源。
- 改造后：占空比 **40%**（PWM 降低 60%），温度与满转时基本相同（46–47°C）。
- **说明**：远程无法听测分贝，以上为占空比口径的客观佐证；实际听感请在板边确认
  （本板无 tach，无法远程确认转速，首次建议目视/听感确认风扇在低速转而非停转）。
- 如需更安静：可把 45–55°C 档降到 35%（`FAN_HOLD_DUTY=35`），但需实测温度；本方案不建议低于 30%。

## 八、安装 / 卸载 / 回滚

### 已完成部署

```
/home/HwHiAiUser/fan_control/{fan_control.sh,opi-fan.service,README.md}   # 板子侧
/etc/systemd/system/opi-fan.service                                       # systemd unit
```

```bash
sudo cp /home/HwHiAiUser/fan_control/opi-fan.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now opi-fan
```

### 卸载 / 回滚

```bash
# A. 卸载服务，恢复官方常满转
sudo systemctl disable --now opi-fan
sudo rm -f /etc/systemd/system/opi-fan.service
sudo systemctl daemon-reload
sudo npu-smi set -t pwm-mode -d 0
sudo npu-smi set -t pwm-duty-ratio -d 100

# B. 完全交还固件自动控速
sudo systemctl disable --now opi-fan
sudo rm -f /etc/systemd/system/opi-fan.service
sudo systemctl daemon-reload
sudo npu-smi set -t pwm-mode -d 1     # auto 模式
```

> 卸载后风扇停在最后一次占空比，不会自动回 100%。

### 一行状态命令

```bash
sudo /home/HwHiAiUser/fan_control/fan_control.sh --status
```

输出示例（实测）：

```
===== OrangePi AI Pro 20T 风扇状态 =====
NPU 温度    : 47 C   (安全阈值 80C, 线性区步进 5%)
PWM 模式    : manual
当前占空比  : 40%
转速曲线    : <42C:30% | 42-45C:滞回 | 45-55C:40% | 55-80C:线性 40%->100% | >80C:100%+告警
服务状态    : systemd opi-fan = active
```

## 九、已知问题与风险提示

1. **310B4 长期高温影响寿命/稳定性**：SoC 长期 >75°C 会加速老化、可能降频。v2 曲线在 75°C 时为 88%、80°C 才 100%；
   若发现长期 75°C 以上，建议把线性区终点提前（`FAN_RAMP_END_TEMP=75` 即 75°C 到 100%）。
2. **最低 30%、40% 起步**：实测空闲 46–47°C 稳定，不要为安静把最低档改到 0–20%。
3. **温度来源单一**：板载无 thermal zone，只有 NPU 芯片温度，读不到主板/环境温度；
   若 `npu-smi` 异常，脚本保持上次占空比并告警，不会失控降速。
4. **journald 异常**：板上 journald 自 11 月 16 日起无新记录，`journalctl -u opi-fan` 为空，
   日志以 `/var/log/opi_fan.log` 为准；v1 测试日志已归档 `/var/log/opi_fan.log.v1-test-20261119`。
5. **Health=Alarm**：改造前后均 Alarm，与风扇无关，建议另行排查。
6. **无转速反馈**：只能靠占空比+噪音判断，首次建议耳测确认。
7. **断电恢复**：`pwm-mode` 掉电可能复位为固件默认（通常满转，安全侧失败）；systemd 开机自启后 5s 内强制 manual。
8. **pwm 设置需 root**：服务以 root 运行（仅 npu-smi 与写日志）。

## 十、文件清单

| 路径 | 说明 |
|---|---|
| `fan_control/fan_control.sh` | 调速器主脚本（仓库 + 板子 `/home/HwHiAiUser/fan_control/`） |
| `fan_control/opi-fan.service` | systemd unit（板子 `/etc/systemd/system/opi-fan.service`） |
| `fan_control/README.md` | 安装/卸载/日志/手动调节/调试说明 |
| `fan_control/REPORT.md` | 本报告 |
