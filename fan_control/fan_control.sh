#!/bin/bash
# =============================================================================
# fan_control.sh - OrangePi AI Pro 20T (Ascend 310B4) 温度自适应风扇调速器
#
# 控制通道: Ascend 驱动 PWM (channel 2)，只能通过 npu-smi 访问：
#   npu-smi set -t pwm-mode -d 0            # manual 模式
#   npu-smi set -t pwm-duty-ratio -d <N>    # N = 0..100
# 板载 sysfs 无 pwm/hwmon 节点，/sys/class/thermal 为空，温度只能由 npu-smi 读取。
#
# 转速曲线 (2026-11-19 第2版):
#   < 42 C          -> 30%          (超低温档，安静)
#   42 - 45 C       -> 滞回保持     (不升档；若当前 >40% 则回到 40%)
#   45 - 55 C       -> 40%          (固定档)
#   55 - 80 C       -> 线性递增 40% -> 100%   (每约 2C 提 5%)
#   >= 80 C         -> 100%         (> 80 C 同时写 ALERT)
#   NPU > 80 C      -> 立即 100% + 告警日志
# 线性区带 5% 步进去抖，避免温度微抖导致占空比频繁变化。
#
# 安全底线:
#   * 任何占空比不低于 30% (MIN_DUTY)，脚本永远不会写 0%
#   * NPU 温度 > 80 C 立即 100%，并写 ALERT 日志
#   * 温度读不到时保持上次占空比，并重试；绝不因此降为 0%
#   * 每次写入后回读校验，失败重试 2 次
#
# 用法:
#   sudo ./fan_control.sh --once                # 按当前温度调一次后退出
#   sudo ./fan_control.sh --loop [秒]           # 守护循环，默认 5 秒一次
#   sudo ./fan_control.sh --status              # 打印当前温度/占空比/模式/最近日志
#   ./fan_control.sh --help
#
# 日志: /var/log/opi_fan.log
# =============================================================================

set -u

# ----------------------------- 可配置参数 -----------------------------------
LOG_FILE="${FAN_LOG:-/var/log/opi_fan.log}"
STATE_FILE="${FAN_STATE:-/run/opi_fan.state}"
INTERVAL="${FAN_INTERVAL:-5}"        # --loop 采样间隔(秒)

MIN_DUTY="${FAN_MIN_DUTY:-30}"       # 最低占空比，安全底线
LOW_TEMP="${FAN_LOW_TEMP:-42}"       # 低于此温度用低温档 MIN_DUTY
HOLD_TEMP="${FAN_HOLD_TEMP:-45}"     # 42..45 为滞回保持区
HOLD_DUTY="${FAN_HOLD_DUTY:-40}"     # 45..55 固定档 = 40%
RAMP_START_TEMP="${FAN_RAMP_START_TEMP:-55}"   # 线性区起点温度
RAMP_END_TEMP="${FAN_RAMP_END_TEMP:-80}"       # 线性区终点温度(到 100%)
RAMP_START_DUTY="${FAN_RAMP_START_DUTY:-40}"
RAMP_END_DUTY="${FAN_RAMP_END_DUTY:-100}"
RAMP_STEP="${FAN_RAMP_STEP:-5}"      # 线性区最小变化步进(%)，去抖
SAFETY_TEMP="${FAN_SAFETY_TEMP:-80}" # 超过此温度强制 100% 并告警

TEMP_RETRIES=3                       # 温度读取重试次数
SET_RETRIES=2                        # 占空比写入重试次数
HEARTBEAT_CYCLES=$(( 300 / INTERVAL ))  # 无变化时每 ~5 分钟记一条心跳
[ "$HEARTBEAT_CYCLES" -lt 1 ] && HEARTBEAT_CYCLES=1

# npu-smi 绝对路径(systemd 环境 PATH 精简，优先解析一次)
NPU_SMI="$(command -v npu-smi 2>/dev/null || true)"
[ -z "$NPU_SMI" ] && NPU_SMI="/usr/local/sbin/npu-smi"
[ -x "$NPU_SMI" ] || NPU_SMI="/usr/local/bin/npu-smi"

LAST_DUTY=""   # 最近一次确认的占空比
LAST_TEMP=""
CHANGED=0
CYCLE=0
LAST_ALERT_TS=0   # 上次高温告警时间戳，避免循环模式下刷屏

# ------------------------------- 工具函数 -----------------------------------
ts() { date '+%Y-%m-%d %H:%M:%S'; }

log() {  # log LEVEL message
    local level="$1"; shift
    mkdir -p "$(dirname "$LOG_FILE")" 2>/dev/null || true
    printf '%s [%s] %s\n' "$(ts)" "$level" "$*" >>"$LOG_FILE" 2>/dev/null || true
    # 前台运行时也回显（systemd 下 stdout 进 journal）
    printf '%s [%s] %s\n' "$(ts)" "$level" "$*" >&2
}

die() { log "ERROR" "$*"; exit 1; }

need_root() {
    if [ "$(id -u)" -ne 0 ]; then
        echo "错误: 需要 root (sudo) 权限来调用 npu-smi set。" >&2
        exit 1
    fi
}

# 读当前 NPU 温度(°C)，失败返回空串
read_temp() {
    local t
    # 测试钩子：FAN_TEST_TEMP=<数字> 时用注入温度替代真实读数（仅用于验证曲线逻辑，
    # 正常 systemd 服务不设置此变量；README 的"调试"一节有用法）
    if [[ "${FAN_TEST_TEMP:-}" =~ ^[0-9]+$ ]]; then
        echo "$FAN_TEST_TEMP"
        return 0
    fi
    t=$("$NPU_SMI" info -t temp -i 0 2>/dev/null \
        | awk -F: '/Temperature/{gsub(/[^0-9]/,"",$2); print $2; exit}')
    case "$t" in
        ''|*[!0-9]*) echo "" ;;
        *) echo "$t" ;;
    esac
}

read_duty() {
    local d
    d=$("$NPU_SMI" info -t pwm-duty-ratio 2>/dev/null \
        | awk -F: '/pwm-duty-ratio/{gsub(/[^0-9]/,"",$2); print $2; exit}')
    case "$d" in
        ''|*[!0-9]*) echo "" ;;
        *) echo "$d" ;;
    esac
}

read_mode() {
    "$NPU_SMI" info -t pwm-mode 2>/dev/null \
        | awk -F: '/pwm-mode/{gsub(/[ \t\r]/,"",$2); print $2; exit}'
}

save_state() {
    printf '%s\t%s\t%s\n' "${LAST_DUTY:-}" "${LAST_TEMP:-}" "$(ts)" >"$STATE_FILE" 2>/dev/null || true
}

load_state_duty() {
    [ -r "$STATE_FILE" ] || { echo ""; return; }
    awk -F'\t' 'NR==1{print $1}' "$STATE_FILE" 2>/dev/null
}

# ------------------------------ 转速曲线 ------------------------------------
# 给定当前占空比 cur 与温度 t，返回目标占空比
decide_duty() {
    local cur="$1" t="$2" target diff span

    # 1) 安全区：>= 80C 一律 100%
    if [ "$t" -ge "$RAMP_END_TEMP" ]; then
        echo "$RAMP_END_DUTY"
        return
    fi

    # 2) 线性区 55..80C: 40% -> 100%
    if [ "$t" -ge "$RAMP_START_TEMP" ]; then
        span=$(( RAMP_END_TEMP - RAMP_START_TEMP ))
        target=$(( RAMP_START_DUTY + (t - RAMP_START_TEMP) * (RAMP_END_DUTY - RAMP_START_DUTY) / span ))
        [ "$target" -gt "$RAMP_END_DUTY" ] && target="$RAMP_END_DUTY"
        [ "$target" -lt "$RAMP_START_DUTY" ] && target="$RAMP_START_DUTY"
        diff=$(( target - cur )); [ "$diff" -lt 0 ] && diff=$(( -diff ))
        # 变化小于步进就不动，避免温度微抖导致频繁调节
        if [ "$diff" -lt "$RAMP_STEP" ]; then
            echo "$cur"
        else
            echo "$target"
        fi
        return
    fi

    # 3) 固定档 45..55C = 40%
    if [ "$t" -ge "$HOLD_TEMP" ]; then
        echo "$HOLD_DUTY"
        return
    fi

    # 4) 滞回保持区 42..45C: 不主动升档；若当前高于 40% 则回落到 40%
    if [ "$t" -ge "$LOW_TEMP" ]; then
        if [ "$cur" -gt "$HOLD_DUTY" ]; then
            echo "$HOLD_DUTY"
        else
            echo "$cur"
        fi
        return
    fi

    # 5) 超低温 < 42C
    echo "$MIN_DUTY"
}

# 确保 pwm-mode = manual(0)；失败返回非 0
ensure_manual_mode() {
    local m
    m=$(read_mode)
    [ "$m" = "manual" ] && return 0
    "$NPU_SMI" set -t pwm-mode -d 0 >/dev/null 2>&1
    m=$(read_mode)
    [ "$m" = "manual" ]
}

# 写占空比并回读校验
apply_duty() {
    local target="$1" attempt d
    for attempt in $(seq 0 "$SET_RETRIES"); do
        ensure_manual_mode || true
        "$NPU_SMI" set -t pwm-duty-ratio -d "$target" >/dev/null 2>&1
        sleep 0.2
        d=$(read_duty)
        if [ "$d" = "$target" ]; then
            return 0
        fi
        log "WARN" "设置占空比 ${target}% 回读为 '${d:-<空>}'，重试 $attempt/$SET_RETRIES"
        sleep 0.5
    done
    return 1
}

# ------------------------------ 单次调节 ------------------------------------
# 返回值: 0 正常已应用 / 4 温度不可用已保持 / 5 写入失败
run_once() {
    local temp attempt cur_duty target

    if [[ "${FAN_TEST_TEMP:-}" =~ ^[0-9]+$ ]]; then
        log "WARN" "TEST: 使用注入温度 ${FAN_TEST_TEMP}C (真实温度未读取)"
    fi

    # 1) 读温度（带重试）
    temp=""
    for attempt in $(seq 1 "$TEMP_RETRIES"); do
        temp=$(read_temp)
        [ -n "$temp" ] && break
        [ "$attempt" -lt "$TEMP_RETRIES" ] && sleep 1
    done

    if [ -z "$temp" ]; then
        # 温度读不到：保持上次占空比，绝不下调
        local last="${LAST_DUTY:-$(load_state_duty)}"
        log "WARN" "温度读取失败(${TEMP_RETRIES}次)；保持占空比 ${last:-未知}%，不调整"
        [ -n "$last" ] && LAST_DUTY="$last"
        return 4
    fi
    LAST_TEMP="$temp"

    # 2) 读当前占空比
    cur_duty=$(read_duty)
    [ -z "$cur_duty" ] && cur_duty="${LAST_DUTY:-$(load_state_duty)}"
    [ -z "$cur_duty" ] && cur_duty="$HOLD_DUTY"   # 未知时取 45..55C 档
    [ "$cur_duty" -lt "$MIN_DUTY" ] && cur_duty="$MIN_DUTY"   # 防止意外低档

    # 3) 安全告警（>80C 必须 100% 且写 ALERT）
    if [ "$temp" -gt "$SAFETY_TEMP" ]; then
        target=100
        if [ "$cur_duty" -ne 100 ]; then
            log "ALERT" "NPU 温度 ${temp}C > ${SAFETY_TEMP}C，强制 100% (原 ${cur_duty}%)"
            LAST_ALERT_TS=$(date +%s)
        elif [ $(( $(date +%s) - LAST_ALERT_TS )) -ge 60 ]; then
            log "ALERT" "NPU 温度 ${temp}C > ${SAFETY_TEMP}C，保持强制 100%"
            LAST_ALERT_TS=$(date +%s)
        fi
    else
        target=$(decide_duty "$cur_duty" "$temp")
        # 安全底线兜底（曲线本身已保证 >=30）
        [ "$target" -lt "$MIN_DUTY" ] && target="$MIN_DUTY"
        [ "$target" -gt 100 ] && target=100
    fi

    # 4) 应用
    if [ "$cur_duty" -ne "$target" ]; then
        if apply_duty "$target"; then
            LAST_DUTY="$target"
            log "INFO" "温度 ${temp}C 占空比 ${cur_duty}% -> ${target}% [$(read_mode)]"
            CHANGED=1
        else
            LAST_DUTY="$cur_duty"
            log "ERROR" "温度 ${temp}C 设置占空比 ${target}% 失败，维持 ${cur_duty}%"
            save_state
            return 5
        fi
    else
        LAST_DUTY="$cur_duty"
        # 无变化：仅心跳
        if [ $(( CYCLE % HEARTBEAT_CYCLES )) -eq 0 ]; then
            log "INFO" "温度 ${temp}C 占空比 ${target}% 保持"
        fi
    fi

    save_state
    return 0
}

# ------------------------------ 循环模式 ------------------------------------
run_loop() {
    local interval="${1:-$INTERVAL}"
    log "INFO" "启动守护循环: 间隔 ${interval}s, 曲线: <${LOW_TEMP}C=${MIN_DUTY}% | ${HOLD_TEMP}-${RAMP_START_TEMP}C=${HOLD_DUTY}% | ${RAMP_START_TEMP}-${RAMP_END_TEMP}C 线性 ${RAMP_START_DUTY}%->${RAMP_END_DUTY}% | >=${RAMP_END_TEMP}C=100%"
    trap 'log "INFO" "收到终止信号，退出守护循环(最后 ${LAST_TEMP:-?}C / ${LAST_DUTY:-?}%)"; exit 0' TERM INT
    while true; do
        CYCLE=$(( CYCLE + 1 ))
        run_once || true      # 单次失败不退出，下一轮重试
        sleep "$interval" &
        wait $! 2>/dev/null   # 可被 SIGTERM 立即打断
    done
}

# ------------------------------ 状态查看 ------------------------------------
show_status() {
    local temp duty mode
    temp=$(read_temp);  [ -z "$temp" ] && temp="N/A"
    duty=$(read_duty);  [ -z "$duty" ] && duty="N/A"
    mode=$(read_mode);  [ -z "$mode" ] && mode="N/A"
    echo "===== OrangePi AI Pro 20T 风扇状态 ====="
    echo "NPU 温度    : ${temp} C   (安全阈值 ${SAFETY_TEMP}C, 线性区步进 ${RAMP_STEP}%)"
    echo "PWM 模式    : ${mode}"
    echo "当前占空比  : ${duty}%"
    echo "转速曲线    : <${LOW_TEMP}C:${MIN_DUTY}% | ${LOW_TEMP}-${HOLD_TEMP}C:滞回 | ${HOLD_TEMP}-${RAMP_START_TEMP}C:${HOLD_DUTY}% | ${RAMP_START_TEMP}-${RAMP_END_TEMP}C:线性 ${RAMP_START_DUTY}%->${RAMP_END_DUTY}% | >${RAMP_END_TEMP}C:100%+告警"
    if systemctl is-active --quiet opi-fan 2>/dev/null; then
        echo "服务状态    : systemd opi-fan = active"
    else
        echo "服务状态    : systemd opi-fan = 未运行/未安装"
    fi
    if [ -r "$LOG_FILE" ]; then
        echo "--- 最近日志 (${LOG_FILE}) ---"
        tail -n 5 "$LOG_FILE" 2>/dev/null
    fi
}

usage() {
    cat <<EOF
OrangePi AI Pro 20T 自适应风扇调速器

用法:
  sudo ./fan_control.sh --once              # 按当前温度调节一次后退出
  sudo ./fan_control.sh --loop [秒]         # 守护循环(默认 5s 一次)
  sudo ./fan_control.sh --status            # 查看温度/占空比/模式/最近日志
  ./fan_control.sh --help                   # 本帮助

转速曲线:
  < ${LOW_TEMP}C: ${MIN_DUTY}% | ${LOW_TEMP}-${HOLD_TEMP}C: 滞回保持 | ${HOLD_TEMP}-${RAMP_START_TEMP}C: ${HOLD_DUTY}%
  ${RAMP_START_TEMP}-${RAMP_END_TEMP}C: 线性 ${RAMP_START_DUTY}% -> ${RAMP_END_DUTY}% | > ${RAMP_END_TEMP}C: 100% + ALERT
  NPU > ${SAFETY_TEMP}C 强制 100%；温度读不到时保持上次占空比；最低 ${MIN_DUTY}%。
环境变量可覆盖: FAN_LOG FAN_INTERVAL FAN_MIN_DUTY FAN_HOLD_DUTY FAN_LOW_TEMP
              FAN_HOLD_TEMP FAN_RAMP_START_TEMP FAN_RAMP_END_TEMP
              FAN_RAMP_START_DUTY FAN_RAMP_END_DUTY FAN_RAMP_STEP FAN_SAFETY_TEMP
EOF
}

# -------------------------------- 入口 --------------------------------------
main() {
    case "${1:-}" in
        --once)   need_root; run_once; rc=$?; [ "$rc" -eq 0 ] && exit 0; exit "$rc" ;;
        --loop)   need_root; shift; run_loop "${1:-$INTERVAL}" ;;
        --status) need_root; show_status ;;
        -h|--help|help) usage ;;
        '')       usage; exit 1 ;;
        *)        echo "未知参数: $1" >&2; usage; exit 1 ;;
    esac
}

main "$@"
