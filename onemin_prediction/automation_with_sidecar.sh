#!/usr/bin/env bash
#
# Launcher for:
#   1) Main index automation  -> run_main.py
#   2) NIFTY futures VWAP+CVD -> futures_vwap_cvd_sidecar.py  (optional)
#
# - Ensures logs directory exists
# - Redirects stdout/stderr as requested
# - Handles Ctrl+C, SIGTERM, and SIGHUP cleanly (kills children + their groups)
# - Supports start|stop|restart|status operations
# - Optionally schedules a system shutdown at a fixed clock time (e.g. 02:00)
#
# Environment variables (optional):
#   PYTHON_BIN        : Python interpreter (default: python)
#   ENABLE_SIDECAR    : "1" to run sidecar, "0" to skip (default: 1)
#   AUTO_SHUTDOWN_AT  : HH:MM 24h time for system poweroff (default: 02:00; empty to disable)
#   CLEAN_START       : "1" to stop stale matching processes before start (default: 1)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# Usage: ./automation_with_sidecar.sh [start|stop|restart|status]
ACTION="${1:-start}"

# ---------- CONFIGURABLE PATHS ----------
PYTHON_BIN="${PYTHON_BIN:-python}"
MAIN_SCRIPT="run_main.py"
SIDECAR_SCRIPT="futures_vwap_cvd_sidecar.py"
LOG_DIR="logs"
MAIN_LOG="${LOG_DIR}/run_main.log"
SIDECAR_LOG="${LOG_DIR}/fut_vwap_cvd_sidecar.log"
RUNTIME_DIR="runtime"
LAUNCHER_PID_FILE="${RUNTIME_DIR}/automation_with_sidecar.pid"
MAIN_PID_FILE="${RUNTIME_DIR}/run_main.pid"
SIDECAR_PID_FILE="${RUNTIME_DIR}/fut_vwap_cvd_sidecar.pid"

# Sidecar control
ENABLE_SIDECAR="${ENABLE_SIDECAR:-1}"

# Auto-shutdown time (24h clock). Set to empty to disable.
AUTO_SHUTDOWN_AT="${AUTO_SHUTDOWN_AT:-02:00}"
CLEAN_START="${CLEAN_START:-1}"

mkdir -p "${LOG_DIR}" "${RUNTIME_DIR}"

MAIN_PID=""
SIDECAR_PID=""

log() {
  echo "[$(date +'%F %T')] $*" >&2
}

pid_is_alive() {
  local pid="$1"
  [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null
}

get_pgid() {
  local pid="$1"
  ps -o pgid= "${pid}" 2>/dev/null | tr -d ' ' || true
}

read_pidfile() {
  local file="$1"
  if [[ -f "${file}" ]]; then
    tr -d '[:space:]' < "${file}" || true
  fi
}

write_pidfile() {
  local file="$1"
  local pid="$2"
  printf '%s\n' "${pid}" > "${file}"
}

clear_pidfiles() {
  rm -f "${LAUNCHER_PID_FILE}" "${MAIN_PID_FILE}" "${SIDECAR_PID_FILE}" || true
}

kill_pid_and_group() {
  local pid="$1"
  local label="${2:-process}"

  if ! pid_is_alive "${pid}"; then
    return 0
  fi

  local pgid
  pgid="$(get_pgid "${pid}")"
  log "Stopping ${label} (PID=${pid}${pgid:+ PGID=${pgid}})..."

  if [[ -n "${pgid}" ]]; then
    kill -TERM "-${pgid}" 2>/dev/null || true
  fi
  kill -TERM "${pid}" 2>/dev/null || true

  local i=0
  while pid_is_alive "${pid}" && (( i < 80 )); do
    sleep 0.1
    i=$((i + 1))
  done

  if pid_is_alive "${pid}"; then
    log "Force killing ${label} (PID=${pid}${pgid:+ PGID=${pgid}})..."
    if [[ -n "${pgid}" ]]; then
      kill -KILL "-${pgid}" 2>/dev/null || true
    fi
    kill -KILL "${pid}" 2>/dev/null || true
  fi
}

stop_pattern_in_project() {
  local pattern="$1"
  local label="$2"
  local pid=""
  local cmd=""

  while IFS= read -r line; do
    pid="${line%% *}"
    cmd="${line#* }"
    [[ -z "${pid}" ]] && continue
    [[ "${pid}" == "$$" ]] && continue
    log "Matched ${label}: PID=${pid} CMD=${cmd}"
    kill_pid_and_group "${pid}" "${label}"
  done < <(pgrep -af "${pattern}" || true)
}

status_report() {
  local pattern="$1"
  local title="$2"
  local any=0
  local pid=""
  local row=""

  log "${title}"
  printf '  %-8s %-8s %-6s %-10s %-6s %s\n' "PID" "PPID" "TTY" "ELAPSED" "STAT" "CMD" >&2
  while IFS= read -r line; do
    pid="${line%% *}"
    [[ -z "${pid}" ]] && continue
    [[ "${pid}" == "$$" ]] && continue
    pid_is_alive "${pid}" || continue
    row="$(ps -o pid=,ppid=,tty=,etime=,stat=,cmd= -p "${pid}" 2>/dev/null || true)"
    [[ -n "${row// }" ]] || continue
    echo "  ${row}" >&2
    any=1
  done < <(pgrep -af "${pattern}" || true)

  if [[ "${any}" -eq 0 ]]; then
    echo "  (none)" >&2
  fi
}

stop_all() {
  local pid=""

  pid="$(read_pidfile "${MAIN_PID_FILE}")"
  [[ -n "${pid}" ]] && kill_pid_and_group "${pid}" "main automation (pidfile)"

  pid="$(read_pidfile "${SIDECAR_PID_FILE}")"
  [[ -n "${pid}" ]] && kill_pid_and_group "${pid}" "futures sidecar (pidfile)"

  # Catch leftovers started from this project directory.
  stop_pattern_in_project "python .*run_main.py" "run_main.py"
  stop_pattern_in_project "python .*futures_vwap_cvd_sidecar.py" "futures_vwap_cvd_sidecar.py"
  stop_pattern_in_project "bash .*automation_with_sidecar.sh" "automation_with_sidecar.sh"

  clear_pidfiles
}

cleanup() {
  # Optional argument: "from_trap" if called from signal handler
  local from_trap="${1:-}"

  log "Cleanup requested, stopping children..."

  if [[ -n "${MAIN_PID}" ]]; then
    kill_pid_and_group "${MAIN_PID}" "main automation"
  fi
  if [[ -n "${SIDECAR_PID}" ]]; then
    kill_pid_and_group "${SIDECAR_PID}" "VWAP+CVD sidecar"
  fi

  clear_pidfiles
  log "All children stopped."

  # If called from a signal handler, terminate the script here
  if [[ "${from_trap}" == "from_trap" ]]; then
    exit 130   # 128+2 = SIGINT; generic "killed by signal" code
  fi
}

start_launcher() {
  if [[ "${CLEAN_START}" == "1" ]]; then
    log "CLEAN_START=1, stopping stale processes first..."
    stop_all
  fi

  write_pidfile "${LAUNCHER_PID_FILE}" "$$"

  # Handle Ctrl+C (INT), TERM, and HUP (terminal close)
  trap 'cleanup "from_trap"' INT TERM HUP

  # ---------- OPTIONAL: schedule system shutdown ----------
  if [[ -n "${AUTO_SHUTDOWN_AT}" ]]; then
    log "Scheduling system shutdown at ${AUTO_SHUTDOWN_AT}..."
    # Use sudo in non-interactive mode to avoid password prompt.
    # If not allowed, print a warning and continue without blocking.
    if sudo -n shutdown -h "${AUTO_SHUTDOWN_AT}" 2>/dev/null; then
      log "Shutdown scheduled successfully for ${AUTO_SHUTDOWN_AT}."
    else
      log "WARNING: Failed to schedule shutdown (no sudo -n perms). Run 'sudo shutdown -h ${AUTO_SHUTDOWN_AT}' manually if needed."
    fi
  fi

  # ---------- START OPTIONAL SIDECAR ----------
  if [[ "${ENABLE_SIDECAR}" == "1" ]]; then
    log "Starting VWAP+CVD sidecar..."
    "${PYTHON_BIN}" "${SIDECAR_SCRIPT}" >> "${SIDECAR_LOG}" 2>&1 &
    SIDECAR_PID=$!
    write_pidfile "${SIDECAR_PID_FILE}" "${SIDECAR_PID}"
    log "VWAP+CVD sidecar PID=${SIDECAR_PID}"
  else
    log "ENABLE_SIDECAR=0 -> sidecar will NOT be started."
  fi

  # ---------- START MAIN AUTOMATION ----------
  log "Starting main automation..."
  "${PYTHON_BIN}" "${MAIN_SCRIPT}" &>> "${MAIN_LOG}" &
  MAIN_PID=$!
  write_pidfile "${MAIN_PID_FILE}" "${MAIN_PID}"
  log "Main automation PID=${MAIN_PID}"

  # ---------- WAIT FOR MAIN AND THEN CLEAN UP ----------
  local main_exit=0
  if wait "${MAIN_PID}"; then
    main_exit=0
  else
    main_exit=$?
  fi

  log "Main automation exited with code ${main_exit}"

  # Normal cleanup (not from trap): also stops sidecar
  cleanup

  log "Launcher exiting with code ${main_exit}"
  exit "${main_exit}"
}

case "${ACTION}" in
  start)
    start_launcher
    ;;
  stop)
    stop_all
    log "Status after stop:"
    status_report "python .*run_main.py|python .*futures_vwap_cvd_sidecar.py|bash .*automation_with_sidecar.sh" "Project processes"
    ;;
  restart)
    stop_all
    start_launcher
    ;;
  status)
    status_report "python .*run_main.py|python .*futures_vwap_cvd_sidecar.py|bash .*automation_with_sidecar.sh" "Project processes"
    ;;
  *)
    echo "Usage: $0 [start|stop|restart|status]" >&2
    exit 2
    ;;
esac
