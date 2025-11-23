#!/usr/bin/env bash
#
# Launcher for:
#   1) Main index automation  -> run_main.py
#   2) NIFTY futures VWAP+CVD -> futures_vwap_cvd_sidecar.py  (optional)
#
# - Ensures logs directory exists
# - Redirects stdout/stderr as requested
# - Handles Ctrl+C and SIGTERM cleanly (kills children)
# - Optionally schedules a system shutdown at a fixed clock time (e.g. 02:00)
#
# Environment variables (optional):
#   PYTHON_BIN        : Python interpreter (default: python)
#   ENABLE_SIDECAR    : "1" to run sidecar, "0" to skip (default: 1)
#   AUTO_SHUTDOWN_AT  : HH:MM 24h time for system poweroff (default: 02:00; empty to disable)

set -euo pipefail

# ---------- CONFIGURABLE PATHS ----------
PYTHON_BIN="${PYTHON_BIN:-python}"
MAIN_SCRIPT="run_main.py"
SIDECAR_SCRIPT="futures_vwap_cvd_sidecar.py"
LOG_DIR="logs"
MAIN_LOG="${LOG_DIR}/run_main.log"
SIDECAR_LOG="${LOG_DIR}/fut_vwap_cvd_sidecar.log"

# Sidecar control (we still recommend running it, but you can disable via env)
ENABLE_SIDECAR="${ENABLE_SIDECAR:-1}"

# Auto-shutdown time (24h clock). Set to empty to disable.
AUTO_SHUTDOWN_AT="${AUTO_SHUTDOWN_AT:-02:00}"

mkdir -p "${LOG_DIR}"

MAIN_PID=""
SIDECAR_PID=""

cleanup() {
  echo "[$(date +'%F %T')] Cleanup requested, stopping children..." >&2

  # Kill main automation if running
  if [[ -n "${MAIN_PID}" ]] && kill -0 "${MAIN_PID}" 2>/dev/null; then
    echo "Stopping main automation (PID=${MAIN_PID})..." >&2
    kill "${MAIN_PID}" 2>/dev/null || true
  fi

  # Kill sidecar if running
  if [[ -n "${SIDECAR_PID}" ]] && kill -0 "${SIDECAR_PID}" 2>/dev/null; then
    echo "Stopping VWAP+CVD sidecar (PID=${SIDECAR_PID})..." >&2
    kill "${SIDECAR_PID}" 2>/dev/null || true
  fi

  # Wait briefly for children to exit
  if [[ -n "${MAIN_PID}" ]]; then
    wait "${MAIN_PID}" 2>/dev/null || true
  fi
  if [[ -n "${SIDECAR_PID}" ]]; then
    wait "${SIDECAR_PID}" 2>/dev/null || true
  fi

  echo "[$(date +'%F %T')] All children stopped. Exiting launcher." >&2
}

# Handle Ctrl+C and TERM from OS (reboot/shutdown or supervisor)
trap cleanup INT TERM

# ---------- OPTIONAL: schedule system shutdown ----------
if [[ -n "${AUTO_SHUTDOWN_AT}" ]]; then
  echo "[$(date +'%F %T')] Scheduling system shutdown at ${AUTO_SHUTDOWN_AT}..." >&2
  # Use sudo; if it fails (no permissions), continue running automation
  if sudo shutdown -h "${AUTO_SHUTDOWN_AT}" 2>/dev/null; then
    echo "[$(date +'%F %T')] Shutdown scheduled successfully for ${AUTO_SHUTDOWN_AT}." >&2
  else
    echo "[$(date +'%F %T')] WARNING: Failed to schedule shutdown. Run 'sudo shutdown -h ${AUTO_SHUTDOWN_AT}' manually if needed." >&2
  fi
fi

# ---------- START OPTIONAL SIDECAR ----------
if [[ "${ENABLE_SIDECAR}" == "1" ]]; then
  echo "[$(date +'%F %T')] Starting VWAP+CVD sidecar..." >&2
  # Append stdout+stderr to sidecar log
  #   python futures_vwap_cvd_sidecar.py >> logs/fut_vwap_cvd_sidecar.log
  "${PYTHON_BIN}" "${SIDECAR_SCRIPT}" >> "${SIDECAR_LOG}" 2>&1 &
  SIDECAR_PID=$!
  echo "[$(date +'%F %T')] VWAP+CVD sidecar PID=${SIDECAR_PID}" >&2
else
  echo "[$(date +'%F %T')] ENABLE_SIDECAR=0 → sidecar will NOT be started." >&2
fi

# ---------- START MAIN AUTOMATION ----------
echo "[$(date +'%F %T')] Starting main automation..." >&2
# Redirect stdout+stderr to run_main.log (overwrite each run)
#   python run_main.py &> logs/run_main.log
"${PYTHON_BIN}" "${MAIN_SCRIPT}" &> "${MAIN_LOG}" &
MAIN_PID=$!
echo "[$(date +'%F %T')] Main automation PID=${MAIN_PID}" >&2

# ---------- WAIT FOR MAIN AND THEN CLEAN UP ----------
MAIN_EXIT=0
if wait "${MAIN_PID}"; then
  MAIN_EXIT=$?
else
  MAIN_EXIT=$?
fi

echo "[$(date +'%F %T')] Main automation exited with code ${MAIN_EXIT}" >&2

# When main exits (normal or error), stop sidecar as well
if [[ -n "${SIDECAR_PID}" ]] && kill -0 "${SIDECAR_PID}" 2>/dev/null; then
  echo "[$(date +'%F %T')] Stopping VWAP+CVD sidecar after main exit..." >&2
  kill "${SIDECAR_PID}" 2>/dev/null || true
  wait "${SIDECAR_PID}" 2>/dev/null || true
fi

echo "[$(date +'%F %T')] Launcher exiting with code ${MAIN_EXIT}" >&2
exit "${MAIN_EXIT}"
