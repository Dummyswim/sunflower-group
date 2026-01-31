#!/usr/bin/env bash
set -euo pipefail

export PYTHON_BIN=${PYTHON_BIN:-"/home/hanumanth/Documents/pyvirtualenv/venv/bin/python"}
export ENABLE_SIDECAR=${ENABLE_SIDECAR:-1}
export AUTO_SHUTDOWN_AT=${AUTO_SHUTDOWN_AT:-02}
export FUT_EXCHANGE_SEGMENT=${FUT_EXCHANGE_SEGMENT:-"NSE_FNO"}
export FUT_OUTPUT_DIR=${FUT_OUTPUT_DIR:-"/home/hanumanth/Documents/sunflower-group/scalperentry/data"}
export FUT_TICKS_PATH=${FUT_TICKS_PATH:-"${FUT_OUTPUT_DIR}/fut_ticks_candles.csv"}
export FUT_SIDECAR_PATH=${FUT_SIDECAR_PATH:-"${FUT_OUTPUT_DIR}/fut_candles.csv"}
export TB_USE_FUT_FLOW=${TB_USE_FUT_FLOW:-"1"}
export TB_FUT_SIDECAR_PATH=${TB_FUT_SIDECAR_PATH:-"/home/hanumanth/Documents/sunflower-group/scalperentry/data/fut_candles.csv"}
export TB_FUT_FLOW_STALE_SEC=${TB_FUT_FLOW_STALE_SEC:-"180"}

MAIN_SCRIPT=${MAIN_SCRIPT:-dhan.py}
MAIN_ARGS=${MAIN_ARGS:-}
ALLOW_TRADEBRAIN_STANDALONE=${ALLOW_TRADEBRAIN_STANDALONE:-0}

LOG_DIR=${LOG_DIR:-logs}
MAIN_LOG=${MAIN_LOG:-"${LOG_DIR}/dhan.log"}
SIDECAR_LOG=${SIDECAR_LOG:-"${LOG_DIR}/fut_sidecar.log"}

MAIN_PID=0
MAIN_PGID=0
SIDECAR_PID=0
SIDECAR_PGID=0
SELF_PGID=0
_CLEANUP_DONE=0

log() {
    printf '%s | %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" >&2
}

get_pgid() {
    ps -o pgid= -p "$1" 2>/dev/null | tr -d ' '
}

kill_group() {
    local pgid="$1"
    if [ -n "$pgid" ] && [ "$pgid" -gt 0 ] 2>/dev/null; then
        if [ "$SELF_PGID" -gt 0 ] && [ "$pgid" -eq "$SELF_PGID" ]; then
            return
        fi
        kill -TERM -"$pgid" 2>/dev/null || true
    fi
}

kill_pid() {
    local pid="$1"
    if [ -n "$pid" ] && [ "$pid" -gt 0 ] 2>/dev/null; then
        kill -TERM "$pid" 2>/dev/null || true
    fi
}

wait_brief() {
    local pid="$1"
    local i
    for i in 1 2 3 4 5; do
        if ! kill -0 "$pid" 2>/dev/null; then
            return 0
        fi
        sleep 0.2
    done
    return 1
}

cleanup() {
    local from_trap="${1:-}"

    if [ "$_CLEANUP_DONE" -eq 1 ]; then
        return
    fi
    _CLEANUP_DONE=1
    trap - INT TERM HUP

    log "cleanup: stopping processes"

    # Kill process groups first (covers grandchildren)
    if [ "$SIDECAR_PGID" -gt 0 ]; then
        kill_group "$SIDECAR_PGID"
    fi
    if [ "$MAIN_PGID" -gt 0 ]; then
        kill_group "$MAIN_PGID"
    fi

    # Fallback to individual PIDs
    if [ "$SIDECAR_PID" -gt 0 ]; then
        kill_pid "$SIDECAR_PID"
    fi
    if [ "$MAIN_PID" -gt 0 ]; then
        kill_pid "$MAIN_PID"
    fi

    # Wait briefly, then force if still alive
    if [ "$SIDECAR_PID" -gt 0 ]; then
        if ! wait_brief "$SIDECAR_PID"; then
            kill -KILL "$SIDECAR_PID" 2>/dev/null || true
        fi
    fi
    if [ "$MAIN_PID" -gt 0 ]; then
        if ! wait_brief "$MAIN_PID"; then
            kill -KILL "$MAIN_PID" 2>/dev/null || true
        fi
    fi

    # Reap
    wait "$SIDECAR_PID" 2>/dev/null || true
    wait "$MAIN_PID" 2>/dev/null || true

    if [ "$from_trap" = "from_trap" ]; then
        exit 130
    fi
}

trap 'cleanup from_trap' INT TERM HUP

mkdir -p "$LOG_DIR"
SELF_PGID=$(get_pgid $$)

if [ -n "$AUTO_SHUTDOWN_AT" ]; then
    shutdown_at="$AUTO_SHUTDOWN_AT"
    if [[ "$shutdown_at" =~ ^[0-9]{1,2}$ ]]; then
        shutdown_at="${shutdown_at}:00"
    fi
    if ! sudo -n shutdown -h "$shutdown_at" >/dev/null 2>&1; then
        log "warn: failed to schedule shutdown at $shutdown_at (sudo -n)"
    else
        log "shutdown scheduled at $shutdown_at"
    fi
fi

if [ "$ENABLE_SIDECAR" = "1" ]; then
    log "starting futures sidecar"
    if command -v setsid >/dev/null 2>&1; then
        setsid "$PYTHON_BIN" futures_sidecar.py >>"$SIDECAR_LOG" 2>&1 &
    else
        "$PYTHON_BIN" futures_sidecar.py >>"$SIDECAR_LOG" 2>&1 &
    fi
    SIDECAR_PID=$!
    SIDECAR_PGID=$(get_pgid "$SIDECAR_PID")
    log "sidecar pid=$SIDECAR_PID pgid=$SIDECAR_PGID"
else
    log "sidecar disabled (ENABLE_SIDECAR=$ENABLE_SIDECAR)"
fi

if [ "$MAIN_SCRIPT" = "tradebrain.py" ] && [ "$ALLOW_TRADEBRAIN_STANDALONE" != "1" ]; then
    log "warn: tradebrain.py has no standalone runner; switching to dhan.py"
    MAIN_SCRIPT="dhan.py"
    MAIN_ARGS=""
    MAIN_LOG=${MAIN_LOG:-"${LOG_DIR}/dhan.log"}
fi

log "starting main (${MAIN_SCRIPT})"
if command -v setsid >/dev/null 2>&1; then
    setsid "$PYTHON_BIN" "$MAIN_SCRIPT" $MAIN_ARGS >>"$MAIN_LOG" 2>&1 &
else
    "$PYTHON_BIN" "$MAIN_SCRIPT" $MAIN_ARGS >>"$MAIN_LOG" 2>&1 &
fi
MAIN_PID=$!
MAIN_PGID=$(get_pgid "$MAIN_PID")
log "main pid=$MAIN_PID pgid=$MAIN_PGID"

set +e
wait "$MAIN_PID"
main_rc=$?
set -e

cleanup
exit "$main_rc"
