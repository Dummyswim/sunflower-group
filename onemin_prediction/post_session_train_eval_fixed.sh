#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$SCRIPT_DIR}"
cd "$PROJECT_ROOT"

# --- inputs ---
TRAIN_LOG_PATH="${TRAIN_LOG_PATH:-data/train_log_v3_schema_v5.jsonl}"
LIVE_LOG_PATH="${LIVE_LOG_PATH:-data/train_log_v3_live.jsonl}"

POST_REBUILD_MANIFEST="${POST_REBUILD_MANIFEST:-1}"
POST_FETCH_MISSING="${POST_FETCH_MISSING:-0}"
POST_BOOTSTRAP_TRAINLOG="${POST_BOOTSTRAP_TRAINLOG:-1}"
POST_VERIFY_COUNTS="${POST_VERIFY_COUNTS:-1}"

CACHE_GLOB="${CACHE_GLOB:-data/intraday_cache/INDEX_*_1m.csv}"
CACHE_MANIFEST_PATH="${CACHE_MANIFEST_PATH:-data/cache_manifest.jsonl}"
CACHE_OUT_DIR="${CACHE_OUT_DIR:-data/intraday_cache}"
HOLIDAY_FILE="${HOLIDAY_FILE:-data/nse_holidays_2025.txt}"
CACHE_START_DATE="${CACHE_START_DATE:-2025-01-01}"
CACHE_END_DATE="${CACHE_END_DATE:-$(date +%Y-%m-%d)}"
CACHE_SYMBOL="${CACHE_SYMBOL:-primary:IDX_I:13}"
CACHE_SECURITY_ID="${CACHE_SECURITY_ID:-13}"
CACHE_EXCHANGE_SEGMENT="${CACHE_EXCHANGE_SEGMENT:-IDX_I}"
CACHE_INSTRUMENT="${CACHE_INSTRUMENT:-INDEX}"

# --- archive today's live log (before training) ---
ARCHIVE_DIR="${ARCHIVE_DIR:-data/archive}"
mkdir -p "$ARCHIVE_DIR"
ARCHIVE_DATE="$(date +%Y%m%d)"
ARCHIVE_PATH="${ARCHIVE_DIR}/train_log_${ARCHIVE_DATE}.jsonl"
ARCHIVE_SRC="${LIVE_LOG_PATH}"
if [ ! -f "$ARCHIVE_SRC" ]; then
  ARCHIVE_SRC="$TRAIN_LOG_PATH"
fi
if [ -f "$ARCHIVE_SRC" ]; then
  cp "$ARCHIVE_SRC" "$ARCHIVE_PATH"
  echo "[POST] Archived ${ARCHIVE_SRC} -> ${ARCHIVE_PATH}"
else
  echo "[POST] Archive skipped (no source log found)"
fi

# --- runtime artifacts (keep logs out of bundles) ---
mkdir -p runtime
export FEATURE_LOG="${FEATURE_LOG:-runtime/feature_log.csv}"
export FEATURE_LOG_HIST="${FEATURE_LOG_HIST:-runtime/feature_log_hist.csv}"
export SIGNALS_PATH="${SIGNALS_PATH:-runtime/signals.jsonl}"

# --- bundle system ---
export MODEL_BUNDLES_DIR="${MODEL_BUNDLES_DIR:-trained_models/bundles}"
export MODEL_PRODUCTION_LINK="${MODEL_PRODUCTION_LINK:-trained_models/production}"

# Schema paths (base + policy)
export FEATURE_SCHEMA_COLS_PATH="${FEATURE_SCHEMA_COLS_PATH:-data/feature_schema_cols.json}"
export POLICY_SCHEMA_COLS_PATH="${POLICY_SCHEMA_COLS_PATH:-trained_models/production/policy_schema_cols.json}"
export POLICY_MOVE_PATH="${POLICY_MOVE_PATH:-trained_models/production/policy_move.json}"
MOVE_HEAD_ATR_MULT="${MOVE_HEAD_ATR_MULT:-0.35}"
MOVE_HEAD_MIN_ROWS="${MOVE_HEAD_MIN_ROWS:-5000}"
MOVE_HEAD_MAX_ROWS="${MOVE_HEAD_MAX_ROWS:-200000}"

# --- cache manifest + fetch (optional) ---
if [ "${POST_REBUILD_MANIFEST}" = "1" ]; then
  echo "[POST] Rebuilding cache manifest -> ${CACHE_MANIFEST_PATH}"
  /home/hanumanth/Documents/pyvirtualenv/venv/bin/python cache_manifest.py \
    --cache_glob "$CACHE_GLOB" \
    --out "$CACHE_MANIFEST_PATH" \
    --start_date "$CACHE_START_DATE" --end_date "$CACHE_END_DATE" \
    --skip_weekends \
    --holiday_file "$HOLIDAY_FILE"
fi

if [ "${POST_FETCH_MISSING}" = "1" ]; then
  echo "[POST] Fetching missing/partial cache days..."
  /home/hanumanth/Documents/pyvirtualenv/venv/bin/python fetch_missing_from_manifest.py \
    --manifest "$CACHE_MANIFEST_PATH" \
    --start_date "$CACHE_START_DATE" --end_date "$CACHE_END_DATE" \
    --skip_weekends \
    --holiday_file "$HOLIDAY_FILE" \
    --include_partial \
    --symbol "$CACHE_SYMBOL" \
    --security_id "$CACHE_SECURITY_ID" \
    --exchange_segment "$CACHE_EXCHANGE_SEGMENT" \
    --instrument "$CACHE_INSTRUMENT" \
    --out_dir "$CACHE_OUT_DIR" \
    --overwrite
fi

# --- bootstrap train log from cache (optional) ---
if [ "${POST_BOOTSTRAP_TRAINLOG}" = "1" ]; then
  echo "[POST] Bootstrapping train log -> ${TRAIN_LOG_PATH}"
  ALLOW_RULE_SIG_FALLBACK="${ALLOW_RULE_SIG_FALLBACK:-1}" \
  RULE_MIN_SIG="${RULE_MIN_SIG:-0.03}" \
  LANE_SCORE_MIN="${LANE_SCORE_MIN:-0.25}" \
  GATE_MARGIN_THR="${GATE_MARGIN_THR:-0.02}" \
  EMA_CHOP_HARD_MIN="${EMA_CHOP_HARD_MIN:-0.10}" \
  TREND_MIN_SIGNALS="${TREND_MIN_SIGNALS:-1}" \
  REQUIRE_SETUP="${REQUIRE_SETUP:-0}" \
  TAPE_REQUIRED_FOR_TRADING="${TAPE_REQUIRED_FOR_TRADING:-0}" \
  FEATURE_SCHEMA_COLS_PATH="$FEATURE_SCHEMA_COLS_PATH" \
  /home/hanumanth/Documents/pyvirtualenv/venv/bin/python bootstrap_trainlog_from_cache_v3.py \
    --cache_glob "$CACHE_GLOB" \
    --out_jsonl "$TRAIN_LOG_PATH" \
    --schema "$FEATURE_SCHEMA_COLS_PATH" \
    --symbol INDEX --bar_min 5 --horizon_min 10
fi

if [ "${POST_VERIFY_COUNTS}" = "1" ]; then
  echo "[POST] Verifying teacher_dir/tradeable counts..."
  /home/hanumanth/Documents/pyvirtualenv/venv/bin/python3 - <<'PY'
import json
from collections import Counter
path="data/train_log_v3_schema_v5.jsonl"
ctr_dir=Counter()
ctr_trade=Counter()
ctr_trade_dir=Counter()
try:
    with open(path,'r',encoding='utf-8') as f:
        for line in f:
            rec=json.loads(line)
            d=str(rec.get('teacher_dir',''))
            t=bool(rec.get('teacher_tradeable'))
            ctr_dir[d]+=1
            if t:
                ctr_trade['tradeable']+=1
                ctr_trade_dir[d]+=1
except FileNotFoundError:
    print(f"[POST][WARN] Train log not found: {path}")
    raise SystemExit(0)
print('teacher_dir counts:', ctr_dir)
print('tradeable rows:', ctr_trade)
print('tradeable by dir:', ctr_trade_dir)
PY
fi

# --- pick offline trainer ---
TRAINER=""
if [ -f "offline_train_regen_v2.py" ]; then
  TRAINER="offline_train_regen_v2.py"
else
  echo "[POST][ERROR] No offline trainer found. Expected offline_train_regen_v2.py"
  ls -la | sed -n '1,200p'
  exit 1
fi

echo "[POST] Running offline trainer: $TRAINER"
/home/hanumanth/Documents/pyvirtualenv/venv/bin/python "$TRAINER" --log "$TRAIN_LOG_PATH"

TRAIN_MOVE_HEAD="${TRAIN_MOVE_HEAD:-1}"
if [ "${TRAIN_MOVE_HEAD}" = "1" ] && [ -f "train_move_head.py" ]; then
  echo "[POST] Training move head (ATR mult=${MOVE_HEAD_ATR_MULT})..."
  /home/hanumanth/Documents/pyvirtualenv/venv/bin/python train_move_head.py \
    --log "$TRAIN_LOG_PATH" \
    --atr_mult "$MOVE_HEAD_ATR_MULT" \
    --min_rows "$MOVE_HEAD_MIN_ROWS" \
    --max_rows "$MOVE_HEAD_MAX_ROWS" \
    --out "$POLICY_MOVE_PATH"
else
  echo "[POST] Move head training skipped (TRAIN_MOVE_HEAD=${TRAIN_MOVE_HEAD})"
fi

echo "[POST] Active production bundle:"
if [ -L "$MODEL_PRODUCTION_LINK" ]; then
  readlink -f "$MODEL_PRODUCTION_LINK" || true
else
  echo "  (production is not a symlink yet) -> $MODEL_PRODUCTION_LINK"
fi

# --- show manifest (if present) ---
if [ -f "trained_models/production/manifest.json" ]; then
  echo "[POST] manifest.json:"
  cat "trained_models/production/manifest.json"
else
  echo "[POST] manifest.json not found in trained_models/production (ok if your trainer doesn't write it)."
fi

# --- HARD validation: schema vs model feature counts (prevents 16/26 regression) ---
/home/hanumanth/Documents/pyvirtualenv/venv/bin/python - <<'PY'
import json
from pathlib import Path
import xgboost as xgb

prod = Path("trained_models/production")
buy_path = prod / "policy_buy.json"
sell_path = prod / "policy_sell.json"
schema_path = prod / "policy_schema_cols.json"

if not buy_path.exists():
    raise SystemExit(f"[POST][ERROR] Missing: {buy_path}")
if not sell_path.exists():
    raise SystemExit(f"[POST][ERROR] Missing: {sell_path}")
if not schema_path.exists():
    raise SystemExit(f"[POST][ERROR] Missing: {schema_path}")

obj = json.loads(schema_path.read_text(encoding="utf-8"))
cols = obj.get("columns") or obj.get("feature_names") or obj.get("features") or obj
if not isinstance(cols, list):
    raise SystemExit("[POST][ERROR] policy_schema_cols.json has unexpected format (expected list under 'columns').")

b_buy = xgb.Booster()
b_buy.load_model(str(buy_path))
b_sell = xgb.Booster()
b_sell.load_model(str(sell_path))

print("[POST] BUY num_features:", b_buy.num_features())
print("[POST] SELL num_features:", b_sell.num_features())
print("[POST] Schema cols:", len(cols))

if b_buy.num_features() != len(cols):
    raise SystemExit("[POST][ERROR] BUY/schema mismatch. Refusing to continue.")
if b_sell.num_features() != len(cols):
    raise SystemExit("[POST][ERROR] SELL/schema mismatch. Refusing to continue.")

print("[POST] Policy models/schema OK ✅")
PY

# --- HARD validation: move head vs schema ---
if [ "${TRAIN_MOVE_HEAD:-0}" = "1" ]; then
  /home/hanumanth/Documents/pyvirtualenv/venv/bin/python - <<'PY'
import json
from pathlib import Path
import os
import xgboost as xgb

prod = Path(os.getenv("MODEL_PRODUCTION_LINK", "trained_models/production"))
move_path = Path(os.getenv("POLICY_MOVE_PATH", str(prod / "policy_move.json")))
schema_path = prod / "policy_schema_cols.json"

if not move_path.exists():
    raise SystemExit(f"[POST][ERROR] Missing move model: {move_path}")
if not schema_path.exists():
    raise SystemExit(f"[POST][ERROR] Missing: {schema_path}")

obj = json.loads(schema_path.read_text(encoding="utf-8"))
cols = obj.get("columns") or obj.get("feature_names") or obj.get("features") or obj
if not isinstance(cols, list):
    raise SystemExit("[POST][ERROR] policy_schema_cols.json has unexpected format (expected list under 'columns').")

booster = xgb.Booster()
booster.load_model(str(move_path))

print("[POST] MOVE num_features:", booster.num_features())
print("[POST] Schema cols:", len(cols))

if booster.num_features() != len(cols):
    raise SystemExit("[POST][ERROR] MOVE/schema mismatch. Refusing to continue.")

print("[POST] Move model/schema OK ✅")
PY
fi

# --- offline eval (non-fatal, but useful) ---
# --- score train log for calibrator ---
SCORED_LOG="${SCORED_LOG:-$LIVE_LOG_PATH}"
SCORE_OUT="${SCORED_LOG}"
SCORE_TMP=""
resolve_path() {
  local p="$1"
  if command -v readlink >/dev/null 2>&1; then
    readlink -f "$p" 2>/dev/null || echo "$p"
  else
    echo "$p"
  fi
}
TRAIN_LOG_REAL="$(resolve_path "$TRAIN_LOG_PATH")"
SCORED_LOG_REAL="$(resolve_path "$SCORED_LOG")"
if [ "$SCORED_LOG_REAL" = "$TRAIN_LOG_REAL" ]; then
  SCORE_TMP="${TRAIN_LOG_PATH%.jsonl}_scored.jsonl"
  SCORE_OUT="${SCORE_TMP}"
  echo "[POST][WARN] SCORED_LOG matches TRAIN_LOG_PATH; scoring to ${SCORE_OUT} then replacing ${SCORED_LOG}."
fi
export SCORED_LOG
echo "[POST] Scoring train log -> ${SCORE_OUT}"
set +e
/home/hanumanth/Documents/pyvirtualenv/venv/bin/python offline_score_trainlog.py --in "$TRAIN_LOG_PATH" --out "$SCORE_OUT" --overwrite
SCORE_STATUS=$?
set -e
if [ "${SCORE_STATUS}" -eq 0 ] && [ -n "${SCORE_TMP}" ]; then
  mv -f "$SCORE_TMP" "$SCORED_LOG"
  echo "[POST] Replaced ${SCORED_LOG} with scored output"
fi
SCORED_SOURCE="$SCORED_LOG"
if [ ! -s "$SCORED_SOURCE" ] && [ -s "$SCORE_OUT" ]; then
  SCORED_SOURCE="$SCORE_OUT"
fi
export SCORED_SOURCE

# --- sanity check: ensure scored log has p_success_raw + tradeable rows ---
echo "[POST] Sanity check: scored log coverage"
/home/hanumanth/Documents/pyvirtualenv/venv/bin/python - <<'PY'
import json
import os
from collections import Counter

path = os.environ.get("SCORED_SOURCE", "")
if not path:
    raise SystemExit("[POST][ERROR] SCORED_SOURCE missing")

ctr_dir = Counter()
ctr_trade = Counter()
ctr_trade_dir = Counter()
ctr_scored = Counter()
ctr_scored_trade = Counter()

try:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or not line.startswith("{"):
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            d = str(rec.get("teacher_dir", "")).upper()
            t = bool(rec.get("teacher_tradeable"))
            model = rec.get("model") or {}
            has_p = model.get("p_success_raw") is not None

            ctr_dir[d] += 1
            if t:
                ctr_trade["tradeable"] += 1
                ctr_trade_dir[d] += 1
            if has_p:
                ctr_scored["p_success_raw"] += 1
                if t:
                    ctr_scored_trade["p_success_raw_tradeable"] += 1
except FileNotFoundError:
    raise SystemExit(f"[POST][ERROR] Scored log not found: {path}")

print("[POST] scored log:", path)
print("[POST] teacher_dir counts:", dict(ctr_dir))
print("[POST] tradeable rows:", dict(ctr_trade))
print("[POST] tradeable by dir:", dict(ctr_trade_dir))
print("[POST] p_success_raw rows:", dict(ctr_scored))
print("[POST] p_success_raw tradeable rows:", dict(ctr_scored_trade))

if ctr_scored.get("p_success_raw", 0) == 0:
    raise SystemExit("[POST][ERROR] No p_success_raw in scored log (calibrator will skip)")
if ctr_scored_trade.get("p_success_raw_tradeable", 0) == 0:
    raise SystemExit("[POST][ERROR] No tradeable rows with p_success_raw (calibrator will skip)")
PY

# --- seed signals from cache if missing/empty ---
if [ ! -s "${SIGNALS_PATH}" ]; then
  echo "[POST] Seeding signals from ${SCORED_SOURCE} -> ${SIGNALS_PATH}"
  /home/hanumanth/Documents/pyvirtualenv/venv/bin/python - <<'PY'
import json
from pathlib import Path
import os

inp = Path(os.environ.get("SCORED_SOURCE", os.environ.get("SCORED_LOG", "")))
out = Path(os.environ.get("SIGNALS_PATH", "runtime/signals.jsonl"))
out.parent.mkdir(parents=True, exist_ok=True)

count = 0
with inp.open("r", encoding="utf-8", errors="ignore") as f_in, out.open("w", encoding="utf-8") as f_out:
    for line in f_in:
        line = line.strip()
        if not line or not line.startswith("{"):
            continue
        try:
            rec = json.loads(line)
        except Exception:
            continue
        model = rec.get("model") or {}
        rule = rec.get("rule_signals") or {}
        gates = rec.get("gates") or {}
        out_rec = {
            "pred_for": rec.get("ts_target_close"),
            "teacher_dir": rec.get("teacher_dir"),
            "teacher_tradeable": bool(rec.get("teacher_tradeable")),
            "policy_success_raw": model.get("p_success_raw"),
            "policy_success_calib": model.get("p_success_calib"),
            "rule_signal": rule.get("rule_sig"),
            "rule_direction": rec.get("teacher_dir"),
            "lane": gates.get("lane"),
            "gate_reasons": gates.get("gate_reasons", []),
        }
        f_out.write(json.dumps(out_rec, ensure_ascii=True) + "\n")
        count += 1

print(f"[POST] Wrote {count} signals -> {out}")
PY
fi

# --- offline eval (non-fatal, but useful) ---
if [ -f "${SIGNALS_PATH}" ] && [ -s "${SIGNALS_PATH}" ]; then
  echo "[POST] Offline eval…"
  /home/hanumanth/Documents/pyvirtualenv/venv/bin/python offline_eval.py || true
else
  echo "[POST] Offline eval skipped (signals file missing/empty): ${SIGNALS_PATH}"
fi

# --- build tradeable-only log for calibrator ---
SCORED_TRADEABLE_LOG="${SCORED_SOURCE%.jsonl}_tradeable.jsonl"
export SCORED_TRADEABLE_LOG
echo "[POST] Building tradeable-only log -> ${SCORED_TRADEABLE_LOG}"
/home/hanumanth/Documents/pyvirtualenv/venv/bin/python - <<'PY'
import json
import os

inp = os.environ.get("SCORED_SOURCE", os.environ.get("SCORED_LOG", ""))
out = os.environ.get("SCORED_TRADEABLE_LOG", "")
if not inp or not out:
    raise SystemExit("[POST][ERROR] Missing SCORED_LOG/SCORED_TRADEABLE_LOG env")
count = 0
with open(inp, "r", encoding="utf-8", errors="ignore") as f_in, open(out, "w", encoding="utf-8") as f_out:
    for line in f_in:
        line = line.strip()
        if not line or not line.startswith("{"):
            continue
        try:
            rec = json.loads(line)
        except Exception:
            continue
        if not rec.get("teacher_tradeable"):
            continue
        model = rec.get("model") or {}
        if model.get("p_success_raw") is None:
            continue
        f_out.write(line + "\n")
        count += 1
print(f"[POST] Wrote {count} tradeable rows -> {out}")
PY

# --- calibrator refresh (non-fatal) ---
CALIB_MIN_ROWS="${CALIB_MIN_ROWS:-200}"
echo "[POST] Calibrator refresh (min_rows=${CALIB_MIN_ROWS})…"
/home/hanumanth/Documents/pyvirtualenv/venv/bin/python calibrator.py --log "$SCORED_TRADEABLE_LOG" --min_rows "$CALIB_MIN_ROWS" || true

# --- seed next session live log ---
NEXT_LIVE_LOG="${NEXT_LIVE_LOG:-data/train_log_v3_live.jsonl}"
if [ "$SCORED_LOG" != "$NEXT_LIVE_LOG" ]; then
  cp "$SCORED_LOG" "$NEXT_LIVE_LOG"
  echo "[POST] Seeded next session live log -> ${NEXT_LIVE_LOG}"
else
  echo "[POST] Next session live log already ${NEXT_LIVE_LOG} (skip copy)"
fi


/home/hanumanth/Documents/pyvirtualenv/venv/bin/python - <<'PY'
import json
cnt=0
with open("data/train_log_v3_live.jsonl","r") as f:
    for line in f:
        r=json.loads(line)
        if r.get("teacher_tradeable") and (r.get("model") or {}).get("p_success_raw") is not None:
            cnt+=1
print("tradeable_scored_rows", cnt)
PY


echo "[POST] Done."

# --- cleanup (optional) ---
POST_CLEANUP="${POST_CLEANUP:-1}"
KEEP_BUNDLES="${KEEP_BUNDLES:-3}"
if [ "${POST_CLEANUP}" = "1" ]; then
  echo "[POST] Cleanup: temp logs + old bundles"
  if [ -n "${SCORE_TMP}" ] && [ -f "${SCORE_TMP}" ]; then
    rm -f "${SCORE_TMP}"
    echo "[POST] Removed ${SCORE_TMP}"
  fi
  if [ -n "${SCORED_TRADEABLE_LOG:-}" ] && [ -f "${SCORED_TRADEABLE_LOG}" ]; then
    rm -f "${SCORED_TRADEABLE_LOG}"
    echo "[POST] Removed ${SCORED_TRADEABLE_LOG}"
  fi
  if [ -d "${MODEL_BUNDLES_DIR}" ]; then
    ls -1dt "${MODEL_BUNDLES_DIR}"/* 2>/dev/null | tail -n +"$((KEEP_BUNDLES + 1))" | xargs -r rm -rf
  fi
fi

# Crontab entry to run this script at 2:00 AM from Sunday to Thursday and save logs to a file
# 0 2 * * 0-4 /bin/bash /home/hanumanth/Documents/sunflower-group/onemin_prediction/post_session_train_eval_fixed.sh >> /home/hanumanth/Documents/sunflower-group/onemin_prediction/logs/post_session_train_eval_fixed.log 2>&1
