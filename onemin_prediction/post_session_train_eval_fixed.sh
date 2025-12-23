#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/Documents/sunflower-group_2/onemin_prediction}"
cd "$PROJECT_ROOT"

# --- inputs ---
if [ -z "${TRAIN_LOG_PATH:-}" ]; then
  echo "[POST][ERROR] TRAIN_LOG_PATH is required."
  exit 1
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
python "$TRAINER" --log "$TRAIN_LOG_PATH"

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
python - <<'PY'
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

# --- offline eval (non-fatal, but useful) ---
echo "[POST] Offline eval…"
python offline_eval.py || true

echo "[POST] Done."
