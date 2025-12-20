#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/Documents/sunflower-group_2/onemin_prediction}"
cd "$PROJECT_ROOT"

# --- inputs ---
TRAIN_LOG_PATH="${TRAIN_LOG_PATH:-data/train_log_v2.jsonl}"

# --- runtime artifacts (keep logs out of bundles) ---
mkdir -p runtime
export FEATURE_LOG="${FEATURE_LOG:-runtime/feature_log.csv}"
export FEATURE_LOG_HIST="${FEATURE_LOG_HIST:-runtime/feature_log_hist.csv}"
export SIGNALS_PATH="${SIGNALS_PATH:-runtime/signals.jsonl}"

# --- bundle system ---
export MODEL_BUNDLES_DIR="${MODEL_BUNDLES_DIR:-trained_models/bundles}"
export MODEL_PRODUCTION_LINK="${MODEL_PRODUCTION_LINK:-trained_models/production}"

# Schema path (used by trainer if needed)
export FEATURE_SCHEMA_COLS_PATH="${FEATURE_SCHEMA_COLS_PATH:-trained_models/production/feature_schema_cols.json}"

# --- pick offline trainer ---
TRAINER=""
if [ -f "offline_train_regen_v2_bundle.py" ]; then
  TRAINER="offline_train_regen_v2_bundle.py"
elif [ -f "offline_train_regen_v2.py" ]; then
  TRAINER="offline_train_regen_v2.py"
else
  echo "[POST][ERROR] No offline trainer found. Expected offline_train_regen_v2_bundle.py or offline_train_regen_v2.py"
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
xgb_path = prod / "xgb_model.json"
schema_path = prod / "feature_schema_cols.json"
neutral_path = prod / "neutral_model.pkl"

if not xgb_path.exists():
    raise SystemExit(f"[POST][ERROR] Missing: {xgb_path}")
if not schema_path.exists():
    raise SystemExit(f"[POST][ERROR] Missing: {schema_path}")

obj = json.loads(schema_path.read_text(encoding="utf-8"))
cols = obj.get("columns") or obj.get("feature_names") or obj.get("features") or obj
if not isinstance(cols, list):
    raise SystemExit("[POST][ERROR] feature_schema_cols.json has unexpected format (expected list under 'columns').")

b = xgb.Booster()
b.load_model(str(xgb_path))

print("[POST] XGB num_features:", b.num_features())
print("[POST] Schema cols:", len(cols))

if b.num_features() != len(cols):
    raise SystemExit("[POST][ERROR] XGB/schema mismatch. Refusing to continue.")

# Neutral model feature check (if sklearn model exposes n_features_in_)
if neutral_path.exists():
    try:
        import joblib
        neu = joblib.load(neutral_path)
        n_in = getattr(neu, "n_features_in_", None)
        print("[POST] Neutral n_features_in_:", n_in)
        if n_in is not None and int(n_in) != len(cols):
            raise SystemExit("[POST][ERROR] Neutral/schema mismatch. Refusing to continue.")
    except Exception as e:
        raise SystemExit(f"[POST][ERROR] Failed loading/checking neutral model: {e}")
else:
    print("[POST] Neutral model missing (neutral_model.pkl). Skipping neutral feature check.")

print("[POST] Model/schema OK ✅")
PY

# --- offline eval (non-fatal, but useful) ---
echo "[POST] Offline eval…"
python offline_eval.py || true

echo "[POST] Done."
