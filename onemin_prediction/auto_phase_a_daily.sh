#!/usr/bin/env bash
set -euo pipefail

### === CONFIG SECTION ===

# Adjust to your project path
PROJECT_ROOT="/home/hanumanth/Documents/sunflower-group_2/onemin_prediction"
ENV_PATH="/home/hanumanth/Documents/pyvirtualenv"
VENV_PATH="${ENV_PATH}/venv"



# Market times (IST) used to build env dates
TRAIN_START_DATE_DEFAULT="2025-08-25 09:30:00"   # first day you want in training
TRAIN_END_CUTOFF_TIME="15:15:00"                 # end of training window (yesterday close)
EVAL_START_TIME="09:15:00"                       # eval day start
EVAL_END_TIME="15:30:00"                         # eval day end

# Which day to run leakage sanity (1=Mon ... 7=Sun)
LEAKAGE_DOW=6   # 6 = Saturday, 5 = Friday if you prefer

### === DATE CALC ===

# Today/yesterday in IST; assumes server clock is IST or close enough for EOD runs
TODAY=$(date +%Y-%m-%d)
YESTERDAY=$(date -d "yesterday" +%Y-%m-%d)

TRAIN_START_DATE="${TRAIN_START_DATE:-$TRAIN_START_DATE_DEFAULT}"
TRAIN_END_DATE="${TRAIN_END_DATE:-${YESTERDAY} ${TRAIN_END_CUTOFF_TIME}}"

EVAL_START_DATE="${EVAL_START_DATE:-${YESTERDAY} ${EVAL_START_TIME}}"
EVAL_END_DATE="${EVAL_END_DATE:-${YESTERDAY} ${EVAL_END_TIME}}"

echo "=== Phase A daily run ==="
echo "PROJECT_ROOT:        ${PROJECT_ROOT}"
echo "TRAIN_START_DATE:    ${TRAIN_START_DATE}"
echo "TRAIN_END_DATE:      ${TRAIN_END_DATE}"
echo "EVAL_START_DATE:     ${EVAL_START_DATE}"
echo "EVAL_END_DATE:       ${EVAL_END_DATE}"
echo

### === ENV EXPORTS FOR PY SCRIPTS ===

export TRAIN_START_DATE
export TRAIN_END_DATE
export EVAL_START_DATE
export EVAL_END_DATE

# Make sure model paths are consistent (adjust if yours differ)
export XGB_PATH="${PROJECT_ROOT}/trained_models/experiments/xgb_2min.json"
export NEUTRAL_PATH="${PROJECT_ROOT}/trained_models/experiments/neutral_2min.pkl"
export Q_MODEL_2MIN_PATH="${PROJECT_ROOT}/trained_models/experiments/q_model_2min.json"

# These are used by offline_eval.py (live signals eval) but harmless here
export SIGNALS_PATH="${PROJECT_ROOT}/trained_models/production/signals.jsonl"
export FEATURE_LOG_HIST="${PROJECT_ROOT}/trained_models/production/feature_log_hist.csv"

### === ACTIVATE VENV ===

cd "${PROJECT_ROOT}"
# shellcheck disable=SC1090
source "${VENV_PATH}/bin/activate"

### === STEP 1: Train 2-min XGB + neutral model (history up to TRAIN_END_DATE) ===

echo ">>> [1/4] offline_train_2min.py"
python offline_train_2min.py

echo ">>> [2/4] offline_eval_2min.py (raw XGB only, walk-forward)"
python offline_eval_2min.py

echo ">>> [3/4] offline_train_q_model_2min.py (Q-model from offline predictions)"
python offline_train_q_model_2min.py

### === STEP 4: Leakage sanity (once per week) ===

DOW=$(date +%u)  # 1=Mon ... 7=Sun

if [[ "${DOW}" -eq "${LEAKAGE_DOW}" ]]; then
  echo ">>> [4/4] Weekly leakage sanity: offline_leakage_sanity_2min.py"
  # Use eval day as leakage test window by default
  export LEAKAGE_START_DATE="${EVAL_START_DATE}"
  export LEAKAGE_END_DATE="${EVAL_END_DATE}"
  python offline_leakage_sanity_2min.py
else
  echo ">>> [4/4] Skipping leakage sanity (today DOW=${DOW}, scheduled DOW=${LEAKAGE_DOW})"
fi

echo "=== Phase A daily run complete ==="
