#!/usr/bin/env bash
# Tuned env preset for next live session (single-source, no duplicate keys).

export PROJECT_ROOT="/home/hanumanth/Documents/sunflower-group/onemin_prediction"

# Paths
export MODEL_BUNDLES_DIR="$PROJECT_ROOT/trained_models/bundles"
export MODEL_PRODUCTION_LINK="$PROJECT_ROOT/trained_models/production"
export FEATURE_SCHEMA_COLS_PATH="$PROJECT_ROOT/data/feature_schema_cols.json"
export POLICY_SCHEMA_COLS_PATH="$MODEL_PRODUCTION_LINK/policy_schema_cols.json"
export POLICY_BUY_PATH="$MODEL_PRODUCTION_LINK/policy_buy.json"
export POLICY_SELL_PATH="$MODEL_PRODUCTION_LINK/policy_sell.json"
export CALIB_BUY_PATH="$MODEL_PRODUCTION_LINK/calib_buy.json"
export CALIB_SELL_PATH="$MODEL_PRODUCTION_LINK/calib_sell.json"
export TRAIN_LOG_PATH="$PROJECT_ROOT/data/train_log_v3_live.jsonl"
export SIGNALS_PATH="$PROJECT_ROOT/runtime/signals.jsonl"
export IDX_TICKS_PATH="$PROJECT_ROOT/runtime/idx_ticks.csv"
export FUT_SIDECAR_PATH="$PROJECT_ROOT/runtime/fut_candles_vwap_cvd.csv"
export FUT_TICKS_PATH="$PROJECT_ROOT/runtime/fut_ticks_vwap_cvd.csv"
export TRAIN_STATUS_PATH="$PROJECT_ROOT/runtime/trainer_status.json"

# Trainer / promotion gates
export PRETRAIN_MIN_DIR_ROWS=200
export PRETRAIN_SAMPLE_MAX=50000
export PRETRAIN_MAX_ZERO_VAR_FRAC=0.40
export PRETRAIN_MIN_POS_RATIO=0.18
export PREPROMO_HOLDOUT_FRAC=0.20
export PREPROMO_MIN_HOLDOUT_ROWS=60
export CALIB_MIN_ROWS=50
export CALIB_MAX_BYTES=150000000
export CALIB_MAX_ROWS=200000
export TRAIN_MAX_ROWS=200000

# Policy gating
export POLICY_MIN_SUCCESS=0.52
export POLICY_MIN_SIZE_MULT=0.15
export POLICY_GATE_USE_RAW=0
export POLICY_CALIB_FALLBACK_RAW=1
export POLICY_RAW_FALLBACK_MIN=0.62
export POLICY_CALIB_RAW_RATIO_MIN=0.55
export POLICY_CALIB_RAW_GAP_MIN=0.18
export POLICY_INELIGIBLE_OVERRIDE_MIN=0.62

# Structure / flow thresholds
export FLOW_STRONG_MIN=0.32
export LANE_SCORE_MIN=0.45
export GATE_MARGIN_THR=0.055
export FLOW_VWAP_EXT=0.00045
export STRUCT_OPPOSE_FLOW_MIN=0.65
export FLOW_CVD_MIN=0.012
export FLOW_LOCK_CVD_MIN=0.03
export FLOW_LOCK_IMB_MIN=0.07
export REQUIRE_SETUP=1
export PEN_NO_SETUP=0.01
export EMA_CHOP_HARD_MIN=0.55
export TAPE_VALID_USE_FUTURES=1

# Regime
export REGIME_HOLD_BARS=3
export REGIME_FLOW_TREND_MIN=0.40
export REGIME_FLOW_CHOP_MAX=0.20
export REGIME_VWAP_TREND_MIN=0.0007
export REGIME_VWAP_CHOP_MAX=0.0003
export VOL_BAND_LOW_MULT=0.85
export VOL_BAND_HIGH_MULT=1.15

# HTF veto
export HTF_VETO_MODE=conditional
export HTF_VETO_SOFT_FLOW_MIN=0.72
export HTF_STRONG_VETO_MIN=0.70

# Reversal-risk
export REVERSAL_IMB_MIN=0.10
export REVERSAL_CVD_MIN=0.00
export REVERSAL_VWAP_MIN=0.0006
export REVERSAL_SLOPE_MIN=0.16
export PEN_REVERSAL_RISK=0.015

# Move head
export USE_MOVE_HEAD=0
export MOVE_HEAD_MODE=model
export MOVE_HEAD_FALLBACK_PROXY=0
export MOVE_EDGE_MIN=0.34
export MOVE_PROXY_ATR_MULT=0.40
export MOVE_EDGE_TREND_ONLY=1
export POLICY_MOVE_PATH="$MODEL_PRODUCTION_LINK/policy_move.json"
export TRAIN_MOVE_HEAD=1
export MOVE_HEAD_ATR_MULT=0.40
export MOVE_HEAD_MIN_ROWS=5000
export MOVE_HEAD_MAX_ROWS=200000

# Runtime hygiene
export RV_ATR_MIN=1.5e-05
export PEN_LANE_SCORE=0.003
export PEN_HYST=0.003
export ENABLE_PHASE1_PROFILE=1
export DYN_TUNE_ENABLE=0
export POST_CLEANUP=1
export KEEP_BUNDLES=3
