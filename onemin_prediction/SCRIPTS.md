# Script Reference

Short descriptions for top-level scripts. Paths are relative to repo root.

- `automation_with_sidecar.sh`: Convenience runner to start main trading loop with the futures sidecar.
- `bootstrap_trainlog_from_cache_v3.py`: Build v3 training records from cached OHLCV files in `data/intraday_cache/`.
- `cache_manifest.py`: Build cache manifest + gap scan for cached days (helps identify missing/partial days).
- `calibrator.py`: Platt calibration loop for BUY probabilities using v3 train logs.
- `core_handler.py`: WebSocket/event handling core for live data.
- `dhan_cache_fetch.py`: Fetch intraday OHLCV from Dhan API into per-day cache CSVs.
- `eval_before_promotion.py`: Walk-forward evaluation gate before model promotion.
- `fetch_missing_from_manifest.py`: Fetch only missing/partial cache days from a manifest using Dhan API.
- `feature_availability.py`: LIVE vs OHLCV feature coverage checks for gating.
- `feature_pipeline.py`: Feature engineering and normalization used by online pipeline.
- `futures_vwap_cvd_sidecar.py`: Optional sidecar to compute futures VWAP/CVD features.
- `logging_setup.py`: Logging setup utilities.
- `main_event_loop_regen.py`: Main live trading loop, produces signals and v3 train records.
- `merge_train_logs_v3.py`: Merge backfill + online v3 logs into canonical with diagnostics.
- `model_bundle.py`: Bundle creation, validation, manifest, and atomic promotion.
- `model_pipeline_regen_v2.py`: Inference pipeline, schema alignment, calibration application.
- `offline_eval.py`: Offline evaluation of signal accuracy and AUC against labels.
- `offline_train_regen_v2.py`: Offline training on canonical v3 logs with gates + promotion.
- `online_trainer_regen_v2_bundle.py`: Background online trainer with gates + bundle promotion.
- `post_session_train_eval_fixed.sh`: Post-session training + bundle validation + eval helper.
- `pretrain_validator.py`: Pre-training validation gates (label skew, zero-var, live coverage).
- `run_main.py`: Entry point for live trading; sets Dhan connection details and config.
- `schema_contract.py`: Schema resolution error type.
- `train_log_utils_v3.py`: v3 log utilities (validation, quarantine, merge helpers).
- `train_record_v3.py`: v3 train record contract + IST timestamp normalization.
- `tune_atr_multipliers.py`: ATR threshold tuning tool for label generation.

Removed/obsolete:
- `offline_check_schema_vs_logs.py`: Removed (legacy CSV/schema check; superseded by v3 validation).
- `online_trainer_regen_v2.py`: Removed (redundant shim; import bundle directly).
