# Script Reference

Short descriptions for top-level scripts. Paths are relative to repo root.

- `automation_with_sidecar.sh`: Convenience runner to start main trading loop with the futures sidecar.
- `bootstrap_trainlog_from_cache_v3.py`: Build SignalContext training records from cached OHLCV files in `data/intraday_cache/`.
- `policy_pipeline.py`: Policy inference pipeline for BUY/SELL success probabilities.
- `cache_manifest.py`: Build cache manifest + gap scan for cached days (helps identify missing/partial days).
- `calibrator.py`: Platt calibration loop for BUY/SELL success probabilities using SignalContext logs.
- `core_handler.py`: WebSocket/event handling core for live data.
- `dhan_cache_fetch.py`: Fetch intraday OHLCV from Dhan API into per-day cache CSVs.
- `eval_before_promotion.py`: Walk-forward evaluation gate before model promotion.
- `fetch_missing_from_manifest.py`: Fetch only missing/partial cache days from a manifest using Dhan API.
- `feature_availability.py`: LIVE vs OHLCV feature coverage checks for gating.
- `feature_pipeline.py`: Feature engineering and normalization used by online pipeline.
- `futures_vwap_cvd_sidecar.py`: Optional sidecar to compute futures VWAP/CVD features.
- `logging_setup.py`: Logging setup utilities.
- `main_event_loop_regen.py`: Main live trading loop, emits signals and SignalContext training records.
- `merge_train_logs_v3.py`: Merge backfill + online SignalContext logs into canonical.
- `model_bundle.py`: Bundle creation, validation, manifest, and atomic promotion for policy artifacts.
- `offline_eval.py`: Offline evaluation of policy success vs labels.
- `offline_train_regen_v2.py`: Offline training on canonical SignalContext logs with gates + promotion.
- `online_trainer_regen_v2_bundle.py`: Background policy trainer with gates + bundle promotion.
- `post_session_train_eval_fixed.sh`: Post-session training + bundle validation + eval helper.
- `pretrain_validator.py`: Pre-training validation gates (label skew, zero-var, live coverage).
- `run_main.py`: Entry point for live trading; sets Dhan connection details and config.
- `schema_contract.py`: Schema resolution error type.
- `signal_context.py`: SignalContext record contract + IST timestamp normalization.
- `signal_log_utils.py`: SignalContext log utilities (validation, quarantine, merge helpers).
- `rule_engine.py`: Shared rule-as-teacher logic for direction and tradeability.
- `tune_atr_multipliers.py`: ATR threshold tuning tool for label generation.

Removed/obsolete:
- `offline_check_schema_vs_logs.py`: Removed (legacy CSV/schema check; superseded by v3 validation).
- `online_trainer_regen_v2.py`: Removed (redundant shim; import bundle directly).
- `model_pipeline_regen_v2.py`: Removed (legacy XGB/neutral inference pipeline).
- `train_log_utils_v3.py`: Removed (superseded by SignalContext utilities).
- `train_record_v3.py`: Removed (superseded by SignalContext contract).
