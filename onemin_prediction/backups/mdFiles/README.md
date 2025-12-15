# onemin_prediction – 2‑minute NIFTY automation (guide + 5‑minute extension)

This repo contains a 2‑minute, probabilities‑only trading pipeline for NIFTY. It ingests live data from Dhan (index + futures sidecar), computes features, produces BUY/SELL/FLAT probabilities, and records signals. This README explains the current flow and how to extend it to a 5‑minute horizon.

## High‑level architecture

- **main_event_loop.py**: Real‑time orchestrator. Subscribes to WS, builds per‑candle features, runs the model pipeline, applies indicator/Q/neutral gating, and emits signals to `trained_models/production/signals.jsonl`.
- **core_handler.py / UnifiedWebSocketHandler**: Websocket handler for index feed; builds candles and basic microstructure features.
- **futures_vwap_cvd_sidecar.py**: Separate process to consume Dhan futures feed (NSE_FNO), compute session VWAP + CVD, and log to CSV for feature ingestion.
- **feature_pipeline.py**: Feature computations (EMAs, TA bundle, microstructure, wick extremes, VWAP reversion flags, CVD divergence, reversal cross features, normalization helpers).
- **model_pipeline.py**: Wraps CNN-LSTM placeholder + XGB directional model + optional neutrality model. Contains indicator modulation (now soft margin shrink) and feature/schema alignment.
- **online_trainer.py**: Background trainer for XGB (directional) and neutrality models from rolling feature logs; includes standardized, balanced logistic for neutrality.
- **offline_train_2min.py**: Builds 2‑minute labeled dataset from historical 1‑minute candles, trains XGB + neutrality, embeds feature schema, writes models.
- **offline_eval_2min.py**: Pure XGB evaluation on a hold‑out day; dumps `offline_2min_with_preds.csv` for Q training.
- **offline_train_q_model_2min.py**: Trains a Q‑model (logistic on p_buy + regime features). Uses `build_q_features_from_row` to mirror live feature construction; skips rows lacking required features.
- **offline_leakage_sanity_2min.py**: Quick AUC sanity (true vs shuffled labels) to detect structural leakage.
- **offline_eval.py**: Evaluates live signals against feature_log_hist (backfilled labels) to gauge live performance.
- **automation scripts**: `auto_phase_a_daily.sh`, `phase_a_daily.sh` to run the daily offline pipeline.

## Data & artifacts

- Live signals: `trained_models/production/signals.jsonl`
- Live feature logs: `trained_models/production/feature_log.csv` and `feature_log_hist.csv`
- Futures sidecar outputs: `trained_models/production/fut_ticks_vwap_cvd.csv`, `fut_candles_vwap_cvd.csv`
- Models: `trained_models/production` or `trained_models/experiments` (XGB, neutrality, Q)
- Offline Q dump: `trained_models/experiments/offline_2min_with_preds.csv`

## Runtime flow (2‑minute)

1) **WS ingest** (index feed) -> `core_handler` emits ticks/candles.
2) **Sidecar futures feed** -> VWAP/CVD CSVs (if available).
3) **Feature build** in `main_event_loop._on_preclose_cb`: EMAs, TA, patterns, SR, microstructure, ToD, wicks, VWAP reversion, CVD divergence, futures deltas, indicator score, MTF consensus; normalization via `FeaturePipeline.normalize_features`.
4) **Model inference** in `model_pipeline`: XGB directional (p_model), optional neutrality (p_flat), indicator modulation (soft shrink/boost), pattern adj.
5) **Q‑model** (if loaded) gets p_buy + regime features (`cvd_divergence`, `vwap_reversion_flag`, `wick_extreme_*`).
6) **Gating** via `_gate_trade_decision`: uses margin, Q, neutral_prob, tuner delta, and optional rule_signal to set tradeable flag.
7) **Logging & signals**: human‑readable 3‑way BUY/SELL/FLAT view, plus signal record to JSONL; staged features for later training.
8) **Background trainer** (optional) refreshes models from feature logs.

## Key defensive choices

- Price sanity checks on futures sidecar; reconnect logic.
- NaN/inf guards on all features; normalization clips bounded keys.
- Neutrality logistic standardized with class_weight, higher max_iter; bypass when data insufficient.
- Indicator gating softened (shrink margin vs. hard clamp) with diagnostics.
- Q‑training skips rows without required features; optional futures fields filled only when present.

## Adapting to a 5‑minute horizon

The existing horizon is 2 minutes (horizon=2 candles at 1‑min candles). To build a 5‑minute version:

1) **Config knobs**
   - In `main_event_loop.py` (and configs), set `candle_interval_seconds = 60` (keep 1‑min candles) and set a new horizon of 5 (or simply run 5‑minute candles end‑to‑end). If you prefer native 5‑minute candles, adjust `candle_interval_seconds = 300` in the connection config and downstream consumers.
   - Update any environment variables or script params that assume “2‑minute” labels.

2) **Labeling logic (offline)**
   - In `offline_train_2min.py`, set `horizon = 5` (or add a param) so label_t compares close_{t+5} vs close_t.
   - Recompute flat tolerance stats on 5‑minute moves; expect larger sigma. You may want a higher `OFFLINE_FLAT_SIGMA_MULT` or points‑based tolerance.
   - Rebuild the offline dataset and retrain XGB + neutrality with the 5‑minute labels.

3) **Feature cadence**
   - If staying on 1‑minute candles with 5‑minute horizon: keep feature lookbacks but ensure they are appropriate (e.g., price history len, ATR windows). Many features already use fixed lists; review `FeaturePipeline.compute_*` windows for sufficiency on slower horizons.
   - If switching to 5‑minute candles: adjust lookbacks proportionally (e.g., EMAs on fewer but larger bars) in both offline and live code paths.

4) **Model retraining**
   - Run `offline_train_2min.py` (renamed/parametrized) with horizon=5 to produce `xgb_5min.json` and `neutral_5min.pkl` (pos_share/minor_share will change). Ensure feature_schema is embedded.
   - Run `offline_eval_2min.py` (parametrized) on a hold‑out day to sanity‑check raw XGB.

5) **Q‑model refresh**
   - Generate new `offline_5min_with_preds.csv` via the eval script (now 5‑minute labels/preds). Confirm `cvd_divergence` / `vwap_reversion_flag` columns are populated (non‑NaN) when futures data exist.
   - Train a fresh Q‑model with `offline_train_q_model_2min.py` (parametrized) and ensure feature order matches live (p_buy, cvd_divergence, vwap_reversion_flag, wick_extreme_up, wick_extreme_down). Drop rows missing required fields; optional futures features can be filled with 0.0 after dropna on required wicks/p_buy.

6) **Gating thresholds**
   - 5‑minute horizon usually yields larger margins; revisit `qmin` base and neutral gating. Start with a slightly higher `qmin` (e.g., 0.12–0.15) and re‑tune via the rolling tuner.
   - Neutral gating: keep skip rules (e.g., neu >= 0.75 => skip) but validate on new horizon.

7) **Logging/paths**
   - Use new artifact names to avoid overwriting 2‑minute models/logs (e.g., `trained_models/production/xgb_5min.json`, `signals_5min.jsonl`).
   - Update any automation scripts (`auto_phase_a_daily.sh`, `phase_a_daily.sh`) to call the 5‑minute variants.

8) **Live pipeline tweaks**
   - In `main_event_loop`, change the horizon used for `target_start` (currently 2 * interval). Set to 5 * interval when building the signal record and staging features.
   - Ensure feature staging and model loading point to the 5‑minute models; schema alignment will still work via the booster attribute.

## Quick file map

- `main_event_loop.py`: live loop, gating helper `_gate_trade_decision`, signal logging (3‑way BUY/SELL/FLAT), model_quality, tuner.
- `model_pipeline.py`: XGB + optional CNN placeholder, indicator modulation (soft shrink/boost), schema alignment.
- `feature_pipeline.py`: feature builders, wicks, VWAP reversion, CVD divergence, reversal crosses, normalization.
- `core_handler.py`: unified WS handler, candle building, microstructure.
- `futures_vwap_cvd_sidecar.py`: futures VWAP/CVD feed; ensure `FUT_SECURITY_ID` set to current contract.
- `offline_train_2min.py`: offline dataset/label build, training, feature schema embed.
- `offline_eval_2min.py`: raw XGB eval + dump for Q training.
- `offline_train_q_model_2min.py`: Q feature reconstruction + logistic training; skips when missing required fields.
- `offline_leakage_sanity_2min.py`: leakage AUC check (now BUY=1, SELL=0).
- `offline_eval.py`: evaluates live signals vs feature logs.
- `logging_setup.py`, `core_handler.py`, `main_event_loop.py`: logging infra and helpers.

## Tips for 5‑minute build

- Keep futures sidecar running; without real CVD/VWAP, Q loses a key regime signal.
- Validate that `cvd_divergence`/`vwap_reversion_flag` are non‑NaN in the offline dump; if all NaN, revisit merge/feature sourcing.
- Start with a raw XGB sanity check (accuracy vs prob bins). If it’s weak (<55%), re‑tune features or expand history.
- Use leakage sanity after major label/feature changes to detect future leakage.
- When switching horizons, regenerate feature statistics (sigma/flat tolerance) and retune neutral thresholds.

## Running the pipeline (2‑minute current)

- Live: `python main_event_loop.py` (with env DHAN_ACCESS_TOKEN, DHAN_CLIENT_ID, model paths). Futures sidecar can be launched separately.
- Offline train: `python offline_train_2min.py` (set TRAIN_START_DATE/END_DATE, XGB_PATH, NEUTRAL_PATH).
- Offline eval: `python offline_eval_2min.py` (set EVAL_START_DATE/END_DATE, XGB_PATH); produces offline_2min_with_preds.csv.
- Q‑train: `python offline_train_q_model_2min.py` after the offline preds dump.
- Leakage sanity: `python offline_leakage_sanity_2min.py` with LEAKAGE_START_DATE/END_DATE.

## Known warnings

- joblib/loky resource_tracker warnings indicate worker pools not closed cleanly; mitigate by using `with Parallel(...)` or shutting executors explicitly.

---
This README summarizes the current 2‑minute system and the steps needed to adapt it to a 5‑minute horizon. Use it as a blueprint to adjust configs, labels, features, models, and gating for the longer timeframe.
