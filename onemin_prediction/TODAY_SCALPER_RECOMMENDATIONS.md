# Today Scalper Review Dec 31 2025

## Summary
Recent sessions (Jan8–Jan9) show SELL-heavy intent, low BUY recall, and tradeable rates under 3%. Dynamic tuning is active and futures features now show variance, but most misses are still blocked by flow gating (`no_direction`) and low-vol/reversal risk gates.

## Root Cause
- Flow gating (`no_direction`) suppresses BUY when flow strength is marginal.
- Low-vol gating (`rv_atr_low`) and reversal-risk penalties suppress early trend entries.
- Move head gating is disabled when `policy_move.json` is missing.

## Current Focus (Jan 9)
Small, reversible tweaks to recover BUY intent without spiking false positives:
- FLOW_STRONG_MIN=0.33
- LANE_SCORE_MIN=0.54
- HTF_VETO_SOFT_FLOW_MIN=0.80
- GATE_MARGIN_THR=0.065

Follow-ups if BUY recall remains low:
- RV_ATR_MIN=1.5e-05
- REVERSAL_IMB_MIN=0.07
- REVERSAL_VWAP_MIN=0.0005
- REVERSAL_SLOPE_MIN=0.12
- PEN_LANE_SCORE=0.005
- PEN_HYST=0.003

## Archived Retune (Dec 31, Sniper config-only)
Tightening proposal retained for reference (use only if false positives spike):

- FLOW_STRONG_MIN: 0.38 -> 0.45
- REGIME_FLOW_TREND_MIN: 0.35 -> 0.42
- REGIME_VWAP_TREND_MIN: 0.0006 -> 0.0008
- LANE_SCORE_MIN: 0.52 -> 0.56
- GATE_MARGIN_THR: 0.065 -> 0.075
- HTF_VETO_SOFT_FLOW_MIN: 0.85 -> 0.90
- POLICY_MIN_SUCCESS: 0.52 -> 0.56
- MOVE_EDGE_MIN: 0.33 -> 0.36
- RV_ATR_MIN: 1e-05 -> 2e-05

## Validation (Archived Retune)
- Re-run bucketization after a session with the retune.
- Target: reduce BUY->FLAT count without increasing FLAT->SELL misses.
- Watch for over-suppression of trend BUYs; if that happens, roll back GATE_MARGIN_THR first.


## Jan 01 2026 Update
### Changes Implemented Today
- Added dynamic threshold tuning that adjusts FLOW_STRONG_MIN, LANE_SCORE_MIN, and GATE_MARGIN_THR at label time and persists overrides in runtime/thresholds.json.
- Extended jan_1 analysis CSV with lane/regime accuracy buckets for live tuning review.
- Added full-packet volume probe script to verify spot vs futures volume availability.
- Confirmed full-packet volume works for futures (NSE_FNO) but not for index (IDX_I), so keep index on RequestCode 15 and futures sidecar for VWAP/CVD.


### How To Verify Today vs Yesterday
- Generate bucket CSV for today’s session and compare against Jan 1 baseline:
  - intent accuracy: target +8–12 points overall.
  - SETUP lane: target >20% accuracy.
  - CHOP/SIDEWAYS regimes: target +10% accuracy.
- Compare mismatch counts:
  - FLAT→BUY/SELL (missed trades) should drop.
  - BUY/SELL→FLAT (false positives) should not increase.
- Check dynamic tuner log lines:
  - `[DYN-TUNE]` updates should be present but not frequent (avoid oscillation).

### Expected Improvements To Evaluate Tomorrow
- Fewer `no_direction` blocks in TREND_DN/SIDEWAYS while preserving SELL precision.
- BUY recall improves without a spike in false positives.
- `rv_atr_low` and `short_reversal_risk` appear less often in missed BUY labels.


## Jan 05 2026 Update (Dynamic Gates + Volume Proxies)
### Changes Implemented
- Volume proxy: candle volume now uses tick_count when real volume is absent (core handler + preclose preview).
- Futures volume integration: latest futures candle volume is merged into spot candle volume when available.
- Dynamic thresholds applied to flow strength in trend-signal scoring (prevents static gating).
- HTF veto gating uses dynamic thresholds in policy override checks.
- REQUIRE_SETUP penalty is skipped for TREND lanes when trend_signals >= 2 (reduces double friction).
- Gate stack cleanup: avoid adding `teacher_ineligible` or `teacher_flat` when `no_direction` is already present.
- HTF strong veto is softened in CHOP by raising the veto threshold (reduces hard veto frequency).
- EMA_CHOP_HARD_MIN is eased in CHOP when flow/trend align and trend_signals >= 2.
- LANE_SCORE_MIN is lowered only when trend_signals >= 2 to admit strong alignment.

### Implementation Checklist (Jan 05 Session)
- [x] Raise HTF strong veto threshold in CHOP (rule engine + policy override) to reduce hard veto frequency.
- [x] Ease EMA_CHOP_HARD_MIN in CHOP only when flow/trend align and trend_signals >= 2.
- [x] Lower LANE_SCORE_MIN only when trend_signals >= 2 (protects weak signals).
- [ ] Confirm htf_veto and ema_chop_hard gate counts drop in CHOP windows (needs more CHOP samples).
- [ ] Confirm tradeable rate does not spike in CHOP (no false positives).

### Saved Reports
- reports/daily_comparison.md

### To Do (Open)
- Compare gate counts vs Jan 5 baseline (need Jan5 log on disk).
- Train policy_move.json so move-head gating is enabled.
- Retrain and compare BUY/SELL/FLAT accuracy + calibration after 1–2 more sessions with volume variance.
- Run a full session with dynamic tuning enabled and compare bucket metrics to Jan 1 baseline.
- If FLAT misses persist (intent FLAT vs label BUY/SELL), loosen FLOW_STRONG_MIN slightly or reduce CHOP/SIDEWAYS multipliers.

### Completed (Jan 8–9)
- No `[TRAIN] Record quarantined` lines; feature/policy schemas match.
- Volume features are non-zero and show variance.
- Gate counts + tradeable rate compared Jan8 vs Jan9 (see report).
- Calibration guard active (writes skipped when brier_after > brier_before).
- SETUP lane accuracy exceeded 20% (Jan9: 10/16).


### Manual Baseline (No New Env Vars)
Use this balanced baseline if you want fewer FLATs without adding any new env variables.
```
export EMA_CHOP_HARD_MIN=0.54
export LANE_SCORE_MIN=0.52
export FLOW_STRONG_MIN=0.32
```
