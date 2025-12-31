# Today Scalper Review Dec 31 2025

## Summary
Incorrect bucket is dominated by BUY->FLAT mismatches. The system keeps BUY intent in low-edge windows, while labels resolve to FLAT. This is driven by stacked soft penalties (require_setup_no_setup + lane_score_override + hysteresis_hold), low-vol gating (rv_atr_low), and conservative label thresholds.

## Root Cause
- Intent stays BUY while gates reduce tradeability; labels are FLAT in low-edge windows.
- Flow/regime thresholds are permissive, so TREND intent appears in chop/sideways pockets.
- Soft penalties stack without a cap, which is accurate for safety but harms prediction alignment.

## Recommended Retune (Sniper, config-only)
Adjust only env values (no new logic) to tighten intent and reduce BUY->FLAT errors:

- FLOW_STRONG_MIN: 0.38 -> 0.45
- REGIME_FLOW_TREND_MIN: 0.35 -> 0.42
- REGIME_VWAP_TREND_MIN: 0.0006 -> 0.0008
- LANE_SCORE_MIN: 0.52 -> 0.56
- GATE_MARGIN_THR: 0.065 -> 0.075
- HTF_VETO_SOFT_FLOW_MIN: 0.85 -> 0.90
- POLICY_MIN_SUCCESS: 0.52 -> 0.56
- MOVE_EDGE_MIN: 0.33 -> 0.36
- RV_ATR_MIN: 1e-05 -> 2e-05

## Validation
- Re-run bucketization after a session with the retune.
- Target: reduce BUY->FLAT count without increasing FLAT->SELL misses.
- Watch for over-suppression of trend BUYs; if that happens, roll back GATE_MARGIN_THR first.
