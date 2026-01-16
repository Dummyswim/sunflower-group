#!/usr/bin/env python3
"""Feature availability checks and live coverage gating."""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Tuple

LIVE_ONLY_FEATURES = [
    "micro_imbalance",
    "micro_slope",
    "fut_cvd_delta",
    "fut_vwap_dev",
    "fut_session_vwap",
    "fut_vol_delta",
    "flow_score",
    "flow_side",
    "flow_fut_cvd",
    "flow_fut_vwap_dev",
    "flow_fut_vol",
    "flow_vwap_side",
    "flow_micro_imb",
    "indicator_score",
]

OHLCV_ONLY_FEATURES = [
    "ema_8",
    "ema_21",
    "ema_50",
    "ta_rsi14",
    "ta_macd_hist",
    "ta_bb_bw",
    "ta_bb_bw_pct",
    "ta_bb_pctb",
    "ta_di_plus",
    "ta_di_minus",
    "ta_di_spread",
    "ta_supertrend_dir",
    "ta_supertrend_flip",
    "atr_1t",
    "rv_10",
    "last_zscore",
    "mean_drift_pct",
    "struct_pivot_swipe_up",
    "struct_pivot_swipe_down",
    "struct_fvg_up_present",
    "struct_fvg_down_present",
    "struct_ob_bull_present",
    "struct_ob_bear_present",
    "wick_extreme_up",
    "wick_extreme_down",
    "tod_sin",
    "tod_cos",
]


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        return v
    except Exception:
        return float(default)


def _record_source(row: Dict[str, Any]) -> str:
    prov = row.get("provenance") or {}
    src = prov.get("record_source_primary") or prov.get("record_source")
    if src is None:
        meta = row.get("meta") or {}
        src = meta.get("record_source_primary") or meta.get("record_source")
    return str(src or "").lower()


def assess_live_coverage(
    rows: Iterable[Dict[str, Any]],
    *,
    min_coverage: float = 0.2,
    eps: float = 1e-9,
) -> Tuple[bool, Dict[str, Any]]:
    total = 0
    live = 0
    src_counts = {"online": 0, "backfill": 0, "other": 0}

    for r in rows:
        total += 1
        src = _record_source(r)
        if src == "online":
            src_counts["online"] += 1
        elif src == "backfill":
            src_counts["backfill"] += 1
        else:
            src_counts["other"] += 1

        if src == "online":
            live += 1
            continue

        feats = r.get("features") or {}
        live_hit = False
        for k in LIVE_ONLY_FEATURES:
            if abs(_safe_float(feats.get(k), 0.0)) > eps:
                live_hit = True
                break
        if live_hit:
            live += 1

    coverage = (float(live) / float(total)) if total else 0.0
    ok = coverage >= float(min_coverage)

    report = {
        "total": total,
        "live": live,
        "coverage": coverage,
        "min_coverage": float(min_coverage),
        "source_counts": src_counts,
        "live_only_features": list(LIVE_ONLY_FEATURES),
    }
    return ok, report
