#!/usr/bin/env python3
"""Pre-training validation gates for v3 train records."""
from __future__ import annotations

import os
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

from feature_availability import assess_live_coverage


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        if np.isfinite(v):
            return v
    except Exception:
        pass
    return float(default)


def _label_counts(rows: Iterable[Dict[str, Any]]) -> Dict[str, int]:
    out = {"BUY": 0, "SELL": 0, "FLAT": 0, "OTHER": 0}
    for r in rows:
        lab = str(r.get("label", "")).upper()
        if lab in out:
            out[lab] += 1
        else:
            out["OTHER"] += 1
    return out


def _feature_stats(rows: List[Dict[str, Any]], schema_cols: List[str]) -> Dict[str, Dict[str, float]]:
    sums = {c: 0.0 for c in schema_cols}
    sumsqs = {c: 0.0 for c in schema_cols}
    counts = {c: 0 for c in schema_cols}

    for r in rows:
        feats = r.get("features") or {}
        for c in schema_cols:
            v = _safe_float(feats.get(c), 0.0)
            sums[c] += v
            sumsqs[c] += v * v
            counts[c] += 1

    stats = {}
    for c in schema_cols:
        n = max(1, counts[c])
        mean = sums[c] / n
        var = (sumsqs[c] / n) - (mean * mean)
        stats[c] = {"mean": mean, "var": var}
    return stats


def validate_pretrain(
    rows: List[Dict[str, Any]],
    *,
    schema_cols: List[str],
) -> Tuple[bool, Dict[str, Any]]:
    report: Dict[str, Any] = {"rows": len(rows)}
    if not rows:
        return False, {"rows": 0, "reasons": ["no_rows"]}

    min_rows = int(os.getenv("PRETRAIN_MIN_ROWS", "5000") or "5000")
    if len(rows) < min_rows:
        return False, {"rows": len(rows), "reasons": ["min_rows"]}

    sample_max = int(os.getenv("PRETRAIN_SAMPLE_MAX", "50000") or "50000")
    sample = rows[-sample_max:] if sample_max and len(rows) > sample_max else rows

    counts = _label_counts(sample)
    report["label_counts"] = counts
    dir_total = counts["BUY"] + counts["SELL"]
    min_dir_ratio = float(os.getenv("PRETRAIN_MIN_DIR_RATIO", "0.2") or "0.2")
    if dir_total > 0:
        min_share = min(counts["BUY"], counts["SELL"]) / float(dir_total)
    else:
        min_share = 0.0
    report["min_dir_ratio"] = min_share

    max_flat_ratio = float(os.getenv("PRETRAIN_MAX_FLAT_RATIO", "0.85") or "0.85")
    flat_ratio = counts["FLAT"] / float(max(1, len(sample)))
    report["flat_ratio"] = flat_ratio

    reasons: List[str] = []
    if dir_total == 0 or min_share < min_dir_ratio:
        reasons.append("label_skew")
    if flat_ratio > max_flat_ratio:
        reasons.append("too_many_flat")

    stats = _feature_stats(sample, schema_cols)
    zero_var_eps = float(os.getenv("PRETRAIN_ZERO_VAR_EPS", "1e-10") or "1e-10")
    zero_var = [k for k, v in stats.items() if abs(v.get("var", 0.0)) <= zero_var_eps]
    report["zero_var_features"] = zero_var
    max_zero_var_frac = float(os.getenv("PRETRAIN_MAX_ZERO_VAR_FRAC", "0.25") or "0.25")
    zero_var_frac = len(zero_var) / float(max(1, len(schema_cols)))
    report["zero_var_frac"] = zero_var_frac
    if zero_var_frac > max_zero_var_frac:
        reasons.append("too_many_zero_var_features")

    live_min = float(os.getenv("PRETRAIN_MIN_LIVE_COVERAGE", "0.2") or "0.2")
    live_ok, live_report = assess_live_coverage(sample, min_coverage=live_min)
    report["live_coverage"] = live_report
    if not live_ok:
        reasons.append("low_live_coverage")

    report["reasons"] = reasons
    return (len(reasons) == 0), report
