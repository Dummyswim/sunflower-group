#!/usr/bin/env python3
"""Pre-training validation gates for v3 train records."""
from __future__ import annotations

import os
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

from feature_availability import assess_live_coverage, LIVE_ONLY_FEATURES
from signal_context import align_features_to_schema, compose_policy_features


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


def _teacher_counts(rows: Iterable[Dict[str, Any]]) -> Dict[str, int]:
    out = {"BUY": 0, "SELL": 0, "FLAT": 0, "OTHER": 0}
    for r in rows:
        d = str(r.get("teacher_dir", "")).upper()
        if d in out:
            out[d] += 1
        else:
            out["OTHER"] += 1
    return out


def _is_live_only_feature(name: str) -> bool:
    if name in LIVE_ONLY_FEATURES:
        return True
    if name.startswith("rule_"):
        base = name[len("rule_"):]
        return base in LIVE_ONLY_FEATURES
    return False


def _policy_feature_stats(rows: List[Dict[str, Any]], schema_cols: List[str]) -> Dict[str, Dict[str, float]]:
    sums = {c: 0.0 for c in schema_cols}
    sumsqs = {c: 0.0 for c in schema_cols}
    counts = {c: 0 for c in schema_cols}

    for r in rows:
        feats = r.get("features") or {}
        rule_signals = r.get("rule_signals") or {}
        gates = r.get("gates") or {}
        teacher_strength = _safe_float(r.get("teacher_strength", 0.0), 0.0)
        pol = compose_policy_features(
            features=feats,
            rule_signals=rule_signals,
            gates=gates,
            teacher_strength=teacher_strength,
        )
        aligned, _, _ = align_features_to_schema(pol, schema_cols)
        for c in schema_cols:
            v = _safe_float(aligned.get(c), 0.0)
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
    tradeable_sample = [r for r in sample if bool(r.get("teacher_tradeable"))]
    teacher_counts = _teacher_counts(tradeable_sample)
    report["teacher_counts_tradeable"] = teacher_counts
    dir_total = teacher_counts["BUY"] + teacher_counts["SELL"]
    min_dir_ratio = float(os.getenv("PRETRAIN_MIN_DIR_RATIO", "0.2") or "0.2")
    if dir_total > 0:
        min_share = min(teacher_counts["BUY"], teacher_counts["SELL"]) / float(dir_total)
    else:
        min_share = 0.0
    report["min_dir_ratio"] = min_share

    max_flat_ratio = float(os.getenv("PRETRAIN_MAX_FLAT_RATIO", "0.85") or "0.85")
    flat_ratio = counts["FLAT"] / float(max(1, len(sample)))
    report["flat_ratio"] = flat_ratio

    reasons: List[str] = []
    min_dir_rows = int(os.getenv("PRETRAIN_MIN_DIR_ROWS", "800") or "800")
    if not tradeable_sample:
        reasons.append("no_teacher_tradeable_rows")
    if dir_total == 0 or min_share < min_dir_ratio:
        reasons.append("teacher_dir_skew")
    if teacher_counts["BUY"] < min_dir_rows or teacher_counts["SELL"] < min_dir_rows:
        reasons.append("too_few_teacher_dir_rows")
    if flat_ratio > max_flat_ratio:
        reasons.append("too_many_flat")

    stats = _policy_feature_stats(sample, schema_cols)
    zero_var_eps = float(os.getenv("PRETRAIN_ZERO_VAR_EPS", "1e-10") or "1e-10")
    zero_var = [
        k for k, v in stats.items()
        if (not _is_live_only_feature(k)) and abs(v.get("var", 0.0)) <= zero_var_eps
    ]
    report["zero_var_features"] = zero_var
    max_zero_var_frac = float(os.getenv("PRETRAIN_MAX_ZERO_VAR_FRAC", "0.25") or "0.25")
    candidates = [k for k in schema_cols if not _is_live_only_feature(k)]
    zero_var_frac = len(zero_var) / float(max(1, len(candidates)))
    report["zero_var_frac"] = zero_var_frac
    if zero_var_frac > max_zero_var_frac:
        reasons.append("too_many_zero_var_features")

    live_min = float(os.getenv("PRETRAIN_MIN_LIVE_COVERAGE", "0.2") or "0.2")
    live_ok, live_report = assess_live_coverage(sample, min_coverage=live_min)
    report["live_coverage"] = live_report
    if not live_ok:
        reasons.append("low_live_coverage")

    # Directional success balance per teacher
    pos_min = float(os.getenv("PRETRAIN_MIN_POS_RATIO", "0.2") or "0.2")
    pos_ratios: Dict[str, float] = {}
    for d in ("BUY", "SELL"):
        sub = [r for r in sample if str(r.get("teacher_dir", "")).upper() == d and bool(r.get("teacher_tradeable"))]
        if not sub:
            pos_ratios[d] = 0.0
            continue
        pos = sum(1 for r in sub if str(r.get("label", "")).upper() == d)
        pos_ratios[d] = pos / float(len(sub))
        if pos_ratios[d] < pos_min or pos_ratios[d] > (1.0 - pos_min):
            reasons.append(f"label_skew_{d.lower()}")
    report["pos_ratios"] = pos_ratios

    report["reasons"] = reasons
    return (len(reasons) == 0), report
