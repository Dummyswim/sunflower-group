#!/usr/bin/env python3
"""Walk-forward evaluation gate before promotion."""
from __future__ import annotations

import os
from typing import Any, Dict, List, Tuple

import numpy as np


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        if np.isfinite(v):
            return v
    except Exception:
        pass
    return float(default)


def _build_matrix(rows: List[Dict[str, Any]], cols: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    X = np.zeros((len(rows), len(cols)), dtype=np.float32)
    y = np.zeros((len(rows),), dtype=np.int32)
    for i, r in enumerate(rows):
        feats = r.get("features") or {}
        for j, c in enumerate(cols):
            X[i, j] = _safe_float(feats.get(c, 0.0), 0.0)
        lab = str(r.get("label", "FLAT")).upper()
        y[i] = 1 if lab == "BUY" else 0
    return X, y


def evaluate_holdout(
    rows: List[Dict[str, Any]],
    schema_cols: List[str],
    xgb_model,
) -> Tuple[bool, Dict[str, Any]]:
    if not rows:
        return False, {"reason": "no_rows"}

    holdout_frac = float(os.getenv("PREPROMO_HOLDOUT_FRAC", "0.1") or "0.1")
    holdout_frac = min(max(holdout_frac, 0.05), 0.5)
    min_holdout = int(os.getenv("PREPROMO_MIN_HOLDOUT_ROWS", "500") or "500")

    rows_sorted = sorted(rows, key=lambda r: r.get("ts_target_close") or "")
    split = int(len(rows_sorted) * (1.0 - holdout_frac))
    holdout = rows_sorted[split:]

    # directional only
    holdout_dir = [r for r in holdout if str(r.get("label", "")).upper() in ("BUY", "SELL")]
    if len(holdout_dir) < min_holdout:
        return False, {"reason": "holdout_too_small", "holdout_dir": len(holdout_dir)}

    X, y = _build_matrix(holdout_dir, schema_cols)
    try:
        p = xgb_model.predict_proba(X)[:, 1]
    except Exception:
        return False, {"reason": "predict_failed"}

    preds = (p >= 0.5).astype(int)
    acc = float(np.mean(preds == y))

    # AUC (rank-based) for robustness
    try:
        order = np.argsort(p)
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(len(p), dtype=float) + 1.0
        n_pos = float(np.sum(y == 1))
        n_neg = float(np.sum(y == 0))
        if n_pos > 0 and n_neg > 0:
            pos_ranks_sum = float(np.sum(ranks[y == 1]))
            auc = (pos_ranks_sum - n_pos * (n_pos + 1.0) / 2.0) / (n_pos * n_neg)
        else:
            auc = 0.0
    except Exception:
        auc = 0.0

    min_acc = float(os.getenv("PREPROMO_MIN_ACC", "0.52") or "0.52")
    min_auc = float(os.getenv("PREPROMO_MIN_AUC", "0.52") or "0.52")

    ok = bool(acc >= min_acc and auc >= min_auc)
    report = {
        "holdout_dir": len(holdout_dir),
        "acc": acc,
        "auc": auc,
        "min_acc": min_acc,
        "min_auc": min_auc,
    }
    return ok, report
