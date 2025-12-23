#!/usr/bin/env python3
"""Walk-forward evaluation gate before promotion (policy BUY/SELL models)."""
from __future__ import annotations

import os
from typing import Any, Dict, List, Tuple

import numpy as np

from signal_context import align_features_to_schema, compose_policy_features


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
        rule_signals = r.get("rule_signals") or {}
        gates = r.get("gates") or {}
        teacher_strength = _safe_float(r.get("teacher_strength", 0.0), 0.0)
        pol = compose_policy_features(
            features=feats,
            rule_signals=rule_signals,
            gates=gates,
            teacher_strength=teacher_strength,
        )
        aligned, _, _ = align_features_to_schema(pol, cols)
        for j, c in enumerate(cols):
            X[i, j] = _safe_float(aligned.get(c, 0.0), 0.0)
        lab = str(r.get("label", "FLAT")).upper()
        teacher_dir = str(r.get("teacher_dir", "")).upper()
        y[i] = 1 if lab == teacher_dir else 0
    return X, y


def _predict_proba(model, X: np.ndarray) -> np.ndarray:
    try:
        if hasattr(model, "predict_proba"):
            p = model.predict_proba(X)[:, 1]
            return np.asarray(p, dtype=float)
    except Exception:
        pass
    try:
        import xgboost as xgb
        dm = xgb.DMatrix(X)
        p = model.predict(dm)
        return np.asarray(p, dtype=float)
    except Exception:
        return np.zeros((X.shape[0],), dtype=float)


def evaluate_holdout(
    rows: List[Dict[str, Any]],
    schema_cols: List[str],
    buy_model,
    sell_model,
) -> Tuple[bool, Dict[str, Any]]:
    if not rows:
        return False, {"reason": "no_rows"}

    holdout_frac = float(os.getenv("PREPROMO_HOLDOUT_FRAC", "0.1") or "0.1")
    holdout_frac = min(max(holdout_frac, 0.05), 0.5)
    min_holdout = int(os.getenv("PREPROMO_MIN_HOLDOUT_ROWS", "500") or "500")

    rows_sorted = sorted(rows, key=lambda r: r.get("ts_target_close") or "")
    split = int(len(rows_sorted) * (1.0 - holdout_frac))
    holdout = rows_sorted[split:]

    report: Dict[str, Any] = {"holdout_total": len(holdout)}

    min_acc = float(os.getenv("PREPROMO_MIN_ACC", "0.52") or "0.52")
    min_auc = float(os.getenv("PREPROMO_MIN_AUC", "0.52") or "0.52")

    ok = True
    for d, model in (("BUY", buy_model), ("SELL", sell_model)):
        holdout_dir = [
            r for r in holdout
            if str(r.get("teacher_dir", "")).upper() == d and bool(r.get("teacher_tradeable"))
        ]
        if len(holdout_dir) < min_holdout:
            report[f"{d.lower()}_reason"] = "holdout_too_small"
            report[f"{d.lower()}_holdout"] = len(holdout_dir)
            ok = False
            continue

        X, y = _build_matrix(holdout_dir, schema_cols)
        p = _predict_proba(model, X)
        if p.size != y.size:
            report[f"{d.lower()}_reason"] = "predict_failed"
            ok = False
            continue

        preds = (p >= 0.5).astype(int)
        acc = float(np.mean(preds == y))

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

        report[f"{d.lower()}_holdout"] = len(holdout_dir)
        report[f"{d.lower()}_acc"] = acc
        report[f"{d.lower()}_auc"] = auc
        report[f"{d.lower()}_min_acc"] = min_acc
        report[f"{d.lower()}_min_auc"] = min_auc

        if acc < min_acc or auc < min_auc:
            ok = False

    return ok, report
