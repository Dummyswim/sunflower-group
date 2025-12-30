#!/usr/bin/env python3
"""
Train a move/edge head (binary classifier) using SignalContext logs.

Target (Option A):
  y_move = 1 if abs(aux_ret_main) > k * (atr_1t / close)
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import xgboost as xgb
except Exception:
    xgb = None

from signal_context import align_features_to_schema, compose_policy_features, compose_policy_schema
from signal_log_utils import load_signal_log
from online_trainer_regen_v2_bundle import _load_schema_cols


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        if np.isfinite(v):
            return v
    except Exception:
        pass
    return float(default)


def _load_base_schema() -> Optional[List[str]]:
    sp = os.getenv("FEATURE_SCHEMA_COLS_PATH", "").strip()
    if sp:
        cols = _load_schema_cols(Path(sp))
        if cols:
            return cols
    cols = _load_schema_cols(Path("data/feature_schema_cols.json"))
    if cols:
        return cols
    production_link = Path(os.getenv("MODEL_PRODUCTION_LINK", "trained_models/production"))
    try:
        if production_link.exists():
            resolved = production_link.resolve()
            cols = _load_schema_cols(resolved / "feature_schema_cols.json")
            if cols:
                return cols
    except Exception:
        pass
    cols = _load_schema_cols(production_link.parent / "feature_schema_cols.json")
    if cols:
        return cols
    return None


def _load_policy_schema() -> Optional[List[str]]:
    sp = os.getenv("POLICY_SCHEMA_COLS_PATH", "").strip()
    if sp:
        cols = _load_schema_cols(Path(sp))
        if cols:
            return cols
    production_link = Path(os.getenv("MODEL_PRODUCTION_LINK", "trained_models/production"))
    try:
        if production_link.exists():
            resolved = production_link.resolve()
            cols = _load_schema_cols(resolved / "policy_schema_cols.json")
            if cols:
                return cols
    except Exception:
        pass
    cols = _load_schema_cols(production_link.parent / "policy_schema_cols.json")
    if cols:
        return cols
    return None


def _build_move_dataset(
    rows: List[Dict[str, Any]],
    cols: List[str],
    *,
    atr_mult: float,
    min_label_weight: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    X = []
    y = []
    w = []
    for r in rows:
        feats = r.get("features") or {}
        prov = r.get("provenance") or {}
        ret = prov.get("aux_ret_main")
        if ret is None:
            continue
        ret = _safe_float(ret, 0.0)
        atr_1t = _safe_float(feats.get("atr_1t", 0.0), 0.0)
        close = _safe_float(feats.get("close", feats.get("last_price", 0.0)), 0.0)
        if atr_1t <= 0.0 or close <= 0.0:
            continue
        atr_ret = atr_1t / max(abs(close), 1e-6)
        thr = float(atr_mult) * float(atr_ret)
        y_move = 1 if abs(ret) > thr else 0

        label_weight = _safe_float(r.get("label_weight", 1.0), 1.0)
        if label_weight < min_label_weight:
            continue

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
        X.append([_safe_float(aligned.get(c, 0.0), 0.0) for c in cols])
        y.append(int(y_move))
        w.append(float(label_weight))

    if not X:
        return np.zeros((0, len(cols)), dtype=np.float32), np.zeros((0,), dtype=np.int32), np.zeros((0,), dtype=np.float32), 0, 0

    X_arr = np.asarray(X, dtype=np.float32)
    y_arr = np.asarray(y, dtype=np.int32)
    w_arr = np.asarray(w, dtype=np.float32)
    pos = int(np.sum(y_arr == 1))
    neg = int(np.sum(y_arr == 0))
    return X_arr, y_arr, w_arr, pos, neg


def _train_move_model(X: np.ndarray, y: np.ndarray, w: np.ndarray, pos: int, neg: int) -> Optional[Any]:
    if xgb is None or X.size == 0:
        return None

    pos = max(1, int(pos))
    neg = max(1, int(neg))
    try:
        scale_pos_weight = float(os.getenv("MOVE_XGB_SCALE_POS_WEIGHT", ""))
    except Exception:
        scale_pos_weight = float(neg / pos) if pos else 1.0

    params = {
        "n_estimators": int(os.getenv("MOVE_XGB_N_ESTIMATORS", "300")),
        "max_depth": int(os.getenv("MOVE_XGB_MAX_DEPTH", "4")),
        "learning_rate": float(os.getenv("MOVE_XGB_LR", "0.05")),
        "subsample": float(os.getenv("MOVE_XGB_SUBSAMPLE", "0.9")),
        "colsample_bytree": float(os.getenv("MOVE_XGB_COLSAMPLE", "0.9")),
        "reg_lambda": float(os.getenv("MOVE_XGB_L2", "1.0")),
        "min_child_weight": float(os.getenv("MOVE_XGB_MIN_CHILD_WEIGHT", "1.0")),
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "n_jobs": int(os.getenv("MOVE_XGB_N_JOBS", "4")),
        "random_state": int(os.getenv("MOVE_XGB_SEED", "7")),
        "scale_pos_weight": float(scale_pos_weight),
    }

    clf = xgb.XGBClassifier(**params)
    clf.fit(X, y, sample_weight=w)
    return clf


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default=os.getenv("TRAIN_LOG_PATH", "data/train_log_v3_schema_v5.jsonl"))
    ap.add_argument("--min_rows", type=int, default=int(os.getenv("MOVE_MIN_ROWS", "5000")))
    ap.add_argument("--max_rows", type=int, default=int(os.getenv("MOVE_MAX_ROWS", "200000")))
    ap.add_argument("--atr_mult", type=float, default=float(os.getenv("MOVE_ATR_MULT", "0.35")))
    ap.add_argument("--out", default=os.getenv("POLICY_MOVE_PATH", "trained_models/production/policy_move.json"))
    args = ap.parse_args()

    if xgb is None:
        raise SystemExit("xgboost not available; cannot train move head")

    base_schema = _load_base_schema()
    if not base_schema:
        raise SystemExit("FEATURE_SCHEMA_COLS_PATH missing or invalid; base schema required")

    policy_schema = _load_policy_schema()
    if not policy_schema:
        policy_schema = compose_policy_schema(list(base_schema))

    rows = load_signal_log(args.log, max_rows=args.max_rows, schema_cols=list(base_schema))
    if len(rows) < args.min_rows:
        raise SystemExit(f"Not enough rows: {len(rows)} < {args.min_rows}")

    min_label_weight = _safe_float(os.getenv("MOVE_MIN_LABEL_WEIGHT", "0.0"), 0.0)
    X, y, w, pos, neg = _build_move_dataset(
        rows,
        list(policy_schema),
        atr_mult=float(args.atr_mult),
        min_label_weight=min_label_weight,
    )
    if X.size == 0:
        raise SystemExit("No usable rows after filtering for aux_ret_main/atr_1t")

    print(f"[MOVE] rows={len(rows)} used={len(y)} pos={pos} neg={neg} cols={len(policy_schema)}")

    model = _train_move_model(X, y, w, pos, neg)
    if model is None:
        raise SystemExit("training failed")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(out_path))
    print(f"[MOVE] wrote model -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
