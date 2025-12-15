#!/usr/bin/env python3
"""
offline_train_calibrator_5min.py

Fit a Platt-style calibrator for the CURRENT 5-minute XGB model and
write it to CALIB_PATH.

Design:
- Reuses offline_train_5min._load_all_features and _select_feature_columns
  so we get *exactly* the same features the XGB was trained on.
- Loads the trained XGB booster from XGB_OUT_PATH.
- Computes raw BUY probabilities on all BUY/SELL rows.
- Fits a 1D logistic regression on logit(p_xgb) -> label (BUY=1, SELL=0).
- Saves {"a": float, "b": float} JSON so model_pipeline can apply:
      p_calib = sigmoid(a * logit(p_xgb) + b)

If data is unsuitable (no XGB model, not enough BUY/SELL, etc.), it exits
cleanly and does NOT write/overwrite calibrator.json.
"""

import json
import os
from pathlib import Path
from typing import Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression

import xgboost as xgb  # type: ignore

# Import helpers from offline_train_regen_v2
import offline_train_regen_v2 as ot  # type: ignore


# Where to write the calibrator (same env var model_pipeline uses)
CALIB_PATH = Path(
    os.getenv(
        "CALIB_PATH",
        str(Path("trained_models/production/calibrator.json")),
    )
)

# Minimum directional rows required for a stable fit
MIN_DIR_ROWS = int(os.getenv("CALIB_OFFLINE_MIN_DIR_ROWS", "500") or "500")


def _load_directional_dataset() -> Tuple[np.ndarray, np.ndarray, list]:
    """
    Load all historical features via offline_train_5min, keep BUY/SELL rows,
    and build X,y plus the feature column list.

    Returns:
        X: 2D np.ndarray of shape (n_samples, n_features)
        y: 1D np.ndarray of 0/1 labels (SELL=0, BUY=1)
        feat_cols: list of feature column names (order used to build X)
    """
    df = ot._load_all_features()
    if df is None or df.empty:
        print("[CALIB] No data from _load_all_features; aborting.")
        return np.empty((0, 0)), np.empty((0,)), []

    # Keep only directional rows
    df_dir = df[df["label"].isin(["BUY", "SELL"])].copy()
    if df_dir.empty:
        print("[CALIB] No BUY/SELL rows in features; aborting.")
        return np.empty((0, 0)), np.empty((0,)), []

    if len(df_dir) < MIN_DIR_ROWS:
        print(f"[CALIB] directional rows insufficient: {len(df_dir)}/{MIN_DIR_ROWS}")
        return np.empty((0, 0)), np.empty((0,)), []

    # Select feature columns using the same logic as offline_train_5min
    feat_cols = ot._select_feature_columns(df_dir)

    X = (
        df_dir[feat_cols]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .values
    )
    y = (df_dir["label"].astype(str).str.upper() == "BUY").astype(int).values

    # Remove any rows that became weird during cleaning
    if X.shape[0] != y.shape[0] or X.shape[0] == 0:
        print("[CALIB] After cleaning, X and y are misaligned or empty.")
        return np.empty((0, 0)), np.empty((0,)), []

    return X, y, feat_cols


def main() -> None:
    # Ensure we actually have a trained booster
    xgb_path = ot.XGB_OUT_PATH
    if not xgb_path.exists():
        print(f"[CALIB] XGB model not found at {xgb_path}; run offline_train_5min.py first.")
        return

    X, y, feat_cols = _load_directional_dataset()
    if X.size == 0 or y.size == 0 or not feat_cols:
        print(
            "[CALIB] Dataset not suitable for calibration yet; "
            "calibrator.json will not be created/updated."
        )
        return

    # Load booster and compute raw BUY probabilities
    booster = xgb.Booster()
    booster.load_model(str(xgb_path))

    dmat = xgb.DMatrix(X, feature_names=feat_cols)
    p = booster.predict(dmat)
    p = np.clip(p, 1e-6, 1.0 - 1e-6)

    # Basic sanity: must have both classes present
    unique, counts = np.unique(y, return_counts=True)
    if len(unique) < 2:
        label_map = {0: "SELL", 1: "BUY"}
        pretty_counts = {label_map.get(int(k), str(k)): int(v) for k, v in zip(unique, counts)}
        print(f"[CALIB] Need both BUY and SELL for calibration; got counts: {pretty_counts}")
        return

    # Platt scaling: logistic on logit(p_xgb)
    logit_p = np.log(p / (1.0 - p)).reshape(-1, 1)

    clf = LogisticRegression(max_iter=800, class_weight="balanced")
    clf.fit(logit_p, y)

    a = float(clf.coef_.ravel()[0])
    b = float(clf.intercept_.ravel()[0])

    CALIB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CALIB_PATH.open("w", encoding="utf-8") as f:
        json.dump({"a": a, "b": b}, f, indent=2)

    print(f"[CALIB] Saved calibrator to {CALIB_PATH} (a={a:.4f}, b={b:.4f})")


if __name__ == "__main__":
    main()
