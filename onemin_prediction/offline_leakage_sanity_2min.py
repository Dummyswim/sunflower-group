"""
offline_leakage_sanity_2min.py

One-off helper to test for structural label leakage in the 2-minute pipeline.

Usage (example):

    export LEAKAGE_START_DATE="2025-11-24 09:15:00"
    export LEAKAGE_END_DATE="2025-11-24 15:30:00"
    python offline_leakage_sanity_2min.py

This will:
    - fetch 1-min candles for the given range,
    - build the 2-min dataset via offline_train_2min.build_2min_dataset,
    - train a small XGB classifier on true labels (BUY vs SELL on directional rows),
    - train another on shuffled labels,
    - report cross-validated ROC AUC for both.

If true-label AUC >> shuffled-label AUC (~0.5), there is likely real signal and
no obvious structural leakage. If both are high, there is probably leakage
(features peeking at the future label).
"""

import os
import logging
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

from offline_train_2min import fetch_intraday_range_2min, build_2min_dataset, _parse_train_datetime

"""
Purpose:
- One-off sanity test for structural leakage.
- Shuffles labels and retrains XGB; if AUC stays >> 0.5,
  your feature pipeline is leaking future information.
NOT for daily use; run only when changing label/feature logic.
"""

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)


def _build_xy(df_2min: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
    df = df_2min.copy()
    if "is_directional" not in df.columns:
        df["is_directional"] = df["label"].isin(["BUY", "SELL"]).astype(int)
    df = df[df["is_directional"] == 1].copy()
    if df.empty:
        raise ValueError("No BUY/SELL rows in leakage dataset.")

    # Directional label 1=BUY, -1=SELL
    df["y_dir"] = np.where(df["label"] == "BUY", 1, -1).astype(int)

    # Drop non-feature columns
    exclude = {"ts", "label", "is_directional", "y_dir"}
    exclude |= {c for c in df.columns if df[c].dtype == "O"}
    feat_cols = [c for c in df.columns if c not in exclude]
    if not feat_cols:
        raise ValueError("No numeric features found for leakage test.")

    X = df[feat_cols].astype(float)
    y = df["y_dir"].values
    logger.info(
        "Leakage check feature matrix shape: X=%s, y=%s, n_features=%d",
        X.shape,
        y.shape,
        len(feat_cols),
    )
    return X, y


def _cv_auc(X: pd.DataFrame, y: np.ndarray, seed: int = 42) -> float:
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    aucs = []
    for train_idx, test_idx in skf.split(X, y):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        model = XGBClassifier(
            max_depth=3,
            n_estimators=80,
            learning_rate=0.1,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            objective="binary:logistic",
            n_jobs=4,
            tree_method="hist",
        )
        model.fit(X_tr, y_tr)
        p = model.predict_proba(X_te)[:, 1]
        auc = roc_auc_score(y_te, p)
        aucs.append(auc)

    return float(np.mean(aucs))


def main() -> None:
    logger.warning("Running leakage sanity test: this script is NOT for daily production use.")

    # 1) Resolve date range
    start_str = (
        os.getenv("LEAK_TEST_START_DATE", "2025-11-24 09:15:00")
        or os.getenv("LEAKAGE_START_DATE")
        or os.getenv("EVAL_START_DATE")
        or os.getenv("TRAIN_START_DATE")
    )
    end_str = (
        os.getenv("LEAK_TEST_END_DATE", "2025-11-24 15:30:00")
        or os.getenv("LEAKAGE_END_DATE")
        or os.getenv("EVAL_END_DATE")
        or os.getenv("TRAIN_END_DATE")
    )
    if not start_str or not end_str:
        logger.error("LEAKAGE_START_DATE/LEAKAGE_END_DATE (or EVAL_*/TRAIN_*) must be set.")
        return

    start_dt = _parse_train_datetime(start_str)
    end_dt = _parse_train_datetime(end_str)
    if end_dt < start_dt:
        logger.error("LEAKAGE_END_DATE must be >= LEAKAGE_START_DATE.")
        return

    logger.info("Leakage check on range: %s -> %s", start_dt, end_dt)

    # 2) Fetch candles and build 2-min dataset
    df_intraday = fetch_intraday_range_2min(start_dt, end_dt)
    if df_intraday is None or df_intraday.empty:
        logger.error("No intraday data fetched for leakage range.")
        return

    df_2min = build_2min_dataset(df_intraday)
    if df_2min is None or df_2min.empty:
        logger.error("No 2-min rows built for leakage range.")
        return

    X, y = _build_xy(df_2min)

    # 3) True labels CV AUC
    auc_true = _cv_auc(X, y)
    logger.info("[LEAKAGE] True-label CV AUC: %.3f", auc_true)

    # 4) Shuffled labels CV AUC
    y_shuffled = np.random.permutation(y)
    auc_shuf = _cv_auc(X, y_shuffled)
    logger.info("[LEAKAGE] Shuffled-label CV AUC: %.3f", auc_shuf)

    logger.info(
        "Leakage sanity summary: if AUC_true ≫ AUC_shuffled (~0.5), signal is likely real; "
        "if both are high, investigate features for future leakage."
    )


if __name__ == "__main__":
    main()
