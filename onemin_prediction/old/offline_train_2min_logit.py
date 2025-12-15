#!/usr/bin/env python3
"""
offline_train_2min_logit.py

Offline evaluation for the trade-window TP/SL outcome model.

- Uses offline_train_2min.build_2min_dataset(), which now:
    * labels each bar by which side (long/short) is more likely to hit TP before SL
      within TRADE_HORIZON_MIN minutes.
    * keeps FLAT for ambiguous / no-edge bars.

All metrics here are computed over BUY/SELL labels (direction of the better trade side),
optionally with FLAT handling where applicable.
"""

import os
import logging
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import joblib

from offline_train_2min import (
    fetch_intraday_range,
    build_2min_dataset,
    _parse_train_datetime,
)

logger = logging.getLogger(__name__)
_high_verbose = os.getenv("LOG_HIGH_VERBOSITY", "1").lower() in ("1", "true", "yes")
_level = logging.DEBUG if _high_verbose else logging.INFO
logging.basicConfig(
    level=_level,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)

# Core features that should already exist in the 2-min dataset
CORE_FEATURES: List[str] = [
    "ema_8",
    "ema_21",
    "micro_slope",
    "micro_imbalance",
    "mean_drift_pct",
    "last_price",
    "last_zscore",
    "atr_1t",
    "atr_3t",
    "rv_10",
    "wick_extreme_up",
    "wick_extreme_down",
    "vwap_reversion_flag",
    "cvd_divergence",
]


def _label_to_int(lbl: str) -> int:
    """Map string label -> int for logistic: BUY=1, SELL=0, FLAT/other=-1."""
    if lbl == "BUY":
        return 1
    if lbl == "SELL":
        return 0
    return -1


def _build_logit_dataset(df: pd.DataFrame) -> Optional[Tuple[np.ndarray, np.ndarray, List[str]]]:
    """Prepare X, y, feature list for logistic training."""
    if df is None or df.empty:
        logger.error("[LOGIT] Empty training DataFrame.")
        return None

    if "label" not in df.columns:
        logger.error("[LOGIT] Dataset missing 'label' column.")
        return None

    df = df.copy()
    df["y"] = df["label"].map(_label_to_int).astype(int)

    # Drop FLAT / invalid rows
    df = df.loc[df["y"] >= 0]
    if df.empty:
        logger.error("[LOGIT] No BUY/SELL rows after filtering; cannot train.")
        return None

    feat_cols = [c for c in CORE_FEATURES if c in df.columns]
    if not feat_cols:
        logger.error("[LOGIT] None of CORE_FEATURES are present in DataFrame.")
        return None

    df_feat = (
        df[feat_cols]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .astype("float64")
    )

    X = df_feat.values
    y = df["y"].values

    # Verification #1 / #16 – ensure finite matrix
    if not np.isfinite(X).all():
        logger.error("[LOGIT] Feature matrix contains NaN or inf after cleaning.")
        return None

    pos_rate = float(y.mean()) if len(y) else 0.0
    logger.info(
        "[LOGIT] Training dataset: n=%d, pos_rate=%.3f, n_features=%d",
        len(y),
        pos_rate,
        len(feat_cols),
    )
    return X, y, feat_cols


def train_logit_model(df: pd.DataFrame, out_path: str) -> None:
    """Train and persist the logistic regression model."""
    dataset = _build_logit_dataset(df)
    if dataset is None:
        return

    X, y, feat_cols = dataset

    # Scale features to improve convergence of lbfgs
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression(
        solver="lbfgs",
        max_iter=int(os.getenv("LOGIT_MAX_ITER", "3000")),  # higher cap to avoid convergence warnings
        class_weight="balanced",
        n_jobs=-1,
        penalty="l2",
    )

    model.fit(X_scaled, y)

    coef_norm = float(np.linalg.norm(model.coef_))
    logger.info("[LOGIT] Logistic model trained (coef_norm=%.3f)", coef_norm)

    payload = {
        "model": model,
        "feature_names": feat_cols,
        "scaler": scaler,
    }

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    tmp_path = out_path + ".tmp"
    joblib.dump(payload, tmp_path)
    os.replace(tmp_path, out_path)
    logger.info("[LOGIT] Saved logistic model to %s", out_path)


def main() -> None:
    start_str = os.getenv("TRAIN_START_DATE", "").strip()
    end_str = os.getenv("TRAIN_END_DATE", "").strip()
    if not start_str or not end_str:
        logger.error("[LOGIT] TRAIN_START_DATE and TRAIN_END_DATE must be set.")
        return

    try:
        start_date = _parse_train_datetime(start_str)
        end_date = _parse_train_datetime(end_str)
    except Exception as e:
        logger.error("[LOGIT] Invalid TRAIN_*_DATE values: %s", e)
        return

    if end_date < start_date:
        logger.error("[LOGIT] TRAIN_END_DATE must be >= TRAIN_START_DATE.")
        return

    out_path = os.getenv("LOGIT_PATH", "").strip()
    if not out_path:
        logger.error("[LOGIT] LOGIT_PATH must be set for logistic model output.")
        return

    df_intraday = fetch_intraday_range(start_date, end_date)
    if df_intraday is None or df_intraday.empty:
        logger.error("[LOGIT] No intraday data returned for training window.")
        return

    df_2min = build_2min_dataset(df_intraday)
    if df_2min is None or df_2min.empty:
        logger.error("[LOGIT] 2-minute dataset empty after build_2min_dataset.")
        return

    train_logit_model(df_2min, out_path)


if __name__ == "__main__":
    main()
