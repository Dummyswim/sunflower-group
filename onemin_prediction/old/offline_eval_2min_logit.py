#!/usr/bin/env python3
"""
offline_eval_2min_logit.py

Offline evaluation for the 2-minute *logistic* direction model.

- Fetches 1-minute intraday OHLCV from Dhan /v2/charts/intraday for a date range.
- Builds a 2-minute direction dataset using build_2min_dataset() from offline_train_2min.py.
- Loads the offline-trained logistic model from LOGIT_PATH.
- Computes:
    * Overall directional accuracy on BUY/SELL labels.
    * Accuracy in bins of predicted buy_prob (reliability curve).
    * High-confidence subset accuracy (buy_prob >= threshold).

Env:
    export DHAN_ACCESS_TOKEN="..."
    export DHAN_CLIENT_ID="..."
    export EVAL_START_DATE="2025-09-01 09:15:00"
    export EVAL_END_DATE="2025-11-27 15:30:00"
    export LOGIT_PATH="trained_models/experiments/logit_2min.pkl"
"""

import os
import logging
from typing import Optional, List, Tuple

import numpy as np
import pandas as pd
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

logger.info("offline_eval_2min_logit started (LOG_HIGH_VERBOSITY=%s)", _high_verbose)


def _dir_label_to_int(lbl: str) -> int:
    """Directional encoding: BUY=+1, SELL=-1, FLAT/other=0."""
    if lbl == "BUY":
        return 1
    if lbl == "SELL":
        return -1
    return 0


def _load_logit_payload(path: str):
    """Load saved logistic payload {model, scaler, feature_names}."""
    if not path or not os.path.exists(path):
        logger.error("[LOGIT_EVAL] LOGIT_PATH does not exist: %s", path)
        return None
    try:
        payload = joblib.load(path)
    except Exception as e:
        logger.error("[LOGIT_EVAL] Failed to load logistic payload: %s", e)
        return None

    model = payload.get("model")
    scaler = payload.get("scaler")
    feat_names = payload.get("feature_names")

    if model is None or scaler is None or not feat_names:
        logger.error(
            "[LOGIT_EVAL] Payload missing model/scaler/feature_names. Keys: %s",
            list(payload.keys()),
        )
        return None

    logger.info(
        "[LOGIT_EVAL] Loaded logistic payload from %s (n_features=%d)",
        path,
        len(feat_names),
    )
    return model, scaler, list(feat_names)


def _build_eval_matrix(
    df: pd.DataFrame,
    feat_names: List[str],
) -> Optional[Tuple[np.ndarray, np.ndarray, List[str]]]:
    """
    Build (X, y_true, feat_cols) from a 2-minute dataset DataFrame for eval.

    - y_true: directional ints (1=BUY, -1=SELL, 0=FLAT).
    - X: features aligned to feat_names from the trained payload.
    """
    if df is None or df.empty:
        logger.error("[LOGIT_EVAL] Empty evaluation dataset")
        return None

    if "label" not in df.columns:
        logger.error("[LOGIT_EVAL] Dataset missing 'label' column")
        return None

    df = df.copy()
    df["dir_true"] = df["label"].map(_dir_label_to_int).fillna(0).astype(int)

    # Use only the features the model was trained with
    available = set(df.columns)
    feat_cols = [c for c in feat_names if c in available]
    missing = [c for c in feat_names if c not in available]

    if missing:
        logger.warning(
            "[LOGIT_EVAL] Missing trained features in eval DataFrame: %s",
            missing,
        )
    if not feat_cols:
        logger.error(
            "[LOGIT_EVAL] None of the trained feature names are present in eval DataFrame."
        )
        return None

    df_feat = (
        df[feat_cols]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .astype("float64")
    )

    X = df_feat.values
    y_true = df["dir_true"].values.astype(int)

    # Verification #1 / #16 – finite matrix check
    if not np.isfinite(X).all():
        logger.error("[LOGIT_EVAL] Feature matrix contains NaN or inf after cleaning.")
        return None

    return X, y_true, feat_cols


def _bin_accuracy(y_true: np.ndarray, p_buy: np.ndarray, bins, label: str = "buy_prob") -> None:
    """
    Print directional accuracy in bins of predicted probability for BUY.

    Only counts BUY/SELL labels (dir_true in {+1,-1}).
    """
    logger.info("=== [LOGIT] Accuracy vs %s bins ===", label)
    y = y_true
    p = p_buy

    mask_dir = np.isin(y, [1, -1])
    if not mask_dir.any():
        logger.warning("[LOGIT] No BUY/SELL labels in dataset; skipping bin analysis.")
        return

    y_dir = y[mask_dir]
    p_dir = p[mask_dir]

    cats = pd.cut(p_dir, bins=bins, include_lowest=True)
    for b in cats.categories:
        mask_bin = (cats == b)
        if not mask_bin.any():
            continue
        y_b = y_dir[mask_bin]
        p_b = p_dir[mask_bin]
        if y_b.size == 0:
            continue
        y_pred = np.sign(p_b - 0.5).astype(int)
        acc = float((y_pred == y_b).mean())
        logger.info("  %s: n=%d, acc=%.3f", str(b), int(y_b.size), acc)


def main() -> None:
    # ----- Config from env -----
    eval_start = os.getenv("EVAL_START_DATE", "").strip() or os.getenv("TRAIN_START_DATE", "").strip()
    eval_end = os.getenv("EVAL_END_DATE", "").strip() or os.getenv("TRAIN_END_DATE", "").strip()

    if not eval_start or not eval_end:
        logger.error(
            "[LOGIT_EVAL] EVAL_START_DATE/EVAL_END_DATE (or TRAIN_*_DATE) must be set "
            "(YYYY-MM-DD or 'YYYY-MM-DD HH:MM:SS')."
        )
        return

    logit_path = os.getenv("LOGIT_PATH", "").strip()
    if not logit_path:
        logger.error("[LOGIT_EVAL] LOGIT_PATH must be set to the logistic model path.")
        return

    try:
        start_date = _parse_train_datetime(eval_start)
        end_date = _parse_train_datetime(eval_end)
    except Exception as e:
        logger.error("[LOGIT_EVAL] Invalid EVAL_START_DATE/EVAL_END_DATE: %s", e)
        return

    if end_date < start_date:
        logger.error("[LOGIT_EVAL] EVAL_END_DATE must be >= EVAL_START_DATE.")
        return

    payload = _load_logit_payload(logit_path)
    if payload is None:
        return
    model, scaler, feat_names = payload

    # ----- Fetch intraday data -----
    logger.info(
        "[LOGIT_EVAL] Fetching intraday candles for evaluation range %s to %s",
        start_date.isoformat(),
        end_date.isoformat(),
    )
    df_intraday = fetch_intraday_range(start_date, end_date)
    if df_intraday is None or df_intraday.empty:
        logger.error("[LOGIT_EVAL] No intraday data fetched for evaluation.")
        return

    # ----- Build 2-minute dataset -----
    df_eval = build_2min_dataset(df_intraday)
    if df_eval is None or df_eval.empty:
        logger.error("[LOGIT_EVAL] 2-minute evaluation dataset is empty.")
        return

    logger.info("[LOGIT_EVAL] 2-minute evaluation dataset rows: %d", len(df_eval))

    # ----- Build feature matrix -----
    out = _build_eval_matrix(df_eval, feat_names)
    if out is None:
        return
    X, y_true, used_feats = out
    logger.info(
        "[LOGIT_EVAL] Feature matrix shape: X=%s, labels=%s, n_features=%d",
        X.shape,
        y_true.shape,
        len(used_feats),
    )

    # ----- Predict buy_prob -----
    X_scaled = scaler.transform(X)
    try:
        proba = model.predict_proba(X_scaled)
    except Exception as e:
        logger.error("[LOGIT_EVAL] model.predict_proba failed: %s", e)
        return

    proba = np.asarray(proba, dtype=float)
    if proba.ndim == 1:
        p_buy = np.clip(proba, 1e-9, 1.0 - 1e-9)
    else:
        # Assume column 1 is BUY
        p_buy = np.clip(proba[:, 1], 1e-9, 1.0 - 1e-9)

    if p_buy.shape[0] != y_true.shape[0]:
        logger.error(
            "[LOGIT_EVAL] Prediction length mismatch: p_buy=%d, y_true=%d",
            p_buy.shape[0],
            y_true.shape[0],
        )
        return

    # ----- Overall directional accuracy (BUY/SELL) -----
    mask_dir = np.isin(y_true, [1, -1])
    if not mask_dir.any():
        logger.error("[LOGIT_EVAL] No BUY/SELL labels in evaluation dataset.")
        return

    y_dir = y_true[mask_dir]
    p_dir = p_buy[mask_dir]

    y_pred_dir = np.sign(p_dir - 0.5).astype(int)
    overall_acc = float((y_pred_dir == y_dir).mean())
    logger.info("[LOGIT_EVAL] Overall directional accuracy (BUY/SELL only): %.3f", overall_acc)
    logger.info(
        "[LOGIT_EVAL] Directional rows: %d out of %d",
        int(mask_dir.sum()),
        int(len(y_true)),
    )

    # ----- Reliability by probability bins -----
    prob_bins_env = os.getenv("OFFLINE2_PROB_BINS", "")
    if prob_bins_env:
        try:
            bins = list(map(float, prob_bins_env.strip("[] ").split(",")))
        except Exception:
            bins = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    else:
        bins = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    _bin_accuracy(y_true, p_buy, bins=bins, label="buy_prob")

    # ----- High-confidence subset -----
    try:
        hc_thresh = float(os.getenv("OFFLINE2_HC_THRESH", "0.7") or "0.7")
    except Exception:
        hc_thresh = 0.7

    mask_hc = mask_dir & (p_buy >= hc_thresh)
    if mask_hc.any():
        y_hc = y_true[mask_hc]
        p_hc = p_buy[mask_hc]
        y_pred_hc = np.sign(p_hc - 0.5).astype(int)
        acc_hc = float((y_pred_hc == y_hc).mean())
        logger.info(
            "[LOGIT_EVAL] High-confidence subset (buy_prob>=%.2f): n=%d, acc=%.3f",
            hc_thresh,
            int(mask_hc.sum()),
            acc_hc,
        )
    else:
        logger.info(
            "[LOGIT_EVAL] High-confidence subset (buy_prob>=%.2f): n=0",
            hc_thresh,
        )


if __name__ == "__main__":
    main()
