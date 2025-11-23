#!/usr/bin/env python3
"""
offline_eval_2min.py

Offline evaluation for the 2-minute direction model.

- Fetches 1-minute intraday OHLCV from Dhan /v2/charts/intraday for a date range.
- Builds a 2-minute direction dataset using build_2min_dataset() from offline_train_2min.py.
- Loads the offline-trained XGB model from XGB_PATH.
- Computes:
    * Overall directional accuracy on BUY/SELL labels.
    * Accuracy in bins of predicted buy_prob (reliability curve).
    * High-confidence subset accuracy (buy_prob >= threshold).

Usage:
    export DHAN_ACCESS_TOKEN="..."
    export DHAN_CLIENT_ID="..."
    export EVAL_START_DATE="2024-11-06"
    export EVAL_END_DATE="2024-11-08"
    export XGB_PATH="trained_models/experiments/xgb_2min.json"
    python offline_eval_2min.py
"""

import os
import logging
from typing import Optional

import numpy as np
import pandas as pd


from offline_train_2min import fetch_intraday_range, build_2min_dataset, _parse_train_datetime



logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
)


def _dir_label_to_int(lbl: str) -> int:
    if lbl == "BUY":
        return 1
    if lbl == "SELL":
        return -1
    return 0


def _load_xgb_model(path: str):
    if not path or not os.path.exists(path):
        logger.error("XGB_PATH does not exist: %s", path)
        return None
    try:
        import xgboost as xgb
        booster = xgb.Booster()
        booster.load_model(path)
        logger.info("Loaded XGB model from %s", path)
        return booster
    except Exception as e:
        logger.error("Failed to load XGB model: %s", e)
        return None


def _build_feature_matrix(df: pd.DataFrame) -> Optional[tuple[np.ndarray, np.ndarray, list[str]]]:
    """
    Build (X, y, feat_cols) from a 2-minute dataset DataFrame:
      - X: feature matrix
      - y: directional labels (1=BUY, -1=SELL, 0=FLAT)
      - feat_cols: list of feature column names used
    """
    if df is None or df.empty:
        logger.error("Empty evaluation dataset")
        return None

    # Map labels
    df = df.copy()
    if "label" not in df.columns:
        logger.error("Dataset missing 'label' column")
        return None
    df["dir_true"] = df["label"].map(_dir_label_to_int).fillna(0).astype(int)

    exclude = {"ts", "label", "dir_true"}
    drop_prefixes = ("meta_", "p_xgb_")
    feat_cols = sorted([
        c for c in df.columns
        if (c not in exclude)
        and (df[c].dtype != "O")
        and not any(c.startswith(p) for p in drop_prefixes)
    ])
    if not feat_cols:
        logger.error("No numeric feature columns found for evaluation.")
        return None

    X = (
        df[feat_cols]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .values
    )
    y = df["dir_true"].values.astype(int)
    return X, y, feat_cols


def _predict_buy_prob(booster, X: np.ndarray) -> np.ndarray:
    """
    Use an xgboost.Booster to get BUY probabilities for each row.
    """
    import xgboost as xgb

    dm = xgb.DMatrix(X)
    p = booster.predict(dm)
    p = np.asarray(p, dtype=float)
    if p.ndim == 1:
        p = np.clip(p, 1e-9, 1.0 - 1e-9)
        return p
    # If model output is multi-class or 2-column [p0,p1], assume second col is BUY
    if p.shape[1] >= 2:
        p1 = np.clip(p[:, 1], 1e-9, 1.0 - 1e-9)
        return p1
    # Fallback
    p_flat = np.clip(p.ravel(), 1e-9, 1.0 - 1e-9)
    return p_flat


def _bin_accuracy(y_true: np.ndarray, p_buy: np.ndarray, bins, label: str = "buy_prob") -> None:
    """
    Print directional accuracy in bins of predicted probability for BUY.
    Only counts BUY/SELL labels (dir_true in {+1,-1}).
    """
    logger.info("=== Accuracy vs %s bins ===", label)
    y = y_true
    p = p_buy

    # directional mask
    mask_dir = np.isin(y, [1, -1])
    if not mask_dir.any():
        logger.warning("No BUY/SELL labels in dataset; skipping bin analysis.")
        return

    y_dir = y[mask_dir]
    p_dir = p[mask_dir]

    # Assign bins
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


def main():
    # ----- Config from env -----    

    eval_start = os.getenv("EVAL_START_DATE", "").strip() or os.getenv("TRAIN_START_DATE", "").strip()
    eval_end = os.getenv("EVAL_END_DATE", "").strip() or os.getenv("TRAIN_END_DATE", "").strip()

    if not eval_start or not eval_end:
        logger.error(
            "EVAL_START_DATE/EVAL_END_DATE (or TRAIN_START_DATE/TRAIN_END_DATE) "
            "must be set (YYYY-MM-DD or 'YYYY-MM-DD HH:MM:SS')."
        )
        return


    xgb_path = os.getenv("XGB_PATH", "").strip()


    try:
        # Reuse the same parser as offline_train_2min; time part is ignored.
        start_date = _parse_train_datetime(eval_start)
        end_date = _parse_train_datetime(eval_end)
    except Exception as e:
        logger.error("Invalid EVAL_START_DATE/EVAL_END_DATE: %s", e)
        return
    
    if end_date < start_date:
        logger.error("EVAL_END_DATE must be >= EVAL_START_DATE.")
        return

    if not xgb_path:
        logger.error("XGB_PATH must be set to the 2-minute XGB model path.")
        return

    booster = _load_xgb_model(xgb_path)
    if booster is None:
        return

    # ----- Fetch intraday data -----
    logger.info("Fetching intraday candles for evaluation range %s to %s",
                start_date.isoformat(), end_date.isoformat())
    df_intraday = fetch_intraday_range(start_date, end_date)
    if df_intraday is None or df_intraday.empty:
        logger.error("No intraday data fetched for evaluation.")
        return

    # ----- Build 2-minute dataset -----
    df_eval = build_2min_dataset(df_intraday)
    if df_eval is None or df_eval.empty:
        logger.error("2-minute evaluation dataset is empty.")
        return

    logger.info("2-minute evaluation dataset rows: %d", len(df_eval))

    # ----- Build feature matrix -----
    out = _build_feature_matrix(df_eval)
    if out is None:
        return
    X, y_true, feat_cols = out
    logger.info("Feature matrix shape: X=%s, labels=%s, n_features=%d",
                X.shape, y_true.shape, len(feat_cols))

    # ----- Predict buy_prob -----
    p_buy = _predict_buy_prob(booster, X)
    if p_buy.shape[0] != y_true.shape[0]:
        logger.error("Prediction length mismatch: p_buy=%d, y_true=%d",
                     p_buy.shape[0], y_true.shape[0])
        return

    # ----- Overall directional accuracy (BUY/SELL) -----
    mask_dir = np.isin(y_true, [1, -1])
    if not mask_dir.any():
        logger.error("No BUY/SELL labels in evaluation dataset.")
        return

    y_dir = y_true[mask_dir]
    p_dir = p_buy[mask_dir]

    y_pred_dir = np.sign(p_dir - 0.5).astype(int)
    overall_acc = float((y_pred_dir == y_dir).mean())
    logger.info("Overall directional accuracy (BUY/SELL only): %.3f", overall_acc)
    logger.info("Directional rows: %d out of %d", int(mask_dir.sum()), int(len(y_true)))

    # ----- Reliability by probability bins -----
    prob_bins_env = os.getenv("OFFLINE2_PROB_BINS", "")
    if prob_bins_env:
        try:
            # Expect a JSON-like list, e.g. "[0.5,0.6,0.7,0.8,0.9,1.0]"
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
            "High-confidence subset (buy_prob>=%.2f): n=%d, acc=%.3f",
            hc_thresh,
            int(mask_hc.sum()),
            acc_hc,
        )
    else:
        logger.info(
            "High-confidence subset (buy_prob>=%.2f): n=0",
            hc_thresh,
        )


if __name__ == "__main__":
    main()
