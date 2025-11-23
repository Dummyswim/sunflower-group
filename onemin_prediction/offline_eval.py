#!/usr/bin/env python3

"""
Offline evaluation and reliability curves for the probabilities-only 2-minute system.

- Joins trained_models/production/signals.jsonl (pre-close predictions) with feature_log_hist.csv
  (2-minute directional labels + features, logged at the target close time).
...
"""

import os
import json
import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from online_trainer import _parse_feature_csv  # re-use existing parser


logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
)


def _load_signals(path: str) -> Optional[pd.DataFrame]:
    if not os.path.exists(path):
        logger.error("Signals file not found: %s", path)
        return None
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                obj = json.loads(ln)
                rows.append(obj)
            except Exception:
                continue
    if not rows:
        logger.error("No valid rows in signals file: %s", path)
        return None
    df = pd.DataFrame(rows)
    if "pred_for" not in df.columns or "buy_prob" not in df.columns:
        logger.error("Signals file missing required columns (pred_for, buy_prob)")
        return None
    return df


def _join_signals_features(
    df_sig: pd.DataFrame,
    df_feat: pd.DataFrame
) -> pd.DataFrame:
    """
    Inner-join signals and feature logs on timestamp:
    signals.pred_for == features.ts
    """
    df_sig = df_sig.copy()
    df_feat = df_feat.copy()

    # Ensure minimal required columns
    if "ts" not in df_feat.columns or "label" not in df_feat.columns:
        raise ValueError("feature_log_hist is missing ts/label columns")

    # Use string keys for join to avoid timezone confusion
    df_sig["pred_for_str"] = df_sig["pred_for"].astype(str)
    df_feat["ts_str"] = df_feat["ts"].astype(str)

    merged = df_sig.merge(
        df_feat,
        left_on="pred_for_str",
        right_on="ts_str",
        how="inner",
        suffixes=("_sig", "_feat")
    )
    return merged


def _dir_label_to_int(lbl: str) -> int:
    if lbl == "BUY":
        return 1
    if lbl == "SELL":
        return -1
    return 0


def _compute_accuracy(merged: pd.DataFrame) -> Tuple[float, float]:
    """
    Returns (overall_acc, tradeable_acc).
    """
    df = merged.copy()
    if "label" not in df.columns:
        logger.error("Merged DataFrame missing 'label' column")
        return 0.0, 0.0

    df["dir_true"] = df["label"].map(_dir_label_to_int).fillna(0).astype(int)
    df["dir_pred"] = np.sign(df["buy_prob"] - 0.5).astype(int)

    mask_dir = df["dir_true"].isin([1, -1])
    if mask_dir.sum() == 0:
        logger.warning("No directional labels (BUY/SELL) in merged data")
        return 0.0, 0.0

    overall_acc = float((df.loc[mask_dir, "dir_pred"] == df.loc[mask_dir, "dir_true"]).mean())

    if "tradeable" in df.columns:
        mask_tr = mask_dir & (df["tradeable"] == True)
        if mask_tr.sum() > 0:
            tradeable_acc = float((df.loc[mask_tr, "dir_pred"] == df.loc[mask_tr, "dir_true"]).mean())
        else:
            tradeable_acc = 0.0
    else:
        tradeable_acc = overall_acc

    return overall_acc, tradeable_acc


def _bin_accuracy(df: pd.DataFrame, col: str, bins) -> None:
    """
    Print accuracy per bin for a probability-like column.
    """
    logger.info("=== Accuracy vs %s bins ===", col)
    if col not in df.columns:
        logger.info("Column %s not present; skipping.", col)
        return

    x = df[col].astype(float)
    df = df.copy()
    df["bin"] = pd.cut(x, bins=bins, include_lowest=True)

    for b in df["bin"].cat.categories:
        sub = df[df["bin"] == b]
        if sub.empty:
            continue
        mask_dir = sub["dir_true"].isin([1, -1])
        if mask_dir.sum() == 0:
            continue
        acc = float((sub.loc[mask_dir, "dir_pred"] == sub.loc[mask_dir, "dir_true"]).mean())
        logger.info("  %s: n=%d, acc=%.3f", str(b), int(mask_dir.sum()), acc)


def main():
    signals_path = os.getenv("SIGNALS_PATH", "trained_models/production/signals.jsonl")
    feat_hist_path = os.getenv("FEATURE_LOG_HIST", "trained_models/production/feature_log_hist.csv")

    logger.info("Loading signals from %s", signals_path)
    df_sig = _load_signals(signals_path)
    if df_sig is None:
        return

    logger.info("Loading feature log from %s", feat_hist_path)
    df_feat = _parse_feature_csv(feat_hist_path, min_rows=0)
    if df_feat is None or df_feat.empty:
        logger.error("No data in feature_log_hist for evaluation")
        return

    try:
        merged = _join_signals_features(df_sig, df_feat)
    except Exception as e:
        logger.error("Join failed: %s", e)
        return

    if merged.empty:
        logger.error("Merged DataFrame is empty (no matching timestamps)")
        return

    logger.info("Merged rows: %d", len(merged))


    # ---- NEW: normalize 'buy_prob' column name after merge ----
    # Because we merged with suffixes=("_sig", "_feat"), overlapping columns
    # like buy_prob become buy_prob_sig / buy_prob_feat.
    buy_col = None
    if "buy_prob_sig" in merged.columns:
        buy_col = "buy_prob_sig"          # prefer signals' pre-close prediction
    elif "buy_prob" in merged.columns:
        buy_col = "buy_prob"
    elif "buy_prob_feat" in merged.columns:
        buy_col = "buy_prob_feat"
    else:
        logger.error(
            "No buy_prob column found after merge "
            "(expected one of: buy_prob_sig, buy_prob, buy_prob_feat). "
            "Available columns: %s",
            list(merged.columns),
        )
        return

    try:
        merged["buy_prob"] = merged[buy_col].astype(float)
    except Exception as e:
        logger.error("Failed to coerce %s to float: %s", buy_col, e)
        return

    # Compute directional ints
    merged["dir_true"] = merged["label"].map(_dir_label_to_int).fillna(0).astype(int)
    merged["dir_pred"] = np.sign(merged["buy_prob"] - 0.5).astype(int)

    overall_acc, tradeable_acc = _compute_accuracy(merged)
    logger.info("Overall directional accuracy (BUY/SELL only): %.3f", overall_acc)
    logger.info("Tradeable directional accuracy: %.3f", tradeable_acc)

    # Restrict to tradeable for bin analysis
    if "tradeable" in merged.columns:
        df_tr = merged[merged["tradeable"] == True].copy()
        logger.info("Tradeable rows: %d", len(df_tr))
    else:
        df_tr = merged

    # Accuracy vs predicted probability (tradeable only)
    prob_bins = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    _bin_accuracy(df_tr, "buy_prob", prob_bins)

    # Accuracy vs neutral_prob (if available)
    if "neutral_prob" in merged.columns:
        neu_bins = [0.0, 0.3, 0.5, 0.7, 1.0]
        _bin_accuracy(df_tr, "neutral_prob", neu_bins)


    # --- Extra: summarize high-confidence tradeable subset ---
    try:
        if "tradeable" in merged.columns:
            mask_tr = (merged["tradeable"] == True)
        else:
            mask_tr = np.ones(len(merged), dtype=bool)

        df_hc = merged[mask_tr & (merged["dir_true"].isin([1, -1]))].copy()
        df_hc = df_hc[df_hc["buy_prob"] >= 0.7]
        if not df_hc.empty:
            acc_hc = float((df_hc["dir_pred"] == df_hc["dir_true"]).mean())
            logger.info(
                "High-confidence subset (buy_prob>=0.7 & tradeable): n=%d, acc=%.3f",
                len(df_hc),
                acc_hc,
            )
        else:
            logger.info("High-confidence subset (buy_prob>=0.7 & tradeable): n=0")
    except Exception as e:
        logger.debug("High-confidence summary skipped: %s", e)


if __name__ == "__main__":
    main()
