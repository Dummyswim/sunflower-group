#!/usr/bin/env python3
"""
offline_eval.py

Offline evaluation for live signals emitted by main_event_loop_regen:

- signals.jsonl (pred_for, buy_prob, neutral_prob, etc.)
- labels come from:
    (A) FEATURE_LOG_HIST (if parseable), else
    (B) TRAIN_LOG_PATH (data/train_log_v2.jsonl)

This makes evaluation robust against CSV schema drift/corruption.
"""

import os
import json
import logging
from typing import Optional, Tuple, List, Dict, Any

import numpy as np
import pandas as pd

from online_trainer_regen_v2_bundle import _parse_feature_csv, load_train_log


logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
)

def _load_signals(path: str) -> Optional[pd.DataFrame]:
    if not os.path.exists(path):
        logger.error("Signals file not found: %s", path)
        return None
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                rows.append(json.loads(ln))
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


def _labels_from_trainlog(train_log_path: str, max_rows: int = 300000) -> Optional[pd.DataFrame]:
    if not os.path.exists(train_log_path):
        logger.error("TRAIN_LOG_PATH not found: %s", train_log_path)
        return None

    rows = load_train_log(train_log_path, max_rows=max_rows)
    if not rows:
        logger.error("No rows in TRAIN_LOG_PATH: %s", train_log_path)
        return None

    out = []
    for r in rows:
        # prefer ts_target_close (what you trained on), else ts
        ts = r.get("ts_target_close") or r.get("ts")
        lbl = r.get("label")
        if ts is None or lbl is None:
            continue
        aux_short = r.get("aux_label_short")
        if aux_short is None:
            aux_short = (r.get("meta") or {}).get("aux_label_short")

        out.append({
            "ts": str(ts),
            "label": str(lbl),
            "tradeable": bool(r.get("tradeable")) if "tradeable" in r else None,
            "aux_label_short": aux_short,
        })

    if not out:
        return None

    df = pd.DataFrame(out)

    # If multiple entries per timestamp, keep the last one (latest view of that bar)
    df = df.dropna(subset=["ts", "label"])
    df = df.drop_duplicates(subset=["ts"], keep="last")
    return df


def _join_signals_labels(df_sig: pd.DataFrame, df_lbl: pd.DataFrame) -> pd.DataFrame:
    df_sig = df_sig.copy()
    df_lbl = df_lbl.copy()

    df_sig["pred_for_str"] = df_sig["pred_for"].astype(str)
    df_lbl["ts_str"] = df_lbl["ts"].astype(str)

    merged = df_sig.merge(
        df_lbl,
        left_on="pred_for_str",
        right_on="ts_str",
        how="inner",
        suffixes=("_sig", "_lbl"),
    )
    return merged


def _dir_label_to_int(lbl: str) -> int:
    if lbl == "BUY":
        return 1
    if lbl == "SELL":
        return -1
    return 0


def _compute_accuracy(df: pd.DataFrame) -> Tuple[float, float]:
    mask_dir = df["dir_true"].isin([1, -1])
    if mask_dir.sum() == 0:
        return 0.0, 0.0

    overall_acc = float((df.loc[mask_dir, "dir_pred"] == df.loc[mask_dir, "dir_true"]).mean())

    if "tradeable" in df.columns:
        mask_tr = mask_dir & (df["tradeable"] == True)
        tradeable_acc = float((df.loc[mask_tr, "dir_pred"] == df.loc[mask_tr, "dir_true"]).mean()) if mask_tr.sum() else 0.0
    else:
        tradeable_acc = overall_acc

    return overall_acc, tradeable_acc


def _compute_auc(dir_true: np.ndarray, buy_prob: np.ndarray) -> Optional[float]:
    try:
        dir_true = np.asarray(dir_true, dtype=float)
        buy_prob = np.asarray(buy_prob, dtype=float)
    except Exception:
        return None

    mask = np.isfinite(dir_true) & np.isfinite(buy_prob) & (dir_true != 0.0)
    if not mask.any():
        return None

    y = (dir_true[mask] == 1.0).astype(float)
    p = buy_prob[mask]

    n_pos = float(y.sum())
    n_neg = float(len(y) - n_pos)
    if n_pos <= 0.0 or n_neg <= 0.0:
        return None

    order = np.argsort(p)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(p), dtype=float) + 1.0
    pos_ranks_sum = float(ranks[y == 1.0].sum())
    auc = (pos_ranks_sum - n_pos * (n_pos + 1.0) / 2.0) / (n_pos * n_neg)
    return float(auc)


def _bin_accuracy(df: pd.DataFrame, col: str, bins) -> None:
    logger.info("=== Accuracy vs %s bins ===", col)
    if col not in df.columns:
        logger.info("Column %s not present; skipping.", col)
        return

    x = df[col].astype(float)
    tmp = df.copy()
    tmp["bin"] = pd.cut(x, bins=bins, include_lowest=True)

    for b in tmp["bin"].cat.categories:
        sub = tmp[tmp["bin"] == b]
        if sub.empty:
            continue
        mask_dir = sub["dir_true"].isin([1, -1])
        if mask_dir.sum() == 0:
            continue
        acc = float((sub.loc[mask_dir, "dir_pred"] == sub.loc[mask_dir, "dir_true"]).mean())
        logger.info("  %s: n=%d, acc=%.3f", str(b), int(mask_dir.sum()), acc)


def _short_horizon_accuracy(df: pd.DataFrame) -> None:
    if "aux_label_short" not in df.columns or df["aux_label_short"].isna().all():
        logger.info("[EVAL] aux_label_short not present; skipping short-horizon accuracy.")
        return

    try:
        aux = (
            df["aux_label_short"]
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .astype(float)
        )
    except Exception as e:
        logger.info("[EVAL] Failed to coerce aux_label_short: %s", e)
        return

    tmp = df.copy()
    tmp["short_true"] = np.sign(aux).astype(int)
    mask_dir = tmp["short_true"].isin([1, -1])
    if mask_dir.sum() == 0:
        logger.info("[EVAL] No directional short-horizon rows; skipping.")
        return

    acc = float((tmp.loc[mask_dir, "dir_pred"] == tmp.loc[mask_dir, "short_true"]).mean())
    logger.info("[EVAL] Short-horizon directional accuracy: n=%d, acc=%.3f", int(mask_dir.sum()), acc)


def main():
    # Prefer production bundle artifacts, but allow runtime override.
    signals_path = os.getenv("SIGNALS_PATH", "trained_models/production/signals.jsonl")
    feat_hist_path = os.getenv("FEATURE_LOG_HIST", "trained_models/production/feature_log_hist.csv")
    train_log_path = os.getenv("TRAIN_LOG_PATH", "data/train_log_v2.jsonl")

    logger.info("Loading signals from %s", signals_path)
    df_sig = _load_signals(signals_path)
    if df_sig is None:
        return

    # 1) Try feature_log_hist.csv
    df_lbl = None
    if os.path.exists(feat_hist_path):
        logger.info("Loading labels from feature_log_hist: %s", feat_hist_path)
        try:
            df_feat = _parse_feature_csv(feat_hist_path, min_rows=0)
            if df_feat is not None and not df_feat.empty and "ts" in df_feat.columns and "label" in df_feat.columns:
                df_lbl = df_feat[["ts", "label"] + ([c for c in ["tradeable", "aux_label_short"] if c in df_feat.columns])].copy()
                df_lbl["ts"] = df_lbl["ts"].astype(str)
                df_lbl = df_lbl.drop_duplicates(subset=["ts"], keep="last")
            else:
                df_lbl = None
        except Exception as e:
            logger.warning("feature_log_hist parse failed (%s). Falling back to TRAIN_LOG_PATH.", e)
            df_lbl = None

    # 2) Fallback to train_log_v2.jsonl
    if df_lbl is None or df_lbl.empty:
        logger.info("Loading labels from TRAIN_LOG_PATH: %s", train_log_path)
        df_lbl = _labels_from_trainlog(train_log_path)
        if df_lbl is None or df_lbl.empty:
            logger.error("No labels available from feature_log_hist or TRAIN_LOG_PATH.")
            return

    merged = _join_signals_labels(df_sig, df_lbl)
    if merged.empty:
        logger.error("Merged DataFrame is empty (no matching timestamps).")
        return

    logger.info("Merged rows: %d", len(merged))

    # Normalize buy_prob to float
    merged["buy_prob"] = pd.to_numeric(merged["buy_prob"], errors="coerce")
    merged = merged.replace([np.inf, -np.inf], np.nan).dropna(subset=["buy_prob", "label"])

    merged["dir_true"] = merged["label"].map(_dir_label_to_int).fillna(0).astype(int)
    merged["dir_pred"] = np.sign(merged["buy_prob"] - 0.5).astype(int)

    overall_acc, tradeable_acc = _compute_accuracy(merged)
    logger.info("Overall directional accuracy (BUY/SELL only): %.3f", overall_acc)
    logger.info("Tradeable directional accuracy: %.3f", tradeable_acc)

    auc = _compute_auc(merged["dir_true"].values, merged["buy_prob"].values)
    if auc is not None:
        mask_dir = merged["dir_true"].isin([1, -1])
        logger.info("Directional AUC: n=%d (BUY/SELL rows), AUC=%.3f", int(mask_dir.sum()), float(auc))

    _bin_accuracy(merged, "buy_prob", bins=[0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    if "neutral_prob" in merged.columns:
        _bin_accuracy(merged, "neutral_prob", bins=[-0.001, 0.3, 0.5, 0.7, 1.0])

    _short_horizon_accuracy(merged)

    # High confidence subset
    if "tradeable" in merged.columns:
        hc = merged[(merged["buy_prob"] >= 0.7) & (merged["tradeable"] == True)]
    else:
        hc = merged[merged["buy_prob"] >= 0.7]

    mask_dir = hc["dir_true"].isin([1, -1])
    if mask_dir.sum() > 0:
        acc = float((hc.loc[mask_dir, "dir_pred"] == hc.loc[mask_dir, "dir_true"]).mean())
        logger.info("High-confidence subset (buy_prob>=0.7 & tradeable): n=%d, acc=%.3f", int(mask_dir.sum()), acc)
    else:
        logger.info("High-confidence subset: n=0")


if __name__ == "__main__":
    main()
