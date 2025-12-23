#!/usr/bin/env python3
"""
Offline evaluation for policy success signals vs labels.

- signals.jsonl: uses teacher_dir + policy_success(_calib)
- labels from TRAIN_LOG_PATH (SignalContext JSONL)
"""
import json
import logging
import os
from typing import Optional, Tuple, List, Dict, Any

import numpy as np
import pandas as pd

from signal_log_utils import load_signal_log

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
    if "pred_for" not in df.columns or "teacher_dir" not in df.columns:
        logger.error("Signals file missing required columns (pred_for, teacher_dir)")
        return None
    return df


def _labels_from_trainlog(train_log_path: str, max_rows: int = 300000) -> Optional[pd.DataFrame]:
    if not os.path.exists(train_log_path):
        logger.error("TRAIN_LOG_PATH not found: %s", train_log_path)
        return None

    rows = load_signal_log(train_log_path, max_rows=max_rows, schema_cols=None)
    if not rows:
        logger.error("No rows in TRAIN_LOG_PATH: %s", train_log_path)
        return None

    out = []
    for r in rows:
        ts = r.get("ts_target_close")
        lbl = r.get("label")
        if ts is None or lbl is None:
            continue
        out.append({
            "ts": str(ts),
            "label": str(lbl),
        })

    if not out:
        return None

    df = pd.DataFrame(out)
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


def _compute_auc(y_true: np.ndarray, scores: np.ndarray) -> Optional[float]:
    try:
        y_true = np.asarray(y_true, dtype=float)
        scores = np.asarray(scores, dtype=float)
    except Exception:
        return None

    mask = np.isfinite(y_true) & np.isfinite(scores)
    if not mask.any():
        return None

    y = y_true[mask]
    p = scores[mask]

    n_pos = float(np.sum(y == 1.0))
    n_neg = float(len(y) - n_pos)
    if n_pos <= 0.0 or n_neg <= 0.0:
        return None

    order = np.argsort(p)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(p), dtype=float) + 1.0
    pos_ranks_sum = float(ranks[y == 1.0].sum())
    auc = (pos_ranks_sum - n_pos * (n_pos + 1.0) / 2.0) / (n_pos * n_neg)
    return float(auc)


def _brier(y_true: np.ndarray, scores: np.ndarray) -> float:
    scores = np.clip(scores.astype(float), 1e-9, 1 - 1e-9)
    return float(np.mean((scores - y_true) ** 2))


def _score_column(df: pd.DataFrame) -> str:
    if "policy_success_calib" in df.columns:
        return "policy_success_calib"
    if "policy_success_raw" in df.columns:
        return "policy_success_raw"
    return ""


def main() -> None:
    sig_path = os.getenv("SIGNALS_PATH", "trained_models/production/signals.jsonl")
    train_log_path = os.getenv("TRAIN_LOG_PATH", "data/train_log_v3_canonical.jsonl")

    df_sig = _load_signals(sig_path)
    if df_sig is None:
        return
    df_lbl = _labels_from_trainlog(train_log_path)
    if df_lbl is None:
        return

    merged = _join_signals_labels(df_sig, df_lbl)
    if merged.empty:
        logger.info("No overlapping timestamps between signals and labels.")
        return

    score_col = _score_column(merged)
    if not score_col:
        logger.info("No policy_success_* fields in signals; nothing to score.")
        return

    for d in ("BUY", "SELL"):
        sub = merged[merged["teacher_dir"].astype(str).str.upper() == d]
        if sub.empty:
            logger.info("[%s] no rows", d)
            continue
        y = (sub["label"].astype(str).str.upper() == d).astype(int).to_numpy()
        p = sub[score_col].astype(float).to_numpy()
        auc = _compute_auc(y, p)
        brier = _brier(y, p)
        logger.info("[%s] n=%d auc=%s brier=%.4f", d, int(len(sub)), f"{auc:.3f}" if auc is not None else "NA", brier)


if __name__ == "__main__":
    main()
