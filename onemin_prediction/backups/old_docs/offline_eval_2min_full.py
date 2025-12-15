#!/usr/bin/env python
"""
offline_eval_2min_full.py

Offline evaluation for the trade-window TP/SL outcome model.

- Uses offline_train_2min.build_2min_dataset(), which now:
    * labels each bar by which side (long/short) is more likely to hit TP before SL
      within TRADE_HORIZON_MIN minutes.
    * keeps FLAT for ambiguous / no-edge bars.

All metrics here are computed over BUY/SELL labels (direction of the better trade side),
optionally with FLAT handling where applicable.
"""

import os
import sys
import json
import logging
from typing import Tuple, Optional

import numpy as np
import pandas as pd

# Optional: if you want to later compare against a logistic baseline,
# you can also import joblib and load logit_2min.pkl here.
# import joblib


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def _setup_logging() -> logging.Logger:
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    ch = logging.StreamHandler(sys.stdout)

    high_verbosity = os.getenv("LOG_HIGH_VERBOSITY", "1") not in ("0", "false", "False")
    ch.setLevel(logging.DEBUG if high_verbosity else logging.INFO)

    fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    ch.setFormatter(logging.Formatter(fmt))
    logger.handlers.clear()
    logger.addHandler(ch)

    return logger


logger = _setup_logging()


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _get_env_float(name: str, default: float) -> float:
    """Parse a float env var with safe fallback and logging."""
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        val = float(raw)
        return val
    except ValueError:
        logger.warning("Env %s=%r is not a valid float; falling back to %f", name, raw, default)
        return default


def _get_default_paths(project_root: Optional[str] = None) -> Tuple[str, str]:
    """
    Returns:
        offline_preds_path, q_model_path
    """
    root = project_root or os.getenv("PROJECT_ROOT") or os.getcwd()
    experiments_dir = os.path.join(root, "trained_models", "experiments")

    offline_preds_path = os.getenv(
        "OFFLINE_2MIN_PREDS_PATH",
        os.path.join(experiments_dir, "offline_2min_with_preds.csv"),
    )

    q_model_path = os.getenv(
        "Q_MODEL_2MIN_PATH",
        os.path.join(experiments_dir, "q_model_2min.json"),
    )

    return offline_preds_path, q_model_path


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def _load_offline_preds(csv_path: str) -> pd.DataFrame:
    """
    Load offline_2min_with_preds.csv and basic sanity checks.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"offline preds file not found: {csv_path}")

    logger.info("Loading offline predictions from %s", csv_path)
    df = pd.read_csv(csv_path)

    # Try to parse timestamp if present
    for ts_col in ("mid_ts", "timestamp", "ts"):
        if ts_col in df.columns:
            df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
            break

    # Normalize label column name if needed
    if "label_2min" not in df.columns and "label" in df.columns:
        df = df.rename(columns={"label": "label_2min"})

    # Basic required columns
    required_cols = ["label_2min", "p_buy"]
    for c in required_cols:
        if c not in df.columns:
            raise KeyError(f"Required column {c!r} not found in {csv_path}")

    # Drop rows without p_buy or label
    before = len(df)
    df = df.dropna(subset=["label_2min", "p_buy"])
    after = len(df)
    if after < before:
        logger.warning("Dropped %d rows with NaN in label_2min or p_buy", before - after)

    return df


def _build_directional_frame(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only BUY/SELL rows and attach directional label (1=BUY, 0=SELL).
    """
    mask = df["label_2min"].isin(["BUY", "SELL"])
    df_dir = df.loc[mask].copy()
    df_dir["y_true"] = (df_dir["label_2min"] == "BUY").astype(int)
    logger.info("Directional dataset: n=%d (BUY/SELL only)", df_dir.shape[0])
    return df_dir


def _compute_bin_stats(p_buy: np.ndarray, y_true: np.ndarray) -> None:
    """
    Print accuracy vs p-buy bins.
    """
    assert p_buy.shape == y_true.shape

    logger.info("=== Accuracy vs buy_prob bins (raw model) ===")
    # Bins: (0.499,0.6], (0.6,0.7], (0.7,0.8], (0.8,0.9], (>0.9 if any)
    edges = [0.499, 0.6, 0.7, 0.8, 0.9, 1.0001]

    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        mask = (p_buy > lo) & (p_buy <= hi)
        n = int(mask.sum())
        if n == 0:
            continue
        acc = float(((p_buy[mask] > 0.5).astype(int) == y_true[mask]).mean())
        logger.info("  (%.3f, %.1f]: n=%d, acc=%.3f", lo, hi, n, acc)


def _load_q_model(path: str) -> Optional[dict]:
    """
    Load Q logistic meta-model JSON (output of offline_train_q_model_2min.py).
    """
    if not os.path.exists(path):
        logger.warning("Q model file not found at %s; skipping Q gating eval.", path)
        return None

    with open(path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    required_keys = ["features", "coef", "intercept", "scaler_mean", "scaler_scale"]
    for k in required_keys:
        if k not in meta:
            raise KeyError(f"Q model JSON missing key {k!r}: {path}")

    return meta


def _compute_q_hat(df_dir: pd.DataFrame, q_meta: dict) -> np.ndarray:
    """
    Compute q_hat for each directional row using the Q meta-model.

    The Q model expects a fixed feature list. We:
      - Construct those columns from df_dir (p_buy from df, others as-is or 0 if missing)
      - Apply the saved StandardScaler parameters
      - Apply logistic regression (coef / intercept) to get q_hat in [0,1]
    """
    feat_names = q_meta["features"]
    scaler_mean = np.asarray(q_meta.get("scaler_mean", [0.0] * len(feat_names)), dtype=float)
    scaler_scale = np.asarray(q_meta.get("scaler_scale", [1.0] * len(feat_names)), dtype=float)
    coef = np.asarray(q_meta["coef"], dtype=float)
    intercept = float(q_meta["intercept"])

    # Guard against zero scale to avoid division-by-zero
    scaler_scale_safe = np.where(scaler_scale == 0.0, 1.0, scaler_scale)

    # Build feature matrix
    data = {}
    for name in feat_names:
        if name == "p_buy":
            data[name] = df_dir["p_buy"].values
        elif name in df_dir.columns:
            data[name] = df_dir[name].fillna(0.0).values
        else:
            # Missing feature: safe default 0.0
            logger.debug("Q feature %r missing in offline preds; filling with 0.0", name)
            data[name] = np.zeros(len(df_dir), dtype=float)

    X = np.column_stack([data[n] for n in feat_names])
    X_std = (X - scaler_mean) / scaler_scale_safe

    z = intercept + np.dot(X_std, coef)
    q_hat = 1.0 / (1.0 + np.exp(-z))

    return q_hat


def _eval_gating(
    df_dir: pd.DataFrame,
    q_hat: np.ndarray,
    p_edge_min: float,
    q_gate_min: float,
) -> None:
    """
    Evaluate performance under a simple gating rule:
        - p_edge = |p_buy - 0.5|
        - gate if p_edge >= p_edge_min AND q_hat >= q_gate_min
    """
    p_buy = df_dir["p_buy"].values
    y_true = df_dir["y_true"].values

    p_edge = np.abs(p_buy - 0.5)
    gate_mask = (p_edge >= p_edge_min) & (q_hat >= q_gate_min)

    n_total = len(df_dir)
    n_gate = int(gate_mask.sum())

    logger.info("=== Q + margin gating summary ===")
    logger.info(
        "Gate condition: |p_buy-0.5| >= %.3f AND q_hat >= %.3f",
        p_edge_min,
        q_gate_min,
    )
    logger.info("Directional rows (total): %d", n_total)
    logger.info("Directional rows (gated in): %d", n_gate)

    if n_gate == 0:
        logger.info("No trades pass the gate; nothing to evaluate.")
        return

    y_pred = (p_buy > 0.5).astype(int)
    acc_all = float((y_pred == y_true).mean())
    acc_gate = float((y_pred[gate_mask] == y_true[gate_mask]).mean())

    logger.info("Overall accuracy (all directional): %.3f", acc_all)
    logger.info("Gated accuracy (subset):          %.3f", acc_gate)

    # Optional: trades per day if we have a timestamp column
    ts_col = None
    for c in ("mid_ts", "timestamp", "ts"):
        if c in df_dir.columns:
            ts_col = c
            break

    if ts_col is not None:
        # Drop NaT to avoid issues
        ts_valid = df_dir[ts_col].dropna()
        n_days = ts_valid.dt.normalize().nunique()
        if n_days > 0:
            trades_per_day = n_gate / float(n_days)
            logger.info("Distinct days in eval set: %d", n_days)
            logger.info("Approx trades/day (gated): %.2f", trades_per_day)
    else:
        logger.info("No timestamp column found; skipping trades/day computation.")

    # Some extra diagnostics
    logger.info("Mean p_edge (gated): %.3f", float(p_edge[gate_mask].mean()))
    logger.info("Mean q_hat (gated):  %.3f", float(q_hat[gate_mask].mean()))


def _eval_gating_grid(
    df_dir: pd.DataFrame,
    q_hat: np.ndarray,
) -> None:
    """
    Explore a small grid of (margin, q_hat) thresholds to see
    how many trades pass the gate and what accuracy they achieve.

    This is purely for offline analysis / debug.
    """
    if df_dir.empty:
        logger.info("Skipping gating grid search: no directional rows.")
        return

    p_buy = df_dir["p_buy"].values
    y_true = df_dir["y_true"].values
    p_edge = np.abs(p_buy - 0.5)

    # Global directional baseline for reference
    y_pred = (p_buy > 0.5).astype(int)
    acc_all = float((y_pred == y_true).mean())

    logger.info("=== Q + margin gating grid search ===")
    logger.info("Baseline accuracy (all directional): %.3f", acc_all)

    # You can tune these sets if needed
    margin_thresholds = [0.08, 0.10, 0.12, 0.15, 0.18, 0.20]
    q_thresholds = [0.50, 0.55, 0.60]

    for m_thr in margin_thresholds:
        for q_thr in q_thresholds:
            gate_mask = (p_edge >= m_thr) & (q_hat >= q_thr)
            n_gate = int(gate_mask.sum())
            if n_gate == 0:
                logger.info(
                    "  margin>=%.3f & q_hat>=%.2f → n=0 (no trades)",
                    m_thr,
                    q_thr,
                )
                continue

            acc_gate = float(
                (y_pred[gate_mask] == y_true[gate_mask]).mean()
            )
            logger.info(
                "  margin>=%.3f & q_hat>=%.2f → n=%d, acc=%.3f",
                m_thr,
                q_thr,
                n_gate,
                acc_gate,
            )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    logger.info("offline_eval_2min_full started (LOG_HIGH_VERBOSITY=%s)",
                os.getenv("LOG_HIGH_VERBOSITY", "1"))

    project_root = os.getenv("PROJECT_ROOT")
    offline_preds_path, q_model_path = _get_default_paths(project_root)

    logger.info("Using offline preds: %s", offline_preds_path)
    logger.info("Using Q model path:  %s", q_model_path)

    # Gating thresholds (you can tune these via env to get ~8 trades/day)
    p_edge_min = _get_env_float("GATE_P_EDGE_MIN", 0.15)  # |p_buy-0.5| threshold
    q_gate_min = _get_env_float("GATE_Q_MIN", 0.55)       # q_hat minimum

    logger.info("GATE_P_EDGE_MIN=%.3f", p_edge_min)
    logger.info("GATE_Q_MIN=%.3f", q_gate_min)

    try:
        df = _load_offline_preds(offline_preds_path)
    except Exception as e:
        logger.exception("Failed to load offline predictions: %s", e)
        sys.exit(1)

    df_dir = _build_directional_frame(df)

    if df_dir.empty:
        logger.error("No directional (BUY/SELL) rows found; nothing to evaluate.")
        sys.exit(1)

    # --- Raw model stats (XGB p_buy from offline_2min_with_preds.csv) ---
    p_buy = df_dir["p_buy"].values
    y_true = df_dir["y_true"].values

    y_pred = (p_buy > 0.5).astype(int)
    overall_acc = float((y_pred == y_true).mean())
    logger.info("Overall directional accuracy (raw XGB): %.3f", overall_acc)

    _compute_bin_stats(p_buy, y_true)

    # --- Q gating evaluation ---
    q_meta = _load_q_model(q_model_path)
    if q_meta is None:
        # Already logged; exit with success since raw stats are still valid
        logger.info("Skipping Q gating stats (no Q model available).")
        return

    try:
        q_hat = _compute_q_hat(df_dir, q_meta)
    except Exception as e:
        logger.exception("Failed to compute q_hat: %s", e)
        sys.exit(1)

    _eval_gating(df_dir, q_hat, p_edge_min=p_edge_min, q_gate_min=q_gate_min)

    # Additional grid search for analysis / tuning
    _eval_gating_grid(df_dir, q_hat)


if __name__ == "__main__":
    main()
