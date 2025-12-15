"""
offline_train_q_model_2min.py

Offline evaluation for the trade-window TP/SL outcome model.

- Uses offline_train_2min.build_2min_dataset(), which now:
    * labels each bar by which side (long/short) is more likely to hit TP before SL
      within TRADE_HORIZON_MIN minutes.
    * keeps FLAT for ambiguous / no-edge bars.

All metrics here are computed over BUY/SELL labels (direction of the better trade side),
optionally with FLAT handling where applicable.
"""

import json
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from feature_pipeline import FeaturePipeline

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

EXP_DIR = os.getenv("EXPERIMENT_DIR", "trained_models/experiments")
DATA_PATH = os.path.join(EXP_DIR, "offline_2min_with_preds.csv")
OUT_PATH = os.path.join(EXP_DIR, "q_model_2min.json")


def load_offline_predictions() -> pd.DataFrame:
    """
    Expect a CSV with:
        ts, label, p_buy, neutral_prob, indicator_score, fut_vol_delta,
        cvd_divergence, vwap_reversion_flag, wick_extreme_up, wick_extreme_down, ...
    You can generate this from offline_eval_2min after prediction.
    """
    if not os.path.exists(DATA_PATH):
        raise SystemExit(f"missing offline prediction dataset: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["p_buy", "label"])
    return df


def _safe_float(x, default: float = 0.0, allow_nan: bool = False) -> float:
    try:
        v = float(x)
        if np.isfinite(v):
            return v
    except Exception:
        pass
    return float("nan") if allow_nan else default


def build_q_features_from_row(row: pd.Series) -> dict:
    """
    Reconstruct Q-model features to match main_event_loop behaviour.

    Prioritises explicit columns if present; otherwise falls back to the
    same helper functions used online. Keeps missing/unknown values as 0.0.
    """
    # Wick extremes: prefer provided columns
    wick_up = _safe_float(row.get("wick_extreme_up", np.nan), default=0.0)
    wick_dn = _safe_float(row.get("wick_extreme_down", np.nan), default=0.0)

    # VWAP reversion flag: use column if present, else try to compute
    if "vwap_reversion_flag" in row:
        vwap_rev = _safe_float(row.get("vwap_reversion_flag", np.nan), default=np.nan, allow_nan=True)
    else:
        try:
            vwap_val = _safe_float(row.get("fut_session_vwap", np.nan), default=np.nan)
            px_hist_df = None
            vwap_rev = _safe_float(FeaturePipeline._compute_vwap_reversion_flag(px_hist_df, vwap_val), default=np.nan, allow_nan=True)
        except Exception:
            vwap_rev = np.nan

    # CVD divergence: use column if present; otherwise compute from fut_cvd_delta + price change if available
    if "cvd_divergence" in row:
        cvd_div = _safe_float(row.get("cvd_divergence", np.nan), default=np.nan, allow_nan=True)
    else:
        try:
            fut_cvd_delta = row.get("fut_cvd_delta", None)
            last_close = _safe_float(row.get("last_price", np.nan), default=0.0)
            prev_close = _safe_float(row.get("prev_price", np.nan), default=last_close)
            px_change = last_close - prev_close
            cvd_div = _safe_float(FeaturePipeline._compute_cvd_divergence(px_change, fut_cvd_delta), default=np.nan, allow_nan=True)
        except Exception:
            cvd_div = np.nan

    return {
        "p_buy": _safe_float(row.get("p_buy", 0.5), default=0.5),
        "cvd_divergence": cvd_div,
        "vwap_reversion_flag": vwap_rev,
        "wick_extreme_up": wick_up,
        "wick_extreme_down": wick_dn,
    }


def build_training_matrix(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
    df = df.copy()
    if "is_directional" not in df.columns:
        df["is_directional"] = df["label"].isin(["BUY", "SELL"]).astype(int)

    # Use only directional rows
    df_dir = df[df["is_directional"] == 1].copy()
    if df_dir.empty:
        raise SystemExit("no directional rows for Q training")

    # Ensure clean inputs and target
    if "correct" not in df_dir.columns:
        df_dir["correct"] = (
            ((df_dir["label"] == "BUY") & (df_dir["p_buy"] >= 0.5))
            | ((df_dir["label"] == "SELL") & (df_dir["p_buy"] < 0.5))
        ).astype(int)

    feature_names = ["p_buy", "cvd_divergence", "vwap_reversion_flag", "wick_extreme_up", "wick_extreme_down"]
    required = ["p_buy", "wick_extreme_up", "wick_extreme_down", "correct"]
    optional = ["cvd_divergence", "vwap_reversion_flag"]

    df_dir = df_dir.replace([np.inf, -np.inf], np.nan)
    df_dir = df_dir.dropna(subset=required)
    if df_dir.empty:
        logger.warning("No rows with required Q features (p_buy + wicks); skipping Q training dataset.")
        return np.empty((0, len(feature_names))), np.empty((0,), dtype=int), feature_names

    # Optional futures fields: keep if present, else neutral-fill to 0.0
    present_optional = [c for c in optional if c in df_dir.columns]
    if present_optional:
        df_dir[present_optional] = df_dir[present_optional].fillna(0.0)

    # Build Q-feature matrix row by row to mirror live computation
    q_rows = []
    for _, r in df_dir.iterrows():
        q_rows.append(build_q_features_from_row(r))

    q_df = pd.DataFrame(q_rows, columns=feature_names).replace([np.inf, -np.inf], np.nan)

    X = q_df[feature_names].values
    y = df_dir["correct"].astype(int).values
    return X, y, feature_names


def main() -> None:
    df = load_offline_predictions()
    X, y, cols = build_training_matrix(df)
    if X.size == 0 or y.size == 0:
        logger.warning("Q training aborted: no usable rows after filtering required features.")
        return

    logger.info("Training Q logistic on %d samples, pos_rate=%.3f", len(y), y.mean())
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clf = LogisticRegression(
        solver="lbfgs",
        max_iter=2000,
    )
    clf.fit(X_scaled, y)

    coef = clf.coef_.ravel().tolist()
    intercept = float(clf.intercept_.ravel()[0])

    payload = {
        "intercept": intercept,
        "coef": coef,
        "features": cols,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
    }
    Path(OUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    logger.info("Saved Q model to %s (features=%s)", OUT_PATH, cols)


if __name__ == "__main__":
    main()
