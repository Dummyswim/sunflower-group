import json
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

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

    df_dir = df_dir.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["p_buy", "cvd_divergence", "vwap_reversion_flag", "wick_extreme_up", "wick_extreme_down", "correct"]
    )

    feature_names = [
        "p_buy",
        "cvd_divergence",
        "vwap_reversion_flag",
        "wick_extreme_up",
        "wick_extreme_down",
    ]
    cols = [c for c in feature_names if c in df_dir.columns]
    if len(cols) != len(feature_names):
        missing = [c for c in feature_names if c not in cols]
        logger.warning("Missing Q features in dataset: %s", missing)
    if not cols:
        raise SystemExit("no Q features found in dataset")

    X = df_dir[cols].values
    y = df_dir["correct"].astype(int).values
    return X, y, cols


def main() -> None:
    df = load_offline_predictions()
    X, y, cols = build_training_matrix(df)

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
