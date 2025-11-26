import json
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

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
    # Use only directional rows
    mask_dir = df["label"].isin(["BUY", "SELL"])
    df = df[mask_dir].copy()
    if df.empty:
        raise SystemExit("no directional rows for Q training")

    # Target: 1 if direction correct
    y_true = (df["label"] == "BUY").astype(int).values
    y_pred = (df["p_buy"] >= 0.5).astype(int)
    y = (y_true == y_pred).astype(int)

    feat_cols = [
        "p_buy",
        "neutral_prob",
        "indicator_score",
        "fut_vol_delta",
        "cvd_divergence",
        "vwap_reversion_flag",
        "wick_extreme_up",
        "wick_extreme_down",
    ]
    cols = [c for c in feat_cols if c in df.columns]
    if not cols:
        raise SystemExit("no Q features found in dataset")

    X = df[cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).values
    return X, y, cols


def main() -> None:
    df = load_offline_predictions()
    X, y, cols = build_training_matrix(df)

    logger.info("Training Q logistic on %d samples, pos_rate=%.3f", len(y), y.mean())
    clf = LogisticRegression(max_iter=500)
    clf.fit(X, y)

    coef = clf.coef_.ravel().tolist()
    intercept = float(clf.intercept_.ravel()[0])

    payload = {"intercept": intercept, "coef": coef, "features": cols}
    Path(OUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    logger.info("Saved Q model to %s (features=%s)", OUT_PATH, cols)


if __name__ == "__main__":
    main()
