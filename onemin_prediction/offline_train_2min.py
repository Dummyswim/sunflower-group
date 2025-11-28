#!/usr/bin/env python3
"""
offline_train_2min.py

Offline trainer for a 2-minute direction model using Dhan historical intraday data.

- Fetches 1-minute OHLCV from Dhan /v2/charts/intraday for a date range.
- Builds a 2-minute direction label:
    label_t = BUY  if close_{t+2} > close_t + tol
             = SELL if close_{t+2} < close_t - tol
             = FLAT otherwise
- Computes features using your existing FeaturePipeline + TA logic.
- Trains:
    - XGBoost directional model (BUY vs SELL)
    - Neutrality LogisticRegression (FLAT vs non-FLAT)
- Saves models to XGB_PATH and NEUTRAL_PATH (env vars).

Usage:
    export DHAN_ACCESS_TOKEN="..."
    export DHAN_CLIENT_ID="..."
    export TRAIN_START_DATE="2024-11-01"
    export TRAIN_END_DATE="2024-11-05"
    export XGB_PATH="trained_models/production/xgb_2min.json"
    export NEUTRAL_PATH="trained_models/production/neutral_2min.pkl"
    python offline_train_2min.py
"""

import os
import json
import logging
from datetime import datetime, timedelta, date
from typing import Optional, List

import numpy as np
import pandas as pd
import requests

from feature_pipeline import FeaturePipeline, TA
from online_trainer import _train_xgb, _train_neutrality

"""
Purpose:
- Build 2-minute labeled dataset (BUY / SELL / FLAT) from 1-minute candles.
- Train XGB directional model (+ neutrality model).
- This script uses *only past information* per reference bar; no future leakage.
"""

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
)

# ---------- DHAN API CONFIG ----------

DHAN_ACCESS_TOKEN = os.getenv("DHAN_ACCESS_TOKEN", "").strip()
DHAN_CLIENT_ID = os.getenv("DHAN_CLIENT_ID", "").strip()
DHAN_HIST_BASE_URL = os.getenv("DHAN_HIST_BASE_URL", "https://api.dhan.co").rstrip("/")
DHAN_HIST_PATH = os.getenv("DHAN_HIST_PATH", "/v2/charts/intraday")

# Instrument details: reuse from run_main.py defaults
NIFTY_SECURITY_ID = int(os.getenv("NIFTY_SECURITY_ID", "13"))
NIFTY_EXCHANGE_SEGMENT = os.getenv("NIFTY_EXCHANGE_SEGMENT", "IDX_I")
INTRADAY_RESOLUTION = os.getenv("INTRADAY_RESOLUTION", "1")  # 1-minute
NIFTY_INSTRUMENT = os.getenv("NIFTY_INSTRUMENT", "INDEX")

def _check_dhan_config() -> bool:
    ok = True
    if not DHAN_ACCESS_TOKEN:
        logger.error("DHAN_ACCESS_TOKEN is not set.")
        ok = False
    if not DHAN_CLIENT_ID:
        logger.error("DHAN_CLIENT_ID is not set.")
        ok = False
    return ok


def fetch_intraday_for_day(dt: date) -> Optional[pd.DataFrame]:
    """
    Fetch 1-minute intraday OHLCV for a single day from Dhan.
    This implementation assumes a POST /v2/charts/intraday with JSON body.
    Adjust fields per https://dhanhq.co/docs/v2/historical-data/ if needed.
    """
    if not _check_dhan_config():
        return None

    url = DHAN_HIST_BASE_URL + DHAN_HIST_PATH
    
    # print(start_dt.strftime("%Y-%m-%dT%H:%M:%S"), end_dt.strftime("%Y-%m-%dT%H:%M:%S"))

    # Construct from/to for the given date in IST. Adjust format if API expects epoch / different TZ.
    start_dt = datetime(dt.year, dt.month, dt.day, 9, 15)
    end_dt = datetime(dt.year, dt.month, dt.day, 15, 30)

    headers = {
        "accept": "application/json",
        # Adjust header names if docs specify differently
        "access-token": DHAN_ACCESS_TOKEN,
        "client-id": DHAN_CLIENT_ID,
        "Content-Type": "application/json",
    }

    
    
    payload = {
        "securityId": str(NIFTY_SECURITY_ID),
        "exchangeSegment": NIFTY_EXCHANGE_SEGMENT,
        "fromDate": start_dt.strftime("%Y-%m-%d %H:%M:%S"),  # <- changed
        "toDate": end_dt.strftime("%Y-%m-%d %H:%M:%S"),      # <- changed
        "interval": INTRADAY_RESOLUTION,
        "instrument": NIFTY_INSTRUMENT,
    }

    print(f"{payload}")
    try:
        logger.info("Fetching intraday for %s from %s", dt.isoformat(), url)
        


        resp = requests.post(url, headers=headers, json=payload, timeout=20)
        if resp.status_code != 200:
            try:
                err = resp.json()
            except Exception:
                err = {}

            if err.get("errorCode") == "DH-905":
                logger.warning(
                    "No intraday data for %s (DH-905: %s)",
                    dt.isoformat(),
                    err.get("errorMessage", "no data"),
                )
                return None

            logger.error("Dhan intraday API returned %s: %s", resp.status_code, resp.text[:200])
            return None

        
        
        data = resp.json()
    except Exception as e:
        logger.error("Dhan intraday request failed: %s", e)
        return None



    # Parse response according to Dhan intraday format:
    # {
    #   "open": [...],
    #   "high": [...],
    #   "low": [...],
    #   "close": [...],
    #   "volume": [...],
    #   "timestamp": [...]
    # }

    rows = []

    # Case 1: old "data"/"candles" list format (keep backward compat)
    records = data.get("data") or data.get("candles")
    if isinstance(records, list) and records:
        for rec in records:
            try:
                if isinstance(rec, dict):
                    ts = pd.to_datetime(rec.get("timestamp"))
                    o = float(rec.get("open"))
                    h = float(rec.get("high"))
                    l = float(rec.get("low"))
                    c = float(rec.get("close"))
                    v = float(rec.get("volume", 0.0))
                else:
                    ts = pd.to_datetime(rec[0])
                    o = float(rec[1])
                    h = float(rec[2])
                    l = float(rec[3])
                    c = float(rec[4])
                    v = float(rec[5]) if len(rec) > 5 else 0.0

                if not np.isfinite(o + h + l + c):
                    continue
                rows.append(
                    {"timestamp": ts, "open": o, "high": h, "low": l, "close": c, "volume": v}
                )
            except Exception:
                continue

    # Case 2: Dhan OHLC arrays at root (your curl response)
    elif all(k in data for k in ("open", "high", "low", "close", "timestamp")):
        opens = data["open"]
        highs = data["high"]
        lows = data["low"]
        closes = data["close"]
        volumes = data.get("volume") or [0.0] * len(opens)
        timestamps = data["timestamp"]

        for ts, o, h, l, c, v in zip(timestamps, opens, highs, lows, closes, volumes):
            try:
                ts_parsed = pd.to_datetime(ts, unit="s") if isinstance(ts, (int, float)) else pd.to_datetime(ts)
                o = float(o); h = float(h); l = float(l); c = float(c)
                v = float(v) if v is not None else 0.0
                if not np.isfinite(o + h + l + c):
                    continue
                rows.append(
                    {"timestamp": ts_parsed, "open": o, "high": h, "low": l, "close": c, "volume": v}
                )
            except Exception:
                continue


    if not rows:
        logger.warning("Parsed 0 valid candles for %s", dt.isoformat())
        return None

    df = pd.DataFrame(rows).set_index("timestamp").sort_index()
    logger.info("Fetched %d candles for %s", len(df), dt.isoformat())
    return df


def fetch_intraday_range(start: date, end: date) -> pd.DataFrame:
    """
    Fetch intraday 1-minute OHLCV for all days in [start, end], inclusive.
    """
    all_dfs: List[pd.DataFrame] = []
    cur = start
    while cur <= end:
        df_day = fetch_intraday_for_day(cur)
        if df_day is not None and not df_day.empty:
            all_dfs.append(df_day)
        cur += timedelta(days=1)

    if not all_dfs:
        logger.error("No data fetched for date range %s to %s", start.isoformat(), end.isoformat())
        return pd.DataFrame()

    df_all = pd.concat(all_dfs, axis=0)
    df_all = df_all[~df_all.index.duplicated(keep="first")].sort_index()
    logger.info("Total candles in range: %d", len(df_all))
    return df_all


def fetch_intraday_range_2min(start: date, end: date) -> pd.DataFrame:
    """
    Alias for fetch_intraday_range; retained for compatibility with leakage tests.
    """
    return fetch_intraday_range(start, end)


def build_2min_dataset(df_candles: pd.DataFrame) -> pd.DataFrame:
    """
    Build features + 2-minute direction labels from 1-minute candles.

    Label definition:
        For each t:
          ref = close_t
          target = close_{t+2}
          move = target - ref
          tol = flat_tolerance_pct * ref

          BUY  if move > +tol
          SELL if move < -tol
          FLAT otherwise
    """
    if df_candles is None or df_candles.empty:
        return pd.DataFrame()

    df = df_candles.copy().sort_index()
    timestamps = df.index.to_list()
    n = len(df)
    logger.info("Building 2-minute dataset from %d candles", n)

    # --- 2-minute move distribution (for FLAT tolerance tuning) ---
    try:
        closes = df["close"].astype(float).values
        if closes.size >= 4:
            # 2-minute horizon: diff between t+2 and t
            moves = closes[2:] - closes[:-2]
            moves = moves[np.isfinite(moves)]
            abs_moves = np.abs(moves)
            if abs_moves.size >= 10:
                sigma_pts = float(abs_moves.std())
                med_pts = float(np.median(abs_moves))
                p95_pts = float(np.percentile(abs_moves, 95))
                logger.info(
                    "2-min abs move stats (pts): sigma=%.4f, median=%.4f, p95=%.4f",
                    sigma_pts,
                    med_pts,
                    p95_pts,
                )

                # Suggested flat tolerance in points:
                # max(cost, sigma_mult * sigma)
                cost_pts = float(os.getenv("OFFLINE_COST_POINTS", "0.0") or "0.0")
                sigma_mult = float(os.getenv("OFFLINE_FLAT_SIGMA_MULT", "0.35") or "0.35")
                min_pts = float(os.getenv("OFFLINE_FLAT_MIN_PTS", "0.0") or "0.0")

                suggested_tol_pts = max(cost_pts, min_pts, sigma_mult * sigma_pts)
                logger.info(
                    "Suggested OFFLINE flat tolerance (pts): %.4f "
                    "(cost=%.4f, sigma_mult=%.3f, min_pts=%.4f)",
                    suggested_tol_pts,
                    cost_pts,
                    sigma_mult,
                    min_pts,
                )
            else:
                logger.info("Not enough 2-min moves for volatility stats (n=%d)", abs_moves.size)
        else:
            logger.info("Not enough candles for 2-min volatility stats (n=%d)", closes.size)
    except Exception as e:
        logger.warning("2-min move distribution analysis failed: %s", e)

    feat_pipe = FeaturePipeline(train_features={})
    rows = []

    # FLAT tolerance configuration
    flat_tolerance_pct = float(os.getenv("OFFLINE_FLAT_TOL_PCT", "0.00010"))  # 0.01% default

    # Optional: use a points-based tolerance derived from move stats.
    use_points = os.getenv("OFFLINE_FLAT_USE_POINTS", "0").strip() == "1"
    flat_tolerance_pts = None
    if use_points:
        try:
            # Allow explicit override, otherwise reuse suggested formula.
            cost_pts = float(os.getenv("OFFLINE_COST_POINTS", "0.0") or "0.0")
            sigma_mult = float(os.getenv("OFFLINE_FLAT_SIGMA_MULT", "0.35") or "0.35")
            min_pts = float(os.getenv("OFFLINE_FLAT_MIN_PTS", "0.0") or "0.0")

            closes = df["close"].astype(float).values
            if closes.size >= 4:
                moves = closes[2:] - closes[:-2]
                moves = moves[np.isfinite(moves)]
                abs_moves = np.abs(moves)
                if abs_moves.size >= 10:
                    sigma_pts = float(abs_moves.std())
                    flat_tolerance_pts = max(cost_pts, min_pts, sigma_mult * sigma_pts)
                    logger.info(
                        "Using points-based FLAT tolerance: %.4f pts "
                        "(cost=%.4f, sigma_mult=%.3f, min_pts=%.4f)",
                        flat_tolerance_pts,
                        cost_pts,
                        sigma_mult,
                        min_pts,
                    )
                    if not np.isfinite(flat_tolerance_pts) or flat_tolerance_pts <= 0:
                        raise ValueError(
                            f"Computed flat_tolerance_pts={flat_tolerance_pts} is invalid; "
                            "check sigma / cost settings before training."
                        )
        except Exception as e:
            logger.warning("Failed to compute points-based FLAT tolerance; falling back to pct: %s", e)
            flat_tolerance_pts = None
            use_points = False

    lookback_prices = 200
    lookback_candles = 500
    horizon = 2  # 2-minute direction

    for i, ts in enumerate(timestamps):
        # need t+2
        if i + horizon >= n:
            continue

        cur = df.iloc[i]
        ref_close = float(cur["close"])
        if not np.isfinite(ref_close) or ref_close <= 0.0:
            continue

        future = df.iloc[i + horizon]
        fut_close = float(future["close"])
        if not np.isfinite(fut_close):
            continue

        move = fut_close - ref_close

        if use_points and flat_tolerance_pts is not None:
            tol = flat_tolerance_pts
        else:
            tol = flat_tolerance_pct * ref_close

        if move > tol:
            label = "BUY"
        elif move < -tol:
            label = "SELL"
        else:
            label = "FLAT"

        # Build features similar to live pipeline
        # price history
        px_hist = df["close"].astype(float).iloc[max(0, i - lookback_prices + 1): i + 1].tolist()
        if len(px_hist) < 5:
            continue

        # Core trend/indicator features
        ema_feats = FeaturePipeline.compute_emas(px_hist)
        ta_feats = TA.compute_ta_bundle(px_hist)

        safe_df = df.iloc[max(0, i - lookback_candles + 1): i + 1]

        pat_feats = FeaturePipeline.compute_candlestick_patterns(
            candles=safe_df.tail(max(3, 5)),
            rvol_window=int(os.getenv("OFFLINE_PAT_RVOL_WINDOW", "5")),
            rvol_thresh=float(os.getenv("OFFLINE_PAT_RVOL_THRESH", "1.2")),
            min_winrate=float(os.getenv("OFFLINE_PAT_MIN_WINRATE", "0.55")),
        ) or {}

        mtf_feats = FeaturePipeline.compute_mtf_pattern_consensus(
            candle_df=safe_df,
            timeframes=["1T", "3T", "5T"],
            rvol_window=int(os.getenv("OFFLINE_MTF_RVOL_WINDOW", "5")),
            rvol_thresh=float(os.getenv("OFFLINE_MTF_RVOL_THRESH", "1.2")),
            min_winrate=float(os.getenv("OFFLINE_MTF_MIN_WINRATE", "0.55")),
        ) or {}

        sr_feats = FeaturePipeline.compute_sr_features(safe_df)

        micro = FeaturePipeline.compute_micro_trend(px_hist)
        micro_slope = float(micro.get("micro_slope", 0.0))
        imbalance = float(micro.get("micro_imbalance", 0.0))
        mean_drift_pct = float(micro.get("mean_drift_pct", 0.0))
        last_z = float(micro.get("last_zscore", 0.0))

        # --- Regime features: realized range / volatility and time-of-day ---
        atr_1t = 0.0
        atr_3t = 0.0
        rv_10 = 0.0
        tod_sin = 0.0
        tod_cos = 0.0

        try:
            if not safe_df.empty:
                hi = safe_df["high"].astype(float)
                lo = safe_df["low"].astype(float)
                rng = (hi - lo).clip(lower=0.0)

                if len(rng) >= 1:
                    atr_1t = float(rng.iloc[-1])
                if len(rng) >= 3:
                    atr_3t = float(rng.tail(3).mean())
                else:
                    atr_3t = atr_1t

            px_arr = np.asarray(px_hist, dtype=float)
            if px_arr.size >= 10:
                diff = np.diff(px_arr)
                if diff.size >= 3:
                    rv_10 = float(np.std(diff[-10:]))
        except Exception:
            atr_1t = atr_3t = rv_10 = 0.0

        try:
            # time-of-day as a cycle (minutes since midnight)
            minute_of_day = ts.hour * 60 + ts.minute
            tod_angle = 2.0 * np.pi * (minute_of_day / (24.0 * 60.0))
            tod_sin = float(np.sin(tod_angle))
            tod_cos = float(np.cos(tod_angle))
        except Exception:
            tod_sin = tod_cos = 0.0

        # --- Reversal / regime flags: wick extremes, VWAP reversion, CVD divergence ---
        try:
            if isinstance(safe_df, pd.DataFrame) and not safe_df.empty:
                last_candle = safe_df.iloc[-1]
                prev_candle = safe_df.iloc[-2] if len(safe_df) >= 2 else last_candle
            else:
                last_candle = None
                prev_candle = None
        except Exception:
            last_candle = None
            prev_candle = None

        try:
            wick_up, wick_down = FeaturePipeline._compute_wick_extremes(last_candle) if last_candle is not None else (0.0, 0.0)
        except Exception:
            wick_up, wick_down = 0.0, 0.0

        # Offline has no futures sidecar by default; use None
        try:
            vwap_val = None
            px_change = float(px_hist[-1] - px_hist[-2]) if len(px_hist) >= 2 else 0.0
            cvd_div = FeaturePipeline._compute_cvd_divergence(px_change, None)
        except Exception:
            vwap_val = None
            cvd_div = 0.0

        try:
            vwap_rev_flag = float(
                FeaturePipeline._compute_vwap_reversion_flag(
                    safe_df if (isinstance(safe_df, pd.DataFrame) and not safe_df.empty) else None,
                    vwap_val,
                )
            ) if (safe_df is not None) else 0.0
        except Exception:
            vwap_rev_flag = 0.0

        rev_cross_feats = FeaturePipeline.compute_reversal_cross_features(
            safe_df.tail(20) if isinstance(safe_df, pd.DataFrame) else pd.DataFrame(),
            {
                "wick_extreme_up": float(wick_up),
                "wick_extreme_down": float(wick_down),
                "vwap_reversion_flag": float(vwap_rev_flag),
                "cvd_divergence": float(cvd_div),
            },
        )

        features_raw = {
            **ema_feats,
            **ta_feats,
            **pat_feats,
            **mtf_feats,
            **sr_feats,
            "micro_slope": micro_slope,
            "micro_imbalance": imbalance,
            "mean_drift_pct": mean_drift_pct,
            "last_price": ref_close,
            "last_zscore": float(last_z),
            "atr_1t": atr_1t,
            "atr_3t": atr_3t,
            "rv_10": rv_10,
            "tod_sin": tod_sin,
            "tod_cos": tod_cos,
            # Reversal/regime flags
            "wick_extreme_up": float(wick_up),
            "wick_extreme_down": float(wick_down),
            "vwap_reversion_flag": float(vwap_rev_flag),
            "cvd_divergence": float(cvd_div),
        }
        features_raw.update(rev_cross_feats)

        # Normalisation scale: std of returns
        try:
            arr_diff = np.diff(np.asarray(px_hist, dtype=float))
            scale = float(np.std(arr_diff)) if arr_diff.size >= 3 else 1.0
            scale = max(1e-6, scale)
        except Exception:
            scale = 1.0

        features_norm = FeaturePipeline.normalize_features(features_raw, scale=scale)
        row = {"ts": ts, "label": label}
        for k, v in features_norm.items():
            try:
                row[k] = float(v)
            except Exception:
                continue
        rows.append(row)

    df_out = pd.DataFrame(rows)
    if df_out.empty:
        logger.error("2-minute dataset is empty after processing.")
        return df_out

    df_out = df_out.drop_duplicates(subset=["ts"], keep="last").sort_values("ts")
    logger.info("Built 2-minute dataset: %d rows (BUY/SELL/FLAT)", len(df_out))
    return df_out


def train_models_2min(df_train: pd.DataFrame, xgb_out_path: str, neutral_out_path: str) -> None:
    """
    Train directional XGB (BUY vs SELL) and neutrality model (FLAT vs non-FLAT)
    on the 2-minute dataset, then save to disk.
    """
    if df_train is None or df_train.empty:
        logger.error("No training data for 2-minute model.")
        return

    # Optional: shuffle labels for leakage sanity check (offline only)
    if os.getenv("SHUFFLE_LABELS_FOR_SANITY", "0") == "1":
        logger.warning("[LEAKAGE] SHUFFLE_LABELS_FOR_SANITY=1 → shuffling labels for sanity test (offline only)")
        df_train = df_train.copy()
        df_train["label"] = np.random.permutation(df_train["label"].values)

    # Build feature matrix
    exclude = {"ts", "label"}
    drop_prefixes = ("meta_", "p_xgb_")
    feat_cols = sorted([
        c for c in df_train.columns
        if c not in exclude
        and df_train[c].dtype != "O"
        and not any(c.startswith(p) for p in drop_prefixes)
    ])
    if not feat_cols:
        logger.error("No numeric feature columns found.")
        return

    df_feat = df_train[feat_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    X_all = df_feat.values
    labels = df_train["label"].astype(str).values

    # Neutrality (FLAT vs non-FLAT)
    y_neu = (labels == "FLAT").astype(int)
    X_neu = X_all
    logger.info("Neutrality dataset: n=%d, pos=%.3f", len(y_neu), float(y_neu.mean()) if len(y_neu) else 0.0)

    # Directional: BUY vs SELL, ignore FLAT
    mask_dir = np.isin(labels, ["BUY", "SELL"])
    df_dir = df_train[mask_dir]
    if df_dir.empty:
        logger.error("No BUY/SELL rows for directional model.")
        return

    X_dir = df_dir[feat_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).values
    y_dir = (df_dir["label"] == "BUY").astype(int).values
    pos_share = float(y_dir.mean()) if len(y_dir) else 0.0
    minor_share = float(min(pos_share, 1.0 - pos_share))
    logger.info(
        "Directional dataset: n=%d, pos_share=%.3f, minor_share=%.3f",
        len(y_dir),
        pos_share,
        minor_share,
    )

    # Ensure no NaNs / inf in features
    if not np.isfinite(X_dir).all():
        raise ValueError("Directional feature matrix contains NaN or inf values.")
    if not np.isfinite(X_neu).all():
        raise ValueError("Neutrality feature matrix contains NaN or inf values.")

    # Train XGB using your existing helper
    xgb_model = _train_xgb(X_dir, y_dir)
    if xgb_model is None:
        logger.error("Offline XGB training returned None.")
        return

    # Embed schema
    try:
        schema = {"feature_names": list(feat_cols)}
        xgb_model.set_attr(feature_schema=json.dumps(schema))
        logger.info("Embedded feature_schema into XGB booster (n=%d)", len(feat_cols))
    except Exception as e:
        logger.warning("Failed to set booster feature_schema attr: %s", e)

    # Train neutrality model
    neutral_model = _train_neutrality(X_neu, y_neu)
    if neutral_model is None:
        logger.error("Offline neutrality training returned None.")
        return

    # Save models
    from pathlib import Path
    Path(os.path.dirname(xgb_out_path) or ".").mkdir(parents=True, exist_ok=True)
    Path(os.path.dirname(neutral_out_path) or ".").mkdir(parents=True, exist_ok=True)

    try:
        import xgboost as xgb
        xgb_model.save_model(xgb_out_path)
        logger.info("Saved 2-minute XGB model to %s", xgb_out_path)
    except Exception as e:
        logger.error("Failed to save XGB model: %s", e)

    try:
        import joblib
        joblib.dump(neutral_model, neutral_out_path)
        logger.info("Saved 2-minute neutrality model to %s", neutral_out_path)
    except Exception as e:
        logger.error("Failed to save neutrality model: %s", e)


def _parse_train_datetime(s: str) -> date:
    """
    Parse TRAIN_START_DATE / TRAIN_END_DATE env vars.

    Accepts:
        - "YYYY-MM-DD"
        - "YYYY-MM-DD HH:MM:SS"

    Returns a date object (time part is ignored for now).
    """
    s = s.strip()
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            dt = datetime.strptime(s, fmt)
            return dt.date()
        except ValueError:
            continue
    raise ValueError(f"Invalid date format {s!r}; expected 'YYYY-MM-DD' or 'YYYY-MM-DD HH:MM:SS'")


def main():
    # Date range from env
    start_str = os.getenv("TRAIN_START_DATE", "").strip()
    end_str = os.getenv("TRAIN_END_DATE", "").strip()


    if not start_str or not end_str:
        logger.error("TRAIN_START_DATE and TRAIN_END_DATE must be set (YYYY-MM-DD or 'YYYY-MM-DD HH:MM:SS').")
        return

    try:
        start_date = _parse_train_datetime(start_str)
        end_date = _parse_train_datetime(end_str)
    except Exception as e:
        logger.error("Invalid TRAIN_START_DATE / TRAIN_END_DATE: %s", e)
        return

    if end_date < start_date:
        logger.error("TRAIN_END_DATE must be >= TRAIN_START_DATE.")
        return

    xgb_out = os.getenv("XGB_PATH", "").strip()
    neutral_out = os.getenv("NEUTRAL_PATH", "").strip()
    if not xgb_out or not neutral_out:
        logger.error("XGB_PATH and NEUTRAL_PATH must be set for offline training outputs.")
        return

    df_intraday = fetch_intraday_range(start_date, end_date)
    if df_intraday.empty:
        return

    df_train = build_2min_dataset(df_intraday)
    if df_train.empty:
        return

    train_models_2min(df_train, xgb_out_path=xgb_out, neutral_out_path=neutral_out)
    logger.info("Offline 2-minute training complete.")


if __name__ == "__main__":
    main()
