#!/usr/bin/env python3
"""
offline_train_2min.py

Offline trainer for a trade-window TP/SL outcome model.

- Fetches 1-minute OHLCV from Dhan /v2/charts/intraday for a date range.
- For each bar t, defines a hypothetical trade with:
      * horizon H = TRADE_HORIZON_MIN minutes (bars)
      * symmetric TP/SL in % of entry price:
            TP = close_t * (1 + TRADE_TP_PCT)
            SL = close_t * (1 - TRADE_SL_PCT)  for BUY
            TP = close_t * (1 - TRADE_TP_PCT)
            SL = close_t * (1 + TRADE_SL_PCT)  for SELL
- In the forward H-minute window, checks which side hits its TP/SL first:

      long_outcome  ∈ {WIN, LOSS, NONE}
      short_outcome ∈ {WIN, LOSS, NONE}

  Combined into a label:
      BUY   if long_outcome == "WIN" and short_outcome != "WIN"
      SELL  if short_outcome == "WIN" and long_outcome != "WIN"
      FLAT  otherwise (no clear edge / ambiguous)

- Builds features from:
      price + imbalance + pivots / FVG / order-blocks + indicators + candlestick patterns
- Trains:
    - XGBoost directional model (BUY vs SELL on non-FLAT rows)
    - Neutrality LogisticRegression (FLAT vs non-FLAT)

Environment:
    TRADE_HORIZON_MIN  : horizon in minutes/bars (e.g. 10)
    TRADE_TP_PCT       : TP as fraction (e.g. 0.0015 for 0.15%)
    TRADE_SL_PCT       : SL as fraction (e.g. 0.0008 for 0.08%)
"""

import os
import json
import logging
import time
from datetime import datetime, timedelta, date
from typing import Optional, List, Dict

import numpy as np
import pandas as pd
import requests

from feature_pipeline import FeaturePipeline, TA
from online_trainer import _train_xgb, _train_neutrality
from intraday_cache_manager import get_cache_manager

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

# Optional on-disk cache for intraday candles
INTRADAY_CACHE_DIR = os.getenv("INTRADAY_CACHE_DIR", "").strip()
INTRADAY_CACHE_ENABLE = os.getenv("INTRADAY_CACHE_ENABLE", "1").strip() in ("1", "true", "yes")

def _check_dhan_config() -> bool:
    ok = True
    if not DHAN_ACCESS_TOKEN:
        logger.error("DHAN_ACCESS_TOKEN is not set.")
        ok = False
    if not DHAN_CLIENT_ID:
        logger.error("DHAN_CLIENT_ID is not set.")
        ok = False
    return ok


def fetch_intraday_for_day(dt: date, max_retries: int = 3) -> Optional[pd.DataFrame]:
    """
    Fetch 1-minute intraday OHLCV for a single day from Dhan.

    With INTRADAY_CACHE_ENABLE=1 and INTRADAY_CACHE_DIR set, this will:
    - First try to load a cached CSV for that date.
    - If missing or broken, fall back to the API and then write the cache.
    This implementation assumes a POST /v2/charts/intraday with JSON body.
    Adjust fields per https://dhanhq.co/docs/v2/historical-data/ if needed.
    """
    if not _check_dhan_config():
        return None

    cache_path = None
    if INTRADAY_CACHE_ENABLE and INTRADAY_CACHE_DIR:
        try:
            os.makedirs(INTRADAY_CACHE_DIR, exist_ok=True)
            cache_path = os.path.join(
                INTRADAY_CACHE_DIR,
                f"{NIFTY_INSTRUMENT}_{dt.strftime('%Y%m%d')}_1m.csv",
            )
            if os.path.exists(cache_path):
                df_cached = pd.read_csv(
                    cache_path,
                    parse_dates=["timestamp"],
                    index_col="timestamp",
                ).sort_index()

                if not df_cached.empty:
                    logger.info(
                        "Loaded %d cached candles for %s from %s",
                        len(df_cached),
                        dt.isoformat(),
                        cache_path,
                    )
                    expected_min = 300
                    if len(df_cached) < expected_min:
                        logger.warning(
                            "Cached intraday file %s has only %d rows (<%d)",
                            cache_path,
                            len(df_cached),
                            expected_min,
                        )
                    return df_cached
        except Exception as e:
            logger.warning(
                "Failed to read intraday cache for %s (%s); refetching from API.",
                dt.isoformat(),
                e,
            )

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
    logger.info("Fetching intraday for %s from %s", dt.isoformat(), url)

    data = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=20)
        except Exception as e:
            logger.error(
                "Network error on %s (attempt %d/%d): %s",
                dt.isoformat(),
                attempt,
                max_retries,
                e,
            )
            if attempt == max_retries:
                return None
            time.sleep(2 * attempt)
            continue

        if resp.status_code == 429 or resp.status_code >= 500:
            logger.warning(
                "Server/rate-limit error %s on %s (attempt %d/%d): %s",
                resp.status_code,
                dt.isoformat(),
                attempt,
                max_retries,
                resp.text[:200],
            )
            if attempt == max_retries:
                return None
            time.sleep(2 * attempt)
            continue

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

        try:
            data = resp.json()
        except Exception as e:
            logger.error("Failed to parse intraday JSON for %s: %s", dt.isoformat(), e)
            return None
        break

    if data is None:
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
    expected_min = 300  # very loose lower bound
    if len(df) < expected_min:
        logger.warning("Suspiciously few candles on %s: got %d", dt.isoformat(), len(df))
    logger.info("Fetched %d candles for %s", len(df), dt.isoformat())

    # Write to cache if requested
    if INTRADAY_CACHE_ENABLE and INTRADAY_CACHE_DIR and cache_path:
        try:
            df_reset = df.reset_index()
            df_reset.to_csv(cache_path, index=False)
            logger.info(
                "Cached %d candles for %s to %s",
                len(df),
                dt.isoformat(),
                cache_path,
            )
        except Exception as e:
            logger.warning(
                "Failed to write intraday cache for %s to %s (%s); continuing without cache.",
                dt.isoformat(),
                cache_path,
                e,
            )

    return df


def fetch_intraday_range(start: date, end: date) -> pd.DataFrame:
    """
    Fetch intraday 1-minute OHLCV for all days in [start, end], inclusive.
    
    Strategy:
    1. Load all cached data first (no API calls)
    2. Identify dates that need API fetching
    3. Fetch only missing dates from API
    4. Merge cached + newly fetched data
    """
    # Initialize cache manager
    cache_mgr = get_cache_manager(INTRADAY_CACHE_DIR, NIFTY_INSTRUMENT)
    
    # Log cache status
    cache_status = cache_mgr.get_cache_summary(start, end)
    logger.info("Cache status: %d cached, %d missing out of %d trading days",
                cache_status["cached_dates"],
                cache_status["missing_dates"],
                cache_status["total_trading_days"])
    
    # Step 1: Load all cached data
    all_dfs: List[pd.DataFrame] = []
    if INTRADAY_CACHE_ENABLE:
        df_cached = cache_mgr.load_cached_data(start, end)
        if not df_cached.empty:
            all_dfs.append(df_cached)
            logger.info("Loaded %d candles from cache", len(df_cached))
    
    # Step 2: Identify missing dates and fetch them
    missing_dates = cache_mgr.get_missing_dates(start, end, exclude_weekends=True)
    
    if missing_dates:
        logger.info("Fetching %d missing dates from API", len(missing_dates))
        for dt in missing_dates:
            df_day = fetch_intraday_for_day(dt)
            if df_day is not None and not df_day.empty:
                all_dfs.append(df_day)
                
                # Cache the newly fetched data
                if INTRADAY_CACHE_ENABLE:
                    cache_mgr.save_cached_data(dt, df_day)
    else:
        logger.info("All dates for range %s to %s are cached", start.isoformat(), end.isoformat())
    
    # Step 3: Consolidate all data
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


def make_trade_outcome_label(
    df_candles: pd.DataFrame,
    idx: int,
    horizon: int,
    tp_pct: float,
    sl_pct: float,
    side: str,
) -> str:
    """
    Determine outcome for a hypothetical trade opened at bar idx.

    side  : "BUY" or "SELL"
    return: "WIN", "LOSS", or "NONE"
    """
    if df_candles is None or df_candles.empty:
        return "NONE"

    n = len(df_candles)
    if idx < 0 or idx >= n - 1:
        return "NONE"

    horizon = max(1, int(horizon))
    if idx + 1 >= n or idx + horizon >= n:
        # Not enough future bars
        return "NONE"

    try:
        entry_row = df_candles.iloc[idx]
        entry_px = float(entry_row["close"])
        if not np.isfinite(entry_px) or entry_px <= 0.0:
            return "NONE"
    except Exception:
        return "NONE"

    try:
        highs = df_candles["high"].astype(float).values
        lows = df_candles["low"].astype(float).values
    except Exception:
        return "NONE"

    if highs.shape[0] != n or lows.shape[0] != n:
        return "NONE"

    if side == "BUY":
        tp = entry_px * (1.0 + float(tp_pct))
        sl = entry_px * (1.0 - float(sl_pct))
        lo_idx = idx + 1
        hi_idx = idx + horizon
        for j in range(lo_idx, hi_idx + 1):
            h = highs[j]
            l = lows[j]
            if not (np.isfinite(h) and np.isfinite(l)):
                continue
            if h >= tp:
                return "WIN"
            if l <= sl:
                return "LOSS"
        return "NONE"

    elif side == "SELL":
        tp = entry_px * (1.0 - float(tp_pct))
        sl = entry_px * (1.0 + float(sl_pct))
        lo_idx = idx + 1
        hi_idx = idx + horizon
        for j in range(lo_idx, hi_idx + 1):
            h = highs[j]
            l = lows[j]
            if not (np.isfinite(h) and np.isfinite(l)):
                continue
            if l <= tp:
                return "WIN"
            if h >= sl:
                return "LOSS"
        return "NONE"

    return "NONE"


def build_2min_dataset(df_candles: pd.DataFrame) -> pd.DataFrame:
    """
    Build a trade-window dataset:
      - labels: BUY/SELL/FLAT per trade-outcome logic (TP/SL vs horizon)
      - features: price + imbalance + structure (pivot/FVG/order-block) + TA + patterns.

    The name is legacy but semantics are now trade-window rather than raw 2-minute direction.
    """
    if df_candles is None or df_candles.empty:
        logger.error("Empty intraday DataFrame for trade-window dataset.")
        return pd.DataFrame()

    df = df_candles.copy()

    # Ensure timestamp index
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df = df.set_index("timestamp")

    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must be indexed by timestamp for build_2min_dataset().")

    df = df.sort_index()

    # Basic validity filter
    try:
        close = df["close"].astype(float)
        high = df["high"].astype(float)
        low = df["low"].astype(float)
    except Exception as e:
        logger.error("Candles missing close/high/low: %s", e)
        return pd.DataFrame()

    mask_ok = close.notna() & high.notna() & low.notna()
    df = df[mask_ok].copy()
    close = close[mask_ok]

    if df.empty:
        logger.error("All rows dropped as invalid in build_2min_dataset.")
        return pd.DataFrame()

    # --- Trade config (dynamic via env) ---
    trade_tp_pct = float(os.getenv("TRADE_TP_PCT", "0.0015") or "0.0015")
    trade_sl_pct = float(os.getenv("TRADE_SL_PCT", "0.0008") or "0.0008")
    trade_horizon_min = int(os.getenv("TRADE_HORIZON_MIN", "10") or "10")
    horizon = max(1, trade_horizon_min)

    rows = []
    idx_list = list(df.index)
    n = len(df)

    for i, ts in enumerate(idx_list):
        # Need enough bars ahead to evaluate the full horizon
        if i + horizon >= n:
            break

        try:
            ref_close = float(close.iloc[i])
            if not np.isfinite(ref_close) or ref_close <= 0.0:
                continue
        except Exception:
            continue

        # --- Trade-outcome labels (long & short) ---
        long_outcome = make_trade_outcome_label(
            df, i, horizon, trade_tp_pct, trade_sl_pct, side="BUY"
        )
        short_outcome = make_trade_outcome_label(
            df, i, horizon, trade_tp_pct, trade_sl_pct, side="SELL"
        )

        if (long_outcome == "WIN") and (short_outcome != "WIN"):
            label = "BUY"
        elif (short_outcome == "WIN") and (long_outcome != "WIN"):
            label = "SELL"
        else:
            label = "FLAT"

        # --- Feature window around reference bar ---
        try:
            hist_lookback = 120
            safe_df = df.iloc[max(0, i - hist_lookback + 1): i + 1].copy()
            px_hist = safe_df["close"].astype(float).values.tolist()
        except Exception:
            safe_df = pd.DataFrame()
            px_hist = []

        if len(px_hist) < 20:
            # Need some price history for micro-trend / TA
            continue

        # Core price/TA
        ema_feats = FeaturePipeline.compute_emas(px_hist)
        ta_feats = TA.compute_ta_bundle(px_hist)

        # Patterns / MTF
        try:
            pat_feats = FeaturePipeline.compute_pattern_features(safe_df)
        except Exception:
            pat_feats = {}

        try:
            mtf_feats = FeaturePipeline.compute_mtf_pattern_features(
                safe_df,
                base_tf="1T",
                higher_tfs=["3T", "5T"],
                rvol_window=int(os.getenv("OFFLINE_MTF_RVOL_WINDOW", "5") or "5"),
                rvol_thresh=float(os.getenv("OFFLINE_MTF_RVOL_THRESH", "1.2") or "1.2"),
                min_winrate=float(os.getenv("OFFLINE_MTF_MIN_WINRATE", "0.55") or "0.55"),
            ) or {}
        except Exception:
            mtf_feats = {}

        # Support/resistance
        try:
            sr_feats = FeaturePipeline.compute_sr_features(safe_df)
        except Exception:
            sr_feats = {}

        # Micro-trend / imbalance
        try:
            micro = FeaturePipeline.compute_micro_trend(px_hist)
            micro_slope = float(micro.get("micro_slope", 0.0))
            imbalance = float(micro.get("micro_imbalance", 0.0))
            mean_drift_pct = float(micro.get("mean_drift_pct", 0.0))
            last_z = float(micro.get("last_zscore", 0.0))
        except Exception:
            micro_slope = imbalance = mean_drift_pct = last_z = 0.0

        # Volatility features
        try:
            atr_1t = float(FeaturePipeline.compute_atr(safe_df, window=14) or 0.0)
            atr_3t = float(FeaturePipeline.compute_atr(safe_df, window=42) or 0.0)
        except Exception:
            atr_1t = atr_3t = 0.0

        try:
            rv_10 = float(FeaturePipeline.compute_realised_vol(px_hist, window=10) or 0.0)
        except Exception:
            rv_10 = 0.0

        # Time-of-day
        try:
            tod_feats = FeaturePipeline.compute_tod_features(ts)
            tod_sin = float(tod_feats.get("tod_sin", 0.0))
            tod_cos = float(tod_feats.get("tod_cos", 0.0))
        except Exception:
            tod_sin = tod_cos = 0.0

        # Wick extremes
        try:
            last_candle = safe_df.iloc[-1]
        except Exception:
            last_candle = None

        try:
            if last_candle is not None:
                wick_up, wick_down = FeaturePipeline._compute_wick_extremes(last_candle)
            else:
                wick_up = wick_down = 0.0
        except Exception:
            wick_up = wick_down = 0.0

        # Offline has no futures sidecar: mark as NaN/missing
        vwap_val = None
        cvd_div = np.nan
        vwap_rev_flag = np.nan

        # Reversal / cross regime
        try:
            rev_cross_feats = FeaturePipeline.compute_reversal_cross_features(
                safe_df.tail(20) if isinstance(safe_df, pd.DataFrame) else pd.DataFrame(),
                {
                    "wick_extreme_up": float(wick_up),
                    "wick_extreme_down": float(wick_down),
                    "vwap_reversion_flag": float(vwap_rev_flag),
                    "cvd_divergence": float(cvd_div),
                },
            )
        except Exception:
            rev_cross_feats = {}

        # NEW: structure bundle (pivot swipe + FVG + order-block)
        try:
            struct_feats = FeaturePipeline.compute_structure_bundle(
                safe_df.tail(40) if isinstance(safe_df, pd.DataFrame) else pd.DataFrame()
            )
        except Exception:
            struct_feats = {}

        features_raw = {
            **ema_feats,
            **ta_feats,
            **pat_feats,
            **mtf_feats,
            **sr_feats,
            **struct_feats,
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
            "wick_extreme_up": float(wick_up),
            "wick_extreme_down": float(wick_down),
            "vwap_reversion_flag": float(vwap_rev_flag),
            "cvd_divergence": float(cvd_div),
        }
        features_raw.update(rev_cross_feats)

        # Normalisation scale
        try:
            arr_diff = np.diff(np.asarray(px_hist, dtype=float))
            scale = float(np.std(arr_diff)) if arr_diff.size >= 3 else 1.0
            scale = max(1e-6, scale)
        except Exception:
            scale = 1.0

        features_norm = FeaturePipeline.normalize_features(features_raw, scale=scale)
        rows.append(
            {
                "ts": ts,
                "label": label,
                "features": json.dumps(features_norm),
            }
        )

    if not rows:
        logger.error("No rows built in build_2min_dataset (trade-window).")
        return pd.DataFrame()

    return pd.DataFrame(rows)


def train_models_2min(df_train: pd.DataFrame, xgb_out_path: str, neutral_out_path: str) -> None:
    """
    Train directional XGB (BUY vs SELL) and neutrality model (FLAT vs non-FLAT)
    on the 2-minute dataset, then save to disk.
    """
    if df_train is None or df_train.empty:
        logger.error("No training data for 2-minute model.")
        return

    # Expand json features if present
    if "features" in df_train.columns and df_train["features"].dtype == object:
        try:
            expanded_rows = []
            for _, r in df_train.iterrows():
                feats = {}
                try:
                    feats = json.loads(r["features"]) if isinstance(r["features"], str) else {}
                except Exception:
                    feats = {}
                feats["ts"] = r.get("ts")
                feats["label"] = r.get("label")
                expanded_rows.append(feats)
            df_train = pd.DataFrame(expanded_rows)
        except Exception as e:
            logger.error("Failed to expand features column: %s", e)
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

    # Show cache status before fetching
    if INTRADAY_CACHE_ENABLE and INTRADAY_CACHE_DIR:
        cache_mgr = get_cache_manager(INTRADAY_CACHE_DIR, NIFTY_INSTRUMENT)
        cache_info = cache_mgr.get_cache_summary(start_date, end_date)
        logger.info("=== CACHE STATUS ===")
        logger.info("Cache directory: %s", cache_info["cache_directory"])
        logger.info("Total cached dates: %d", cache_info["total_cached_files"])
        logger.info("Date range: %s to %s", cache_info["date_range_start"], cache_info["date_range_end"])
        logger.info("Trading days in range: %d", cache_info["total_trading_days"])
        logger.info("Cached: %d, Missing: %d", cache_info["cached_dates"], cache_info["missing_dates"])
        if cache_info["missing_dates"] > 0 and cache_info["missing_date_list"]:
            logger.info("First missing dates: %s", ", ".join(cache_info["missing_date_list"][:3]))
        logger.info("===================")
    
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
