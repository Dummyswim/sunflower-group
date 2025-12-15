#!/usr/bin/env python3
"""
offline_train_5min.py

Offline trainer for the 5-minute trade-window model using historical
feature logs (feature_log_hist.csv + feature_log.csv).

Design:
- No SPOT_PATH / spot_5min_history.csv dependency.
- Uses the same CSV parsing & feature selection logic as online_trainer.py.
- Trains:
    * Directional XGB: BUY vs SELL (all directional rows).
    * Neutrality LogisticRegression: FLAT vs non-FLAT (for logging / diagnostics).
- Saves:
    * xgb_5min_trade_window.json
    * neutral_5min.pkl
    * xgb_5min_feature_schema.json
    * xgb_5min_feature_stats.json
"""

import glob
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib  # type: ignore
import numpy as np
import pandas as pd
import xgboost as xgb  # type: ignore
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from feature_pipeline import FeaturePipeline, TA
from backups.old_docs.main_event_loop import compute_tp_sl_direction_label

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

# ---------------------------------------------------------------------
# Paths / config
# ---------------------------------------------------------------------

PROD_DIR = Path(os.getenv("PRODUCTION_DIR", "trained_models/production"))

FEATURE_LOG_HIST_PATH = Path(
    os.getenv("FEATURE_LOG_HIST_PATH", str(PROD_DIR / "feature_log_hist.csv"))
)
FEATURE_LOG_PATH = Path(
    os.getenv("FEATURE_LOG_PATH", str(PROD_DIR / "feature_log.csv"))
)

XGB_OUT_PATH = PROD_DIR / "xgb_5min_trade_window.json"
NEUTRAL_OUT_PATH = PROD_DIR / "neutral_5min.pkl"
SCHEMA_OUT_PATH = PROD_DIR / "xgb_5min_feature_schema.json"
STATS_OUT_PATH = PROD_DIR / "xgb_5min_feature_stats.json"

MIN_ROWS_PER_FILE = int(os.getenv("OFFLINE_MIN_ROWS", "500") or "500")

# Cache/backfill config
CACHE_STATE_PATH = "trained_models/production/offline_cache_progress.json"
CACHE_GLOB = os.getenv("OFFLINE_SPOT_CACHE_GLOB", "INDEX_*_1m.csv")
FUT_CANDLES_PATH = os.getenv("OFFLINE_FUT_CANDLES_PATH", "fut_candles_vwap_cvd.csv")

INTRADAY_CACHE_BASE = os.getenv(
    "OFFLINE_SPOT_CACHE_GLOB",
    "/home/hanumanth/Documents/sunflower-group_2/onemin_prediction/data/intraday_cache",
)
CACHE_1M_GLOB = os.path.join(INTRADAY_CACHE_BASE, "INDEX_*_1m.csv")
USE_1M_CACHE = str(os.getenv("OFFLINE_USE_1M_CACHE", "0")).lower() in (
    "1",
    "true",
    "yes",
)
MAX_CACHE_DAYS = int(os.getenv("OFFLINE_MAX_CACHE_DAYS", "260") or "260")

# Optional: prefer a compact, high-information subset if available
PREFERRED_FEATURES: Tuple[str, ...] = (
    # Trend / level
    "ema_8",
    "ema_21",
    "ema_50",
    "last_price",
    # Momentum / micro-trend
    "ta_rsi14",
    "ta_macd_hist",
    "ta_bb_pctb",
    "ta_bb_bw",
    "micro_slope",
    "micro_imbalance",
    "mean_drift_pct",
    "last_zscore",
    # Volatility
    "atr_1t",
    "rv_10",
    # Time-of-day
    "tod_sin",
    "tod_cos",
    # Structure bundle
    "struct_pivot_swipe_up",
    "struct_pivot_swipe_down",
    "struct_fvg_up_present",
    "struct_fvg_down_present",
    "struct_ob_bull_present",
    "struct_ob_bear_present",
    # Micro-structure & futures features
    "wick_extreme_up",
    "wick_extreme_down",
    "vwap_reversion_flag",
    "cvd_divergence",
    # Extra TA strength / participation
    "ta_adx",
    "ta_mfi",
    "ta_obv_z",
)

# ---------------------------------------------------------------------
# 1. Parse feature logs (mirrors online_trainer._parse_feature_csv)
# ---------------------------------------------------------------------


def _parse_feature_csv(path: Path, min_rows: int = 200) -> Optional[pd.DataFrame]:
    """
    Parse feature_log-style CSV into a DataFrame.

    Expected CSV line layout (same as feature_log.csv):
      ts,decision,label,buy_prob,alpha,tradeable,is_flat,<...,>,"features=k=v;k=v;..."

    We expand the packed feature map into columns; futures features such as
    cvd_divergence / vwap_reversion_flag are taken directly from the log, so
    no training–serving skew.
    """
    try:
        if not path.exists():
            logger.info("[OFFLINE] Feature log not found: %s", path)
            return None

        # Wide CSV fast-path (headered logs or backfill output)
        try:
            df_head = pd.read_csv(path, nrows=5)
            if "ts" in df_head.columns and "label" in df_head.columns:
                df = pd.read_csv(path)
                if len(df) < min_rows:
                    logger.info(
                        "[OFFLINE] Not enough rows in %s: %d/%d",
                        path,
                        len(df),
                        min_rows,
                    )
                    return None

                if "decision" not in df.columns:
                    df["decision"] = "USER"
                if "buy_prob" not in df.columns:
                    df["buy_prob"] = 0.5
                if "alpha" not in df.columns:
                    df["alpha"] = 0.0
                if "tradeable" not in df.columns:
                    df["tradeable"] = True
                if "is_flat" not in df.columns:
                    df["is_flat"] = df["label"].astype(str).str.upper().eq("FLAT")

                df = df.drop_duplicates(subset=["ts"], keep="last")
                return df
        except Exception as e:
            logger.debug(f"[OFFLINE] Wide CSV parse fallback failed: {e}")

        rows: List[Dict] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                toks = [t.strip() for t in line.split(",") if t.strip() != ""]
                if len(toks) < 8:
                    continue

                try:
                    ts = toks[0]
                    decision = toks[1]
                    label = toks[2]
                    buy_prob = float(toks[3])
                    alpha = float(toks[4]) if toks[4] not in ("", "None", "nan") else 0.0
                    tradeable = toks[5].lower() == "true"
                    is_flat = toks[6].lower() == "true"

                    feat_map: Dict[str, float] = {}
                    for tok in toks[8:]:
                        if not tok:
                            continue
                        if tok.startswith("features="):
                            # packed k=v;k=v;...
                            try:
                                _, packed = tok.split("=", 1)
                                for kv in packed.split(";"):
                                    kv = kv.strip()
                                    if not kv or "=" not in kv:
                                        continue
                                    k, v = kv.split("=", 1)
                                    try:
                                        feat_map[k.strip()] = float(v.strip())
                                    except Exception:
                                        continue
                            except Exception:
                                continue
                            continue
                        if tok.startswith("latent="):
                            # latent embeddings are ignored in this simplified architecture
                            continue
                        if "=" in tok:
                            k, v = tok.split("=", 1)
                            try:
                                feat_map[k.strip()] = float(v.strip())
                            except Exception:
                                continue

                    row = {
                        "ts": ts,
                        "decision": decision,
                        "label": label,
                        "buy_prob": buy_prob,
                        "alpha": alpha,
                        "tradeable": tradeable,
                        "is_flat": is_flat,
                    }
                    row.update(feat_map)
                    rows.append(row)
                except Exception:
                    # skip malformed lines
                    continue

        if len(rows) < min_rows:
            logger.info(
                "[OFFLINE] Not enough rows in %s: %d/%d",
                path,
                len(rows),
                min_rows,
            )
            return None

        df = pd.DataFrame(rows)
        df = df.drop_duplicates(subset=["ts"], keep="last")
        return df

    except Exception as exc:
        logger.error(
            "[OFFLINE] Parse feature CSV failed for %s: %s", path, exc, exc_info=True
        )
        return None


def _load_all_features() -> Optional[pd.DataFrame]:
    """
    Load historical + recent feature logs, sort, de-dup.

    This is the only data source for offline 5-minute training, so futures
    features (cvd_divergence, vwap_reversion_flag, etc.) will be identical
    to what the live pipeline used at the time.
    """
    dfs: List[pd.DataFrame] = []

    df_hist = _parse_feature_csv(FEATURE_LOG_HIST_PATH, min_rows=MIN_ROWS_PER_FILE)
    if df_hist is not None:
        dfs.append(df_hist)

    df_live = _parse_feature_csv(FEATURE_LOG_PATH, min_rows=MIN_ROWS_PER_FILE)
    if df_live is not None:
        dfs.append(df_live)

    if not dfs:
        logger.error("[OFFLINE] No feature logs loaded; cannot train.")
        return None

    df = pd.concat(dfs, ignore_index=True)
    df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
    df = df.dropna(subset=["ts"])
    df = df.sort_values("ts").drop_duplicates(subset=["ts"], keep="last")

    logger.info("[OFFLINE] Loaded combined features: %d rows", len(df))
    return df


def _load_cache_state() -> set:
    if not os.path.exists(CACHE_STATE_PATH):
        return set()
    try:
        with open(CACHE_STATE_PATH, "r") as f:
            obj = json.load(f)
        return set(obj.get("processed_files", []))
    except Exception:
        return set()


def _save_cache_state(processed: set):
    obj = {"processed_files": sorted(list(processed))}
    with open(CACHE_STATE_PATH, "w") as f:
        json.dump(obj, f, indent=2)


def _load_fut_sidecar(path: str) -> Optional[pd.DataFrame]:
    """
    Load futures VWAP/CVD sidecar candles and compute fut_* features per row.

    Expected columns in fut_candles_vwap_cvd.csv:
        timestamp, open, high, low, close, volume, ticks, session_vwap, cvd

    We normalise deltas similar to _read_latest_fut_features in main_event_loop.
    """
    if not path or not os.path.exists(path):
        logger.warning("[OFFLINE] Futures sidecar not found at %s", path)
        return None

    try:
        df = pd.read_csv(path)
    except Exception as e:
        logger.error("[OFFLINE] Failed to read futures sidecar at %s: %s", path, e)
        return None

    if df.empty:
        logger.warning("[OFFLINE] Futures sidecar is empty: %s", path)
        return None

    cols = {c.lower(): c for c in df.columns}

    # Headerless sidecar fallback: if we can't find needed cols, re-read without header
    missing_core = (
        not any(k in cols for k in ("volume", "vol"))
        or not any(k in cols for k in ("cvd", "cum_cvd", "cumulative_cvd"))
        or not any(k in cols for k in ("session_vwap", "vwap", "fut_session_vwap"))
    )
    if missing_core:
        try:
            df_no_header = pd.read_csv(path, header=None)
            expected = [
                "timestamp",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "tick_count",
                "session_vwap",
                "cvd",
                "extra",
            ]
            df_no_header = df_no_header.iloc[:, : len(expected)]
            df_no_header.columns = expected[: df_no_header.shape[1]]
            df = df_no_header
            cols = {c.lower(): c for c in df.columns}
            logger.info("[OFFLINE] Re-read futures sidecar without header (assigned defaults).")
        except Exception as e:
            logger.error("[OFFLINE] Headerless futures sidecar parse failed: %s", e)
            return None

    ts_col = cols.get("timestamp") or cols.get("ts") or list(df.columns)[0]
    df["ts"] = pd.to_datetime(df[ts_col], utc=True).dt.tz_convert(None)

    def _map(col_name: str, *aliases: str) -> Optional[str]:
        for a in (col_name,) + aliases:
            c = cols.get(a)
            if c:
                return c
        return None

    vol_col = _map("volume", "vol", "candle_vol")
    cvd_col = _map("cvd", "cum_cvd", "cumulative_cvd")
    vwap_col = _map("session_vwap", "vwap", "fut_session_vwap")

    if vol_col is None or cvd_col is None or vwap_col is None:
        logger.error(
            "[OFFLINE] Futures sidecar missing volume/cvd/vwap columns (have=%s)",
            df.columns.tolist(),
        )
        return None

    df["fut_session_vwap"] = pd.to_numeric(df[vwap_col], errors="coerce")
    df["_vol"] = pd.to_numeric(df[vol_col], errors="coerce").fillna(0.0)
    df["_cvd"] = pd.to_numeric(df[cvd_col], errors="coerce").fillna(0.0)

    df["_cvd_prev"] = df["_cvd"].shift(1)
    df["_vol_prev"] = df["_vol"].shift(1).fillna(0.0)
    df["_cum_vol"] = df["_vol"].cumsum()

    cvd_delta = (df["_cvd"] - df["_cvd_prev"]).fillna(0.0)
    vol_delta = df["_vol"]

    cur_vol = df["_cum_vol"].replace(0.0, 1.0)
    cvd_norm = np.tanh(cvd_delta / cur_vol.replace(0.0, 1.0))
    vol_norm = np.tanh(vol_delta / 10000.0)

    out = pd.DataFrame(
        {
            "ts": df["ts"],
            "fut_session_vwap": df["fut_session_vwap"].astype(float),
            "fut_cvd_delta": cvd_norm.astype(float),
            "fut_vol_delta": vol_norm.astype(float),
        }
    )

    out = out.dropna(subset=["ts"]).drop_duplicates(subset=["ts"], keep="last")
    out = out.sort_values("ts").reset_index(drop=True)
    logger.info("[OFFLINE] Loaded futures sidecar features: n=%d", len(out))
    return out


if not hasattr(FeaturePipeline, "compute_feature_bundle"):
    def _compute_feature_bundle(closes: List[float], window_df: pd.DataFrame) -> Dict[str, float]:
        feat: Dict[str, float] = {}
        try:
            feat.update(FeaturePipeline.compute_emas(closes, periods=[8, 21, 50]))
        except Exception:
            pass

        try:
            candle_df_ta = window_df.copy()
            if candle_df_ta.empty:
                candle_df_ta = pd.DataFrame({"close": pd.Series(closes)})
            feat.update(TA.compute_ta_bundle(candle_df_ta, feat))
        except Exception:
            pass

        try:
            micro = FeaturePipeline.compute_micro_trend(closes[-64:])
            feat.update(micro)
        except Exception:
            pass

        try:
            feat["rv_10"] = FeaturePipeline.compute_rvol(
                window_df["volume"].to_numpy(dtype=float), window=10
            )
        except Exception:
            feat["rv_10"] = 0.0

        try:
            feat["atr_1t"] = FeaturePipeline.compute_atr_from_candles(window_df.tail(14))
        except Exception:
            feat["atr_1t"] = 0.0

        try:
            feat.update(FeaturePipeline.compute_tod_features(window_df["ts"].iloc[-1]))
        except Exception:
            pass

        try:
            feat.update(FeaturePipeline.compute_structure_features(window_df))
        except Exception:
            pass

        try:
            feat.update(FeaturePipeline.compute_pattern_features(window_df))
        except Exception:
            pass

        try:
            wick_up, wick_down = FeaturePipeline.compute_wick_extremes(window_df.iloc[-1])
        except Exception:
            wick_up, wick_down = 0.0, 0.0
        feat["wick_extreme_up"] = float(wick_up)
        feat["wick_extreme_down"] = float(wick_down)

        return FeaturePipeline.normalize_features(feat)

    FeaturePipeline.compute_feature_bundle = staticmethod(_compute_feature_bundle)


def _load_spot_candles_from_cache() -> Optional[pd.DataFrame]:
    """
    Load 1m spot candles from cached CSVs matching CACHE_1M_GLOB.

    Assumes each file looks like: INDEX_YYYYMMDD_1m.csv with columns:
        timestamp, open, high, low, close, volume
    """
    import glob

    paths = sorted(glob.glob(CACHE_1M_GLOB))
    if not paths:
        logger.warning("[OFFLINE] No 1m cache files found for glob: %s", CACHE_1M_GLOB)
        return None

    if len(paths) > MAX_CACHE_DAYS:
        logger.info(
            "[OFFLINE] Truncating cache files: %d -> %d (MAX_CACHE_DAYS)",
            len(paths),
            MAX_CACHE_DAYS,
        )
        paths = paths[-MAX_CACHE_DAYS:]

    dfs: List[pd.DataFrame] = []
    for p in paths:
        try:
            df = pd.read_csv(p)
        except Exception as e:
            logger.warning("[OFFLINE] Failed to read %s: %s (skipping)", p, e)
            continue
        if df.empty:
            continue

        cols = {c.lower(): c for c in df.columns}
        ts_col = cols.get("timestamp") or cols.get("ts") or list(df.columns)[0]
        df["ts"] = pd.to_datetime(df[ts_col])
        for k in ("open", "high", "low", "close", "volume"):
            c = cols.get(k)
            if c is None:
                logger.error("[OFFLINE] Missing %s in %s; skipping file", k, p)
                df = None
                break
            df[k] = pd.to_numeric(df[c], errors="coerce")
        if df is None:
            continue
        df = df.dropna(subset=["ts"]).sort_values("ts")
        dfs.append(df[["ts", "open", "high", "low", "close", "volume"]])

    if not dfs:
        logger.error("[OFFLINE] No valid 1m cache data loaded.")
        return None

    all_df = pd.concat(dfs, ignore_index=True)
    all_df = all_df.drop_duplicates(subset=["ts"]).sort_values("ts").reset_index(drop=True)
    logger.info("[OFFLINE] Loaded 1m spot cache: n=%d", len(all_df))
    return all_df


def _build_features_from_cache() -> Optional[pd.DataFrame]:
    """
    Build 5-min training rows from 1m spot cache + futures sidecar.

    This does NOT write CSVs; it just returns a DataFrame with:
        ts, label, tradeable, is_flat, and all feature columns.
    """
    spot_df = _load_spot_candles_from_cache()
    if spot_df is None or spot_df.empty:
        return None

    fut_df = _load_fut_sidecar(FUT_CANDLES_PATH)
    if fut_df is None or fut_df.empty:
        logger.warning("[OFFLINE] Proceeding without futures sidecar; fut_* features will be 0.")
        fut_df = None

    spot_df = spot_df.sort_values("ts").reset_index(drop=True)
    if fut_df is not None:
        fut_df = fut_df.sort_values("ts").reset_index(drop=True)

    closes = spot_df["close"].astype(float).tolist()
    vols = spot_df["volume"].astype(float).tolist()

    rows: List[Dict[str, object]] = []
    fp = FeaturePipeline(train_features={})

    horizon_min = int(os.getenv("OFFLINE_TP_SL_HORIZON_MIN", "10") or "10")
    horizon_min = max(1, horizon_min)
    base_tp_pct = float(os.getenv("TRADE_TP_PCT", "0.0015") or "0.0015")
    base_sl_pct = float(os.getenv("TRADE_SL_PCT", "0.0008") or "0.0008")
    vol_k = float(os.getenv("LABEL_VOL_K", "0.0") or "0.0")

    for i in range(len(spot_df)):
        ts = spot_df.loc[i, "ts"]
        if ts.minute % 5 != 0:
            continue
        if i < 20:
            continue

        px_hist = closes[: i + 1]
        vol_hist = vols[: i + 1]

        window_df = spot_df.iloc[max(0, i - 80): i + 1].copy()

        feat: Dict[str, float] = {}
        feat.update(FeaturePipeline.compute_emas(px_hist, periods=[8, 21, 50]))
        try:
            candle_df_ta = window_df if not window_df.empty else pd.DataFrame({"close": pd.Series(px_hist)})
            feat.update(TA.compute_ta_bundle(candle_df_ta, feat))
        except Exception:
            pass

        try:
            micro = FeaturePipeline.compute_micro_trend(px_hist[-64:])
            feat.update(micro)
        except Exception:
            pass

        try:
            px_arr = np.array(px_hist[-64:], dtype=float)
            if px_arr.size >= 2:
                px_last = float(px_arr[-1])
                if px_arr.size >= 32:
                    px_mean = float(np.mean(px_arr[-32:]))
                    px_std = float(np.std(px_arr[-32:]))
                else:
                    px_mean = float(np.mean(px_arr))
                    px_std = float(np.std(px_arr))
                last_z = (px_last - px_mean) / max(1e-9, px_std)
            else:
                last_z = 0.0
        except Exception:
            last_z = 0.0
        feat["last_zscore"] = float(last_z)

        try:
            feat["rv_10"] = FeaturePipeline.compute_rvol(np.array(vol_hist), window=10)
        except Exception:
            feat["rv_10"] = 0.0

        try:
            feat["atr_1t"] = FeaturePipeline.compute_atr_from_candles(window_df.tail(14))
        except Exception:
            feat["atr_1t"] = 0.0

        feat.update(FeaturePipeline.compute_tod_features(ts))

        try:
            struct_feats = FeaturePipeline.compute_structure_features(window_df)
            feat.update(struct_feats)
        except Exception:
            pass

        try:
            pat_feats = FeaturePipeline.compute_pattern_features(window_df)
            feat.update(pat_feats)
        except Exception:
            pass

        try:
            wick_up, wick_down = FeaturePipeline.compute_wick_extremes(window_df.iloc[-1])
        except Exception:
            wick_up, wick_down = 0.0, 0.0
        feat["wick_extreme_up"] = float(wick_up)
        feat["wick_extreme_down"] = float(wick_down)

        fut_row = None
        if fut_df is not None:
            j = fut_df["ts"].searchsorted(ts, side="right") - 1
            if j >= 0:
                fut_row = fut_df.iloc[j]

        vwap_val = None
        cvd_delta = None
        if fut_row is not None:
            try:
                vwap_val = float(fut_row["fut_session_vwap"])
            except Exception:
                vwap_val = None
            try:
                cvd_delta = float(fut_row["fut_cvd_delta"])
            except Exception:
                cvd_delta = None

        try:
            vwap_rev = FeaturePipeline._compute_vwap_reversion_flag(window_df, vwap_val)
        except Exception:
            vwap_rev = 0.0
        feat["vwap_reversion_flag"] = float(vwap_rev)

        try:
            price_change = float(window_df["close"].iloc[-1] - window_df["close"].iloc[-2])
            cvd_div = FeaturePipeline._compute_cvd_divergence(price_change, cvd_delta)
        except Exception:
            cvd_div = 0.0
        feat["cvd_divergence"] = float(cvd_div)

        feat_norm = FeaturePipeline.normalize_features(feat)

        label = compute_tp_sl_direction_label(
            spot_df,
            i,
            horizon_bars=max(1, int(horizon_min)),
            base_tp_pct=base_tp_pct,
            base_sl_pct=base_sl_pct,
            rv10=feat_norm.get("rv_10"),
            atr1=feat_norm.get("atr_1t"),
            vol_k=vol_k,
        )
        if label is None:
            continue

        rows.append(
            {
                "ts": ts,
                "label": label,
                "tradeable": True,
                "is_flat": label == "FLAT",
                **feat_norm,
            }
        )

    if not rows:
        logger.error("[OFFLINE] No training rows built from cache.")
        return None

    df_features = pd.DataFrame(rows)
    df_features = df_features.dropna(subset=["ts"]).sort_values("ts").reset_index(drop=True)
    logger.info(
        "[OFFLINE] Built features from cache: n=%d (5-min decision points)",
        len(df_features),
    )
    return df_features


def _load_all_features_with_cache() -> Optional[pd.DataFrame]:
    """
    Try normal feature logs first; if empty and OFFLINE_USE_1M_CACHE=1,
    build features directly from cached 1m candles + futures sidecar.
    """
    df = _load_all_features()
    if df is not None and not df.empty:
        logger.info(
            "[OFFLINE] Loaded combined features from logs: n=%d (no cache backfill needed)",
            len(df),
        )
        return df

    if not USE_1M_CACHE:
        logger.error(
            "[OFFLINE] No feature logs and OFFLINE_USE_1M_CACHE=0; cannot train."
        )
        return df

    logger.info(
        "[OFFLINE] Feature logs empty; building training set from cached 1m data + futures sidecar."
    )
    df_cache = _build_features_from_cache()
    if df_cache is None or df_cache.empty:
        logger.error("[OFFLINE] Cache-based feature build failed; no data to train on.")
        return None

    return df_cache


def backfill_from_cache_if_needed() -> Optional[pd.DataFrame]:
    """
    Builds feature rows from cached spot + futures data ONLY if live feature logs are empty.
    Prevents recomputing old days using a progress file.
    """
    if os.path.exists(FEATURE_LOG_HIST_PATH) and os.path.getsize(FEATURE_LOG_HIST_PATH) > 0:
        logger.info("[BACKFILL] feature_log_hist.csv already populated, skipping cache backfill.")
        return None

    logger.info("[BACKFILL] Starting offline backfill from cached 1m data...")

    processed = _load_cache_state()

    fut_df = _load_fut_sidecar(FUT_CANDLES_PATH)
    if fut_df is None or fut_df.empty:
        logger.warning("[BACKFILL] Futures features missing, fut_* will be zeros.")

    all_files = sorted(glob.glob(CACHE_GLOB))
    if not all_files:
        logger.error("[BACKFILL] No cached INDEX files matching glob: %s", CACHE_GLOB)
        return None

    rows: List[Dict[str, object]] = []

    for file in all_files:
        fname = os.path.basename(file)
        if fname in processed:
            logger.info("[BACKFILL] Skipping already processed file: %s", fname)
            continue

        logger.info("[BACKFILL] Processing spot file: %s", fname)

        df1 = pd.read_csv(file)
        df1["ts"] = pd.to_datetime(df1["timestamp"])
        df1 = df1.sort_values("ts")

        closes = df1["close"].tolist()

        for i in range(len(df1)):
            ts = df1["ts"].iloc[i]
            if ts.minute % 5 != 0:
                continue
            if i < 20:
                continue

            window = df1.iloc[max(0, i - 80): i + 1].copy()

            feat = FeaturePipeline.compute_feature_bundle(
                closes=closes[: i + 1],
                window_df=window,
            )

            fut_row = None
            if fut_df is not None:
                j = fut_df["ts"].searchsorted(ts, side="right") - 1
                if j >= 0:
                    fut_row = fut_df.iloc[j]

            feat["vwap_reversion_flag"] = FeaturePipeline._compute_vwap_reversion_flag(
                window, fut_row["fut_session_vwap"] if fut_row is not None else None
            )

            feat["cvd_divergence"] = FeaturePipeline._compute_cvd_divergence(
                float(window["close"].iloc[-1] - window["close"].iloc[-2]),
                fut_row["fut_cvd_delta"] if fut_row is not None else None
            )

            fwd = df1[(df1["ts"] > ts) & (df1["ts"] <= ts + pd.Timedelta(minutes=10))]
            dir_label = compute_tp_sl_direction_label(
                df=fwd,
                idx=0,
                horizon_bars=len(fwd),
                base_tp_pct=0.0015,
                base_sl_pct=0.0008,
                rv10=feat.get("rv_10", 0.0),
                atr1=feat.get("atr_1t", 0.0),
                vol_k=0.0,
            )

            if dir_label is None:
                continue

            row: Dict[str, object] = {"ts": ts, "label": dir_label}
            row.update(feat)
            rows.append(row)

        processed.add(fname)
        _save_cache_state(processed)

    if not rows:
        logger.error("[BACKFILL] No rows created from cache!")
        return None

    df = pd.DataFrame(rows).sort_values("ts")
    df.to_csv(FEATURE_LOG_HIST_PATH, index=False)

    logger.info("[BACKFILL] Backfill complete, wrote %d rows", len(df))
    return df


# ---------------------------------------------------------------------
# 2. Build datasets (mirrors online_trainer._build_datasets)
# ---------------------------------------------------------------------


def _select_feature_columns(df: pd.DataFrame) -> List[str]:
    """
    Choose feature columns:
      - Drop non-numeric and meta/p_xgb_/aux_ret*/aux_label*.
      - If all PREFERRED_FEATURES exist, restrict to them (minimal set).
      - Otherwise, use full numeric set (online/offline parity).

    NOTE:
      * aux_ret_main / aux_ret_short / aux_label_short are explicitly
        excluded because they are forward-looking diagnostics derived
        from future returns and must NEVER be used as model inputs.
    """
    # Meta / non-feature columns
    exclude = {
        "ts",
        "decision",
        "label",
        "buy_prob",
        "alpha",
        "tradeable",
        "is_flat",
        # Forward-looking diagnostics (must not be used as features)
        "aux_ret_main",
        "aux_ret_short",
        "aux_label_short",
    }
    drop_prefixes = ("meta_", "p_xgb_")

    numeric_cols: List[str] = []
    for c in df.columns:
        if c in exclude:
            continue
        if any(c.startswith(p) for p in drop_prefixes):
            continue
        if df[c].dtype == "O":
            continue
        numeric_cols.append(c)

    # Log what we dropped for transparency
    try:
        dropped = [c for c in df.columns if c in exclude or any(c.startswith(p) for p in drop_prefixes)]
        logger.info(
            "[OFFLINE] Column filter: kept=%d numeric feature cols, dropped=%d meta/aux cols",
            len(numeric_cols),
            len(dropped),
        )
        if any(c.startswith("aux_ret_") or c == "aux_label_short" for c in dropped):
            logger.info(
                "[OFFLINE] aux_ret_* / aux_label_short detected and excluded from features "
                "(used only for diagnostics, not training)."
            )
    except Exception:
        # logging failure should not break training
        pass

    preferred_available = [c for c in PREFERRED_FEATURES if c in numeric_cols]
    if preferred_available and len(preferred_available) >= 10:
        feat_cols = sorted(preferred_available)
        logger.info(
            "[OFFLINE] Using preferred minimal feature set: n=%d", len(feat_cols)
        )
    else:
        feat_cols = sorted(numeric_cols)
        logger.info(
            "[OFFLINE] Using full numeric feature set: n=%d (no minimal subset)",
            len(feat_cols),
        )

    return feat_cols


def _build_datasets(
    df: pd.DataFrame,
) -> Tuple[
    Optional[Tuple[np.ndarray, np.ndarray]],
    Optional[Tuple[np.ndarray, np.ndarray]],
    List[str],
]:
    """
    Build (X_dir, y_dir) and (X_neu, y_neu) + feature column list.
    """
    try:
        if "label" not in df.columns:
            logger.error("[OFFLINE] Missing label column in feature logs.")
            return None, None, []

        # High-visibility label distribution for sanity
        try:
            total_rows = int(df.shape[0])
            n_buy = int((df["label"] == "BUY").sum())
            n_sell = int((df["label"] == "SELL").sum())
            n_flat = int((df["label"] == "FLAT").sum())

            logger.info(
                "[OFFLINE] Label distribution: total=%d | BUY=%d | SELL=%d | FLAT=%d | dir_share=%.3f",
                total_rows,
                n_buy,
                n_sell,
                n_flat,
                (n_buy + n_sell) / total_rows if total_rows > 0 else 0.0,
            )
        except Exception as e:
            logger.debug(f"[OFFLINE] Label distribution logging failed (ignored): {e}")

        y_neu = (df["label"] == "FLAT").astype(int).values

        feat_cols = _select_feature_columns(df)

        X_all = (
            df[feat_cols]
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .values
        )
        neu_ds: Optional[Tuple[np.ndarray, np.ndarray]] = (X_all, y_neu)

        mask_dir = df["label"].isin(["BUY", "SELL"])
        df_dir = df.loc[mask_dir].copy()
        if df_dir.empty:
            logger.error("[OFFLINE] No BUY/SELL rows for directional training.")
            dir_ds = None
        else:
            y_dir = (df_dir["label"] == "BUY").astype(int).values
            X_dir = (
                df_dir[feat_cols]
                .replace([np.inf, -np.inf], np.nan)
                .fillna(0.0)
                .values
            )
            dir_ds = (X_dir, y_dir)

        try:
            neu_pos = float(np.mean(y_neu)) if len(y_neu) else 0.0
            dir_size = int(df_dir.shape[0]) if not df_dir.empty else 0
            logger.info(
                "[OFFLINE] Dataset sizes: dir=%d rows | neu=%d rows | neu_pos=%.3f",
                dir_size,
                len(df),
                neu_pos,
            )
        except Exception:
            pass

        return dir_ds, neu_ds, feat_cols

    except Exception as exc:
        logger.error("[OFFLINE] Dataset build failed: %s", exc, exc_info=True)
        return None, None, []


# ---------------------------------------------------------------------
# 3. Train models
# ---------------------------------------------------------------------


def _train_xgb(X: np.ndarray, y: np.ndarray) -> xgb.Booster:
    """
    Train a directional BUY vs SELL XGBoost classifier.

    Slight L2 regularization and small tree depth, tuned for stability.
    """
    dtrain = xgb.DMatrix(X, label=y)
    pos_rate = float(np.mean(y)) if len(y) else 0.5
    scale_pos_weight = float((1.0 - pos_rate) / max(pos_rate, 1e-6))

    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "max_depth": int(os.getenv("XGB_MAX_DEPTH", "4") or "4"),
        "eta": float(os.getenv("XGB_ETA", "0.05") or "0.05"),
        "subsample": float(os.getenv("XGB_SUBSAMPLE", "0.8") or "0.8"),
        "colsample_bytree": float(os.getenv("XGB_COLSAMPLE", "0.8") or "0.8"),
        "lambda": float(os.getenv("XGB_L2", "1.0") or "1.0"),
        "min_child_weight": float(os.getenv("XGB_MIN_CHILD", "5.0") or "5.0"),
        "max_delta_step": 0,
        "tree_method": os.getenv("XGB_TREE_METHOD", "hist"),
        "scale_pos_weight": scale_pos_weight,
    }

    num_boost_round = int(os.getenv("XGB_NUM_ROUNDS", "200") or "200")
    logger.info(
        "[OFFLINE] Training 5-min XGB: n=%d, pos_rate=%.3f, scale_pos_weight=%.3f, rounds=%d",
        len(y),
        pos_rate,
        scale_pos_weight,
        num_boost_round,
    )
    booster = xgb.train(params, dtrain, num_boost_round=num_boost_round)
    return booster


def _train_neutrality(X: np.ndarray, y: np.ndarray) -> Pipeline:
    """
    Train FLAT vs non-FLAT logistic regression (for logging / diagnostics).
    """
    logger.info(
        "[OFFLINE] Training 5-min neutrality model: n=%d, pos_rate=%.3f",
        len(y),
        float(np.mean(y)) if len(y) else 0.0,
    )
    pipe = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=800, class_weight="balanced")),
        ]
    )
    pipe.fit(X, y)
    return pipe


# ---------------------------------------------------------------------
# 4. Save artefacts
# ---------------------------------------------------------------------


def _save_feature_schema(feat_cols: List[str]) -> None:
    schema = {"feature_cols": feat_cols}
    SCHEMA_OUT_PATH.write_text(json.dumps(schema, indent=2))
    logger.info("[OFFLINE] Saved feature schema: %s", SCHEMA_OUT_PATH)


def _save_feature_stats(df: pd.DataFrame, feat_cols: List[str]) -> None:
    stats: Dict[str, Dict[str, float]] = {}
    for c in feat_cols:
        try:
            arr = (
                df[c]
                .replace([np.inf, -np.inf], np.nan)
                .dropna()
                .astype(float)
                .values
            )
            if arr.size == 0:
                continue
            stats[c] = {
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
            }
        except Exception:
            continue
    STATS_OUT_PATH.write_text(json.dumps(stats, indent=2))
    logger.info("[OFFLINE] Saved feature stats: %s", STATS_OUT_PATH)


def _ensure_prod_dir() -> None:
    PROD_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# 5. Main
# ---------------------------------------------------------------------


def main() -> None:
    _ensure_prod_dir()

    df = _load_all_features()
    if (df is None or df.empty) and USE_1M_CACHE:
        df = backfill_from_cache_if_needed()
    if df is None or df.empty:
        logger.error("[OFFLINE] No data to train on.")
        return

    dir_ds, neu_ds, feat_cols = _build_datasets(df)
    if dir_ds is None or neu_ds is None:
        logger.error("[OFFLINE] Dataset build failed; aborting.")
        return

    X_dir, y_dir = dir_ds
    X_neu, y_neu = neu_ds

    if X_dir.size == 0 or X_neu.size == 0:
        logger.error("[OFFLINE] Empty X_dir or X_neu; aborting.")
        return

    booster = _train_xgb(X_dir, y_dir)

    # Embed schema on booster so AdaptiveModelPipeline can enforce it
    try:
        schema_attr = {"feature_names": list(feat_cols)}
        booster.set_attr(feature_schema=json.dumps(schema_attr))
        logger.info(
            "[OFFLINE] Embedded feature_schema into 5-min XGB booster (n=%d)",
            len(feat_cols),
        )
    except Exception as exc:
        logger.warning(
            "[OFFLINE] Failed to set 5-min booster feature_schema attr: %s", exc
        )

    booster.save_model(str(XGB_OUT_PATH))
    logger.info("[OFFLINE] Saved directional XGB model to %s", XGB_OUT_PATH)

    neutrality_model = _train_neutrality(X_neu, y_neu)
    joblib.dump(neutrality_model, NEUTRAL_OUT_PATH)
    logger.info("[OFFLINE] Saved neutrality model to %s", NEUTRAL_OUT_PATH)

    _save_feature_schema(feat_cols)
    _save_feature_stats(df, feat_cols)

    logger.info("[OFFLINE] 5-minute offline training complete.")


if __name__ == "__main__":
    main()
