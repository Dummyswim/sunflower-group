#!/usr/bin/env python3
"""
bootstrap_trainlog_from_cache_v3.py

Bootstraps SignalContext training records from cached OHLCV.

Fixes schema loader to support multiple schema file formats:
- {"feature_names":[...]}            (your feature_schema_v*.json)
- {"columns":[...]}                  (common cols file)
- {"feature_schema_cols":[...]}      (legacy)
- {"features": {"name": idx, ...}}   (index mapping)
- {"name": {"dtype":"float"}, ...}   (type mapping)
- plain list ["f1","f2",...]

Everything else is identical to v2.

Tip: If you want the exact live base schema, point --schema to your newest
feature_schema_v*.json, e.g. feature_schema_v20251210_093838.json.
"""

from __future__ import annotations
import argparse
import glob
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from signal_context import build_signal_context
from signal_log_utils import append_jsonl
from rule_engine import (
    compute_flow_signal,
    compute_rule_hierarchy,
    compute_structure_score,
    compute_ta_rule_signal,
    DecisionState,
    decide_trade,
)
from feature_pipeline import FeaturePipeline, TA

def _resample_ohlcv(df1: pd.DataFrame, bar_min: int) -> pd.DataFrame:
    rule = f"{bar_min}min"
    out = pd.DataFrame({
        "open":  df1["open"].resample(rule).first(),
        "high":  df1["high"].resample(rule).max(),
        "low":   df1["low"].resample(rule).min(),
        "close": df1["close"].resample(rule).last(),
        "volume": df1["volume"].resample(rule).sum() if "volume" in df1.columns else 0.0,
    }).dropna(subset=["open","high","low","close"])
    return out


def _ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()


def _rsi14(close: pd.Series) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0.0)
    dn = (-d).clip(lower=0.0)
    rs = up.rolling(14, min_periods=14).mean() / dn.rolling(14, min_periods=14).mean().replace(0.0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def _macd_hist(close: pd.Series) -> pd.Series:
    macd, signal, hist = _macd(close)
    return hist


def _macd(close: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
    ema12 = _ema(close, 12)
    ema26 = _ema(close, 26)
    macd = ema12 - ema26
    signal = _ema(macd, 9)
    hist = macd - signal
    return macd, signal, hist


def _indicator_score(ind_feats: Dict[str, float]) -> float:
    weights = {
        "ema_trend": 0.35,
        "micro_slope": 0.25,
        "imbalance": 0.20,
        "mean_drift": 0.20,
    }
    score = 0.0
    for k, w in weights.items():
        try:
            v = float(ind_feats.get(k, 0.0))
        except Exception:
            v = 0.0
        if not np.isfinite(v):
            v = 0.0
        v = float(np.clip(v, -1.0, 1.0))
        score += w * v
    return float(np.clip(score, -1.0, 1.0))


def _bb(close: pd.Series, n: int = 20, k: float = 2.0) -> Tuple[pd.Series,pd.Series,pd.Series]:
    ma = close.rolling(n, min_periods=n).mean()
    sd = close.rolling(n, min_periods=n).std(ddof=0)
    upper = ma + k * sd
    lower = ma - k * sd
    return lower, ma, upper


def _atr14(df: pd.DataFrame) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    pc = c.shift(1)
    tr = pd.concat([(h - l).abs(), (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.rolling(14, min_periods=14).mean()


def _pivot_swipe(df: pd.DataFrame, lookback: int = 10) -> Tuple[pd.Series,pd.Series]:
    prev_high = df["high"].shift(1).rolling(lookback, min_periods=lookback).max()
    prev_low  = df["low"].shift(1).rolling(lookback, min_periods=lookback).min()
    swipe_up = (df["high"] > prev_high) & (df["close"] < prev_high)
    swipe_dn = (df["low"]  < prev_low)  & (df["close"] > prev_low)
    return swipe_up.astype(float), swipe_dn.astype(float)


def _fvg_flags(df: pd.DataFrame) -> Tuple[pd.Series,pd.Series]:
    h2 = df["high"].shift(2)
    l2 = df["low"].shift(2)
    up = (df["low"] > h2).astype(float)
    dn = (df["high"] < l2).astype(float)
    return up, dn


def _wick_extremes(df: pd.DataFrame) -> Tuple[pd.Series,pd.Series]:
    body = (df["close"] - df["open"]).abs()
    rng = (df["high"] - df["low"]).replace(0.0, np.nan)
    upper = (df["high"] - df[["open","close"]].max(axis=1)) / rng
    lower = (df[["open","close"]].min(axis=1) - df["low"]) / rng
    wick_up = ((upper > 0.60) & (body / rng < 0.25)).astype(float)
    wick_dn = ((lower > 0.60) & (body / rng < 0.25)).astype(float)
    return wick_up, wick_dn


def _rv10(close: pd.Series) -> pd.Series:
    r = close.pct_change()
    return r.rolling(10, min_periods=10).std(ddof=0)


def _zscore(close: pd.Series, n: int = 50) -> pd.Series:
    m = close.rolling(n, min_periods=n).mean()
    s = close.rolling(n, min_periods=n).std(ddof=0).replace(0.0, np.nan)
    return (close - m) / s


def _label_barrier(df: pd.DataFrame, i: int, horizon_bars: int, tp_atr_mult: float, sl_atr_mult: float) -> Tuple[str, float]:
    if i + horizon_bars >= len(df):
        return ("FLAT", 0.0)
    entry = float(df["close"].iat[i])
    atr = float(df["atr14"].iat[i])
    if entry <= 0 or not np.isfinite(atr) or atr <= 0:
        return ("FLAT", 0.0)
    atr_pct = atr / entry
    tp = tp_atr_mult * atr_pct
    sl = sl_atr_mult * atr_pct
    slc = df.iloc[i+1:i+1+horizon_bars]
    up = (float(slc["high"].max()) - entry) / entry
    dn = (entry - float(slc["low"].min())) / entry
    tgt_close = float(df["close"].iat[i + horizon_bars])
    aux_ret_main = (tgt_close - entry) / entry
    if up >= tp and dn < sl:
        return ("BUY", aux_ret_main)
    if dn >= tp and up < sl:
        return ("SELL", aux_ret_main)
    return ("FLAT", aux_ret_main)


def _load_schema(schema_path: str) -> List[str]:
    p = Path(schema_path)
    if not p.exists():
        raise FileNotFoundError(f"schema not found: {schema_path}")
    data = json.loads(p.read_text(encoding="utf-8"))

    # 1) direct list
    if isinstance(data, list) and all(isinstance(x, str) for x in data):
        return list(data)

    # 2) dict with known keys
    if isinstance(data, dict):
        for key in ("feature_names", "columns", "feature_schema_cols", "feature_cols", "features"):
            if key in data:
                v = data[key]
                # list of strings
                if isinstance(v, list) and all(isinstance(x, str) for x in v):
                    return list(v)
                # dict mapping feature->index
                if isinstance(v, dict) and all(isinstance(k, str) for k in v.keys()):
                    # if values are ints, sort by index
                    if all(isinstance(iv, int) for iv in v.values()):
                        return [k for k, _ in sorted(v.items(), key=lambda kv: kv[1])]
                    # if values are dicts (dtype), just take keys sorted
                    return sorted(list(v.keys()))

        # 3) try any dict value that looks like list of strings
        for k, v in data.items():
            if isinstance(v, list) and v and all(isinstance(x, str) for x in v):
                return list(v)

        # 4) dict mapping feature->something
        if all(isinstance(k, str) for k in data.keys()):
            # mapping to int indices
            if all(isinstance(iv, int) for iv in data.values()):
                return [k for k, _ in sorted(data.items(), key=lambda kv: kv[1])]
            # mapping to dtype dicts/strings
            return sorted(list(data.keys()))

    raise ValueError(f"Unrecognized schema format in {schema_path}")


def build_trainlog_from_cache(
    cache_glob: str,
    out_jsonl: str,
    schema_path: str,
    symbol: str,
    bar_min: int,
    horizon_min: int,
    tp_atr_mult: float,
    sl_atr_mult: float,
    append: bool,
) -> int:
    files = sorted(glob.glob(cache_glob))
    if not files:
        raise SystemExit(f"No files matched: {cache_glob}")

    schema = _load_schema(schema_path)
    decision_state = DecisionState(f"backfill:{symbol}")

    horizon_bars = max(1, int(round(horizon_min / float(bar_min))))
    mode = "a" if append and Path(out_jsonl).exists() else "w"

    out_path = Path(out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    quarantine_path = str(out_path.with_name(out_path.stem + "_quarantine.jsonl"))

    n_written = 0

    with out_path.open(mode, encoding="utf-8") as f_out:
        for fp in files:
            df1 = pd.read_csv(fp)
            if "timestamp" not in df1.columns:
                continue

            df1["timestamp"] = pd.to_datetime(df1["timestamp"], errors="coerce")
            df1 = df1.dropna(subset=["timestamp"]).set_index("timestamp").sort_index()

            for c in ("open","high","low","close","volume"):
                if c in df1.columns:
                    df1[c] = pd.to_numeric(df1[c], errors="coerce")
            df1 = df1.dropna(subset=["open","high","low","close"])

            df = _resample_ohlcv(df1, bar_min=bar_min)
            if len(df) < 60:
                continue

            # OHLCV-derived features (matches your v20251210 schema names)
            df["ema_8"] = _ema(df["close"], 8)
            df["ema_9"] = _ema(df["close"], 9)
            df["ema_15"] = _ema(df["close"], 15)
            df["ema_21"] = _ema(df["close"], 21)
            df["ema_50"] = _ema(df["close"], 50)
            df["ta_rsi14"] = _rsi14(df["close"])
            macd_line, macd_signal, macd_hist = _macd(df["close"])
            df["ta_macd_line"] = macd_line
            df["ta_macd_signal"] = macd_signal
            df["ta_macd_hist"] = macd_hist

            lo, mid, up = _bb(df["close"], 20, 2.0)
            df["ta_bb_bw"] = (up - lo) / mid.replace(0.0, np.nan)
            df["ta_bb_pctb"] = (df["close"] - lo) / (up - lo).replace(0.0, np.nan)
            df["ta_bb_mid"] = mid

            try:
                k, d = TA.stoch_kd(df["high"], df["low"], df["close"])
                df["ta_stoch_k"] = k
                df["ta_stoch_d"] = d
            except Exception:
                df["ta_stoch_k"] = 50.0
                df["ta_stoch_d"] = 50.0

            try:
                df["ta_cci"] = TA.cci(df["high"], df["low"], df["close"])
            except Exception:
                df["ta_cci"] = 0.0
            try:
                df["ta_adx"] = TA.adx(df["high"], df["low"], df["close"])
            except Exception:
                df["ta_adx"] = 0.0
            try:
                vol = df["volume"] if "volume" in df.columns else pd.Series([0.0] * len(df), index=df.index)
                df["ta_mfi"] = TA.mfi(df["high"], df["low"], df["close"], vol)
            except Exception:
                df["ta_mfi"] = 50.0
            try:
                df["ta_mom14"] = TA.momentum(df["close"], window=14)
            except Exception:
                df["ta_mom14"] = 0.0
            try:
                vol = df["volume"] if "volume" in df.columns else pd.Series([0.0] * len(df), index=df.index)
                obv = TA.obv(df["close"], vol)
                obv_mean = obv.rolling(100, min_periods=10).mean()
                obv_std = obv.rolling(100, min_periods=10).std(ddof=0).replace(0.0, np.nan)
                df["ta_obv_z"] = ((obv - obv_mean) / obv_std).fillna(0.0)
            except Exception:
                df["ta_obv_z"] = 0.0

            df["atr14"] = _atr14(df)
            df["atr_1t"] = df["atr14"]
            df["atr_3t"] = (df["high"] - df["low"]).rolling(15, min_periods=1).mean()

            df["rv_10"] = _rv10(df["close"])
            df["last_zscore"] = _zscore(df["close"], 50)
            df["mean_drift_pct"] = df["close"].pct_change().rolling(10, min_periods=10).mean()

            df["struct_pivot_swipe_up"], df["struct_pivot_swipe_down"] = _pivot_swipe(df, lookback=10)
            df["struct_fvg_up_present"], df["struct_fvg_down_present"] = _fvg_flags(df)
            df["wick_extreme_up"], df["wick_extreme_down"] = _wick_extremes(df)

            # time-of-day (match live session encoding)
            idx = pd.DatetimeIndex(df.index)
            mins = (idx.hour * 60 + idx.minute).astype(float)
            mins = np.maximum(0.0, mins - (9 * 60 + 15))
            ang = (2.0 * np.pi * mins) / 375.0
            df["tod_sin"] = np.sin(ang)
            df["tod_cos"] = np.cos(ang)

            # not derivable from pure OHLCV: set neutral (per-row overrides may update)
            neutral_cols = (
                "micro_imbalance", "micro_slope", "imbalance",
                "mean_drift", "std_dltp_short", "price_range_tightness",
                "cvd_divergence", "vwap_reversion_flag",
                "struct_ob_bull_present", "struct_ob_bear_present",
                "struct_ob_bull_dist", "struct_ob_bear_dist",
                "struct_fvg_up_size", "struct_fvg_down_size",
                "struct_pivot_is_swing_high", "struct_pivot_is_swing_low",
                "struct_pivot_dist_from_high", "struct_pivot_dist_from_low",
                "fut_session_vwap", "fut_vwap_dev", "fut_cvd_delta", "fut_vol_delta",
                "flow_score", "flow_side", "flow_fut_cvd", "flow_fut_vwap_dev",
                "flow_vwap_side", "flow_fut_vol",
                "rev_cross_upper_wick_cvd", "rev_cross_upper_wick_vwap",
                "rev_cross_lower_wick_cvd", "rev_cross_lower_wick_vwap",
                "spread",
            )
            df.loc[:, list(neutral_cols)] = 0.0
            df["last_price"] = df["close"]

            # ensure schema keys exist (imputed zeros flagged in meta)
            imputed_cols: List[str] = []
            missing_cols = [k for k in schema if k not in df.columns]
            if missing_cols:
                df = df.assign(**{k: 0.0 for k in missing_cols})
                imputed_cols.extend(missing_cols)

            df = df.dropna(subset=["atr14","ta_rsi14","ema_8","ema_21","ema_50"], how="any")

            for i in range(len(df)):
                if i + horizon_bars >= len(df):
                    continue

                ts_ref = df.index[i]
                ts_tgt = df.index[i + horizon_bars]

                label, aux_ret_main = _label_barrier(df, i, horizon_bars, tp_atr_mult, sl_atr_mult)

                window_df = df.iloc[max(0, i - 240):i + 1].copy()
                if window_df.empty:
                    continue

                try:
                    prices = pd.to_numeric(window_df["close"], errors="coerce").dropna().astype(float).tolist()
                except Exception:
                    prices = []

                try:
                    pattern_features = FeaturePipeline.compute_candlestick_patterns(
                        candles=window_df.tail(8),
                        rvol_window=5,
                        rvol_thresh=1.2,
                        min_winrate=0.55,
                    ) or {}
                except Exception:
                    pattern_features = {}

                try:
                    mtf = FeaturePipeline.compute_mtf_pattern_consensus(
                        candle_df=window_df,
                        timeframes=["1T", "3T", "5T"],
                        rvol_window=5,
                        rvol_thresh=1.2,
                        min_winrate=0.55,
                    ) or {}
                except Exception:
                    mtf = {}

                try:
                    sr = FeaturePipeline.compute_sr_features(
                        candle_df=window_df,
                        timeframes=["1T", "3T", "5T"],
                    ) or {}
                except Exception:
                    sr = {}

                try:
                    structure_feats = FeaturePipeline.compute_structure_bundle(window_df.tail(40))
                except Exception:
                    structure_feats = {}

                try:
                    ema_module_feats, _ = FeaturePipeline.compute_ema_module(
                        decision_df=window_df,
                        filter_df=window_df,
                        decision_tf=f"{bar_min}T",
                        filter_tf=f"{bar_min}T",
                    )
                except Exception:
                    ema_module_feats, ema_meta = {}, {}

                micro = FeaturePipeline.compute_micro_trend(prices[-32:]) if prices else {
                    "micro_slope": 0.0,
                    "micro_imbalance": 0.0,
                    "mean_drift_pct": 0.0,
                    "last_zscore": 0.0,
                }

                ema_fast = float(df["ema_8"].iat[i]) if "ema_8" in df.columns else 0.0
                ema_slow = float(df["ema_21"].iat[i]) if "ema_21" in df.columns else 0.0
                ema_trend = 1.0 if ema_fast > ema_slow else (-1.0 if ema_fast < ema_slow else 0.0)

                indicator_features = {
                    "ema_trend": float(ema_trend),
                    "micro_slope": float(micro.get("micro_slope", 0.0)),
                    "imbalance": float(micro.get("micro_imbalance", 0.0)),
                    "mean_drift": float(micro.get("mean_drift_pct", 0.0)),
                }
                indicator_score = _indicator_score(indicator_features)

                feats: Dict[str, float] = {}
                for k in schema:
                    try:
                        v = df[k].iat[i]
                    except Exception:
                        v = 0.0
                    try:
                        v = float(v)
                        if not np.isfinite(v):
                            v = 0.0
                    except Exception:
                        v = 0.0
                    feats[k] = v
                schema_version = os.getenv("SCHEMA_VERSION", "schema_v4")
                label_version = os.getenv("LABEL_VERSION", "label_v4")
                pipeline_version = os.getenv("PIPELINE_VERSION", "pipeline_v4")
                label_weight = 1.0 if label != "FLAT" else 0.7

                features_raw = dict(feats)
                features_raw.update(pattern_features)
                features_raw.update(mtf)
                features_raw.update(sr)
                features_raw.update(structure_feats)
                features_raw.update(ema_module_feats)
                features_raw.setdefault("ema_bias_5t", float(ema_trend))
                features_raw.setdefault("ema_regime_chop_5t", 0.0)
                features_raw.setdefault("ema_entry_tag", 0.0)
                features_raw.setdefault("ema_entry_side", 0.0)
                features_raw.setdefault("ema15_break_veto", 0.0)

                mean_drift_pct = float(micro.get("mean_drift_pct", 0.0))
                mean_drift = float(mean_drift_pct)
                mean_drift_pct = mean_drift_pct * 100.0

                features_raw["micro_slope"] = float(micro.get("micro_slope", 0.0))
                features_raw["micro_imbalance"] = float(micro.get("micro_imbalance", 0.0))
                features_raw["imbalance"] = float(micro.get("micro_imbalance", 0.0))
                features_raw["mean_drift"] = float(mean_drift)
                features_raw["mean_drift_pct"] = float(mean_drift_pct)
                features_raw["last_zscore"] = float(micro.get("last_zscore", features_raw.get("last_zscore", 0.0)))
                features_raw["last_price"] = float(df["close"].iat[i])
                features_raw["ema_trend"] = float(ema_trend)
                features_raw["indicator_score"] = float(indicator_score)

                try:
                    px_tail = np.asarray(window_df["close"].tail(12), dtype=float)
                    if px_tail.size >= 2:
                        features_raw["std_dltp_short"] = float(np.std(np.diff(px_tail)))
                    else:
                        features_raw["std_dltp_short"] = 0.0
                except Exception:
                    features_raw["std_dltp_short"] = 0.0

                try:
                    rng_recent = window_df["high"].tail(5).max() - window_df["low"].tail(5).min()
                    rng_base = (window_df["high"] - window_df["low"]).tail(20).mean()
                    denom = float(rng_base) if np.isfinite(rng_base) and rng_base > 0 else 1.0
                    tight = 1.0 - min(1.0, float(rng_recent) / denom)
                    features_raw["price_range_tightness"] = float(np.clip(tight, 0.0, 1.0))
                except Exception:
                    features_raw["price_range_tightness"] = 0.0

                try:
                    last_candle = window_df.iloc[-1]
                    wick_up, wick_down = FeaturePipeline._compute_wick_extremes(last_candle)
                except Exception:
                    wick_up, wick_down = 0.0, 0.0
                features_raw["wick_extreme_up"] = float(wick_up)
                features_raw["wick_extreme_down"] = float(wick_down)
                features_raw["vwap_reversion_flag"] = float(features_raw.get("vwap_reversion_flag", 0.0))
                features_raw["cvd_divergence"] = float(features_raw.get("cvd_divergence", 0.0))

                rev_cross_feats = FeaturePipeline.compute_reversal_cross_features(
                    window_df.tail(20),
                    {
                        "wick_extreme_up": float(wick_up),
                        "wick_extreme_down": float(wick_down),
                        "cvd_divergence": float(features_raw.get("cvd_divergence", 0.0)),
                        "vwap_reversion_flag": float(features_raw.get("vwap_reversion_flag", 0.0)),
                    },
                )
                features_raw.update(rev_cross_feats)

                try:
                    features_raw["open"] = float(df["open"].iat[i])
                    features_raw["high"] = float(df["high"].iat[i])
                    features_raw["low"] = float(df["low"].iat[i])
                    features_raw["close"] = float(df["close"].iat[i])
                except Exception:
                    pass

                for k in ("fut_session_vwap", "fut_vwap_dev", "fut_cvd_delta", "fut_vol_delta"):
                    features_raw.setdefault(k, 0.0)

                ta_rule = compute_ta_rule_signal(features_raw)
                rule_weight_ind = float(os.getenv("RULE_WEIGHT_IND", "0.50"))
                rule_weight_mtf = float(os.getenv("RULE_WEIGHT_MTF", "0.35"))
                rule_weight_pat = float(os.getenv("RULE_WEIGHT_PAT", "0.15"))
                rule_weight_ta = float(os.getenv("RULE_WEIGHT_TA", "0.00"))
                mtf_cons = float(mtf.get("mtf_consensus", 0.0)) if mtf else 0.0
                pat_adj = float(pattern_features.get("probability_adjustment", 0.0)) if pattern_features else 0.0
                rule_sig = (rule_weight_ind * indicator_score) + (rule_weight_mtf * mtf_cons) + (rule_weight_pat * pat_adj) + (rule_weight_ta * ta_rule)
                rule_sig = float(np.clip(rule_sig, -1.0, 1.0))

                flow_score, flow_side, _, _, _, vwap_side = compute_flow_signal(features_raw)
                features_raw.pop("flow_regime", None)
                struct_score, struct_side = compute_structure_score(features_raw)
                features_raw["struct_side"] = float(struct_side)
                features_raw["struct_score"] = float(struct_score)


                is_bull_setup = bool(
                    float(features_raw.get("struct_pivot_swipe_up", 0.0)) > 0.0
                    or float(features_raw.get("struct_fvg_up_present", 0.0)) > 0.0
                    or float(features_raw.get("struct_ob_bull_present", 0.0)) > 0.0
                )
                is_bear_setup = bool(
                    float(features_raw.get("struct_pivot_swipe_down", 0.0)) > 0.0
                    or float(features_raw.get("struct_fvg_down_present", 0.0)) > 0.0
                    or float(features_raw.get("struct_ob_bear_present", 0.0)) > 0.0
                )
                any_setup = is_bull_setup or is_bear_setup
                ambiguous_setup = is_bull_setup and is_bear_setup
                fast_setup_ready = bool(
                    (is_bull_setup and flow_side > 0) or (is_bear_setup and flow_side < 0)
                )
                features_raw["is_bull_setup"] = int(is_bull_setup)
                features_raw["is_bear_setup"] = int(is_bear_setup)
                features_raw["is_any_setup"] = int(any_setup)
                features_raw["fast_setup_ready"] = int(fast_setup_ready)

                feats = {}
                for k in schema:
                    try:
                        v = float(features_raw.get(k, 0.0))
                        if not np.isfinite(v):
                            v = 0.0
                    except Exception:
                        v = 0.0
                    feats[k] = v

                rule_dir, base_dir, conflict_level, conflict_reasons = compute_rule_hierarchy(
                    name=symbol,
                    rule_sig=rule_sig,
                    features_raw=features_raw,
                    mtf=mtf,
                    is_bull_setup=is_bull_setup,
                    is_bear_setup=is_bear_setup,
                    any_setup=any_setup,
                    ambiguous_setup=ambiguous_setup,
                )

                teacher_strength = float(abs(rule_sig))
                decision = decide_trade(
                    state=decision_state,
                    cfg=None,
                    features_raw=features_raw,
                    mtf=mtf,
                    rule_dir=rule_dir,
                    conflict_level=conflict_level,
                    conflict_reasons=conflict_reasons,
                    teacher_strength=teacher_strength,
                    is_bull_setup=is_bull_setup,
                    is_bear_setup=is_bear_setup,
                    safe_df=None,
                )

                teacher_dir = str(decision.get("intent", rule_dir)).upper()
                if teacher_dir not in ("BUY", "SELL"):
                    teacher_dir = "FLAT"

                gates = {
                    "lane": decision.get("lane"),
                    "tape_conflict_level": conflict_level,
                    "tape_conflict_reasons": list(conflict_reasons or []),
                    "gate_reasons": list(decision.get("gate_reasons", []) or []),
                }

                rule_signals = {
                    "rule_sig": float(rule_sig),
                    "flow_side": float(flow_side),
                    "flow_score": float(flow_score),
                    "vwap_side": float(vwap_side),
                    "struct_side": float(struct_side),
                    "struct_score": float(struct_score),
                    "trend_side": float(features_raw.get("ema_bias_5t", 0.0)),
                    "mtf_consensus": float(mtf_cons),
                    "indicator_score": float(indicator_score),
                    "pattern_adj": float(pat_adj),
                    "ta_rule": float(ta_rule),
                }

                provenance = {
                    "record_source": "backfill",
                    "feature_imputed": imputed_cols,
                    "aux_ret_main": float(aux_ret_main),
                    "scored": False,
                    "tick_count": 0,
                }

                record, errors = build_signal_context(
                    schema_cols=schema,
                    schema_version=schema_version,
                    label_version=label_version,
                    pipeline_version=pipeline_version,
                    symbol=symbol,
                    bar_min=bar_min,
                    horizon_min=horizon_min,
                    ts_ref_start=ts_ref,
                    ts_target_close=ts_tgt,
                    label=label,
                    label_source="cache_bootstrap",
                    label_weight=label_weight,
                    features=feats,
                    rule_signals=rule_signals,
                    gates=gates,
                    teacher_dir=teacher_dir,
                    teacher_tradeable=bool(decision.get("tradeable", False)),
                    teacher_strength=teacher_strength,
                    provenance=provenance,
                    model={},
                )

                if record is None or errors:
                    qrec = {
                        "record_version": "sc_v1",
                        "errors": errors,
                        "record": {
                            "symbol": symbol,
                            "ts_target_close": str(ts_tgt),
                            "label": label,
                            "features": feats,
                            "provenance": provenance,
                        },
                    }
                    append_jsonl(quarantine_path, qrec)
                    continue

                f_out.write(json.dumps(record, separators=(",", ":"), ensure_ascii=False) + "\n")
                n_written += 1

    return n_written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_glob", default=os.getenv("OFFLINE_SPOT_CACHE_GLOB", ""), help="Glob like data/intraday_cache/INDEX_*_1m.csv")
    ap.add_argument("--out_jsonl", default=os.getenv("TRAIN_LOG_PATH", ""), help="Output SignalContext JSONL")
    ap.add_argument("--schema", default=os.getenv("FEATURE_SCHEMA_COLS_PATH", ""), help="Schema path")
    ap.add_argument("--symbol", default=os.getenv("TRAIN_SYMBOL","INDEX"))
    ap.add_argument("--bar_min", type=int, default=int(os.getenv("BAR_MIN", "5")))
    ap.add_argument("--horizon_min", type=int, default=int(os.getenv("HORIZON_MIN", "10")))
    ap.add_argument("--tp_atr_mult", type=float, default=float(os.getenv("TP_ATR_MULT", "0.90")))
    ap.add_argument("--sl_atr_mult", type=float, default=float(os.getenv("SL_ATR_MULT", "0.60")))
    ap.add_argument("--append", action="store_true")
    args = ap.parse_args()

    if not args.cache_glob:
        raise SystemExit("cache_glob missing (set --cache_glob or OFFLINE_SPOT_CACHE_GLOB)")
    if not args.out_jsonl:
        raise SystemExit("out_jsonl missing (set --out_jsonl or TRAIN_LOG_PATH)")
    if not args.schema:
        raise SystemExit("schema missing (set --schema or FEATURE_SCHEMA_COLS_PATH)")

    n = build_trainlog_from_cache(
        cache_glob=args.cache_glob,
        out_jsonl=args.out_jsonl,
        schema_path=args.schema,
        symbol=args.symbol,
        bar_min=args.bar_min,
        horizon_min=args.horizon_min,
        tp_atr_mult=args.tp_atr_mult,
        sl_atr_mult=args.sl_atr_mult,
        append=args.append,
    )
    print("wrote_records:", n)
    print("out:", args.out_jsonl)


if __name__ == "__main__":
    main()
