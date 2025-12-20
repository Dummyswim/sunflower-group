#!/usr/bin/env python3
"""
bootstrap_trainlog_from_cache_v3.py

Fixes schema loader to support multiple schema file formats:
- {"feature_names":[...]}            (your feature_schema_v*.json)
- {"columns":[...]}                  (common cols file)
- {"feature_schema_cols":[...]}      (legacy)
- {"features": {"name": idx, ...}}   (index mapping)
- {"name": {"dtype":"float"}, ...}   (type mapping)
- plain list ["f1","f2",...]

Everything else is identical to v2.

Tip: If you want the exact live schema, point --schema to your newest
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

from train_record_v3 import build_train_record_v3
from train_log_utils_v3 import append_jsonl

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
    ema12 = _ema(close, 12)
    ema26 = _ema(close, 26)
    macd = ema12 - ema26
    signal = _ema(macd, 9)
    return macd - signal


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
            df["ema_21"] = _ema(df["close"], 21)
            df["ema_50"] = _ema(df["close"], 50)
            df["ta_rsi14"] = _rsi14(df["close"])
            df["ta_macd_hist"] = _macd_hist(df["close"])

            lo, mid, up = _bb(df["close"], 20, 2.0)
            df["ta_bb_bw"] = (up - lo) / mid.replace(0.0, np.nan)
            df["ta_bb_pctb"] = (df["close"] - lo) / (up - lo).replace(0.0, np.nan)

            df["atr14"] = _atr14(df)
            df["atr_1t"] = df["atr14"]

            df["rv_10"] = _rv10(df["close"])
            df["last_zscore"] = _zscore(df["close"], 50)
            df["mean_drift_pct"] = df["close"].pct_change().rolling(10, min_periods=10).mean()

            df["struct_pivot_swipe_up"], df["struct_pivot_swipe_down"] = _pivot_swipe(df, lookback=10)
            df["struct_fvg_up_present"], df["struct_fvg_down_present"] = _fvg_flags(df)
            df["wick_extreme_up"], df["wick_extreme_down"] = _wick_extremes(df)

            # time-of-day
            mins = (df.index.hour * 60 + df.index.minute).astype(float)
            ang = (2.0 * np.pi * mins) / (24.0 * 60.0)
            df["tod_sin"] = np.sin(ang)
            df["tod_cos"] = np.cos(ang)

            # not derivable from pure OHLCV: set neutral
            for k in ("micro_imbalance","micro_slope","cvd_divergence","vwap_reversion_flag",
                      "struct_ob_bull_present","struct_ob_bear_present"):
                df[k] = 0.0
            df["last_price"] = df["close"]

            # ensure schema keys exist (imputed zeros flagged in meta)
            imputed_cols: List[str] = []
            for k in schema:
                if k not in df.columns:
                    df[k] = 0.0
                    imputed_cols.append(k)

            df = df.dropna(subset=["atr14","ta_rsi14","ema_8","ema_21","ema_50"], how="any")

            for i in range(len(df)):
                if i + horizon_bars >= len(df):
                    continue

                ts_ref = df.index[i]
                ts_tgt = df.index[i + horizon_bars]

                label, aux_ret_main = _label_barrier(df, i, horizon_bars, tp_atr_mult, sl_atr_mult)

                feats: Dict[str, float] = {}
                for k in schema:
                    v = df[k].iat[i]
                    try:
                        v = float(v)
                        if not np.isfinite(v):
                            v = 0.0
                    except Exception:
                        v = 0.0
                    feats[k] = v
                schema_version = os.getenv("SCHEMA_VERSION", "schema_v3")
                label_version = os.getenv("LABEL_VERSION", "label_v3")
                pipeline_version = os.getenv("PIPELINE_VERSION", "pipeline_v3")

                meta = {
                    "record_source": "backfill",
                    "feature_imputed": imputed_cols,
                    "aux_ret_main": float(aux_ret_main),
                }

                record, errors = build_train_record_v3(
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
                    label_weight=1.0 if label != "FLAT" else 0.7,
                    buy_prob=0.5,
                    alpha=0.0,
                    tradeable=True,
                    is_flat=(label == "FLAT"),
                    tick_count=0,
                    features=feats,
                    meta=meta,
                )

                if record is None or errors:
                    qrec = {
                        "record_version": "v3",
                        "errors": errors,
                        "record": {
                            "symbol": symbol,
                            "ts_target_close": str(ts_tgt),
                            "label": label,
                            "features": feats,
                            "meta": meta,
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
    ap.add_argument("--out_jsonl", default=os.getenv("TRAIN_LOG_PATH", ""), help="Output train_log_v2.jsonl")
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
