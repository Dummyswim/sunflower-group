#!/usr/bin/env python3
"""
tune_atr_multipliers.py

Grid-search TP_ATR_MULT / SL_ATR_MULT on your cached 1-minute OHLCV files to
target a healthier label mix (less unwanted FLAT) without hallucinating direction.

Scalper intent:
- Label BUY/SELL only when price *actually* moved enough (volatility-scaled).
- Keep FLAT as "didn't pay after risk", not "I got scared".

Usage:
  python tune_atr_multipliers.py \
    --cache_glob "data/intraday_cache/INDEX_*_1m.csv" \
    --bar_min 5 \
    --horizon_min 10 \
    --tp_grid 0.50 0.60 0.70 0.80 0.90 \
    --sl_grid 0.50 0.60 0.70 0.80 0.90 \
    --target_flat 0.60 \
    --max_rows_per_file 20000

Outputs:
- Prints top candidates with label mix + simple quality stats.
"""

from __future__ import annotations
import argparse
import glob
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd


def _resample_ohlcv(df1: pd.DataFrame, bar_min: int) -> pd.DataFrame:
    # df1 indexed by timestamp
    df = df1.sort_index()
    rule = f"{bar_min}min"
    out = pd.DataFrame({
        "open":  df["open"].resample(rule).first(),
        "high":  df["high"].resample(rule).max(),
        "low":   df["low"].resample(rule).min(),
        "close": df["close"].resample(rule).last(),
        "volume": df["volume"].resample(rule).sum() if "volume" in df.columns else 0.0,
    }).dropna(subset=["open","high","low","close"])
    return out


def _atr14(df: pd.DataFrame) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    pc = c.shift(1)
    tr = pd.concat([(h - l).abs(), (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.rolling(14, min_periods=14).mean()


def _barrier_label(df: pd.DataFrame, i: int, horizon_bars: int, tp_atr: float, sl_atr: float) -> Tuple[str, float]:
    """
    Returns (label, aux_ret_main)
    aux_ret_main is signed return to horizon close (used for FLAT recycle later).
    """
    if i + horizon_bars >= len(df):
        return ("FLAT", 0.0)

    entry = float(df["close"].iat[i])
    if entry <= 0:
        return ("FLAT", 0.0)

    atr = float(df["atr14"].iat[i])
    if not np.isfinite(atr) or atr <= 0:
        return ("FLAT", 0.0)

    atr_pct = atr / entry
    tp = tp_atr * atr_pct
    sl = sl_atr * atr_pct

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


@dataclass
class MixStats:
    tp: float
    sl: float
    n: int
    buy: float
    sell: float
    flat: float
    dir: float
    # crude "direction usefulness" proxy:
    mean_abs_ret: float
    mean_abs_ret_dir: float


def _stats_for(df: pd.DataFrame, horizon_bars: int, tp: float, sl: float) -> MixStats:
    labels: List[str] = []
    rets: List[float] = []
    for i in range(len(df)):
        lab, r = _barrier_label(df, i, horizon_bars, tp, sl)
        if i + horizon_bars < len(df) and np.isfinite(df["atr14"].iat[i]):
            labels.append(lab)
            rets.append(float(r))

    if not labels:
        return MixStats(tp, sl, 0, 0, 0, 1, 0, 0, 0)

    s = pd.Series(labels)
    vc = s.value_counts(normalize=True).to_dict()
    buy = float(vc.get("BUY", 0.0))
    sell = float(vc.get("SELL", 0.0))
    flat = float(vc.get("FLAT", 0.0))
    dirp = 1.0 - flat

    arr = np.array(rets, dtype=float)
    mean_abs_ret = float(np.nanmean(np.abs(arr))) if arr.size else 0.0
    dir_mask = np.array([lab != "FLAT" for lab in labels], dtype=bool)
    mean_abs_ret_dir = float(np.nanmean(np.abs(arr[dir_mask]))) if dir_mask.any() else 0.0

    return MixStats(tp, sl, int(len(labels)), buy, sell, flat, dirp, mean_abs_ret, mean_abs_ret_dir)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_glob", required=True)
    ap.add_argument("--bar_min", type=int, default=5)
    ap.add_argument("--horizon_min", type=int, default=10)
    ap.add_argument("--tp_grid", type=float, nargs="+", default=[0.5,0.6,0.7,0.8,0.9])
    ap.add_argument("--sl_grid", type=float, nargs="+", default=[0.5,0.6,0.7,0.8,0.9])
    ap.add_argument("--target_flat", type=float, default=0.60)
    ap.add_argument("--max_files", type=int, default=0, help="0 = all")
    ap.add_argument("--max_rows_per_file", type=int, default=0, help="0 = all")
    args = ap.parse_args()

    files = sorted(glob.glob(args.cache_glob))
    if args.max_files and args.max_files > 0:
        files = files[:args.max_files]
    if not files:
        raise SystemExit(f"No files matched: {args.cache_glob}")

    horizon_bars = max(1, int(round(args.horizon_min / float(args.bar_min))))

    # Accumulate 5m bars across files (streamy but simple)
    df_all = []
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
        if args.max_rows_per_file and args.max_rows_per_file > 0:
            df1 = df1.iloc[:args.max_rows_per_file]
        df5 = _resample_ohlcv(df1, bar_min=args.bar_min)
        df_all.append(df5)

    if not df_all:
        raise SystemExit("No usable OHLCV rows found after parsing.")

    df = pd.concat(df_all).sort_index()
    df["atr14"] = _atr14(df)
    df = df.dropna(subset=["atr14"])

    print(f"bars={len(df)} horizon_bars={horizon_bars} files={len(files)}")

    results: List[MixStats] = []
    for tp in args.tp_grid:
        for sl in args.sl_grid:
            st = _stats_for(df, horizon_bars=horizon_bars, tp=float(tp), sl=float(sl))
            if st.n > 0:
                results.append(st)

    # Rank: close to target FLAT, then buy/sell balance, then higher abs-ret on directional labels
    def score(s: MixStats) -> Tuple[float,float,float]:
        flat_err = abs(s.flat - args.target_flat)
        bal_err = abs(s.buy - s.sell)
        # prefer more directional quality
        qual = -s.mean_abs_ret_dir
        return (flat_err, bal_err, qual)

    results.sort(key=score)

    print("\nTop 12 candidates:")
    print("tp  sl   n     FLAT   BUY    SELL   DIR   |mean|ret(dir)")
    for s in results[:12]:
        print(f"{s.tp:>3.2f} {s.sl:>3.2f} {s.n:>6d}  {s.flat:>5.3f}  {s.buy:>5.3f}  {s.sell:>5.3f}  {s.dir:>5.3f}   {s.mean_abs_ret_dir:>8.5f}")

    best = results[0]
    print("\nRecommended starting point:")
    print(f"TP_ATR_MULT={best.tp:.2f}  SL_ATR_MULT={best.sl:.2f}  (targets FLAT≈{args.target_flat:.2f})")


if __name__ == "__main__":
    main()
