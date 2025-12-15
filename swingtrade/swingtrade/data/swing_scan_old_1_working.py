#!/usr/bin/env python3
"""
swing_scan_v2.py

Daily swing scanner for NSE cash equities using DhanHQ v2 Historical API.

Key additions (v2):
- Weekly overlay: weekly EMA20/EMA50, weekly close>EMA20 flag (+ optional EMA20>EMA50)
- Breakout trigger flags: breakout_today (close > prior 20D high), vol_confirm, actionable_now (breakout + 2x vol)
- ADX14 soft gate: adx_ok (ADX>18 or rising 3 bars), bonus if ADX>22
- MACD histogram expansion (small tie-break bonus)
- Candlestick flags: bullish engulfing / hammer near resistance (small bonus + reasons)
- 52-week distance features: high_252, dist_52w_high, near_52w_12pct
- strong_ok tier: prox20<=0.04 AND (bbw_pct<=0.80 OR atrp_ratio<=0.90) AND not extended AND weekly ok
- Plain-English report split: Actionable now first, then Watchlist

Usage:
  export DHAN_ACCESS_TOKEN="<token>"   # or DHAN_TOKEN_B64
  python swing_scan_v2.py --master-csv api-scrip-master.csv --watchlist watchlist.txt

Outputs:
  - swing_scan_report.csv
  - Swing_Scan_Findings_and_Recommendations.md

Notes:
- For daily candles, toDate is treated as non-inclusive by Dhan for daily history.
  We use tomorrow as toDate to include today's candle.
"""

from __future__ import annotations

import argparse
import base64
import dataclasses
import datetime as dt
import logging
import os
import sys
import time
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import requests


# -----------------------------
# Logging
# -----------------------------
LOG = logging.getLogger("swing_scan")


def setup_logging(level: str, log_file: str | None = None) -> None:
    lvl = getattr(logging, level.upper(), None)
    if not isinstance(lvl, int):
        raise ValueError(f"Invalid log level: {level}")

    handlers: List[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(
        level=lvl,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        handlers=handlers,
    )


# -----------------------------
# Dhan REST
# -----------------------------
DHAN_BASE = "https://api.dhan.co/v2"


class DhanHTTPError(RuntimeError):
    pass


def _looks_like_jwt(s: str) -> bool:
    return isinstance(s, str) and s.count(".") >= 2 and len(s) > 40


def get_access_token() -> str:
    """
    Supports either:
    - DHAN_ACCESS_TOKEN = "<jwt>"
    - DHAN_TOKEN_B64 = base64("client_id:jwt")  OR base64("<jwt>")
    """
    tok = os.getenv("DHAN_ACCESS_TOKEN", "").strip()
    if tok:
        return tok

    b64 = os.getenv("DHAN_TOKEN_B64", "").strip()
    if not b64:
        raise RuntimeError("Missing DHAN_ACCESS_TOKEN or DHAN_TOKEN_B64 env var")

    try:
        raw = base64.b64decode(b64).decode("utf-8", errors="replace").strip()
    except Exception as e:
        raise RuntimeError("DHAN_TOKEN_B64 is not valid base64") from e

    if ":" in raw and not _looks_like_jwt(raw):
        raw = raw.split(":", 1)[1].strip()

    if not _looks_like_jwt(raw):
        LOG.warning("Access token does not look like a JWT. Double-check your env vars.")

    return raw


def dhan_post(path: str, payload: dict, access_token: str, timeout: int = 30) -> dict:
    url = f"{DHAN_BASE}{path}"
    headers = {"Content-Type": "application/json", "access-token": access_token}
    LOG.debug("POST %s payload=%s", url, payload)

    r = requests.post(url, headers=headers, json=payload, timeout=timeout)
    if r.status_code >= 400:
        raise DhanHTTPError(f"[HTTP {r.status_code}] {r.text.strip()}")

    try:
        return r.json()
    except Exception as e:
        raise DhanHTTPError(f"Non-JSON response: HTTP {r.status_code} {r.text[:200]}") from e


def fetch_daily_ohlcv(
    *,
    security_id: str,
    exchange_segment: str,
    instrument: str,
    from_date: str,
    to_date: str,
    expiry_code: int = 0,
    oi: bool = False,
    access_token: str,
) -> pd.DataFrame:
    """
    Fetch daily candles.

    Dhan v2 requires: securityId, exchangeSegment, instrument, fromDate, toDate.
    toDate is treated as non-inclusive for daily history -> use tomorrow to include today.
    """
    payload = {
        "securityId": str(security_id),
        "exchangeSegment": exchange_segment,
        "instrument": instrument,
        "expiryCode": int(expiry_code),
        "oi": bool(oi),
        "fromDate": from_date,
        "toDate": to_date,
    }

    obj = dhan_post("/charts/historical", payload, access_token=access_token)

    if not obj or "timestamp" not in obj or not obj["timestamp"]:
        return pd.DataFrame()

    ts = pd.to_datetime(pd.Series(obj["timestamp"], dtype="int64"), unit="s", utc=True)
    ts = ts.dt.tz_convert("Asia/Kolkata")
    out = pd.DataFrame(
        {
            "open": pd.Series(obj.get("open", []), dtype="float64"),
            "high": pd.Series(obj.get("high", []), dtype="float64"),
            "low": pd.Series(obj.get("low", []), dtype="float64"),
            "close": pd.Series(obj.get("close", []), dtype="float64"),
            "volume": pd.Series(obj.get("volume", []), dtype="float64"),
        }
    )
    out.index = ts
    out.index.name = "dt"
    out = out.sort_index()
    out = out.dropna(subset=["open", "high", "low", "close"])
    return out


# -----------------------------
# Instrument resolution
# -----------------------------
@dataclasses.dataclass(frozen=True)
class Instrument:
    symbol: str
    security_id: str
    exchange_segment: str  # e.g. NSE_EQ
    instrument: str        # e.g. EQUITY
    expiry_code: int = 0


def load_master(master_csv: str) -> pd.DataFrame:
    usecols = [
        "SEM_EXM_EXCH_ID",
        "SEM_SMST_SECURITY_ID",
        "SEM_INSTRUMENT_NAME",
        "SEM_EXPIRY_CODE",
        "SEM_TRADING_SYMBOL",
        "SEM_CUSTOM_SYMBOL",
        "SM_SYMBOL_NAME",
        "SEM_SERIES",
    ]
    df = pd.read_csv(master_csv, usecols=usecols, low_memory=False)
    for c in ["SEM_EXM_EXCH_ID", "SEM_INSTRUMENT_NAME", "SEM_TRADING_SYMBOL", "SEM_CUSTOM_SYMBOL", "SM_SYMBOL_NAME", "SEM_SERIES"]:
        df[c] = df[c].astype(str).str.strip()
    df["SEM_SMST_SECURITY_ID"] = df["SEM_SMST_SECURITY_ID"].astype(str).str.strip()
    return df


def _exchange_segment(exch: str, instr: str) -> str:
    if exch == "NSE" and instr == "EQUITY":
        return "NSE_EQ"
    if exch == "BSE" and instr == "EQUITY":
        return "BSE_EQ"
    if exch == "MCX":
        return "MCX_COMM"
    if exch == "NSE" and instr in ("FUTSTK", "OPTSTK", "FUTIDX", "OPTIDX"):
        return "NSE_FNO"
    if exch == "NSE" and instr in ("FUTCUR", "OPTCUR"):
        return "NSE_CURRENCY"
    return f"{exch}_EQ"


def resolve_symbol(
    sym: str,
    master: pd.DataFrame,
    prefer_exchange: str = "NSE",
    prefer_instrument: str = "EQUITY",
    prefer_series: str = "EQ",
) -> Optional[Instrument]:
    s = str(sym).strip()
    if not s:
        return None

    if s.isdigit() and len(s) >= 3:
        return Instrument(symbol=s, security_id=s, exchange_segment="NSE_EQ", instrument="EQUITY", expiry_code=0)

    m = master
    cand = m[m["SEM_TRADING_SYMBOL"].str.upper() == s.upper()]
    if not cand.empty:
        cand1 = cand[
            (cand["SEM_EXM_EXCH_ID"] == prefer_exchange)
            & (cand["SEM_INSTRUMENT_NAME"] == prefer_instrument)
            & (cand["SEM_SERIES"] == prefer_series)
        ]
        if cand1.empty:
            cand1 = cand[(cand["SEM_EXM_EXCH_ID"] == prefer_exchange) & (cand["SEM_INSTRUMENT_NAME"] == prefer_instrument)]
        if cand1.empty:
            cand1 = cand
        row = cand1.iloc[0]
        instr = row["SEM_INSTRUMENT_NAME"]
        exch = row["SEM_EXM_EXCH_ID"]
        return Instrument(
            symbol=s,
            security_id=row["SEM_SMST_SECURITY_ID"],
            exchange_segment=_exchange_segment(exch, instr),
            instrument=instr,
            expiry_code=int(row.get("SEM_EXPIRY_CODE", 0) or 0),
        )

    mask = (
        m["SEM_CUSTOM_SYMBOL"].str.upper().str.contains(s.upper(), na=False)
        | m["SM_SYMBOL_NAME"].str.upper().str.contains(s.upper(), na=False)
    )
    cand = m[mask]
    if cand.empty:
        return None

    cand = cand[(cand["SEM_EXM_EXCH_ID"] == prefer_exchange) & (cand["SEM_INSTRUMENT_NAME"] == prefer_instrument)]
    cand = cand[cand["SEM_SERIES"] == prefer_series]
    if cand.empty:
        return None

    row = cand.iloc[0]
    instr = row["SEM_INSTRUMENT_NAME"]
    exch = row["SEM_EXM_EXCH_ID"]
    return Instrument(
        symbol=s,
        security_id=row["SEM_SMST_SECURITY_ID"],
        exchange_segment=_exchange_segment(exch, instr),
        instrument=instr,
        expiry_code=int(row.get("SEM_EXPIRY_CODE", 0) or 0),
    )


# -----------------------------
# Indicators
# -----------------------------
EPS = 1e-9


def ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False, min_periods=span).mean()


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=n).mean()


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).rolling(n, min_periods=n).mean()
    down = (-d.clip(upper=0)).rolling(n, min_periods=n).mean()
    rs = up / (down.replace(0, np.nan))
    return 100 - (100 / (1 + rs))


def bollinger_width(close: pd.Series, n: int = 20, k: float = 2.0) -> pd.Series:
    ma = close.rolling(n, min_periods=n).mean()
    sd = close.rolling(n, min_periods=n).std(ddof=0)
    upper = ma + k * sd
    lower = ma - k * sd
    return (upper - lower) / (ma.replace(0, np.nan) + EPS)


def macd_hist(close: pd.Series, fast: int = 12, slow: int = 26, sig: int = 9) -> pd.Series:
    m = ema(close, fast) - ema(close, slow)
    s = ema(m, sig)
    return m - s


def adx(df: pd.DataFrame, n: int = 14) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"]

    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr1 = (high - low).abs()
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr_w = tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1 / n, adjust=False, min_periods=n).mean() / (atr_w + EPS)
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1 / n, adjust=False, min_periods=n).mean() / (atr_w + EPS)
    dx = 100 * (plus_di - minus_di).abs() / ((plus_di + minus_di) + EPS)
    return dx.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


# -----------------------------
# Patterns (flags only)
# -----------------------------
def bullish_engulfing(df: pd.DataFrame) -> bool:
    if df is None or len(df) < 2:
        return False
    prev = df.iloc[-2]
    cur = df.iloc[-1]
    prev_bear = prev["close"] < prev["open"]
    cur_bull = cur["close"] > cur["open"]
    if not (prev_bear and cur_bull):
        return False
    return (cur["close"] >= prev["open"]) and (cur["open"] <= prev["close"])


def hammer(df: pd.DataFrame) -> bool:
    if df is None or len(df) < 1:
        return False
    cur = df.iloc[-1]
    o, h, l, c = float(cur["open"]), float(cur["high"]), float(cur["low"]), float(cur["close"])
    body = abs(c - o) + EPS
    lower = min(o, c) - l
    upper = h - max(o, c)
    return (lower >= 1.5 * body) and (upper <= 0.5 * body)


# -----------------------------
# Weekly overlay
# -----------------------------
def weekly_bars_from_daily(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    last_day = pd.Timestamp(df.index[-1]).tz_convert("Asia/Kolkata").date()
    wk = df.resample("W-FRI", label="right", closed="right").agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
    )
    wk = wk.dropna(subset=["open", "high", "low", "close"])
    wk = wk[wk.index.date <= last_day]  # drop future label weeks
    return wk



# -----------------------------
# VRVP (Visible Range Volume Profile) - daily approximation (fast Option A)
# -----------------------------
VRVP_LOOKBACK_DAYS = 120
VRVP_BIN_PCT = 0.0025        # 0.25% of price
VRVP_BIN_ATR_MULT = 0.50     # 0.5 * ATR as alternative bin size
VRVP_LVN_Q = 0.25            # bottom quartile bins are LVN (low-volume nodes)
VRVP_HVN_Q = 0.75            # top quartile bins are HVN (high-volume nodes)
VRVP_OVERHEAD_ATR = 2.0      # evaluate "air pocket" overhead within N ATR above resistance


def compute_vrvp_features(df: pd.DataFrame) -> dict:
    """
    Fast daily VRVP approximation:
      - Put each day's volume into a bin at typical price (H+L+C)/3.
      - Build a volume histogram across bins over the last VRVP_LOOKBACK_DAYS.

    Returns:
      poc_price, dist_to_poc, poc_support (0/1),
      lvn_overhead (0/1), hvn_overhead_close (0/1), vrvp_bin_size
    """
    if df is None or df.empty:
        return {}

    n = len(df)
    look = df.tail(min(n, VRVP_LOOKBACK_DAYS))
    if len(look) < 60:
        return {}

    tp = (look["high"].astype(float) + look["low"].astype(float) + look["close"].astype(float)) / 3.0
    vol = look["volume"].astype(float).fillna(0.0)

    last_close = float(df["close"].iloc[-1]) if "close" in df and pd.notna(df["close"].iloc[-1]) else float("nan")
    if not (last_close > 0):
        return {}

    # ATR for adaptive bin sizing
    atr14 = atr(df, 14)
    atr_abs = float(atr14.iloc[-1]) if len(atr14) and pd.notna(atr14.iloc[-1]) else float("nan")

    bin_size = max(last_close * VRVP_BIN_PCT, (atr_abs * VRVP_BIN_ATR_MULT) if (atr_abs == atr_abs and atr_abs > 0) else 0.0)
    if not (bin_size > 0):
        return {}

    tp_min = float(tp.min())
    tp_max = float(tp.max())
    if not (tp_max > tp_min):
        return {"vrvp_bin_size": bin_size}

    pad = bin_size * 0.5
    left = tp_min - pad
    right = tp_max + pad
    import numpy as _np
    bins = _np.arange(left, right + bin_size, bin_size)
    if len(bins) < 8:
        return {"vrvp_bin_size": bin_size}

    bidx = _np.digitize(tp.to_numpy(), bins) - 1
    bidx = _np.clip(bidx, 0, len(bins) - 2)
    vhist = _np.bincount(bidx, weights=vol.to_numpy(), minlength=len(bins) - 1)

    if vhist.sum() <= 0:
        return {"vrvp_bin_size": bin_size}

    centers = bins[:-1] + 0.5 * bin_size
    poc_i = int(_np.argmax(vhist))
    poc_price = float(centers[poc_i])
    dist_to_poc = (poc_price - last_close) / last_close

    nz = vhist[vhist > 0]
    if len(nz) >= 10:
        lvn_thr = float(_np.quantile(nz, VRVP_LVN_Q))
        hvn_thr = float(_np.quantile(nz, VRVP_HVN_Q))
    else:
        med = float(_np.median(nz)) if len(nz) else 0.0
        lvn_thr = med * 0.6
        hvn_thr = med * 1.4

    # POC support: POC is below close and not far (<= ~1.2 ATR)
    if atr_abs == atr_abs and atr_abs > 0:
        poc_support = float((last_close >= poc_price) and ((last_close - poc_price) <= 1.2 * atr_abs))
        overhead_end = None
    else:
        poc_support = float((last_close >= poc_price) and ((last_close - poc_price) / last_close <= 0.03))

    # Overhead corridor (above 20D resistance) for "air pocket" test
    high20 = df["high"].rolling(20, min_periods=20).max()
    res = float(high20.iloc[-1]) if len(high20) and pd.notna(high20.iloc[-1]) else last_close
    start = res

    if atr_abs == atr_abs and atr_abs > 0:
        end = res + VRVP_OVERHEAD_ATR * atr_abs
    else:
        end = res * 1.06

    mask = (centers > start) & (centers <= end)
    corridor = vhist[mask]
    if corridor.size >= 3:
        lvn_overhead = float(_np.max(corridor) <= lvn_thr)
        hvn_overhead_close = float(_np.max(corridor) >= hvn_thr)
    else:
        lvn_overhead = 0.0
        hvn_overhead_close = 0.0

    return {
        "poc_price": poc_price,
        "dist_to_poc": dist_to_poc,
        "poc_support": poc_support,
        "lvn_overhead": lvn_overhead,
        "hvn_overhead_close": hvn_overhead_close,
        "vrvp_bin_size": bin_size,
    }

# -----------------------------
# Scoring
# -----------------------------
def pct_rank(s: pd.Series, asc: bool = True) -> pd.Series:
    return s.rank(pct=True, ascending=asc)


def build_reasons(row: pd.Series) -> str:
    reasons: List[str] = []

    if row.get("actionable_now", 0):
        reasons.append("Actionable now: breakout + 2× volume")

    if row.get("wk_trend_ok", 0):
        reasons.append("Weekly close > weekly EMA20")
    if row.get("wk_trend_health", 0):
        reasons.append("Weekly EMA20 > weekly EMA50")

    if row.get("trend_up", 0) >= 1:
        reasons.append("EMA50 > EMA200 (daily uptrend)")
    if row.get("above_ema50", 0) >= 1:
        reasons.append("Close above EMA50")

    if np.isfinite(row.get("prox20", np.nan)) and row["prox20"] <= 0.04:
        reasons.append(f"Within {row['prox20']*100:.1f}% of 20D high")
    elif np.isfinite(row.get("prox20", np.nan)) and row["prox20"] <= 0.06:
        reasons.append(f"Within {row['prox20']*100:.1f}% of 20D high (mild)")

    if np.isfinite(row.get("prox55", np.nan)) and row["prox55"] <= 0.03:
        reasons.append(f"Within {row['prox55']*100:.1f}% of 55D high")

    if np.isfinite(row.get("bbw_pct", np.nan)) and row["bbw_pct"] <= 0.80:
        reasons.append("Strong squeeze (BB width vs 120D median)")
    elif np.isfinite(row.get("bbw_pct", np.nan)) and row["bbw_pct"] <= 0.85:
        reasons.append("BB width contracted (squeeze-ish)")

    if np.isfinite(row.get("atrp_ratio", np.nan)) and row["atrp_ratio"] <= 0.90:
        reasons.append("ATR% contracting (5D < 20D)")

    if row.get("vol_dry", 0) >= 1:
        reasons.append("Volume drying up in base")

    if row.get("adx_ok", 0) >= 1:
        reasons.append(f"ADX supportive ({row.get('adx14', np.nan):.1f})")
    if row.get("adx_strong", 0) >= 1:
        reasons.append("ADX strong (>22)")

    if row.get("macd_hist_expand", 0) >= 1:
        reasons.append("MACD histogram expanding")

    if row.get("pat_engulf", 0) >= 1:
        reasons.append("Bullish engulfing")
    if row.get("pat_hammer", 0) >= 1:
        reasons.append("Hammer candle")

    if row.get("near_52w_12pct", 0) >= 1:
        reasons.append("Near 52W high (within 12%)")

    # VRVP reasons (daily approximation)
    if row.get("poc_support", 0) >= 1:
        reasons.append("VRVP: strong support below (POC)")
    if row.get("lvn_overhead", 0) >= 1:
        reasons.append("VRVP: low-volume air pocket overhead")
    if row.get("hvn_overhead_close", 0) >= 1:
        reasons.append("VRVP: heavy volume overhead (may stall)")

    if row.get("vol_confirm", 0) >= 1 and not row.get("actionable_now", 0):
        reasons.append("Volume confirmation (>1.5× 20D avg)")

    if row.get("extended_penalty", 0) > 0:
        reasons.append("Slightly extended vs EMA20 (avoid chasing)")

    return " | ".join(reasons) if reasons else "—"


def score_one(df: pd.DataFrame) -> Optional[dict]:
    # Minimum bars: EMA200 needs ~200. Keep a small buffer.
    if df is None or df.empty or len(df) < 210:
        return None

    df = df.copy()
    c = df["close"]
    v = df["volume"]

    ema20 = ema(c, 20)
    ema50 = ema(c, 50)
    ema200 = ema(c, 200)

    atr14 = atr(df, 14)
    atrp = (atr14 / (c.replace(0, np.nan) + EPS)) * 100.0
    rsi14 = rsi(c, 14)
    bbw = bollinger_width(c, 20, 2.0)

    high20 = df["high"].rolling(20, min_periods=20).max()
    high55 = df["high"].rolling(55, min_periods=55).max()
    high252 = df["high"].rolling(252, min_periods=252).max()

    vol20 = v.rolling(20, min_periods=20).mean()
    vol10 = v.rolling(10, min_periods=10).mean()

    # Weekly overlay
    wk = weekly_bars_from_daily(df)
    wk_close = np.nan
    wk_ema20 = np.nan
    wk_ema50 = np.nan
    wk_trend_ok = 0.0
    wk_trend_health = 0.0
    if not wk.empty and len(wk) >= 25:
        wk_close_s = wk["close"]
        wk_ema20_s = ema(wk_close_s, 20)
        wk_ema50_s = ema(wk_close_s, 50)
        wk_close = float(wk_close_s.iloc[-1])
        wk_ema20 = float(wk_ema20_s.iloc[-1]) if np.isfinite(wk_ema20_s.iloc[-1]) else np.nan
        wk_ema50 = float(wk_ema50_s.iloc[-1]) if np.isfinite(wk_ema50_s.iloc[-1]) else np.nan
        if np.isfinite(wk_ema20):
            wk_trend_ok = float(wk_close > wk_ema20)
        if np.isfinite(wk_ema20) and np.isfinite(wk_ema50):
            wk_trend_health = float(wk_ema20 > wk_ema50)

    # ADX
    adx14 = adx(df, 14)
    adx_val = float(adx14.iloc[-1]) if np.isfinite(adx14.iloc[-1]) else np.nan
    adx_rising_3 = False
    if len(adx14.dropna()) >= 4:
        d3 = adx14.diff().tail(3)
        adx_rising_3 = bool((d3 > 0).all())
    adx_ok = float((np.isfinite(adx_val) and adx_val > 18.0) or adx_rising_3)
    adx_strong = float(np.isfinite(adx_val) and adx_val > 22.0)

    # MACD histogram
    mh = macd_hist(c)
    mh_val = float(mh.iloc[-1]) if np.isfinite(mh.iloc[-1]) else np.nan
    mh_delta = float(mh.diff().iloc[-1]) if len(mh.dropna()) >= 2 and np.isfinite(mh.diff().iloc[-1]) else np.nan
    macd_hist_expand = float(np.isfinite(mh_delta) and mh_delta > 0)

    last = df.iloc[-1]
    idx = df.index[-1]
    close = float(last["close"])
    if not np.isfinite(close) or close <= 0:
        return None

    trend_up = float(np.isfinite(ema50.iloc[-1]) and np.isfinite(ema200.iloc[-1]) and (ema50.iloc[-1] > ema200.iloc[-1]))
    above_ema50 = float(np.isfinite(ema50.iloc[-1]) and (close > float(ema50.iloc[-1])))
    above_ema20 = float(np.isfinite(ema20.iloc[-1]) and (close > float(ema20.iloc[-1])))

    prox20 = float((float(high20.iloc[-1]) - close) / close) if np.isfinite(high20.iloc[-1]) else np.nan
    prox55 = float((float(high55.iloc[-1]) - close) / close) if np.isfinite(high55.iloc[-1]) else np.nan

    bbw_now = float(bbw.iloc[-1]) if np.isfinite(bbw.iloc[-1]) else np.nan
    bbw_med120 = float(bbw.rolling(120, min_periods=120).median().iloc[-1]) if len(bbw.dropna()) >= 140 else np.nan
    bbw_pct = float(bbw_now / (bbw_med120 + EPS)) if np.isfinite(bbw_now) and np.isfinite(bbw_med120) else np.nan

    atrp_now = float(atrp.iloc[-1]) if np.isfinite(atrp.iloc[-1]) else np.nan
    atrp5 = atrp.rolling(5, min_periods=5).mean()
    atrp20 = atrp.rolling(20, min_periods=20).mean()
    atrp_ratio = float(atrp5.iloc[-1] / (atrp20.iloc[-1] + EPS)) if np.isfinite(atrp5.iloc[-1]) and np.isfinite(atrp20.iloc[-1]) else np.nan

    vol_dry = float(np.isfinite(vol20.iloc[-1]) and np.isfinite(vol10.iloc[-1]) and (vol10.iloc[-1] < 0.8 * vol20.iloc[-1]))

    ext = (close - float(ema20.iloc[-1])) / (float(atr14.iloc[-1]) + EPS) if np.isfinite(ema20.iloc[-1]) and np.isfinite(atr14.iloc[-1]) else 0.0
    extended_penalty = float(max(0.0, ext - 1.8))
    not_extended = float(extended_penalty <= 0.0)

    turnover = float((c.tail(20).mean() * v.tail(20).mean()) if len(c) >= 20 else 0.0)
    ret_20 = float(c.pct_change(20).iloc[-1]) if len(c) > 21 and np.isfinite(c.pct_change(20).iloc[-1]) else np.nan
    ret_60 = float(c.pct_change(60).iloc[-1]) if len(c) > 61 and np.isfinite(c.pct_change(60).iloc[-1]) else np.nan

    r = float(rsi14.iloc[-1]) if np.isfinite(rsi14.iloc[-1]) else np.nan
    rsi_ok = float(np.isfinite(r) and 55 <= r <= 70)

    high_252 = float(high252.iloc[-1]) if np.isfinite(high252.iloc[-1]) else np.nan
    dist_52w_high = float((high_252 - close) / close) if np.isfinite(high_252) else np.nan
    near_52w_12pct = float(np.isfinite(dist_52w_high) and dist_52w_high <= 0.12)

    # Breakout trigger (no lookahead)
    high20_prev = float(high20.shift(1).iloc[-1]) if np.isfinite(high20.shift(1).iloc[-1]) else np.nan
    vol20_prev = float(vol20.shift(1).iloc[-1]) if np.isfinite(vol20.shift(1).iloc[-1]) else np.nan
    breakout_today = float(np.isfinite(high20_prev) and close > high20_prev)
    vol_confirm = float(np.isfinite(vol20_prev) and float(last["volume"]) > 1.5 * vol20_prev)
    vol_surge = float(np.isfinite(vol20_prev) and float(last["volume"]) > 2.0 * vol20_prev)
    actionable_now = float(bool(breakout_today and vol_surge))

    pat_engulf = float(bullish_engulfing(df) and np.isfinite(prox20) and prox20 <= 0.06)
    pat_hammer = float(hammer(df) and np.isfinite(prox20) and prox20 <= 0.06)

    # VRVP (daily approximation)
    vrvp = compute_vrvp_features(df)

    return {
        "asof": str(idx),
        "close": close,
        "trend_up": trend_up,
        "above_ema50": above_ema50,
        "above_ema20": above_ema20,
        "prox20": prox20,
        "prox55": prox55,
        "breakout_lvl_20d": float(high20.iloc[-1]) if np.isfinite(high20.iloc[-1]) else np.nan,
        "breakout_lvl_55d": float(high55.iloc[-1]) if np.isfinite(high55.iloc[-1]) else np.nan,
        "bbw_now": bbw_now,
        "bbw_pct": bbw_pct,
        "atrp_now": atrp_now,
        "atrp_ratio": atrp_ratio,
        "vol_dry": vol_dry,
        "extended_penalty": extended_penalty,
        "not_extended": not_extended,
        "turnover": turnover,
        "ret_20": ret_20,
        "ret_60": ret_60,
        "rsi14": r,
        "rsi_ok": rsi_ok,
        "wk_close": wk_close,
        "wk_ema20": wk_ema20,
        "wk_ema50": wk_ema50,
        "wk_trend_ok": wk_trend_ok,
        "wk_trend_health": wk_trend_health,
        "breakout_today": breakout_today,
        "vol_confirm": vol_confirm,
        "vol_surge": vol_surge,
        "actionable_now": actionable_now,
        "adx14": adx_val,
        "adx_ok": adx_ok,
        "adx_strong": adx_strong,
        "macd_hist": mh_val,
        "macd_hist_delta": mh_delta,
        "macd_hist_expand": macd_hist_expand,
        "pat_engulf": pat_engulf,
        "pat_hammer": pat_hammer,
        "high_252": high_252,
        "dist_52w_high": dist_52w_high,
        "near_52w_12pct": near_52w_12pct,
        # VRVP
        "poc_price": float(vrvp.get("poc_price", np.nan)),
        "dist_to_poc": float(vrvp.get("dist_to_poc", np.nan)),
        "poc_support": float(vrvp.get("poc_support", 0.0)),
        "lvn_overhead": float(vrvp.get("lvn_overhead", 0.0)),
        "hvn_overhead_close": float(vrvp.get("hvn_overhead_close", 0.0)),
        "vrvp_bin_size": float(vrvp.get("vrvp_bin_size", np.nan)),
    }


def final_score_table(rows: List[dict]) -> pd.DataFrame:
    out = pd.DataFrame(rows).copy()
    if out.empty:
        return out

    out["r_turnover"] = pct_rank(out["turnover"].fillna(0), asc=True)
    out["r_ret60"] = pct_rank(out["ret_60"].fillna(-1), asc=True)
    out["r_prox20"] = 1 - pct_rank(out["prox20"].fillna(1), asc=True)
    out["r_squeeze"] = 1 - pct_rank(out["bbw_pct"].fillna(10), asc=True)
    out["r_atr_contr"] = 1 - pct_rank(out["atrp_ratio"].fillna(10), asc=True)
    out["r_low_atr"] = 1 - pct_rank(out["atrp_now"].fillna(50), asc=True)
    out["r_not_extended"] = 1 - pct_rank(out["extended_penalty"].fillna(10), asc=True)

    out["hard_ok"] = (
        (out["trend_up"] >= 1)
        & (out["above_ema50"] >= 1)
        & (out["turnover"] > 2e7)
        & (out["prox20"] < 0.06)
    )

    squeeze_ok = (out["bbw_pct"] <= 0.80) | (out["atrp_ratio"] <= 0.90)
    out["strong_ok"] = (
        (out["hard_ok"])
        & (out["prox20"] <= 0.04)
        & squeeze_ok
        & (out["not_extended"] >= 1)
        & (out["wk_trend_ok"] >= 1)
    )

    adx_bonus = np.where(out["adx_strong"] >= 1, 1.0, np.where(out["adx_ok"] >= 1, 0.6, 0.0))
    macd_bonus = np.where(out["macd_hist_expand"] >= 1, 1.0, 0.0)
    pat_bonus = np.where((out["pat_engulf"] >= 1) | (out["pat_hammer"] >= 1), 1.0, 0.0)
    wk_bonus = np.where(out["wk_trend_ok"] >= 1, 1.0, 0.0)

    out["score"] = (
        0.22 * out["r_turnover"]
        + 0.20 * out["r_ret60"]
        + 0.18 * out["r_prox20"]
        + 0.16 * out["r_squeeze"]
        + 0.10 * out["r_atr_contr"]
        + 0.08 * out["r_low_atr"]
        + 0.06 * out["r_not_extended"]
        + 0.03 * wk_bonus
        + 0.03 * adx_bonus
        + 0.02 * macd_bonus
        + 0.02 * out["vol_dry"].fillna(0).astype(float)
        + 0.02 * out["rsi_ok"].fillna(0).astype(float)
        + 0.12 * out["lvn_overhead"].fillna(0).astype(float)
        + 0.08 * out["poc_support"].fillna(0).astype(float)
        + 0.01 * pat_bonus
    )

    out = out.sort_values(["actionable_now", "strong_ok", "hard_ok", "score"], ascending=[False, False, False, False])
    out["reasons"] = out.apply(build_reasons, axis=1)
    return out



# -----------------------------
# Human-readable trade plan sentences
# -----------------------------
ENTRY_BUFFER_PCT = 0.002      # 0.2% above resistance
SL_ATR_MULT = 1.5             # stop distance in ATRs
STOP_BELOW_BREAKOUT_PCT = 0.001  # 0.1% below breakout level (tiny buffer)
R_MULT = 2.0                  # target = entry + R_MULT * risk


def _fmt_inr(x: float) -> str:
    """Format INR with commas. Falls back gracefully for NaN."""
    try:
        if x is None or not np.isfinite(float(x)):
            return "—"
        v = float(x)
        return f"₹{v:,.2f}"
    except Exception:
        return "—"


def _pct(x: float) -> str:
    try:
        if x is None or not np.isfinite(float(x)):
            return "—"
        return f"{float(x)*100:.1f}%"
    except Exception:
        return "—"


def build_action_sentence(row: pd.Series) -> str:
    """
    Generate a common-man sentence:
    - Watchlist mode: wait for breakout, then enter; stop; first target; % moves.
    - Actionable now: breakout+volume already happened; gives entry/stop/target plan.

    Uses only columns we already compute (close, breakout_lvl_20d, atrp_now, vol_confirm/actionable_now).
    """
    sym = str(row.get("symbol", "—"))
    close = float(row["close"]) if "close" in row and np.isfinite(row["close"]) else np.nan
    res = float(row["breakout_lvl_20d"]) if "breakout_lvl_20d" in row and np.isfinite(row["breakout_lvl_20d"]) else np.nan
    atrp_now = float(row["atrp_now"]) if "atrp_now" in row and np.isfinite(row["atrp_now"]) else np.nan

    # Compute ATR absolute from ATR% (ATR% = ATR/close*100)
    atr_abs = close * (atrp_now / 100.0) if np.isfinite(close) and np.isfinite(atrp_now) else np.nan

    # Entry plan: slightly above resistance (breakout level) if known, else "near current"
    entry = res * (1.0 + ENTRY_BUFFER_PCT) if np.isfinite(res) else close

    # Stop plan: tighter of (below breakout) vs (entry - SL_ATR_MULT*ATR)
    stop_candidates = []
    if np.isfinite(res):
        stop_candidates.append(res * (1.0 - STOP_BELOW_BREAKOUT_PCT))
    if np.isfinite(atr_abs) and np.isfinite(entry):
        stop_candidates.append(entry - SL_ATR_MULT * atr_abs)

    # tighter stop for longs = higher stop price (closer to entry)
    finite_stops = [x for x in stop_candidates if np.isfinite(x)]
    stop = max(finite_stops) if finite_stops else np.nan

    risk = (entry - stop) if np.isfinite(entry) and np.isfinite(stop) else np.nan
    target = (entry + R_MULT * risk) if np.isfinite(risk) and risk > 0 else np.nan

    stop_pct = (stop / entry - 1.0) if np.isfinite(stop) and np.isfinite(entry) and entry > 0 else np.nan
    tgt_pct = (target / entry - 1.0) if np.isfinite(target) and np.isfinite(entry) and entry > 0 else np.nan

    actionable = bool(row.get("actionable_now", 0) >= 1)
    vol_confirm = bool(row.get("vol_confirm", 0) >= 1)
    vol_surge = bool(row.get("vol_surge", 0) >= 1)

    vrvp_hint = []
    if bool(row.get("lvn_overhead", 0) >= 1):
        vrvp_hint.append("less traffic above")
    if bool(row.get("poc_support", 0) >= 1):
        vrvp_hint.append("support below")
    vrvp_txt = (" (" + ", ".join(vrvp_hint) + ")") if vrvp_hint else ""

    if actionable:
        return (
            f"**{sym}**{vrvp_txt}: Breakout is already happening. If you want to trade it, consider entry near {_fmt_inr(entry)}; "
            f"keep a stop near {_fmt_inr(stop)} ({_pct(stop_pct)} from entry). "
            f"First target near {_fmt_inr(target)} ({_pct(tgt_pct)} from entry)."
        )

    # Watchlist sentence
    if np.isfinite(res):
        vol_txt = "with strong volume" if (vol_surge or vol_confirm) else "and watch for strong volume"
        return (
            f"**{sym}**{vrvp_txt}: Wait for price to close above resistance at {_fmt_inr(res)} {vol_txt}. "
            f"After that breakout, enter around {_fmt_inr(entry)}; place stop near {_fmt_inr(stop)} ({_pct(stop_pct)}). "
            f"A reasonable first target is {_fmt_inr(target)} ({_pct(tgt_pct)})."
        )

    # Fallback (missing breakout level)
    return (
        f"**{sym}**{vrvp_txt}: Keep this on watch. Wait for a clear breakout above recent highs with strong volume. "
        f"If it moves up, enter near {_fmt_inr(entry)}; keep a stop below recent support; "
        f"aim for ~2× the risk as the first target."
    )

def write_plain_english_report(
    *,
    table: pd.DataFrame,
    failures: List[Tuple[str, str]],
    out_path: str,
    report_csv_path: str,
    asof: str,
    top_n: int = 30,
) -> None:
    if table is None or table.empty:
        return

    total = int(len(table))
    hard_ok = int(table["hard_ok"].sum()) if "hard_ok" in table.columns else 0
    strong_ok = int(table["strong_ok"].sum()) if "strong_ok" in table.columns else 0
    actionable = int(table["actionable_now"].sum()) if "actionable_now" in table.columns else 0
    fail_n = int(len(failures)) if failures else 0

    checks: List[Tuple[str, bool]] = []
    checks.append(("Scores present for all rows", int(table["score"].isna().sum()) == 0))
    if "rsi14" in table.columns:
        checks.append(("RSI stays in 0–100 range", bool(((table["rsi14"] >= 0) & (table["rsi14"] <= 100)).all())))
    if "atrp_now" in table.columns:
        checks.append(("ATR% (atrp_now) is positive", bool((table["atrp_now"] > 0).all())))
    if "breakout_lvl_20d" in table.columns:
        checks.append(("20D breakout level is >= close", bool(((table["breakout_lvl_20d"] + 1e-9) >= table["close"]).all())))

    actionable_df = table[table["actionable_now"] >= 1].head(top_n)
    watchlist_df = table[table["actionable_now"] < 1].head(top_n)

    cols = ["symbol", "score", "close", "breakout_lvl_20d", "prox20", "bbw_pct", "atrp_ratio", "adx14", "wk_trend_ok", "strong_ok", "reasons"]
    cols = [c for c in cols if c in table.columns]

    lines: List[str] = []
    lines.append("# Swing Scan: Findings & Plain-English Recommendations\n")
    lines.append(f"**As-of candle:** {asof}\n")
    lines.append(f"**Files:**\n- Ranked CSV: `{report_csv_path}`\n- This summary: `{os.path.basename(out_path)}`\n")

    lines.append("## 1) Quick verdict\n")
    lines.append(
        f"- Symbols with usable data: **{total}**\n"
        f"- Passed mild hard filters: **{hard_ok}**\n"
        f"- Passed strong tier (tighter + weekly aligned): **{strong_ok}**\n"
        f"- Actionable now (breakout + 2× vol): **{actionable}**\n"
        f"- Skipped/failed: **{fail_n}**\n"
    )

    lines.append("\n## 2) What the scanner is selecting (plain English)\n")
    lines.append(
        "This scanner looks for **coiled-spring** breakouts in uptrends:\n"
        "- Daily uptrend (EMA50 > EMA200) and close above EMA50\n"
        "- Volatility contraction (BB width vs own 120D median and/or ATR% contracting)\n"
        "- Price sitting close to resistance (near 20D high)\n"
        "- Weekly confirmation to avoid chop (weekly close > weekly EMA20)\n\n"
        "**Actionable now** is only when the stock breaks above the prior 20D high\n"
        "and does it with **2×** the 20D average volume.\n"
    )

    lines.append("\n## 3) Data quality checks\n")
    for name, ok in checks:
        lines.append(f"- {'✅' if ok else '❌'} {name}\n")

    if not actionable_df.empty:
        lines.append("\n## 4) Actionable now (breakout + 2× volume)\n")
        lines.append("These already broke out with strong volume. If you trade them, use a simple entry/stop/target plan:")
        for _, r in actionable_df.iterrows():
            lines.append("- " + build_action_sentence(r))

    lines.append("\n## 5) Watchlist (near breakout, waiting for trigger)\n")
    lines.append(
        "These are close to resistance. The clean action is: wait for a daily close above resistance with volume confirmation, then enter.\n"
    )
    for _, r in watchlist_df.iterrows():
        lines.append("- " + build_action_sentence(r))

    lines.append("\n## 6) Simple trigger + risk plan\n")
    lines.append(
        "- **Trigger:** daily close > prior 20D high (`breakout_today = 1`)\n"
        "- **Confirmation:** volume > 1.5× 20D avg (`vol_confirm = 1`) or 2× for strict\n"
        "- **Stop idea:** below breakout level OR 1.5× ATR(14) (tighter wins)\n"
        "- **First target idea:** 2× risk or next swing resistance zone\n"
    )

    if failures:
        lines.append("\n## 7) Skips / failures (first 30)\n")
        for sym, reason in failures[:30]:
            lines.append(f"- {sym}: {reason}\n")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def read_watchlist(path: str) -> List[str]:
    p = path.lower()
    if p.endswith(".csv"):
        d = pd.read_csv(path)
        for col in ["symbol", "trading_symbol", "SEM_TRADING_SYMBOL"]:
            if col in d.columns:
                return [str(x).strip() for x in d[col].dropna().tolist() if str(x).strip()]
        return [str(x).strip() for x in d.iloc[:, 0].dropna().tolist() if str(x).strip()]
    else:
        syms: List[str] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if s and not s.startswith("#"):
                    syms.append(s)
        return syms


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--master-csv", default="api-scrip-master.csv", help="Dhan instrument master CSV")
    ap.add_argument("--watchlist", required=True, help="Symbols file (txt or csv)")
    ap.add_argument("--days", type=int, default=730, help="Lookback days for daily history (use >= 540 for robust weekly/52W stats)")
    ap.add_argument("--top", type=int, default=30, help="Top N picks to print")
    ap.add_argument("--out", default="swing_scan_report.csv", help="Output report CSV")
    ap.add_argument("--english-out", default="Swing_Scan_Findings_and_Recommendations.md", help="Plain-English summary (markdown)")
    ap.add_argument("--cache-dir", default="cache_ohlc", help="Store fetched candles as JSON for re-use")
    ap.add_argument("--sleep", type=float, default=0.25, help="Sleep between API calls (seconds)")
    ap.add_argument("--log-level", default="INFO", help="Logging level: DEBUG/INFO/WARNING/ERROR")
    ap.add_argument("--log-file", default="", help="Optional log file path (also logs to console)")
    args = ap.parse_args()

    setup_logging(args.log_level, args.log_file or None)

    if args.days < 260:
        LOG.warning("days=%s is low. Weekly EMA50 needs more history; recommend >= 365.", args.days)
    if args.top <= 0:
        raise ValueError("--top must be > 0")
    if args.sleep < 0:
        raise ValueError("--sleep must be >= 0")

    token = get_access_token()
    master = load_master(args.master_csv)
    os.makedirs(args.cache_dir, exist_ok=True)

    watch = read_watchlist(args.watchlist)
    if not watch:
        LOG.error("Watchlist is empty.")
        return 2

    today = dt.date.today()
    to_date = (today + dt.timedelta(days=1)).strftime("%Y-%m-%d")
    from_date = (today - dt.timedelta(days=args.days)).strftime("%Y-%m-%d")

    LOG.info("Scanning symbols=%d from=%s to=%s (toDate non-inclusive)", len(watch), from_date, to_date)

    rows: List[dict] = []
    failures: List[Tuple[str, str]] = []

    for i, sym in enumerate(watch, 1):
        inst = resolve_symbol(sym, master)
        if inst is None:
            failures.append((sym, "Not found in master CSV"))
            LOG.warning("(%d/%d) Skip %s: not found in instrument master", i, len(watch), sym)
            continue

        if not (inst.exchange_segment == "NSE_EQ" and inst.instrument == "EQUITY"):
            failures.append((sym, f"Skipped non-NSE_EQ equity: {inst.exchange_segment}/{inst.instrument}"))
            LOG.debug("(%d/%d) Skip %s: %s/%s", i, len(watch), sym, inst.exchange_segment, inst.instrument)
            continue

        cache_path = os.path.join(args.cache_dir, f"{sym}.json")
        df: Optional[pd.DataFrame] = None

        if os.path.exists(cache_path):
            try:
                df = pd.read_json(cache_path, convert_dates=["dt"])
                if "dt" in df.columns:
                    df = df.set_index("dt")
                df.index = pd.to_datetime(df.index)
                if df.index.tz is None:
                    df.index = df.index.tz_localize("Asia/Kolkata")
                LOG.debug("(%d/%d) %s cache loaded rows=%d", i, len(watch), sym, len(df))
            except Exception as e:
                LOG.debug("(%d/%d) %s cache read failed: %s", i, len(watch), sym, e)
                df = None

        if df is None or df.empty:
            try:
                df = fetch_daily_ohlcv(
                    security_id=inst.security_id,
                    exchange_segment=inst.exchange_segment,
                    instrument=inst.instrument,
                    expiry_code=inst.expiry_code,
                    oi=False,
                    from_date=from_date,
                    to_date=to_date,
                    access_token=token,
                )
                if df.empty:
                    failures.append((sym, "No data returned"))
                    LOG.warning("(%d/%d) Failed %s: no data returned", i, len(watch), sym)
                    continue

                df.reset_index().rename(columns={"dt": "dt"}).to_json(cache_path, orient="records", date_format="iso")
                LOG.debug("(%d/%d) %s fetched rows=%d cached=%s", i, len(watch), sym, len(df), cache_path)
            except Exception as e:
                failures.append((sym, str(e)[:200]))
                LOG.warning("(%d/%d) Failed %s: %s", i, len(watch), sym, e)
                continue
            finally:
                time.sleep(args.sleep)

        met = score_one(df)
        if met is None:
            n = int(len(df)) if df is not None else 0
            first_dt = str(df.index[0]) if df is not None and n else "—"
            last_dt = str(df.index[-1]) if df is not None and n else "—"
            msg = f"Insufficient history (<210 bars) or bad data (rows={n}, first={first_dt}, last={last_dt})"
            failures.append((sym, msg))
            LOG.debug("(%d/%d) Skip %s: %s", i, len(watch), sym, msg)
            continue


        met["symbol"] = sym
        met["security_id"] = inst.security_id
        rows.append(met)

        if LOG.isEnabledFor(logging.DEBUG):
            LOG.debug("(%d/%d) %s prox20=%.4f bbw_pct=%.3f atrp_ratio=%.3f wk_ok=%s actionable=%s vrvp_lvn=%s vrvp_poc=%s",
                      i, len(watch), sym,
                      met.get("prox20", np.nan), met.get("bbw_pct", np.nan), met.get("atrp_ratio", np.nan),
                      int(met.get("wk_trend_ok", 0)), int(met.get("actionable_now", 0)),
                      int(met.get("lvn_overhead", 0)), int(met.get("poc_support", 0)))

    if not rows:
        LOG.error("No symbols produced usable data.")
        if failures:
            LOG.info("Failures (first 30):")
            for s, reason in failures[:30]:
                LOG.info("  - %s: %s", s, reason)
        return 3

    table = final_score_table(rows)
    table.to_csv(args.out, index=False)
    LOG.info("Saved ranked CSV: %s rows=%d", args.out, len(table))

    try:
        write_plain_english_report(
            table=table,
            failures=failures,
            out_path=args.english_out,
            report_csv_path=args.out,
            asof=str(table["asof"].iloc[0]) if "asof" in table.columns and len(table) else "—",
            top_n=max(args.top, 30),
        )
        LOG.info("Saved plain-English report: %s", args.english_out)
    except Exception as e:
        LOG.warning("Failed to write english report: %s", e)

    top = table.head(args.top)
    print("\n=== TOP PICKS ===")
    cols = ["symbol", "score", "actionable_now", "strong_ok", "hard_ok", "close", "breakout_lvl_20d", "prox20", "bbw_pct", "adx14", "reasons"]
    cols = [c for c in cols if c in top.columns]
    with pd.option_context("display.max_colwidth", 160):
        print(top[cols].to_string(index=False))

    if failures:
        print("\n=== FAILURES / SKIPS (first 30) ===", file=sys.stderr)
        for s, reason in failures[:30]:
            print(f"- {s}: {reason}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
