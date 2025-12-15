#!/usr/bin/env python3
"""
Swing trade scanner (daily) for NSE instruments via Dhan v2 historical API.

Key design goals:
- Lean, single-file script.
- High-visibility logging for debugging (INFO/DEBUG).
- No lookahead bias in "actionable now": compares today's close to yesterday's resistance level.
- Plain-English markdown report with actionable/watchlist sections, separated by instrument bucket:
  EQUITY / ETF / REIT / SME / OTHER

Master CSV notes:
- Excel Col F -> SEM_TRADING_SYMBOL (symbol you pass in watchlist)
- Excel Col H -> SEM_CUSTOM_SYMBOL (often contains "ETF" for ETFs)
"""

from __future__ import annotations

import argparse
import atexit
import base64
import dataclasses
import datetime as dt
import io
import json
import hashlib
import logging
import math
import os
import re
import time
import subprocess
import shutil
from urllib.parse import quote
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests


# Optional deps (kept optional to keep the script lightweight)
try:
    from bs4 import BeautifulSoup  # type: ignore
except Exception:  # pragma: no cover
    BeautifulSoup = None  # type: ignore



# -----------------------------
# Cache freshness + Screener fundamentals
# -----------------------------
CACHE_SCHEMA_VERSION = 2

DEFAULT_OHLC_CACHE_TTL_HOURS = 36.0  # daily candles; avoids "stale forever" but doesn't spam API
DEFAULT_FUND_CACHE_TTL_HOURS = 24.0  # Screener fundamentals
SCREENER_BASE = "https://www.screener.in"

# Fundamental guardrails (tune as you like)
FUND_YOY_VETO_PCT = -10.0     # <= -10% YoY net profit growth => veto
FUND_YOY_CAUTION_PCT = 5.0    # < +5% YoY => caution

# Helpers for parsing Screener quarterly tables
_MONTH = {"Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6, "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12}
# Screener sometimes returns obfuscated quarter headers to non-browser clients (e.g. 'Sp 3', 'D 4', 'Jn 5').
# Accept 1-3 letter month tokens and 1-4 digit years; normalize later.
_QCOL_RE = re.compile(r"^([A-Za-z]{1,3})\s*(\d{1,4})$")
_MONTH_ALIASES = {
    'SP': 'Sep',  # Sep
    'S': 'Sep',
    'D': 'Dec',   # Dec
    'DC': 'Dec',
    'JN': 'Jun',  # Jun
    'J': 'Jun',
    'MR': 'Mar',  # Mar
    'MY': 'May',
    'FB': 'Feb',
    'AU': 'Aug',
    'OC': 'Oct',
    'NO': 'Nov',
}
_ZERO_WIDTH = re.compile(r"[\u200b\u200c\u200d\ufeff]")


def _norm_text(x: object) -> str:
    s = "" if x is None else str(x)
    s = s.replace("\xa0", " ")
    s = _ZERO_WIDTH.sub("", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _flatten_col(c: object) -> str:
    """Flatten pandas read_html columns (may be MultiIndex) into a useful string.

    Screener tables sometimes come back as MultiIndex like ('Sep', '2025') or ('Sep 2025', '₹ Cr').
    We try to preserve month+year when present instead of dropping to the last level.
    """
    if isinstance(c, (tuple, list)):
        parts = [_norm_text(p) for p in c if p is not None]
        parts = [p for p in parts if p and p.lower() != "nan" and not p.lower().startswith("unnamed")]
        if not parts:
            return ""
        # If it looks like (Mon, YYYY) combine.
        if len(parts) >= 2 and re.fullmatch(r"[A-Za-z]{2,3}", parts[0]) and re.fullmatch(r"\d{2,4}", parts[1]):
            return _norm_text(f"{parts[0]} {parts[1]}")
        return _norm_text(" ".join(parts))
    return _norm_text(c)


def _clean_number(v):
    if v is None:
        return None
    if isinstance(v, (int, float, np.integer, np.floating)):
        return float(v)
    s = _norm_text(v)
    if not s or s.lower() in ("nan", "-"):
        return None
    neg = False
    if "(" in s and ")" in s:
        neg = True
    s = s.replace(",", "")
    s = re.sub(r"[^0-9.\\-]+", "", s)
    if not s or s in ("-", ".", "-."):
        return None
    try:
        x = float(s)
        return -x if neg else x
    except Exception:
        return None


def _qkey(label: str):
    """Return sortable key for a quarter column label.

    Normalizes both normal headers ('Sep 2025') and obfuscated ones like:
      - 'Sp 3'  -> 'Sep 2023'
      - 'D 4'   -> 'Dec 2024'
      - 'Jn 5'  -> 'Jun 2025'
    """
    s = _norm_text(label)
    if not s:
        return None
    m = _QCOL_RE.match(s)
    if not m:
        return None
    mon_raw = m.group(1).strip()
    yr_raw = m.group(2).strip()

    mon_key = re.sub(r"[^A-Za-z]", "", mon_raw).upper()
    if not mon_key:
        return None
    # normalize month token
    mon = _MONTH_ALIASES.get(mon_key)
    if mon is None:
        # try 3-letter title case
        mon = mon_key[:3].title()
    if mon not in _MONTH:
        return None

    # normalize year token
    try:
        y = int(re.sub(r"[^0-9]", "", yr_raw))
    except Exception:
        return None
    # Heuristic: Screener obfuscation often truncates 2023 -> 3 or 23.
    if y < 10:
        y = 2020 + y
    elif y < 100:
        y = 2000 + y

    return y * 100 + _MONTH[mon]


def parse_net_profit_yoy_from_quarters_df(df: pd.DataFrame, sym: str, logger: logging.Logger) -> Dict[str, object]:
    """Parse YoY net profit growth from a Screener-like *Quarterly Results* table.

    This function is intentionally defensive because Screener pages sometimes obfuscate label text
    (e.g., 'Net Profit' becomes 'Nt Proit') and sometimes return partial tables to bots.

    Strategy:
      1) Detect orientation (metrics-as-rows). If likely transposed, transpose and retry.
      2) Find a *label column* (row names) using heuristics instead of exact text.
      3) Locate the net profit row:
           - exact match (net profit)
           - fuzzy match against 'netprofit' (handles missing letters)
           - positional fallback: row immediately above the 'EPS' row
      4) Identify quarter columns by numeric content and preserve their appearance order.
      5) YoY uses last quarter vs 4 quarters back (same quarter last year).
    """
    def _out(status: str, **kw) -> Dict[str, object]:
        o: Dict[str, object] = {
            "symbol": sym,
            "status": status,
            "note": kw.pop("note", None),
            "source": kw.pop("source", None),
            "yoy_net_profit_growth_pct": None,
            "curr_q_label": None,
            "prev_yoy_q_label": None,
            "curr_net_profit": None,
            "prev_yoy_net_profit": None,
        }
        o.update(kw)
        # Backward-compat key (some older code/logs used 'yoy')
        if "yoy" not in o:
            o["yoy"] = o.get("yoy_net_profit_growth_pct")
        return o

    if df is None or df.empty:
        return _out("no_data", note="empty_df")

    d0 = df.copy()
    d0.columns = [_flatten_col(c) for c in d0.columns]

    # --- Detect label column heuristically ---
    def _find_label_col(d: pd.DataFrame) -> Optional[str]:
        best = None
        best_score = -1.0
        for c in d.columns[: min(6, len(d.columns))]:
            try:
                s = d[c].astype(str).map(_norm_text)
            except Exception:
                continue
            # proportion of cells that are NOT numeric-looking
            nn = float((s.map(lambda x: _clean_number(x) is None)).mean())
            # bonus if it contains "EPS"/"PDF"/"+" which is common in label column
            bonus = 0.0
            if s.str.contains(r"\beps\b", case=False, regex=True).any():
                bonus += 0.25
            if s.str.contains(r"\bpdf\b", case=False, regex=True).any():
                bonus += 0.10
            if s.str.contains(r"\+", regex=True).any():
                bonus += 0.05
            score = nn + bonus
            if score > best_score:
                best_score = score
                best = c
        if best is None:
            return None
        # require it to look reasonably like labels
        return best if best_score >= 0.55 else None

    # --- Determine if table seems transposed ---
    def _looks_transposed(d: pd.DataFrame) -> bool:
        if d.shape[0] < 5 or d.shape[1] < 5:
            return False
        # If the first column is mostly numeric and the columns look like periods, it's probably transposed.
        try:
            s0 = d.iloc[:, 0].astype(str).map(_norm_text)
            nonnum0 = float((s0.map(lambda x: _clean_number(x) is None)).mean())
        except Exception:
            nonnum0 = 0.0
        # columns that look like periods (month/quarter-ish) even if obfuscated
        colish = [str(c) for c in d.columns[1:]]
        period_like = sum(1 for c in colish if re.search(r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|Sp|Jn|D)\s*\d", _norm_text(c), flags=re.I))
        return (nonnum0 < 0.35) and (period_like >= max(3, int(0.5 * len(colish))))

    candidates = [d0]
    if _looks_transposed(d0):
        candidates.append(d0.T.reset_index())

    for attempt_i, d in enumerate(candidates, start=1):
        label_col = _find_label_col(d)
        if not label_col:
            continue

        try:
            dd = d.copy()
            dd[label_col] = dd[label_col].astype(str).map(_norm_text)
            dd = dd.set_index(label_col)
        except Exception:
            continue

        # --- Find net profit row key (robust) ---
        idx_list = [str(x) for x in dd.index.tolist()]
        # 1) exact
        net_key = next((k for k in idx_list if re.search(r"net\s*profit", _norm_text(k), flags=re.I)), None)

        # 2) fuzzy (handles missing letters)
        if net_key is None:
            try:
                import difflib
                tgt = "netprofit"
                best_k = None
                best_r = 0.0
                for k in idx_list:
                    z = re.sub(r"[^a-z]", "", _norm_text(k).lower())
                    if not z:
                        continue
                    r = difflib.SequenceMatcher(None, z, tgt).ratio()
                    if r > best_r:
                        best_r = r
                        best_k = k
                if best_k is not None and best_r >= 0.62:
                    net_key = best_k
            except Exception:
                net_key = None

        # 3) positional fallback using EPS row
        if net_key is None:
            eps_pos = None
            for j, k in enumerate(idx_list):
                if re.search(r"\beps\b", _norm_text(k), flags=re.I):
                    eps_pos = j
                    break
            if eps_pos is not None and eps_pos > 0:
                net_key = idx_list[eps_pos - 1]
                logger.debug("[FUND] %s net_profit via EPS fallback: chosen_row=%s", sym, net_key)

        if net_key is None:
            logger.debug("[FUND] %s SKIP no_net_profit_row attempt=%d label_col=%s idx_sample=%s",
                         sym, attempt_i, label_col, idx_list[:12])
            continue

        net_row = dd.loc[net_key]
        if isinstance(net_row, pd.DataFrame):
            net_row = net_row.iloc[0]

        # --- Identify quarter columns (numeric content, keep original order) ---
        cols = [c for c in dd.columns]
        # drop columns that are clearly not quarters
        drop_pat = re.compile(r"(ttm|yoy|fy|year|annual|growth|ratio)", flags=re.I)
        cols = [c for c in cols if not drop_pat.search(_norm_text(c))]

        # Keep only columns that have a usable number in net_row; fallback to columns with numeric density
        qcols = [c for c in cols if _clean_number(net_row.get(c)) is not None]

        if len(qcols) < 5:
            qcols = []
            for c in cols:
                try:
                    s = dd[c].astype(str).map(_norm_text)
                    num_ct = int(s.map(lambda x: _clean_number(x) is not None).sum())
                    if num_ct >= max(2, int(0.25 * len(dd))):
                        qcols.append(c)
                except Exception:
                    continue

        if len(qcols) < 5:
            logger.debug("[FUND] %s SKIP too_few_quarter_cols attempt=%d label_col=%s cols=%s",
                         sym, attempt_i, label_col, dd.columns.tolist()[:12])
            continue

        curr_c = qcols[-1]
        prev_c = qcols[-5]  # same quarter last year

        curr_v = _clean_number(net_row.get(curr_c))
        prev_v = _clean_number(net_row.get(prev_c))

        if curr_v is None or prev_v in (None, 0, 0.0):
            logger.debug("[FUND] %s SKIP curr/prev missing_or_zero curr=%s(%s) prev=%s(%s)",
                         sym, curr_c, curr_v, prev_c, prev_v)
            continue

        yoy = (float(curr_v) - float(prev_v)) / float(prev_v) * 100.0
        return _out(
            "ok",
            yoy_net_profit_growth_pct=float(yoy),
            curr_q_label=str(curr_c),
            prev_yoy_q_label=str(prev_c),
            curr_net_profit=float(curr_v),
            prev_yoy_net_profit=float(prev_v),
            net_row_key=str(net_key),
            label_col=str(label_col),
        )

    # If we got here, nothing worked.
    logger.debug("[FUND] %s SKIP no_label_or_net_profit cols=%s", sym, d0.columns.tolist()[:12])
    return _out("no_data", note="net_profit_not_found")


def _utcnow() -> dt.datetime:
    return dt.datetime.now(tz=dt.timezone.utc)


def _cache_age_hours_from_meta(meta: dict) -> Optional[float]:
    try:
        s = (meta or {}).get("fetched_at_utc")
        if not s:
            return None
        t = dt.datetime.fromisoformat(str(s).replace("Z", "+00:00"))
        if t.tzinfo is None:
            t = t.replace(tzinfo=dt.timezone.utc)
        return (_utcnow() - t).total_seconds() / 3600.0
    except Exception:
        return None


def _cache_is_fresh(
    *,
    meta: Optional[dict],
    cache_path: Path,
    ttl_hours: float,
    want_from: str,
    want_to: str,
) -> bool:
    """Freshness policy (simple + effective):
    - New cache schema: must match requested from/to dates, and be newer than ttl_hours.
    - Legacy cache: use file mtime as proxy, and require it to be newer than ttl_hours.
    """
    ttl_hours = float(ttl_hours) if ttl_hours is not None else float(DEFAULT_OHLC_CACHE_TTL_HOURS)
    ttl_hours = max(0.0, ttl_hours)

    if meta and isinstance(meta, dict):
        if str(meta.get("fromDate", "")) != str(want_from) or str(meta.get("toDate", "")) != str(want_to):
            return False
        age_h = _cache_age_hours_from_meta(meta)
        return bool(age_h is not None and age_h <= ttl_hours)

    # legacy cache
    try:
        mtime = dt.datetime.fromtimestamp(cache_path.stat().st_mtime, tz=dt.timezone.utc)
        age_h = (_utcnow() - mtime).total_seconds() / 3600.0
        return bool(age_h <= ttl_hours)
    except Exception:
        return False


def _fund_flag_from_growth(yoy_pct: Optional[float]) -> str:
    if yoy_pct is None or (isinstance(yoy_pct, float) and not np.isfinite(yoy_pct)):
        return "NA"
    if float(yoy_pct) <= FUND_YOY_VETO_PCT:
        return "VETO"
    if float(yoy_pct) < FUND_YOY_CAUTION_PCT:
        return "CAUTION"
    return "OK"


def _as_float(x) -> Optional[float]:
    try:
        if x is None:
            return None
        s = str(x).strip()
        if not s or s in ("-", "—", "NA", "N/A"):
            return None
        s = s.replace(",", "")
        return float(s)
    except Exception:
        return None



def _extract_quarterly_results_table_html(html: str) -> Optional[str]:
    """Return HTML for the *Quarterly Results* table (consolidated) if present.

    Notes:
    - We keep BeautifulSoup optional. If it's missing, we just return None and fall back to read_html(all tables).
    """
    if not html:
        return None
    if BeautifulSoup is None:  # optional dependency
        return None
    try:
        soup = BeautifulSoup(html, "html.parser")

        # Preferred: stable section id + data-result-table wrapper.
        sec = soup.find("section", id="quarters")
        if sec:
            holder = sec.find(attrs={"data-result-table": True})
            if holder:
                tbl = holder.find("table")
                if tbl:
                    return str(tbl)

        # Fallback: find heading and take the next table.
        h = soup.find(lambda t: getattr(t, "name", None) in ("h2", "h3") and _norm_text(t.get_text()).lower() == "quarterly results")
        if h:
            tbl = h.find_next("table")
            if tbl:
                return str(tbl)

        return None
    except Exception:
        return None


def _tickertape_headers() -> Dict[str, str]:
    # Based on public Tickertape API usage patterns.
    return {
        "Accept": "application/json, text/plain, */*",
        "accept-version": "7.9.0",
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36",
        "Referer": "https://www.tickertape.in/",
    }


def _tickertape_search_sid(symbol: str, sess: requests.Session, logger: logging.Logger) -> Optional[str]:
    sym = symbol.strip().upper()
    try:
        r = sess.get(
            "https://api.tickertape.in/search",
            params={"text": sym, "types": "stock", "pageNumber": 0},
            headers=_tickertape_headers(),
            timeout=25,
        )
        if r.status_code != 200:
            logger.debug("[FUND][TT] %s search http=%s", sym, r.status_code)
            return None
        js = r.json() if r.content else {}
        items = (((js or {}).get("data") or {}).get("items") or [])
        if not isinstance(items, list) or not items:
            return None

        # Prefer exact ticker match when possible.
        for it in items:
            if isinstance(it, dict) and str(it.get("ticker", "")).upper() == sym and it.get("sid"):
                return str(it["sid"])
        # Otherwise first hit.
        sid = items[0].get("sid") if isinstance(items[0], dict) else None
        return str(sid) if sid else None
    except Exception as e:
        logger.debug("[FUND][TT] %s search failed: %s", sym, e)
        return None


def _normalize_period_label(x: object) -> Optional[str]:
    s = _norm_text(x)
    if not s:
        return None
    # already like "Sep 2025"
    if _QCOL_RE.match(s):
        return s
    # ISO like 2025-09-30
    m = re.match(r"^(\d{4})-(\d{2})", s)
    if m:
        y = int(m.group(1))
        mo = int(m.group(2))
        inv = {v: k for k, v in _MONTH.items()}
        mon = inv.get(mo)
        if mon:
            return f"{mon} {y}"
    # "Q1 FY25" style -> not supported here
    return s


def _tickertape_income_to_quarters_df(df: pd.DataFrame, logger: logging.Logger) -> Optional[pd.DataFrame]:
    """Convert Tickertape income DF to a Screener-like wide DF (Metric x quarters)."""
    if df is None or df.empty:
        return None

    # If it already looks wide with quarter columns, keep it.
    cols = [_flatten_col(c) for c in df.columns]
    if any(_qkey(c) for c in cols):
        out = df.copy()
        out.columns = cols
        return out

    # If long-form: name/period/value
    lowcols = {str(c).lower(): c for c in df.columns}
    if all(k in lowcols for k in ("name", "period", "value")):
        try:
            wide = df.pivot_table(index=lowcols["name"], columns=lowcols["period"], values=lowcols["value"], aggfunc="first")
            wide = wide.reset_index().rename(columns={lowcols["name"]: "Metric"})
            wide.columns = [_flatten_col(c) for c in wide.columns]
            return wide
        except Exception:
            pass

    # Nested values list (common in API payloads).
    name_col = None
    for cand in ("name", "label", "title", "metric", "item"):
        if cand in lowcols:
            name_col = lowcols[cand]
            break

    # detect list-ish column
    values_col = None
    for c in df.columns:
        try:
            sample = df[c].dropna().head(3).tolist()
            if sample and any(isinstance(v, (list, tuple)) for v in sample):
                values_col = c
                break
        except Exception:
            continue

    if not name_col or not values_col:
        return None

    rows = []
    for _, r in df.iterrows():
        metric = _norm_text(r.get(name_col))
        if not metric:
            continue
        vals = r.get(values_col)
        if not isinstance(vals, (list, tuple)):
            continue
        pm: Dict[str, object] = {}
        for it in vals:
            if isinstance(it, dict):
                per = _normalize_period_label(it.get("period") or it.get("label") or it.get("name") or it.get("key"))
                val = it.get("value")
                if per is None:
                    continue
                pm[per] = val
        if pm:
            rows.append({"Metric": metric, **pm})

    if not rows:
        return None
    wide = pd.DataFrame(rows)
    wide.columns = [_flatten_col(c) for c in wide.columns]
    return wide


def _tickertape_profit_growth(symbol: str, sess: requests.Session, logger: logging.Logger) -> Dict[str, object]:
    sym = symbol.strip().upper()
    sid = _tickertape_search_sid(sym, sess, logger)
    if not sid:
        return {"symbol": sym, "status": "no_data", "note": "tickertape_no_sid", "source": "tickertape"}

    try:
        r = sess.get(
            f"https://api.tickertape.in/stocks/financials/income/{sid}/interim/normal",
            params={"count": 12},
            headers=_tickertape_headers(),
            timeout=25,
        )
        if r.status_code != 200:
            return {"symbol": sym, "status": "http_error", "http_status": int(r.status_code), "note": "tickertape_income_http", "source": "tickertape"}
        js = r.json() if r.content else {}
        df = pd.DataFrame((js or {}).get("data") or [])
        qdf = _tickertape_income_to_quarters_df(df, logger)
        if qdf is None or qdf.empty:
            return {"symbol": sym, "status": "no_data", "note": "tickertape_income_unparseable", "source": "tickertape"}

        # Reuse the screener net-profit parser so behavior stays consistent.
        parsed = parse_net_profit_yoy_from_quarters_df(qdf, sym, logger)
        parsed["source"] = "tickertape"
        return parsed
    except Exception as e:
        return {"symbol": sym, "status": "parse_error", "note": f"tickertape_exception:{e}", "source": "tickertape"}


def fetch_profit_growth_screener(
    *,
    symbol: str,
    cache_dir: Path,
    ttl_hours: float = DEFAULT_FUND_CACHE_TTL_HOURS,
    session: Optional[requests.Session] = None,
    sleep_s: float = 0.4,
    html_override: Optional[str] = None,
    force_refresh: bool = False,
) -> Dict[str, object]:
    """Fetch YoY Net Profit growth.

    Root-cause fix:
    - Some tickers (e.g., WELINV) have EMPTY quarterly tables on the /consolidated/ page,
      but the *non-consolidated* page has full quarterly data.
    - So we try BOTH endpoints with plain requests (fast, stable), before any other fallback.

    Order:
      1) Screener consolidated page (best when available)
      2) Screener normal page (critical fallback for many tickers)
      3) Optional: curl fallback (FUND_USE_CURL=1) when manual curl works but requests doesn't
      4) Tickertape API fallback (optional, FUND_FALLBACK_TICKERTAPE=0 to disable)

    Output keys (stable):
      - status: ok | no_data | http_error | parse_error
      - yoy_net_profit_growth_pct, curr_q_label, prev_yoy_q_label, curr_net_profit, prev_yoy_net_profit
      - source: screener_requests | tickertape
    """

    sym = str(symbol).strip().upper()
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{sym}.json"

    want_ttl = float(ttl_hours) if ttl_hours is not None else float(DEFAULT_FUND_CACHE_TTL_HOURS)
    want_ttl = max(0.0, want_ttl)

    def _default_out(status: str, **kw) -> Dict[str, object]:
        o = {
            "symbol": sym,
            "status": status,
            "yoy_net_profit_growth_pct": None,
            "curr_q_label": None,
            "prev_yoy_q_label": None,
            "curr_net_profit": None,
            "prev_yoy_net_profit": None,
            "source": kw.pop("source", "screener_requests"),
            "note": kw.pop("note", None),
        }
        o.update(kw)
        return o

    def _cache_write(data: Dict[str, object], http_status: Optional[int], url: str):
        try:
            meta = {
                "schema_version": 2,
                "fetched_at_utc": _utcnow().isoformat(),
                "ttl_hours": float(want_ttl),
                "url": url,
                "http_status": http_status,
                "source": data.get("source"),
            }
            cache_obj = {"_meta": meta, "data": data}
            cache_path.write_text(json.dumps(cache_obj), encoding="utf-8")
            LOG.debug("[FUND] %s cached -> %s (status=%s source=%s url=%s)", sym, str(cache_path), data.get("status"), data.get("source"), url)
        except Exception as e:
            LOG.debug("[FUND] %s cache write failed: %s", sym, e)

    # 1) cache read (backward compatible)
    if html_override is not None:
        LOG.debug("[FUND] %s cache bypass (html_override provided)", sym)
    elif force_refresh:
        LOG.debug("[FUND] %s cache bypass (force_refresh)", sym)
    elif cache_path.exists():
        try:
            js = json.loads(cache_path.read_text(encoding="utf-8", errors="replace"))
            if isinstance(js, dict) and "data" in js:
                meta = js.get("_meta") or {}
                age_h = _cache_age_hours_from_meta(meta)
                if age_h is not None and age_h <= want_ttl:
                    data = js.get("data")
                    if isinstance(data, dict):
                        LOG.debug("[FUND] %s cache hit -> %s age_h=%.2f status=%s source=%s yoy=%s",
                                  sym, str(cache_path), age_h, data.get("status"), data.get("source"), data.get("yoy_net_profit_growth_pct"))
                        return data
            # legacy: file is directly the data dict
            if isinstance(js, dict) and "status" in js and "yoy_net_profit_growth_pct" in js:
                LOG.debug("[FUND] %s legacy cache hit (no meta)", sym)
                return js
        except Exception as e:
            LOG.debug("[FUND] %s cache read failed: %s", sym, e)

    sess = session or requests.Session()

    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": SCREENER_BASE,
    }

    def _parse_from_html(html: str) -> Dict[str, object]:
        """Parse YoY Net Profit growth from a Screener company page HTML.

        We intentionally avoid brute-forcing *every* HTML table because it creates noisy debug logs
        and slows scans on large watchlists. Instead we:
          1) Prefer the extracted 'Quarterly Results' table (fast path).
          2) If that table is present but has *no quarter columns*, treat it as blocked/empty
             and return quickly so the caller can try the non-consolidated URL.
          3) Otherwise, score tables by how 'quarter-like' they are and only parse the top few.
        """
        if not html:
            return _default_out("no_data", note="empty_html")

        extracted = None
        try:
            extracted = _extract_quarterly_results_table_html(html)
        except Exception:
            extracted = None

        extracted_tables: List[pd.DataFrame] = []
        if extracted:
            try:
                extracted_tables = pd.read_html(io.StringIO(extracted))
            except Exception:
                extracted_tables = []

        # If the quarterly table is present but quarter columns are missing, this is the
        # exact symptom we see on bot-blocked consolidated pages (e.g. WELINV).
        for t in extracted_tables:
            try:
                cols_flat = [_flatten_col(c) for c in t.columns]
                qcols = [c for c in cols_flat if _qkey(c) is not None]
                first_col = _norm_text(cols_flat[0]) if cols_flat else ""
                if first_col.lower().startswith("sales") and len(qcols) < 4:
                    return _default_out("no_data", note="quarterly_table_blocked_or_empty", cols_preview=cols_flat[:12])
            except Exception:
                continue

        all_tables: List[pd.DataFrame] = []
        try:
            all_tables = pd.read_html(io.StringIO(html))
        except Exception:
            all_tables = []

        candidates = extracted_tables + all_tables
        if not candidates:
            return _default_out("no_data", note="no_tables_found")

        def _qcount_first_col(df: pd.DataFrame) -> int:
            try:
                s = df.iloc[:, 0].astype(str).map(_norm_text)
                return int(sum(1 for v in s if _qkey(v) is not None))
            except Exception:
                return 0

        def _table_score(df: pd.DataFrame) -> int:
            try:
                cols_flat = [_flatten_col(c) for c in df.columns]
                qcols = sum(1 for c in cols_flat if _qkey(c) is not None)
                qrows = _qcount_first_col(df)
                score = 10 * max(qcols, qrows)

                # bonus for the typical metric labels
                try:
                    lab = df.iloc[:, 0].astype(str).map(_norm_text).str.lower()
                    if lab.str.contains(r"\bsales\b").any():
                        score += 5
                    if lab.str.contains(r"net\s*profit|profit\s*after\s*tax|pat").any():
                        score += 5
                    if lab.str.contains(r"\beps\b").any():
                        score += 2
                except Exception:
                    pass

                # Prefer wider (more columns), but cap to avoid biasing giant tables
                score += min(5, int(df.shape[1] // 2))
                return int(score)
            except Exception:
                return -1

        scored: List[Tuple[int, pd.DataFrame]] = []
        for t in candidates:
            if t is None or t.empty:
                continue
            # quick filter: ignore tiny tables that cannot hold quarterly + yoy
            if t.shape[0] < 4 and t.shape[1] < 4:
                continue
            s = _table_score(t)
            if s <= 0:
                continue
            scored.append((s, t))

        if not scored:
            return _default_out("no_data", note="no_candidate_tables")

        scored.sort(key=lambda x: x[0], reverse=True)
        top = [t for _, t in scored[:5]]

        best: Optional[Dict[str, object]] = None
        for t in top:
            try:
                parsed = parse_net_profit_yoy_from_quarters_df(t, sym, LOG)
            except Exception as e:
                LOG.debug("[FUND] %s parse error: %s", sym, e)
                continue

            if parsed.get("status") == "ok" and parsed.get("yoy_net_profit_growth_pct") is not None:
                return parsed

            # keep the most informative failure
            if best is None or (best.get("note") in (None, "", "no_label_or_net_profit") and parsed.get("note") not in (None, "", "no_label_or_net_profit")):
                best = parsed

        return best or _default_out("no_data", note="net_profit_not_found_in_top_tables")

    def _fetch_url(url: str) -> Tuple[Optional[int], str, str]:
        """Returns (http_status, title, html)."""
        tries = 3
        http_status: Optional[int] = None
        title = ""
        html = ""
        for attempt in range(1, tries + 1):
            try:
                r = sess.get(url, headers=headers, timeout=30)
                http_status = int(r.status_code)
                html = r.text or ""
                m = re.search(r"<title>(.*?)</title>", html, flags=re.I | re.S)
                title = (_norm_text(m.group(1)) if m else "")[:120]
                LOG.debug("[FUND] %s GET %s HTTP=%s attempt=%d/%d title=%s len=%d", sym, url, http_status, attempt, tries, title, len(html))
                if http_status == 429 and attempt < tries:
                    time.sleep(attempt * max(0.25, sleep_s))
                    continue
                break
            except Exception as e:
                LOG.debug("[FUND] %s HTTP fetch error url=%s attempt=%d/%d: %s", sym, url, attempt, tries, e)
                if attempt >= tries:
                    http_status = None
                    title = ""
                    html = ""
                    break
                time.sleep(attempt * max(0.25, sleep_s))

        # Optional fallback: use curl (different TLS/browser fingerprint). Helpful when manual curl works but requests doesn't.
        if (not html or http_status != 200) and os.getenv("FUND_USE_CURL", "0").strip() == "1":
            if shutil.which("curl"):
                try:
                    cmd = [
                        "curl", "-L", "-k", "-s",
                        "-A", headers.get("User-Agent", "Mozilla/5.0"),
                        "-H", f"Accept: {headers.get('Accept','*/*')}",
                        "-H", f"Accept-Language: {headers.get('Accept-Language','en-US,en;q=0.9')}",
                        "-H", f"Referer: {headers.get('Referer', SCREENER_BASE)}",
                        "-w", "\nCURL_HTTP_CODE:%{http_code}\n",
                        url,
                    ]
                    cp = subprocess.run(cmd, capture_output=True, text=True, timeout=45)
                    out = (cp.stdout or "")
                    mcode = re.search(r"\nCURL_HTTP_CODE:(\d{3})\n\s*$", out)
                    if mcode:
                        http_status = int(mcode.group(1))
                        out = re.sub(r"\nCURL_HTTP_CODE:\d{3}\n\s*$", "", out)
                    html = out
                    m = re.search(r"<title>(.*?)</title>", html, flags=re.I | re.S)
                    title = (_norm_text(m.group(1)) if m else "")[:120]
                    LOG.debug("[FUND] %s curl GET %s HTTP=%s len=%d", sym, url, http_status, len(html))
                except Exception as e:
                    LOG.debug("[FUND] %s curl fallback failed url=%s: %s", sym, url, e)

        return http_status, title, html

    # 2) HTML source selection
    if html_override is not None:
        html = str(html_override)
        out = _parse_from_html(html)
        if not out.get("source"):
            out["source"] = "screener_requests"
        out.setdefault("title", "html_override")
        if out.get("status") == "ok":
            _cache_write(out, 200, "html_override")
        return out

    url_candidates = [
        f"{SCREENER_BASE}/company/{quote(sym)}/consolidated/",
        f"{SCREENER_BASE}/company/{quote(sym)}/",
    ]

    last_fail: Optional[Dict[str, object]] = None

    for url in url_candidates:
        http_status, title, html = _fetch_url(url)
        if not html or http_status != 200:
            last_fail = _default_out("http_error", http_status=http_status, note="screener_http_error", title=title, url=url)
            continue

        out = _parse_from_html(html)
        if not out.get("source"):
            out["source"] = "screener_requests"
        out.setdefault("title", title)
        out["url"] = url
        out["http_status"] = http_status

        if out.get("status") == "ok":
            _cache_write(out, http_status, url)
            time.sleep(max(0.0, float(sleep_s)))
            return out

        last_fail = out
        LOG.debug("[FUND] %s parse failed on url=%s status=%s note=%s", sym, url, out.get("status"), out.get("note"))

        # Dump HTML for manual inspection if enabled
        if os.getenv("FUND_DUMP_HTML_ON_FAIL", "1").strip() != "0":
            try:
                dump_dir = cache_dir / "_html"
                dump_dir.mkdir(parents=True, exist_ok=True)
                tag = "consolidated" if url.endswith("/consolidated/") else "standalone"
                dump_path = dump_dir / f"{sym}__{tag}.html"
                dump_path.write_text(html, encoding="utf-8")
                LOG.debug("[FUND] %s dumped html -> %s", sym, str(dump_path))
            except Exception as e:
                LOG.debug("[FUND] %s html dump failed: %s", sym, e)

    # 3) fallback source: Tickertape API (optional)
    if os.getenv("FUND_FALLBACK_TICKERTAPE", "1").strip() != "0":
        tt = _tickertape_profit_growth(sym, sess, LOG)
        if tt.get("status") == "ok":
            tt["source"] = "tickertape"
            _cache_write(tt, 200, "tickertape")
            time.sleep(max(0.0, float(sleep_s)))
            return tt

        note = (last_fail or {}).get("note") or ""
        last_fail = _default_out(
            "no_data",
            source="screener_requests",
            note=f"{note}; tickertape:{tt.get('note') or tt.get('status')}",
        )

    out_final = last_fail or _default_out("no_data", note="no_usable_source")
    if not out_final.get("source"):
        out_final["source"] = "screener_requests"
    _cache_write(out_final, out_final.get("http_status"), out_final.get("url") or "unknown")
    time.sleep(max(0.0, float(sleep_s)))
    return out_final

# -----------------------------
# Logging
# -----------------------------
LOG = logging.getLogger("swing_scan")


def setup_logging(level: str, log_file: Optional[str]) -> None:
    LOG.setLevel(getattr(logging, level.upper(), logging.INFO))
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")
    handlers: List[logging.Handler] = []

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    handlers.append(sh)

    if log_file:
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        handlers.append(fh)

    for h in handlers:
        h.setLevel(getattr(logging, level.upper(), logging.INFO))
        LOG.addHandler(h)

    # Quiet urllib3 unless DEBUG
    if getattr(logging, level.upper(), logging.INFO) <= logging.DEBUG:
        logging.getLogger("urllib3").setLevel(logging.DEBUG)
    else:
        logging.getLogger("urllib3").setLevel(logging.WARNING)


# -----------------------------
# Dhan Auth
# -----------------------------
_JWT_RE = re.compile(r"^[A-Za-z0-9_\-]+=*\.[A-Za-z0-9_\-]+=*\.[A-Za-z0-9_\-]+=*$")

def _looks_like_jwt(s: str) -> bool:
    s = (s or "").strip()
    return bool(_JWT_RE.match(s))

def _normalize_token(raw: str) -> str:
    """Handle common token formats.

    - Some envs store base64 of "clientid:token". Dhan expects only the access token.
    - If we see a single ':' and the full string is not a JWT, use the RHS.
    """
    raw = (raw or "").strip()
    if not raw:
        return raw
    if ':' in raw and not _looks_like_jwt(raw):
        # Common pattern: "clientid:accesstoken"
        raw = raw.split(':', 1)[1].strip()
    return raw

def _decode_token_b64(tok_b64: str) -> str:
    """Decode base64 if it looks like base64, else return as-is."""
    tok_b64 = (tok_b64 or "").strip()
    if not tok_b64:
        return ""
    try:
        return base64.b64decode(tok_b64).decode("utf-8", errors="replace").strip()
    except Exception:
        return tok_b64


def get_dhan_token() -> tuple[str, str]:
    """Returns (token, source_env).

    Prefer DHAN_ACCESS_TOKEN (raw) over DHAN_TOKEN_B64 to avoid the common footgun where an old
    DHAN_TOKEN_B64 remains set and silently overrides a fresh token.

    Also normalizes the common "clientid:token" format to just "token".
    """
    if os.getenv("DHAN_ACCESS_TOKEN"):
        return _normalize_token(os.getenv("DHAN_ACCESS_TOKEN", "")), "DHAN_ACCESS_TOKEN"
    if os.getenv("DHAN_TOKEN"):
        return _normalize_token(os.getenv("DHAN_TOKEN", "")), "DHAN_TOKEN"
    if os.getenv("DHAN_TOKEN_B64"):
        return _normalize_token(_decode_token_b64(os.getenv("DHAN_TOKEN_B64", ""))), "DHAN_TOKEN_B64"
    raise RuntimeError("Missing Dhan token. Set DHAN_ACCESS_TOKEN (preferred) or DHAN_TOKEN or DHAN_TOKEN_B64.")




# -----------------------------
# Token helpers (expiry visibility + optional renew)
# -----------------------------
def get_dhan_client_id() -> Optional[str]:
    """
    Needed only if you want to auto-renew a *web-generated* token via /v2/RenewToken.
    Set one of these:
      - DHAN_CLIENT_ID
      - DHAN_CLIENTID
      - DHAN_DHANCLIENTID
    """
    for k in ("DHAN_CLIENT_ID", "DHAN_CLIENTID", "DHAN_DHANCLIENTID"):
        v = os.getenv(k)
        if v and v.strip():
            return v.strip()
    return None


def _jwt_exp_utc(token: str) -> Optional[dt.datetime]:
    """
    Best-effort decode JWT expiry without verifying signature.
    This is only for logging visibility (helps debug DH-901).
    """
    try:
        parts = token.split(".")
        if len(parts) < 2:
            return None
        payload_b64 = parts[1] + "==="
        payload = json.loads(base64.urlsafe_b64decode(payload_b64.encode("utf-8")).decode("utf-8"))
        exp = payload.get("exp")
        if exp:
            return dt.datetime.fromtimestamp(int(exp), tz=dt.timezone.utc)
        return None
    except Exception:
        return None


def renew_token_if_possible(token: str) -> Optional[str]:
    """Attempt to renew token via Dhan /v2/RenewToken.

    This only works for web-generated tokens and requires a client id header.
    We keep this best-effort and purely optional, because most users just paste a fresh token.

    Returns a *new* token string if successful, else None.
    """
    client_id = get_dhan_client_id()
    if not client_id:
        return None

    url = "https://api.dhan.co/v2/RenewToken"
    headers = {"access-token": str(token).strip(), "dhanClientId": str(client_id).strip()}
    try:
        LOG.warning("Attempting token renew via /v2/RenewToken (dhanClientId=%s)...", client_id)
        r = requests.post(url, headers=headers, timeout=30)
        if r.status_code != 200:
            LOG.warning("Token renew failed [HTTP %s]: %s", r.status_code, r.text)
            return None

        js = r.json() if r.content else {}
        new_tok = (
            js.get("accessToken")
            or js.get("token")
            or js.get("access-token")
            or (js.get("data") or {}).get("accessToken")
        )
        if not new_tok:
            LOG.warning("Token renew response did not contain accessToken. Raw=%s", js)
            return None

        new_tok = str(new_tok).strip()
        if not new_tok:
            return None

        LOG.warning("Token renewed successfully.")
        return new_tok
    except Exception as e:
        LOG.warning("Token renew exception: %s", e)
        return None


def token_smoke_test(token: str) -> bool:
    """
    Very lightweight validation call. Holdings endpoint requires only access-token in docs. 
    If you get 401 here, your token is invalid/expired.
    """
    url = "https://api.dhan.co/v2/holdings"
    try:
        r = requests.get(url, headers={"access-token": token, "Content-Type": "application/json"}, timeout=20)
        if r.status_code == 200:
            return True
        LOG.warning("Token smoke-test failed [HTTP %s]: %s", r.status_code, r.text)
        return False
    except Exception as e:
        LOG.warning("Token smoke-test exception: %s", e)
        return False

# -----------------------------
# Master resolution
# -----------------------------
@dataclasses.dataclass(frozen=True)
class Instrument:
    symbol: str
    security_id: str
    exchange_segment: str  # NSE_EQ / etc
    instrument: str        # EQUITY / etc (Dhan historical expects "EQUITY" for segment E)
    expiry_code: int = 0
    exch_instrument_type: str = "EQUITY"  # ETF/REIT/INVIT/EQUITY etc.
    series: str = "EQ"                   # EQ/RR/SM/ST etc.
    custom_symbol: str = ""              # Excel col H
    asset_bucket: str = "EQUITY"         # EQUITY/ETF/REIT/SME/OTHER


def load_master(master_csv: str) -> pd.DataFrame:
    """
    Load Dhan master CSV.

    Excel column mapping reference:
    - Col F => SEM_TRADING_SYMBOL
    - Col H => SEM_CUSTOM_SYMBOL (often includes ETF name)
    """
    usecols = [
        "SEM_EXM_EXCH_ID",
        "SEM_SEGMENT",
        "SEM_SMST_SECURITY_ID",
        "SEM_INSTRUMENT_NAME",
        "SEM_EXCH_INSTRUMENT_TYPE",
        "SEM_EXPIRY_CODE",
        "SEM_TRADING_SYMBOL",
        "SEM_CUSTOM_SYMBOL",
        "SM_SYMBOL_NAME",
        "SEM_SERIES",
    ]
    df = pd.read_csv(master_csv, usecols=usecols, low_memory=False, dtype=str).fillna("")
    for c in usecols:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()
    df["SEM_EXPIRY_CODE"] = pd.to_numeric(df["SEM_EXPIRY_CODE"], errors="coerce").fillna(0).astype(int)
    return df


def _exchange_segment(exch: str, instr: str) -> str:
    # For E-segment symbols (equity/ETF/REIT), Dhan uses NSE_EQ
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


def bucket_asset(exch_instrument_type: str, series: str, custom_symbol: str) -> str:
    """
    Uses:
    - SEM_EXCH_INSTRUMENT_TYPE (most reliable)
    - SEM_CUSTOM_SYMBOL (Excel col H, often contains 'ETF')
    - SEM_SERIES (SM/ST => SME, RR => REIT series)
    """
    t = (exch_instrument_type or "").strip().upper()
    s = (series or "").strip().upper()
    c = (custom_symbol or "").strip().upper()

    if "ETF" in t or " ETF" in c or c.endswith("ETF"):
        return "ETF"
    if t in ("REIT", "INVIT"):
        return "REIT"
    if s in ("SM", "ST"):
        return "SME"
    if t and t not in ("EQUITY", "EQ", "ES", "STOCK"):
        return t
    return "EQUITY"


def resolve_symbol(
    sym: str,
    master: pd.DataFrame,
    prefer_exchange: str = "NSE",
    prefer_instrument: str = "EQUITY",
) -> Optional[Instrument]:
    s = str(sym).strip()
    if not s:
        return None

    # Allow passing a security id directly
    if s.isdigit() and len(s) >= 3:
        return Instrument(
            symbol=s,
            security_id=s,
            exchange_segment="NSE_EQ",
            instrument="EQUITY",
            expiry_code=0,
            exch_instrument_type="EQUITY",
            series="EQ",
            custom_symbol="",
            asset_bucket="EQUITY",
        )

    m = master
    cand = m[m["SEM_TRADING_SYMBOL"].str.upper() == s.upper()]
    if not cand.empty:
        cand1 = cand[(cand["SEM_EXM_EXCH_ID"] == prefer_exchange) & (cand["SEM_INSTRUMENT_NAME"] == prefer_instrument)]
        if cand1.empty:
            cand1 = cand
        row = cand1.iloc[0]
        instr = row.get("SEM_INSTRUMENT_NAME", "EQUITY")
        exch = row.get("SEM_EXM_EXCH_ID", prefer_exchange)
        eit = row.get("SEM_EXCH_INSTRUMENT_TYPE", "EQUITY")
        series = row.get("SEM_SERIES", "")
        custom = row.get("SEM_CUSTOM_SYMBOL", "")
        bucket = bucket_asset(eit, series, custom)
        return Instrument(
            symbol=s,
            security_id=row["SEM_SMST_SECURITY_ID"],
            exchange_segment=_exchange_segment(exch, instr),
            instrument=instr,
            expiry_code=int(row.get("SEM_EXPIRY_CODE", 0) or 0),
            exch_instrument_type=eit,
            series=series,
            custom_symbol=custom,
            asset_bucket=bucket,
        )

    # fallback fuzzy match
    mask = (
        m["SEM_CUSTOM_SYMBOL"].str.upper().str.contains(s.upper(), na=False)
        | m["SM_SYMBOL_NAME"].str.upper().str.contains(s.upper(), na=False)
    )
    cand = m[mask]
    if cand.empty:
        return None
    cand = cand[(cand["SEM_EXM_EXCH_ID"] == prefer_exchange) & (cand["SEM_INSTRUMENT_NAME"] == prefer_instrument)]
    if cand.empty:
        return None
    row = cand.iloc[0]
    instr = row.get("SEM_INSTRUMENT_NAME", "EQUITY")
    exch = row.get("SEM_EXM_EXCH_ID", prefer_exchange)
    eit = row.get("SEM_EXCH_INSTRUMENT_TYPE", "EQUITY")
    series = row.get("SEM_SERIES", "")
    custom = row.get("SEM_CUSTOM_SYMBOL", "")
    bucket = bucket_asset(eit, series, custom)
    return Instrument(
        symbol=s,
        security_id=row["SEM_SMST_SECURITY_ID"],
        exchange_segment=_exchange_segment(exch, instr),
        instrument=instr,
        expiry_code=int(row.get("SEM_EXPIRY_CODE", 0) or 0),
        exch_instrument_type=eit,
        series=series,
        custom_symbol=custom,
        asset_bucket=bucket,
    )


# -----------------------------
# Indicators
# -----------------------------
EPS = 1e-9


def ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False, min_periods=span).mean()


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0)
    down = (-delta).clip(lower=0)
    ma_up = up.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    ma_down = down.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    rs = ma_up / (ma_down + EPS)
    return 100 - (100 / (1 + rs))


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def bollinger(close: pd.Series, n: int = 20, k: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
    ma = close.rolling(n, min_periods=n).mean()
    sd = close.rolling(n, min_periods=n).std(ddof=0)
    upper = ma + k * sd
    lower = ma - k * sd
    return ma, upper, lower


def adx(df: pd.DataFrame, n: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr = pd.concat([(high - low).abs(), (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
    atr_n = tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()

    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1 / n, adjust=False, min_periods=n).mean() / (atr_n + EPS)
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1 / n, adjust=False, min_periods=n).mean() / (atr_n + EPS)

    dx = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di + EPS)).fillna(0.0)
    return dx.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def macd_hist(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    fast_ema = ema(close, fast)
    slow_ema = ema(close, slow)
    macd_line = fast_ema - slow_ema
    sig = ema(macd_line, signal)
    return macd_line - sig


def bullish_engulfing(df: pd.DataFrame) -> bool:
    if len(df) < 2:
        return False
    o1, c1 = df["open"].iloc[-2], df["close"].iloc[-2]
    o2, c2 = df["open"].iloc[-1], df["close"].iloc[-1]
    # prev red, current green and current body engulfs prev body
    return (c1 < o1) and (c2 > o2) and (c2 >= o1) and (o2 <= c1)


def hammer(df: pd.DataFrame) -> bool:
    if len(df) < 1:
        return False
    o, h, l, c = df["open"].iloc[-1], df["high"].iloc[-1], df["low"].iloc[-1], df["close"].iloc[-1]
    body = abs(c - o)
    lower = min(c, o) - l
    upper = h - max(c, o)
    return (lower > 1.5 * (body + EPS)) and (upper <= 0.6 * (body + EPS))


def weekly_from_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Resample daily OHLCV to weekly (Fri close)."""
    if df.empty:
        return df
    w = df.resample("W-FRI").agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
    )
    return w.dropna()


# -----------------------------
# VRVP (daily approximation, Option A) + Option C weights
# -----------------------------
VRVP_LOOKBACK_DAYS = 120
VRVP_BIN_PCT = 0.0025
VRVP_BIN_ATR_MULT = 0.50
VRVP_LVN_Q = 0.25
VRVP_HVN_Q = 0.75
VRVP_OVERHEAD_ATR = 2.0

# -----------------------------
# Cost-awareness (rough delivery round-trip) for small accounts
# -----------------------------
APPROX_COST_PCT = 0.0035      # ~0.35% round-trip estimate (taxes/fees/slippage). Tune as needed.
MIN_TARGET_PCT_FLOOR = 0.01   # minimum 1% target for swing delivery (noise/gaps).
MIN_TARGET_COST_MULT = 3.0    # require target at least 3× estimated costs.


def compute_vrvp_features(df: pd.DataFrame) -> Dict[str, float]:
    if df is None or df.empty or len(df) < 60:
        return {}

    look = df.tail(min(len(df), VRVP_LOOKBACK_DAYS))
    tp = (look["high"].astype(float) + look["low"].astype(float) + look["close"].astype(float)) / 3.0
    vol = look["volume"].astype(float).fillna(0.0)

    last_close = float(df["close"].iloc[-1]) if pd.notna(df["close"].iloc[-1]) else np.nan
    if not (np.isfinite(last_close) and last_close > 0):
        return {}

    atr14 = atr(df, 14)
    atr_abs = float(atr14.iloc[-1]) if pd.notna(atr14.iloc[-1]) else np.nan
    bin_size = max(last_close * VRVP_BIN_PCT, (atr_abs * VRVP_BIN_ATR_MULT) if np.isfinite(atr_abs) and atr_abs > 0 else 0.0)
    if not (np.isfinite(bin_size) and bin_size > 0):
        return {}

    tp_min, tp_max = float(tp.min()), float(tp.max())
    if not (np.isfinite(tp_min) and np.isfinite(tp_max) and tp_max > tp_min):
        return {"vrvp_bin_size": float(bin_size)}

    pad = 0.5 * bin_size
    bins = np.arange(tp_min - pad, tp_max + pad + bin_size, bin_size)
    if len(bins) < 8:
        return {"vrvp_bin_size": float(bin_size)}

    bidx = np.digitize(tp.to_numpy(), bins) - 1
    bidx = np.clip(bidx, 0, len(bins) - 2)
    vhist = np.bincount(bidx, weights=vol.to_numpy(), minlength=len(bins) - 1)
    if vhist.sum() <= 0:
        return {"vrvp_bin_size": float(bin_size)}

    centers = bins[:-1] + 0.5 * bin_size
    poc_i = int(np.argmax(vhist))
    poc_price = float(centers[poc_i])
    dist_to_poc = float((poc_price - last_close) / last_close)

    nz = vhist[vhist > 0]
    if len(nz) >= 10:
        lvn_thr = float(np.quantile(nz, VRVP_LVN_Q))
        hvn_thr = float(np.quantile(nz, VRVP_HVN_Q))
    else:
        med = float(np.median(nz)) if len(nz) else 0.0
        lvn_thr = med * 0.6
        hvn_thr = med * 1.4

    # POC support: below price and not too far (<= 1.2 ATR or <=3%)
    if np.isfinite(atr_abs) and atr_abs > 0:
        poc_support = float((last_close >= poc_price) and ((last_close - poc_price) <= 1.2 * atr_abs))
        end = None
    else:
        poc_support = float((last_close >= poc_price) and ((last_close - poc_price) / last_close <= 0.03))

    # Overhead corridor from resistance up to ~2 ATR
    high20 = df["high"].rolling(20, min_periods=20).max()
    res = float(high20.iloc[-1]) if pd.notna(high20.iloc[-1]) else last_close
    if np.isfinite(atr_abs) and atr_abs > 0:
        end = res + VRVP_OVERHEAD_ATR * atr_abs
    else:
        end = res * 1.06

    mask = (centers > res) & (centers <= end)
    corridor = vhist[mask]
    if corridor.size >= 3:
        lvn_overhead = float(np.max(corridor) <= lvn_thr)
        hvn_overhead_close = float(np.max(corridor) >= hvn_thr)
    else:
        lvn_overhead = 0.0
        hvn_overhead_close = 0.0

    return {
        "poc_price": poc_price,
        "dist_to_poc": dist_to_poc,
        "poc_support": poc_support,
        "lvn_overhead": lvn_overhead,
        "hvn_overhead_close": hvn_overhead_close,
        "vrvp_bin_size": float(bin_size),
    }


# -----------------------------
# Scoring
# -----------------------------
def safe_div(a: float, b: float) -> float:
    return a / (b + EPS)


def score_one(df: pd.DataFrame) -> Optional[Dict[str, float]]:
    if df is None or df.empty:
        return None
    if len(df) < 210:
        return None

    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    vol = df["volume"].astype(float).fillna(0.0)

    ema20 = ema(close, 20)
    ema50 = ema(close, 50)
    ema200 = ema(close, 200)

    trend_up = float(ema50.iloc[-1] > ema200.iloc[-1])
    above_ema50 = float(close.iloc[-1] > ema50.iloc[-1])
    above_ema20 = float(close.iloc[-1] > ema20.iloc[-1])

    # Resistance levels and proximity (no lookahead for breakout trigger)
    high20 = high.rolling(20, min_periods=20).max()
    high55 = high.rolling(55, min_periods=55).max()
    breakout_lvl_20d = float(high20.iloc[-1]) if pd.notna(high20.iloc[-1]) else np.nan
    breakout_lvl_55d = float(high55.iloc[-1]) if pd.notna(high55.iloc[-1]) else np.nan

    prox20 = float(safe_div((breakout_lvl_20d - close.iloc[-1]), breakout_lvl_20d)) if np.isfinite(breakout_lvl_20d) else np.nan
    prox55 = float(safe_div((breakout_lvl_55d - close.iloc[-1]), breakout_lvl_55d)) if np.isfinite(breakout_lvl_55d) else np.nan

    # Bollinger width and squeeze normalization
    bb_mid, bb_up, bb_dn = bollinger(close, 20, 2.0)
    bbw_now = float(safe_div((bb_up.iloc[-1] - bb_dn.iloc[-1]), bb_mid.iloc[-1])) if pd.notna(bb_mid.iloc[-1]) else np.nan
    bbw = (bb_up - bb_dn) / (bb_mid + EPS)
    bbw_med = bbw.rolling(120, min_periods=60).median()
    bbw_pct = float(safe_div(bbw.iloc[-1], bbw_med.iloc[-1])) if pd.notna(bbw_med.iloc[-1]) else np.nan

    # ATR percent and contraction ratio
    atr14 = atr(df, 14)
    atr_abs = float(atr14.iloc[-1]) if pd.notna(atr14.iloc[-1]) else np.nan
    atrp_now = float(safe_div(atr_abs, close.iloc[-1]) * 100.0) if np.isfinite(atr_abs) else np.nan
    atrp5 = (atr14 / (close + EPS) * 100.0).rolling(5, min_periods=5).mean()
    atrp20 = (atr14 / (close + EPS) * 100.0).rolling(20, min_periods=20).mean()
    atrp_ratio = float(safe_div(atrp5.iloc[-1], atrp20.iloc[-1])) if pd.notna(atrp20.iloc[-1]) else np.nan

    # Volume dry-up and turnover
    vol10 = vol.rolling(10, min_periods=10).mean()
    vol20 = vol.rolling(20, min_periods=20).mean()
    vol_dry = float(safe_div(vol10.iloc[-1], vol20.iloc[-1]) < 0.8) if pd.notna(vol20.iloc[-1]) else 0.0
    price20 = close.rolling(20, min_periods=20).mean()
    turnover = float(price20.iloc[-1] * vol20.iloc[-1]) if pd.notna(price20.iloc[-1]) and pd.notna(vol20.iloc[-1]) else 0.0

    # Returns
    ret_20 = float(safe_div(close.iloc[-1], close.iloc[-21]) - 1.0) if len(close) > 21 else np.nan
    ret_60 = float(safe_div(close.iloc[-1], close.iloc[-61]) - 1.0) if len(close) > 61 else np.nan

    # RSI
    rsi14 = float(rsi(close, 14).iloc[-1])
    rsi_ok = float(55 <= rsi14 <= 72)

    # Weekly overlay
    wk = weekly_from_daily(df)
    wk_close = float(wk["close"].iloc[-1]) if len(wk) else np.nan
    wk_ema20 = float(ema(wk["close"], 20).iloc[-1]) if len(wk) >= 20 else np.nan
    wk_ema50 = float(ema(wk["close"], 50).iloc[-1]) if len(wk) >= 50 else np.nan
    wk_trend_ok = float(np.isfinite(wk_ema20) and wk_close > wk_ema20)
    wk_trend_health = float(np.isfinite(wk_ema20) and np.isfinite(wk_ema50) and wk_ema20 > wk_ema50)

    # Breakout trigger (no lookahead)
    high20_prev = float(high20.shift(1).iloc[-1]) if pd.notna(high20.shift(1).iloc[-1]) else np.nan
    breakout_today = float(np.isfinite(high20_prev) and close.iloc[-1] > high20_prev)
    vol_confirm = float(pd.notna(vol20.iloc[-1]) and vol.iloc[-1] > 2.0 * float(vol20.iloc[-1]))
    vol_surge = vol_confirm  # same semantic
    actionable_now = float(bool(breakout_today and vol_confirm))

    # ADX
    adx14 = float(adx(df, 14).iloc[-1])
    adx_ok = float(adx14 >= 18)
    adx_strong = float(adx14 > 22)

    # MACD histogram expansion
    mh = macd_hist(close)
    macd_hist_now = float(mh.iloc[-1])
    macd_hist_delta = float(mh.iloc[-1] - mh.iloc[-2]) if len(mh) >= 2 else 0.0
    macd_hist_expand = float((macd_hist_now > 0) and (macd_hist_delta > 0))

    # Candles
    pat_engulf = float(bullish_engulfing(df) and np.isfinite(prox20) and prox20 <= 0.06)
    pat_hammer = float(hammer(df) and np.isfinite(prox20) and prox20 <= 0.06)

    # 52W high
    high_252 = float(high.rolling(252, min_periods=252).max().iloc[-1]) if len(df) >= 252 else np.nan
    dist_52w_high = float(safe_div((high_252 - close.iloc[-1]), high_252)) if np.isfinite(high_252) else np.nan
    near_52w_12pct = float(np.isfinite(dist_52w_high) and dist_52w_high <= 0.12)

    # Extended vs EMA20 (avoid chasing)
    if np.isfinite(atr_abs) and atr_abs > 0:
        ext_z = float((close.iloc[-1] - ema20.iloc[-1]) / (atr_abs + EPS))
        extended_penalty = float(max(0.0, ext_z - 1.8) / 3.0)  # 0..~1 small penalty
        not_extended = float(ext_z <= 1.8)
    else:
        extended_penalty = 0.0
        not_extended = 1.0

    # VRVP
    vrvp = compute_vrvp_features(df)
    poc_price = float(vrvp.get("poc_price", np.nan))
    dist_to_poc = float(vrvp.get("dist_to_poc", np.nan))
    poc_support = float(vrvp.get("poc_support", 0.0))
    lvn_overhead = float(vrvp.get("lvn_overhead", 0.0))
    hvn_overhead_close = float(vrvp.get("hvn_overhead_close", 0.0))
    vrvp_bin_size = float(vrvp.get("vrvp_bin_size", np.nan))

    # Filters / tiers
    hard_ok = bool(trend_up and above_ema50 and np.isfinite(prox20) and prox20 <= 0.06 and turnover >= 2e7)
    strong_ok = bool(np.isfinite(prox20) and prox20 <= 0.04 and not_extended and ((np.isfinite(bbw_pct) and bbw_pct <= 0.80) or (np.isfinite(atrp_ratio) and atrp_ratio <= 0.90)))

    # Score (simple, stable)
    squeeze_score = 0.0
    if np.isfinite(bbw_pct):
        squeeze_score += max(0.0, min(1.0, 1.2 - bbw_pct))  # <1.2 good
    if np.isfinite(atrp_ratio):
        squeeze_score += max(0.0, min(1.0, 1.2 - atrp_ratio))

    prox_score = 0.0
    if np.isfinite(prox20):
        prox_score = max(0.0, 1.0 - (prox20 / 0.06))

    score = (
        0.18 * trend_up
        + 0.08 * above_ema50
        + 0.10 * wk_trend_ok
        + 0.05 * wk_trend_health
        + 0.16 * prox_score
        + 0.12 * max(0.0, min(1.0, squeeze_score / 2.0))
        + 0.08 * float(np.isfinite(atrp_ratio) and atrp_ratio < 1.0)
        + 0.06 * float(not_extended)
        + 0.05 * float(np.isfinite(ret_60) and ret_60 > 0)
        + 0.04 * float(rsi_ok)
        + 0.04 * float(macd_hist_expand)
        + 0.03 * float(adx_ok)
        + 0.02 * float(adx_strong)
        + 0.01 * float(pat_engulf or pat_hammer)
        # Option C (hybrid with VRVP Option A)
        + 0.12 * lvn_overhead
        + 0.08 * poc_support
        - 0.06 * extended_penalty
        - 0.04 * hvn_overhead_close
    )

    return {
        "asof": str(df.index[-1]),
        "close": float(close.iloc[-1]),
        "trend_up": float(trend_up),
        "above_ema50": float(above_ema50),
        "above_ema20": float(above_ema20),
        "prox20": float(prox20),
        "prox55": float(prox55),
        "breakout_lvl_20d": float(breakout_lvl_20d),
        "breakout_lvl_55d": float(breakout_lvl_55d),
        "bbw_now": float(bbw_now),
        "bbw_pct": float(bbw_pct),
        "atrp_now": float(atrp_now),
        "atrp_ratio": float(atrp_ratio),
        "vol_dry": float(vol_dry),
        "extended_penalty": float(extended_penalty),
        "not_extended": float(not_extended),
        "turnover": float(turnover),
        "ret_20": float(ret_20),
        "ret_60": float(ret_60),
        "rsi14": float(rsi14),
        "rsi_ok": float(rsi_ok),
        "wk_close": float(wk_close),
        "wk_ema20": float(wk_ema20),
        "wk_ema50": float(wk_ema50),
        "wk_trend_ok": float(wk_trend_ok),
        "wk_trend_health": float(wk_trend_health),
        "breakout_today": float(breakout_today),
        "vol_confirm": float(vol_confirm),
        "vol_surge": float(vol_surge),
        "actionable_now": float(actionable_now),
        "adx14": float(adx14),
        "adx_ok": float(adx_ok),
        "adx_strong": float(adx_strong),
        "macd_hist": float(macd_hist_now),
        "macd_hist_delta": float(macd_hist_delta),
        "macd_hist_expand": float(macd_hist_expand),
        "pat_engulf": float(pat_engulf),
        "pat_hammer": float(pat_hammer),
        "high_252": float(high_252),
        "dist_52w_high": float(dist_52w_high),
        "near_52w_12pct": float(near_52w_12pct),
        # VRVP
        "poc_price": float(poc_price),
        "dist_to_poc": float(dist_to_poc),
        "poc_support": float(poc_support),
        "lvn_overhead": float(lvn_overhead),
        "hvn_overhead_close": float(hvn_overhead_close),
        "vrvp_bin_size": float(vrvp_bin_size),
        # tiers
        "hard_ok": bool(hard_ok),
        "strong_ok": bool(strong_ok),
        "score": float(score),
    }


def build_reasons(row: pd.Series) -> str:
    reasons: List[str] = []
    bucket = str(row.get("asset_bucket", "EQUITY")).strip().upper() or "EQUITY"
    if bucket != "EQUITY":
        reasons.append(f"Instrument type: {bucket}")

    # Fundamentals overlay
    yoy = row.get("profit_growth_yoy_pct", np.nan)
    fflag = str(row.get("fundamental_flag", "NA")).strip().upper() or "NA"
    if np.isfinite(yoy):
        reasons.append(f"Net Profit YoY: {yoy:+.1f}%")
    if fflag == "VETO":
        reasons.append("Fundamentals: VETO (weak/negative net profit growth)")
    elif fflag == "CAUTION":
        reasons.append("Fundamentals: CAUTION (low net profit growth)")

    if row.get("actionable_now", 0) >= 1:
        reasons.append("Actionable now: breakout + 2× volume")
    if row.get("wk_trend_ok", 0) >= 1:
        reasons.append("Weekly close > weekly EMA20")
    if row.get("wk_trend_health", 0) >= 1:
        reasons.append("Weekly EMA20 > weekly EMA50")
    if row.get("trend_up", 0) >= 1:
        reasons.append("EMA50 > EMA200 (daily uptrend)")
    if row.get("above_ema50", 0) >= 1:
        reasons.append("Close above EMA50")

    prox20 = row.get("prox20", np.nan)
    if np.isfinite(prox20):
        if prox20 <= 0.011:
            reasons.append("Within ~1.1% of 20D high")
        elif prox20 <= 0.04:
            reasons.append("Within 4% of 20D high")
        elif prox20 <= 0.06:
            reasons.append("Close to 20D resistance")

    bbw_pct = row.get("bbw_pct", np.nan)
    if np.isfinite(bbw_pct) and bbw_pct <= 0.80:
        reasons.append("Strong squeeze (BB width vs 120D median)")
    elif np.isfinite(bbw_pct) and bbw_pct <= 1.00:
        reasons.append("BB width contracted (squeeze-ish)")

    atrp_ratio = row.get("atrp_ratio", np.nan)
    if np.isfinite(atrp_ratio) and atrp_ratio < 1.0:
        reasons.append("ATR% contracting (5D < 20D)")

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

    if row.get("not_extended", 1) < 1:
        reasons.append("Slightly extended vs EMA20 (avoid chasing)")

    # VRVP
    if row.get("poc_support", 0) >= 1:
        reasons.append("VRVP: strong support below (POC)")
    if row.get("lvn_overhead", 0) >= 1:
        reasons.append("VRVP: low-volume air pocket overhead")
    if row.get("hvn_overhead_close", 0) >= 1:
        reasons.append("VRVP: heavy volume overhead (may stall)")

    return " | ".join(reasons)



def min_target_after_costs_ok(entry: float, target: float) -> bool:
    """True if projected target is meaningfully larger than estimated costs/noise."""
    if not (np.isfinite(entry) and np.isfinite(target) and entry > 0):
        return False
    tgt_pct = (target / entry) - 1.0
    min_pct = max(MIN_TARGET_PCT_FLOOR, MIN_TARGET_COST_MULT * APPROX_COST_PCT)
    return bool(tgt_pct >= min_pct)

def build_action_sentence(row: pd.Series) -> str:
    sym = str(row.get("symbol", "—"))
    bucket = str(row.get("asset_bucket", "EQUITY")).strip().upper() or "EQUITY"
    tag = f" [{bucket}]" if bucket != "EQUITY" else ""
    sym_disp = sym + tag

    close = float(row.get("close", np.nan))
    res = float(row.get("breakout_lvl_20d", np.nan))
    atrp = float(row.get("atrp_now", np.nan))
    not_ext = bool(row.get("not_extended", 1) >= 1)

    # Fundamentals summary
    yoy = row.get("profit_growth_yoy_pct", np.nan)
    fflag = str(row.get("fundamental_flag", "NA")).strip().upper() or "NA"
    fund_txt = ""
    if np.isfinite(yoy):
        fund_txt = f"Net Profit YoY {yoy:+.1f}%"
    if fflag == "VETO":
        fund_txt = (fund_txt + "; " if fund_txt else "") + "Fundamental VETO"
    elif fflag == "CAUTION":
        fund_txt = (fund_txt + "; " if fund_txt else "") + "Fundamental CAUTION"
    if fund_txt:
        fund_txt = f" (fundamentals: {fund_txt})"

    # Simple swing plan (keep conservative)
    entry = res * 1.002 if np.isfinite(res) else close

    if np.isfinite(atrp) and atrp > 0:
        atr_abs = close * (atrp / 100.0)
        stop_atr = entry - 1.5 * atr_abs
    else:
        stop_atr = entry * 0.97
    stop_res = res * 0.999 if np.isfinite(res) else stop_atr
    stop = min(stop_atr, stop_res)

    risk = max(EPS, entry - stop)
    target = entry + 2.0 * risk
    tgt_ok = min_target_after_costs_ok(entry, target)

    caution_bits = []
    if not tgt_ok:
        caution_bits.append("target is small after charges")
    if not not_ext:
        caution_bits.append("price is stretched, wait for a pullback")
    if bool(row.get("hvn_overhead_close", 0) >= 1):
        caution_bits.append("heavy selling traffic overhead")
    if fflag == "CAUTION":
        caution_bits.append("profit growth is weak")
    if fflag == "VETO":
        caution_bits.append("profit growth is negative/weak (veto)")

    vrvp_bits = []
    if bool(row.get("lvn_overhead", 0) >= 1):
        vrvp_bits.append("less traffic above")
    if bool(row.get("poc_support", 0) >= 1):
        vrvp_bits.append("support below")

    hints = []
    if vrvp_bits:
        hints.append(", ".join(vrvp_bits))
    if caution_bits:
        hints.append("caution: " + "; ".join(caution_bits))

    hint_txt = f" ({' | '.join(hints)})" if hints else ""
    state = str(row.get("actionable_state", "Watchlist"))

    if state == "Fundamental_Veto":
        return (
            f"**{sym_disp}**{fund_txt}{hint_txt}: Technicals may look good, but fundamentals veto this setup. "
            f"Skip for swing entries unless you have a separate thesis. If you still track it, treat it as watch-only."
        )

    if state == "Actionable":
        return (
            f"**{sym_disp}**{fund_txt}{hint_txt}: Breakout already confirmed. "
            f"Enter near ₹{entry:.2f}. Stop near ₹{stop:.2f} ({(stop/entry-1)*100:.1f}%). "
            f"First target near ₹{target:.2f} ({(target/entry-1)*100:.1f}%)."
        )

    if state == "Actionable_Caution":
        return (
            f"**{sym_disp}**{fund_txt}{hint_txt}: Breakout is happening, but there are caution flags. "
            f"Better action: wait for 1–3 days of holding above resistance or a pullback closer to support/EMA20, then enter. "
            f"If you still enter, consider smaller size. "
            f"Reference levels: entry ~ ₹{entry:.2f}, stop ~ ₹{stop:.2f}, first target ~ ₹{target:.2f}."
        )

    if np.isfinite(res):
        return (
            f"**{sym_disp}**{fund_txt}{hint_txt}: Wait for a daily close above resistance near ₹{res:.2f} with higher volume, then enter. "
            f"If it closes back below ₹{stop:.2f}, exit. First target near ₹{target:.2f}."
        )

    return f"**{sym_disp}**{fund_txt}{hint_txt}: Keep on watch. Wait for a clean breakout and volume confirmation."


# -----------------------------
# Dhan historical fetch
# -----------------------------
DHAN_HIST_URL = "https://api.dhan.co/v2/charts/historical"


def fetch_daily_ohlcv(
    token_box: Dict[str, str],
    inst: Instrument,
    from_date: str,
    to_date: str,
    cache_dir: Path,
    sleep_s: float = 0.25,
    cache_ttl_hours: float = DEFAULT_OHLC_CACHE_TTL_HOURS,
) -> pd.DataFrame:
    """Fetch (and cache) daily OHLCV from Dhan historical endpoint.

    Cache behavior:
    - Backward compatible: old caches that are just the raw Dhan JSON will still parse.
    - Freshness: cache is reused only if it matches the requested from/to dates AND is newer than cache_ttl_hours.
      (Legacy cache uses file mtime.)
    - New cache schema wraps the raw payload with `_meta` so we can safely expire stale caches.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{inst.symbol}.json"

    payload = {
        "securityId": str(inst.security_id),
        "exchangeSegment": str(inst.exchange_segment),
        "instrument": str(inst.instrument),
        "expiryCode": int(inst.expiry_code),
        "oi": False,
        "fromDate": from_date,
        "toDate": to_date,
    }
    headers = {
        "access-token": token_box["token"],
        "Accept": "application/json",
        "Content-Type": "application/json",
    }

    def _epoch_unit_from_sample(x: float) -> str:
        # epoch ms is ~1.7e12, epoch s is ~1.7e9
        try:
            v = float(x)
        except Exception:
            return "s"
        return "ms" if v > 1e11 else "s"

    def _parse_array_style(js: dict) -> pd.DataFrame:
        ts = js.get("timestamp")
        if not isinstance(ts, list) or len(ts) == 0:
            LOG.debug("_parse_array_style: no timestamp list")
            return pd.DataFrame()

        ts_ser = pd.to_numeric(pd.Series(ts), errors="coerce").dropna()
        if ts_ser.empty:
            return pd.DataFrame()

        unit = _epoch_unit_from_sample(ts_ser.iloc[-1])
        idx = pd.to_datetime(ts_ser.to_numpy().astype("int64"), unit=unit, utc=True)
        idx = pd.DatetimeIndex(idx).tz_convert("Asia/Kolkata")

        cols = {
            "open": js.get("open", []),
            "high": js.get("high", []),
            "low": js.get("low", []),
            "close": js.get("close", []),
            "volume": js.get("volume", []),
        }

        lens = [len(v) for v in cols.values() if isinstance(v, list)] + [len(idx)]
        n = int(min(lens)) if lens else 0
        if n <= 0:
            return pd.DataFrame()

        df = pd.DataFrame(
            {
                "open": pd.to_numeric(pd.Series(cols["open"][:n]), errors="coerce").reset_index(drop=True),
                "high": pd.to_numeric(pd.Series(cols["high"][:n]), errors="coerce").reset_index(drop=True),
                "low": pd.to_numeric(pd.Series(cols["low"][:n]), errors="coerce").reset_index(drop=True),
                "close": pd.to_numeric(pd.Series(cols["close"][:n]), errors="coerce").reset_index(drop=True),
                "volume": pd.to_numeric(pd.Series(cols["volume"][:n]), errors="coerce").reset_index(drop=True),
            }
        )
        df.index = idx[:n]
        df = df.dropna(subset=["open", "high", "low", "close"])
        df = df.sort_index()
        df = df[~df.index.duplicated(keep="last")]
        df.index.name = "dt"
        return df

    def _parse_candles_style(js: dict) -> pd.DataFrame:
        candles = None
        if isinstance(js, dict):
            data = js.get("data") or js.get("Data") or js.get("result") or js
            if isinstance(data, dict):
                candles = data.get("candles") or data.get("Candles") or data.get("candle") or data.get("Candle")
            if candles is None and "candles" in js:
                candles = js.get("candles")

        if not candles:
            return pd.DataFrame()

        arr = np.array(candles, dtype=object)
        if arr.ndim != 2 or arr.shape[1] < 6:
            return pd.DataFrame()

        ts = arr[:, 0]
        if isinstance(ts[0], (int, float, np.integer, np.floating)):
            unit = _epoch_unit_from_sample(ts[0])
            idx = pd.to_datetime(np.asarray(ts, dtype="int64"), unit=unit, utc=True)
        else:
            idx = pd.to_datetime(ts, utc=True, errors="coerce")

        idx = pd.DatetimeIndex(idx).tz_convert("Asia/Kolkata")

        df = pd.DataFrame(
            {
                "open": pd.to_numeric(arr[:, 1], errors="coerce"),
                "high": pd.to_numeric(arr[:, 2], errors="coerce"),
                "low": pd.to_numeric(arr[:, 3], errors="coerce"),
                "close": pd.to_numeric(arr[:, 4], errors="coerce"),
                "volume": pd.to_numeric(arr[:, 5], errors="coerce"),
            },
            index=idx,
        ).dropna(subset=["open", "high", "low", "close"])

        df = df.sort_index()
        df = df[~df.index.duplicated(keep="last")]
        df.index.name = "dt"
        return df

    def _parse_payload_to_df(payload_js: dict) -> pd.DataFrame:
        df0 = pd.DataFrame()
        if isinstance(payload_js, dict) and isinstance(payload_js.get("timestamp"), list):
            df0 = _parse_array_style(payload_js)
        if df0.empty and isinstance(payload_js, dict):
            df0 = _parse_candles_style(payload_js)
        return df0

    # 1) Cache read (fresh only)
    if cache_path.exists():
        try:
            raw = json.loads(cache_path.read_text(encoding="utf-8", errors="replace"))
            meta = None
            payload_js = raw
            if isinstance(raw, dict) and ("payload" in raw) and ("_meta" in raw):
                meta = raw.get("_meta")
                payload_js = raw.get("payload") or {}
            if _cache_is_fresh(meta=meta, cache_path=cache_path, ttl_hours=cache_ttl_hours, want_from=from_date, want_to=to_date):
                df_cached = _parse_payload_to_df(payload_js if isinstance(payload_js, dict) else {})
                if not df_cached.empty:
                    LOG.debug("%s cache hit rows=%d age_h=%s", inst.symbol, len(df_cached), _cache_age_hours_from_meta(meta or {}) if meta else "legacy")
                    return df_cached
            else:
                LOG.debug("%s cache stale (ttl_hours=%.2f)", inst.symbol, float(cache_ttl_hours))
        except Exception as e:
            LOG.debug("%s cache read/parse failed: %s", inst.symbol, e)

    # 2) Fetch
    LOG.debug("POST %s payload=%s", DHAN_HIST_URL, payload)
    r = requests.post(DHAN_HIST_URL, headers=headers, json=payload, timeout=30)

    if r.status_code != 200:
        try:
            js_err = r.json() if r.content else {}
        except Exception:
            js_err = {}

        err_code = str(js_err.get("errorCode", "") or "")
        if r.status_code == 401 and err_code in ("DH-901", "807", "808", "809"):
            new_tok = renew_token_if_possible(token_box["token"])
            if new_tok:
                headers["access-token"] = new_tok
                r2 = requests.post(DHAN_HIST_URL, headers=headers, json=payload, timeout=30)
                if r2.status_code == 200:
                    token_box["token"] = new_tok
                    r = r2
                else:
                    raise RuntimeError(f"Dhan historical error for {inst.symbol} [HTTP {r2.status_code}]: {r2.text}")
            else:
                raise RuntimeError(f"Dhan historical error for {inst.symbol} [HTTP {r.status_code}]: {r.text}")
        else:
            raise RuntimeError(f"Dhan historical error for {inst.symbol} [HTTP {r.status_code}]: {r.text}")

    payload_js = r.json()

    # 3) Parse
    df = _parse_payload_to_df(payload_js if isinstance(payload_js, dict) else {})
    if df.empty:
        raise RuntimeError(
            f"No candle data present for {inst.symbol} (response keys={list(payload_js.keys()) if isinstance(payload_js, dict) else type(payload_js)})."
        )

    # 4) Cache write (new schema, backward compatible)
    try:
        meta = {
            "schema_version": CACHE_SCHEMA_VERSION,
            "fetched_at_utc": _utcnow().isoformat(),
            "ttl_hours": float(cache_ttl_hours),
            "endpoint": DHAN_HIST_URL,
            "fromDate": from_date,
            "toDate": to_date,
            "securityId": str(inst.security_id),
            "exchangeSegment": str(inst.exchange_segment),
            "instrument": str(inst.instrument),
            "expiryCode": int(inst.expiry_code),
            "rows": int(len(df)),
            "last_bar_date": str(df.index[-1].date()) if len(df) else None,
        }
        cache_obj = {"_meta": meta, "payload": payload_js}
        cache_path.write_text(json.dumps(cache_obj), encoding="utf-8")
    except Exception as e:
        LOG.debug("%s cache write failed: %s", inst.symbol, e)

    time.sleep(max(0.0, float(sleep_s)))
    return df


# -----------------------------
# Runner# -----------------------------
# Runner
# -----------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--master-csv", required=True)
    p.add_argument("--watchlist", required=True, help="Text file: one symbol per line.")
    p.add_argument("--out-csv", default="swing_scan_report.csv")
    p.add_argument("--out-md", default="Swing_Scan_Findings_and_Recommendations.md")
    p.add_argument("--cache-dir", default="cache_ohlc")
    p.add_argument("--log-level", default="INFO")
    p.add_argument("--log-file", default=None)
    p.add_argument("--days", type=int, default=365, help="History lookback in days (daily candles).")
    p.add_argument("--top-n", type=int, default=30)

    # Cache freshness (fix: stale data forever)
    p.add_argument("--cache-ttl-hours", type=float, default=DEFAULT_OHLC_CACHE_TTL_HOURS, help="OHLC cache freshness window.")

    # Fundamentals overlay (Screener.in)
    p.add_argument("--no-fundamentals", action="store_true", help="Disable Screener.in fundamentals overlay.")
    p.add_argument("--fund-ttl-hours", type=float, default=DEFAULT_FUND_CACHE_TTL_HOURS, help="Fundamentals cache freshness window.")
    p.add_argument("--fund-force-refresh", action="store_true", help="Bypass fundamentals cache and re-parse/re-fetch Screener data.")
    p.add_argument("--screener-sleep", type=float, default=0.4, help="Sleep between Screener requests (polite rate-limit).")
    p.add_argument("--screener-html-file", default="", help="(debug) Use saved Screener HTML file instead of fetching")

    p.add_argument("--token-smoke-test", action="store_true", help="Call /v2/holdings once to validate token early.")
    p.add_argument("--dhan-client-id", default=None, help="Optional: used for /v2/RenewToken auto-renew (also sets DHAN_CLIENT_ID).")
    return p.parse_args()


def read_watchlist(path: str) -> List[str]:
    syms: List[str] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        syms.append(s)
    return syms


def write_markdown(df: pd.DataFrame, out_md: str, top_n: int) -> None:
    lines: List[str] = []
    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M")

    lines.append("# Swing Scan Findings and Recommendations\n")
    lines.append(f"_Generated: {now}_\n")
    lines.append("This report is plain-English. It tells you what to do next for each symbol.\n\n")

    # Rank
    df = df.copy()
    df["reasons"] = df.apply(build_reasons, axis=1)

    # Top-N overall
    df = df.sort_values(["score"], ascending=False)
    top = df.head(top_n)

    lines.append("## Summary\n")
    lines.append(f"- Symbols scanned: **{len(df)}**\n")
    lines.append(f"- Top shortlist shown: **{len(top)}**\n\n")

    veto = top[top.get("actionable_state", "Watchlist") == "Fundamental_Veto"]
    actionable = top[top.get("actionable_state", "Watchlist") == "Actionable"]
    caution = top[top.get("actionable_state", "Watchlist") == "Actionable_Caution"]
    watch = top[top.get("actionable_state", "Watchlist") == "Watchlist"]

    def section(title: str, sub: str, frame: pd.DataFrame) -> None:
        lines.append(f"## {title}\n")
        lines.append(sub + "\n")
        if frame.empty:
            lines.append("_None in this section._\n")
            return
        for bucket, grp in frame.groupby("asset_bucket", dropna=False):
            b = str(bucket).strip().upper() if str(bucket).strip() else "EQUITY"
            lines.append(f"### {b}\n")
            for _, r in grp.iterrows():
                lines.append("- " + build_action_sentence(r))
            lines.append("")

    section(
        "Fundamental veto (skip / deprioritize)",
        f"These may look technically attractive, but quarterly net profit YoY is <= {FUND_YOY_VETO_PCT:.0f}%. Treat as watch-only unless you have a strong separate thesis.",
        veto,
    )

    section(
        "Actionable (breakout + volume confirmed)",
        "These broke out on a daily close with strong volume and look clean. Use the entry/stop/target plan.",
        actionable,
    )

    section(
        "Actionable with caution",
        "Breakout signal is present, but caution flags exist (extended price, heavy volume overhead, or weak fundamentals). Prefer waiting for hold/pullback or reduce size.",
        caution,
    )

    section(
        "Watchlist (near breakout, waiting for trigger)",
        "These are close to resistance. Wait for a daily close above resistance with higher volume, then enter.",
        watch,
    )

    lines.append("## Details table (top shortlist)\n")
    cols = [
        "symbol", "asset_bucket", "actionable_state",
        "close", "score",
        "profit_growth_yoy_pct", "fundamental_flag",
        "hard_ok", "strong_ok",
        "turnover", "prox20", "bbw_pct", "atrp_ratio", "adx14", "rsi14",
    ]
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "|".join(["---"] * len(cols)) + "|")
    for _, r in top[cols].iterrows():
        row = []
        for c in cols:
            v = r[c]
            if isinstance(v, (float, np.floating)):
                if c == "profit_growth_yoy_pct":
                    row.append(f"{v:+.1f}" if np.isfinite(v) else "NA")
                else:
                    row.append(f"{v:.4g}" if np.isfinite(v) else "NA")
            else:
                row.append(str(v))
        lines.append("| " + " | ".join(row) + " |")

    Path(out_md).write_text("\n".join(lines), encoding="utf-8")
    LOG.info("Wrote markdown report: %s", out_md)
def main() -> int:
    args = parse_args()
    setup_logging(args.log_level, args.log_file)

    token, token_src = get_dhan_token()
    tok_len = len(token)
    tok_hash = hashlib.sha1(token.encode("utf-8", errors="ignore")).hexdigest()[:10]
    LOG.info("Using Dhan token from %s (len=%d sha1=%s)", token_src, tok_len, tok_hash)
    if args.dhan_client_id:
        os.environ["DHAN_CLIENT_ID"] = str(args.dhan_client_id).strip()
    exp = _jwt_exp_utc(token)
    if exp:
        LOG.info("Token expiry (UTC) appears to be: %s", exp.isoformat())
    else:
        LOG.info("Token expiry not decoded (still OK).")
    if args.token_smoke_test:
        ok = token_smoke_test(token)
        if not ok:
            LOG.error("Token smoke-test failed: access-token invalid/expired (DH-901 likely). Generate a fresh token in Dhan Web or renew it.")
            return 2

    master = load_master(args.master_csv)
    watch = read_watchlist(args.watchlist)

    # token holder so auto-renew updates token for subsequent API calls
    token_box = {"token": token}

    # Date range
    today = dt.date.today()
    from_date = (today - dt.timedelta(days=args.days)).strftime("%Y-%m-%d")
    to_date = today.strftime("%Y-%m-%d")

    LOG.info(
        "Scanning symbols=%d from=%s to=%s (cache_ttl_hours=%.1f; fundamentals=%s)",
        len(watch),
        from_date,
        to_date,
        float(args.cache_ttl_hours),
        "off" if args.no_fundamentals else "on",
    )

    cache_dir = Path(args.cache_dir)
    fund_cache_dir = cache_dir / "_fundamentals"

    rows: List[Dict[str, float]] = []
    failures: List[Tuple[str, str]] = []

    screener_sess = requests.Session() if not args.no_fundamentals else None

    screener_html_override = None
    if getattr(args, "screener_html_file", ""):
        try:
            pth = Path(args.screener_html_file).expanduser()
            if pth.exists():
                screener_html_override = pth.read_text(encoding="utf-8", errors="ignore")
                LOG.info("Using screener-html-file override: %s (len=%d)", str(pth), len(screener_html_override))
            else:
                LOG.warning("screener-html-file not found: %s", str(pth))
        except Exception as e:
            LOG.warning("Failed reading screener-html-file: %s", e)

    for i, sym in enumerate(watch, start=1):
        inst = resolve_symbol(sym, master)
        if not inst:
            failures.append((sym, "Symbol not found in master CSV"))
            LOG.warning("(%d/%d) Failed %s: not found in master CSV", i, len(watch), sym)
            continue

        try:
            df = fetch_daily_ohlcv(
                token_box,
                inst,
                from_date,
                to_date,
                cache_dir,
                cache_ttl_hours=float(args.cache_ttl_hours),
            )
            LOG.debug("(%d/%d) %s fetched rows=%d cached=%s", i, len(watch), sym, len(df), str(cache_dir / f"{inst.symbol}.json"))

            met = score_one(df)
            if met is None:
                failures.append((sym, f"Insufficient history (<210 bars) or bad data (rows={len(df)})"))
                LOG.debug("(%d/%d) Skip %s: insufficient history or bad data", i, len(watch), sym)
                continue

            met["symbol"] = sym
            met["security_id"] = inst.security_id
            met["asset_bucket"] = inst.asset_bucket
            met["exch_instrument_type"] = inst.exch_instrument_type
            met["series"] = inst.series
            met["custom_symbol"] = inst.custom_symbol

            # Fundamentals overlay (default ON for EQUITY only)
            met["profit_growth_yoy_pct"] = np.nan
            met["fundamental_flag"] = "NA"
            met["fundamental_q_curr"] = ""
            met["fundamental_q_prev_yoy"] = ""

            if not args.no_fundamentals and (inst.asset_bucket != "EQUITY"):
                LOG.debug(
                    "[FUND] %s skipped (non-equity bucket=%s exch_type=%s instr=%s series=%s)",
                    sym, inst.asset_bucket, inst.exch_instrument_type, inst.instrument, inst.series
                )

            if (not args.no_fundamentals) and (inst.asset_bucket == "EQUITY"):
                f = fetch_profit_growth_screener(
                    symbol=sym,
                    cache_dir=fund_cache_dir,
                    ttl_hours=float(args.fund_ttl_hours),
                    session=screener_sess,
                    sleep_s=float(args.screener_sleep),
                    html_override=screener_html_override,
                    force_refresh=bool(args.fund_force_refresh),
                )
                yoy = f.get("yoy_net_profit_growth_pct")
                if yoy is not None and np.isfinite(float(yoy)):
                    met["profit_growth_yoy_pct"] = float(yoy)
                met["fundamental_q_curr"] = str(f.get("curr_q_label") or "")
                met["fundamental_q_prev_yoy"] = str(f.get("prev_yoy_q_label") or "")
                met["fundamental_flag"] = _fund_flag_from_growth(met.get("profit_growth_yoy_pct", np.nan))

            # 4-state actionable label: technicals + fundamentals
            fflag = str(met.get("fundamental_flag", "NA")).strip().upper() or "NA"
            if fflag == "VETO":
                met["actionable_state"] = "Fundamental_Veto"
            else:
                if int(met.get("actionable_now", 0)) >= 1:
                    clean_tech = (int(met.get("not_extended", 1)) >= 1) and (int(met.get("hvn_overhead_close", 0)) < 1)
                    if clean_tech and fflag != "CAUTION":
                        met["actionable_state"] = "Actionable"
                    else:
                        met["actionable_state"] = "Actionable_Caution"
                else:
                    met["actionable_state"] = "Watchlist"

            rows.append(met)

            if LOG.isEnabledFor(logging.DEBUG):
                LOG.debug(
                    "(%d/%d) %s bucket=%s prox20=%.4f bbw_pct=%.3f atrp_ratio=%.3f wk_ok=%s actionable=%s fund=%s(%s)",
                    i, len(watch), sym, inst.asset_bucket,
                    met.get("prox20", np.nan), met.get("bbw_pct", np.nan), met.get("atrp_ratio", np.nan),
                    int(met.get("wk_trend_ok", 0)), int(met.get("actionable_now", 0)),
                    met.get("fundamental_flag", "NA"), met.get("profit_growth_yoy_pct", np.nan),
                )

        except Exception as e:
            failures.append((sym, str(e)))
            LOG.warning("(%d/%d) Failed for %s: %s", i, len(watch), sym, e)

    if not rows:
        LOG.error("No symbols produced usable data.")
        if failures:
            LOG.info("Failures (first 30):")
            for s, reason in failures[:30]:
                LOG.info("  - %s: %s", s, reason)
        return 3

    out_df = pd.DataFrame(rows)
    out_df["reasons"] = out_df.apply(build_reasons, axis=1)

    out_df.to_csv(args.out_csv, index=False)
    LOG.info("Wrote CSV report: %s (rows=%d)", args.out_csv, len(out_df))

    write_markdown(out_df, args.out_md, args.top_n)

    if failures:
        LOG.info("Failures (first 30):")
        for s, reason in failures[:30]:
            LOG.info("  - %s: %s", s, reason)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
