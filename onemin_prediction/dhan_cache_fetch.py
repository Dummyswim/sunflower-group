#!/usr/bin/env python3
"""Fetch intraday OHLCV from Dhan and write per-day cache files."""
from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import requests
except Exception:
    requests = None

import pandas as pd

API_URL = "https://api.dhan.co/v2/charts/intraday"

logger = logging.getLogger(__name__)
IST = timezone(timedelta(hours=5, minutes=30))


def _parse_ts(val: Any) -> Optional[str]:
    if val is None:
        return None
    try:
        if isinstance(val, (int, float)):
            v = float(val)
            # epoch in ms or seconds
            if v > 1e12:
                dt = datetime.fromtimestamp(v / 1000.0, tz=timezone.utc)
            else:
                dt = datetime.fromtimestamp(v, tz=timezone.utc)
            dt = dt.astimezone(IST)
            return dt.strftime("%Y-%m-%d %H:%M:%S")
        s = str(val)
        try:
            dt = datetime.fromisoformat(s)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=IST)
            else:
                dt = dt.astimezone(IST)
            return dt.strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            return s
    except Exception:
        return None


def _parse_candles(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    data = payload.get("data") if isinstance(payload, dict) else payload
    if data is None and isinstance(payload, dict):
        data = payload
    if data is None:
        return []

    # Case 1: list of candles
    if isinstance(data, list):
        candles = data
    elif isinstance(data, dict) and isinstance(data.get("candles"), list):
        candles = data.get("candles")
    else:
        candles = None

    rows: List[Dict[str, Any]] = []
    if candles:
        for c in candles:
            # Expect [ts, o, h, l, c, v, (optional) oi]
            if not isinstance(c, (list, tuple)) or len(c) < 6:
                continue
            ts = _parse_ts(c[0])
            if not ts:
                continue
            rows.append({
                "timestamp": ts,
                "open": c[1],
                "high": c[2],
                "low": c[3],
                "close": c[4],
                "volume": c[5],
            })
        return rows

    # Case 2: dict of arrays
    if isinstance(data, dict):
        keys = ("timestamp", "open", "high", "low", "close", "volume")
        if all(k in data for k in keys):
            ts_list = data.get("timestamp") or []
            if ts_list:
                logger.debug("[FETCH] timestamp raw sample=%s", ts_list[:3])
                logger.debug("[FETCH] timestamp parsed sample=%s", [_parse_ts(x) for x in ts_list[:3]])
            for i in range(len(ts_list)):
                ts = _parse_ts(ts_list[i])
                if not ts:
                    continue
                rows.append({
                    "timestamp": ts,
                    "open": data.get("open", [None])[i],
                    "high": data.get("high", [None])[i],
                    "low": data.get("low", [None])[i],
                    "close": data.get("close", [None])[i],
                    "volume": data.get("volume", [None])[i],
                })
            return rows

    return []


def fetch_day(
    *,
    token: str,
    security_id: str,
    exchange_segment: str,
    instrument: str,
    interval: str,
    from_date: str,
    to_date: str,
) -> List[Dict[str, Any]]:
    if requests is None:
        raise RuntimeError("requests not available")

    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "access-token": token,
    }
    payload = {
        "securityId": str(security_id),
        "exchangeSegment": str(exchange_segment),
        "instrument": str(instrument),
        "interval": str(interval),
        "fromDate": from_date,
        "toDate": to_date,
    }

    resp = requests.post(API_URL, headers=headers, json=payload, timeout=30)
    logger.debug("[FETCH] POST %s payload=%s", API_URL, payload)
    logger.debug("[FETCH] status=%s body=%s", resp.status_code, resp.text[:500])
    if resp.status_code != 200:
        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:200]}")

    try:
        data = resp.json()
        if isinstance(data, dict):
            logger.debug("[FETCH] response keys=%s", list(data.keys()))
    except Exception as e:
        raise RuntimeError(f"JSON parse failed: {e}")

    return _parse_candles(data)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start_date", required=True, help="YYYY-MM-DD")
    ap.add_argument("--end_date", required=True, help="YYYY-MM-DD (inclusive)")
    ap.add_argument("--out_dir", default="data/intraday_cache")
    ap.add_argument("--symbol", default="INDEX")
    ap.add_argument("--interval", default="1", help="minute interval: 1,5,15,25,60")
    ap.add_argument("--security_id", default=os.getenv("DHAN_SECURITY_ID", "13"))
    ap.add_argument("--exchange_segment", default=os.getenv("DHAN_EXCHANGE_SEGMENT", "IDX_I"))
    ap.add_argument("--instrument", default=os.getenv("DHAN_INSTRUMENT", "INDEX"))
    ap.add_argument("--token", default=os.getenv("DHAN_ACCESS_TOKEN", ""))
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--sleep_sec", type=float, default=float(os.getenv("DHAN_RATE_LIMIT_SLEEP", "0.3") or "0.3"))
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    if args.debug:
        logging.basicConfig(level=logging.DEBUG, format="%(asctime)s | %(levelname)s | %(message)s")
    else:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    if not args.token:
        raise SystemExit("DHAN_ACCESS_TOKEN missing")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    start = datetime.strptime(args.start_date, "%Y-%m-%d").date()
    end = datetime.strptime(args.end_date, "%Y-%m-%d").date()

    cur = start
    total_written = 0
    while cur <= end:
        day_str = cur.strftime("%Y-%m-%d")
        next_day = (cur + timedelta(days=1)).strftime("%Y-%m-%d")
        out_path = out_dir / f"{args.symbol}_{cur.strftime('%Y%m%d')}_1m.csv"

        if out_path.exists() and not args.overwrite:
            cur += timedelta(days=1)
            continue

        try:
            rows = fetch_day(
                token=args.token,
                security_id=args.security_id,
                exchange_segment=args.exchange_segment,
                instrument=args.instrument,
                interval=args.interval,
                from_date=day_str,
                to_date=next_day,
            )
        except Exception as e:
            print(f"[FETCH] {day_str} failed: {e}")
            cur += timedelta(days=1)
            continue

        if not rows:
            print(f"[FETCH] {day_str} no rows")
            cur += timedelta(days=1)
            continue

        df = pd.DataFrame(rows)
        df = df.dropna(subset=["timestamp", "open", "high", "low", "close"])\
               .sort_values("timestamp")
        df.to_csv(out_path, index=False)
        total_written += 1
        print(f"[FETCH] wrote {out_path} rows={len(df)}")

        if args.sleep_sec > 0:
            import time
            time.sleep(args.sleep_sec)

        cur += timedelta(days=1)

    print(f"done written_files={total_written}")


if __name__ == "__main__":
    main()
