#!/usr/bin/env python3
"""Fetch only missing/partial days from cache manifest using Dhan API."""
from __future__ import annotations

import argparse
import os
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from cache_manifest import scan_cache_completeness
from dhan_cache_fetch import fetch_day


def _date_to_filename(symbol: str, day: str, suffix: str) -> str:
    dt = datetime.strptime(day, "%Y-%m-%d")
    return f"{symbol}_{dt.strftime('%Y%m%d')}_{suffix}.csv"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--start_date", required=True)
    ap.add_argument("--end_date", required=True)
    ap.add_argument("--out_dir", default="data/intraday_cache")
    ap.add_argument("--symbol", default="INDEX")
    ap.add_argument("--interval", default="1")
    ap.add_argument("--file_suffix", default="1m")
    ap.add_argument("--security_id", default=os.getenv("DHAN_SECURITY_ID", "13"))
    ap.add_argument("--exchange_segment", default=os.getenv("DHAN_EXCHANGE_SEGMENT", "IDX_I"))
    ap.add_argument("--instrument", default=os.getenv("DHAN_INSTRUMENT", "INDEX"))
    ap.add_argument("--token", default=os.getenv("DHAN_ACCESS_TOKEN", ""))
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--include_partial", action="store_true")
    ap.add_argument("--skip_weekends", action="store_true")
    ap.add_argument("--holiday_file", default="")
    ap.add_argument("--sleep_sec", type=float, default=float(os.getenv("DHAN_RATE_LIMIT_SLEEP", "0.3") or "0.3"))
    args = ap.parse_args()

    if not args.token:
        raise SystemExit("DHAN_ACCESS_TOKEN missing")

    res = scan_cache_completeness(
        args.manifest,
        args.start_date,
        args.end_date,
        skip_weekends=args.skip_weekends,
        holiday_file=args.holiday_file,
    )

    targets = list(res.get("missing", []))
    if args.include_partial:
        targets.extend(res.get("partial", []))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    total_written = 0
    for day in targets:
        out_path = out_dir / _date_to_filename(args.symbol, day, args.file_suffix)
        if out_path.exists() and not args.overwrite:
            continue

        next_day = (datetime.strptime(day, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
        try:
            rows = fetch_day(
                token=args.token,
                security_id=args.security_id,
                exchange_segment=args.exchange_segment,
                instrument=args.instrument,
                interval=args.interval,
                from_date=day,
                to_date=next_day,
            )
        except Exception as e:
            print(f"[FETCH] {day} failed: {e}")
            continue

        if not rows:
            print(f"[FETCH] {day} no rows")
            continue

        df = pd.DataFrame(rows)
        df = df.dropna(subset=["timestamp", "open", "high", "low", "close"]).sort_values("timestamp")
        df.to_csv(out_path, index=False)
        total_written += 1
        print(f"[FETCH] wrote {out_path} rows={len(df)}")

        if args.sleep_sec > 0:
            import time
            time.sleep(args.sleep_sec)

    print(f"done written_files={total_written}")


if __name__ == "__main__":
    main()
