#!/usr/bin/env python3
"""Cache manifest builder + gap scanner for intraday_cache."""
from __future__ import annotations

import argparse
import json
import os
import re
import glob
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Dict, List, Set

import pandas as pd

DATE_RE = re.compile(r"_(\d{8})_1m\.csv$")


def _parse_date_from_name(path: str) -> str:
    m = DATE_RE.search(path)
    if not m:
        return ""
    ds = m.group(1)
    try:
        return datetime.strptime(ds, "%Y%m%d").date().isoformat()
    except Exception:
        return ""


def build_cache_manifest(cache_glob: str, out_path: str, expected_bars: int) -> int:
    files = sorted(glob.glob(cache_glob))
    if not files:
        raise SystemExit(f"No files matched: {cache_glob}")

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    n = 0
    with out.open("w", encoding="utf-8") as f:
        for fp in files:
            fp = Path(fp)
            try:
                df = pd.read_csv(fp, usecols=["timestamp"])
            except Exception:
                continue
            if df.empty:
                continue
            try:
                ts = pd.to_datetime(df["timestamp"], errors="coerce")
            except Exception:
                continue
            ts = ts.dropna()
            if ts.empty:
                continue

            day = _parse_date_from_name(fp.name) or ts.iloc[0].date().isoformat()
            count = int(len(ts))
            first_ts = str(ts.iloc[0])
            last_ts = str(ts.iloc[-1])
            status = "complete" if count >= expected_bars else "partial"

            rec = {
                "day": day,
                "file": str(fp),
                "bars": count,
                "first_ts": first_ts,
                "last_ts": last_ts,
                "expected_bars": int(expected_bars),
                "status": status,
            }
            f.write(json.dumps(rec, separators=(",", ":")) + "\n")
            n += 1

    return n


def _load_holidays(path: str) -> Set[str]:
    if not path:
        return set()
    p = Path(path)
    if not p.exists():
        return set()
    out: Set[str] = set()
    try:
        raw = p.read_text(encoding="utf-8")
    except Exception:
        return out
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # allow comma-separated
        parts = [x.strip() for x in line.split(",") if x.strip()]
        for part in parts:
            try:
                d = datetime.strptime(part, "%Y-%m-%d").date().isoformat()
                out.add(d)
            except Exception:
                continue
    return out


def _is_trading_day(d: date, *, skip_weekends: bool, holidays: Set[str]) -> bool:
    if skip_weekends and d.weekday() >= 5:
        return False
    if d.isoformat() in holidays:
        return False
    return True


def scan_cache_completeness(
    manifest_path: str,
    start_date: str,
    end_date: str,
    *,
    skip_weekends: bool = False,
    holiday_file: str = "",
) -> Dict[str, List[str]]:
    start = datetime.strptime(start_date, "%Y-%m-%d").date()
    end = datetime.strptime(end_date, "%Y-%m-%d").date()
    holidays = _load_holidays(holiday_file)

    by_day: Dict[str, Dict[str, str]] = {}
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            day = rec.get("day")
            if day:
                by_day[day] = rec

    missing: List[str] = []
    partial: List[str] = []

    cur = start
    while cur <= end:
        if not _is_trading_day(cur, skip_weekends=skip_weekends, holidays=holidays):
            cur += timedelta(days=1)
            continue
        d = cur.isoformat()
        rec = by_day.get(d)
        if not rec:
            missing.append(d)
        else:
            if rec.get("status") != "complete":
                partial.append(d)
        cur += timedelta(days=1)

    return {"missing": missing, "partial": partial}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_glob", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--expected_bars", type=int, default=int(os.getenv("EXPECTED_BARS_PER_DAY", "375")))
    ap.add_argument("--scan", action="store_true")
    ap.add_argument("--start_date")
    ap.add_argument("--end_date")
    ap.add_argument("--skip_weekends", action="store_true")
    ap.add_argument("--holiday_file", default="")
    args = ap.parse_args()

    n = build_cache_manifest(args.cache_glob, args.out, args.expected_bars)
    print(f"manifest_rows={n} out={args.out}")

    if args.scan:
        if not args.start_date or not args.end_date:
            raise SystemExit("--scan requires --start_date and --end_date")
        res = scan_cache_completeness(
            args.out,
            args.start_date,
            args.end_date,
            skip_weekends=args.skip_weekends,
            holiday_file=args.holiday_file,
        )
        print("missing_days=", len(res["missing"]))
        print("partial_days=", len(res["partial"]))
        if res["missing"]:
            print("missing=", ",".join(res["missing"]))
        if res["partial"]:
            print("partial=", ",".join(res["partial"]))


if __name__ == "__main__":
    main()
