#!/usr/bin/env python3
"""Replay captured index ticks through UnifiedWebSocketHandler."""
from __future__ import annotations

import argparse
import asyncio
import csv
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import pandas as pd

from core_handler import UnifiedWebSocketHandler


def _build_config(interval_sec: int) -> SimpleNamespace:
    return SimpleNamespace(
        nifty_security_id=13,
        nifty_exchange_segment="IDX_I",
        candle_interval_seconds=int(interval_sec),
        max_buffer_size=20000,
        max_candles_stored=100000,
        price_sanity_min=1.0,
        price_sanity_max=1000000.0,
        use_arrival_time=True,
        preclose_lead_seconds=10,
        preclose_completion_buffer_sec=2,
        idx_ticks_path="",
    )


def _parse_ts(val: str):
    ts = pd.to_datetime(val, errors="coerce")
    if pd.isna(ts):
        return None
    if ts.tzinfo is None:
        ts = ts.tz_localize("Asia/Kolkata")
    return ts.to_pydatetime()


async def _run_replay(args) -> int:
    ticks_path = Path(args.ticks)
    if not ticks_path.exists():
        raise FileNotFoundError(f"tick file not found: {ticks_path}")

    cfg = _build_config(args.interval_sec)
    ws = UnifiedWebSocketHandler(cfg)

    async def _on_candle(_row, _full_hist):
        return

    ws.on_candle = _on_candle

    processed = 0
    skipped = 0
    start = time.perf_counter()
    prev_ts: Optional[float] = None

    with ticks_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if args.max_ticks > 0 and processed >= args.max_ticks:
                break
            ts = _parse_ts(str(row.get("ts_ist", "")))
            if ts is None:
                skipped += 1
                continue
            try:
                ltp = float(row.get("ltp", "0") or 0.0)
            except Exception:
                skipped += 1
                continue
            packet_type = str(row.get("packet_type", "ticker") or "ticker")
            tick = {
                "timestamp": ts,
                "packet_type": packet_type,
                "ltp": ltp,
            }
            await ws._process_tick(tick)
            processed += 1

            if args.speed > 0:
                cur_ts = ts.timestamp()
                if prev_ts is not None:
                    gap = max(0.0, cur_ts - prev_ts)
                    if gap > 0:
                        await asyncio.sleep(gap / args.speed)
                prev_ts = cur_ts

    await ws.disconnect(stop_running=True)

    elapsed = max(1e-9, time.perf_counter() - start)
    tps = processed / elapsed
    candles = len(ws.candle_data) if isinstance(ws.candle_data, pd.DataFrame) else 0

    out_candles = Path(args.out_candles)
    out_candles.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(ws.candle_data, pd.DataFrame) and not ws.candle_data.empty:
        ws.candle_data.to_csv(out_candles, index=True)

    print(f"processed_ticks={processed}")
    print(f"skipped_ticks={skipped}")
    print(f"candles={candles}")
    print(f"elapsed_sec={elapsed:.3f}")
    print(f"ticks_per_sec={tps:.2f}")
    print(f"candles_csv={out_candles}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Replay index ticks through handler")
    ap.add_argument("--ticks", required=True, help="Path to IDX tick CSV (IDX_TICKS_PATH output)")
    ap.add_argument("--out-candles", default="runtime/replay_idx_candles.csv", help="Output candles CSV path")
    ap.add_argument("--interval-sec", type=int, default=300, help="Candle interval in seconds")
    ap.add_argument("--speed", type=float, default=0.0, help="Replay speed multiplier; 0=as fast as possible")
    ap.add_argument("--max-ticks", type=int, default=0, help="Limit ticks processed; 0=all")
    args = ap.parse_args()

    return asyncio.run(_run_replay(args))


if __name__ == "__main__":
    raise SystemExit(main())
