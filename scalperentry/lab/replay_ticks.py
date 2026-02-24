#!/usr/bin/env python3
"""Deterministic post-session replay for TradeBrain.

Feeds tick rows from FUT tick CSV into TradeBrain.process_tick() and writes isolated
signal/arm JSONL outputs for analysis.
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tradebrain import TradeBrain, TradeBrainConfig, setup_logger


def _parse_ts(x: str) -> Optional[datetime]:
    try:
        dt = datetime.fromisoformat((x or "").strip())
        return dt if dt.tzinfo else None
    except Exception:
        return None


def _build_brain(out_dir: Path, sidecar_path: str, log_level: int) -> TradeBrain:
    # Avoid mixed runs when output directory already exists.
    for name in (
        "replay_tradebrain.log",
        "replay_tradebrain_arm.log",
        "replay_signal.jsonl",
        "replay_arm.jsonl",
    ):
        p = out_dir / name
        if p.exists():
            p.unlink()

    cfg = TradeBrainConfig.from_env()
    cfg = replace(
        cfg,
        log_file=str(out_dir / "replay_tradebrain.log"),
        arm_log_file=str(out_dir / "replay_tradebrain_arm.log"),
        jsonl_path=str(out_dir / "replay_signal.jsonl"),
        arm_jsonl_path=str(out_dir / "replay_arm.jsonl"),
        write_all=True,
        write_hold=True,
        fut_sidecar_path=sidecar_path,
        fut_sidecar_poll_ms=0,
    )
    cfg.validate()
    log = setup_logger("tradebrain.replay", cfg.log_file, log_level, False)
    return TradeBrain(cfg, log)


def _iter_ticks(tick_csv: Path):
    # Expected sidecar tick schema: ts, ltp, ... (ltp at index 1)
    with tick_csv.open("r", encoding="utf-8", newline="") as f:
        rdr = csv.reader(f)
        for row in rdr:
            if len(row) < 2:
                continue
            ts = _parse_ts(row[0])
            if ts is None:
                continue
            try:
                ltp = float(row[1])
            except Exception:
                continue
            if not (ltp > 0):
                continue
            recv_ms = int(ts.timestamp() * 1000)
            yield {
                "timestamp": ts.isoformat(),
                "ltp": ltp,
                "recv_ms": recv_ms,
                "recv_ns": recv_ms * 1_000_000,
            }


def _load_sidecar_rows(sidecar_csv: Path):
    rows = []
    with sidecar_csv.open("r", encoding="utf-8", newline="") as f:
        for line in f:
            raw = line.strip()
            if not raw:
                continue
            ts_txt = raw.split(",", 1)[0]
            ts = _parse_ts(ts_txt)
            if ts is None:
                continue
            rows.append((ts, raw))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description="Replay tick CSV through TradeBrain")
    ap.add_argument("--ticks", required=True, help="Path to fut_ticks_candles.csv")
    ap.add_argument("--sidecar", required=True, help="Path to fut_candles.csv")
    ap.add_argument("--out", required=True, help="Output directory for replay artifacts")
    ap.add_argument("--from", dest="from_ts", default="", help="Inclusive ISO timestamp filter")
    ap.add_argument("--to", dest="to_ts", default="", help="Inclusive ISO timestamp filter")
    ap.add_argument(
        "--warmup-minutes",
        type=int,
        default=120,
        help="Replay warmup minutes before --from (default: 120)",
    )
    ap.add_argument("--log-level", default="INFO", help="Logging level")
    args = ap.parse_args()

    tick_csv = Path(args.ticks)
    sidecar = Path(args.sidecar)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not tick_csv.exists():
        raise FileNotFoundError(f"ticks file not found: {tick_csv}")
    if not sidecar.exists():
        raise FileNotFoundError(f"sidecar file not found: {sidecar}")

    from_ts = _parse_ts(args.from_ts) if args.from_ts else None
    to_ts = _parse_ts(args.to_ts) if args.to_ts else None
    warmup_minutes = max(0, int(args.warmup_minutes))
    warmup_from = (from_ts - timedelta(minutes=warmup_minutes)) if from_ts else None

    sidecar_rows = _load_sidecar_rows(sidecar)
    runtime_sidecar = out_dir / "replay_sidecar.csv"
    runtime_sidecar.write_text("", encoding="utf-8")

    lvl = getattr(logging, str(args.log_level).upper(), logging.INFO)
    tb = _build_brain(out_dir, str(runtime_sidecar), lvl)

    total = 0
    fed_total = 0
    fed_window = 0
    dropped = 0
    sidecar_rows_written = 0
    side_i = 0
    try:
        with runtime_sidecar.open("a", encoding="utf-8", newline="") as sidecar_out:
            for tick in _iter_ticks(tick_csv):
                total += 1
                ts = _parse_ts(str(tick.get("timestamp", "")))
                if ts is None:
                    dropped += 1
                    continue
                if warmup_from and ts < warmup_from:
                    continue
                if to_ts and ts > to_ts:
                    continue

                bucket_ts = ts.replace(second=0, microsecond=0)
                # Feed only sidecar rows that would have existed at this replay timestamp.
                while side_i < len(sidecar_rows) and sidecar_rows[side_i][0] < bucket_ts:
                    sidecar_out.write(sidecar_rows[side_i][1] + "\n")
                    side_i += 1
                    sidecar_rows_written += 1
                sidecar_out.flush()

                tb.process_tick(tick)
                fed_total += 1
                if (from_ts is None) or (ts >= from_ts):
                    fed_window += 1
    finally:
        tb.close()

    summary = {
        "ticks_total": total,
        "ticks_fed_total": fed_total,
        "ticks_fed_window": fed_window,
        "ticks_dropped": dropped,
        "sidecar_rows_written": sidecar_rows_written,
        "warmup_minutes": warmup_minutes,
        "out_dir": str(out_dir),
        "signal_jsonl": str(out_dir / "replay_signal.jsonl"),
        "arm_jsonl": str(out_dir / "replay_arm.jsonl"),
    }
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
