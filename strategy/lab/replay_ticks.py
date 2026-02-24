#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
import sys
from typing import Dict, Iterator, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scalp_multi_engines import EngineManager, MultiEngineRelay, OneMinuteCandleBuilder, _build_engines


def _build_logger(log_path: str, verbose: bool) -> logging.Logger:
    logger = logging.getLogger("tick_replay")
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh = logging.FileHandler(log_path)
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    return logger


def _iter_ticks(path: Path) -> Iterator[Tuple[int, Dict]]:
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            raw = line.strip()
            if not raw:
                continue
            row = json.loads(raw)
            tick = row.get("tick") if isinstance(row, dict) and isinstance(row.get("tick"), dict) else row
            if isinstance(tick, dict) and tick.get("kind") == "dhan_quote_packet":
                yield i, tick


def main() -> int:
    ap = argparse.ArgumentParser(description="Replay recorded tick stream through strategy engines.")
    ap.add_argument("--ticks", required=True, help="Input tick NDJSON file (recorded by STRAT_TICK_LOG_ENABLED).")
    ap.add_argument("--jsonl-out", required=True, help="Output strategy JSONL path.")
    ap.add_argument("--log-out", required=True, help="Output replay log path.")
    ap.add_argument("--gap-fill", action="store_true", help="Enable synthetic gap-fill candles during replay.")
    ap.add_argument("--max-gap-fill-minutes", type=int, default=3, help="Max synthetic candles to fill per gap.")
    ap.add_argument("--verbose", action="store_true", help="Enable debug logs.")
    ap.add_argument("--truncate-output", action="store_true", help="Delete output files before replay.")
    args = ap.parse_args()

    ticks_path = Path(args.ticks)
    jsonl_out = Path(args.jsonl_out)
    log_out = Path(args.log_out)

    if not ticks_path.exists():
        raise SystemExit(f"Tick file not found: {ticks_path}")

    jsonl_out.parent.mkdir(parents=True, exist_ok=True)
    log_out.parent.mkdir(parents=True, exist_ok=True)
    if args.truncate_output:
        if jsonl_out.exists():
            jsonl_out.unlink()
        if log_out.exists():
            log_out.unlink()

    log = _build_logger(str(log_out), verbose=args.verbose)
    log.info("[REPLAY] ticks=%s jsonl_out=%s", ticks_path, jsonl_out)

    # Keep replay deterministic and self-contained.
    os.environ["STRAT_JSONL"] = str(jsonl_out)
    os.environ["STRAT_MODE"] = os.getenv("STRAT_MODE", "manual")
    os.environ["STRAT_WRITE_CANDLES"] = os.getenv("STRAT_WRITE_CANDLES", "1")
    os.environ["STRAT_WRITE_HOLD"] = os.getenv("STRAT_WRITE_HOLD", "1")

    engines = _build_engines(log=log)
    if not engines:
        raise SystemExit("No engines enabled for replay. Set ENABLE_ENGINES.")

    mgr = EngineManager(
        engines=engines,
        jsonl_path=str(jsonl_out),
        log=log,
        write_hold=True,
        write_candles=True,
    )
    relay = MultiEngineRelay(
        mgr=mgr,
        candle_builder=OneMinuteCandleBuilder(
            gap_fill=bool(args.gap_fill),
            max_gap_fill_minutes=max(0, int(args.max_gap_fill_minutes)),
        ),
        log=log,
        tick_recorder=None,
    )

    seen = 0
    bad = 0
    for ln, tick in _iter_ticks(ticks_path):
        try:
            relay.on_tick(tick)
            seen += 1
        except Exception as e:
            bad += 1
            log.exception("[REPLAY] failed tick line=%s err=%s", ln, e)

    log.info("[REPLAY] completed ticks_seen=%s ticks_failed=%s", seen, bad)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
