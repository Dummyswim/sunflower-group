#!/usr/bin/env python3
"""Evaluate replay outputs (signal + arm JSONL) with compact KPIs."""

from __future__ import annotations

import argparse
import json
from collections import Counter, deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Deque, Dict, Iterable, Optional


@dataclass
class OpenTrade:
    side: str
    entry_ts: datetime
    entry_px: float


def _ts(x: object) -> Optional[datetime]:
    try:
        s = str(x or "")
        if not s:
            return None
        dt = datetime.fromisoformat(s)
        return dt if dt.tzinfo else None
    except Exception:
        return None


def _entry_side(obj: Dict[str, object]) -> str:
    s = str(obj.get("suggestion", "") or "")
    if "LONG" in s:
        return "LONG"
    if "SHORT" in s:
        return "SHORT"
    return str(obj.get("entry_side", "") or "")


def _exit_side(obj: Dict[str, object], open_trade: Optional[OpenTrade]) -> str:
    if open_trade is not None:
        return open_trade.side
    p = str(obj.get("pos_side_before", "") or obj.get("pos_side", "") or "")
    if p in ("LONG", "SHORT"):
        return p
    s = str(obj.get("suggestion", "") or "")
    if "LONG" in s:
        return "LONG"
    if "SHORT" in s:
        return "SHORT"
    return ""


def _pnl_points(side: str, entry_px: float, exit_px: float) -> float:
    if side == "LONG":
        return exit_px - entry_px
    if side == "SHORT":
        return entry_px - exit_px
    return 0.0


def _event_ts(obj: Dict[str, object]) -> Optional[datetime]:
    for key in ("ts_ist", "ts", "ts_utc", "_candle_ts_ist", "_candle_ts_utc", "_candle_ts"):
        dt = _ts(obj.get(key))
        if dt is not None:
            return dt
    return None


def _iter_jsonl(path: Path) -> Iterable[Dict[str, object]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict):
                yield obj


def main() -> int:
    ap = argparse.ArgumentParser(description="Evaluate replay JSONL outputs")
    ap.add_argument("--signal", required=True, help="Signal JSONL path")
    ap.add_argument("--arm", required=True, help="Arm JSONL path")
    ap.add_argument("--cost-per-trade", type=float, default=0.0, help="Fixed cost in points per completed trade")
    ap.add_argument("--from", dest="from_ts", default="", help="Inclusive ISO timestamp filter")
    ap.add_argument("--to", dest="to_ts", default="", help="Inclusive ISO timestamp filter")
    args = ap.parse_args()

    signal_path = Path(args.signal)
    arm_path = Path(args.arm)
    if not signal_path.exists():
        raise FileNotFoundError(signal_path)
    if not arm_path.exists():
        raise FileNotFoundError(arm_path)
    from_ts = _ts(args.from_ts) if args.from_ts else None
    to_ts = _ts(args.to_ts) if args.to_ts else None

    open_trade: Optional[OpenTrade] = None
    trades = 0
    wins = 0
    pnl_gross = 0.0
    side_count = Counter()
    side_wins = Counter()
    exit_reason_count = Counter()
    hold_reason_count = Counter()
    veto_reason_count = Counter()
    quick_stopouts = 0

    for obj in _iter_jsonl(signal_path):
        evt_ts = _event_ts(obj)
        if from_ts and evt_ts and evt_ts < from_ts:
            continue
        if to_ts and evt_ts and evt_ts > to_ts:
            continue
        suggestion = str(obj.get("suggestion", "") or "")
        if suggestion in ("HOLD_CONFLICT", "HOLD_WAIT_CONFIRM", "COOLDOWN_ACTIVE", "HOLD_WEAK_EDGE"):
            hold_reason_count[str(obj.get("reason", "") or "")] += 1
        if suggestion == "ENTRY_VETO":
            veto_reason_count[str(obj.get("reason", "") or "")] += 1

        if suggestion.startswith("ENTRY_") and suggestion != "ENTRY_VETO":
            if open_trade is None:
                side = _entry_side(obj)
                ts = _ts(obj.get("ts_ist") or obj.get("ts"))
                try:
                    px = float(obj.get("entry_px") or obj.get("px") or 0.0)
                except Exception:
                    px = 0.0
                if side in ("LONG", "SHORT") and ts is not None and px > 0.0:
                    open_trade = OpenTrade(side=side, entry_ts=ts, entry_px=px)
            continue

        if suggestion.startswith("EXIT_"):
            if open_trade is None:
                continue
            ts = _ts(obj.get("ts_ist") or obj.get("ts"))
            if ts is None:
                continue
            try:
                px = float(obj.get("px") or obj.get("sl_fill_px") or 0.0)
            except Exception:
                px = 0.0
            if px <= 0.0:
                continue

            side = _exit_side(obj, open_trade)
            pnl = _pnl_points(side, open_trade.entry_px, px)
            pnl_gross += pnl
            trades += 1
            side_count[side] += 1
            if pnl > 0:
                wins += 1
                side_wins[side] += 1
            exit_reason = str(obj.get("suggestion") or "")
            exit_reason_count[exit_reason] += 1

            hold_min = (ts - open_trade.entry_ts).total_seconds() / 60.0
            if exit_reason == "EXIT_INIT_SL" and hold_min <= 1.0:
                quick_stopouts += 1

            open_trade = None

    cost_total = float(args.cost_per_trade) * float(trades)
    pnl_net = pnl_gross - cost_total
    expectancy = pnl_net / trades if trades > 0 else 0.0
    win_rate = wins / trades if trades > 0 else 0.0

    # Arm churn stats
    arm_by_engine: Dict[str, Deque[str]] = {
        "micro": deque(),
        "ema915": deque(),
    }
    arm_events = Counter()
    arm_flips = Counter()

    for obj in _iter_jsonl(arm_path):
        evt_ts = _event_ts(obj)
        if from_ts and evt_ts and evt_ts < from_ts:
            continue
        if to_ts and evt_ts and evt_ts > to_ts:
            continue
        eng = str(obj.get("engine", "") or "")
        if eng not in ("micro", "ema915"):
            continue
        sug = str(obj.get("suggestion", "") or "")
        side = "LONG" if "LONG" in sug else ("SHORT" if "SHORT" in sug else "")
        if side not in ("LONG", "SHORT"):
            continue
        arm_events[eng] += 1
        dq = arm_by_engine[eng]
        if dq and dq[-1] != side:
            arm_flips[eng] += 1
        dq.append(side)

    print("== Replay KPI ==")
    print(f"trades={trades}")
    print(f"win_rate={win_rate:.3f}")
    print(f"pnl_gross={pnl_gross:.2f}")
    print(f"cost_total={cost_total:.2f}")
    print(f"pnl_net={pnl_net:.2f}")
    print(f"expectancy={expectancy:.2f}")
    print(f"quick_stopouts_le_1m={quick_stopouts}")

    print("\n== Side Stats ==")
    for side in ("LONG", "SHORT"):
        n = side_count[side]
        w = side_wins[side]
        wr = (w / n) if n > 0 else 0.0
        print(f"{side}: trades={n} wins={w} win_rate={wr:.3f}")

    print("\n== Exit Reasons ==")
    for k, v in exit_reason_count.most_common():
        print(f"{k}: {v}")

    print("\n== Hold Reasons ==")
    for k, v in hold_reason_count.most_common(15):
        print(f"{k}: {v}")

    print("\n== Entry Veto Reasons ==")
    for k, v in veto_reason_count.most_common(15):
        print(f"{k}: {v}")

    print("\n== Arm Churn ==")
    for eng in ("micro", "ema915"):
        ev = arm_events[eng]
        flips = arm_flips[eng]
        fr = (flips / ev) if ev > 0 else 0.0
        print(f"{eng}: events={ev} flips={flips} flip_rate={fr:.3f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
