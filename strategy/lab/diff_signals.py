#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Optional, Tuple


def _suggestion_map(path: Path) -> Dict[str, Tuple]:
    out: Dict[str, Tuple] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            raw = line.strip()
            if not raw:
                continue
            row = json.loads(raw)
            if row.get("kind") != "DECISION":
                continue
            if row.get("decision") != "SUGGEST":
                continue
            ts = str(row.get("candle_end_ist") or "")
            w = row.get("winner") or {}
            key = (
                str(w.get("engine") or ""),
                str(w.get("direction") or ""),
                round(float(w.get("entry_price") or 0.0), 4),
                round(float(w.get("stop_loss") or 0.0), 4),
                round(float(w.get("take_profit_1") or 0.0), 4),
                int(w.get("quality_score") or 0),
            )
            if ts:
                out[ts] = key
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Diff SUGGEST decisions between two strategy JSONL runs.")
    ap.add_argument("--baseline", required=True, help="Baseline strategy JSONL.")
    ap.add_argument("--candidate", required=True, help="Candidate strategy JSONL.")
    ap.add_argument("--csv-out", default="", help="Optional CSV output for per-minute changes.")
    args = ap.parse_args()

    baseline = Path(args.baseline)
    candidate = Path(args.candidate)
    if not baseline.exists():
        raise SystemExit(f"Missing baseline: {baseline}")
    if not candidate.exists():
        raise SystemExit(f"Missing candidate: {candidate}")

    b = _suggestion_map(baseline)
    c = _suggestion_map(candidate)

    keys = sorted(set(b.keys()) | set(c.keys()))
    same = 0
    changed = 0
    added = 0
    removed = 0
    rows = []

    for ts in keys:
        bv: Optional[Tuple] = b.get(ts)
        cv: Optional[Tuple] = c.get(ts)
        if bv is None and cv is not None:
            added += 1
            rows.append((ts, "added", "", repr(cv)))
        elif bv is not None and cv is None:
            removed += 1
            rows.append((ts, "removed", repr(bv), ""))
        elif bv == cv:
            same += 1
        else:
            changed += 1
            rows.append((ts, "changed", repr(bv), repr(cv)))

    print("=== Signal Diff Summary ===")
    print(f"baseline_suggests={len(b)}")
    print(f"candidate_suggests={len(c)}")
    print(f"same={same}")
    print(f"changed={changed}")
    print(f"added={added}")
    print(f"removed={removed}")

    if args.csv_out:
        out = Path(args.csv_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["candle_end_ist", "status", "baseline", "candidate"])
            w.writerows(rows)
        print(f"wrote_csv={out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

