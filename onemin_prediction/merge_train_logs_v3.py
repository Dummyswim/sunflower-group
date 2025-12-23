#!/usr/bin/env python3
"""Merge SignalContext logs into a canonical file with deterministic sort/dedupe."""
from __future__ import annotations

import argparse

from signal_log_utils import merge_signal_logs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="Output canonical JSONL path")
    ap.add_argument("--source", action="append", default=[], help="NAME=PATH for each source")
    args = ap.parse_args()

    sources = []
    for item in args.source:
        if "=" not in item:
            raise SystemExit(f"invalid --source '{item}' (expected NAME=PATH)")
        name, path = item.split("=", 1)
        sources.append((name, path))

    if not sources:
        raise SystemExit("no sources provided")

    stats = merge_signal_logs(sources, out_path=args.out)
    print(
        "merged_rows_in={total_in} merged_rows_valid={total_valid} merged_rows_dupe={total_dupe} "
        "merged_rows_out={total_out} out={out}".format(out=args.out, **stats)
    )


if __name__ == "__main__":
    main()
