#!/usr/bin/env python3
"""Merge v3 train logs into a canonical file with deterministic sort/dedupe."""
from __future__ import annotations

import argparse

from train_log_utils_v3 import merge_train_logs_v3


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

    stats = merge_train_logs_v3(sources, out_path=args.out)
    print(
        "merged_rows_in={rows_in} merged_rows_valid={rows_valid} merged_rows_out={rows_out} "
        "merged_rows_dupe={rows_dupe} bad_version={rows_bad_version} parse_err={rows_parse_err} "
        "out_of_order={out_of_order} out={out}".format(out=args.out, **stats)
    )
    for name, s in (stats.get("source_stats") or {}).items():
        print(
            f"source={name} in={s.get('in', 0)} kept={s.get('kept', 0)} "
            f"dupe={s.get('dupe', 0)} invalid={s.get('invalid', 0)} "
            f"parse_err={s.get('parse_err', 0)}"
        )


if __name__ == "__main__":
    main()
