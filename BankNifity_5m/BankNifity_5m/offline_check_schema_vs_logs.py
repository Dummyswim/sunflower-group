#!/usr/bin/env python3
"""
Quick sanity check: feature_schema.json vs feature_log(_hist).csv

- Prints which columns are expected by the model schema.
- Shows which of those are missing in logs, and which log columns are unused.
"""

import json
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent

SCHEMA_PATH = PROJECT_ROOT / "feature_schema.json"
LOG_PATH = PROJECT_ROOT / "trained_models" / "production" / "feature_log.csv"
LOG_HIST_PATH = PROJECT_ROOT / "trained_models" / "production" / "feature_log_hist.csv"


def main() -> None:
    if not SCHEMA_PATH.exists():
        print(f"[CHECK] No schema at {SCHEMA_PATH}")
        return

    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    feat_names = schema.get("feature_names", [])
    print(f"[CHECK] Schema features (n={len(feat_names)}):")
    print(" ", ", ".join(feat_names))

    # Use hist if available, otherwise current
    log_p = LOG_HIST_PATH if LOG_HIST_PATH.exists() else LOG_PATH
    if not log_p.exists():
        print(f"[CHECK] No feature log found at {log_p}")
        return

    df_head = pd.read_csv(log_p, nrows=5)
    log_cols = df_head.columns.tolist()

    print(f"\n[CHECK] Log columns (n={len(log_cols)}) from {log_p.name}:")
    print(" ", ", ".join(log_cols))

    schema_set = set(feat_names)
    log_set = set(log_cols)

    missing_in_logs = schema_set - log_set
    extra_in_logs = log_set - schema_set

    print("\n[CHECK] Features expected by model but missing in logs:")
    print(" ", sorted(missing_in_logs) if missing_in_logs else "  (none)")

    print("\n[CHECK] Log columns not used by the current model schema:")
    print(" ", sorted(extra_in_logs) if extra_in_logs else "  (none)")


if __name__ == "__main__":
    main()
