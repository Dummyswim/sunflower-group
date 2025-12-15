#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


def _load_schema_cols(schema_path: str) -> List[str]:
    obj = json.loads(Path(schema_path).read_text(encoding="utf-8"))
    if isinstance(obj, dict) and "columns" in obj and isinstance(obj["columns"], list):
        return [str(x) for x in obj["columns"]]
    if isinstance(obj, list):
        return [str(x) for x in obj]
    raise ValueError(f"Unrecognized schema format: {schema_path}")


def _iter_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                yield json.loads(ln)
            except Exception:
                continue


def _to_float_or_blank(v: Any):
    try:
        if v is None:
            return ""
        x = float(v)
        return x if np.isfinite(x) else ""
    except Exception:
        return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_log", default=os.getenv("TRAIN_LOG_PATH", "data/train_log_v2.jsonl"))
    ap.add_argument("--schema", default=os.getenv("FEATURE_SCHEMA_COLS_PATH", "trained_models/production/feature_schema_cols.json"))
    ap.add_argument("--out_daily", default=os.getenv("FEATURE_LOG", "runtime/feature_log.csv"))
    ap.add_argument("--out_hist", default=os.getenv("FEATURE_LOG_HIST", "runtime/feature_log_hist.csv"))
    ap.add_argument("--hist_cap", type=int, default=int(os.getenv("FEATURE_LOG_HIST_CAP", "200000")))
    ap.add_argument("--daily_tail", type=int, default=int(os.getenv("FEATURE_LOG_DAILY_TAIL", "4000")))
    args = ap.parse_args()

    schema_cols = _load_schema_cols(args.schema)

    meta_cols = [
        "ts", "decision", "label", "buy_prob", "alpha",
        "tradeable", "is_flat", "ticks", "neutral_prob",
        "aux_label_short"
    ]
    fieldnames = meta_cols + schema_cols

    rows: List[Dict[str, Any]] = []

    for r in _iter_jsonl(args.train_log):
        feats = r.get("features") or {}
        ts = r.get("ts_target_close") or r.get("ts")
        if ts is None:
            continue

        label = str(r.get("label", "FLAT"))
        tradeable = bool(r.get("tradeable")) if "tradeable" in r else True

        out: Dict[str, Any] = {k: "" for k in fieldnames}
        out["ts"] = str(ts)
        out["decision"] = "TRAINLOG"
        out["label"] = label
        out["tradeable"] = tradeable
        out["is_flat"] = (label == "FLAT")
        out["ticks"] = int(r.get("ticks", 0) or 0)

        # probs may not exist in trainlog; keep blank if missing
        out["buy_prob"] = _to_float_or_blank(r.get("buy_prob"))
        out["neutral_prob"] = _to_float_or_blank(r.get("neutral_prob"))
        out["alpha"] = _to_float_or_blank(r.get("alpha"))

        # optional short-horizon label
        out["aux_label_short"] = _to_float_or_blank(r.get("aux_label_short"))

        # schema features (fixed)
        for c in schema_cols:
            out[c] = _to_float_or_blank(feats.get(c))

        rows.append(out)

    if not rows:
        raise SystemExit("No rows parsed from train log.")

    df = pd.DataFrame(rows)

    # de-dupe by ts (keep last)
    if "ts" in df.columns:
        df = df.drop_duplicates(subset=["ts"], keep="last")

    # cap hist size
    if args.hist_cap > 0 and len(df) > args.hist_cap:
        df = df.tail(args.hist_cap)

    Path(Path(args.out_hist).parent).mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_hist, index=False)

    # daily = tail of hist
    df_daily = df.tail(args.daily_tail) if args.daily_tail > 0 else df
    Path(Path(args.out_daily).parent).mkdir(parents=True, exist_ok=True)
    df_daily.to_csv(args.out_daily, index=False)

    print(f"Wrote hist={args.out_hist} rows={len(df)} cols={len(df.columns)}")
    print(f"Wrote daily={args.out_daily} rows={len(df_daily)} cols={len(df_daily.columns)}")


if __name__ == "__main__":
    main()
