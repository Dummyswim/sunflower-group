#!/usr/bin/env python3
"""Train log utilities for v3 records (validation, quarantine, loading, merge helpers)."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, List, Optional, Tuple
from collections import deque

from train_record_v3 import RECORD_VERSION, compute_schema_hash, validate_train_record_v3


def _default_quarantine_path(train_log_path: str) -> str:
    base = Path(train_log_path)
    return str(base.with_name(base.stem + "_quarantine.jsonl"))


def append_jsonl(path: str, rec: Dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, separators=(",", ":"), ensure_ascii=False) + "\n")


def validate_or_quarantine(
    rec: Dict[str, Any],
    *,
    schema_cols: Optional[List[str]],
    train_log_path: str,
    quarantine_path: Optional[str] = None,
) -> bool:
    errors = validate_train_record_v3(rec, schema_cols=schema_cols)
    if errors:
        qp = quarantine_path or _default_quarantine_path(train_log_path)
        qrec = {
            "record_version": RECORD_VERSION,
            "errors": errors,
            "record": rec,
        }
        append_jsonl(qp, qrec)
        return False
    return True


def load_train_log_v3(
    path: str,
    *,
    schema_cols: Optional[List[str]] = None,
    max_rows: int = 50000,
) -> List[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return []

    if max_rows and max_rows > 0:
        rows: Deque[Dict[str, Any]] = deque(maxlen=max_rows)
    else:
        rows = deque()
    target_hash = compute_schema_hash(schema_cols) if schema_cols else None

    with p.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or not line.startswith("{"):
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if rec.get("record_version") != RECORD_VERSION:
                continue
            if target_hash and rec.get("feature_schema_hash") != target_hash:
                continue
            if schema_cols:
                errs = validate_train_record_v3(rec, schema_cols=schema_cols)
                if errs:
                    continue
            rows.append(rec)

    return list(rows)


def summarize_sources(rows: Iterable[Dict[str, Any]]) -> Dict[str, int]:
    counts = {"online": 0, "backfill": 0, "other": 0}
    for r in rows:
        meta = r.get("meta") or {}
        src = meta.get("record_source_primary") or meta.get("record_source") or ""
        src = str(src).lower()
        if src == "online":
            counts["online"] += 1
        elif src == "backfill":
            counts["backfill"] += 1
        else:
            counts["other"] += 1
    return counts


def data_range(rows: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    ts_vals = [r.get("ts_target_close") for r in rows if r.get("ts_target_close")]
    if not ts_vals:
        return {}
    ts_vals = sorted(ts_vals)
    return {"start": ts_vals[0], "end": ts_vals[-1]}


def merge_train_logs_v3(
    sources: Iterable[Tuple[str, str]],
    *,
    out_path: str,
) -> Dict[str, Any]:
    """
    sources: iterable of (source_name, path)
    Dedup key: (symbol, bar_min, ts_target_close, label_version, feature_schema_hash)
    Keeps first seen (source order).
    """
    seen: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
    total_in = 0
    total_valid = 0
    total_dupe = 0
    total_parse_err = 0
    total_bad_version = 0
    out_of_order = 0
    source_stats: Dict[str, Dict[str, int]] = {}

    for src_name, path in sources:
        p = Path(path)
        if not p.exists():
            continue
        source_stats.setdefault(src_name, {"in": 0, "kept": 0, "dupe": 0, "invalid": 0, "parse_err": 0})
        last_ts = None
        with p.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line or not line.startswith("{"):
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    total_parse_err += 1
                    source_stats[src_name]["parse_err"] += 1
                    continue
                source_stats[src_name]["in"] += 1
                total_in += 1
                if rec.get("record_version") != RECORD_VERSION:
                    total_bad_version += 1
                    source_stats[src_name]["invalid"] += 1
                    continue
                total_valid += 1
                ts_val = rec.get("ts_target_close")
                if ts_val is not None and last_ts is not None and ts_val < last_ts:
                    out_of_order += 1
                if ts_val is not None:
                    last_ts = ts_val
                key = (
                    rec.get("symbol"),
                    rec.get("bar_min"),
                    rec.get("ts_target_close"),
                    rec.get("label_version"),
                    rec.get("feature_schema_hash"),
                )
                if key not in seen:
                    meta = rec.setdefault("meta", {})
                    meta["record_source_primary"] = src_name
                    meta["record_sources"] = [src_name]
                    seen[key] = rec
                    source_stats[src_name]["kept"] += 1
                else:
                    total_dupe += 1
                    source_stats[src_name]["dupe"] += 1
                    meta = seen[key].setdefault("meta", {})
                    srcs = meta.get("record_sources")
                    if isinstance(srcs, list) and src_name not in srcs:
                        srcs.append(src_name)
                    elif srcs is None:
                        meta["record_sources"] = [src_name]

    # sort by ts_target_close for deterministic output
    rows = list(seen.values())
    rows.sort(key=lambda r: (r.get("ts_target_close") or "", r.get("symbol") or ""))

    out_tmp = str(Path(out_path).with_suffix(".tmp"))
    out_dir = Path(out_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_tmp, "w", encoding="utf-8") as f:
        for rec in rows:
            f.write(json.dumps(rec, separators=(",", ":"), ensure_ascii=False) + "\n")
    os.replace(out_tmp, out_path)

    return {
        "rows_in": total_in,
        "rows_valid": total_valid,
        "rows_out": len(rows),
        "rows_dupe": total_dupe,
        "rows_bad_version": total_bad_version,
        "rows_parse_err": total_parse_err,
        "out_of_order": out_of_order,
        "source_stats": source_stats,
    }
