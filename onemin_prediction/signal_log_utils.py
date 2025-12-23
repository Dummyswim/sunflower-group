#!/usr/bin/env python3
"""SignalContext log utilities (validation, loading, merge helpers)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, List, Optional, Tuple
from collections import deque

from signal_context import RECORD_VERSION, compute_schema_hash, validate_signal_context


def append_jsonl(path: str, rec: Dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, separators=(",", ":"), ensure_ascii=False) + "\n")


def _default_quarantine_path(train_log_path: str) -> str:
    base = Path(train_log_path)
    return str(base.with_name(base.stem + "_quarantine.jsonl"))


def validate_or_quarantine(
    rec: Dict[str, Any],
    *,
    schema_cols: Optional[List[str]],
    train_log_path: str,
    quarantine_path: Optional[str] = None,
) -> bool:
    errors = validate_signal_context(rec, schema_cols=schema_cols)
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


def load_signal_log(
    path: str,
    *,
    schema_cols: Optional[List[str]] = None,
    max_rows: int = 50000,
    require_scored: Optional[bool] = None,
    require_teacher_tradeable: Optional[bool] = None,
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
                errs = validate_signal_context(rec, schema_cols=schema_cols)
                if errs:
                    continue
            if require_scored is not None:
                scored = bool((rec.get("provenance") or {}).get("scored", False))
                if scored != bool(require_scored):
                    continue
            if require_teacher_tradeable is not None:
                if bool(rec.get("teacher_tradeable")) != bool(require_teacher_tradeable):
                    continue
            rows.append(rec)

    return list(rows)


def summarize_sources(rows: Iterable[Dict[str, Any]]) -> Dict[str, int]:
    counts = {"online": 0, "backfill": 0, "other": 0}
    for r in rows:
        prov = r.get("provenance") or {}
        src = prov.get("record_source_primary") or prov.get("record_source") or ""
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


def merge_signal_logs(
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

    for src_name, path in sources:
        p = Path(path)
        if not p.exists():
            continue
        with p.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line or not line.startswith("{"):
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                total_in += 1
                if rec.get("record_version") != RECORD_VERSION:
                    continue
                key = (
                    rec.get("symbol"),
                    rec.get("bar_min"),
                    rec.get("ts_target_close"),
                    rec.get("label_version"),
                    rec.get("feature_schema_hash"),
                )
                if key in seen:
                    total_dupe += 1
                    continue
                seen[key] = rec
                total_valid += 1

    rows = list(seen.values())
    rows.sort(key=lambda r: (r.get("ts_target_close") or "", r.get("symbol") or ""))

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for rec in rows:
            f.write(json.dumps(rec, separators=(",", ":"), ensure_ascii=False) + "\n")

    return {
        "total_in": total_in,
        "total_valid": total_valid,
        "total_dupe": total_dupe,
        "total_out": len(rows),
    }
