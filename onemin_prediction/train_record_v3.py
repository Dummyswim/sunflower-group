#!/usr/bin/env python3
"""
Train record contract (v3) + timestamp normalization + schema hashing.
"""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, Iterable, List, Optional, Tuple

IST = timezone(timedelta(hours=5, minutes=30))

RECORD_VERSION = "v3"


def _is_tz_aware(dt: datetime) -> bool:
    return dt.tzinfo is not None and dt.tzinfo.utcoffset(dt) is not None


def normalize_ts_ist(ts: Any) -> Optional[str]:
    """Normalize timestamp to IST ISO-8601 with timezone offset (+05:30)."""
    if ts is None:
        return None
    if isinstance(ts, datetime):
        dt = ts
    else:
        try:
            # Try ISO first; fall back to pandas-style string handling via fromisoformat
            dt = datetime.fromisoformat(str(ts))
        except Exception:
            return None

    if not _is_tz_aware(dt):
        dt = dt.replace(tzinfo=IST)
    else:
        dt = dt.astimezone(IST)

    dt = dt.replace(microsecond=0)
    return dt.isoformat()


def compute_schema_hash(cols: Iterable[str]) -> str:
    joined = "\n".join([str(c) for c in cols])
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def align_features_to_schema(
    features: Dict[str, Any],
    schema_cols: List[str],
) -> Tuple[Dict[str, float], List[str], List[str]]:
    missing: List[str] = []
    extra: List[str] = []

    if features is None:
        features = {}

    feat_keys = set(features.keys())
    schema_set = set(schema_cols)

    missing = [k for k in schema_cols if k not in feat_keys]
    extra = [k for k in feat_keys if k not in schema_set]

    out: Dict[str, float] = {}
    for k in schema_cols:
        v = features.get(k)
        try:
            fv = float(v)
        except Exception:
            fv = float("nan")
        out[k] = fv

    return out, missing, extra


def build_train_record_v3(
    *,
    schema_cols: List[str],
    schema_version: str,
    label_version: str,
    pipeline_version: str,
    symbol: str,
    bar_min: int,
    horizon_min: int,
    ts_ref_start: Any,
    ts_target_close: Any,
    label: str,
    label_source: str,
    label_weight: float,
    buy_prob: float,
    alpha: float,
    tradeable: bool,
    is_flat: bool,
    tick_count: int,
    features: Dict[str, Any],
    meta: Optional[Dict[str, Any]] = None,
) -> Tuple[Optional[Dict[str, Any]], List[str]]:
    errors: List[str] = []

    if not schema_cols:
        errors.append("schema_cols_missing")

    ts_ref_norm = normalize_ts_ist(ts_ref_start)
    ts_tgt_norm = normalize_ts_ist(ts_target_close)
    if not ts_ref_norm or not ts_tgt_norm:
        errors.append("timestamp_invalid_or_missing")

    if not schema_version:
        errors.append("schema_version_missing")
    if not label_version:
        errors.append("label_version_missing")
    if not pipeline_version:
        errors.append("pipeline_version_missing")

    aligned, missing, extra = align_features_to_schema(features, schema_cols)
    if missing:
        errors.append(f"missing_features:{','.join(missing)}")
    if extra:
        errors.append(f"extra_features:{','.join(extra)}")

    # Reject NaN/inf values
    for k, v in aligned.items():
        if v != v or v in (float("inf"), float("-inf")):
            errors.append(f"non_finite_feature:{k}")
            break

    if errors:
        return None, errors

    rec = {
        "record_version": RECORD_VERSION,
        "schema_version": str(schema_version),
        "feature_schema_hash": compute_schema_hash(schema_cols),
        "label_version": str(label_version),
        "pipeline_version": str(pipeline_version),
        "symbol": str(symbol),
        "bar_min": int(bar_min),
        "horizon_min": int(horizon_min),
        "ts_ref_start": ts_ref_norm,
        "ts_target_close": ts_tgt_norm,
        "label": str(label),
        "label_source": str(label_source),
        "label_weight": float(label_weight),
        "buy_prob": float(buy_prob),
        "alpha": float(alpha),
        "tradeable": bool(tradeable),
        "is_flat": bool(is_flat),
        "tick_count": int(tick_count),
        "features": aligned,
        "meta": meta or {},
    }
    return rec, []


def validate_train_record_v3(
    rec: Dict[str, Any],
    schema_cols: Optional[List[str]] = None,
) -> List[str]:
    errors: List[str] = []
    if not isinstance(rec, dict):
        return ["record_not_dict"]

    if rec.get("record_version") != RECORD_VERSION:
        errors.append("record_version_mismatch")

    for key in (
        "schema_version",
        "feature_schema_hash",
        "label_version",
        "pipeline_version",
        "symbol",
        "bar_min",
        "horizon_min",
        "ts_ref_start",
        "ts_target_close",
        "label",
        "label_source",
        "label_weight",
        "buy_prob",
        "alpha",
        "tradeable",
        "is_flat",
        "tick_count",
        "features",
    ):
        if key not in rec:
            errors.append(f"missing_field:{key}")

    for ts_key in ("ts_ref_start", "ts_target_close"):
        ts_val = rec.get(ts_key)
        if ts_val:
            norm = normalize_ts_ist(ts_val)
            if not norm:
                errors.append(f"timestamp_invalid:{ts_key}")
        else:
            errors.append(f"timestamp_missing:{ts_key}")

    feats = rec.get("features")
    if not isinstance(feats, dict):
        errors.append("features_not_dict")
    else:
        if schema_cols:
            schema_set = set(schema_cols)
            feat_set = set(feats.keys())
            missing = [k for k in schema_cols if k not in feat_set]
            extra = [k for k in feat_set if k not in schema_set]
            if missing:
                errors.append(f"missing_features:{','.join(missing)}")
            if extra:
                errors.append(f"extra_features:{','.join(extra)}")

        # finite check
        for k, v in feats.items():
            try:
                fv = float(v)
            except Exception:
                errors.append(f"feature_not_float:{k}")
                break
            if fv != fv or fv in (float("inf"), float("-inf")):
                errors.append(f"non_finite_feature:{k}")
                break

    return errors
