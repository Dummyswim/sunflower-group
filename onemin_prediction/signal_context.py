#!/usr/bin/env python3
"""
Canonical SignalContext contract (single source of truth for live + training).
"""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

IST = timezone(timedelta(hours=5, minutes=30))
RECORD_VERSION = "sc_v1"
DEFAULT_RULE_SIGNAL_KEYS = (
    "rule_sig",
    "flow_side",
    "flow_score",
    "vwap_side",
    "struct_side",
    "struct_score",
    "trend_side",
    "mtf_consensus",
    "indicator_score",
    "pattern_adj",
    "ta_rule",
)


def normalize_ts_ist(ts: Any) -> Optional[str]:
    """Normalize timestamp to IST ISO-8601 with timezone offset (+05:30)."""
    if ts is None:
        return None
    if isinstance(ts, datetime):
        dt = ts
    else:
        try:
            dt = datetime.fromisoformat(str(ts))
        except Exception:
            return None

    if dt.tzinfo is None or dt.tzinfo.utcoffset(dt) is None:
        dt = dt.replace(tzinfo=IST)
    else:
        dt = dt.astimezone(IST)

    dt = dt.replace(microsecond=0)
    return dt.isoformat()


def compute_schema_hash(cols: Iterable[str]) -> str:
    joined = "\n".join([str(c) for c in cols])
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        if np.isfinite(v):
            return v
    except Exception:
        pass
    return float(default)


def align_features_to_schema(
    features: Dict[str, Any],
    schema_cols: List[str],
) -> Tuple[Dict[str, float], List[str], List[str]]:
    if features is None:
        features = {}
    feat_keys = set(features.keys())
    schema_set = set(schema_cols)

    missing = [k for k in schema_cols if k not in feat_keys]
    extra = [k for k in feat_keys if k not in schema_set]

    aligned: Dict[str, float] = {}
    for k in schema_cols:
        aligned[k] = _safe_float(features.get(k), 0.0)

    return aligned, missing, extra


def build_signal_context(
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
    features: Dict[str, Any],
    rule_signals: Optional[Dict[str, Any]] = None,
    gates: Optional[Dict[str, Any]] = None,
    teacher_dir: str = "FLAT",
    teacher_tradeable: bool = False,
    teacher_strength: float = 0.0,
    provenance: Optional[Dict[str, Any]] = None,
    model: Optional[Dict[str, Any]] = None,
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

    if errors:
        return None, errors

    teacher_dir = str(teacher_dir or "FLAT").upper()
    if teacher_dir not in ("BUY", "SELL", "FLAT"):
        teacher_dir = "FLAT"

    prov = provenance or {}
    prov.setdefault("record_source", "unknown")
    prov.setdefault("scored", False)
    prov["feature_missing"] = missing
    prov["feature_extra"] = extra

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
        "teacher_dir": teacher_dir,
        "teacher_tradeable": bool(teacher_tradeable),
        "teacher_strength": float(teacher_strength),
        "rule_signals": rule_signals or {},
        "gates": gates or {},
        "features": aligned,
        "provenance": prov,
        "model": model or {},
    }

    return rec, []


def validate_signal_context(
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
        "teacher_dir",
        "teacher_tradeable",
        "teacher_strength",
        "rule_signals",
        "gates",
        "features",
        "provenance",
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
    elif schema_cols:
        _, missing, extra = align_features_to_schema(feats, schema_cols)
        if missing:
            errors.append(f"missing_features:{','.join(missing)}")
        if extra:
            errors.append(f"extra_features:{','.join(extra)}")

    return errors


def compose_policy_features(
    *,
    features: Dict[str, Any],
    rule_signals: Optional[Dict[str, Any]] = None,
    gates: Optional[Dict[str, Any]] = None,
    teacher_strength: float = 0.0,
) -> Dict[str, float]:
    """Create policy model feature map from base features + rule signals + gating."""
    out: Dict[str, float] = {}
    for k, v in (features or {}).items():
        out[str(k)] = _safe_float(v, 0.0)

    for k, v in (rule_signals or {}).items():
        out[f"rule_{k}"] = _safe_float(v, 0.0)

    out["teacher_strength"] = _safe_float(teacher_strength, 0.0)

    lane = str((gates or {}).get("lane", "")).upper()
    out["lane_code"] = 1.0 if lane == "SETUP" else (2.0 if lane == "TREND" else 0.0)

    conflict = str((gates or {}).get("tape_conflict_level", "")).lower()
    out["conflict_code"] = 2.0 if conflict == "red" else (1.0 if conflict == "yellow" else 0.0)

    gate_reasons = (gates or {}).get("gate_reasons") or []
    out["gate_count"] = _safe_float(len(gate_reasons), 0.0)

    return out


def compose_policy_schema(
    base_schema_cols: List[str],
    *,
    rule_signal_keys: Optional[Iterable[str]] = None,
) -> List[str]:
    """Build policy feature schema from base schema + rule/gate fields."""
    rule_keys = list(rule_signal_keys) if rule_signal_keys is not None else list(DEFAULT_RULE_SIGNAL_KEYS)
    out = list(base_schema_cols)
    out.extend([f"rule_{k}" for k in rule_keys])
    out.extend(["teacher_strength", "lane_code", "conflict_code", "gate_count"])
    return out
