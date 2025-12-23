#!/usr/bin/env python3
"""
Online trainer for rule-as-teacher policy models (BUY/SELL) with bundle promotion.

Env:
- TRAIN_LOG_PATH: SignalContext JSONL file
- MODEL_BUNDLES_DIR: default 'trained_models/bundles'
- MODEL_PRODUCTION_LINK: default 'trained_models/production'
- FEATURE_SCHEMA_COLS_PATH: base schema file (default: trained_models/production/feature_schema_cols.json)
- POLICY_SCHEMA_COLS_PATH: policy schema file (default: trained_models/production/policy_schema_cols.json)
- POLICY_BUY_PATH / POLICY_SELL_PATH: optional copy-out paths
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

try:
    import xgboost as xgb
except Exception:
    xgb = None

from eval_before_promotion import evaluate_holdout
from model_bundle import (
    make_bundle_dir,
    bundle_paths,
    atomic_write_json,
    atomic_write_text,
    write_manifest,
    validate_bundle_feature_counts,
    promote_bundle_symlink,
)
from pretrain_validator import validate_pretrain
from schema_contract import SchemaResolutionError
from signal_context import align_features_to_schema, compose_policy_features, compose_policy_schema
from signal_log_utils import load_signal_log, summarize_sources, data_range

logger = logging.getLogger(__name__)


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        if np.isfinite(v):
            return v
    except Exception:
        pass
    return float(default)


def _load_schema_cols(path: Path) -> Optional[List[str]]:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None

    if isinstance(data, dict):
        cols = data.get("columns") or data.get("feature_names") or data.get("features")
        if isinstance(cols, list) and all(isinstance(x, str) for x in cols) and cols:
            return list(cols)
        if "columns" in data and isinstance(data["columns"], dict):
            return list(data["columns"].keys())

    if isinstance(data, list) and all(isinstance(x, str) for x in data) and data:
        return list(data)

    return None


def _get_schema_from_pipeline(pipeline_ref) -> Optional[List[str]]:
    try:
        names = getattr(pipeline_ref, "feature_schema_names", None)
        if names and isinstance(names, list) and all(isinstance(x, str) for x in names):
            return list(names)
    except Exception:
        pass
    return None


def _load_base_schema() -> Optional[List[str]]:
    sp = os.getenv("FEATURE_SCHEMA_COLS_PATH", "").strip()
    if sp:
        cols = _load_schema_cols(Path(sp))
        if cols:
            return cols

    cols = _load_schema_cols(Path("data/feature_schema_cols.json"))
    if cols:
        return cols

    production_link = Path(os.getenv("MODEL_PRODUCTION_LINK", "trained_models/production"))
    if production_link.exists():
        try:
            resolved = production_link.resolve()
            cols = _load_schema_cols(resolved / "feature_schema_cols.json")
            if cols:
                return cols
        except Exception:
            pass

    cols = _load_schema_cols(production_link.parent / "feature_schema_cols.json")
    if cols:
        return cols

    return None


def _load_policy_schema() -> Optional[List[str]]:
    sp = os.getenv("POLICY_SCHEMA_COLS_PATH", "").strip()
    if sp:
        cols = _load_schema_cols(Path(sp))
        if cols:
            return cols

    production_link = Path(os.getenv("MODEL_PRODUCTION_LINK", "trained_models/production"))
    if production_link.exists():
        try:
            resolved = production_link.resolve()
            cols = _load_schema_cols(resolved / "policy_schema_cols.json")
            if cols:
                return cols
        except Exception:
            pass

    cols = _load_schema_cols(production_link.parent / "policy_schema_cols.json")
    if cols:
        return cols

    return None


def _infer_base_schema_from_rows(rows: Iterable[Dict[str, Any]], max_rows: int = 2000) -> Optional[List[str]]:
    keys: set[str] = set()
    for i, r in enumerate(rows):
        if i >= max_rows:
            break
        feats = r.get("features") or {}
        for k in feats.keys():
            keys.add(str(k))

    if not keys:
        return None
    return sorted(keys)


def _infer_policy_schema_from_rows(rows: Iterable[Dict[str, Any]], max_rows: int = 2000) -> Optional[List[str]]:
    keys: set[str] = set()
    for i, r in enumerate(rows):
        if i >= max_rows:
            break
        feats = r.get("features") or {}
        rule_signals = r.get("rule_signals") or {}
        gates = r.get("gates") or {}
        teacher_strength = _safe_float(r.get("teacher_strength", 0.0), 0.0)
        pol = compose_policy_features(
            features=feats,
            rule_signals=rule_signals,
            gates=gates,
            teacher_strength=teacher_strength,
        )
        for k in pol.keys():
            keys.add(str(k))

    if not keys:
        return None
    return sorted(keys)


@dataclass
class PolicyDataset:
    X: np.ndarray
    y: np.ndarray
    w: np.ndarray
    cols: List[str]
    n_rows: int
    pos: int
    neg: int


def _build_matrix(rows: List[Dict[str, Any]], cols: List[str], teacher_dir: str) -> PolicyDataset:
    X = np.zeros((len(rows), len(cols)), dtype=np.float32)
    y = np.zeros((len(rows),), dtype=np.int32)
    w = np.ones((len(rows),), dtype=np.float32)

    dir_norm = str(teacher_dir).upper()
    for i, r in enumerate(rows):
        feats = r.get("features") or {}
        rule_signals = r.get("rule_signals") or {}
        gates = r.get("gates") or {}
        teacher_strength = _safe_float(r.get("teacher_strength", 0.0), 0.0)
        pol = compose_policy_features(
            features=feats,
            rule_signals=rule_signals,
            gates=gates,
            teacher_strength=teacher_strength,
        )
        aligned, _, _ = align_features_to_schema(pol, cols)
        for j, c in enumerate(cols):
            X[i, j] = _safe_float(aligned.get(c, 0.0), 0.0)
        lab = str(r.get("label", "FLAT")).upper()
        y[i] = 1 if lab == dir_norm else 0
        w[i] = _safe_float(r.get("label_weight", 1.0), 1.0)

    pos = int(np.sum(y == 1))
    neg = int(np.sum(y == 0))
    return PolicyDataset(X=X, y=y, w=w, cols=list(cols), n_rows=int(len(rows)), pos=pos, neg=neg)


def build_policy_dataset(rows: List[Dict[str, Any]], schema_cols: List[str], teacher_dir: str) -> PolicyDataset:
    dir_norm = str(teacher_dir).upper()
    subset = [
        r for r in rows
        if str(r.get("teacher_dir", "")).upper() == dir_norm
        and bool(r.get("teacher_tradeable"))
    ]
    return _build_matrix(subset, list(schema_cols), dir_norm)


def train_policy_model(data: PolicyDataset, *, seed: int = 7) -> Optional[Any]:
    if xgb is None or data.X.size == 0:
        return None

    pos = max(1, data.pos)
    neg = max(1, data.neg)
    try:
        scale_pos_weight = float(os.getenv("POLICY_XGB_SCALE_POS_WEIGHT", ""))
    except Exception:
        scale_pos_weight = float(neg / pos) if pos else 1.0

    params = {
        "n_estimators": int(os.getenv("POLICY_XGB_N_ESTIMATORS", "300")),
        "max_depth": int(os.getenv("POLICY_XGB_MAX_DEPTH", "4")),
        "learning_rate": float(os.getenv("POLICY_XGB_LR", "0.05")),
        "subsample": float(os.getenv("POLICY_XGB_SUBSAMPLE", "0.9")),
        "colsample_bytree": float(os.getenv("POLICY_XGB_COLSAMPLE", "0.9")),
        "reg_lambda": float(os.getenv("POLICY_XGB_L2", "1.0")),
        "min_child_weight": float(os.getenv("POLICY_XGB_MIN_CHILD_WEIGHT", "1.0")),
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "n_jobs": int(os.getenv("POLICY_XGB_N_JOBS", "4")),
        "random_state": int(os.getenv("POLICY_XGB_SEED", str(seed)) or seed),
        "scale_pos_weight": float(scale_pos_weight),
    }

    clf = xgb.XGBClassifier(**params)
    clf.fit(data.X, data.y, sample_weight=data.w)
    try:
        meta = json.dumps({"feature_names": list(data.cols)}, separators=(",", ":"))
        clf.get_booster().set_attr(feature_schema=meta)
    except Exception as e:
        logger.warning("[TRAIN] failed embedding feature_schema: %s", e)
    return clf


def _write_bundle_and_promote(
    *,
    buy_model,
    sell_model,
    schema_cols: List[str],
    base_schema_cols: Optional[List[str]],
    notes: Dict[str, Any],
    data_range_info: Optional[Dict[str, Any]] = None,
    source_counts: Optional[Dict[str, Any]] = None,
    validation_report: Optional[Dict[str, Any]] = None,
    buy_out_path: str,
    sell_out_path: str,
) -> Path:
    bundles_dir = Path(os.getenv("MODEL_BUNDLES_DIR", "trained_models/bundles"))
    production_link = Path(os.getenv("MODEL_PRODUCTION_LINK", "trained_models/production"))

    bundle_dir = make_bundle_dir(bundles_dir)
    bp = bundle_paths(bundle_dir)

    buy_model.save_model(str(bp.policy_buy_path))
    sell_model.save_model(str(bp.policy_sell_path))

    atomic_write_json(bp.schema_cols_json, {"columns": list(schema_cols)})
    atomic_write_text(bp.schema_cols_txt, "\n".join(schema_cols))
    if base_schema_cols:
        atomic_write_json(bp.base_schema_cols_json, {"columns": list(base_schema_cols)})
        atomic_write_text(bp.base_schema_cols_txt, "\n".join(base_schema_cols))

    calib_buy = os.getenv("CALIB_BUY_PATH", "").strip()
    calib_sell = os.getenv("CALIB_SELL_PATH", "").strip()
    if calib_buy:
        try:
            csrc = Path(calib_buy)
            if csrc.exists():
                bp.calib_buy_path.write_bytes(csrc.read_bytes())
        except Exception:
            pass
    if calib_sell:
        try:
            csrc = Path(calib_sell)
            if csrc.exists():
                bp.calib_sell_path.write_bytes(csrc.read_bytes())
        except Exception:
            pass

    ok, reason = validate_bundle_feature_counts(
        bp.policy_buy_path,
        bp.policy_sell_path,
        list(schema_cols),
    )
    if not ok:
        raise RuntimeError(f"bundle validation failed: {reason}")

    write_manifest(
        bp.manifest_path,
        schema_cols=list(schema_cols),
        notes=notes,
        data_range=data_range_info,
        source_counts=source_counts,
        validation_report=validation_report,
    )

    promote_bundle_symlink(bundle_dir, production_link)

    try:
        Path(buy_out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(buy_out_path).write_bytes(bp.policy_buy_path.read_bytes())
        Path(sell_out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(sell_out_path).write_bytes(bp.policy_sell_path.read_bytes())
    except Exception:
        pass

    return bundle_dir


def _train_and_promote_sync(
    feature_log_path: str,
    buy_out_path: str,
    sell_out_path: str,
    min_rows: int,
    max_rows: int,
    pipeline_ref=None,
) -> Dict[str, Any]:
    policy_schema = _get_schema_from_pipeline(pipeline_ref) or _load_policy_schema()
    base_schema = _load_base_schema()

    if base_schema:
        rows = load_signal_log(feature_log_path, max_rows=max_rows, schema_cols=list(base_schema))
    else:
        rows = load_signal_log(feature_log_path, max_rows=max_rows, schema_cols=None)
        base_schema = _infer_base_schema_from_rows(rows)

    if not policy_schema and base_schema:
        policy_schema = compose_policy_schema(list(base_schema))
    if not policy_schema:
        policy_schema = _infer_policy_schema_from_rows(rows)

    if not policy_schema:
        raise SchemaResolutionError("POLICY_SCHEMA_COLS_PATH missing and schema inference failed")

    if len(rows) < int(min_rows):
        return {"status": "wait", "rows": len(rows)}

    ok, report = validate_pretrain(rows, schema_cols=list(policy_schema))
    if not ok:
        reasons = report.get("reasons") or []
        soft_block = {"min_rows", "low_live_coverage"}
        if reasons and set(reasons).issubset(soft_block):
            return {
                "status": "wait",
                "reason": "pretrain_insufficient",
                "report": report,
                "rows": len(rows),
            }
        return {"status": "error", "reason": "pretrain_validation_failed", "report": report}

    buy_data = build_policy_dataset(rows, schema_cols=policy_schema, teacher_dir="BUY")
    sell_data = build_policy_dataset(rows, schema_cols=policy_schema, teacher_dir="SELL")

    buy_model = train_policy_model(buy_data)
    sell_model = train_policy_model(sell_data)

    if buy_model is None or sell_model is None:
        return {
            "status": "skip",
            "rows": len(rows),
            "buy_rows": int(buy_data.n_rows),
            "sell_rows": int(sell_data.n_rows),
            "cols": len(schema),
        }

    eval_ok, eval_report = evaluate_holdout(rows, list(policy_schema), buy_model, sell_model)
    if not eval_ok:
        return {"status": "error", "reason": "eval_gate_failed", "report": eval_report}

    notes = {
        "trainer": "online_trainer_policy",
        "rows": len(rows),
        "buy_rows": int(buy_data.n_rows),
        "sell_rows": int(sell_data.n_rows),
        "min_rows": int(min_rows),
        "pretrain_report": report,
        "eval_report": eval_report,
    }
    src_counts = summarize_sources(rows)
    drange = data_range(rows)

    bundle_dir = _write_bundle_and_promote(
        buy_model=buy_model,
        sell_model=sell_model,
        schema_cols=list(policy_schema),
        base_schema_cols=list(base_schema) if base_schema else None,
        notes=notes,
        data_range_info=drange,
        source_counts=src_counts,
        validation_report={"pretrain": report, "eval": eval_report},
        buy_out_path=buy_out_path,
        sell_out_path=sell_out_path,
    )

    return {
        "status": "ok",
        "bundle_dir": bundle_dir,
        "buy_model": buy_model,
        "sell_model": sell_model,
        "rows": len(rows),
        "buy_rows": int(buy_data.n_rows),
        "sell_rows": int(sell_data.n_rows),
        "cols": len(policy_schema),
    }


async def background_trainer_loop(
    feature_log_path: str,
    buy_out_path: str,
    sell_out_path: str,
    min_rows: int = 500,
    interval_sec: float = 30.0,
    pipeline_ref=None,
):
    last_mtime = None

    while True:
        try:
            p = Path(feature_log_path)
            if p.exists():
                try:
                    mtime = p.stat().st_mtime
                except Exception:
                    mtime = None

                if mtime and last_mtime and mtime == last_mtime:
                    await asyncio.sleep(interval_sec)
                    continue

                max_rows = int(os.getenv("TRAIN_MAX_ROWS", "50000"))
                result = await asyncio.to_thread(
                    _train_and_promote_sync,
                    str(p),
                    buy_out_path,
                    sell_out_path,
                    int(min_rows),
                    int(max_rows),
                    pipeline_ref,
                )

                status = result.get("status")
                if status == "wait":
                    report = result.get("report") or {}
                    reasons = report.get("reasons") or []
                    if reasons:
                        live_report = report.get("live_coverage") or {}
                        coverage = float(live_report.get("coverage", 0.0))
                        min_cov = float(live_report.get("min_coverage", 0.0))
                        src = live_report.get("source_counts") or {}
                        logger.info(
                            "[TRAIN] waiting: pretrain %s coverage=%.3f min=%.3f online=%d backfill=%d",
                            ",".join(str(r) for r in reasons),
                            coverage,
                            min_cov,
                            int(src.get("online", 0)),
                            int(src.get("backfill", 0)),
                        )
                    else:
                        logger.info("[TRAIN] waiting: rows=%d", int(result.get("rows", 0)))
                elif status == "ok":
                    logger.info("[TRAIN] promoted bundle → %s", str(result.get("bundle_dir")))
                    if pipeline_ref is not None and hasattr(pipeline_ref, "replace_models"):
                        try:
                            pipeline_ref.replace_models(
                                buy_model=result.get("buy_model"),
                                sell_model=result.get("sell_model"),
                            )
                            logger.info("[TRAIN] pipeline_ref models replaced")
                        except Exception:
                            logger.debug("[TRAIN] pipeline_ref replace failed", exc_info=True)
                elif status == "error":
                    logger.error("[TRAIN] error: %s", result.get("reason"))
                elif status == "skip":
                    logger.info(
                        "[TRAIN] skipped: rows=%d buy_rows=%d sell_rows=%d cols=%d",
                        int(result.get("rows", 0)),
                        int(result.get("buy_rows", 0)),
                        int(result.get("sell_rows", 0)),
                        int(result.get("cols", 0)),
                    )

                last_mtime = mtime
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error("[TRAIN] loop error: %s", e, exc_info=True)
        await asyncio.sleep(interval_sec)
