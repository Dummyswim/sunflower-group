# online_trainer_regen_v3.py
"""
Online trainer (v3) with model-bundle promotion.

Key long-term guarantees:
1) Schema + XGB + Neutral are written into a NEW bundle dir.
2) Bundle is validated (feature counts match).
3) 'trained_models/production' is atomically repointed (symlink swap).
4) Live pipeline hot-swaps models (optional) after promotion.

This prevents:
- schema cols becoming "1" due to partial/invalid writes
- model expects N != inference columns M
- half-written artifacts inside production

Env (important):
- TRAIN_LOG_PATH: JSONL file (v3 train records only)
- MODEL_BUNDLES_DIR: default 'trained_models/bundles'
- MODEL_PRODUCTION_LINK: default 'trained_models/production'
- XGB_OUT_PATH / NEUTRAL_OUT_PATH are still accepted for compatibility.

Notes:
- TRAIN_LOG_PATH must contain v3 train records; legacy CSV is not supported.
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
    import pandas as pd
except Exception:
    pd = None

try:
    import xgboost as xgb
except Exception:
    xgb = None

try:
    from sklearn.linear_model import LogisticRegression
except Exception:
    LogisticRegression = None

from model_bundle import (
    make_bundle_dir,
    bundle_paths,
    atomic_write_json,
    atomic_write_text,
    write_manifest,
    validate_bundle_feature_counts,
    promote_bundle_symlink,
)
from train_log_utils_v3 import load_train_log_v3, summarize_sources, data_range
from pretrain_validator import validate_pretrain
from eval_before_promotion import evaluate_holdout
from schema_contract import SchemaResolutionError

logger = logging.getLogger(__name__)


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        if np.isfinite(v):
            return v
    except Exception:
        pass
    return float(default)


def _parse_feature_csv(path: str, min_rows: int = 200):
    """
    Utility for calibrator: load feature log CSV into a DataFrame.
    Keeps only numeric probability/score columns where possible.
    """
    if pd is None:
        return None
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    if min_rows and len(df) < int(min_rows):
        return None
    # Normalize common column names
    for c in ("buy_prob","p_buy"):
        if c in df.columns and "buy_prob" not in df.columns:
            df["buy_prob"] = df[c]
    for c in ("sell_prob","p_sell"):
        if c in df.columns and "sell_prob" not in df.columns:
            df["sell_prob"] = df[c]
    for c in ("neutral_prob","p_neutral"):
        if c in df.columns and "neutral_prob" not in df.columns:
            df["neutral_prob"] = df[c]
    return df


def load_train_log(path: str, max_rows: int = 50000, schema_cols: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    return load_train_log_v3(path, schema_cols=schema_cols, max_rows=max_rows)


@dataclass
class Dataset:
    X: np.ndarray
    y: np.ndarray
    w: np.ndarray
    cols: List[str]


def _get_schema_from_pipeline(pipeline_ref) -> Optional[List[str]]:
    # Prefer explicit schema in pipeline
    try:
        names = getattr(pipeline_ref, "feature_schema_names", None)
        if names and isinstance(names, list) and all(isinstance(x, str) for x in names):
            return list(names)
    except Exception:
        pass
    # If pipeline wraps a model with schema on it
    try:
        nm = getattr(getattr(pipeline_ref, "xgb", None), "feature_schema_names", None)
        if nm and isinstance(nm, list):
            return list(nm)
    except Exception:
        pass
    return None


def _build_matrix(rows: List[Dict[str, Any]], cols: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    X = np.zeros((len(rows), len(cols)), dtype=np.float32)
    w = np.ones((len(rows),), dtype=np.float32)
    for i, r in enumerate(rows):
        feats = r.get("features") or {}
        for j, c in enumerate(cols):
            X[i, j] = _safe_float(feats.get(c, 0.0), 0.0)
        w[i] = _safe_float(r.get("label_weight", 1.0), 1.0)
    return X, w


def _recycle_flat_to_dir(labels: List[str], aux_ret: np.ndarray, eps: float, flip_weight: float, weights: np.ndarray) -> None:
    """
    Convert some FLAT rows into BUY/SELL (low weight) if return magnitude is meaningful.
    """
    if aux_ret.size != len(labels):
        return
    for i, lab in enumerate(labels):
        if lab != "FLAT":
            continue
        r = float(aux_ret[i])
        if abs(r) >= eps:
            labels[i] = "BUY" if r > 0 else "SELL"
            weights[i] = min(weights[i], flip_weight)


def build_directional_dataset(rows: List[Dict[str, Any]], schema_cols: List[str]) -> Dataset:
    # keep only BUY/SELL; optionally recycle FLAT
    labels = [str(r.get("label", "FLAT")) for r in rows]
    cols = list(schema_cols)

    X, w = _build_matrix(rows, cols)

    # optional recycle
    eps = float(os.getenv("FLAT_RECYCLE_EPS", "0.0008"))
    flip_weight = float(os.getenv("FLAT_RECYCLE_WEIGHT", "0.35"))
    aux_vals = []
    for r in rows:
        feats = r.get("features") or {}
        aux_val = feats.get("aux_ret_main")
        if aux_val is None:
            aux_val = (r.get("meta") or {}).get("aux_ret_main")
        aux_vals.append(_safe_float(aux_val, 0.0))
    aux = np.array(aux_vals, dtype=np.float32)
    _recycle_flat_to_dir(labels, aux, eps=eps, flip_weight=flip_weight, weights=w)

    y = np.array([1 if lab == "BUY" else 0 for lab in labels if lab in ("BUY","SELL")], dtype=np.int32)
    keep = np.array([lab in ("BUY","SELL") for lab in labels], dtype=bool)
    return Dataset(X=X[keep], y=y, w=w[keep], cols=cols)


def build_neutral_dataset(rows: List[Dict[str, Any]], schema_cols: List[str]) -> Dataset:
    labels = [str(r.get("label", "FLAT")) for r in rows]
    cols = list(schema_cols)
    X, w = _build_matrix(rows, cols)
    # neutral label: 1 if FLAT else 0
    y = np.array([1 if lab == "FLAT" else 0 for lab in labels], dtype=np.int32)
    return Dataset(X=X, y=y, w=w, cols=cols)


def train_xgb_dir(data: Dataset):
    if xgb is None or data.X.size == 0:
        return None
    # Simple robust defaults; tune via env if desired
    params = {
        "n_estimators": int(os.getenv("XGB_N_ESTIMATORS", "300")),
        "max_depth": int(os.getenv("XGB_MAX_DEPTH", "4")),
        "learning_rate": float(os.getenv("XGB_LR", "0.05")),
        "subsample": float(os.getenv("XGB_SUBSAMPLE", "0.9")),
        "colsample_bytree": float(os.getenv("XGB_COLSAMPLE", "0.9")),
        "reg_lambda": float(os.getenv("XGB_L2", "1.0")),
        "min_child_weight": float(os.getenv("XGB_MIN_CHILD_WEIGHT", "1.0")),
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "n_jobs": int(os.getenv("XGB_N_JOBS", "4")),
        "random_state": 7,
    }
    clf = xgb.XGBClassifier(**params)
    clf.fit(data.X, data.y, sample_weight=data.w)
    # Embed schema into booster
    try:
        meta = json.dumps({"feature_names": list(data.cols)}, separators=(",", ":"))
        clf.get_booster().set_attr(feature_schema=meta)
    except Exception as e:
        logger.warning("[TRAIN] failed embedding feature_schema: %s", e)
    return clf


def train_neutral(data: Dataset):
    if LogisticRegression is None or data.X.size == 0:
        return None
    # weights already included; keep it simple
    try:
        neutral_max_iter = int(os.getenv("NEUTRAL_MAX_ITER", "2000") or "2000")
    except Exception:
        neutral_max_iter = 2000
    solver = os.getenv("NEUTRAL_SOLVER", "lbfgs") or "lbfgs"
    clf = LogisticRegression(max_iter=neutral_max_iter, n_jobs=1, solver=solver)
    try:
        clf.fit(data.X, data.y, sample_weight=data.w)
    except TypeError:
        clf.fit(data.X, data.y)
    return clf


def _write_bundle_and_promote(
    *,
    xgb_model,
    neutral_model,
    schema_cols: List[str],
    notes: Dict[str, Any],
    data_range: Optional[Dict[str, Any]] = None,
    source_counts: Optional[Dict[str, Any]] = None,
    validation_report: Optional[Dict[str, Any]] = None,
    xgb_out_path: str,
    neutral_out_path: str,
) -> Path:
    bundles_dir = Path(os.getenv("MODEL_BUNDLES_DIR", "trained_models/bundles"))
    production_link = Path(os.getenv("MODEL_PRODUCTION_LINK", "trained_models/production"))

    bundle_dir = make_bundle_dir(bundles_dir)
    bp = bundle_paths(bundle_dir)

    # Save artifacts inside bundle
    xgb_model.save_model(str(bp.xgb_path))
    try:
        import joblib
        joblib.dump(neutral_model, str(bp.neutral_path))
    except Exception as e:
        raise RuntimeError(f"failed saving neutral model: {e}")

    # schema cols (atomic writes)
    atomic_write_json(bp.schema_cols_json, {"columns": list(schema_cols)})
    atomic_write_text(bp.schema_cols_txt, "\n".join(schema_cols))

    # Optionally carry forward calibrator from current production, if exists
    calib_env = os.getenv("CALIB_PATH", "").strip()
    if calib_env:
        try:
            csrc = Path(calib_env)
            if csrc.exists():
                bp.calib_path.write_bytes(csrc.read_bytes())
        except Exception:
            pass

    # Validate feature counts match schema
    ok, reason = validate_bundle_feature_counts(bp.xgb_path, schema_cols, neutral_path=bp.neutral_path)
    if not ok:
        raise RuntimeError(f"bundle validation failed: {reason}")

    # Manifest
    write_manifest(
        bp.manifest_path,
        schema_cols=schema_cols,
        notes=notes,
        data_range=data_range,
        source_counts=source_counts,
        validation_report=validation_report,
    )

    # Promote symlink atomically
    promote_bundle_symlink(bundle_dir, production_link)

    # Also (optional) update legacy out paths to point at production link
    # This keeps older tooling happy.
    try:
        # xgb_out_path likely points into production; ensure file exists there
        Path(xgb_out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(xgb_out_path).write_bytes(bp.xgb_path.read_bytes())
        Path(neutral_out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(neutral_out_path).write_bytes(bp.neutral_path.read_bytes())
    except Exception:
        pass

    return bundle_dir


def _train_and_promote_sync(
    feature_log_path: str,
    xgb_out_path: str,
    neutral_out_path: str,
    min_rows: int,
    max_rows: int,
    pipeline_ref=None,
) -> Dict[str, Any]:
    schema = _get_schema_from_pipeline(pipeline_ref)
    if not schema:
        schema_path = Path(os.getenv("FEATURE_SCHEMA_COLS_PATH", "")).expanduser()
        if schema_path and schema_path.exists():
            try:
                schema = json.loads(schema_path.read_text(encoding="utf-8")).get("columns")
            except Exception:
                schema = None

    if not schema:
        raise SchemaResolutionError("FEATURE_SCHEMA_COLS_PATH missing or pipeline schema unavailable")

    rows = load_train_log(feature_log_path, max_rows=max_rows, schema_cols=list(schema))
    if len(rows) < int(min_rows):
        return {"status": "wait", "rows": len(rows)}

    ok, report = validate_pretrain(rows, schema_cols=list(schema))
    if not ok:
        return {"status": "error", "reason": "pretrain_validation_failed", "report": report}

    dir_data = build_directional_dataset(rows, schema_cols=schema)
    neu_data = build_neutral_dataset(rows, schema_cols=schema)

    xgb_model = train_xgb_dir(dir_data)
    neu_model = train_neutral(neu_data)

    if xgb_model is None or neu_model is None:
        return {
            "status": "skip",
            "rows": len(rows),
            "dir_rows": int(dir_data.y.size),
            "neu_rows": int(neu_data.y.size),
            "cols": len(schema),
        }

    eval_ok, eval_report = evaluate_holdout(rows, list(schema), xgb_model)
    if not eval_ok:
        return {"status": "error", "reason": "eval_gate_failed", "report": eval_report}

    notes = {
        "trainer": "online_trainer_regen_v3",
        "rows": len(rows),
        "dir_rows": int(dir_data.y.size),
        "neu_rows": int(neu_data.y.size),
        "min_rows": int(min_rows),
        "pretrain_report": report,
        "eval_report": eval_report,
    }
    src_counts = summarize_sources(rows)
    drange = data_range(rows)

    bundle_dir = _write_bundle_and_promote(
        xgb_model=xgb_model,
        neutral_model=neu_model,
        schema_cols=list(schema),
        notes=notes,
        data_range=drange,
        source_counts=src_counts,
        validation_report={"pretrain": report, "eval": eval_report},
        xgb_out_path=xgb_out_path,
        neutral_out_path=neutral_out_path,
    )

    return {
        "status": "ok",
        "bundle_dir": bundle_dir,
        "xgb_model": xgb_model,
        "neu_model": neu_model,
        "rows": len(rows),
        "dir_rows": int(dir_data.y.size),
        "neu_rows": int(neu_data.y.size),
        "cols": len(schema),
    }


async def background_trainer_loop(
    feature_log_path: str,
    xgb_out_path: str,
    neutral_out_path: str,
    min_rows: int = 500,
    interval_sec: float = 30.0,
    pipeline_ref=None,
):
    """
    Watch training log; train & promote bundles when new data arrives.
    """
    last_mtime = None

    while True:
        try:
            p = Path(feature_log_path)
            if p.exists():
                mtime = None
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
                    xgb_out_path,
                    neutral_out_path,
                    int(min_rows),
                    int(max_rows),
                    pipeline_ref,
                )

                status = result.get("status")
                if status == "wait":
                    logger.info("[TRAIN] waiting: rows=%d (<%d)", int(result.get("rows", 0)), int(min_rows))
                    last_mtime = mtime
                    await asyncio.sleep(interval_sec)
                    continue

                if status == "skip":
                    logger.warning("[TRAIN] training skipped (models None). xgb=%s neu=%s", False, False)
                    last_mtime = mtime
                    await asyncio.sleep(interval_sec)
                    continue

                if status == "error":
                    logger.error("[TRAIN] error: %s", str(result.get("reason", "unknown")))
                    last_mtime = mtime
                    await asyncio.sleep(interval_sec)
                    continue

                if status != "ok":
                    last_mtime = mtime
                    await asyncio.sleep(interval_sec)
                    continue

                logger.info(
                    "[TRAIN] rows=%d dir=%d neu=%d cols=%d",
                    int(result.get("rows", 0)),
                    int(result.get("dir_rows", 0)),
                    int(result.get("neu_rows", 0)),
                    int(result.get("cols", 0)),
                )
                logger.info("[TRAIN] promoted bundle → %s", str(result.get("bundle_dir")))

                # hot-swap into live pipeline if possible
                try:
                    if pipeline_ref is not None and hasattr(pipeline_ref, "replace_models"):
                        pipeline_ref.replace_models(xgb=result.get("xgb_model"), neutral=result.get("neu_model"))
                        logger.info("[TRAIN] pipeline_ref models replaced")
                except Exception as e:
                    logger.warning("[TRAIN] pipeline hot-swap failed: %s", e)

                last_mtime = mtime

        except SchemaResolutionError as e:
            logger.error("[TRAIN] schema resolution failed: %s", e)
        except Exception as e:
            logger.warning("[TRAIN] loop error: %s", e)

        await asyncio.sleep(interval_sec)
