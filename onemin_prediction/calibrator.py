#!/usr/bin/env python3
"""
Platt calibrator for XGB BUY probability.

Fixes addressed:
1) Persist and apply "inverted" orientation:
   - When raw p has AUC <= 0.5, we try (1 - p).
   - If inverted wins, we *record* inverted=True in calibrator.json.
2) Provide a "true Platt" map for inference (a,b on logit(p)), plus diagnostics:
   - auc, base_rate_buy, brier_before/after, generated_at, n, source, inverted
3) Avoid n=0 purgatory:
   - Robust JSONL tail reader (skips partial/bad lines; reads last N MB).
4) Keep the main loop responsive:
   - This file only fits lightweight LogisticRegression on logit(p).

Entry point used by main_event_loop_regen.py:
    from calibrator import background_calibrator_loop
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple, List

import numpy as np
import pandas as pd

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
except Exception:  # pragma: no cover
    LogisticRegression = None  # type: ignore
    roc_auc_score = None  # type: ignore

logger = logging.getLogger(__name__)


def _atomic_write_json(path: str, obj: Dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
    os.replace(tmp, path)


def _extract_meta_p_xgb_raw(val: Any) -> Optional[float]:
    """Pull meta_p_xgb_raw from nested features dict or JSON string."""
    if isinstance(val, dict):
        return val.get("meta_p_xgb_raw")
    if isinstance(val, str):
        try:
            obj = json.loads(val)
        except Exception:
            return None
        if isinstance(obj, dict):
            return obj.get("meta_p_xgb_raw")
    return None


def _validate_calibrator_json(path: str) -> bool:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        logger.warning(f"[CALIB] validate failed to read {path}: {e}")
        return False

    try:
        a = float(data.get("a"))
        b = float(data.get("b"))
        n = int(data.get("n", -1))
    except Exception:
        logger.warning(f"[CALIB] validate failed: missing a/b/n in {path}")
        return False

    if not np.isfinite(a) or not np.isfinite(b) or n < 0:
        logger.warning(f"[CALIB] validate failed: invalid a/b/n in {path}")
        return False

    try:
        auc = float(data.get("auc"))
    except Exception:
        logger.warning(f"[CALIB] validate warning: auc missing in {path}")
        return False

    if not (0.5 <= auc <= 1.0):
        logger.warning(f"[CALIB] validate failed: auc out of range ({auc:.4f}) in {path}")
        return False

    return True


def _read_jsonl_tail(path: str, *, max_bytes: int, max_rows: int) -> pd.DataFrame:
    """
    Read last ~max_bytes from a JSONL file and parse into a DataFrame.
    Skips partial/invalid lines safely.
    """
    try:
        size = os.path.getsize(path)
    except Exception:
        size = 0

    start = 0
    if size > max_bytes:
        start = max(0, size - max_bytes)

    rows: List[Dict[str, Any]] = []
    with open(path, "rb") as f:
        if start:
            f.seek(start)
            # Drop partial first line
            _ = f.readline()
        data = f.read()

    for line in data.splitlines():
        try:
            s = line.decode("utf-8", errors="ignore").strip()
            if not s or not s.startswith("{"):
                continue
            rows.append(json.loads(s))
        except Exception:
            continue

    if not rows:
        return pd.DataFrame()

    if len(rows) > max_rows:
        rows = rows[-max_rows:]

    return pd.DataFrame(rows)


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def _logit_clip_arr(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return np.log(p / (1.0 - p))


def _brier(y_true: np.ndarray, p_hat: np.ndarray) -> float:
    p_hat = np.clip(p_hat, 1e-9, 1 - 1e-9)
    return float(np.mean((p_hat - y_true) ** 2))


def _monotonic_grid_ok(a: float, b: float) -> bool:
    # With logit input, monotonic calibration requires a > 0
    if not np.isfinite(a) or not np.isfinite(b) or a <= 0.0:
        return False
    # quick numeric sanity over a coarse grid
    grid = np.linspace(0.02, 0.98, 25)
    x = _logit_clip_arr(grid)
    z = a * x + b
    q = _sigmoid(z)
    return bool(np.all(np.diff(q) > -1e-6))


@dataclass
class CalibFit:
    a: float
    b: float
    n: int
    auc: float
    brier_before: float
    brier_after: float
    base_rate_buy: float
    inverted: bool
    source: str


def _fit_platt(dfd: pd.DataFrame) -> Optional[CalibFit]:
    if LogisticRegression is None:
        logger.warning("[CALIB] sklearn not available → cannot fit calibration")
        return None

    if dfd is None or dfd.empty:
        logger.debug("[CALIB] fit skipped: empty frame")
        return None

    # Required columns
    if "meta_p_xgb_raw" not in dfd.columns or "label" not in dfd.columns:
        logger.debug("[CALIB] fit skipped: missing required columns")
        return None

    # Build arrays
    p_raw = np.asarray(dfd["meta_p_xgb_raw"].values, dtype=float)
    y = np.asarray((dfd["label"].astype(str) == "BUY").astype(int).values, dtype=int)

    # Drop NaNs/infs
    m = np.isfinite(p_raw)
    p_raw = p_raw[m]
    y = y[m]
    n = int(len(y))
    if n < 10:
        logger.debug("[CALIB] fit skipped: too few samples n=%d", n)
        return None

    # Skill gate: require AUC > 0.5; allow inversion (1-p) if it helps
    auc = 0.5
    auc_inv = 0.5
    inverted = False
    try:
        if len(np.unique(y)) == 2 and roc_auc_score is not None:
            auc = float(roc_auc_score(y, p_raw))
    except Exception:
        auc = 0.5

    if not (auc > 0.5):
        try:
            if len(np.unique(y)) == 2 and roc_auc_score is not None:
                auc_inv = float(roc_auc_score(y, 1.0 - p_raw))
        except Exception:
            auc_inv = 0.5

        if auc_inv > 0.5:
            p_raw = 1.0 - p_raw
            auc = auc_inv
            inverted = True
        else:
            logger.debug("[CALIB] fit skipped: auc<=0.5 and inverted also <=0.5")
            return None

    # Fit Platt on logit(p)
    x = _logit_clip_arr(np.clip(p_raw, 1e-9, 1 - 1e-9)).reshape(-1, 1)
    clf = LogisticRegression(
        max_iter=int(os.getenv("CALIB_LOGIT_MAX_ITER", "1000") or "1000"),
        solver="lbfgs",
        class_weight="balanced",
    )
    clf.fit(x, y)
    a = float(clf.coef_.ravel()[0])
    b = float(clf.intercept_.ravel()[0])

    if not _monotonic_grid_ok(a, b):
        logger.debug("[CALIB] fit skipped: monotonic check failed a=%.6f b=%.6f", a, b)
        return None

    # Holdout tail for Brier comparison
    n_val = max(50, min(200, n // 5))
    y_val = y[-n_val:]
    p_val = np.clip(p_raw[-n_val:], 1e-9, 1 - 1e-9)
    z_val = a * _logit_clip_arr(p_val) + b
    p_cal_val = _sigmoid(z_val)

    brier_before = _brier(y_val, p_val)
    brier_after = _brier(y_val, p_cal_val)

    base_rate_buy = float(np.mean(y)) if n else 0.5

    return CalibFit(
        a=a,
        b=b,
        n=n,
        auc=float(auc),
        brier_before=float(brier_before),
        brier_after=float(brier_after),
        base_rate_buy=float(base_rate_buy),
        inverted=bool(inverted),
        source="",
    )


async def background_calibrator_loop(
    feature_log_path: str,
    calib_out_path: str,
    pipeline_ref,
    interval_sec: int = 1200,
    min_dir_rows: int = 120,
) -> None:
    """
    Periodically fit Platt calibration on recent train logs and write calibrator.json.

    NOTE: Despite the arg name, main_event_loop passes train_log_path here (JSONL).
    """
    logger.info(f"[CALIB] Calibrator started (every {interval_sec}s) → {calib_out_path}")

    # Tunables
    try:
        recent_rows = int(os.getenv("CALIB_RECENT_ROWS", "1200") or "1200")
    except Exception:
        recent_rows = 1200

    try:
        max_bytes = int(os.getenv("CALIB_TAIL_MAX_BYTES", str(10 * 1024 * 1024)) or str(10 * 1024 * 1024))
    except Exception:
        max_bytes = 10 * 1024 * 1024

    try:
        max_rows = int(os.getenv("CALIB_TAIL_MAX_ROWS", "60000") or "60000")
    except Exception:
        max_rows = 60000

    try:
        min_auc = float(os.getenv("CALIB_MIN_AUC", "0.53") or "0.53")
    except Exception:
        min_auc = 0.53

    try:
        brier_tol = float(os.getenv("CALIB_BRIER_TOL", "0.002") or "0.002")
    except Exception:
        brier_tol = 0.002

    # Repeat forever
    while True:
        try:
            await asyncio.sleep(max(10, int(interval_sec)))

            if not feature_log_path or not os.path.exists(feature_log_path):
                logger.info(f"[CALIB] train log missing: {feature_log_path}")
                continue

            df = _read_jsonl_tail(feature_log_path, max_bytes=max_bytes, max_rows=max_rows)
            if df.empty:
                logger.info("[CALIB] no rows parsed from train log tail")
                continue
            if "record_version" in df.columns:
                df = df[df["record_version"] == "v3"].copy()
                if df.empty:
                    logger.info("[CALIB] no v3 rows in train log tail")
                    continue

            # Normalize columns that might exist with different names.
            # Prefer features.meta_p_xgb_raw, then explicit top-level columns, then buy_prob.
            if "meta_p_xgb_raw" not in df.columns:
                if "features" in df.columns:
                    try:
                        df["meta_p_xgb_raw"] = df["features"].apply(_extract_meta_p_xgb_raw)
                    except Exception:
                        pass
            if "meta_p_xgb_raw" not in df.columns:
                if "meta" in df.columns:
                    try:
                        df["meta_p_xgb_raw"] = df["meta"].apply(_extract_meta_p_xgb_raw)
                    except Exception:
                        pass
            if "meta_p_xgb_raw" not in df.columns:
                for alt in ("p_model_raw", "p_xgb_raw", "p_raw"):
                    if alt in df.columns:
                        df["meta_p_xgb_raw"] = df[alt]
                        break
            if "meta_p_xgb_raw" not in df.columns and "buy_prob" in df.columns:
                df["meta_p_xgb_raw"] = df["buy_prob"]
            elif "meta_p_xgb_raw" in df.columns and "buy_prob" in df.columns:
                # fill missing raw values from buy_prob as a last resort
                try:
                    df["meta_p_xgb_raw"] = df["meta_p_xgb_raw"].where(
                        pd.notna(df["meta_p_xgb_raw"]),
                        df["buy_prob"],
                    )
                except Exception:
                    pass

            if "label" not in df.columns and "dir_label" in df.columns:
                df["label"] = df["dir_label"]

            if "label" not in df.columns or "meta_p_xgb_raw" not in df.columns:
                logger.info("[CALIB] missing required columns in tail → skip")
                continue

            # Filter to directionals
            dfd = df[df["label"].isin(["BUY", "SELL"])].copy()
            dfd = dfd.replace([np.inf, -np.inf], np.nan).dropna(subset=["meta_p_xgb_raw"])
            if len(dfd) < min_dir_rows:
                logger.info(f"[CALIB] insufficient dir rows n={len(dfd)} (<{min_dir_rows}) → skip")
                continue

            # Candidate slices (order matters)
            candidates: List[Tuple[str, pd.DataFrame]] = []
            # Prefer tradeable-only if present
            if "suggest_tradeable" in dfd.columns:
                tr = dfd[dfd["suggest_tradeable"].astype(bool)].copy()
                if len(tr) >= min_dir_rows:
                    candidates.append(("recent_tr", tr.tail(recent_rows)))
                    candidates.append(("full_tr", tr))
            elif "tradeable" in dfd.columns:
                tr = dfd[dfd["tradeable"].astype(bool)].copy()
                if len(tr) >= min_dir_rows:
                    candidates.append(("recent_tr", tr.tail(recent_rows)))
                    candidates.append(("full_tr", tr))

            candidates.append(("recent_all", dfd.tail(recent_rows)))
            candidates.append(("full_all", dfd))

            accepted: Optional[CalibFit] = None
            for name, dd in candidates:
                try:
                    fit = _fit_platt(dd)
                    if fit is None:
                        logger.info(f"[CALIB] candidate={name} skipped (no skill / invalid)")
                        continue

                    # Apply gates
                    if float(fit.auc) < float(min_auc):
                        logger.debug(
                            "[CALIB] reject candidate=%s: auc=%.4f < min_auc=%.4f",
                            name,
                            float(fit.auc),
                            float(min_auc),
                        )
                        logger.info(f"[CALIB] candidate={name} auc={fit.auc:.3f} < {min_auc:.3f} → skip")
                        continue
                    if float(fit.brier_after) > float(fit.brier_before) + float(brier_tol):
                        logger.debug(
                            "[CALIB] reject candidate=%s: brier_before=%.6f brier_after=%.6f tol=%.6f",
                            name,
                            float(fit.brier_before),
                            float(fit.brier_after),
                            float(brier_tol),
                        )
                        logger.info(
                            f"[CALIB] candidate={name} brier {fit.brier_before:.6f}->{fit.brier_after:.6f} worsened → skip"
                        )
                        continue

                    fit.source = name
                    accepted = fit
                    break
                except Exception as e:
                    logger.debug(f"[CALIB] candidate {name} failed: {e}")

            if accepted is None:
                logger.info("[CALIB] rejected all candidates")
                continue

            out = {
                "a": float(accepted.a),
                "b": float(accepted.b),
                "n": int(accepted.n),
                "auc": float(accepted.auc),
                "base_rate_buy": float(accepted.base_rate_buy),
                "brier_before": float(accepted.brier_before),
                "brier_after": float(accepted.brier_after),
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "inverted": bool(accepted.inverted),
                "source": str(accepted.source),
            }
            _atomic_write_json(calib_out_path, out)
            _validate_calibrator_json(calib_out_path)

            logger.info(
                "[CALIB] wrote calibration (%s) n=%d a=%.4f b=%.4f brier %.6f->%.6f inverted=%d",
                accepted.source,
                accepted.n,
                accepted.a,
                accepted.b,
                accepted.brier_before,
                accepted.brier_after,
                1 if accepted.inverted else 0,
            )

            # Hot reload into pipeline
            try:
                ok = bool(pipeline_ref.reload_calibration(calib_out_path))
                logger.info(f"[CALIB] hot-reload {'ok' if ok else 'skipped'}")
            except Exception as e:
                logger.warning(f"[CALIB] hot-reload failed: {e}")

        except asyncio.CancelledError:
            logger.info("[CALIB] Calibrator loop cancelled")
            raise
        except Exception as e:
            logger.warning(f"[CALIB] loop error: {e}")
