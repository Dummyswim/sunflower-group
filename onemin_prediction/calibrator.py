#!/usr/bin/env python3
"""
Platt calibrator for policy success probabilities (BUY/SELL).

Fits sigmoid(a * logit(p_raw) + b) per direction on SignalContext logs.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

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


def _read_jsonl_tail(path: str, *, max_bytes: int, max_rows: int) -> List[Dict[str, Any]]:
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

    if max_rows and len(rows) > max_rows:
        rows = rows[-max_rows:]

    return rows


def _extract_model_p_raw(val: Any) -> Optional[float]:
    if isinstance(val, dict):
        p = val.get("p_success_raw")
        if p is None:
            p = val.get("policy_success_raw")
        if p is None:
            return None
        try:
            p = float(p)
        except Exception:
            return None
        return p if np.isfinite(p) else None
    if isinstance(val, str):
        try:
            obj = json.loads(val)
        except Exception:
            return None
        return _extract_model_p_raw(obj)
    return None


def _extract_arrays(
    rows: List[Dict[str, Any]],
    *,
    direction: str,
    require_tradeable: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    p_vals: List[float] = []
    y_vals: List[int] = []
    dir_norm = str(direction).upper()

    for r in rows:
        if str(r.get("teacher_dir", "")).upper() != dir_norm:
            continue
        if require_tradeable and not bool(r.get("teacher_tradeable")):
            continue
        label = str(r.get("label", "")).upper()
        if label not in ("BUY", "SELL", "FLAT"):
            continue
        p_raw = _extract_model_p_raw(r.get("model"))
        if p_raw is None:
            continue
        p_vals.append(float(p_raw))
        y_vals.append(1 if label == dir_norm else 0)

    if not p_vals:
        return np.array([]), np.array([])

    return np.asarray(p_vals, dtype=float), np.asarray(y_vals, dtype=int)


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def _logit_clip_arr(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return np.log(p / (1.0 - p))


def _brier(y_true: np.ndarray, p_hat: np.ndarray) -> float:
    p_hat = np.clip(p_hat, 1e-9, 1 - 1e-9)
    return float(np.mean((p_hat - y_true) ** 2))


def _monotonic_grid_ok(a: float, b: float) -> bool:
    if not np.isfinite(a) or not np.isfinite(b) or a <= 0.0:
        return False
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
    brier_before_eval: Optional[float]
    brier_after_eval: Optional[float]
    n_eval: int
    base_rate: float
    inverted: bool
    direction: str


def _fit_platt(
    p_raw: np.ndarray,
    y: np.ndarray,
    direction: str,
    *,
    eval_p: Optional[np.ndarray] = None,
    eval_y: Optional[np.ndarray] = None,
) -> Optional[CalibFit]:
    if LogisticRegression is None:
        logger.warning("[CALIB] sklearn not available → cannot fit calibration")
        return None

    if p_raw.size == 0 or y.size == 0:
        logger.debug("[CALIB] fit skipped: empty arrays")
        return None

    m = np.isfinite(p_raw)
    p_raw = p_raw[m]
    y = y[m]
    n = int(len(y))
    if n < 10:
        logger.debug("[CALIB] fit skipped: too few samples n=%d", n)
        return None

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

    if not (auc > 0.5):
        logger.debug("[CALIB] fit skipped: auc<=0.5 (dir=%s)", direction)
        return None

    x = _logit_clip_arr(p_raw).reshape(-1, 1)

    try:
        clf = LogisticRegression(max_iter=200)
        clf.fit(x, y)
        a = float(clf.coef_[0][0])
        b = float(clf.intercept_[0])
    except Exception as e:
        logger.warning("[CALIB] fit failed: %s", e)
        return None

    if not _monotonic_grid_ok(a, b):
        logger.debug("[CALIB] fit skipped: non-monotonic map a=%.4f b=%.4f", a, b)
        return None

    p_fit = _sigmoid(a * x + b)
    base = float(np.mean(y)) if y.size else 0.0

    brier_before = _brier(y, p_raw)
    brier_after = _brier(y, p_fit)
    brier_before_eval = None
    brier_after_eval = None
    n_eval = 0

    if eval_p is not None and eval_y is not None and len(eval_y) > 0:
        eval_p = np.asarray(eval_p, dtype=float)
        eval_y = np.asarray(eval_y, dtype=int)
        eval_x = _logit_clip_arr(eval_p)
        eval_fit = _sigmoid(a * eval_x + b)
        brier_before_eval = _brier(eval_y, eval_p)
        brier_after_eval = _brier(eval_y, eval_fit)
        n_eval = int(len(eval_y))

    return CalibFit(
        a=a,
        b=b,
        n=n,
        auc=float(auc),
        brier_before=brier_before,
        brier_after=brier_after,
        brier_before_eval=brier_before_eval,
        brier_after_eval=brier_after_eval,
        n_eval=n_eval,
        base_rate=base,
        inverted=inverted,
        direction=str(direction).upper(),
    )


def _write_calib(path: str, fit: CalibFit, source: str) -> bool:
    tol = float(os.getenv("CALIB_IMPROVE_TOL", "0.0") or "0.0")
    before = fit.brier_before_eval if fit.brier_before_eval is not None else fit.brier_before
    after = fit.brier_after_eval if fit.brier_after_eval is not None else fit.brier_after
    if after > (before + tol):
        logger.warning(
            "[CALIB] skip write for %s: brier_after %.6f > brier_before %.6f",
            fit.direction,
            after,
            before,
        )
        return False
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    payload = {
        "a": fit.a,
        "b": fit.b,
        "n": fit.n,
        "auc": fit.auc,
        "brier_before": fit.brier_before,
        "brier_after": fit.brier_after,
        "brier_before_eval": fit.brier_before_eval,
        "brier_after_eval": fit.brier_after_eval,
        "n_eval": fit.n_eval,
        "base_rate": fit.base_rate,
        "inverted": fit.inverted,
        "direction": fit.direction,
        "source": source,
        "generated_at": now,
    }
    _atomic_write_json(path, payload)
    return True


def _fit_direction(
    rows: List[Dict[str, Any]],
    *,
    direction: str,
    min_rows: int,
    source: str,
    holdout_frac: float,
    min_holdout: int,
) -> Optional[CalibFit]:
    p_raw, y = _extract_arrays(rows, direction=direction, require_tradeable=True)
    if len(y) < min_rows:
        logger.debug("[CALIB] %s skipped: n=%d < min_rows=%d", direction, len(y), min_rows)
        return None
    eval_p = None
    eval_y = None
    if holdout_frac > 0.0:
        holdout_n = int(len(y) * holdout_frac)
        if holdout_n >= min_holdout and (len(y) - holdout_n) >= min_rows:
            eval_p = p_raw[-holdout_n:]
            eval_y = y[-holdout_n:]
            p_raw = p_raw[:-holdout_n]
            y = y[:-holdout_n]
    return _fit_platt(p_raw, y, direction, eval_p=eval_p, eval_y=eval_y)


async def background_calibrator_loop(
    feature_log_path: str,
    calib_buy_out_path: str,
    calib_sell_out_path: str,
    *,
    interval_sec: float = 1200.0,
    min_dir_rows: int = 200,
    max_bytes: int = 6_000_000,
    max_rows: int = 50000,
):
    last_mtime = None
    holdout_frac = float(os.getenv("CALIB_HOLDOUT_FRAC", "0.2") or "0.2")
    min_holdout = int(os.getenv("CALIB_MIN_HOLDOUT", "50") or "50")

    while True:
        try:
            p = Path(feature_log_path)
            if not p.exists():
                await asyncio.sleep(interval_sec)
                continue

            try:
                mtime = p.stat().st_mtime
            except Exception:
                mtime = None

            if mtime and last_mtime and mtime == last_mtime:
                await asyncio.sleep(interval_sec)
                continue

            rows = _read_jsonl_tail(str(p), max_bytes=max_bytes, max_rows=max_rows)
            if not rows:
                await asyncio.sleep(interval_sec)
                continue

            fit_buy = _fit_direction(
                rows,
                direction="BUY",
                min_rows=min_dir_rows,
                source=str(p),
                holdout_frac=holdout_frac,
                min_holdout=min_holdout,
            )
            fit_sell = _fit_direction(
                rows,
                direction="SELL",
                min_rows=min_dir_rows,
                source=str(p),
                holdout_frac=holdout_frac,
                min_holdout=min_holdout,
            )

            if fit_buy:
                wrote = _write_calib(calib_buy_out_path, fit_buy, source=str(p))
                if wrote:
                    logger.info("[CALIB] BUY updated n=%d auc=%.3f", fit_buy.n, fit_buy.auc)
                else:
                    logger.info("[CALIB] BUY skipped (brier_after worse)")
            else:
                logger.info("[CALIB] BUY skipped (insufficient rows/skill)")

            if fit_sell:
                wrote = _write_calib(calib_sell_out_path, fit_sell, source=str(p))
                if wrote:
                    logger.info("[CALIB] SELL updated n=%d auc=%.3f", fit_sell.n, fit_sell.auc)
                else:
                    logger.info("[CALIB] SELL skipped (brier_after worse)")
            else:
                logger.info("[CALIB] SELL skipped (insufficient rows/skill)")

            last_mtime = mtime
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error("[CALIB] loop error: %s", e, exc_info=True)
        await asyncio.sleep(interval_sec)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default=os.getenv("TRAIN_LOG_PATH", ""))
    ap.add_argument("--buy", default=os.getenv("CALIB_BUY_PATH", "calib_buy.json"))
    ap.add_argument("--sell", default=os.getenv("CALIB_SELL_PATH", "calib_sell.json"))
    ap.add_argument("--min_rows", type=int, default=int(os.getenv("CALIB_MIN_ROWS", "200") or "200"))
    ap.add_argument("--holdout_frac", type=float, default=float(os.getenv("CALIB_HOLDOUT_FRAC", "0.2") or "0.2"))
    ap.add_argument("--min_holdout", type=int, default=int(os.getenv("CALIB_MIN_HOLDOUT", "50") or "50"))
    args = ap.parse_args()

    if not args.log:
        raise SystemExit("TRAIN_LOG_PATH missing")

    rows = _read_jsonl_tail(args.log, max_bytes=6_000_000, max_rows=50000)
    if not rows:
        raise SystemExit("no rows found")

    fit_buy = _fit_direction(
        rows,
        direction="BUY",
        min_rows=args.min_rows,
        source=args.log,
        holdout_frac=float(args.holdout_frac),
        min_holdout=int(args.min_holdout),
    )
    fit_sell = _fit_direction(
        rows,
        direction="SELL",
        min_rows=args.min_rows,
        source=args.log,
        holdout_frac=float(args.holdout_frac),
        min_holdout=int(args.min_holdout),
    )

    if fit_buy:
        wrote = _write_calib(args.buy, fit_buy, source=args.log)
        if wrote:
            print(f"[CALIB] BUY written -> {args.buy}")
        else:
            print("[CALIB] BUY skipped (brier_after worse)")
    else:
        print("[CALIB] BUY skipped")

    if fit_sell:
        wrote = _write_calib(args.sell, fit_sell, source=args.log)
        if wrote:
            print(f"[CALIB] SELL written -> {args.sell}")
        else:
            print("[CALIB] SELL skipped (brier_after worse)")
    else:
        print("[CALIB] SELL skipped")
