import asyncio
import json
import logging
import os
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)

def _atomic_write_json(path: str, obj: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, p)


async def background_calibrator_loop(
    feature_log_path: str,
    calib_out_path: str,
    pipeline_ref,
    interval_sec: int = 1200,
    min_dir_rows: int = 500
):
    """
    Periodically fit Platt calibration on recent logs and write a,b to calib_out_path.
    STRICT: trains only on meta_p_xgb_raw; skips cycle if missing.
    Validates monotonicity and Brier score improvement before writing.
    Auto hot-reloads into pipeline via pipeline_ref.reload_calibration().
    """
    logger.info(f"[CALIB] Calibrator started (every {interval_sec}s) → {calib_out_path}")
    last_mtime = None
    
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-z))
    
    def _logit_clip_arr(p: np.ndarray) -> np.ndarray:
        p = np.clip(p, 1e-9, 1 - 1e-9)
        return np.log(p / (1.0 - p))
    
    def _brier(y_true: np.ndarray, p_hat: np.ndarray) -> float:
        p_hat = np.clip(p_hat, 1e-9, 1 - 1e-9)
        return float(np.mean((p_hat - y_true) ** 2))
    
    def _monotonic_grid_ok(a: float, b: float) -> bool:
        # a must be positive and grid mapping must be strictly increasing
        if not np.isfinite(a) or not np.isfinite(b) or a <= 0.0:
            return False
        grid = np.linspace(0.05, 0.95, 19)
        lg = _logit_clip_arr(grid)
        q = _sigmoid(a * lg + b)
        return bool(np.all(np.diff(q) > 0.0))
    
    while True:
        try:
            await asyncio.sleep(interval_sec)
            
            if not os.path.exists(feature_log_path):
                logger.info(f"[CALIB] feature_log not found: {feature_log_path}")
                continue
            
            mtime = os.path.getmtime(feature_log_path)
            if last_mtime is not None and mtime <= last_mtime:
                logger.debug("[CALIB] Feature log unchanged; skipping")
                continue
            last_mtime = mtime
            
            try:
                from online_trainer import _parse_feature_csv
                df = _parse_feature_csv(feature_log_path, min_rows=min_dir_rows)
            except Exception as e:
                logger.warning(f"[CALIB] parse failed: {e}")
                df = None
            
            if df is None or df.empty:
                logger.info(f"[CALIB] Not enough rows yet for calibration (min_dir_rows={min_dir_rows})")
                continue
            
            # directional, tradeable only
            m = (df["label"].isin(["BUY", "SELL"])) & (df["tradeable"] == True)
            dfd = df[m].copy()
            if dfd.empty or len(dfd) < min_dir_rows:
                logger.info(f"[CALIB] directional/tradeable rows insufficient: {len(dfd)}/{min_dir_rows}")
                continue
            
            # STRICT: use only meta_p_xgb_raw; do not fallback to buy_prob
            if "meta_p_xgb_raw" not in dfd.columns:
                logger.info("[CALIB] meta_p_xgb_raw not found in logs; skipping this cycle to avoid corrupt calibration")
                continue
            
            p = np.asarray(dfd["meta_p_xgb_raw"].values, dtype=float)
            y = np.asarray((dfd["label"] == "BUY").astype(int).values, dtype=int)
            x = _logit_clip_arr(p).reshape(-1, 1)
            
            # Fit logistic on logit(p) -> y
            a = b = None
            try:
                from sklearn.linear_model import LogisticRegression
                clf = LogisticRegression(max_iter=200, solver="lbfgs")
                clf.fit(x, y)
                a = float(clf.coef_.ravel()[0])
                b = float(clf.intercept_.ravel()[0])
            except Exception as e:
                logger.warning(f"[CALIB] sklearn fit failed: {e}")
                continue
            
            if not np.isfinite(a) or not np.isfinite(b):
                logger.info("[CALIB] invalid coefficients; skipping write")
                continue
            
            # Holdout validation: last 200 rows (or 20% if smaller)
            n = len(dfd)
            n_val = max(50, min(200, n // 5))
            y_val = y[-n_val:]
            p_raw_val = np.clip(p[-n_val:], 1e-9, 1 - 1e-9)
            z_val = a * _logit_clip_arr(p_raw_val) + b
            p_cal_val = _sigmoid(z_val)
            brier_raw = _brier(y_val, p_raw_val)
            brier_cal = _brier(y_val, p_cal_val)
            
            # Sanity checks: monotonicity and no major degradation
            if not _monotonic_grid_ok(a, b):
                logger.info(f"[CALIB] rejected (non-monotonic or non-positive slope): a={a:.6f}, b={b:.6f}")
                continue
            if brier_cal > (brier_raw + 0.002):
                logger.info(f"[CALIB] rejected (Brier worsened): raw={brier_raw:.5f} calib={brier_cal:.5f}")
                continue
            
            meta = {"a": a, "b": b, "n": int(n), "source": "meta_p_xgb_raw"}
            _atomic_write_json(calib_out_path, meta)
            logger.info(f"[CALIB] wrote: a={a:.6f} b={b:.6f} (n={n}, src=meta_p_xgb_raw) → {calib_out_path}")
            
            # Live hot-reload into pipeline
            try:
                ok = bool(pipeline_ref.reload_calibration(calib_out_path))
                logger.info(f"[CALIB] hot-reload {'ok' if ok else 'skipped'}")
            except Exception as e:
                logger.warning(f"[CALIB] hot-reload failed: {e}")
        
        except asyncio.CancelledError:
            logger.info("[CALIB] Calibrator cancelled")
            break
        except Exception as e:
            logger.error(f"[CALIB] loop error: {e}", exc_info=True)


