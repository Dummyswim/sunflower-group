# calibrator.py
import asyncio
import json
import logging
import os
import numpy as np
from pathlib import Path
from datetime import datetime
from logging_setup import log_every
import pandas as pd
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
                log_every("calib-missing-log", 60, logger.info, f"[CALIB] feature_log not found: {feature_log_path}")
                continue


            mtime_daily = os.path.getmtime(feature_log_path)
            hist_path = os.getenv("FEATURE_LOG_HIST", "feature_log_hist.csv")
            mtime_hist = os.path.getmtime(hist_path) if os.path.exists(hist_path) else 0.0
            mtime_combined = max(mtime_daily, mtime_hist)
            if last_mtime is not None and mtime_combined <= last_mtime:
                log_every("calib-unchanged", 30, logger.debug, "[CALIB] Feature logs unchanged; skipping")
                continue
            last_mtime = mtime_combined



            hist_path = os.getenv("FEATURE_LOG_HIST", "feature_log_hist.csv")
            try:
                from online_trainer import _parse_feature_csv
                df_daily = _parse_feature_csv(feature_log_path, min_rows=0)
                df_hist = _parse_feature_csv(hist_path, min_rows=0) if os.path.exists(hist_path) else None
                
                if df_daily is None and df_hist is None:
                    df = None
                elif df_daily is not None and df_hist is not None:
                    df = (pd.concat([df_hist, df_daily], ignore_index=True)
                          .drop_duplicates(subset=["ts"], keep="last"))
                else:
                    df = df_daily if df_daily is not None else df_hist
            except Exception as e:
                logger.warning(f"[CALIB] parse failed: {e}")
                df = None

            # Optional freshness cap
            if isinstance(df, pd.DataFrame):
                cap_rows = int(os.getenv("CALIB_MAX_ROWS", "4000") or "4000")
                if cap_rows > 0 and len(df) > cap_rows:
                    df = df.tail(cap_rows)

            if df is None or (isinstance(df, pd.DataFrame) and df.empty):
                log_every("calib-not-enough", 60, logger.info, f"[CALIB] Not enough rows yet for calibration (min_dir_rows={min_dir_rows})")
                continue



            use_all_dir = os.getenv("CALIB_USE_ALL_DIR", "").strip().lower() in ("1", "true", "yes")
            if use_all_dir:
                m = (df["label"].isin(["BUY", "SELL"]))
            else:
                m = (df["label"].isin(["BUY", "SELL"])) & (df["tradeable"] == True)

            dfd = df[m].copy()
            if dfd.empty or len(dfd) < min_dir_rows:
                log_every("calib-dir-insufficient", 120, logger.info, f"[CALIB] directional rows insufficient: {len(dfd)}/{min_dir_rows} (all_dir={use_all_dir})")
                continue

            if "meta_p_xgb_raw" not in dfd.columns:
                logger.info("[CALIB] meta_p_xgb_raw not found in logs; skipping this cycle to avoid corrupt calibration")
                continue

            dfd = dfd.replace([np.inf, -np.inf], np.nan)
            before = len(dfd)
            dfd = dfd.dropna(subset=["meta_p_xgb_raw"])
            after = len(dfd)
            if after < min_dir_rows:
                log_every("calib-dropna-insufficient", 120, logger.info, f"[CALIB] rows after dropna insufficient: {after}/{min_dir_rows} (before={before})")
                continue

            p = np.asarray(dfd["meta_p_xgb_raw"].values, dtype=float)
            y = np.asarray((dfd["label"] == "BUY").astype(int).values, dtype=int)
            x = _logit_clip_arr(p).reshape(-1, 1)

            a = b = None
            try:
                from sklearn.linear_model import LogisticRegression
                clf = LogisticRegression(max_iter=500, solver="lbfgs", class_weight=None)
                clf.fit(x, y)
                a = float(clf.coef_.ravel()[0])
                b = float(clf.intercept_.ravel()[0])
            except Exception as e:
                logger.warning(f"[CALIB] sklearn fit failed: {e}")
                continue

            if not np.isfinite(a) or not np.isfinite(b):
                logger.info("[CALIB] invalid coefficients; skipping write")
                continue

            n = len(dfd)
            n_val = max(50, min(200, n // 5))
            y_val = y[-n_val:]
            p_raw_val = np.clip(p[-n_val:], 1e-9, 1 - 1e-9)
            z_val = a * _logit_clip_arr(p_raw_val) + b
            p_cal_val = _sigmoid(z_val)
            brier_raw = _brier(y_val, p_raw_val)
            brier_cal = _brier(y_val, p_cal_val)

            if not _monotonic_grid_ok(a, b):
                logger.info(f"[CALIB] rejected (non-monotonic or non-positive slope): a={a:.6f}, b={b:.6f}")
                continue
            if brier_cal > (brier_raw + 0.002):
                logger.info(f"[CALIB] rejected (Brier worsened): raw={brier_raw:.5f} calib={brier_cal:.5f}")
                continue

            meta = {
                "a": a,
                "b": b,
                "n": int(n),
                "source": "meta_p_xgb_raw",
                "last_success_ts": datetime.utcnow().isoformat() + "Z"
            }
            _atomic_write_json(calib_out_path, meta)
            logger.info(f"[CALIB] wrote a,b (n={n}, src=meta_p_xgb_raw) → {calib_out_path} | a={a:.6f} b={b:.6f}")

            try:
                if getattr(pipeline_ref, "_calib_bypass", False):
                    logger.info("[CALIB] hot-reload skipped: pipeline bypass active")
                else:
                    ok = bool(pipeline_ref.reload_calibration(calib_out_path))
                    logger.info(f"[CALIB] hot-reload {'ok' if ok else 'skipped'}")
            except Exception as e:
                logger.warning(f"[CALIB] hot-reload failed: {e}")

        except asyncio.CancelledError:
            logger.info("[CALIB] Calibrator cancelled")
            break
        except Exception as e:
            logger.error(f"[CALIB] loop error: {e}", exc_info=True)
