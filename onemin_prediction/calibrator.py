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

def _logit_clip(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return np.log(p / (1.0 - p))

async def background_calibrator_loop(
    feature_log_path: str,
    calib_out_path: str,
    pipeline_ref,
    interval_sec: int = 1200,
    min_dir_rows: int = 500
):
    """
    Periodically fit Platt calibration on recent logs and write a,b to calib_out_path.
    Prefers p_xgb_raw from feature logs; falls back to buy_prob if missing.
    Auto hot-reloads into pipeline via pipeline_ref.reload_calibration().
    """
    logger.info(f"[CALIB] Calibrator started (every {interval_sec}s) → {calib_out_path}")
    last_mtime = None
    
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
                # Reuse trainer's robust parser
                from online_trainer import _parse_feature_csv
                df = _parse_feature_csv(feature_log_path, min_rows=min_dir_rows)
            except Exception as e:
                logger.warning(f"[CALIB] parse failed: {e}")
                df = None
            
            if df is None or df.empty:
                logger.info(f"[CALIB] Not enough rows yet for calibration (min_dir_rows={min_dir_rows})")
                continue
            
            # Filter to directional, tradeable rows
            m = (df["label"].isin(["BUY", "SELL"])) & (df["tradeable"] == True)
            dfd = df[m].copy()
            if dfd.empty or len(dfd) < min_dir_rows:
                logger.info(f"[CALIB] directional/tradeable rows insufficient: {len(dfd)}/{min_dir_rows}")
                continue
            

            # Prefer raw XGB p from logs (if provided by pipeline); else fallback to buy_prob
            p_col = "meta_p_xgb_raw" if "meta_p_xgb_raw" in dfd.columns else "buy_prob"


            
            p = np.asarray(dfd[p_col].values, dtype=float)
            y = np.asarray((dfd["label"] == "BUY").astype(int).values, dtype=int)
            
            # Guarded logit
            x = _logit_clip(p).reshape(-1, 1)
            
            # Fit logistic regression on logit(p) → y
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
                logger.warning("[CALIB] invalid coefficients; skipping write")
                continue
            
            meta = {
                "a": a,
                "b": b,
                "n": int(len(dfd)),
                "source": p_col,
            }
            _atomic_write_json(calib_out_path, meta)
            logger.info(f"[CALIB] wrote: a={a:.6f} b={b:.6f} (n={len(dfd)}, src={p_col}) → {calib_out_path}")
            
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
