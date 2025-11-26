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
    min_dir_rows: int = 220
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
            hist_path = os.getenv("FEATURE_LOG_HIST", "trained_models/production/feature_log_hist.csv")
            mtime_hist = os.path.getmtime(hist_path) if os.path.exists(hist_path) else 0.0
            mtime_combined = max(mtime_daily, mtime_hist)
            if last_mtime is not None and mtime_combined <= last_mtime:
                log_every("calib-unchanged", 30, logger.debug, "[CALIB] Feature logs unchanged; skipping")
                continue
            last_mtime = mtime_combined



            hist_path = os.getenv("FEATURE_LOG_HIST", "trained_models/production/feature_log_hist.csv")
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


            # --- Adaptive multi-slice, class-balanced calibration ---

            use_all_dir = os.getenv("CALIB_USE_ALL_DIR", "").strip().lower() in ("1", "true", "yes")

            # Base directional dataset (respect tradeable when requested and present)
            if use_all_dir or ("tradeable" not in df.columns):
                mask_dir = (df["label"].isin(["BUY", "SELL"]))
            else:
                mask_dir = (df["label"].isin(["BUY", "SELL"])) & (df["tradeable"] == True)

            dfd_all = df[mask_dir].copy()
            if dfd_all.empty or len(dfd_all) < min_dir_rows:
                log_every(
                    "calib-dir-insufficient",
                    120,
                    logger.info,
                    f"[CALIB] directional rows insufficient: {len(dfd_all)}/{min_dir_rows} (all_dir={use_all_dir})",
                )
                continue



            if "meta_p_xgb_raw" not in dfd_all.columns:
                logger.info("[CALIB] meta_p_xgb_raw not found in logs; skipping this cycle to avoid corrupt calibration")
                continue

            # Clean NaNs/Infs only for the calibration source column
            dfd_all = dfd_all.replace([np.inf, -np.inf], np.nan)
            dfd_all = dfd_all.dropna(subset=["meta_p_xgb_raw"])

            if len(dfd_all) < min_dir_rows:
                log_every(
                    "calib-dropna-insufficient",
                    120,
                    logger.info,
                    f"[CALIB] rows after cleaning insufficient: {len(dfd_all)}/{min_dir_rows}",
                )
                continue

            # Compose candidate slices
            recent_rows = int(os.getenv("CALIB_RECENT_ROWS", "1200") or "1200")
            candidates = []

            # 1) recent/full tradeable-filtered (only if not using all_dir and column exists)
            if (not use_all_dir) and ("tradeable" in df.columns):
                dfd_tr = dfd_all.copy()
                if not dfd_tr.empty:
                    candidates.append(("recent_tr", dfd_tr.tail(recent_rows)))
                    candidates.append(("full_tr", dfd_tr))



            # 2) recent/full all directionals (ignoring tradeable)
            dfd_all_dir = df[df["label"].isin(["BUY", "SELL"])].copy()
            dfd_all_dir = dfd_all_dir.replace([np.inf, -np.inf], np.nan)
            dfd_all_dir = dfd_all_dir.dropna(subset=["meta_p_xgb_raw"])

            if len(dfd_all_dir) >= min_dir_rows:
                candidates.append(("recent_all", dfd_all_dir.tail(recent_rows)))
                candidates.append(("full_all", dfd_all_dir))



            # Helper: fit class-balanced Platt and score
            from sklearn.linear_model import LogisticRegression
            from sklearn.metrics import roc_auc_score

            def _fit_and_score(dfd_slice: pd.DataFrame):
                p = np.asarray(dfd_slice["meta_p_xgb_raw"].values, dtype=float)
                y = np.asarray((dfd_slice["label"] == "BUY").astype(int).values, dtype=int)


                # Fast skill gate: require AUC > 0.5
                try:
                    if len(np.unique(y)) == 2:
                        auc = float(roc_auc_score(y, p))
                    else:
                        auc = 0.5
                except Exception:
                    auc = 0.5

                # If AUC <= 0.5, try inverted mapping (1 - p) before giving up.
                if not (auc > 0.5):
                    try:
                        if len(np.unique(y)) == 2:
                            auc_inv = float(roc_auc_score(y, 1.0 - p))
                        else:
                            auc_inv = 0.5
                    except Exception:
                        auc_inv = 0.5

                    if auc_inv > 0.5:
                        # Use inverted probabilities going forward
                        p = 1.0 - p
                        auc = auc_inv
                    else:
                        # No positive skill in either direction for this slice
                        return None



                x = _logit_clip_arr(p).reshape(-1, 1)
                clf = LogisticRegression(max_iter=500, solver="lbfgs", class_weight="balanced")
                clf.fit(x, y)
                a = float(clf.coef_.ravel()[0])
                b = float(clf.intercept_.ravel()[0])
                n = len(y)

                # --- NEW: Minimum slope guard to avoid confidence collapse ---
                try:
                    min_slope = float(os.getenv("CALIB_MIN_SLOPE", "0.20"))
                except Exception:
                    min_slope = 0.20

                if not np.isfinite(a) or a < min_slope:
                    logger.info(
                        "[CALIB] reject candidate due to weak slope: a=%.4f < min_slope=%.4f (auc=%.3f, n=%d)",
                        float(a), float(min_slope), float(auc), int(n)
                    )
                    return None

                # Validation on a small holdout tail
                n_val = max(50, min(200, n // 5))
                y_val = y[-n_val:]
                p_raw_val = np.clip(p[-n_val:], 1e-9, 1 - 1e-9)
                z_val = a * _logit_clip_arr(p_raw_val) + b
                p_cal_val = _sigmoid(z_val)

                brier_raw = _brier(y_val, p_raw_val)
                brier_cal = _brier(y_val, p_cal_val)

                return {
                    "a": a,
                    "b": b,
                    "n": int(n),
                    "auc": float(auc),
                    "brier_raw": float(brier_raw),
                    "brier_cal": float(brier_cal),
                }

            # Try candidates in order; accept the first valid one
            accepted = None
            for name, d in candidates:
                try:
                    if d is None or d.empty or len(d) < min_dir_rows:
                        continue
                    res = _fit_and_score(d)
                    if res is None:
                        logger.info(f"[CALIB] candidate={name} skipped: AUC ≤ 0.5 or invalid data")
                        continue

                    a, b = res["a"], res["b"]
                    mono_ok = _monotonic_grid_ok(a, b)
                    logger.info(
                        "[CALIB] candidate=%s n=%d auc=%.3f a=%.6f b=%.6f raw=%.5f calib=%.5f mono_ok=%s",
                        name,
                        res["n"],
                        res["auc"],
                        a,
                        b,
                        res["brier_raw"],
                        res["brier_cal"],
                        mono_ok,
                    )

                    if not mono_ok:
                        continue
                    if res["brier_cal"] > (res["brier_raw"] + 0.002):
                        continue

                    accepted = (name, res)
                    break
                except Exception as e:
                    logger.debug(f"[CALIB] candidate {name} failed: {e}")

            # Maintain a small rejection streak counter on the function object
            if not hasattr(background_calibrator_loop, "_reject_streak"):
                setattr(background_calibrator_loop, "_reject_streak", 0)

            if accepted is None:
                background_calibrator_loop._reject_streak += 1
                logger.info(
                    "[CALIB] rejected all candidates (streak=%d)",
                    background_calibrator_loop._reject_streak,
                )
                # Auto-bypass after repeated failures
                try:
                    if (
                        background_calibrator_loop._reject_streak >= 3
                        and hasattr(pipeline_ref, "_calib_bypass")
                        and not getattr(pipeline_ref, "_calib_bypass", False)
                    ):
                        setattr(pipeline_ref, "_calib_bypass", True)
                        logger.info("[CALIB-HEALTH] Bypass ENABLED (repeated calibration rejections)")
                except Exception:
                    pass
                continue

            # Accept best candidate and write/hot-reload
            background_calibrator_loop._reject_streak = 0
            name, best = accepted
            a, b, n = best["a"], best["b"], best["n"]
            auc = best.get("auc", 0.5)

            MIN_A = float(os.getenv("CALIB_MIN_SLOPE", "0.2"))
            MIN_AUC = float(os.getenv("CALIB_MIN_AUC", "0.53"))

            if float(auc) < MIN_AUC:
                logger.info(
                    "[CALIB] auc=%.3f below MIN_AUC=%.3f → skip overwrite, keep previous calib",
                    float(auc), float(MIN_AUC)
                )
                continue

            if abs(float(a)) < MIN_A:
                logger.info(
                    "[CALIB] slope a=%.4f below MIN_A=%.3f → skip overwrite, keep previous calib",
                    float(a), float(MIN_A)
                )
                continue

            try:
                if hasattr(pipeline_ref, "_calib_bypass") and getattr(pipeline_ref, "_calib_bypass", False):
                    setattr(pipeline_ref, "_calib_bypass", False)
                    logger.info("[CALIB-HEALTH] Bypass DISABLED (valid calibration found)")
            except Exception:
                pass

            meta = {
                "a": float(a),
                "b": float(b),
                "n": int(n),
                "auc": float(auc),  # NEW: persist skill for downstream model_quality
                "source": "meta_p_xgb_raw",
                "last_success_ts": datetime.utcnow().isoformat() + "Z",
            }
            _atomic_write_json(calib_out_path, meta)
            logger.info(
                "[CALIB] wrote a,b (n=%d, src=meta_p_xgb_raw, candidate=%s) → %s | a=%.6f b=%.6f",
                n,
                name,
                calib_out_path,
                a,
                b,
            )

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
