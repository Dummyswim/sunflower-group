# online_trainer.py
"""
Online trainer for probabilities-only system.
Parses feature_log.csv and trains:
1. Directional classifier (XGBoost) on BUY vs SELL (all directionals)
2. Neutrality classifier (LogisticRegression) on HOLD/FLAT
Saves models for hot-reload.
"""
import os, json
import logging
import asyncio
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List
from logging_setup import log_every

logger = logging.getLogger(__name__)

# Preferred minimal feature set (shared with offline trainer)
PREFERRED_FEATURES: Tuple[str, ...] = (
    "ema_8",
    "ema_21",
    "ema_50",
    "last_price",
    "ta_rsi14",
    "ta_macd_hist",
    "ta_bb_pctb",
    "ta_bb_bw",
    "micro_slope",
    "micro_imbalance",
    "mean_drift_pct",
    "last_zscore",
    "atr_1t",
    "rv_10",
    "tod_sin",
    "tod_cos",
    "struct_pivot_swipe_up",
    "struct_pivot_swipe_down",
    "struct_fvg_up_present",
    "struct_fvg_down_present",
    "struct_ob_bull_present",
    "struct_ob_bear_present",
    "wick_extreme_up",
    "wick_extreme_down",
    "vwap_reversion_flag",
    "cvd_divergence",
    "ta_adx",
    "ta_mfi",
    "ta_obv_z",
)

OFFLINE_STATS_PATH = os.getenv(
    "OFFLINE_STATS_PATH", "trained_models/production/xgb_5min_feature_stats.json"
)

_SKEW_CACHE: Dict[str, Optional[object]] = {"mtime": None, "stats": None}

def _atomic_write_json(path: str, obj: dict) -> None:
    from pathlib import Path
    import os as _os
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
    _os.replace(tmp, p)

def _parse_feature_csv(path: str, min_rows: int = 200) -> Optional[pd.DataFrame]:
    try:
        if not os.path.exists(path):
            logger.info(f"[TRAIN] Feature log not found: {path}")
            return None

        # --- Fast path: wide CSV with headers (backfill/offline style) ---
        try:
            df_head = pd.read_csv(path, nrows=5)
            if "ts" in df_head.columns and "label" in df_head.columns:
                df = pd.read_csv(path)
                if len(df) < min_rows:
                    log_every(
                        "train-parse-not-enough",
                        60,
                        logger.info,
                        f"[TRAIN] Not enough rows yet: {len(df)}/{min_rows}",
                    )
                    return None

                # Ensure required meta columns exist with sane defaults
                if "decision" not in df.columns:
                    df["decision"] = "USER"
                if "buy_prob" not in df.columns:
                    df["buy_prob"] = 0.5
                if "alpha" not in df.columns:
                    df["alpha"] = 0.0
                if "tradeable" not in df.columns:
                    df["tradeable"] = True
                if "is_flat" not in df.columns:
                    df["is_flat"] = df["label"].astype(str).str.upper().eq("FLAT")

                df = df.drop_duplicates(subset=["ts"], keep="last")
                return df
        except Exception as e:
            logger.debug(f"[TRAIN] Wide CSV parse fallback failed: {e}")

        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                toks = [t.strip() for t in line.split(",") if t.strip() != ""]
                if len(toks) < 8:
                    continue
                try:
                    ts = toks[0]
                    decision = toks[1]
                    label = toks[2]
                    buy_prob = float(toks[3])
                    alpha = float(toks[4]) if toks[4] not in ("", "None", "nan") else 0.0
                    tradeable = toks[5].lower() == "true"
                    is_flat = toks[6].lower() == "true"
                    feat_map: Dict[str, float] = {}
                    for tok in toks[8:]:
                        if not tok:
                            continue
                        if tok.startswith("features="):
                            _, packed = tok.split("=", 1)
                            for sub in packed.split(";"):
                                if "=" not in sub:
                                    continue
                                k, v = sub.split("=", 1)
                                try:
                                    feat_map[k.strip()] = float(v.strip())
                                except Exception:
                                    continue
                            continue
                        if tok.startswith("latent="):
                            continue
                        if "=" in tok:
                            k, v = tok.split("=", 1)
                            try:
                                feat_map[k.strip()] = float(v.strip())
                            except Exception:
                                continue
                    row = {
                        "ts": ts,
                        "decision": decision,
                        "label": label,
                        "buy_prob": buy_prob,
                        "alpha": alpha,
                        "tradeable": tradeable,
                        "is_flat": is_flat,
                    }
                    row.update(feat_map)
                    rows.append(row)
                except Exception:
                    continue
        if len(rows) < min_rows:
            log_every("train-parse-not-enough", 60, logger.info, f"[TRAIN] Not enough rows yet: {len(rows)}/{min_rows}")
            return None
        df = pd.DataFrame(rows)
        df = df.drop_duplicates(subset=["ts"], keep="last")
        return df
    except Exception as e:
        logger.error(f"[TRAIN] Parse feature CSV failed: {e}", exc_info=True)
        return None

def _select_feature_columns(df: pd.DataFrame) -> List[str]:
    exclude = {"ts", "decision", "label", "buy_prob", "alpha", "tradeable", "is_flat"}
    drop_prefixes = ("meta_", "p_xgb_")

    numeric_cols = [
        c
        for c in df.columns
        if c not in exclude
        and df[c].dtype != "O"
        and not any(c.startswith(p) for p in drop_prefixes)
    ]

    preferred_available = [c for c in PREFERRED_FEATURES if c in numeric_cols]
    if len(preferred_available) >= 10:
        feat_cols = sorted(preferred_available)
        logger.info("[TRAIN] Using preferred minimal feature set: n=%d", len(feat_cols))
    else:
        feat_cols = sorted(numeric_cols)
        logger.info(
            "[TRAIN] Using full numeric feature set: n=%d (no minimal subset)",
            len(feat_cols),
        )

    return feat_cols


def _build_datasets(df: pd.DataFrame) -> Tuple[
    Optional[Tuple[np.ndarray, np.ndarray]],
    Optional[Tuple[np.ndarray, np.ndarray]],
    List[str]
]:
    try:
        y_neu = (df["label"] == "FLAT").astype(int).values

        feat_cols = _select_feature_columns(df)

        X_neu = (
            df[feat_cols]
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .values
        )
        neu_ds = (X_neu, y_neu[:len(df)])

        mask_dir = (df["label"].isin(["BUY", "SELL"]))
        df_dir = df[mask_dir].copy()
        if df_dir.empty:
            logger.warning("[TRAIN] No directional rows (BUY/SELL) after filtering; skipping directional model.")
            return None, neu_ds, feat_cols

        # Optional: strength diagnostics from aux_ret_main (if available)
        try:
            if "aux_ret_main" in df_dir.columns:
                abs_ret = df_dir["aux_ret_main"].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
                q50 = float(abs_ret.quantile(0.50))
                q75 = float(abs_ret.quantile(0.75))
                q90 = float(abs_ret.quantile(0.90))
                logger.info(
                    "[TRAIN] Directional strength |aux_ret_main|: q50=%.5f q75=%.5f q90=%.5f",
                    q50,
                    q75,
                    q90,
                )
        except Exception:
            pass

        y_dir = (df_dir["label"] == "BUY").astype(int).values
        X_dir = (
            df_dir[feat_cols]
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .values
        )

        try:
            neu_pos = float(np.mean(y_neu)) if len(y_neu) else 0.0
            dir_size = int(df_dir.shape[0])
            logger.info(
                "[TRAIN] Dataset sizes: dir=%d rows | neu=%d rows | neu_pos=%.3f",
                dir_size,
                len(df),
                neu_pos,
            )
        except Exception:
            pass

        return (X_dir, y_dir), neu_ds, feat_cols
    except Exception as e:
        logger.error(f"[TRAIN] Dataset build failed: {e}", exc_info=True)
        return None, None, []


def _load_offline_stats() -> Optional[Dict[str, Dict[str, float]]]:
    """Load offline feature mean/std once, cached by mtime."""
    try:
        path = Path(OFFLINE_STATS_PATH)
        if not path.exists():
            return None

        mtime = path.stat().st_mtime
        if _SKEW_CACHE["mtime"] == mtime and _SKEW_CACHE["stats"] is not None:
            return _SKEW_CACHE["stats"]  # type: ignore[return-value]

        with path.open("r", encoding="utf-8") as f:
            stats = json.load(f)

        if not isinstance(stats, dict):
            return None

        _SKEW_CACHE["mtime"] = mtime
        _SKEW_CACHE["stats"] = stats
        logger.info("[SKEW] Loaded offline stats from %s", OFFLINE_STATS_PATH)
        return stats
    except Exception as e:
        logger.debug(f"[SKEW] Failed to load offline stats: {e}")
        return None


def _check_feature_skew(
    df: pd.DataFrame,
    feat_cols: List[str],
    k_sigma: float = 4.0,
    max_alerts: int = 10,
    min_std: float = 1e-4,
) -> None:
    """
    Compare online feature means to offline mean/std and log alerts
    when |mean_online - mean_offline| > k_sigma * std_offline.

    Features with effectively zero offline variance (std < min_std) are skipped
    to avoid bogus z-scores from numerical noise.
    """
    stats = _load_offline_stats()
    if not stats:
        return

    alerts: List[str] = []

    for c in feat_cols:
        if c not in stats:
            continue
        try:
            offline_mean = float(stats[c].get("mean", 0.0))
            offline_std = float(stats[c].get("std", 0.0))
            if (
                not np.isfinite(offline_std)
                or offline_std <= 0.0
                or offline_std < min_std
            ):
                continue

            series = (
                df[c]
                .replace([np.inf, -np.inf], np.nan)
                .dropna()
                .astype(float)
            )
            if series.empty:
                continue

            online_mean = float(series.mean())
            z = abs(online_mean - offline_mean) / offline_std
            if z >= k_sigma:
                alerts.append(
                    f"{c}: z={z:.1f} off={offline_mean:.4g} on={online_mean:.4g}"
                )
                if len(alerts) >= max_alerts:
                    break
        except Exception:
            # Be robust against any one bad column
            continue

    if alerts:
        logger.warning(
            "[SKEW] mean-shift alerts (>|%.1f|σ) on %d features: %s",
            k_sigma,
            len(alerts),
            "; ".join(alerts),
        )


def _train_xgb(X: np.ndarray, y: np.ndarray):
    try:
        import xgboost as xgb
        dm = xgb.DMatrix(X, label=y)
        pos = float(np.sum(y == 1))
        neg = float(np.sum(y == 0))
        spw = float(neg / max(1.0, pos)) if pos > 0 else 1.0
        params = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "max_depth": 5,
            "eta": 0.05,
            "subsample": 0.9,
            "colsample_bytree": 0.8,
            "min_child_weight": 1.0,
            "reg_lambda": 1.0,  # lambda_l2 equivalent
            "alpha": 0.0,
            "scale_pos_weight": float(np.clip(spw, 0.5, 10.0)),
        }
        bst = xgb.train(params, dm, num_boost_round=200)
        return bst
    except Exception as e:
        logger.error(f"[TRAIN] XGB training failed: {e}", exc_info=True)
        return None

def _train_neutrality(X: np.ndarray, y: np.ndarray):
    """
    Train a FLAT vs non-FLAT neutrality classifier with standardization and stability guards.
    """
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
    except Exception as e:
        logger.error(f"[TRAIN] Neutrality model imports failed: {e}", exc_info=True)
        return None

    try:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int).ravel()

        if X.size == 0 or y.size == 0:
            logger.warning("[NEUTRAL] Empty X or y; skipping neutrality model.")
            return None

        if X.ndim != 2:
            try:
                X = X.reshape(len(y), -1)
            except Exception as e:
                logger.error(f"[NEUTRAL] Failed to reshape X: {e}", exc_info=True)
                return None

        mask = np.isfinite(X).all(axis=1)
        mask &= np.isfinite(y)
        X = X[mask]
        y = y[mask]

        if X.shape[0] < 200:
            logger.info("[NEUTRAL] Only %d clean rows; skipping neutrality model (unstable).", X.shape[0])
            return None

        pipe = Pipeline(
            steps=[
                ("scaler", StandardScaler(with_mean=True, with_std=True)),
                ("clf", LogisticRegression(
                    solver="lbfgs",
                    max_iter=1500,
                    C=0.5,
                    tol=1e-3,
                    n_jobs=-1,
                    class_weight="balanced",
                )),
            ]
        )

        logger.info(
            "[NEUTRAL] Training neutrality model: n=%d, pos_rate=%.3f, n_features=%d",
            X.shape[0],
            float(y.mean()) if y.size else 0.0,
            X.shape[1],
        )

        pipe.fit(X, y)

        try:
            coef_norm = float(np.linalg.norm(pipe.named_steps["clf"].coef_))
        except Exception:
            coef_norm = float("nan")

        logger.info("[NEUTRAL] Neutrality model trained successfully (coef_norm=%.3f)", coef_norm)
        return pipe
    except Exception as e:
        logger.error(f"[TRAIN] Neutrality model training failed: {e}", exc_info=True)
        return None

async def background_trainer_loop(
    feature_log_path: str,
    xgb_out_path: str,
    neutral_out_path: str,
    pipeline_ref,
    interval_sec: int = 300,
    min_rows: int = 100
):
    logger.info(f"[TRAIN] Background trainer started (every {interval_sec}s)")
    last_mtime = None

    while True:
        try:
            if not os.path.exists(feature_log_path):
                log_every("train-missing-log", 60, logger.info, f"[TRAIN] Feature log not found: {feature_log_path}")
                await asyncio.sleep(interval_sec)
                continue

            mtime_daily = os.path.getmtime(feature_log_path)
            hist_path = os.getenv("FEATURE_LOG_HIST", "trained_models/production/feature_log_hist.csv")
            mtime_hist = os.path.getmtime(hist_path) if os.path.exists(hist_path) else 0.0
            mtime_combined = max(mtime_daily, mtime_hist)
            if last_mtime is not None and mtime_combined <= last_mtime:
                log_every("train-unchanged", 30, logger.debug, "[TRAIN] Feature logs unchanged since last check")
                await asyncio.sleep(interval_sec)
                continue
            last_mtime = mtime_combined



            hist_path = os.getenv("FEATURE_LOG_HIST", "trained_models/production/feature_log_hist.csv")
            df_daily = _parse_feature_csv(feature_log_path, min_rows=0)
            df_hist = _parse_feature_csv(hist_path, min_rows=0) if os.path.exists(hist_path) else None

            if df_daily is None and df_hist is None:
                log_every("train-not-enough", 60, logger.info, "[TRAIN] Not enough data to train (no logs found)")
                await asyncio.sleep(interval_sec)
                continue

            if df_daily is not None and df_hist is not None:
                df = pd.concat([df_hist, df_daily], ignore_index=True)
            elif df_daily is not None:
                df = df_daily
            else:
                df = df_hist




            df = df.drop_duplicates(subset=["ts"], keep="last")
            cap_rows = int(os.getenv("TRAIN_MAX_ROWS", "4000") or "4000")
            if cap_rows > 0 and len(df) > cap_rows:
                df = df.tail(cap_rows)

            if len(df) < min_rows:
                log_every("train-not-enough", 60, logger.info, f"[TRAIN] Not enough data to train (rows={len(df)}/{min_rows})")
                await asyncio.sleep(interval_sec)
                continue
            dir_ds, neu_ds, feat_cols = _build_datasets(df)

            # Align trainer feature space to the XGB booster schema so XGB and neutrality share the same input space
            schema_names = getattr(pipeline_ref, "feature_schema_names", None)
            if schema_names and feat_cols:
                original_n = len(feat_cols)
                schema_set = set(schema_names)
                aligned_cols = [c for c in feat_cols if c in schema_set]

                if not aligned_cols:
                    logger.warning(
                        "[TRAIN] No overlap between feature columns (n=%d) and XGB schema (n=%d); skipping this cycle",
                        original_n,
                        len(schema_names),
                    )
                    await asyncio.sleep(interval_sec)
                    continue

                logger.info(
                    "[TRAIN] Aligned feature columns to XGB schema: n=%d (was %d, schema_n=%d)",
                    len(aligned_cols),
                    original_n,
                    len(schema_names),
                )
                feat_cols = aligned_cols

                # Rebuild neutrality dataset on aligned columns
                try:
                    y_neu = (df["label"] == "FLAT").astype(int).values
                    X_neu = df[feat_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).values
                    neu_ds = (X_neu, y_neu[:len(df)])
                except Exception as e:
                    logger.error(f"[TRAIN] Rebuild neu_ds with aligned schema failed: {e}", exc_info=True)
                    neu_ds = None

                # Rebuild directional dataset on aligned columns
                try:
                    mask_dir = df["label"].isin(["BUY", "SELL"])
                    df_dir = df[mask_dir].copy()
                    if df_dir.empty:
                        dir_ds = None
                    else:
                        y_dir = (df_dir["label"] == "BUY").astype(int).values
                        X_dir = df_dir[feat_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).values
                        dir_ds = (X_dir, y_dir)
                except Exception as e:
                    logger.error(f"[TRAIN] Rebuild dir_ds with aligned schema failed: {e}", exc_info=True)
                    dir_ds = None





            # Skew monitor: compare online feature distribution vs offline baseline
            try:
                _check_feature_skew(df, feat_cols)
            except Exception as e:
                logger.debug(f"[SKEW] check failed: {e}")

            # Gate XGB training/hot-reload (env-configurable)
            try:
                # Default lowered to allow training on smaller recent buffers
                min_dir_rows = int(os.getenv("TRAIN_MIN_DIR_ROWS", "120"))
            except Exception:
                min_dir_rows = 120
            try:
                min_minor_share = float(os.getenv("TRAIN_MIN_MINOR_SHARE", "0.30"))
            except Exception:
                min_minor_share = 0.30


            
            dir_ok = False
            dir_stats = "n/a"
            if dir_ds is not None:
                Xd, yd = dir_ds
                n_dir = int(Xd.shape[0])
                if n_dir > 0:
                    pos_share = float(np.mean(yd))
                    minor_share = float(min(pos_share, 1.0 - pos_share))
                    dir_ok = (n_dir >= min_dir_rows) and (minor_share >= min_minor_share)
                    dir_stats = f"n={n_dir} pos_share={pos_share:.3f} minor_share={minor_share:.3f}"
            else:
                dir_stats = "none"

            logger.info(f"[TRAIN] Directional dataset check: {dir_stats} -> {'OK' if dir_ok else 'SKIP'}")


            xgb_model = None
            neutral_model = None

            if dir_ds is not None and dir_ok:
                Xd, yd = dir_ds
                xgb_model = _train_xgb(Xd, yd)
                if xgb_model is None:
                    logger.warning("[TRAIN] Skipping XGB save: training returned None")
            else:
                xgb_model = None
                logger.info("[TRAIN] Skipping XGB training/hot-reload due to insufficient directional dataset")

            if neu_ds is not None:
                Xn, yn = neu_ds
                neutral_model = _train_neutrality(Xn, yn)
                if neutral_model is None:
                    logger.warning("[TRAIN] Skipping neutrality save: training returned None")

            try:
                if xgb_model is not None and feat_cols:
                    xgb_model.set_attr(feature_schema=json.dumps({"feature_names": list(feat_cols)}))
                    logger.info(f"[TRAIN] Embedded feature_schema into booster (n={len(feat_cols)})")
            except Exception as e:
                logger.warning(f"[TRAIN] Failed to set booster feature_schema attr: {e}")

            # Persist schema first
            schema_ok = False
            try:

                if feat_cols and (xgb_model is not None):

                    from pathlib import Path
                    from datetime import datetime, timezone
                    schema = {
                        "feature_names": list(feat_cols),
                        "num_features": int(len(feat_cols)),
                        "written_at": datetime.now(timezone.utc).isoformat()
                    }
                    base_dir = Path(xgb_out_path).parent
                    schema_path = base_dir / "feature_schema.json"
                    ver = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
                    schema_ver_path = base_dir / f"feature_schema_v{ver}.json"
                    _atomic_write_json(str(schema_ver_path), schema)
                    _atomic_write_json(str(schema_path), schema)
                    logger.info(f"[TRAIN] Saved feature schema: {schema_path} n={schema['num_features']} (snapshot={schema_ver_path.name})")

                    if pipeline_ref is not None and hasattr(pipeline_ref, "set_feature_schema"):
                        pipeline_ref.set_feature_schema(schema["feature_names"])
                        logger.info("[TRAIN] Feature schema hot-reloaded into pipeline")
                    schema_ok = True
                else:
                    logger.info("[TRAIN] Skipping schema save: no features or models not trained")
            except Exception as e:
                logger.error(f"[TRAIN] Failed to persist/hotload feature schema: {e}", exc_info=True)
                schema_ok = False

            if not schema_ok:
                logger.warning("[TRAIN] Schema not saved; skipping model saves this cycle to avoid misalignment")
            else:
                try:
                    if xgb_model is not None:
                        xgb_model.save_model(xgb_out_path)
                        logger.info(f"[TRAIN] Saved XGB to {xgb_out_path}")
                except Exception as e:
                    logger.warning(f"[TRAIN] Failed to save XGB: {e}")
                try:
                    if neutral_model is not None:
                        import joblib as _joblib
                        _joblib.dump(neutral_model, neutral_out_path)
                        logger.info(f"[TRAIN] Saved Neutral model to {neutral_out_path}")
                except Exception as e:
                    logger.warning(f"[TRAIN] Failed to save neutrality model: {e}")

            # Hot reload models into pipeline
            try:
                if pipeline_ref is not None:
                    import xgboost as xgb
                    xgb_wrapped = None
                    if xgb_model is not None:
                        booster = xgb_model
                        class _BoosterWrapper:
                            def __init__(self, booster):
                                self.booster = booster
                                self.is_dummy = False
                                self.name = "XGBBooster"
                            def predict_proba(self, X):
                                dm = xgb.DMatrix(X)
                                p = self.booster.predict(dm)
                                import numpy as np
                                if p.ndim == 1:
                                    p = np.clip(p, 1e-9, 1 - 1e-9)
                                    return np.stack([1 - p, p], axis=1)
                                return p
                        xgb_wrapped = _BoosterWrapper(booster)
                    pipeline_ref.replace_models(xgb=xgb_wrapped, neutral=neutral_model)
                    logger.info("[TRAIN] Hot-reloaded models into pipeline")
            except Exception as e:
                logger.error(f"[TRAIN] Hot-reload into pipeline failed: {e}", exc_info=True)
        except asyncio.CancelledError:
            logger.info("[TRAIN] Background trainer cancelled")
            break
        except Exception as e:
            logger.error(f"[TRAIN] Trainer loop error: {e}", exc_info=True)
        finally:
            await asyncio.sleep(interval_sec)
