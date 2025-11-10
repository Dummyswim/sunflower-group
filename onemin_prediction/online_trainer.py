"""
Online trainer for AR-NMS system.
Parses feature_log.csv and trains:
1. Directional classifier (XGBoost) on BUY vs SELL
2. Neutrality classifier (LogisticRegression) on HOLD/FLAT
Saves models for hot-reload.
"""

import os, json
import logging
import asyncio
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List


logger = logging.getLogger(__name__)


def _parse_feature_csv(path: str, min_rows: int = 200) -> Optional[pd.DataFrame]:
    """Parse feature log CSV into DataFrame."""
    try:
        if not os.path.exists(path):
            logger.info(f"[TRAIN] Feature log not found: {path}")
            return None
        
        # Read raw text for robustness (features are k=v tokens)
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
                    alpha = float(toks[4])
                    tradeable = toks[5].lower() == "true"
                    is_flat = toks[6].lower() == "true"
                    
                    # Remaining tokens are features and latent
                    feat_map: Dict[str, float] = {}
                    for tok in toks[8:]:
                        if not tok:
                            continue
                        # Expand packed "features" cell like "a=1;b=2;..."
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
                        # Skip latent
                        if tok.startswith("latent="):
                            continue
                        # Normal k=v tokens
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
                    # Merge features
                    row.update(feat_map)
                    rows.append(row)
                except Exception:
                    continue
        
        if len(rows) < min_rows:
            logger.info(f"[TRAIN] Not enough rows yet: {len(rows)}/{min_rows}")
            return None
        
        df = pd.DataFrame(rows)
        # Drop duplicated timestamps if any
        df = df.drop_duplicates(subset=["ts"], keep="last")
        return df
    except Exception as e:
        logger.error(f"[TRAIN] Parse feature CSV failed: {e}", exc_info=True)
        return None




def _build_datasets(df: pd.DataFrame) -> Tuple[
    Optional[Tuple[np.ndarray, np.ndarray]], 
    Optional[Tuple[np.ndarray, np.ndarray]], 
    List[str]
]:
    """Build directional and neutrality datasets and return feature column order."""
    try:
        # Neutral target: HOLD/FLAT indicator
        y_neu = ((df["is_flat"] == True) | (df["decision"] == "HOLD")).astype(int).values
        
        # Select numeric feature columns (exclude known meta)
        exclude = {"ts","decision","label","buy_prob","alpha","tradeable","is_flat"}
        feat_cols = sorted([c for c in df.columns if c not in exclude and df[c].dtype != "O"])
        
        # Build neutrality dataset over full df
        X_neu = df[feat_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).values
        neu_ds = (X_neu, y_neu[:len(df)])
        
        # Directional: BUY vs SELL only and tradeable True
        mask_dir = (df["label"].isin(["BUY", "SELL"])) & (df["tradeable"] == True)
        df_dir = df[mask_dir].copy()
        
        if df_dir.empty:
            # No directional dataset, but neutrality is available
            return None, neu_ds, feat_cols
        
        y_dir = (df_dir["label"] == "BUY").astype(int).values
        X_dir = df_dir[feat_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).values
        
        return (X_dir, y_dir), neu_ds, feat_cols
    except Exception as e:
        logger.error(f"[TRAIN] Dataset build failed: {e}", exc_info=True)
        return None, None, []






def _train_xgb(X: np.ndarray, y: np.ndarray):
    """Train XGBoost directional classifier."""
    try:
        import xgboost as xgb
        dm = xgb.DMatrix(X, label=y)
        params = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "max_depth": 5,
            "eta": 0.08,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 1.0,
            "lambda": 1.0,
            "alpha": 0.0,
        }
        bst = xgb.train(params, dm, num_boost_round=200)
        return bst
    except Exception as e:
        logger.error(f"[TRAIN] XGB training failed: {e}", exc_info=True)
        return None


def _train_neutrality(X: np.ndarray, y: np.ndarray):
    """Train neutrality classifier (LogisticRegression)."""
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline
        
        clf = Pipeline([
            ("scaler", StandardScaler(with_mean=False)),
            ("lr", LogisticRegression(max_iter=200, solver="lbfgs"))
        ])
        clf.fit(X, y)
        return clf
    except Exception as e:
        logger.error(f"[TRAIN] Neutrality model training failed: {e}", exc_info=True)
        return None




async def background_trainer_loop(
    feature_log_path: str,
    xgb_out_path: str,
    neutral_out_path: str,
    pipeline_ref,
    interval_sec: int = 600,
    min_rows: int = 300
):
    """
    Periodically trains models from feature logs and hot-reloads into the pipeline.
    
    Args:
        feature_log_path: Path to feature_log.csv
        xgb_out_path: Output path for XGB model
        neutral_out_path: Output path for neutrality model
        pipeline_ref: Reference to AdaptiveModelPipeline for hot-reload
        interval_sec: Training interval in seconds
        min_rows: Minimum rows required before training
    """
    logger.info(f"[TRAIN] Background trainer started (every {interval_sec}s)")
    last_mtime = None
    
    while True:
        try:
            await asyncio.sleep(interval_sec)
            
            if not os.path.exists(feature_log_path):
                continue
            
            mtime = os.path.getmtime(feature_log_path)
            if last_mtime is not None and mtime <= last_mtime:
                continue
            last_mtime = mtime
            
            # Parse feature log
            df = _parse_feature_csv(feature_log_path, min_rows=min_rows)
            if df is None:
                continue

            # Build datasets
            dir_ds, neu_ds, feat_cols = _build_datasets(df)

            xgb_model = None
            neutral_model = None

            # Train directional model
            if dir_ds is not None:
                Xd, yd = dir_ds
                xgb_model = _train_xgb(Xd, yd)
                if xgb_model is not None:
                    try:
                        xgb_model.save_model(xgb_out_path)
                        logger.info(f"[TRAIN] Saved XGB to {xgb_out_path}")
                    except Exception as e:
                        logger.warning(f"[TRAIN] Failed to save XGB: {e}")

            # Train neutrality model
            if neu_ds is not None:
                Xn, yn = neu_ds
                neutral_model = _train_neutrality(Xn, yn)
                if neutral_model is not None:
                    try:
                        import joblib
                        joblib.dump(neutral_model, neutral_out_path)
                        logger.info(f"[TRAIN] Saved Neutral model to {neutral_out_path}")
                    except Exception as e:
                        logger.warning(f"[TRAIN] Failed to save neutrality model: {e}")


            # ========== PERSIST SCHEMA AND HOT-RELOAD ==========
            try:
                # Persist schema only if non-empty and at least one model is trained
                if feat_cols and (xgb_model is not None or neutral_model is not None):
                    from pathlib import Path
                    # Use module-level json (imported at top) to avoid shadowing
                    schema = {
                        "feature_names": list(feat_cols),
                        "num_features": int(len(feat_cols))
                    }
                    schema_path = Path(xgb_out_path).parent / "feature_schema.json"
                    schema_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(schema_path, "w", encoding="utf-8") as f:
                        # Use module-level json instead of local import
                        json.dump(schema, f, indent=2)
                    logger.info(f"[TRAIN] Saved feature schema: {schema_path} n={schema['num_features']}")
                    
                    # Hot-reload schema into pipeline
                    if pipeline_ref is not None and hasattr(pipeline_ref, "set_feature_schema"):
                        pipeline_ref.set_feature_schema(schema["feature_names"])
                        logger.info("[TRAIN] Feature schema hot-reloaded into pipeline")
                else:
                    logger.info("[TRAIN] Skipping schema save: no features or models not trained")
            except Exception as e:
                logger.warning(f"[TRAIN] Failed to persist/hotload feature schema: {e}")


            # Hot reload models
            try:
                if pipeline_ref is not None:
                    # Wrap XGB booster
                    if xgb_model is not None:
                        import xgboost as xgb
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
                    else:
                        xgb_wrapped = None
                    
                    pipeline_ref.replace_models(xgb=xgb_wrapped, neutral=neutral_model)
            except Exception as e:
                logger.error(f"[TRAIN] Hot-reload into pipeline failed: {e}", exc_info=True)
        
        except asyncio.CancelledError:
            logger.info("[TRAIN] Background trainer cancelled")
            break
        except Exception as e:
            logger.error(f"[TRAIN] Trainer loop error: {e}", exc_info=True)
            await asyncio.sleep(5)



