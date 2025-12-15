# model_pipeline.py
"""
Adaptive model pipeline (probabilities-only).
- XGB probabilities (+ optional Platt calibration)
- Indicator/pattern/MTF conservative modulation
- Optional neutrality probability
"""
import json
import math
import numpy as np
from typing import Dict, Optional, Tuple
import logging
import os
from logging_setup import log_every

logger = logging.getLogger(__name__)

def _safe_booster_num_features(xgb_obj) -> Optional[int]:
    """Return booster.num_features() if available."""
    try:
        b = None
        if hasattr(xgb_obj, 'booster'):
            b = getattr(xgb_obj, 'booster')
        elif hasattr(xgb_obj, 'get_booster'):
            b = xgb_obj.get_booster()
        if b is None:
            return None
        if hasattr(b, 'num_features'):
            return int(b.num_features())
    except Exception:
        return None
    return None


def _load_schema_from_disk() -> Optional[list]:
    """
    Disk fallback for feature schema. Source of truth order:
      1) FEATURE_SCHEMA_COLS_PATH (explicit)
      2) feature_schema_cols.json next to XGB_PATH
      3) feature_schema_v*.json newest in same dir as XGB_PATH
    Returns list of feature names or None.
    """
    try:
        import glob, json, os
        from pathlib import Path
        p = (os.getenv('FEATURE_SCHEMA_COLS_PATH','') or '').strip()
        if p and Path(p).exists():
            data = json.loads(Path(p).read_text(encoding='utf-8'))
            if isinstance(data, dict) and 'feature_names' in data:
                return list(data['feature_names'])
            if isinstance(data, dict) and 'columns' in data:
                return list(data['columns'])
            if isinstance(data, list):
                return list(data)
        xgb_path = (os.getenv('XGB_PATH','') or '').strip()
        if xgb_path:
            d = Path(xgb_path).resolve().parent
            p2 = d / 'feature_schema_cols.json'
            if p2.exists():
                data = json.loads(p2.read_text(encoding='utf-8'))
                if isinstance(data, dict) and 'columns' in data:
                    return list(data['columns'])
                if isinstance(data, dict) and 'feature_names' in data:
                    return list(data['feature_names'])
                if isinstance(data, list):
                    return list(data)
            candidates = sorted(d.glob('feature_schema_v*.json'))
            if candidates:
                newest = candidates[-1]
                data = json.loads(newest.read_text(encoding='utf-8'))
                if isinstance(data, dict) and 'feature_names' in data:
                    return list(data['feature_names'])
                if isinstance(data, dict) and 'columns' in data:
                    return list(data['columns'])
                if isinstance(data, list):
                    return list(data)
    except Exception as e:
        logger.warning(f"[SCHEMA] Disk fallback load failed: {e}")
    return None


class AdaptiveModelPipeline:
    def __init__(
        self,
        xgb,
        base_weights: Optional[Dict[str, float]] = None,
        neutral_model: Optional[object] = None
    ):
        self.xgb = xgb
        self._xgb_disabled_reason = None  # set when schema/model mismatch
        self.neutral_model = neutral_model

        if base_weights is None:
            base_weights = {
                'ema_trend': 0.35,
                'micro_slope': 0.25,
                'imbalance': 0.20,
                'mean_drift': 0.20
            }
        self.weights = dict(base_weights)
        self.min_w = 0.05
        self.max_w = 0.80

        self.feature_schema_names = None
        self.feature_schema_size = None

        # Calibration
        self._calib_bypass = False
        self._calib_a = None
        self._calib_b = None
        self.last_p_xgb_raw = None
        self.last_p_xgb_calib = None
        self.calibrator_ready = False
        self.platt_path = os.getenv("CALIB_PATH", "").strip() or None
        self._calib_mtime = None
        # Global estimate of current model skill ("weak" / "ok" / "strong")
        self.model_quality = "unknown"

        try:
            calib_path = self.platt_path
            if calib_path and os.path.exists(calib_path):
                if self.reload_calibration(calib_path):
                    try:
                        self._calib_mtime = os.path.getmtime(calib_path)
                    except Exception:
                        self._calib_mtime = None
                    self.calibrator_ready = True
            else:
                logger.info("[CALIB] No calibration file; raw XGB probabilities will be used")
        except Exception as e:
            logger.warning(f"[CALIB] Failed to load calibration: {e}")

        # If starting with bypass enabled, mark quality as weak baseline
        if getattr(self, "_calib_bypass", False):
            try:
                self._update_model_quality(auc=0.5, slope=0.0)
            except Exception:
                pass

        logger.info("Adaptive model pipeline initialized (probabilities-only)")





    def _apply_calibration(self, p: float) -> float:
        """
        Apply Platt calibration to a raw BUY probability.

        Behaviour:
        - If _calib_bypass is True, calibration is effectively disabled:
          we return the raw probability (clipped) without extra damping.
        - Otherwise, use current (a,b), with guards against
          non-monotonic or extreme inversions.
        """
        try:
            # Respect dynamic bypass flag
            if getattr(self, "_calib_bypass", False):
                # True bypass: trust raw XGB output, only clip for numerical safety.
                p_raw = float(np.clip(p, 1e-9, 1.0 - 1e-9))
                logger.debug(
                    "[CALIB] bypass active → using raw probability p=%.3f",
                    p_raw,
                )
                return p_raw

            if self._calib_a is None or self._calib_b is None:
                return float(p)

            a = float(self._calib_a)
            b = float(self._calib_b)
            if not np.isfinite(a) or not np.isfinite(b) or a <= 0.0:
                logger.info(f"[CALIB-GUARD] invalid coefficients a={a} b={b} → using raw")
                return float(p)

            p = float(np.clip(p, 1e-9, 1.0 - 1e-9))
            logit = float(np.log(p / (1.0 - p)))
            z = (a * logit) + b
            q = 1.0 / (1.0 + np.exp(-z))
            q = float(np.clip(q, 1e-6, 1.0 - 1e-6))

            # Guard against large sign inversions
            if (p - 0.5) * (q - 0.5) < 0 and abs(q - p) > 0.20:
                logger.info(f"[CALIB-GUARD] large inversion Δ={q - p:+.3f} (p={p:.3f} → q={q:.3f}) → using raw")
                return float(p)

            # Clamp overly aggressive deltas
            max_delta = 0.12 + 0.25 * abs(p - 0.5)
            delta = q - p
            if abs(delta) > max_delta:
                q_clamped = float(p + np.clip(delta, -max_delta, max_delta))
                logger.info(
                    f"[CALIB-GUARD] clamp Δ={delta:+.3f}→{(q_clamped - p):+.3f} (p={p:.3f} → q={q:.3f})"
                )
                return q_clamped

            return q
        except Exception as e:
            logger.warning(f"[CALIB] calibration failed, using raw p (error={e})")
            try:
                return float(p)
            except Exception:
                return 0.5

    def _update_model_quality(self, auc: float | None, slope: float | None) -> None:
        """
        Map calibration skill into a coarse model_quality bucket.

        This is used by indicator blending to decide how much to trust raw p
        when indicators / futures strongly disagree.
        """
        try:
            a = float(slope) if slope is not None else 0.0
        except Exception:
            a = 0.0

        try:
            auc_val = float(auc) if auc is not None else 0.5
        except Exception:
            auc_val = 0.5

        # Env-tunable thresholds, but with sane defaults
        try:
            ok_auc = float(os.getenv("MODEL_OK_AUC", "0.53"))
            ok_slope = float(os.getenv("MODEL_OK_SLOPE", "0.15"))
            strong_auc = float(os.getenv("MODEL_STRONG_AUC", "0.57"))
            strong_slope = float(os.getenv("MODEL_STRONG_SLOPE", "0.25"))
        except Exception:
            ok_auc, ok_slope = 0.53, 0.15
            strong_auc, strong_slope = 0.57, 0.25

        if auc_val >= strong_auc and a >= strong_slope:
            q = "strong"
        elif auc_val >= ok_auc and a >= ok_slope:
            q = "ok"
        else:
            q = "weak"

        if q != getattr(self, "model_quality", "unknown"):
            logger.info(
                "[CALIB-QUALITY] model_quality=%s (auc=%.3f, slope=%.3f)",
                q,
                auc_val,
                a,
            )
        self.model_quality = q




    def reload_calibration(self, path: Optional[str] = None) -> bool:
        try:
            if not path:
                path = os.getenv("CALIB_PATH", "").strip()
            if not path or not os.path.exists(path):
                logger.info(f"[CALIB] reload skipped; file missing: {path or '(empty)'}")
                return False
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            a = float(data.get("a"))
            b = float(data.get("b"))
            n = int(data.get("n", 0))

            def _mono_ok(a_: float, b_: float) -> bool:
                if not np.isfinite(a_) or not np.isfinite(b_) or a_ <= 0.0:
                    return False
                grid = np.linspace(0.05, 0.95, 19)
                lg = np.log(np.clip(grid, 1e-9, 1-1e-9) / (1 - np.clip(grid, 1e-9, 1-1e-9)))
                q = 1.0 / (1.0 + np.exp(-(a_ * lg + b_)))
                return bool(np.all(np.diff(q) > 0.0))
            if not _mono_ok(a, b):
                logger.info(f"[CALIB] reload rejected: non-monotonic or invalid a={a:.6f} b={b:.6f} (n={n})")
                return False

            min_a = float(os.getenv("CALIB_MIN_SLOPE", "0.2"))
            if abs(a) < min_a:
                logger.info(
                    "[CALIB] loaded weak slope a=%.4f (<%.2f) → enabling calib bypass",
                    a, min_a
                )
                self._calib_bypass = True
            else:
                self._calib_bypass = False

            self._calib_a, self._calib_b = a, b
            logger.info(f"[CALIB] reloaded: a={a:.6f} b={b:.6f} (n={n})")
            # update coarse model quality estimate from persisted auc (if present)
            try:
                auc = float(data.get("auc", 0.5))
            except Exception:
                auc = 0.5
            try:
                self._update_model_quality(auc=auc, slope=self._calib_a)
            except Exception:
                pass
            return True
        except Exception as e:
            logger.warning(f"[CALIB] reload failed: {e}")
            return False

    def reload_calibrator_if_exists(self):
        """Hot-reload Platt calibrator if background job writes it later."""
        import os
        try:
            path = getattr(self, "platt_path", None) or os.getenv("CALIB_PATH", "").strip()
            if not path or not os.path.exists(path):
                return False
            try:
                mtime = os.path.getmtime(path)
            except Exception:
                mtime = None

            if mtime is not None and getattr(self, "_calib_mtime", None) == mtime:
                return False

            ok = bool(self.reload_calibration(path))
            if ok:
                try:
                    self._calib_mtime = mtime or os.path.getmtime(path)
                except Exception:
                    self._calib_mtime = None
                self.calibrator_ready = True
            return ok
        except Exception:
            return False

    def set_feature_schema(self, names):
        try:
            self.feature_schema_names = list(names) if names else None
            self.feature_schema_size = len(self.feature_schema_names) if self.feature_schema_names else None
            logger.info(f"[SCHEMA] Feature schema set: n={self.feature_schema_size}")
        except Exception as e:
            logger.warning(f"[SCHEMA] Failed to set schema: {e}")

    def _align_features_to_schema(self, names, values):
        import numpy as np
        try:
            vals = np.asarray(values, dtype=float).ravel().tolist()
            if not self.feature_schema_names or not names:
                x = np.asarray(vals, dtype=float).reshape(1, -1)
                x[~np.isfinite(x)] = 0.0
                return x
            m = {}
            for n, v in zip(names, vals):
                try:
                    fv = float(v)
                    if np.isfinite(fv):
                        m[str(n)] = fv
                except Exception:
                    continue
            aligned = [m.get(s, 0.0) for s in self.feature_schema_names]
            x = np.asarray(aligned, dtype=float).reshape(1, -1)
            x[~np.isfinite(x)] = 0.0
            return x
        except Exception:
            x = np.asarray(values, dtype=float).reshape(1, -1)
            x[~np.isfinite(x)] = 0.0
            return x

    def compute_indicator_score(self, features: Dict[str, float]) -> float:
        """
        Lightweight indicator score with volatility-aware micro weighting.
        """
        try:
            score = 0.0
            vol_short = float(features.get("std_dltp_short", 0.0))
            tight = float(features.get("price_range_tightness", 1.0))
            if (vol_short > 0.0) and (tight < 0.98):
                micro_factor = 1.25
            elif (vol_short <= 0.0) or (tight >= 0.995):
                micro_factor = 0.60
            else:
                micro_factor = 1.0
            for key, base_weight in self.weights.items():
                val = float(features.get(key, 0.0))
                w = base_weight
                if key == "micro_slope":
                    tight_ctx = float(features.get("price_range_tightness", 1.0)) >= 0.995
                    val = float(np.tanh(val)) if tight_ctx else val
                    w = float(np.clip(base_weight * micro_factor, self.min_w, self.max_w))
                score += w * val
            return float(score)
        except Exception as e:
            logger.warning(f"Indicator score computation failed: {e}")
            return 0.0

    def _apply_indicator_modulation(
        self,
        p: float,
        indicator_score: float,
        mtf_consensus: float,
        neutral_prob: float,
        margin: float,
        engineered_features: Optional[Dict[str, float]] = None,
    ) -> Tuple[float, float]:
        """
        Indicator / MTF / futures inputs are kept as diagnostic features and
        for the Q-model, but they no longer directly modify the core XGB
        probability. This keeps live behaviour aligned with offline_eval_2min
        and offline_eval_2min_full, which both assume p_buy is the raw model
        output.
        """

        try:
            raw_p = float(p)
        except Exception:
            raw_p = 0.5
        p_clipped = max(0.0, min(1.0, raw_p))

        try:
            ind_val = float(indicator_score)
        except Exception:
            ind_val = 0.0
        mtf_val = float(mtf_consensus) if mtf_consensus is not None else 0.0
        fut_cvd = 0.0
        if engineered_features is not None:
            fut_cvd = float(engineered_features.get("cvd_divergence", 0.0))

        logger.debug(
            "[IND] modulation disabled → p=%.3f (indicator=%.3f, mtf=%.3f, fut_cvd=%.6f, neutral=%.3f)",
            p_clipped,
            ind_val,
            mtf_val,
            fut_cvd,
            float(neutral_prob),
        )

        # No modulation: return probability and neutral as-is
        return p_clipped, neutral_prob

    def predict(
        self,
        live_tensor: np.ndarray,
        engineered_features: list,
        indicator_score: Optional[float] = None,
        pattern_prob_adjustment: Optional[float] = None,
        engineered_feature_names: Optional[list] = None,
        mtf_consensus: Optional[float] = None
    ) -> Tuple[np.ndarray, Optional[float]]:
        """
        Run the full inference stack for a single candle.

        Pipeline:
          1) Align features to the booster schema (guards NaN/Inf).
          2) XGB predict_proba -> (optional) calibration.
          3) Indicator/MTF modulation (currently logging-only).
          4) Pattern adjustment (logging-only).
          5) Return final [[p_sell, p_buy]] and optional neutrality prob.

        Returns:
          - signal_probs: np.array([[p_sell, p_buy]])
          - neutral_prob (float in [0,1]) or None
        """
        try:
            try:
                self.reload_calibrator_if_exists()
            except Exception:
                pass
            # Align features to schema
            try:
                xgb_input = self._align_features_to_schema(engineered_feature_names, engineered_features)
                if np.isnan(xgb_input).any() or np.isinf(xgb_input).any():
                    xgb_input = np.nan_to_num(xgb_input, nan=0.0, posinf=0.0, neginf=0.0)
                    logger.debug("[SCHEMA] NaN/Inf detected in aligned vector; sanitized to 0.0")
                logger.debug(f"[SCHEMA] Inference vector shaped to {xgb_input.shape} (schema_n={self.feature_schema_size})")
            except Exception as e:
                logger.error(f"Schema alignment failed: {e}", exc_info=True)
                xgb_input = np.asarray(engineered_features, dtype=float).reshape(1, -1)

            # XGB predict
            try:
                signal_probs = self.xgb.predict_proba(xgb_input)
                if not isinstance(signal_probs, np.ndarray):
                    signal_probs = np.array(signal_probs, dtype=float)
                if signal_probs.ndim != 2 or signal_probs.shape[1] != 2:
                    logger.warning(f"Unexpected XGB output shape: {signal_probs.shape}")
                    signal_probs = np.array([[0.5, 0.5]], dtype=float)
                try:
                    raw_buy = float(signal_probs[0][1])
                    self.last_p_xgb_raw = float(np.clip(raw_buy, 1e-9, 1 - 1e-9))
                except Exception:
                    self.last_p_xgb_raw = None
                try:
                    if self.last_p_xgb_raw is not None:
                        cal_buy = self._apply_calibration(self.last_p_xgb_raw)
                        self.last_p_xgb_calib = float(np.clip(cal_buy, 1e-6, 1.0 - 1e-6))
    
                        if abs(self.last_p_xgb_calib - self.last_p_xgb_raw) > 1e-12:
                            signal_probs = np.array([[1.0 - self.last_p_xgb_calib, self.last_p_xgb_calib]], dtype=float)
                            logger.debug("[CALIB] p_buy raw=%.3f → calib=%.3f",self.last_p_xgb_raw,self.last_p_xgb_calib,)
                    else:
                        self.last_p_xgb_calib = None
                except Exception as e:
                    logger.debug(f"[CALIB] Skipped: {e}")
                    self.last_p_xgb_calib = None
            except Exception as e:
                logger.error(f"XGB prediction failed: {e}")
                signal_probs = np.array([[0.5, 0.5]], dtype=float)
                self.last_p_xgb_raw = None
                self.last_p_xgb_calib = None

            # Build engineered feature map for downstream lookups (futures, etc.)
            engineered_feature_map: Dict[str, float] = {}
            try:
                if engineered_feature_names and engineered_features:
                    engineered_feature_map = {
                        str(k): float(v)
                        for k, v in zip(engineered_feature_names, engineered_features)
                    }
            except Exception:
                engineered_feature_map = {}

            # Neutrality (optional)
            neutral_prob = None
            try:
                if self.neutral_model is not None:
                    if hasattr(self.neutral_model, "predict_proba"):
                        np.seterr(all='ignore')
                        p = self.neutral_model.predict_proba(xgb_input)
                        p = np.asarray(p, dtype=float)
                        if p.ndim == 2 and p.shape[1] >= 2:
                            neutral_prob = float(p[0, 1])
                        else:
                            neutral_prob = float(p.ravel()[0])
                    elif hasattr(self.neutral_model, "predict"):
                        raw = float(self.neutral_model.predict(xgb_input).ravel()[0])
                        neutral_prob = 1.0 / (1.0 + np.exp(-raw))
            except Exception as e:
                logger.debug(f"Neutrality model inference failed: {e}")
                neutral_prob = None

            # Indicator / MTF modulation (with futures CVD awareness)
            p_model = float(signal_probs[0][1])
            p_after_indicator = p_model
            try:
                margin = abs(p_model - 0.5)
                mtf_val = float(mtf_consensus) if mtf_consensus is not None else 0.0
                ind_val = float(indicator_score) if indicator_score is not None else 0.0
                neutral_in = float(neutral_prob) if neutral_prob is not None else 0.0
                p_after_indicator, neutral_prob = self._apply_indicator_modulation(
                    p=p_model,
                    indicator_score=ind_val,
                    mtf_consensus=mtf_val,
                    neutral_prob=neutral_in,
                    margin=margin,
                    engineered_features=engineered_feature_map,
                )
                p_after_indicator = float(np.clip(p_after_indicator, 1e-4, 1 - 1e-4))
                signal_probs = np.array(
                    [[1.0 - p_after_indicator, p_after_indicator]],
                    dtype=float,
                )
            except Exception as e:
                logger.warning(f"Indicator modulation failed: {e}")



            # Pattern adjustment (logging only; no probability modulation)
            p_after_pattern = float(signal_probs[0][1])
            try:
                if pattern_prob_adjustment is not None:
                    adj_in = float(pattern_prob_adjustment)
                    logger.debug(
                        "[PATTERN] modulation disabled → adj_in=%+.3f, p stays %.3f",
                        adj_in,
                        p_after_pattern,
                    )
                # signal_probs left unchanged
            except Exception as e:
                logger.warning(f"Pattern adjustment logging failed: {e}")



            p_final = float(signal_probs[0][1])
            # Final log (no weak-model hard clip; p_final reflects indicator + pattern)
            logger.debug(
                "[PRED] p_model=%.3f p_after_indicator=%.3f p_after_pattern=%.3f p_final=%.3f neutral_prob=%s",
                p_model,
                p_after_indicator,
                p_after_pattern,
                p_final,
                str(neutral_prob) if neutral_prob is not None else "na",
            )
       
                   
            return signal_probs, neutral_prob
        except Exception as e:
            logger.error(f"Prediction pipeline failed: {e}", exc_info=True)
            return np.array([[0.5, 0.5]], dtype=float), None


    def replace_models(self, xgb=None, neutral=None):
        """
        Hot-reload models. Schema source of truth: embedded in booster only.
        Disk fallback is enabled via FEATURE_SCHEMA_COLS_PATH / feature_schema_cols.json next to XGB_PATH.
        """
        try:
            if xgb is not None:
                self.xgb = xgb
                logger.info("[MODELS] XGB model hot-reloaded")
                
                # Always load schema from booster (source of truth)
                try:
                    if hasattr(self.xgb, "booster") and hasattr(self.xgb.booster, "attr"):
                        meta = self.xgb.booster.attr("feature_schema")
                        if meta:
                            try:
                                data = json.loads(meta)
                                names = data.get("feature_names") or data.get("features")
                                if names:
                                    self.set_feature_schema(names)
                                    logger.info(f"[SCHEMA] Loaded from booster.attr('feature_schema'): n={len(names)}")
                                else:
                                    logger.warning("[SCHEMA] Booster has feature_schema attr but no feature_names found")
                            except json.JSONDecodeError as e:
                                logger.warning(f"[SCHEMA] Failed to parse booster feature_schema attr: {e}")
                        else:
                            logger.info("[SCHEMA] Booster has no embedded feature_schema; current schema retained")
                        # Disk fallback (explicit + co-located schema files)
                        if not self.feature_schema_names:
                            names = _load_schema_from_disk()
                            if names:
                                self.set_feature_schema(names)
                                logger.info(f"[SCHEMA] Loaded from disk fallback: n={len(names)}")
                                nfeat = _safe_booster_num_features(self.xgb)
                                if nfeat is not None and self.feature_schema_names and len(self.feature_schema_names) != nfeat:
                                    self._xgb_disabled_reason = f"schema_n={len(self.feature_schema_names)} != booster_n={nfeat} (set XGB_PATH/FEATURE_SCHEMA_COLS_PATH to matching artifacts)"
                                    logger.error(f"[SCHEMA] {self._xgb_disabled_reason}")

                    else:
                        logger.debug("[SCHEMA] Booster does not support attr method; schema unchanged")
                except Exception as e:
                    logger.warning(f"[SCHEMA] Booster schema load failed: {e}")

            if neutral is not None:
                self.neutral_model = neutral
                logger.info("[MODELS] Neutrality model hot-reloaded")
                
        except Exception as e:
            logger.error(f"Model hot-reload failed: {e}", exc_info=True)



def create_default_pipeline(xgb, neutral_model=None, **kwargs) -> AdaptiveModelPipeline:
    defaults = {
        'base_weights': {
            'ema_trend': 0.35,
            'micro_slope': 0.25,
            'imbalance': 0.20,
            'mean_drift': 0.20
        }
    }
    defaults.update(kwargs)
    pipe = AdaptiveModelPipeline(xgb, neutral_model=neutral_model, **defaults)
    try:
        names = _load_schema_from_disk()
        if names:
            pipe.set_feature_schema(names)
            nfeat = _safe_booster_num_features(pipe.xgb)
            if nfeat is not None and pipe.feature_schema_names and len(pipe.feature_schema_names) != nfeat:
                pipe._xgb_disabled_reason = f"schema_n={len(pipe.feature_schema_names)} != booster_n={nfeat} (set XGB_PATH/FEATURE_SCHEMA_COLS_PATH to matching artifacts)"
                logger.error(f"[SCHEMA] {pipe._xgb_disabled_reason}")
    except Exception as e:
        logger.warning(f"[SCHEMA] Failed to apply disk schema: {e}")
    return pipe