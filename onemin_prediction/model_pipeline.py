# model_pipeline.py
"""
Adaptive model pipeline (probabilities-only).
- XGB probabilities (+ optional Platt calibration)
- Indicator/pattern/MTF conservative modulation
- Optional neutrality probability
"""
import json
import numpy as np
from typing import Dict, Optional, Tuple
import logging
import os
from logging_setup import log_every

logger = logging.getLogger(__name__)

class AdaptiveModelPipeline:
    def __init__(
        self,
        cnn_lstm,
        xgb,
        base_weights: Optional[Dict[str, float]] = None,
        neutral_model: Optional[object] = None
    ):
        self.cnn_lstm = cnn_lstm
        self.xgb = xgb
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

        try:
            calib_path = os.getenv("CALIB_PATH", "").strip()
            if calib_path and os.path.exists(calib_path):
                self.reload_calibration(calib_path)
            else:
                logger.info("[CALIB] No calibration file; raw XGB probabilities will be used")
        except Exception as e:
            logger.warning(f"[CALIB] Failed to load calibration: {e}")

        logger.info("Adaptive model pipeline initialized (probabilities-only)")





    def _apply_calibration(self, p: float) -> float:
        """
        Apply Platt calibration to a raw BUY probability.

        Behaviour:
        - If _calib_bypass is True, we never trust raw margins fully:
          we shrink toward 0.5 so extreme values are softened.
        - Otherwise, use current (a,b), with guards against
          non-monotonic or extreme inversions.
        """
        try:
            # Respect dynamic bypass flag
            if getattr(self, "_calib_bypass", False):
                # When bypassed, never trust raw margins fully.
                # Shrink towards 0.5 so 0.9 behaves more like ~0.66–0.7.
                try:
                    shrink_env = os.getenv("CALIB_BYPASS_SHRINK", "0.4") or "0.4"
                    shrink = float(shrink_env)
                except Exception:
                    shrink = 0.4
                shrink = max(0.1, min(0.8, shrink))
                p_raw = float(np.clip(p, 1e-9, 1.0 - 1e-9))
                p_shrunk = 0.5 + shrink * (p_raw - 0.5)
                p_shrunk = float(np.clip(p_shrunk, 1e-6, 1.0 - 1e-6))
                logger.info(
                    "[CALIB] bypass active → shrink raw p=%.3f to %.3f (shrink=%.2f)",
                    p_raw, p_shrunk, shrink
                )
                return p_shrunk

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
        except Exception:
            # On any error, fall back to raw
            return float(p)




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
            self._calib_a, self._calib_b = a, b
            logger.info(f"[CALIB] reloaded: a={a:.6f} b={b:.6f} (n={n})")
            return True
        except Exception as e:
            logger.warning(f"[CALIB] reload failed: {e}")
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
        Returns:
          - signal_probs: np.array([[p_sell, p_buy]])
          - neutral_prob (float in [0,1]) or None
        """
        try:
            # latent (optional, for logging/debug)
            try:
                latent_features = self.cnn_lstm.predict(live_tensor)
                if not isinstance(latent_features, np.ndarray):
                    latent_features = np.array(latent_features, dtype=float)
                latent_features = np.atleast_1d(latent_features).ravel()
                logger.debug(f"[LATENT] shape={getattr(latent_features, 'shape', None)} "
                             f"first3={latent_features[:3].tolist() if latent_features.size >= 3 else latent_features.tolist()}")
            except Exception as e:
                logger.debug(f"CNN-LSTM latent skipped: {e}")

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

            # Conservative indicator modulation
            p_model = float(signal_probs[0][1])
            p_after_indicator = p_model
            if indicator_score is not None:
                try:
                    indicator_norm = 0.5 + 0.5 * np.tanh(indicator_score)
                    margin = abs(p_model - 0.5)
                    agree = (np.sign(indicator_norm - 0.5) == np.sign(p_model - 0.5)) and (np.sign(p_model - 0.5) != 0)
                    strong_cons = (mtf_consensus is not None) and (abs(float(mtf_consensus)) >= 0.60)
                    cap = 0.04 if strong_cons else 0.02
                    if margin <= 0.12 and agree:
                        blended = 0.9 * p_model + 0.1 * indicator_norm
                        delta = float(np.clip(blended - p_model, -cap, cap))
                        p_after_indicator = float(np.clip(p_model + delta, 0.0, 1.0))
                        signal_probs = np.array([[1.0 - p_after_indicator, p_after_indicator]], dtype=float)
                        logger.debug("[BLEND] gated: p_model=%.3f indicator=%.3f → p_after_indicator=%.3f (Δ=%+.3f, cap=%.3f)",p_model, indicator_norm, p_after_indicator, delta, cap)
                    else:
                      log_every("blend-skip",10,logger.debug,"[BLEND] skipped: margin=%.3f agree=%s p_model=%.3f indicator=%.3f",margin, agree, p_model, indicator_norm)
                                              

                except Exception as e:
                    logger.warning(f"Indicator modulation failed: {e}")



            # Pattern adjustment (adaptive, neutral-aware and conflict-aware)
            p_after_pattern = float(signal_probs[0][1])
            try:
                if pattern_prob_adjustment is not None:
                    adj_in = float(pattern_prob_adjustment)
                    p_cur = float(signal_probs[0][1])

                    ind = float(indicator_score) if indicator_score is not None else 0.0
                    ind_abs = abs(ind)
                    strong_cons = (mtf_consensus is not None) and (abs(float(mtf_consensus)) >= 0.60)
                    sign_pat = np.sign(adj_in)
                    sign_ind = np.sign(ind)
                    sign_mtf = np.sign(float(mtf_consensus)) if (mtf_consensus is not None) else 0.0

                    base_cap = 0.10 if strong_cons else 0.08

                    # Neutral-aware shrink
                    neu = float(engineered_features[engineered_feature_names.index("neutral_prob")] if "neutral_prob" in engineered_feature_names else 0.5)
                    def _smoothstep(x, x0=0.55, x1=0.90):
                        if x <= x0: return 0.0
                        if x >= x1: return 1.0
                        t = (x - x0) / max(1e-9, (x1 - x0))
                        return t * t * (3 - 2 * t)
                    s_neu = _smoothstep(neu, 0.55, 0.90)
                    cap_ctx = base_cap * (1.0 - 0.65 * s_neu)  # up to -65% at high neutral

                    # Conflict-aware shrink (unless strong consensus)
                    conflict = ((sign_pat != 0) and ((sign_ind != 0 and sign_pat != sign_ind) or
                                                    (sign_mtf != 0 and sign_pat != sign_mtf)))
                    if conflict and not strong_cons:
                        cap_ctx *= 0.5

                    # Weak-indicator shrink (keep original behavior)
                    if (ind_abs < 0.25) and not strong_cons:
                        cap_ctx = min(cap_ctx, 0.02)

                    # Clip incoming adjustment to base safety, then to context cap
                    adj = float(np.clip(adj_in, -base_cap, base_cap))
                    
                    
                    if abs(adj) > cap_ctx:
        
                        log_every("pattern-cap2",10,logger.debug,"[PATTERN] ctx-cap applied: %+.3f → %+.3f (neu=%.3f, ind_abs=%.3f, strong_cons=%s)",adj,float(np.sign(adj) * cap_ctx),neu,ind_abs,strong_cons,)                        

                        
                        
                        adj = float(np.sign(adj) * cap_ctx)

                    # Flip rule: block flips in high-neutral contexts unless consensus is strong and aligned
                    would_flip = ((p_cur - 0.5) * ((p_cur + adj) - 0.5)) < 0
                    aligned_consensus = (strong_cons and
                                        (sign_mtf == np.sign(p_cur - 0.5)) and
                                        (sign_ind == np.sign(p_cur - 0.5)))
                    if would_flip and (neu >= 0.70) and (not aligned_consensus):
                        room = max(1e-3, abs(p_cur - 0.5) - 1e-3)
                        adj = float(np.sign(adj) * min(abs(adj), room))
                           
                        logger.debug("[PATTERN] flip blocked by high-neutral ctx: adj→%+.3f (p_cur=%.3f, neu=%.3f, strong_cons=%s)",adj, p_cur, neu, strong_cons)
                                          
                                          

                    p_after_pattern = float(np.clip(p_cur + adj, 0.0, 1.0))
                    signal_probs = np.array([[1.0 - p_after_pattern, p_after_pattern]], dtype=float)
                     
                    logger.debug("[PATTERN] adj_in=%+.3f → adj=%+.3f | p: %.3f → %.3f",adj_in, adj, p_cur, p_after_pattern)
                                  
            except Exception as e:
                logger.warning(f"Pattern adjustment failed: {e}")



            # Final log
            logger.debug("[PRED] p_model=%.3f p_after_indicator=%.3f p_after_pattern=%.3f neutral_prob=%s schema_n=%s",
                p_model,
                p_after_indicator,
                p_after_pattern,
                str(neutral_prob) if neutral_prob is not None else "na",
                str(self.feature_schema_size),
            )
       
                   
            return signal_probs, neutral_prob
        except Exception as e:
            logger.error(f"Prediction pipeline failed: {e}", exc_info=True)
            return np.array([[0.5, 0.5]], dtype=float), None


    def replace_models(self, xgb=None, neutral=None):
        """
        Hot-reload models. Schema source of truth: embedded in booster only.
        Disk fallback is removed to prevent stale schema issues.
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
                    else:
                        logger.debug("[SCHEMA] Booster does not support attr method; schema unchanged")
                except Exception as e:
                    logger.warning(f"[SCHEMA] Booster schema load failed: {e}")

            if neutral is not None:
                self.neutral_model = neutral
                logger.info("[MODELS] Neutrality model hot-reloaded")
                
        except Exception as e:
            logger.error(f"Model hot-reload failed: {e}", exc_info=True)



def create_default_pipeline(cnn_lstm, xgb, neutral_model=None, **kwargs) -> AdaptiveModelPipeline:
    defaults = {
        'base_weights': {
            'ema_trend': 0.35,
            'micro_slope': 0.25,
            'imbalance': 0.20,
            'mean_drift': 0.20
        }
    }
    defaults.update(kwargs)
    return AdaptiveModelPipeline(cnn_lstm, xgb, neutral_model=neutral_model, **defaults)
