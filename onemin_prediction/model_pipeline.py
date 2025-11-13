"""
Unified model pipeline with adaptive weight tuning.
Consolidates: hybrid_model.py, indicator_weight_tuner.py
Eliminates redundant file separation for cleaner architecture.
"""
import asyncio
import json
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple
import logging
import os
logger = logging.getLogger(__name__)


class AdaptiveModelPipeline:
    """
    Unified hybrid model with integrated adaptive weight tuning.
    
    Features:
    - CNN-LSTM + XGBoost ensemble predictions
    - RL-based confidence adjustment
    - Proactive indicator weight learning from hitrate logs
    - Volume-free, ATR-free design
    - Toggleable rule-based blending
    
    Eliminates separate IndicatorWeightTuner class for better cohesion.
    """
    
    
    def __init__(
        self, 
        cnn_lstm, 
        xgb, 
        rl_agent, 
        base_weights: Optional[Dict[str, float]] = None,
        base_alpha_buy: float = 0.6, 
        lr: float = 0.05, 
        ema_anchor: float = 0.35, 
        min_w: float = 0.05, 
        max_w: float = 0.80,
        neutral_model: Optional[object] = None  # NEW

    ):

    
        """
        Initialize adaptive model pipeline.
        
        Args:
            cnn_lstm: Trained CNN-LSTM model with predict() method
            xgb: Trained XGBoost model with predict_proba() method
            rl_agent: RL agent with adjust_confidence() method and threshold attribute
            base_weights: Initial indicator weights (default: balanced)
            base_alpha_buy: Base buy confidence threshold
            lr: Learning rate for weight updates
            ema_anchor: Fixed weight for EMA trend (not tuned)
            min_w: Minimum allowed weight
            max_w: Maximum allowed weight
        """
        self.cnn_lstm = cnn_lstm
        self.xgb = xgb
        self.rl_agent = rl_agent
        self.base_alpha_buy = float(base_alpha_buy)
        
        # Initialize weights with defaults if not provided
        if base_weights is None:

            base_weights = {
                'ema_trend': 0.35,
                'micro_slope': 0.25,  # reduced default
                'imbalance': 0.20,
                'mean_drift': 0.20
            }

        
        # Integrated weight tuning parameters
        self.weights = dict(base_weights)
        self.lr = float(lr)
        self.ema_anchor = float(ema_anchor)
        self.min_w = float(min_w)
        self.max_w = float(max_w)
        self.tunable_keys = ['micro_slope', 'imbalance', 'mean_drift']
        
        # Validation
        self._validate_weights()
        
        
        
                
        # Feature schema tracking
        self.feature_schema_names = None
        self.feature_schema_size = None

        
        logger.info("Adaptive model pipeline initialized")
        logger.info(f"Base weights: {self.weights}")
        logger.info(f"Tunable indicators: {self.tunable_keys}")

        # Neutrality model initialization
        self.neutral_model = neutral_model
        logger.info(f"Neutrality model: {'present' if self.neutral_model is not None else 'absent'}")



        # Optional probability calibration (Platt)
        self._calib_a = None
        self._calib_b = None
        self.last_p_xgb_raw = None  # NEW: last raw XGB p (pre-calibration)
        self.last_p_xgb_calib = None  # NEW: last calibrated p (pre-blend)

                

        try:
            calib_path = os.getenv("CALIB_PATH", "").strip()
            if calib_path and os.path.exists(calib_path):
                self.reload_calibration(calib_path)
            else:
                logger.info("[CALIB] No calibration file; raw XGB probabilities will be used")
        except Exception as e:
            logger.warning(f"[CALIB] Failed to load calibration: {e}")

        logger.info(f"Learning rate: {self.lr}, EMA anchor: {self.ema_anchor}")





    def _apply_calibration(self, p: float) -> float:
        """
        Apply optional Platt calibration in logit space: q = sigmoid(a*logit(p)+b).
        Guardrails:
        - Reject invalid/negative slope
        - Reject large side-flipping inversions across 0.5
        - Clamp excessive same-side deltas, strongest near 0.5
        """
        try:
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

            # Guard 1: large inversion across 0.5
            if (p - 0.5) * (q - 0.5) < 0 and abs(q - p) > 0.20:
                logger.info(f"[CALIB-GUARD] large inversion Δ={q - p:+.3f} (p={p:.3f} → q={q:.3f}) → using raw")
                return float(p)

            # Guard 2: clamp excessive same-side delta (stop 0.405→0.074 type collapses)
            # 0.12 max delta at center, increasing slightly towards edges
            max_delta = 0.12 + 0.25 * abs(p - 0.5)  # in [0.12, 0.245]
            delta = q - p
            if abs(delta) > max_delta:
                q_clamped = float(p + np.clip(delta, -max_delta, max_delta))
                logger.info(f"[CALIB-GUARD] clamp Δ={delta:+.3f}→{(q_clamped - p):+.3f} (p={p:.3f} → q={q:.3f})")
                return q_clamped

            return q
        except Exception:
            return float(p)



    def reload_calibration(self, path: Optional[str] = None) -> bool:
        """
        Reload Platt calibration coefficients from JSON file and apply live.
        Rejects non-positive slope or non-monotonic mappings.
        """
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
            
            # simple monotonic grid test
            def _mono_ok(a_: float, b_: float) -> bool:
                if not np.isfinite(a_) or not np.isfinite(b_) or a_ <= 0.0:
                    return False
                grid = np.linspace(0.05, 0.95, 19)
                lg = np.log(np.clip(grid, 1e-9, 1-1e-9) / (1-np.clip(grid, 1e-9, 1-1e-9)))
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



    def disable_neutrality(self, reason: str = "") -> None:
        """Disable neutrality gating for this session with a visible reason."""
        try:
            self.neutral_model = None
            if reason:
                logger.info(f"[NEUTRAL] gating disabled: {reason}")
            else:
                logger.info("[NEUTRAL] gating disabled")
        except Exception:
            pass



    def _validate_weights(self):
        """Validate weight configuration."""
        total = sum(self.weights.values())
        if not (0.95 <= total <= 1.05):
            logger.warning(f"Weights sum to {total:.3f}, normalizing to 1.0")
            norm = 1.0 / max(1e-9, total)
            self.weights = {k: v * norm for k, v in self.weights.items()}



    def set_feature_schema(self, names):
        """Set feature schema for alignment during inference."""
        try:
            self.feature_schema_names = list(names) if names else None
            self.feature_schema_size = len(self.feature_schema_names) if self.feature_schema_names else None
            logger.info(f"[SCHEMA] Feature schema set: n={self.feature_schema_size}")
        except Exception as e:
            logger.warning(f"[SCHEMA] Failed to set schema: {e}")

    def _align_features_to_schema(self, names, values):
        """
        Map (names, values) to the persisted schema order, dropping unknowns and filling missing with 0.0.
        """
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




    def predict(
        self, 
        live_tensor: np.ndarray, 
        engineered_features: list, 
        recent_profit_factor: float,
        indicator_score: Optional[float] = None,
        pattern_prob_adjustment: Optional[float] = None,
        engineered_feature_names: Optional[list] = None,
        mtf_consensus: Optional[float] = None  # NEW
    ) -> Tuple[np.ndarray, float, Optional[float]]:


        """
        Generate ensemble prediction with optional indicator modulation.
        
        Returns:
            (signal_probs, adjusted_alpha_buy, neutral_prob)
            - signal_probs: np.array([[p_sell, p_buy]])
            - adjusted_alpha_buy: RL-adjusted confidence threshold (float)
            - neutral_prob: probability of flat/neutral (float in [0,1]) or None
        """


        try:
            # ========== CNN-LSTM LATENT FEATURES (not used for XGB input) ==========
            try:
                latent_features = self.cnn_lstm.predict(live_tensor)
                if not isinstance(latent_features, np.ndarray):
                    latent_features = np.array(latent_features, dtype=float)
                latent_features = np.atleast_1d(latent_features).ravel()
                logger.debug(f"[LATENT] shape={getattr(latent_features, 'shape', None)} "
                        f"first3={latent_features[:3].tolist() if latent_features.size >= 3 else latent_features.tolist()}")
                
            except Exception as e:
                logger.warning(f"CNN-LSTM prediction failed: {e}")
                latent_features = np.zeros(8, dtype=float)

            # ========== ENGINEERED FEATURES ==========
            try:
                ef_vals = np.asarray(engineered_features, dtype=float).ravel().tolist()
            except Exception as e:
                logger.warning(f"Feature engineering failed: {e}")
                ef_vals = [0.0]

            # ========== ALIGN TO SCHEMA & XGB INPUT (NO LATENT CONCAT) ==========
            try:
                xgb_input = self._align_features_to_schema(engineered_feature_names, ef_vals)
                
                
                try:
                    if np.isnan(xgb_input).any() or np.isinf(xgb_input).any():
                        xgb_input = np.nan_to_num(xgb_input, nan=0.0, posinf=0.0, neginf=0.0)
                        logger.debug("[SCHEMA] NaN/Inf detected in aligned vector; sanitized to 0.0")
                except Exception:
                    pass
            
                logger.debug(f"[SCHEMA] Inference vector shaped to {xgb_input.shape} "
                            f"(schema_n={self.feature_schema_size})")
            except Exception as e:
                logger.error(f"Schema alignment failed: {e}", exc_info=True)
                xgb_input = np.asarray(ef_vals, dtype=float).reshape(1, -1)

            # ========== XGB PREDICTION ==========
            try:
                signal_probs = self.xgb.predict_proba(xgb_input)
                
                if not isinstance(signal_probs, np.ndarray):
                    signal_probs = np.array(signal_probs, dtype=float)
                if signal_probs.ndim != 2 or signal_probs.shape[1] != 2:
                    logger.warning(f"Unexpected XGB output shape: {signal_probs.shape}")
                    signal_probs = np.array([[0.5, 0.5]], dtype=float)
                
                # Capture raw p before calibration
                try:
                    raw_buy = float(signal_probs[0][1])
                    self.last_p_xgb_raw = float(np.clip(raw_buy, 1e-9, 1 - 1e-9))
                except Exception:
                    self.last_p_xgb_raw = None
                
                # ========== OPTIONAL PROBABILITY CALIBRATION ==========
                try:
                    if self.last_p_xgb_raw is not None:
                        cal_buy = self._apply_calibration(self.last_p_xgb_raw)
                        self.last_p_xgb_calib = float(np.clip(cal_buy, 1e-6, 1.0 - 1e-6))
                        if abs(self.last_p_xgb_calib - self.last_p_xgb_raw) > 1e-12:
                            signal_probs = np.array([[1.0 - self.last_p_xgb_calib, self.last_p_xgb_calib]], dtype=float)
                            logger.info(f"[CALIB] p_buy raw={self.last_p_xgb_raw:.3f} → calib={self.last_p_xgb_calib:.3f}")
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

            
        


            # ========== NEUTRALITY MODEL (OPTIONAL) ==========
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
                # Detect common schema mismatches and disable permanently to cut noise
                if "expecting" in str(e).lower() or "features" in str(e).lower():
                    self.disable_neutrality(f"schema mismatch detected ({e})")
                else:
                    logger.debug(f"Neutrality model inference failed: {e}")
                neutral_prob = None




            # ========== INDICATOR MODULATION (OPTIONAL) ==========
            p_model = float(signal_probs[0][1])
            p_after_indicator = p_model

            if indicator_score is not None:
                try:
                    indicator_norm = 0.5 + 0.5 * np.tanh(indicator_score)
                    margin = abs(p_model - 0.5)
                    agree = (np.sign(indicator_norm - 0.5) == np.sign(p_model - 0.5)) and (np.sign(p_model - 0.5) != 0)
                    
                    # NEW: adaptive cap if MTF consensus is strong
                    strong_cons = (mtf_consensus is not None) and (abs(float(mtf_consensus)) >= 0.66)
                    cap = 0.04 if strong_cons else 0.02
                    
                    if margin <= 0.05 and agree:
                        blended = 0.9 * p_model + 0.1 * indicator_norm
                        delta = float(np.clip(blended - p_model, -cap, cap))
                        p_after_indicator = float(np.clip(p_model + delta, 0.0, 1.0))
                        signal_probs = np.array([[1.0 - p_after_indicator, p_after_indicator]], dtype=float)
                        logger.info(f"[BLEND] gated: p_model={p_model:.3f} indicator={indicator_norm:.3f} "
                                    f"→ p_after_indicator={p_after_indicator:.3f} (Δ={delta:+.3f}, cap={cap:.3f}) "
                                    f"mtf={mtf_consensus if mtf_consensus is not None else 'na'} strong_cons={strong_cons}")
                    else:
                        logger.info(f"[BLEND] skipped: margin={margin:.3f} agree={agree} "
                                    f"p_model={p_model:.3f} indicator={indicator_norm:.3f} "
                                    f"mtf={mtf_consensus if mtf_consensus is not None else 'na'}")
                except Exception as e:
                    logger.warning(f"Indicator modulation failed: {e}")



            # ========== PATTERN PROBABILITY ADJUSTMENT (OPTIONAL) ==========
            p_after_pattern = float(signal_probs[0][1])
            try:
                if pattern_prob_adjustment is not None:
                    adj_in = float(pattern_prob_adjustment)
                    adj = float(np.clip(adj_in, -0.08, 0.08))
                    p_cur = float(signal_probs[0][1])

                    # Context: consensus strength and indicator magnitude
                    strong_cons = (mtf_consensus is not None) and (abs(float(mtf_consensus)) >= 0.66)
                    ind_abs = abs(float(indicator_score)) if indicator_score is not None else 0.0

                    # Weak-context cap (indicator weak and no strong consensus)
                    cap = 0.02 if (ind_abs < 0.25 and not strong_cons) else 0.08
                    if abs(adj) > cap:
                        logger.info(f"[PATTERN] weak-indicator cap applied: {adj:+.3f} → {np.sign(adj)*cap:+.3f} "
                                    f"(ind={ind_abs:.3f}, strong_cons={strong_cons})")
                        adj = float(np.sign(adj) * cap)

                    # Flip-prevent near 0.5 in weak context
                    would_flip = ((p_cur - 0.5) * ((p_cur + adj) - 0.5)) < 0
                    if would_flip and (ind_abs < 0.30) and (not strong_cons):
                        room = max(1e-3, abs(p_cur - 0.5) - 1e-3)
                        adj = float(np.sign(adj) * min(abs(adj), room))
                        logger.info(f"[PATTERN] flip-prevent near 0.5 (weak ctx): adj→{adj:+.3f} (p_cur={p_cur:.3f})")

                    p_after_pattern = float(np.clip(p_cur + adj, 0.0, 1.0))
                    signal_probs = np.array([[1.0 - p_after_pattern, p_after_pattern]], dtype=float)
                    logger.info(f"[PATTERN] adj_in={adj_in:+.3f} → adj={adj:+.3f} | p: {p_cur:.3f} → {p_after_pattern:.3f}")
            except Exception as e:
                logger.warning(f"Pattern adjustment failed: {e}")




                        
            # ========== RL + NEUTRALITY-BASED CONFIDENCE ADJUSTMENT ==========
            try:
                rl_alpha = float(self.rl_agent.adjust_confidence(recent_profit_factor))
            except Exception as e:
                logger.warning(f"RL adjustment failed: {e}")
                rl_alpha = self.base_alpha_buy

            adjusted_alpha = rl_alpha  # pre-neutrality RL alpha (what we actually start from)

            # Neutrality gating (optional)
            try:
                if neutral_prob is not None and np.isfinite(neutral_prob):
                    if neutral_prob <= 0.35 or neutral_prob >= 0.65:
                        delta_raw = (neutral_prob - 0.5) * 0.04
                        delta = float(np.clip(delta_raw, -0.02, 0.02))
                    else:
                        delta = 0.0
                    pre = adjusted_alpha
                    adjusted_alpha = float(np.clip(pre + delta, 0.50, 0.80))
                    logger.info(f"[NEUTRAL] gated: p_neutral={neutral_prob:.3f} | alpha {pre:.3f} → {adjusted_alpha:.3f} (Δ={delta:+.3f})")
            except Exception as e:
                logger.debug(f"Neutrality alpha adjust failed: {e}")

            # ========== FINAL PREDICTION LOG ==========
            logger.info(
                f"[PRED] p_model={p_model:.3f} p_after_indicator={p_after_indicator:.3f} "
                f"p_after_pattern={p_after_pattern:.3f} | alpha_rl={rl_alpha:.3f} "
                f"alpha_final={adjusted_alpha:.3f} neutral_prob={neutral_prob if neutral_prob is not None else 'na'} "
                f"recentPF={float(recent_profit_factor):.3f} schema_n={self.feature_schema_size}"
            )
            
            return signal_probs, adjusted_alpha, neutral_prob
 

        except Exception as e:
            logger.error(f"Prediction pipeline failed: {e}", exc_info=True)
            
            return np.array([[0.5, 0.5]], dtype=float), self.base_alpha_buy, None





    # ========== INTEGRATED WEIGHT TUNING ==========
        
    async def update_weights_from_file(self, hitrate_path: str) -> Optional[Dict[str, float]]:
        """
        Update indicator weights from hitrate log file (proactive learning).
        Returns updated weights dict when a significant update occurs; otherwise None.
        """
        try:
            path = Path(hitrate_path)
            if not path.exists():
                logger.debug(f"Hitrate file not found: {hitrate_path}")
                return None

            # Read and parse JSONL
            records = []
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

            if not records:
                logger.debug("No valid records in hitrate file")
                return None

            # Aggregate hits/misses per indicator
            stats = {k: {'hits': 0, 'total': 0} for k in self.tunable_keys}
            for rec in records:
                ind = rec.get('indicator')
                if ind not in self.tunable_keys:
                    continue
                hit = int(rec.get('hit', 0))
                miss = int(rec.get('miss', 0))
                stats[ind]['hits'] += hit
                stats[ind]['total'] += (hit + miss)

            # Compute accuracies with Laplace smoothing
            accuracies = {}
            for ind in self.tunable_keys:
                hits = stats[ind]['hits']
                total = stats[ind]['total']
                acc = (hits + 1) / (total + 2) if total > 0 else 0.5
                accuracies[ind] = float(acc)

            logger.debug(f"Hitrate accuracies: {accuracies}")

            # Update weights via gradient ascent
            new_weights = dict(self.weights)
            for ind in self.tunable_keys:
                acc = accuracies[ind]
                current_w = self.weights[ind]
                gradient = acc - 0.5
                delta = self.lr * gradient
                new_w = float(np.clip(current_w + delta, self.min_w, self.max_w))
                new_weights[ind] = new_w

            # Keep EMA trend anchored
            new_weights['ema_trend'] = self.ema_anchor

            # Normalize to sum to 1.0
            total = sum(new_weights.values())
            if total > 0:
                norm = 1.0 / total
                new_weights = {k: v * norm for k, v in new_weights.items()}

            # Only emit if significant change
            try:
                keys = set(new_weights.keys()) | set(self.weights.keys())
                max_delta = max(abs(new_weights.get(k, 0.0) - self.weights.get(k, 0.0)) for k in keys)
            except Exception:
                max_delta = 0.0

            epsilon = 1e-4
            if max_delta < epsilon:
                logger.debug(f"[WEIGHTS] Insignificant change (max |Δw|={max_delta:.6g} < {epsilon}); suppressing INFO log")
                return None

            # Atomic update
            self.weights = new_weights
            logger.info(f"Weights updated: {self.weights}")
            return self.weights

        except Exception as e:
            logger.error(f"Weight update failed: {e}", exc_info=True)
            return None



    def get_weights(self) -> Dict[str, float]:
        """Get current indicator weights."""
        return dict(self.weights)



    def compute_indicator_score(self, features: Dict[str, float]) -> float:
        """
        Compute weighted indicator score from features with volatility-aware micro weighting.
        - Boost micro inputs when realized short-term volatility is active with loose range.
        - Suppress micro inputs in tight/low-vol regimes.
        """
        try:
            score = 0.0
            
            # Volatility and tightness context if present
            vol_short = float(features.get("std_dltp_short", 0.0))
            tight = float(features.get("price_range_tightness", 1.0))  # near 1.0 means very tight
            
            # Heuristic factors (kept minimal and bounded)
            # If market is active (looser range and some realized vol), boost micro; else suppress
            

                        
            if (vol_short > 0.0) and (tight < 0.98):
                micro_factor = 1.25
            elif (vol_short <= 0.0) or (tight >= 0.995):
                micro_factor = 0.60  # tighter suppression in tight regimes
            else:
                micro_factor = 1.0


            
            
            
            for key, base_weight in self.weights.items():
                val = float(features.get(key, 0.0))
                w = base_weight
                if key == "micro_slope":
                    w = float(np.clip(base_weight * micro_factor, self.min_w, self.max_w))
                score += w * val
            
            return float(score)
        except Exception as e:
            logger.warning(f"Indicator score computation failed: {e}")
            return 0.0



    def __repr__(self):
        return (f"AdaptiveModelPipeline(weights={self.weights}, "
                f"lr={self.lr}, ema_anchor={self.ema_anchor})")



    def replace_models(self, xgb=None, neutral=None):
        """
        Hot-reload XGB and/or neutrality models without restarting the pipeline.
        """
        try:
            


            if xgb is not None:
                self.xgb = xgb
                logger.info("XGB model hot-reloaded")



                # Fallback: load schema embedded in booster if file-based schema missing
                try:
                    if self.feature_schema_names is None and hasattr(self.xgb, "booster") and hasattr(self.xgb.booster, "attr"):
                        meta = self.xgb.booster.attr("feature_schema")
                        if meta:
                            try:
                                data = json.loads(meta)
                                names = data.get("feature_names") or data.get("features")
                                if names:
                                    self.set_feature_schema(names)
                                    logger.info("[SCHEMA] Loaded from booster.attr('feature_schema')")
                                else:
                                    logger.warning("[SCHEMA] Booster attribute present but no names field")
                            except Exception as e:
                                logger.warning(f"[SCHEMA] Failed to parse booster feature_schema attr: {e}")
                        else:
                            logger.info("[SCHEMA] No schema attribute in booster")
                except Exception as e:
                    logger.warning(f"[SCHEMA] Booster schema load failed: {e}")


                
                # Try to load persisted feature schema and apply via the official setter
                try:
                    # Prefer the new feature_schema.json saved by trainer
                    base_dir = os.path.dirname(os.getenv("XGB_PATH", "models/xgb_model.json")) or "."
                    schema_candidates = [
                        os.path.join(base_dir, "feature_schema.json"),                         # new
                        os.getenv("XGB_PATH", "models/xgb_model.json").replace(".json", "_schema.json")  # legacy
                    ]
                    schema = None
                    for sp in schema_candidates:
                        if os.path.exists(sp):
                            # Use module-level json (imported at top) to avoid shadowing in this scope
                            with open(sp, "r", encoding="utf-8") as f:
                                schema = json.load(f)
                            logger.info(f"Loaded feature schema: {sp}")
                            break
                    
                    names = None
                    if isinstance(schema, dict):
                        # support both formats {feature_names: [...]} and {features: [...]}
                        names = schema.get("feature_names") or schema.get("features")
                    if names:
                        self.set_feature_schema(names)
                    else:
                        logger.warning("Feature schema file found but missing keys; skipping set_feature_schema")
                except Exception as e:
                    logger.warning(f"Failed to load/apply feature schema: {e}")

            
            
            if neutral is not None:
                self.neutral_model = neutral
                logger.info("Neutrality model hot-reloaded")
        except Exception as e:
            logger.error(f"Model hot-reload failed: {e}", exc_info=True)



# ========== BACKGROUND WEIGHT REFRESH TASK ==========

async def weight_refresh_loop(
    pipeline: AdaptiveModelPipeline, 
    hitrate_path: str, 
    refresh_seconds: int = 30
):
    """
    Background task to periodically refresh weights from hitrate logs.
    Skips if file unchanged and quiet when changes are tiny.
    """
    logger.info(f"Starting weight refresh loop (every {refresh_seconds}s)")
    import os, time
    last_mtime = None

    while True:
        try:
            try:
                mtime = os.path.getmtime(hitrate_path) if os.path.exists(hitrate_path) else None
            except Exception:
                mtime = None

            if mtime is None:
                logger.debug("[WEIGHTS] Hitrate file not found")
            elif last_mtime is not None and mtime <= last_mtime:
                logger.debug("[WEIGHTS] Hitrate unchanged since last check")
            else:
                # File changed → attempt update
                updated = await pipeline.update_weights_from_file(hitrate_path)
                if updated:
                    logger.info(f"[WEIGHTS] Refreshed: {updated}")
                else:
                    logger.debug("[WEIGHTS] No significant delta or no valid records; skipping log")
                last_mtime = mtime

        except asyncio.CancelledError:
            logger.info("Weight refresh loop cancelled")
            break
        except Exception as e:
            logger.error(f"Weight refresh error: {e}", exc_info=True)
        await asyncio.sleep(refresh_seconds)



# ========== UTILITY FUNCTIONS ==========


def create_default_pipeline(cnn_lstm, xgb, rl_agent, neutral_model=None, **kwargs) -> AdaptiveModelPipeline:
    """
    Factory function to create pipeline with sensible defaults.
    
    Args:
        cnn_lstm: Trained CNN-LSTM model
        xgb: Trained XGBoost model
        rl_agent: RL agent
        neutral_model: Optional neutrality classifier
        **kwargs: Override default parameters
    
    Returns:
        Configured AdaptiveModelPipeline instance
    """
    defaults = {
        'base_weights': {
            'ema_trend': 0.35,
            'micro_slope': 0.25,  # reduced default
            'imbalance': 0.20,
            'mean_drift': 0.20
        },
        'base_alpha_buy': 0.6,
        'lr': 0.05,
        'ema_anchor': 0.35,
        'min_w': 0.05,
        'max_w': 0.80
    }
    defaults.update(kwargs)
    
    return AdaptiveModelPipeline(cnn_lstm, xgb, rl_agent, neutral_model=neutral_model, **defaults)




def validate_hitrate_log(hitrate_path: str) -> bool:
    """
    Validate hitrate log file format.
    
    Args:
        hitrate_path: Path to JSONL file
    
    Returns:
        True if valid, False otherwise
    """
    try:
        path = Path(hitrate_path)
        if not path.exists():
            return False
        
        with open(path, 'r') as f:
            for i, line in enumerate(f):
                if i >= 10:  # Check first 10 lines
                    break
                line = line.strip()
                if not line:
                    continue
                
                rec = json.loads(line)
                required = ['timestamp', 'indicator', 'hit', 'miss']
                if not all(k in rec for k in required):
                    logger.warning(f"Invalid record at line {i+1}: missing keys")
                    return False
        
        return True
    except Exception as e:
        logger.error(f"Hitrate log validation failed: {e}")
        return False
