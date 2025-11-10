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


def _sign(x: float) -> int:
    """Helper for sign extraction with safe type handling."""
    try:
        x = float(x)
        if x > 0:
            return 1
        elif x < 0:
            return -1
        return 0
    except (TypeError, ValueError):
        return 0


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
                'micro_slope': 0.35,
                'imbalance': 0.15,
                'mean_drift': 0.15
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

        
        logger.info(f"Learning rate: {self.lr}, EMA anchor: {self.ema_anchor}")

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
        engineered_feature_names: Optional[list] = None  # NEW
    ) -> Tuple[np.ndarray, float]:
        """
        Generate ensemble prediction with optional indicator modulation.
        
        Returns:
            (signal_probs, adjusted_alpha_buy)
            - signal_probs: np.array([[p_sell, p_buy]])
            - adjusted_alpha_buy: RL-adjusted confidence threshold
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
                latent_features = np.zeros(8, dtype=float)  # safe default

            # ========== ENGINEERED FEATURES ==========
            try:
                ef_vals = np.asarray(engineered_features, dtype=float).ravel().tolist()
            except Exception as e:
                logger.warning(f"Feature engineering failed: {e}")
                ef_vals = [0.0]

            # ========== ALIGN TO SCHEMA & XGB INPUT (NO LATENT CONCAT) ==========
            try:
                xgb_input = self._align_features_to_schema(engineered_feature_names, ef_vals)
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
            except Exception as e:
                logger.error(f"XGB prediction failed: {e}")
                signal_probs = np.array([[0.5, 0.5]], dtype=float)

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
                logger.debug(f"Neutrality model inference failed: {e}")
                neutral_prob = None

            # ========== INDICATOR MODULATION (OPTIONAL) ==========
            if indicator_score is not None:
                try:
                    buy_prob = float(signal_probs[0][1])
                    indicator_norm = 0.5 + 0.5 * np.tanh(indicator_score)
                    blended_buy_prob = 0.5 * buy_prob + 0.5 * indicator_norm
                    signal_probs = np.array([[1.0 - blended_buy_prob, blended_buy_prob]], dtype=float)
                    logger.debug(f"Indicator blend: model={buy_prob:.3f}, "
                                f"indicator={indicator_norm:.3f}, "
                                f"blended={blended_buy_prob:.3f}")
                except Exception as e:
                    logger.warning(f"Indicator modulation failed: {e}")

                # Deterministic fallback if almost flat
                try:
                    buy_prob_now = float(signal_probs[0][1])
                    if not np.isfinite(buy_prob_now) or abs(buy_prob_now - 0.5) < 0.02:
                        k = 2.0
                        p = float(1.0 / (1.0 + np.exp(-k * float(indicator_score))))
                        p = 0.0 if not np.isfinite(p) else min(max(p, 0.0), 1.0)
                        signal_probs = np.array([[1.0 - p, p]], dtype=float)
                except Exception as e:
                    logger.debug(f"Fallback mapping failed: {e}")

            # ========== PATTERN PROBABILITY ADJUSTMENT (OPTIONAL) ==========
            try:
                if pattern_prob_adjustment is not None:
                    buy_prob = float(signal_probs[0][1])
                    adjusted_buy_prob = np.clip(buy_prob + float(pattern_prob_adjustment), 0.0, 1.0)
                    signal_probs = np.array([[1.0 - adjusted_buy_prob, adjusted_buy_prob]], dtype=float)
                    logger.debug(f"Pattern adjustment: {float(pattern_prob_adjustment):+.3f} | "
                                f"buy_prob: {buy_prob:.3f} → {adjusted_buy_prob:.3f}")
            except Exception as e:
                logger.warning(f"Pattern adjustment failed: {e}")

            # ========== RL + NEUTRALITY-BASED CONFIDENCE ADJUSTMENT ==========
            try:
                adjusted_alpha = float(self.rl_agent.adjust_confidence(recent_profit_factor))
            except Exception as e:
                logger.warning(f"RL adjustment failed: {e}")
                adjusted_alpha = self.base_alpha_buy
            
            try:
                if neutral_prob is not None and np.isfinite(neutral_prob):
                    delta = (neutral_prob - 0.5) * 0.12
                    pre = adjusted_alpha
                    adjusted_alpha = float(np.clip(adjusted_alpha + delta, 0.50, 0.80))
                    logger.info(f"[NEUTRAL] p_neutral={neutral_prob:.3f} | alpha {pre:.3f} → {adjusted_alpha:.3f} (Δ={delta:+.3f})")
            except Exception as e:
                logger.debug(f"Neutrality alpha adjust failed: {e}")

            # ========== FINAL PREDICTION LOG ==========
            logger.info(f"[PRED] p_buy={float(signal_probs[0][1]):.3f} p_sell={float(signal_probs[0][0]):.3f} "
                        f"alpha={adjusted_alpha:.3f} recentPF={float(recent_profit_factor):.3f} "
                        f"schema_n={self.feature_schema_size}")

            return signal_probs, adjusted_alpha

        except Exception as e:
            logger.error(f"Prediction pipeline failed: {e}", exc_info=True)
            return np.array([[0.5, 0.5]], dtype=float), self.base_alpha_buy















    # ========== INTEGRATED WEIGHT TUNING ==========
    async def update_weights_from_file(self, hitrate_path: str) -> Optional[Dict[str, float]]:
        """
        Update indicator weights from hitrate log file (proactive learning).
        
        Reads JSONL file with format:
        {"timestamp": "...", "indicator": "micro_slope", "hit": 1, "miss": 0}
        
        Args:
            hitrate_path: Path to JSONL hitrate log
        
        Returns:
            Updated weights dict if successful, None otherwise
        """
        try:
            path = Path(hitrate_path)
            if not path.exists():
                logger.debug(f"Hitrate file not found: {hitrate_path}")
                return None

            # Read and parse JSONL
            records = []
            with open(path, 'r') as f:
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
                # Laplace smoothing: (hits + 1) / (total + 2)
                acc = (hits + 1) / (total + 2) if total > 0 else 0.5
                accuracies[ind] = float(acc)

            logger.info(f"Hitrate accuracies: {accuracies}")

            # Update weights via gradient ascent
            new_weights = dict(self.weights)
            
            for ind in self.tunable_keys:
                acc = accuracies[ind]
                current_w = self.weights[ind]
                
                # Gradient: (accuracy - 0.5) pushes weight up if acc > 0.5
                gradient = acc - 0.5
                delta = self.lr * gradient
                
                new_w = current_w + delta
                new_w = np.clip(new_w, self.min_w, self.max_w)
                new_weights[ind] = float(new_w)

            # Keep EMA trend anchored
            new_weights['ema_trend'] = self.ema_anchor

            # Normalize to sum to 1.0
            total = sum(new_weights.values())
            if total > 0:
                norm = 1.0 / total
                new_weights = {k: v * norm for k, v in new_weights.items()}

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
        Compute weighted indicator score from features.
        
        Args:
            features: Dict with keys matching weight keys
        
        Returns:
            Weighted sum of indicators
        """
        try:
            score = 0.0
            for key, weight in self.weights.items():
                val = features.get(key, 0.0)
                score += weight * float(val)
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
    
    Args:
        pipeline: AdaptiveModelPipeline instance
        hitrate_path: Path to JSONL hitrate log
        refresh_seconds: Refresh interval in seconds
    """
    logger.info(f"Starting weight refresh loop (every {refresh_seconds}s)")
    
    while True:
        try:
            new_weights = await pipeline.update_weights_from_file(hitrate_path)
            if new_weights:
                logger.info(f"[WEIGHTS] Refreshed: {new_weights}")
            else:
                logger.debug("[WEIGHTS] No update (file missing or empty)")
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
            'micro_slope': 0.35,
            'imbalance': 0.15,
            'mean_drift': 0.15
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
