"""
Unified feature engineering and model utilities.
Consolidates: feature_engineering.py, drift_detector.py, smart_order_executor.py
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging

try: 
    from scipy.stats import ks_2samp # type: ignore
except Exception: 
    ks_2samp = None

logger = logging.getLogger(__name__)


class FeaturePipeline:
    """
    All-in-one feature engineering, drift detection, and order execution logic.
    Eliminates redundant file separation.
    """
    
    def __init__(self, train_features: Dict, rl_agent):
        self.train_features = train_features
        self.rl_agent = rl_agent
        logger.info("Feature pipeline initialized")

    # ========== FEATURE ENGINEERING ==========
    @staticmethod
    def compute_emas(prices: List[float], periods: Optional[List[int]] = None) -> Dict[str, float]:
        """Compute EMAs for given periods."""

        if periods is None:
            periods = [8, 21, 50]
                    
        s = pd.Series(prices, dtype='float64') if prices else pd.Series([], dtype='float64')
        out = {}
        for p in periods:
            try:
                out[f'ema_{p}'] = float(s.ewm(span=min(p, max(1, len(s))), adjust=False).mean().iloc[-1]) if len(s) else 0.0
            except Exception:
                out[f'ema_{p}'] = 0.0
                
        logger.debug(f"EMAs computed (len={len(prices)}): {out}")
        return out


    @staticmethod
    def order_flow_dynamics(order_books: List[Dict], window: int = 5) -> Dict[str, float]:
        """Compute spread from order books (volume-free)."""
        try:
            ob_df = pd.DataFrame(order_books[-window:])
            if ob_df.empty:
                return {'spread': 0.0}
            
            spread = (ob_df['ask_price'].astype(float).mean() - 
                     ob_df['bid_price'].astype(float).mean()) if 'ask_price' in ob_df and 'bid_price' in ob_df else 0.0
            return {'spread': float(spread)}
        except Exception:
            return {'spread': 0.0}

    @staticmethod
    def volatility_regime_tag(rng_value: float, rng_history: List[float]) -> str:
        """Tag volatility regime using price range percentiles."""
        try:
            prc = np.percentile(rng_history, [33, 66]) if len(rng_history) >= 3 else [rng_value, rng_value]
            if rng_value < prc[0]:
                return 'low'
            elif rng_value < prc[1]:
                return 'medium'
            else:
                return 'high'
        except Exception:
            return 'unknown'

    @staticmethod
    def normalize_features(features: Dict, scale: float = 1.0) -> Dict[str, float]:
        """Normalize numeric features by scale."""
        denom = scale if (isinstance(scale, (int, float)) and scale > 0) else 1.0
        out = {}
        for k, v in features.items():
            try:
                out[k] = float(v) / denom
            except Exception:
                pass  # Skip non-numeric
        return out

    # ========== DRIFT DETECTION ==========
    def detect_drift(self, live_features: Dict) -> Dict[str, Dict[str, float]]:
        """
        Detect concept drift using KS-statistics.
        Integrated from drift_detector.py.
        """
        drift_stats = {}
        for feat in self.train_features:
            try:
                base = np.asarray(self.train_features[feat], dtype=float)
                live_val = live_features.get(feat, None)
                if live_val is None:
                    continue
                
                live_arr = np.asarray(live_val if hasattr(live_val, '__len__') else [live_val], dtype=float)
                if base.size == 0 or live_arr.size == 0:
                    continue
                
                stat, pval = ks_2samp(base, live_arr, alternative='two-sided', mode='auto')
                drift_stats[feat] = {'ks_stat': float(stat), 'p_value': float(pval)}
            except Exception:
                continue
        return drift_stats

    # ========== SMART ORDER EXECUTION ==========
    def place_order(self, price: float, fill_prob: float, time_waited: float, 
                   get_mid_price_func) -> float:
        """
        Adaptive limit order execution with RL feedback.
        Integrated from smart_order_executor.py.
        """
        try:
            price = float(price) if price is not None else 0.0
            fp = float(fill_prob) if fill_prob is not None else 0.0
            tw = float(time_waited) if time_waited is not None else 0.0
        except Exception:
            price, fp, tw = 0.0, 0.0, 0.0

        try:
            threshold = float(getattr(self.rl_agent, 'threshold', 0.5))
        except Exception:
            threshold = 0.5

        if fp < threshold and tw > 5:
            try:
                mid = float(get_mid_price_func()) if callable(get_mid_price_func) else float(get_mid_price_func)
            except Exception:
                mid = price
            price = self._adjust_limit_towards_mid(price, mid, fraction=0.25)
        
        return price

    @staticmethod
    def _adjust_limit_towards_mid(price: float, mid_price: float, fraction: float) -> float:
        """Adjust limit price towards mid."""
        return price + fraction * (mid_price - price)
