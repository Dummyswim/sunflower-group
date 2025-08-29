"""
Advanced technical indicators with ATR scaling and momentum.
"""
import numpy as np
import pandas as pd
import logging
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

try:
    import talib
    HAS_TALIB = True
except ImportError:
    HAS_TALIB = False

class AdvancedIndicators:
    """Enhanced technical indicators with volatility adjustment."""
    
    @staticmethod
    def calculate_atr(df: pd.DataFrame, period: int = 14) -> Optional[float]:
        """Calculate Average True Range."""
        if not HAS_TALIB or len(df) < period + 1:
            return None
        
        try:
            atr = talib.ATR(df['high'].values, df['low'].values, 
                          df['close'].values, timeperiod=period)
            return float(atr[-1]) if not np.isnan(atr[-1]) else None
        except Exception as e:
            logger.error(f"ATR calculation failed: {e}")
            return None
    
    @staticmethod
    def calculate_volatility_ratio(df: pd.DataFrame, atr: Optional[float]) -> float:
        """
        Calculate volatility ratio for confidence scaling.
        Returns a multiplier: >1 for high volatility, <1 for low.
        """
        if atr is None or len(df) == 0:
            return 1.0
        
        last_price = df['close'].iloc[-1]
        if last_price == 0:
            return 1.0
        
        atr_pct = atr / last_price
        logger.debug(f"ATR: {atr}, Price: {last_price}, ATR%: {atr_pct:.4f}")
        
        if atr_pct > 0.01:  # >1% volatility
            return 1.2  # Boost confidence
        elif atr_pct < 0.002:  # <0.2% volatility
            return 0.8  # Dampen confidence
        else:
            return 1.0
    
    @staticmethod
    def calculate_momentum(df: pd.DataFrame, periods: int = 3) -> float:
        """Calculate price momentum over specified periods."""
        if len(df) < periods:
            return 0.0
        
        closes = df['close'].tail(periods).values
        if len(closes) < 2:
            return 0.0
        
        # Simple momentum: (last - first) / first
        momentum = (closes[-1] - closes[0]) / closes[0] if closes[0] != 0 else 0
        return momentum
    
    @staticmethod
    def calculate_volume_profile(df: pd.DataFrame, periods: int = 20) -> Dict:
        """Calculate volume profile statistics."""
        if 'volume' not in df.columns or len(df) < periods:
            return {"avg_volume": 0, "volume_ratio": 1.0, "volume_trend": "neutral"}
        
        recent_vol = df['volume'].tail(periods)
        current_vol = df['volume'].iloc[-1] if len(df) > 0 else 0
        avg_vol = recent_vol.mean()
        
        volume_ratio = current_vol / avg_vol if avg_vol > 0 else 1.0
        
        # Volume trend
        if len(recent_vol) >= 3:
            vol_ma = recent_vol.rolling(3).mean()
            if vol_ma.iloc[-1] > vol_ma.iloc[-2]:
                volume_trend = "increasing"
            elif vol_ma.iloc[-1] < vol_ma.iloc[-2]:
                volume_trend = "decreasing"
            else:
                volume_trend = "neutral"
        else:
            volume_trend = "neutral"
        
        return {
            "avg_volume": float(avg_vol),
            "volume_ratio": float(volume_ratio),
            "volume_trend": volume_trend
        }
    
    @staticmethod
    def calculate_support_resistance(df: pd.DataFrame, window: int = 20) -> Dict:
        """Calculate dynamic support and resistance levels."""
        if len(df) < window:
            return {"support": None, "resistance": None}
        
        recent = df.tail(window)
        
        # Find local highs and lows
        highs = recent['high'].values
        lows = recent['low'].values
        
        # Simple approach: use percentiles
        resistance = np.percentile(highs, 90)
        support = np.percentile(lows, 10)
        
        current_price = df['close'].iloc[-1]
        
        return {
            "support": float(support),
            "resistance": float(resistance),
            "position": "near_support" if abs(current_price - support) / support < 0.01
                       else "near_resistance" if abs(current_price - resistance) / resistance < 0.01
                       else "middle"
        }
