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
        """Calculate ATR with robust fallback."""
        if len(df) < 2:  # Need at least 2 candles
            return None
        
        try:
            # Try TA-Lib first if available
            if HAS_TALIB and len(df) >= period:
                atr = talib.ATR(df['high'].values, df['low'].values, 
                            df['close'].values, timeperiod=period)
                if not np.isnan(atr[-1]):
                    return float(atr[-1])
        except:
            pass
        
        # Fallback calculation
        try:
            # Use simple range if not enough data for ATR
            if len(df) < period:
                # Use average range of available candles
                ranges = df['high'] - df['low']
                return float(ranges.mean()) if len(ranges) > 0 else None
            
            # Full ATR calculation
            high_low = df['high'] - df['low']
            high_close = np.abs(df['high'] - df['close'].shift())
            low_close = np.abs(df['low'] - df['close'].shift())
            
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = ranges.max(axis=1)
            atr_value = true_range.rolling(period).mean().iloc[-1]
            
            return float(atr_value) if not pd.isna(atr_value) else None
            
        except Exception as e:
            logger.error(f"ATR calculation failed: {e}")
            # Last resort: return 1% of current price
            if len(df) > 0:
                return float(df['close'].iloc[-1] * 0.01)
            return None

    
    @staticmethod
    def calculate_volatility_ratio(df: pd.DataFrame, atr: Optional[float]) -> float:
        """
        Calculate volatility ratio for confidence scaling.
        Returns a multiplier: >1 for high volatility, <1 for low.
        """
        if atr is None or len(df) == 0:
            # Use price-based volatility as fallback
            price_volatility = df['close'].pct_change().std() * np.sqrt(252)
            return 1.0 + (price_volatility - 0.15) * 2  # Scale around 15% annual vol

        
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
        """
        Calculate volume profile statistics using only real volume data.
        Returns default values if volume data is unavailable.
        """
        volume_profile = {
            "avg_volume": 0.0,
            "volume_ratio": 1.0,
            "volume_trend": "neutral",
            "data_quality": "no_data",
            "has_volume": False,
            "volume_percentile": 50.0,
            "volume_momentum": 0.0,
            "price_volume_correlation": 0.0
        }
        
        # Check if volume data exists and is valid
        if 'volume' not in df.columns:
            logger.warning("Volume column missing from DataFrame")
            return volume_profile
        
        # Check if we have any non-zero volume data
        if df['volume'].sum() == 0 or df['volume'].isna().all():
            logger.warning("No valid volume data available")
            return volume_profile
        
        # If we have valid volume data, calculate metrics
        recent_vol = df['volume'].tail(periods)
        
        # Filter out zero/NaN values for calculations
        valid_volumes = recent_vol[recent_vol > 0]
        
        if len(valid_volumes) == 0:
            logger.warning("No valid volume in recent period")
            return volume_profile
        
        # Calculate average volume
        avg_vol = float(valid_volumes.mean())
        current_vol = float(df['volume'].iloc[-1]) if len(df) > 0 else 0
        
        volume_profile["avg_volume"] = avg_vol
        volume_profile["has_volume"] = True
        volume_profile["data_quality"] = "real"
        
        # Calculate volume ratio (current vs average)
        volume_profile["volume_ratio"] = current_vol / avg_vol if avg_vol > 0 else 1.0
        
        # Calculate volume percentile (where does current volume rank)
        if len(valid_volumes) >= 5:
            percentile = (valid_volumes < current_vol).sum() / len(valid_volumes) * 100
            volume_profile["volume_percentile"] = float(percentile)
        
        # Volume trend calculation using EMA
        if len(recent_vol) >= 5:
            # Short and long EMAs for volume
            vol_ema_short = recent_vol.ewm(span=3, adjust=False).mean()
            vol_ema_long = recent_vol.ewm(span=periods, adjust=False).mean()
            
            # Determine trend
            if vol_ema_short.iloc[-1] > vol_ema_long.iloc[-1] * 1.1:
                volume_profile["volume_trend"] = "increasing"
            elif vol_ema_short.iloc[-1] < vol_ema_long.iloc[-1] * 0.9:
                volume_profile["volume_trend"] = "decreasing"
            else:
                volume_profile["volume_trend"] = "stable"
            
            # Volume momentum (rate of change)
            if len(recent_vol) >= 10:
                vol_momentum = (recent_vol.iloc[-1] - recent_vol.iloc[-10]) / recent_vol.iloc[-10] if recent_vol.iloc[-10] > 0 else 0
                volume_profile["volume_momentum"] = float(vol_momentum)
        
        # Price-Volume correlation
        if len(df) >= periods and 'close' in df.columns:
            price_changes = df['close'].pct_change().tail(periods)
            volume_changes = df['volume'].pct_change().tail(periods)
            
            # Remove NaN values for correlation
            valid_mask = ~(price_changes.isna() | volume_changes.isna())
            if valid_mask.sum() >= 5:
                correlation = price_changes[valid_mask].corr(volume_changes[valid_mask])
                if not pd.isna(correlation):
                    volume_profile["price_volume_correlation"] = float(correlation)
        
        # Add volume profile zones (high/medium/low volume areas)
        if len(df) >= 20:
            volume_profile["volume_zones"] = AdvancedIndicators._calculate_volume_zones(df.tail(20))
        
        return volume_profile
    @staticmethod
    def _calculate_volume_zones(df: pd.DataFrame) -> Dict:
        """
        Calculate volume distribution zones for better volume profile analysis.
        """
        zones = {
            "high_volume_price": 0.0,
            "low_volume_price": 0.0,
            "volume_weighted_price": 0.0
        }
        
        if 'volume' not in df.columns or df['volume'].sum() == 0:
            return zones
        
        # Calculate VWAP (Volume Weighted Average Price)
        vwap = (df['close'] * df['volume']).sum() / df['volume'].sum()
        zones["volume_weighted_price"] = float(vwap)
        
        # Find price at highest volume
        max_vol_idx = df['volume'].idxmax()
        if not pd.isna(max_vol_idx):
            zones["high_volume_price"] = float(df.loc[max_vol_idx, 'close'])
        
        # Find price at lowest volume (excluding zeros)
        non_zero_vol = df[df['volume'] > 0]
        if len(non_zero_vol) > 0:
            min_vol_idx = non_zero_vol['volume'].idxmin()
            if not pd.isna(min_vol_idx):
                zones["low_volume_price"] = float(non_zero_vol.loc[min_vol_idx, 'close'])
        
        return zones


    def enhanced_market_context(self, df: pd.DataFrame, news_sentiment=None, vix_level=None) -> Dict:
        """Enhanced market context with additional inputs."""
        context = self.get_market_context(df)
        
        # Add VIX-based volatility regime
        if vix_level:
            context['volatility_regime'] = 'high' if vix_level > 20 else 'normal'
        
        # Add trend strength using ADX if available
        if HAS_TALIB and len(df) >= 14:
            try:
                adx = talib.ADX(df['high'].values, df['low'].values, 
                            df['close'].values, timeperiod=14)
                if not np.isnan(adx[-1]):
                    context['trend_strength'] = float(adx[-1])
            except:
                pass
        
        # Add news sentiment if provided
        if news_sentiment:
            context['news_sentiment'] = news_sentiment
        
        return context

    
    @staticmethod
    def calculate_support_resistance(df: pd.DataFrame, window: int = 20) -> Dict:
        """Enhanced S/R calculation using actual price pivots."""
        if len(df) < window:
            return {"support": None, "resistance": None}
        
        # Find actual pivot points
        highs = df['high'].rolling(5, center=True).max() == df['high']
        lows = df['low'].rolling(5, center=True).min() == df['low']
        
        pivot_highs = df[highs]['high'].tail(10)
        pivot_lows = df[lows]['low'].tail(10)
        
        current_price = df['close'].iloc[-1]
        
        # Find nearest support/resistance
        resistance = pivot_highs[pivot_highs > current_price].min() if len(pivot_highs[pivot_highs > current_price]) > 0 else current_price * 1.02
        support = pivot_lows[pivot_lows < current_price].max() if len(pivot_lows[pivot_lows < current_price]) > 0 else current_price * 0.98
        
        return {
            "support": float(support),
            "resistance": float(resistance),
            "position": "near_support" if abs(current_price - support) / support < 0.01
                    else "near_resistance" if abs(current_price - resistance) / resistance < 0.01
                    else "middle"
        }
        
    @staticmethod
    def get_market_context(df: pd.DataFrame) -> Dict:
        """Determine market conditions for better pattern interpretation."""
        if len(df) < 50:
            return {
                'trend': 'unknown',
                'volatility': 'normal',
                'trend_strength': 0.0
            }
        
        sma_20 = df['close'].rolling(20).mean()
        sma_50 = df['close'].rolling(50).mean()
        
        # Trend determination
        if sma_20.iloc[-1] > sma_50.iloc[-1] and sma_20.iloc[-5] < sma_50.iloc[-5]:
            trend = "bullish_crossover"
        elif sma_20.iloc[-1] < sma_50.iloc[-1] and sma_20.iloc[-5] > sma_50.iloc[-5]:
            trend = "bearish_crossover"
        elif sma_20.iloc[-1] > sma_50.iloc[-1]:
            trend = "bullish"
        else:
            trend = "bearish"
        
        # Volatility
        returns = df['close'].pct_change()
        volatility = returns.std() * np.sqrt(252)
        
        return {
            'trend': trend,
            'volatility': 'high' if volatility > 0.25 else 'low',
            'trend_strength': abs(sma_20.iloc[-1] - sma_50.iloc[-1]) / sma_50.iloc[-1]
        }
            