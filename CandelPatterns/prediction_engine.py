"""
Intelligent prediction engine using weighted voting and pattern confidence.
"""
import logging
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

class PredictionEngine:
    """
    Advanced prediction using pattern signals, momentum, and volatility.
    """
    
    def __init__(self, momentum_weight: float = 0.15, 
                 pattern_weight: float = 0.70,
                 volume_weight: float = 0.15):
        """
        Initialize prediction engine with component weights.
        
        Args:
            momentum_weight: Weight for momentum component
            pattern_weight: Weight for pattern signals
            volume_weight: Weight for volume analysis
        """
        self.momentum_weight = momentum_weight
        self.pattern_weight = pattern_weight
        self.volume_weight = volume_weight
        
        # Ensure weights sum to 1
        total = momentum_weight + pattern_weight + volume_weight
        if abs(total - 1.0) > 0.01:
            logger.warning(f"Weights don't sum to 1.0 ({total}), normalizing...")
            self.momentum_weight /= total
            self.pattern_weight /= total
            self.volume_weight /= total
            
    def predict(self, patterns: List[Dict], momentum: float, volume_profile: Dict,
            atr_ratio: float, support_resistance: Dict, 
            df: pd.DataFrame = None, market_context: Dict = None) -> Dict:
        
        """
        Generate prediction based on all signals.
        
        Returns:
            Dictionary with direction, confidence, and reasoning
        """
        # Use market_context to adjust predictions
        if market_context:
            trend = market_context.get('trend', 'unknown')
            if trend == 'bullish' and total_score < 0:
                total_score *= 0.7  # Reduce bearish signals in bullish trend
            elif trend == 'bearish' and total_score > 0:
                total_score *= 0.7  # Reduce bullish signals in bearish trend
                        
        # Weight patterns by historical accuracy
        patterns = self._weight_patterns_by_accuracy(patterns)
        
        # Component scores
        pattern_score = self._calculate_pattern_score(patterns)
        momentum_score = self._calculate_momentum_score(momentum)
        volume_score = self._calculate_volume_score(volume_profile)
        
        # Weighted combination
        total_score = (
            self.pattern_weight * pattern_score +
            self.momentum_weight * momentum_score +
            self.volume_weight * volume_score
        )
        logger.info(f"Pre-ATR score: {total_score:.3f}, ATR ratio: {atr_ratio:.3f}")
        
        # Consider not multiplying, but adjusting instead
        if atr_ratio < 0.5:  # Very low volatility
            total_score *= 0.8
        elif atr_ratio > 1.5:  # High volatility
            total_score *= 1.2
                    
        # # Apply volatility adjustment
        # total_score *= atr_ratio
        
        # Support/resistance adjustment
        sr_adjustment = self._calculate_sr_adjustment(support_resistance, total_score)
        total_score += sr_adjustment
        
        # Convert score to prediction
        prediction = self._score_to_prediction(total_score)
        
        # ADD THIS: Validate with higher timeframe if DataFrame provided
        if df is not None and len(df) >= 15:  # Need at least 3 candles for 15-min timeframe
            if not self._check_higher_timeframe(df, prediction):
                # Reduce confidence if higher timeframe conflicts
                prediction['confidence'] *= 0.7
                prediction['warning'] = 'Higher timeframe conflict'
                        
        # Add reasoning
        prediction["reasoning"] = self._generate_reasoning(
            patterns, momentum, volume_profile, support_resistance, pattern_score
        )
        
        # Add component scores for transparency
        prediction["components"] = {
            "pattern_score": pattern_score,
            "momentum_score": momentum_score,
            "volume_score": volume_score,
            "total_score": total_score
        }
        
        return prediction

    def _calculate_pattern_score(self, patterns: List[Dict]) -> float:
        if not patterns:
            return 0.0
        
        bullish_score = 0.0
        bearish_score = 0.0
        
        for pattern in patterns:
            confidence = pattern["confidence"]
            strength = pattern["strength"]
            weight = confidence * strength
            
            if pattern["direction"] == "bullish":
                bullish_score += weight
            else:
                bearish_score += weight
        
        # Return net directional bias
        total = bullish_score + bearish_score
        if total > 0:
            return (bullish_score - bearish_score) / total
        return 0.0

    
    def _weight_patterns_by_accuracy(self, patterns: List[Dict]) -> List[Dict]:
        """Weight patterns by their historical accuracy."""
        weighted_patterns = []
        
        for pattern in patterns:
            # Get historical accuracy for this pattern
            hit_rate = pattern.get('hit_rate', 0.5)
            
            # Patterns with <50% accuracy get negative weight
            if hit_rate < 0.5:
                pattern['effective_confidence'] = pattern['confidence'] * (1 - hit_rate)
                pattern['direction'] = 'bearish' if pattern['direction'] == 'bullish' else 'bullish'
            else:
                pattern['effective_confidence'] = pattern['confidence'] * hit_rate
            
            weighted_patterns.append(pattern)
        
        return weighted_patterns

    
    def _calculate_momentum_score(self, momentum: float) -> float:
        """Convert momentum to normalized score."""
        # Sigmoid-like transformation
        return np.tanh(momentum * 10)  # Scale and bound to [-1, 1]
    
        
    def _calculate_volume_score(self, volume_profile: Dict) -> float:
        """Calculate score from volume analysis."""
        # Check if real volume data exists
        if not volume_profile.get("has_volume", False):
            return 0.0  # Neutral score when no volume data
        
        score = 0.0
        
        # Volume ratio contribution
        vol_ratio = volume_profile.get("volume_ratio", 1.0)
        if vol_ratio > 1.5:  # High volume
            score += 0.3
        elif vol_ratio < 0.5:  # Low volume
            score -= 0.3
        
        # Volume trend contribution
        trend = volume_profile.get("volume_trend", "neutral")
        if trend == "increasing":
            score += 0.2
        elif trend == "decreasing":
            score -= 0.2
        
        # Price-volume correlation contribution
        correlation = volume_profile.get("price_volume_correlation", 0)
        score += correlation * 0.1
        
        return np.clip(score, -1, 1)


    def _calculate_sr_adjustment(self, sr: Dict, current_score: float) -> float:
        """Adjust score based on support/resistance levels."""
        position = sr.get("position", "middle")
        
        if position == "near_resistance" and current_score > 0:
            # Reduce bullish bias near resistance
            return -0.1
        elif position == "near_support" and current_score < 0:
            # Reduce bearish bias near support
            return 0.1
        
        return 0.0
    
    def _score_to_prediction(self, score: float) -> Dict:
        """Convert numerical score to prediction."""
        # Determine direction
        if score > 0.1:
            direction = "bullish"
        elif score < -0.1:
            direction = "bearish"
        else:
            direction = "neutral"
        
        # Calculate confidence (0 to 1)
        confidence = min(1.0, abs(score))
        
        # Determine strength
        if confidence > 0.7:
            strength = "strong"
        elif confidence > 0.4:
            strength = "moderate"
        else:
            strength = "weak"
        
        return {
            "direction": direction,
            "confidence": confidence,
            "strength": strength,
            "score": score
        }
    
    def _generate_reasoning(self, patterns, momentum, volume, sr, pattern_score) -> str:
        """Generate human-readable reasoning for the prediction."""
        reasons = []
        
        # Pattern reasoning
        if patterns:
            pattern_names = [p["name"] for p in patterns[:3]]  # Top 3
            if pattern_score > 0:
                reasons.append(f"Bullish patterns detected: {', '.join(pattern_names)}")
            elif pattern_score < 0:
                reasons.append(f"Bearish patterns detected: {', '.join(pattern_names)}")
        
        # Momentum reasoning
        if abs(momentum) > 0.01:
            direction = "positive" if momentum > 0 else "negative"
            reasons.append(f"Price momentum is {direction} ({momentum:.2%})")
        
        # Volume reasoning
        vol_trend = volume.get("volume_trend", "neutral")
        if vol_trend != "neutral":
            reasons.append(f"Volume is {vol_trend}")
        
        # Support/resistance reasoning
        position = sr.get("position", "middle")
        if position != "middle":
            reasons.append(f"Price is {position.replace('_', ' ')}")
        
        return " | ".join(reasons) if reasons else "No clear signals"

    def _check_higher_timeframe(self, df: pd.DataFrame, prediction: Dict) -> bool:
        """Confirm signal with higher timeframe trend."""
        # Create 15-minute candles from 5-minute data
        df_15m = df.resample('15min').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        })
        
        # Check if higher timeframe agrees
        trend_15m = self._calculate_trend(df_15m.tail(10))
        
        if prediction['direction'] == 'bearish' and trend_15m > 0:
            logger.warning("Bearish signal conflicts with 15m uptrend")
            return False
        
        if not isinstance(df.index, pd.DatetimeIndex):
            logger.warning("Cannot check higher timeframe - index is not DatetimeIndex")
            
        return True

    def _calculate_trend(self, df: pd.DataFrame) -> float:
        """Calculate trend using simple linear regression."""
        if len(df) < 2:
            return 0.0
        x = np.arange(len(df))
        y = df['close'].values
        coeffs = np.polyfit(x, y, 1)
        return coeffs[0]  # Slope indicates trend

    def validate_with_higher_timeframe(self, signal: Dict, df: pd.DataFrame) -> bool:
        """
        Validate signal with higher timeframe analysis.
        
        Args:
            signal: Current signal dictionary
            df: DataFrame with OHLC data
            
        Returns:
            True if higher timeframe confirms signal
        """
        if len(df) < 60:  # Need at least 60 5-min candles for hourly analysis
            return True  # Can't validate, assume valid
        
        # Create 15-minute candles
        df_15m = df.tail(30).resample('15min').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }) if isinstance(df.index, pd.DatetimeIndex) else None
        
        # Create hourly candles
        df_hourly = df.tail(60).resample('60T').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }) if isinstance(df.index, pd.DatetimeIndex) else None
        
        if df_15m is None or df_hourly is None:
            return True
        
        # Calculate trends
        trend_15m = self._calculate_trend(df_15m) if len(df_15m) > 1 else 0
        trend_hourly = self._calculate_trend(df_hourly) if len(df_hourly) > 1 else 0
        
        # Validate signal direction
        if signal['direction'] == 'bullish':
            return trend_15m >= 0 and trend_hourly >= 0
        elif signal['direction'] == 'bearish':
            return trend_15m <= 0 and trend_hourly <= 0
        
        return True
