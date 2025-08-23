"""
Enhanced technical indicators module with signal accuracy and duration prediction.
"""
import pandas as pd
import numpy as np
import logging
from typing import Dict, Optional, Tuple, List, Any
from scipy import stats
from collections import deque

logger = logging.getLogger(__name__)

class SignalAccuracyTracker:
    """Track and calculate signal accuracy metrics."""
    
    def __init__(self, window_size: int = 100):
        """
        Initialize accuracy tracker.
        
        Args:
            window_size: Number of signals to track for accuracy calculation
        """
        self.window_size = window_size
        self.signal_history = deque(maxlen=window_size)
        self.prediction_history = deque(maxlen=window_size)
        logger.debug(f"SignalAccuracyTracker initialized with window_size={window_size}")
    
    def add_signal(self, signal_data: Dict):
        """Add a signal to history for tracking."""
        self.signal_history.append(signal_data)
        logger.debug(f"Signal added to history: {signal_data.get('signal', 'unknown')}")
    
    def add_prediction(self, prediction_data: Dict):
        """Add a prediction for validation."""
        self.prediction_history.append(prediction_data)
        logger.debug("Prediction added to history")
    
    def calculate_accuracy_metrics(self) -> Dict:
        """
        Calculate accuracy metrics based on historical signals.
        
        Returns:
            Dictionary with accuracy metrics
        """
        try:
            if len(self.signal_history) < 10:
                logger.debug("Insufficient history for accuracy calculation")
                return {
                    "signal_accuracy": 50.0,
                    "confidence_sustain": 50.0,
                    "win_rate": 50.0,
                    "avg_duration_accuracy": 50.0
                }
            
            # Calculate win rate (signals that moved in predicted direction)
            correct_signals = 0
            total_signals = 0
            
            for i in range(1, len(self.signal_history)):
                prev_signal = self.signal_history[i-1]
                curr_signal = self.signal_history[i]
                
                if prev_signal.get('action') == 'buy':
                    if curr_signal.get('price', 0) > prev_signal.get('price', 0):
                        correct_signals += 1
                    total_signals += 1
                elif prev_signal.get('action') == 'sell':
                    if curr_signal.get('price', 0) < prev_signal.get('price', 0):
                        correct_signals += 1
                    total_signals += 1
            
            win_rate = (correct_signals / max(total_signals, 1)) * 100
            
            # Calculate confidence sustenance (how well confidence correlates with outcome)
            confidence_correlation = self._calculate_confidence_correlation()
            
            # Calculate duration accuracy
            duration_accuracy = self._calculate_duration_accuracy()
            
            # Overall signal accuracy (weighted average)
            signal_accuracy = (win_rate * 0.5 + confidence_correlation * 0.3 + duration_accuracy * 0.2)
            
            metrics = {
                "signal_accuracy": round(signal_accuracy, 1),
                "confidence_sustain": round(confidence_correlation, 1),
                "win_rate": round(win_rate, 1),
                "avg_duration_accuracy": round(duration_accuracy, 1)
            }
            
            logger.debug(f"Accuracy metrics calculated: {metrics}")
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating accuracy metrics: {e}")
            return {
                "signal_accuracy": 50.0,
                "confidence_sustain": 50.0,
                "win_rate": 50.0,
                "avg_duration_accuracy": 50.0
            }
    
    def _calculate_confidence_correlation(self) -> float:
        """Calculate how well confidence scores correlate with actual outcomes."""
        try:
            if len(self.signal_history) < 5:
                return 50.0
            
            confidences = []
            outcomes = []
            
            for i in range(1, len(self.signal_history)):
                prev_signal = self.signal_history[i-1]
                curr_signal = self.signal_history[i]
                
                confidence = prev_signal.get('confidence', 50)
                confidences.append(confidence)
                
                # Calculate outcome (1 for correct, 0 for incorrect)
                if prev_signal.get('action') == 'buy':
                    outcome = 1 if curr_signal.get('price', 0) > prev_signal.get('price', 0) else 0
                elif prev_signal.get('action') == 'sell':
                    outcome = 1 if curr_signal.get('price', 0) < prev_signal.get('price', 0) else 0
                else:
                    outcome = 0.5
                
                outcomes.append(outcome)
            
            if len(confidences) > 1 and len(outcomes) > 1:
                correlation = np.corrcoef(confidences, outcomes)[0, 1]
                if not np.isnan(correlation):
                    return abs(correlation) * 100
            
            return 50.0
            
        except Exception as e:
            logger.error(f"Error calculating confidence correlation: {e}")
            return 50.0
    
    def _calculate_duration_accuracy(self) -> float:
        """Calculate accuracy of duration predictions."""
        try:
            if len(self.prediction_history) < 5:
                return 50.0
            
            accurate_predictions = 0
            total_predictions = 0
            
            for prediction in self.prediction_history:
                predicted_duration = prediction.get('predicted_duration', 0)
                actual_duration = prediction.get('actual_duration', 0)
                
                if predicted_duration > 0 and actual_duration > 0:
                    error_pct = abs(predicted_duration - actual_duration) / predicted_duration
                    if error_pct < 0.3:  # Within 30% error
                        accurate_predictions += 1
                    total_predictions += 1
            
            if total_predictions > 0:
                return (accurate_predictions / total_predictions) * 100
            
            return 50.0
            
        except Exception as e:
            logger.error(f"Error calculating duration accuracy: {e}")
            return 50.0


class TechnicalIndicators:
    """Enhanced technical indicators with improved calculations."""
    
    @staticmethod
    def calculate_ema(prices: pd.Series, period: int) -> pd.Series:
        """Calculate Exponential Moving Average with validation."""
        try:
            if len(prices) < period:
                logger.warning(f"Insufficient data for EMA calculation: {len(prices)} < {period}")
                return pd.Series(index=prices.index)
            return prices.ewm(span=period, adjust=False).mean()
        except Exception as e:
            logger.error(f"EMA calculation error: {e}")
            return pd.Series(index=prices.index)
    
    @staticmethod
    def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Average True Range with proper error handling."""
        try:
            if len(high) < period or len(low) < period or len(close) < period:
                logger.warning(f"Insufficient data for ATR calculation")
                return pd.Series(index=close.index)
                
            tr1 = high - low
            tr2 = abs(high - close.shift())
            tr3 = abs(low - close.shift())
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(window=period).mean()
            
            logger.debug(f"ATR calculated successfully, last value: {atr.iloc[-1]:.2f}")
            return atr
            
        except Exception as e:
            logger.error(f"ATR calculation error: {e}")
            return pd.Series(index=close.index)
    
    @staticmethod
    def calculate_signal_momentum(indicator_series: pd.Series, lookback: int = 10) -> Dict:
        """Calculate momentum and rate of change for signal strength prediction."""
        try:
            if len(indicator_series) < lookback:
                logger.debug("Insufficient data for momentum calculation")
                return {"momentum": 0, "acceleration": 0, "trend_consistency": 0, "slope": 0}
            
            recent = indicator_series.tail(lookback)
            
            # Calculate momentum (rate of change)
            momentum = (recent.iloc[-1] - recent.iloc[0]) / max(lookback, 1)
            
            # Calculate acceleration (second derivative)
            if len(recent) >= 3:
                first_half_momentum = (recent.iloc[lookback//2] - recent.iloc[0]) / max(lookback//2, 1)
                second_half_momentum = (recent.iloc[-1] - recent.iloc[lookback//2]) / max(lookback//2, 1)
                acceleration = second_half_momentum - first_half_momentum
            else:
                acceleration = 0
            
            # Calculate trend consistency (R-squared of linear regression)
            try:
                x = np.arange(len(recent))
                y = recent.values
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
                trend_consistency = r_value ** 2  # R-squared
            except:
                trend_consistency = 0
                slope = 0
            
            result = {
                "momentum": momentum,
                "acceleration": acceleration,
                "trend_consistency": trend_consistency,
                "slope": slope
            }
            
            logger.debug(f"Momentum calculated: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Momentum calculation error: {e}")
            return {"momentum": 0, "acceleration": 0, "trend_consistency": 0, "slope": 0}
    
    @staticmethod
    def predict_signal_duration(indicators: Dict, ohlcv_df: pd.DataFrame) -> Dict:
        """
        Predict how long a signal is likely to hold based on indicator momentum.
        
        Returns:
            Dictionary with duration predictions
        """
        try:
            logger.debug("Calculating signal duration prediction")
            
            predictions = {
                "estimated_minutes": 0,
                "estimated_candles": 0,
                "confidence": "low",
                "strength_trend": "neutral",
                "key_levels": {},
                "momentum_status": {}
            }
            
            # Analyze MACD momentum
            if indicators.get("macd") and "histogram_series" in indicators["macd"]:
                macd_momentum = TechnicalIndicators.calculate_signal_momentum(
                    indicators["macd"]["histogram_series"]
                )
                predictions["momentum_status"]["macd"] = macd_momentum
            
            # Analyze RSI momentum
            if indicators.get("rsi") and "rsi_series" in indicators["rsi"]:
                rsi_momentum = TechnicalIndicators.calculate_signal_momentum(
                    indicators["rsi"]["rsi_series"]
                )
                predictions["momentum_status"]["rsi"] = rsi_momentum
            
            # Analyze Supertrend stability
            supertrend_stable_candles = 0
            if indicators.get("supertrend") and "direction_series" in indicators["supertrend"]:
                direction_series = indicators["supertrend"]["direction_series"].dropna()
                if len(direction_series) > 0:
                    current_direction = direction_series.iloc[-1]
                    
                    # Count consecutive candles in same direction
                    for i in range(len(direction_series) - 1, -1, -1):
                        if direction_series.iloc[i] == current_direction:
                            supertrend_stable_candles += 1
                        else:
                            break
            
            # Calculate average momentum across indicators
            total_momentum = 0
            total_consistency = 0
            indicator_count = 0
            
            for indicator_name, momentum_data in predictions["momentum_status"].items():
                if momentum_data:
                    total_momentum += abs(momentum_data.get("momentum", 0))
                    total_consistency += momentum_data.get("trend_consistency", 0)
                    indicator_count += 1
            
            if indicator_count > 0:
                avg_momentum = total_momentum / indicator_count
                avg_consistency = total_consistency / indicator_count
                
                # Estimate duration based on momentum and consistency
                base_duration = 5  # Base duration in candles
                
                # Momentum multiplier
                if avg_momentum > 0.5:
                    momentum_multiplier = 3
                elif avg_momentum > 0.3:
                    momentum_multiplier = 2
                elif avg_momentum > 0.1:
                    momentum_multiplier = 1.5
                else:
                    momentum_multiplier = 1
                
                # Consistency multiplier
                consistency_multiplier = 1 + avg_consistency
                
                # Stability bonus
                stability_bonus = min(supertrend_stable_candles / 10, 2)
                
                # Calculate final estimates
                estimated_candles = int(base_duration * momentum_multiplier * 
                                       consistency_multiplier * stability_bonus)
                estimated_candles = max(1, min(estimated_candles, 30))  # Cap between 1-30
                
                predictions["estimated_candles"] = estimated_candles
                predictions["estimated_minutes"] = estimated_candles  # For minute candles
                
                # Determine confidence level
                if avg_consistency > 0.7 and supertrend_stable_candles > 5:
                    predictions["confidence"] = "high"
                elif avg_consistency > 0.5 and supertrend_stable_candles > 3:
                    predictions["confidence"] = "medium"
                else:
                    predictions["confidence"] = "low"
                
                # Determine strength trend
                avg_acceleration = sum(m.get("acceleration", 0) for m in 
                                     predictions["momentum_status"].values()) / max(indicator_count, 1)
                
                if avg_acceleration > 0.01:
                    predictions["strength_trend"] = "strengthening"
                elif avg_acceleration < -0.01:
                    predictions["strength_trend"] = "weakening"
                else:
                    predictions["strength_trend"] = "stable"
            
            # Identify key support/resistance levels
            if len(ohlcv_df) >= 20:
                recent_highs = ohlcv_df['high'].tail(20)
                recent_lows = ohlcv_df['low'].tail(20)
                
                predictions["key_levels"] = {
                    "resistance": round(recent_highs.max(), 2),
                    "support": round(recent_lows.min(), 2),
                    "pivot": round((recent_highs.max() + recent_lows.min() + 
                                  ohlcv_df['close'].iloc[-1]) / 3, 2)
                }
            
            logger.debug(f"Duration prediction: {predictions}")
            return predictions
            
        except Exception as e:
            logger.error(f"Signal duration prediction error: {e}", exc_info=True)
            return {
                "estimated_minutes": 0,
                "estimated_candles": 0,
                "confidence": "low",
                "strength_trend": "neutral"
            }
    
    # [Keep all existing indicator methods: calculate_macd, calculate_rsi, etc.]
    # ... (rest of the indicator methods remain the same but with added logging)


class SignalGenerator:
    """Generate weighted trading signals with accuracy tracking."""
    
    def __init__(self):
        """Initialize signal generator with accuracy tracker."""
        self.accuracy_tracker = SignalAccuracyTracker()
        logger.info("SignalGenerator initialized with accuracy tracking")
    
    def calculate_weighted_signal_with_metrics(
        self, 
        indicators: Dict, 
        weights: Dict, 
        ohlcv_df: pd.DataFrame,
        current_price: float
    ) -> Dict:
        """
        Calculate weighted signal with accuracy metrics and duration prediction.
        
        Args:
            indicators: Dictionary of calculated indicators
            weights: Dictionary of indicator weights
            ohlcv_df: OHLC dataframe
            current_price: Current market price
            
        Returns:
            Dictionary with signal, metrics, and predictions
        """
        try:
            logger.debug("Calculating weighted signal with metrics")
            
            # Get base signal analysis
            signal_result = self._calculate_weighted_signal(indicators, weights)
            
            # Add duration prediction
            duration_prediction = TechnicalIndicators.predict_signal_duration(indicators, ohlcv_df)
            signal_result["duration_prediction"] = duration_prediction
            
            # Calculate accuracy metrics
            accuracy_metrics = self.accuracy_tracker.calculate_accuracy_metrics()
            signal_result["accuracy_metrics"] = accuracy_metrics
            
            # Track this signal
            self.accuracy_tracker.add_signal({
                "timestamp": pd.Timestamp.now(),
                "signal": signal_result["composite_signal"],
                "action": signal_result["action"],
                "confidence": signal_result["confidence"],
                "price": current_price,
                "predicted_duration": duration_prediction.get("estimated_candles", 0)
            })
            
            logger.info(f"Signal generated: {signal_result['composite_signal']} "
                       f"(Accuracy: {accuracy_metrics['signal_accuracy']:.1f}%)")
            
            return signal_result
            
        except Exception as e:
            logger.error(f"Error calculating signal with metrics: {e}", exc_info=True)
            return self._get_default_signal_result()
    
    def _calculate_weighted_signal(self, indicators: Dict, weights: Dict) -> Dict:
        """
        Calculate weighted signal from multiple indicators.
        
        Returns:
            Dictionary with composite signal and contributions
        """
        try:
            signal_values = {
                "strong_buy": 1.0,
                "buy": 0.75,
                "bullish": 0.5,
                "weak_buy": 0.25,
                "neutral": 0.0,
                "weak_sell": -0.25,
                "bearish": -0.5,
                "sell": -0.75,
                "strong_sell": -1.0,
                "overbought": -0.5,
                "oversold": 0.5
            }
            
            weighted_sum = 0
            contributions = {}
            active_indicators = 0
            
            for indicator_name, indicator_data in indicators.items():
                if indicator_name in weights and indicator_data:
                    signal = indicator_data.get("signal", "neutral")
                    signal_value = signal_values.get(signal, 0)
                    weight = weights.get(indicator_name, 0)
                    contribution = signal_value * weight
                    
                    weighted_sum += contribution
                    contributions[indicator_name] = {
                        "signal": signal,
                        "value": signal_value,
                        "weight": weight,
                        "contribution": round(contribution, 3)
                    }
                    
                    if signal != "neutral":
                        active_indicators += 1
            
            # Determine composite signal
            if weighted_sum >= 0.75:
                composite_signal = "STRONG_BUY"
                action = "buy"
            elif weighted_sum >= 0.5:
                composite_signal = "BUY"
                action = "buy"
            elif weighted_sum >= 0.25:
                composite_signal = "WEAK_BUY"
                action = "hold"
            elif weighted_sum <= -0.75:
                composite_signal = "STRONG_SELL"
                action = "sell"
            elif weighted_sum <= -0.5:
                composite_signal = "SELL"
                action = "sell"
            elif weighted_sum <= -0.25:
                composite_signal = "WEAK_SELL"
                action = "hold"
            else:
                composite_signal = "NEUTRAL"
                action = "hold"
            
            # Calculate confidence (0-100%)
            confidence = min(abs(weighted_sum) * 100, 100)
            
            result = {
                "composite_signal": composite_signal,
                "action": action,
                "weighted_score": round(weighted_sum, 3),
                "confidence": round(confidence, 1),
                "active_indicators": active_indicators,
                "contributions": contributions
            }
            
            logger.debug(f"Weighted signal calculated: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Error in weighted signal calculation: {e}")
            return self._get_default_signal_result()
    
    def _get_default_signal_result(self) -> Dict:
        """Get default signal result for error cases."""
        return {
            "composite_signal": "NEUTRAL",
            "action": "hold",
            "weighted_score": 0.0,
            "confidence": 0.0,
            "active_indicators": 0,
            "contributions": {},
            "accuracy_metrics": {
                "signal_accuracy": 50.0,
                "confidence_sustain": 50.0
            },
            "duration_prediction": {
                "estimated_minutes": 0,
                "estimated_candles": 0,
                "confidence": "low"
            }
        }
