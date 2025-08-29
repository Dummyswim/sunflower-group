"""
Signal duration and persistence prediction for RSI, MACD, VWAP.
Complete implementation with all methods.
"""
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass
from collections import deque
from datetime import datetime
from datetime import datetime, timezone, timedelta
import pytz  # if you prefer pytz

logger = logging.getLogger(__name__)

@dataclass
class SignalPersistence:
    """Data class for signal persistence information."""
    expected_candles: int
    expected_minutes: int
    confidence_level: float
    strength_decay_rate: float
    critical_levels: Dict[str, float]
    momentum_status: str
    volatility_assessment: str
    risk_factors: List[str]

class SignalDurationPredictor:
    """Predict signal duration for 5-15 minute windows."""
    
    def __init__(self):
        """Initialize predictor."""
        self.signal_history = deque(maxlen=500)
        self.prediction_accuracy = deque(maxlen=100)
        logger.info("SignalDurationPredictor initialized")
    
    def predict_signal_duration(self, df: pd.DataFrame, indicators: Dict, 
                               signal_result: Dict, timeframe_minutes: int = 5) -> SignalPersistence:
        """
        Predict signal duration optimized for 5-15 minute holding periods.
        """
        try:
            # Analyze each indicator's momentum
            rsi_analysis = self._analyze_rsi_momentum(indicators)
            macd_analysis = self._analyze_macd_momentum(indicators)
            vwap_analysis = self._analyze_vwap_trend(indicators)
            bb_analysis = self._analyze_bollinger_volatility(indicators)
            obv_analysis = self._analyze_obv_trend(indicators)
            
            # Composite analysis
            analyses = [rsi_analysis, macd_analysis, vwap_analysis, bb_analysis, obv_analysis]
            
            # Calculate expected duration
            momentum_scores = [a['momentum'] for a in analyses if 'momentum' in a]
            avg_momentum = np.mean(momentum_scores) if momentum_scores else 0.5
            
            # Determine expected candles (1-3 for 5-15 minutes)
            if avg_momentum > 0.8:
                expected_candles = 3  # 15 minutes
                confidence = 0.85
                momentum_status = "strong"
            elif avg_momentum > 0.6:
                expected_candles = 2  # 10 minutes
                confidence = 0.70
                momentum_status = "moderate"
            else:
                expected_candles = 1  # 5 minutes
                confidence = 0.50
                momentum_status = "weak"
            
            # Adjust for signal strength
            signal_strength = abs(signal_result.get('weighted_score', 0))
            if signal_strength > 0.7:
                confidence = min(confidence * 1.2, 1.0)
            elif signal_strength < 0.3:
                confidence = confidence * 0.8
            
            # Identify risk factors
            risk_factors = self._identify_risk_factors(analyses, indicators, df)
            
            # Calculate decay rate
            decay_rate = self._calculate_decay_rate(analyses)
            
            # Get critical levels
            critical_levels = self._get_critical_levels(indicators)
            
            # Assess volatility
            volatility = self._assess_volatility(indicators, df)
            
            persistence = SignalPersistence(
                expected_candles=expected_candles,
                expected_minutes=expected_candles * timeframe_minutes,
                confidence_level=confidence,
                strength_decay_rate=decay_rate,
                critical_levels=critical_levels,
                momentum_status=momentum_status,
                volatility_assessment=volatility,
                risk_factors=risk_factors
            )
            
            # Store for accuracy tracking
            self.signal_history.append({
                'timestamp': datetime.now(),
                'prediction': persistence,
                'signal': signal_result.get('composite_signal', 'UNKNOWN')
            })
            
            logger.debug(f"Duration prediction: {expected_candles} candles "
                        f"({expected_candles * timeframe_minutes} min), "
                        f"Confidence: {confidence:.1%}")
            
            return persistence
            
        except Exception as e:
            logger.error(f"Duration prediction error: {e}")
            return self._get_default_persistence(timeframe_minutes)
    
    def _analyze_rsi_momentum(self, indicators: Dict) -> Dict:
        """Analyze RSI momentum and trend."""
        try:
            rsi_series = indicators.get('rsi', pd.Series([50]))
            if len(rsi_series) < 3:
                return {'momentum': 0.5, 'trend': 'neutral'}
            
            current_rsi = rsi_series.iloc[-1]
            if np.isnan(current_rsi):
                return {'momentum': 0.5, 'trend': 'neutral'}
                
            rsi_change = current_rsi - rsi_series.iloc[-3]
            
            # Calculate momentum
            if current_rsi > 70 and rsi_change > 0:
                momentum = 0.9
            elif current_rsi < 30 and rsi_change < 0:
                momentum = 0.9
            elif 50 < current_rsi < 70 and rsi_change > 0:
                momentum = 0.7
            elif 30 < current_rsi < 50 and rsi_change < 0:
                momentum = 0.7
            else:
                momentum = 0.5
            
            trend = 'bullish' if rsi_change > 0 else 'bearish' if rsi_change < 0 else 'neutral'
            
            return {
                'momentum': momentum,
                'trend': trend,
                'value': current_rsi,
                'change': rsi_change
            }
            
        except Exception as e:
            logger.error(f"RSI momentum analysis error: {e}")
            return {'momentum': 0.5, 'trend': 'neutral'}
    
    def _analyze_macd_momentum(self, indicators: Dict) -> Dict:
        """Analyze MACD momentum and trend."""
        try:
            macd_data = indicators.get('macd', {})
            if not macd_data or 'hist' not in macd_data:
                return {'momentum': 0.5, 'trend': 'neutral'}
            
            hist_series = macd_data.get('hist', pd.Series([0]))
            if len(hist_series) < 3:
                return {'momentum': 0.5, 'trend': 'neutral'}
            
            current_hist = hist_series.iloc[-1]
            if np.isnan(current_hist):
                return {'momentum': 0.5, 'trend': 'neutral'}
                
            hist_change = current_hist - hist_series.iloc[-3]
            
            # Calculate momentum based on histogram
            momentum = min(abs(current_hist) * 10, 1.0)
            
            # Adjust for trend
            if current_hist > 0 and hist_change > 0:
                momentum = min(momentum * 1.2, 1.0)
            elif current_hist < 0 and hist_change < 0:
                momentum = min(momentum * 1.2, 1.0)
            
            trend = 'bullish' if current_hist > 0 else 'bearish' if current_hist < 0 else 'neutral'
            
            return {
                'momentum': momentum,
                'trend': trend,
                'histogram': current_hist,
                'change': hist_change
            }
            
        except Exception as e:
            logger.error(f"MACD momentum analysis error: {e}")
            return {'momentum': 0.5, 'trend': 'neutral'}
    
    def _analyze_vwap_trend(self, indicators: Dict) -> Dict:
        """Analyze VWAP trend strength."""
        try:
            vwap = indicators.get('vwap', pd.Series([0])).iloc[-1]
            price = indicators.get('price', 0)
            
            if vwap == 0 or np.isnan(vwap):
                return {'momentum': 0.5, 'trend': 'neutral'}
            
            # Calculate distance from VWAP
            distance_pct = abs((price - vwap) / vwap * 100)
            
            # Momentum based on distance
            if distance_pct > 1.0:
                momentum = 0.8
            elif distance_pct > 0.5:
                momentum = 0.6
            else:
                momentum = 0.4
            
            trend = 'bullish' if price > vwap else 'bearish' if price < vwap else 'neutral'
            
            return {
                'momentum': momentum,
                'trend': trend,
                'distance_pct': distance_pct
            }
            
        except Exception as e:
            logger.error(f"VWAP trend analysis error: {e}")
            return {'momentum': 0.5, 'trend': 'neutral'}
    
    def _analyze_bollinger_volatility(self, indicators: Dict) -> Dict:
        """Analyze Bollinger Bands for volatility."""
        try:
            bb = indicators.get('bollinger', {})
            if not bb or 'upper' not in bb:
                return {'momentum': 0.5, 'volatility': 'normal'}
            
            upper = bb.get('upper', pd.Series([0])).iloc[-1]
            lower = bb.get('lower', pd.Series([0])).iloc[-1]
            middle = bb.get('middle', pd.Series([0])).iloc[-1]
            price = indicators.get('price', 0)
            
            if middle == 0 or np.isnan(middle):
                return {'momentum': 0.5, 'volatility': 'unknown'}
            
            # Calculate band width
            band_width_pct = (upper - lower) / middle * 100
            
            # Position within bands
            if upper > lower:
                position = (price - lower) / (upper - lower)
            else:
                position = 0.5
            
            # Momentum based on position
            if position > 0.9 or position < 0.1:
                momentum = 0.8
            elif position > 0.7 or position < 0.3:
                momentum = 0.6
            else:
                momentum = 0.4
            
            # Volatility assessment
            if band_width_pct < 1:
                volatility = 'low'
            elif band_width_pct > 3:
                volatility = 'high'
            else:
                volatility = 'normal'
            
            return {
                'momentum': momentum,
                'volatility': volatility,
                'position': position,
                'band_width_pct': band_width_pct
            }
            
        except Exception as e:
            logger.error(f"Bollinger volatility analysis error: {e}")
            return {'momentum': 0.5, 'volatility': 'unknown'}
    
    def _analyze_obv_trend(self, indicators: Dict) -> Dict:
        """Analyze OBV trend strength."""
        try:
            obv_series = indicators.get('obv', pd.Series([0]))
            if len(obv_series) < 5:
                return {'momentum': 0.5, 'trend': 'neutral'}
            
            # Calculate OBV slope
            recent_obv = obv_series.iloc[-5:].values
            obv_slope = np.polyfit(range(len(recent_obv)), recent_obv, 1)[0]
            
            # Normalize slope
            momentum = min(abs(obv_slope) / 10000, 1.0)
            trend = 'bullish' if obv_slope > 0 else 'bearish' if obv_slope < 0 else 'neutral'
            
            return {
                'momentum': momentum,
                'trend': trend,
                'slope': obv_slope
            }
            
        except Exception as e:
            logger.error(f"OBV trend analysis error: {e}")
            return {'momentum': 0.5, 'trend': 'neutral'}
    
    def _identify_risk_factors(self, analyses: List[Dict], 
                               indicators: Dict, df: pd.DataFrame) -> List[str]:
        """Identify potential risk factors."""
        risk_factors = []
        
        try:
            # Check for divergence
            trends = [a.get('trend', 'neutral') for a in analyses]
            if len(set(trends)) > 1 and 'bullish' in trends and 'bearish' in trends:
                risk_factors.append("indicator_divergence")
            
            # Check volatility
            volatility_assessments = [a.get('volatility', 'normal') for a in analyses if 'volatility' in a]
            if 'high' in volatility_assessments:
                risk_factors.append("high_volatility")
            
            # Check RSI extremes
            rsi = indicators.get('rsi', pd.Series([50])).iloc[-1] if 'rsi' in indicators else 50
            if rsi > 80 or rsi < 20:
                risk_factors.append("rsi_extreme")
            
            # Check for weak momentum
            momentum_scores = [a.get('momentum', 0.5) for a in analyses]
            if np.mean(momentum_scores) < 0.4:
                risk_factors.append("weak_momentum")
            
            # Check recent price action
            if len(df) >= 5:
                recent_volatility = df['close'].iloc[-5:].std() / df['close'].iloc[-5:].mean()
                if recent_volatility > 0.02:
                    risk_factors.append("choppy_price_action")
            
        except Exception as e:
            logger.error(f"Risk factor identification error: {e}")
            risk_factors.append("analysis_error")
        
        return risk_factors
    
    def _calculate_decay_rate(self, analyses: List[Dict]) -> float:
        """Calculate expected signal strength decay rate."""
        try:
            momentum_scores = [a.get('momentum', 0.5) for a in analyses]
            avg_momentum = np.mean(momentum_scores) if momentum_scores else 0.5
            
            # Invert momentum to get decay rate
            decay_rate = 1.0 - avg_momentum
            
            # Ensure reasonable bounds
            decay_rate = max(0.1, min(0.9, decay_rate))
            
            return decay_rate
            
        except Exception as e:
            logger.error(f"Decay rate calculation error: {e}")
            return 0.5
    
    def _assess_volatility(self, indicators: Dict, df: pd.DataFrame) -> str:
        """Assess overall market volatility."""
        try:
            bb = indicators.get('bollinger', {})
            if not bb or 'upper' not in bb:
                return "unknown"
            
            upper = bb.get('upper', pd.Series([0])).iloc[-1]
            lower = bb.get('lower', pd.Series([0])).iloc[-1]
            middle = bb.get('middle', pd.Series([0])).iloc[-1]
            
            if middle > 0 and not np.isnan(middle):
                band_width_pct = (upper - lower) / middle * 100
                
                if band_width_pct < 1:
                    return "very_low"
                elif band_width_pct < 2:
                    return "low"
                elif band_width_pct < 3:
                    return "normal"
                elif band_width_pct < 4:
                    return "high"
                else:
                    return "very_high"
            
            return "unknown"
            
        except Exception as e:
            logger.error(f"Volatility assessment error: {e}")
            return "unknown"
    
    def _get_critical_levels(self, indicators: Dict) -> Dict[str, float]:
        """Get critical indicator levels."""
        try:
            levels = {}
            
            if 'rsi' in indicators:
                rsi_val = indicators['rsi'].iloc[-1]
                if not np.isnan(rsi_val):
                    levels['rsi'] = rsi_val
            
            if 'macd' in indicators and 'macd' in indicators['macd']:
                macd_val = indicators['macd']['macd'].iloc[-1]
                if not np.isnan(macd_val):
                    levels['macd'] = macd_val
            
            if 'vwap' in indicators:
                vwap_val = indicators['vwap'].iloc[-1]
                if not np.isnan(vwap_val):
                    levels['vwap'] = vwap_val
            
            if 'price' in indicators:
                levels['price'] = indicators['price']
            
            return levels
            
        except Exception as e:
            logger.error(f"Critical levels error: {e}")
            return {}
    
    def _get_default_persistence(self, timeframe_minutes: int) -> SignalPersistence:
        """Return default persistence on error."""
        return SignalPersistence(
            expected_candles=1,
            expected_minutes=timeframe_minutes,
            confidence_level=0.3,
            strength_decay_rate=0.5,
            critical_levels={},
            momentum_status="unknown",
            volatility_assessment="unknown",
            risk_factors=["prediction_error"]
        )


class EnhancedSignalFormatter:
    """Format signals for Telegram alerts with enhanced details."""
    
    @staticmethod
    def format_duration_alert(current_price: float, signal_result: Dict,
                              duration_prediction: SignalPersistence,
                              timestamp: datetime) -> str:
        """Format comprehensive alert message with duration prediction."""
        try:
            # Determine signal type
            signal_type = signal_result.get('composite_signal', 'UNKNOWN')
            if 'BUY' in signal_type:
                signal_marker = "[GREEN]"
                action = "BUY"
            elif 'SELL' in signal_type:
                signal_marker = "[RED]"
                action = "SELL"
            else:
                signal_marker = "[WHITE]"
                action = "HOLD"

            # Convert timestamp to IST
            ist = pytz.timezone("Asia/Kolkata")
            timestamp_ist = timestamp.astimezone(ist)

            # Format message
            message_parts = [
                f"{signal_marker} <b>{signal_type} Signal</b> {signal_marker}",
                "=" * 20,
                f"Price: Rs{current_price:.2f}",
                f"Action: <b>{action}</b>",
                "",
                "<b>Signal Metrics:</b>",
                f"* Strength: {abs(signal_result.get('weighted_score', 0)):.1%}",
                f"* Confidence: {signal_result.get('confidence', 0):.0f}%",
                f"* Active Indicators: {signal_result.get('active_indicators', 0)}/5",
                "",
                "<b>Duration Prediction:</b>",
                f"* Expected: {duration_prediction.expected_minutes} minutes",
                f"* Confidence: {duration_prediction.confidence_level:.0%}",
                f"* Momentum: {duration_prediction.momentum_status}",
                f"* Volatility: {duration_prediction.volatility_assessment}",
            ]
            
            # Add critical levels
            if duration_prediction.critical_levels:
                message_parts.extend([
                    "",
                    "<b>Key Levels:</b>",
                    f"* RSI: {duration_prediction.critical_levels.get('rsi', 0):.1f}",
                    f"* VWAP: Rs{duration_prediction.critical_levels.get('vwap', 0):.2f}",
                ])
            
            # Add risk factors if any
            if duration_prediction.risk_factors:
                message_parts.extend([
                    "",
                    "<b>Risk Factors:</b>",
                    *[f"* {risk}" for risk in duration_prediction.risk_factors[:3]]
                ])
            
            # Add indicator signals
            if 'signals' in signal_result:
                signals = signal_result['signals']
                message_parts.extend([
                    "",
                    "<b>Indicator Signals:</b>",
                    f"* RSI: {'BUY' if signals.get('rsi', 0) > 0 else 'SELL' if signals.get('rsi', 0) < 0 else 'NEUTRAL'}",
                    f"* MACD: {'BUY' if signals.get('macd', 0) > 0 else 'SELL' if signals.get('macd', 0) < 0 else 'NEUTRAL'}",
                    f"* VWAP: {'ABOVE' if signals.get('vwap', 0) > 0 else 'BELOW' if signals.get('vwap', 0) < 0 else 'AT'}",
                ])

            # Add timestamp in IST
            message_parts.extend([
                "",
                f"Time: {timestamp_ist.strftime('%H:%M:%S IST')}"
            ])
            
            return "\n".join(message_parts)
            
        except Exception as e:
            logger.error(f"Alert formatting error: {e}")
            return f"Signal Alert: {signal_result.get('composite_signal', 'UNKNOWN')} at Rs{current_price:.2f}"
