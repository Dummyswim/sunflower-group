"""
Signal duration and persistence prediction module.
Analyzes how long signals are likely to remain valid based on indicator dynamics.
"""
import numpy as np
import pandas as pd
import logging
from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
from scipy import stats
from collections import deque

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
    """
    Predicts how long trading signals are likely to remain valid
    based on multiple technical indicators and market dynamics.
    """
    
    def __init__(self):
        self.signal_history = deque(maxlen=500)
        self.prediction_accuracy = deque(maxlen=100)
        
    def predict_signal_duration(self, 
                               df: pd.DataFrame,
                               indicators: Dict,
                               signal_result: Dict,
                               timeframe_minutes: int = 1) -> SignalPersistence:
        """
        Predict how long a signal will remain valid.
        
        Args:
            df: OHLCV dataframe
            indicators: Dictionary of calculated indicators
            signal_result: Current signal analysis result
            timeframe_minutes: Minutes per candle
            
        Returns:
            SignalPersistence object with duration predictions
        """
        try:
            # Analyze each indicator's momentum and stability
            ichimoku_analysis = self._analyze_ichimoku_persistence(indicators.get('ichimoku', {}), df)
            stochastic_analysis = self._analyze_stochastic_momentum(indicators.get('stochastic', {}))
            obv_analysis = self._analyze_volume_sustainability(indicators.get('obv', {}), df)
            bollinger_analysis = self._analyze_volatility_expansion(indicators.get('bollinger', {}))
            adx_analysis = self._analyze_trend_strength_duration(indicators.get('adx', {}))
            atr_analysis = self._analyze_volatility_cycles(indicators.get('atr', {}))
            
            # Calculate composite persistence score
            persistence_data = self._calculate_composite_persistence(
                ichimoku_analysis,
                stochastic_analysis,
                obv_analysis,
                bollinger_analysis,
                adx_analysis,
                atr_analysis,
                signal_result
            )
            
            # Estimate signal duration
            expected_candles = self._estimate_candle_duration(persistence_data, df)
            expected_minutes = expected_candles * timeframe_minutes
            
            # Calculate strength decay rate
            decay_rate = self._calculate_decay_rate(persistence_data, df)
            
            # Identify critical levels
            critical_levels = self._identify_critical_levels(df, indicators)
            
            # Assess momentum and volatility
            momentum_status = self._assess_momentum_status(persistence_data)
            volatility_assessment = self._assess_volatility_impact(
                bollinger_analysis, 
                atr_analysis
            )
            
            # Identify risk factors
            risk_factors = self._identify_risk_factors(
                persistence_data,
                indicators,
                df
            )
            
            # Calculate confidence level
            confidence = self._calculate_prediction_confidence(
                persistence_data,
                signal_result
            )
            
            return SignalPersistence(
                expected_candles=expected_candles,
                expected_minutes=expected_minutes,
                confidence_level=confidence,
                strength_decay_rate=decay_rate,
                critical_levels=critical_levels,
                momentum_status=momentum_status,
                volatility_assessment=volatility_assessment,
                risk_factors=risk_factors
            )
            
        except Exception as e:
            logger.error(f"Signal duration prediction error: {e}")
            return SignalPersistence(
                expected_candles=0,
                expected_minutes=0,
                confidence_level=0,
                strength_decay_rate=1.0,
                critical_levels={},
                momentum_status="Unknown",
                volatility_assessment="Unknown",
                risk_factors=["Prediction error"]
            )
    
    def _analyze_ichimoku_persistence(self, ichimoku: Dict, df: pd.DataFrame) -> Dict:
        """Analyze Ichimoku Cloud for trend persistence."""
        try:
            if not ichimoku or 'tenkan' not in ichimoku:
                return {'persistence': 0, 'trend_strength': 0}
            
            tenkan = ichimoku['tenkan']
            kijun = ichimoku['kijun']
            senkou_a = ichimoku['senkou_a']
            senkou_b = ichimoku['senkou_b']
            
            # Calculate cloud thickness (volatility indicator)
            cloud_thickness = abs(senkou_a.iloc[-1] - senkou_b.iloc[-1])
            avg_price = df['close'].iloc[-1]
            relative_thickness = (cloud_thickness / avg_price) * 100
            
            # Analyze trend momentum
            tenkan_kijun_distance = abs(tenkan.iloc[-1] - kijun.iloc[-1])
            relative_distance = (tenkan_kijun_distance / avg_price) * 100
            
            # Check for trend consistency
            price_above_cloud = df['close'].iloc[-1] > max(senkou_a.iloc[-1], senkou_b.iloc[-1])
            price_below_cloud = df['close'].iloc[-1] < min(senkou_a.iloc[-1], senkou_b.iloc[-1])
            
            # Calculate trend persistence score
            persistence_score = 0
            if price_above_cloud or price_below_cloud:
                persistence_score += 0.4
            
            if relative_thickness > 1.0:  # Strong cloud
                persistence_score += 0.3
            
            if relative_distance > 0.5:  # Strong momentum
                persistence_score += 0.3
            
            # Check historical cloud breaks
            recent_breaks = 0
            for i in range(-10, -1):
                if i >= -len(df):
                    prev_above = df['close'].iloc[i-1] > max(senkou_a.iloc[i-1], senkou_b.iloc[i-1])
                    curr_above = df['close'].iloc[i] > max(senkou_a.iloc[i], senkou_b.iloc[i])
                    if prev_above != curr_above:
                        recent_breaks += 1
            
            # Adjust for stability
            if recent_breaks > 2:
                persistence_score *= 0.7
            
            return {
                'persistence': persistence_score,
                'trend_strength': relative_distance,
                'cloud_thickness': relative_thickness,
                'recent_breaks': recent_breaks,
                'position': 'above_cloud' if price_above_cloud else 'below_cloud' if price_below_cloud else 'in_cloud'
            }
            
        except Exception as e:
            logger.error(f"Ichimoku persistence analysis error: {e}")
            return {'persistence': 0, 'trend_strength': 0}
    
    def _analyze_stochastic_momentum(self, stochastic: Dict) -> Dict:
        """Analyze Stochastic for momentum sustainability."""
        try:
            if not stochastic or 'k' not in stochastic:
                return {'momentum': 0, 'reversal_risk': 0}
            
            k = stochastic['k']
            d = stochastic['d']
            
            # Current values
            k_current = k.iloc[-1]
            d_current = d.iloc[-1]
            
            # Calculate momentum strength
            momentum_strength = abs(k_current - 50) / 50
            
            # Check for divergence
            k_slope = (k.iloc[-1] - k.iloc[-5]) / 5 if len(k) >= 5 else 0
            d_slope = (d.iloc[-1] - d.iloc[-5]) / 5 if len(d) >= 5 else 0
            
            divergence = abs(k_slope - d_slope)
            
            # Assess reversal risk
            reversal_risk = 0
            if k_current > 80 or k_current < 20:  # Extreme zones
                reversal_risk += 0.4
            
            if divergence > 2:  # Significant divergence
                reversal_risk += 0.3
            
            # Check for momentum consistency
            consistent_direction = all(k.diff().iloc[-3:] > 0) or all(k.diff().iloc[-3:] < 0)
            
            momentum_persistence = momentum_strength * (1 - reversal_risk)
            if consistent_direction:
                momentum_persistence *= 1.2
            
            return {
                'momentum': momentum_persistence,
                'reversal_risk': reversal_risk,
                'k_value': k_current,
                'd_value': d_current,
                'divergence': divergence,
                'consistent': consistent_direction
            }
            
        except Exception as e:
            logger.error(f"Stochastic momentum analysis error: {e}")
            return {'momentum': 0, 'reversal_risk': 0}
    
    def _analyze_volume_sustainability(self, obv: Dict, df: pd.DataFrame) -> Dict:
        """Analyze OBV for volume trend sustainability."""
        try:
            if not obv or 'obv' not in obv:
                return {'volume_support': 0, 'trend_confirmation': 0}
            
            obv_line = obv['obv']
            obv_signal = obv['signal']
            
            # Calculate volume momentum
            obv_momentum = (obv_line.iloc[-1] - obv_line.iloc[-10]) / 10 if len(obv_line) >= 10 else 0
            
            # Check price-volume correlation
            price_change = (df['close'].iloc[-1] - df['close'].iloc[-10]) / df['close'].iloc[-10] if len(df) >= 10 else 0
            
            # Volume confirmation
            volume_price_alignment = 1 if (obv_momentum > 0 and price_change > 0) or \
                                          (obv_momentum < 0 and price_change < 0) else 0
            
            # Calculate volume trend strength
            obv_above_signal = obv_line.iloc[-1] > obv_signal.iloc[-1]
            signal_distance = abs(obv_line.iloc[-1] - obv_signal.iloc[-1]) / obv_signal.iloc[-1] if obv_signal.iloc[-1] != 0 else 0
            
            # Volume sustainability score
            sustainability = 0
            if volume_price_alignment:
                sustainability += 0.5
            
            if obv_above_signal:
                sustainability += 0.3
            
            if signal_distance > 0.05:  # Significant distance from signal
                sustainability += 0.2
            
            return {
                'volume_support': sustainability,
                'trend_confirmation': volume_price_alignment,
                'obv_momentum': obv_momentum,
                'signal_distance': signal_distance,
                'above_signal': obv_above_signal
            }
            
        except Exception as e:
            logger.error(f"Volume sustainability analysis error: {e}")
            return {'volume_support': 0, 'trend_confirmation': 0}
    
    def _analyze_volatility_expansion(self, bollinger: Dict) -> Dict:
        """Analyze Bollinger Bands for volatility cycles."""
        try:
            if not bollinger or 'upper' not in bollinger:
                return {'volatility_state': 'unknown', 'expansion_phase': 0}
            
            upper = bollinger['upper']
            lower = bollinger['lower']
            middle = bollinger['middle']
            width = bollinger['width']
            percent_b = bollinger['percent_b']
            
            # Current volatility state
            current_width = width.iloc[-1]
            avg_width = width.rolling(20).mean().iloc[-1]
            
            # Determine volatility phase
            if current_width < avg_width * 0.7:
                volatility_state = 'squeeze'
                expansion_potential = 0.8
            elif current_width > avg_width * 1.3:
                volatility_state = 'expansion'
                expansion_potential = 0.2
            else:
                volatility_state = 'normal'
                expansion_potential = 0.5
            
            # Band position analysis
            band_position = 'upper' if percent_b.iloc[-1] > 0.8 else 'lower' if percent_b.iloc[-1] < 0.2 else 'middle'
            
            # Calculate mean reversion probability
            reversion_probability = 0
            if percent_b.iloc[-1] > 1:  # Above upper band
                reversion_probability = min((percent_b.iloc[-1] - 1) * 2, 0.8)
            elif percent_b.iloc[-1] < 0:  # Below lower band
                reversion_probability = min(abs(percent_b.iloc[-1]) * 2, 0.8)
            
            return {
                'volatility_state': volatility_state,
                'expansion_phase': expansion_potential,
                'band_position': band_position,
                'reversion_probability': reversion_probability,
                'relative_width': current_width / avg_width if avg_width > 0 else 1
            }
            
        except Exception as e:
            logger.error(f"Volatility expansion analysis error: {e}")
            return {'volatility_state': 'unknown', 'expansion_phase': 0}
    
    def _analyze_trend_strength_duration(self, adx: Dict) -> Dict:
        """Analyze ADX for trend strength and expected duration."""
        try:
            if not adx or 'adx' not in adx:
                return {'trend_strength': 0, 'trend_maturity': 'unknown'}
            
            adx_line = adx['adx']
            plus_di = adx['plus_di']
            minus_di = adx['minus_di']
            
            current_adx = adx_line.iloc[-1]
            
            # Trend strength classification
            if current_adx > 40:
                trend_strength = 'very_strong'
                persistence_factor = 0.9
            elif current_adx > 25:
                trend_strength = 'strong'
                persistence_factor = 0.7
            elif current_adx > 20:
                trend_strength = 'moderate'
                persistence_factor = 0.5
            else:
                trend_strength = 'weak'
                persistence_factor = 0.2
            
            # Trend direction clarity
            di_spread = abs(plus_di.iloc[-1] - minus_di.iloc[-1])
            direction_clarity = min(di_spread / 20, 1.0)
            
            # ADX momentum (is trend strengthening or weakening?)
            adx_slope = (adx_line.iloc[-1] - adx_line.iloc[-5]) / 5 if len(adx_line) >= 5 else 0
            
            if adx_slope > 1:
                trend_maturity = 'developing'
                maturity_factor = 1.2
            elif adx_slope < -1:
                trend_maturity = 'weakening'
                maturity_factor = 0.7
            else:
                trend_maturity = 'stable'
                maturity_factor = 1.0
            
            # Calculate expected persistence
            expected_persistence = persistence_factor * direction_clarity * maturity_factor
            
            return {
                'trend_strength': trend_strength,
                'trend_maturity': trend_maturity,
                'persistence_score': expected_persistence,
                'adx_value': current_adx,
                'di_spread': di_spread,
                'adx_momentum': adx_slope
            }
            
        except Exception as e:
            logger.error(f"Trend strength duration analysis error: {e}")
            return {'trend_strength': 0, 'trend_maturity': 'unknown'}
    
    def _analyze_volatility_cycles(self, atr: Dict) -> Dict:
        """Analyze ATR for volatility cycles and risk assessment."""
        try:
            if not atr or 'atr' not in atr:
                return {'volatility_cycle': 'unknown', 'risk_level': 0}
            
            atr_line = atr['atr']
            atr_percent = atr['atr_percent']
            
            # Current volatility level
            current_atr_pct = atr_percent.iloc[-1]
            avg_atr_pct = atr_percent.rolling(20).mean().iloc[-1]
            
            # Volatility cycle phase
            volatility_ratio = current_atr_pct / avg_atr_pct if avg_atr_pct > 0 else 1
            
            if volatility_ratio < 0.7:
                volatility_cycle = 'low'
                breakout_potential = 0.7
            elif volatility_ratio > 1.3:
                volatility_cycle = 'high'
                breakout_potential = 0.3
            else:
                volatility_cycle = 'normal'
                breakout_potential = 0.5
            
            # Risk assessment
            risk_level = min(current_atr_pct / 2, 1.0)  # Normalize to 0-1
            
            # Volatility trend
            atr_slope = (atr_percent.iloc[-1] - atr_percent.iloc[-5]) / 5 if len(atr_percent) >= 5 else 0
            volatility_trend = 'increasing' if atr_slope > 0.1 else 'decreasing' if atr_slope < -0.1 else 'stable'
            
            return {
                'volatility_cycle': volatility_cycle,
                'risk_level': risk_level,
                'breakout_potential': breakout_potential,
                'volatility_trend': volatility_trend,
                'atr_ratio': volatility_ratio
            }
            
        except Exception as e:
            logger.error(f"Volatility cycles analysis error: {e}")
            return {'volatility_cycle': 'unknown', 'risk_level': 0}
    
    def _calculate_composite_persistence(self, *analyses) -> Dict:
        """Calculate composite persistence score from all analyses."""
        try:
            ichimoku, stochastic, obv, bollinger, adx, atr, signal_result = analyses
            
            # Weight individual persistence scores
            persistence_scores = {
                'ichimoku': ichimoku.get('persistence', 0) * 0.20,
                'stochastic': stochastic.get('momentum', 0) * 0.15,
                'obv': obv.get('volume_support', 0) * 0.15,
                'bollinger': (1 - bollinger.get('reversion_probability', 0)) * 0.15,
                'adx': adx.get('persistence_score', 0) * 0.20,
                'atr': atr.get('breakout_potential', 0) * 0.15
            }
            
            # Calculate total persistence
            total_persistence = sum(persistence_scores.values())
            
            # Adjust for signal strength
            signal_strength_multiplier = min(abs(signal_result.get('weighted_score', 0)) + 0.5, 1.5)
            adjusted_persistence = total_persistence * signal_strength_multiplier
            
            return {
                'total_persistence': adjusted_persistence,
                'individual_scores': persistence_scores,
                'ichimoku_analysis': ichimoku,
                'stochastic_analysis': stochastic,
                'obv_analysis': obv,
                'bollinger_analysis': bollinger,
                'adx_analysis': adx,
                'atr_analysis': atr
            }
            
        except Exception as e:
            logger.error(f"Composite persistence calculation error: {e}")
            return {'total_persistence': 0, 'individual_scores': {}}
    
    def _estimate_candle_duration(self, persistence_data: Dict, df: pd.DataFrame) -> int:
        """Estimate how many candles the signal will remain valid."""
        try:
            base_persistence = persistence_data.get('total_persistence', 0)
            
            # Base candle estimate (exponential scaling)
            if base_persistence >= 0.8:
                base_candles = 15
            elif base_persistence >= 0.6:
                base_candles = 10
            elif base_persistence >= 0.4:
                base_candles = 6
            elif base_persistence >= 0.2:
                base_candles = 3
            else:
                base_candles = 1
            
            # Adjust for trend strength
            adx_analysis = persistence_data.get('adx_analysis', {})
            if adx_analysis.get('trend_strength') == 'very_strong':
                base_candles = int(base_candles * 1.5)
            elif adx_analysis.get('trend_strength') == 'weak':
                base_candles = int(base_candles * 0.7)
            
            # Adjust for volatility
            atr_analysis = persistence_data.get('atr_analysis', {})
            if atr_analysis.get('volatility_cycle') == 'high':
                base_candles = int(base_candles * 0.8)
            elif atr_analysis.get('volatility_cycle') == 'low':
                base_candles = int(base_candles * 1.2)
            
            # Cap at reasonable limits
            return min(max(base_candles, 1), 30)
            
        except Exception as e:
            logger.error(f"Candle duration estimation error: {e}")
            return 1
    
    def _calculate_decay_rate(self, persistence_data: Dict, df: pd.DataFrame) -> float:
        """Calculate expected signal strength decay rate."""
        try:
            # Factors affecting decay rate
            volatility_factor = persistence_data.get('atr_analysis', {}).get('risk_level', 0.5)
            momentum_factor = persistence_data.get('stochastic_analysis', {}).get('reversal_risk', 0.5)
            trend_factor = 1 - (persistence_data.get('adx_analysis', {}).get('persistence_score', 0.5))
            
            # Calculate composite decay rate (higher = faster decay)
            decay_rate = (volatility_factor * 0.3 + momentum_factor * 0.4 + trend_factor * 0.3)
            
            # Normalize to percentage per candle
            decay_per_candle = decay_rate * 0.1  # 10% max decay per candle
            
            return min(decay_per_candle, 0.2)  # Cap at 20% per candle
            
        except Exception as e:
            logger.error(f"Decay rate calculation error: {e}")
            return 0.1
    
    def _identify_critical_levels(self, df: pd.DataFrame, indicators: Dict) -> Dict[str, float]:
        """Identify critical price levels that could invalidate the signal."""
        try:
            current_price = df['close'].iloc[-1]
            levels = {}
            
            # Bollinger Bands levels
            if 'bollinger' in indicators:
                levels['bollinger_upper'] = indicators['bollinger']['upper'].iloc[-1]
                levels['bollinger_lower'] = indicators['bollinger']['lower'].iloc[-1]
                levels['bollinger_middle'] = indicators['bollinger']['middle'].iloc[-1]
            
            # ATR-based stop levels
            if 'atr' in indicators:
                levels['atr_stop_long'] = indicators['atr']['stop_loss_long'].iloc[-1]
                levels['atr_stop_short'] = indicators['atr']['stop_loss_short'].iloc[-1]
            
            # Ichimoku levels
            if 'ichimoku' in indicators:
                levels['ichimoku_kijun'] = indicators['ichimoku']['kijun'].iloc[-1]
                levels['cloud_top'] = max(
                    indicators['ichimoku']['senkou_a'].iloc[-1],
                    indicators['ichimoku']['senkou_b'].iloc[-1]
                )
                levels['cloud_bottom'] = min(
                    indicators['ichimoku']['senkou_a'].iloc[-1],
                    indicators['ichimoku']['senkou_b'].iloc[-1]
                )
            
            # Recent support/resistance
            recent_high = df['high'].rolling(20).max().iloc[-1]
            recent_low = df['low'].rolling(20).min().iloc[-1]
            levels['recent_high'] = recent_high
            levels['recent_low'] = recent_low
            
            return levels
            
        except Exception as e:
            logger.error(f"Critical levels identification error: {e}")
            return {}
    
    def _assess_momentum_status(self, persistence_data: Dict) -> str:
        """Assess overall momentum status."""
        try:
            stoch = persistence_data.get('stochastic_analysis', {})
            adx = persistence_data.get('adx_analysis', {})
            
            momentum_score = stoch.get('momentum', 0)
            trend_strength = adx.get('adx_value', 0)
            
            if momentum_score > 0.7 and trend_strength > 25:
                return "Strong momentum - Signal likely to persist"
            elif momentum_score > 0.4 or trend_strength > 20:
                return "Moderate momentum - Signal stable for now"
            elif momentum_score < 0.2 and trend_strength < 20:
                return "Weak momentum - Signal may fade quickly"
            else:
                return "Mixed momentum - Monitor closely"
                
        except Exception as e:
            logger.error(f"Momentum assessment error: {e}")
            return "Unknown momentum status"
    
    def _assess_volatility_impact(self, bollinger_analysis: Dict, atr_analysis: Dict) -> str:
        """Assess volatility impact on signal duration."""
        try:
            vol_state = bollinger_analysis.get('volatility_state', 'unknown')
            vol_cycle = atr_analysis.get('volatility_cycle', 'unknown')
            
            if vol_state == 'squeeze' and vol_cycle == 'low':
                return "Low volatility - Breakout imminent, signal may accelerate"
            elif vol_state == 'expansion' and vol_cycle == 'high':
                return "High volatility - Signal may be short-lived"
            elif vol_state == 'normal' and vol_cycle == 'normal':
                return "Normal volatility - Standard signal duration expected"
            else:
                return f"Volatility: {vol_state}/{vol_cycle}"
                
        except Exception as e:
            logger.error(f"Volatility assessment error: {e}")
            return "Unknown volatility impact"
    
    def _identify_risk_factors(self, persistence_data: Dict, indicators: Dict, df: pd.DataFrame) -> List[str]:
        """Identify factors that could terminate the signal early."""
        risks = []
        
        try:
            # Check stochastic extremes
            stoch = persistence_data.get('stochastic_analysis', {})
            if stoch.get('reversal_risk', 0) > 0.6:
                risks.append("High reversal risk from overbought/oversold levels")
            
            # Check Bollinger Band extremes
            bb = persistence_data.get('bollinger_analysis', {})
            if bb.get('reversion_probability', 0) > 0.6:
                risks.append("Mean reversion likely from band extremes")
            
            # Check trend weakness
            adx = persistence_data.get('adx_analysis', {})
            if adx.get('trend_strength') == 'weak':
                risks.append("Weak trend may not sustain signal")
            
            # Check volume divergence
            obv = persistence_data.get('obv_analysis', {})
            if not obv.get('trend_confirmation', False):
                risks.append("Volume not confirming price movement")
            
            # Check Ichimoku cloud proximity
            ich = persistence_data.get('ichimoku_analysis', {})
            if ich.get('position') == 'in_cloud':
                risks.append("Price in Ichimoku cloud - unclear direction")
            
            # Check for high volatility
            atr = persistence_data.get('atr_analysis', {})
            if atr.get('risk_level', 0) > 0.7:
                risks.append("High volatility increases signal uncertainty")
            
            if not risks:
                risks.append("No significant risks identified")
                
        except Exception as e:
            logger.error(f"Risk identification error: {e}")
            risks.append("Unable to assess risks")
        
        return risks
    
    def _calculate_prediction_confidence(self, persistence_data: Dict, signal_result: Dict) -> float:
        """Calculate confidence level for the duration prediction."""
        try:
            # Base confidence from signal strength
            base_confidence = signal_result.get('confidence', 50)
            
            # Adjust for indicator agreement
            active_indicators = signal_result.get('active_indicators', 0)
            indicator_factor = min(active_indicators / 6, 1.0)
            
            # Adjust for trend clarity
            adx = persistence_data.get('adx_analysis', {})
            trend_clarity = 0
            if adx.get('trend_strength') in ['strong', 'very_strong']:
                trend_clarity = 0.2
            
            # Calculate final confidence
            confidence = base_confidence * indicator_factor + (trend_clarity * 100)
            
            return min(confidence, 95)  # Cap at 95%
            
        except Exception as e:
            logger.error(f"Confidence calculation error: {e}")
            return 0


class EnhancedSignalFormatter:
    """Format enhanced signals with duration predictions."""
    
    @staticmethod
    def format_duration_alert(current_price: float, 
                            signal_result: Dict,
                            duration_prediction: SignalPersistence,
                            timestamp: datetime) -> str:
        """
        Format alert message with signal duration predictions.
        """
        try:
            # Signal header
            signal_emoji = {
                "STRONG BUY": "🚀",
                "BUY": "📈",
                "STRONG SELL": "⚠️",
                "SELL": "📉",
                "NEUTRAL": "↔️"
            }.get(signal_result['composite_signal'], "❓")
            
            message = f"{signal_emoji} <b>{signal_result['composite_signal']} SIGNAL</b>\n"
            message += f"━━━━━━━━━━━━━━━━━━\n"
            message += f"💰 Price: ₹{current_price:,.2f}\n"
            message += f"📊 Score: {signal_result['weighted_score']:.3f}\n"
            message += f"🎯 Confidence: {signal_result['confidence']:.1f}%\n\n"
            
            # Duration prediction section
            message += "<b>⏱️ SIGNAL DURATION FORECAST:</b>\n"
            message += f"📍 Expected Duration: {duration_prediction.expected_candles} candles "
            message += f"(~{duration_prediction.expected_minutes} minutes)\n"
            message += f"📉 Strength Decay: {duration_prediction.strength_decay_rate:.1%} per candle\n"
            message += f"🎯 Prediction Confidence: {duration_prediction.confidence_level:.1f}%\n\n"
            
            # Momentum and volatility assessment
            message += "<b>📊 MARKET DYNAMICS:</b>\n"
            message += f"💪 {duration_prediction.momentum_status}\n"
            message += f"📈 {duration_prediction.volatility_assessment}\n\n"
            
            # Critical levels
            if duration_prediction.critical_levels:
                message += "<b>🎯 KEY LEVELS TO WATCH:</b>\n"
                for level_name, level_value in sorted(duration_prediction.critical_levels.items(), 
                                                     key=lambda x: x[1], reverse=True)[:5]:
                    level_label = level_name.replace('_', ' ').title()
                    message += f"• {level_label}: ₹{level_value:,.2f}\n"
                message += "\n"
            
            # Risk factors
            if duration_prediction.risk_factors:
                message += "<b>⚠️ RISK FACTORS:</b>\n"
                for risk in duration_prediction.risk_factors[:3]:
                    message += f"• {risk}\n"
                message += "\n"
            
            # Time-based recommendations
            message += "<b>⏰ RECOMMENDED ACTIONS:</b>\n"
            if duration_prediction.expected_candles <= 3:
                message += "• ⚡ Short-term signal - Consider quick action\n"
                message += "• 🔍 Monitor closely for reversal signs\n"
            elif duration_prediction.expected_candles <= 10:
                message += "• ⏱️ Medium-term signal - Normal position sizing\n"
                message += "• 📊 Review after 5 candles\n"
            else:
                message += "• 💎 Strong persistent signal - Can hold position\n"
                message += "• 📈 Trail stop-loss as signal progresses\n"
            
            # Decay timeline
            if duration_prediction.expected_candles > 1:
                message += f"\n<b>📉 SIGNAL STRENGTH TIMELINE:</b>\n"
                current_strength = abs(signal_result['weighted_score'])
                for i in [1, 3, 5, 10]:
                    if i <= duration_prediction.expected_candles:
                        projected_strength = current_strength * (1 - duration_prediction.strength_decay_rate * i)
                        message += f"• After {i} candles: {projected_strength:.2f} strength\n"
            
            message += f"\n⏰ {timestamp.strftime('%Y-%m-%d %H:%M:%S')}"
            
            return message
            
        except Exception as e:
            logger.error(f"Duration alert formatting error: {e}")
            return "Error formatting duration alert"
