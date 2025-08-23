"""
Enhanced signal validation to prevent false positives.
"""
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional
from collections import deque
from scipy import stats

logger = logging.getLogger(__name__)

class SignalValidator:
    """Validate trading signals to reduce false positives."""
    
    def __init__(self, config):
        """Initialize signal validator."""
        self.config = config
        self.signal_history = deque(maxlen=100)
        self.false_positive_patterns = deque(maxlen=50)
        self.validation_stats = {
            'total_signals': 0,
            'validated_signals': 0,
            'rejected_signals': 0,
            'false_positives': 0
        }
    
    def validate_signal(self, 
                        signal_result: Dict,
                        df: pd.DataFrame,
                        indicators: Dict) -> Tuple[bool, Dict]:
        """
        Comprehensive signal validation.
        
        Returns:
            Tuple of (is_valid, validation_details)
        """
        try:
            validations = {
                'volume_confirmation': self._validate_volume_confirmation(df, signal_result),
                'trend_alignment': self._validate_trend_alignment(indicators, signal_result),
                'volatility_check': self._validate_volatility_conditions(indicators),
                'divergence_check': self._check_divergences(df, indicators),
                'support_resistance': self._validate_support_resistance(df, signal_result),
                'momentum_quality': self._validate_momentum_quality(indicators),
                'signal_consistency': self._validate_signal_consistency(signal_result)
            }
            
            # Calculate validation score
            validation_score = sum(v['passed'] for v in validations.values()) / len(validations)
            
            # Determine if signal is valid
            is_valid = (
                validation_score >= 0.6 and
                validations['volume_confirmation']['passed'] and
                validations['trend_alignment']['passed'] and
                not validations['divergence_check'].get('bearish_divergence', False)
            )
            
            # Track statistics
            self.validation_stats['total_signals'] += 1
            if is_valid:
                self.validation_stats['validated_signals'] += 1
            else:
                self.validation_stats['rejected_signals'] += 1
            
            # Store in history
            self.signal_history.append({
                'timestamp': pd.Timestamp.now(),
                'signal': signal_result.get('composite_signal'),
                'valid': is_valid,
                'score': validation_score,
                'validations': validations
            })
            
            validation_details = {
                'is_valid': is_valid,
                'validation_score': validation_score,
                'validations': validations,
                'rejection_reasons': self._get_rejection_reasons(validations),
                'confidence_adjustment': self._calculate_confidence_adjustment(validation_score)
            }
            
            logger.debug(f"Signal validation: {is_valid} (score: {validation_score:.2f})")
            
            return is_valid, validation_details
            
        except Exception as e:
            logger.error(f"Signal validation error: {e}")
            return False, {'error': str(e)}
    
    def _validate_volume_confirmation(self, df: pd.DataFrame, signal_result: Dict) -> Dict:
        """Validate that volume supports the signal."""
        try:
            recent_volume = df['volume'].tail(10)
            avg_volume = df['volume'].tail(50).mean()
            current_volume = df['volume'].iloc[-1]
            
            # Check volume surge
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
            volume_increasing = recent_volume.iloc[-1] > recent_volume.iloc[-5]
            
            # Volume should increase on breakouts
            if 'BUY' in signal_result.get('composite_signal', ''):
                passed = volume_ratio > 1.2 or volume_increasing
            elif 'SELL' in signal_result.get('composite_signal', ''):
                passed = volume_ratio > 1.1
            else:
                passed = True
            
            return {
                'passed': passed,
                'volume_ratio': volume_ratio,
                'volume_trend': 'increasing' if volume_increasing else 'decreasing',
                'message': f"Volume ratio: {volume_ratio:.2f}"
            }
            
        except Exception as e:
            logger.error(f"Volume validation error: {e}")
            return {'passed': False, 'error': str(e)}
    
    def _validate_trend_alignment(self, indicators: Dict, signal_result: Dict) -> Dict:
        """Check if signal aligns with overall trend."""
        try:
            trend_indicators = []
            
            # Check Ichimoku trend
            if 'ichimoku' in indicators:
                ich = indicators['ichimoku']
                if 'close' in ich:
                    price = ich['close'].iloc[-1]
                    cloud_top = max(ich['senkou_a'].iloc[-1], ich['senkou_b'].iloc[-1])
                    cloud_bottom = min(ich['senkou_a'].iloc[-1], ich['senkou_b'].iloc[-1])
                    
                    if price > cloud_top:
                        trend_indicators.append('bullish')
                    elif price < cloud_bottom:
                        trend_indicators.append('bearish')
                    else:
                        trend_indicators.append('neutral')
            
            # Check ADX trend
            if 'adx' in indicators:
                adx = indicators['adx']
                if adx['plus_di'].iloc[-1] > adx['minus_di'].iloc[-1]:
                    trend_indicators.append('bullish')
                else:
                    trend_indicators.append('bearish')
            
            # Determine consensus trend
            bullish_count = trend_indicators.count('bullish')
            bearish_count = trend_indicators.count('bearish')
            
            if bullish_count > bearish_count:
                overall_trend = 'bullish'
            elif bearish_count > bullish_count:
                overall_trend = 'bearish'
            else:
                overall_trend = 'neutral'
            
            # Check alignment
            signal = signal_result.get('composite_signal', '')
            aligned = (
                ('BUY' in signal and overall_trend == 'bullish') or
                ('SELL' in signal and overall_trend == 'bearish') or
                ('NEUTRAL' in signal)
            )
            
            return {
                'passed': aligned,
                'overall_trend': overall_trend,
                'trend_indicators': trend_indicators,
                'message': f"Trend: {overall_trend}, Signal: {signal}"
            }
            
        except Exception as e:
            logger.error(f"Trend alignment error: {e}")
            return {'passed': False, 'error': str(e)}
    
    def _validate_volatility_conditions(self, indicators: Dict) -> Dict:
        """Check if volatility conditions are favorable."""
        try:
            if 'atr' not in indicators or 'bollinger' not in indicators:
                return {'passed': True, 'message': 'Insufficient data'}
            
            atr = indicators['atr']
            bb = indicators['bollinger']
            
            # Check ATR levels
            current_atr_pct = atr['atr_percent'].iloc[-1]
            avg_atr_pct = atr['atr_percent'].rolling(20).mean().iloc[-1]
            
            # Check Bollinger Band width
            bb_width = bb['width'].iloc[-1]
            avg_bb_width = bb['width'].rolling(20).mean().iloc[-1]
            
            # Extreme volatility check
            extreme_volatility = (
                current_atr_pct > avg_atr_pct * 2 or
                bb_width > avg_bb_width * 2
            )
            
            # Volatility squeeze check
            volatility_squeeze = (
                bb_width < avg_bb_width * 0.5
            )
            
            # Favorable conditions: not extreme, potential breakout from squeeze
            favorable = not extreme_volatility
            
            return {
                'passed': favorable,
                'extreme_volatility': extreme_volatility,
                'volatility_squeeze': volatility_squeeze,
                'atr_ratio': current_atr_pct / avg_atr_pct if avg_atr_pct > 0 else 1,
                'bb_width_ratio': bb_width / avg_bb_width if avg_bb_width > 0 else 1,
                'message': 'Extreme volatility detected' if extreme_volatility else 'Normal volatility'
            }
            
        except Exception as e:
            logger.error(f"Volatility validation error: {e}")
            return {'passed': True, 'error': str(e)}
    
    def _check_divergences(self, df: pd.DataFrame, indicators: Dict) -> Dict:
        """Check for price-indicator divergences."""
        try:
            divergences = {}
            
            # Price trend
            price_trend = 'up' if df['close'].iloc[-1] > df['close'].iloc[-10] else 'down'
            
            # Check OBV divergence
            if 'obv' in indicators:
                obv = indicators['obv']['obv']
                obv_trend = 'up' if obv.iloc[-1] > obv.iloc[-10] else 'down'
                divergences['obv_divergence'] = price_trend != obv_trend
            
            # Check Stochastic divergence
            if 'stochastic' in indicators:
                stoch_k = indicators['stochastic']['k']
                stoch_trend = 'up' if stoch_k.iloc[-1] > stoch_k.iloc[-5] else 'down'
                divergences['stoch_divergence'] = price_trend != stoch_trend
            
            # Bearish divergence is particularly concerning
            bearish_divergence = (
                price_trend == 'up' and 
                any(divergences.values())
            )
            
            return {
                'passed': not bearish_divergence,
                'bearish_divergence': bearish_divergence,
                'divergences': divergences,
                'price_trend': price_trend,
                'message': 'Bearish divergence detected' if bearish_divergence else 'No concerning divergences'
            }
            
        except Exception as e:
            logger.error(f"Divergence check error: {e}")
            return {'passed': True, 'error': str(e)}
    
    def _validate_support_resistance(self, df: pd.DataFrame, signal_result: Dict) -> Dict:
        """Check proximity to support/resistance levels."""
        try:
            current_price = df['close'].iloc[-1]
            
            # Calculate recent support/resistance
            recent_high = df['high'].rolling(20).max().iloc[-1]
            recent_low = df['low'].rolling(20).min().iloc[-1]
            
            # Distance to levels
            distance_to_resistance = (recent_high - current_price) / current_price
            distance_to_support = (current_price - recent_low) / current_price
            
            # Check signal validity based on S/R
            signal = signal_result.get('composite_signal', '')
            
            if 'BUY' in signal:
                # Buy signals are better near support
                favorable = distance_to_support < 0.02  # Within 2% of support
            elif 'SELL' in signal:
                # Sell signals are better near resistance  
                favorable = distance_to_resistance < 0.02  # Within 2% of resistance
            else:
                favorable = True
            
            return {
                'passed': favorable,
                'resistance': recent_high,
                'support': recent_low,
                'distance_to_resistance': distance_to_resistance,
                'distance_to_support': distance_to_support,
                'message': f"Price: {current_price:.2f}, Support: {recent_low:.2f}, Resistance: {recent_high:.2f}"
            }
            
        except Exception as e:
            logger.error(f"Support/Resistance validation error: {e}")
            return {'passed': True, 'error': str(e)}
    
    def _validate_momentum_quality(self, indicators: Dict) -> Dict:
        """Assess momentum quality across indicators."""
        try:
            momentum_scores = []
            
            # ADX momentum
            if 'adx' in indicators:
                adx_value = indicators['adx']['adx'].iloc[-1]
                if adx_value > 25:
                    momentum_scores.append(1.0)
                elif adx_value > 20:
                    momentum_scores.append(0.5)
                else:
                    momentum_scores.append(0.0)
            
            # Stochastic momentum
            if 'stochastic' in indicators:
                k = indicators['stochastic']['k'].iloc[-1]
                if 30 < k < 70:  # Mid-range, good for continuation
                    momentum_scores.append(0.7)
                elif k < 20 or k > 80:  # Extreme, reversal risk
                    momentum_scores.append(0.3)
                else:
                    momentum_scores.append(0.5)
            
            # Average momentum quality
            avg_momentum = np.mean(momentum_scores) if momentum_scores else 0.5
            
            return {
                'passed': avg_momentum > 0.4,
                'momentum_quality': avg_momentum,
                'momentum_scores': momentum_scores,
                'message': f"Momentum quality: {avg_momentum:.2f}"
            }
            
        except Exception as e:
            logger.error(f"Momentum validation error: {e}")
            return {'passed': True, 'error': str(e)}
    
    def _validate_signal_consistency(self, signal_result: Dict) -> Dict:
        """Check internal consistency of signal components."""
        try:
            individual_signals = signal_result.get('individual_signals', {})
            
            # Count bullish vs bearish signals
            bullish = sum(1 for s in individual_signals.values() if s > 0)
            bearish = sum(1 for s in individual_signals.values() if s < 0)
            neutral = sum(1 for s in individual_signals.values() if s == 0)
            
            total = len(individual_signals)
            
            # Calculate consistency score
            if total > 0:
                max_agreement = max(bullish, bearish) / total
                consistency_score = max_agreement
            else:
                consistency_score = 0
            
            # Strong signals should have high consistency
            signal_type = signal_result.get('composite_signal', '')
            if 'STRONG' in signal_type:
                required_consistency = 0.7
            else:
                required_consistency = 0.5
            
            passed = consistency_score >= required_consistency
            
            return {
                'passed': passed,
                'consistency_score': consistency_score,
                'bullish_count': bullish,
                'bearish_count': bearish,
                'neutral_count': neutral,
                'required_consistency': required_consistency,
                'message': f"Consistency: {consistency_score:.2f} (required: {required_consistency})"
            }
            
        except Exception as e:
            logger.error(f"Consistency validation error: {e}")
            return {'passed': False, 'error': str(e)}
    
    def _get_rejection_reasons(self, validations: Dict) -> List[str]:
        """Extract reasons for signal rejection."""
        reasons = []
        
        for name, validation in validations.items():
            if not validation.get('passed', False):
                message = validation.get('message', f"{name} failed")
                reasons.append(message)
        
        return reasons
    
    def _calculate_confidence_adjustment(self, validation_score: float) -> float:
        """Calculate confidence adjustment based on validation."""
        # Linear adjustment: 0.5 validation = no change, 1.0 = +20%, 0.0 = -40%
        return (validation_score - 0.5) * 0.4
    
    def get_validation_stats(self) -> Dict:
        """Get validation statistics."""
        if self.validation_stats['total_signals'] > 0:
            accuracy = (self.validation_stats['validated_signals'] / 
                       self.validation_stats['total_signals']) * 100
        else:
            accuracy = 0
        
        return {
            **self.validation_stats,
            'validation_accuracy': accuracy,
            'recent_signals': list(self.signal_history)[-10:]
        }
