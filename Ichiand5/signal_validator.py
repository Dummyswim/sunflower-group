"""
Enhanced signal validation with detailed rejection reasons.
"""
import numpy as np
import pandas as pd
import logging
from typing import Dict, Tuple, List, Optional
from collections import deque
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class SignalValidator:
    """Validate signals with multiple criteria."""
    
    def __init__(self, config):
        """Initialize validator with configuration."""
        self.config = config
        self.signal_history = deque(maxlen=100)
        logger.info("SignalValidator initialized")
    
    def validate_signal(self, signal_result: Dict, df: pd.DataFrame, 
                       indicators: Dict) -> Tuple[bool, Dict]:
        """
        Validate signal with multiple checks.
        Returns (is_valid, validation_details).
        """
        try:
            validations = {}
            rejection_reasons = []
            
            # Volume Confirmation
            volume_check = self._validate_volume_confirmation(df)
            validations['volume'] = volume_check
            if not volume_check['passed']:
                rejection_reasons.append(f"Volume too low: {volume_check['recent_vol']:.0f}")
            
            # Trend Alignment
            trend_check = self._validate_trend_alignment(indicators, signal_result)
            validations['trend'] = trend_check
            if not trend_check['passed']:
                rejection_reasons.append("Trend misalignment")
            
            # Volatility Check
            volatility_check = self._validate_volatility_conditions(indicators)
            validations['volatility'] = volatility_check
            if not volatility_check['passed']:
                rejection_reasons.append(f"Volatility issue: {volatility_check.get('reason', 'unknown')}")
            
            # Momentum Quality
            momentum_check = self._validate_momentum_quality(indicators)
            validations['momentum'] = momentum_check
            if not momentum_check['passed']:
                rejection_reasons.append("Weak momentum")
            
            # Signal Consistency
            consistency_check = self._validate_signal_consistency(signal_result)
            validations['consistency'] = consistency_check
            if not consistency_check['passed']:
                rejection_reasons.append("Inconsistent signals")
            
            # Risk Management
            risk_check = self._validate_risk_management(signal_result, df)
            validations['risk'] = risk_check
            if not risk_check['passed']:
                rejection_reasons.append(f"Risk too high: {risk_check.get('reason', 'unknown')}")
            
            # Calculate overall validation score
            passed_count = sum(1 for v in validations.values() if v.get('passed', False))
            total_checks = len(validations)
            validation_score = passed_count / total_checks if total_checks > 0 else 0
            
            # Determine if valid
            is_valid = validation_score >= self.config.SIGNAL_VALIDATION_SCORE
            
            # Add to history
            self.signal_history.append({
                'timestamp': datetime.now(),
                'signal': signal_result.get('composite_signal', 'UNKNOWN'),
                'score': signal_result.get('weighted_score', 0),
                'validation_score': validation_score,
                'valid': is_valid
            })
            
            if is_valid:
                logger.info(f"Signal validated: {signal_result.get('composite_signal')}, "
                           f"Score: {validation_score:.2f}")
            else:
                logger.debug(f"Signal rejected: {', '.join(rejection_reasons)}")
            
            return is_valid, {
                'score': validation_score,
                'details': validations,
                'rejection_reasons': rejection_reasons,
                'passed_checks': passed_count,
                'total_checks': total_checks
            }
            
        except Exception as e:
            logger.error(f"Signal validation error: {e}")
            return False, {'score': 0, 'error': str(e)}
    
    def _validate_volume_confirmation(self, df: pd.DataFrame) -> Dict:
        """Validate volume is above average."""
        try:
            if len(df) < 10:
                return {"passed": True, "reason": "Insufficient data"}
            
            recent_vol = df['volume'].iloc[-3:].mean()
            avg_vol = df['volume'].iloc[-20:].mean()
            
            if avg_vol == 0:
                return {"passed": True, "reason": "No volume data"}
            
            passed = recent_vol >= avg_vol * self.config.VOLUME_MULTIPLIER
            
            return {
                "passed": passed,
                "recent_vol": recent_vol,
                "avg_vol": avg_vol,
                "multiplier": recent_vol / avg_vol if avg_vol > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"Volume validation error: {e}")
            return {"passed": True, "error": str(e)}
    
    def _validate_trend_alignment(self, indicators: Dict, 
                                  signal_result: Dict) -> Dict:
        """Validate signal aligns with trend indicators."""
        try:
            rsi = indicators.get('rsi', pd.Series([50])).iloc[-1] if 'rsi' in indicators else 50
            macd_data = indicators.get('macd', {})
            vwap = indicators.get('vwap', pd.Series([0])).iloc[-1] if 'vwap' in indicators else 0
            price = indicators.get('price', 0)
            signal_direction = signal_result.get('weighted_score', 0)
            
            # Get MACD values safely
            macd = macd_data.get('macd', pd.Series([0])).iloc[-1] if 'macd' in macd_data else 0
            macd_signal = macd_data.get('signal', pd.Series([0])).iloc[-1] if 'signal' in macd_data else 0
            
            # Check alignment
            bullish_aligned = (
                signal_direction > 0 and 
                rsi > 40 and 
                macd > macd_signal and 
                price > vwap
            )
            
            bearish_aligned = (
                signal_direction < 0 and 
                rsi < 60 and 
                macd < macd_signal and 
                price < vwap
            )
            
            neutral = abs(signal_direction) < 0.2
            
            passed = bullish_aligned or bearish_aligned or neutral
            
            return {
                "passed": passed,
                "rsi": rsi,
                "macd_vs_signal": macd - macd_signal,
                "price_vs_vwap": price - vwap,
                "alignment": "bullish" if bullish_aligned else "bearish" if bearish_aligned else "neutral"
            }
            
        except Exception as e:
            logger.error(f"Trend validation error: {e}")
            return {"passed": True, "error": str(e)}
    
    def _validate_volatility_conditions(self, indicators: Dict) -> Dict:
        """Validate volatility is within acceptable range."""
        try:
            bb = indicators.get('bollinger', {})
            if not bb or 'upper' not in bb:
                return {"passed": True, "reason": "No Bollinger data"}
            
            upper = bb.get('upper', pd.Series([0])).iloc[-1] if 'upper' in bb else 0
            lower = bb.get('lower', pd.Series([0])).iloc[-1] if 'lower' in bb else 0
            middle = bb.get('middle', pd.Series([0])).iloc[-1] if 'middle' in bb else 0
            
            if middle == 0:
                return {"passed": True, "reason": "Invalid middle band"}
            
            band_width = upper - lower
            band_width_pct = (band_width / middle) * 100
            
            too_low = band_width_pct < 0.5
            too_high = band_width_pct > 5.0
            
            passed = not (too_low or too_high)
            
            reason = ""
            if too_low:
                reason = "Volatility too low"
            elif too_high:
                reason = "Volatility too high"
            
            return {
                "passed": passed,
                "band_width": band_width,
                "band_width_pct": band_width_pct,
                "reason": reason
            }
            
        except Exception as e:
            logger.error(f"Volatility validation error: {e}")
            return {"passed": True, "error": str(e)}
    
    def _validate_momentum_quality(self, indicators: Dict) -> Dict:
        """Validate momentum indicators show strength."""
        try:
            rsi = indicators.get('rsi', pd.Series([50])).iloc[-1] if 'rsi' in indicators else 50
            macd_data = indicators.get('macd', {})
            macd_hist = macd_data.get('hist', pd.Series([0])).iloc[-1] if 'hist' in macd_data else 0
            
            rsi_momentum = abs(rsi - 50)
            macd_strength = abs(macd_hist)
            
            passed = rsi_momentum > 10 or macd_strength > 0.05
            
            return {
                "passed": passed,
                "rsi": rsi,
                "rsi_momentum": rsi_momentum,
                "macd_hist": macd_hist,
                "macd_strength": macd_strength
            }
            
        except Exception as e:
            logger.error(f"Momentum validation error: {e}")
            return {"passed": True, "error": str(e)}
    
    def _validate_signal_consistency(self, signal_result: Dict) -> Dict:
        """Validate signals are consistent across indicators."""
        try:
            signals = signal_result.get('signals', {})
            if not signals:
                return {"passed": True, "reason": "No signals"}
            
            bullish = sum(1 for v in signals.values() if v > 0)
            bearish = sum(1 for v in signals.values() if v < 0)
            neutral = sum(1 for v in signals.values() if v == 0)
            
            total_active = bullish + bearish
            
            if total_active == 0:
                passed = False
                reason = "No active signals"
            elif bullish > 0 and bearish > 0:
                ratio = max(bullish, bearish) / total_active
                passed = ratio >= 0.7
                reason = "Mixed signals" if not passed else "Dominant direction"
            else:
                passed = True
                reason = "Consistent signals"
            
            return {
                "passed": passed,
                "bullish": bullish,
                "bearish": bearish,
                "neutral": neutral,
                "reason": reason
            }
            
        except Exception as e:
            logger.error(f"Consistency validation error: {e}")
            return {"passed": True, "error": str(e)}

    def _validate_risk_management(self, signal_result: Dict, df: pd.DataFrame) -> Dict:
        """Validate risk parameters are acceptable."""
        try:
            if len(df) < 20:
                return {"passed": True, "reason": "Insufficient data"}
            
            recent_returns = df['close'].pct_change().iloc[-20:]
            volatility = recent_returns.std() * 100
            
            # FIX: Count only non-NEUTRAL signals from signal_history
            recent_signals = [s for s in self.signal_history 
                if (datetime.now() - s['timestamp']).seconds < 3600 
                and s.get('valid', True)  # Check if it WAS validated
                and s.get('signal') not in ['NEUTRAL', 'NO_SIGNAL', 'ERROR']
    ]
                        
            signal_frequency = len(recent_signals)
            
            too_volatile = volatility > 3.0
            too_frequent = signal_frequency > 10
            
            passed = not (too_volatile or too_frequent)
            
            reason = ""
            if too_volatile:
                reason = f"High volatility: {volatility:.2f}%"
            elif too_frequent:
                reason = f"Too frequent: {signal_frequency} signals/hour"
                
            logger.debug(f"Passed: {passed}, Reason: {reason}, Risk check - Volatility: {volatility:.2f}%, Frequency: {signal_frequency}/hr")
            return {
                "passed": passed,
                "volatility": volatility,
                "signal_frequency": signal_frequency,
                "reason": reason
            }
            
        except Exception as e:
            logger.error(f"Risk validation error: {e}")
            return {"passed": True, "error": str(e)}
    
    def get_validation_summary(self) -> Dict:
        """Get summary of recent validations."""
        if not self.signal_history:
            return {
                "total_signals": 0,
                "valid_signals": 0,
                "rejection_rate": 0,
                "avg_validation_score": 0
            }
        
        total = len(self.signal_history)
        valid = sum(1 for s in self.signal_history if s.get('valid', False))
        avg_score = np.mean([s.get('validation_score', 0) for s in self.signal_history])
        
        return {
            "total_signals": total,
            "valid_signals": valid,
            "rejection_rate": (total - valid) / total * 100 if total > 0 else 0,
            "avg_validation_score": avg_score
        }
