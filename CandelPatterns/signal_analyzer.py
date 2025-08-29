"""
Signal quality analyzer for calculating accuracy, sustenance, and duration metrics.
"""
import logging
from typing import Dict, List
import numpy as np
import pandas as pd
from collections import deque

logger = logging.getLogger(__name__)

class SignalQualityAnalyzer:
    """
    Analyzes signal quality including accuracy, confidence sustenance, and duration.
    """
    
    def __init__(self):
        """Initialize signal quality analyzer."""
        self.signal_history = deque(maxlen=500)
        self.pattern_durations = {}
        self.pattern_sustenance = {}
        self.active_signals = []
        
        # Performance tracking
        self.signal_outcomes = {
            'total': 0,
            'successful': 0,
            'failed': 0,
            'avg_duration': 0,
            'avg_sustenance': 0
        }
        
        logger.info("SignalQualityAnalyzer initialized")
    
    def calculate_signal_metrics(self, patterns: List[Dict], 
                                df: pd.DataFrame, 
                                prediction: Dict,
                                indicators: Dict) -> Dict:
        """
        Calculate comprehensive signal quality metrics.
        
        Returns:
            Dict containing:
            - accuracy_score: Expected accuracy of the signal (0-1)
            - confidence_sustenance: How long confidence will sustain (0-1)
            - expected_duration_minutes: Expected signal validity period
            - signal_strength: Overall signal strength (WEAK/MODERATE/STRONG)
            - risk_reward_ratio: Risk/reward calculation
        """
        metrics = {
            'accuracy_score': 0.0,
            'confidence_sustenance': 0.0,
            'expected_duration_minutes': 0,
            'signal_strength': 'WEAK',
            'risk_reward_ratio': 1.0,
            'entry_price': df['close'].iloc[-1] if len(df) > 0 else 0,
            'stop_loss': 0,
            'take_profit': 0
        }
        
        if not patterns or df.empty:
            return metrics
        
        # Calculate accuracy score
        metrics['accuracy_score'] = self._calculate_accuracy_score(patterns, indicators)
        
        # Calculate confidence sustenance
        metrics['confidence_sustenance'] = self._calculate_sustenance(
            patterns, prediction, df
        )
        
        # Predict signal duration
        metrics['expected_duration_minutes'] = self._predict_duration(
            patterns, df, indicators
        )
        
        # Calculate risk/reward and targets
        risk_reward = self._calculate_risk_reward(df, prediction, indicators)
        metrics.update(risk_reward)
        
        # Determine overall signal strength
        metrics['signal_strength'] = self._determine_signal_strength(metrics)
        
        # Track signal
        self._track_signal(patterns, metrics)
        
        return metrics
    
    def _calculate_accuracy_score(self, patterns: List[Dict], 
                                 indicators: Dict) -> float:
        """Calculate expected accuracy based on patterns and indicators."""
        if not patterns:
            return 0.0
        
        # Weight patterns by confidence and historical performance
        accuracy_scores = []
        weights = []
        
        for pattern in patterns:
            # Base accuracy from pattern confidence
            base_accuracy = pattern.get('confidence', 0.5)
            
            # Adjust for pattern type
            pattern_type = pattern.get('type', 'neutral')
            if pattern_type == 'reversal':
                type_multiplier = 1.1  # Reversals are typically more reliable
            elif pattern_type == 'continuation':
                type_multiplier = 0.9  # Continuations need trend confirmation
            else:
                type_multiplier = 1.0
            
            # Adjust for historical hit rate
            hit_rate = pattern.get('hit_rate', 0.5)
            historical_weight = 0.3 if pattern.get('sample_size', 0) > 20 else 0.1
            
            # Combine factors
            accuracy = (
                base_accuracy * type_multiplier * (1 - historical_weight) +
                hit_rate * historical_weight
            )
            
            accuracy_scores.append(accuracy)
            weights.append(pattern.get('strength', 0.5))
        
        # Calculate weighted average
        if sum(weights) > 0:
            weighted_accuracy = np.average(accuracy_scores, weights=weights)
        else:
            weighted_accuracy = np.mean(accuracy_scores)
        
        # Adjust for market conditions
        momentum = indicators.get('momentum', 0)
        if abs(momentum) > 0.02:  # Strong momentum
            weighted_accuracy *= 1.1
        elif abs(momentum) < 0.005:  # Low momentum
            weighted_accuracy *= 0.9
        
        return np.clip(weighted_accuracy, 0, 1)
    
    def _calculate_sustenance(self, patterns: List[Dict], 
                             prediction: Dict, 
                             df: pd.DataFrame) -> float:
        """Calculate how long the signal confidence will sustain."""
        if not patterns:
            return 0.0
        
        base_confidence = prediction.get('confidence', 0.5)
        
        # Analyze pattern types
        reversal_count = sum(1 for p in patterns if p.get('type') == 'reversal')
        continuation_count = sum(1 for p in patterns if p.get('type') == 'continuation')
        total_patterns = len(patterns)
        
        # Base sustenance on pattern type distribution
        if reversal_count > continuation_count:
            # Reversals typically have shorter but stronger signals
            type_factor = 0.7
        elif continuation_count > reversal_count:
            # Continuations can sustain longer in trending markets
            type_factor = 0.85
        else:
            type_factor = 0.75
        
        # Adjust for number of confirming patterns
        pattern_consensus = min(1.0, total_patterns / 3)  # Max benefit at 3+ patterns
        
        # Calculate volatility impact
        if len(df) >= 20:
            recent_volatility = df['close'].pct_change().tail(20).std()
            if recent_volatility > 0.02:  # High volatility
                volatility_factor = 0.8  # Reduces sustenance
            elif recent_volatility < 0.005:  # Low volatility
                volatility_factor = 1.1  # Increases sustenance
            else:
                volatility_factor = 1.0
        else:
            volatility_factor = 1.0
        
        # Combine factors
        sustenance = base_confidence * type_factor * pattern_consensus * volatility_factor
        
        # Check historical sustenance for these patterns
        for pattern in patterns[:3]:  # Top 3 patterns
            pattern_name = pattern.get('display_name', pattern.get('name'))
            if pattern_name in self.pattern_sustenance:
                historical_sustenance = np.mean(self.pattern_sustenance[pattern_name])
                sustenance = 0.7 * sustenance + 0.3 * historical_sustenance
        
        return np.clip(sustenance, 0, 1)

    def _predict_duration(self, patterns: List[Dict], 
                        df: pd.DataFrame,
                        indicators: Dict) -> int:
        """Predict how long the signal will remain valid in minutes."""
        # Base duration adjusted for 5-minute candles
        base_duration = 25  # Base 25 minutes (5 candles)
        
        # Adjust based on pattern types
        for pattern in patterns:
            pattern_name = pattern.get('display_name', pattern.get('name'))
            pattern_type = pattern.get('type', 'neutral')
            
            # Strong reversal patterns last longer
            if pattern_type == 'reversal' and pattern.get('confidence', 0) > 0.7:
                base_duration += 15  # 3 more candles
            # Continuation patterns in trending market
            elif pattern_type == 'continuation' and abs(indicators.get('momentum', 0)) > 0.01:
                base_duration += 10  # 2 more candles
            # Neutral patterns are shorter
            elif pattern_type == 'neutral':
                base_duration -= 10  # 2 fewer candles
            
            # Check historical duration for this pattern
            if pattern_name in self.pattern_durations:
                hist_durations = self.pattern_durations[pattern_name]
                if len(hist_durations) > 5:
                    hist_list = list(hist_durations)
                    avg_historical = np.mean(hist_list[-10:])
                    base_duration = int((base_duration + avg_historical) / 2)
        
        # Adjust for volatility
        atr = indicators.get('atr', 0)
        if atr and len(df) > 0:
            atr_pct = atr / df['close'].iloc[-1] if df['close'].iloc[-1] > 0 else 0
            
            if atr_pct > 0.02:  # High volatility = shorter duration
                base_duration = int(base_duration * 0.7)
            elif atr_pct < 0.005:  # Low volatility = longer duration
                base_duration = int(base_duration * 1.3)
        
        # Clamp between 10-120 minutes (2-24 candles)
        return max(10, min(120, base_duration))

    def _calculate_risk_reward(self, df: pd.DataFrame, 
                            prediction: Dict,
                            indicators: Dict) -> Dict:
        """Calculate risk/reward ratio and price targets."""
        if df.empty:
            return {'risk_reward_ratio': 1.0, 'stop_loss': 0, 'take_profit': 0}
        
        current_price = float(df['close'].iloc[-1])
        
        # Calculate ATR-based targets
        atr = indicators.get('atr', current_price * 0.01)
        
        # Get support/resistance levels with proper null checks
        sr = indicators.get('support_resistance', {})
        support = sr.get('support')
        resistance = sr.get('resistance')
        
        # Ensure valid support/resistance values
        if support is None or pd.isna(support):
            support = current_price * 0.98
        if resistance is None or pd.isna(resistance):
            resistance = current_price * 1.02
        
        if prediction['direction'] == 'bullish':
            # Calculate recent low with null check
            if len(df) >= 5:
                recent_low = df['low'].tail(5).min()
                if pd.isna(recent_low):
                    recent_low = current_price * 0.99
            else:
                recent_low = current_price * 0.99
            
            # Stop loss below recent low or support
            stop_loss = min(recent_low, support) * 0.995
            
            # Take profit at resistance or ATR-based target
            take_profit = max(resistance, current_price + (2 * atr))
            
        elif prediction['direction'] == 'bearish':
            # Calculate recent high with null check
            if len(df) >= 5:
                recent_high = df['high'].tail(5).max()
                if pd.isna(recent_high):
                    recent_high = current_price * 1.01
            else:
                recent_high = current_price * 1.01
            
            stop_loss = max(recent_high, resistance) * 1.005
            take_profit = min(support, current_price - (2 * atr))
        else:
            # Neutral direction
            return {'risk_reward_ratio': 1.0, 'stop_loss': 0, 'take_profit': 0}
        
        # Calculate risk and reward
        risk = abs(current_price - stop_loss)
        reward = abs(take_profit - current_price)
        
        # Calculate ratio
        risk_reward_ratio = reward / risk if risk > 0 else 1.0
        
        return {
            'risk_reward_ratio': round(risk_reward_ratio, 2),
            'stop_loss': round(stop_loss, 2),
            'take_profit': round(take_profit, 2)
        }


    # def _calculate_risk_reward(self, df: pd.DataFrame, 
    #                           prediction: Dict,
    #                           indicators: Dict) -> Dict:
    #     """Calculate risk/reward ratio and price targets."""
    #     if df.empty:
    #         return {'risk_reward_ratio': 1.0, 'stop_loss': 0, 'take_profit': 0}
        
    #     current_price = float(df['close'].iloc[-1])
        
    #     # Calculate ATR-based targets
    #     atr = indicators.get('atr', current_price * 0.01)  # Default 1% if no ATR
        
    #     # Get support/resistance levels
    #     sr = indicators.get('support_resistance', {})
    #     support = sr.get('support', current_price * 0.98)
    #     resistance = sr.get('resistance', current_price * 1.02)
        
    #     if prediction['direction'] == 'bullish':
    #         recent_low = df['low'].tail(5).min() if len(df) >= 5 else current_price * 0.98
    #             # Stop loss below recent low or support
    #         stop_loss = min(
    #             recent_low if recent_low is not None else current_price * 0.98,
    #             support if support is not None else current_price * 0.98
    #            ) * 0.995  # Small buffer
            
    #         # Take profit at resistance or ATR-based target
    #         take_profit = max(
    #             resistance,
    #             current_price + (2 * atr)
    #         )
            
    #     elif prediction['direction'] == 'bearish':
    #         # Stop loss above recent high or resistance
    #         stop_loss = max(
    #             df['high'].tail(5).max(),
    #             resistance
    #         ) * 1.005  # Small buffer
            
    #         # Take profit at support or ATR-based target
    #         take_profit = min(
    #             support,
    #             current_price - (2 * atr)
    #         )
    #     else:
    #         return {'risk_reward_ratio': 1.0, 'stop_loss': 0, 'take_profit': 0}
        
    #     # Calculate risk and reward
    #     risk = abs(current_price - stop_loss)
    #     reward = abs(take_profit - current_price)
        
    #     # Calculate ratio
    #     risk_reward_ratio = reward / risk if risk > 0 else 1.0
        
    #     return {
    #         'risk_reward_ratio': round(risk_reward_ratio, 2),
    #         'stop_loss': round(stop_loss, 2),
    #         'take_profit': round(take_profit, 2)
    #     }
    
    def _determine_signal_strength(self, metrics: Dict) -> str:
        """Determine overall signal strength based on all metrics."""
        # Calculate composite score
        score = 0
        
        # Accuracy component (40% weight)
        score += metrics['accuracy_score'] * 0.4
        
        # Sustenance component (30% weight)
        score += metrics['confidence_sustenance'] * 0.3
        
        # Risk/Reward component (20% weight)
        rr_score = min(1.0, metrics['risk_reward_ratio'] / 3)  # Normalize to 0-1
        score += rr_score * 0.2
        
        # Duration component (10% weight)
        duration_score = min(1.0, metrics['expected_duration_minutes'] / 30)
        score += duration_score * 0.1
        
        # Determine strength category
        if score > 0.7:
            return 'STRONG'
        elif score > 0.5:
            return 'MODERATE'
        else:
            return 'WEAK'
    
    def _track_signal(self, patterns: List[Dict], metrics: Dict):
        """Track signal for performance analysis."""
        signal_data = {
            'timestamp': pd.Timestamp.now(),
            'patterns': [p.get('display_name', p.get('name')) for p in patterns[:3]],
            'metrics': metrics.copy(),
            'active': True
        }
        
        self.signal_history.append(signal_data)
        self.active_signals.append(signal_data)
        
        # Clean up expired signals
        self._cleanup_expired_signals()
    
    def _cleanup_expired_signals(self):
        """Remove expired signals from active list."""
        current_time = pd.Timestamp.now()
        updated_active = []
        
        for signal in self.active_signals:
            elapsed_minutes = (current_time - signal['timestamp']).total_seconds() / 60
            expected_duration = signal['metrics'].get('expected_duration_minutes', 15)
            
            if elapsed_minutes < expected_duration:
                updated_active.append(signal)
            else:
                # Signal expired, update duration history
                for pattern_name in signal['patterns']:
                    if pattern_name not in self.pattern_durations:
                        self.pattern_durations[pattern_name] = deque(maxlen=50)
                    self.pattern_durations[pattern_name].append(elapsed_minutes)
        
        self.active_signals = updated_active
    
    def update_signal_outcome(self, was_successful: bool, actual_duration: int = None):
        """Update signal outcome for performance tracking."""
        self.signal_outcomes['total'] += 1
        
        if was_successful:
            self.signal_outcomes['successful'] += 1
        else:
            self.signal_outcomes['failed'] += 1
        
        if actual_duration:
            # Update average duration
            current_avg = self.signal_outcomes['avg_duration']
            total = self.signal_outcomes['total']
            self.signal_outcomes['avg_duration'] = (
                (current_avg * (total - 1) + actual_duration) / total
            )
    
    def get_performance_summary(self) -> Dict:
        """Get signal quality performance summary."""
        if self.signal_outcomes['total'] == 0:
            success_rate = 0
        else:
            success_rate = self.signal_outcomes['successful'] / self.signal_outcomes['total']
        
        return {
            'total_signals': self.signal_outcomes['total'],
            'success_rate': success_rate,
            'avg_duration': self.signal_outcomes['avg_duration'],
            'active_signals': len(self.active_signals),
            'pattern_durations': {
                pattern: np.mean(durations) if durations else 0
                for pattern, durations in self.pattern_durations.items()
            }
        }
