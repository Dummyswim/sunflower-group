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
    
    def predict(self, 
                patterns: List[Dict],
                momentum: float,
                volume_profile: Dict,
                atr_ratio: float,
                support_resistance: Dict) -> Dict:
        """
        Generate prediction based on all signals.
        
        Returns:
            Dictionary with direction, confidence, and reasoning
        """
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

    
    # def _calculate_pattern_score(self, patterns: List[Dict]) -> float:
    #     """Calculate weighted score from pattern detections."""
    #     if not patterns:
    #         return 0.0
        
    #     score = 0.0
    #     total_weight = 0.0
        
    #     for pattern in patterns:
    #         # Weight by confidence and strength
    #         weight = pattern["confidence"] * pattern["strength"]
            
    #         # Adjust for pattern type
    #         if pattern["type"] == "reversal":
    #             weight *= 1.2  # Reversals are stronger signals
    #         elif pattern["type"] == "continuation":
    #             weight *= 0.8  # Continuations are weaker
            
    #         # Use theoretical_prob or confidence as the prior
    #         prior = pattern.get("theoretical_prob", pattern.get("confidence", 0.5))
    #         if pattern["direction"] == "bullish":
    #             score += weight * (prior - 0.5)
    #         else:
    #             score -= weight * (prior - 0.5)            
            
    #         total_weight += weight
        
    #     # Normalize
    #     if total_weight > 0:
    #         score /= total_weight
        
    #     return np.clip(score, -1, 1)
    
    def _calculate_momentum_score(self, momentum: float) -> float:
        """Convert momentum to normalized score."""
        # Sigmoid-like transformation
        return np.tanh(momentum * 10)  # Scale and bound to [-1, 1]
    
    def _calculate_volume_score(self, volume_profile: Dict) -> float:
        """Calculate score from volume analysis."""
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
