"""
Candlestick Pattern Detection for Index Scalping
"""
import pandas as pd
import numpy as np
import logging
from typing import Dict 

logger = logging.getLogger(__name__)

class CandlestickPatternDetector:
    """Detects high-probability candlestick patterns for scalping."""
    
    def __init__(self):
        self.patterns = []
        
    def detect_patterns(self, df: pd.DataFrame) -> dict:
        """Detect actionable patterns in last 5 candles."""
        if len(df) < 5:
            return {'pattern': 'none', 'signal': 'neutral', 'confidence': 0}
        
        last_5 = df.tail(5)
        patterns_found = []
        
        # Pattern 1: Three White Soldiers / Three Black Crows
        if self._detect_three_soldiers(last_5):
            patterns_found.append({
                'name': 'three_white_soldiers',
                'signal': 'LONG',
                'confidence': 80
            })
        elif self._detect_three_crows(last_5):
            patterns_found.append({
                'name': 'three_black_crows', 
                'signal': 'SHORT',
                'confidence': 80
            })
        
        # Pattern 2: Momentum Burst
        momentum = self._detect_momentum_burst(last_5)
        if momentum['detected']:
            patterns_found.append(momentum)
        
        # Pattern 3: Support/Resistance Test
        sr_test = self._detect_sr_test(df)
        if sr_test['detected']:
            patterns_found.append(sr_test)
        
        # Pattern 4: Exhaustion
        exhaustion = self._detect_exhaustion(last_5)
        if exhaustion['detected']:
            patterns_found.append(exhaustion)
        
        if patterns_found:
            # Return highest confidence pattern
            best = max(patterns_found, key=lambda x: x['confidence'])
            logger.info(f"Pattern detected: {best['name']} - {best['signal']}")
            return best
        
        return {'pattern': 'none', 'signal': 'neutral', 'confidence': 0}
    
    def _detect_three_soldiers(self, df: pd.DataFrame) -> bool:
        """Three consecutive green candles with increasing closes."""
        if len(df) < 3:
            return False
        
        last_3 = df.tail(3)
        all_green = all(last_3['close'] > last_3['open'])
        increasing = all(last_3['close'].diff().dropna() > 0)
        
        return all_green and increasing
    
    def _detect_three_crows(self, df: pd.DataFrame) -> bool:
        """Three consecutive red candles with decreasing closes."""
        if len(df) < 3:
            return False
            
        last_3 = df.tail(3)
        all_red = all(last_3['close'] < last_3['open'])
        decreasing = all(last_3['close'].diff().dropna() < 0)
        
        return all_red and decreasing
    
    def _detect_momentum_burst(self, df: pd.DataFrame) -> dict:
        """Detect sudden momentum increase."""
        if len(df) < 5:
            return {'detected': False}
        
        # Calculate rate of change
        roc = (df['close'].iloc[-1] - df['close'].iloc[-3]) / df['close'].iloc[-3] * 100
        
        if abs(roc) > 0.15:  # 0.15% move in 3 candles
            return {
                'detected': True,
                'name': 'momentum_burst',
                'signal': 'LONG' if roc > 0 else 'SHORT',
                'confidence': min(90, 60 + abs(roc) * 100),
                'momentum': roc
            }
        
        return {'detected': False}
    
    def _detect_sr_test(self, df: pd.DataFrame) -> dict:
        """Detect support/resistance test."""
        if len(df) < 20:
            return {'detected': False}
        
        # Find recent highs/lows
        recent_high = df['high'].tail(20).max()
        recent_low = df['low'].tail(20).min()
        current = df['close'].iloc[-1]
        
        # Test resistance
        if abs(current - recent_high) / recent_high < 0.001:  # Within 0.1%
            return {
                'detected': True,
                'name': 'resistance_test',
                'signal': 'SHORT',
                'confidence': 70,
                'level': recent_high
            }
        
        # Test support
        if abs(current - recent_low) / recent_low < 0.001:  # Within 0.1%
            return {
                'detected': True,
                'name': 'support_test',
                'signal': 'LONG',
                'confidence': 70,
                'level': recent_low
            }
        
        return {'detected': False}
    
    def _detect_exhaustion(self, df: pd.DataFrame) -> dict:
        """Detect exhaustion after strong move."""
        if len(df) < 5:
            return {'detected': False}
        
        # Check for extended move
        move_5_candles = (df['close'].iloc[-1] - df['close'].iloc[-5]) / df['close'].iloc[-5] * 100
        
        # Check for declining momentum
        last_candle_range = df['high'].iloc[-1] - df['low'].iloc[-1]
        prev_candle_range = df['high'].iloc[-2] - df['low'].iloc[-2]
        
        if abs(move_5_candles) > 0.3 and last_candle_range < prev_candle_range * 0.5:
            return {
                'detected': True,
                'name': 'exhaustion',
                'signal': 'SHORT' if move_5_candles > 0 else 'LONG',
                'confidence': 75,
                'move': move_5_candles
            }
        
        return {'detected': False}


class ResistanceDetector:
    """Detects dynamic support and resistance levels from live data."""
    
    def detect_levels(self, df: pd.DataFrame, lookback: int = 100) -> Dict:
        """Detect support/resistance from actual price action."""
        if len(df) < lookback:
            lookback = len(df)
        
        levels = {
            'strong_resistance': [],
            'moderate_resistance': [],
            'strong_support': [],
            'moderate_support': []
        }
        
        # Method 1: Find swing highs/lows
        for i in range(2, lookback - 2):
            # Resistance (swing high)
            if (df['high'].iloc[-i] > df['high'].iloc[-i-1] and 
                df['high'].iloc[-i] > df['high'].iloc[-i-2] and
                df['high'].iloc[-i] > df['high'].iloc[-i+1] and 
                df['high'].iloc[-i] > df['high'].iloc[-i+2]):
                
                level = df['high'].iloc[-i]
                touches = self._count_touches(df, level, is_resistance=True)
                
                if touches >= 3:
                    levels['strong_resistance'].append(level)
                elif touches >= 2:
                    levels['moderate_resistance'].append(level)
            
            # Support (swing low)
            if (df['low'].iloc[-i] < df['low'].iloc[-i-1] and 
                df['low'].iloc[-i] < df['low'].iloc[-i-2] and
                df['low'].iloc[-i] < df['low'].iloc[-i+1] and 
                df['low'].iloc[-i] < df['low'].iloc[-i+2]):
                
                level = df['low'].iloc[-i]
                touches = self._count_touches(df, level, is_resistance=False)
                
                if touches >= 3:
                    levels['strong_support'].append(level)
                elif touches >= 2:
                    levels['moderate_support'].append(level)
        
        # Method 2: Round numbers (psychological levels)
        current_price = df['close'].iloc[-1]
        round_levels = [
            round(current_price / 100) * 100,  # Nearest 100
            round(current_price / 50) * 50,     # Nearest 50
            round(current_price / 25) * 25      # Nearest 25
        ]
        
        for level in round_levels:
            if abs(current_price - level) / current_price < 0.01:  # Within 1%
                if level > current_price:
                    levels['moderate_resistance'].append(level)
                else:
                    levels['moderate_support'].append(level)
        
        # Clean and sort
        for key in levels:
            levels[key] = sorted(list(set(levels[key])))
        
        # Find nearest levels
        nearest_resistance = min(levels['strong_resistance'] + levels['moderate_resistance'], 
                               default=current_price + 100, 
                               key=lambda x: abs(x - current_price) if x > current_price else float('inf'))
        nearest_support = max(levels['strong_support'] + levels['moderate_support'],
                            default=current_price - 100,
                            key=lambda x: abs(x - current_price) if x < current_price else float('-inf'))
        
        return {
            'levels': levels,
            'nearest_resistance': nearest_resistance,
            'nearest_support': nearest_support,
            'current_price': current_price
        }
    
    def _count_touches(self, df: pd.DataFrame, level: float, is_resistance: bool, tolerance: float = 0.001) -> int:
        """Count how many times price touched a level."""
        touches = 0
        price_range = level * tolerance
        
        for i in range(len(df)):
            if is_resistance:
                if abs(df['high'].iloc[i] - level) < price_range:
                    touches += 1
            else:
                if abs(df['low'].iloc[i] - level) < price_range:
                    touches += 1
        
        return touches
