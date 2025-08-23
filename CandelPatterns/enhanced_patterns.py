"""
Enhanced pattern recognition with multiple candle patterns.
Inspired by successful trading bot repositories.
"""
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

class EnhancedPatternRecognition:
    """
    Advanced candlestick pattern recognition beyond TA-Lib.
    Includes custom patterns from successful trading strategies.
    """
    
    def __init__(self):
        """Initialize enhanced pattern recognition."""
        self.pattern_functions = {
            'three_line_strike': self._detect_three_line_strike,
            'inside_bar': self._detect_inside_bar,
            'pin_bar': self._detect_pin_bar,
            'fakey': self._detect_fakey,
            'two_bar_reversal': self._detect_two_bar_reversal,
            'key_reversal': self._detect_key_reversal,
            'island_reversal': self._detect_island_reversal,
            'hook_reversal': self._detect_hook_reversal,
            'three_drives': self._detect_three_drives,
            'abcd_pattern': self._detect_abcd_pattern
        }
        
        logger.info(f"Enhanced patterns initialized: {len(self.pattern_functions)} patterns")
    
    def detect_all_patterns(self, df: pd.DataFrame) -> List[Dict]:
        """Detect all custom patterns in the data."""
        if len(df) < 10:
            return []
        
        detections = []
        
        for pattern_name, detect_func in self.pattern_functions.items():
            try:
                result = detect_func(df)
                if result:
                    detections.append({
                        'name': pattern_name,
                        'type': result.get('type', 'neutral'),
                        'direction': result.get('direction', 'neutral'),
                        'strength': result.get('strength', 0.5),
                        'confidence': result.get('confidence', 0.5),
                        'description': result.get('description', '')
                    })
            except Exception as e:
                logger.debug(f"Pattern {pattern_name} detection failed: {e}")
        
        return detections
    
    def _detect_three_line_strike(self, df: pd.DataFrame) -> Optional[Dict]:
        """Detect Three Line Strike pattern."""
        if len(df) < 4:
            return None
        
        # Get last 4 candles
        last_4 = df.tail(4)
        
        # Check for 3 consecutive bullish/bearish candles followed by reversal
        first_3 = last_4.iloc[:3]
        last = last_4.iloc[-1]
        
        # Bullish three line strike
        if all(first_3['close'] > first_3['open']) and \
           last['open'] > first_3['close'].iloc[-1] and \
           last['close'] < first_3['open'].iloc[0]:
            return {
                'type': 'reversal',
                'direction': 'bearish',
                'strength': 0.8,
                'confidence': 0.75,
                'description': 'Three bullish candles followed by bearish engulfing'
            }
        
        # Bearish three line strike
        if all(first_3['close'] < first_3['open']) and \
           last['open'] < first_3['close'].iloc[-1] and \
           last['close'] > first_3['open'].iloc[0]:
            return {
                'type': 'reversal',
                'direction': 'bullish',
                'strength': 0.8,
                'confidence': 0.75,
                'description': 'Three bearish candles followed by bullish engulfing'
            }
        
        return None
    
    def _detect_inside_bar(self, df: pd.DataFrame) -> Optional[Dict]:
        """Detect Inside Bar pattern."""
        if len(df) < 2:
            return None
        
        prev = df.iloc[-2]
        curr = df.iloc[-1]
        
        # Current bar completely inside previous bar
        if curr['high'] < prev['high'] and curr['low'] > prev['low']:
            # Determine direction based on context
            trend = self._calculate_trend(df.tail(10))
            
            return {
                'type': 'continuation',
                'direction': 'bullish' if trend > 0 else 'bearish',
                'strength': 0.6,
                'confidence': 0.65,
                'description': 'Inside bar indicating consolidation'
            }
        
        return None
    
    def _detect_pin_bar(self, df: pd.DataFrame) -> Optional[Dict]:
        """Detect Pin Bar (Hammer/Shooting Star) pattern."""
        if len(df) < 1:
            return None
        
        candle = df.iloc[-1]
        body = abs(candle['close'] - candle['open'])
        upper_wick = candle['high'] - max(candle['close'], candle['open'])
        lower_wick = min(candle['close'], candle['open']) - candle['low']
        total_range = candle['high'] - candle['low']
        
        if total_range == 0:
            return None
        
        body_ratio = body / total_range
        
        # Bullish pin bar (hammer)
        if lower_wick > 2 * body and upper_wick < body and body_ratio < 0.3:
            return {
                'type': 'reversal',
                'direction': 'bullish',
                'strength': 0.7,
                'confidence': 0.7,
                'description': 'Bullish pin bar with long lower wick'
            }
        
        # Bearish pin bar (shooting star)
        if upper_wick > 2 * body and lower_wick < body and body_ratio < 0.3:
            return {
                'type': 'reversal',
                'direction': 'bearish',
                'strength': 0.7,
                'confidence': 0.7,
                'description': 'Bearish pin bar with long upper wick'
            }
        
        return None
    
    def _detect_fakey(self, df: pd.DataFrame) -> Optional[Dict]:
        """Detect Fakey pattern (false breakout of inside bar)."""
        if len(df) < 3:
            return None
        
        # Check for inside bar two candles ago
        if df.iloc[-2]['high'] < df.iloc[-3]['high'] and \
           df.iloc[-2]['low'] > df.iloc[-3]['low']:
            
            # Check for false breakout
            if df.iloc[-1]['close'] > df.iloc[-3]['high'] and \
               df.iloc[-1]['close'] < df.iloc[-1]['open']:
                return {
                    'type': 'reversal',
                    'direction': 'bearish',
                    'strength': 0.75,
                    'confidence': 0.7,
                    'description': 'Fakey pattern - false breakout above'
                }
            
            if df.iloc[-1]['close'] < df.iloc[-3]['low'] and \
               df.iloc[-1]['close'] > df.iloc[-1]['open']:
                return {
                    'type': 'reversal',
                    'direction': 'bullish',
                    'strength': 0.75,
                    'confidence': 0.7,
                    'description': 'Fakey pattern - false breakout below'
                }
        
        return None
    
    def _detect_two_bar_reversal(self, df: pd.DataFrame) -> Optional[Dict]:
        """Detect Two Bar Reversal pattern."""
        if len(df) < 2:
            return None
        
        prev = df.iloc[-2]
        curr = df.iloc[-1]
        
        # Bullish reversal
        if prev['close'] < prev['open'] and \
           curr['close'] > curr['open'] and \
           curr['close'] > prev['high']:
            return {
                'type': 'reversal',
                'direction': 'bullish',
                'strength': 0.65,
                'confidence': 0.6,
                'description': 'Two bar bullish reversal'
            }
        
        # Bearish reversal
        if prev['close'] > prev['open'] and \
           curr['close'] < curr['open'] and \
           curr['close'] < prev['low']:
            return {
                'type': 'reversal',
                'direction': 'bearish',
                'strength': 0.65,
                'confidence': 0.6,
                'description': 'Two bar bearish reversal'
            }
        
        return None
    
    def _detect_key_reversal(self, df: pd.DataFrame) -> Optional[Dict]:
        """Detect Key Reversal pattern."""
        if len(df) < 2:
            return None
        
        prev = df.iloc[-2]
        curr = df.iloc[-1]
        
        # Bullish key reversal
        if curr['low'] < prev['low'] and curr['close'] > prev['high']:
            return {
                'type': 'reversal',
                'direction': 'bullish',
                'strength': 0.8,
                'confidence': 0.75,
                'description': 'Bullish key reversal'
            }
        
        # Bearish key reversal
        if curr['high'] > prev['high'] and curr['close'] < prev['low']:
            return {
                'type': 'reversal',
                'direction': 'bearish',
                'strength': 0.8,
                'confidence': 0.75,
                'description': 'Bearish key reversal'
            }
        
        return None
    
    def _detect_island_reversal(self, df: pd.DataFrame) -> Optional[Dict]:
        """Detect Island Reversal pattern."""
        if len(df) < 3:
            return None
        
        # Check for gaps
        gap_up = df.iloc[-2]['low'] > df.iloc[-3]['high']
        gap_down = df.iloc[-2]['high'] < df.iloc[-3]['low']
        
        # Bearish island reversal
        if gap_up and df.iloc[-1]['high'] < df.iloc[-2]['low']:
            return {
                'type': 'reversal',
                'direction': 'bearish',
                'strength': 0.85,
                'confidence': 0.8,
                'description': 'Bearish island reversal'
            }
        
        # Bullish island reversal
        if gap_down and df.iloc[-1]['low'] > df.iloc[-2]['high']:
            return {
                'type': 'reversal',
                'direction': 'bullish',
                'strength': 0.85,
                'confidence': 0.8,
                'description': 'Bullish island reversal'
            }
        
        return None
    
    def _detect_hook_reversal(self, df: pd.DataFrame) -> Optional[Dict]:
        """Detect Hook Reversal pattern."""
        if len(df) < 2:
            return None
        
        prev = df.iloc[-2]
        curr = df.iloc[-1]
        
        # Bullish hook
        if prev['close'] < prev['open'] and \
           curr['open'] < prev['low'] and \
           curr['close'] > prev['close']:
            return {
                'type': 'reversal',
                'direction': 'bullish',
                'strength': 0.7,
                'confidence': 0.65,
                'description': 'Bullish hook reversal'
            }
        
        # Bearish hook
        if prev['close'] > prev['open'] and \
           curr['open'] > prev['high'] and \
           curr['close'] < prev['close']:
            return {
                'type': 'reversal',
                'direction': 'bearish',
                'strength': 0.7,
                'confidence': 0.65,
                'description': 'Bearish hook reversal'
            }
        
        return None
    
    def _detect_three_drives(self, df: pd.DataFrame) -> Optional[Dict]:
        """Detect Three Drives pattern."""
        if len(df) < 7:
            return None
        
        # Simplified three drives detection
        highs = df['high'].tail(7).values
        lows = df['low'].tail(7).values
        
        # Look for three successive higher highs or lower lows
        higher_highs = sum(1 for i in range(1, len(highs)) if highs[i] > highs[i-1])
        lower_lows = sum(1 for i in range(1, len(lows)) if lows[i] < lows[i-1])
        
        if higher_highs >= 3:
            return {
                'type': 'reversal',
                'direction': 'bearish',
                'strength': 0.75,
                'confidence': 0.7,
                'description': 'Three drives to top'
            }
        
        if lower_lows >= 3:
            return {
                'type': 'reversal',
                'direction': 'bullish',
                'strength': 0.75,
                'confidence': 0.7,
                'description': 'Three drives to bottom'
            }
        
        return None
    
    def _detect_abcd_pattern(self, df: pd.DataFrame) -> Optional[Dict]:
        """Detect ABCD harmonic pattern."""
        if len(df) < 4:
            return None
        
        # Simplified ABCD detection using price swings
        prices = df['close'].tail(4).values
        
        # Bullish ABCD
        if prices[0] > prices[1] and prices[2] > prices[1] and \
           prices[3] < prices[2] and abs(prices[2] - prices[1]) > abs(prices[3] - prices[2]):
            return {
                'type': 'reversal',
                'direction': 'bullish',
                'strength': 0.7,
                'confidence': 0.65,
                'description': 'Bullish ABCD pattern'
            }
        
        # Bearish ABCD
        if prices[0] < prices[1] and prices[2] < prices[1] and \
           prices[3] > prices[2] and abs(prices[1] - prices[2]) > abs(prices[2] - prices[3]):
            return {
                'type': 'reversal',
                'direction': 'bearish',
                'strength': 0.7,
                'confidence': 0.65,
                'description': 'Bearish ABCD pattern'
            }
        
        return None
    
    def _calculate_trend(self, df: pd.DataFrame) -> float:
        """Calculate trend direction using linear regression."""
        if len(df) < 2:
            return 0.0
        
        x = np.arange(len(df))
        y = df['close'].values
        
        # Simple linear regression
        coeffs = np.polyfit(x, y, 1)
        return coeffs[0]  # Slope indicates trend
