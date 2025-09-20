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
        
        df = df.replace([np.nan, np.inf, -np.inf], 0) 
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
            # logger.info(f"Pattern detected: {best['name']} - {best['signal']}")
            logger.debug(f"Pattern detected: {best['name']} - {best['signal']}")
            return best
        
        return {'pattern': 'none', 'signal': 'neutral', 'confidence': 0}
    
    # def _detect_three_soldiers(self, df: pd.DataFrame) -> bool:
    #     """Three consecutive green candles with increasing closes."""
    #     if len(df) < 3:
    #         return False
        
    #     last_3 = df.tail(3)
    #     all_green = all(last_3['close'] > last_3['open'])
    #     increasing = all(last_3['close'].diff().dropna() > 0)
        
    #     return all_green and increasing
    
    
    def _detect_three_soldiers(self, df: pd.DataFrame) -> bool: 
        """Three White Soldiers: 3 green candles; opens inside prior body; strong bodies; small upper wicks.""" 
        if len(df) < 3: 
            return False 
        
        last_3 = df.tail(3).copy()
        if not all(last_3['close'] > last_3['open']):
            return False
        if not all(last_3['close'].diff().dropna() > 0):
            return False

        for i in range(-2, 1):
            prev = last_3.iloc[i-1]
            cur = last_3.iloc[i]
            body_prev = abs(prev['close'] - prev['open'])
            range_prev = max(prev['high'] - prev['low'], 1e-6)
            if body_prev / range_prev < 0.5:
                return False
            lo_body = min(prev['open'], prev['close'])
            hi_body = max(prev['open'], prev['close'])
            if not (lo_body <= cur['open'] <= hi_body):
                return False
            cur_body = cur['close'] - cur['open']
            upper_wick = cur['high'] - max(cur['close'], cur['open'])
            if cur_body <= 0 or upper_wick > 0.5 * cur_body:
                return False
        logger.debug("Pattern OK: Three White Soldiers")
        return True

    
    
    def _detect_three_crows(self, df: pd.DataFrame) -> bool: 
        """Three Black Crows: 3 red candles; opens inside prior body; strong bodies; small lower wicks.""" 
        if len(df) < 3: 
            return False 
        
        last_3 = df.tail(3).copy()
    
        if not all(last_3['close'] < last_3['open']):
            return False
        if not all(last_3['close'].diff().dropna() < 0):
            return False
        for i in range(-2, 1):
            prev = last_3.iloc[i-1]
            cur = last_3.iloc[i]
            body_prev = abs(prev['close'] - prev['open'])
            range_prev = max(prev['high'] - prev['low'], 1e-6)
            if body_prev / range_prev < 0.5:
                return False
            lo_body = min(prev['open'], prev['close'])
            hi_body = max(prev['open'], prev['close'])
            if not (lo_body <= cur['open'] <= hi_body):
                return False
            
            cur_body = cur['open'] - cur['close'] 
            lower_wick = min(cur['open'], cur['close']) - cur['low'] 
            if cur_body <= 0 or lower_wick > 0.5 * cur_body: 
                return False
            
            
            
        logger.debug("Pattern OK: Three Black Crows")
        return True

    
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
        """Detect support/resistance test with longer momentum and breakout evidence.

        Rules:
        - At resistance + bullish momentum + breakout evidence at day high → breakout_potential (LONG)
        - At resistance without breakout evidence or with bearish momentum → resistance_test (SHORT)
        - At support + bearish momentum + breakdown evidence at day low → breakdown_potential (SHORT)
        - At support without breakdown evidence or with bullish momentum → support_test (LONG)
        """
        if len(df) < 20:
            return {'detected': False}

        window = df.tail(20).copy()
        recent_high = float(window['high'].max())
        recent_low = float(window['low'].min())
        current = float(df['close'].iloc[-1])

        # Proximity thresholds
        tol_res = 0.001
        tol_sup = 0.001
        near_resistance = (recent_high > 0) and (abs(current - recent_high) / recent_high < tol_res)
        near_support = (recent_low > 0) and (abs(current - recent_low) / recent_low < tol_sup)

        # Momentum: 20-bar slope + RSI(14)
        closes_short = df['close'].tail(12).astype(float).values
        closes_long = df['close'].tail(20).astype(float).values
        try:
            x_short = np.arange(len(closes_short))
            x_long = np.arange(len(closes_long))
            slope_short = float(np.polyfit(x_short, closes_short, 1)[0])
            slope_long = float(np.polyfit(x_long, closes_long, 1)[0])
        except Exception:
            slope_short, slope_long = 0.0, 0.0

        try:
            delta14 = pd.Series(df['close'].tail(30).astype(float).values).diff()
            gain14 = delta14.clip(lower=0).rolling(14).mean()
            loss14 = (-delta14.clip(upper=0)).rolling(14).mean().replace(0, 1e-10)
            rs14 = (gain14 / loss14).fillna(0)
            rsi14 = float((100 - (100 / (1 + rs14))).iloc[-1])
        except Exception:
            rsi14 = 50.0

        # Recent pressure
        recent = df.tail(2)
        recent_green = int((recent['close'] > recent['open']).sum())
        recent_red = 2 - recent_green

        # Momentum decision (longer horizon)
        bullish_mom = ((slope_long > 0 and rsi14 > 50) or (recent_green >= 2))
        bearish_mom = ((slope_long < 0 and rsi14 < 50) or (recent_red >= 2))

        # Day high/low context
        try:
            day_mask = df.index.date == df.index[-1].date()
            day_high = float(df.loc[day_mask, 'high'].max())
            day_low = float(df.loc[day_mask, 'low'].min())
        except Exception:
            day_high, day_low = recent_high, recent_low

        # Breakout evidence thresholds
        breakout_buf = 0.0005  # 0.05%
        last_close = float(df['close'].iloc[-1])
        last_high = float(df['high'].iloc[-1])
        breakout_evidence = (
            (last_close > recent_high * (1 + breakout_buf) or last_high > recent_high * (1 + breakout_buf))
            and slope_long > 0 and rsi14 >= 55
        )

        logger.debug(
            f"SR ctx → cur={current:.2f}, H20={recent_high:.2f}, L20={recent_low:.2f}, "
            f"dayH={day_high:.2f}, dayL={day_low:.2f}, near_res={near_resistance}, near_sup={near_support}, "
            f"slope_short={slope_short:.5f}, slope_long={slope_long:.5f}, RSI14={rsi14:.1f}, "
            f"breakout_ev={breakout_evidence}"
        )

        if near_resistance:
            if bullish_mom and breakout_evidence and abs(day_high - recent_high) / max(1.0, recent_high) < 0.002:
                logger.info("SR: Resistance + bullish momentum + breakout evidence at day high → breakout_potential (LONG)")
                return {
                    'detected': True, 'name': 'breakout_potential', 'signal': 'LONG',
                    'confidence': 80, 'level': recent_high, 'slope_long': slope_long, 'rsi14': rsi14
                }
            else:
                logger.info("SR: Resistance without breakout evidence → resistance_test (SHORT)")
                return {
                    'detected': True, 'name': 'resistance_test', 'signal': 'SHORT',
                    'confidence': 75, 'level': recent_high, 'slope_long': slope_long, 'rsi14': rsi14
                }

        if near_support:
            breakdown_evidence = (
                (last_close < recent_low * (1 - breakout_buf) or float(df['low'].iloc[-1]) < recent_low * (1 - breakout_buf))
                and slope_long < 0 and rsi14 <= 45
            )
            if bearish_mom and breakdown_evidence and abs(day_low - recent_low) / max(1.0, recent_low) < 0.002:
                logger.info("SR: Support + bearish momentum + breakdown evidence at day low → breakdown_potential (SHORT)")
                return {
                    'detected': True, 'name': 'breakdown_potential', 'signal': 'SHORT',
                    'confidence': 80, 'level': recent_low, 'slope_long': slope_long, 'rsi14': rsi14
                }
            else:
                logger.info("SR: Support without breakdown evidence → support_test (LONG)")
                return {
                    'detected': True, 'name': 'support_test', 'signal': 'LONG',
                    'confidence': 75, 'level': recent_low, 'slope_long': slope_long, 'rsi14': rsi14
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
