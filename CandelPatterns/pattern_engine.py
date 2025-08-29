"""
Advanced pattern recognition engine with all 61 TA-Lib patterns and probabilities.
Self-learning capabilities with rolling accuracy tracking.
"""
import json
import logging
import os
from collections import deque
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

try:
    import talib
    HAS_TALIB = True
except ImportError:
    HAS_TALIB = False
    logging.warning("TA-Lib not installed. Pattern detection disabled.")

logger = logging.getLogger(__name__)

class PatternEngine:
    """
    Enhanced pattern recognition with complete TA-Lib pattern support.
    """
    
    # Complete pattern mapping with display names
    PATTERNS = {
        'CDL2CROWS': 'Two Crows',
        'CDL3BLACKCROWS': 'Three Black Crows',
        'CDL3INSIDE': 'Three Inside Up/Down',
        'CDL3LINESTRIKE': 'Three-Line Strike',
        'CDL3OUTSIDE': 'Three Outside Up/Down',
        'CDL3STARSINSOUTH': 'Three Stars In The South',
        'CDL3WHITESOLDIERS': 'Three Advancing White Soldiers',
        'CDLABANDONEDBABY': 'Abandoned Baby',
        'CDLADVANCEBLOCK': 'Advance Block',
        'CDLBELTHOLD': 'Belt-hold',
        'CDLBREAKAWAY': 'Breakaway',
        'CDLCLOSINGMARUBOZU': 'Closing Marubozu',
        'CDLCONCEALBABYSWALL': 'Concealing Baby Swallow',
        'CDLCOUNTERATTACK': 'Counterattack',
        'CDLDARKCLOUDCOVER': 'Dark Cloud Cover',
        'CDLDOJI': 'Doji',
        'CDLDOJISTAR': 'Doji Star',
        'CDLDRAGONFLYDOJI': 'Dragonfly Doji',
        'CDLENGULFING': 'Engulfing Pattern',
        'CDLEVENINGDOJISTAR': 'Evening Doji Star',
        'CDLEVENINGSTAR': 'Evening Star',
        'CDLGAPSIDESIDEWHITE': 'Up/Down-gap side-by-side white lines',
        'CDLGRAVESTONEDOJI': 'Gravestone Doji',
        'CDLHAMMER': 'Hammer',
        'CDLHANGINGMAN': 'Hanging Man',
        'CDLHARAMI': 'Harami Pattern',
        'CDLHARAMICROSS': 'Harami Cross Pattern',
        'CDLHIGHWAVE': 'High-Wave Candle',
        'CDLHIKKAKE': 'Hikkake Pattern',
        'CDLHIKKAKEMOD': 'Modified Hikkake Pattern',
        'CDLHOMINGPIGEON': 'Homing Pigeon',
        'CDLIDENTICAL3CROWS': 'Identical Three Crows',
        'CDLINNECK': 'In-Neck Pattern',
        'CDLINVERTEDHAMMER': 'Inverted Hammer',
        'CDLKICKING': 'Kicking',
        'CDLKICKINGBYLENGTH': 'Kicking - bull/bear determined by the longer marubozu',
        'CDLLADDERBOTTOM': 'Ladder Bottom',
        'CDLLONGLEGGEDDOJI': 'Long Legged Doji',
        'CDLLONGLINE': 'Long Line Candle',
        'CDLMARUBOZU': 'Marubozu',
        'CDLMATCHINGLOW': 'Matching Low',
        'CDLMATHOLD': 'Mat Hold',
        'CDLMORNINGDOJISTAR': 'Morning Doji Star',
        'CDLMORNINGSTAR': 'Morning Star',
        'CDLONNECK': 'On-Neck Pattern',
        'CDLPIERCING': 'Piercing Pattern',
        'CDLRICKSHAWMAN': 'Rickshaw Man',
        'CDLRISEFALL3METHODS': 'Rising/Falling Three Methods',
        'CDLSEPARATINGLINES': 'Separating Lines',
        'CDLSHOOTINGSTAR': 'Shooting Star',
        'CDLSHORTLINE': 'Short Line Candle',
        'CDLSPINNINGTOP': 'Spinning Top',
        'CDLSTALLEDPATTERN': 'Stalled Pattern',
        'CDLSTICKSANDWICH': 'Stick Sandwich',
        'CDLTAKURI': 'Takuri (Dragonfly Doji with very long lower shadow)',
        'CDLTASUKIGAP': 'Tasuki Gap',
        'CDLTHRUSTING': 'Thrusting Pattern',
        'CDLTRISTAR': 'Tristar Pattern',
        'CDLUNIQUE3RIVER': 'Unique 3 River',
        'CDLUPSIDEGAP2CROWS': 'Upside Gap Two Crows',
        'CDLXSIDEGAP3METHODS': 'Upside/Downside Gap Three Methods'
    }
    
    # Historical probabilities for all patterns
    PATTERN_PROBABILITIES = {
        'Hammer': {'bullish': 0.68, 'bearish': 0.32, 'type': 'reversal'},
        'Shooting Star': {'bullish': 0.35, 'bearish': 0.65, 'type': 'reversal'},
        'Doji': {'bullish': 0.50, 'bearish': 0.50, 'type': 'neutral'},
        'Engulfing Pattern': {'bullish': 0.72, 'bearish': 0.28, 'type': 'reversal'},
        'Harami Pattern': {'bullish': 0.55, 'bearish': 0.45, 'type': 'reversal'},
        'Morning Star': {'bullish': 0.78, 'bearish': 0.22, 'type': 'reversal'},
        'Evening Star': {'bullish': 0.25, 'bearish': 0.75, 'type': 'reversal'},
        'Marubozu': {'bullish': 0.70, 'bearish': 0.30, 'type': 'continuation'},
        'Three Black Crows': {'bullish': 0.20, 'bearish': 0.80, 'type': 'reversal'},
        'Three Advancing White Soldiers': {'bullish': 0.82, 'bearish': 0.18, 'type': 'reversal'},
        'Two Crows': {'bullish': 0.28, 'bearish': 0.72, 'type': 'reversal'},
        'Three Inside Up/Down': {'bullish': 0.65, 'bearish': 0.35, 'type': 'reversal'},
        'Three-Line Strike': {'bullish': 0.83, 'bearish': 0.17, 'type': 'reversal'},
        'Three Outside Up/Down': {'bullish': 0.68, 'bearish': 0.32, 'type': 'reversal'},
        'Three Stars In The South': {'bullish': 0.74, 'bearish': 0.26, 'type': 'reversal'},
        'Abandoned Baby': {'bullish': 0.70, 'bearish': 0.30, 'type': 'reversal'},
        'Advance Block': {'bullish': 0.37, 'bearish': 0.63, 'type': 'reversal'},
        'Belt-hold': {'bullish': 0.66, 'bearish': 0.34, 'type': 'continuation'},
        'Breakaway': {'bullish': 0.63, 'bearish': 0.37, 'type': 'reversal'},
        'Closing Marubozu': {'bullish': 0.68, 'bearish': 0.32, 'type': 'continuation'},
        'Concealing Baby Swallow': {'bullish': 0.73, 'bearish': 0.27, 'type': 'reversal'},
        'Counterattack': {'bullish': 0.60, 'bearish': 0.40, 'type': 'reversal'},
        'Dark Cloud Cover': {'bullish': 0.32, 'bearish': 0.68, 'type': 'reversal'},
        'Doji Star': {'bullish': 0.52, 'bearish': 0.48, 'type': 'neutral'},
        'Dragonfly Doji': {'bullish': 0.69, 'bearish': 0.31, 'type': 'reversal'},
        'Evening Doji Star': {'bullish': 0.27, 'bearish': 0.73, 'type': 'reversal'},
        'Up/Down-gap side-by-side white lines': {'bullish': 0.64, 'bearish': 0.36, 'type': 'continuation'},
        'Gravestone Doji': {'bullish': 0.31, 'bearish': 0.69, 'type': 'reversal'},
        'Hanging Man': {'bullish': 0.33, 'bearish': 0.67, 'type': 'reversal'},
        'Harami Cross Pattern': {'bullish': 0.53, 'bearish': 0.47, 'type': 'reversal'},
        'High-Wave Candle': {'bullish': 0.48, 'bearish': 0.52, 'type': 'neutral'},
        'Hikkake Pattern': {'bullish': 0.58, 'bearish': 0.42, 'type': 'reversal'},
        'Modified Hikkake Pattern': {'bullish': 0.61, 'bearish': 0.39, 'type': 'reversal'},
        'Homing Pigeon': {'bullish': 0.71, 'bearish': 0.29, 'type': 'continuation'},
        'Identical Three Crows': {'bullish': 0.22, 'bearish': 0.78, 'type': 'reversal'},
        'In-Neck Pattern': {'bullish': 0.38, 'bearish': 0.62, 'type': 'continuation'},
        'Inverted Hammer': {'bullish': 0.65, 'bearish': 0.35, 'type': 'reversal'},
        'Kicking': {'bullish': 0.75, 'bearish': 0.25, 'type': 'reversal'},
        'Kicking - bull/bear determined by the longer marubozu': {'bullish': 0.73, 'bearish': 0.27, 'type': 'reversal'},
        'Ladder Bottom': {'bullish': 0.77, 'bearish': 0.23, 'type': 'reversal'},
        'Long Legged Doji': {'bullish': 0.49, 'bearish': 0.51, 'type': 'neutral'},
        'Long Line Candle': {'bullish': 0.67, 'bearish': 0.33, 'type': 'continuation'},
        'Matching Low': {'bullish': 0.72, 'bearish': 0.28, 'type': 'reversal'},
        'Mat Hold': {'bullish': 0.76, 'bearish': 0.24, 'type': 'continuation'},
        'Morning Doji Star': {'bullish': 0.76, 'bearish': 0.24, 'type': 'reversal'},
        'On-Neck Pattern': {'bullish': 0.36, 'bearish': 0.64, 'type': 'continuation'},
        'Piercing Pattern': {'bullish': 0.69, 'bearish': 0.31, 'type': 'reversal'},
        'Rickshaw Man': {'bullish': 0.50, 'bearish': 0.50, 'type': 'neutral'},
        'Rising/Falling Three Methods': {'bullish': 0.71, 'bearish': 0.29, 'type': 'continuation'},
        'Separating Lines': {'bullish': 0.64, 'bearish': 0.36, 'type': 'continuation'},
        'Short Line Candle': {'bullish': 0.51, 'bearish': 0.49, 'type': 'neutral'},
        'Spinning Top': {'bullish': 0.48, 'bearish': 0.52, 'type': 'neutral'},
        'Stalled Pattern': {'bullish': 0.35, 'bearish': 0.65, 'type': 'reversal'},
        'Stick Sandwich': {'bullish': 0.70, 'bearish': 0.30, 'type': 'reversal'},
        'Takuri (Dragonfly Doji with very long lower shadow)': {'bullish': 0.71, 'bearish': 0.29, 'type': 'reversal'},
        'Tasuki Gap': {'bullish': 0.62, 'bearish': 0.38, 'type': 'continuation'},
        'Thrusting Pattern': {'bullish': 0.39, 'bearish': 0.61, 'type': 'continuation'},
        'Tristar Pattern': {'bullish': 0.66, 'bearish': 0.34, 'type': 'reversal'},
        'Unique 3 River': {'bullish': 0.74, 'bearish': 0.26, 'type': 'reversal'},
        'Upside Gap Two Crows': {'bullish': 0.29, 'bearish': 0.71, 'type': 'reversal'},
        'Upside/Downside Gap Three Methods': {'bullish': 0.68, 'bearish': 0.32, 'type': 'continuation'}
    }
    
    def __init__(self, window: int = 100, default_prob: float = 0.55, 
                 history_file: Optional[str] = None):
        """Initialize enhanced pattern engine."""
        self.window = window
        self.default_prob = default_prob
        self.history_file = history_file
        
        # Initialize pattern functions
        self.pattern_funcs = {}
        if HAS_TALIB:
            for func_name in self.PATTERNS.keys():
                try:
                    self.pattern_funcs[func_name] = getattr(talib, func_name)
                except AttributeError:
                    logger.warning(f"Pattern function not found: {func_name}")
        
        # Initialize history tracking
        self.pattern_history: Dict[str, deque] = {
            name: deque(maxlen=window) for name in self.PATTERNS.keys()
        }
        
        # Performance tracking
        self.pattern_performance = {
            name: {'detected': 0, 'correct': 0, 'accuracy': 0.0}
            for name in self.PATTERNS.keys()
        }
        
        # Load historical data
        self._load_history()
        
        logger.info(f"PatternEngine initialized with {len(self.pattern_funcs)} patterns")
    
    def detect_patterns(self, df: pd.DataFrame) -> List[Dict]:
        """
        Detect all candlestick patterns with enhanced metadata.
        """
        if not HAS_TALIB or len(df) < 5:
            return []
        
        detections = []
        
        # Prepare OHLC arrays
        try:
            o = df['open'].values.astype(float)
            h = df['high'].values.astype(float)
            l = df['low'].values.astype(float)
            c = df['close'].values.astype(float)
        except Exception as e:
            logger.error(f"Data preparation error: {e}")
            return []
        
        for func_name, func in self.pattern_funcs.items():
            try:
                # Execute pattern detection
                result = func(o, h, l, c)
                
                # Check last value
                if isinstance(result, np.ndarray) and len(result) > 0:
                    value = int(result[-1])
                else:
                    continue
                
                if value != 0:  # Pattern detected
                    # Get pattern metadata
                    display_name = self.PATTERNS[func_name]
                    probs = self.PATTERN_PROBABILITIES.get(display_name, {})
                    
                    # Determine direction from value and probabilities
                    if value > 0:  # Bullish signal
                        direction = "bullish"
                        confidence_base = probs.get('bullish', 0.55)
                    else:  # Bearish signal
                        direction = "bearish"
                        confidence_base = probs.get('bearish', 0.55)
                    
                    # Get historical performance
                    hit_rate = self.get_hit_rate(func_name)
                    sample_size = len(self.pattern_history.get(func_name, []))
                    
                    # Calculate adjusted confidence
                    if sample_size >= 20:
                        # Blend historical and theoretical
                        confidence = 0.7 * confidence_base + 0.3 * hit_rate
                    elif sample_size >= 10:
                        confidence = 0.85 * confidence_base + 0.15 * hit_rate
                    else:
                        confidence = confidence_base
                    
                    detection = {
                        "name": func_name,
                        "display_name": display_name,
                        "value": value,
                        "direction": direction,
                        "type": probs.get('type', 'neutral'),
                        "confidence": confidence,
                        "theoretical_prob": confidence_base,
                        "hit_rate": hit_rate,
                        "sample_size": sample_size,
                        "strength": abs(value) / 100.0,
                        "timestamp": df.index[-1] if hasattr(df.index[-1], 'strftime') else str(df.index[-1])
                    }
                    
                    detections.append(detection)
                    self.pattern_performance[func_name]['detected'] += 1
                    
            except Exception as e:
                logger.debug(f"Pattern {func_name} detection error: {e}")
                continue
        
        # Sort by confidence
        detections.sort(key=lambda x: x['confidence'], reverse=True)
        
        return detections
    
    def update_pattern_performance(self, pattern_name: str, was_correct: bool):
        """Update pattern performance with actual results."""
        if pattern_name not in self.pattern_history:
            self.pattern_history[pattern_name] = deque(maxlen=self.window)
        
        # Update history
        self.pattern_history[pattern_name].append(1 if was_correct else 0)
        
        # Update performance stats
        if pattern_name in self.pattern_performance:
            if was_correct:
                self.pattern_performance[pattern_name]['correct'] += 1
            
            total = self.pattern_performance[pattern_name]['detected']
            correct = self.pattern_performance[pattern_name]['correct']
            
            if total > 0:
                self.pattern_performance[pattern_name]['accuracy'] = correct / total
        
        logger.debug(f"Updated {pattern_name}: {'✓' if was_correct else '✗'} "
                    f"(Accuracy: {self.get_hit_rate(pattern_name):.1%})")
    
    def get_hit_rate(self, pattern_name: str) -> float:
        """Get rolling hit rate for a pattern."""
        history = self.pattern_history.get(pattern_name, [])
        if len(history) < 5:
            # Use theoretical probability if insufficient data
            display_name = self.PATTERNS.get(pattern_name, pattern_name)
            probs = self.PATTERN_PROBABILITIES.get(display_name, {})
            return max(probs.get('bullish', 0.5), probs.get('bearish', 0.5))
        
        return float(np.mean(history))
    
    def get_pattern_statistics(self) -> Dict:
        """Get comprehensive pattern statistics."""
        stats = {}
        
        for pattern_name in self.PATTERNS.keys():
            display_name = self.PATTERNS[pattern_name]
            history = self.pattern_history.get(pattern_name, [])
            perf = self.pattern_performance.get(pattern_name, {})
            
            stats[display_name] = {
                'detected_count': perf.get('detected', 0),
                'accuracy': perf.get('accuracy', 0.0),
                'hit_rate': self.get_hit_rate(pattern_name),
                'sample_size': len(history),
                'theoretical_prob': self.PATTERN_PROBABILITIES.get(display_name, {})
            }
        
        return stats
    
    def _load_history(self):
        """Load pattern history from file."""
        if self.history_file and os.path.exists(self.history_file):
            try:
                with open(self.history_file, 'r') as f:
                    data = json.load(f)
                
                # Load history
                for name, values in data.get('history', {}).items():
                    if name in self.pattern_history:
                        self.pattern_history[name] = deque(
                            values[-self.window:], 
                            maxlen=self.window
                        )
                
                # Load performance
                for name, perf in data.get('performance', {}).items():
                    if name in self.pattern_performance:
                        self.pattern_performance[name].update(perf)
                
                logger.info(f"Loaded pattern history from {self.history_file}")
                
            except Exception as e:
                logger.error(f"Failed to load pattern history: {e}")
    
    def save_history(self):
        """Save pattern history and performance."""
        if self.history_file:
            try:
                data = {
                    'history': {
                        name: list(hist) 
                        for name, hist in self.pattern_history.items()
                    },
                    'performance': self.pattern_performance,
                    'timestamp': str(pd.Timestamp.now())
                }
                
                os.makedirs(os.path.dirname(self.history_file), exist_ok=True)
                
                with open(self.history_file, 'w') as f:
                    json.dump(data, f, indent=2)
                
                logger.debug(f"Saved pattern history to {self.history_file}")
                
            except Exception as e:
                logger.error(f"Failed to save pattern history: {e}")
