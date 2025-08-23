"""
Advanced pattern recognition engine with self-learning capabilities.
"""
import json
import logging
import os
from collections import deque
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional

try:
    import talib
    HAS_TALIB = True
except ImportError:
    HAS_TALIB = False
    logging.warning("TA-Lib not installed. Pattern detection disabled.")

logger = logging.getLogger(__name__)

class PatternEngine:
    """
    Self-learning pattern recognition engine with rolling accuracy tracking.
    """
    
    def __init__(self, window: int = 100, default_prob: float = 0.55, 
                 history_file: Optional[str] = None):
        """
        Initialize pattern engine.
        
        Args:
            window: Rolling window for hit-rate calculation
            default_prob: Default probability when no history exists
            history_file: Optional file to persist pattern history
        """
        self.window = window
        self.default_prob = default_prob
        self.history_file = history_file
        
        # Get all TA-Lib pattern functions
        self.pattern_funcs = {}
        if HAS_TALIB:
            pattern_groups = talib.get_function_groups().get("Pattern Recognition", [])
            for func_name in pattern_groups:
                try:
                    self.pattern_funcs[func_name] = getattr(talib, func_name)
                except AttributeError:
                    pass
        
        # Initialize history tracking
        self.pattern_history: Dict[str, deque] = {
            name: deque(maxlen=window) for name in self.pattern_funcs
        }
        
        # Pattern metadata
        self.pattern_types = self._classify_patterns()
        
        # Load historical performance if available
        self._load_history()
        
        logger.info(f"PatternEngine initialized with {len(self.pattern_funcs)} patterns")
    
    def _classify_patterns(self) -> Dict[str, str]:
        """Classify patterns by type (reversal, continuation, etc.)"""
        classifications = {
            # Reversal patterns
            "CDLENGULFING": "reversal",
            "CDLHARAMI": "reversal",
            "CDLHARAMICROSS": "reversal",
            "CDLMORNINGSTAR": "reversal",
            "CDLEVENINGSTAR": "reversal",
            "CDLSHOOTINGSTAR": "reversal",
            "CDLHAMMER": "reversal",
            "CDLINVERTEDHAMMER": "reversal",
            "CDLHANGINGMAN": "reversal",
            "CDLDOJI": "reversal",
            "CDLDRAGONFLYDOJI": "reversal",
            "CDLGRAVESTONEDOJI": "reversal",
            
            # Continuation patterns
            "CDLSPINNINGTOP": "continuation",
            "CDLLONGLEGGEDDOJI": "continuation",
            "CDLRICKSHAWMAN": "continuation",
            "CDLRISEFALL3METHODS": "continuation",
            "CDLGAPSIDESIDEWHITE": "continuation",
            
            # Trend patterns
            "CDL3WHITESOLDIERS": "trend",
            "CDL3BLACKCROWS": "trend",
            "CDL3LINESTRIKE": "trend",
            "CDLKICKING": "trend",
            "CDLBREAKAWAY": "trend",
        }
        
        # Default classification for unspecified patterns
        default_class = "neutral"
        result = {}
        for name in self.pattern_funcs:
            result[name] = classifications.get(name, default_class)
        
        return result
    
    def detect_patterns(self, df: pd.DataFrame) -> List[Dict]:
        """
        Detect all candlestick patterns in the data.
        
        Returns:
            List of detected patterns with metadata
        """
        if not HAS_TALIB or len(df) < 5:
            return []
        
        detections = []
        o = df['open'].values
        h = df['high'].values
        l = df['low'].values
        c = df['close'].values
        
        for name, func in self.pattern_funcs.items():
            try:
                # Run pattern detection
                result = func(o, h, l, c)
                if isinstance(result, np.ndarray):
                    value = int(result[-1])  # Latest bar
                else:
                    value = int(result)
                
                if value != 0:  # Pattern detected
                    # Get historical performance
                    prior = self.get_prior(name)
                    hit_rate = self.get_hit_rate(name)
                    sample_size = len(self.pattern_history.get(name, []))
                    
                    detection = {
                        "name": name,
                        "value": value,  # >0 bullish, <0 bearish
                        "direction": "bullish" if value > 0 else "bearish",
                        "type": self.pattern_types.get(name, "neutral"),
                        "prior": prior,
                        "hit_rate": hit_rate,
                        "sample_size": sample_size,
                        "confidence": self._calculate_confidence(prior, sample_size),
                        "strength": abs(value) / 100.0  # Normalize strength
                    }
                    detections.append(detection)
                    
            except Exception as e:
                logger.debug(f"Pattern {name} detection failed: {e}")
                continue
        
        return detections
    
    def update_pattern_performance(self, pattern_name: str, was_correct: bool):
        """Update the rolling performance history for a pattern."""
        if pattern_name not in self.pattern_history:
            self.pattern_history[pattern_name] = deque(maxlen=self.window)
        
        self.pattern_history[pattern_name].append(1 if was_correct else 0)
        logger.debug(f"Updated {pattern_name}: {'✓' if was_correct else '✗'}")
    
    def get_prior(self, pattern_name: str) -> float:
        """Get the prior probability for a pattern based on history."""
        history = self.pattern_history.get(pattern_name, [])
        if len(history) < 5:  # Need minimum samples
            return self.default_prob
        return float(np.mean(history))
    
    def get_hit_rate(self, pattern_name: str) -> float:
        """Get the historical hit rate for a pattern."""
        history = self.pattern_history.get(pattern_name, [])
        if not history:
            return 0.0
        return float(np.mean(history))
    
    def _calculate_confidence(self, prior: float, sample_size: int) -> float:
        """
        Calculate confidence based on prior and sample size.
        More samples = higher confidence in the prior.
        """
        if sample_size < 5:
            return 0.5  # Low confidence
        elif sample_size < 20:
            # Interpolate between default and actual prior
            weight = sample_size / 20.0
            return self.default_prob * (1 - weight) + prior * weight
        else:
            return prior
    
    def _load_history(self):
        """Load pattern history from file if exists."""
        if self.history_file and os.path.exists(self.history_file):
            try:
                with open(self.history_file, 'r') as f:
                    data = json.load(f)
                for name, values in data.items():
                    if name in self.pattern_history:
                        self.pattern_history[name] = deque(values[-self.window:], 
                                                          maxlen=self.window)
                logger.info(f"Loaded pattern history from {self.history_file}")
            except Exception as e:
                logger.error(f"Failed to load pattern history: {e}")
    
    def save_history(self):
        """Save pattern history to file."""
        if self.history_file:
            try:
                data = {name: list(hist) for name, hist in self.pattern_history.items()}
                with open(self.history_file, 'w') as f:
                    json.dump(data, f)
                logger.debug(f"Saved pattern history to {self.history_file}")
            except Exception as e:
                logger.error(f"Failed to save pattern history: {e}")
