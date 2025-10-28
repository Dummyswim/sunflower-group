"""
Candlestick Pattern Detection for Index Scalping (TA‑Lib–backed)
This module detects classic TA‑Lib CDL patterns + custom Tweezer Top/Bottom and optional Rounding.
It returns the top candidate for the last bar with {'name','signal','confidence'} and logs the decision.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)

try:
    import talib as ta
    _HAS_TALIB = True
except Exception:
    _HAS_TALIB = False


def _np_series(x: pd.Series) -> np.ndarray:
    return pd.to_numeric(x, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy()


def _talib_func(name: str):
    return getattr(ta, name, None) if _HAS_TALIB else None


# Define TA‑Lib patterns
TA_PATTERNS = {
    # Requested new patterns
    "CDLINVERTEDHAMMER": _talib_func("CDLINVERTEDHAMMER"),
    "CDLPIERCING": _talib_func("CDLPIERCING"),
    "CDLHARAMI": _talib_func("CDLHARAMI"),
    "CDL3WHITESOLDIERS": _talib_func("CDL3WHITESOLDIERS"),
    "CDL3BLACKCROWS": _talib_func("CDL3BLACKCROWS"),
    "CDLDARKCLOUDCOVER": _talib_func("CDLDARKCLOUDCOVER"),
    "CDLABANDONEDBABY": _talib_func("CDLABANDONEDBABY"),
    "CDLSPINNINGTOP": _talib_func("CDLSPINNINGTOP"),
    "CDLTRISTAR": _talib_func("CDLTRISTAR"),
    "CDLSTICKSANDWICH": _talib_func("CDLSTICKSANDWICH"),
    # Existing/common
    "CDLENGULFING": _talib_func("CDLENGULFING"),
    "CDLHAMMER": _talib_func("CDLHAMMER"),
    "CDLSHOOTINGSTAR": _talib_func("CDLSHOOTINGSTAR"),
}


class CandlestickPatternDetector:
    """
    Detects high-probability candlestick patterns for scalping.
    Returns a single top pattern for the last bar with:
    {'name': str, 'signal': 'LONG'|'SHORT'|'NEUTRAL', 'confidence': int}
    Compatible with the rest of the pipeline (analyzer, charts, alerts).
    """

    def __init__(self, config=None):
        self.config = config
        # Defaults if config not injected
        self._min_strength = int(getattr(config, "pattern_min_strength", 50)) if config else 50
        self._enable_talib = bool(getattr(config, "enable_talib_patterns", True)) if config else True
        self._names = list(getattr(config, "candlestick_patterns", [])) if config else [
            "CDLINVERTEDHAMMER", "CDLPIERCING", "CDLHARAMI", "CDL3WHITESOLDIERS", "CDL3BLACKCROWS",
            "CDLDARKCLOUDCOVER", "CDLABANDONEDBABY", "CDLSPINNINGTOP", "CDLTRISTAR", "CDLSTICKSANDWICH",
            "CDLENGULFING", "CDLHAMMER", "CDLSHOOTINGSTAR"
        ]
        self._enable_tw = bool(getattr(config, "enable_custom_tweezer", True)) if config else True
        self._tw_tol_bps = float(getattr(config, "tweezer_tolerance_bps", 5.0)) if config else 5.0
        self._enable_round = bool(getattr(config, "enable_rounding_patterns", False)) if config else False
        self._round_win = int(getattr(config, "rounding_window", 20)) if config else 20

    def detect_patterns(self, df: pd.DataFrame) -> Dict[str, object]:
        """
        Detect actionable TA‑Lib candlestick patterns (and custom) on the last bar.
        Returns: {'name': str, 'signal': 'LONG'|'SHORT'|'NEUTRAL', 'confidence': int}
        High-visibility INFO logs name the recognized pattern for users.
        """
        # NaN/Inf safe frame
        if df is None or df.empty or len(df) < 2:
            return {'name': 'NONE', 'signal': 'NEUTRAL', 'confidence': 0}

        data = df.replace([np.nan, np.inf, -np.inf], 0.0)

        # 1) TA‑Lib patterns
        merged: Dict[str, int] = {}
        if self._enable_talib and _HAS_TALIB and self._names:
            raw = self._detect_talib_scores(data.tail(50), self._names)
            merged.update(raw)

        # 2) Custom Tweezer Top/Bottom (TA‑Lib doesn’t provide these)
        if self._enable_tw:
            tw = self._detect_tweezer(data, self._tw_tol_bps)  # {'TWEEZER_TOP': -1, 'TWEEZER_BOTTOM': +1}
            for k, v in tw.items():
                merged[k] = int(v * 100)  # scale to ±100 so min_strength applies uniformly

        # 3) Optional rounding top/bottom proxy (noisy on 5m; disabled by default)
        if self._enable_round:
            rnd = self._detect_rounding(data, self._round_win)  # {'ROUNDING_TOP':1} or {'ROUNDING_BOTTOM':1}
            for k, v in rnd.items():
                merged[k] = int(v * 100)

        


        # ENHANCED: Require completed bar for pattern confirmation
        if df is None or df.empty or len(df) < 3:
            logger.info("[PATTERN] Insufficient data for confirmation")
            return {'name': 'NONE', 'signal': 'NEUTRAL', 'confidence': 0}

        # Verify last bar is closed (not forming)
        last_bar = df.iloc[-1]
        is_forming = False  # Assume closed unless marked otherwise
        if hasattr(df.index[-1], 'second') and df.index[-1].second != 0:
            is_forming = True  # Bar timestamp not at boundary
            logger.warning("[PATTERN] ⚠️ Last bar is forming - skipping pattern detection")
            return {'name': 'NONE', 'signal': 'NEUTRAL', 'confidence': 0}

        # Pick top pattern by |score| ≥ min_strength (ONLY ON CLOSED BARS)
        top_name, top_val = self._select_top(merged, self._min_strength)
        if not top_name:
            logger.info("[PATTERN] No pattern meets minimum strength threshold")


            
            # If nothing strong, still surface a Spinning Top if it’s pronounced (indecision/veto)
            st = merged.get("CDLSPINNINGTOP", 0)
            if abs(int(st)) >= self._min_strength:
                logger.info("[PATTERN] Indecision: CDLSPINNINGTOP (%+d) — veto in compression", int(st))
                
        
            return {'name': 'NONE', 'signal': 'NEUTRAL', 'confidence': 0}

        # Map sign to LONG/SHORT, compute a confidence 60–90 band
        direction = 'LONG' if top_val > 0 else 'SHORT' if top_val < 0 else 'NEUTRAL'
        conf = int(min(90, 60 + min(100, abs(int(top_val))) * 0.2))


        # Treat Spinning Top as true indecision (no boost)
        if top_name == "CDLSPINNINGTOP":
            logger.info("[PATTERN] Indecision: %s (%+d) — veto in compression", top_name, int(top_val))
            return {'name': top_name, 'signal': 'NEUTRAL', 'confidence': 0}

        # High-visibility INFO for users
        logger.info("[PATTERN] Recognized: %s (%+d) | dir=%s",
                    top_name, int(top_val), "BULLISH" if direction == 'LONG' else "BEARISH")
                    

        # Log custom tweezer specifically (user-friendly naming)
        if top_name in ("TWEEZER_TOP", "TWEEZER_BOTTOM"):
            logger.info("[PATTERN] Tweezer detected: %s → %s", top_name,
                        "bearish" if top_name.endswith("TOP") else "bullish")
            

        return {'name': top_name, 'signal': direction, 'confidence': conf}

    # ===== Internals =====

    def _detect_talib_scores(self, df: pd.DataFrame, names: List[str]) -> Dict[str, int]:
        if not _HAS_TALIB or df is None or df.empty:
            return {}
        o = _np_series(df["open"])
        h = _np_series(df["high"])
        l = _np_series(df["low"])
        c = _np_series(df["close"])
        out: Dict[str, int] = {}
        for name in names:
            func = TA_PATTERNS.get(name)
            if func is None:
                continue
            try:
                                
                                
                v = func(o, h, l, c)
                last = v[-1] if v is not None and len(v) else 0
                val = int(last) if np.isfinite(last) else 0
                out[name] = val
                logger.debug(f"[PATTERN] TA-Lib {name} last={last} → val={val}")


            except Exception as e:
                logger.debug("[PATTERN] TA‑Lib %s failed: %s", name, e)
        return out

    def _select_top(self, scores: Dict[str, int], min_strength: int) -> Tuple[Optional[str], int]:
        top_name, top_val = None, 0
        for k, v in (scores or {}).items():
            vi = int(v)
            if abs(vi) >= int(min_strength) and abs(vi) > abs(int(top_val)):
                top_name, top_val = k, vi
        return top_name, int(top_val)

    def _detect_tweezer(self, df: pd.DataFrame, tol_bps: float) -> Dict[str, int]:
        """
        Tweezer Top/Bottom:
          - Top: highs of last 2 candles within tolerance → bearish bias (-1)
          - Bottom: lows of last 2 candles within tolerance → bullish bias (+1)
        """
        out = {}
        if df is None or len(df) < 2:
            return out
        last2 = df.tail(2)
        h1, h2 = float(last2["high"].iloc[-2]), float(last2["high"].iloc[-1])
        l1, l2 = float(last2["low"].iloc[-2]), float(last2["low"].iloc[-1])
        if not (np.isfinite(h1) and np.isfinite(h2) and np.isfinite(l1) and np.isfinite(l2)):
            return out
        px = float(last2["close"].iloc[-1]) or 1.0
        tol = abs(px) * (tol_bps / 10_000.0)
        if abs(h1 - h2) <= tol:
            out["TWEEZER_TOP"] = -1
        if abs(l1 - l2) <= tol:
            out["TWEEZER_BOTTOM"] = +1
        return out

    def _detect_rounding(self, df: pd.DataFrame, window: int) -> Dict[str, int]:
        """
        Lightweight rounding proxy via quadratic curvature sign over `window` bars.
        (Noisy on 5m; disabled by default in config.)
        """
        out = {}
        if df is None or len(df) < window:
            return out
        closes = _np_series(df["close"])[-window:]
        x = np.arange(window, dtype=float)
        try:
            a, b, c = np.polyfit(x, closes, 2)
            if np.isfinite(a):
                if a < 0:
                    out["ROUNDING_TOP"] = 1
                elif a > 0:
                    out["ROUNDING_BOTTOM"] = 1
        except Exception as e:
            logger.debug("[PATTERN] Rounding fit failed: %s", e)
        return out

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
            denom = max(abs(float(current_price)), 1e-9)
            # protect against division by zero 
            if abs(current_price - level) / denom < 0.01:  # Within 1%
                
                if level > current_price:
                    levels['moderate_resistance'].append(level)
                else:
                    levels['moderate_support'].append(level)
                            
        # Clean and sort
        for key in levels:
            levels[key] = sorted(list(set(levels[key])))
        
        # Find nearest levels

        upside = [x for x in (levels['strong_resistance'] + levels['moderate_resistance']) if x > current_price]
        downside = [x for x in (levels['strong_support'] + levels['moderate_support']) if x < current_price]
        nearest_resistance = min(upside, default=current_price + 100, key=lambda x: abs(x - current_price))
        nearest_support = max(downside, default=current_price - 100, key=lambda x: abs(x - current_price))

        
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