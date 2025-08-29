"""
Technical indicators module: RSI, MACD, VWAP, Bollinger, OBV.
"""
import numpy as np
import pandas as pd
import logging
from typing import Dict, Optional, Tuple
import talib
from datetime import datetime

logger = logging.getLogger(__name__)

class TechnicalIndicators:
    """Calculate technical indicators for trading signals."""
    
    @staticmethod
    def calculate_rsi(df: pd.DataFrame, params: Dict) -> pd.Series:
        """Calculate RSI with validation."""
        try:
            period = params.get("period", 9)
            if len(df) < period:
                logger.warning(f"Insufficient data for RSI: {len(df)} < {period}")
                return pd.Series([50] * len(df), index=df.index)
            
            rsi = talib.RSI(df['close'].values, timeperiod=period)
            logger.debug(f"RSI calculated: {rsi[-1]:.2f}" if len(rsi) > 0 and not np.isnan(rsi[-1]) else "RSI empty")
            return pd.Series(rsi, index=df.index)
            
        except Exception as e:
            logger.error(f"RSI calculation error: {e}")
            return pd.Series([50] * len(df), index=df.index)
    
    @staticmethod
    def calculate_macd(df: pd.DataFrame, params: Dict) -> Dict[str, pd.Series]:
        """Calculate MACD with validation."""
        try:
            fast = params.get("fastperiod", 8)
            slow = params.get("slowperiod", 17)
            signal = params.get("signalperiod", 9)
            
            if len(df) < slow:
                logger.warning(f"Insufficient data for MACD: {len(df)} < {slow}")
                zeros = pd.Series([0] * len(df), index=df.index)
                return {"macd": zeros, "signal": zeros, "hist": zeros}
            
            macd, macdsignal, macdhist = talib.MACD(
                df['close'].values, 
                fastperiod=fast, 
                slowperiod=slow, 
                signalperiod=signal
            )
            
            return {
                "macd": pd.Series(macd, index=df.index),
                "signal": pd.Series(macdsignal, index=df.index),
                "hist": pd.Series(macdhist, index=df.index)
            }
            
        except Exception as e:
            logger.error(f"MACD calculation error: {e}")
            zeros = pd.Series([0] * len(df), index=df.index)
            return {"macd": zeros, "signal": zeros, "hist": zeros}
    
    @staticmethod
    def calculate_vwap(df: pd.DataFrame, params: Dict) -> pd.Series:
        """Calculate VWAP with validation."""
        try:
            window = params.get("window", 3)
            
            typical_price = (df['high'] + df['low'] + df['close']) / 3
            pv = typical_price * df['volume']
            
            pv_sum = pv.rolling(window=window, min_periods=1).sum()
            vol_sum = df['volume'].rolling(window=window, min_periods=1).sum()
            
            vwap = pv_sum / vol_sum
            vwap = vwap.fillna(df['close'])
            
            logger.debug(f"VWAP calculated: {vwap.iloc[-1]:.2f}" if len(vwap) > 0 else "VWAP empty")
            return vwap
            
        except Exception as e:
            logger.error(f"VWAP calculation error: {e}")
            return df['close'].copy()
    
    @staticmethod
    def calculate_bollinger_bands(df: pd.DataFrame, params: Dict) -> Dict[str, pd.Series]:
        """Calculate Bollinger Bands with validation."""
        try:
            period = params.get("period", 10)
            stddev = params.get("stddev", 2)
            
            if len(df) < period:
                logger.warning(f"Insufficient data for Bollinger: {len(df)} < {period}")
                return {
                    "upper": df['close'].copy(),
                    "middle": df['close'].copy(),
                    "lower": df['close'].copy()
                }
            
            upper, middle, lower = talib.BBANDS(
                df['close'].values,
                timeperiod=period,
                nbdevup=stddev,
                nbdevdn=stddev,
                matype=0
            )
            
            return {
                "upper": pd.Series(upper, index=df.index),
                "middle": pd.Series(middle, index=df.index),
                "lower": pd.Series(lower, index=df.index)
            }
            
        except Exception as e:
            logger.error(f"Bollinger calculation error: {e}")
            return {
                "upper": df['close'].copy(),
                "middle": df['close'].copy(),
                "lower": df['close'].copy()
            }
    
    @staticmethod
    def calculate_obv(df: pd.DataFrame, params: Dict) -> pd.Series:
        """Calculate OBV with validation."""
        try:
            if len(df) < 2:
                logger.warning("Insufficient data for OBV")
                return pd.Series([0] * len(df), index=df.index)
            
            obv = talib.OBV(df['close'].values, df['volume'].values.astype(float))
            return pd.Series(obv, index=df.index)
            
        except Exception as e:
            logger.error(f"OBV calculation error: {e}")
            return pd.Series([0] * len(df), index=df.index)


class SignalGenerator:
    """Generate trading signals from indicators."""
    
    def __init__(self, config):
        """Initialize with configuration."""
        self.config = config
        logger.info("SignalGenerator initialized")
    
    def calculate_weighted_signal(self, indicators: Dict) -> Dict:
        """Calculate weighted composite signal from all indicators."""
        try:
            weights = self.config.INDICATOR_WEIGHTS
            signals = {}
            active_count = 0
            total_score = 0.0
            confidence = 0
            crossovers = {}
            
            # Process each indicator
            rsi_signal, rsi_cross = self._get_rsi_signal(indicators, self.config.RSI_PARAMS)
            signals['rsi'] = rsi_signal
            if rsi_cross:
                crossovers['rsi'] = rsi_cross
            
            macd_signal, macd_cross = self._get_macd_signal(indicators)
            signals['macd'] = macd_signal
            if macd_cross:
                crossovers['macd'] = macd_cross
            
            vwap_signal, vwap_cross = self._get_vwap_signal(indicators)
            signals['vwap'] = vwap_signal
            if vwap_cross:
                crossovers['vwap'] = vwap_cross
            
            bb_signal, bb_cross = self._get_bollinger_signal(indicators)
            signals['bollinger'] = bb_signal
            if bb_cross:
                crossovers['bollinger'] = bb_cross
            
            obv_signal = self._get_obv_signal(indicators)
            signals['obv'] = obv_signal
            
            # Calculate weighted score
            for indicator_name, signal_value in signals.items():
                weight = weights.get(indicator_name, 0)
                total_score += weight * signal_value
                
                if signal_value != 0:
                    active_count += 1
                    confidence += abs(signal_value) * weight * 100
            
            # Add bonus confidence for crossovers
            if crossovers:
                confidence += 10 * len(crossovers)
            
            composite_signal = self._determine_composite_signal(total_score, active_count)
            
            result = {
                "score": total_score,
                "weighted_score": total_score,
                "confidence": min(confidence, 100),
                "active_indicators": active_count,
                "signals": signals,
                "composite_signal": composite_signal,
                "crossovers": crossovers,
                "timestamp": datetime.now()
            }
            
            logger.debug(f"Signal generated: {composite_signal}, Score: {total_score:.3f}, "
                        f"Confidence: {confidence:.1f}%, Active: {active_count}")
            
            return result
            
        except Exception as e:
            logger.error(f"Signal generation error: {e}")
            return self._get_default_signal_result()
    
    def _get_rsi_signal(self, indicators: Dict, params: Dict) -> Tuple[float, Optional[str]]:
        """Get RSI signal and check for crossovers."""
        try:
            if 'rsi' not in indicators or indicators['rsi'].empty:
                return 0, None
            
            rsi = indicators['rsi'].iloc[-1]
            if np.isnan(rsi):
                return 0, None
                
            prev_rsi = indicators['rsi'].iloc[-2] if len(indicators['rsi']) > 1 else rsi
            
            overbought = params['overbought']
            oversold = params['oversold']
            
            crossover = None
            if prev_rsi <= oversold < rsi:
                crossover = "bullish"
            elif prev_rsi >= overbought > rsi:
                crossover = "bearish"
            
            if rsi > overbought:
                return -1, crossover
            elif rsi < oversold:
                return 1, crossover
            elif rsi > 60:
                return -0.5, crossover
            elif rsi < 40:
                return 0.5, crossover
            else:
                return 0, crossover
                
        except Exception as e:
            logger.error(f"RSI signal error: {e}")
            return 0, None
    
    def _get_macd_signal(self, indicators: Dict) -> Tuple[float, Optional[str]]:
        """Get MACD signal and check for crossovers."""
        try:
            if 'macd' not in indicators:
                return 0, None
            
            macd_data = indicators['macd']
            if macd_data['macd'].empty:
                return 0, None
            
            macd = macd_data['macd'].iloc[-1]
            signal = macd_data['signal'].iloc[-1]
            hist = macd_data['hist'].iloc[-1]
            
            if np.isnan(macd) or np.isnan(signal) or np.isnan(hist):
                return 0, None
            
            prev_macd = macd_data['macd'].iloc[-2] if len(macd_data['macd']) > 1 else macd
            prev_signal = macd_data['signal'].iloc[-2] if len(macd_data['signal']) > 1 else signal
            
            crossover = None
            if prev_macd <= prev_signal and macd > signal:
                crossover = "bullish"
            elif prev_macd >= prev_signal and macd < signal:
                crossover = "bearish"
            
            if hist > 0:
                strength = min(abs(hist) / 0.1, 1.0)
                return strength, crossover
            elif hist < 0:
                strength = -min(abs(hist) / 0.1, 1.0)
                return strength, crossover
            else:
                return 0, crossover
                
        except Exception as e:
            logger.error(f"MACD signal error: {e}")
            return 0, None
    
    def _get_vwap_signal(self, indicators: Dict) -> Tuple[float, Optional[str]]:
        """Get VWAP signal and check for crossovers."""
        try:
            if 'vwap' not in indicators or 'price' not in indicators:
                return 0, None
            
            price = indicators['price']
            vwap = indicators['vwap'].iloc[-1]
            
            if np.isnan(vwap) or vwap == 0:
                return 0, None
            
            prev_price = indicators.get('prev_price', price)
            prev_vwap = indicators['vwap'].iloc[-2] if len(indicators['vwap']) > 1 else vwap
            
            crossover = None
            if prev_price <= prev_vwap and price > vwap:
                crossover = "bullish"
            elif prev_price >= prev_vwap and price < vwap:
                crossover = "bearish"
            
            distance_pct = (price - vwap) / vwap * 100
            
            if distance_pct > 0.5:
                return min(distance_pct / 2, 1.0), crossover
            elif distance_pct < -0.5:
                return max(distance_pct / 2, -1.0), crossover
            else:
                return 0, crossover
                
        except Exception as e:
            logger.error(f"VWAP signal error: {e}")
            return 0, None
    
    def _get_bollinger_signal(self, indicators: Dict) -> Tuple[float, Optional[str]]:
        """Get Bollinger Bands signal and check for band touches."""
        try:
            if 'bollinger' not in indicators or 'price' not in indicators:
                return 0, None
            
            bb = indicators['bollinger']
            price = indicators['price']
            
            upper = bb['upper'].iloc[-1]
            middle = bb['middle'].iloc[-1]
            lower = bb['lower'].iloc[-1]
            
            if np.isnan(upper) or np.isnan(lower) or np.isnan(middle):
                return 0, None
            
            crossover = None
            if price <= lower:
                crossover = "lower_touch"
            elif price >= upper:
                crossover = "upper_touch"
            
            band_width = upper - lower
            if band_width > 0:
                position = (price - lower) / band_width
                
                if position > 0.9:
                    return -0.8, crossover
                elif position < 0.1:
                    return 0.8, crossover
                elif position > 0.7:
                    return -0.3, crossover
                elif position < 0.3:
                    return 0.3, crossover
                else:
                    return 0, crossover
            
            return 0, crossover
            
        except Exception as e:
            logger.error(f"Bollinger signal error: {e}")
            return 0, None
    
    def _get_obv_signal(self, indicators: Dict) -> float:
        """Get OBV signal based on trend."""
        try:
            if 'obv' not in indicators or indicators['obv'].empty:
                return 0
            
            if len(indicators['obv']) < 3:
                return 0
            
            recent_obv = indicators['obv'].iloc[-3:]
            obv_slope = np.polyfit(range(len(recent_obv)), recent_obv.values, 1)[0]
            
            if obv_slope > 0:
                return min(obv_slope / 10000, 0.5)
            elif obv_slope < 0:
                return max(obv_slope / 10000, -0.5)
            else:
                return 0
                
        except Exception as e:
            logger.error(f"OBV signal error: {e}")
            return 0
    
    def _determine_composite_signal(self, score: float, active_count: int) -> str:
        """Determine composite signal from score and active indicators."""
        if active_count == 0:
            return "NO_SIGNAL"
        
        if score >= 0.7:
            return "STRONG_BUY"
        elif score >= 0.3:
            return "BUY"
        elif score <= -0.7:
            return "STRONG_SELL"
        elif score <= -0.3:
            return "SELL"
        else:
            return "NEUTRAL"
    
    def _get_default_signal_result(self) -> Dict:
        """Return default signal result on error."""
        return {
            "score": 0,
            "weighted_score": 0,
            "confidence": 0,
            "active_indicators": 0,
            "signals": {},
            "composite_signal": "ERROR",
            "crossovers": {},
            "timestamp": datetime.now()
        }
