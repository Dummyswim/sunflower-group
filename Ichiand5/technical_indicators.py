"""
Comprehensive technical indicators module with all six required indicators.
"""
import numpy as np
import pandas as pd
import logging
from typing import Dict, Tuple, Optional, Union
import talib
from scipy import stats

logger = logging.getLogger(__name__)

class TechnicalIndicators:
    """Calculate all technical indicators for trading signals."""
    
    @staticmethod
    def calculate_ichimoku(df: pd.DataFrame, params: Dict) -> Dict[str, pd.Series]:
        """
        Calculate Ichimoku Cloud components for trend analysis.
        
        Returns:
            Dictionary with tenkan, kijun, senkou_a, senkou_b, chikou
        """
        try:
            high = df['high']
            low = df['low']
            close = df['close']
            
            # Tenkan-sen (Conversion Line)
            high_tenkan = high.rolling(window=params['tenkan_period']).max()
            low_tenkan = low.rolling(window=params['tenkan_period']).min()
            tenkan_sen = (high_tenkan + low_tenkan) / 2
            
            # Kijun-sen (Base Line)
            high_kijun = high.rolling(window=params['kijun_period']).max()
            low_kijun = low.rolling(window=params['kijun_period']).min()
            kijun_sen = (high_kijun + low_kijun) / 2
            
            # Senkou Span A (Leading Span A)
            senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(params['displacement'])
            
            # Senkou Span B (Leading Span B)
            high_senkou = high.rolling(window=params['senkou_span_b_period']).max()
            low_senkou = low.rolling(window=params['senkou_span_b_period']).min()
            senkou_span_b = ((high_senkou + low_senkou) / 2).shift(params['displacement'])
            
            # Chikou Span (Lagging Span)
            chikou_span = close.shift(-params['displacement'])
            
            return {
                'tenkan': tenkan_sen,
                'kijun': kijun_sen,
                'senkou_a': senkou_span_a,
                'senkou_b': senkou_span_b,
                'chikou': chikou_span
            }
            
        except Exception as e:
            logger.error(f"Ichimoku calculation error: {e}")
            return {}
    
    @staticmethod
    def calculate_stochastic(df: pd.DataFrame, params: Dict) -> Dict[str, pd.Series]:
        """
        Calculate Stochastic Oscillator for momentum analysis.
        
        Returns:
            Dictionary with %K and %D values
        """
        try:
            high = df['high'].values
            low = df['low'].values
            close = df['close'].values
            
            # Calculate Stochastic using talib
            k, d = talib.STOCH(
                high, low, close,
                fastk_period=params['k_period'],
                slowk_period=params['smooth_k'],
                slowk_matype=0,
                slowd_period=params['d_period'],
                slowd_matype=0
            )
            
            return {
                'k': pd.Series(k, index=df.index),
                'd': pd.Series(d, index=df.index),
                'overbought': params['overbought'],
                'oversold': params['oversold']
            }
            
        except Exception as e:
            logger.error(f"Stochastic calculation error: {e}")
            return {}
    
    @staticmethod
    def calculate_obv(df: pd.DataFrame, params: Dict) -> Dict[str, pd.Series]:
        """
        Calculate On-Balance Volume for volume trend analysis.
        
        Returns:
            Dictionary with OBV and signal line
        """
        try:
            close = df['close'].values
            volume = df['volume'].values
            
            # Calculate OBV using talib
            obv = talib.OBV(close, volume)
            obv_series = pd.Series(obv, index=df.index)
            
            # Calculate signal line (EMA of OBV)
            obv_ema = obv_series.ewm(span=params['ema_period'], adjust=False).mean()
            
            # Calculate divergence
            obv_normalized = (obv_series - obv_series.rolling(20).mean()) / obv_series.rolling(20).std()
            
            return {
                'obv': obv_series,
                'signal': obv_ema,
                'normalized': obv_normalized
            }
            
        except Exception as e:
            logger.error(f"OBV calculation error: {e}")
            return {}
    
    @staticmethod
    def calculate_bollinger_bands(df: pd.DataFrame, params: Dict) -> Dict[str, pd.Series]:
        """
        Calculate Bollinger Bands for volatility analysis.
        
        Returns:
            Dictionary with upper, middle, lower bands and %B
        """
        try:
            close = df['close'].values
            
            # Calculate Bollinger Bands using talib
            upper, middle, lower = talib.BBANDS(
                close,
                timeperiod=params['period'],
                nbdevup=params['num_std'],
                nbdevdn=params['num_std'],
                matype=0
            )
            
            # Calculate %B (position within bands)
            percent_b = (close - lower) / (upper - lower)
            
            # Calculate band width
            band_width = (upper - lower) / middle
            
            return {
                'upper': pd.Series(upper, index=df.index),
                'middle': pd.Series(middle, index=df.index),
                'lower': pd.Series(lower, index=df.index),
                'percent_b': pd.Series(percent_b, index=df.index),
                'width': pd.Series(band_width, index=df.index)
            }
            
        except Exception as e:
            logger.error(f"Bollinger Bands calculation error: {e}")
            return {}
    
    @staticmethod
    def calculate_adx(df: pd.DataFrame, params: Dict) -> Dict[str, Union[pd.Series, float]]:
        """
        Calculate Average Directional Index for trend strength.
        
        Returns:
            Dictionary with ADX, +DI, -DI values
        """
        try:
            high = df['high'].values
            low = df['low'].values
            close = df['close'].values
            
            # Calculate ADX using talib
            adx = talib.ADX(high, low, close, timeperiod=params['period'])
            plus_di = talib.PLUS_DI(high, low, close, timeperiod=params['period'])
            minus_di = talib.MINUS_DI(high, low, close, timeperiod=params['period'])
            
            return {
                'adx': pd.Series(adx, index=df.index),
                'plus_di': pd.Series(plus_di, index=df.index),
                'minus_di': pd.Series(minus_di, index=df.index),
                'strong_trend': params['strong_trend'],
                'weak_trend': params['weak_trend']
            }
            
        except Exception as e:
            logger.error(f"ADX calculation error: {e}")
            return {}
    
    @staticmethod
    def calculate_atr(df: pd.DataFrame, params: Dict) -> Dict[str, pd.Series]:
        """
        Calculate Average True Range for volatility measurement.
        
        Returns:
            Dictionary with ATR and normalized ATR
        """
        try:
            high = df['high'].values
            low = df['low'].values
            close = df['close'].values
            
            # Calculate ATR using talib
            atr = talib.ATR(high, low, close, timeperiod=params['period'])
            atr_series = pd.Series(atr, index=df.index)
            
            # Calculate normalized ATR (as percentage of close)
            atr_percent = (atr_series / df['close']) * 100
            
            # Calculate stop-loss levels
            stop_loss_long = df['close'] - (atr_series * params['multiplier'])
            stop_loss_short = df['close'] + (atr_series * params['multiplier'])
            
            return {
                'atr': atr_series,
                'atr_percent': atr_percent,
                'stop_loss_long': stop_loss_long,
                'stop_loss_short': stop_loss_short
            }
            
        except Exception as e:
            logger.error(f"ATR calculation error: {e}")
            return {}


class SignalGenerator:
    """Generate weighted trading signals from multiple indicators."""
    
    def __init__(self):
        self.signal_history = []
        
    def calculate_weighted_signal(self, df: pd.DataFrame, indicators: Dict, 
                                 weights: Dict[str, float]) -> Dict:
        """
        Calculate weighted signal from all indicators.
        
        Returns:
            Dictionary with composite signal, score, and confidence
        """
        try:
            signals = {}
            signal_strengths = {}
            
            # Ichimoku Cloud Signal
            ichimoku_signal = self._ichimoku_signal(indicators.get('ichimoku', {}))
            signals['ichimoku'] = ichimoku_signal['signal']
            signal_strengths['ichimoku'] = ichimoku_signal['strength']
            
            # Stochastic Signal
            stochastic_signal = self._stochastic_signal(indicators.get('stochastic', {}))
            signals['stochastic'] = stochastic_signal['signal']
            signal_strengths['stochastic'] = stochastic_signal['strength']
            
            # OBV Signal
            obv_signal = self._obv_signal(indicators.get('obv', {}))
            signals['obv'] = obv_signal['signal']
            signal_strengths['obv'] = obv_signal['strength']
            
            # Bollinger Bands Signal
            bb_signal = self._bollinger_signal(indicators.get('bollinger', {}))
            signals['bollinger'] = bb_signal['signal']
            signal_strengths['bollinger'] = bb_signal['strength']
            
            # ADX Signal
            adx_signal = self._adx_signal(indicators.get('adx', {}))
            signals['adx'] = adx_signal['signal']
            signal_strengths['adx'] = adx_signal['strength']
            
            # ATR Signal (for position sizing/risk)
            atr_signal = self._atr_signal(indicators.get('atr', {}))
            signals['atr'] = atr_signal['signal']
            signal_strengths['atr'] = atr_signal['strength']
            
            # Calculate weighted score
            weighted_score = 0
            total_weight = 0
            active_indicators = 0
            
            for indicator, signal in signals.items():
                if signal != 0 and indicator in weights:
                    weighted_score += signal * signal_strengths[indicator] * weights[indicator]
                    total_weight += weights[indicator]
                    active_indicators += 1
            
            # Normalize score
            if total_weight > 0:
                weighted_score = weighted_score / total_weight
            
            # Calculate confidence
            confidence = self._calculate_confidence(signals, signal_strengths)
            
            # Determine composite signal
            composite_signal = self._determine_signal_type(weighted_score)
            
            # Check for crossovers
            crossovers = self._detect_crossovers(indicators)
            
            return {
                'composite_signal': composite_signal,
                'weighted_score': round(weighted_score, 3),
                'confidence': round(confidence, 1),
                'active_indicators': active_indicators,
                'individual_signals': signals,
                'signal_strengths': signal_strengths,
                'crossovers': crossovers
            }
            
        except Exception as e:
            logger.error(f"Signal generation error: {e}")
            return {
                'composite_signal': 'NEUTRAL',
                'weighted_score': 0,
                'confidence': 0,
                'active_indicators': 0
            }
    
    def _ichimoku_signal(self, ichimoku: Dict) -> Dict[str, float]:
        """Generate signal from Ichimoku Cloud."""
        try:
            if not ichimoku or 'tenkan' not in ichimoku:
                return {'signal': 0, 'strength': 0}
            
            tenkan = ichimoku['tenkan'].iloc[-1]
            kijun = ichimoku['kijun'].iloc[-1]
            senkou_a = ichimoku['senkou_a'].iloc[-1]
            senkou_b = ichimoku['senkou_b'].iloc[-1]
            close = ichimoku.get('close', pd.Series([0])).iloc[-1]
            
            signal = 0
            strength = 0
            
            # Bullish signals
            if tenkan > kijun:  # Tenkan above Kijun
                signal += 0.5
                strength += 0.3
            
            if close > max(senkou_a, senkou_b):  # Price above cloud
                signal += 0.5
                strength += 0.4
            
            # Bearish signals
            if tenkan < kijun:  # Tenkan below Kijun
                signal -= 0.5
                strength += 0.3
            
            if close < min(senkou_a, senkou_b):  # Price below cloud
                signal -= 0.5
                strength += 0.4
            
            # Normalize strength
            strength = min(strength, 1.0)
            
            return {'signal': signal, 'strength': strength}
            
        except Exception as e:
            logger.error(f"Ichimoku signal error: {e}")
            return {'signal': 0, 'strength': 0}
    
    def _stochastic_signal(self, stochastic: Dict) -> Dict[str, float]:
        """Generate signal from Stochastic Oscillator."""
        try:
            if not stochastic or 'k' not in stochastic:
                return {'signal': 0, 'strength': 0}
            
            k = stochastic['k'].iloc[-1]
            d = stochastic['d'].iloc[-1]
            k_prev = stochastic['k'].iloc[-2]
            d_prev = stochastic['d'].iloc[-2]
            
            signal = 0
            strength = 0
            
            # Oversold with bullish crossover
            if k < stochastic['oversold'] and k > d and k_prev <= d_prev:
                signal = 1
                strength = 0.8
            # Overbought with bearish crossover
            elif k > stochastic['overbought'] and k < d and k_prev >= d_prev:
                signal = -1
                strength = 0.8
            # Momentum signals
            elif k > 50 and k > d:
                signal = 0.5
                strength = 0.5
            elif k < 50 and k < d:
                signal = -0.5
                strength = 0.5
            
            return {'signal': signal, 'strength': strength}
            
        except Exception as e:
            logger.error(f"Stochastic signal error: {e}")
            return {'signal': 0, 'strength': 0}
    
    def _obv_signal(self, obv: Dict) -> Dict[str, float]:
        """Generate signal from On-Balance Volume."""
        try:
            if not obv or 'obv' not in obv:
                return {'signal': 0, 'strength': 0}
            
            obv_val = obv['obv'].iloc[-1]
            obv_signal = obv['signal'].iloc[-1]
            obv_norm = obv['normalized'].iloc[-1]
            
            signal = 0
            strength = 0
            
            # Volume confirmation
            if obv_val > obv_signal and obv_norm > 0.5:
                signal = 0.7
                strength = min(abs(obv_norm) / 2, 1.0)
            elif obv_val < obv_signal and obv_norm < -0.5:
                signal = -0.7
                strength = min(abs(obv_norm) / 2, 1.0)
            elif obv_val > obv_signal:
                signal = 0.3
                strength = 0.4
            elif obv_val < obv_signal:
                signal = -0.3
                strength = 0.4
            
            return {'signal': signal, 'strength': strength}
            
        except Exception as e:
            logger.error(f"OBV signal error: {e}")
            return {'signal': 0, 'strength': 0}
    
    def _bollinger_signal(self, bollinger: Dict) -> Dict[str, float]:
        """Generate signal from Bollinger Bands."""
        try:
            if not bollinger or 'upper' not in bollinger:
                return {'signal': 0, 'strength': 0}
            
            percent_b = bollinger['percent_b'].iloc[-1]
            width = bollinger['width'].iloc[-1]
            width_avg = bollinger['width'].rolling(20).mean().iloc[-1]
            
            signal = 0
            strength = 0
            
            # Band squeeze/expansion
            volatility_factor = width / width_avg if width_avg > 0 else 1
            
            # Oversold/Overbought
            if percent_b < 0:  # Below lower band
                signal = 0.8
                strength = min(abs(percent_b), 1.0)
            elif percent_b > 1:  # Above upper band
                signal = -0.8
                strength = min(percent_b - 1, 1.0)
            elif percent_b < 0.2:  # Near lower band
                signal = 0.4
                strength = 0.5
            elif percent_b > 0.8:  # Near upper band
                signal = -0.4
                strength = 0.5
            
            # Adjust for volatility
            if volatility_factor < 0.5:  # Squeeze
                strength *= 1.2
            
            return {'signal': signal, 'strength': min(strength, 1.0)}
            
        except Exception as e:
            logger.error(f"Bollinger signal error: {e}")
            return {'signal': 0, 'strength': 0}
    
    def _adx_signal(self, adx: Dict) -> Dict[str, float]:
        """Generate signal from ADX indicator."""
        try:
            if not adx or 'adx' not in adx:
                return {'signal': 0, 'strength': 0}
            
            adx_val = adx['adx'].iloc[-1]
            plus_di = adx['plus_di'].iloc[-1]
            minus_di = adx['minus_di'].iloc[-1]
            
            signal = 0
            strength = 0
            
            # Strong trend confirmation
            if adx_val > adx['strong_trend']:
                if plus_di > minus_di:
                    signal = 0.8
                else:
                    signal = -0.8
                strength = min(adx_val / 50, 1.0)
            # Weak trend
            elif adx_val < adx['weak_trend']:
                signal = 0  # No trend
                strength = 0.2
            # Moderate trend
            else:
                if plus_di > minus_di:
                    signal = 0.5
                else:
                    signal = -0.5
                strength = 0.5
            
            return {'signal': signal, 'strength': strength}
            
        except Exception as e:
            logger.error(f"ADX signal error: {e}")
            return {'signal': 0, 'strength': 0}
    
    def _atr_signal(self, atr: Dict) -> Dict[str, float]:
        """Generate signal from ATR (for risk management)."""
        try:
            if not atr or 'atr_percent' not in atr:
                return {'signal': 0, 'strength': 0}
            
            atr_pct = atr['atr_percent'].iloc[-1]
            atr_avg = atr['atr_percent'].rolling(20).mean().iloc[-1]
            
            # ATR mainly for position sizing, but can indicate breakouts
            signal = 0
            strength = 0
            
            volatility_ratio = atr_pct / atr_avg if atr_avg > 0 else 1
            
            # High volatility might indicate breakout
            if volatility_ratio > 1.5:
                strength = 0.6
            elif volatility_ratio < 0.5:
                strength = 0.3
            else:
                strength = 0.4
            
            return {'signal': signal, 'strength': strength}
            
        except Exception as e:
            logger.error(f"ATR signal error: {e}")
            return {'signal': 0, 'strength': 0}
    
    def _calculate_confidence(self, signals: Dict, strengths: Dict) -> float:
        """Calculate overall confidence level."""
        try:
            # Count agreeing signals
            bullish = sum(1 for s in signals.values() if s > 0)
            bearish = sum(1 for s in signals.values() if s < 0)
            
            # Calculate average strength
            avg_strength = np.mean(list(strengths.values())) if strengths else 0
            
            # Agreement ratio
            total_signals = len(signals)
            agreement = max(bullish, bearish) / total_signals if total_signals > 0 else 0
            
            # Confidence formula
            confidence = (agreement * 60 + avg_strength * 40)
            
            return min(confidence, 100)
            
        except Exception as e:
            logger.error(f"Confidence calculation error: {e}")
            return 0
    
    def _determine_signal_type(self, score: float) -> str:
        """Determine signal type from weighted score."""
        if score >= 0.7:
            return "STRONG BUY"
        elif score >= 0.4:
            return "BUY"
        elif score <= -0.7:
            return "STRONG SELL"
        elif score <= -0.4:
            return "SELL"
        else:
            return "NEUTRAL"
    
    def _detect_crossovers(self, indicators: Dict) -> Dict[str, bool]:
        """Detect important crossovers."""
        crossovers = {}
        
        try:
            # Ichimoku crossover
            if 'ichimoku' in indicators:
                ich = indicators['ichimoku']
                if len(ich.get('tenkan', [])) > 1:
                    tenkan_cross = (
                        ich['tenkan'].iloc[-1] > ich['kijun'].iloc[-1] and
                        ich['tenkan'].iloc[-2] <= ich['kijun'].iloc[-2]
                    )
                    crossovers['ichimoku_bullish'] = tenkan_cross
            
            # Stochastic crossover
            if 'stochastic' in indicators:
                stoch = indicators['stochastic']
                if len(stoch.get('k', [])) > 1:
                    stoch_cross = (
                        stoch['k'].iloc[-1] > stoch['d'].iloc[-1] and
                        stoch['k'].iloc[-2] <= stoch['d'].iloc[-2]
                    )
                    crossovers['stochastic_bullish'] = stoch_cross
            
        except Exception as e:
            logger.error(f"Crossover detection error: {e}")
        
        return crossovers
