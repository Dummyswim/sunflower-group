"""
Technical analysis module - VOLUME INDICATORS REMOVED, EMA ADDED
Optimized for NIFTY50 index trading without volume data
"""
import pandas as pd
import numpy as np
import logging
from typing import Dict, Optional, List, Any
from datetime import datetime

try:
    import talib
except ImportError:
    talib = None
    logging.warning("TA-Lib not installed. Using fallback calculations.")

logger = logging.getLogger(__name__)

class ConsolidatedTechnicalAnalysis:
    """Technical analysis for index trading (no volume indicators)."""
    
    def __init__(self, config):
        self.config = config
        self.signal_history = []
        logger.info("Technical Analysis initialized (volume-free)")
        self._last_valid_rsi = {'5m': None, '15m': None}
    
    async def calculate_all_indicators(self, df: pd.DataFrame, timeframe: str = "5m") -> Dict:
        """Calculate technical indicators without volume dependencies."""
        
        # Input validation
        if df is None or df.empty:
            logger.warning("Empty dataframe provided")
            return self._get_default_indicators(df)
        
        min_required = self.config.min_data_points
        if len(df) < min_required:
            logger.warning(f"Insufficient data: {len(df)} < {min_required}")
            return self._get_default_indicators(df)
        
        # Ensure proper data types
        try:
            for col in ['open', 'high', 'low', 'close']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce').ffill()
                    df[col] = df[col].astype(np.float64)
        except Exception as e:
            logger.error(f"Data type conversion error: {e}")
            df = df.ffill()
        
        indicators = {}
        
        # Calculate each indicator with error handling
        try:
            indicators['rsi'] = self._calculate_rsi(df, timeframe)
            logger.debug(f"RSI calculated: {indicators['rsi']['value']:.2f}")
        except Exception as e:
            logger.error(f"RSI calculation failed: {e}")
            indicators['rsi'] = self._get_default_rsi()
        
        try:
            indicators['macd'] = self._calculate_macd(df, timeframe)
            logger.debug(f"MACD histogram: {indicators['macd']['histogram']:.4f}")
        except Exception as e:
            logger.error(f"MACD calculation failed: {e}")
            indicators['macd'] = self._get_default_macd()
        
        try:
            indicators['ema'] = self._calculate_ema(df, timeframe)
            logger.debug(f"EMA signal: {indicators['ema']['signal']}")
        except Exception as e:
            logger.error(f"EMA calculation failed: {e}")
            indicators['ema'] = self._get_default_ema()
        
        try:
            indicators['bollinger'] = self._calculate_bollinger(df, timeframe)
            logger.debug(f"Bollinger position: {indicators['bollinger']['position']:.2f}")
        except Exception as e:
            logger.error(f"Bollinger calculation failed: {e}")
            indicators['bollinger'] = self._get_default_bollinger()
        
        try:
            indicators['keltner'] = self._calculate_keltner(df, timeframe)
            logger.debug(f"Keltner signal: {indicators['keltner']['signal']}")
        except Exception as e:
            logger.error(f"Keltner calculation failed: {e}")
            indicators['keltner'] = self._get_default_keltner()
        
        try:
            indicators['supertrend'] = self._calculate_supertrend(df, timeframe)
            logger.debug(f"Supertrend: {indicators['supertrend']['trend']}")
        except Exception as e:
            logger.error(f"Supertrend calculation failed: {e}")
            indicators['supertrend'] = self._get_default_supertrend()
        
        try:
            indicators['impulse'] = self._calculate_impulse(df, timeframe)
            logger.debug(f"Impulse state: {indicators['impulse']['state']}")
        except Exception as e:
            logger.error(f"Impulse calculation failed: {e}")
            indicators['impulse'] = self._get_default_impulse()
        
        try:
            indicators['atr'] = self._calculate_atr(df, timeframe)
            logger.info("[ATR] %.2f (period=%d)", indicators['atr'].get('value', 0.0), int(indicators['atr'].get('period', 14)))
            
        except Exception as e:
            logger.error(f"ATR calculation failed: {e}")
            indicators['atr'] = {'value': 0.0, 'period': 14}
        

        try:
            if getattr(self.config, 'enable_oi_integration', True):
                indicators['oi'] = self._calculate_oi_context(df, timeframe)
                logger.debug("[OI] context=%s ΔOI%%=%.2f (oi=%s, Δ=%s)", 
                            indicators['oi'].get('signal', 'neutral'), 
                            float(indicators['oi'].get('oi_change_pct', 0.0)), 
                            str(indicators['oi'].get('oi', 0)), 
                            str(indicators['oi'].get('oi_change', 0)))
                
        except Exception as e:
            logger.error(f"OI context calculation failed: {e}")
        

        
        
        # Add current price and volatility
        indicators['price'] = float(df['close'].iloc[-1]) if not df.empty else 0
        indicators['volatility'] = self._calculate_volatility(df, timeframe)
        

        logger.info(f"Calculated {len(indicators)} indicators successfully [{timeframe}]")

        logger.info("[IND] RSI=%s | MACD_hist=%.4f | EMA_sig=%s | BB_bw=%s",
            f"{indicators['rsi'].get('value', 50):.1f}" if 'rsi' in indicators else 'NA',
            float(indicators['macd'].get('histogram', 0.0)) if 'macd' in indicators else 0.0,
            indicators['ema'].get('signal', 'NA') if 'ema' in indicators else 'NA',
            f"{indicators['bollinger'].get('bandwidth', 0.0):.2f}" if 'bollinger' in indicators else 'NA')
        



        return indicators
    
    
    
    
    def _calculate_atr(self, df: pd.DataFrame, timeframe: str = "5m", period: int = 14) -> Dict:
        """ATR(14) proxy using True Range; NaN/Inf safe."""
        try:
            if len(df) < period + 2:
                return {'value': 0.0, 'period': period}
            high = pd.to_numeric(df['high'], errors='coerce').replace([np.inf, -np.inf], np.nan)
            low = pd.to_numeric(df['low'], errors='coerce').replace([np.inf, -np.inf], np.nan)
            close = pd.to_numeric(df['close'], errors='coerce').replace([np.inf, -np.inf], np.nan)
            prev_close = close.shift(1)
            tr = pd.concat([(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1).dropna()
            atr = tr.rolling(window=period).mean().dropna()
            val = float(atr.iloc[-1]) if not atr.empty and np.isfinite(atr.iloc[-1]) else 0.0
            return {'value': val, 'period': period}
        except Exception:
            return {'value': 0.0, 'period': period}


    
    def _calculate_volatility(self, df: pd.DataFrame, timeframe: str = "5m") -> Dict:
        """Calculate price volatility using standard deviation."""
        try:
            if len(df) < 20:
                return {'value': 1.0, 'level': 'normal'}
            
            # Calculate returns
            returns = df['close'].pct_change().dropna()
            
            # Calculate volatility (annualized)
            volatility = returns.std() * np.sqrt(252) * 100  # Annualized volatility percentage
            
            # Determine volatility level
            if volatility < 10:
                level = 'low'
            elif volatility < 20:
                level = 'normal'
            elif volatility < 30:
                level = 'high'
            else:
                level = 'extreme'
            
            return {
                'value': round(volatility, 2),
                'level': level,
                'recent_std': round(returns.tail(20).std() * 100, 2)
            }
        except Exception as e:
            logger.error(f"Volatility calculation error: {e}")
            return {'value': 15.0, 'level': 'normal', 'recent_std': 1.0}
    
    def _calculate_ema(self, df: pd.DataFrame, timeframe: str = "5m") -> Dict:
        """Calculate EMA crossover signals."""
        try:
            ema_params = self.config.get_ema_params(timeframe)
            if talib:
                ema_short = talib.EMA(df['close'].values, timeperiod=ema_params['short_period'])
                ema_medium = talib.EMA(df['close'].values, timeperiod=ema_params['medium_period'])
                ema_long = talib.EMA(df['close'].values, timeperiod=ema_params['long_period'])
            else:
                # Fallback calculation
                ema_short = df['close'].ewm(span=ema_params['short_period'], adjust=False).mean().values
                ema_medium = df['close'].ewm(span=ema_params['medium_period'], adjust=False).mean().values
                ema_long = df['close'].ewm(span=ema_params['long_period'], adjust=False).mean().values
            
            current_price = df['close'].iloc[-1]
            
            # Check crossovers and position
            signal = 'neutral'
            crossover = None
            
            # Bullish conditions
            if (ema_short[-1] > ema_medium[-1] > ema_long[-1]):
                signal = 'bullish'
                if len(ema_short) > 2 and ema_short[-2] <= ema_medium[-2]:
                    crossover = 'golden_cross'
            # Bearish conditions
            elif (ema_short[-1] < ema_medium[-1] < ema_long[-1]):
                signal = 'bearish'
                if len(ema_short) > 2 and ema_short[-2] >= ema_medium[-2]:
                    crossover = 'death_cross'
            # Price above all EMAs
            elif current_price > ema_short[-1] and current_price > ema_medium[-1]:
                signal = 'above'
            # Price below all EMAs
            elif current_price < ema_short[-1] and current_price < ema_medium[-1]:
                signal = 'below'
            
            return {
                'short': float(ema_short[-1]),
                'medium': float(ema_medium[-1]),
                'long': float(ema_long[-1]),
                'signal': signal,
                'crossover': crossover,
                'short_series': pd.Series(ema_short, index=df.index),
                'medium_series': pd.Series(ema_medium, index=df.index),
                'long_series': pd.Series(ema_long, index=df.index)
            }
        except Exception as e:
            logger.error(f"EMA calculation error: {e}")
            return self._get_default_ema()
    
    def _calculate_rsi(self, df: pd.DataFrame, timeframe: str = "5m") -> Dict:
        """Calculate RSI indicator."""
        
        try:
            rsi_params = self.config.get_rsi_params(timeframe)
            period = rsi_params['period']
            
            if talib:
                rsi_series = talib.RSI(df['close'].values, timeperiod=period)
            else:
                delta = df['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
                loss_safe = loss.replace(0, 1e-10)
                loss_safe = loss_safe.where(loss_safe != 0, 1e-10)
                

                rs = gain / loss_safe.replace(0, 1e-10)
                rs = rs.fillna(100)
                rs = rs.replace([np.inf, -np.inf], 100)
                rsi_series = 100 - (100 / (1 + rs))

                rsi_series = rsi_series.values
            
            fallback_used = False
            if len(rsi_series) > 0:
                last_val = rsi_series[-1]
                
                if pd.notna(last_val) and not np.isinf(last_val) and 0 <= last_val <= 100:
                    current_rsi = float(last_val)
                else:
                    # Context‑preserving fallback: use last valid RSI with slight decay, else shorter window, else 50
                    prev_vals = pd.Series(rsi_series).replace([np.inf, -np.inf], np.nan).dropna()
                    if len(prev_vals) >= 1:
                        last_valid = float(prev_vals.iloc[-1])
                        self._last_valid_rsi[timeframe] = last_valid
                        current_rsi = last_valid * 0.95 + 50.0 * 0.05
                    elif self._last_valid_rsi.get(timeframe) is not None:
                        current_rsi = float(self._last_valid_rsi[timeframe]) * 0.95 + 50.0 * 0.05
                    else:
                        # fallback to short window proxy
                        short = max(5, period // 2)
                        try:
                            delta2 = df['close'].diff()
                            gain2 = (delta2.where(delta2 > 0, 0)).rolling(window=short).mean()
                            loss2 = (-delta2.where(delta2 < 0, 0)).rolling(window=short).mean().replace(0, 1e-10)
                            rs2 = (gain2 / loss2).fillna(1.0)
                            alt = float((100 - (100 / (1 + rs2))).iloc[-1])
                            current_rsi = alt if np.isfinite(alt) and 0 <= alt <= 100 else 50.0
                        except Exception:
                            current_rsi = 50.0
                    fallback_used = True
                    logger.info(f"[RSI] Fallback used → {current_rsi:.1f} (period={period}, tf={timeframe})")
            else:
                # empty series: last valid if exists, else 50
                if self._last_valid_rsi.get(timeframe) is not None:
                    current_rsi = float(self._last_valid_rsi[timeframe]) * 0.95 + 50.0 * 0.05
                else:
                    current_rsi = 50.0
                fallback_used = True
                logger.info(f"[RSI] Fallback used → {current_rsi:.1f} (empty series, tf={timeframe})")


            # Persist last valid RSI when not a fallback
            if not fallback_used:
                self._last_valid_rsi[timeframe] = current_rsi

            # Sanitize final RSI value
            try:
                current_rsi = float(current_rsi)
                if not np.isfinite(current_rsi):
                    current_rsi = 50.0
                else:
                    current_rsi = float(np.clip(current_rsi, 0.0, 100.0))
            except Exception:
                current_rsi = 50.0

            # Determine signal
            if current_rsi > rsi_params['overbought']:
                signal = 'overbought'
            elif current_rsi < rsi_params['oversold']:
                signal = 'oversold'
            else:
                signal = 'neutral'


            
            divergence = self._check_rsi_divergence(df, rsi_series)
            
            return {
                'value': current_rsi,
                'signal': signal,
                'divergence': divergence,
                'rsi_series': pd.Series(rsi_series, index=df.index),
                'fallback': fallback_used
            }
        except Exception as e:
            logger.error(f"RSI calculation error: {e}")
            return self._get_default_rsi()



    def _check_rsi_divergence(self, df: pd.DataFrame, rsi_series: np.ndarray) -> str:
        """Check for RSI divergence patterns."""
        try:
            if len(df) < 20 or len(rsi_series) < 20:
                return 'none'
            
            # Get recent peaks and troughs
            price_recent = df['close'].iloc[-20:].values
            rsi_recent = rsi_series[-20:]
            
            # Simple divergence check
            price_trend = np.polyfit(range(len(price_recent)), price_recent, 1)[0]
            valid_rsi = rsi_recent[~np.isnan(rsi_recent)]
            
            if len(valid_rsi) >= 2:
                rsi_trend = np.polyfit(range(len(valid_rsi)), valid_rsi, 1)[0]
            else:
                rsi_trend = 0

            
            if price_trend > 0 and rsi_trend < 0:
                return 'bearish_divergence'
            elif price_trend < 0 and rsi_trend > 0:
                return 'bullish_divergence'
            
            return 'none'
        except:
            return 'none'
    
    def _calculate_macd(self, df: pd.DataFrame, timeframe: str = "5m") -> Dict:
        """Calculate MACD indicator."""

        try:
            macd_params = self.config.get_macd_params(timeframe)
            if talib:
                macd, signal, hist = talib.MACD(
                    df['close'].values,
                    fastperiod=macd_params['fastperiod'],
                    slowperiod=macd_params['slowperiod'],
                    signalperiod=macd_params['signalperiod']
                )
            else:
                # Fallback MACD calculation
                exp1 = df['close'].ewm(span=macd_params['fastperiod'], adjust=False).mean()
                exp2 = df['close'].ewm(span=macd_params['slowperiod'], adjust=False).mean()
                macd = exp1 - exp2
                signal = macd.ewm(span=macd_params['signalperiod'], adjust=False).mean()

                hist = macd - signal
                macd = macd.values
                signal = signal.values
                hist = hist.values
            
            current_hist = float(hist[-1]) if len(hist) > 0 and not np.isnan(hist[-1]) else 0

            if np.isnan(current_hist) or np.isinf(current_hist):
                current_hist = 0.0
                        
                        
            # Determine signal
            signal_type = 'neutral'
            if current_hist > 0:
                signal_type = 'bullish'
                # Check for strengthening
                if len(hist) > 2 and hist[-1] > hist[-2]:
                    signal_type = 'bullish_strengthening'
            elif current_hist < 0:
                signal_type = 'bearish'
                # Check for strengthening
                if len(hist) > 2 and hist[-1] < hist[-2]:
                    signal_type = 'bearish_strengthening'
            
            # Check for crossover
            crossover = None
            if len(hist) > 2:
                if hist[-2] <= 0 and hist[-1] > 0:
                    crossover = 'bullish_crossover'
                elif hist[-2] >= 0 and hist[-1] < 0:
                    crossover = 'bearish_crossover'


            # ensure finite values
            macd_val = float(macd[-1]) if len(macd) > 0 else 0.0
            sig_val = float(signal[-1]) if len(signal) > 0 else 0.0
            if np.isnan(macd_val) or np.isinf(macd_val):
                macd_val = 0.0
            if np.isnan(sig_val) or np.isinf(sig_val):
                sig_val = 0.0






            hist_slope_closed = 0.0
            hist_slope_forming = 0.0
            try:
                ser = pd.Series(hist).astype(float)
                # CRITICAL FIX: Use only CLOSED bars for slope calculation
                # Exclude last bar (forming) for closed slope
                if len(ser) >= 5:
                    closed_bars = ser.iloc[:-1]  # Exclude forming bar
                    diffs_closed = closed_bars.diff().dropna().tail(3)
                    if len(diffs_closed) >= 3:
                        hist_slope_closed = float(diffs_closed.mean())
                        logger.info(f"[MACD-SLOPE] Closed slope={hist_slope_closed:+.6f} (last 3 closed diffs)")
                    else:
                        logger.warning("[MACD-SLOPE] Insufficient closed bars for slope calculation")
                
                # Forming slope: only for pre-close analysis reference (NOT for decisions)
                if len(ser) >= 2:
                    hist_slope_forming = float(ser.iloc[-1] - ser.iloc[-2])
                    logger.debug(f"[MACD-SLOPE] Forming slope={hist_slope_forming:+.6f} (reference only)")
            except Exception as e:
                logger.error(f"[MACD-SLOPE] Calculation error: {e}")
                pass



            return {
                'macd': macd_val,
                'signal': sig_val,
                'histogram': current_hist,
                'signal_type': signal_type,
                'crossover': crossover,
                'macd_series': pd.Series(macd, index=df.index),
                'signal_series': pd.Series(signal, index=df.index),
                'histogram_series': pd.Series(hist, index=df.index),
                'hist_slope_closed': hist_slope_closed,    # ADD THIS
                'hist_slope_forming': hist_slope_forming   # ADD THIS
            }

        except Exception as e:
            logger.error(f"MACD calculation error: {e}")
            return self._get_default_macd()
    
    def _calculate_bollinger(self, df: pd.DataFrame, timeframe: str = "5m") -> Dict:
        """Calculate Bollinger Bands."""

        try:
            bollinger_params = self.config.get_bollinger_params(timeframe)
            period = bollinger_params['period']
            stddev = bollinger_params['stddev']
              
            if talib:
                upper, middle, lower = talib.BBANDS(
                    df['close'].values,
                    timeperiod=period,
                    nbdevup=stddev,
                    nbdevdn=stddev
                )
            else:
                # Fallback Bollinger Bands calculation
                middle = df['close'].rolling(window=period).mean()
                std = df['close'].rolling(window=period).std()
                upper = middle + (std * stddev)
                lower = middle - (std * stddev)
                upper = upper.values
                middle = middle.values
                lower = lower.values
            
            current_price = float(df['close'].iloc[-1])
            current_upper = float(upper[-1]) if not np.isnan(upper[-1]) else current_price + 100
            current_lower = float(lower[-1]) if not np.isnan(lower[-1]) else current_price - 100
            current_middle = float(middle[-1]) if not np.isnan(middle[-1]) else current_price
            
            # Calculate bandwidth
            bandwidth = (current_upper - current_lower) / current_middle * 100 if current_middle > 0 else 0
            
            # Calculate position (0 to 1)
            if current_upper > current_lower:
                position = (current_price - current_lower) / (current_upper - current_lower)
            else:
                position = 0.5
            
            position = max(0, min(1, position))  # Clamp between 0 and 1
            
            # Determine signal
            if position > 0.95:
                signal = 'overbought'
            elif position < 0.05:
                signal = 'oversold'
            elif position > 0.8:
                signal = 'near_upper'
            elif position < 0.2:
                signal = 'near_lower'
            else:
                signal = 'neutral'
            
            # Check for squeeze (low volatility)
            is_squeeze = bandwidth < 10  # Bandwidth less than 10%
            
            return {
                'upper': current_upper,
                'middle': current_middle,
                'lower': current_lower,
                'position': position,
                'signal': signal,
                'bandwidth': round(bandwidth, 2),
                'is_squeeze': is_squeeze,
                'upper_series': pd.Series(upper, index=df.index),
                'middle_series': pd.Series(middle, index=df.index),
                'lower_series': pd.Series(lower, index=df.index)
            }
        except Exception as e:
            logger.error(f"Bollinger calculation error: {e}")
            return self._get_default_bollinger()
    
    def _calculate_keltner(self, df: pd.DataFrame, timeframe: str = "5m") -> Dict:
        """Calculate Keltner Channels using High-Low range instead of ATR."""
        try:
            period = self.config.keltner_params['period']
            multiplier = self.config.keltner_params['multiplier']
            
            # Calculate EMA of typical price
            typical_price = (df['high'] + df['low'] + df['close']) / 3
            
            if talib:
                ema = talib.EMA(typical_price.values, timeperiod=period)
            else:
                ema = typical_price.ewm(span=period, adjust=False).mean().values
            
            # Use High-Low range instead of ATR (since we don't have volume)
            hl_range = (df['high'] - df['low']).rolling(window=period).mean()
            
            # Calculate channels
            upper = ema + (multiplier * hl_range.values)
            lower = ema - (multiplier * hl_range.values)
            
            current_price = float(df['close'].iloc[-1])
            current_upper = float(upper[-1]) if not np.isnan(upper[-1]) else current_price + 100
            current_lower = float(lower[-1]) if not np.isnan(lower[-1]) else current_price - 100
            current_middle = float(ema[-1]) if not np.isnan(ema[-1]) else current_price
            
            # Calculate position
            if current_upper > current_lower:
                position = (current_price - current_lower) / (current_upper - current_lower)
            else:
                position = 0.5
            
            position = max(0, min(1, position))
            
            # Determine signal
            if current_price > current_upper:
                signal = 'above_upper'
            elif current_price < current_lower:
                signal = 'below_lower'
            else:
                signal = 'within'
            
            return {
                'upper': current_upper,
                'middle': current_middle,
                'lower': current_lower,
                'position': position,
                'signal': signal,
                'upper_series': pd.Series(upper, index=df.index),
                'middle_series': pd.Series(ema, index=df.index),
                'lower_series': pd.Series(lower, index=df.index)
            }
        except Exception as e:
            logger.error(f"Keltner calculation error: {e}")
            return self._get_default_keltner()
    
    def _calculate_supertrend(self, df: pd.DataFrame, timeframe: str = "5m") -> Dict:
        """Calculate Supertrend using HL/2 and range-based multiplier."""
        try:
            supertrend_params = self.config.get_supertrend_params(timeframe)
            period = supertrend_params['period']
            multiplier = supertrend_params['multiplier']
            
            # Calculate HL average
            hl_avg = (df['high'] + df['low']) / 2
            
            # Use range instead of ATR
            hl_range = (df['high'] - df['low']).rolling(window=period).mean()
            
            # Calculate bands
            upper_band = hl_avg + (multiplier * hl_range)
            lower_band = hl_avg - (multiplier * hl_range)
            
            # Initialize Supertrend
            supertrend = pd.Series(index=df.index, dtype=float)
            direction = pd.Series(index=df.index, dtype=float)
            
            for i in range(period, len(df)):
                if i == period:
                    supertrend.iloc[i] = upper_band.iloc[i]
                    direction.iloc[i] = -1
                else:
                    # Previous values
                    prev_st = supertrend.iloc[i-1]
                    
                    # Current close
                    curr_close = df['close'].iloc[i]
                    
                    # Upper and lower bands
                    curr_upper = upper_band.iloc[i]
                    curr_lower = lower_band.iloc[i]
                    
                    # Supertrend logic
                    if prev_st == upper_band.iloc[i-1]:
                        if curr_close <= curr_upper:
                            supertrend.iloc[i] = curr_upper
                            direction.iloc[i] = -1
                        else:
                            supertrend.iloc[i] = curr_lower
                            direction.iloc[i] = 1
                    else:
                        if curr_close >= curr_lower:
                            supertrend.iloc[i] = curr_lower
                            direction.iloc[i] = 1
                        else:
                            supertrend.iloc[i] = curr_upper
                            direction.iloc[i] = -1
            
            # Get current values
            if not supertrend.dropna().empty:
                current_st = float(supertrend.dropna().iloc[-1])
                current_direction = float(direction.dropna().iloc[-1])
                
                if current_direction == 1:
                    trend = 'bullish'
                elif current_direction == -1:
                    trend = 'bearish'
                else:
                    trend = 'neutral'
            else:
                current_st = float(df['close'].iloc[-1])
                current_direction = 0
                trend = 'neutral'
            
            return { 
                    'value': current_st, 
                    'direction': current_direction, 
                    'trend': trend, 
                    'signal': trend, # expose as signal so analyzer and consensus read it 
                    'supertrend_series': supertrend.ffill(), 
                    'direction_series': direction.fillna(0) 
                }
                    
        except Exception as e:
            logger.error(f"Supertrend calculation error: {e}")
            return self._get_default_supertrend()
    
    def _calculate_impulse(self, df: pd.DataFrame, timeframe: str = "5m") -> Dict:
        """Calculate Impulse MACD System."""
        try:
            # Calculate MACD histogram
            if talib:
                macd, signal, hist = talib.MACD(df['close'].values)
                ema_13 = talib.EMA(df['close'].values, timeperiod=13)
            else:
                # Fallback
                exp1 = df['close'].ewm(span=12, adjust=False).mean()
                exp2 = df['close'].ewm(span=26, adjust=False).mean()
                macd = exp1 - exp2
                signal = macd.ewm(span=9, adjust=False).mean()
                hist = (macd - signal).values
                ema_13 = df['close'].ewm(span=13, adjust=False).mean().values
            
            # Determine impulse state
            state = 'blue'  # Default neutral
            
            if len(hist) > 1 and len(ema_13) > 1:
                macd_rising = hist[-1] > hist[-2]
                ema_rising = ema_13[-1] > ema_13[-2]
                
                if macd_rising and ema_rising:
                    state = 'green'  # Buy signal
                elif not macd_rising and not ema_rising:
                    state = 'red'    # Sell signal
                else:
                    state = 'blue'   # Neutral
            
            # Calculate strength
            histogram_strength = abs(hist[-1]) if len(hist) > 0 else 0
            ema_slope = (ema_13[-1] - ema_13[-2]) / ema_13[-2] * 100 if len(ema_13) > 1 and ema_13[-2] != 0 else 0
            
            return {
                'state': state,
                'histogram': float(hist[-1]) if len(hist) > 0 else 0,
                'ema_slope': float(ema_slope),
                'strength': float(histogram_strength)
            }
        except Exception as e:
            logger.error(f"Impulse MACD calculation error: {e}")
            return self._get_default_impulse()
    
    
        

    def _calculate_oi_context(self, df: pd.DataFrame, timeframe: str = "5m") -> Dict:
        """
        Compute OI change context from df['oi'] vs price change over last bar.
        Returns a small, NaN/Inf-safe context indicator.
        """
        try:
            if df is None or df.empty or 'oi' not in df.columns or len(df) < 2:
                return {'signal': 'neutral', 'oi': 0, 'oi_change': 0, 'oi_change_pct': 0.0}
            
            
            oi2 = pd.to_numeric(df['oi'].tail(2), errors='coerce').ffill().fillna(0).astype(float)
            px2 = pd.to_numeric(df['close'].tail(2), errors='coerce').ffill().fillna(0).astype(float)
            
            if len(oi2) < 2 or len(px2) < 2:
                return {'signal': 'neutral', 'oi': float(oi2.iloc[-1]) if len(oi2) else 0.0, 'oi_change': 0, 'oi_change_pct': 0.0}
            
            oi_now, oi_prev = float(oi2.iloc[-1]), float(oi2.iloc[-2])
            px_now, px_prev = float(px2.iloc[-1]), float(px2.iloc[-2])
            
            oi_change = oi_now - oi_prev
            denom = abs(oi_prev) if oi_prev != 0 else 1.0
            oi_change_pct = (oi_change / denom) * 100.0
            
            price_up = px_now > px_prev
            price_down = px_now < px_prev
            oi_up = oi_change > 0
            oi_down = oi_change < 0

            # Map to common futures context
            # Price↑ & OI↑ → Long build-up; Price↓ & OI↑ → Short build-up
            # Price↑ & OI↓ → Short covering; Price↓ & OI↓ → Long unwinding
            signal = 'neutral'
            if price_up and oi_up:
                signal = 'long_build_up'
            elif price_up and oi_down:
                signal = 'short_covering'
            elif price_down and oi_up:
                signal = 'short_build_up'
            elif price_down and oi_down:
                signal = 'long_unwinding'

            return {
                'signal': signal,
                'oi': oi_now,
                'oi_change': oi_change,
                'oi_change_pct': oi_change_pct
            }
        except Exception:
            return {'signal': 'neutral', 'oi': 0, 'oi_change': 0, 'oi_change_pct': 0.0}


    
    # Default value methods
    def _get_default_indicators(self, df: pd.DataFrame) -> Dict:
        """Return default indicator values when insufficient data."""
        current_price = float(df['close'].iloc[-1]) if not df.empty else 0
        
        return {
            'rsi': self._get_default_rsi(),
            'macd': self._get_default_macd(),
            'ema': self._get_default_ema(),
            'bollinger': self._get_default_bollinger(),
            'keltner': self._get_default_keltner(),
            'supertrend': self._get_default_supertrend(),
            'impulse': self._get_default_impulse(),
            'price': current_price,
            'volatility': {'value': 15.0, 'level': 'normal'}
        }
    
    def _get_default_rsi(self) -> Dict:
        return {
            'value': 50,
            'signal': 'neutral',
            'divergence': 'none',
            'rsi_series': pd.Series([50])
        }
    

    
    def _get_default_macd(self) -> Dict:
        return {
            'macd': 0.0,
            'signal': 0.0,
            'histogram': 0.0,
            'signal_type': 'neutral',
            'crossover': None,
            'macd_series': pd.Series([0.0]),
            'signal_series': pd.Series([0.0]),
            'histogram_series': pd.Series([0.0]),
            'hist_slope_closed': 0.0,
            'hist_slope_forming': 0.0
        }

    
    def _get_default_ema(self) -> Dict:
        return {
            'short': 0,
            'medium': 0,
            'long': 0,
            'signal': 'neutral',
            'crossover': None,
            'short_series': pd.Series([0]),
            'medium_series': pd.Series([0]),
            'long_series': pd.Series([0])
        }
    
    def _get_default_bollinger(self) -> Dict:
        return {
            'upper': 0,
            'middle': 0,
            'lower': 0,
            'position': 0.5,
            'signal': 'neutral',
            'bandwidth': 0,
            'is_squeeze': False,
            'upper_series': pd.Series([0]),
            'middle_series': pd.Series([0]),
            'lower_series': pd.Series([0])
        }
    
    def _get_default_keltner(self) -> Dict:
        return {
            'upper': 0,
            'middle': 0,
            'lower': 0,
            'position': 0.5,
            'signal': 'within',
            'upper_series': pd.Series([0]),
            'middle_series': pd.Series([0]),
            'lower_series': pd.Series([0])
        }
    
    def _get_default_supertrend(self) -> Dict:

        return { 
                'value': 0, 
                'direction': 0, 
                'trend': 'neutral', 
                'signal': 'neutral', 
                'supertrend_series': pd.Series([0]), 
                'direction_series': pd.Series([0]) 
                }
    
    
    def _get_default_impulse(self) -> Dict:
        return {
            'state': 'blue',
            'histogram': 0,
            'ema_slope': 0,
            'strength': 0
        }
