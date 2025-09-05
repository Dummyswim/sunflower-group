# websocket_client.py - ENHANCED VERSION WITH COMPREHENSIVE LOGGING

import base64
import json
import logging
import ssl
import struct
import threading
import time
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import collections
import websocket
from typing import Optional, Dict, List, Tuple, Callable
from config import CONFIG

logger = logging.getLogger(__name__)

class EnhancedDhanWebSocketClient:
    """WebSocket client with comprehensive logging and analysis."""
    
    # Dhan API v2 Feed Request Codes
    SUBSCRIBE_FEED = 15
    UNSUBSCRIBE_FEED = 16
    SUBSCRIBE_ORDER_STATUS = 17
    UNSUBSCRIBE_ORDER_STATUS = 18
    SUBSCRIBE_DPR = 19
    MARKET_STATUS = 21
    HEARTBEAT = 50
    
    # Dhan Response Codes (from binary stream)
    TICKER_PACKET = 2
    QUOTE_PACKET = 4
    OI_PACKET = 5
    PREV_CLOSE_PACKET = 6
    MARKET_STATUS_PACKET = 7
    FULL_PACKET = 8
    DISCONNECT_PACKET = 50
    
    def __init__(self, access_token_b64: str, client_id_b64: str, telegram_bot):
        """Initialize WebSocket client with enhanced logging."""
        try:
            # Decode credentials
            self.access_token = base64.b64decode(access_token_b64).decode("utf-8")
            self.client_id = base64.b64decode(client_id_b64).decode("utf-8")
            
            # Import components
            from technical_indicators import TechnicalIndicators, SignalGenerator
            from signal_prediction import SignalDurationPredictor, EnhancedSignalFormatter
            from signal_validator import SignalValidator
            
            # Initialize components
            self.telegram_bot = telegram_bot
            self.technical_indicators = TechnicalIndicators()
            self.signal_generator = SignalGenerator(CONFIG)
            self.signal_validator = SignalValidator(CONFIG)
            self.signal_monitor = None
            self.duration_predictor = SignalDurationPredictor()
            self.signal_formatter = EnhancedSignalFormatter()
            
            # Metrics callback
            self.metrics_callback = None
            
            # Pattern recognition
            self.detected_patterns = []
            
            # Chart generator
            self.chart_generator = None
            if CONFIG.alert_with_charts:
                try:
                    from chart_generator import ChartGenerator
                    self.chart_generator = ChartGenerator(CONFIG)
                except ImportError:
                    logger.warning("Chart generator not available")
            
            # WebSocket state
            self.ws = None
            self.connected = False
            self.running = True
            
            # Enhanced data storage
            self.tick_buffer = collections.deque(maxlen=CONFIG.MAX_BUFFER_SIZE)
            self.minute_ohlcv = pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])
            self.current_5min_data = []
            self.last_5min_timestamp = None
            
            # Market analysis
            self.market_stats = {
                'session_high': 0,
                'session_low': float('inf'),
                'session_open': 0,
                'current_price': 0,
                'day_volume': 0,
                'price_changes': [],
                'volatility': 0,
                'trend': 'NEUTRAL',
                'support_levels': [],
                'resistance_levels': []
            }
            
            # Volume tracking
            self.volume_analysis = {
                'last_cumulative': 0,
                'current_period': 0,
                'average_volume': 0,
                'volume_trend': 'NEUTRAL',
                'volume_spikes': []
            }
            
            # Signal tracking
            self.signal_stats = {
                'total_generated': 0,
                'bullish_count': 0,
                'bearish_count': 0,
                'neutral_count': 0,
                'strong_signals': 0,
                'weak_signals': 0,
                'last_signal': None,
                'signal_history': collections.deque(maxlen=50),
                'last_analysis': {}
            }
            
            # Alert management
            self.alert_stats = {
                'total_sent': 0,
                'last_alert_time': None,
                'alert_history': collections.deque(maxlen=100),
                'cooldown_active': False
            }
            
            # Performance tracking
            self.performance = {
                'start_time': datetime.now(),
                'tick_count': 0,
                'candle_count': 0,
                'analysis_count': 0,
                'packets_received': {'quote': 0, 'ticker': 0, 'full': 0, 'prev_close': 0, 'oi': 0, 'other': 0}
            }
            
            # Connection management
            self.connection_stats = {
                'connect_time': None,
                'reconnect_count': 0,
                'last_heartbeat': None,
                'heartbeat_failures': 0,
                'connection_retry_count': 0,
                'max_retries': 5,
                'retry_delay': 5
            }
            
            # Additional tracking
            self.last_price = 0
            self.last_logged_price = 0
            self.last_cumulative_volume = 0
            self.current_period_volume = 0
            self.heartbeat_thread = None
            self.reconnect_thread = None
            
            logger.info("🚀 EnhancedDhanWebSocketClient initialized with comprehensive logging")
            
        except Exception as e:
            logger.error(f"Initialization error: {e}")
            raise

    def set_metrics_callback(self, callback: Callable):
        """Set callback for updating main metrics."""
        self.metrics_callback = callback

    def set_signal_monitor(self, monitor):
        """Attach signal monitor for tracking."""
        self.signal_monitor = monitor
        logger.info("Signal monitor attached")

    def _update_market_stats(self, price: float, volume: int = 0):
        """Update market statistics with new data."""
        try:
            self.market_stats['current_price'] = price
            
            # Update session high/low
            if price > self.market_stats['session_high']:
                self.market_stats['session_high'] = price
                logger.info(f"📈 NEW SESSION HIGH: ₹{price:.2f}")
            
            if price < self.market_stats['session_low']:
                self.market_stats['session_low'] = price
                logger.info(f"📉 NEW SESSION LOW: ₹{price:.2f}")
            
            # Track price changes for volatility
            self.market_stats['price_changes'].append(price)
            if len(self.market_stats['price_changes']) > 100:
                self.market_stats['price_changes'].pop(0)
            
            # Calculate volatility
            if len(self.market_stats['price_changes']) > 10:
                prices = np.array(self.market_stats['price_changes'])
                returns = np.diff(prices) / prices[:-1]
                self.market_stats['volatility'] = np.std(returns) * np.sqrt(252) * 100
            
            # Update volume stats
            if volume > 0:
                self.market_stats['day_volume'] += volume
                self.volume_analysis['current_period'] += volume
                
        except Exception as e:
            logger.error(f"Market stats update error: {e}")

    def _detect_market_patterns(self, df: pd.DataFrame) -> List[Dict]:
        """Detect chart patterns in price data."""
        patterns = []
        
        try:
            if len(df) < 20:
                return patterns
            
            closes = df['close'].values
            highs = df['high'].values
            lows = df['low'].values
            
            # Detect Head and Shoulders
            if self._is_head_and_shoulders(highs, lows):
                patterns.append({
                    'name': 'Head and Shoulders',
                    'type': 'BEARISH',
                    'confidence': 75
                })
                logger.info("🔍 PATTERN: Head and Shoulders detected (BEARISH)")
            
            # Detect Double Top
            if self._is_double_top(highs):
                patterns.append({
                    'name': 'Double Top',
                    'type': 'BEARISH',
                    'confidence': 70
                })
                logger.info("🔍 PATTERN: Double Top detected (BEARISH)")
            
            # Detect Double Bottom
            if self._is_double_bottom(lows):
                patterns.append({
                    'name': 'Double Bottom',
                    'type': 'BULLISH',
                    'confidence': 70
                })
                logger.info("🔍 PATTERN: Double Bottom detected (BULLISH)")
            
            # Detect Triangle patterns
            triangle = self._detect_triangle(highs, lows)
            if triangle:
                patterns.append(triangle)
                logger.info(f"🔍 PATTERN: {triangle['name']} detected ({triangle['type']})")
            
            # Detect Flag pattern
            if self._is_flag_pattern(closes):
                trend = 'BULLISH' if closes[-1] > closes[0] else 'BEARISH'
                patterns.append({
                    'name': 'Flag Pattern',
                    'type': trend,
                    'confidence': 65
                })
                logger.info(f"🔍 PATTERN: Flag pattern detected ({trend})")
            
            self.detected_patterns = patterns
            
            if self.metrics_callback and patterns:
                self.metrics_callback('patterns_detected', len(patterns))
            
        except Exception as e:
            logger.error(f"Pattern detection error: {e}")
        
        return patterns

    def _calculate_support_resistance(self, df: pd.DataFrame) -> Tuple[List[float], List[float]]:
        """Calculate support and resistance levels."""
        try:
            if len(df) < 20:
                return [], []
            
            # Use recent highs and lows
            recent_data = df.tail(50)
            highs = recent_data['high'].values
            lows = recent_data['low'].values
            closes = recent_data['close'].values
            
            # Find pivot points
            pivot = (highs[-1] + lows[-1] + closes[-1]) / 3
            
            # Calculate support levels
            support1 = 2 * pivot - highs[-1]
            support2 = pivot - (highs[-1] - lows[-1])
            support3 = lows[-1] - 2 * (highs[-1] - pivot)
            
            # Calculate resistance levels
            resistance1 = 2 * pivot - lows[-1]
            resistance2 = pivot + (highs[-1] - lows[-1])
            resistance3 = highs[-1] + 2 * (pivot - lows[-1])
            
            supports = sorted([s for s in [support1, support2, support3] if s > 0])
            resistances = sorted([r for r in [resistance1, resistance2, resistance3] if r > 0])
            
            self.market_stats['support_levels'] = supports
            self.market_stats['resistance_levels'] = resistances
            
            current_price = closes[-1]
            nearest_support = min(supports, key=lambda x: abs(x - current_price)) if supports else 0
            nearest_resistance = min(resistances, key=lambda x: abs(x - current_price)) if resistances else 0
            
            logger.info(f"📊 S/R LEVELS: Support: ₹{nearest_support:.2f} | "
                       f"Resistance: ₹{nearest_resistance:.2f}")
            
            return supports, resistances
            
        except Exception as e:
            logger.error(f"Support/Resistance calculation error: {e}")
            return [], []

    def _analyze_volume_pattern(self, df: pd.DataFrame) -> str:
        """Analyze volume patterns and trends."""
        try:
            if len(df) < 10:
                return "INSUFFICIENT_DATA"
            
            volumes = df['volume'].tail(20).values
            avg_volume = np.mean(volumes)
            recent_volume = np.mean(volumes[-5:])
            
            # Detect volume trend
            if recent_volume > avg_volume * 1.5:
                trend = "HIGH_VOLUME"
                logger.info(f"📊 VOLUME ALERT: High volume detected ({recent_volume/avg_volume:.1f}x average)")
            elif recent_volume > avg_volume * 1.2:
                trend = "INCREASING"
            elif recent_volume < avg_volume * 0.7:
                trend = "LOW_VOLUME"
                logger.info(f"📊 VOLUME ALERT: Low volume detected ({recent_volume/avg_volume:.1f}x average)")
            elif recent_volume < avg_volume * 0.9:
                trend = "DECREASING"
            else:
                trend = "NORMAL"
            
            self.volume_analysis['volume_trend'] = trend
            self.volume_analysis['average_volume'] = avg_volume
            
            # Detect volume spikes
            for i in range(1, len(volumes)):
                if volumes[i] > volumes[i-1] * 2:
                    self.volume_analysis['volume_spikes'].append({
                        'timestamp': df.index[-20+i],
                        'volume': volumes[i],
                        'multiplier': volumes[i] / volumes[i-1]
                    })
            
            return trend
            
        except Exception as e:
            logger.error(f"Volume analysis error: {e}")
            return "ERROR"

    def _process_tick(self, tick_data: Dict):
        """Enhanced tick processing with detailed logging."""
        try:
            if not tick_data or "timestamp" not in tick_data:
                return
            
            # Validate and extract data
            ltp = tick_data.get("ltp", 0)
            if ltp <= 0:
                return
            
            # Update performance counter
            self.performance['tick_count'] += 1
            
            # Update market stats
            volume = tick_data.get("volume", 0)
            self._update_market_stats(ltp, volume)
            
            # Log significant price movements - FIXED: Check for non-zero value
            if hasattr(self, 'last_logged_price') and self.last_logged_price > 0:
                price_change = ltp - self.last_logged_price
                price_change_pct = (price_change / self.last_logged_price) * 100
                
                if abs(price_change_pct) > 0.1:  # Log if change > 0.1%
                    direction = "📈" if price_change > 0 else "📉"
                    logger.info(f"{direction} PRICE MOVE: ₹{self.last_logged_price:.2f} → ₹{ltp:.2f} "
                            f"({price_change:+.2f}, {price_change_pct:+.2%})")
                    self.last_logged_price = ltp
            else:
                self.last_logged_price = ltp
                self.market_stats['session_open'] = ltp
                logger.info(f"📊 SESSION OPEN: ₹{ltp:.2f}")
            
            # Create synthetic volume for index
            if volume == 0:
                if hasattr(self, 'last_price') and self.last_price > 0:  # ALSO CHECK last_price > 0
                    price_change = abs(ltp - self.last_price)
                    price_change_pct = (price_change / self.last_price) * 100
                    
                    # Enhanced synthetic volume calculation
                    base_volume = 1000
                    movement_volume = int(price_change_pct * 5000)
                    tick_volume = 100 * (self.performance['tick_count'] % 10 + 1)
                    synthetic_volume = base_volume + movement_volume + tick_volume
                    
                    tick_data["volume"] = synthetic_volume
                    
                    if self.performance['tick_count'] % 25 == 0:
                        logger.info(f"📊 SYNTHETIC VOLUME: {synthetic_volume} "
                                f"(price moved {price_change:.2f}, "
                                f"volatility: {self.market_stats['volatility']:.2f}%)")
                else:
                    tick_data["volume"] = 1000
                
                self.last_price = ltp
            
            # Add to buffer
            self.tick_buffer.append(tick_data)
            
            # Process 5-minute candle logic
            current_time = tick_data["timestamp"]
            minute = current_time.minute
            rounded_minute = (minute // 5) * 5
            current_5min = current_time.replace(minute=rounded_minute, second=0, microsecond=0)
            
            # Track candle progress
            if self.last_5min_timestamp is None:
                self.last_5min_timestamp = current_5min
                self.current_5min_data = [tick_data]
                logger.info(f"🕐 CANDLE START: {current_5min.strftime('%H:%M')} "
                        f"(Need {CONFIG.MIN_DATA_POINTS} candles for analysis)")
                
            elif self.last_5min_timestamp != current_5min:
                # Complete previous candle
                self._create_5min_candle(self.last_5min_timestamp, self.current_5min_data)
                
                # Start new candle
                self.last_5min_timestamp = current_5min
                self.current_5min_data = [tick_data]
                
                # Log candle progress
                candles_collected = len(self.minute_ohlcv)
                candles_needed = CONFIG.MIN_DATA_POINTS
                if candles_collected < candles_needed:
                    remaining = candles_needed - candles_collected
                    eta_minutes = remaining * 5
                    logger.info(f"🕐 NEW CANDLE: {current_5min.strftime('%H:%M')} | "
                            f"Progress: {candles_collected}/{candles_needed} "
                            f"({remaining} remaining, ETA: ~{eta_minutes} min)")
                else:
                    logger.info(f"🕐 NEW CANDLE: {current_5min.strftime('%H:%M')} | "
                            f"Ready for analysis ({candles_collected} candles)")
            else:
                # Accumulate data for current candle
                self.current_5min_data.append(tick_data)
                
                # Log candle building progress periodically
                if len(self.current_5min_data) % 20 == 0:
                    elapsed = (current_time - self.last_5min_timestamp).seconds
                    remaining = 300 - elapsed  # 5 minutes = 300 seconds
                    logger.debug(f"Building candle: {len(self.current_5min_data)} ticks, "
                            f"{remaining}s remaining")
                
        except Exception as e:
            logger.error(f"Tick processing error: {e}", exc_info=True)


    def _create_5min_candle(self, candle_timestamp: datetime, period_ticks: List[Dict]):
        """Create 5-minute candle with enhanced analysis."""
        try:
            if not period_ticks:
                return
            
            # Extract price data
            prices = [tick.get("ltp", 0) for tick in period_ticks if tick.get("ltp", 0) > 0]
            if not prices:
                return
            
            # Calculate OHLC
            open_price = prices[0]
            high_price = max(prices)
            low_price = min(prices)
            close_price = prices[-1]
            
            # Calculate volume
            volumes = [tick.get("volume", 0) for tick in period_ticks]
            candle_volume = sum(volumes)
            
            # Calculate additional metrics
            price_range = high_price - low_price
            body_size = abs(close_price - open_price)
            is_bullish = close_price > open_price
            
            # Determine candle type
            if body_size < price_range * 0.1:
                candle_type = "DOJI"
            elif body_size > price_range * 0.8:
                candle_type = "MARUBOZU"
            elif is_bullish and open_price - low_price > body_size * 2:
                candle_type = "HAMMER"
            elif not is_bullish and high_price - open_price > body_size * 2:
                candle_type = "SHOOTING_STAR"
            else:
                candle_type = "NORMAL"
            
            # Create candle
            candle = {
                "timestamp": candle_timestamp,
                "open": open_price,
                "high": high_price,
                "low": low_price,
                "close": close_price,
                "volume": candle_volume
            }
            
            # Add to dataframe
            new_candle_df = pd.DataFrame([candle]).set_index("timestamp")
            if self.minute_ohlcv.empty:
                self.minute_ohlcv = new_candle_df
            else:
                self.minute_ohlcv = pd.concat([self.minute_ohlcv, new_candle_df])
            
            # Limit dataframe size
            if len(self.minute_ohlcv) > 500:
                self.minute_ohlcv = self.minute_ohlcv.tail(500)
            
            # Update performance
            self.performance['candle_count'] += 1
            
            # Enhanced candle logging
            direction = "🟢" if is_bullish else "🔴"
            logger.info(f"{direction} CANDLE COMPLETE [{candle_timestamp.strftime('%H:%M')}]: "
                       f"O:{open_price:.2f} H:{high_price:.2f} L:{low_price:.2f} C:{close_price:.2f} | "
                       f"Vol:{candle_volume:,} | "
                       f"Range:₹{price_range:.2f} | "
                       f"Type:{candle_type} | "
                       f"Ticks:{len(period_ticks)}")
            
            # Calculate trend after enough candles
            if len(self.minute_ohlcv) >= 5:
                recent_closes = self.minute_ohlcv['close'].tail(5).values
                if all(recent_closes[i] > recent_closes[i-1] for i in range(1, len(recent_closes))):
                    self.market_stats['trend'] = "STRONG_UP"
                    logger.info("📈 TREND: Strong uptrend detected")
                elif all(recent_closes[i] < recent_closes[i-1] for i in range(1, len(recent_closes))):
                    self.market_stats['trend'] = "STRONG_DOWN"
                    logger.info("📉 TREND: Strong downtrend detected")
                elif recent_closes[-1] > recent_closes[0]:
                    self.market_stats['trend'] = "UP"
                elif recent_closes[-1] < recent_closes[0]:
                    self.market_stats['trend'] = "DOWN"
                else:
                    self.market_stats['trend'] = "SIDEWAYS"
            
            # Trigger analysis if enough data
            if len(self.minute_ohlcv) >= CONFIG.MIN_DATA_POINTS:
                logger.info(f"✅ DATA READY: {len(self.minute_ohlcv)} candles available for analysis")
                self.analyze_and_alert(self.minute_ohlcv)
            else:
                remaining = CONFIG.MIN_DATA_POINTS - len(self.minute_ohlcv)
                progress_pct = (len(self.minute_ohlcv) / CONFIG.MIN_DATA_POINTS) * 100
                logger.info(f"📊 DATA COLLECTION: {len(self.minute_ohlcv)}/{CONFIG.MIN_DATA_POINTS} candles "
                          f"({progress_pct:.1f}%) | "
                          f"{remaining} more needed | "
                          f"ETA: ~{remaining * 5} minutes")
                
        except Exception as e:
            logger.error(f"Candle creation error: {e}", exc_info=True)

    def analyze_and_alert(self, df: pd.DataFrame):
        """Enhanced analysis with comprehensive logging."""
        try:
            if df.empty or len(df) < CONFIG.MIN_DATA_POINTS:
                return
            
            self.performance['analysis_count'] += 1
            
            logger.info("=" * 50)
            logger.info(f"🔬 ANALYSIS #{self.performance['analysis_count']} STARTED")
            
            # Calculate all indicators
            indicators = {
                "rsi": self.technical_indicators.calculate_rsi(df, CONFIG.RSI_PARAMS),
                "macd": self.technical_indicators.calculate_macd(df, CONFIG.MACD_PARAMS),
                "vwap": self.technical_indicators.calculate_vwap(df, CONFIG.VWAP_PARAMS),
                "bollinger": self.technical_indicators.calculate_bollinger_bands(df, CONFIG.BOLLINGER_PARAMS),
                "obv": self.technical_indicators.calculate_obv(df, CONFIG.OBV_PARAMS),
                "price": df['close'].iloc[-1]
            }
            
            # Store for status reporting
            self.signal_stats['last_analysis'] = {
                'rsi': indicators['rsi'].iloc[-1] if not indicators['rsi'].empty else 50,
                'macd_signal': indicators['macd']['signal'].iloc[-1] if 'signal' in indicators['macd'] else 0,
                'volume_trend': self.volume_analysis['volume_trend'],
                'signal_strength': 0,
                'pattern': None,
                'pattern_confidence': 0
            }
            
            # Log indicator values
            current_price = indicators['price']        
            rsi_value = indicators['rsi'].iloc[-1] if not indicators['rsi'].empty else 50
            macd_histogram = indicators['macd']['hist'].iloc[-1] if 'hist' in indicators['macd'] else 0
            
            logger.info(f"📊 INDICATORS:")
            logger.info(f"  • Price: ₹{current_price:.2f}")
            logger.info(f"  • RSI: {rsi_value:.1f}")
            logger.info(f"  • MACD Histogram: {macd_histogram:.4f}")
            logger.info(f"  • VWAP: ₹{indicators['vwap'].iloc[-1]:.2f}")

            # Calculate OBV trend from the series
            obv_trend = "N/A"
            if 'obv' in indicators and not indicators['obv'].empty and len(indicators['obv']) >= 3:
                recent_obv = indicators['obv'].iloc[-3:]
                obv_slope = np.polyfit(range(len(recent_obv)), recent_obv.values, 1)[0]
                if obv_slope > 0:
                    obv_trend = "BULLISH"
                elif obv_slope < 0:
                    obv_trend = "BEARISH"
                else:
                    obv_trend = "NEUTRAL"
                logger.info(f"  • OBV: {indicators['obv'].iloc[-1]:.0f} (Trend: {obv_trend})")
            else:
                logger.info(f"  • OBV: N/A")
                

            if 'bollinger' in indicators:
                bb = indicators['bollinger']
                logger.info(f"  • Bollinger: Upper={bb['upper'].iloc[-1]:.2f}, "
                            f"Middle={bb['middle'].iloc[-1]:.2f}, Lower={bb['lower'].iloc[-1]:.2f}")
                
                            
            # Detect patterns
            patterns = self._detect_market_patterns(df)
            if patterns:
                logger.info(f"📍 PATTERNS DETECTED: {len(patterns)}")
                for pattern in patterns:
                    logger.info(f"  • {pattern['name']}: {pattern['type']} "
                              f"(Confidence: {pattern['confidence']}%)")
                
                # Update last analysis with pattern info
                self.signal_stats['last_analysis']['pattern'] = patterns[0]['name']
                self.signal_stats['last_analysis']['pattern_confidence'] = patterns[0]['confidence']
            
            # Calculate support/resistance
            supports, resistances = self._calculate_support_resistance(df)
            
            # Analyze volume
            volume_trend = self._analyze_volume_pattern(df)
            logger.info(f"📊 VOLUME ANALYSIS: {volume_trend}")
            
            # Generate signal
            signal_result = self.signal_generator.calculate_weighted_signal(indicators)
            self.signal_stats['total_generated'] += 1
            
            # Update last analysis
            self.signal_stats['last_analysis']['signal_strength'] = signal_result['weighted_score']
            
            # Track signal types
            signal_type = signal_result['composite_signal']
            if 'BULLISH' in signal_type:
                self.signal_stats['bullish_count'] += 1
            elif 'BEARISH' in signal_type:
                self.signal_stats['bearish_count'] += 1
            else:
                self.signal_stats['neutral_count'] += 1
            
            # Log signal details
            logger.info(f"🎯 SIGNAL GENERATED:")
            logger.info(f"  • Type: {signal_type}")
            logger.info(f"  • Strength: {signal_result['weighted_score']:.2%}")
            logger.info(f"  • Confidence: {signal_result['confidence']:.1f}%")
            logger.info(f"  • Active Indicators: {signal_result['active_indicators']}/{signal_result['total_indicators']}")
            
            # Log individual indicator contributions
            if 'indicator_scores' in signal_result:
                logger.info(f"  • Indicator Contributions:")
                for ind, score in signal_result['indicator_scores'].items():
                    logger.info(f"    - {ind}: {score:.2%}")
            
            # Validate signal
            is_valid, validation_details = self.signal_validator.validate_signal(
                signal_result, df, indicators
            )
            
            if not is_valid:
                reasons = validation_details.get('rejection_reasons', [])
                logger.info(f"❌ SIGNAL REJECTED:")
                for reason in reasons:
                    logger.info(f"  • {reason}")
            else:
                logger.info(f"✅ SIGNAL VALIDATED")
                
                # Check alert conditions
                if self._should_alert(signal_result):
                    # Predict duration
                    persistence = self.duration_predictor.predict_signal_duration(
                        df, indicators, signal_result, timeframe_minutes=5
                    )
                    
                    logger.info(f"⏱️ DURATION PREDICTION:")
                    logger.info(f"  • Expected: {persistence.expected_minutes} minutes")
                    logger.info(f"  • Range: {persistence.min_minutes}-{persistence.max_minutes} minutes")
                    logger.info(f"  • Confidence: {persistence.confidence_level}")
                    
                    # Send alert
                    self._send_enhanced_duration_alert(
                        current_price, indicators, signal_result, persistence
                    )
                    
                    # Update metrics
                    if self.metrics_callback:
                        self.metrics_callback('signals_generated', self.signal_stats['total_generated'])
                        self.metrics_callback('alerts_sent', self.alert_stats['total_sent'])
                    
                    # Generate chart
                    if self.chart_generator and CONFIG.alert_with_charts:
                        self._generate_and_send_chart(df, indicators, signal_result, persistence)
                else:
                    if self.alert_stats['cooldown_active']:
                        remaining = CONFIG.COOLDOWN_SECONDS - (datetime.now() - self.alert_stats['last_alert_time']).seconds
                        logger.info(f"⏳ ALERT COOLDOWN: {remaining}s remaining")
                    else:
                        logger.info(f"📊 Signal below threshold (min: {CONFIG.MIN_SIGNAL_STRENGTH:.1%})")
            
            logger.info("=" * 50)
            
        except Exception as e:
            logger.error(f"Analysis error: {e}", exc_info=True)

    def _should_alert(self, signal_result: Dict) -> bool:
        """Enhanced alert condition checking with logging."""
        try:
            # Check signal strength
            strength = abs(signal_result.get('weighted_score', 0))
            if strength < CONFIG.MIN_SIGNAL_STRENGTH:
                logger.debug(f"Signal strength {strength:.2%} below minimum {CONFIG.MIN_SIGNAL_STRENGTH:.2%}")
                return False
            
            # Check confidence
            confidence = signal_result.get('confidence', 0)
            if confidence < CONFIG.MIN_CONFIDENCE:
                logger.debug(f"Confidence {confidence:.1f}% below minimum {CONFIG.MIN_CONFIDENCE}%")
                return False
            
            # Check active indicators
            active = signal_result.get('active_indicators', 0)
            if active < CONFIG.MIN_ACTIVE_INDICATORS:
                logger.debug(f"Active indicators {active} below minimum {CONFIG.MIN_ACTIVE_INDICATORS}")
                return False
            
            # Check cooldown
            if self.alert_stats['last_alert_time']:
                elapsed = (datetime.now() - self.alert_stats['last_alert_time']).total_seconds()
                if elapsed < CONFIG.COOLDOWN_SECONDS:
                    self.alert_stats['cooldown_active'] = True
                    remaining = CONFIG.COOLDOWN_SECONDS - elapsed
                    logger.debug(f"Cooldown active: {remaining:.0f}s remaining")
                    return False
                else:
                    self.alert_stats['cooldown_active'] = False
            
            # Track strong signals
            if strength > 0.7:
                self.signal_stats['strong_signals'] += 1
                logger.info(f"💪 STRONG SIGNAL DETECTED: {strength:.2%}")
            
            return True
            
        except Exception as e:
            logger.error(f"Alert check error: {e}")
            return False

    def _send_enhanced_duration_alert(self, current_price: float, indicators: Dict,
                                     signal_result: Dict, duration_prediction):
        """Send alert with comprehensive information."""
        try:
            # Log alert details
            logger.info("🔔 " + "=" * 45)
            logger.info(f"🔔 ALERT TRIGGERED")
            logger.info(f"🔔 Signal: {signal_result['composite_signal']}")
            logger.info(f"🔔 Price: ₹{current_price:.2f}")
            logger.info(f"🔔 Strength: {signal_result['weighted_score']:.2%}")
            logger.info(f"🔔 Expected Duration: {duration_prediction.expected_minutes} minutes")
            logger.info("🔔 " + "=" * 45)
            
            # Format message
            message = self.signal_formatter.format_duration_alert(
                current_price,
                signal_result,
                duration_prediction,
                datetime.now()
            )
            
            # Add pattern information if available
            if self.detected_patterns:
                pattern_text = "\n<b>📍 Patterns:</b>\n"
                for pattern in self.detected_patterns[:3]:  # Top 3 patterns
                    pattern_text += f"• {pattern['name']} ({pattern['confidence']}%)\n"
                message += pattern_text
            
            # Add support/resistance levels
            # if self.market_stats['support_levels'] and self.market_stats['resistance_levels']:
            #     # s_r_text = f"\n<b>📊 Key Levels:</b>\n"
            #     s_r_text += f"Support: ₹{self.market_stats['support_levels'][0]:.2f}\n"
            #     s_r_text += f"Resistance: ₹{self.market_stats['resistance_levels'][0]:.2f}"
            #     message += s_r_text
            
            # Send to Telegram
            if self.telegram_bot:
                success = self.telegram_bot.send_message(message)
                if success:
                    self.alert_stats['last_alert_time'] = datetime.now()
                    self.alert_stats['total_sent'] += 1
                    
                    # Track signal
                    signal_data = {
                        "timestamp": datetime.now(),
                        "signal_type": signal_result['composite_signal'],
                        "score": signal_result['weighted_score'],
                        "price": current_price,
                        "expected_duration": duration_prediction.expected_minutes,
                        "confidence": duration_prediction.confidence_level,
                        "patterns": self.detected_patterns
                    }
                    
                    self.alert_stats['alert_history'].append(signal_data)
                    self.signal_stats['last_signal'] = signal_data
                    
                    # Add to monitor
                    if self.signal_monitor:
                        self.signal_monitor.add_signal(signal_data)
                    
                    logger.info(f"✅ Alert #{self.alert_stats['total_sent']} sent successfully")
                else:
                    logger.error("❌ Failed to send alert to Telegram")
                    
        except Exception as e:
            logger.error(f"Alert sending error: {e}", exc_info=True)

    def _generate_and_send_chart(self, df: pd.DataFrame, indicators: Dict, 
                                 signal_result: Dict, persistence):
        """Generate and send chart to Telegram."""
        try:
            timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
            chart_path = f"images/signal_{timestamp_str}.png"
            
            generated_path = self.chart_generator.generate_signal_chart(
                df, indicators, signal_result, output_path=chart_path
            )
            
            if generated_path and self.telegram_bot:
                caption = (
                    f"Signal: {signal_result['composite_signal']}\n"
                    f"Score: {signal_result['weighted_score']:.2%}\n"
                    f"Expected Duration: {persistence.expected_minutes} min"
                )
                self.telegram_bot.send_photo(generated_path, caption)
                
        except Exception as e:
            logger.error(f"Chart generation error: {e}")

    def get_detailed_status(self) -> Dict:
        """Get comprehensive system status."""
        try:
            status = {
                'connected': self.connected,
                'current_price': self.market_stats['current_price'],
                'day_change_pct': 0,
                'volatility': f"{self.market_stats['volatility']:.2f}%",
                'candles_collected': len(self.minute_ohlcv),
                'last_analysis': None,
                'active_signals': []
            }
            
            # Calculate day change
            if self.market_stats['session_open'] > 0:
                day_change = self.market_stats['current_price'] - self.market_stats['session_open']
                status['day_change_pct'] = (day_change / self.market_stats['session_open'])
            
            # Add last analysis info
            if self.signal_stats['last_analysis']:
                status['last_analysis'] = self.signal_stats['last_analysis']
            
            # Add active signals from alert history
            for alert in list(self.alert_stats['alert_history'])[-5:]:
                if (datetime.now() - alert['timestamp']).seconds < alert.get('expected_duration', 0) * 60:
                    current_pnl = ((self.market_stats['current_price'] - alert['price']) / alert['price'])
                    status['active_signals'].append({
                        'type': alert['signal_type'],
                        'entry_price': alert['price'],
                        'timestamp': alert['timestamp'],
                        'pnl_pct': current_pnl,
                        'expected_duration': alert.get('expected_duration', 'N/A')
                    })
            
            return status
            
        except Exception as e:
            logger.error(f"Status generation error: {e}")
            return {'connected': self.connected, 'error': str(e)}

    def get_full_report(self) -> Dict:
        """Generate comprehensive system report."""
        try:
            uptime = datetime.now() - self.performance['start_time']
            
            report = {
                'health_status': 'HEALTHY' if self.connected else 'DISCONNECTED',
                'uptime': str(uptime).split('.')[0],
                'memory_usage': 'N/A',  # Would need psutil
                'market_analysis': {
                    'trend': self.market_stats['trend'],
                    'trend_strength': abs(self.market_stats.get('volatility', 0) / 100),
                    'support': self.market_stats['support_levels'][0] if self.market_stats['support_levels'] else 0,
                    'resistance': self.market_stats['resistance_levels'][0] if self.market_stats['resistance_levels'] else 0,
                    'sentiment': self._calculate_market_sentiment()
                }
            }
            
            return report
            
        except Exception as e:
            logger.error(f"Report generation error: {e}")
            return {'error': str(e)}

    def _calculate_market_sentiment(self) -> str:
        """Calculate overall market sentiment."""
        bullish_score = self.signal_stats['bullish_count']
        bearish_score = self.signal_stats['bearish_count']
        
        if bullish_score > bearish_score * 1.5:
            return "VERY_BULLISH"
        elif bullish_score > bearish_score:
            return "BULLISH"
        elif bearish_score > bullish_score * 1.5:
            return "VERY_BEARISH"
        elif bearish_score > bullish_score:
            return "BEARISH"
        else:
            return "NEUTRAL"

    def get_status(self) -> Dict:
        """Get client status."""
        return {
            'connected': self.connected,
            'running': self.running,
            'packets_received': self.performance['packets_received'],
            'candles_count': len(self.minute_ohlcv),
            'last_tick': self.performance.get('tick_count', 0),
            'alerts_sent': self.alert_stats['total_sent']
        }

    # Pattern detection helper methods
    def _is_head_and_shoulders(self, highs, lows):
        """Detect head and shoulders pattern."""
        if len(highs) < 5:
            return False
        # Simplified detection logic
        return highs[2] > highs[1] and highs[2] > highs[3] and highs[1] > highs[0] and highs[3] > highs[4]
    
    def _is_double_top(self, highs):
        """Detect double top pattern."""
        if len(highs) < 5:
            return False
        # Simplified detection logic
        peak1 = max(highs[:len(highs)//2])
        peak2 = max(highs[len(highs)//2:])
        return abs(peak1 - peak2) / peak1 < 0.02  # Within 2%
    
    def _is_double_bottom(self, lows):
        """Detect double bottom pattern."""
        if len(lows) < 5:
            return False
        # Simplified detection logic
        trough1 = min(lows[:len(lows)//2])
        trough2 = min(lows[len(lows)//2:])
        return abs(trough1 - trough2) / trough1 < 0.02  # Within 2%
    
    def _detect_triangle(self, highs, lows):
        """Detect triangle patterns."""
        if len(highs) < 5:
            return None
        # Simplified triangle detection
        highs_trend = highs[-1] - highs[0]
        lows_trend = lows[-1] - lows[0]
        
        if highs_trend < 0 and lows_trend > 0:
            return {'name': 'Symmetrical Triangle', 'type': 'NEUTRAL', 'confidence': 60}
        elif highs_trend < 0 and abs(lows_trend) < abs(highs_trend) * 0.2:
            return {'name': 'Descending Triangle', 'type': 'BEARISH', 'confidence': 65}
        elif lows_trend > 0 and abs(highs_trend) < abs(lows_trend) * 0.2:
            return {'name': 'Ascending Triangle', 'type': 'BULLISH', 'confidence': 65}
        return None
    
    def _is_flag_pattern(self, closes):
        """Detect flag pattern."""
        if len(closes) < 10:
            return False
        # Simplified flag detection
        first_half = closes[:len(closes)//2]
        second_half = closes[len(closes)//2:]
        return np.std(second_half) < np.std(first_half) * 0.5

    # ============== ORIGINAL METHODS FROM UPLOADED FILE ==============

    def _parse_json_response(self, json_data: Dict) -> Optional[Dict]:
        """Parse JSON response from WebSocket."""
        try:
            # Handle different JSON response types
            if 'type' in json_data:
                response_type = json_data.get('type')
                
                if response_type == 'market_status':
                    logger.info(f"Market status: {json_data.get('status', 'unknown')}")
                    return None
                    
                elif response_type == 'subscription_status':
                    logger.info(f"Subscription status: {json_data.get('message', 'unknown')}")
                    return None
                    
                elif response_type == 'error':
                    logger.error(f"Server error: {json_data.get('message', 'unknown')}")
                    return None
            
            # Check if it's a data packet
            if 'data' in json_data:
                return self._parse_json_data_packet(json_data['data'])
                
            return None
            
        except Exception as e:
            logger.debug(f"JSON response parsing error: {e}")
            return None

    def _parse_json_data_packet(self, data: Dict) -> Optional[Dict]:
        """Parse JSON data packet (fallback for non-binary data)."""
        try:
            # Extract relevant fields if they exist
            result = {}
            
            if 'ltp' in data:
                result['ltp'] = float(data['ltp'])
            if 'volume' in data:
                result['volume'] = int(data['volume'])
            if 'timestamp' in data:
                result['timestamp'] = datetime.fromtimestamp(data['timestamp'])
            else:
                result['timestamp'] = datetime.now()
                
            if result.get('ltp'):
                return result
                
            return None
            
        except Exception as e:
            logger.debug(f"JSON data packet parsing error: {e}")
            return None

    def _parse_binary_packet(self, data: bytes) -> Optional[Dict]:
        """Parse binary packet - handles both standard and extended formats."""
        try:
            if not data or len(data) < 8:
                return None
            
            # Get packet type from first byte
            packet_type = data[0]
            
            # Route to appropriate parser
            if packet_type == self.TICKER_PACKET and len(data) >= 16:
                return self._parse_ticker_packet(data)
            elif packet_type == self.QUOTE_PACKET and len(data) >= 50:
                # Handle both 50-byte and 66-byte Quote packets
                return self._parse_quote_packet(data)
            elif packet_type == self.FULL_PACKET and len(data) >= 162:
                return self._parse_full_packet(data)
            elif packet_type == self.PREV_CLOSE_PACKET and len(data) >= 16:
                return self._parse_prev_close_packet(data)
            elif packet_type == self.OI_PACKET and len(data) >= 12:
                return self._parse_oi_packet(data)
            elif packet_type == self.DISCONNECT_PACKET:
                self._handle_disconnect_packet(data)
                return None
            else:
                self.performance['packets_received']['other'] += 1
                return None
                
        except Exception as e:
            logger.debug(f"Binary parsing error: {e}")
            return None

    def _parse_ticker_packet(self, data: bytes) -> Optional[Dict]:
        """Parse Ticker Packet."""
        try:
            if len(data) < 16:
                logger.debug(f"Ticker packet too short: {len(data)} bytes")
                return None
                
            # Parse header (bytes 0-7)
            response_code = data[0]
            if response_code != self.TICKER_PACKET:
                return None
                
            message_length = struct.unpack('<H', data[1:3])[0]
            exchange_segment = data[3]
            security_id = struct.unpack('<I', data[4:8])[0]
            
            if security_id == CONFIG.NIFTY_SECURITY_ID:
                # Parse ticker data (bytes 8-15)
                ltp = struct.unpack('<f', data[8:12])[0]
                ltt = struct.unpack('<I', data[12:16])[0]
                
                if CONFIG.PRICE_SANITY_MIN <= ltp <= CONFIG.PRICE_SANITY_MAX:
                    self.performance['packets_received']['ticker'] += 1
                    
                    # Convert timestamp
                    try:
                        timestamp = datetime.fromtimestamp(ltt)
                    except:
                        timestamp = datetime.now()
                    
                    logger.debug(f"Ticker packet parsed: LTP={ltp:.2f}, Time={timestamp.strftime('%H:%M:%S')}")
                    
                    return {
                        "timestamp": timestamp,
                        "ltp": ltp,
                        "volume": 0,  # Ticker doesn't have volume
                        "packet_type": "ticker"
                    }
                else:
                    logger.warning(f"Ticker price sanity check failed: {ltp}")
                    
        except Exception as e:
            logger.error(f"Ticker packet parse error: {e}", exc_info=True)
        return None

    def _parse_quote_packet(self, data: bytes) -> Optional[Dict]:
        """Parse Quote Packet - Handles both 50 and 66 byte variants."""
        try:
            # Accept both 50-byte and 66-byte packets
            if len(data) < 50:
                logger.debug(f"Quote packet too short: {len(data)} bytes")
                return None
            
            # Parse Response Header (bytes 0-7)
            response_code = data[0]
            if response_code != self.QUOTE_PACKET:
                return None
            
            message_length = struct.unpack('<H', data[1:3])[0]
            exchange_segment = data[3]
            security_id = struct.unpack('<I', data[4:8])[0]
            
            # Verify this is NIFTY50 data
            if security_id != CONFIG.NIFTY_SECURITY_ID:
                return None
            
            # Parse Quote Data (bytes 8-49)
            ltp = struct.unpack('<f', data[8:12])[0]
            ltq = struct.unpack('<h', data[12:14])[0]
            ltt = struct.unpack('<I', data[14:18])[0]
            atp = struct.unpack('<f', data[18:22])[0]
            volume = struct.unpack('<I', data[22:26])[0]
            total_sell_qty = struct.unpack('<I', data[26:30])[0]
            total_buy_qty = struct.unpack('<I', data[30:34])[0]
            open_value = struct.unpack('<f', data[34:38])[0]
            close_value = struct.unpack('<f', data[38:42])[0]
            high_value = struct.unpack('<f', data[42:46])[0]
            low_value = struct.unpack('<f', data[46:50])[0]
            
            # If 66-byte packet, might have additional fields (ignore for now)
            if len(data) == 66:
                logger.debug("Extended Quote packet received (66 bytes)")
            
            # Validate price
            if not (CONFIG.PRICE_SANITY_MIN <= ltp <= CONFIG.PRICE_SANITY_MAX):
                logger.warning(f"Price sanity check failed: {ltp}")
                return None
            
            # Convert timestamp
            try:
                trade_time = datetime.fromtimestamp(ltt)
            except:
                trade_time = datetime.now()
            
            parsed_data = {
                "timestamp": trade_time,
                "ltp": ltp,
                "ltq": ltq,
                "ltt": ltt,
                "atp": atp,
                "volume": volume,  # Will be 0 for index
                "total_sell_qty": total_sell_qty,
                "total_buy_qty": total_buy_qty,
                "open": open_value if open_value > 0 else ltp,
                "close": close_value if close_value > 0 else ltp,
                "high": high_value if high_value > 0 else ltp,
                "low": low_value if low_value > 0 else ltp,
                "packet_type": "quote",
                "security_id": security_id
            }
            
            self.performance['packets_received']['quote'] += 1
            if self.performance['packets_received']['quote'] % 10 == 0:
                logger.info(f"Quote packet #{self.performance['packets_received']['quote']}: "
                        f"LTP={ltp:.2f}, Volume={volume}, "
                        f"OHLC=[{open_value:.2f}, {high_value:.2f}, {low_value:.2f}, {close_value:.2f}]")
            
            return parsed_data
            
        except Exception as e:
            logger.error(f"Quote packet parsing error: {e}", exc_info=True)
            return None

    def _parse_full_packet(self, data: bytes) -> Optional[Dict]:
        """Parse Full Packet."""
        try:
            if len(data) < 162:
                return None
                
            # Parse header (bytes 0-7)
            response_code = data[0]
            if response_code != self.FULL_PACKET:
                return None
                
            message_length = struct.unpack('<H', data[1:3])[0]
            exchange_segment = data[3]
            security_id = struct.unpack('<I', data[4:8])[0]
            
            if security_id == CONFIG.NIFTY_SECURITY_ID:
                # Parse main data (bytes 8-61)
                ltp = struct.unpack('<f', data[8:12])[0]
                ltq = struct.unpack('<h', data[12:14])[0]
                ltt = struct.unpack('<I', data[14:18])[0]
                atp = struct.unpack('<f', data[18:22])[0]
                volume = struct.unpack('<I', data[22:26])[0]
                total_sell_qty = struct.unpack('<I', data[26:30])[0]
                total_buy_qty = struct.unpack('<I', data[30:34])[0]
                oi = struct.unpack('<I', data[34:38])[0]
                oi_high = struct.unpack('<I', data[38:42])[0]
                oi_low = struct.unpack('<I', data[42:46])[0]
                
                # OHLC values
                open_value = struct.unpack('<f', data[46:50])[0]
                close_value = struct.unpack('<f', data[50:54])[0]
                high_value = struct.unpack('<f', data[54:58])[0]
                low_value = struct.unpack('<f', data[58:62])[0]
                
                if CONFIG.PRICE_SANITY_MIN <= ltp <= CONFIG.PRICE_SANITY_MAX:
                    self.performance['packets_received']['full'] += 1
                    
                    # Parse market depth (bytes 62-161) - 5 levels of 20 bytes each
                    market_depth = []
                    for i in range(5):
                        start = 62 + (i * 20)
                        bid_qty = struct.unpack('<I', data[start:start+4])[0]
                        ask_qty = struct.unpack('<I', data[start+4:start+8])[0]
                        bid_orders = struct.unpack('<H', data[start+8:start+10])[0]
                        ask_orders = struct.unpack('<H', data[start+10:start+12])[0]
                        bid_price = struct.unpack('<f', data[start+12:start+16])[0]
                        ask_price = struct.unpack('<f', data[start+16:start+20])[0]
                        
                        market_depth.append({
                            'bid_qty': bid_qty,
                            'ask_qty': ask_qty,
                            'bid_orders': bid_orders,
                            'ask_orders': ask_orders,
                            'bid_price': bid_price,
                            'ask_price': ask_price
                        })
                    
                    try:
                        timestamp = datetime.fromtimestamp(ltt)
                    except:
                        timestamp = datetime.now()
                    
                    logger.debug(f"Full packet parsed: LTP={ltp:.2f}, Volume={volume}, OI={oi}")
                    
                    return {
                        "timestamp": timestamp,
                        "ltp": ltp,
                        "ltq": ltq,
                        "volume": volume,
                        "open": open_value if open_value > 0 else ltp,
                        "high": high_value if high_value > 0 else ltp,
                        "low": low_value if low_value > 0 else ltp,
                        "close": close_value if close_value > 0 else ltp,
                        "total_buy_qty": total_buy_qty,
                        "total_sell_qty": total_sell_qty,
                        "oi": oi,
                        "market_depth": market_depth,
                        "packet_type": "full"
                    }
        except Exception as e:
            logger.error(f"Full packet parse error: {e}", exc_info=True)
        return None

    def _parse_prev_close_packet(self, data: bytes) -> Optional[Dict]:
        """Parse Previous Close packet."""
        try:
            if len(data) < 16:
                return None
                
            # Parse header
            response_code = data[0]
            if response_code != self.PREV_CLOSE_PACKET:
                return None
                
            message_length = struct.unpack('<H', data[1:3])[0]
            exchange_segment = data[3]
            security_id = struct.unpack('<I', data[4:8])[0]
            
            if security_id == CONFIG.NIFTY_SECURITY_ID:
                prev_close = struct.unpack('<f', data[8:12])[0]
                prev_oi = struct.unpack('<I', data[12:16])[0]
                
                self.performance['packets_received']['prev_close'] += 1
                logger.info(f"Previous close: {prev_close:.2f}, Previous OI: {prev_oi}")
                return {
                    "packet_type": "prev_close",
                    "prev_close": prev_close,
                    "prev_oi": prev_oi
                }
        except Exception as e:
            logger.debug(f"Prev close packet parse error: {e}")
        return None

    def _parse_oi_packet(self, data: bytes) -> Optional[Dict]:
        """Parse OI Data packet."""
        try:
            if len(data) < 12:
                return None
                
            # Parse header
            response_code = data[0]
            if response_code != self.OI_PACKET:
                return None
                
            message_length = struct.unpack('<H', data[1:3])[0]
            exchange_segment = data[3]
            security_id = struct.unpack('<I', data[4:8])[0]
            
            if security_id == CONFIG.NIFTY_SECURITY_ID:
                oi = struct.unpack('<I', data[8:12])[0]
                self.performance['packets_received']['oi'] += 1
                return {
                    "packet_type": "oi",
                    "oi": oi,
                    "timestamp": datetime.now()
                }
        except Exception as e:
            logger.debug(f"OI packet parse error: {e}")
        return None

    def _handle_disconnect_packet(self, data: bytes):
        """Handle disconnection packet."""
        try:
            if len(data) >= 10:
                disconnect_code = struct.unpack('<H', data[8:10])[0]
                logger.warning(f"Disconnect packet received with code: {disconnect_code}")
                
                # Trigger reconnection if needed
                if self.running:
                    self._schedule_reconnect()
        except Exception as e:
            logger.error(f"Disconnect packet handling error: {e}")

    def start_heartbeat(self):
        """Start heartbeat thread to keep connection alive."""
        if self.heartbeat_thread and self.heartbeat_thread.is_alive():
            return
            
        def heartbeat_loop():
            """Send heartbeat every 10 seconds."""
            while self.running and self.connected:
                try:
                    # Send heartbeat message
                    if self.ws and self.connected:
                        heartbeat_msg = {
                            "RequestCode": self.HEARTBEAT
                        }
                        self.ws.send(json.dumps(heartbeat_msg))
                        self.connection_stats['last_heartbeat'] = time.time()
                        logger.debug("Heartbeat sent")
                    
                    time.sleep(10)
                    
                except Exception as e:
                    logger.error(f"Heartbeat error: {e}")
                    time.sleep(5)
        
        self.heartbeat_thread = threading.Thread(target=heartbeat_loop, daemon=True)
        self.heartbeat_thread.start()
        logger.info("Heartbeat thread started")

    def connect_with_retry(self):
        """Connect to WebSocket with retry logic."""
        while self.running and self.connection_stats['connection_retry_count'] < self.connection_stats['max_retries']:
            try:
                logger.info(f"Connection attempt {self.connection_stats['connection_retry_count'] + 1}/{self.connection_stats['max_retries']}")
                self.connect()
                
                # Wait for connection
                timeout = 15
                while not self.connected and timeout > 0:
                    time.sleep(1)
                    timeout -= 1
                
                if self.connected:
                    logger.info("Successfully connected to WebSocket")
                    self.connection_stats['connection_retry_count'] = 0
                    self.connection_stats['connect_time'] = datetime.now()
                    return True
                else:
                    raise ConnectionError("Connection timeout")
                    
            except Exception as e:
                self.connection_stats['connection_retry_count'] += 1
                logger.error(f"Connection attempt {self.connection_stats['connection_retry_count']} failed: {e}")
                
                if self.connection_stats['connection_retry_count'] < self.connection_stats['max_retries']:
                    delay = self.connection_stats['retry_delay'] * self.connection_stats['connection_retry_count']
                    logger.info(f"Retrying in {delay} seconds...")
                    time.sleep(delay)
                else:
                    logger.error("Max retries reached. Connection failed.")
                    if self.telegram_bot:
                        self.telegram_bot.send_message(
                            "⚠️ <b>Connection Failed</b>\n"
                            f"Unable to connect after {self.connection_stats['max_retries']} attempts.\n"
                            "Please check credentials and network."
                        )
                    return False
        
        return False

    def _schedule_reconnect(self):
        """Schedule reconnection attempt."""
        if not self.reconnect_thread or not self.reconnect_thread.is_alive():
            self.reconnect_thread = threading.Thread(
                target=self._reconnect_handler, 
                daemon=True
            )
            self.reconnect_thread.start()

    def _reconnect_handler(self):
        """Handle reconnection with retry."""
        logger.info("Starting reconnection handler...")
        time.sleep(2)  # Brief delay before reconnecting
        
        if self.ws:
            try:
                self.ws.close()
            except:
                pass
        
        self.connected = False
        self.connection_stats['reconnect_count'] += 1
        
        if self.connect_with_retry():
            logger.info("Reconnection successful")
            if self.telegram_bot:
                self.telegram_bot.send_message(
                    "✅ <b>Reconnected</b>\n"
                    "WebSocket connection restored."
                )
        else:
            logger.error("Reconnection failed")

    def on_open(self, ws):
        """Handle WebSocket connection open."""
        try:
            logger.info("WebSocket connection established")
            self.connected = True
            self.connection_stats['connection_retry_count'] = 0
            
            # Subscribe to Quote packets (type 4) which contain volume data
            subscription = {
                "RequestCode": self.SUBSCRIBE_FEED,
                "InstrumentCount": 1,
                "InstrumentList": [{
                    "ExchangeSegment": "IDX_I",
                    "SecurityId": str(CONFIG.NIFTY_SECURITY_ID)
                }]
            }
            
            # Send subscription without SubscriptionCode to get default feed
            ws.send(json.dumps(subscription))
            logger.info(f"Subscribed to NIFTY50 default feed (ID: {CONFIG.NIFTY_SECURITY_ID})")
            
            # Wait a moment then request Quote feed specifically
            time.sleep(0.5)
            
            # Now request Quote packet subscription (code 4)
            quote_subscription = {
                "RequestCode": 15,  # Subscribe
                "InstrumentCount": 1,
                "InstrumentList": [{
                    "ExchangeSegment": "IDX_I",
                    "SecurityId": str(CONFIG.NIFTY_SECURITY_ID)
                }]
            }
            ws.send(json.dumps(quote_subscription))
            logger.info("Requested Quote feed for volume data")
            
            # Start heartbeat
            self.start_heartbeat()
            
            # Send connection notification
            if self.telegram_bot:
                self.telegram_bot.send_message(
                    "✅ <b>Trading System Connected</b>\n"
                    f"Feed: Quote + Ticker packets\n"
                    f"Candles: 5-minute\n"
                    f"Indicators: RSI, MACD, VWAP, BB, OBV\n"
                    f"Pattern Recognition: Enabled\n"
                    f"Time: {datetime.now().strftime('%H:%M:%S')}"
                )
                
        except Exception as e:
            logger.error(f"Error in on_open: {e}", exc_info=True)

    def on_message(self, ws, message):
        """Process incoming WebSocket messages."""
        try:
            data = None
            # logger.debug(f"Message received: {type(message)}, length: {len(message) if message else 0}")
            
            # Parse based on message type
            if isinstance(message, str):
                try:
                    json_data = json.loads(message)
                    data = self._parse_json_response(json_data)
                except json.JSONDecodeError:
                    pass
            
            elif isinstance(message, bytes):
                data = self._parse_binary_packet(message)
            
            # Process valid tick data
            if data and 'ltp' in data:
                self._process_tick(data)
                
                # Log statistics periodically
                if self.performance['tick_count'] % 50 == 0:
                    total = sum(self.performance['packets_received'].values())
                    logger.info(f"📡 PACKETS ({total} total): "
                              f"Quote:{self.performance['packets_received']['quote']} "
                              f"Ticker:{self.performance['packets_received']['ticker']} "
                              f"Full:{self.performance['packets_received']['full']}")
                    
        except Exception as e:
            logger.error(f"Message processing error: {e}", exc_info=True)

    def on_error(self, ws, error):
        """Handle WebSocket errors."""
        logger.error(f"WebSocket error: {error}")
        
        # Check if it's a connection error
        if "connection" in str(error).lower():
            self._schedule_reconnect()

    def on_close(self, ws, close_status_code, close_msg):
        """Handle WebSocket connection close."""
        logger.warning(f"WebSocket closed - Code: {close_status_code}, Message: {close_msg}")
        self.connected = False
        
        # Attempt reconnection if not shutting down
        if self.running:
            self._schedule_reconnect()

    def connect(self):
        """Connect to Dhan WebSocket API."""
        try:
            # Build connection URL
            ws_url = (
                f"wss://api-feed.dhan.co"
                f"?version=2"
                f"&token={self.access_token}"
                f"&clientId={self.client_id}"
                f"&authType=2"
            )
            
            logger.info("Connecting to Dhan WebSocket...")
            
            self.ws = websocket.WebSocketApp(
                ws_url,
                on_open=self.on_open,
                on_message=self.on_message,
                on_error=self.on_error,
                on_close=self.on_close
            )
            
            # Run in separate thread
            wst = threading.Thread(
                target=self.ws.run_forever,
                kwargs={"sslopt": {"cert_reqs": ssl.CERT_NONE}}
            )
            wst.daemon = True
            wst.start()
            
        except Exception as e:
            logger.error(f"Connection error: {e}")
            raise

    def disconnect(self):
        """Gracefully disconnect WebSocket."""
        try:
            self.running = False
            self.connected = False
            
            if self.ws:
                self.ws.close()
                logger.info("WebSocket disconnected")
                
        except Exception as e:
            logger.error(f"Disconnect error: {e}")
