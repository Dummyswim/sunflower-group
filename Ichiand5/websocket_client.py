"""
Enhanced Dhan WebSocket client with comprehensive indicator integration.
Fixed version with proper imports and error handling.
"""
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
from typing import Optional, Dict, Any, List
from pathlib import Path

# Ensure imports are available
try:
    from technical_indicators import TechnicalIndicators, SignalGenerator
    from telegram_bot import TelegramBot
    from signal_prediction import (
        SignalDurationPredictor, 
        SignalPersistence,
        EnhancedSignalFormatter
    )
    import config
    from chart_generator import ChartGenerator
    from signal_validator import SignalValidator

except ImportError as e:
    logging.error(f"Import error: {e}")
    raise

logger = logging.getLogger(__name__)

class EnhancedDhanWebSocketClient:
    """Enhanced WebSocket client with proper error handling and validation."""
    
    def __init__(self, access_token_b64: str, client_id_b64: str, telegram_bot: TelegramBot):
        """Initialize enhanced WebSocket client with all indicators."""
        try:
            # Validate inputs
            if not access_token_b64 or not client_id_b64:
                raise ValueError("Credentials cannot be empty")
            
            self.access_token = base64.b64decode(access_token_b64).decode("utf-8")
            self.client_id = base64.b64decode(client_id_b64).decode("utf-8")
            
            # Validate decoded credentials
            if not self.access_token or not self.client_id:
                raise ValueError("Invalid credentials after decoding")
                
            # Add new components
            self.chart_generator = ChartGenerator(config) if config.alert_with_charts else None
            self.signal_validator = SignalValidator(config)
            self.signal_monitor = signal_monitor
        
        except Exception as e:
            logger.error(f"Failed to decode credentials: {e}")
            raise ValueError("Invalid base64 encoded credentials")
        
        self.telegram_bot = telegram_bot
        self.ws = None
        self.connected = False
        self.running = True  # Flag to control loops
        
        # Enhanced data storage with validation
        self.tick_buffer = collections.deque(maxlen=config.MAX_BUFFER_SIZE)
        self.minute_ohlcv = pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])
        self.last_minute_processed = None
        
        # Signal management
        self.signal_generator = SignalGenerator()
        self.duration_predictor = SignalDurationPredictor()  # Initialize duration predictor
        self.last_alert_time = None
        self.alert_history = collections.deque(maxlen=100)
        self.consecutive_signals = 0
        self.last_signal_type = None
        
        # Connection management
        self.connection_retry_count = 0
        self.max_retries = 5
        self.heartbeat_thread = None
        
        # Performance monitoring
        self.tick_count = 0
        self.last_tick_time = None
        self.processing_times = collections.deque(maxlen=100)
        
        logger.info("Enhanced DhanWebSocketClient initialized with 6 indicators")
    
    def on_open(self, ws):
        """Handle WebSocket connection opened."""
        try:
            logger.info("WebSocket connection established")
            self.connected = True
            self.connection_retry_count = 0
            
            # Start heartbeat
            self.start_heartbeat()
            
            # Subscribe to Nifty50
            subscription = {
                "RequestCode": 15,
                "InstrumentCount": 1,
                "InstrumentList": [{
                    "ExchangeSegment": config.NIFTY_EXCHANGE_SEGMENT,
                    "SecurityId": str(config.NIFTY_SECURITY_ID)
                }]
            }
            
            ws.send(json.dumps(subscription))
            logger.info(f"Subscribed to Nifty50 (ID: {config.NIFTY_SECURITY_ID})")
            
            # Send connection notification
            if self.telegram_bot:
                indicators_list = ", ".join(config.INDICATOR_WEIGHTS.keys())
                self.telegram_bot.send_message(
                    "✅ <b>Enhanced Trading System Connected</b>\n"
                    "📊 Monitoring: NIFTY 50\n"
                    f"📈 Indicators: {indicators_list}\n"
                    f"⚙️ Min Data Points: {config.MIN_DATA_POINTS}\n"
                    f"⏱️ Alert Cooldown: {config.COOLDOWN_SECONDS}s\n"
                    f"💪 Min Signal Strength: {config.MIN_SIGNAL_STRENGTH}"
                )
        except Exception as e:
            logger.error(f"Error in on_open: {e}", exc_info=True)
    
    def on_message(self, ws, message):
        """Process incoming WebSocket messages with performance tracking."""
        start_time = time.time()
        try:
            data = self._parse_packet(message)
            if data:
                self._process_tick(data)
                self.tick_count += 1
                
                # Track processing time
                processing_time = (time.time() - start_time) * 1000  # ms
                self.processing_times.append(processing_time)
                
                # Log performance stats every 100 ticks
                if self.tick_count % 100 == 0:
                    if self.processing_times:
                        avg_time = np.mean(self.processing_times)
                        logger.debug(f"Processed {self.tick_count} ticks, Avg time: {avg_time:.2f}ms")
                    
        except Exception as e:
            logger.error(f"Message processing error: {e}", exc_info=True)
    
    def _parse_packet(self, message) -> Optional[Dict]:
        """Enhanced packet parsing with validation."""
        try:
            if isinstance(message, bytes):
                packet_length = len(message)
                
                # Define packet parsers based on length
                parsers = {
                    16: self._parse_tick_packet,
                    32: self._parse_detailed_packet,
                    44: self._parse_market_depth_packet,
                    108: self._parse_full_packet
                }
                
                parser = parsers.get(packet_length)
                if parser:
                    return parser(message)
                else:
                    logger.debug(f"Unhandled packet length: {packet_length}")
                    return None
            else:
                return self._parse_json_message(str(message))
                
        except Exception as e:
            logger.error(f"Packet parsing error: {e}")
            return None
    
    def _parse_tick_packet(self, data: bytes) -> Optional[Dict]:
        """Parse basic tick packet with validation."""
        try:
            if len(data) != 16:
                logger.warning(f"Invalid tick packet length: {len(data)}")
                return None
            
            code, msg_len, exch_seg, sec_id, ltp, ltt = struct.unpack("<B H B I f I", data)
            
            if sec_id != config.NIFTY_SECURITY_ID:
                return None
            
            # Validate price
            if not (config.PRICE_SANITY_MIN <= ltp <= config.PRICE_SANITY_MAX):
                logger.warning(f"Price sanity check failed: {ltp}")
                return None
            
            return {
                "timestamp": datetime.now(),
                "ltp": ltp,
                "ltt": ltt,
                "packet_type": "tick"
            }
        except struct.error as e:
            logger.error(f"Tick packet unpack error: {e}")
            return None
        except Exception as e:
            logger.error(f"Tick packet parse error: {e}")
            return None
    
    def _parse_detailed_packet(self, data: bytes) -> Optional[Dict]:
        """Parse detailed market data packet with validation."""
        try:
            if len(data) != 32:
                logger.warning(f"Invalid detailed packet length: {len(data)}")
                return None
            
            unpacked = struct.unpack("<B H B I f h I f I I I", data)
            code, msg_len, exch_seg, sec_id, ltp, ltq, ltt, atp, volume, total_sell, total_buy = unpacked
            
            if sec_id != config.NIFTY_SECURITY_ID:
                return None
            
            # Validate price
            if not (config.PRICE_SANITY_MIN <= ltp <= config.PRICE_SANITY_MAX):
                logger.warning(f"Price sanity check failed: {ltp}")
                return None
            
            return {
                "timestamp": datetime.now(),
                "ltp": ltp,
                "ltq": ltq,
                "ltt": ltt,
                "atp": atp,
                "volume": volume,
                "total_sell": total_sell,
                "total_buy": total_buy,
                "packet_type": "detailed"
            }
        except struct.error as e:
            logger.error(f"Detailed packet unpack error: {e}")
            return None
        except Exception as e:
            logger.error(f"Detailed packet parse error: {e}")
            return None
    
    def _parse_market_depth_packet(self, data: bytes) -> Optional[Dict]:
        """Parse market depth packet - placeholder for future implementation."""
        logger.debug("Market depth packet received but not implemented")
        return None
    
    def _parse_full_packet(self, data: bytes) -> Optional[Dict]:
        """Parse full market data packet - placeholder for future implementation."""
        logger.debug("Full packet received but not implemented")
        return None
    
    def _parse_json_message(self, message: str) -> Optional[Dict]:
        """Parse JSON messages with error handling."""
        try:
            data = json.loads(message)
            
            if "Touchline" in data:
                touchline = data["Touchline"]
                return {
                    "timestamp": datetime.now(),
                    "open": touchline.get("Open", 0),
                    "high": touchline.get("High", 0),
                    "low": touchline.get("Low", 0),
                    "close": touchline.get("Close", 0),
                    "ltp": touchline.get("LastTradedPrice", 0),
                    "volume": touchline.get("TotalTradedQuantity", 0),
                    "packet_type": "touchline"
                }
            
            return None
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}")
            return None
        except Exception as e:
            logger.error(f"JSON message parse error: {e}")
            return None
    
    def _process_tick(self, tick_data: Dict):
        """Process tick data and create minute candles with validation."""
        try:
            if not tick_data:
                return
            
            # Validate tick data
            if "timestamp" not in tick_data:
                logger.warning("Tick data missing timestamp")
                return
            
            # Add to buffer
            self.tick_buffer.append(tick_data)
            self.last_tick_time = tick_data["timestamp"]
            
            # Get current minute
            current_minute = tick_data["timestamp"].replace(second=0, microsecond=0)
            
            # Check if new minute
            if self.last_minute_processed != current_minute:
                if self.last_minute_processed is not None:
                    self._create_minute_candle(self.last_minute_processed)
                self.last_minute_processed = current_minute
                
        except Exception as e:
            logger.error(f"Tick processing error: {e}", exc_info=True)
    
    def _create_minute_candle(self, minute_timestamp: datetime):
        """Create minute OHLCV candle from tick data with validation."""
        try:
            # Filter ticks for this minute
            minute_ticks = [
                t for t in self.tick_buffer 
                if t["timestamp"].replace(second=0, microsecond=0) == minute_timestamp
                and "ltp" in t
            ]
            
            if not minute_ticks:
                logger.debug(f"No valid ticks for minute {minute_timestamp}")
                return
            
            # Calculate OHLCV
            prices = [t["ltp"] for t in minute_ticks]
            volumes = [t.get("volume", 0) for t in minute_ticks if "volume" in t]
            
            if not prices:
                logger.warning("No prices found in minute ticks")
                return
            
            candle = {
                "timestamp": minute_timestamp,
                "open": prices[0],
                "high": max(prices),
                "low": min(prices),
                "close": prices[-1],
                "volume": volumes[-1] if volumes else 0
            }
            
            # Validate candle data
            if candle["high"] < candle["low"]:
                logger.error(f"Invalid candle: high < low")
                return
            
            # Add to dataframe
            new_candle_df = pd.DataFrame([candle]).set_index("timestamp")
            self.minute_ohlcv = pd.concat([self.minute_ohlcv, new_candle_df])
            
            # Keep only last 500 candles to prevent memory issues
            if len(self.minute_ohlcv) > 500:
                self.minute_ohlcv = self.minute_ohlcv.tail(500)
            
            logger.debug(f"Candle: O={candle['open']:.2f}, H={candle['high']:.2f}, "
                        f"L={candle['low']:.2f}, C={candle['close']:.2f}, V={candle['volume']}")
            
            # Analyze if enough data
            if len(self.minute_ohlcv) >= config.MIN_DATA_POINTS:
                self._analyze_and_alert()
            else:
                logger.debug(f"Waiting for more data: {len(self.minute_ohlcv)}/{config.MIN_DATA_POINTS}")
                
        except Exception as e:
            logger.error(f"Candle creation error: {e}", exc_info=True)
    
    def _analyze_and_alert(self):
        """Perform comprehensive technical analysis with all 6 indicators."""
        try:
            # Prepare data
            df = self.minute_ohlcv.tail(config.MIN_DATA_POINTS).copy()
            
            if df.empty or len(df) < config.MIN_DATA_POINTS:
                logger.warning("Insufficient data for analysis")
                return
            
            logger.debug(f"Analyzing {len(df)} candles")
            
            # Calculate all indicators
            indicators = {}
            
            # 1. Ichimoku Cloud
            indicators['ichimoku'] = TechnicalIndicators.calculate_ichimoku(
                df, config.ICHIMOKU_PARAMS
            )
            if indicators['ichimoku']:
                indicators['ichimoku']['close'] = df['close']
            
            # 2. Stochastic Oscillator
            indicators['stochastic'] = TechnicalIndicators.calculate_stochastic(
                df, config.STOCHASTIC_PARAMS
            )
            
            # 3. On-Balance Volume
            indicators['obv'] = TechnicalIndicators.calculate_obv(
                df, config.OBV_PARAMS
            )
            
            # 4. Bollinger Bands
            indicators['bollinger'] = TechnicalIndicators.calculate_bollinger_bands(
                df, config.BOLLINGER_PARAMS
            )
            
            # 5. ADX
            indicators['adx'] = TechnicalIndicators.calculate_adx(
                df, config.ADX_PARAMS
            )
            
            # 6. ATR
            indicators['atr'] = TechnicalIndicators.calculate_atr(
                df, config.ATR_PARAMS
            )
            
            # Validate indicators
            if not all(indicators.values()):
                logger.warning("Some indicators failed to calculate")
            
            # Generate weighted signal
            signal_result = self.signal_generator.calculate_weighted_signal(
                df, indicators, config.INDICATOR_WEIGHTS
            )
            
            # Predict signal duration
            duration_prediction = self.duration_predictor.predict_signal_duration(
                df, 
                indicators, 
                signal_result,
                timeframe_minutes=1
            )
            
            # Track consecutive signals
            if signal_result['composite_signal'] == self.last_signal_type:
                self.consecutive_signals += 1
            else:
                self.consecutive_signals = 1
                self.last_signal_type = signal_result['composite_signal']
            
            # Log analysis
            logger.info(f"Analysis: {signal_result['composite_signal']} "
                       f"(Score: {signal_result['weighted_score']}, "
                       f"Confidence: {signal_result['confidence']}%, "
                       f"Expected Duration: {duration_prediction.expected_candles} candles)")
            
            # Check alert conditions
            if self._should_alert(signal_result):
                self._send_enhanced_duration_alert(
                    df['close'].iloc[-1], 
                    indicators, 
                    signal_result,
                    duration_prediction
                )

            # NEW: Validate signal
            is_valid, validation_details = self.signal_validator.validate_signal(
                signal_result, df, indicators
            )
            
            if not is_valid:
                logger.info(f"Signal rejected: {validation_details.get('rejection_reasons')}")
                return
            
            # NEW: Generate chart if enabled
            chart_path = None
            if self.chart_generator and CONFIG.alert_with_charts:
                chart_path = self.chart_generator.generate_signal_chart(
                    df, indicators, signal_result
                )
            
            # Send alert with chart
            if chart_path:
                self.telegram_bot.send_chart(message, chart_path)
            else:
                self.telegram_bot.send_message(message)
                
        except Exception as e:
            logger.error(f"Analysis error: {e}")
                            
    
    def _should_alert(self, signal_result: Dict) -> bool:
        """Enhanced alert conditions with multiple checks."""
        try:
            # Check signal strength
            if abs(signal_result.get('weighted_score', 0)) < config.MIN_SIGNAL_STRENGTH:
                logger.debug(f"Signal too weak: {signal_result.get('weighted_score', 0)}")
                return False
            
            # Check confidence
            if signal_result.get('confidence', 0) < config.MIN_CONFIDENCE:
                logger.debug(f"Confidence too low: {signal_result.get('confidence', 0)}")
                return False
            
            # Check active indicators
            if signal_result.get('active_indicators', 0) < config.MIN_ACTIVE_INDICATORS:
                logger.debug(f"Too few active indicators: {signal_result.get('active_indicators', 0)}")
                return False
            
            # Check cooldown
            if self.last_alert_time:
                elapsed = (datetime.now() - self.last_alert_time).total_seconds()
                if elapsed < config.COOLDOWN_SECONDS:
                    remaining = config.COOLDOWN_SECONDS - elapsed
                    logger.debug(f"Alert cooldown active: {remaining:.0f}s remaining")
                    return False
            
            # Extra check for strong signals or crossovers
            is_strong = signal_result.get('composite_signal', '') in ['STRONG BUY', 'STRONG SELL']
            has_crossover = any(signal_result.get('crossovers', {}).values())
            
            # Require consecutive signals for non-strong signals
            if not is_strong and not has_crossover and self.consecutive_signals < 2:
                logger.debug("Waiting for signal confirmation")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Alert condition check error: {e}")
            return False
    
    def _send_enhanced_duration_alert(self, current_price: float, indicators: Dict, 
                                      signal_result: Dict, duration_prediction: SignalPersistence):
        """Send alert with signal duration predictions."""
        try:
            # Format message with duration predictions
            message = EnhancedSignalFormatter.format_duration_alert(
                current_price,
                signal_result,
                duration_prediction,
                datetime.now()
            )
            
            # Send to Telegram
            if self.telegram_bot:
                success = self.telegram_bot.send_message(message)
                if success:
                    self.last_alert_time = datetime.now()
                    self.alert_history.append({
                        "timestamp": datetime.now(),
                        "signal": signal_result['composite_signal'],
                        "score": signal_result['weighted_score'],
                        "price": current_price,
                        "expected_duration": duration_prediction.expected_candles,
                        "confidence": duration_prediction.confidence_level
                    })
                    logger.info(f"Duration-enhanced alert sent: {signal_result['composite_signal']} "
                               f"for {duration_prediction.expected_candles} candles")
                else:
                    logger.error("Failed to send duration alert")
                    
        except Exception as e:
            logger.error(f"Duration alert error: {e}", exc_info=True)
    
    def start_heartbeat(self):
        """Start heartbeat thread to maintain connection."""
        def heartbeat():
            while self.connected and self.running:
                try:
                    if self.ws:
                        self.ws.send(json.dumps({"ping": 1}))
                        logger.debug("Heartbeat sent")
                    time.sleep(30)
                except Exception as e:
                    logger.error(f"Heartbeat error: {e}")
                    if not self.running:
                        break
                    
        self.heartbeat_thread = threading.Thread(target=heartbeat, daemon=True)
        self.heartbeat_thread.start()
        logger.debug("Heartbeat thread started")
    
    def on_error(self, ws, error):
        """Handle WebSocket errors."""
        logger.error(f"WebSocket error: {error}")
        
    def on_close(self, ws, close_status_code, close_msg):
        """Handle WebSocket closure with auto-reconnect."""
        logger.warning(f"WebSocket closed - Code: {close_status_code}, Message: {close_msg}")
        self.connected = False
        
        if not self.running:
            logger.info("WebSocket closed intentionally")
            return
        
        if self.connection_retry_count < self.max_retries:
            self.connection_retry_count += 1
            wait_time = min(2 ** self.connection_retry_count, 60)
            logger.info(f"Reconnection attempt {self.connection_retry_count}/{self.max_retries} in {wait_time}s")
            
            if self.telegram_bot:
                self.telegram_bot.send_message(
                    f"⚠️ WebSocket disconnected\n"
                    f"🔄 Reconnecting in {wait_time}s..."
                )
            
            time.sleep(wait_time)
            
            if self.running:
                self.connect()
        else:
            logger.error("Max reconnection attempts reached")
            if self.telegram_bot:
                self.telegram_bot.send_message(
                    "❌ Connection failed after maximum retries.\n"
                    "Please check the system."
                )
    
    def connect(self):
        """Establish WebSocket connection with validation."""
        try:
            # Validate credentials
            if not self.access_token or not self.client_id:
                raise ValueError("Missing credentials")
            
            ws_url = (f"wss://api-feed.dhan.co?version=2&token={self.access_token}"
                     f"&clientId={self.client_id}&authType=2")
            
            logger.debug("Creating WebSocket connection")
            
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
            
            # Wait for connection with timeout
            timeout = 15
            while not self.connected and timeout > 0 and self.running:
                time.sleep(1)
                timeout -= 1
            
            if not self.connected and self.running:
                raise ConnectionError("WebSocket connection timeout")
                
            logger.info("WebSocket connected successfully")
            
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
            
            # Send final stats
            if self.telegram_bot and self.tick_count > 0:
                avg_processing = np.mean(self.processing_times) if self.processing_times else 0
                self.telegram_bot.send_message(
                    f"📊 <b>Session Statistics</b>\n"
                    f"Total Ticks: {self.tick_count}\n"
                    f"Candles Created: {len(self.minute_ohlcv)}\n"
                    f"Alerts Sent: {len(self.alert_history)}\n"
                    f"Avg Processing: {avg_processing:.2f}ms"
                )
        except Exception as e:
            logger.error(f"Disconnect error: {e}")



    def set_signal_monitor(self, monitor):
        """Set the signal monitor for tracking."""
        self.signal_monitor = monitor
        logger.info("Signal monitor attached to WebSocket client")

    def _send_enhanced_duration_alert(self, current_price: float, indicators: Dict, 
                                    signal_result: Dict, duration_prediction):
        """Send alert with signal duration predictions and track in monitor."""
        try:
            # Format message with duration predictions
            message = EnhancedSignalFormatter.format_duration_alert(
                current_price,
                signal_result,
                duration_prediction,
                datetime.now()
            )
            
            # Send to Telegram
            if self.telegram_bot:
                success = self.telegram_bot.send_message(message)
                if success:
                    self.last_alert_time = datetime.now()
                    
                    # Create signal data for monitoring
                    signal_data = {
                        "timestamp": datetime.now(),
                        "signal_type": signal_result['composite_signal'],
                        "score": signal_result['weighted_score'],
                        "price": current_price,
                        "expected_duration": duration_prediction.expected_candles,
                        "confidence": duration_prediction.confidence_level
                    }
                    
                    # Add to alert history
                    self.alert_history.append(signal_data)
                    
                    # Track in signal monitor if available
                    if hasattr(self, 'signal_monitor') and self.signal_monitor:
                        self.signal_monitor.add_signal(signal_data)
                    
                    logger.info(f"Duration-enhanced alert sent: {signal_result['composite_signal']} "
                            f"for {duration_prediction.expected_candles} candles")
                else:
                    logger.error("Failed to send duration alert")
                    
        except Exception as e:
            logger.error(f"Duration alert error: {e}", exc_info=True)
