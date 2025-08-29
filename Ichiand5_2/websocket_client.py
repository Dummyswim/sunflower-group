# websocket_client.py - CRITICAL FIXES ONLY

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
from config import CONFIG

logger = logging.getLogger(__name__)

class EnhancedDhanWebSocketClient:
    """Enhanced WebSocket client with fixed method calls and tick processing."""
    
    def __init__(self, access_token_b64: str, client_id_b64: str, telegram_bot):
        """Initialize with proper components."""
        try:
            self.access_token = base64.b64decode(access_token_b64).decode("utf-8")
            self.client_id = base64.b64decode(client_id_b64).decode("utf-8")
            
            # Import components
            from technical_indicators import TechnicalIndicators, SignalGenerator
            from signal_prediction import SignalDurationPredictor, EnhancedSignalFormatter
            from chart_generator import ChartGenerator
            from signal_validator import SignalValidator
            from signal_monitor import SignalMonitor
            
            # Initialize components
            self.telegram_bot = telegram_bot
            self.signal_generator = SignalGenerator(CONFIG)
            self.chart_generator = ChartGenerator(CONFIG) if CONFIG.alert_with_charts else None
            self.signal_validator = SignalValidator(CONFIG)
            self.signal_monitor = None  # Set later via set_signal_monitor
            self.duration_predictor = SignalDurationPredictor()
            
            # WebSocket state
            self.ws = None
            self.connected = False
            self.running = True
            
            # FIX 1: Enhanced data storage for proper tick accumulation
            self.tick_buffer = collections.deque(maxlen=CONFIG.MAX_BUFFER_SIZE)
            self.minute_ohlcv = pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])
            self.current_minute_data = []  # Store all ticks for current minute
            self.last_minute_timestamp = None
            
            # Signal management
            self.last_alert_time = None
            self.alert_history = collections.deque(maxlen=100)
            
            # Connection management
            self.connection_retry_count = 0
            self.max_retries = 5
            self.heartbeat_thread = None
            
            # Performance monitoring
            self.tick_count = 0
            self.last_tick_time = None
            
            logger.info("EnhancedDhanWebSocketClient initialized")
            
        except Exception as e:
            logger.error(f"Initialization error: {e}")
            raise

    def on_open(self, ws):
        """Handle WebSocket connection with proper Dhan v2 subscription."""
        try:
            logger.info("WebSocket connection established")
            self.connected = True
            self.connection_retry_count = 0
            
            # FIX 4: Proper Dhan API v2 subscription format [[2]]
            subscription = {
                "RequestCode": 15,
                "InstrumentCount": 1,
                "InstrumentList": [{
                    "ExchangeSegment": "IDX_I",  # Index segment for NIFTY
                    "SecurityId": str(CONFIG.NIFTY_SECURITY_ID),
                    "SubscriptionCode": 3  # Subscribe to tick + market depth
                }]
            }
            
            ws.send(json.dumps(subscription))
            logger.info(f"Subscription request sent for Nifty50 (ID: {CONFIG.NIFTY_SECURITY_ID})")
            
            # Start heartbeat
            self.start_heartbeat()
            
            # Send connection notification
            if self.telegram_bot:
                self.telegram_bot.send_message(
                    "✅ <b>Trading System Connected</b>\n"
                    f"📊 Monitoring: NIFTY 50\n"
                    f"📈 Indicators: RSI, MACD, VWAP, BB, OBV\n"
                    f"⏰ Time: {datetime.now().strftime('%H:%M:%S')}"
                )
                
        except Exception as e:
            logger.error(f"Error in on_open: {e}", exc_info=True)

    def on_message(self, ws, message):
        """Process messages with improved parsing."""
        try:
            # FIX 3: Enhanced packet parsing [[2]]
            data = None
            
            # Try JSON first
            if isinstance(message, str):
                try:
                    json_data = json.loads(message)
                    data = self._parse_json_response(json_data)
                except json.JSONDecodeError:
                    pass
            
            # Try binary packet
            elif isinstance(message, bytes):
                data = self._parse_binary_packet(message)
            
            # Process if we got valid data
            if data and 'ltp' in data:
                self._process_tick(data)
                self.tick_count += 1
                
                # FIX 5: Enhanced logging every 10 ticks
                if self.tick_count % 10 == 0:
                    logger.info(f"Processed {self.tick_count} ticks, Last price: {data.get('ltp', 'N/A'):.2f}")
                    
        except Exception as e:
            logger.error(f"Message processing error: {e}", exc_info=True)

    def _parse_json_response(self, json_data: Dict) -> Optional[Dict]:
        """Parse JSON response from Dhan WebSocket."""
        try:
            # Handle subscription confirmation
            if json_data.get("type") == "subscription_status":
                logger.info(f"Subscription status: {json_data.get('status')}")
                return None
            
            # Handle market data
            if "data" in json_data:
                market_data = json_data["data"]
                return {
                    "timestamp": datetime.now(),
                    "ltp": float(market_data.get("LTP", market_data.get("ltp", 0))),
                    "open": float(market_data.get("open", 0)),
                    "high": float(market_data.get("high", 0)),
                    "low": float(market_data.get("low", 0)),
                    "close": float(market_data.get("close", market_data.get("LTP", 0))),
                    "volume": int(market_data.get("volume", 0))
                }
            
            # Handle tick data
            if "LTP" in json_data:
                return {
                    "timestamp": datetime.now(),
                    "ltp": float(json_data["LTP"]),
                    "volume": int(json_data.get("volume", 0))
                }
                
            return None
            
        except Exception as e:
            logger.debug(f"JSON parse error: {e}")
            return None

    def _parse_binary_packet(self, data: bytes) -> Optional[Dict]:
        """Parse binary packet from Dhan WebSocket."""
        try:
            packet_length = len(data)
            
            # Basic tick packet (16 bytes)
            if packet_length >= 16:
                # Try to unpack basic tick structure
                try:
                    values = struct.unpack("<BHBIf", data[:12])
                    packet_code, msg_len, exch_seg, security_id, ltp = values
                    
                    if security_id == CONFIG.NIFTY_SECURITY_ID:
                        # Validate price
                        if CONFIG.PRICE_SANITY_MIN <= ltp <= CONFIG.PRICE_SANITY_MAX:
                            return {
                                "timestamp": datetime.now(),
                                "ltp": ltp,
                                "packet_type": "tick"
                            }
                except:
                    pass
            
            # Try other packet formats if basic fails
            if packet_length >= 32:
                # Detailed packet with volume
                try:
                    values = struct.unpack("<BHBIffI", data[:24])
                    packet_code, msg_len, exch_seg, security_id, ltp, atp, volume = values
                    
                    if security_id == CONFIG.NIFTY_SECURITY_ID:
                        if CONFIG.PRICE_SANITY_MIN <= ltp <= CONFIG.PRICE_SANITY_MAX:
                            return {
                                "timestamp": datetime.now(),
                                "ltp": ltp,
                                "volume": volume,
                                "packet_type": "detailed"
                            }
                except:
                    pass
                    
            return None
            
        except Exception as e:
            logger.debug(f"Binary packet parse error: {e}")
            return None

    def _process_tick(self, tick_data: Dict):
        """FIX 2: Properly accumulate ticks for each minute."""
        try:
            if not tick_data or "timestamp" not in tick_data:
                return
            
            # Must have valid price
            ltp = tick_data.get("ltp", 0)
            if ltp <= 0:
                return
            
            # Add to buffer
            self.tick_buffer.append(tick_data)
            self.last_tick_time = tick_data["timestamp"]
            
            # Get minute timestamp
            current_minute = tick_data["timestamp"].replace(second=0, microsecond=0)
            
            # Initialize or check for new minute
            if self.last_minute_timestamp is None:
                self.last_minute_timestamp = current_minute
                self.current_minute_data = [tick_data]
                logger.debug(f"Starting new minute: {current_minute.strftime('%H:%M')}")
                
            elif self.last_minute_timestamp != current_minute:
                # New minute started - create candle for previous minute
                if self.current_minute_data:
                    self._create_minute_candle(self.last_minute_timestamp, self.current_minute_data)
                
                # Start collecting for new minute
                self.last_minute_timestamp = current_minute
                self.current_minute_data = [tick_data]
                logger.debug(f"New minute: {current_minute.strftime('%H:%M')}")
                
            else:
                # Same minute - add to current collection
                self.current_minute_data.append(tick_data)
                
        except Exception as e:
            logger.error(f"Tick processing error: {e}", exc_info=True)

    def _create_minute_candle(self, minute_timestamp: datetime, minute_ticks: List[Dict]):
        """Create OHLCV candle from collected minute ticks."""
        try:
            if not minute_ticks:
                logger.debug(f"No ticks for minute {minute_timestamp.strftime('%H:%M')}")
                return
            
            # Extract prices and volumes
            prices = [tick.get("ltp", 0) for tick in minute_ticks if tick.get("ltp", 0) > 0]
            volumes = [tick.get("volume", 0) for tick in minute_ticks]
            
            if not prices:
                logger.debug("No valid prices in minute ticks")
                return
            
            # Create candle
            candle = {
                "timestamp": minute_timestamp,
                "open": prices[0],
                "high": max(prices),
                "low": min(prices),
                "close": prices[-1],
                "volume": max(volumes) if volumes else 0  # Use max volume seen
            }
            
            # Validate candle
            if candle["high"] < candle["low"]:
                logger.error("Invalid candle: high < low")
                return
            
            # Add to dataframe
            new_candle_df = pd.DataFrame([candle]).set_index("timestamp")
            if self.minute_ohlcv.empty:
                self.minute_ohlcv = new_candle_df
            else:
                self.minute_ohlcv = pd.concat([self.minute_ohlcv, new_candle_df])
            
            # Limit size
            if len(self.minute_ohlcv) > 500:
                self.minute_ohlcv = self.minute_ohlcv.tail(500)
            
            logger.info(f"Candle created for {minute_timestamp.strftime('%H:%M')}: "
                       f"O:{candle['open']:.2f} H:{candle['high']:.2f} "
                       f"L:{candle['low']:.2f} C:{candle['close']:.2f} V:{candle['volume']}")
            
            # FIX 1: Call correct method with dataframe parameter
            if len(self.minute_ohlcv) >= CONFIG.MIN_DATA_POINTS:
                self.analyze_and_alert(self.minute_ohlcv)  # FIXED: proper method call
            else:
                remaining = CONFIG.MIN_DATA_POINTS - len(self.minute_ohlcv)
                logger.info(f"Building data: {len(self.minute_ohlcv)}/{CONFIG.MIN_DATA_POINTS} "
                           f"({remaining} more candles needed)")
                
        except Exception as e:
            logger.error(f"Candle creation error: {e}", exc_info=True)

    def analyze_and_alert(self, df: pd.DataFrame):
        """Analyze data and send alerts - FIXED VERSION."""
        try:
            if df.empty or len(df) < CONFIG.MIN_DATA_POINTS:
                logger.debug("Insufficient data for analysis")
                return
            
            # Import here to avoid circular imports
            from technical_indicators import TechnicalIndicators
            
            # Calculate all indicators
            indicators = {
                "rsi": TechnicalIndicators.calculate_rsi(df, CONFIG.RSI_PARAMS),
                "macd": TechnicalIndicators.calculate_macd(df, CONFIG.MACD_PARAMS),
                "vwap": TechnicalIndicators.calculate_vwap(df, CONFIG.VWAP_PARAMS),
                "bollinger": TechnicalIndicators.calculate_bollinger_bands(df, CONFIG.BOLLINGER_PARAMS),
                "obv": TechnicalIndicators.calculate_obv(df, CONFIG.OBV_PARAMS),
                "price": df['close'].iloc[-1]
            }
            
            # Generate signal
            signal_result = self.signal_generator.calculate_weighted_signal(indicators)
            
            # FIX 5: Enhanced logging for debugging
            logger.debug(f"Analysis at {datetime.now().strftime('%H:%M:%S')}: "
                        f"Signal: {signal_result['composite_signal']}, "
                        f"Score: {signal_result['weighted_score']:.3f}, "
                        f"Confidence: {signal_result['confidence']:.1f}%, "
                        f"Active: {signal_result['active_indicators']}")
            
            # Validate signal
            is_valid, validation_details = self.signal_validator.validate_signal(
                signal_result, df, indicators
            )
            
            if not is_valid:
                logger.info(f"Signal rejected: {validation_details.get('rejection_reasons')}")
                return
            
            # Check if we should alert
            if self._should_alert(signal_result):
                # Predict duration
                persistence = self.duration_predictor.predict_signal_duration(
                    df, indicators, signal_result,
                    timeframe_minutes=5
                )
                
                # Send alert
                self._send_enhanced_duration_alert(
                    df['close'].iloc[-1],
                    indicators,
                    signal_result,
                    persistence
                )
                
                # Generate chart if enabled
                if self.chart_generator and CONFIG.alert_with_charts:
                    try:
                        timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
                        chart_path = f"images/signal_{timestamp_str}.png"
                        
                        generated_path = self.chart_generator.generate_signal_chart(
                            df, indicators, signal_result,
                            output_path=chart_path
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
                        
        except Exception as e:
            logger.error(f"Analysis error: {e}", exc_info=True)

    def _should_alert(self, signal_result: Dict) -> bool:
        """Check if alert should be sent."""
        try:
            # Check signal strength
            if abs(signal_result.get('weighted_score', 0)) < CONFIG.MIN_SIGNAL_STRENGTH:
                return False
            
            # Check confidence
            if signal_result.get('confidence', 0) < CONFIG.MIN_CONFIDENCE:
                return False
            
            # Check active indicators
            if signal_result.get('active_indicators', 0) < CONFIG.MIN_ACTIVE_INDICATORS:
                return False
            
            # Check cooldown
            if self.last_alert_time:
                elapsed = (datetime.now() - self.last_alert_time).total_seconds()
                if elapsed < CONFIG.COOLDOWN_SECONDS:
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Alert check error: {e}")
            return False

    def _send_enhanced_duration_alert(self, current_price: float, indicators: Dict,
                                     signal_result: Dict, duration_prediction):
        """Send formatted alert to Telegram."""
        try:
            from signal_prediction import EnhancedSignalFormatter
            
            # Format message
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
                    
                    # Track signal
                    signal_data = {
                        "timestamp": datetime.now(),
                        "signal_type": signal_result['composite_signal'],
                        "score": signal_result['weighted_score'],
                        "price": current_price,
                        "expected_duration": duration_prediction.expected_candles,
                        "confidence": duration_prediction.confidence_level
                    }
                    
                    self.alert_history.append(signal_data)
                    
                    # Add to monitor if available
                    if self.signal_monitor:
                        self.signal_monitor.add_signal(signal_data)
                    
                    logger.info(f"Alert sent: {signal_result['composite_signal']} at {current_price:.2f}")
                    
        except Exception as e:
            logger.error(f"Alert sending error: {e}")

    def start_heartbeat(self):
        """Maintain connection with periodic heartbeat."""
        def heartbeat():
            while self.connected and self.running:
                try:
                    if self.ws:
                        # Send Dhan heartbeat packet
                        heartbeat_packet = {"RequestCode": 50}
                        self.ws.send(json.dumps(heartbeat_packet))
                        logger.debug("Heartbeat sent")
                    time.sleep(30)
                except Exception as e:
                    logger.debug(f"Heartbeat error: {e}")
                    if not self.running:
                        break
        
        self.heartbeat_thread = threading.Thread(target=heartbeat, daemon=True)
        self.heartbeat_thread.start()

    def connect(self):
        """Connect to Dhan WebSocket with proper URL."""
        try:
            # Build connection URL for Dhan API v2
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
            
            # Run in thread
            wst = threading.Thread(
                target=self.ws.run_forever,
                kwargs={"sslopt": {"cert_reqs": ssl.CERT_NONE}}
            )
            wst.daemon = True
            wst.start()
            
            # Wait for connection
            timeout = 15
            while not self.connected and timeout > 0 and self.running:
                time.sleep(1)
                timeout -= 1
            
            if not self.connected:
                raise ConnectionError("WebSocket connection timeout")
            
            logger.info("WebSocket connected successfully")
            
        except Exception as e:
            logger.error(f"Connection error: {e}")
            raise

    def on_error(self, ws, error):
        """Handle errors."""
        logger.error(f"WebSocket error: {error}")

    def on_close(self, ws, close_status_code, close_msg):
        """Handle connection closure."""
        logger.warning(f"WebSocket closed - Code: {close_status_code}, Message: {close_msg}")
        self.connected = False

    def disconnect(self):
        """Disconnect WebSocket."""
        try:
            self.running = False
            self.connected = False
            
            if self.ws:
                self.ws.close()
                logger.info("WebSocket disconnected")
                
        except Exception as e:
            logger.error(f"Disconnect error: {e}")

    def set_signal_monitor(self, monitor):
        """Set signal monitor."""
        self.signal_monitor = monitor
        logger.info("Signal monitor attached")
