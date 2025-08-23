"""
Enhanced Dhan WebSocket client with improved data handling and analysis.
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

from technical_indicators import TechnicalIndicators, SignalGenerator
from chart_utils import format_enhanced_alert, plot_enhanced_chart
import config

logger = logging.getLogger(__name__)

class DhanWebSocketClient:
    def __init__(self, access_token_b64: str, client_id_b64: str, telegram_bot):
        """Initialize enhanced WebSocket client."""
        try:
            self.access_token = base64.b64decode(access_token_b64).decode("utf-8")
            self.client_id = base64.b64decode(client_id_b64).decode("utf-8")
        except Exception as e:
            logger.error(f"Failed to decode credentials: {e}")
            raise ValueError("Invalid base64 encoded credentials")
        
        self.telegram_bot = telegram_bot
        self.ws = None
        self.connected = False
        
        # Data storage
        self.tick_buffer = collections.deque(maxlen=config.MAX_BUFFER_SIZE)
        self.minute_ohlcv = pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])
        self.last_minute_processed = None
        
        # Alert management
        self.last_alert_time = None
        self.alert_history = collections.deque(maxlen=100)
        self.alert_lock = threading.Lock()
        
        # Connection management
        self.connection_retry_count = 0
        self.max_retries = 5
        self.heartbeat_thread = None
        self.data_thread = None
        
        # Initialize components
        self.signal_generator = SignalGenerator()
        
        logger.info("Enhanced DhanWebSocketClient initialized")
    
    def on_open(self, ws):
        """Handle WebSocket connection opened."""
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
            self.telegram_bot.send_message(
                "✅ <b>WebSocket Connected</b>\n"
                "📊 Monitoring Nifty50\n"
                f"⚙️ Min Data Points: {config.MIN_DATA_POINTS}\n"
                f"⏱️ Cooldown: {config.COOLDOWN_SECONDS}s"
            )
    
    def on_message(self, ws, message):
        """Process incoming WebSocket messages."""
        try:
            data = self._parse_packet(message)
            if data:
                self._process_tick(data)
        except Exception as e:
            logger.error(f"Message processing error: {e}", exc_info=True)
    
    def _parse_packet(self, message) -> Optional[Dict]:
        """Enhanced packet parsing with better error handling."""
        try:
            if isinstance(message, bytes):
                # Handle different packet structures based on length
                packet_parsers = {
                    16: self._parse_16_byte_packet,
                    32: self._parse_32_byte_packet,
                    44: self._parse_44_byte_packet,  # Add more packet types
                    108: self._parse_108_byte_packet
                }
                
                parser = packet_parsers.get(len(message))
                if parser:
                    return parser(message)
                else:
                    logger.debug(f"Unknown packet length: {len(message)} bytes")
                    return None
            else:
                # JSON message
                return self._parse_json_message(message)
                
        except Exception as e:
            logger.error(f"Packet parsing error: {e}")
            return None
    
    def _parse_16_byte_packet(self, data: bytes) -> Optional[Dict]:
        """Parse 16-byte tick packet."""
        try:
            code, msg_len, exch_seg, sec_id, ltp, ltt = struct.unpack("<B H B I f I", data)
            
            if sec_id != config.NIFTY_SECURITY_ID:
                return None
                
            if not (config.PRICE_SANITY_MIN <= ltp <= config.PRICE_SANITY_MAX):
                logger.warning(f"Price sanity check failed: {ltp}")
                return None
            
            return {
                "timestamp": datetime.now(),
                "ltp": ltp,
                "ltt": ltt,
                "packet_type": "tick_16"
            }
        except Exception as e:
            logger.error(f"16-byte packet parse error: {e}")
            return None
    
    def _parse_32_byte_packet(self, data: bytes) -> Optional[Dict]:
        """Parse 32-byte detailed tick packet."""
        try:
            unpacked = struct.unpack("<B H B I f h I f I I I", data)
            code, msg_len, exch_seg, sec_id, ltp, ltq, ltt, atp, volume, total_sell, total_buy = unpacked
            
            if sec_id != config.NIFTY_SECURITY_ID:
                return None
                
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
                "packet_type": "tick_32"
            }
        except Exception as e:
            logger.error(f"32-byte packet parse error: {e}")
            return None
    
    def _parse_44_byte_packet(self, data: bytes) -> Optional[Dict]:
        """Parse 44-byte market depth packet."""
        # Implementation for 44-byte packets if needed
        return None
    
    def _parse_108_byte_packet(self, data: bytes) -> Optional[Dict]:
        """Parse 108-byte full market data packet."""
        # Implementation for 108-byte packets if needed
        return None
    
    def _parse_json_message(self, message: str) -> Optional[Dict]:
        """Parse JSON WebSocket message."""
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
                    "packet_type": "json_touchline"
                }
            
            return None
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}")
            return None
    
    def _process_tick(self, tick_data: Dict):
        """Process tick data and update minute candles."""
        try:
            # Add to buffer
            self.tick_buffer.append(tick_data)
            
            # Get current minute
            current_minute = tick_data["timestamp"].replace(second=0, microsecond=0)
            
            # Check if new minute
            if self.last_minute_processed != current_minute:
                if self.last_minute_processed is not None:
                    self._create_minute_candle(self.last_minute_processed)
                self.last_minute_processed = current_minute
                
        except Exception as e:
            logger.error(f"Tick processing error: {e}")
    
    def _create_minute_candle(self, minute_timestamp: datetime):
        """Create minute OHLCV candle from tick data."""
        try:
            # Filter ticks for this minute
            minute_ticks = [
                t for t in self.tick_buffer 
                if t["timestamp"].replace(second=0, microsecond=0) == minute_timestamp
                and "ltp" in t
            ]
            
            if not minute_ticks:
                return
            
            # Calculate OHLCV
            prices = [t["ltp"] for t in minute_ticks]
            volumes = [t.get("volume", 0) for t in minute_ticks if "volume" in t]
            
            candle = {
                "timestamp": minute_timestamp,
                "open": prices[0],
                "high": max(prices),
                "low": min(prices),
                "close": prices[-1],
                "volume": volumes[-1] if volumes else 0
            }
            
            # Add to dataframe
            self.minute_ohlcv = pd.concat([
                self.minute_ohlcv,
                pd.DataFrame([candle]).set_index("timestamp")
            ]).tail(500)  # Keep last 500 candles
            
            logger.debug(f"Created minute candle: O={candle['open']:.2f}, H={candle['high']:.2f}, "
                        f"L={candle['low']:.2f}, C={candle['close']:.2f}, V={candle['volume']}")
            
            # Check for signals
            if len(self.minute_ohlcv) >= config.MIN_DATA_POINTS:
                self._analyze_and_alert()
                
        except Exception as e:
            logger.error(f"Candle creation error: {e}")


    def _analyze_and_alert(self):
        """Perform technical analysis and send alerts if conditions are met."""
        with self.alert_lock:
            try:
                # Prepare data
                df = self.minute_ohlcv.tail(config.MIN_DATA_POINTS)
                prices = pd.Series(df['close'].values, index=df.index)
                volumes = pd.Series(df['volume'].values, index=df.index)
                highs = pd.Series(df['high'].values, index=df.index)
                lows = pd.Series(df['low'].values, index=df.index)
                
                # Calculate all indicators
                indicators = {
                    "macd": TechnicalIndicators.calculate_macd(prices),
                    "rsi": TechnicalIndicators.calculate_rsi(prices),
                    "vwap": TechnicalIndicators.calculate_vwap(prices, volumes),
                    "keltner": TechnicalIndicators.calculate_keltner_channels(prices, highs, lows),
                    "supertrend": TechnicalIndicators.calculate_supertrend(prices, highs, lows),
                    "impulse": TechnicalIndicators.calculate_impulse_macd(prices)
                }
                
                # Generate weighted signal with duration prediction
                signal_result = self.signal_generator.calculate_weighted_signal_with_duration(
                    indicators, 
                    config.INDICATOR_WEIGHTS,
                    self.minute_ohlcv.tail(100)  # Use last 100 candles for duration prediction
                )
                
                # Log analysis
                logger.info(f"Signal Analysis: {signal_result['composite_signal']} "
                        f"(Score: {signal_result['weighted_score']}, "
                        f"Confidence: {signal_result['confidence']}%)")
                
                # Log duration prediction
                if 'duration_prediction' in signal_result:
                    duration = signal_result['duration_prediction']
                    logger.info(f"Duration Prediction: {duration['estimated_minutes']} mins, "
                            f"Confidence: {duration['confidence']}")
                
                # Check alert conditions
                if self._should_alert(signal_result):
                    self._send_alert(prices.iloc[-1], indicators, signal_result)
                    
            except Exception as e:
                logger.error(f"Analysis error: {e}", exc_info=True)




    def _should_alert(self, signal_result: Dict) -> bool:
        """Determine if alert should be sent based on signal and cooldown."""
        try:
            # Check signal strength
            if abs(signal_result['weighted_score']) < 0.5:
                return False
            
            # Check confidence
            if signal_result['confidence'] < 60:
                return False
            
            # Check cooldown
            if self.last_alert_time:
                elapsed = (datetime.now() - self.last_alert_time).total_seconds()
                if elapsed < config.COOLDOWN_SECONDS:
                    logger.info(f"Alert suppressed - cooldown active ({config.COOLDOWN_SECONDS - elapsed:.0f}s remaining)")
                    return False
            
            # Check for false positives
            if signal_result['active_indicators'] < 4:
                logger.info("Alert suppressed - insufficient active indicators")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Alert condition check error: {e}")
            return False
    
    def _send_alert(self, current_price: float, indicators: Dict, signal_result: Dict):
        """Send enhanced alert with detailed analysis."""
        try:
            # Import the new formatting function
            from chart_utils import format_enhanced_alert_with_duration
            
            # Format message with duration
            message = format_enhanced_alert_with_duration(
                current_price,
                indicators,
                signal_result,
                datetime.now()
            )
            
            # Generate chart
            chart_path = "images/analysis_chart.png"
            plot_enhanced_chart(
                self.minute_ohlcv.tail(100),
                indicators,
                signal_result,
                chart_path
            )
            
            # Send via Telegram
            if self.telegram_bot:
                success = self.telegram_bot.send_chart(message, chart_path)
                if success:
                    self.last_alert_time = datetime.now()
                    self.alert_history.append({
                        "timestamp": datetime.now(),
                        "signal": signal_result['composite_signal'],
                        "price": current_price,
                        "duration_prediction": signal_result.get("duration_prediction", {})
                    })
                    logger.info("Alert sent successfully")
                else:
                    logger.error("Failed to send alert")
                    
        except Exception as e:
            logger.error(f"Alert sending error: {e}")



    def start_heartbeat(self):
        """Start heartbeat thread to keep connection alive."""
        def heartbeat():
            while self.connected:
                try:
                    if self.ws:
                        self.ws.send(json.dumps({"ping": 1}))
                    time.sleep(30)
                except Exception as e:
                    logger.error(f"Heartbeat error: {e}")
                    
        self.heartbeat_thread = threading.Thread(target=heartbeat, daemon=True)
        self.heartbeat_thread.start()
    
    def on_error(self, ws, error):
        """Handle WebSocket errors."""
        logger.error(f"WebSocket error: {error}")
        
    def on_close(self, ws, close_status_code, close_msg):
        """Handle WebSocket closure with auto-reconnect."""
        logger.warning(f"WebSocket closed - Code: {close_status_code}, Message: {close_msg}")
        self.connected = False
        
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
            self.connect()
    
    def connect(self):
        """Establish WebSocket connection with proper error handling."""
        try:
            # Validate credentials first
            if not self.access_token or not self.client_id:
                raise ValueError("Missing credentials")
            
            ws_url = (f"wss://api-feed.dhan.co?version=2&token={self.access_token}"
                     f"&clientId={self.client_id}&authType=2")
            
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
            
            # Wait for connection
            timeout = 15
            while not self.connected and timeout > 0:
                time.sleep(1)
                timeout -= 1
            
            if not self.connected:
                raise ConnectionError("WebSocket connection timeout")
                
            logger.info("WebSocket connected successfully")
            
        except Exception as e:
            logger.error(f"Connection error: {e}")
            raise
    
    def disconnect(self):
        """Gracefully disconnect WebSocket."""
        if self.ws:
            self.connected = False
            self.ws.close()
            logger.info("WebSocket disconnected by user")
