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
import os
from technical_indicators import TechnicalIndicators, SignalGenerator, EnhancedSignalGenerator
from chart_utils import format_enhanced_alert_with_duration, plot_enhanced_chart
import config

logger = logging.getLogger(__name__)

class DhanWebSocketClient:
    def __init__(self, access_token_b64: str, client_id_b64: str, telegram_bot):
        """Initialize enhanced WebSocket client with validation."""
        # Validate config first
        self._validate_config()
        
        try:
            self.access_token = base64.b64decode(access_token_b64.strip()).decode("utf-8").strip()
            self.client_id = base64.b64decode(client_id_b64.strip()).decode("utf-8").strip()
        except Exception as e:
            logger.error(f"Failed to decode credentials: {e}")
            raise ValueError("Invalid base64 encoded credentials")
        
        
        # Initialize minute_ohlcv with proper dtypes
        self.minute_ohlcv = pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])
        self.minute_ohlcv = self.minute_ohlcv.astype({
            'open': 'float64',
            'high': 'float64',
            'low': 'float64', 
            'close': 'float64',
            'volume': 'int64'
        })
            
        self.telegram_bot = telegram_bot
        self.ws = None
        self.connected = False
        
        # Data storage
        self.tick_buffer = collections.deque(maxlen=config.MAX_BUFFER_SIZE)
        self.minute_ohlcv = pd.DataFrame()
        self.last_minute_processed = None
        self.current_minute_data = []
        
        # Alert management
        self.last_alert_time = None
        self.alert_history = collections.deque(maxlen=100)
        self.alert_lock = threading.Lock()
        
        # Connection management
        self.connection_retry_count = 0
        self.max_retries = 5
        self.heartbeat_thread = None
        
        # Initialize components
        self.signal_generator = SignalGenerator()
        self.enhanced_generator = EnhancedSignalGenerator()
        
        # Track last known volume for estimation
        self.last_known_volume = 0
        self.volume_estimator = VolumeEstimator()
        
        logger.info("Enhanced DhanWebSocketClient initialized")

    def _validate_config(self):
        """Validate required configuration fields."""
        required_fields = [
            'NIFTY_SECURITY_ID', 'NIFTY_EXCHANGE_SEGMENT',
            'MIN_DATA_POINTS', 'MAX_BUFFER_SIZE',
            'PRICE_SANITY_MIN', 'PRICE_SANITY_MAX'
        ]
        
        for field in required_fields:
            if not hasattr(config, field):
                logger.error(f"Missing required config field: {field}")
                raise ValueError(f"Configuration missing: {field}")
            
        logger.info("Configuration validation successful")


    def on_open(self, ws):
        """Handle WebSocket connection with correct Dhan subscription."""
        logger.info("WebSocket connection established")
        self.connected = True

        # # Fetch historical volume data on connection
        # self.fetch_historical_volume()
        
        # # Schedule periodic volume updates
        # self._schedule_volume_fetch()
                
        # According to Dhan docs, subscription format for indices
        subscription = {
            "RequestCode": 15,  # Subscribe request
            "InstrumentCount": 1,
            "InstrumentList": [{
                "ExchangeSegment": "IDX_I",  # Index segment
                "SecurityId": "13"  # NIFTY 50
            }]
        }
        
        ws.send(json.dumps(subscription))
        logger.info(f"Subscription sent: {json.dumps(subscription)}")
        logger.debug(f"Subscription sent: {json.dumps(subscription)}")
        
        # After subscription, request full market data
        time.sleep(0.5)
        
        mode_request = {
            "RequestCode": 17,  # Mode change request
            "InstrumentCount": 1,
            "InstrumentList": [{
                "ExchangeSegment": "IDX_I",
                "SecurityId": "13",
                "QuoteMode": 4  # Full quote mode
            }]
        }
        
        ws.send(json.dumps(mode_request))
        logger.info(f"Requested full quote mode {json.dumps(mode_request)}")

        # Send connection notification
        if self.telegram_bot:
            self.telegram_bot.send_message(
                "✅ <b>WebSocket Connected</b>\n"
                "📊 Monitoring Nifty50\n"
                f"⚙️ Min Data Points: {config.MIN_DATA_POINTS}\n"
                f"⏱️ Cooldown: {config.COOLDOWN_SECONDS}s\n"
                f"📈 Signal Thresholds: Buy={config.BUY_THRESHOLD}, Sell={config.SELL_THRESHOLD}"
            )
    
    def on_message(self, ws, message):
        """Process incoming WebSocket messages."""
        try:
            data = self._parse_message(message)
            if data:
                self._process_tick(data)
        except Exception as e:
            logger.error(f"Message processing error: {e}", exc_info=True)
    
    def _parse_message(self, message) -> Optional[Dict]:
        """Parse WebSocket message based on Dhan format."""
        try:
            if isinstance(message, bytes):
                # Log packet size for debugging
                # logger.debug(f"Received binary packet of size: {len(message)} bytes")
                return self._parse_binary_packet(message)
            else:
                # Text/JSON message
                return self._parse_json_message(message)
                
        except Exception as e:
            logger.error(f"Message parsing error: {e}")
            return None

    def _parse_binary_packet(self, data: bytes) -> Optional[Dict]:
        """Parse binary packet based on size, not packet code."""
        try:
            packet_size = len(data)
            # Add counter for debugging
            if not hasattr(self, 'packet_count'):
                self.packet_count = 0
            self.packet_count += 1
            
            if self.packet_count % 100 == 0:  # Log every 100 packets
                logger.info(f"Received {self.packet_count} packets, last size: {packet_size}")
                
            # Parse based on packet size patterns
            if packet_size == 16:
                # Could be ticker or minimal quote
                return self._parse_16_byte_packet(data)
            elif packet_size == 32:
                return self._parse_32_byte_quote(data)
            elif packet_size == 44:
                return self._parse_quote_packet(data)
            elif packet_size == 50:
                return self._parse_50_byte_packet(data)
            elif packet_size == 66:
                return self._parse_66_byte_packet(data)
            elif packet_size == 184:
                return self._parse_full_quote(data)
            elif packet_size == 492:
                return self._parse_market_depth_full(data)
            else:
                # Try to extract at least LTP if possible
                if packet_size >= 8:
                    return self._parse_minimal_packet(data)
                else:
                    logger.debug(f"Unknown packet size: {packet_size}")
                    return None
                    
        except Exception as e:
            logger.error(f"Binary packet parsing error: {e}")
            return None
           

    def _parse_minimal_packet(self, data: bytes) -> Optional[Dict]:
        """Parse minimal packet to extract LTP."""
        try:
            # Try to extract LTP from position 4-8 (common location)
            if len(data) >= 8:
                ltp = struct.unpack('<f', data[4:8])[0]
                
                # Sanity check the price
                if config.PRICE_SANITY_MIN <= ltp <= config.PRICE_SANITY_MAX:
                    return {
                        "timestamp": datetime.now(),
                        "ltp": ltp,
                        "volume": self.volume_estimator.get_current_estimate(),
                        "packet_type": "minimal"
                    }
            return None
        except:
            return None

    def _parse_16_byte_packet(self, data: bytes) -> Optional[Dict]:
        """Parse 16-byte packet from Dhan WebSocket."""
        try:
            logger.debug(f"16-byte packet hex: {data.hex()}")
            
            # Dhan's 16-byte packet structure for indices
            # Byte 0: Packet type (02, 06, etc.)
            # Byte 1: Exchange segment code
            # Bytes 2-3: Padding or flags
            # Bytes 4-7: LTP (float, little-endian)
            # Bytes 8-15: Additional data (timestamp, etc.)
            
            packet_type = data[0]
            exchange_code = data[1]
            
            # For IDX_I (indices), exchange code should be 16 (0x10)
            if exchange_code == 0x10:  # 16 in decimal for indices
                # Extract LTP from bytes 4-7
                try:
                    ltp = struct.unpack('<f', data[4:8])[0]
                    
                    # Validate price
                    if config.PRICE_SANITY_MIN <= ltp <= config.PRICE_SANITY_MAX:
                        logger.info(f"Parsed packet type {packet_type}: LTP={ltp:.2f}")
                        
                        # Try to extract timestamp if available
                        timestamp_val = struct.unpack('<I', data[8:12])[0]
                        
                        return {
                            "timestamp": datetime.now(),
                            "ltp": ltp,
                            "ltt": timestamp_val,
                            "volume": self.volume_estimator.estimate_volume(ltp, 0),
                            "packet_type": f"idx_packet_{packet_type}"
                        }
                except:
                    pass
            
            # Alternative parsing for different packet structures
            # Try extracting float values from different positions
            for offset in [4, 8]:
                try:
                    value = struct.unpack('<f', data[offset:offset+4])[0]
                    if config.PRICE_SANITY_MIN <= value <= config.PRICE_SANITY_MAX:
                        # logger.info(f"Found valid price at offset {offset}: {value:.2f}")
                        return {
                            "timestamp": datetime.now(),
                            "ltp": value,
                            "volume": self.volume_estimator.estimate_volume(value, 0),
                            "packet_type": f"ticker_offset{offset}"
                        }
                except:
                    continue
            
            logger.debug("Could not parse 16-byte packet")
            return None
            
        except Exception as e:
            logger.error(f"16-byte packet parse error: {e}")
            return None




    # def _parse_16_byte_packet(self, data: bytes) -> Optional[Dict]:
    #     """Parse 16-byte ticker packet."""
    #     try:
    #         # Basic ticker format: exchange(1) + segment(1) + security_id(2) + ltp(4) + ltt(4) + padding(4)
    #         packet_type, exchange, security_id_high, security_id_low = struct.unpack("<BBBB", data[:4])
    #         security_id = (security_id_high << 8) | security_id_low
            
    #         if security_id != config.NIFTY_SECURITY_ID:
    #             return None
            
    #         ltp = struct.unpack("<f", data[4:8])[0]
    #         ltt = struct.unpack("<I", data[8:12])[0]
            
    #         return {
    #             "timestamp": datetime.now(),
    #             "ltp": ltp,
    #             "ltt": ltt,
    #             "volume": self.volume_estimator.estimate_volume(ltp, 0),
    #             "packet_type": "ticker_16"
    #         }
    #     except Exception as e:
    #         logger.error(f"16-byte packet parse error: {e}")
    #         return None


    def _parse_50_byte_packet(self, data: bytes) -> Optional[Dict]:
        """Parse 50-byte packet."""
        try:
            logger.debug(f"50-byte packet first 16 bytes: {data[:16].hex()}")
            
            # Try to find price data in this packet
            for offset in range(0, min(47, len(data)-3), 4):
                try:
                    value = struct.unpack('<f', data[offset:offset+4])[0]
                    if config.PRICE_SANITY_MIN <= value <= config.PRICE_SANITY_MAX:
                        # logger.info(f"Found price in 50-byte packet at offset {offset}: {value:.2f}")
                        return {
                            "timestamp": datetime.now(),
                            "ltp": value,
                            "volume": self.volume_estimator.estimate_volume(value, 0),
                            "packet_type": "quote_50"
                        }
                except:
                    continue
            return None
        except Exception as e:
            logger.error(f"50-byte packet error: {e}")
            return None

    def _parse_66_byte_packet(self, data: bytes) -> Optional[Dict]:
        """Parse 66-byte packet."""
        try:
            # logger.debug(f"66-byte packet first 16 bytes: {data[:16].hex()}")
            
            # Try common positions for price data
            for offset in [4, 8, 12, 16, 20, 24]:
                try:
                    value = struct.unpack('<f', data[offset:offset+4])[0]
                    if config.PRICE_SANITY_MIN <= value <= config.PRICE_SANITY_MAX:
                        # logger.info(f"Found price in 66-byte packet at offset {offset}: {value:.2f}")
                        
                        # Try to get volume if available
                        volume = 0
                        try:
                            volume = struct.unpack('<I', data[offset+8:offset+12])[0]
                        except:
                            volume = self.volume_estimator.estimate_volume(value, 0)
                        
                        return {
                            "timestamp": datetime.now(),
                            "ltp": value,
                            "volume": volume,
                            "packet_type": "quote_66"
                        }
                except:
                    continue
            return None
        except Exception as e:
            logger.error(f"66-byte packet error: {e}")
            return None

    
    def _parse_32_byte_quote(self, data: bytes) -> Optional[Dict]:
        """Parse 32-byte quote packet with volume."""
        try:
            # Dhan 32-byte structure
            if len(data) < 32:
                return None
                
            # Parse based on expected structure
            packet_type = data[0]
            exchange = data[1]
            security_id = struct.unpack("<H", data[2:4])[0]
            
            if security_id != config.NIFTY_SECURITY_ID:
                return None
            
            ltp = struct.unpack("<f", data[4:8])[0]
            ltq = struct.unpack("<I", data[8:12])[0]
            ltt = struct.unpack("<I", data[12:16])[0]
            atp = struct.unpack("<f", data[16:20])[0]
            volume = struct.unpack("<I", data[20:24])[0]
            
            # Estimate volume if zero
            if volume == 0:
                volume = self.volume_estimator.estimate_volume(ltp, ltq)
            else:
                self.volume_estimator.update_last_volume(volume)
            
            return {
                "timestamp": datetime.now(),
                "ltp": ltp,
                "ltq": ltq,
                "ltt": ltt,
                "atp": atp,
                "volume": volume,
                "packet_type": "quote_32"
            }
        except Exception as e:
            logger.error(f"32-byte quote parse error: {e}")
            return None
    
    def _parse_ticker_packet(self, data: bytes) -> Optional[Dict]:
        """Parse ticker packet (minimal data)."""
        try:
            if len(data) < 16:
                return None
                
            # Ticker packet structure
            packet_type, exchange = struct.unpack("<BB", data[:2])
            security_id = struct.unpack("<H", data[2:4])[0]
            
            if security_id != config.NIFTY_SECURITY_ID:
                return None
            
            ltp = struct.unpack("<f", data[4:8])[0]
            ltt = struct.unpack("<I", data[8:12])[0]
            
            return {
                "timestamp": datetime.now(),
                "ltp": ltp,
                "ltt": ltt,
                "volume": self.volume_estimator.estimate_volume(ltp, 0),
                "packet_type": "ticker"
            }
        except Exception as e:
            logger.error(f"Ticker packet parse error: {e}")
            return None

    def _parse_quote_packet(self, data: bytes) -> Optional[Dict]:
        """Parse quote packet with OHLC."""
        try:
            if len(data) < 44:
                return None
                
            # Quote packet structure  
            packet_type = data[0]
            security_id = struct.unpack('<I', data[4:8])[0]
            
            if security_id != config.NIFTY_SECURITY_ID:
                return None
                
            ltp = struct.unpack('<f', data[8:12])[0]
            open_p = struct.unpack('<f', data[12:16])[0]
            high = struct.unpack('<f', data[16:20])[0]
            low = struct.unpack('<f', data[20:24])[0]
            close = struct.unpack('<f', data[24:28])[0]
            volume = struct.unpack('<I', data[28:32])[0]
            
            if volume == 0:
                volume = self.volume_estimator.estimate_volume(ltp, 0)
            else:
                self.volume_estimator.update_last_volume(volume)
            
            return {
                "timestamp": datetime.now(),
                "ltp": ltp,
                "open": open_p,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
                "packet_type": "quote"
            }
        except Exception as e:
            logger.error(f"Quote packet parse error: {e}")
            return None
    
    def _parse_full_quote(self, data: bytes) -> Optional[Dict]:
        """Parse 184-byte full quote packet."""
        try:
            # Parse key fields from full quote
            security_id = struct.unpack('<I', data[4:8])[0]
            
            if security_id != config.NIFTY_SECURITY_ID:
                return None
            
            ltp = struct.unpack('<f', data[8:12])[0]
            ltq = struct.unpack('<I', data[12:16])[0]
            ltt = struct.unpack('<I', data[16:20])[0]
            
            # OHLC data
            open_price = struct.unpack('<f', data[20:24])[0]
            high = struct.unpack('<f', data[24:28])[0]
            low = struct.unpack('<f', data[28:32])[0]
            close = struct.unpack('<f', data[32:36])[0]
            
            # Volume and value (64-bit)
            volume = struct.unpack('<Q', data[36:44])[0]
            value = struct.unpack('<d', data[44:52])[0]
            
            # Update volume estimator
            if volume > 0:
                self.volume_estimator.update_last_volume(volume)
            else:
                volume = self.volume_estimator.estimate_volume_with_value(ltp, value)
            
            return {
                "timestamp": datetime.now(),
                "ltp": ltp,
                "ltq": ltq,
                "ltt": ltt,
                "open": open_price,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
                "value": value,
                "packet_type": "full_quote"
            }
        except Exception as e:
            logger.error(f"Full quote parse error: {e}")
            return None
    
    def _parse_market_depth_packet(self, data: bytes) -> Optional[Dict]:
        """Parse market depth packet."""
        try:
            if len(data) < 20:
                return None
                
            security_id = struct.unpack('<I', data[4:8])[0]
            if security_id != config.NIFTY_SECURITY_ID:
                return None
                
            ltp = struct.unpack('<f', data[8:12])[0]
            
            return {
                "timestamp": datetime.now(),
                "ltp": ltp,
                "volume": self.volume_estimator.get_current_estimate(),
                "packet_type": "depth"
            }
        except Exception as e:
            logger.error(f"Market depth parse error: {e}")
            return None

    def _parse_market_depth_full(self, data: bytes) -> Optional[Dict]:
        """Parse full market depth packet (492 bytes)."""
        try:
            if len(data) < 492:
                return None
                
            # Extract key fields
            security_id = struct.unpack('<I', data[4:8])[0]
            if security_id != config.NIFTY_SECURITY_ID:
                return None
                
            ltp = struct.unpack('<f', data[8:12])[0]
            
            # Extract total buy/sell quantities for volume estimation
            total_buy_qty = struct.unpack('<I', data[100:104])[0]
            total_sell_qty = struct.unpack('<I', data[200:204])[0]
            
            estimated_volume = (total_buy_qty + total_sell_qty) // 2
            if estimated_volume > 0:
                self.volume_estimator.update_last_volume(estimated_volume)
            else:
                estimated_volume = self.volume_estimator.get_current_estimate()
            
            return {
                "timestamp": datetime.now(),
                "ltp": ltp,
                "volume": estimated_volume,
                "total_buy_qty": total_buy_qty,
                "total_sell_qty": total_sell_qty,
                "packet_type": "depth_full"
            }
        except Exception as e:
            logger.error(f"Full depth parse error: {e}")
            return None

    def _parse_oi_packet(self, data: bytes) -> Optional[Dict]:
        """Parse open interest packet."""
        # OI packets not relevant for Nifty50 index
        return None

    def _parse_prev_close_packet(self, data: bytes) -> Optional[Dict]:
        """Parse previous close packet."""
        try:
            if len(data) < 12:
                return None
                
            security_id = struct.unpack('<I', data[4:8])[0]
            if security_id != config.NIFTY_SECURITY_ID:
                return None
                
            prev_close = struct.unpack('<f', data[8:12])[0]
            
            # Store for reference
            logger.info(f"Previous close: {prev_close:.2f}")
            self.volume_estimator.set_previous_close(prev_close)
            return None
            
        except Exception as e:
            logger.error(f"Prev close parse error: {e}")
            return None

    def _parse_market_status_packet(self, data: bytes) -> Optional[Dict]:
        """Parse market status packet."""
        try:
            if len(data) >= 2:
                status = data[1]
                status_map = {
                    1: "Pre-Open",
                    2: "Open",
                    3: "Closed",
                    4: "Post-Close"
                }
                logger.info(f"Market status: {status_map.get(status, f'Unknown ({status})')}")
            return None
        except Exception as e:
            logger.error(f"Market status parse error: {e}")
            return None
    
    def _parse_json_message(self, message: str) -> Optional[Dict]:
        """Parse JSON WebSocket message."""
        try:
            data = json.loads(message)
            
            # Handle different JSON message types
            if "type" in data:
                if data["type"] == "quote":
                    return {
                        "timestamp": datetime.now(),
                        "ltp": data.get("ltp", 0),
                        "volume": data.get("volume", 0),
                        "packet_type": "json_quote"
                    }
                elif data["type"] == "error":
                    logger.error(f"Server error: {data.get('message', 'Unknown error')}")
                elif data["type"] == "subscription_status":
                    logger.info(f"Subscription status: {data.get('status', 'Unknown')}")
            
            return None
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}")
            return None
    
    def _process_tick(self, tick_data: Dict):
        """Process tick data and update minute candles."""
        try:
            ltp = tick_data.get("ltp", 0)
            packet_type = tick_data.get("packet_type", "unknown")
            
            # Validate tick data
            if ltp < config.PRICE_SANITY_MIN or ltp > config.PRICE_SANITY_MAX:
                logger.warning(f"Price sanity check failed for {packet_type}: LTP={ltp} "
                             f"(expected {config.PRICE_SANITY_MIN}-{config.PRICE_SANITY_MAX})")
                logger.debug(f"Full tick data: {tick_data}")
                return
            logger.debug(f"Processing {packet_type} tick: LTP={ltp}, Volume={tick_data.get('volume', 0)}")            
            
            
            
            # Add to current minute data
            self.current_minute_data.append(tick_data)
            
            # Get current minute
            current_minute = tick_data["timestamp"].replace(second=0, microsecond=0)
            
            # Check if new minute
            if self.last_minute_processed != current_minute:
                if self.last_minute_processed is not None:
                    self._create_minute_candle(self.last_minute_processed)
                
                self.last_minute_processed = current_minute
                self.current_minute_data = [tick_data]
                
        except Exception as e:
            logger.error(f"Tick processing error: {e}")
            
                
    def _create_minute_candle(self, minute_timestamp: datetime):
        """Create minute OHLCV candle from tick data."""
        try:
            if not self.current_minute_data:
                return
            
            # Extract prices and volumes - ENSURE FLOAT CONVERSION
            prices = []
            volumes = []
            
            for tick in self.current_minute_data:
                if "ltp" in tick and tick["ltp"] > 0:
                    # Convert to float explicitly
                    price = float(tick["ltp"])
                    prices.append(price)
                if "volume" in tick:
                    # Convert to int explicitly
                    vol = int(tick["volume"])
                    volumes.append(vol)
            
            if not prices:
                return
            
            # Get volume (use last non-zero or estimate)
            volume = 0
            if volumes:
                for v in reversed(volumes):
                    if v > 0:
                        volume = v
                        break
                
                if volume == 0:
                    volume = self.volume_estimator.estimate_from_price_action(prices)
            else:
                volume = self.volume_estimator.get_current_estimate()
            
            # Ensure all values are proper types
            candle = {
                "timestamp": minute_timestamp,
                "open": float(prices[0]),
                "high": float(max(prices)),
                "low": float(min(prices)),
                "close": float(prices[-1]),
                "volume": int(volume)
            }




            # Create new candle DataFrame with explicit dtypes
            new_candle_df = pd.DataFrame([candle]).set_index("timestamp")
            new_candle_df = new_candle_df.astype({
                'open': 'float64',
                'high': 'float64', 
                'low': 'float64',
                'close': 'float64',
                'volume': 'int64'
            })
            
            # If minute_ohlcv is empty, initialize with proper dtypes
            if len(self.minute_ohlcv) == 0:
                self.minute_ohlcv = new_candle_df
            else:
                # Ensure existing DataFrame has correct dtypes before concatenation
                self.minute_ohlcv = self.minute_ohlcv.astype({
                    'open': 'float64',
                    'high': 'float64',
                    'low': 'float64',
                    'close': 'float64',
                    'volume': 'int64'
                })
                self.minute_ohlcv = pd.concat([self.minute_ohlcv, new_candle_df], axis=0)
                
            
            # Keep only last 500 candles
            self.minute_ohlcv = self.minute_ohlcv.tail(500)
            
            logger.debug(f"Created minute candle: O={candle['open']:.2f}, H={candle['high']:.2f}, "
                        f"L={candle['low']:.2f}, C={candle['close']:.2f}, V={candle['volume']}")
            
            # Check for signals
            if len(self.minute_ohlcv) >= config.MIN_DATA_POINTS:
                self._analyze_and_alert()
                
        except Exception as e:
            logger.error(f"Candle creation error: {e}", exc_info=True)
         
    def _analyze_and_alert(self):
        """Enhanced analysis with predictive signal generation."""
        with self.alert_lock:
            try:
                # Prepare data
                df = self.minute_ohlcv.tail(config.MIN_DATA_POINTS)

                # Detect market regime
                market_regime = self.detect_market_regime(df)
                
                # Calculate all indicators
                indicators = self._calculate_all_indicators(df)
                
                # Add market regime to indicators for signal generation
                indicators['market_regime'] = {'regime': market_regime}
                                
                
                # Use enhanced signal generator for better predictions
                signal_result = self.enhanced_generator.generate_trading_signal(
                    indicators,
                    df,
                    config.INDICATOR_WEIGHTS
                )
                
                # Log analysis results
                logger.info(f"Signal: {signal_result['composite_signal']} | "
                           f"Confidence: {signal_result['confidence']:.1f}% | "
                           f"Score: {signal_result['weighted_score']:.3f} | "
                           f"Entry: {signal_result.get('entry_price', 0):.2f}")
                
                # Check alert conditions with market context
                if self._should_alert_with_context(signal_result, df):
                    self._send_alert(df['close'].iloc[-1], indicators, signal_result)
                                    
            except Exception as e:
                logger.error(f"Analysis error: {e}", exc_info=True)

    def _calculate_all_indicators(self, df: pd.DataFrame) -> Dict:
        """Calculate all technical indicators with type validation."""
        try:
            # CRITICAL FIX: Create a fresh copy and force conversion
            df_clean = pd.DataFrame()
            df_clean['open'] = pd.to_numeric(df['open'], errors='coerce').astype('float64')
            df_clean['high'] = pd.to_numeric(df['high'], errors='coerce').astype('float64')
            df_clean['low'] = pd.to_numeric(df['low'], errors='coerce').astype('float64')
            df_clean['close'] = pd.to_numeric(df['close'], errors='coerce').astype('float64')
            df_clean['volume'] = pd.to_numeric(df['volume'], errors='coerce').fillna(0).astype('int64')
            df_clean.index = df.index
            
            # Drop any rows with NaN prices
            df_clean = df_clean.dropna(subset=['open', 'high', 'low', 'close'])
            
            if len(df_clean) == 0:
                logger.error("No valid data after type conversion")
                return {}
            
            # Use the clean DataFrame for calculations
            prices = pd.Series(df_clean['close'].values, index=df_clean.index, dtype='float64')
            volumes = pd.Series(df_clean['volume'].values, index=df_clean.index, dtype='int64')
            highs = pd.Series(df_clean['high'].values, index=df_clean.index, dtype='float64')
            lows = pd.Series(df_clean['low'].values, index=df_clean.index, dtype='float64')
            
            # Debug logging
            logger.debug(f"DataFrame dtypes after cleaning: {df_clean.dtypes.to_dict()}")
            logger.debug(f"Sample close values: {prices.head().tolist()}")
            logger.debug(f"Close series dtype: {prices.dtype}")
            
            return {
                "macd": TechnicalIndicators.calculate_macd(prices),
                "rsi": TechnicalIndicators.calculate_rsi(prices),
                "vwap": TechnicalIndicators.calculate_vwap(prices, volumes),
                "keltner": TechnicalIndicators.calculate_keltner_channels(prices, highs, lows),
                "supertrend": TechnicalIndicators.calculate_supertrend(prices, highs, lows),
                "impulse": TechnicalIndicators.calculate_impulse_macd(prices)
            }
        except Exception as e:
            logger.error(f"Indicator calculation error: {e}", exc_info=True)
            return {}
    
    def _should_alert_with_context(self, signal_result: Dict, df: pd.DataFrame) -> bool:
        """Enhanced alert conditions with market context awareness."""
        try:
            # Get market context
            price_change_pct = ((df['close'].iloc[-1] - df['close'].iloc[-5]) / df['close'].iloc[-5]) * 100
            volatility = df['close'].pct_change().std() * 100
            
            # Dynamic thresholds based on market conditions
            if abs(price_change_pct) > 0.5:  # Trending market
                min_confidence = config.MIN_CONFIDENCE_FOR_ALERT * 0.75
                min_score = config.MIN_SCORE_FOR_ALERT * 0.75
            else:  # Ranging market
                min_confidence = config.MIN_CONFIDENCE_FOR_ALERT
                min_score = config.MIN_SCORE_FOR_ALERT
            
            # Check basic conditions
            if signal_result['confidence'] < min_confidence:
                logger.debug(f"Confidence too low: {signal_result['confidence']:.1f}% < {min_confidence}")
                return False
            
            if abs(signal_result['weighted_score']) < min_score:
                logger.debug(f"Score too low: {abs(signal_result['weighted_score']):.3f} < {min_score}")
                return False
            
            # Override for very strong signals
            if signal_result['composite_signal'] in ['STRONG_BUY', 'STRONG_SELL']:
                if signal_result['confidence'] > 60:
                    logger.info("Strong signal detected - overriding cooldown")
                    return True
            
            # Normal cooldown check
            if self.last_alert_time:
                elapsed = (datetime.now() - self.last_alert_time).total_seconds()
                
                # Dynamic cooldown based on market volatility
                cooldown = config.COOLDOWN_SECONDS
                if volatility > 1.0:
                    cooldown = cooldown // 2  # Halve cooldown in volatile markets
                
                if elapsed < cooldown:
                    logger.info(f"Cooldown active ({cooldown - elapsed:.0f}s remaining)")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Alert condition error: {e}")
            return False
    
    def _send_alert(self, current_price: float, indicators: Dict, signal_result: Dict):
        """Send enhanced alert with detailed analysis."""
        try:
            # Format message
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
                        "confidence": signal_result['confidence']
                    })
                    logger.info("Alert sent successfully")
                    
        except Exception as e:
            logger.error(f"Alert sending error: {e}")

    def fetch_historical_volume(self):
        """Fetch real volume data from Dhan API if WebSocket fails."""
        try:
            import requests
            from datetime import datetime, timedelta
            
            logger.info("Fetching historical volume data from REST API")
            
            # Get last 100 candles for volume reference
            end_date = datetime.now()
            start_date = end_date - timedelta(hours=24)
            
            # Dhan API endpoint for historical data
            url = "https://api.dhan.co/marketfeed/historical"
            
            headers = {
                "access-token": self.access_token,
                "Content-Type": "application/json"
            }
            
            payload = {
                "securityId": str(config.NIFTY_SECURITY_ID),
                "exchangeSegment": config.NIFTY_EXCHANGE_SEGMENT,
                "instrument": "INDEX",
                "fromDate": start_date.strftime("%Y-%m-%d"),
                "toDate": end_date.strftime("%Y-%m-%d"),
                "interval": "1"  # 1 minute interval
            }
            
            response = requests.post(url, json=payload, headers=headers)
            
            if response.status_code == 200:
                data = response.json()
                if data and 'data' in data:
                    volumes = []
                    for candle in data['data']:
                        if 'volume' in candle and candle['volume'] > 0:
                            volumes.append(candle['volume'])
                    
                    if volumes:
                        avg_volume = sum(volumes) / len(volumes)
                        self.volume_estimator.avg_volume_per_minute = int(avg_volume)
                        logger.info(f"Updated average volume: {avg_volume:,.0f}")
                        
                        # Update volume history
                        for vol in volumes[-100:]:  # Keep last 100
                            self.volume_estimator.volume_history.append(vol)
                        
                        return True
                        
            logger.warning(f"Failed to fetch volume data: {response.status_code}")
            return False
            
        except Exception as e:
            logger.error(f"Error fetching historical volume: {e}")
            return False
    
    def detect_market_regime(self, df: pd.DataFrame = None) -> str:
        """
        Detect if market is trending or ranging.
        
        Returns:
            str: 'trending', 'ranging', or 'undefined'
        """
        try:
            if df is None:
                df = self.minute_ohlcv
            
            if len(df) < 20:
                logger.debug("Insufficient data for regime detection")
                return "undefined"
            
            # Method 1: ATR-based volatility analysis
            atr = df['high'] - df['low']
            atr_std = atr.rolling(20).std()
            atr_mean = atr.mean()
            
            # Low volatility relative to mean indicates ranging
            if atr_std.iloc[-1] < atr_mean * 0.3:
                regime = "ranging"
            else:
                # Method 2: Directional movement analysis
                closes = df['close'].values[-20:]
                
                # Calculate linear regression slope
                x = np.arange(len(closes))
                slope = np.polyfit(x, closes, 1)[0]
                
                # Calculate R-squared for trend strength
                y_pred = np.polyval(np.polyfit(x, closes, 1), x)
                ss_res = np.sum((closes - y_pred) ** 2)
                ss_tot = np.sum((closes - np.mean(closes)) ** 2)
                r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
                
                # Strong directional movement with high R-squared = trending
                if abs(slope) > df['close'].mean() * 0.001 and r_squared > 0.6:
                    regime = "trending"
                else:
                    regime = "ranging"
            
            # Method 3: ADX indicator (if we want to add it)
            # adx = self._calculate_adx(df)
            # if adx > 25:
            #     regime = "trending"
            
            logger.info(f"Market regime detected: {regime}")
            
            # Store regime in instance for other methods to use
            self.current_market_regime = regime
            
            # Adjust parameters based on regime
            if regime == "trending":
                # In trending markets, be more responsive
                self.signal_sensitivity = 1.2
                logger.debug("Increased signal sensitivity for trending market")
            else:
                # In ranging markets, be more selective
                self.signal_sensitivity = 0.8
                logger.debug("Decreased signal sensitivity for ranging market")
            
            return regime
            
        except Exception as e:
            logger.error(f"Market regime detection error: {e}")
            return "undefined"
    
    def _schedule_volume_fetch(self):
        """Schedule periodic volume data fetching."""
        def fetch_volume_periodically():
            while self.connected:
                try:
                    # Fetch every 30 minutes
                    time.sleep(1800)
                    self.fetch_historical_volume()
                except Exception as e:
                    logger.error(f"Periodic volume fetch error: {e}")
        
        # Start background thread for volume fetching
        volume_thread = threading.Thread(target=fetch_volume_periodically, daemon=True)
        volume_thread.start()
        logger.info("Scheduled periodic volume fetching")

    
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
        """Handle WebSocket closure."""
        logger.warning(f"WebSocket closed - Code: {close_status_code}, Message: {close_msg}")
        self.connected = False
        
        # Send disconnection notification
        if self.telegram_bot:
            self.telegram_bot.send_message("⚠️ WebSocket disconnected - Attempting reconnection...")
        
        # Attempt reconnection
        if self.connection_retry_count < self.max_retries:
            self.connection_retry_count += 1
            wait_time = min(2 ** self.connection_retry_count, 60)
            logger.info(f"Reconnecting in {wait_time}s (Attempt {self.connection_retry_count}/{self.max_retries})...")
            time.sleep(wait_time)
            self.connect()
    
    def connect(self):
        """Establish WebSocket connection."""
        try:
            ws_url = (f"wss://api-feed.dhan.co?version=2"
                     f"&token={self.access_token}"
                     f"&clientId={self.client_id}"
                     f"&authType=2")
            
            self.ws = websocket.WebSocketApp(
                ws_url,
                on_open=self.on_open,
                on_message=self.on_message,
                on_error=self.on_error,
                on_close=self.on_close
            )
            
            wst = threading.Thread(
                target=self.ws.run_forever,
                kwargs={
                    "sslopt": {"cert_reqs": ssl.CERT_NONE},
                    "ping_interval": 30,
                    "ping_timeout": 10
                }
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
                
        except Exception as e:
            logger.error(f"Connection error: {e}")
            raise
    
    def disconnect(self):
        """Gracefully disconnect WebSocket."""
        if self.ws:
            self.connected = False
            self.ws.close()
            logger.info("WebSocket disconnected")


class VolumeEstimator:
    """Estimate volume when not available from feed."""
    
    def __init__(self):
        self.avg_volume_per_minute = 50000  # Average for Nifty
        self.last_volume = 0
        self.volume_history = collections.deque(maxlen=100)
        self.previous_close = None
        
    def estimate_volume(self, price: float, quantity: int) -> int:
        """Estimate volume based on price and quantity."""
        if quantity > 0:
            estimated = quantity * 100
            self.last_volume = estimated
            self.volume_history.append(estimated)
            return estimated
        
        # Return moving average or default
        if self.volume_history:
            return int(np.mean(self.volume_history))
        return self.avg_volume_per_minute
    
    def estimate_volume_with_value(self, price: float, value: float) -> int:
        """Estimate volume from traded value."""
        if value > 0 and price > 0:
            estimated = int(value / price)
            self.last_volume = estimated
            self.volume_history.append(estimated)
            return estimated
        return self.get_current_estimate()
    
    def estimate_from_price_action(self, prices: List[float]) -> int:
        """Estimate volume from price volatility."""
        if len(prices) < 2:
            return self.get_current_estimate()
        
        # Higher volatility usually means higher volume
        volatility = np.std(prices) / np.mean(prices)
        volume_multiplier = 1 + (volatility * 10)
        
        base_volume = self.avg_volume_per_minute
        if self.volume_history:
            base_volume = int(np.mean(self.volume_history))
        
        estimated = int(base_volume * volume_multiplier)
        self.last_volume = estimated
        return estimated
    
    def update_last_volume(self, volume: int):
        """Update last known volume."""
        if volume > 0:
            self.last_volume = volume
            self.volume_history.append(volume)
    
    def get_current_estimate(self) -> int:
        """Get current volume estimate."""
        if self.last_volume > 0:
            return self.last_volume
        if self.volume_history:
            return int(np.mean(self.volume_history))
        return self.avg_volume_per_minute
    
    def set_previous_close(self, price: float):
        """Set previous close price for reference."""
        self.previous_close = price
