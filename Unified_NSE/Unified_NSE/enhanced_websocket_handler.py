"""
DhanHQ v2 WebSocket handler - Enhanced with Synthetic Volume Generation
Based on volume creation logic from websocket_client.py
"""
import asyncio
import struct
import json
import logging
from typing import Dict, Optional, Any, List
import websockets
import pandas as pd
import numpy as np
import base64

logger = logging.getLogger(__name__)

from datetime import datetime, timezone, timedelta 
IST = timezone(timedelta(hours=5, minutes=30))

class EnhancedWebSocketHandler:
    """WebSocket handler for DhanHQ v2 with synthetic volume generation for indices."""
    
    # DhanHQ v2 Packet Types (from websocket_client.py)
    PACKET_TYPES = {
        8: "ticker",            # Ticker packet (LTP only)
        16: "quote",           # Quote packet (16 bytes for ticker)
        32: "index_full",      # Index full packet
        44: "equity_full",     # Equity/FNO full packet
        50: "quote_extended",  # Quote packet (50 bytes standard)
        66: "quote_full",      # Quote packet (66 bytes extended) 
        162: "full_packet",    # Full packet with market depth
        184: "market_depth",   # 20-depth market data
        492: "market_depth_50" # 50-depth market data
    }
    
    # Response codes from websocket_client.py
    TICKER_PACKET = 2
    QUOTE_PACKET = 4
    OI_PACKET = 5
    PREV_CLOSE_PACKET = 6
    MARKET_STATUS_PACKET = 7
    FULL_PACKET = 8
    DISCONNECT_PACKET = 50
    
    def __init__(self, config):
        logger.info("Initializing Enhanced WebSocket Handler with Synthetic Volume")
        self.config = config
        self.websocket = None
        self.authenticated = False
        self.running = True
        
        # Data buffers
        self.tick_buffer = []
        self.candle_data = pd.DataFrame()
        self.current_candle = {
            'ticks': [],
            'start_time': None
        }
        
        # Volume tracking (synthetic for indices)
        self.last_price = None
        self.last_cumulative_volume = 0
        self.current_period_volume = 0
        self.synthetic_volume_base = 1000  # Base synthetic volume
        
        
        # Callbacks 
        self.on_tick = None 
        self.on_candle = None 
        self.on_error = None 
        self.on_preclose = None # NEW: pre-close analysis callback
        
        # Statistics
        self.packet_stats = {packet_type: 0 for packet_type in self.PACKET_TYPES.values()}
        self.packet_stats.update({'ticker': 0, 'quote': 0, 'full': 0, 'oi': 0, 'other': 0})
        self.last_packet_time = None
        self.tick_count = 0
        self._diag_ticks_left = 50  # one-time startup diagnostic
        

        self.boundary_task = None
        self.data_watchdog_task = None
        self._last_subscribe_time = None



        # Pre-close and boundary close state 
        self._preclose_fired_for_bucket = None
        self._preclose_lock = asyncio.Lock()

                
        self._bucket_closed = False
        
        logger.info(f"Configuration: SecurityId={config.nifty_security_id}, "
                   f"Interval={config.candle_interval_seconds}s, "
                   f"MaxBuffer={config.max_buffer_size}")
    

    def _normalize_tick_ts(self, ltt: int) -> datetime:
        """
        Normalize exchange timestamp to IST.
        Tries both UTC->IST and direct IST epoch; picks the one closest to now(IST).
        """
        now_ist = datetime.now(IST)
        ts_utc_to_ist = datetime.fromtimestamp(ltt, tz=timezone.utc).astimezone(IST)
        ts_direct_ist = datetime.fromtimestamp(ltt, tz=IST)

        # Choose the ts closer to current IST time
        if abs((now_ist - ts_utc_to_ist).total_seconds()) <= abs((now_ist - ts_direct_ist).total_seconds()):
            return ts_utc_to_ist
        return ts_direct_ist
        


    
    

    
    

    def _assemble_candle(self, start_time: datetime, ticks: List[Dict]) -> pd.DataFrame: 
        """Build an OHLCV candle from accumulated ticks for this bucket.""" 
        try: 
            prices = [t.get('ltp', 0.0) for t in ticks if t.get('ltp', 0.0) > 0] 
            if not prices: 
                return pd.DataFrame() 
            open_price = float(prices[0]) 
            high = float(max(prices)) 
            low = float(min(prices)) 
            close = float(prices[-1])
            if self.current_period_volume > 0:
                candle_volume = int(self.current_period_volume)
            else:
                price_range = max(high - low, 0.0)
                volatility = (price_range / close) * 100 if close > 0 else 0
                candle_volume = 5000 + int(volatility * 10000) + len(ticks) * 100

            return pd.DataFrame([{
                'timestamp': start_time,
                'open': open_price,
                'high': high,
                'low': low,
                'close': close,
                'volume': candle_volume,
                'tick_count': len(ticks)
            }]).set_index('timestamp')
        except Exception as e:
            logger.debug(f"_assemble_candle error: {e}")
            return pd.DataFrame()



    async def _maybe_fire_preclose(self, now_ts: datetime): 
        """Trigger a pre-close preview once per bucket near its end."""
        try: 
            if not self.on_preclose or not self.current_candle['start_time']: 
                return 
            start = self.current_candle['start_time'] 
            interval_min = max(1, self.config.candle_interval_seconds // 60) 
            close_time = start + timedelta(minutes=interval_min) 
            lead = getattr(self.config, 'preclose_lead_seconds', 10)  # Use config value directly
            preclose_time = close_time - timedelta(seconds=lead)
            
            # logger.debug(
            #     f"Pre-close check → now={now_ts.strftime('%H:%M:%S')}, "
            #     f"start={start.strftime('%H:%M:%S')}, close={close_time.strftime('%H:%M:%S')}, "
            #     f"lead={lead}s"
            # )

            has_ticks = bool(self.current_candle.get('ticks'))
            fired = (self._preclose_fired_for_bucket == start)

            # Decision logic: Only fire once, and only after preclose_time, if not already fired and there are ticks
            if now_ts < preclose_time:
                # logger.debug(
                #     f"[Pre-Close] Skip: lead window not reached (now={now_ts.strftime('%H:%M:%S')}, "
                #     f"pre={preclose_time.strftime('%H:%M:%S')})"
                # )
                return
            if fired:
                logger.debug(
                    f"[Pre-Close] Skip: already fired for bucket start={start.strftime('%H:%M:%S')}"
                )
                return
            if not has_ticks:
                logger.info(
                    f"[Pre-Close] Skip: no ticks in current bucket (start={start.strftime('%H:%M:%S')})"
                )
                return


            # All conditions met: fire pre-close preview (race-safe)
            async with self._preclose_lock:
                if self._preclose_fired_for_bucket == start:
                    logger.debug(f"[Pre-Close] Skip inside lock: already fired for start={start.strftime('%H:%M:%S')}")
                    return

                preview = self._assemble_candle(start, self.current_candle['ticks'])
                if not preview.empty:
                    # Mark as fired BEFORE awaiting the long call
                    self._preclose_fired_for_bucket = start

                    logger.info("=" * 60)
                    logger.info("⏳ Pre-close checkpoint: start=%s close=%s fired_at=%s",
                                start.strftime('%H:%M:%S'), close_time.strftime('%H:%M:%S'), now_ts.strftime('%H:%M:%S'))
                    logger.info("=" * 60)
                    logger.info("continue logging")

                                        
                    await self.on_preclose(
                        preview,
                        self.candle_data.copy() if not self.candle_data.empty else preview.copy()
                    )
                    

        except Exception as e:
            logger.debug(f"Pre-close skipped: {e}")


    async def _boundary_close_loop(self): 
        """Close the current bucket at the time boundary without waiting for the next tick.""" 
        logger.info("Starting boundary close loop") 

        
        while self.running: 
            try: 
                await asyncio.sleep(1.0) 
                # Check every second
                now_ts = datetime.now(IST) 
                start = self.current_candle.get('start_time')
                            
                # Heartbeat every 30s
                if int(now_ts.timestamp()) % 30 == 0:
                    logger.debug(f"Boundary loop alive at {now_ts.strftime('%H:%M:%S')}")

                # Always check boundary close each second
                if start and self.current_candle.get('ticks'):
                    close_time = start + timedelta(seconds=self.config.candle_interval_seconds)
                    if now_ts >= close_time and not self._bucket_closed:

                        logger.info("=" * 60)
                        logger.info("⏱️ Boundary close %s→%s — creating candle & dispatching callbacks",
                                    start.strftime('%H:%M:%S'), close_time.strftime('%H:%M:%S'))
                        logger.info("=" * 60)
                        logger.info("continue logging")


                        
                        await self._create_candle(start, self.current_candle['ticks'])
                        self._bucket_closed = True

                          
                            
                            
                    
                        
            except Exception as e: 
                logger.error(f"Boundary close loop error: {e}", exc_info=True) 
                await asyncio.sleep(1.0)


        
    def _calculate_synthetic_volume(self, ltp: float) -> int:
        """Calculate synthetic volume based on price movement (from websocket_client.py)."""
        try:
            if self.last_price is None:
                # First tick - use base volume
                self.last_price = ltp
                return self.synthetic_volume_base
            
            # Calculate price change
            price_change = abs(ltp - self.last_price)
            
            if self.last_price and self.last_price > 0:
                price_change_pct = (price_change / self.last_price) * 100
            else:
                price_change_pct = 0
                            
            
            # Synthetic volume based on price movement and tick activity
            movement_volume = int(price_change_pct * 5000)
            synthetic_volume = self.synthetic_volume_base + movement_volume
            
            # Update last price
            self.last_price = ltp
            
            # Log significant movements
            if self.tick_count % 50 == 0 and price_change_pct > 0:
                logger.info(f"Synthetic volume: {synthetic_volume} (price moved {price_change:.2f}, {price_change_pct:.3f}%)")
            
            return synthetic_volume
            
        except Exception as e:
            logger.error(f"Error calculating synthetic volume: {e}")
            return self.synthetic_volume_base
    
    async def connect(self):
        """Establish WebSocket connection with retry logic."""
        logger.info("Starting WebSocket connection to DhanHQ v2")
        max_attempts = self.config.max_reconnect_attempts
        attempt = 0

        # Reset per-connection state
        self.tick_buffer.clear()
        self.candle_data = pd.DataFrame()
        self.current_candle = {'ticks': [], 'start_time': None}
        self.current_period_volume = 0
        self._preclose_fired_for_bucket = None
        self._bucket_closed = False
        logger.debug("Per-connection state reset")

        
        while attempt < max_attempts and self.running:
            try:
                attempt += 1
                logger.info(f"Connection attempt {attempt}/{max_attempts}")
                
                # Decode credentials
                access_token = base64.b64decode(self.config.dhan_access_token_b64).decode("utf-8")
                client_id = base64.b64decode(self.config.dhan_client_id_b64).decode("utf-8")
                logger.debug(f"Credentials decoded for client: {client_id[:4]}****")
                
                
                # Build WebSocket URL
                ws_url = (
                    f"wss://api-feed.dhan.co?version=2"
                    f"&token={access_token}"
                    f"&clientId={client_id}"
                    f"&authType=2"
                )
                
                # Connect with appropriate settings
                self.websocket = await websockets.connect(
                    ws_url,
                    ping_interval=30,
                    ping_timeout=10,
                    max_size=10 * 1024 * 1024,
                    compression=None
                )
                
                self.authenticated = True
                logger.info("WebSocket connected successfully")

                # Reset per-connection timing state for watchdog/metrics (expanded to reset more states if needed)
                self.last_packet_time = None
                self._last_subscribe_time = None  # Additional reset for subscribe timestamp to handle edge cases [[5]]
                logger.debug("Per-connection timing reset (last_packet_time=None, _last_subscribe_time=None)")

                
                # Subscribe to market data
                await self.subscribe()


                logger.info("WebSocket connection established and subscribed")
                # Start boundary close loop once with error handling
                if not self.boundary_task or self.boundary_task.done():
                    self.boundary_task = asyncio.create_task(self._boundary_close_loop())
                    self.boundary_task.add_done_callback(self._handle_task_exception)
                # Start data-stall watchdog once
                if not self.data_watchdog_task or self.data_watchdog_task.done():
                    self.data_watchdog_task = asyncio.create_task(self._data_watchdog_loop())
                    self.data_watchdog_task.add_done_callback(self._handle_task_exception)

                return True



            except Exception as e:
                logger.error(f"Connection attempt {attempt} failed: {e}")
                
                if attempt < max_attempts:
                    delay = min(self.config.reconnect_delay_base * (2 ** (attempt - 1)), 60)
                    logger.info(f"Retrying in {delay} seconds...")
                    await asyncio.sleep(delay)
        
        logger.error("Failed to establish WebSocket connection after all attempts")
        return False
    
    
    def _handle_task_exception(self, task):
        """Handle exceptions from background tasks."""
        try:
            task.result()
        except asyncio.CancelledError:
            pass  # Task was cancelled, this is expected
        except Exception as e:
            logger.error(f"Background task error: {e}", exc_info=True)
            
            

    async def subscribe(self):
        """Subscribe to NIFTY50 market data feed - request Quote packets for volume."""
        logger.info("Subscribing to NIFTY50 market data")
        
        try:
            # DhanHQ v2 subscription format - request all packet types
            subscription = {
                "RequestCode": 15,  # Subscribe for live feed
                "InstrumentCount": 1,
                "InstrumentList": [{
                    "ExchangeSegment": self.config.nifty_exchange_segment,
                    "SecurityId": str(self.config.nifty_security_id)  # int, not str str(self.config.nifty_security_id)
                }]
            }  

            # Send subscription and wait for response
            try:
                logger.info(f"[Subscribe] Sending subscription at {datetime.now(IST).strftime('%H:%M:%S')} with params: {subscription}")
                
                await self.websocket.send(json.dumps(subscription))
                self._last_subscribe_time = datetime.now(IST)
                logger.info(f"[Subscribe] _last_subscribe_time set to {self._last_subscribe_time.strftime('%H:%M:%S')}")
                logger.info("[Subscribe] Waiting for market data...")
                
                # Wait for subscription confirmation
                await asyncio.sleep(2)
                
                # Check if we received any response
                if self.tick_count == 0:
                    logger.warning("[Subscribe] No ticks received after subscription - checking market status")
                    # Try sending a heartbeat/status request
                    status_request = {"RequestCode": 7}  # Market status request
                    await self.websocket.send(json.dumps(status_request))
                    logger.info("[Subscribe] Sent market status request")
                    
            except Exception as e:
                logger.error(f"[Subscribe] Error during subscription: {e}")
                raise



            
        except Exception as e:
            logger.error(f"Subscription error: {e}")
            raise
    
    def _parse_ticker_packet(self, data: bytes) -> Optional[Dict]:
        """Parse Ticker packet (16 bytes) - adapted from websocket_client.py."""
        if len(data) < 16:
            return None
        
        try:
            # Parse header (bytes 0-7)
            response_code = data[0]
            if response_code != self.TICKER_PACKET:
                return None
            
            message_length = struct.unpack('<H', data[1:3])[0]
            exchange_segment = data[3]
            security_id = struct.unpack('<I', data[4:8])[0]
            
            if security_id != self.config.nifty_security_id:
                return None
            
            # Parse ticker data (bytes 8-15)
            ltp = struct.unpack('<f', data[8:12])[0]

            if not np.isfinite(ltp):
                return None
                            
            ltt = struct.unpack('<I', data[12:16])[0]
            
            # Sanity check
            if not (self.config.price_sanity_min <= ltp <= self.config.price_sanity_max):
                logger.warning(f"Price sanity check failed: {ltp}")
                return None
        

            # Convert timestamp (robust)
            try:
                ts = self._normalize_tick_ts(ltt)
                now_ist = datetime.now(IST)
                # if parsed time is implausible (older than 1hr or pre-2000), fall back to now
                if abs((now_ist - ts).total_seconds()) > 3600 or ts.year < 2000:
                    timestamp = now_ist
                else:
                    timestamp = ts
            except Exception:
                timestamp = datetime.now(IST)



            
            # Calculate synthetic volume for index
            synthetic_volume = self._calculate_synthetic_volume(ltp)
            self.current_period_volume += synthetic_volume
            
            self.packet_stats['ticker'] += 1
            
            logger.debug(f"Ticker: LTP={ltp:.2f}, SyntheticVol={synthetic_volume}")
            
            return {
                'timestamp': timestamp,
                'packet_type': 'ticker',
                'ltp': ltp,
                'volume': synthetic_volume  # Synthetic volume for index
            }
            
        except Exception as e:
            logger.error(f"Error parsing ticker packet: {e}")
            return None
    
    def _parse_quote_packet(self, data: bytes) -> Optional[Dict]:
        """Parse Quote packet (50 or 66 bytes) - adapted from websocket_client.py."""
        if len(data) < 50:
            return None
        
        try:
            # Parse Response Header (bytes 0-7)
            response_code = data[0]
            if response_code != self.QUOTE_PACKET:
                return None
            
            message_length = struct.unpack('<H', data[1:3])[0]
            exchange_segment = data[3]
            security_id = struct.unpack('<I', data[4:8])[0]
            
            if security_id != self.config.nifty_security_id:
                return None
            
            # Parse Quote Data (bytes 8-49)
            ltp = struct.unpack('<f', data[8:12])[0]
            if not np.isfinite(ltp):
                return None
                                        
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
            
            # Validate price
            if not (self.config.price_sanity_min <= ltp <= self.config.price_sanity_max):
                logger.warning(f"Price sanity check failed: {ltp}")
                return None

            # Convert timestamp (robust)
            try:
                ts = self._normalize_tick_ts(ltt)
                now_ist = datetime.now(IST)
                if abs((now_ist - ts).total_seconds()) > 3600 or ts.year < 2000:
                    timestamp = now_ist
                else:
                    timestamp = ts
            except Exception:
                timestamp = datetime.now(IST)


            # Handle volume for index (usually 0)
            if volume == 0:
                # Calculate synthetic volume based on price movement and market activity
                synthetic_volume = self._calculate_synthetic_volume(ltp)
                
                # Add volume based on buy/sell quantities if available
                if total_buy_qty > 0 or total_sell_qty > 0:
                    activity_volume = (total_buy_qty + total_sell_qty) // 1000
                    synthetic_volume += activity_volume
                
                volume = synthetic_volume
                self.current_period_volume += synthetic_volume
            else:
                # Real volume (shouldn't happen for index but handle it)
                if volume > self.last_cumulative_volume:
                    volume_change = volume - self.last_cumulative_volume
                    self.current_period_volume += volume_change
                    self.last_cumulative_volume = volume
            
            self.packet_stats['quote'] += 1
            
            if self.packet_stats['quote'] % 10 == 0:
                logger.info(f"Quote #{self.packet_stats['quote']}: LTP={ltp:.2f}, "
                           f"Volume={volume:,} (synthetic), "
                           f"OHLC=[{open_value:.2f},{high_value:.2f},{low_value:.2f},{close_value:.2f}]")

            return {
                'timestamp': timestamp,
                'packet_type': 'quote',
                'ltp': ltp,
                'ltq': ltq,
                'atp': atp,
                'volume': volume,
                'total_sell_qty': total_sell_qty,
                'total_buy_qty': total_buy_qty,
                'open': open_value if open_value > 0 else ltp,
                'high': high_value if high_value > 0 else ltp,
                'low': low_value if low_value > 0 else ltp,
                'close': close_value if close_value > 0 else ltp
            }

        except Exception as e:
            logger.error(f"Error parsing quote packet: {e}")
            return None
    
    def _parse_full_packet(self, data: bytes) -> Optional[Dict]:
        """Parse Full packet (162 bytes) with market depth - adapted from websocket_client.py."""
        if len(data) < 162:
            return None
        
        try:
            # Parse header (bytes 0-7)
            response_code = data[0]
            if response_code != self.FULL_PACKET:
                return None
            
            message_length = struct.unpack('<H', data[1:3])[0]
            exchange_segment = data[3]
            security_id = struct.unpack('<I', data[4:8])[0]
            
            if security_id != self.config.nifty_security_id:
                return None
            
            # Parse main data (bytes 8-61)
            ltp = struct.unpack('<f', data[8:12])[0]
            if not np.isfinite(ltp):
                return None
                        
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
            
            # Validate price
            if not (self.config.price_sanity_min <= ltp <= self.config.price_sanity_max):
                logger.warning(f"Price sanity check failed: {ltp}")
                return None
            
            # Calculate synthetic volume for index
            if volume == 0:
                # Enhanced synthetic volume calculation for full packet
                synthetic_volume = self._calculate_synthetic_volume(ltp)
                
                # Add volume based on OHLC spread
                if high_value > 0 and low_value > 0:
                    price_range = high_value - low_value
                    volatility = (price_range / ltp) * 100 if ltp > 0 else 0
                    volatility_volume = int(volatility * 10000)
                    synthetic_volume += volatility_volume
                
                # Add volume based on market activity
                if total_buy_qty > 0 or total_sell_qty > 0:
                    activity_volume = (total_buy_qty + total_sell_qty) // 1000
                    synthetic_volume += activity_volume
                
                volume = synthetic_volume
                self.current_period_volume += synthetic_volume
            else:
                # Real volume
                if volume > self.last_cumulative_volume:
                    volume_change = volume - self.last_cumulative_volume
                    self.current_period_volume += volume_change
                    self.last_cumulative_volume = volume
            
            # Parse market depth (5 levels)
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
                ts = self._normalize_tick_ts(ltt)
                now_ist = datetime.now(IST)
                if abs((now_ist - ts).total_seconds()) > 3600 or ts.year < 2000:
                    timestamp = now_ist
                else:
                    timestamp = ts
            except Exception:
                timestamp = datetime.now(IST)


            self.packet_stats['full'] += 1
            
            logger.info(f"Full packet: LTP={ltp:.2f}, Volume={volume:,} (synthetic), "
                       f"OI={oi}, Depth levels=5")
            
            return {
                'timestamp': timestamp,
                'packet_type': 'full',
                'ltp': ltp,
                'ltq': ltq,
                'volume': volume,
                'open': open_value if open_value > 0 else ltp,
                'high': high_value if high_value > 0 else ltp,
                'low': low_value if low_value > 0 else ltp,
                'close': close_value if close_value > 0 else ltp,
                'total_buy_qty': total_buy_qty,
                'total_sell_qty': total_sell_qty,
                'oi': oi,
                'market_depth': market_depth
            }
            
        except Exception as e:
            logger.error(f"Error parsing full packet: {e}")
            return None
    
    async def _process_tick(self, tick_data: Dict):
        """Process incoming tick data with synthetic volume support."""
        if not tick_data:
            return
        
        try:

            # Update last packet time
            self.last_packet_time = datetime.now(IST)
            self.tick_count += 1
            
            # Get volume (already synthetic if needed)
            volume = tick_data.get('volume', 0)
            
            # Log volume updates periodically
            if self.tick_count % 20 == 0:
                logger.info(f"Tick #{self.tick_count}: Type={tick_data.get('packet_type')}, "
                           f"LTP={tick_data.get('ltp', 0):.2f}, Volume={volume:,}")
            
            # Add to buffer
            self.tick_buffer.append(tick_data)
            if len(self.tick_buffer) > self.config.max_buffer_size:
                self.tick_buffer.pop(0)
            
            # Candle management
            current_time = tick_data['timestamp']
            
            interval_min = max(1, self.config.candle_interval_seconds // 60)
            bucket_min = (current_time.minute // interval_min) * interval_min 
            candle_start = current_time.replace(minute=bucket_min, second=0, microsecond=0)

            # One-time diagnostic to verify bucket advancement
            if self._diag_ticks_left > 0:
                self._diag_ticks_left -= 1
                logger.info(f"DBG: tick_ts={current_time.strftime('%H:%M:%S')} bucket={candle_start.strftime('%H:%M:%S')} "
                            f"start={self.current_candle['start_time'].strftime('%H:%M:%S') if self.current_candle['start_time'] else 'None'} "
                            f"ticks={len(self.current_candle['ticks']) if self.current_candle['ticks'] else 0}")
                logger.debug(f"DBG: tick_ts={current_time.strftime('%H:%M:%S')} bucket={candle_start.strftime('%H:%M:%S')} "
                            f"start={self.current_candle['start_time'].strftime('%H:%M:%S') if self.current_candle['start_time'] else 'None'} "
                            f"ticks={len(self.current_candle['ticks']) if self.current_candle['ticks'] else 0}")

                

            if self.current_candle['start_time'] != candle_start: 
                # Complete previous candle if not already boundary-closed 
                if self.current_candle['start_time'] and self.current_candle['ticks'] and not self._bucket_closed: 
                    await self._create_candle(self.current_candle['start_time'], self.current_candle['ticks'])

                # Start new candle and reset state
                self.current_candle = {
                    'start_time': candle_start,
                    'ticks': [tick_data]
                }
                self.current_period_volume = 0
                self._bucket_closed = False
                self._preclose_fired_for_bucket = None
                logger.debug(f"New candle period started: {candle_start.strftime('%H:%M:%S')}")
            else:
                self.current_candle['ticks'].append(tick_data)

            
        
        
            # Safety flush: if the current bucket exceeds interval and has ticks, close it
            try:
                if (self.current_candle['start_time'] 
                    and (current_time - self.current_candle['start_time']).total_seconds() 
                        >= self.config.candle_interval_seconds 
                    and len(self.current_candle['ticks']) >= 3):
                    await self._create_candle(
                        self.current_candle['start_time'],
                        self.current_candle['ticks']
                    )
                    # start a new bucket from current_time
                    interval_min = max(1, self.config.candle_interval_seconds // 60)
                    bucket_min = (current_time.minute // interval_min) * interval_min
                    self.current_candle = {
                        'start_time': current_time.replace(minute=bucket_min, second=0, microsecond=0),
                        'ticks': []
                    }
                    self.current_period_volume = 0
            except Exception as e:
                logger.debug(f"Safety flush skipped: {e}")



            # Pre-close preview 
            try: 
                await self._maybe_fire_preclose(current_time) 
            except Exception as e: 
                logger.debug(f"Pre-close check failed: {e}")

        




            # Trigger tick callback
            if self.on_tick:
                await self.on_tick(tick_data)
                
        except Exception as e:
            logger.error(f"Tick processing error: {e}", exc_info=True)
    
    async def _create_candle(self, timestamp: datetime, ticks: List[Dict]):
        """Create OHLCV candle with synthetic volume."""
        if not ticks:
            return
        
        try:
            # Extract prices
            prices = [t['ltp'] for t in ticks if 'ltp' in t and t['ltp'] > 0]
            if not prices:
                logger.warning("No valid prices in ticks for candle creation")
                return
            
            # Calculate OHLC
            open_price = prices[0]
            high = max(prices)
            low = min(prices)
            close = prices[-1]
            
            # Calculate volume for candle
            if self.current_period_volume > 0:
                # Use accumulated synthetic volume for the period
                candle_volume = self.current_period_volume
            else:
                # Fallback: create synthetic volume based on candle characteristics
                price_range = high - low
                volatility = (price_range / close) * 100 if close > 0 else 0
                tick_count = len(ticks)
                
                # Synthetic volume formula
                base_volume = 5000
                volatility_volume = int(volatility * 10000)
                activity_volume = tick_count * 100
                candle_volume = base_volume + volatility_volume + activity_volume
            
            # NSE market hours: 09:15–15:30 IST 
            hhmm = timestamp.hour * 100 + timestamp.minute 
            if timestamp.weekday() >= 5 or not (915 <= hhmm <= 1530): 
                logger.debug(f"Skipping candle outside market hours: {timestamp.strftime('%H:%M:%S')}") 
                logger.info(f"Skipping candle outside market hours: {timestamp.strftime('%H:%M:%S')}")
                return
            
            # # Create candle
            # candle = pd.DataFrame([{
            #     'timestamp': timestamp,
            #     'open': open_price,
            #     'high': high,
            #     'low': low,
            #     'close': close,
            #     'volume': candle_volume,  # Always has value for indicators
            #     'tick_count': len(ticks)
            # }]).set_index('timestamp')
            
            # logger.info(f"Candle Created: {timestamp.strftime('%H:%M:%S')} | "
            #            f"O:{open_price:.2f} H:{high:.2f} L:{low:.2f} C:{close:.2f} | "
            #            f"Volume:{candle_volume:,} (synthetic) | Ticks:{len(ticks)}")
            
             

            # Label candle by START time for real-time context
            interval_min = max(1, self.config.candle_interval_seconds // 60) 
            candle_start = timestamp 
            candle = pd.DataFrame([{
                'timestamp': candle_start,  # Use start time for immediate context
                'open': open_price,
                'high': high,
                'low': low,
                'close': close,
                'volume': candle_volume,
                'tick_count': len(ticks)
            }]).set_index('timestamp')


            # Update logging to show both times
            candle_end = candle_start + timedelta(minutes=interval_min)
            logger.info(f"Candle Created: {candle_start.strftime('%H:%M:%S')}-{candle_end.strftime('%H:%M:%S')} | "
                    f"O:{open_price:.2f} H:{high:.2f} L:{low:.2f} C:{close:.2f} | "
                    f"Volume:{candle_volume:,} (synthetic) | Ticks:{len(ticks)}")

                        
            # Update candle data
            if self.candle_data.empty:
                self.candle_data = candle
            else:
                self.candle_data = pd.concat([self.candle_data, candle])
                self.candle_data = self.candle_data.tail(self.config.max_candles_stored)
            
            
            self._bucket_closed = True
            
            # Trigger candle callback
            if self.on_candle:
                await self.on_candle(candle, self.candle_data)
                
        except Exception as e:
            logger.error(f"Candle creation error: {e}", exc_info=True)
    
    async def process_messages(self):
        """Main message processing loop with enhanced packet parsing."""
        logger.info("Starting DhanHQ v2 message processing with synthetic volume")
        
        message_count = 0
        error_count = 0
        last_status_log = datetime.now()
        
        try:
            async for message in self.websocket:
                message_count += 1
                if message_count <= 10:
                    logger.info(f"[process_messages] Received message #{message_count} at {datetime.now(IST).strftime('%H:%M:%S')}")
                try:
                    if isinstance(message, bytes):
                        # logger.info(f"[DEBUG] Binary message size: {len(message)} bytes, first byte: {message[0] if message else 'empty'}")
                        # Binary packet - market data
                        packet_size = len(message)
                        tick_data = None
                        
                        # First check if it's a response code packet
                        if packet_size >= 8:
                            response_code = message[0]
                            
                            # Route based on response code first
                            if response_code == self.TICKER_PACKET and packet_size == 16:
                                tick_data = self._parse_ticker_packet(message)
                            elif response_code == self.QUOTE_PACKET and packet_size >= 50:
                                tick_data = self._parse_quote_packet(message)
                            elif response_code == self.FULL_PACKET and packet_size == 162:
                                tick_data = self._parse_full_packet(message)
                            elif response_code == self.PREV_CLOSE_PACKET and packet_size == 16:
                                tick_data = self._parse_prev_close_packet(message)
                            elif response_code == self.OI_PACKET and packet_size == 12:
                                tick_data = self._parse_oi_packet(message)
                            elif response_code == self.DISCONNECT_PACKET:
                                self._handle_disconnect_packet(message)
                            else:
                                # Fallback to size-based parsing for legacy support
                                packet_type = self.PACKET_TYPES.get(packet_size, f"unknown_{packet_size}")
                                
                                if packet_size == 8:
                                    tick_data = self._parse_ticker_8(message)
                                elif packet_size == 16:
                                    tick_data = self._parse_quote_16(message)
                                elif packet_size == 32:
                                    tick_data = self._parse_index_full_32(message)
                                elif packet_size == 44:
                                    tick_data = self._parse_equity_full_44(message)
                                elif packet_size == 50 or packet_size == 66:
                                    tick_data = self._parse_quote_packet(message)
                                elif packet_size == 162:
                                    tick_data = self._parse_full_packet(message)
                                elif packet_size == 184:
                                    tick_data = self._parse_market_depth_184(message)
                                else:
                                    self.packet_stats['other'] += 1
                        
                        if tick_data:
                            await self._process_tick(tick_data)
                            
                    else:
                        # Text message - control messages
                        logger.info(f"[DEBUG] Text message: {message[:200] if message else 'empty'}")
                        await self._handle_text_message(message)
                    
                    # Periodic status logging (every 60 seconds)
                    if (datetime.now() - last_status_log).total_seconds() > 60:
                        logger.info(f"Status: Messages={message_count}, Errors={error_count}, "
                                   f"Ticks={self.tick_count}, CurrentVolume={self.current_period_volume:,}")
                        logger.info(f"Packet stats: {self.packet_stats}")
                        last_status_log = datetime.now()
                        
                except Exception as e:
                    error_count += 1
                    logger.error(f"Message processing error: {e}")
                    
                    if error_count > 50:
                        logger.critical("Too many errors, attempting reconnection")
                        break
                        
        except websockets.exceptions.ConnectionClosed as e:
            logger.warning(f"WebSocket connection closed: {e}")
            self.authenticated = False
        except Exception as e:
            logger.error(f"Fatal error in message loop: {e}", exc_info=True)
            if self.on_error:
                await self.on_error(e)
    


    async def _data_watchdog_loop(self):
        """Monitor for data stall; resubscribe once, then reconnect if still stalled."""
        stall_secs = int(getattr(self.config, 'data_stall_seconds', 15))
        retry_secs = int(getattr(self.config, 'data_stall_reconnect_seconds', 30))
        did_resubscribe = False
        logger.info("Starting data-stall watchdog")
                
        while self.running:
            try:
                await asyncio.sleep(1)
                now = datetime.now(IST)
                # logger.debug(
                #     f"[Watchdog] now={now.strftime('%H:%M:%S')}, "
                #     f"_last_subscribe_time={self._last_subscribe_time.strftime('%H:%M:%S') if self._last_subscribe_time else 'None'}, "
                #     f"last_packet_time={self.last_packet_time.strftime('%H:%M:%S') if self.last_packet_time else 'None'}, "
                #     f"did_resubscribe={did_resubscribe}"
                # )
                

                # If we never got any packets since connect
                if self.last_packet_time is None:
                    if self._last_subscribe_time:
                        since_sub = (now - self._last_subscribe_time).total_seconds()
                        logger.info(f"[Watchdog] {since_sub:.1f}s since subscribe, no packets yet.")
                        
                        
                        
                    # After stall_secs from subscribe, try resubscribe once
                    if self._last_subscribe_time and (now - self._last_subscribe_time).total_seconds() >= stall_secs and not did_resubscribe:
                        logger.warning(f"No market data for {stall_secs}s after subscribe — re-subscribing")
                        try:
                            await self.subscribe()
                            logger.info(f"[Watchdog] Resubscribe triggered at {now.strftime('%H:%M:%S')}")
                            did_resubscribe = True
                        except Exception as e:
                            logger.error(f"Resubscribe failed: {e}")
                    # After retry_secs, still no packets — reconnect
                    if self._last_subscribe_time and (now - self._last_subscribe_time).total_seconds() >= retry_secs:

                        logger.warning(f"No market data for {retry_secs}s — reconnecting WebSocket")
                        try:
                            await self.disconnect(stop_running=False)  # Keep running for internal retry [[8]]
                        finally:
                            logger.info(f"[Watchdog] Reconnect triggered at {now.strftime('%H:%M:%S')}")
                            break

  
                else:
                    # Got data — reset watchdog state
                    did_resubscribe = False
                    since_last_packet = (now - self.last_packet_time).total_seconds()
                    # logger.debug(f"[Watchdog] Last packet received {since_last_packet:.1f}s ago.")
            except asyncio.CancelledError:
                logger.debug("Data-stall watchdog cancelled")
                break
            except Exception as e:
                logger.error(f"Data-stall watchdog error: {e}", exc_info=True)
                await asyncio.sleep(2)


        
    
        
    async def run_forever(self):
        """Connect, process, and auto-reconnect until self.running is False."""
        backoff = self.config.reconnect_delay_base
        while self.running:
            try:
                ok = await self.connect()
                if not ok:
                    logger.error("Connect failed, honoring backoff")
                    await asyncio.sleep(min(backoff, 60))
                    backoff = min(backoff * 2, 60)
                    continue

                # Reset backoff after a successful connect
                backoff = self.config.reconnect_delay_base

                await self.process_messages()

                # If process_messages returns without exception, it means the server closed or loop ended
                    
                if self.running:
                    logger.warning("Message loop ended; attempting reconnection")
                    await self.disconnect(stop_running=False)  # Internal reconnect without shutdown [[2]]
                    await asyncio.sleep(backoff)
                    backoff = min(backoff * 2, 60)  # Exponential backoff for resilience [[4]]


                    
            except asyncio.CancelledError:
                logger.info("run_forever cancelled")
                break
            except Exception as e:
                logger.error(f"run_forever error: {e}", exc_info=True)
                if self.running:
                    await asyncio.sleep(min(backoff, 60))
                    backoff = min(backoff * 2, 60)
    
    
    
    # Include remaining methods from original file...
    def _parse_ticker_8(self, data: bytes) -> Optional[Dict]:
        """Parse 8-byte ticker packet (LTP only) - legacy support."""
        # This is kept for compatibility but shouldn't be used for Dhan v2
        if len(data) < 8:
            return None
        
        try:
            security_id = struct.unpack('<I', data[0:4])[0]
            ltp = struct.unpack('<f', data[4:8])[0]
            
            if security_id != self.config.nifty_security_id:
                return None
            
            if not (self.config.price_sanity_min <= ltp <= self.config.price_sanity_max):
                logger.warning(f"Price sanity check failed: {ltp}")
                return None
            
            # Calculate synthetic volume
            synthetic_volume = self._calculate_synthetic_volume(ltp)
            self.current_period_volume += synthetic_volume
            
            logger.debug(f"Legacy Ticker: LTP={ltp:.2f}, SyntheticVol={synthetic_volume}")
            
            return {
                'timestamp': datetime.now(IST),
                'packet_type': 'ticker',
                'ltp': ltp,
                'volume': synthetic_volume
            }
            
        except Exception as e:
            logger.error(f"Error parsing ticker packet: {e}")
            return None
    
    def _parse_prev_close_packet(self, data: bytes) -> Optional[Dict]:
        """Parse Previous Close packet (16 bytes)."""
        if len(data) < 16:
            return None
        
        try:
            response_code = data[0]
            if response_code != self.PREV_CLOSE_PACKET:
                return None
            
            message_length = struct.unpack('<H', data[1:3])[0]
            exchange_segment = data[3]
            security_id = struct.unpack('<I', data[4:8])[0]
            
            if security_id == self.config.nifty_security_id:
                prev_close = struct.unpack('<f', data[8:12])[0]
                prev_oi = struct.unpack('<I', data[12:16])[0]
                
                self.packet_stats['prev_close'] = self.packet_stats.get('prev_close', 0) + 1
                logger.info(f"Previous close: {prev_close:.2f}, Previous OI: {prev_oi}")
                
                return {
                    "packet_type": "prev_close",
                    "prev_close": prev_close,
                    "prev_oi": prev_oi,
                    "timestamp": datetime.now(IST)
                }
        except Exception as e:
            logger.debug(f"Prev close packet parse error: {e}")
        return None
    
    def _parse_oi_packet(self, data: bytes) -> Optional[Dict]:
        """Parse OI Data packet (12 bytes)."""
        if len(data) < 12:
            return None
        
        try:
            response_code = data[0]
            if response_code != self.OI_PACKET:
                return None
            
            message_length = struct.unpack('<H', data[1:3])[0]
            exchange_segment = data[3]
            security_id = struct.unpack('<I', data[4:8])[0]
            
            if security_id == self.config.nifty_security_id:
                oi = struct.unpack('<I', data[8:12])[0]
                
                self.packet_stats['oi'] = self.packet_stats.get('oi', 0) + 1
                
                return {
                    "packet_type": "oi",
                    "oi": oi,
                    "timestamp": datetime.now(IST)
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
                
                # Could trigger reconnection logic here if needed
        except Exception as e:
            logger.error(f"Disconnect packet handling error: {e}")
    
    # Include remaining original methods that don't need modification
    def _parse_quote_16(self, data: bytes) -> Optional[Dict]:
        """Parse 16-byte quote packet - legacy support."""
        # Similar to original but with synthetic volume
        if len(data) < 16:
            return None
        
        try:
            packet_type = struct.unpack('<I', data[0:4])[0]
            security_id = struct.unpack('<I', data[4:8])[0]
            ltp = struct.unpack('<f', data[8:12])[0]
            if not np.isfinite(ltp):
                return None
                        
            close = struct.unpack('<f', data[12:16])[0]
            
            if security_id != self.config.nifty_security_id:
                return None
            
            if not (self.config.price_sanity_min <= ltp <= self.config.price_sanity_max):
                logger.warning(f"Price sanity check failed: {ltp}")
                return None
            
            # Calculate synthetic volume
            synthetic_volume = self._calculate_synthetic_volume(ltp)
            self.current_period_volume += synthetic_volume
            
            logger.debug(f"Quote16: LTP={ltp:.2f}, Close={close:.2f}, SyntheticVol={synthetic_volume}")
            
            return {
                'timestamp': datetime.now(IST),
                'packet_type': 'quote',
                'ltp': ltp,
                'close': close,
                'volume': synthetic_volume
            }
            
        except Exception as e:
            logger.error(f"Error parsing quote packet: {e}")
            return None
    
    def _parse_index_full_32(self, data: bytes) -> Optional[Dict]:
        """Parse 32-byte index full packet with synthetic volume."""
        if len(data) < 32:
            return None
        
        try:
            security_id = struct.unpack('<I', data[4:8])[0]
            
            if security_id != self.config.nifty_security_id:
                return None
            
            ltp = struct.unpack('<f', data[8:12])[0]
            if not np.isfinite(ltp):
                return None   
                   
            open_price = struct.unpack('<f', data[12:16])[0]
            high = struct.unpack('<f', data[16:20])[0]
            low = struct.unpack('<f', data[20:24])[0]
            close = struct.unpack('<f', data[24:28])[0]
            
            # Calculate synthetic volume based on OHLC spread
            synthetic_volume = self._calculate_synthetic_volume(ltp)
            
            # Add extra volume based on volatility
            if high > 0 and low > 0:
                price_range = high - low
                volatility = (price_range / ltp) * 100 if ltp > 0 else 0
                volatility_volume = int(volatility * 5000)
                synthetic_volume += volatility_volume
            
            self.current_period_volume += synthetic_volume
            
            logger.debug(f"Index Full: OHLC=[{open_price:.2f},{high:.2f},{low:.2f},{close:.2f}], "
                        f"LTP={ltp:.2f}, SyntheticVol={synthetic_volume:,}")
            
            return {
                'timestamp': datetime.now(IST),
                'packet_type': 'index_full',
                'ltp': ltp,
                'open': open_price,
                'high': high,
                'low': low,
                'close': close,
                'volume': synthetic_volume
            }
            
        except Exception as e:
            logger.error(f"Error parsing index full packet: {e}")
            return None
    
    def _parse_equity_full_44(self, data: bytes) -> Optional[Dict]:
        """Parse 44-byte equity/FNO full packet."""
        if len(data) < 44:
            return None
        
        try:
            security_id = struct.unpack('<I', data[4:8])[0]
            
            if security_id != self.config.nifty_security_id:
                return None
            
            ltp = struct.unpack('<f', data[8:12])[0]
            if not np.isfinite(ltp):
                return None
            open_price = struct.unpack('<f', data[12:16])[0]
            high = struct.unpack('<f', data[16:20])[0]
            low = struct.unpack('<f', data[20:24])[0]
            close = struct.unpack('<f', data[24:28])[0]
            volume = struct.unpack('<I', data[28:32])[0]
            oi = struct.unpack('<I', data[32:36])[0]
            last_traded_time = struct.unpack('<I', data[36:40])[0]
            exchange_time = struct.unpack('<I', data[40:44])[0]
            
            # Handle volume
            if volume == 0:
                # Calculate synthetic volume for index
                synthetic_volume = self._calculate_synthetic_volume(ltp)
                
                # Add extra based on OHLC
                if high > 0 and low > 0:
                    price_range = high - low
                    volatility = (price_range / ltp) * 100 if ltp > 0 else 0
                    volatility_volume = int(volatility * 7500)
                    synthetic_volume += volatility_volume
                
                volume = synthetic_volume
                self.current_period_volume += synthetic_volume
            else:
                # Real volume
                if volume > self.last_cumulative_volume:
                    volume_change = volume - self.last_cumulative_volume
                    self.current_period_volume += volume_change
                    self.last_cumulative_volume = volume
            
            logger.info(f"Equity Full: LTP={ltp:.2f}, Volume={volume:,}, OI={oi:,}")
            
            return {
                'timestamp': datetime.now(IST),
                'packet_type': 'equity_full',
                'ltp': ltp,
                'open': open_price,
                'high': high,
                'low': low,
                'close': close,
                'volume': volume,
                'oi': oi,
                'ltt': last_traded_time,
                'exchange_time': exchange_time
            }
            
        except Exception as e:
            logger.error(f"Error parsing equity full packet: {e}")
            return None
    
    def _parse_market_depth_184(self, data: bytes) -> Optional[Dict]:
        """Parse 184-byte market depth packet."""
        if len(data) < 184:
            return None
        
        try:
            security_id = struct.unpack('<I', data[4:8])[0]
            
            if security_id != self.config.nifty_security_id:
                return None
            
            ltp = struct.unpack('<f', data[8:12])[0]
            if not np.isfinite(ltp):
                return None
                                        
            # Calculate synthetic volume
            synthetic_volume = self._calculate_synthetic_volume(ltp)
            self.current_period_volume += synthetic_volume
            
            logger.debug(f"Market Depth: LTP={ltp:.2f}, SyntheticVol={synthetic_volume}")
            
            return {
                'timestamp': datetime.now(IST),
                'packet_type': 'market_depth',
                'ltp': ltp,
                'volume': synthetic_volume
            }
            
        except Exception as e:
            logger.error(f"Error parsing market depth packet: {e}")
            return None
    
    async def _handle_text_message(self, message: str):
        """Handle text/JSON control messages from server."""
        try:
            data = json.loads(message)
            logger.debug(f"Text message received: {data}")
            
            if isinstance(data, dict):
                # Handle different response types
                if 'type' in data:
                    msg_type = data['type']
                    
                    if msg_type in ['success', 'subscription_success']:
                        logger.info(f"Server success: {data.get('message', 'Subscription confirmed')}")
                    elif msg_type == 'error':
                        logger.error(f"Server error: {data.get('message', 'Unknown error')}")
                    elif msg_type == 'heartbeat':
                        logger.debug("Heartbeat received")
                    else:
                        logger.debug(f"Server message type: {msg_type}")
                        
                elif 'ResponseCode' in data:
                    code = data['ResponseCode']
                    msg = data.get('ResponseMessage', '')
                    
                    if code == 200:
                        logger.info(f"Subscription successful: {msg}")
                    elif code == 401:
                        logger.critical(f"Authentication failed: {msg}")
                        self.authenticated = False
                    elif code >= 400:
                        logger.error(f"Error {code}: {msg}")
                        
                elif 'market_status' in data:
                    status = data.get('market_status')
                    logger.info(f"Market status: {status}")
                    
                else:
                    logger.debug(f"Unhandled message: {data}")
                    
        except json.JSONDecodeError:
            logger.debug(f"Non-JSON text message received")
        except Exception as e:
            logger.error(f"Error handling text message: {e}")
    


    # async def disconnect(self, stop_running: bool = True):
    #     """Gracefully disconnect from WebSocket.
    #     stop_running=True → full shutdown
    #     stop_running=False → internal reconnect (keep run_forever alive)
    #     """
    #     logger.info("Disconnecting from DhanHQ WebSocket")
    #     if stop_running:
    #         self.running = False
    #         logger.debug("Disconnect mode: full shutdown (running=False)")
    #     else:
    #         logger.debug("Disconnect mode: internal reconnect (running=True)")

    #     # Cancel boundary loop if running (enhanced with safer exception handling)
    #     try:
    #         if getattr(self, 'boundary_task', None) and not self.boundary_task.done():
    #             self.boundary_task.cancel()
    #             await self.boundary_task  # Await to ensure clean cancellation [[3]]
    #     except asyncio.CancelledError:
    #         logger.debug("Boundary loop task cancelled")
    #     except Exception as e:
    #         logger.debug(f"Boundary task cancel failed (ignored): {e}")

    #     # Cancel data-stall watchdog if running (enhanced similarly)
    #     try:
    #         if getattr(self, 'data_watchdog_task', None) and not self.data_watchdog_task.done():
    #             self.data_watchdog_task.cancel()
    #             await self.data_watchdog_task  # Await for proper cleanup [[6]]
    #     except asyncio.CancelledError:
    #         logger.debug("Data-stall watchdog cancelled")
    #     except Exception as e:
    #         logger.debug(f"Data watchdog cancel failed (ignored): {e}")

    #     if self.websocket:
    #         try:
    #             logger.info(f"[Subscribe] WebSocket state: open={self.websocket.open}, closed={self.websocket.closed}")
                
    #             logger.info(f"Final packet statistics: {self.packet_stats}")
    #             logger.info(f"Total ticks processed: {self.tick_count}")
    #             logger.info(f"Final synthetic volume: {self.current_period_volume:,}")
    #             await self.websocket.close()  # Ensure async close to avoid abnormal errors [[7]]
    #             logger.info("WebSocket disconnected successfully")
    #         except Exception as e:
    #             logger.error(f"Error during disconnect: {e}")

    #     self.authenticated = False
        
        
            

    async def disconnect(self, stop_running: bool = True):
        """Gracefully disconnect from WebSocket.
        stop_running=True → full shutdown
        stop_running=False → internal reconnect (keep run_forever alive)
        """
        logger.info("Disconnecting from DhanHQ WebSocket")
        if stop_running:
            self.running = False
            logger.debug("Disconnect mode: full shutdown (running=False)")
        else:
            logger.debug("Disconnect mode: internal reconnect (running=True)")

        # Cancel boundary loop if running (enhanced with safer exception handling)
        try:
            if getattr(self, 'boundary_task', None) and not self.boundary_task.done():
                self.boundary_task.cancel()
                await self.boundary_task  # Await to ensure clean cancellation [[3]]
        except asyncio.CancelledError:
            logger.debug("Boundary loop task cancelled")
        except Exception as e:
            logger.debug(f"Boundary task cancel failed (ignored): {e}")

        # Cancel data-stall watchdog if running (enhanced similarly)
        try:
            if getattr(self, 'data_watchdog_task', None) and not self.data_watchdog_task.done():
                self.data_watchdog_task.cancel()
                await self.data_watchdog_task  # Await for proper cleanup [[6]]
        except asyncio.CancelledError:
            logger.debug("Data-stall watchdog cancelled")
        except Exception as e:
            logger.debug(f"Data watchdog cancel failed (ignored): {e}")

        if self.websocket:
            try:
                # Safely log state without assuming attributes exist
                if hasattr(self.websocket, 'open') and hasattr(self.websocket, 'closed'):
                    logger.info(f"[Subscribe] WebSocket state: open={self.websocket.open}, closed={self.websocket.closed}")
                else:
                    logger.info("[Subscribe] WebSocket state: unknown (attributes missing)")
                
                logger.info(f"Final packet statistics: {self.packet_stats}")
                logger.info(f"Total ticks processed: {self.tick_count}")
                logger.info(f"Final synthetic volume: {self.current_period_volume:,}")
                await self.websocket.close()  # Ensure async close to avoid abnormal errors [[7]]
                logger.info("WebSocket disconnected successfully")
            except Exception as e:
                logger.error(f"Error during disconnect: {e}")

        self.authenticated = False


        