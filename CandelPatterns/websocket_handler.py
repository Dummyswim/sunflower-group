"""
DhanHQ WebSocket handler - DhanHQ v2 compliant, robust, and async.
Fixed for websockets v15.x compatibility and proper shutdown handling.
Enhanced with comprehensive logging for debugging.
"""
import asyncio
import json
import logging
import struct
import time
from datetime import datetime, timezone
from typing import Callable, Dict, Optional, Any, List
import websockets
from websockets.exceptions import WebSocketException, ConnectionClosed
from websockets.client import WebSocketClientProtocol

logger = logging.getLogger(__name__)

class DhanWebSocketHandler:
    """
    DhanHQ WebSocket handler with proper v2 API implementation and enhanced logging.
    Fixed for websockets v15.x compatibility.
    """
    def __init__(self, client_id: str, access_token: str):
        self.client_id = client_id
        self.access_token = access_token
        self.ws_url = (
            f"wss://api-feed.dhan.co?version=2"
            f"&token={access_token}"
            f"&clientId={client_id}"
            f"&authType=2"
        )
        self.websocket = None
        self.running = True
        self.authenticated = False
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 10
        self.reconnect_delay = 5
        self._binary_buffer = b''
        self._shutdown_event = asyncio.Event()
        
        # Statistics tracking
        self.stats = {
            'packets_received': 0,
            'packets_processed': 0,
            'ticks_parsed': 0,
            'heartbeats': 0,
            'errors': 0,
            'last_packet_time': None
        }

        # Callbacks
        self.on_tick = None
        self.on_connect = None
        self.on_disconnect = None
        self.on_error = None

        logger.info(f"[WS INIT] DhanWebSocketHandler initialized (v2)")
        logger.info(f"[WS INIT] Client ID: {client_id[:10]}...")
        logger.info(f"[WS INIT] WS URL: {self.ws_url.replace(access_token, 'TOKEN_HIDDEN')}")

    def _is_connection_open(self) -> bool:
        """Check if WebSocket connection is open (compatible with websockets v15.x)"""
        if not self.websocket:
            return False
        
        # For websockets v15.x, check if connection exists and can send
        try:
            # Check if the connection is in a state where we can send messages
            return self.websocket.state.name == 'OPEN'
        except:
            # Fallback: try to check if we can access the transport
            try:
                return self.websocket.transport is not None and not self.websocket.transport.is_closing()
            except:
                return False

    async def connect(self):
        """Establish WebSocket connection to DhanHQ."""
        try:
            logger.info(f"[WS CONNECT] Starting connection to DhanHQ WebSocket...")
            logger.debug(f"[WS CONNECT] Connection parameters: ping_interval=30, ping_timeout=10")
            
            self.websocket = await websockets.connect(
                self.ws_url,
                ping_interval=30,
                ping_timeout=10,
                close_timeout=10,
                max_size=10 * 1024 * 1024
            )
            
            self.running = True
            self.authenticated = True
            self.reconnect_attempts = 0
            
            logger.info(f"[WS CONNECT] ✅ WebSocket connected successfully")
            
            # Check connection state properly for v15.x
            is_open = self._is_connection_open()
            logger.info(f"[WS CONNECT] Connection state: open={is_open}")
            logger.debug(f"[WS CONNECT] WebSocket object: {self.websocket}")
            
            if self.on_connect:
                logger.debug("[WS CONNECT] Calling on_connect callback")
                await self.on_connect()
                
            return True
            
        except Exception as e:
            logger.error(f"[WS CONNECT] ❌ Connection failed: {type(e).__name__}: {e}")
            logger.debug(f"[WS CONNECT] Full error: {e}", exc_info=True)
            self.websocket = None
            self.authenticated = False
            return False

    async def subscribe(self, instruments: List[Dict]):
        """
        Subscribe to market data for instruments with detailed logging.
        """
        if not self.websocket or not self._is_connection_open():
            logger.error("[WS SUBSCRIBE] Cannot subscribe - WebSocket not connected or closed")
            return False

        try:
            logger.info(f"[WS SUBSCRIBE] Preparing subscription for {len(instruments)} instruments")
            
            # Log each instrument being subscribed
            for idx, inst in enumerate(instruments):
                logger.debug(f"[WS SUBSCRIBE] Instrument {idx+1}: ExchangeSegment={inst.get('exchangeSegment', 'N/A')}, SecurityId={inst.get('securityId', 'N/A')}")
            
            subscription_data = {
                "RequestCode": 15,  # 21 = Full Feed, 15 = Touchline
                "InstrumentCount": len(instruments),
                "InstrumentList": [
                    {
                        "ExchangeSegment": inst.get("exchangeSegment") or inst.get("ExchangeSegment"),
                        "SecurityId": str(inst.get("securityId") or inst.get("SecurityId"))
                    }
                    for inst in instruments
                ]
            }
            
            message = json.dumps(subscription_data)
            logger.info(f"[WS SUBSCRIBE] Sending subscription request with RequestCode=15 (Touchline)")
            logger.debug(f"[WS SUBSCRIBE] Full subscription message: {message}")
            
            await self.websocket.send(message)
            
            logger.info(f"[WS SUBSCRIBE] ✅ Subscription request sent successfully")
            logger.debug(f"[WS SUBSCRIBE] Waiting 500ms for subscription confirmation...")
            await asyncio.sleep(0.5)
            
            return True
            
        except Exception as e:
            logger.error(f"[WS SUBSCRIBE] ❌ Subscription failed: {type(e).__name__}: {e}")
            logger.debug(f"[WS SUBSCRIBE] Full error:", exc_info=True)
            return False

    async def run(self):
        """Main WebSocket event loop with comprehensive logging and proper shutdown handling."""
        logger.info("[WS RUN] Starting WebSocket event loop...")
        logger.debug(f"[WS RUN] Initial state: running={self.running}, authenticated={self.authenticated}")
        
        while self.running:
            try:
                # Check for shutdown signal
                if self._shutdown_event.is_set():
                    logger.info("[WS RUN] Shutdown signal received, stopping...")
                    break
                
                if not self.websocket or not self._is_connection_open():
                    logger.info(f"[WS RUN] WebSocket not connected")
                    logger.info(f"[WS RUN] Attempting connection (attempt {self.reconnect_attempts + 1})")
                    
                    connected = await self.connect()
                    
                    if not connected:
                        logger.error(f"[WS RUN] Connection failed (attempt {self.reconnect_attempts + 1}/{self.max_reconnect_attempts})")
                        await self._handle_reconnect()
                        continue
                    
                    logger.info("[WS RUN] Connection established, starting message loop")

                try:
                    logger.debug("[WS RUN] Entering message receive loop")
                    message_count = 0
                    
                    # Use asyncio timeout to allow periodic checks
                    while self.running and self._is_connection_open():
                        try:
                            # Wait for message with timeout to allow shutdown checks
                            message = await asyncio.wait_for(
                                self.websocket.recv(),
                                timeout=1.0  # 1 second timeout
                            )
                            
                            message_count += 1
                            self.stats['packets_received'] += 1
                            self.stats['last_packet_time'] = datetime.now()
                            
                            if isinstance(message, bytes):
                                logger.debug(f"[WS RUN] Received binary message #{message_count} ({len(message)} bytes)")
                                logger.debug(f"[WS RUN] Binary data hex: {message.hex()[:32]}...")
                                await self._process_binary_message(message)
                            else:
                                logger.debug(f"[WS RUN] Received text message #{message_count}")
                                await self._process_text_message(message)
                                
                            # Log stats periodically
                            if message_count % 100 == 0:
                                logger.info(f"[WS STATS] Messages: {message_count}, Ticks: {self.stats['ticks_parsed']}, Heartbeats: {self.stats['heartbeats']}")
                                
                        except asyncio.TimeoutError:
                            # Timeout is normal, just check if we should continue
                            if self._shutdown_event.is_set():
                                logger.info("[WS RUN] Shutdown requested during message loop")
                                break
                            continue
                        
                        except ConnectionClosed as e:
                            logger.warning(f"[WS RUN] Connection closed: code={e.code}, reason={e.reason}")
                            self.authenticated = False
                            await self._handle_disconnect()
                            break
                            
                except asyncio.CancelledError:
                    logger.info("[WS RUN] Message loop cancelled")
                    break
                    
                except Exception as e:
                    logger.error(f"[WS RUN] Error processing messages: {type(e).__name__}: {e}")
                    logger.debug("[WS RUN] Full error:", exc_info=True)
                    await self._handle_disconnect()
                    
            except asyncio.CancelledError:
                logger.info("[WS RUN] Run loop cancelled")
                break
                
            except Exception as e:
                logger.error(f"[WS RUN] Unexpected error in run loop: {type(e).__name__}: {e}")
                logger.debug("[WS RUN] Full error:", exc_info=True)
                
                if self.running:
                    await asyncio.sleep(1)
                
        logger.info("[WS RUN] Event loop ended")

    async def _process_text_message(self, message: str):
        """Process text/JSON messages from WebSocket with detailed logging."""
        try:
            logger.debug(f"[WS TEXT] Processing text message: {message[:100]}...")
            
            data = json.loads(message)
            logger.info(f"[WS TEXT] Received JSON message with keys: {list(data.keys())}")
            logger.debug(f"[WS TEXT] Full JSON data: {json.dumps(data, indent=2)}")
            
            # Check for response codes
            if "ResponseCode" in data:
                code = data["ResponseCode"]
                logger.info(f"[WS TEXT] Response Code: {code}")
                if code == 200:
                    logger.info("[WS TEXT] ✅ Subscription confirmed successfully")
                else:
                    logger.warning(f"[WS TEXT] ⚠️ Non-success response code: {code}")
            
            # Check for errors
            if "error" in data or "Error" in data:
                error_msg = data.get("error") or data.get("Error")
                logger.error(f"[WS TEXT] ❌ Error message received: {error_msg}")
                if self.on_error:
                    await self.on_error(data)
                    
        except json.JSONDecodeError as e:
            logger.debug(f"[WS TEXT] Non-JSON text message: {message[:100]}... (Error: {e})")
        except Exception as e:
            logger.error(f"[WS TEXT] Processing error: {type(e).__name__}: {e}")
            logger.debug("[WS TEXT] Full error:", exc_info=True)

    async def _process_binary_message(self, message: bytes):
        """Process binary messages with comprehensive packet analysis logging."""
        try:
            logger.debug(f"[WS BINARY] Processing binary message ({len(message)} bytes)")
            # logger.debug(f"[WS BINARY] Buffer before: {len(self._binary_buffer)} bytes")
            
            self._binary_buffer += message
            # logger.debug(f"[WS BINARY] Buffer after: {len(self._binary_buffer)} bytes")
            
            packets_in_message = 0
            
            while len(self._binary_buffer) > 0:
                if len(self._binary_buffer) < 1:
                    logger.debug("[WS BINARY] Buffer too small for packet type")
                    break
                    
                packet_type = self._binary_buffer[0]
                logger.debug(f"[WS BINARY] Packet type: {packet_type}")
                
                # DhanHQ v2 packet sizes with detailed logging
                if packet_type == 1:  # Heartbeat
                    packet_size = 1
                    packet_name = "HEARTBEAT"
                elif packet_type == 2:  # Index/Touchline tick (16 bytes)
                    packet_size = 16
                    packet_name = "INDEX_TICK"
                elif packet_type == 3:  # Full tick
                    packet_size = 44
                    packet_name = "FULL_TICK"
                elif packet_type == 6:  # Control/Status packet
                    packet_size = 16
                    packet_name = "CONTROL"
                else:
                    logger.warning(f"[WS BINARY] Unknown packet type: {packet_type}, skipping byte")
                    logger.debug(f"[WS BINARY] Next 16 bytes: {self._binary_buffer[:16].hex()}")
                    self._binary_buffer = self._binary_buffer[1:]
                    continue
                
                logger.debug(f"[WS BINARY] Packet identified: {packet_name} (size={packet_size})")
                
                if len(self._binary_buffer) < packet_size:
                    logger.debug(f"[WS BINARY] Insufficient buffer for {packet_name}: have {len(self._binary_buffer)}, need {packet_size}")
                    break
                
                # Extract complete packet
                packet_data = self._binary_buffer[:packet_size]
                self._binary_buffer = self._binary_buffer[packet_size:]
                packets_in_message += 1
                
                logger.debug(f"[WS BINARY] Extracted {packet_name} packet: {packet_data.hex()}")
                
                # Parse based on type
                if packet_type == 1:
                    self.stats['heartbeats'] += 1
                    logger.info(f"[WS HEARTBEAT] Heartbeat #{self.stats['heartbeats']} received")
                    
                elif packet_type == 2:
                    logger.debug(f"[WS BINARY] Parsing INDEX_TICK packet")
                    tick = self._parse_index_tick_v2(packet_data)
                    if tick:
                        self.stats['ticks_parsed'] += 1
                        # logger.info(f"[WS TICK] Index tick parsed: SecurityID={tick['security_id']}, LTP={tick['ltp']:.2f}, Time={tick['timestamp']}")
                        if self.on_tick:
                            await self.on_tick(tick)
                    else:
                        logger.warning("[WS BINARY] Failed to parse INDEX_TICK packet")
                        
                elif packet_type == 6:
                    logger.info("[WS CONTROL] Control packet received (type=6)")
                    logger.debug(f"[WS CONTROL] Control packet data: {packet_data.hex()}")
                
                self.stats['packets_processed'] += 1
                
            logger.debug(f"[WS BINARY] Processed {packets_in_message} packets from message")
            logger.debug(f"[WS BINARY] Remaining buffer: {len(self._binary_buffer)} bytes")
                    
        except Exception as e:
            self.stats['errors'] += 1
            logger.error(f"[WS BINARY] Processing error: {type(e).__name__}: {e}")
            logger.debug(f"[WS BINARY] Buffer state at error: {len(self._binary_buffer)} bytes")
            logger.debug("[WS BINARY] Full error:", exc_info=True)

    def _parse_index_tick_v2(self, data: bytes) -> Optional[Dict]:
        """Parse 16-byte index tick packet with detailed field logging."""
        if len(data) != 16:
            logger.error(f"[WS PARSE] Invalid INDEX_TICK size: {len(data)} bytes (expected 16)")
            return None
        
        try:
            logger.debug(f"[WS PARSE] Parsing INDEX_TICK: {data.hex()}")
            
            # Parse fields
            packet_type = data[0]
            security_id = struct.unpack('<I', data[4:8])[0]
            ltp = struct.unpack('<f', data[8:12])[0]
            timestamp = struct.unpack('<I', data[12:16])[0]
            
            logger.debug(f"[WS PARSE] Parsed fields:")
            logger.debug(f"  - Packet Type: {packet_type}")
            logger.debug(f"  - Security ID: {security_id}")
            logger.debug(f"  - LTP: {ltp:.2f}")
            logger.debug(f"  - Timestamp: {timestamp}")
            
            tick = {
                'security_id': security_id,
                'ltp': ltp,
                'timestamp': timestamp,
                'exchange_segment': 0,
                'volume': 0  # Add default volume since index doesn't have volume
            }
            
            logger.info(f"[WS PARSE] ✅ Successfully parsed tick: ID={security_id}, Price={ltp:.2f}")
            return tick
            
        except Exception as e:
            logger.error(f"[WS PARSE] Failed to parse INDEX_TICK: {type(e).__name__}: {e}")
            logger.debug(f"[WS PARSE] Raw data that failed: {data.hex()}")
            logger.debug("[WS PARSE] Full error:", exc_info=True)
            return None

    async def _handle_disconnect(self):
        """Handle WebSocket disconnection with detailed logging."""
        logger.warning("[WS DISCONNECT] Handling disconnection...")
        logger.info(f"[WS DISCONNECT] Session stats: {self.stats}")
        
        if self.on_disconnect:
            logger.debug("[WS DISCONNECT] Calling on_disconnect callback")
            await self.on_disconnect()
            
        self.websocket = None
        self.authenticated = False
        
        if self.running and not self._shutdown_event.is_set():
            logger.info("[WS DISCONNECT] Will attempt reconnection (still running)")
            await self._handle_reconnect()
        else:
            logger.info("[WS DISCONNECT] Not reconnecting (running=False or shutdown requested)")

    async def _handle_reconnect(self):
        """Handle reconnection with exponential backoff and detailed logging."""
        if self.reconnect_attempts >= self.max_reconnect_attempts:
            logger.error(f"[WS RECONNECT] Max reconnection attempts reached ({self.max_reconnect_attempts}). Stopping.")
            self.running = False
            return
            
        self.reconnect_attempts += 1
        delay = min(self.reconnect_delay * (2 ** (self.reconnect_attempts - 1)), 300)
        
        logger.info(f"[WS RECONNECT] Scheduling reconnection attempt {self.reconnect_attempts}/{self.max_reconnect_attempts}")
        logger.info(f"[WS RECONNECT] Waiting {delay} seconds before reconnecting...")
        
        # Use interruptible sleep
        try:
            await asyncio.wait_for(
                self._shutdown_event.wait(),
                timeout=delay
            )
            # If we get here, shutdown was requested
            logger.info("[WS RECONNECT] Shutdown requested during reconnect delay")
            return
        except asyncio.TimeoutError:
            # Normal timeout, proceed with reconnection
            pass
        
        if not self._shutdown_event.is_set():
            logger.info(f"[WS RECONNECT] Starting reconnection attempt {self.reconnect_attempts}")

    async def disconnect(self):
        """Gracefully disconnect WebSocket with status logging."""
        logger.info("[WS CLOSE] Initiating graceful disconnection...")
        logger.info(f"[WS CLOSE] Final stats: {self.stats}")
        
        # Signal shutdown
        self.running = False
        self.authenticated = False
        self._shutdown_event.set()
        
        if self.websocket and self._is_connection_open():
            try:
                logger.debug("[WS CLOSE] Closing WebSocket connection...")
                await asyncio.wait_for(self.websocket.close(), timeout=5.0)
                logger.info("[WS CLOSE] WebSocket connection closed")
            except asyncio.TimeoutError:
                logger.warning("[WS CLOSE] Close operation timed out")
            except Exception as e:
                logger.error(f"[WS CLOSE] Error during close: {e}")
            finally:
                self.websocket = None
        else:
            logger.debug("[WS CLOSE] WebSocket already closed or None")
            
        logger.info("[WS CLOSE] ✅ Disconnection complete")
