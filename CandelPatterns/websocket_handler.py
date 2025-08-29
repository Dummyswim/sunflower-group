"""
DhanHQ WebSocket handler - DhanHQ v2 compliant, robust, and async.
Fixed packet parsing for correct NIFTY price extraction.
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

logger = logging.getLogger(__name__)

class DhanWebSocketHandler:
    """
    DhanHQ WebSocket handler with proper v2 API implementation.
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

        # Callbacks
        self.on_tick = None
        self.on_connect = None
        self.on_disconnect = None
        self.on_error = None

        logger.info("[WS] DhanWebSocketHandler initialized (v2)")

    async def connect(self):
        """Establish WebSocket connection to DhanHQ."""
        try:
            logger.info(f"[WS] Connecting to DhanHQ WebSocket...")
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
            logger.info("[WS] ✅ WebSocket connected successfully to DhanHQ")
            if self.on_connect:
                await self.on_connect()
            return True
        except Exception as e:
            logger.error(f"[WS] Connection failed: {e}")
            self.websocket = None
            self.authenticated = False
            return False

    async def subscribe(self, instruments: List[Dict]):
        """
        Subscribe to market data for instruments.
        """
        if not self.websocket:
            logger.error("[WS] Cannot subscribe - not connected")
            return False

        try:
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
            logger.info(f"[WS][DEBUG] Sending subscription: {message}")
            await self.websocket.send(message)
            logger.info(f"[WS] ✅ Subscription request sent: {subscription_data}")
            await asyncio.sleep(0.5)
            return True
        except Exception as e:
            logger.error(f"[WS] Subscription failed: {e}", exc_info=True)
            return False

    async def run(self):
        """Main WebSocket event loop."""
        logger.info("[WS] Starting WebSocket event loop...")
        while self.running:
            try:
                if not self.websocket or getattr(self.websocket, "closed", True):
                    logger.info("[WS] WebSocket not connected, attempting connection...")
                    connected = await self.connect()
                    if not connected:
                        logger.error(f"[WS] Connection failed (attempt {self.reconnect_attempts + 1}/{self.max_reconnect_attempts})")
                        await self._handle_reconnect()
                        continue

                try:
                    async for message in self.websocket:
                        if isinstance(message, bytes):
                            await self._process_binary_message(message)
                        else:
                            await self._process_text_message(message)
                except ConnectionClosed as e:
                    logger.warning(f"[WS] Connection closed: {e}")
                    self.authenticated = False
                    await self._handle_disconnect()
                except Exception as e:
                    logger.error(f"[WS] Error processing messages: {e}")
                    await self._handle_disconnect()
            except Exception as e:
                logger.error(f"[WS] Unexpected error in run loop: {e}", exc_info=True)
                await asyncio.sleep(1)
        logger.info("[WS] Event loop ended")

    async def _process_text_message(self, message: str):
        """Process text/JSON messages from WebSocket."""
        try:
            data = json.loads(message)
            logger.debug(f"[WS] JSON message received: {data}")
            if "ResponseCode" in data and data["ResponseCode"] == 200:
                logger.info("[WS] ✅ Subscription confirmed")
            if "error" in data or "Error" in data:
                logger.error(f"[WS] Error message: {data}")
                if self.on_error:
                    await self.on_error(data)
        except json.JSONDecodeError:
            logger.debug(f"[WS] Non-JSON text message: {message[:100]}")
        except Exception as e:
            logger.error(f"[WS] Text message processing error: {e}")

    async def _process_binary_message(self, message: bytes):
        """Process binary messages with DhanHQ v2 fixed packet sizes."""
        try:
            self._binary_buffer += message
            
            while len(self._binary_buffer) > 0:
                if len(self._binary_buffer) < 1:
                    break
                    
                packet_type = self._binary_buffer[0]
                
                # DhanHQ v2 packet sizes
                if packet_type == 1:  # Heartbeat
                    packet_size = 1
                elif packet_type == 2:  # Index/Touchline tick (16 bytes)
                    packet_size = 16
                elif packet_type == 3:  # Full tick
                    packet_size = 44
                elif packet_type == 6:  # Control/Status packet
                    packet_size = 16
                else:
                    logger.warning(f"[WS] Unknown packet type: {packet_type}")
                    self._binary_buffer = self._binary_buffer[1:]
                    continue
                
                if len(self._binary_buffer) < packet_size:
                    break
                
                # Extract complete packet
                packet_data = self._binary_buffer[:packet_size]
                self._binary_buffer = self._binary_buffer[packet_size:]
                
                # Parse based on type
                if packet_type == 1:
                    logger.debug("[WS] Heartbeat received")
                elif packet_type == 2:
                    tick = self._parse_index_tick_v2(packet_data)
                    if tick and self.on_tick:
                        await self.on_tick(tick)
                elif packet_type == 6:
                    logger.debug("[WS] Control packet received (ignored)")
                    
        except Exception as e:
            logger.error(f"[WS] Binary message processing error: {e}")

    def _parse_index_tick_v2(self, data: bytes) -> Optional[Dict]:
        """
        Parse 16-byte index tick packet.
        CORRECTED: The price is stored as little-endian FLOAT directly!
        """
        if len(data) != 16:
            return None
        
        try:
            # Byte 0: packet type (already processed)
            # Bytes 1-3: padding/flags
            # Bytes 4-7: Security ID (little-endian integer)
            security_id = struct.unpack('<I', data[4:8])[0]
            
            # Bytes 8-11: LTP as little-endian FLOAT (not integer!)
            ltp = struct.unpack('<f', data[8:12])[0]  # <-- THIS IS THE FIX!
            # No division by 100 needed!
            
            # Bytes 12-15: Timestamp (little-endian integer)
            timestamp = struct.unpack('<I', data[12:16])[0]
            
            tick = {
                'security_id': security_id,
                'ltp': ltp,
                'timestamp': timestamp,
                'exchange_segment': 0  # IDX_I
            }
            
            logger.debug(f"[WS] Index tick: ID={security_id} @ {ltp:.2f}")
            return tick
            
        except Exception as e:
            logger.error(f"[WS] Index tick parsing error: {e}")
            return None


    async def _handle_disconnect(self):
        """Handle WebSocket disconnection."""
        logger.warning("[WS] Handling disconnect...")
        if self.on_disconnect:
            await self.on_disconnect()
        self.websocket = None
        self.authenticated = False
        if self.running:
            await self._handle_reconnect()

    async def _handle_reconnect(self):
        """Handle reconnection with exponential backoff."""
        if self.reconnect_attempts >= self.max_reconnect_attempts:
            logger.error("[WS] Max reconnection attempts reached. Stopping.")
            self.running = False
            return
        self.reconnect_attempts += 1
        delay = min(self.reconnect_delay * (2 ** (self.reconnect_attempts - 1)), 300)
        logger.info(f"[WS] Reconnecting in {delay}s (attempt {self.reconnect_attempts}/{self.max_reconnect_attempts})")
        await asyncio.sleep(delay)

    async def disconnect(self):
        """Gracefully disconnect WebSocket."""
        logger.info("[WS] Disconnecting...")
        self.running = False
        self.authenticated = False
        if self.websocket and not getattr(self.websocket, "closed", True):
            await self.websocket.close()
            self.websocket = None
        logger.info("[WS] Disconnected.")
