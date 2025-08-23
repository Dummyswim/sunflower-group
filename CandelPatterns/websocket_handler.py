"""
DhanHQ WebSocket handler with robust packet handling and reconnection.
Based on DhanHQ API v2 documentation.
"""
import asyncio
import json
import logging
import struct
import time
from datetime import datetime, timezone
from typing import Callable, Dict, Optional, Any, List
import websockets
from websockets.exceptions import WebSocketException
import base64

logger = logging.getLogger(__name__)

class DhanWebSocketHandler:
    """
    DhanHQ WebSocket handler with automatic reconnection and packet management.
    """
    
    def __init__(self, client_id: str, access_token: str):
        """Initialize WebSocket handler with DhanHQ credentials."""
        self.client_id = client_id
        self.access_token = access_token
        self.ws_url = "wss://api-feed.dhan.co"
        
        self.websocket = None
        self.running = False
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 10
        self.reconnect_delay = 5
        
        # Callbacks
        self.on_tick = None
        self.on_connect = None
        self.on_disconnect = None
        self.on_error = None
        
        # Buffer for incomplete packets
        self.packet_buffer = bytearray()
        self.max_buffer_size = 1024 * 1024  # 1MB max buffer
        
        logger.info("DhanWebSocketHandler initialized")
    
    async def connect(self):
        """Establish WebSocket connection to DhanHQ."""
        try:
            logger.info(f"Connecting to {self.ws_url}")
            
            # Build connection URL with auth
            auth_header = {
                "Authorization": f"Bearer {self.access_token}",
                "Client-Id": self.client_id
            }
            
            self.websocket = await websockets.connect(
                self.ws_url,
                extra_headers=auth_header,
                ping_interval=30,
                ping_timeout=10,
                close_timeout=10
            )
            
            self.running = True
            self.reconnect_attempts = 0
            
            logger.info("WebSocket connected successfully")
            
            if self.on_connect:
                await self.on_connect()
            
            return True
            
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            return False
    
    async def subscribe(self, instruments: List[Dict]):
        """
        Subscribe to market data for instruments.
        
        Args:
            instruments: List of dicts with 'securityId' and 'exchangeSegment'
        """
        if not self.websocket:
            logger.error("WebSocket not connected")
            return False
        
        try:
            subscription_data = {
                "RequestCode": 15,  # Subscribe request
                "InstrumentCount": len(instruments),
                "InstrumentList": instruments
            }
            
            message = json.dumps(subscription_data)
            await self.websocket.send(message)
            
            logger.info(f"Subscribed to {len(instruments)} instruments")
            return True
            
        except Exception as e:
            logger.error(f"Subscription failed: {e}")
            return False
    
    async def run(self):
        """Main WebSocket event loop."""
        while self.running:
            try:
                if not self.websocket:
                    if not await self.connect():
                        await self._handle_reconnect()
                        continue
                
                # Receive and process messages
                async for message in self.websocket:
                    await self._process_message(message)
                    
            except WebSocketException as e:
                logger.error(f"WebSocket error: {e}")
                await self._handle_disconnect()
                
            except Exception as e:
                logger.error(f"Unexpected error: {e}", exc_info=True)
                await asyncio.sleep(1)
    
    async def _process_message(self, message: bytes):
        """Process incoming WebSocket message with proper packet handling."""
        try:
            # Add to buffer
            self.packet_buffer.extend(message)
            
            # Check buffer size limit
            if len(self.packet_buffer) > self.max_buffer_size:
                logger.warning("Buffer overflow, clearing")
                self.packet_buffer.clear()
                return
            
            # Process complete packets
            while len(self.packet_buffer) >= 4:
                # Read packet length (first 4 bytes)
                packet_length = struct.unpack('<I', self.packet_buffer[:4])[0]
                
                # Check if we have complete packet
                if len(self.packet_buffer) < packet_length:
                    break  # Wait for more data
                
                # Extract packet
                packet = self.packet_buffer[:packet_length]
                self.packet_buffer = self.packet_buffer[packet_length:]
                
                # Parse packet
                await self._parse_packet(packet)
                
        except Exception as e:
            logger.error(f"Message processing error: {e}")
            self.packet_buffer.clear()
    
    async def _parse_packet(self, packet: bytes):
        """Parse DhanHQ data packet."""
        try:
            # Skip length prefix
            data = packet[4:]
            
            # Parse based on packet type
            packet_type = data[0] if data else 0
            
            if packet_type == 2:  # Tick data
                tick_data = self._parse_tick_data(data[1:])
                if tick_data and self.on_tick:
                    await self.on_tick(tick_data)
                    
            elif packet_type == 4:  # Order update
                logger.debug("Order update received")
                
            else:
                logger.debug(f"Unknown packet type: {packet_type}")
                
        except Exception as e:
            logger.error(f"Packet parsing error: {e}")
    
    def _parse_tick_data(self, data: bytes) -> Optional[Dict]:
        """Parse tick data from binary format."""
        try:
            # DhanHQ tick structure (adjust based on actual API)
            if len(data) < 44:  # Minimum tick size
                return None
            
            # Unpack tick data
            tick = {}
            offset = 0
            
            # Security ID (4 bytes)
            tick['security_id'] = struct.unpack('<I', data[offset:offset+4])[0]
            offset += 4
            
            # LTP (4 bytes, divide by 100 for actual price)
            tick['ltp'] = struct.unpack('<I', data[offset:offset+4])[0] / 100
            offset += 4
            
            # Timestamp (8 bytes)
            tick['timestamp'] = struct.unpack('<Q', data[offset:offset+8])[0]
            offset += 8
            
            # OHLC if available
            if len(data) >= offset + 16:
                tick['open'] = struct.unpack('<I', data[offset:offset+4])[0] / 100
                offset += 4
                tick['high'] = struct.unpack('<I', data[offset:offset+4])[0] / 100
                offset += 4
                tick['low'] = struct.unpack('<I', data[offset:offset+4])[0] / 100
                offset += 4
                tick['close'] = struct.unpack('<I', data[offset:offset+4])[0] / 100
                offset += 4
            
            # Volume if available
            if len(data) >= offset + 8:
                tick['volume'] = struct.unpack('<Q', data[offset:offset+8])[0]
            
            return tick
            
        except Exception as e:
            logger.error(f"Tick parsing error: {e}")
            return None
    
    async def _handle_disconnect(self):
        """Handle WebSocket disconnection."""
        logger.warning("WebSocket disconnected")
        
        if self.on_disconnect:
            await self.on_disconnect()
        
        self.websocket = None
        
        if self.running:
            await self._handle_reconnect()
    
    async def _handle_reconnect(self):
        """Handle reconnection with exponential backoff."""
        if self.reconnect_attempts >= self.max_reconnect_attempts:
            logger.error("Max reconnection attempts reached")
            self.running = False
            return
        
        self.reconnect_attempts += 1
        delay = min(self.reconnect_delay * (2 ** (self.reconnect_attempts - 1)), 300)
        
        logger.info(f"Reconnecting in {delay}s (attempt {self.reconnect_attempts})")
        await asyncio.sleep(delay)
    
    async def disconnect(self):
        """Gracefully disconnect WebSocket."""
        self.running = False
        
        if self.websocket:
            await self.websocket.close()
            self.websocket = None
        
        logger.info("WebSocket disconnected")
