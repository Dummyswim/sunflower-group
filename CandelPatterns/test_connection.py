"""
Test script to verify DhanHQ credentials and WebSocket connection.
Compatible with different websockets library versions.
"""
import asyncio
import base64
import logging
import json
import config

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

async def test_connection():
    """Test WebSocket connection with proper header handling."""
    try:
        # Decode credentials
        client_id = config.decode_b64(config.DHAN_CLIENT_ID_B64)
        access_token = config.decode_b64(config.DHAN_ACCESS_TOKEN_B64)
        
        logger.info(f"Client ID: {client_id[:10]}..." if len(client_id) > 10 else client_id)
        logger.info(f"Token length: {len(access_token)}")
        
        import websockets
        
        # Check websockets version
        ws_version = websockets.__version__
        logger.info(f"Websockets version: {ws_version}")
        
        ws_url = "wss://api-feed.dhan.co"
        
        # Method 1: Try with subprotocol and origin (DhanHQ v2 compatible)
        try:
            logger.info(f"Attempting connection to {ws_url}...")
            
            # Build auth headers in URL or as subprotocol
            auth_url = f"{ws_url}?access_token={access_token}&client_id={client_id}"
            
            async with websockets.connect(
                auth_url,
                ping_interval=30,
                ping_timeout=10
            ) as websocket:
                logger.info("✅ Connected successfully!")
                
                # Try to subscribe
                subscription = {
                    "RequestCode": 15,
                    "InstrumentCount": 1,
                    "InstrumentList": [{
                        "securityId": config.NIFTY_SECURITY_ID,
                        "exchangeSegment": config.NIFTY_EXCHANGE_SEGMENT
                    }]
                }
                
                await websocket.send(json.dumps(subscription))
                logger.info("Subscription message sent")
                
                # Wait for response
                response = await asyncio.wait_for(websocket.recv(), timeout=5)
                logger.info(f"Received response: {len(response)} bytes")
                
        except Exception as e:
            logger.warning(f"Method 1 failed: {e}")
            
            # Method 2: Try with authorization in first message
            logger.info("Trying alternative connection method...")
            
            async with websockets.connect(
                ws_url,
                ping_interval=30,
                ping_timeout=10
            ) as websocket:
                logger.info("✅ Connected (method 2)")
                
                # Send auth as first message
                auth_message = {
                    "RequestCode": 11,  # Auth request
                    "LoginType": "access_token",
                    "Token": access_token,
                    "ClientId": client_id
                }
                
                await websocket.send(json.dumps(auth_message))
                logger.info("Auth message sent")
                
                # Wait for auth response
                auth_response = await asyncio.wait_for(websocket.recv(), timeout=5)
                logger.info(f"Auth response: {auth_response[:100]}...")
                
                # Now try subscription
                subscription = {
                    "RequestCode": 15,
                    "InstrumentCount": 1,
                    "InstrumentList": [{
                        "securityId": config.NIFTY_SECURITY_ID,
                        "exchangeSegment": config.NIFTY_EXCHANGE_SEGMENT
                    }]
                }
                
                await websocket.send(json.dumps(subscription))
                logger.info("Subscription sent")
                
                response = await asyncio.wait_for(websocket.recv(), timeout=5)
                logger.info(f"Subscription response: {len(response)} bytes")
                
    except Exception as e:
        logger.error(f"Connection test failed: {e}", exc_info=True)

if __name__ == "__main__":
    asyncio.run(test_connection())
