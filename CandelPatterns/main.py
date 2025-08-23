"""
Main entry point for the optimized pattern recognition trading system.
Integrates WebSocket, enhanced patterns, and improved workflow.
"""
import asyncio
import logging
import signal
import sys
import time
from datetime import datetime, timezone
from typing import Optional

# Setup logging first
from logging_setup import setup_logging
setup_logging(console_level=logging.INFO)

logger = logging.getLogger(__name__)

# Import components
import config
from alert_system import AlertSystem
from websocket_handler import DhanWebSocketHandler
from enhanced_patterns import EnhancedPatternRecognition

class TradingBot:
    """
    Main trading bot orchestrator with WebSocket integration.
    """
    
    def __init__(self, mode: str = "live", replay_csv: Optional[str] = None):
        """Initialize trading bot."""
        self.mode = mode
        self.replay_csv = replay_csv
        
        # Decode credentials
        self.client_id = config.decode_b64(config.DHAN_CLIENT_ID_B64)
        self.access_token = config.decode_b64(config.DHAN_ACCESS_TOKEN_B64)
        
        # Initialize components
        self.alert_system = AlertSystem(mode=mode, replay_csv=replay_csv)
        self.enhanced_patterns = EnhancedPatternRecognition()
        
        # WebSocket handler (for live mode)
        self.ws_handler = None
        if mode == "live":
            self.ws_handler = DhanWebSocketHandler(
                self.client_id,
                self.access_token
            )
            self._setup_ws_callbacks()
        
        # Shutdown flag
        self.shutdown_flag = False
        
        logger.info(f"TradingBot initialized in {mode} mode")
    
    def _setup_ws_callbacks(self):
        """Setup WebSocket callbacks."""
        self.ws_handler.on_tick = self.on_tick_received
        self.ws_handler.on_connect = self.on_ws_connected
        self.ws_handler.on_disconnect = self.on_ws_disconnected
        self.ws_handler.on_error = self.on_ws_error
    
    async def on_tick_received(self, tick_data: dict):
        """Handle incoming tick data."""
        try:
            # Extract required fields
            timestamp = datetime.fromtimestamp(
                tick_data.get('timestamp', time.time()),
                tz=timezone.utc
            )
            price = tick_data.get('ltp', 0)
            volume = tick_data.get('volume', 0)
            
            # Process tick
            new_candle = self.alert_system.process_tick(timestamp, price, volume)
            
            if new_candle:
                logger.debug(f"New candle formed at {timestamp}")
            
        except Exception as e:
            logger.error(f"Tick processing error: {e}")
    
    async def on_ws_connected(self):
        """Handle WebSocket connection."""
        logger.info("WebSocket connected, subscribing to instruments")
        
        # Subscribe to NIFTY
        instruments = [{
            'securityId': config.NIFTY_SECURITY_ID,
            'exchangeSegment': config.NIFTY_EXCHANGE_SEGMENT
        }]
        
        await self.ws_handler.subscribe(instruments)
    
    async def on_ws_disconnected(self):
        """Handle WebSocket disconnection."""
        logger.warning("WebSocket disconnected")
    
    async def on_ws_error(self, error):
        """Handle WebSocket error."""
        logger.error(f"WebSocket error: {error}")
    
    async def run_live(self):
        """Run bot in live mode with WebSocket."""
        logger.info("Starting live trading mode")
        
        try:
            # Start WebSocket connection
            await self.ws_handler.run()
            
        except KeyboardInterrupt:
            logger.info("Received interrupt signal")
        except Exception as e:
            logger.error(f"Live mode error: {e}", exc_info=True)
        finally:
            await self.shutdown()
    
    def run_replay(self):
        """Run bot in replay mode."""
        logger.info("Starting replay mode")
        
        try:
            self.alert_system.run_replay(speed=0.0)
        except KeyboardInterrupt:
            logger.info("Received interrupt signal")
        except Exception as e:
            logger.error(f"Replay mode error: {e}", exc_info=True)
        finally:
            self.shutdown_sync()
    
    async def shutdown(self):
        """Async shutdown procedure."""
        if self.shutdown_flag:
            return
        
        self.shutdown_flag = True
        logger.info("Initiating shutdown")
        
        # Disconnect WebSocket
        if self.ws_handler:
            await self.ws_handler.disconnect()
        
        # Shutdown alert system
        self.alert_system.shutdown()
        
        # Print final statistics
        stats = self.alert_system.get_statistics()
        logger.info("="*50)
        logger.info("SESSION STATISTICS")
        logger.info("="*50)
        for key, value in stats.items():
            if not isinstance(value, dict):
                logger.info(f"{key}: {value}")
        
        logger.info("Shutdown complete")
    
    def shutdown_sync(self):
        """Synchronous shutdown for non-async context."""
        if self.shutdown_flag:
            return
        
        self.shutdown_flag = True
        self.alert_system.shutdown()
        logger.info("Shutdown complete")

def signal_handler(signum, frame):
    """Handle shutdown signals."""
    logger.info(f"Received signal {signum}")
    sys.exit(0)

def main():
    """Main entry point."""
    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser(description="Pattern Recognition Trading Bot")
    parser.add_argument(
        "--mode",
        choices=["live", "replay"],
        default="live",
        help="Operating mode"
    )
    parser.add_argument(
        "--csv",
        type=str,
        help="CSV file path for replay mode"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    
    args = parser.parse_args()
    
    # Adjust logging level
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate arguments
    if args.mode == "replay" and not args.csv:
        logger.error("CSV file required for replay mode")
        sys.exit(1)
    
    # Create and run bot
    bot = TradingBot(mode=args.mode, replay_csv=args.csv)
    
    if args.mode == "live":
        # Run async event loop
        try:
            asyncio.run(bot.run_live())
        except KeyboardInterrupt:
            pass
    else:
        # Run synchronous replay
        bot.run_replay()

if __name__ == "__main__":
    main()
