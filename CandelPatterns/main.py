"""
Main entry point for the pattern recognition trading system.
Enhanced with proper async handling and signal quality metrics.
"""
import asyncio
import logging
import signal
import sys
import time
from datetime import datetime, timezone
from typing import Optional
import argparse

# Setup logging first
from logging_setup import setup_logging
setup_logging(console_level=logging.INFO)

logger = logging.getLogger(__name__)

# Import components
import config
from alert_system import AlertSystem
from signal_analyzer import SignalQualityAnalyzer
from websocket_handler import DhanWebSocketHandler
from enhanced_patterns import EnhancedPatternRecognition

class TradingBot:
    """
    Main trading bot orchestrator with enhanced signal quality analysis.
    """
    
    def __init__(self, mode: str = "live", replay_csv: Optional[str] = None):
        """Initialize trading bot with signal quality analyzer."""
        self.mode = mode
        self.replay_csv = replay_csv
        
        # Decode credentials
        try:
            self.client_id = config.decode_b64(config.DHAN_CLIENT_ID_B64)
            self.access_token = config.decode_b64(config.DHAN_ACCESS_TOKEN_B64)
            logger.info(f"Credentials decoded - {self.access_token}")
            logger.info(f"{self.client_id}")
            logger.info(f"Credentials decoded - Client ID length: {len(self.client_id)}")
        except Exception as e:
            logger.error(f"Failed to decode credentials: {e}")
            self.client_id = ""
            self.access_token = ""
        
        # Initialize components
        self.alert_system = AlertSystem(mode=mode, replay_csv=replay_csv)
        self.signal_analyzer = SignalQualityAnalyzer()
        self.enhanced_patterns = EnhancedPatternRecognition()
        
        # WebSocket handler (for live mode)
        self.ws_handler = None
        if mode == "live":
            if self.client_id and self.access_token:
                self.ws_handler = DhanWebSocketHandler(
                    self.client_id,
                    self.access_token
                )
                self._setup_ws_callbacks()
                logger.info("WebSocket handler created successfully")
            else:
                logger.error("Cannot create WebSocket handler - missing credentials")
        
        # Shutdown flag
        self.shutdown_flag = False
        
        # Performance tracking
        self.session_stats = {
            'start_time': datetime.now(),
            'ticks_processed': 0,
            'candles_formed': 0,
            'patterns_detected': 0,
            'alerts_sent': 0
        }
        
        logger.info(f"[OK] TradingBot initialized in {mode} mode")
    
    def _setup_ws_callbacks(self):
        """Setup WebSocket event callbacks."""
        self.ws_handler.on_tick = self.on_tick_received
        self.ws_handler.on_connect = self.on_ws_connected
        self.ws_handler.on_disconnect = self.on_ws_disconnected
        self.ws_handler.on_error = self.on_ws_error
        logger.debug("WebSocket callbacks configured")
    
    async def on_tick_received(self, tick_data: dict):
        """Handle incoming tick data with enhanced processing."""
        try:
            self.session_stats['ticks_processed'] += 1
            
            # Extract and validate tick data
            timestamp = datetime.fromtimestamp(
                tick_data.get('timestamp', time.time()),
                tz=timezone.utc
            )
            price = tick_data.get('ltp', 0)
            volume = tick_data.get('volume', 0)
            
            # Validate price
            if price <= 0:
                logger.warning(f"Invalid price received: {price}")
                return
            
            # Process tick through alert system
            new_candle = self.alert_system.process_tick(timestamp, price, volume)
            
            if new_candle:
                self.session_stats['candles_formed'] += 1
                logger.debug(f"New candle #{self.session_stats['candles_formed']} at {timestamp}")
            
            # Log progress every 100 ticks
            if self.session_stats['ticks_processed'] % 100 == 0:
                logger.info(f"Progress: {self.session_stats['ticks_processed']} ticks, "
                          f"{self.session_stats['candles_formed']} candles")
            
        except Exception as e:
            logger.error(f"Tick processing error: {e}", exc_info=True)
    
    async def on_ws_connected(self):
        """Handle successful WebSocket connection."""
        logger.info("[CONNECTED] WebSocket connected successfully")
        
        # Subscribe to NIFTY with validation
        instruments = [{
            'securityId': config.NIFTY_SECURITY_ID,
            'exchangeSegment': config.NIFTY_EXCHANGE_SEGMENT
        }]
        
        success = await self.ws_handler.subscribe(instruments)
        if success:
            logger.info(f"[OK] Subscribed to {len(instruments)} instruments")
        else:
            logger.error("[ERROR] Subscription failed")
    
    async def on_ws_disconnected(self):
        """Handle WebSocket disconnection."""
        logger.warning("[DISCONNECTED] WebSocket disconnected")
        
        # Log session stats on disconnect
        self._log_session_stats()
    
    async def on_ws_error(self, error):
        """Handle WebSocket errors."""
        logger.error(f"[WS_ERROR] WebSocket error: {error}")
    
    async def run_live(self):
        """Run bot in live mode with enhanced monitoring."""
        logger.info("[START] Starting live trading mode")
        
        if not self.ws_handler:
            logger.error("[ERROR] WebSocket handler not initialized - check credentials")
            return
        
        try:
            logger.info("[INFO] Attempting to connect to WebSocket...")
            # Start WebSocket connection
            await self.ws_handler.run()
            
        except KeyboardInterrupt:
            logger.info("[STOP] Received interrupt signal")
        except Exception as e:
            logger.error(f"[ERROR] Live mode error: {e}", exc_info=True)
        finally:
            await self.shutdown()
    
    def run_replay(self):
        """Run bot in replay mode with comprehensive analysis."""
        logger.info("[START] Starting replay mode")
        
        try:
            # Run replay through alert system
            self.alert_system.run_replay(speed=0.0)
            
            # Get signal quality analysis
            signal_performance = self.signal_analyzer.get_performance_summary()
            
            # Display enhanced results
            self._display_replay_results(signal_performance)
            
        except KeyboardInterrupt:
            logger.info("[STOP] Received interrupt signal")
        except Exception as e:
            logger.error(f"[ERROR] Replay mode error: {e}", exc_info=True)
        finally:
            self.shutdown_sync()
    
    def _display_replay_results(self, signal_performance: dict):
        """Display comprehensive replay results with signal quality metrics."""
        logger.info("=" * 60)
        logger.info("[COMPLETE] REPLAY ANALYSIS COMPLETE")
        logger.info("=" * 60)
        
        # Basic statistics
        stats = self.alert_system.get_statistics()
        logger.info(f"Mode: {stats['mode']}")
        logger.info(f"Duration: {stats['session_duration']}")
        logger.info(f"Candles Processed: {stats['candles_processed']}")
        logger.info(f"Patterns Detected: {stats['total_patterns_detected']}")
        logger.info(f"Alerts Sent: {stats['total_alerts_sent']}")
        
        # Prediction accuracy
        logger.info(f"\n[STATS] Prediction Performance:")
        logger.info(f"  Overall Accuracy: {stats['prediction_accuracy']:.1%}")
        logger.info(f"  Win Rate: {stats['win_rate']:.1%}")
        
        # Signal quality metrics
        logger.info(f"\n[SIGNAL] Signal Quality Metrics:")
        logger.info(f"  Total Signals: {signal_performance['total_signals']}")
        logger.info(f"  Success Rate: {signal_performance['success_rate']:.1%}")
        logger.info(f"  Avg Duration: {signal_performance['avg_duration']:.1f} minutes")
        logger.info(f"  Active Signals: {signal_performance['active_signals']}")
        
        # Pattern durations
        if signal_performance.get('pattern_durations'):
            logger.info(f"\n[PATTERNS] Average Pattern Durations:")
            for pattern, duration in sorted(signal_performance['pattern_durations'].items())[:10]:
                if duration > 0:
                    logger.info(f"  {pattern}: {duration:.1f} minutes")
        
        logger.info("=" * 60)
    
    def _log_session_stats(self):
        """Log current session statistics."""
        duration = (datetime.now() - self.session_stats['start_time']).total_seconds() / 60
        
        logger.info(f"\n[STATS] Session Statistics:")
        logger.info(f"  Duration: {duration:.1f} minutes")
        logger.info(f"  Ticks: {self.session_stats['ticks_processed']}")
        logger.info(f"  Candles: {self.session_stats['candles_formed']}")
        logger.info(f"  Patterns: {self.session_stats['patterns_detected']}")
        logger.info(f"  Alerts: {self.session_stats['alerts_sent']}")
    
    async def shutdown(self):
        """Async shutdown procedure with cleanup."""
        if self.shutdown_flag:
            return
        
        self.shutdown_flag = True
        logger.info("[SHUTDOWN] Initiating graceful shutdown...")
        
        # Disconnect WebSocket
        if self.ws_handler:
            await self.ws_handler.disconnect()
        
        # Shutdown alert system
        self.alert_system.shutdown()
        
        # Save signal quality data
        signal_performance = self.signal_analyzer.get_performance_summary()
        
        # Print final statistics
        self._print_final_stats(signal_performance)
        
        logger.info("[OK] Shutdown complete")
    
    def shutdown_sync(self):
        """Synchronous shutdown for non-async context."""
        if self.shutdown_flag:
            return
        
        self.shutdown_flag = True
        logger.info("[SHUTDOWN] Shutting down...")
        
        self.alert_system.shutdown()
        self._log_session_stats()
        
        logger.info("[OK] Shutdown complete")
    
    def _print_final_stats(self, signal_performance: dict):
        """Print comprehensive final statistics."""
        stats = self.alert_system.get_statistics()
        
        logger.info("=" * 60)
        logger.info("[REPORT] FINAL SESSION REPORT")
        logger.info("=" * 60)
        
        # Session info
        logger.info(f"Session Duration: {stats['session_duration']}")
        logger.info(f"Total Patterns: {stats['total_patterns_detected']}")
        logger.info(f"Total Alerts: {stats['total_alerts_sent']}")
        
        # Performance
        logger.info(f"\nPrediction Accuracy: {stats['prediction_accuracy']:.1%}")
        logger.info(f"Win Rate: {stats['win_rate']:.1%}")
        
        # Signal quality
        logger.info(f"\nSignal Quality:")
        logger.info(f"  Success Rate: {signal_performance['success_rate']:.1%}")
        logger.info(f"  Avg Duration: {signal_performance['avg_duration']:.1f} min")
        
        logger.info("=" * 60)

def signal_handler(signum, frame):
    """Handle system signals for graceful shutdown."""
    logger.info(f"[SIGNAL] Received signal {signum}")
    sys.exit(0)

def validate_config():
    """Validate configuration before starting."""
    issues = []
    
    # Check credentials
    if not config.DHAN_ACCESS_TOKEN_B64:
        issues.append("DHAN_ACCESS_TOKEN_B64 not configured")
    if not config.DHAN_CLIENT_ID_B64:
        issues.append("DHAN_CLIENT_ID_B64 not configured")
    
    # Check Telegram (optional)
    if not config.TELEGRAM_BOT_TOKEN_B64:
        logger.warning("Telegram bot token not configured (alerts disabled)")
    if not config.TELEGRAM_CHAT_ID:
        logger.warning("Telegram chat ID not configured (alerts disabled)")
    
    if issues:
        logger.error("Configuration issues found:")
        for issue in issues:
            logger.error(f"  - {issue}")
        return False
    
    logger.info("[OK] Configuration validated successfully")
    return True

def main():
    """Main entry point with enhanced argument parsing."""
    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Advanced Pattern Recognition Trading Bot with Signal Quality Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Live trading:
    python main.py --mode live
    
  Backtesting with CSV:
    python main.py --mode replay --csv data/nifty_2024.csv
    
  Debug mode:
    python main.py --mode live --debug
        """
    )
    
    parser.add_argument(
        "--mode",
        choices=["live", "replay"],
        default="live",
        help="Operating mode (default: live)"
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
    parser.add_argument(
        "--no-telegram",
        action="store_true",
        help="Disable Telegram alerts"
    )
    
    args = parser.parse_args()
    
    # Adjust logging level
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Debug mode enabled")
    
    # Disable Telegram if requested
    if args.no_telegram:
        config.TELEGRAM_BOT_TOKEN_B64 = ""
        config.TELEGRAM_CHAT_ID = ""
        logger.info("Telegram alerts disabled")
    
    # Validate arguments
    if args.mode == "replay" and not args.csv:
        logger.error("[ERROR] CSV file required for replay mode")
        parser.print_help()
        sys.exit(1)
    
    # Validate configuration (skip for replay mode)
    if args.mode == "live" and not validate_config():
        logger.error("[ERROR] Configuration validation failed")
        sys.exit(1)
    
    # Display startup banner
    logger.info("=" * 60)
    logger.info("[START] ADVANCED PATTERN RECOGNITION TRADING SYSTEM")
    logger.info("=" * 60)
    logger.info(f"Mode: {args.mode.upper()}")
    if args.csv:
        logger.info(f"CSV: {args.csv}")
    logger.info("=" * 60)
    
    # Create and run bot
    bot = TradingBot(mode=args.mode, replay_csv=args.csv)
    
    if args.mode == "live":
        # Run async event loop
        try:
            logger.info("[INFO] Starting async event loop...")
            asyncio.run(bot.run_live())
        except KeyboardInterrupt:
            logger.info("[STOP] Interrupted by user")
        except Exception as e:
            logger.error(f"[FATAL] Unhandled exception: {e}", exc_info=True)
    else:
        # Run synchronous replay
        bot.run_replay()

if __name__ == "__main__":
    main()
