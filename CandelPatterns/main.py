"""
Main entry point for the pattern recognition trading system.
Enhanced with proper async handling, signal quality metrics, and comprehensive validation.
"""
import asyncio
import logging
import signal
import sys
import time
from datetime import datetime, timezone
from typing import Optional
import argparse
import traceback

# Setup logging first
from logging_setup import setup_logging
setup_logging(console_level=logging.INFO)

logger = logging.getLogger(__name__)

# Import components with proper error handling
try:
    import config
    from alert_system import AlertSystem
    from signal_analyzer import SignalQualityAnalyzer
    from websocket_handler import DhanWebSocketHandler
    from enhanced_patterns import EnhancedPatternRecognition
except ImportError as e:
    logger.critical(f"Failed to import required modules: {e}")
    sys.exit(1)


class TradingBot:
    """
    Main trading bot orchestrator with enhanced signal quality analysis.
    """
    
    def __init__(self, mode: str = "live", replay_csv: Optional[str] = None):
        """Initialize trading bot with signal quality analyzer."""
        logger.info(f"Initializing TradingBot in {mode} mode")
        
        # Validate mode parameter
        if mode not in ["live", "replay"]:
            raise ValueError(f"Invalid mode: {mode}. Must be 'live' or 'replay'")
        
        self.mode = mode
        self.replay_csv = replay_csv
        
        # Initialize credentials with proper error handling
        try:
            self.client_id = config.decode_b64(config.DHAN_CLIENT_ID_B64)
            self.access_token = config.decode_b64(config.DHAN_ACCESS_TOKEN_B64)
            
            # Validate decoded credentials
            if not self.client_id or not self.access_token:
                raise ValueError("Empty credentials after decoding")
                
            logger.info(f"Credentials decoded successfully - Client ID length: {len(self.client_id)}")
            
        except Exception as e:
            logger.error(f"Failed to decode credentials: {e}")
            self.client_id = ""
            self.access_token = ""
            
            # Only fail if in live mode
            if mode == "live":
                logger.critical("Cannot proceed in live mode without valid credentials")
                raise
        
        # Initialize components with error handling
        try:
            self.alert_system = AlertSystem(mode=mode, replay_csv=replay_csv)
            self.signal_analyzer = SignalQualityAnalyzer()
            self.enhanced_patterns = EnhancedPatternRecognition()
            logger.info("Core components initialized successfully")
            
        except Exception as e:
            logger.critical(f"Failed to initialize core components: {e}")
            raise
        
        # WebSocket handler (for live mode)
        self.ws_handler = None
        if mode == "live":
            if self.client_id and self.access_token:
                try:
                    self.ws_handler = DhanWebSocketHandler(
                        self.client_id,
                        self.access_token
                    )
                    self._setup_ws_callbacks()
                    logger.info("WebSocket handler created successfully")
                except Exception as e:
                    logger.error(f"Failed to create WebSocket handler: {e}")
                    raise
            else:
                logger.error("Cannot create WebSocket handler - missing credentials")
                raise ValueError("WebSocket requires valid credentials")
        
        # Shutdown flag and lock for thread safety
        self.shutdown_flag = False
        self._shutdown_lock = asyncio.Lock() if mode == "live" else None
        
        # Performance tracking
        self.session_stats = {
            'start_time': datetime.now(),
            'ticks_processed': 0,
            'candles_formed': 0,
            'patterns_detected': 0,
            'alerts_sent': 0,
            'errors_count': 0,
            'last_tick_time': None
        }
        
        logger.info(f"TradingBot initialization complete in {mode} mode")
    
    def _setup_ws_callbacks(self):
        """Setup WebSocket event callbacks with validation."""
        if not self.ws_handler:
            raise RuntimeError("WebSocket handler not initialized")
            
        self.ws_handler.on_tick = self.on_tick
        self.ws_handler.on_connect = self.on_ws_connected
        self.ws_handler.on_disconnect = self.on_ws_disconnected
        self.ws_handler.on_error = self.on_ws_error
        logger.debug("WebSocket callbacks configured")

    async def on_tick(self, tick_data: dict):
        """Process incoming tick with volume simulation for indices."""
        try:
            timestamp = datetime.now(timezone.utc)
            price = tick_data.get('ltp', 0)
            
            # For index data without volume, simulate based on price movement
            volume = tick_data.get('volume', 0)
            if volume == 0 and hasattr(self, 'last_price'):
                # Simulate volume based on price change
                price_change = abs(price - self.last_price) / self.last_price if self.last_price else 0
                volume = int(1000 * (1 + price_change * 100))  # Synthetic volume
            
            self.last_price = price
            
            # Process tick
            new_candle = self.alert_system.process_tick(timestamp, price, volume)
            
            if new_candle:
                logger.debug(f"New candle #{self.session_stats['candles_formed']} at {timestamp}")
                logger.info(f"New candle #{self.session_stats['candles_formed']} at {timestamp}")
                
        except Exception as e:
            logger.error(f"Tick processing error: {e}")
    
    def _validate_price(self, price: float) -> bool:
        """Validate price data with comprehensive checks."""
        if price <= 0:
            logger.warning(f"Invalid price: {price} <= 0")
            return False
        
        # Check against configured bounds
        if hasattr(config, 'PRICE_SANITY_MIN') and price < config.PRICE_SANITY_MIN:
            logger.warning(f"Price {price} below minimum threshold {config.PRICE_SANITY_MIN}")
            return False
            
        if hasattr(config, 'PRICE_SANITY_MAX') and price > config.PRICE_SANITY_MAX:
            logger.warning(f"Price {price} above maximum threshold {config.PRICE_SANITY_MAX}")
            return False
        
        return True
    
    def _check_data_staleness(self):
        """Check if data stream has become stale."""
        if self.session_stats['last_tick_time']:
            time_since_last = (datetime.now() - self.session_stats['last_tick_time']).seconds
            if time_since_last > 60:  # 1 minute threshold
                logger.warning(f"Data stream potentially stale - {time_since_last}s since last tick")
    
    def _log_progress(self):
        """Log current progress with detailed statistics."""
        error_rate = (self.session_stats['errors_count'] / 
                     max(1, self.session_stats['ticks_processed'])) * 100
        
        logger.info(
            f"Progress: {self.session_stats['ticks_processed']} ticks, "
            f"{self.session_stats['candles_formed']} candles, "
            f"Error rate: {error_rate:.2f}%"
        )
    
    async def on_ws_connected(self):
        """Handle successful WebSocket connection with retry logic."""
        logger.info("[CONNECTED] WebSocket connected successfully")
        
        # Subscribe to instruments with retry
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                instruments = [{
                    'securityId': config.NIFTY_SECURITY_ID,
                    'exchangeSegment': config.NIFTY_EXCHANGE_SEGMENT
                }]
                
                success = await self.ws_handler.subscribe(instruments)
                if success:
                    logger.info(f"[OK] Subscribed to {len(instruments)} instruments")
                    break
                else:
                    retry_count += 1
                    logger.warning(f"Subscription failed, retry {retry_count}/{max_retries}")
                    await asyncio.sleep(2 ** retry_count)  # Exponential backoff
                    
            except Exception as e:
                logger.error(f"Subscription error: {e}")
                retry_count += 1
                await asyncio.sleep(2 ** retry_count)
        
        if retry_count >= max_retries:
            logger.error("[ERROR] Failed to subscribe after retries")
            await self.shutdown()
    
    async def on_ws_disconnected(self):
        """Handle WebSocket disconnection with cleanup."""
        logger.warning("[DISCONNECTED] WebSocket disconnected")
        
        # Log session stats on disconnect
        self._log_session_stats()
        
        # Attempt reconnection if not shutting down
        if not self.shutdown_flag:
            logger.info("Attempting to reconnect...")
            # Reconnection handled by WebSocket handler
    
    async def on_ws_error(self, error):
        """Handle WebSocket errors with detailed logging."""
        logger.error(f"[WS_ERROR] WebSocket error: {error}")
        self.session_stats['errors_count'] += 1
        
        # Log error details
        if isinstance(error, dict):
            for key, value in error.items():
                logger.error(f"  {key}: {value}")
    
    async def run_live(self):
        """Run bot in live mode with enhanced monitoring and recovery."""
        logger.info("[START] Starting live trading mode")
        
        if not self.ws_handler:
            logger.error("[ERROR] WebSocket handler not initialized - check credentials")
            return
        
        # Setup graceful shutdown handlers
        loop = asyncio.get_event_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, lambda: asyncio.create_task(self.shutdown()))
        
        try:
            logger.info("[INFO] Starting WebSocket connection...")
            
            # Create monitoring task
            monitor_task = asyncio.create_task(self._monitor_health())
            
            # Start WebSocket connection
            ws_task = asyncio.create_task(self.ws_handler.run())
            
            # Wait for tasks
            await asyncio.gather(ws_task, monitor_task)
            
        except KeyboardInterrupt:
            logger.info("[STOP] Received interrupt signal")
        except Exception as e:
            logger.error(f"[ERROR] Live mode error: {e}", exc_info=True)
            self.session_stats['errors_count'] += 1
        finally:
            await self.shutdown()
    
    async def _monitor_health(self):
        """Monitor system health and performance."""
        while not self.shutdown_flag:
            try:
                await asyncio.sleep(60)  # Check every minute
                
                # Check data freshness
                self._check_data_staleness()
                
                # Log statistics
                uptime = (datetime.now() - self.session_stats['start_time']).seconds / 60
                # logger.info(f"Health check - Uptime: {uptime:.1f} min, "
                #           f"Errors: {self.session_stats['errors_count']}")
                
                # Check memory usage (optional)
                try:
                    import psutil
                    process = psutil.Process()
                    memory_mb = process.memory_info().rss / 1024 / 1024
                    logger.info(f"Memory usage: {memory_mb:.1f} MB")
                except ImportError:
                    pass
                    
            except Exception as e:
                logger.error(f"Health monitor error: {e}")
    
    def run_replay(self):
        """Run bot in replay mode with comprehensive analysis."""
        logger.info("[START] Starting replay mode")
        
        if not self.replay_csv:
            logger.error("No CSV file specified for replay mode")
            return
        
        try:
            # Validate CSV file exists
            import os
            if not os.path.exists(self.replay_csv):
                raise FileNotFoundError(f"CSV file not found: {self.replay_csv}")
            
            # Run replay through alert system
            self.alert_system.run_replay(speed=0.0)
            
            # Get signal quality analysis
            signal_performance = self.signal_analyzer.get_performance_summary()
            
            # Display enhanced results
            self._display_replay_results(signal_performance)
            
        except FileNotFoundError as e:
            logger.error(f"[ERROR] {e}")
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
        
        # Safely access dictionary keys with defaults
        logger.info(f"Mode: {stats.get('mode', 'unknown')}")
        logger.info(f"Duration: {stats.get('session_duration', 'N/A')}")
        logger.info(f"Candles Processed: {stats.get('candles_processed', 0)}")
        logger.info(f"Patterns Detected: {stats.get('total_patterns_detected', 0)}")
        logger.info(f"Alerts Sent: {stats.get('total_alerts_sent', 0)}")
        
        # Prediction accuracy
        logger.info(f"\n[STATS] Prediction Performance:")
        logger.info(f"  Overall Accuracy: {stats.get('prediction_accuracy', 0):.1%}")
        logger.info(f"  Win Rate: {stats.get('win_rate', 0):.1%}")
        
        # Signal quality metrics
        if signal_performance:
            logger.info(f"\n[SIGNAL] Signal Quality Metrics:")
            logger.info(f"  Total Signals: {signal_performance.get('total_signals', 0)}")
            logger.info(f"  Success Rate: {signal_performance.get('success_rate', 0):.1%}")
            logger.info(f"  Avg Duration: {signal_performance.get('avg_duration', 0):.1f} minutes")
            logger.info(f"  Active Signals: {signal_performance.get('active_signals', 0)}")
            
            # Pattern durations
            pattern_durations = signal_performance.get('pattern_durations', {})
            if pattern_durations:
                logger.info(f"\n[PATTERNS] Average Pattern Durations:")
                sorted_patterns = sorted(pattern_durations.items(), 
                                       key=lambda x: x[1], reverse=True)[:10]
                for pattern, duration in sorted_patterns:
                    if duration > 0:
                        logger.info(f"  {pattern}: {duration:.1f} minutes")
        
        logger.info("=" * 60)
    
    def _log_session_stats(self):
        """Log current session statistics with detailed metrics."""
        duration = (datetime.now() - self.session_stats['start_time']).total_seconds() / 60
        
        # Calculate rates
        tick_rate = self.session_stats['ticks_processed'] / max(1, duration)
        candle_rate = self.session_stats['candles_formed'] / max(1, duration)
        
        logger.info(f"\n[STATS] Session Statistics:")
        logger.info(f"  Duration: {duration:.1f} minutes")
        logger.info(f"  Ticks: {self.session_stats['ticks_processed']} ({tick_rate:.1f}/min)")
        logger.info(f"  Candles: {self.session_stats['candles_formed']} ({candle_rate:.1f}/min)")
        logger.info(f"  Patterns: {self.session_stats['patterns_detected']}")
        logger.info(f"  Alerts: {self.session_stats['alerts_sent']}")
        logger.info(f"  Errors: {self.session_stats['errors_count']}")
    
    async def shutdown(self):
        """Async shutdown procedure with comprehensive cleanup."""
        # Use lock to prevent multiple shutdowns
        if self._shutdown_lock:
            async with self._shutdown_lock:
                if self.shutdown_flag:
                    return
                self.shutdown_flag = True
        else:
            if self.shutdown_flag:
                return
            self.shutdown_flag = True
        
        logger.info("[SHUTDOWN] Initiating graceful shutdown...")
        
        try:
            # Disconnect WebSocket with timeout
            if self.ws_handler:
                try:
                    await asyncio.wait_for(
                        self.ws_handler.disconnect(),
                        timeout=5.0
                    )
                except asyncio.TimeoutError:
                    logger.warning("WebSocket disconnect timed out")
            
            # Shutdown alert system
            self.alert_system.shutdown()
            
            # Save signal quality data
            signal_performance = self.signal_analyzer.get_performance_summary()
            
            # Print final statistics
            self._print_final_stats(signal_performance)
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
        finally:
            logger.info("[OK] Shutdown complete")
    
    def shutdown_sync(self):
        """Synchronous shutdown for non-async context."""
        if self.shutdown_flag:
            return
        
        self.shutdown_flag = True
        logger.info("[SHUTDOWN] Shutting down...")
        
        try:
            self.alert_system.shutdown()
            self._log_session_stats()
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
        
        logger.info("[OK] Shutdown complete")
    
    def _print_final_stats(self, signal_performance: dict):
        """Print comprehensive final statistics with error handling."""
        try:
            stats = self.alert_system.get_statistics()
            
            logger.info("=" * 60)
            logger.info("[REPORT] FINAL SESSION REPORT")
            logger.info("=" * 60)
            
            # Session info
            logger.info(f"Session Duration: {stats.get('session_duration', 'N/A')}")
            logger.info(f"Total Patterns: {stats.get('total_patterns_detected', 0)}")
            logger.info(f"Total Alerts: {stats.get('total_alerts_sent', 0)}")
            logger.info(f"Total Errors: {self.session_stats['errors_count']}")
            
            # Performance
            logger.info(f"\nPrediction Accuracy: {stats.get('prediction_accuracy', 0):.1%}")
            logger.info(f"Win Rate: {stats.get('win_rate', 0):.1%}")
            
            # Signal quality
            if signal_performance:
                logger.info(f"\nSignal Quality:")
                logger.info(f"  Success Rate: {signal_performance.get('success_rate', 0):.1%}")
                logger.info(f"  Avg Duration: {signal_performance.get('avg_duration', 0):.1f} min")
            
            logger.info("=" * 60)
            
        except Exception as e:
            logger.error(f"Error printing final stats: {e}")


def signal_handler(signum, frame):
    """Handle system signals for graceful shutdown."""
    logger.info(f"[SIGNAL] Received signal {signum}")
    sys.exit(0)


def validate_config():
    """Validate configuration with comprehensive checks."""
    issues = []
    warnings = []
    
    # Check required credentials for live mode
    if not config.DHAN_ACCESS_TOKEN_B64:
        issues.append("DHAN_ACCESS_TOKEN_B64 not configured")
    if not config.DHAN_CLIENT_ID_B64:
        issues.append("DHAN_CLIENT_ID_B64 not configured")
    
    # Check optional Telegram configuration
    if not config.TELEGRAM_BOT_TOKEN_B64:
        warnings.append("Telegram bot token not configured (alerts disabled)")
    if not config.TELEGRAM_CHAT_ID:
        warnings.append("Telegram chat ID not configured (alerts disabled)")
    
    # Validate numeric configurations
    try:
        if config.MIN_CANDLES_FOR_ANALYSIS < 1:
            issues.append("MIN_CANDLES_FOR_ANALYSIS must be >= 1")
        if config.PATTERN_WINDOW < 10:
            warnings.append("PATTERN_WINDOW < 10 may reduce pattern detection accuracy")
        if config.COOLDOWN_SECONDS < 0:
            issues.append("COOLDOWN_SECONDS must be >= 0")
    except AttributeError as e:
        issues.append(f"Missing configuration attribute: {e}")
    
    # Log warnings
    for warning in warnings:
        logger.warning(f"Configuration warning: {warning}")
    
    # Check for critical issues
    if issues:
        logger.error("Configuration issues found:")
        for issue in issues:
            logger.error(f"  - {issue}")
        return False
    
    logger.info("[OK] Configuration validated successfully")
    return True


def main():
    """Main entry point with enhanced argument parsing and error handling."""
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
    
  Without Telegram:
    python main.py --mode live --no-telegram
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
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate configuration and exit"
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
    
    # Validate configuration
    if args.mode == "live" or args.validate_only:
        if not validate_config():
            if args.validate_only:
                sys.exit(1)
            elif args.mode == "live":
                logger.error("[ERROR] Configuration validation failed")
                sys.exit(1)
    
    if args.validate_only:
        logger.info("Configuration validation complete")
        sys.exit(0)
    
    # Display startup banner
    logger.info("=" * 60)
    logger.info("[START] ADVANCED PATTERN RECOGNITION TRADING SYSTEM")
    logger.info("=" * 60)
    logger.info(f"Mode: {args.mode.upper()}")
    if args.csv:
        logger.info(f"CSV: {args.csv}")
    logger.info(f"Debug: {args.debug}")
    logger.info(f"Telegram: {'Disabled' if args.no_telegram else 'Enabled'}")
    logger.info("=" * 60)
    
    # Create and run bot with error handling
    bot = None
    try:
        bot = TradingBot(mode=args.mode, replay_csv=args.csv)
        
        if args.mode == "live":
            # Run async event loop
            logger.info("[INFO] Starting async event loop...")
            asyncio.run(bot.run_live())
        else:
            # Run synchronous replay
            bot.run_replay()
            
    except KeyboardInterrupt:
        logger.info("[STOP] Interrupted by user")
    except Exception as e:
        logger.critical(f"[FATAL] Unhandled exception: {e}", exc_info=True)
        sys.exit(1)
    finally:
        # Ensure cleanup
        if bot and not bot.shutdown_flag:
            if args.mode == "live":
                asyncio.run(bot.shutdown())
            else:
                bot.shutdown_sync()


if __name__ == "__main__":
    main()
