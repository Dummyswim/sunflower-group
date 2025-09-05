"""
Main application entry point for 5-minute candle trading system.
Enhanced with comprehensive logging and monitoring.
"""
import os
import sys
import signal
import logging
import time
import threading
from pathlib import Path
from typing import Optional, Dict
from datetime import datetime, timedelta
import json

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Import configuration first
from config import CONFIG

# Import other modules
from logging_setup import setup_logging
from telegram_bot import TelegramBot
from websocket_client import EnhancedDhanWebSocketClient
from signal_monitor import SignalMonitor

# Global components
client: Optional[EnhancedDhanWebSocketClient] = None
monitor: Optional[SignalMonitor] = None
telegram_bot: Optional[TelegramBot] = None
logger: Optional[logging.Logger] = None
shutdown_event = threading.Event()

# Performance tracking
performance_metrics = {
    'start_time': None,
    'signals_generated': 0,
    'alerts_sent': 0,
    'patterns_detected': 0,
    'last_signal': None,
    'win_rate': 0.0,
    'avg_signal_strength': 0.0
}

def signal_handler(signum, frame):
    """Handle termination signals gracefully."""
    global shutdown_event, client, monitor, logger, performance_metrics
    
    if logger:
        logger.info(f"Received signal {signum}. Initiating graceful shutdown...")
        
        # Log final statistics
        if performance_metrics['start_time']:
            runtime = datetime.now() - performance_metrics['start_time']
            logger.info("=" * 60)
            logger.info("SESSION SUMMARY")
            logger.info(f"Runtime: {runtime}")
            logger.info(f"Total Signals Generated: {performance_metrics['signals_generated']}")
            logger.info(f"Alerts Sent: {performance_metrics['alerts_sent']}")
            logger.info(f"Patterns Detected: {performance_metrics['patterns_detected']}")
            logger.info(f"Average Signal Strength: {performance_metrics['avg_signal_strength']:.2%}")
            logger.info("=" * 60)
    
    shutdown_event.set()
    
    # Cleanup
    if client:
        try:
            client.disconnect()
        except Exception as e:
            if logger:
                logger.error(f"Error disconnecting client: {e}")
    
    if monitor:
        try:
            monitor.stop()
        except Exception as e:
            if logger:
                logger.error(f"Error stopping monitor: {e}")
    
    sys.exit(0)

def health_monitor():
    """Enhanced health monitoring with detailed status reporting."""
    global client, monitor, logger, telegram_bot, performance_metrics
    
    last_health_check = time.time()
    health_check_interval = 60  # seconds
    detailed_log_interval = 300  # 5 minutes for detailed status
    last_detailed_log = time.time()
    reconnect_attempts = 0
    max_reconnect_attempts = 3
    
    while not shutdown_event.is_set():
        try:
            current_time = time.time()
            
            # Regular health check
            if current_time - last_health_check >= health_check_interval:
                last_health_check = current_time
                
                # Check WebSocket health
                if client:
                    if hasattr(client, 'connected') and not client.connected:
                        logger.warning("🔴 WebSocket disconnected, attempting recovery...")
                        reconnect_attempts += 1
                        
                        if reconnect_attempts <= max_reconnect_attempts:
                            try:
                                success = client.connect_with_retry()
                                if success:
                                    reconnect_attempts = 0
                                    logger.info("✅ Reconnection successful")
                                else:
                                    logger.error(f"❌ Reconnection attempt {reconnect_attempts}/{max_reconnect_attempts} failed")
                            except Exception as e:
                                logger.error(f"Reconnection error: {e}")
                        else:
                            logger.error("⚠️ Max reconnection attempts reached")
                            if telegram_bot:
                                telegram_bot.send_message(
                                    "<b>🚨 Critical Error</b>\n"
                                    "Unable to maintain WebSocket connection.\n"
                                    "Manual intervention may be required."
                                )
                                logger.debug(f"{telegram_bot}")
                    else:
                        reconnect_attempts = 0
                
                # Get comprehensive status
                if client and hasattr(client, 'get_detailed_status'):
                    status = client.get_detailed_status()
                    
                    # Log market metrics
                    logger.info(f"📊 MARKET STATUS: "
                              f"Price: ₹{status.get('current_price', 0):.2f} | "
                              f"Day Change: {status.get('day_change_pct', 0):.2%} | "
                              f"Volatility: {status.get('volatility', 'N/A')}")
                    
                    # Log candle progress
                    candles_collected = status.get('candles_collected', 0)
                    candles_required = CONFIG.MIN_DATA_POINTS
                    candles_remaining = max(0, candles_required - candles_collected)
                    
                    if candles_remaining > 0:
                        logger.info(f"📈 CANDLE PROGRESS: {candles_collected}/{candles_required} "
                                  f"({candles_remaining} remaining) | "
                                  f"Time to full data: ~{candles_remaining * 5} minutes")
                    else:
                        logger.info(f"✅ CANDLE DATA: Complete ({candles_collected} candles) | "
                                  f"Ready for analysis")
                    
                    # Log signal analysis status
                    if status.get('last_analysis'):
                        analysis = status['last_analysis']
                        logger.info(f"🎯 SIGNAL ANALYSIS: "
                                  f"RSI: {analysis.get('rsi', 0):.1f} | "
                                  f"MACD: {analysis.get('macd_signal', 'N/A')} | "
                                  f"Volume Trend: {analysis.get('volume_trend', 'N/A')} | "
                                  f"Signal Strength: {analysis.get('signal_strength', 0):.2%}")
                        
                        # Log pattern if detected
                        if analysis.get('pattern'):
                            logger.info(f"🔍 PATTERN DETECTED: {analysis['pattern']} | "
                                      f"Confidence: {analysis.get('pattern_confidence', 0):.1%}")
                            performance_metrics['patterns_detected'] += 1
                    
                    # Log active signals
                    if status.get('active_signals'):
                        for signal in status['active_signals']:
                            elapsed = (datetime.now() - signal['timestamp']).seconds // 60
                            logger.info(f"📍 ACTIVE SIGNAL: {signal['type']} | "
                                      f"Entry: ₹{signal['entry_price']:.2f} | "
                                      f"Current P/L: {signal.get('pnl_pct', 0):.2%} | "
                                      f"Duration: {elapsed} min | "
                                      f"Expected: {signal.get('expected_duration', 'N/A')} min")
                
                # Monitor metrics
                if monitor:
                    metrics = monitor.get_metrics()
                    if metrics.get('total_signals', 0) > 0:
                        logger.info(f"📊 MONITOR STATS: "
                                  f"Active Signals: {metrics.get('active_signals', 0)} | "
                                  f"Today's Signals: {metrics.get('today_signals', 0)} | "
                                  f"Success Rate: {metrics.get('success_rate', 0):.1%} | "
                                  f"Avg Duration: {metrics.get('avg_duration', 0):.1f} min")
                
                # Log packet statistics with more detail
                if client and hasattr(client, 'packets_received'):
                    stats = client.packets_received
                    total_packets = sum(stats.values())
                    if total_packets > 0:
                        logger.info(f"📡 PACKET STATS (Total: {total_packets}): "
                                  f"Quote: {stats['quote']} ({stats['quote']/total_packets*100:.1f}%) | "
                                  f"Ticker: {stats['ticker']} ({stats['ticker']/total_packets*100:.1f}%) | "
                                  f"Full: {stats['full']} ({stats['full']/total_packets*100:.1f}%)")
            
            # Detailed status report every 5 minutes
            if current_time - last_detailed_log >= detailed_log_interval:
                last_detailed_log = current_time
                logger.info("=" * 60)
                logger.info("📋 DETAILED SYSTEM REPORT")
                
                if client and hasattr(client, 'get_full_report'):
                    report = client.get_full_report()
                    
                    # System health
                    logger.info(f"System Health: {report.get('health_status', 'Unknown')}")
                    logger.info(f"Uptime: {report.get('uptime', 'N/A')}")
                    logger.info(f"Memory Usage: {report.get('memory_usage', 'N/A')}")
                    
                    # Trading metrics
                    logger.info(f"Signals Generated: {performance_metrics['signals_generated']}")
                    logger.info(f"Alerts Sent: {performance_metrics['alerts_sent']}")
                    logger.info(f"Patterns Detected: {performance_metrics['patterns_detected']}")
                    
                    # Market analysis
                    if report.get('market_analysis'):
                        ma = report['market_analysis']
                        logger.info(f"Market Trend: {ma.get('trend', 'N/A')}")
                        logger.info(f"Trend Strength: {ma.get('trend_strength', 0):.2%}")
                        logger.info(f"Support: ₹{ma.get('support', 0):.2f}")
                        logger.info(f"Resistance: ₹{ma.get('resistance', 0):.2f}")
                        logger.info(f"Market Sentiment: {ma.get('sentiment', 'N/A')}")
                
                logger.info("=" * 60)
            
            time.sleep(1)
            
        except Exception as e:
            logger.error(f"Health monitor error: {e}")
            time.sleep(5)

def validate_environment() -> bool:
    """Validate required environment variables."""
    required_vars = [
        'DHAN_TOKEN_B64',
        'DHAN_CLIENT_B64', 
        'TELEGRAM_TOKEN_B64',
        'TELEGRAM_CHAT_ID'
    ]
    
    missing = []
    for var in required_vars:
        if not os.getenv(var):
            missing.append(var)
    
    if missing:
        print(f"ERROR: Missing environment variables: {', '.join(missing)}")
        return False
    
    logger.info(f"✅ Environment validation successful")
    return True

def main():
    """Enhanced main application with comprehensive monitoring."""
    global client, monitor, logger, telegram_bot, performance_metrics
    
    try:
        # Setup directories
        Path("logs").mkdir(exist_ok=True)
        Path("images").mkdir(exist_ok=True)
        Path("data").mkdir(exist_ok=True)
        
        # Setup logging
        setup_logging(CONFIG.log_file, getattr(logging, CONFIG.log_level))
        logger = logging.getLogger(__name__)
        
        # Start performance tracking
        performance_metrics['start_time'] = datetime.now()
        
        logger.info("=" * 60)
        logger.info("🚀 ENHANCED NIFTY50 TRADING SYSTEM V3.0")
        logger.info("📊 Real-time Market Analysis & Signal Generation")
        logger.info("=" * 60)
        logger.info(f"System Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Configuration:")
        logger.info(f"  • Timeframe: 5-minute candles")
        logger.info(f"  • Min Data Points: {CONFIG.MIN_DATA_POINTS} candles")
        logger.info(f"  • Signal Threshold: {CONFIG.MIN_SIGNAL_STRENGTH:.1%}")
        logger.info(f"  • Min Confidence: {CONFIG.MIN_CONFIDENCE}%")
        logger.info(f"  • Alert Cooldown: {CONFIG.COOLDOWN_SECONDS} seconds")
        logger.info(f"  • Indicators: RSI, MACD, VWAP, Bollinger, OBV")
        logger.info("=" * 60)
        
        # Validate environment
        if not validate_environment():
            logger.error("Environment validation failed")
            sys.exit(1)
        
        # Validate configuration
        if not CONFIG.validate():
            logger.error("Configuration validation failed")
            sys.exit(1)
        
        logger.info("✅ Configuration validated successfully")
        
        # Register signal handlers
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Initialize Telegram bot
        logger.info("📱 Initializing Telegram bot...")
        telegram_bot = TelegramBot(
            CONFIG.telegram_token_b64,
            CONFIG.telegram_chat_id
        )
        
        # Format enhanced startup message
        startup_message = (
            "<b>🚀 Enhanced Trading System V3.0</b>\n"
            "=" * 25 + "\n"
            "<b>📊 System Configuration:</b>\n"
            f"• Data Feed: Real-time WebSocket\n"
            f"• Candles: 5-minute intervals\n"
            f"• Required Data: {CONFIG.MIN_DATA_POINTS} candles\n"
            f"• Collection Time: ~{CONFIG.MIN_DATA_POINTS * 5} minutes\n\n"
            "<b>📈 Technical Indicators:</b>\n"
            "• RSI (Momentum)\n"
            "• MACD (Trend)\n"
            "• VWAP (Volume-Price)\n"
            "• Bollinger Bands (Volatility)\n"
            "• OBV (Volume Trend)\n\n"
            "<b>🎯 Signal Parameters:</b>\n"
            f"• Min Strength: {CONFIG.MIN_SIGNAL_STRENGTH:.1%}\n"
            f"• Min Confidence: {CONFIG.MIN_CONFIDENCE}%\n"
            f"• Cooldown: {CONFIG.COOLDOWN_SECONDS}s\n\n"
            "<b>✨ Features:</b>\n"
            "• Pattern Recognition\n"
            "• Signal Duration Prediction\n"
            "• Auto-reconnection\n"
            "• Real-time Charts\n\n"
            f"<b>🕐 Start Time:</b> {datetime.now().strftime('%H:%M:%S')}"
        )
        
        if not telegram_bot.send_message(startup_message):
            logger.warning("⚠️ Telegram test failed - continuing anyway")
        else:
            logger.info("✅ Telegram notification sent")
        
        # Initialize signal monitor
        logger.info("🔍 Initializing signal monitor...")
        monitor = SignalMonitor(CONFIG, telegram_bot)
        monitor.start()
        logger.info("✅ Signal monitor started")
        
        # Initialize WebSocket client
        logger.info("📡 Initializing WebSocket client...")
        client = EnhancedDhanWebSocketClient(
            CONFIG.DHAN_ACCESS_TOKEN_B64,
            CONFIG.DHAN_CLIENT_ID_B64,
            telegram_bot
        )
        
        # Set performance metrics callback
        def update_metrics(metric_type, value):
            global performance_metrics
            if metric_type in performance_metrics:
                performance_metrics[metric_type] = value
        
        client.set_metrics_callback(update_metrics)
        
        # Add monitor to client
        if hasattr(client, 'set_signal_monitor'):
            client.set_signal_monitor(monitor)
            logger.info("✅ Signal monitor attached to client")
        
        # Connect with retry logic
        logger.info("🔌 Connecting to Dhan WebSocket with retry...")
        if not client.connect_with_retry():
            logger.error("❌ Failed to establish WebSocket connection")
            telegram_bot.send_message(
                "<b>❌ Connection Failed</b>\n"
                "Unable to connect to market data feed.\n"
                "Please check credentials and try again."
            )
            sys.exit(1)
        
        logger.info("✅ Successfully connected to market data feed")
        
        # Start health monitoring
        health_thread = threading.Thread(target=health_monitor, daemon=True)
        health_thread.start()
        logger.info("✅ Health monitoring started")
        
        logger.info("=" * 60)
        logger.info("🟢 SYSTEM READY - Waiting for market data...")
        logger.info("Press Ctrl+C to stop")
        logger.info("=" * 60)
        
        # Main loop
        while not shutdown_event.is_set():
            time.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("⚠️ Keyboard interrupt received")
    except Exception as e:
        logger.error(f"🔴 Fatal error: {e}", exc_info=True)
        
        if telegram_bot:
            try:
                error_msg = f"<b>🔴 System Error</b>\n{str(e)[:200]}"
                telegram_bot.send_message(error_msg)
            except:
                pass
    finally:
        shutdown_event.set()
        
        # Log final statistics
        if performance_metrics['start_time']:
            runtime = datetime.now() - performance_metrics['start_time']
            logger.info("=" * 60)
            logger.info("📊 FINAL SESSION SUMMARY")
            logger.info(f"Total Runtime: {runtime}")
            logger.info(f"Signals Generated: {performance_metrics['signals_generated']}")
            logger.info(f"Alerts Sent: {performance_metrics['alerts_sent']}")
            logger.info(f"Patterns Detected: {performance_metrics['patterns_detected']}")
            logger.info("=" * 60)
        
        if client:
            try:
                client.disconnect()
            except Exception as e:
                if logger:
                    logger.error(f"Error during final disconnect: {e}")
        
        if monitor:
            try:
                monitor.stop()
            except Exception as e:
                if logger:
                    logger.error(f"Error stopping monitor: {e}")
        
        if logger:
            logger.info("🔴 System shutdown complete")

if __name__ == "__main__":
    main()
