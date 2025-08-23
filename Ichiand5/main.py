"""
Main application with all references properly defined.
"""
import os
import sys
import signal
import logging
import time
import threading
from pathlib import Path
from typing import Optional
from datetime import datetime

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
logger: Optional[logging.Logger] = None
shutdown_event = threading.Event()

def signal_handler(signum, frame):
    """Handle termination signals gracefully."""
    global shutdown_event, client, monitor, logger
    
    if logger:
        logger.info(f"Received signal {signum}. Initiating graceful shutdown...")
    
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
    """Background health monitoring thread."""
    global client, monitor, logger
    
    last_health_check = time.time()
    health_check_interval = 60  # seconds
    
    while not shutdown_event.is_set():
        try:
            current_time = time.time()
            
            if current_time - last_health_check >= health_check_interval:
                last_health_check = current_time
                
                # Check WebSocket health
                if client:
                    if hasattr(client, 'connected') and not client.connected:
                        logger.warning("WebSocket disconnected, attempting recovery...")
                        try:
                            client.connect()
                        except Exception as e:
                            logger.error(f"Reconnection failed: {e}")
                
                # Check signal monitor health
                if monitor and hasattr(monitor, 'is_healthy'):
                    if not monitor.is_healthy():
                        logger.warning("Signal monitor unhealthy, restarting...")
                        monitor.restart()
                
                # Log system metrics
                if monitor:
                    metrics = monitor.get_metrics()
                    logger.info(f"System metrics: {metrics}")
            
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
    
    return True

def main():
    """Enhanced main application entry point."""
    global client, monitor, logger
    
    telegram_bot: Optional[TelegramBot] = None
    
    try:
        # Setup directories
        Path("logs").mkdir(exist_ok=True)
        Path("images").mkdir(exist_ok=True)
        Path("data").mkdir(exist_ok=True)
        
        # Setup logging
        setup_logging(CONFIG.log_file, getattr(logging, CONFIG.log_level))
        logger = logging.getLogger(__name__)
        
        logger.info("=" * 60)
        logger.info("🚀 ENHANCED NIFTY50 TRADING SYSTEM V2.0")
        logger.info("📊 6 Technical Indicators with Signal Validation")
        logger.info("=" * 60)
        
        # Validate environment
        if not validate_environment():
            logger.error("Environment validation failed")
            sys.exit(1)
        
        # Validate configuration
        if not CONFIG.validate():
            logger.error("Configuration validation failed")
            sys.exit(1)
        
        # Register signal handlers
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Initialize Telegram bot
        logger.info("Initializing Telegram bot...")
        telegram_bot = TelegramBot(
            CONFIG.telegram_token_b64,
            CONFIG.telegram_chat_id
        )
        
        # Format startup message with all required variables defined
        cooldown = CONFIG.COOLDOWN_SECONDS
        strength = CONFIG.MIN_SIGNAL_STRENGTH  
        confidence = CONFIG.MIN_CONFIDENCE
        
        startup_message = (
            "🚀 <b>Enhanced Trading System V2.0</b>\n"
            "━━━━━━━━━━━━━━━━━━━\n"
            "📊 Indicators: Ichimoku, Stochastic, OBV, Bollinger, ADX, ATR\n"
            "✅ Signal Validation: Enabled\n"
            "📈 Charts: Enabled\n"
            f"⏱️ Alert Cooldown: {cooldown}s\n"
            f"💪 Min Signal Strength: {strength}\n"
            f"🎯 Min Confidence: {confidence}%"
        )
        
        if not telegram_bot.send_message(startup_message):
            logger.warning("Telegram test failed - continuing anyway")
        
        # Initialize signal monitor
        logger.info("Initializing signal monitor...")
        monitor = SignalMonitor(CONFIG, telegram_bot)
        monitor.start()
        
        # Initialize WebSocket client
        logger.info("Initializing WebSocket client...")
        client = EnhancedDhanWebSocketClient(
            CONFIG.DHAN_ACCESS_TOKEN_B64,
            CONFIG.DHAN_CLIENT_ID_B64,
            telegram_bot
        )
        
        # Add monitor to client if possible
        if hasattr(client, 'signal_monitor'):
            client.signal_monitor = monitor
        
        # Connect to WebSocket
        logger.info("Connecting to Dhan WebSocket...")
        client.connect()
        
        # Start health monitoring
        health_thread = threading.Thread(target=health_monitor, daemon=True)
        health_thread.start()
        
        logger.info("System running. Press Ctrl+C to stop.")
        
        # Main loop
        while not shutdown_event.is_set():
            time.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        
        if telegram_bot:
            try:
                error_msg = f"❌ System Error: {str(e)[:200]}"
                telegram_bot.send_message(error_msg)
            except:
                pass
    finally:
        shutdown_event.set()
        
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
            logger.info("System shutdown complete")

if __name__ == "__main__":
    main()
