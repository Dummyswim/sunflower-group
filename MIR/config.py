"""
Configuration module for trading alert system with enhanced settings.
"""
import base64
import os
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Create necessary directories
Path("logs").mkdir(exist_ok=True)
Path("images").mkdir(exist_ok=True)

# API Credentials (ensure these are properly encoded)
# Example: base64.b64encode("your_actual_token".encode()).decode()
DHAN_ACCESS_TOKEN_B64 = os.getenv("DHAN_TOKEN_B64", "")  # Use environment variables for security
DHAN_CLIENT_ID_B64 = os.getenv("DHAN_CLIENT_B64", "")
TELEGRAM_BOT_TOKEN_B64 = os.getenv("TELEGRAM_TOKEN_B64", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

# System Settings
COOLDOWN_SECONDS = 120  # Reduced from 180 for more frequent alerts
MIN_DATA_POINTS = 35  # Increased for better indicator accuracy
MAX_BUFFER_SIZE = 5000  # Increased buffer for more data
LOG_FILE = "logs/trading_system_1.log"

# Technical Indicator Settings
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
RSI_PERIOD = 14
VWAP_PERIOD = 20
KELTNER_PERIOD = 20
KELTNER_MULTIPLIER = 2.0
SUPERTREND_PERIOD = 10
SUPERTREND_MULTIPLIER = 3.0

# Indicator Weights for Signal Generation
INDICATOR_WEIGHTS = {
    "macd": 0.25,
    "rsi": 0.20,
    "vwap": 0.20,
    "keltner": 0.15,
    "supertrend": 0.15,
    "impulse": 0.05
}

# Signal Thresholds
BUY_THRESHOLD = 0.3           # Reduced from 0.60
SELL_THRESHOLD = -0.3         # Reduced from -0.60
STRONG_SIGNAL_THRESHOLD = 0.75  # 75% for strong signals

# Enhanced Signal Thresholds
STRONG_BUY_THRESHOLD = 0.6    # Reduced from 0.75 for more signals
STRONG_SELL_THRESHOLD = -0.6  # Reduced from -0.75
# Alert Settings
MIN_CONFIDENCE_FOR_ALERT = 40  # Reduced from 60
MIN_SCORE_FOR_ALERT = 0.3     # Reduced from 0.5




# Market Data Settings
NIFTY_SECURITY_ID = 13
NIFTY_EXCHANGE_SEGMENT = "IDX_I"
PRICE_SANITY_MIN = 15000
PRICE_SANITY_MAX = 35000

def validate_config():
    """Validate configuration settings with proper error handling."""
    try:
        logger.info("Validating configuration settings...")
        
        if not all([DHAN_ACCESS_TOKEN_B64, DHAN_CLIENT_ID_B64]):
            logger.error("Missing Dhan API credentials")
            raise ValueError("Missing Dhan API credentials")
            
        if not all([TELEGRAM_BOT_TOKEN_B64, TELEGRAM_CHAT_ID]):
            logger.error("Missing Telegram credentials")
            raise ValueError("Missing Telegram credentials")
            
        weight_sum = sum(INDICATOR_WEIGHTS.values())
        if abs(weight_sum - 1.0) > 0.001:  # Allow small floating point errors
            logger.error(f"Indicator weights sum to {weight_sum}, expected 1.0")
            raise ValueError(f"Indicator weights must sum to 1.0, got {weight_sum}")
            
        logger.info("Configuration validation successful")
        return True
        
    except Exception as e:
        logger.error(f"Configuration validation failed: {e}")
        raise