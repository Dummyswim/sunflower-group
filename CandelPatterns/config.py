"""
Enhanced configuration with pattern recognition settings.
Fixed encoding issues and added WebSocket configurations.
"""
import base64
import os
from typing import Dict, Any
from pathlib import Path

# Create necessary directories
Path("logs").mkdir(exist_ok=True)
Path("charts").mkdir(exist_ok=True)


# File Paths
LOG_FILE = "logs/pattern_alerts.log"
CHART_DIR = "charts"
PATTERN_HISTORY_FILE = "charts/pattern_history.json"

# API Credentials (base64 encoded)
DHAN_ACCESS_TOKEN_B64 = os.getenv("DHAN_TOKEN_B64", "")  # Use environment variables for security
DHAN_CLIENT_ID_B64 = os.getenv("DHAN_CLIENT_B64", "")
TELEGRAM_BOT_TOKEN_B64 = os.getenv("TELEGRAM_TOKEN_B64", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

# WebSocket Settings
NIFTY_SECURITY_ID = 13
NIFTY_EXCHANGE_SEGMENT = "IDX_I"  # 0
EXCHANGE_SEGMENT_MAP = {
    "IDX_I": 0, "NSE_EQ": 1, "NSE_FNO": 2, "NSE_CURRENCY": 3,
    "BSE_EQ": 4, "BSE_FNO": 5, "BSE_CURRENCY": 6, "MCX": 7,
}

# Pattern Recognition Settings
PATTERN_WINDOW = 100
DEFAULT_PATTERN_PROB = 0.55
MIN_PATTERNS_FOR_ALERT = 1
PATTERN_CONFIDENCE_THRESHOLD = 0.2

# Technical Analysis Settings
ATR_PERIOD = 14
RSI_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
MOMENTUM_LOOKBACK = 3

# Risk Management
ATR_HIGH_VOLATILITY = 0.01
ATR_LOW_VOLATILITY = 0.002
VOLATILITY_BOOST_FACTOR = 1.2
VOLATILITY_DAMPEN_FACTOR = 0.8

# Data Management
OHLC_WINDOW = 100
MIN_CANDLES_FOR_ANALYSIS = 10
COOLDOWN_SECONDS = 300  # 5 minutes
MAX_BUFFER_SIZE = 2000
CANDLE_TIMEFRAME_MINUTES = 5  # 5-minute candles

# Pattern window can stay the same (100 candles = 500 minutes of data)
PATTERN_WINDOW = 100

# Price Sanity Checks (for NIFTY)
PRICE_SANITY_MIN = 10000
PRICE_SANITY_MAX = 30000

def decode_b64(encoded: str) -> str:
    """Decode base64 encoded string."""
    return base64.b64decode(encoded).decode("utf-8")

