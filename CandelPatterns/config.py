# File: config.py
"""
Enhanced configuration with pattern recognition settings - CALIBRATED VERSION
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

# API Credentials (unchanged)
DHAN_ACCESS_TOKEN_B64 = os.getenv("DHAN_TOKEN_B64", "")
DHAN_CLIENT_ID_B64 = os.getenv("DHAN_CLIENT_B64", "")
TELEGRAM_BOT_TOKEN_B64 = os.getenv("TELEGRAM_TOKEN_B64", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

# WebSocket Settings (unchanged)
NIFTY_SECURITY_ID = 13
NIFTY_EXCHANGE_SEGMENT = "IDX_I"
EXCHANGE_SEGMENT_MAP = {
    "IDX_I": 0, "NSE_EQ": 1, "NSE_FNO": 2, "NSE_CURRENCY": 3,
    "BSE_EQ": 4, "BSE_FNO": 5, "BSE_CURRENCY": 6, "MCX": 7,
}

# ============= CALIBRATED SETTINGS =============

# INCREASED: Minimum confidence thresholds
MIN_CONFIDENCE_FOR_SIGNAL = 0.65  # Increased from 0.50 to 65%
PATTERN_CONSENSUS_REQUIRED = 0.70  # Increased from 0.60 to 70%
PATTERN_CONFIDENCE_THRESHOLD = 0.55  # Increased from 0.2 to 55%

# Pattern Settings
PATTERN_WINDOW = 100  # Keep same for 5-min candles (500 min history)
DEFAULT_PATTERN_PROB = 0.60  # Increased from 0.55
MIN_PATTERNS_FOR_ALERT = 2  # Keep same
MAX_NEUTRAL_PATTERN_RATIO = 0.20  # Reduced from 0.30 - stricter

# NEW: Require minimum trend strength for signals
MIN_TREND_STRENGTH = 0.01  # Increased from 0.005 (1% minimum)
PATTERN_HIERARCHY_ENABLED = True

# REFINED Pattern Hierarchy - Focus on high-accuracy patterns
PATTERN_HIERARCHY = {
    'CDL3WHITESOLDIERS': 10,    # 82% accuracy - highest priority
    'CDL3BLACKCROWS': 10,       # 80% accuracy
    'CDLMORNINGSTAR': 9,        # 78% accuracy
    'CDLEVENINGSTAR': 9,        # 75% accuracy
    'CDLENGULFING': 8,          # 72% accuracy
    'three_drives': 8,          # Custom pattern - proven effective
    'CDLHAMMER': 7,             # 68% accuracy
    'CDLINVERTEDHAMMER': 7,     # 65% accuracy (your detected pattern)
    'CDLSHOOTINGSTAR': 6,       # 65% accuracy
    'CDLHARAMI': 4,             # 55% accuracy - lower priority
    'CDLDOJI': 2,               # 50% accuracy - lowest priority
    'CDLSPINNINGTOP': 1,        # 48% accuracy - avoid
}

# Technical Indicators - FIXED for ATR calculation
ATR_PERIOD = 14
RSI_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
MOMENTUM_LOOKBACK = 5  # Increased from 3 for better trend detection

# NEW: Minimum data requirements
MIN_CANDLES_FOR_ATR = 20  # Ensure enough data for ATR
MIN_CANDLES_FOR_ANALYSIS = 20  # Increased from 10

# Risk Management - CALIBRATED
ATR_HIGH_VOLATILITY = 0.015  # 1.5% (increased from 0.01)
ATR_LOW_VOLATILITY = 0.005   # 0.5% (increased from 0.002)
VOLATILITY_BOOST_FACTOR = 1.15  # Reduced from 1.2
VOLATILITY_DAMPEN_FACTOR = 0.85  # Increased from 0.8

# NEW: Momentum filters
MIN_MOMENTUM_FOR_SIGNAL = 0.001  # Require 0.1% momentum alignment
MAX_MOMENTUM_CONFLICT = -0.002   # Reject if momentum opposes by >0.2%

# Data Management
OHLC_WINDOW = 100
COOLDOWN_SECONDS = 300  # Increased from 300 (10 minutes)
MAX_BUFFER_SIZE = 2000
CANDLE_TIMEFRAME_MINUTES = 5  # Correct - keep 5-minute

# Price Sanity Checks for NIFTY
PRICE_SANITY_MIN = 15000  # Updated for current NIFTY range
PRICE_SANITY_MAX = 30000

# NEW: Performance tracking thresholds
MIN_ACCURACY_FOR_ALERTS = 0.55  # Stop alerts if accuracy drops below 55%
MIN_SAMPLE_SIZE = 20  # Minimum predictions before enforcing accuracy check

# NEW: Volume requirements
REQUIRE_VOLUME_CONFIRMATION = True
MIN_VOLUME_RATIO = 0.8  # Require at least 80% of average volume

def decode_b64(encoded: str) -> str:
    """Decode base64 encoded string."""
    return base64.b64decode(encoded).decode("utf-8") if encoded else ""

HIGH_CONFIDENCE_PATTERNS = [
    'CDL3WHITESOLDIERS', 'CDL3BLACKCROWS', 'CDLMORNINGSTAR', 
    'CDLEVENINGSTAR', 'CDLENGULFING', 'three_drives'
]