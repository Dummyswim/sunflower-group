"""
Configuration module for 5-minute candle trading system.
All parameters validated and properly defined.
"""
import os
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

@dataclass
class TradingConfig:
    """Trading system configuration with validation."""
    
    # Credentials (from environment)
    telegram_token_b64: str = field(default_factory=lambda: os.getenv("TELEGRAM_TOKEN_B64", ""))
    telegram_chat_id: str = field(default_factory=lambda: os.getenv("TELEGRAM_CHAT_ID", ""))
    DHAN_ACCESS_TOKEN_B64: str = field(default_factory=lambda: os.getenv("DHAN_TOKEN_B64", ""))
    DHAN_CLIENT_ID_B64: str = field(default_factory=lambda: os.getenv("DHAN_CLIENT_B64", ""))
    
    # Market Configuration
    NIFTY_EXCHANGE_SEGMENT: int = 0
    NIFTY_SECURITY_ID: int = 13
    PRICE_SANITY_MIN: float = 10000.0
    PRICE_SANITY_MAX: float = 30000.0
    
    # Data Management
    MIN_DATA_POINTS: int = 30
    MAX_BUFFER_SIZE: int = 10000
    CANDLE_INTERVAL: int = 300  # 5 minutes in seconds
    
    # Alert Management
    COOLDOWN_SECONDS: int = 60
    MIN_SIGNAL_STRENGTH: float = 0.2
    MIN_CONFIDENCE: int = 20
    MIN_ACTIVE_INDICATORS: int = 2
    
    # Signal Duration
    SIGNAL_SUSTAIN_THRESHOLD: float = 0.6
    SIGNAL_ACCURACY_WINDOW: int = 10
    MAX_SIGNAL_DURATION_MINUTES: int = 15
    MIN_SIGNAL_DURATION_MINUTES: int = 5
    
    # Signal Validation
    SIGNAL_VALIDATION_SCORE: float = 0.4
    VOLUME_MULTIPLIER: float = 1.2
    
    # Indicator Weights (must sum to 1.0)
    INDICATOR_WEIGHTS: Dict[str, float] = field(default_factory=lambda: {
        "rsi": 0.30,
        "macd": 0.30,
        "vwap": 0.20,
        "bollinger": 0.10,
        "obv": 0.10
    })
    
    # RSI Parameters
    RSI_PARAMS: Dict = field(default_factory=lambda: {
        "period": 9,
        "overbought": 75,
        "oversold": 25,
        "neutral_zone": (40, 60)
    })
    
    # MACD Parameters
    MACD_PARAMS: Dict = field(default_factory=lambda: {
        "fastperiod": 8,
        "slowperiod": 17,
        "signalperiod": 9
    })
    
    # VWAP Parameters
    VWAP_PARAMS: Dict = field(default_factory=lambda: {
        "window": 3
    })
    
    # Bollinger Bands Parameters
    BOLLINGER_PARAMS: Dict = field(default_factory=lambda: {
        "period": 10,
        "stddev": 2
    })
    
    # OBV Parameters
    OBV_PARAMS: Dict = field(default_factory=lambda: {
        "window": 10
    })
    
    # Charting Configuration
    alert_with_charts: bool = True
    CHART_STYLE: str = 'seaborn'
    chart_save_path: str = 'images/'
    
    # Logging Configuration
    log_file: str = "logs/nifty_alerts.log"
    log_level: str = "INFO"
    
    # Performance Tracking
    track_performance: bool = True
    performance_report_interval: int = 3600
    
    def validate(self) -> bool:
        """Validate configuration parameters."""
        try:
            if not all([self.telegram_token_b64, self.telegram_chat_id, 
                       self.DHAN_ACCESS_TOKEN_B64, self.DHAN_CLIENT_ID_B64]):
                logger.error("Missing required credentials")
                return False
            
            total_weight = sum(self.INDICATOR_WEIGHTS.values())
            if abs(total_weight - 1.0) > 1e-6:
                logger.error(f"Indicator weights sum to {total_weight}, must be 1.0")
                return False
            
            if not 0 < self.MIN_SIGNAL_STRENGTH <= 1:
                logger.error("Invalid MIN_SIGNAL_STRENGTH")
                return False
            
            if not 0 < self.MIN_CONFIDENCE <= 100:
                logger.error("Invalid MIN_CONFIDENCE")
                return False
            
            logger.info("Configuration validated successfully")
            return True
            
        except Exception as e:
            logger.error(f"Configuration validation error: {e}")
            return False
    
    def get_summary(self) -> str:
        """Get configuration summary for logging."""
        return (
            f"Trading Config:\n"
            f"  Data Points: {self.MIN_DATA_POINTS}\n"
            f"  Candle Interval: {self.CANDLE_INTERVAL}s\n"
            f"  Signal Strength: {self.MIN_SIGNAL_STRENGTH}\n"
            f"  Confidence: {self.MIN_CONFIDENCE}%\n"
            f"  Cooldown: {self.COOLDOWN_SECONDS}s\n"
            f"  Indicators: {', '.join(self.INDICATOR_WEIGHTS.keys())}"
        )

# Create global config instance
CONFIG = TradingConfig()
