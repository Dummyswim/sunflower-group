"""
Configuration module with CONFIG instance properly defined.
"""
import os
import logging
from typing import Dict, Any
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

def get_env_var(var_name: str, default: str = "") -> str:
    """Safely get environment variable."""
    value = os.getenv(var_name, default)
    if not value and not default:
        logger.warning(f"Environment variable {var_name} not set")
    return value

@dataclass
class TradingConfig:
    """Configuration with all required attributes."""
    
    # Credentials
    telegram_token_b64: str = field(default_factory=lambda: get_env_var("TELEGRAM_TOKEN_B64"))
    telegram_chat_id: str = field(default_factory=lambda: get_env_var("TELEGRAM_CHAT_ID"))
    
    # WebSocket credentials (matching websocket_client.py usage)
    DHAN_ACCESS_TOKEN_B64: str = field(default_factory=lambda: get_env_var("DHAN_TOKEN_B64"))
    DHAN_CLIENT_ID_B64: str = field(default_factory=lambda: get_env_var("DHAN_CLIENT_B64"))
    
    # Market Configuration
    NIFTY_EXCHANGE_SEGMENT: int = 0
    NIFTY_SECURITY_ID: int = 13
    
    # Data Management
    MIN_DATA_POINTS: int = 200
    MAX_BUFFER_SIZE: int = 10000
    CANDLE_INTERVAL: int = 60
    
    # Price Validation
    PRICE_SANITY_MIN: float = 10000.0
    PRICE_SANITY_MAX: float = 30000.0
    
    # Alert Management
    COOLDOWN_SECONDS: int = 300
    MIN_SIGNAL_STRENGTH: float = 0.6
    MIN_CONFIDENCE: int = 65
    MIN_ACTIVE_INDICATORS: int = 4
    
    # Signal Duration
    SIGNAL_SUSTAIN_THRESHOLD: float = 0.7
    SIGNAL_ACCURACY_WINDOW: int = 10
    
    # Indicator Weights
    INDICATOR_WEIGHTS: Dict[str, float] = field(default_factory=lambda: {
        "ichimoku": 0.20,
        "stochastic": 0.15,
        "obv": 0.15,
        "bollinger": 0.15,
        "adx": 0.20,
        "atr": 0.15
    })
    
    # Indicator Parameters
    ICHIMOKU_PARAMS: Dict = field(default_factory=lambda: {
        "tenkan_period": 9,
        "kijun_period": 26,
        "senkou_span_b_period": 52,
        "displacement": 26
    })
    
    STOCHASTIC_PARAMS: Dict = field(default_factory=lambda: {
        "k_period": 14,
        "d_period": 3,
        "smooth_k": 3,
        "overbought": 80,
        "oversold": 20
    })
    
    BOLLINGER_PARAMS: Dict = field(default_factory=lambda: {
        "period": 20,
        "num_std": 2
    })
    
    ADX_PARAMS: Dict = field(default_factory=lambda: {
        "period": 14,
        "strong_trend": 25,
        "weak_trend": 20
    })
    
    ATR_PARAMS: Dict = field(default_factory=lambda: {
        "period": 14,
        "multiplier": 1.5
    })
    
    OBV_PARAMS: Dict = field(default_factory=lambda: {
        "ema_period": 20
    })
    
    # Charts
    alert_with_charts: bool = True
    ENABLE_CHARTS: bool = True
    CHART_WIDTH: int = 14
    CHART_HEIGHT: int = 10
    CHART_DPI: int = 100
    CHART_STYLE: str = "seaborn-v0_8-darkgrid"
    
    # Logging
    log_level: str = "INFO"
    log_file: str = "logs/trading_system.log"
    LOG_FILE: str = "logs/trading_system.log"  # Compatibility
    LOG_LEVEL: str = "INFO"  # Compatibility
    
    # Monitoring
    cooldown_seconds: int = 300  # lowercase version
    min_signal_strength: float = 0.6  # lowercase version
    min_confidence: int = 65  # lowercase version
    
    def validate(self) -> bool:
        """Validate configuration."""
        try:
            errors = []
            
            # Check credentials
            if not self.DHAN_ACCESS_TOKEN_B64:
                errors.append("Missing DHAN_TOKEN_B64")
            if not self.DHAN_CLIENT_ID_B64:
                errors.append("Missing DHAN_CLIENT_B64")
            if not self.telegram_token_b64:
                errors.append("Missing TELEGRAM_TOKEN_B64")
            if not self.telegram_chat_id:
                errors.append("Missing TELEGRAM_CHAT_ID")
            
            # Validate weights sum
            weight_sum = sum(self.INDICATOR_WEIGHTS.values())
            if abs(weight_sum - 1.0) > 0.001:
                errors.append(f"Indicator weights sum to {weight_sum}, not 1.0")
            
            # Validate price range
            if self.PRICE_SANITY_MIN >= self.PRICE_SANITY_MAX:
                errors.append("Invalid price sanity range")
            
            if errors:
                for error in errors:
                    logger.error(f"Config validation error: {error}")
                return False
            
            logger.info("Configuration validated successfully")
            return True
            
        except Exception as e:
            logger.error(f"Validation error: {e}")
            return False

# Create global CONFIG instance that can be imported
CONFIG = TradingConfig()

# Also expose individual items for backward compatibility
DHAN_ACCESS_TOKEN_B64 = CONFIG.DHAN_ACCESS_TOKEN_B64
DHAN_CLIENT_ID_B64 = CONFIG.DHAN_CLIENT_ID_B64
TELEGRAM_BOT_TOKEN_B64 = CONFIG.telegram_token_b64
TELEGRAM_CHAT_ID = CONFIG.telegram_chat_id
NIFTY_EXCHANGE_SEGMENT = CONFIG.NIFTY_EXCHANGE_SEGMENT
NIFTY_SECURITY_ID = CONFIG.NIFTY_SECURITY_ID
MIN_DATA_POINTS = CONFIG.MIN_DATA_POINTS
MAX_BUFFER_SIZE = CONFIG.MAX_BUFFER_SIZE
PRICE_SANITY_MIN = CONFIG.PRICE_SANITY_MIN
PRICE_SANITY_MAX = CONFIG.PRICE_SANITY_MAX
COOLDOWN_SECONDS = CONFIG.COOLDOWN_SECONDS
MIN_SIGNAL_STRENGTH = CONFIG.MIN_SIGNAL_STRENGTH
MIN_CONFIDENCE = CONFIG.MIN_CONFIDENCE
MIN_ACTIVE_INDICATORS = CONFIG.MIN_ACTIVE_INDICATORS
INDICATOR_WEIGHTS = CONFIG.INDICATOR_WEIGHTS
ICHIMOKU_PARAMS = CONFIG.ICHIMOKU_PARAMS
STOCHASTIC_PARAMS = CONFIG.STOCHASTIC_PARAMS
BOLLINGER_PARAMS = CONFIG.BOLLINGER_PARAMS
ADX_PARAMS = CONFIG.ADX_PARAMS
ATR_PARAMS = CONFIG.ATR_PARAMS
OBV_PARAMS = CONFIG.OBV_PARAMS
LOG_FILE = CONFIG.LOG_FILE
LOG_LEVEL = CONFIG.LOG_LEVEL
