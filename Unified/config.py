"""
Enhanced configuration with multi-timeframe support and optimized parameters
"""
import os
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class UnifiedTradingConfig:
    """Optimized trading configuration for 5m and 15m timeframes."""
    
    # ============== CREDENTIALS ==============
    telegram_token_b64: str = field(default_factory=lambda: os.getenv("TELEGRAM_TOKEN_B64", ""))
    telegram_chat_id: str = field(default_factory=lambda: os.getenv("TELEGRAM_CHAT_ID", ""))
    dhan_access_token_b64: str = field(default_factory=lambda: os.getenv("DHAN_TOKEN_B64", ""))
    dhan_client_id_b64: str = field(default_factory=lambda: os.getenv("DHAN_CLIENT_B64", ""))
    
    # ============== MARKET CONFIGURATION ==============
    nifty_exchange_segment: str = "IDX_I"
    nifty_security_id: int = 13
    price_sanity_min: float = 15000.0
    price_sanity_max: float = 35000.0
    
    # ============== DATA MANAGEMENT ==============
    candle_interval_seconds: int = 300  # 5 minutes
    candle_interval_seconds_15m: int = 900  # 15 minutes
    min_data_points: int = 30  # Minimum for indicator calculation
    max_buffer_size: int = 10000
    max_candles_stored: int = 500
    
    # ============== PERSISTENT STORAGE ==============
    use_persistent_storage: bool = True
    rolling_window_days_5m: int = 30
    rolling_window_days_15m: int = 60
    auto_save_interval: int = 60  # Save every 60 seconds
    
    # ============== ALERT MANAGEMENT (UPDATED) ==============
    base_cooldown_seconds: int = 30
    min_signal_strength: float = 0.10  # Slightly decreased from 0.30
    min_confidence: float = 50.0  # RAISED FROM 40 to 50
    min_active_indicators: int = 2  # REDUCED FROM 3 to 2
    confidence_hard_floor: float = 25.0  # NEW: Hard floor
    
    # Dynamic cooldown factors
    strong_signal_cooldown_factor: float = 0.5
    weak_signal_cooldown_factor: float = 1.5
    
    # ============== PRICE ACTION VALIDATION (NEW) ==============
    price_action_validation: bool = True
    price_action_lookback: int = 3  # Check last 3 candles
    price_action_min_body_ratio: float = 0.3  # Min body/total ratio for strong candle
    
    # ============== SIGNAL VALIDATION ==============
    signal_validation_score: float = 0.4
    trend_alignment_threshold: float = 0.6
    multi_timeframe_alignment: bool = True  # NEW
    
    # ============== RISK MANAGEMENT ==============
    stop_loss_percentage: float = 1.0  # 1% stop loss
    take_profit_percentage: float = 2.0  # 2% take profit
    trailing_stop_percentage: float = 0.5  # 0.5% trailing stop
    
    # ============== SIGNAL DURATION ==============
    signal_sustain_threshold: float = 0.6
    signal_accuracy_window: int = 10
    max_signal_duration_minutes: int = 15
    min_signal_duration_minutes: int = 5
    
    # ============== OPTIMIZED RSI PARAMETERS (5m/15m) ==============
    # rsi_params_5m: Dict = field(default_factory=lambda: {
    #     "period": 10,  # OPTIMIZED FOR 5-MIN
    #     "overbought": 65,  # LOWERED FROM 70
    #     "oversold": 35,  # RAISED FROM 30
    #     "neutral_zone": (40, 60)
    # })
    
    rsi_params_5m: Dict = field(default_factory=lambda: {
        "period": 9,      # Faster for scalping (was 10)
        "overbought": 70, # Higher for indices (was 65)
        "oversold": 30,   # Lower for indices (was 35)
        "neutral_zone": (40, 60)
    })


    rsi_params_15m: Dict = field(default_factory=lambda: {
        "period": 14,  # Standard for 15-min
        "overbought": 65,
        "oversold": 35,
        "neutral_zone": (40, 60)
    })
    
    # ============== OPTIMIZED MACD PARAMETERS ==============
    macd_params_5m: Dict = field(default_factory=lambda: {
        "fastperiod": 8,  # OPTIMIZED FROM 12
        "slowperiod": 17,  # OPTIMIZED FROM 26
        "signalperiod": 9
    })
    
    macd_params_15m: Dict = field(default_factory=lambda: {
        "fastperiod": 12,
        "slowperiod": 26,
        "signalperiod": 9
    })
    
    # ============== OPTIMIZED EMA PARAMETERS ==============
    # ema_params_5m: Dict = field(default_factory=lambda: {
    #     "short_period": 5,  # OPTIMIZED FOR 5-MIN
    #     "medium_period": 10,
    #     "long_period": 20
    # })
    

    ema_params_5m: Dict = field(default_factory=lambda: {
        "short_period": 5,   # keep fast trigger for scalps
        "medium_period": 13, # wider mid to reduce flip-flops
        "long_period": 34    # slower long to anchor trend
    })

    
    ema_params_15m: Dict = field(default_factory=lambda: {
        "short_period": 9,  # OPTIMIZED FOR 15-MIN
        "medium_period": 21,
        "long_period": 50
    })
    
    # ============== OPTIMIZED BOLLINGER PARAMETERS ==============
    bollinger_params_5m: Dict = field(default_factory=lambda: {
        "period": 10,  # Faster response for 5-min scalping
        "stddev": 2.0  # Standard deviation for indices (not 1.5)
    })
    
    bollinger_params_15m: Dict = field(default_factory=lambda: {
        "period": 20,
        "stddev": 2.0
    })
    
    # ============== KELTNER PARAMETERS ==============
    keltner_params: Dict = field(default_factory=lambda: {
        "period": 20,
        "multiplier": 2.0
    })
    
    # ============== SUPERTREND PARAMETERS ==============
    supertrend_params_5m: Dict = field(default_factory=lambda: {
        "period": 7,  # OPTIMIZED FOR 5-MIN
        "multiplier": 1.5
    })
    
    supertrend_params_15m: Dict = field(default_factory=lambda: {
        "period": 10,
        "multiplier": 3.0
    })
    
    
    # ============== INDICATOR WEIGHTS (Leading vs Lagging) ==============
    # indicator_weights: Dict[str, float] = field(default_factory=lambda: {
    #     # Leading indicators (predict) - 50%
    #     "rsi": 0.15,        # Leading
    #     "macd": 0.20,       # Leading
    #     "bollinger": 0.15,  # Leading
        
    #     # Lagging indicators (confirm) - 50%
    #     "ema": 0.20,        # Lagging
    #     "supertrend": 0.20, # Lagging
    #     "keltner": 0.10,    # Mixed
    # })

    indicator_weights: Dict[str, float] = field(default_factory=lambda: {
        # Leading indicators (predict)
        "rsi": 0.15,
        "macd": 0.22,       # +0.02 (momentum)
        "bollinger": 0.10,  # −0.05 (reduce over-penalizing continuation)
        
        # Lagging/confirm
        "ema": 0.25,        # +0.05 (trend structure)
        "supertrend": 0.18, # −0.02 (less lag weight)
        "keltner": 0.10
    })

    # ============== INDICATOR GROUPS ==============
    indicator_groups: Dict[str, list] = field(default_factory=lambda: {
        "leading": ["rsi", "macd", "bollinger"],  # Predict
        "lagging": ["ema", "supertrend"],         # Confirm
        "volatility": ["bollinger", "keltner"],   # Volatility
    })
    
    # Minimum agreement within groups
    min_group_consensus: float = 0.6
    
    # ============== TREND ALIGNMENT PARAMETERS ==============
    trend_lookback: int = 20
    trend_ma_period: int = 20
    trend_strength_threshold: float = 0.02
    
    # ============== MULTI-TIMEFRAME SETTINGS (NEW) ==============
    mtf_enabled: bool = True
    mtf_primary_timeframe: str = "5m"
    mtf_higher_timeframe: str = "15m"
    mtf_alignment_required: bool = True
    mtf_trend_weight: float = 0.7  # Weight of higher timeframe trend
    
    # ============== OTHER CONFIGURATIONS ==============
    enable_charts: bool = True
    chart_style: str = 'seaborn-v0_8'
    chart_save_path: str = 'images/'
    chart_dpi: int = 100
    chart_candles_to_show: int = 50
    
    enable_health_monitor: bool = True
    health_check_interval: int = 60
    max_reconnect_attempts: int = 10
    reconnect_delay_base: int = 5
    
    track_performance: bool = True
    performance_window: int = 100
    performance_report_interval: int = 3600
    
    log_file: str = "logs/unified_trading.log"
    log_level: str = "INFO"
    log_rotation_size: int = 10485760
    log_backup_count: int = 5
    
    enable_duration_prediction: bool = True
    enable_market_structure: bool = True
    enable_signal_validation: bool = True
    enable_risk_management: bool = True
    
    def get_rsi_params(self, timeframe: str) -> Dict:
        """Get RSI parameters for timeframe."""
        return self.rsi_params_5m if timeframe == "5m" else self.rsi_params_15m
    
    def get_macd_params(self, timeframe: str) -> Dict:
        """Get MACD parameters for timeframe."""
        return self.macd_params_5m if timeframe == "5m" else self.macd_params_15m
    
    def get_ema_params(self, timeframe: str) -> Dict:
        """Get EMA parameters for timeframe."""
        return self.ema_params_5m if timeframe == "5m" else self.ema_params_15m
    
    def get_bollinger_params(self, timeframe: str) -> Dict:
        """Get Bollinger parameters for timeframe."""
        return self.bollinger_params_5m if timeframe == "5m" else self.bollinger_params_15m
    
    def get_supertrend_params(self, timeframe: str) -> Dict:
        """Get Supertrend parameters for timeframe."""
        return self.supertrend_params_5m if timeframe == "5m" else self.supertrend_params_15m
    
    def validate(self) -> bool:
        """Validate configuration parameters."""
        try:
            # Check credentials
            if not all([self.telegram_token_b64, self.telegram_chat_id,
                       self.dhan_access_token_b64, self.dhan_client_id_b64]):
                logger.error("Missing required credentials")
                return False
            
            # Validate indicator weights sum to 1.0
            weight_sum = sum(self.indicator_weights.values())
            if abs(weight_sum - 1.0) > 0.001:
                logger.warning(f"Indicator weights sum to {weight_sum}, normalizing...")
                # Normalize weights
                for key in self.indicator_weights:
                    self.indicator_weights[key] /= weight_sum
                logger.info("Indicator weights normalized to 1.0")
            
            # Validate ranges
            if not 0 < self.min_signal_strength <= 1:
                logger.error(f"Invalid min_signal_strength: {self.min_signal_strength}")
                return False
            
            if not 0 < self.min_confidence <= 100:
                logger.error(f"Invalid min_confidence: {self.min_confidence}")
                return False
            
            # Create required directories
            for path in [self.log_file, self.chart_save_path, "data"]:
                Path(path).parent.mkdir(parents=True, exist_ok=True)
            
            logger.info("Configuration validated successfully")
            logger.info(f"✓ Min Confidence: {self.min_confidence}%")
            logger.info(f"✓ Min Active Indicators: {self.min_active_indicators}")
            logger.info(f"✓ Price Action Validation: {self.price_action_validation}")
            logger.info(f"✓ Multi-Timeframe Alignment: {self.multi_timeframe_alignment}")
            logger.info(f"✓ Persistent Storage: {self.use_persistent_storage}")
            
            return True
            
        except Exception as e:
            logger.error(f"Configuration validation error: {e}")
            return False

    def get_summary(self) -> str:
        """Get configuration summary."""
        return f"""
========== ENHANCED CONFIGURATION ==========
Market: NIFTY50 (ID: {self.nifty_security_id})
Primary Timeframe: {self.mtf_primary_timeframe}
Higher Timeframe: {self.mtf_higher_timeframe}
Signal Confidence: {self.min_confidence}% (Floor: {self.confidence_hard_floor}%)
Min Active Indicators: {self.min_active_indicators}
Price Action Validation: {self.price_action_validation}
Multi-Timeframe: {self.multi_timeframe_alignment}
Persistent Storage: {self.use_persistent_storage}
==========================================
"""

# Create global config instance
config = UnifiedTradingConfig()
