"""
Enhanced configuration with multi-timeframe support and optimized parameters
"""
import os
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path
import math


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
    sensex_exchange_segment: str = "IDX_I"
    sensex_security_id: int = 51
    price_sanity_min: float = 75000.0
    price_sanity_max: float = 90000.0
    
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
    trend_alignment_threshold: float = 0.6  # More reasonable threshold
    debug_mode: bool = True  # Enable debug logging
    multi_timeframe_alignment: bool = True  # NEW

    # ============== RISK MANAGEMENT ============== 
    stop_loss_percentage: float = 1.0 # 1% stop loss 
    take_profit_percentage: float = 2.0 # 2% take profit 
    trailing_stop_percentage: float = 0.5 # 0.5% trailing stop 
    min_risk_reward_floor: float = 1.0 # NEW: hard floor for actionable alerts
    # Minimum volatility range as % of price for 5m scalps (keeps SL/TP realistic) 
    min_volatility_range_pct: float = 0.002 # 0.20%
    
    # ============== SIGNAL DURATION ==============
    signal_sustain_threshold: float = 0.6
    signal_accuracy_window: int = 10
    max_signal_duration_minutes: int = 15
    min_signal_duration_minutes: int = 5
    
    
    # Weak‑MTF band extra penalty (0.50–0.65); 0.0 keeps current behavior
    weak_mtf_band_extra_penalty: float = 0.01


    # Volatility caps for intraday scalps (constrain TP/SL by local vol)
    tp_volatility_cap_multiple: float = 1.8  # TP cannot be farther than 1.8× recent 5m volatility range
    sl_volatility_cap_multiple: float = 1.0  # SL cannot be farther than 1.0× recent 5m volatility range

    # Stricter gate for pre-close predictions (forming bar is noisier)
    preclose_min_mtf_score: float = 0.60 # Reduced from 0.6

    # Require more breadth for directional alerts; candidates can use the lower global value
    min_active_indicators_for_alert: int = 4


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
        "rsi": 0.17,
        "macd": 0.26,       # +0.02 (momentum)
        "bollinger": 0.10,  # −0.05 (reduce over-penalizing continuation)
        
        # Lagging/confirm
        "ema": 0.19,        # +0.05 (trend structure)
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
    
    
    # ============== EXTREME CONTEXT & WITHDRAW MONITOR ==============
    extreme_price_pos_hi: float = 0.95
    extreme_price_pos_lo: float = 0.05
    extreme_rsi_5m: float = 85.0
    extreme_rsi_15m: float = 75.0
    confidence_cap_at_extreme: float = 78.0  # cap unless breakout evidence

    withdraw_monitor_enabled: bool = True
    withdraw_confidence_min: float = 70.0
    withdraw_min_dwell_sec: int = 10        # wait at least this much after open
    withdraw_window_sec: int = 300          # one 5m candle
    withdraw_adverse_points: float = 12.0   # e.g., sensex pts
    withdraw_adverse_pct_of_tp: float = 0.40
    withdraw_adverse_ticks_cluster: int = 3
    withdraw_check_interval_sec: int = 5

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
    
    log_file: str = "logs/unified_trading_1.log"
    log_level: str = "INFO"
    log_rotation_size: int = 10485760
    log_backup_count: int = 5
    
    enable_duration_prediction: bool = True
    enable_market_structure: bool = True
    enable_signal_validation: bool = True
    enable_risk_management: bool = True
    

    # Risk tapering for burst momentum setups
    enable_rr_taper: bool = True
    rr_taper_floor: float = 0.80           # min R:R for high-confidence bursts
    rr_taper_confidence_min: float = 0.75  # only taper when conf >= 75%
    rr_taper_burst_strength: float = 0.30  # |weighted_score| >= 0.30 defines 'burst'

    # ATR settings for volatility-scaled SL/TP
    atr_period: int = 14
    atr_multiplier_sl: float = 0.6         # stop = 0.6*ATR for bursts
    atr_multiplier_tp: float = 1.2         # target = 1.2*ATR for bursts

    # Conditional breadth policy
    enable_conditional_breadth: bool = True
    conditional_breadth_mtf_min: float = 0.65

    # MTF adaptive thresholds
    mtf_threshold_available: float = 0.50
    mtf_threshold_limited: float = 0.60
    mtf_consistency_window: int = 5
    mtf_consistency_adjust: float = 0.05   # slide threshold ±0.05

    # Feature flags
    enable_symmetric_mtf_conflict: bool = True
    enable_rsi50_confirmation: bool = True
    enable_momentum_slope_guard: bool = True


    # Logging and release-gate feature flags (high-verbosity until stable)
    verbose_logging: bool = True
    require_expansion_for_promotion: bool = True


    # ============== Dynamic MTF threshold tuning ============== 
    mtf_dynamic_enable: bool = True
    mtf_dynamic_min: float = 0.40
    mtf_dynamic_max: float = 0.70
    mtf_adj_trend_agree: float = -0.05
    mtf_adj_ranging_no_room: float = +0.05
    mtf_adj_squeeze: float = -0.05
    mtf_adj_extreme_rsi: float = +0.05
    mtf_adj_open_close: float = +0.03
    mtf_squeeze_bandwidth: float = 8.0
    mtf_rsi_extreme_buy: float = 75.0
    mtf_rsi_extreme_sell: float = 25.0


    
    # Pre-close alerts (analyze about-to-close bar before boundary) 
    preclose_lead_seconds: int = 15 # analyze N seconds before close (min 5s)
    
    
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
            

            # Finite checks for critical floats
            crit_pairs = [
                ('min_signal_strength', self.min_signal_strength),
                ('min_confidence', self.min_confidence),
                ('preclose_min_mtf_score', self.preclose_min_mtf_score),
                ('weak_mtf_band_extra_penalty', self.weak_mtf_band_extra_penalty),
            ]
            for name, val in crit_pairs:
                try:
                    v = float(val)
                except Exception:
                    logger.error(f"{name} is not numeric: {val}")
                    return False
                if not math.isfinite(v):
                    logger.error(f"{name} is NaN/Inf: {val}")
                    return False
                        
                        
            
            # Create required directories (file vs directory safe)
            Path(self.log_file).parent.mkdir(parents=True, exist_ok=True)
            Path(self.chart_save_path).mkdir(parents=True, exist_ok=True)
            Path("data").mkdir(parents=True, exist_ok=True)


            logger.info("Configuration validated successfully") 
            
            logger.info("[LOG] Verbose logging mode: %s", getattr(self, 'verbose_logging', True))
            logger.info("[GATE] require_expansion_for_promotion=%s", getattr(self, 'require_expansion_for_promotion', True))
            logger.info("[CFG] preclose_min_mtf_score=%.2f | weak_mtf_band_extra_penalty=%.3f | preclose_lead_seconds=%ds",
                        self.preclose_min_mtf_score, self.weak_mtf_band_extra_penalty, self.preclose_lead_seconds)
            logger.info("continue logging")

            
            logger.info(f"✓ Min Confidence: {self.min_confidence}%") 
            logger.info(f"✓ Min Active Indicators: {self.min_active_indicators}") 
            logger.info(f"✓ Price Action Validation: {self.price_action_validation}") 
            logger.info(f"✓ Multi-Timeframe Alignment: {self.multi_timeframe_alignment}") 
            logger.info(f"✓ Persistent Storage: {self.use_persistent_storage}")
            
            logger.info(f"✓ R:R Floor: {self.min_risk_reward_floor:.2f}")
            mv = float(getattr(self, 'min_volatility_range_pct', 0.002))
            logger.info(f"✓ Min Volatility Range (pct): {mv:.3f}")
            logger.info(f"✓ MTF Threshold (at-close): {self.trend_alignment_threshold:.2f}")


            
            return True
            
        except Exception as e:
            logger.error(f"Configuration validation error: {e}")
            return False

    def get_summary(self) -> str:
        """Get configuration summary."""
        return f"""
========== ENHANCED CONFIGURATION ==========
Market: SENSEX (ID: {self.sensex_security_id})
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
