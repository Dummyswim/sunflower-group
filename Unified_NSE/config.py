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
    nifty_exchange_segment: str = "IDX_I"
    nifty_security_id: int = 13
    price_sanity_min: float = 15000.0
    price_sanity_max: float = 35000.0
    
    # ============== DATA MANAGEMENT ==============
    candle_interval_seconds: int = 300
    candle_interval_seconds_15m: int = 900
    min_data_points: int = 30
    max_buffer_size: int = 10000
    max_candles_stored: int = 500
    
    # ============== PERSISTENT STORAGE ==============
    use_persistent_storage: bool = True
    rolling_window_days_5m: int = 30
    rolling_window_days_15m: int = 60
    auto_save_interval: int = 60
    
    # ============== ALERT MANAGEMENT ==============
    base_cooldown_seconds: int = 30
    min_signal_strength: float = 0.10
    min_confidence: float = 50.0
    min_active_indicators: int = 4
    min_active_indicators_calibration: int = 2
    confidence_hard_floor: float = 25.0
    strong_signal_cooldown_factor: float = 0.5
    weak_signal_cooldown_factor: float = 1.5
    
    # ============== PRICE ACTION VALIDATION ==============
    price_action_validation: bool = True
    price_action_lookback: int = 3
    price_action_min_body_ratio: float = 0.3
    
    # ============== SIGNAL VALIDATION ==============
    signal_validation_score: float = 0.4
    trend_alignment_threshold: float = 0.6
    debug_mode: bool = True
    multi_timeframe_alignment: bool = True
    
    # ============== RISK MANAGEMENT ==============
    stop_loss_percentage: float = 1.0
    take_profit_percentage: float = 2.0
    trailing_stop_percentage: float = 0.5
    min_risk_reward_floor: float = 1.0
    min_volatility_range_pct: float = 0.002
    
    # ============== SIGNAL DURATION ==============
    signal_sustain_threshold: float = 0.6
    signal_accuracy_window: int = 10
    max_signal_duration_minutes: int = 15
    min_signal_duration_minutes: int = 5
    
    weak_mtf_band_extra_penalty: float = 0.01
    tp_volatility_cap_multiple: float = 1.8
    sl_volatility_cap_multiple: float = 1.0
    preclose_min_mtf_score: float = 0.60
    min_active_indicators_for_alert: int = 4
    
    # ============== MOMENTUM EXHAUSTION ==============
    enable_momentum_exhaustion: bool = True
    momentum_exhaustion_rsi_threshold: float = 75.0
    momentum_exhaustion_rsi_low: float = 25.0
    momentum_exhaustion_macd_bars: int = 3
    momentum_exhaustion_divergence_check: bool = True
    
    # ============== OPTIMIZED RSI PARAMETERS ==============
    rsi_params_5m: Dict = field(default_factory=lambda: {
        "period": 9,
        "overbought": 70,
        "oversold": 30,
        "neutral_zone": (40, 60)
    })
    
    rsi_params_15m: Dict = field(default_factory=lambda: {
        "period": 14,
        "overbought": 65,
        "oversold": 35,
        "neutral_zone": (40, 60)
    })
    
    # ============== OPTIMIZED MACD PARAMETERS ==============
    macd_params_5m: Dict = field(default_factory=lambda: {
        "fastperiod": 8,
        "slowperiod": 17,
        "signalperiod": 9
    })
    
    macd_params_15m: Dict = field(default_factory=lambda: {
        "fastperiod": 12,
        "slowperiod": 26,
        "signalperiod": 9
    })
    
    # ============== OPTIMIZED EMA PARAMETERS ==============
    ema_params_5m: Dict = field(default_factory=lambda: {
        "short_period": 5,
        "medium_period": 13,
        "long_period": 34
    })
    
    ema_params_15m: Dict = field(default_factory=lambda: {
        "short_period": 9,
        "medium_period": 21,
        "long_period": 50
    })
    
    # ============== OPTIMIZED BOLLINGER PARAMETERS ==============
    bollinger_params_5m: Dict = field(default_factory=lambda: {
        "period": 10,
        "stddev": 2.0
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
        "period": 7,
        "multiplier": 1.5
    })
    
    supertrend_params_15m: Dict = field(default_factory=lambda: {
        "period": 10,
        "multiplier": 3.0
    })
    
    # ============== INDICATOR WEIGHTS ==============
    indicator_weights: Dict[str, float] = field(default_factory=lambda: {
        "rsi": 0.25,
        "macd": 0.15,
        "bollinger": 0.12,
        "ema": 0.20,
        "supertrend": 0.18,
        "keltner": 0.10
    })
    
    # ============== INDICATOR GROUPS ==============
    indicator_groups: Dict[str, list] = field(default_factory=lambda: {
        "leading": ["rsi", "macd", "bollinger"],
        "lagging": ["ema", "supertrend"],
        "volatility": ["bollinger", "keltner"],
    })
    
    min_group_consensus: float = 0.6
    
    # ============== TREND ALIGNMENT PARAMETERS ==============
    trend_lookback: int = 20
    trend_ma_period: int = 20
    trend_strength_threshold: float = 0.02
    
    # ============== MULTI-TIMEFRAME SETTINGS ==============
    mtf_enabled: bool = True
    mtf_primary_timeframe: str = "5m"
    mtf_higher_timeframe: str = "15m"
    mtf_alignment_required: bool = True
    mtf_trend_weight: float = 0.7
    
    # ============== EXTREME CONTEXT & WITHDRAW MONITOR ==============
    extreme_price_pos_hi: float = 0.95
    extreme_price_pos_lo: float = 0.05
    extreme_rsi_5m: float = 85.0
    extreme_rsi_15m: float = 75.0
    confidence_cap_at_extreme: float = 78.0
    
    withdraw_monitor_enabled: bool = True
    withdraw_confidence_min: float = 70.0
    withdraw_min_dwell_sec: int = 10
    withdraw_window_sec: int = 300
    withdraw_adverse_points: float = 12.0
    withdraw_adverse_pct_of_tp: float = 0.40
    withdraw_adverse_ticks_cluster: int = 3
    withdraw_check_interval_sec: int = 5
    
    # ============== HEALTH MONITORING ==============
    enable_health_monitor: bool = True
    health_check_interval: int = 60
    max_reconnect_attempts: int = 10
    reconnect_delay_base: int = 5
    
    # Watchdog (used by WebSocket handler)
    data_stall_seconds: int = 15
    data_stall_reconnect_seconds: int = 30
    
    # ============== PERFORMANCE TRACKING ==============
    track_performance: bool = True
    performance_window: int = 100
    performance_report_interval: int = 3600
    
    # ============== LOGGING ==============
    log_file: str = "logs/unified_trading_1.log"
    log_level: str = "INFO"
    log_rotation_size: int = 10485760
    log_backup_count: int = 5
    
    # ============== FEATURE FLAGS ==============
    enable_duration_prediction: bool = True
    enable_market_structure: bool = True
    enable_signal_validation: bool = True
    enable_risk_management: bool = True
    
    # ============== RISK TAPERING ==============
    enable_rr_taper: bool = True
    rr_taper_floor: float = 0.80
    rr_taper_confidence_min: float = 64.0
    rr_taper_burst_strength: float = 0.30
    rr_taper_confidence_soft_min: float = 0.64
    rr_taper_mtf_min_strong: float = 0.70
    
    # ATR settings
    atr_period: int = 14
    atr_multiplier_sl: float = 0.6
    atr_multiplier_tp: float = 1.2
    
    # ============== CONDITIONAL BREADTH ==============
    enable_conditional_breadth: bool = True
    conditional_breadth_mtf_min: float = 0.65
    
    # ============== MTF ADAPTIVE THRESHOLDS ==============
    mtf_threshold_available: float = 0.50
    mtf_threshold_limited: float = 0.60
    mtf_consistency_window: int = 5
    mtf_consistency_adjust: float = 0.05
    
    # ============== DYNAMIC MTF TUNING ==============
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
    
    # ============== ADDITIONAL FLAGS ==============
    enable_symmetric_mtf_conflict: bool = True
    enable_rsi50_confirmation: bool = True
    enable_momentum_slope_guard: bool = True
    verbose_logging: bool = True
    require_expansion_for_promotion: bool = True
    
    # ============== LOSING STREAK PROTECTION ==============
    enable_losing_streak_protection: bool = True
    losing_streak_threshold: int = 3
    losing_streak_confidence_boost: float = 10.0
    losing_streak_pause_after: int = 5
    losing_streak_pause_minutes: int = 30
    losing_streak_reset_after_wins: int = 2
    
    # ============== REGIME ADAPTIVE R:R ==============
    rr_floor_use_adaptive: bool = True
    rr_floor_ranging: float = 0.60
    rr_floor_trending: float = 1.00
    
    # ============== PRE-CLOSE SETTINGS ==============
    enable_packet_checksum_validation: bool = False
    preclose_lead_seconds: int = 15
    preclose_completion_buffer_sec: int = 1
    
    # ============== HIT-RATE TRACKING ==============
    hitrate_base_path: str = "logs/hitrate"
    hitrate_rotate_daily: bool = True
    hitrate_keep_days: int = 60
    hitrate_symlink_latest: bool = True
    
    # ============== REGIME DETECTION ==============
    mtf_threshold_pullback: float = 0.40
    mtf_threshold_reversal: float = 0.60
    mtf_regime_detection: bool = True
    
    # ============== CANDLESTICK PATTERNS ==============
    enable_talib_patterns: bool = True
    candlestick_patterns: list = field(default_factory=lambda: [
        "CDLINVERTEDHAMMER",
        "CDLPIERCING",
        "CDLHARAMI",
        "CDL3WHITESOLDIERS",
        "CDL3BLACKCROWS",
        "CDLDARKCLOUDCOVER",
        "CDLABANDONEDBABY",
        "CDLSPINNINGTOP",
        "CDLTRISTAR",
        "CDLSTICKSANDWICH",
        "CDLENGULFING",
        "CDLHAMMER",
        "CDLSHOOTINGSTAR"
    ])
    
    enable_custom_tweezer: bool = True
    tweezer_tolerance_bps: float = 5.0
    enable_rounding_patterns: bool = False
    rounding_window: int = 20
    pattern_min_strength: int = 50
    pattern_as_confirmation_only: bool = True
    require_pattern_confirmation: bool = False
    
    # ============== BORDERLINE MTF ==============
    enable_mtf_borderline_soft_allow: bool = True
    mtf_borderline_min: float = 0.55
    mtf_borderline_max: float = 0.60
    mtf_borderline_conf_penalty: float = 12.0
    
    # ============== HTF GUARDS ==============
    buy_guard_htf_enabled: bool = True
    buy_guard_mtf_threshold: float = 0.70
    sell_guard_htf_enabled: bool = True
    sell_guard_mtf_threshold: float = 0.70
    
    slope_soft_allow_min_mag: float = 0.15
    session_ranging_strength_delta: float = 0.02
    
    # ============== OI/PCR INTEGRATION ==============
    enable_oi_integration: bool = True
    oi_context_boost: float = 0.03
    oi_min_change_pct: float = 0.10
    
    # ============== SUPPLY/DEMAND ==============
    enable_supply_demand_integration: bool = True
    sd_zone_distance_bps: float = 8.0
    sd_context_boost: float = 0.03
    
    enable_pattern_location_quality: bool = True
    
    # ==================== 1m NEXT-MINUTE ENGINE (consolidated, deduplicated) ====================
    enable_next_minute_engine: bool = True
    next_minute_predict_second: int = 57
    next_minute_resolve_second: int = 10
    next_minute_optional_alerts: bool = False
    
    # Micro tick buffer and time-bounded windows
    micro_tick_window: int = 200
    micro_window_sec_1m: int = 25
    micro_min_ticks_1m: int = 30
    micro_short_window_sec_1m: int = 8
    micro_short_min_ticks_1m: int = 12
    
    # Micro thresholds and guards
    micro_imbalance_min: float = 0.30
    micro_slope_min: float = 0.15
    micro_noise_sigma_mult: float = 2.0
    micro_persistence_min_checks: int = 1
    micro_macd_slope_soft_min: float = 0.05
    micro_extreme_imb_min: float = 0.60
    quiet_mtf_in_1m: bool = True
    
    # Forecast prior weights (correctness-first)
    next_minute_macd_prior_weight: float = 3.8 # was 2.5
    next_minute_imb_weight: float = 55.0
    next_minute_slope_weight: float = 10.0
    next_minute_rsi_prior_weight: float = 0.09 # was 0.05
    
    # Display and safety guards
    next_minute_forecast_neutral_band: float = 0.03
    next_minute_drift_guard_pct: float = 0.01
    
    # Location extremes and soft MTF demotion
    next_minute_mtf_extreme_pos_hi: float = 0.95
    next_minute_mtf_extreme_pos_lo: float = 0.05
    next_minute_mtf_conf_penalty_pp: float = 8.0
    next_minute_use_soft_mtf: bool = True
    
    # Micro override when tape is very strong
    next_minute_micro_override_imb: float = 0.50
    next_minute_micro_override_slope: float = 0.25
    
    # Dual-window blend and reversal handling
    next_minute_imb_short_weight: float = 0.60
    next_minute_imb_long_weight: float = 0.40
    next_minute_reversal_penalty_pp: float = 8.0
    next_minute_reversal_penalty_cap_pp: float = 14.0
    

    # Significance/acceleration assists
    next_minute_signif_z_min_base: float = 0.90
    next_minute_signif_z_min_trend: float = 0.80
    next_minute_accel_assist_delta: float = 0.40
    next_minute_accel_assist_imb: float = 0.12

        
    # Micro conflict arbitration and guards (1m-only)
    next_minute_boundary_flip_ratio: float = 1.40   # |imbS| must exceed |imbL| by this ratio (or see momΔ)
    next_minute_boundary_flip_mom_delta: float = 0.25
    next_minute_slope_conflict_min: float = 0.45    # min |slope| in both windows to arbitrate
    next_minute_imb_conflict_cap: float = 0.20      # only arbitrate if |imb| <= this
    next_minute_imb_conflict_replacement: float = 0.15
    next_minute_vol_shock_mult: float = 1.60        # stdΔ shock factor vs ref to tighten passes
    next_minute_drift_conflict_mul: float = 2.0     # x drift_guard to neutralize on drift conflict


    
    # Sign-aware extreme tuning
    next_minute_extreme_relax_pp: float = 0.12  # relax threshold by 12pp when fading from the edge
    next_minute_extreme_harden_pp: float = 0.05  # harden threshold by 5pp when trading into edge

    # Slope assist and edge override
    next_minute_slope_assist_min: float = 0.35  # minimum |slope| for slope-assist allowance
    next_minute_edge_override_short_imb: float = 0.20  # min |imbalance_short| at edge for override
    next_minute_edge_override_short_slp: float = 0.35  # min |slope_short| at edge for override

    # Display neutral band (probability) - UPDATED VALUE
    next_minute_forecast_neutral_band_pp: float = 0.02  # ±2pp, was 0.03




    # Online micro-only model for 1m probability (no indicators)
    enable_next_minute_ml: bool = True
    nm_ml_lr: float = 0.03
    nm_ml_l2: float = 0.0005
    nm_ml_blend_alpha: float = 0.70   # final_prob = alpha*rule_prob + (1-alpha)*ml_prob
    nm_ml_assist_enabled: bool = False  # keep False for stability initially
    nm_ml_assist_pp: float = 8.0       # optional assist: add/sub pp to prior if model is very confident
    nm_ml_assist_confident: float = 0.62  # >0.62 or <0.38 qualifies as "confident"

    
    # ============== LIBERAL PRE-GATE THRESHOLDS ==============
    liberal_min_abs_score: float = 0.05
    liberal_min_mtf: float = 0.50
    
    # ============== PIVOT SWIPE SETUP ==============
    enable_pivot_swipe: bool = True
    pivot_swipe_bps_tolerance: float = 6.0
    pivot_swipe_levels: list = field(default_factory=lambda: ["PDH", "PDL", "SWING_5m"])
    pivot_swipe_min_reclaim_closes: int = 1
    pivot_swipe_weight: float = 0.06
    pivot_swipe_as_confirmation_only: bool = True
    
    # ============== IMBALANCE STRUCTURE SETUP ==============
    enable_imbalance_structure: bool = True
    imbalance_c2_min_body_ratio: float = 0.60
    imbalance_min_gap_bars: int = 1
    ema_widen_pair: tuple = (20, 50)
    ema_widen_min_bps: float = 8.0
    imbalance_weight: float = 0.07
    imbalance_as_confirmation_only: bool = True
    
    # ============== CHARTS ==============
    enable_charts: bool = True
    chart_style: str = 'seaborn-v0_8'
    chart_save_path: str = 'images/'
    chart_dpi: int = 100
    chart_candles_to_show: int = 50
    
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
            
            if not (0.0 <= self.weak_mtf_band_extra_penalty <= 0.1):
                logger.error(f"Invalid weak_mtf_band_extra_penalty: {self.weak_mtf_band_extra_penalty}")
                return False
            
            if not (5 <= self.preclose_lead_seconds <= 60):
                logger.error(f"Invalid preclose_lead_seconds: {self.preclose_lead_seconds}")
                return False
            
            # Validate new parameters
            new_params = [
                ('weak_mtf_band_extra_penalty', self.weak_mtf_band_extra_penalty, 0.0, 0.1),
                ('preclose_lead_seconds', self.preclose_lead_seconds, 5, 60),
                ('mtf_borderline_conf_penalty', self.mtf_borderline_conf_penalty, 0.0, 50.0),
            ]
            
            for name, value, min_val, max_val in new_params:
                try:
                    v = float(value)
                except Exception:
                    logger.error(f"{name} is not numeric: {value}")
                    return False
                if not (min_val <= v <= max_val):
                    logger.error(f"Invalid {name}: {value} (must be {min_val}-{max_val})")
                    return False
            
            # Create required directories
            Path(self.log_file).parent.mkdir(parents=True, exist_ok=True)
            Path(self.chart_save_path).mkdir(parents=True, exist_ok=True)
            Path("data").mkdir(parents=True, exist_ok=True)
            
            # Log configuration status
            logger.info("[WS] checksum_validation=%s", getattr(self, 'enable_packet_checksum_validation', False))
            logger.info("Configuration validated successfully")
            logger.info("[LOG] Verbose logging mode: %s", getattr(self, 'verbose_logging', True))
            logger.info("[GATE] require_expansion_for_promotion=%s", getattr(self, 'require_expansion_for_promotion', True))
            logger.info("[CFG] preclose_min_mtf_score=%.2f | weak_mtf_band_extra_penalty=%.3f | preclose_lead_seconds=%ds",
                        self.preclose_min_mtf_score, self.weak_mtf_band_extra_penalty, self.preclose_lead_seconds)
            logger.info("[SOFT-ALLOW] slope_soft_allow_min_mag=%.3f", self.slope_soft_allow_min_mag)
            logger.info("[SESSION] ranging_strength_delta=%.3f", self.session_ranging_strength_delta)
            
            logger.info("[SETUP] PivotSwipe: enabled=%s tol=%.1f bps levels=%s weight=%.2f confirm_only=%s",
                        getattr(self, 'enable_pivot_swipe', True),
                        float(getattr(self, 'pivot_swipe_bps_tolerance', 6.0)),
                        getattr(self, 'pivot_swipe_levels', []),
                        float(getattr(self, 'pivot_swipe_weight', 0.06)),
                        getattr(self, 'pivot_swipe_as_confirmation_only', True))
            
            logger.info("[SETUP] Imbalance: enabled=%s C2_body>=%.2f EMA_pair=%s widen_min=%.1f bps weight=%.2f confirm_only=%s",
                        getattr(self, 'enable_imbalance_structure', True),
                        float(getattr(self, 'imbalance_c2_min_body_ratio', 0.60)),
                        str(getattr(self, 'ema_widen_pair', (20, 50))),
                        float(getattr(self, 'ema_widen_min_bps', 8.0)),
                        float(getattr(self, 'imbalance_weight', 0.07)),
                        getattr(self, 'imbalance_as_confirmation_only', True))
            
            logger.info("[PATTERN] talib=%s | enabled=%s | min_strength=%d | confirm_only=%s | require=%s",
                        True, self.enable_talib_patterns, self.pattern_min_strength,
                        self.pattern_as_confirmation_only, self.require_pattern_confirmation)
            logger.info("[PATTERN] list=%s", self.candlestick_patterns)
            
            logger.info("[OI] integration=%s | boost=%.3f | min_change_pct=%.2f%%",
                        getattr(self, 'enable_oi_integration', True),
                        float(getattr(self, 'oi_context_boost', 0.03)),
                        float(getattr(self, 'oi_min_change_pct', 0.10)))
            
            # NEW: Enhanced 1m configuration logging
            logger.info("[CFG-1m] next_minute weights → imb=%.1f slope=%.1f macd=%.1f rsi=%.2f | windows: long=%ds/%d ticks, short=%ds/%d ticks",
                        self.next_minute_imb_weight, self.next_minute_slope_weight,
                        self.next_minute_macd_prior_weight, self.next_minute_rsi_prior_weight,
                        self.micro_window_sec_1m, self.micro_min_ticks_1m,
                        self.micro_short_window_sec_1m, self.micro_short_min_ticks_1m)
            
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
