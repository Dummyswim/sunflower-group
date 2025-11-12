"""
Runner script for AR-NMS system with minimal config and model stubs.
"""
from types import SimpleNamespace
import asyncio
import base64
import numpy as np
import logging, os
from main_event_loop import main_loop



try:
    import joblib
except ImportError:
    joblib = None
    logging.warning("joblib not available; neutrality model loading will be skipped")


# Dummy model stubs (replace with your trained models)
class DummyCNNLSTM:
    def predict(self, x):
        return np.zeros((8,), dtype=float)



class DummyRL:
    threshold = 0.6  # Base confidence threshold
    
    def adjust_confidence(self, pf):
        """
        Adjust confidence based on recent profit factor.
        pf > 1.0: increase confidence (lower threshold)
        pf < 1.0: decrease confidence (higher threshold)
        """
        try:
            pf = float(pf)
            if pf > 1.2:
                return max(0.55, self.threshold - 0.05)  # More aggressive
            elif pf < 0.8:
                return min(0.70, self.threshold + 0.10)  # More conservative
            else:
                return self.threshold  # Neutral
        except Exception:
            return self.threshold



# ========== CONFIGURATION ==========
# Replace these with your actual credentials
# DHAN_ACCESS_TOKEN  : str = field(default_factory=lambda: os.getenv("DHAN_ACCESS_TOKEN", ""))
# DHAN_CLIENT_ID : str = field(default_factory=lambda: os.getenv("DHAN_CLIENT_ID", ""))
# TELEGRAM_BOT_TOKEN : str = field(default_factory=lambda: os.getenv("TELEGRAM_BOT_TOKEN", ""))
# TELEGRAM_CHAT_ID : str = field(default_factory=lambda: os.getenv("TELEGRAM_CHAT_ID", ""))

DHAN_ACCESS_TOKEN = os.getenv("DHAN_ACCESS_TOKEN", "") 
DHAN_CLIENT_ID = os.getenv("DHAN_CLIENT_ID", "") 
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "") 
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
REQUIRE_TRAINED_MODELS = os.getenv("REQUIRE_TRAINED_MODELS", "0") in ("1", "true", "True")


config = SimpleNamespace(
    # Required for EnhancedWebSocketHandler
    dhan_access_token_b64=base64.b64encode(DHAN_ACCESS_TOKEN.encode()).decode(),
    dhan_client_id_b64=base64.b64encode(DHAN_CLIENT_ID.encode()).decode(),
    nifty_security_id=13,
    nifty_exchange_segment="IDX_I",
    candle_interval_seconds=60,

    # Timestamp mode: True = use arrival time (wall-clock IST), False = use vendor LTT
    use_arrival_time=True,  # Set to False to use vendor timestamps with auto-detection
    
    max_buffer_size=5000,
    max_reconnect_attempts=5,
    reconnect_delay_base=2,
    max_candles_stored=2000,
    price_sanity_min=1.0,
    price_sanity_max=100000.0,
    
    # Optional toggles
    enable_packet_checksum_validation=False,
    data_stall_seconds=15,
    data_stall_reconnect_seconds=30,
    
    # Main loop toggles
    enable_rule_blend=True,
    hitrate_path="hitrate.txt",
    weight_learning_rate=0.05,
    weight_refresh_seconds=30,
    
    # Microstructure windows
    micro_tick_window=400,
    micro_window_sec_1m=45,
    micro_min_ticks_1m=24,
    micro_short_window_sec_1m=12,
    micro_short_min_ticks_1m=8,
    micro_last3s_window_sec_1m=3,
    
    # Gate, tolerance, and feature logging
    gate_std_short_epsilon=1e-6,
    gate_tightness_threshold=0.995,
    flat_tolerance_pct=0.0002,  # 0.02%
    
    # Flat evaluation tunables (Option B)
    flat_dyn_k_range=0.20,        # 20% of (high-low) range contributes to tolerance
    flat_min_points=0.20,         # absolute floor in points
    flat_tolerance_max_pts=1.0,  # enable cap at 1.0 points by default (tunable)


    # Optional probability calibration file for XGB (Platt: p' = sigmoid(a*p + b))
    calib_path=os.getenv("CALIB_PATH", "").strip(),

    
    feature_log_path="feature_log.csv",
    
    # Pre-close options
    preclose_lead_seconds=10,
    preclose_completion_buffer_sec=1,
    

    # Pattern detection parameters
    pattern_rvol_window=5,           # Window for RVOL baseline
    pattern_rvol_threshold=1.2,      # RVOL confirmation threshold
    pattern_min_winrate=0.55,        # Minimum winrate to use pattern
        
        
    # Dynamic alpha tuning
    target_hold_ratio=0.30,
    alpha_tune_step=0.01,  # gentler hold-ratio tuning
    alpha_min=0.52,
    alpha_max=0.72,
    deadband_eps=0.05,
    consensus_threshold=0.66,
    indicator_bias_threshold=0.20,
    
    # Pattern timeframes for MTF
    pattern_timeframes=["1T", "3T", "5T"],

    # Online trainer configuration
    trainer_interval_sec=600,  # Train every 10 minutes
    trainer_min_rows=300,      # Minimum rows before training

    # Hysteresis controls and decision logging verbosity
    hysteresis_enable=True,
    hysteresis_respect_alpha=True,
    hysteresis_flip_buffer=0.01,  # unconditional flip buffer over alpha
    hysteresis_advantage=0.02,    # p_new - p_prev advantage to flip (if not unconditional)
    decision_log_verbose=True,    # log raw vs final decision at prediction time

    # Hysteresis telemetry
    hysteresis_telemetry_enable=True,
    hysteresis_telemetry_interval_sec=3600,  # hourly summary by default

        
)




# Minimal train_features for DriftDetector

train_features = {
    'ema_8': np.random.normal(size=500).tolist(),
    'ema_21': np.random.normal(size=500).tolist(),
    'spread': np.random.normal(size=500).tolist(),
    'micro_slope': np.random.normal(size=500).tolist(),
    'micro_imbalance': np.random.normal(size=500).tolist(),
    'mean_drift_pct': np.random.normal(size=500).tolist(),
    'last_price': np.random.normal(size=500).tolist(),
    'last_zscore': np.random.normal(size=500).tolist()
}



def _load_or_dummy_models():
    cnn = None
    xgb_model = None
    neutral_model = None
    
    cnn_path = os.getenv("CNN_LSTM_PATH", "").strip()
    xgb_path = os.getenv("XGB_PATH", "").strip()
    neutral_path = os.getenv("NEUTRAL_PATH", "").strip()
    
    # CNN-LSTM optional
    if cnn_path:
        try:
            os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
            from tensorflow import keras
            cnn = keras.models.load_model(cnn_path)
            logging.info(f"Loaded CNN-LSTM: {cnn_path}")
        except Exception as e:
            logging.warning(f"Failed to load CNN-LSTM, using dummy: {e}")
    else:
        logging.info("CNN_LSTM_PATH not set; using dummy CNN-LSTM")
    
    # XGB required
    if not xgb_path or not os.path.exists(xgb_path):
        logging.critical("XGB_PATH is required and must exist. Set XGB_PATH to a valid model file.")
        raise SystemExit(2)
    
    try:
        import xgboost as xgb
        booster = xgb.Booster()
        booster.load_model(xgb_path)
        
        class _BoosterWrapper:
            def __init__(self, booster):
                self.booster = booster
                self.is_dummy = False
                self.name = "XGBBooster"
            def predict_proba(self, X):
                dm = xgb.DMatrix(X)
                p = self.booster.predict(dm)
                import numpy as np
                if p.ndim == 1:
                    p = np.clip(p, 1e-9, 1 - 1e-9)
                    return np.stack([1 - p, p], axis=1)
                return p
        
        xgb_model = _BoosterWrapper(booster)
        logging.info(f"Loaded XGB: {xgb_path}")
    except Exception as e:
        logging.critical(f"Failed to load XGB model at {xgb_path}: {e}")
        raise SystemExit(2)
    
    # Neutrality required
    if not neutral_path or not os.path.exists(neutral_path):
        logging.critical("NEUTRAL_PATH is required and must exist. Set NEUTRAL_PATH to a valid model file.")
        raise SystemExit(3)
    
    if joblib is None:
        logging.critical("joblib is required to load NEUTRAL_PATH")
        raise SystemExit(3)
    try:
        neutral_model = joblib.load(neutral_path)
        logging.info(f"Loaded Neutrality model: {neutral_path}")
    except Exception as e:
        logging.critical(f"Failed to load Neutrality model at {neutral_path}: {e}")
        raise SystemExit(3)
    
    return (cnn if cnn is not None else DummyCNNLSTM(),
            xgb_model,
            neutral_model)




# Make globals visible to main()
global cnn_model, xgb_model, neutral_model

cnn_model, xgb_model, neutral_model = _load_or_dummy_models()





async def main():
    await main_loop(
        config=config,
        cnn_lstm=cnn_model,
        xgb=xgb_model,
        rl_agent=DummyRL(),
        train_features=train_features,
        token_b64=base64.b64encode(TELEGRAM_BOT_TOKEN.encode()).decode(),
        chat_id=TELEGRAM_CHAT_ID,
        neutral_model=neutral_model,  # NEW
    )


if __name__ == "__main__":

    # logging.basicConfig(
    #     level=logging.INFO,
    #     format="%(asctime)s | %(levelname)s | %(name)s | [%(funcName)s:%(lineno)d] | %(message)s"
    # )

    # Import the enhanced logging setup with IST timezone
    from logging_setup import setup_logging2 as setup_logging
    
        
    # Configure logging (console colored, file UTF-8 without ANSI)
    # Keep high verbosity in early stage via LOG_HIGH_VERBOSITY=1 (default)
    _high_verbose = os.getenv("LOG_HIGH_VERBOSITY", "1") in ("1", "true", "True")
    _console_level = logging.DEBUG if _high_verbose else logging.INFO
    _file_level = logging.DEBUG if _high_verbose else logging.INFO

    setup_logging(
        logfile="logs/unified_trading.log",
        console_level=_console_level,
        file_level=_file_level,
        enable_colors_console=True,
        enable_colors_file=False,         # avoid ANSI in file
        max_bytes=10485760,               # 10MB
        backup_count=5,
        heartbeat_cooldown_sec=0.0        # you can set 30 to dedupe heartbeats in file
    )

    
    logging.info(f"REQUIRE_TRAINED_MODELS={REQUIRE_TRAINED_MODELS} (env)")
    logging.info(f"XGB_PATH={os.getenv('XGB_PATH','') or '(not set)'} | NEUTRAL_PATH={os.getenv('NEUTRAL_PATH','') or '(not set)'}")
    
    logging.info(f"CALIB_PATH={os.getenv('CALIB_PATH','') or '(not set)'} (env -- model_pipeline will auto-load if present)")


    # Visibility: log selected decision/hysteresis toggles
    try:

        logging.info(
            "Hysteresis/telemetry: enable=%s respect_alpha=%s flip_buffer=%.3f advantage=%.3f "
            "| decision_log_verbose=%s | telem_enable=%s telem_interval=%ss",
            getattr(config, "hysteresis_enable", True),
            getattr(config, "hysteresis_respect_alpha", True),
            float(getattr(config, "hysteresis_flip_buffer", 0.01) or 0.01),
            float(getattr(config, "hysteresis_advantage", 0.02) or 0.02),
            getattr(config, "decision_log_verbose", True),
            getattr(config, "hysteresis_telemetry_enable", True),
            int(getattr(config, "hysteresis_telemetry_interval_sec", 3600) or 3600),
        )


    except Exception:
        pass


    
    # Reduce third-party chatter
    logging.getLogger("websockets").setLevel(logging.INFO)
    logging.getLogger("asyncio").setLevel(logging.INFO)
    
    asyncio.run(main())
