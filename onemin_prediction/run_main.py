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



class DummyXGB:
    is_dummy = True
    name = "DummyXGB"
    
    def predict_proba(self, X):
        """
        Generate random predictions for testing.
        Replace with trained model for production.
        """
        import numpy as np
        import random
        buy_prob = random.uniform(0.3, 0.7)  # Random between 30-70%
        return np.array([[1.0 - buy_prob, buy_prob]], dtype=float)

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
logging.info(f"REQUIRE_TRAINED_MODELS={REQUIRE_TRAINED_MODELS} (env)")


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
    micro_window_sec_1m=25,
    micro_min_ticks_1m=30,
    micro_short_window_sec_1m=8,
    micro_short_min_ticks_1m=12,
    micro_last3s_window_sec_1m=3,
    
    # Gate, tolerance, and feature logging
    gate_std_short_epsilon=1e-6,
    gate_tightness_threshold=0.995,
    flat_tolerance_pct=0.0002,  # 0.02%
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
    alpha_tune_step=0.02,
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

        
)




# Minimal train_features for DriftDetector
train_features = {
    'ema_8': np.random.normal(size=500).tolist(),
    'ema_21': np.random.normal(size=500).tolist(),
    'spread': np.random.normal(size=500).tolist(),
    'micro_slope': np.random.normal(size=500).tolist(),
    'micro_imbalance': np.random.normal(size=500).tolist(),
    'mean_drift_pct': np.random.normal(size=500).tolist(),
    'last_price': np.random.normal(size=500).tolist()
}

def _load_or_dummy_models():
    cnn = None
    xgb_model = None
    neutral_model = None  # NEW
    
    cnn_path = os.getenv("CNN_LSTM_PATH", "").strip()
    xgb_path = os.getenv("XGB_PATH", "").strip()
    
    # Load CNN-LSTM only if a path is provided
    if cnn_path:
        try:
            # Suppress TF INFO logs (including oneDNN/CUDA notices)
            os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
            from tensorflow import keras
            cnn = keras.models.load_model(cnn_path)
            logging.info(f"Loaded CNN-LSTM: {cnn_path}")
        except Exception as e:
            logging.warning(f"Failed to load CNN-LSTM, using dummy: {e}")
    else:
        logging.info("CNN_LSTM_PATH not set; using dummy CNN-LSTM")
    
    # Load XGBoost only if a path is provided
    if xgb_path:
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
                    import numpy as np
                    dm = xgb.DMatrix(X)
                    p = self.booster.predict(dm)
                    if p.ndim == 1:
                        p = np.clip(p, 1e-9, 1 - 1e-9)
                        return np.stack([1 - p, p], axis=1)
                    return p

            
            xgb_model = _BoosterWrapper(booster)
            logging.info(f"Loaded XGB: {xgb_path}")
        except Exception as e:
            logging.warning(f"Failed to load XGB, using dummy: {e}")

    else:
        logging.info("XGB_PATH not set; using dummy XGB")
    

    
    # Load Neutrality Model if provided
    neutral_path = os.getenv("NEUTRAL_PATH", "").strip()

    if neutral_path and joblib is not None:
        try:
            neutral_model = joblib.load(neutral_path)
            logging.info(f"Loaded Neutrality model: {neutral_path}")
        except Exception as e:
            logging.warning(f"Failed to load Neutrality model, continuing without: {e}")
    else:
        if neutral_path and joblib is None:
            logging.warning("NEUTRAL_PATH set but joblib not available")
        else:
            logging.info("NEUTRAL_PATH not set; running without neutral model")
    
    # Enforce trained models in production if requested
    if REQUIRE_TRAINED_MODELS:
        if xgb_model is None:
            logging.critical("REQUIRE_TRAINED_MODELS=1 but XGB_PATH is not set or failed to load")
            raise SystemExit(2)
        if neutral_model is None:
            logging.critical("REQUIRE_TRAINED_MODELS=1 but NEUTRAL_PATH is not set or failed to load")
            raise SystemExit(3)
            
    return (cnn if cnn is not None else DummyCNNLSTM(),
            xgb_model if xgb_model is not None else DummyXGB(),
            neutral_model)



# Make globals visible to main()
global cnn_model, xgb_model, neutral_model

cnn_model, xgb_model, neutral_model = _load_or_dummy_models()




logging.info(
    f"Backends: cnn={getattr(cnn_model, 'name', type(cnn_model).__name__)}, "
    f"xgb={getattr(xgb_model, 'name', type(xgb_model).__name__)} "
    f"(is_dummy={getattr(xgb_model, 'is_dummy', 'unknown')}), "
    f"neutral={'present' if neutral_model is not None else 'absent'}, "
    f"XGB_PATH={os.getenv('XGB_PATH','') or '(not set)'} "
    f"NEUTRAL_PATH={os.getenv('NEUTRAL_PATH','') or '(not set)'}"
)




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
    from logging_setup import setup_logging
    
    # Configure logging with IST timezone
    setup_logging(
        logfile="logs/unified_trading.log",
        console_level=logging.DEBUG,  # Increased verbosity for debugging
        file_level=logging.DEBUG,
        enable_colors=True,
        max_bytes=10485760,  # 10MB
        backup_count=5
    )
    
    # Reduce third-party chatter
    logging.getLogger("websockets").setLevel(logging.INFO)
    logging.getLogger("asyncio").setLevel(logging.INFO)
    
    asyncio.run(main())
