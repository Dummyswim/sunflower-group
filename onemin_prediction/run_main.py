"""
Runner script for AR-NMS system with minimal config and model stubs.
"""
from types import SimpleNamespace
import asyncio
import base64
import numpy as np
import logging, os
from main_event_loop import main_loop
import random



# Dummy model stubs (replace with your trained models)
class DummyCNNLSTM:
    def predict(self, x):
        return np.zeros((8,), dtype=float)


class DummyXGB:
    def predict_proba(self, X):
        """
        Generate random predictions for testing.
        Replace with trained model for production.
        """
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
 

config = SimpleNamespace(
    # Required for EnhancedWebSocketHandler
    dhan_access_token_b64=base64.b64encode(DHAN_ACCESS_TOKEN.encode()).decode(),
    dhan_client_id_b64=base64.b64encode(DHAN_CLIENT_ID.encode()).decode(),
    nifty_security_id=13,
    nifty_exchange_segment="IDX_I",
    candle_interval_seconds=60,
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
    
    # Lazy import to avoid hard dependency when using dummies
    try:
        from tensorflow import keras
    except Exception:
        keras = None
    
    try:
        import xgboost as xgb
    except Exception:
        xgb = None
    
    try:
        cnn_path = os.getenv("CNN_LSTM_PATH", "").strip()
        if keras and cnn_path:
            cnn = keras.models.load_model(cnn_path)
            logging.info(f"Loaded CNN-LSTM: {cnn_path}")
    except Exception as e:
        logging.warning(f"Failed to load CNN-LSTM, using dummy: {e}")
    
    try:
        xgb_path = os.getenv("XGB_PATH", "").strip()
        if xgb and xgb_path:
            booster = xgb.Booster()
            booster.load_model(xgb_path)
            
            class _BoosterWrapper:
                def __init__(self, booster):
                    self.booster = booster
                
                def predict_proba(self, X):
                    import numpy as np
                    dm = xgb.DMatrix(X)
                    p = self.booster.predict(dm)
                    if p.ndim == 1:
                        p = np.clip(p, 1e-9, 1-1e-9)
                        return np.stack([1-p, p], axis=1)
                    return p
            
            xgb_model = _BoosterWrapper(booster)
            logging.info(f"Loaded XGB: {xgb_path}")
    except Exception as e:
        logging.warning(f"Failed to load XGB, using dummy: {e}")
    
    return (cnn if cnn is not None else DummyCNNLSTM(),
            xgb_model if xgb_model is not None else DummyXGB())



cnn_model, xgb_model = _load_or_dummy_models()
async def main():
    await main_loop(
        config=config,
        cnn_lstm=cnn_model,
        xgb=xgb_model,
        rl_agent=DummyRL(),
        train_features=train_features,
        token_b64=base64.b64encode(TELEGRAM_BOT_TOKEN.encode()).decode(),
        chat_id=TELEGRAM_CHAT_ID,
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
