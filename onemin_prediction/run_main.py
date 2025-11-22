# run_main.py
"""
Runner script for probabilities-only one-minute predictor.
Emits p(BUY)/p(SELL) per minute; user decides what to trade.
"""
from types import SimpleNamespace
import asyncio
import base64
import numpy as np
import logging, os
from main_event_loop import main_loop
import shutil
from collections import deque


try:
    import joblib
except ImportError:
    joblib = None
    logging.warning("[MODELS] joblib not available; neutrality model cannot be loaded (NEUTRAL_PATH is required)")

# Dummy model stub for latent extraction (optional)
class DummyCNNLSTM:
    def predict(self, x):
        return np.zeros((8,), dtype=float)

# ========== CONFIGURATION ==========
DHAN_ACCESS_TOKEN = os.getenv("DHAN_ACCESS_TOKEN", "")
DHAN_CLIENT_ID = os.getenv("DHAN_CLIENT_ID", "")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
REQUIRE_TRAINED_MODELS = os.getenv("REQUIRE_TRAINED_MODELS", "0") in ("1", "true", "True")

config = SimpleNamespace(
    # Provider credentials (encoded)
    dhan_access_token_b64=base64.b64encode(DHAN_ACCESS_TOKEN.encode()).decode(),
    dhan_client_id_b64=base64.b64encode(DHAN_CLIENT_ID.encode()).decode(),
    nifty_security_id=13,
    nifty_exchange_segment="IDX_I",
    candle_interval_seconds=60,

    # User control and signal output
    user_trade_control=True,
    emit_prob_only=True,                 # NEW: probabilities-only mode
    suggest_tradeable_from_Q=True,       # optional UI hint
    signals_path="logs/signals.jsonl",

    # Timekeeping
    use_arrival_time=True,

    # WebSocket & buffers
    max_buffer_size=5000,
    max_reconnect_attempts=5,
    reconnect_delay_base=2,
    max_candles_stored=2000,
    price_sanity_min=1.0,
    price_sanity_max=100000.0,
    ws_ping_interval=30,
    ws_ping_timeout=10,
    enable_packet_checksum_validation=False,
    data_stall_seconds=15,
    data_stall_reconnect_seconds=30,

    # Feature logging
    feature_log_path="feature_log.csv",

    # Pre-close
    preclose_lead_seconds=1,
    preclose_completion_buffer_sec=1,

    # Patterns
    pattern_timeframes=["1T", "3T", "5T"],
    pattern_rvol_window=5,
    pattern_rvol_threshold=1.2,
    pattern_min_winrate=0.55,


    # Flat labelling (relax to increase directional rows)
    flat_tolerance_pct=0.00010,  # 0.01%
    flat_dyn_k_range=0.20,
    flat_min_points=0.20,
    flat_tolerance_max_pts=3.0,

    # Calibrator cadence (fit on >=220 directional rows)
    calib_interval_sec=300,
    calib_min_rows=220,

)

# Minimal drift baseline seed (will be refreshed from feature_log)
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

def _load_models():
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
            logging.info(f"[MODELS] CNN-LSTM loaded from {cnn_path}")
        except Exception as e:
            logging.warning(f"Failed to load CNN-LSTM, using dummy: {e}")
    else:
        logging.info("[MODELS] CNN-LSTM not configured; using dummy")

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
        logging.info(f"[MODELS] XGB loaded from {xgb_path}")
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
        logging.info(f"[MODELS] Neutrality model loaded from {neutral_path}")
    except Exception as e:
        logging.critical(f"Failed to load Neutrality model at {neutral_path}: {e}")
        raise SystemExit(3)

    return (cnn if cnn is not None else DummyCNNLSTM(),
            xgb_model,
            neutral_model)

# Globals for main()
global cnn_model, xgb_model, neutral_model
cnn_model, xgb_model, neutral_model = _load_models()

async def main():
    await main_loop(
        config=config,
        cnn_lstm=cnn_model,
        xgb=xgb_model,
        train_features=train_features,
        token_b64=base64.b64encode(TELEGRAM_BOT_TOKEN.encode()).decode(),
        chat_id=TELEGRAM_CHAT_ID,
        neutral_model=neutral_model,
    )


def _tail_lines(path: str, n: int = 2000) -> list[str]:
    try:
        dq = deque(maxlen=max(1, n))
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    dq.append(line.rstrip("\n"))
        return list(dq)
    except Exception:
        return []

def _count_directionals(path: str, max_lines: int = 200000) -> int:
    try:
        n = 0
        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= max_lines:
                    break
                if not line.strip():
                    continue
                parts = line.split(",", 3)
                if len(parts) >= 3:
                    label = parts[2].strip()
                    if label in ("BUY", "SELL"):
                        n += 1
        return n
    except Exception:
        return 0


def bootstrap_feature_log(hist_path: str, daily_path: str, dir_threshold: int = 500, tail_rows: int = 2000) -> None:
    try:
        if not os.path.exists(hist_path):
            return
        if not os.path.exists(daily_path):
            lines = _tail_lines(hist_path, n=tail_rows)
            if lines:
                with open(daily_path, "w", encoding="utf-8") as f:
                    f.write("\n".join(lines) + "\n")
                logging.info("[BOOT] Bootstrapped %s from %s (%d rows)", daily_path, hist_path, len(lines))
            return
        
        dir_count = _count_directionals(daily_path)
        if dir_count >= dir_threshold:
            logging.info("[BOOT] Daily feature log already has %d directional rows (>= %d); no bootstrap merge", dir_count, dir_threshold)
            return
        
        with open(daily_path, "r", encoding="utf-8") as f:
            daily_lines = [ln.strip() for ln in f if ln.strip()]
        
        hist_tail = _tail_lines(hist_path, n=tail_rows)
        merged, seen = [], set()
        for ln in hist_tail + daily_lines:
            ts = ln.split(",", 1)[0].strip()
            if ts and ts not in seen:
                seen.add(ts)
                merged.append(ln)
        
        with open(daily_path, "w", encoding="utf-8") as f:
            f.write("\n".join(merged) + "\n")
        logging.info("[BOOT] Merged %s with tail of %s. Rows=%d (dir before=%d, threshold=%d)", 
                     daily_path, hist_path, len(merged), dir_count, dir_threshold)
    except Exception as e:
        logging.warning("[BOOT] Bootstrap skipped: %s", e)



if __name__ == "__main__":
    from logging_setup import setup_logging2 as setup_logging, start_dynamic_level_watcher, stop_dynamic_level_watcher

    _high_verbose = os.getenv("LOG_HIGH_VERBOSITY", "1").lower() in ("1", "true", "yes")
    _console_level = logging.DEBUG if _high_verbose else logging.INFO
    _file_level = logging.DEBUG if _high_verbose else logging.INFO

    setup_logging(
        logfile="logs/unified_trading.log",
        console_level=_console_level,
        file_level=_file_level,
        enable_colors_console=True,
        enable_colors_file=False,
        max_bytes=10_485_760,
        backup_count=5,
        heartbeat_cooldown_sec=30.0,
        heartbeat_cooldown_console_sec=0.0,
        telegram_alerts=True,
        telegram_min_level=logging.ERROR
    )

    # Force DEBUG for core modules while stabilizing
    try:
        for mod in ("main_event_loop", "core_handler", "feature_pipeline",
                    "model_pipeline", "online_trainer", "calibrator"):
            logging.getLogger(mod).setLevel(logging.DEBUG)
        logging.info("[LOG] Forced DEBUG level for core modules")
    except Exception as e:
        logging.debug(f"[LOG] Force DEBUG failed (ignored): {e}")


    try:
        start_dynamic_level_watcher(config_path="logs/log_level.json", poll_sec=2.0)
        logging.info("[LOG] Dynamic log-level watcher started (logs/log_level.json)")
    except Exception as e:
        logging.debug(f"[LOG] Log watcher not started: {e}")

    logging.info("[BOOT] REQUIRE_TRAINED_MODELS=%s (env)", REQUIRE_TRAINED_MODELS)
    logging.info("XGB_PATH=%s | NEUTRAL_PATH=%s",
                os.getenv("XGB_PATH","") or "(not set)",
                os.getenv("NEUTRAL_PATH","") or "(not set)")
    logging.info("CALIB_PATH=%s (env -- model_pipeline will auto-load if present)",
                os.getenv("CALIB_PATH","") or "(not set)")

    # Reduce third-party chatter
    logging.getLogger("websockets").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)



    # Bootstrap daily feature log from historical when directional rows are below threshold
    try:
        hist_path = os.getenv("FEATURE_LOG_HIST", "feature_log_hist.csv")
        daily_path = config.feature_log_path
        dir_threshold = int(getattr(config, "calib_min_rows", 300))
        bootstrap_feature_log(hist_path, daily_path, dir_threshold=dir_threshold, tail_rows=2000)
    except Exception as e:
        logging.debug(f"[BOOT] Bootstrap failed (ignored): {e}")



    asyncio.run(main())

    try:
        stop_dynamic_level_watcher()
        logging.info("[LOG] Dynamic log-level watcher stopped")
    except Exception:
        pass
