# run_main.py
"""
Runner script for probabilities-only 5-minute directional predictor.
Emits p(BUY)/p(SELL) for a 5-minute horizon; user decides what to trade.
"""

from types import SimpleNamespace
import asyncio
import base64
import numpy as np
import logging, os
from main_event_loop_regen import main_loop
from collections import deque


try:
    import joblib
except ImportError:
    joblib = None
    logging.warning("[MODELS] joblib not available; neutrality model cannot be loaded (NEUTRAL_PATH is required)")

# ========== CONFIGURATION ==========
DHAN_ACCESS_TOKEN = os.getenv("DHAN_ACCESS_TOKEN", "")
DHAN_CLIENT_ID = os.getenv("DHAN_CLIENT_ID", "")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

config = SimpleNamespace(
    # Instrument / subscription
    symbol=os.getenv("TRAIN_SYMBOL", "IDX_I:NIFTY 50"),
    nifty_security_id=13,
    nifty_exchange_segment="IDX_I",

    # Provider credentials (encoded)
    dhan_access_token_b64=base64.b64encode(DHAN_ACCESS_TOKEN.encode()).decode(),
    dhan_client_id_b64=base64.b64encode(DHAN_CLIENT_ID.encode()).decode(),

    # --- TIMEFRAME / CANDLE CONFIG -----------------------------------------
    # Use 5-minute candles by default; override with CANDLE_INTERVAL_SECONDS
    # if you want to experiment with other bar sizes.
    candle_interval_seconds=int(os.getenv("CANDLE_INTERVAL_SECONDS", "300")),

    # User control and signal output
    user_trade_control=True,
    emit_prob_only=True,                 # NEW: probabilities-only mode
    suggest_tradeable_from_Q=True,       # optional UI hint
    signals_path="trained_models/production/signals.jsonl",

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
    ws_ping_timeout=30,
    enable_packet_checksum_validation=False,
    data_stall_seconds=15,
    data_stall_reconnect_seconds=30,

    # Feature logging
    feature_log_path=os.getenv("FEATURE_LOG", "feature_log.csv"),
    
     
    # Pre-close
    preclose_lead_seconds=1,
    preclose_completion_buffer_sec=1,

    # Patterns
    pattern_timeframes=["1T", "3T", "5T"],
    pattern_rvol_window=5,
    pattern_rvol_threshold=1.2,
    pattern_min_winrate=0.55,

    # Trade-window TP/SL config (replaces flat tolerance)
    trade_horizon_min=int(os.getenv("TRADE_HORIZON_MIN", "10") or "10"),
    trade_tp_pct=float(os.getenv("TRADE_TP_PCT", "0.0015") or "0.0015"),
    trade_sl_pct=float(os.getenv("TRADE_SL_PCT", "0.0008") or "0.0008"),

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
    xgb_model = None
    neutral_model = None

    xgb_default = "trained_models/production/xgb_5min_trade_window.json"
    neutral_default = "trained_models/production/neutral_5min.pkl"

    xgb_path = (os.getenv("XGB_PATH", "") or xgb_default).strip()
    neutral_path = (os.getenv("NEUTRAL_PATH", "") or neutral_default).strip()

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

    return xgb_model, neutral_model

# Globals for main()
global xgb_model, neutral_model
xgb_model, neutral_model = _load_models()

async def main():
    await main_loop(
        config=config,
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

    # Force DEBUG for core modules while stabilizing (only when LOG_HIGH_VERBOSITY=1)
    if _high_verbose:
        try:
            for mod in ("main_event_loop_regen", "core_handler", "feature_pipeline",
                        "model_pipeline_regen_v2", "online_trainer_regen_v2", "calibrator"):
                logging.getLogger(mod).setLevel(logging.DEBUG)
            logging.info("[LOG] Forced DEBUG level for core modules (LOG_HIGH_VERBOSITY=1)")
        except Exception as e:
            logging.debug(f"[LOG] Force DEBUG failed (ignored): {e}")


    try:
        start_dynamic_level_watcher(config_path="logs/log_level.json", poll_sec=2.0)
        logging.info("[LOG] Dynamic log-level watcher started (logs/log_level.json)")
    except Exception as e:
        logging.debug(f"[LOG] Log watcher not started: {e}")

    logging.info("XGB_PATH=%s | NEUTRAL_PATH=%s",
                os.getenv("XGB_PATH","") or "(not set)",
                os.getenv("NEUTRAL_PATH","") or "(not set)")
    logging.info("CALIB_PATH=%s (env -- model_pipeline will auto-load if present)",
                os.getenv("CALIB_PATH","") or "(not set)")

    # Reduce third-party chatter
    logging.getLogger("websockets").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)



    # Bootstrap daily feature log from historical when directional rows are below threshold
    if os.getenv("BOOTSTRAP_FEATURE_LOG", "0").lower() in ("1", "true", "yes"):
        try:
            hist_path = os.getenv("FEATURE_LOG_HIST", "trained_models/production/feature_log_hist.csv")
            daily_path = config.feature_log_path
            dir_threshold = int(getattr(config, "calib_min_rows", 300))
            bootstrap_feature_log(hist_path, daily_path, dir_threshold=dir_threshold, tail_rows=2000)
        except Exception as e:
            logging.debug(f"[BOOT] Bootstrap failed (ignored): {e}")
    else:
        logging.info("[BOOT] Bootstrap disabled (set BOOTSTRAP_FEATURE_LOG=1 to enable).")



    asyncio.run(main())

    try:
        stop_dynamic_level_watcher()
        logging.info("[LOG] Dynamic log-level watcher stopped")
    except Exception:
        pass
