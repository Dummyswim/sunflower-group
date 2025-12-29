# run_main.py
"""
Runner script for probabilities-only 5-minute predictor.
Emits policy success probability for teacher-defined direction.
"""

from types import SimpleNamespace
import asyncio
import base64
import numpy as np
import logging, os
from main_event_loop_regen import main_loop
from policy_pipeline import load_policy_models



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
    train_log_path=os.getenv("TRAIN_LOG_PATH", "data/train_log_v3_canonical.jsonl"),

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
    preclose_lead_seconds=10,
    preclose_completion_buffer_sec=2,

    # Patterns
    pattern_timeframes=["1T", "3T", "5T"],
    pattern_rvol_window=5,
    pattern_rvol_threshold=1.2,
    pattern_min_winrate=0.55,

    # Trade-window TP/SL config (replaces flat tolerance)
    trade_horizon_min=int(os.getenv("TRADE_HORIZON_MIN", "10") or "10"),
    trade_tp_pct=float(os.getenv("TRADE_TP_PCT", "0.0015") or "0.0015"),
    trade_sl_pct=float(os.getenv("TRADE_SL_PCT", "0.0008") or "0.0008"),

    # Calibrator cadence (fit on >=CALIB_MIN_ROWS directional rows)
    calib_interval_sec=300,
    calib_min_rows=int(os.getenv("CALIB_MIN_ROWS", "220") or "220"),

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

if not os.getenv("FEATURE_SCHEMA_COLS_PATH"):
    os.environ["FEATURE_SCHEMA_COLS_PATH"] = "data/feature_schema_cols.json"
if not os.getenv("POLICY_SCHEMA_COLS_PATH"):
    os.environ["POLICY_SCHEMA_COLS_PATH"] = "trained_models/production/policy_schema_cols.json"

def _load_policy_pipeline():
    buy_default = "trained_models/production/policy_buy.json"
    sell_default = "trained_models/production/policy_sell.json"
    schema_default = "trained_models/production/policy_schema_cols.json"
    calib_buy_default = "trained_models/production/calib_buy.json"
    calib_sell_default = "trained_models/production/calib_sell.json"

    buy_path = (os.getenv("POLICY_BUY_PATH", "") or buy_default).strip()
    sell_path = (os.getenv("POLICY_SELL_PATH", "") or sell_default).strip()
    schema_path = (os.getenv("POLICY_SCHEMA_COLS_PATH", "") or schema_default).strip()
    calib_buy_path = (os.getenv("CALIB_BUY_PATH", "") or calib_buy_default).strip()
    calib_sell_path = (os.getenv("CALIB_SELL_PATH", "") or calib_sell_default).strip()

    if not buy_path or not os.path.exists(buy_path):
        logging.critical("POLICY_BUY_PATH is required and must exist.")
        raise SystemExit(2)
    if not sell_path or not os.path.exists(sell_path):
        logging.critical("POLICY_SELL_PATH is required and must exist.")
        raise SystemExit(2)
    if not schema_path or not os.path.exists(schema_path):
        logging.critical("POLICY_SCHEMA_COLS_PATH is required and must exist.")
        raise SystemExit(2)

    pipe = load_policy_models(
        buy_path=buy_path,
        sell_path=sell_path,
        schema_path=schema_path,
        calib_buy_path=calib_buy_path,
        calib_sell_path=calib_sell_path,
    )
    return pipe


# Globals for main()
global policy_pipe
policy_pipe = _load_policy_pipeline()

async def main():
    await main_loop(
        config=config,
        policy_pipe=policy_pipe,
        train_features=train_features,
        token_b64=base64.b64encode(TELEGRAM_BOT_TOKEN.encode()).decode(),
        chat_id=TELEGRAM_CHAT_ID,
    )


if __name__ == "__main__":
    from logging_setup import setup_logging2 as setup_logging, start_dynamic_level_watcher, stop_dynamic_level_watcher

    def _safe_getenv_bool(key: str, default: bool = False) -> bool:
        val = os.getenv(key)
        if val is None:
            return bool(default)
        return str(val).strip().lower() in ("1", "true", "yes", "y", "on")

    _high_verbose = _safe_getenv_bool("LOG_HIGH_VERBOSITY", default=True)
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
                        "policy_pipeline", "online_trainer_regen_v2_bundle", "calibrator"):
                logging.getLogger(mod).setLevel(logging.DEBUG)
            logging.info("[LOG] Forced DEBUG level for core modules (LOG_HIGH_VERBOSITY=1)")
        except Exception as e:
            logging.debug(f"[LOG] Force DEBUG failed (ignored): {e}")


    try:
        start_dynamic_level_watcher(config_path="logs/log_level.json", poll_sec=2.0)
        logging.info("[LOG] Dynamic log-level watcher started (logs/log_level.json)")
    except Exception as e:
        logging.debug(f"[LOG] Log watcher not started: {e}")

    logging.info("POLICY_BUY_PATH=%s | POLICY_SELL_PATH=%s",
                os.getenv("POLICY_BUY_PATH","") or "(not set)",
                os.getenv("POLICY_SELL_PATH","") or "(not set)")
    logging.info("CALIB_BUY_PATH=%s | CALIB_SELL_PATH=%s",
                os.getenv("CALIB_BUY_PATH","") or "(not set)",
                os.getenv("CALIB_SELL_PATH","") or "(not set)")
    logging.info("FEATURE_SCHEMA_COLS_PATH=%s | POLICY_SCHEMA_COLS_PATH=%s",
                os.getenv("FEATURE_SCHEMA_COLS_PATH","") or "(not set)",
                os.getenv("POLICY_SCHEMA_COLS_PATH","") or "(not set)")

    # Reduce third-party chatter
    logging.getLogger("websockets").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)



    asyncio.run(main())

    try:
        stop_dynamic_level_watcher()
        logging.info("[LOG] Dynamic log-level watcher stopped")
    except Exception:
        pass
