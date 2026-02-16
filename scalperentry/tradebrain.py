"""
tradebrain.py - Master Refactor v2.1 (Index-safe, price-only scalper core)

Key upgrades (no volume / no broker CVD required):
- Anchor & Pressure Engine: HMA(20) midline + ATR bands + signed distance.
- CVD replacement: CLV (pressure), Body% (doji filter), Path Efficiency + Travel/Tick + Smoothness.
- Shock overhaul: close-prev_close, dynamic threshold, WATCH then confirm before entry.
- Anti-climax guard: blocks selling the bottom during absorption.
- HMA hysteresis: 5-bar normalized slope drives bias state (prevents flip-flop).
- Fast Exit Engine: intra-tick hard stop + intrabar SL/TP fills + BE + trailing.
- Harvest Mode: overextension tightens trail; exits require CLV flip (no panic exits).

Design goals: minimal bloat, clean condition stack, safe math, robust logging.
"""

from __future__ import annotations

import json
import logging
import math
import os
import threading
import time
import urllib.error
import urllib.request
from collections import deque
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone, timedelta
from logging.handlers import RotatingFileHandler
from typing import Any, Deque, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

# --- Build Metadata ---
DATA_MODE = "quote"

_EPS = 1e-9
IST_TZ = timezone(timedelta(hours=5, minutes=30))


class ISTFormatter(logging.Formatter):
    def formatTime(self, record, datefmt=None):
        dt = datetime.fromtimestamp(record.created, IST_TZ)
        base = dt.strftime(datefmt) if datefmt else dt.strftime("%Y-%m-%d %H:%M:%S")
        off = dt.strftime("%z")
        off = (off[:3] + ":" + off[3:]) if len(off) == 5 else off
        return f"{base}{off}"


def setup_logger(name: str, log_file: str, level: int, also_console: bool) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers = []
    logger.propagate = False

    os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)

    fmt = ISTFormatter("%(asctime)s | %(levelname)s | %(message)s")
    fh = RotatingFileHandler(log_file, maxBytes=5_000_000, backupCount=5, encoding="utf-8")
    fh.setFormatter(fmt)
    fh.setLevel(level)
    logger.addHandler(fh)

    if also_console:
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        sh.setLevel(level)
        logger.addHandler(sh)

    return logger


@dataclass(frozen=True)
class TradeBrainConfig:
    # Logging
    log_file: str = os.getenv("TB_LOG_FILE", "logs/tradebrain.log")
    log_level: int = logging.INFO
    log_to_console: bool = os.getenv("TB_LOG_TO_CONSOLE", "0") == "1"

    # Anchor & Trend
    hma_period: int = int(os.getenv("TB_HMA_PERIOD", "20"))
    atr_period: int = int(os.getenv("TB_ATR_PERIOD", "14"))
    stretch_limit: float = float(os.getenv("TB_STRETCH_LIMIT_ATR", "2.0"))  # band = HMA ± stretch_limit*ATR

    # Directionality / chop filter
    min_path_eff_filter: float = float(os.getenv("TB_MIN_PATH_EFF_FILTER", "0.03"))  # tuned for NIFTY 1m tick-travel PE
    entry_path_eff: float = float(os.getenv("TB_ENTRY_PATH_EFF", "0.12"))  # tuned for NIFTY 1m tick-travel PE
    entry_anchor_min_atr: float = float(os.getenv("TB_ENTRY_ANCHOR_MIN_ATR", "0.30"))  # keep entries off anchor
    max_anchor_dist_atr: float = float(os.getenv("TB_MAX_ANCHOR_DIST_ATR", "1.5"))  # max distance from anchor for entries

    # Contextual entry vetoes (ATR-normalized; "soft" vetoes return HOLD_WAIT_CONFIRM)
    late_breakout_anchor_atr: float = float(os.getenv("TB_LATE_BREAKOUT_ANCHOR_ATR", "0.90"))
    lip_breakout_atr: float = float(os.getenv("TB_LIP_BREAKOUT_ATR", "0.20"))
    climax_streak: int = int(os.getenv("TB_CLIMAX_STREAK", "4"))
    climax_anchor_atr: float = float(os.getenv("TB_CLIMAX_ANCHOR_ATR", "0.8"))
    flow_bias_veto: float = float(os.getenv("TB_FLOW_BIAS_VETO", "0.15"))

    # Scalper context awareness (SR / absorption / latency)
    support_lookback: int = int(os.getenv("TB_SUPPORT_LOOKBACK", "45"))  # local structure (1m)
    sr_buffer_atr: float = float(os.getenv("TB_SR_BUFFER_ATR", "0.35"))  # too close to floor/ceiling
    entry_rr_min: float = float(os.getenv("TB_ENTRY_RR_MIN", "0.8"))  # min reward/risk vs hard stop

    absorb_wick_pct: float = float(os.getenv("TB_ABSORB_WICK_PCT", "0.45"))
    absorb_clv_min: float = float(os.getenv("TB_ABSORB_CLV_MIN", "0.20"))
    absorb_cooldown_ms: int = int(os.getenv("TB_ABSORB_COOLDOWN_MS", "120000"))  # 2 min

    # Latency guard (only for tick engines: micro/ema915). 0 disables.
    entry_latency_max_ms: int = int(os.getenv("TB_ENTRY_LATENCY_MAX_MS", "1000"))

    # Persist confirmed 1m candles so session SR survives restarts (JSONL, filtered by IST date)
    persist_candles: bool = os.getenv("TB_PERSIST_CANDLES", "1") == "1"
    candle_db_file: str = os.getenv("TB_CANDLE_DB_FILE", "data/candles_session.jsonl")

    # HTF Trend Filter (3m/5m proxy via 15/25-bar HMA slope; ATR-normalized per minute)
    # Example: 0.04 means ~0.04 ATR per minute downward qualifies as "bear HTF" when sustained on both windows.
    htf_slope_thresh: float = float(os.getenv("TB_HTF_SLOPE_THRESH", "0.04"))
    htf_local_lookback: int = int(os.getenv("TB_HTF_LOCAL_LOOKBACK", "15"))  # defines local shelf (resistance)

    # Reversal proven: require break + hold above local_res before allowing LONGs in bearish HTF
    # hold is counted in CLOSED 1m candles.
    htf_reversal_hold_bars: int = int(os.getenv("TB_HTF_REV_HOLD_BARS", "2"))
    htf_reversal_buffer_ticks: int = int(os.getenv("TB_HTF_REV_BUFFER_TICKS", "1"))

    # Micro hardening (for proactive_* micro intents)
    # This is your "catch move early" entry; require stronger candle evidence and stricter flow neutrality.
    micro_harden_clv: float = float(os.getenv("TB_MICRO_HARDEN_CLV", "0.40"))
    micro_harden_eff: float = float(os.getenv("TB_MICRO_HARDEN_EFF", "0.25"))
    micro_flow_veto: float = float(os.getenv("TB_MICRO_FLOW_VETO", "0.05"))  # strict: slight opposing flow blocks
    # Market-driven hysteresis (no fixed time cooldowns)
    # These operate on a normalized margin in [-1..1] derived from candle metrics.
    ready_margin: float = float(os.getenv("TB_READY_MARGIN", "0.18"))  # becomes READY_LONG/READY_SHORT
    flip_margin: float = float(os.getenv("TB_FLIP_MARGIN", "0.28"))  # required to flip setup side
    conflict_margin: float = float(os.getenv("TB_CONFLICT_MARGIN", "0.08"))  # HOLD_CONFLICT when abs(margin) < this

    # Tick-intent hysteresis (uses engine score units)
    intent_conflict_score_delta: float = float(os.getenv("TB_INTENT_CONFLICT_SCORE_DELTA", "1200"))
    intent_flip_score_delta: float = float(os.getenv("TB_INTENT_FLIP_SCORE_DELTA", "2000"))

    # Reversal-proof (market evidence; not time)
    reverse_min_move_atr: float = float(os.getenv("TB_REVERSE_MIN_MOVE_ATR", "0.80"))
    reverse_require_anchor_cross: bool = os.getenv("TB_REVERSE_REQUIRE_ANCHOR_CROSS", "1") == "1"

    # Pressure thresholds
    min_clv_confirm: float = float(os.getenv("TB_MIN_CLV_CONFIRM", "0.20"))  # momentum confirm
    strong_clv: float = float(os.getenv("TB_STRONG_CLV", "0.80"))  # clean sweep pattern
    min_body_pct: float = float(os.getenv("TB_MIN_BODY_PCT", "0.08"))  # allow wick-rejection candles

    # Shock system
    shock_atr_mult: float = float(os.getenv("TB_SHOCK_ATR_MULT", "1.5"))
    shock_points: float = float(os.getenv("TB_SHOCK_POINTS", "15.0"))
    shock_confirm_candles: int = int(os.getenv("TB_SHOCK_CONFIRM_CANDLES", "3"))
    shock_lock_ms: int = int(os.getenv("TB_SHOCK_LOCK_MS", "60000"))
    shock_expiry_minutes: int = int(os.getenv("TB_SHOCK_EXPIRY_MINUTES", "3"))

    # Exits / Trade Mgmt
    hard_stop_points: float = float(os.getenv("TB_HARD_STOP_POINTS", "20.0"))  # intra-tick hard stop vs entry
    tp_atr_mult: float = float(os.getenv("TB_TP_ATR_MULT", "2.5"))  # fixed TP when not runner
    move_to_be_atr: float = float(os.getenv("TB_MOVE_TO_BE_ATR", "0.35"))
    be_buffer_points: float = float(os.getenv("TB_BE_BUFFER_POINTS", "0.5"))
    trail_atr: float = float(os.getenv("TB_TRAIL_ATR", "2.0"))
    harvest_trail_atr: float = float(os.getenv("TB_HARVEST_TRAIL_ATR", "0.5"))
    min_init_sl_points: float = float(os.getenv("TB_MIN_INIT_SL_POINTS", "4.0"))
    min_init_sl_atr: float = float(os.getenv("TB_MIN_INIT_SL_ATR", "0.45"))
    hard_stop_atr_mult: float = float(os.getenv("TB_HARD_STOP_ATR_MULT", "1.5"))
    cooldown_reclaim_atr: float = float(os.getenv("TB_COOLDOWN_RECLAIM_ATR", "0.25"))
    cooldown_min_bars: int = int(os.getenv("TB_COOLDOWN_MIN_BARS", "1"))
    stall_bars: int = int(os.getenv("TB_STALL_BARS", "3"))
    stall_min_mfe_atr: float = float(os.getenv("TB_STALL_MIN_MFE_ATR", "0.15"))

    # ------------------------ Regime-aware, non-erratic enhancements ------------------------
    # A) Proactive Break Acceptance Lite
    proactive_lite_enable: bool = os.getenv("TB_PROACTIVE_LITE_ENABLE", "1") == "1"
    proactive_lite_energy_min: float = float(os.getenv("TB_PROACTIVE_LITE_ENERGY_MIN", "0.52"))
    proactive_lite_abs_margin_min: float = float(os.getenv("TB_PROACTIVE_LITE_ABS_MARGIN_MIN", "0.25"))
    proactive_lite_abs_margin_min_ext: float = float(os.getenv("TB_PROACTIVE_LITE_ABS_MARGIN_MIN_EXT", "0.18"))
    proactive_lite_extension_atr_min: float = float(os.getenv("TB_PROACTIVE_LITE_EXTENSION_ATR_MIN", "1.5"))
    proactive_lite_path_eff_min: float = float(os.getenv("TB_PROACTIVE_LITE_PATH_EFF_MIN", "0.08"))
    proactive_lite_conf_min: float = float(os.getenv("TB_PROACTIVE_LITE_CONF_MIN", "0.30"))
    proactive_lite_latency_max_ms: int = int(os.getenv("TB_PROACTIVE_LITE_LATENCY_MAX_MS", "600"))

    # B) Exit-type aware cooldown + reclaim override
    cooldown_exit_init_sl_bars: int = int(os.getenv("TB_COOLDOWN_EXIT_INIT_SL_BARS", "1"))
    cooldown_exit_be_bars: int = int(os.getenv("TB_COOLDOWN_EXIT_BE_BARS", "1"))
    cooldown_exit_hard_stop_bars: int = int(os.getenv("TB_COOLDOWN_EXIT_HARD_STOP_BARS", "3"))
    cooldown_exit_trail_bars: int = int(os.getenv("TB_COOLDOWN_EXIT_TRAIL_BARS", "0"))
    cooldown_reclaim_override_energy_min: float = float(os.getenv("TB_COOLDOWN_RECLAIM_OVERRIDE_ENERGY_MIN", "0.55"))
    cooldown_reclaim_override_abs_margin_min: float = float(os.getenv("TB_COOLDOWN_RECLAIM_OVERRIDE_ABS_MARGIN_MIN", "0.25"))

    # C) Flip Firewall + margin floors
    flip_lockout_candles: int = int(os.getenv("TB_FLIP_LOCKOUT_CANDLES", "3"))
    min_abs_margin_entry: float = float(os.getenv("TB_MIN_ABS_MARGIN_ENTRY", "0.15"))
    min_abs_margin_entry_flip: float = float(os.getenv("TB_MIN_ABS_MARGIN_ENTRY_FLIP", "0.20"))
    flip_override_energy_min: float = float(os.getenv("TB_FLIP_OVERRIDE_ENERGY_MIN", "0.60"))
    flip_override_abs_margin_min: float = float(os.getenv("TB_FLIP_OVERRIDE_ABS_MARGIN_MIN", "0.20"))
    flip_override_ema915_age_ms: int = int(os.getenv("TB_FLIP_OVERRIDE_EMA915_AGE_MS", "350"))

    # D) SL sanity hard gates
    max_sl_points: float = float(os.getenv("TB_MAX_SL_POINTS", "12.0"))
    max_sl_atr_mult: float = float(os.getenv("TB_MAX_SL_ATR_MULT", "1.5"))
    block_on_sl_hint_rejected: bool = os.getenv("TB_BLOCK_ON_SL_HINT_REJECTED", "1") == "1"

    # Trailing arming and early-hit debounce (hard stop always enforced)
    trail_start_atr: float = float(os.getenv("TB_TRAIL_START_ATR", "0.8"))
    trail_debounce_ms: int = int(os.getenv("TB_TRAIL_DEBOUNCE_MS", "25000"))

    # E) Regime-aware trailing profile
    regime_trailing_enable: bool = os.getenv("TB_REGIME_TRAILING_ENABLE", "1") == "1"
    trail_breakout_start_atr: float = float(os.getenv("TB_TRAIL_BREAKOUT_START_ATR", "0.9"))
    trail_breakout_dist_atr: float = float(os.getenv("TB_TRAIL_BREAKOUT_DIST_ATR", "0.9"))
    trail_trend_start_atr: float = float(os.getenv("TB_TRAIL_TREND_START_ATR", "0.7"))
    trail_trend_dist_atr: float = float(os.getenv("TB_TRAIL_TREND_DIST_ATR", "0.7"))
    trail_disable_in_chop: bool = os.getenv("TB_TRAIL_DISABLE_IN_CHOP", "1") == "1"

    # Wick exit gating (reduce churn in strong trends)
    wick_exit_min_profit_atr: float = float(os.getenv("TB_WICK_EXIT_MIN_PROFIT_ATR", "0.35"))
    wick_threshold: float = float(os.getenv("TB_WICK_THRESHOLD", "0.30"))  # volatility-aware default

    # Squeeze
    squeeze_lookback: int = int(os.getenv("TB_SQUEEZE_LOOKBACK", "120"))
    squeeze_pct: float = float(os.getenv("TB_SQUEEZE_PCT", "20.0"))  # bottom percentile of range

    # System / IO
    jsonl_path: str = os.getenv("TB_JSONL", "tradebrain_signal.jsonl")
    arm_jsonl_path: str = os.getenv("TB_ARM_JSONL", "tradebrain_arm.jsonl")
    write_all: bool = os.getenv("TB_WRITE_ALL", "1") == "1"
    write_hold: bool = os.getenv("TB_WRITE_HOLD", "0") == "1"

    # Optional: duplicate ARM stream into main JSONL for unified analysis
    dup_arm_to_main: bool = os.getenv("TB_DUP_ARM_TO_MAIN", "0") == "1"

    # F) Anti-overtrade limiter after impulse exits (TRAIL/TP)
    impulse_reentry_window_min: int = int(os.getenv("TB_IMPULSE_REENTRY_WINDOW_MIN", "5"))
    impulse_reentry_max: int = int(os.getenv("TB_IMPULSE_REENTRY_MAX", "1"))
    impulse_reentry_energy_min: float = float(os.getenv("TB_IMPULSE_REENTRY_ENERGY_MIN", "0.60"))
    impulse_reentry_abs_margin_min: float = float(os.getenv("TB_IMPULSE_REENTRY_ABS_MARGIN_MIN", "0.15"))

    # Tiered notifications (best-effort webhook)
    notify_enable: bool = os.getenv("TB_NOTIFY_ENABLE", "1") == "1"
    notify_webhook_url: str = os.getenv("TB_NOTIFY_WEBHOOK_URL", "").strip()
    notify_debounce_arm_s: int = int(os.getenv("TB_NOTIFY_DEBOUNCE_ARM_S", "60"))
    notify_debounce_go_s: int = int(os.getenv("TB_NOTIFY_DEBOUNCE_GO_S", "60"))
    notify_debounce_manage_s: int = int(os.getenv("TB_NOTIFY_DEBOUNCE_MANAGE_S", "20"))
    notify_latency_downgrade_ms: int = int(os.getenv("TB_NOTIFY_LATENCY_DOWNGRADE_MS", "600"))
    notify_arm_edgescore: float = float(os.getenv("TB_NOTIFY_ARM_EDGESCORE", "0.55"))
    notify_go_edgescore: float = float(os.getenv("TB_NOTIFY_GO_EDGESCORE", "0.70"))

    # Optional: Futures sidecar (Option C). Read once per 1m candle close.
    use_fut_flow: bool = os.getenv("TB_USE_FUT_FLOW", "1") == "1"
    fut_sidecar_path: str = os.getenv("TB_FUT_SIDECAR_PATH", "data/fut_candles.csv")
    fut_flow_stale_sec: int = int(os.getenv("TB_FUT_FLOW_STALE_SEC", "180"))
    fut_flow_fail_mode: str = os.getenv("TB_FUT_FLOW_FAIL_MODE", "neutral").strip().lower()
    fut_sidecar_poll_ms: int = int(os.getenv("TB_FUT_SIDECAR_POLL_MS", "1000"))

    # Activity-weighted pressure guard (Option B). Prevent CLV from overconfident signals in low-activity tape.
    activity_w_low: float = float(os.getenv("TB_ACTIVITY_W_LOW", "0.70"))
    activity_w_high: float = float(os.getenv("TB_ACTIVITY_W_HIGH", "1.30"))
    min_activity_w_confirm: float = float(os.getenv("TB_MIN_ACTIVITY_W_CONFIRM", "0.35"))

    # Limits
    max_candles: int = int(os.getenv("TB_MAX_CANDLES", "450"))  # ~full session of 1m
    max_ticks_per_candle: int = int(os.getenv("TB_MAX_TICKS_PER_CANDLE", "20000"))  # loop safety
    min_ready_candles: int = int(os.getenv("TB_MIN_READY_CANDLES", "10"))

    # Entry guards
    respect_bias: bool = os.getenv("TB_RESPECT_BIAS", "0") == "1"

    # EMA915
    ema915_arm_min_age_ms: int = int(os.getenv("TB_EMA915_ARM_MIN_AGE_MS", "200"))
    ema915_break_buffer_atr: float = float(os.getenv("TB_EMA915_BREAK_BUFFER_ATR", "0.12"))
    ema915_break_buffer_ticks: int = int(os.getenv("TB_EMA915_BREAK_BUFFER_TICKS", "3"))

    # Tick/micro params
    tick_size: float = float(os.getenv("TB_TICK_SIZE", "0.05"))
    micro_atr_fallback: float = float(os.getenv("TB_MICRO_ATR_FALLBACK", "12.0"))
    micro_accept_vel_z: float = float(os.getenv("TB_MICRO_ACCEPT_VEL_Z", "1.0"))
    micro_accept_acc_z_max_against: float = float(os.getenv("TB_MICRO_ACCEPT_ACC_Z_MAX_AGAINST", "3.0"))
    micro_arm_buffer_atr: float = float(os.getenv("TB_MICRO_ARM_BUFFER_ATR", "0.12"))
    micro_arm_buffer_ticks: int = int(os.getenv("TB_MICRO_ARM_BUFFER_TICKS", "20"))

    # Timebase
    use_exchange_timestamps: bool = os.getenv("TB_USE_EXCHANGE_TIMESTAMPS", "0") == "1"
    allow_chop_trades: bool = os.getenv("TB_ALLOW_CHOP_TRADES", "0") == "1"

    # ------------------------ Price Action Engine (3-candle) ------------------------
    pa_enable: bool = os.getenv("TB_PA_ENABLE", "1") == "1"
    pa_min_strength: float = float(os.getenv("TB_PA_MIN_STRENGTH", "0.62"))
    pa_veto_strength: float = float(os.getenv("TB_PA_VETO_STRENGTH", "0.78"))
    pa_dom_body_pct: float = float(os.getenv("TB_PA_DOM_BODY_PCT", "0.55"))
    pa_dom_clv: float = float(os.getenv("TB_PA_DOM_CLV", "0.65"))
    pa_dom_range_atr: float = float(os.getenv("TB_PA_DOM_RANGE_ATR", "0.75"))
    pa_rej_wick_pct: float = float(os.getenv("TB_PA_REJ_WICK_PCT", "0.45"))
    pa_rej_clv_max: float = float(os.getenv("TB_PA_REJ_CLV_MAX", "0.25"))
    pa_uturn_sweep_atr: float = float(os.getenv("TB_PA_UTURN_SWEEP_ATR", "0.18"))
    pa_uturn_reclaim_clv: float = float(os.getenv("TB_PA_UTURN_RECLAIM_CLV", "0.35"))
    allow_chop_pa_exceptions: bool = os.getenv("TB_ALLOW_CHOP_PA_EXCEPTIONS", "1") == "1"
    chop_pa_min_strength: float = float(os.getenv("TB_CHOP_PA_MIN_STRENGTH", "0.72"))
    pa_countertrend_min_energy: float = float(os.getenv("TB_PA_COUNTERTREND_MIN_ENERGY", "0.72"))
    pa_countertrend_min_margin: float = float(os.getenv("TB_PA_COUNTERTREND_MIN_MARGIN", "0.28"))
    pa_countertrend_allow_chop: bool = os.getenv("TB_PA_COUNTERTREND_ALLOW_CHOP", "0") == "1"

    # ------------------------ Breakout Energy Window ------------------------
    breakout_enable: bool = os.getenv("TB_BREAKOUT_ENABLE", "1") == "1"
    breakout_energy_entry_min: float = float(os.getenv("TB_BREAKOUT_ENERGY_ENTRY_MIN", "0.62"))
    breakout_energy_keep_min: float = float(os.getenv("TB_BREAKOUT_ENERGY_KEEP_MIN", "0.42"))
    breakout_energy_decay_base: float = float(os.getenv("TB_BREAKOUT_ENERGY_DECAY_BASE", "0.74"))
    breakout_energy_decay_eff_w: float = float(os.getenv("TB_BREAKOUT_ENERGY_DECAY_EFF_W", "0.22"))
    breakout_energy_decay_smooth_w: float = float(os.getenv("TB_BREAKOUT_ENERGY_DECAY_SMOOTH_W", "0.10"))
    breakout_kill_reclaim_atr: float = float(os.getenv("TB_BREAKOUT_KILL_RECLAIM_ATR", "0.35"))
    breakout_runaway_atr: float = float(os.getenv("TB_BREAKOUT_RUNAWAY_ATR", "1.25"))

    # ------------------------ Retest State ------------------------
    retest_runaway_atr_min: float = float(os.getenv("TB_RETEST_RUNAWAY_ATR_MIN", "0.85"))
    retest_runaway_atr_max: float = float(os.getenv("TB_RETEST_RUNAWAY_ATR_MAX", "1.55"))
    retest_invalid_atr: float = float(os.getenv("TB_RETEST_INVALID_ATR", "0.25"))
    retest_touch_ticks: int = int(os.getenv("TB_RETEST_TOUCH_TICKS", "2"))

    # ------------------------ Shock Energy ------------------------
    shock_energy_min: float = float(os.getenv("TB_SHOCK_ENERGY_MIN", "0.25"))
    shock_energy_decay_base: float = float(os.getenv("TB_SHOCK_ENERGY_DECAY_BASE", "0.78"))
    shock_energy_decay_eff_w: float = float(os.getenv("TB_SHOCK_ENERGY_DECAY_EFF_W", "0.18"))
    shock_energy_decay_act_w: float = float(os.getenv("TB_SHOCK_ENERGY_DECAY_ACT_W", "0.12"))

    # ------------------------ Dynamic TP/Runner ------------------------
    tp_mult_chop_min: float = float(os.getenv("TB_TP_MULT_CHOP_MIN", "0.9"))
    tp_mult_chop_max: float = float(os.getenv("TB_TP_MULT_CHOP_MAX", "1.4"))
    tp_mult_trend_min: float = float(os.getenv("TB_TP_MULT_TREND_MIN", "1.4"))
    tp_mult_trend_max: float = float(os.getenv("TB_TP_MULT_TREND_MAX", "2.4"))
    tp_mult_break_min: float = float(os.getenv("TB_TP_MULT_BREAK_MIN", "1.8"))
    tp_mult_break_max: float = float(os.getenv("TB_TP_MULT_BREAK_MAX", "3.2"))
    runner_conf_break: float = float(os.getenv("TB_RUNNER_CONF_BREAK", "0.78"))
    runner_conf_trend: float = float(os.getenv("TB_RUNNER_CONF_TREND", "0.82"))
    runner_energy_min: float = float(os.getenv("TB_RUNNER_ENERGY_MIN", "0.72"))
    runner_pa_min: float = float(os.getenv("TB_RUNNER_PA_MIN", "0.70"))

    # ------------------------ Tick engine anchor alignment ------------------------
    anchor_align_max_against_atr: float = float(os.getenv("TB_ANCHOR_ALIGN_MAX_AGAINST_ATR", "0.60"))

    def validate(self) -> None:
        if self.hma_period < 5:
            raise ValueError("hma_period must be >= 5")
        if self.atr_period < 2:
            raise ValueError("atr_period must be >= 2")
        if self.stretch_limit <= 0:
            raise ValueError("stretch_limit must be > 0")
        if not (0.0 < self.min_path_eff_filter < 1.0):
            raise ValueError("min_path_eff_filter must be in (0,1)")
        if not (0.0 < self.entry_path_eff < 1.0):
            raise ValueError("entry_path_eff must be in (0,1)")
        if self.entry_anchor_min_atr < 0:
            raise ValueError("entry_anchor_min_atr must be >= 0")
        if self.max_anchor_dist_atr <= 0:
            raise ValueError("max_anchor_dist_atr must be > 0")

        # Contextual entry veto sanity
        if self.late_breakout_anchor_atr < 0:
            raise ValueError("late_breakout_anchor_atr must be >= 0")
        if not (0.0 < self.lip_breakout_atr <= 1.0):
            raise ValueError("lip_breakout_atr must be in (0,1]")
        if self.climax_streak < 1:
            raise ValueError("climax_streak must be >= 1")
        if self.climax_anchor_atr < 0:
            raise ValueError("climax_anchor_atr must be >= 0")
        if not (0.0 <= self.flow_bias_veto <= 1.0):
            raise ValueError("flow_bias_veto must be in [0,1]")
        if self.support_lookback < 10:
            raise ValueError("support_lookback must be >= 10")
        if self.sr_buffer_atr < 0:
            raise ValueError("sr_buffer_atr must be >= 0")
        if self.entry_rr_min < 0:
            raise ValueError("entry_rr_min must be >= 0")
        if not (0.0 < self.absorb_wick_pct <= 1.0):
            raise ValueError("absorb_wick_pct must be in (0,1]")
        if not (0.0 <= self.absorb_clv_min <= 1.0):
            raise ValueError("absorb_clv_min must be in [0,1]")
        if self.absorb_cooldown_ms < 0:
            raise ValueError("absorb_cooldown_ms must be >= 0")
        if self.entry_latency_max_ms < 0:
            raise ValueError("entry_latency_max_ms must be >= 0")
        if not (0.0 <= self.ready_margin <= 1.0):
            raise ValueError("ready_margin must be in [0,1]")
        if not (0.0 <= self.flip_margin <= 1.0):
            raise ValueError("flip_margin must be in [0,1]")
        if not (0.0 <= self.conflict_margin <= 1.0):
            raise ValueError("conflict_margin must be in [0,1]")
        if self.intent_conflict_score_delta < 0:
            raise ValueError("intent_conflict_score_delta must be >= 0")
        if self.intent_flip_score_delta < 0:
            raise ValueError("intent_flip_score_delta must be >= 0")
        if self.reverse_min_move_atr < 0:
            raise ValueError("reverse_min_move_atr must be >= 0")
        if not (-1.0 <= self.min_clv_confirm <= 1.0):
            raise ValueError("min_clv_confirm must be in [-1,1]")
        if not (0.0 < self.min_body_pct < 1.0):
            raise ValueError("min_body_pct must be in (0,1)")
        if not (0.0 < self.strong_clv <= 1.0):
            raise ValueError("strong_clv must be in (0,1]")
        if self.shock_atr_mult <= 0:
            raise ValueError("shock_atr_mult must be > 0")
        if self.shock_points <= 0:
            raise ValueError("shock_points must be > 0")
        if self.shock_confirm_candles < 1:
            raise ValueError("shock_confirm_candles must be >= 1")
        if self.shock_lock_ms < 0:
            raise ValueError("shock_lock_ms must be >= 0")
        if self.shock_expiry_minutes < 1:
            raise ValueError("shock_expiry_minutes must be >= 1")
        if self.hard_stop_points <= 0:
            raise ValueError("hard_stop_points must be > 0")
        if self.tp_atr_mult <= 0:
            raise ValueError("tp_atr_mult must be > 0")
        if self.move_to_be_atr <= 0:
            raise ValueError("move_to_be_atr must be > 0")
        if self.be_buffer_points < 0:
            raise ValueError("be_buffer_points must be >= 0")
        if self.min_init_sl_points <= 0:
            raise ValueError("min_init_sl_points must be > 0")
        if self.min_init_sl_atr < 0:
            raise ValueError("min_init_sl_atr must be >= 0")
        if self.hard_stop_atr_mult <= 0:
            raise ValueError("hard_stop_atr_mult must be > 0")
        if self.cooldown_reclaim_atr < 0:
            raise ValueError("cooldown_reclaim_atr must be >= 0")
        if self.cooldown_min_bars < 0:
            raise ValueError("cooldown_min_bars must be >= 0")
        if self.stall_bars < 0:
            raise ValueError("stall_bars must be >= 0")
        if self.stall_min_mfe_atr < 0:
            raise ValueError("stall_min_mfe_atr must be >= 0")
        if not (0.0 <= self.proactive_lite_energy_min <= 1.0):
            raise ValueError("proactive_lite_energy_min must be in [0,1]")
        if not (0.0 <= self.proactive_lite_abs_margin_min <= 1.0):
            raise ValueError("proactive_lite_abs_margin_min must be in [0,1]")
        if not (0.0 <= self.proactive_lite_abs_margin_min_ext <= 1.0):
            raise ValueError("proactive_lite_abs_margin_min_ext must be in [0,1]")
        if self.proactive_lite_extension_atr_min < 0:
            raise ValueError("proactive_lite_extension_atr_min must be >= 0")
        if not (0.0 <= self.proactive_lite_path_eff_min <= 1.0):
            raise ValueError("proactive_lite_path_eff_min must be in [0,1]")
        if not (0.0 <= self.proactive_lite_conf_min <= 1.0):
            raise ValueError("proactive_lite_conf_min must be in [0,1]")
        if self.proactive_lite_latency_max_ms < 0:
            raise ValueError("proactive_lite_latency_max_ms must be >= 0")
        for k, v in (
            ("cooldown_exit_init_sl_bars", self.cooldown_exit_init_sl_bars),
            ("cooldown_exit_be_bars", self.cooldown_exit_be_bars),
            ("cooldown_exit_hard_stop_bars", self.cooldown_exit_hard_stop_bars),
            ("cooldown_exit_trail_bars", self.cooldown_exit_trail_bars),
            ("flip_lockout_candles", self.flip_lockout_candles),
            ("flip_override_ema915_age_ms", self.flip_override_ema915_age_ms),
            ("impulse_reentry_window_min", self.impulse_reentry_window_min),
            ("impulse_reentry_max", self.impulse_reentry_max),
            ("notify_debounce_arm_s", self.notify_debounce_arm_s),
            ("notify_debounce_go_s", self.notify_debounce_go_s),
            ("notify_debounce_manage_s", self.notify_debounce_manage_s),
            ("notify_latency_downgrade_ms", self.notify_latency_downgrade_ms),
        ):
            if int(v) < 0:
                raise ValueError(f"{k} must be >= 0")
        if not (0.0 <= self.cooldown_reclaim_override_energy_min <= 1.0):
            raise ValueError("cooldown_reclaim_override_energy_min must be in [0,1]")
        if not (0.0 <= self.cooldown_reclaim_override_abs_margin_min <= 1.0):
            raise ValueError("cooldown_reclaim_override_abs_margin_min must be in [0,1]")
        if not (0.0 <= self.min_abs_margin_entry <= 1.0):
            raise ValueError("min_abs_margin_entry must be in [0,1]")
        if not (0.0 <= self.min_abs_margin_entry_flip <= 1.0):
            raise ValueError("min_abs_margin_entry_flip must be in [0,1]")
        if not (0.0 <= self.flip_override_energy_min <= 1.0):
            raise ValueError("flip_override_energy_min must be in [0,1]")
        if not (0.0 <= self.flip_override_abs_margin_min <= 1.0):
            raise ValueError("flip_override_abs_margin_min must be in [0,1]")
        if self.max_sl_points <= 0:
            raise ValueError("max_sl_points must be > 0")
        if self.max_sl_atr_mult <= 0:
            raise ValueError("max_sl_atr_mult must be > 0")
        if self.trail_atr <= 0 or self.harvest_trail_atr <= 0:
            raise ValueError("trail_atr and harvest_trail_atr must be > 0")
        if self.trail_start_atr < 0:
            raise ValueError("trail_start_atr must be >= 0")
        if self.trail_debounce_ms < 0:
            raise ValueError("trail_debounce_ms must be >= 0")
        if self.trail_breakout_start_atr < 0:
            raise ValueError("trail_breakout_start_atr must be >= 0")
        if self.trail_breakout_dist_atr <= 0:
            raise ValueError("trail_breakout_dist_atr must be > 0")
        if self.trail_trend_start_atr < 0:
            raise ValueError("trail_trend_start_atr must be >= 0")
        if self.trail_trend_dist_atr <= 0:
            raise ValueError("trail_trend_dist_atr must be > 0")
        if self.wick_exit_min_profit_atr < 0:
            raise ValueError("wick_exit_min_profit_atr must be >= 0")
        if not (0.05 <= self.wick_threshold <= 0.90):
            raise ValueError("wick_threshold should be in [0.05,0.90]")
        if self.squeeze_lookback < 20:
            raise ValueError("squeeze_lookback must be >= 20")
        if not (1.0 <= self.squeeze_pct <= 50.0):
            raise ValueError("squeeze_pct must be in [1,50]")
        if self.max_candles < 50:
            raise ValueError("max_candles must be >= 50")
        if self.max_ticks_per_candle < 100:
            raise ValueError("max_ticks_per_candle must be >= 100")
        if self.min_ready_candles < 5:
            raise ValueError("min_ready_candles must be >= 5")

        # HTF / micro hardening validation
        if self.htf_slope_thresh < 0:
            raise ValueError("htf_slope_thresh must be >= 0")
        if self.htf_local_lookback < 5:
            raise ValueError("htf_local_lookback must be >= 5")
        if self.htf_reversal_hold_bars < 1:
            raise ValueError("htf_reversal_hold_bars must be >= 1")
        if self.htf_reversal_buffer_ticks < 0:
            raise ValueError("htf_reversal_buffer_ticks must be >= 0")
        if not (0.0 <= self.micro_harden_clv <= 1.0):
            raise ValueError("micro_harden_clv must be in [0,1]")
        if not (0.0 <= self.micro_harden_eff <= 1.0):
            raise ValueError("micro_harden_eff must be in [0,1]")
        if not (0.0 <= self.micro_flow_veto <= 1.0):
            raise ValueError("micro_flow_veto must be in [0,1]")

        # Sidecar / activity validation
        if self.fut_flow_stale_sec < 1:
            raise ValueError("TB_FUT_FLOW_STALE_SEC must be >= 1")
        if self.fut_flow_fail_mode not in ("neutral", "hold"):
            raise ValueError("TB_FUT_FLOW_FAIL_MODE must be 'neutral' or 'hold'")
        if self.fut_sidecar_poll_ms < 100:
            raise ValueError("TB_FUT_SIDECAR_POLL_MS must be >= 100")
        if self.activity_w_high <= self.activity_w_low:
            raise ValueError("TB_ACTIVITY_W_HIGH must be > TB_ACTIVITY_W_LOW")
        if not (0.0 <= self.min_activity_w_confirm <= 1.0):
            raise ValueError("TB_MIN_ACTIVITY_W_CONFIRM must be between 0 and 1")
        if self.tick_size <= 0:
            raise ValueError("TB_TICK_SIZE must be > 0")
        if self.micro_atr_fallback <= 0:
            raise ValueError("TB_MICRO_ATR_FALLBACK must be > 0")
        if self.micro_accept_vel_z < 0:
            raise ValueError("TB_MICRO_ACCEPT_VEL_Z must be >= 0")
        if self.micro_accept_acc_z_max_against < 0:
            raise ValueError("TB_MICRO_ACCEPT_ACC_Z_MAX_AGAINST must be >= 0")
        if self.micro_arm_buffer_atr < 0:
            raise ValueError("TB_MICRO_ARM_BUFFER_ATR must be >= 0")
        if self.micro_arm_buffer_ticks < 0:
            raise ValueError("TB_MICRO_ARM_BUFFER_TICKS must be >= 0")
        if self.ema915_break_buffer_atr < 0:
            raise ValueError("TB_EMA915_BREAK_BUFFER_ATR must be >= 0")
        if self.ema915_break_buffer_ticks < 0:
            raise ValueError("TB_EMA915_BREAK_BUFFER_TICKS must be >= 0")
        if not (0.0 <= self.pa_min_strength <= 1.0):
            raise ValueError("pa_min_strength must be in [0,1]")
        if not (0.0 <= self.pa_veto_strength <= 1.0):
            raise ValueError("pa_veto_strength must be in [0,1]")
        if not (0.0 < self.pa_dom_body_pct <= 1.0):
            raise ValueError("pa_dom_body_pct must be in (0,1]")
        if not (0.0 <= self.pa_dom_clv <= 1.0):
            raise ValueError("pa_dom_clv must be in [0,1]")
        if self.pa_dom_range_atr < 0:
            raise ValueError("pa_dom_range_atr must be >= 0")
        if not (0.0 < self.pa_rej_wick_pct <= 1.0):
            raise ValueError("pa_rej_wick_pct must be in (0,1]")
        if not (0.0 <= self.pa_rej_clv_max <= 1.0):
            raise ValueError("pa_rej_clv_max must be in [0,1]")
        if self.pa_uturn_sweep_atr < 0:
            raise ValueError("pa_uturn_sweep_atr must be >= 0")
        if not (0.0 <= self.pa_uturn_reclaim_clv <= 1.0):
            raise ValueError("pa_uturn_reclaim_clv must be in [0,1]")
        if not (0.0 <= self.chop_pa_min_strength <= 1.0):
            raise ValueError("chop_pa_min_strength must be in [0,1]")
        if not (0.0 <= self.pa_countertrend_min_energy <= 1.0):
            raise ValueError("pa_countertrend_min_energy must be in [0,1]")
        if not (0.0 <= self.pa_countertrend_min_margin <= 1.0):
            raise ValueError("pa_countertrend_min_margin must be in [0,1]")
        if not (0.0 <= self.breakout_energy_entry_min <= 1.0):
            raise ValueError("breakout_energy_entry_min must be in [0,1]")
        if not (0.0 <= self.breakout_energy_keep_min <= 1.0):
            raise ValueError("breakout_energy_keep_min must be in [0,1]")
        if not (0.0 < self.breakout_energy_decay_base <= 1.0):
            raise ValueError("breakout_energy_decay_base must be in (0,1]")
        if self.breakout_energy_decay_eff_w < 0 or self.breakout_energy_decay_smooth_w < 0:
            raise ValueError("breakout_energy_decay_* weights must be >= 0")
        if self.breakout_kill_reclaim_atr < 0:
            raise ValueError("breakout_kill_reclaim_atr must be >= 0")
        if self.breakout_runaway_atr < 0:
            raise ValueError("breakout_runaway_atr must be >= 0")
        if self.retest_runaway_atr_min < 0 or self.retest_runaway_atr_max < 0:
            raise ValueError("retest_runaway_atr_* must be >= 0")
        if self.retest_runaway_atr_min > self.retest_runaway_atr_max:
            raise ValueError("retest_runaway_atr_min must be <= retest_runaway_atr_max")
        if self.retest_invalid_atr < 0:
            raise ValueError("retest_invalid_atr must be >= 0")
        if self.retest_touch_ticks < 0:
            raise ValueError("retest_touch_ticks must be >= 0")
        if not (0.0 <= self.shock_energy_min <= 1.0):
            raise ValueError("shock_energy_min must be in [0,1]")
        if not (0.0 < self.shock_energy_decay_base <= 1.0):
            raise ValueError("shock_energy_decay_base must be in (0,1]")
        if self.shock_energy_decay_eff_w < 0 or self.shock_energy_decay_act_w < 0:
            raise ValueError("shock_energy_decay_* weights must be >= 0")
        for a, b, nm in [
            (self.tp_mult_chop_min, self.tp_mult_chop_max, "tp_mult_chop"),
            (self.tp_mult_trend_min, self.tp_mult_trend_max, "tp_mult_trend"),
            (self.tp_mult_break_min, self.tp_mult_break_max, "tp_mult_break"),
        ]:
            if a <= 0 or b <= 0:
                raise ValueError(f"{nm}_min/max must be > 0")
            if a > b:
                raise ValueError(f"{nm}_min must be <= {nm}_max")
        if not (0.0 <= self.runner_conf_break <= 1.0):
            raise ValueError("runner_conf_break must be in [0,1]")
        if not (0.0 <= self.runner_conf_trend <= 1.0):
            raise ValueError("runner_conf_trend must be in [0,1]")
        if not (0.0 <= self.runner_energy_min <= 1.0):
            raise ValueError("runner_energy_min must be in [0,1]")
        if not (0.0 <= self.runner_pa_min <= 1.0):
            raise ValueError("runner_pa_min must be in [0,1]")
        if self.anchor_align_max_against_atr < 0:
            raise ValueError("anchor_align_max_against_atr must be >= 0")
        if not (0.0 <= self.impulse_reentry_energy_min <= 1.0):
            raise ValueError("impulse_reentry_energy_min must be in [0,1]")
        if not (0.0 <= self.impulse_reentry_abs_margin_min <= 1.0):
            raise ValueError("impulse_reentry_abs_margin_min must be in [0,1]")
        if not (0.0 <= self.notify_arm_edgescore <= 1.0):
            raise ValueError("notify_arm_edgescore must be in [0,1]")
        if not (0.0 <= self.notify_go_edgescore <= 1.0):
            raise ValueError("notify_go_edgescore must be in [0,1]")

    @staticmethod
    def from_env() -> "TradeBrainConfig":
        cfg = TradeBrainConfig()
        log_file = _env("TB_LOG_FILE", cfg.log_file, aliases=["TB_FULL_LOG_FILE"])
        log_level = _parse_level(_env("TB_LOG_LEVEL", str(cfg.log_level), aliases=["TB_FULL_LOG_LEVEL"]))
        log_to_console = _env_bool("TB_LOG_TO_CONSOLE", cfg.log_to_console, aliases=["TB_FULL_LOG_TO_CONSOLE"])

        default_jsonl = cfg.jsonl_path
        if not os.path.exists(default_jsonl) and os.path.exists("tradebrain_signal.jsonl"):
            default_jsonl = "tradebrain_signal.jsonl"
        jsonl_path = _env("TB_JSONL", default_jsonl, aliases=["TB_FULL_JSONL"])
        arm_jsonl_path = _env("TB_ARM_JSONL", cfg.arm_jsonl_path, aliases=["TB_FULL_ARM_JSONL"])
        write_hold = _env_bool("TB_WRITE_HOLD", cfg.write_hold, aliases=["TB_FULL_WRITE_HOLD"])
        write_all = _env_bool("TB_WRITE_ALL", cfg.write_all, aliases=["TB_FULL_WRITE_ALL"])
        return replace(
            cfg,
            log_file=log_file,
            log_level=log_level,
            log_to_console=log_to_console,
            jsonl_path=jsonl_path,
            arm_jsonl_path=arm_jsonl_path,
            write_hold=write_hold,
            write_all=write_all,
        )


@dataclass
class EntryIntent:
    engine: str  # "micro" | "ema915" | ...
    side: str  # "LONG" | "SHORT"
    entry_px: float
    ts: datetime
    reason: str
    score: float  # arbitration score
    sl_hint: Optional[float] = None
    tp_hint: Optional[float] = None
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Position:
    side: str  # "LONG" or "SHORT"
    entry_px: float
    entry_ts: datetime
    entry_bucket_utc: Optional[datetime]
    hard_sl: float
    sl: float
    sl_init: float
    tp: Optional[float]
    best_px: float
    entry_atr: float = 0.0
    be_trigger_points: float = 0.0
    trail_arm_points: float = 0.0
    rr_est: Optional[float] = None
    regime: str = "UNKNOWN"
    setup_quality: str = "C"
    confidence: float = 0.0
    is_be: bool = False
    is_runner: bool = False
    is_harvest: bool = False
    trail_atr_current: float = 0.0
    trail_armed: bool = False
    trail_arm_ts: Optional[datetime] = None
    engine: str = "candle"
    why: str = ""


@dataclass
class PendingShock:
    side: str  # "LONG" or "SHORT"
    mid_px: float
    energy: float
    origin: str = "shock"
    best_px: float = 0.0
    expires_at: Optional[datetime] = None
    last_candle_key: Optional[str] = None


@dataclass
class BreakoutState:
    side: str  # "LONG" or "SHORT"
    level: float
    origin: str
    energy: float
    best_px: float
    last_candle_key: Optional[str] = None


class TradeBrain:
    def __init__(self, cfg: TradeBrainConfig, log: logging.Logger):
        cfg.validate()
        self.cfg = cfg
        self.log = log

        self._process_lock = threading.Lock()
        self._lock = threading.Lock()
        self.candles: Deque[Dict[str, Any]] = deque(maxlen=self.cfg.max_candles)
        self.current_candle: Optional[Dict[str, Any]] = None

        # Session structure (HOD/LOD) + scalper cooldown (used by absorption veto)
        self.session_high: float = -float("inf")
        self.session_low: float = float("inf")
        self._entry_cooldown_until_ms_by_side: Dict[str, int] = {"LONG": 0, "SHORT": 0}
        self._session_day_ist: Optional[str] = None

        self.pos: Optional[Position] = None
        self.bias: Optional[str] = None  # "LONG" or "SHORT" or None
        self.pending_shock: Optional[PendingShock] = None
        self._breakout_state: Optional[BreakoutState] = None
        self._last_pa: Optional[Dict[str, Any]] = None
        self._last_fut_cvd: float = 0.0

        # Latest futures snapshot carried into tick-level gating (stale-checked)
        self._last_fut_flow_snapshot: Optional[Dict[str, Any]] = None
        self._last_fut_flow_ts_utc: Optional[datetime] = None
        self._fut_sidecar_cache_lock = threading.Lock()
        self._fut_sidecar_cache: Optional[Dict[str, Any]] = None
        self._fut_sidecar_cache_status: str = "UNSET"
        self._fut_sidecar_cache_fetch_ms: int = 0

        # Flip-control: never flip within the same 1m candle bucket
        self._last_exit_bucket_utc: Optional[datetime] = None

        # Market-driven setup memory (prevents flip-flops)
        self._setup_side: Optional[str] = None  # "LONG" / "SHORT" / None
        self._setup_margin: float = 0.0  # last computed margin in [-1..1]
        self._setup_candle_key: Optional[str] = None  # candle key we last updated setup on

        # Reversal context (for reversal-proof)
        self._last_exit_side: Optional[str] = None
        self._last_exit_px: Optional[float] = None
        self._last_exit_ms: int = 0
        self._stopout_cooldown_by_side: Dict[str, Optional[Dict[str, Any]]] = {"LONG": None, "SHORT": None}

        # HTF LONG reversal proof state (break + hold above shelf)
        self._htf_long_rev_level: Optional[float] = None
        self._htf_long_rev_hold: int = 0
        self._htf_long_rev_key: Optional[str] = None

        # Conflict logging de-dupe (avoid spamming HOLD_CONFLICT)
        self._hold_conflict_key: Optional[str] = None
        self._intent_conflict_info: Optional[Dict[str, Any]] = None
        self._retest_ctx: Optional[Dict[str, Any]] = None

        # Diagnostic state
        self._last_diag: Dict[str, Any] = {}
        self._range_hist: Deque[float] = deque(maxlen=self.cfg.squeeze_lookback)
        self._travel_hist: Deque[float] = deque(maxlen=60)  # baseline for PAV proxy
        self._tpt_hist: Deque[float] = deque(maxlen=60)

        # Tick-time microstructure engine
        # Store arrival-ms (not feed timestamps) so 150–400ms windows are real
        self._tbuf: Deque[Tuple[int, float]] = deque(maxlen=400)
        self._armed: Optional[Dict[str, Any]] = None
        self._last_vel: float = 0.0
        self._last_vel_ms: Optional[int] = None
        self._last_ms_eval: Optional[int] = None
        self._MS_EVAL_EVERY = 50
        self._ARM_TIMEOUT_MS = 1500
        self._HOLD_FAST_MS = 150
        self._HOLD_SLOW_MS = 400
        self._MICRO_ATR_FALLBACK = float(cfg.micro_atr_fallback)
        self._last_px: float = 0.0
        self._TICK_SIZE = float(cfg.tick_size)
        self._arm_log_key: Optional[Tuple[str, str, str]] = None
        self._last_entry_side: Optional[str] = None
        self._last_entry_ms: int = 0
        self._rearm_required: Dict[str, bool] = {"micro": False, "ema915": False}
        self._rearm_epoch_ms: Dict[str, int] = {"micro": 0, "ema915": 0}
        self._shock_lock_side: Optional[str] = None
        self._shock_lock_until_ms: int = 0
        self._fut_flow_warn_min: Optional[str] = None
        self._last_recv_ms: Optional[int] = None
        self._hold_wait_candle_key: Optional[str] = None
        self._hold_wait_reasons: set[str] = set()
        self._impulse_reentry_guard_by_side: Dict[str, Optional[Dict[str, Any]]] = {"LONG": None, "SHORT": None}
        self._notify_last_sent_by_key: Dict[str, int] = {}

        # EMA915 tick engine state (micro-bars + EMAs)
        self._ema915_bar_ms = int(getattr(cfg, "ema915_bar_ms", 1000))
        self._ema915_min_angle = float(getattr(cfg, "ema915_min_angle_deg", 30.0))
        self._ema915_max_angle = float(getattr(cfg, "ema915_max_angle_deg", 80.0))
        self._ema915_min_angle_deg = self._ema915_min_angle
        self._ema915_max_angle_deg = self._ema915_max_angle
        self._ema915_slope_lb = int(getattr(cfg, "ema915_slope_lookback_bars", 3))
        self._ema915_touch_pad = float(getattr(cfg, "ema915_touch_pad_pts", 0.75))
        self._ema915_arm_timeout_ms = int(getattr(cfg, "ema915_arm_timeout_ms", 1500))
        self._ema915_arm_min_age_ms = int(getattr(cfg, "ema915_arm_min_age_ms", 200))
        self._ema915_eval_every_ms = int(getattr(cfg, "ema915_eval_every_ms", 50))

        self._ema915_curbar: Optional[Dict[str, Any]] = None
        self._ema915_bars: Deque[Dict[str, Any]] = deque(maxlen=1200)
        self._ema9: Optional[float] = None
        self._ema15: Optional[float] = None
        self._ema9_hist: Deque[float] = deque(maxlen=50)
        self._ema15_hist: Deque[float] = deque(maxlen=50)
        self._ema915_armed: Optional[Dict[str, Any]] = None
        self._ema915_last_eval_ms: int = 0

        # Writer setup (safe even if jsonl_path has no directory)
        out_dir = os.path.dirname(self.cfg.jsonl_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        arm_dir = os.path.dirname(self.cfg.arm_jsonl_path)
        if arm_dir:
            os.makedirs(arm_dir, exist_ok=True)

        log_dir = os.path.dirname(self.cfg.log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

        candle_path = str(getattr(self.cfg, "candle_db_file", "") or "")
        candle_dir = os.path.dirname(candle_path)
        if candle_dir:
            os.makedirs(candle_dir, exist_ok=True)

        # Hydrate today's session candles on startup so SR survives restarts.
        if bool(getattr(self.cfg, "persist_candles", False)):
            try:
                with self._lock:
                    self._load_session_candles_locked()
            except Exception:
                self.log.exception("load_session_candles failed")
        if self.candles:
            cts = self.candles[-1].get("ts")
            if isinstance(cts, datetime):
                self._session_day_ist = cts.astimezone(IST_TZ).strftime("%Y-%m-%d")

        self._write_lock = threading.Lock()
        self.jsonl_file = open(self.cfg.jsonl_path, "a", encoding="utf-8", buffering=1)
        self._arm_write_lock = threading.Lock()
        self.arm_jsonl_file = open(self.cfg.arm_jsonl_path, "a", encoding="utf-8", buffering=1)

    # ------------------------ Public API ------------------------
    def on_tick(self, tick: Dict[str, Any]) -> None:
        self.process_tick(tick)

    def process_tick(self, tick: Dict[str, Any]) -> None:
        """Single-threaded wrapper for main tick pipeline."""
        with self._process_lock:
            self._process_tick_unsafe(tick)

    def _process_tick_unsafe(self, tick: Dict[str, Any]) -> None:
        """Main entry point for incoming ticks. Safe under bad packets."""
        try:
            ltp = _safe_float(tick.get("ltp"))
            if ltp is None:
                return

            recv_ns = int(tick.get("recv_ns") or time.time_ns())
            recv_ms = int(tick.get("recv_ms") or (recv_ns // 1_000_000))
            recv_ts_utc = datetime.fromtimestamp(recv_ms / 1000.0, tz=timezone.utc)
            self._last_recv_ms = recv_ms
            try:
                self._refresh_fut_sidecar_cache(recv_ms)
            except Exception:
                self.log.exception("refresh_fut_sidecar_cache failed")

            raw_tick_ts = tick.get("timestamp")
            tick_ts_utc = _parse_ts(raw_tick_ts) if raw_tick_ts is not None else None
            tick_ts_fixed, latency_ms, latency_status = _fix_tick_time(
                tick_ts_utc,
                recv_ts_utc,
                drift_sec=int(getattr(self.cfg, "ts_drift_sec", 19800)),
                drift_tol_sec=int(getattr(self.cfg, "ts_drift_tolerance_sec", 120)),
                max_abs_latency_ms=int(getattr(self.cfg, "max_abs_latency_ms", 300000)),
            )

            # Internal timebase is configurable:
            # - recv_ts_utc (default; execution-coherent)
            # - corrected exchange tick time (optional)
            if bool(getattr(self.cfg, "use_exchange_timestamps", False)) and tick_ts_fixed is not None:
                ts = tick_ts_fixed
            else:
                ts = recv_ts_utc
            candle_ts = ts.replace(second=0, microsecond=0)

            with self._lock:
                self._maybe_roll_session_locked(candle_ts)
                if self._shock_lock_side is not None and recv_ms >= int(self._shock_lock_until_ms):
                    self._shock_lock_side = None

                # Guard against out-of-order ticks (prevents stale exits/candle corruption)
                latest_ts = None
                if self.current_candle is not None:
                    latest_ts = self.current_candle.get("ts")
                elif self.candles:
                    latest_ts = self.candles[-1].get("ts")
                if latest_ts is not None and candle_ts < latest_ts:
                    return

                # Kill stale shocks even if candle close is slow / feed lags
                if self.pending_shock is not None:
                    exp = getattr(self.pending_shock, "expires_at", None)
                    if isinstance(exp, datetime) and _now_utc() > exp:
                        self.pending_shock = None

                # 1) Candle management (close previous minute)
                if self.current_candle and candle_ts > self.current_candle["ts"]:
                    self._close_candle_locked()

                if self.current_candle is None:
                    self.current_candle = {
                        "ts": candle_ts,
                        "open": ltp,
                        "high": ltp,
                        "low": ltp,
                        "close": ltp,
                        "ticks": 0,
                        "total_travel": 0.0,
                        "rv2": 0.0,
                        "last_px": ltp,
                    }

                c = self.current_candle

                # Infinite-loop / runaway safety: ignore absurd tick bursts
                if c["ticks"] >= self.cfg.max_ticks_per_candle:
                    # Do NOT stop processing during tick explosions.
                    # Instead, mark overload and log once (prevents silent drops in fast markets).
                    if not bool(c.get("_tick_overrun", False)):
                        c["_tick_overrun"] = True
                        c["_tick_overrun_at"] = int(c.get("ticks", 0))
                        try:
                            self.log.warning(
                                "tick_overrun: minute=%s ticks=%s >= max_ticks_per_candle=%s; continuing without dropping ticks",
                                str(c.get("ts")),
                                int(c.get("ticks", 0)),
                                int(self.cfg.max_ticks_per_candle),
                            )
                        except Exception:
                            pass


                delta = ltp - float(c["last_px"])
                c["high"] = max(float(c["high"]), ltp)
                c["low"] = min(float(c["low"]), ltp)
                c["total_travel"] = float(c["total_travel"]) + abs(delta)
                c["rv2"] = float(c["rv2"]) + (delta * delta)
                c["last_px"] = ltp
                c["close"] = ltp
                c["ticks"] = int(c["ticks"]) + 1

                self._last_px = ltp

                # 2) Intra-tick exits (serialized under lock)
                exit_decision = None
                if self.pos is not None:
                    exit_decision = self._check_intra_tick_exits(ltp, ts, recv_ms=recv_ms)
                if isinstance(exit_decision, dict):
                    self._write_signal(exit_decision, self._mk_tick_metrics({}, ltp, ts, recv_ms=recv_ms, recv_ns=recv_ns))
                    return

                # 3) SignalBus arbitration (only if flat)
                if self.pos is None:
                    m0 = self._micro_diag_fallback_locked()
                    m0["recv_ms"] = recv_ms
                    m0["tick_ts_utc"] = tick_ts_fixed.isoformat() if tick_ts_fixed else None
                    m0["tick_ts_raw"] = raw_tick_ts
                    m0["latency_ms"] = latency_ms
                    m0["latency_status"] = latency_status
                    fut0, fut_age0 = self._get_fut_flow_snapshot(ts)
                    if fut0 is not None:
                        m0["fut_flow"] = fut0
                        m0["fut_flow_status"] = "OK"
                    else:
                        if fut_age0 is None:
                            m0["fut_flow_status"] = "MISSING"
                        elif fut_age0 < 0:
                            m0["fut_flow_status"] = "FUTURE_TS"
                        else:
                            m0["fut_flow_status"] = "STALE"
                    prev = None
                    if self.candles:
                        prev = dict(self.candles[-1])
                    elif self.current_candle and int(self.current_candle.get("ticks", 0)) >= 10:
                        prev = dict(self.current_candle)

                    intents: List[EntryIntent] = []
                    if prev is not None:
                        mi = self._micro_intent_locked(ltp, ts, recv_ms=recv_ms, recv_ns=recv_ns, m0=m0, prev=prev)
                        if mi:
                            intents.append(mi)

                        ei = self._ema915_intent_locked(ltp, ts, recv_ms=recv_ms, recv_ns=recv_ns, m0=m0, prev=prev)
                        if ei:
                            intents.append(ei)

                    winner = self._resolve_entry_intents(intents)
                    # If we had a genuine conflict (both sides close), emit HOLD_CONFLICT once per candle.
                    if winner is None and self._intent_conflict_info and self.current_candle is not None:
                        ckey = str(self.current_candle.get("ts"))
                        if self._hold_conflict_key != ckey:
                            self._hold_conflict_key = ckey
                            info = dict(self._intent_conflict_info)
                            self._write_signal(
                                {
                                    "suggestion": "HOLD_CONFLICT",
                                    "reason": "tick_intent_conflict",
                                    **info,
                                },
                                self._mk_tick_metrics(m0, ltp, ts, recv_ms=recv_ms, recv_ns=recv_ns),
                            )
                    if winner is not None:
                        self._commit_entry_intent_locked(winner, m0)

        except Exception:
            self.log.exception("process_tick failed")

    def close(self) -> None:
        """Flush & close file handles."""
        try:
            with self._write_lock:
                self.jsonl_file.flush()
                self.jsonl_file.close()
            with self._arm_write_lock:
                self.arm_jsonl_file.flush()
                self.arm_jsonl_file.close()
        except Exception:
            self.log.exception("close failed")

    def _maybe_roll_session_locked(self, ts: datetime) -> None:
        """Reset session-bound state when IST trading date changes."""
        day_key = ts.astimezone(IST_TZ).strftime("%Y-%m-%d")
        if self._session_day_ist is None:
            self._session_day_ist = day_key
            return
        if day_key == self._session_day_ist:
            return

        self._session_day_ist = day_key
        self.candles.clear()
        self.current_candle = None
        self.session_high = -float("inf")
        self.session_low = float("inf")
        self._range_hist.clear()
        self._travel_hist.clear()
        self._tpt_hist.clear()
        self.bias = None
        self.pending_shock = None
        self._setup_side = None
        self._setup_margin = 0.0
        self._setup_candle_key = None
        self._htf_long_rev_level = None
        self._htf_long_rev_hold = 0
        self._htf_long_rev_key = None
        self._hold_conflict_key = None
        self._hold_wait_candle_key = None
        self._hold_wait_reasons.clear()
        self._retest_ctx = None
        self._last_diag = {}
        self._last_fut_cvd = 0.0
        self._last_fut_flow_snapshot = None
        self._last_fut_flow_ts_utc = None
        self._stopout_cooldown_by_side = {"LONG": None, "SHORT": None}
        self._impulse_reentry_guard_by_side = {"LONG": None, "SHORT": None}
        self._notify_last_sent_by_key.clear()
        self._breakout_state = None
        self._last_pa = None
        self.log.info("session_rollover_ist=%s reset_session_state=1", day_key)

    def _should_emit_hold_wait_locked(self, reason: str, candle_key: str, *, engine: str = "", want_side: str = "") -> bool:
        """Rate limit HOLD_WAIT_CONFIRM to once per (reason,engine,side) per candle."""
        if self._hold_wait_candle_key != candle_key:
            self._hold_wait_candle_key = candle_key
            self._hold_wait_reasons.clear()
        dedupe_key = f"{reason}|{engine}|{want_side}"
        if dedupe_key in self._hold_wait_reasons:
            return False
        self._hold_wait_reasons.add(dedupe_key)
        return True

    def _refresh_fut_sidecar_cache(self, recv_ms: int) -> None:
        """Prefetch sidecar line outside main trading lock to reduce lock-held file I/O."""
        if not bool(getattr(self.cfg, "use_fut_flow", False)):
            return
        poll_ms = int(getattr(self.cfg, "fut_sidecar_poll_ms", 1000) or 1000)
        poll_ms = max(100, poll_ms)

        with self._fut_sidecar_cache_lock:
            if (int(recv_ms) - int(self._fut_sidecar_cache_fetch_ms)) < poll_ms:
                return
            self._fut_sidecar_cache_fetch_ms = int(recv_ms)

        status = "MISSING_FILE"
        fut: Optional[Dict[str, Any]] = None
        fut_path = str(getattr(self.cfg, "fut_sidecar_path", "") or "")
        if fut_path and os.path.exists(fut_path):
            line = _tail_last_line(fut_path)
            if not line:
                status = "EMPTY_FILE"
            else:
                parsed = _parse_fut_candle_row(line)
                if isinstance(parsed, dict) and isinstance(parsed.get("ts"), datetime):
                    fut = parsed
                    status = "OK"
                else:
                    status = "PARSE_ERROR"

        with self._fut_sidecar_cache_lock:
            self._fut_sidecar_cache = dict(fut) if isinstance(fut, dict) else None
            self._fut_sidecar_cache_status = status

    def _get_fut_sidecar_cache(self) -> Tuple[Optional[Dict[str, Any]], str]:
        with self._fut_sidecar_cache_lock:
            fut = dict(self._fut_sidecar_cache) if isinstance(self._fut_sidecar_cache, dict) else None
            status = str(self._fut_sidecar_cache_status or "UNSET")
        return fut, status

    # ------------------------ Candle Persistence / Session SR ------------------------
    def _update_session_extremes_locked(self, candle: Dict[str, Any]) -> None:
        """Update session HOD/LOD from a confirmed 1m candle."""
        try:
            h = float(candle.get("high", 0.0) or 0.0)
            l = float(candle.get("low", 0.0) or 0.0)
            if _is_finite_pos(h):
                self.session_high = max(float(self.session_high), h)
            if _is_finite_pos(l):
                self.session_low = min(float(self.session_low), l)
        except Exception:
            pass

    def _persist_candle_locked(self, candle: Dict[str, Any]) -> None:
        """Append confirmed 1m candle to JSONL (for restart-safe session SR)."""
        if not bool(getattr(self.cfg, "persist_candles", False)):
            return
        path = str(getattr(self.cfg, "candle_db_file", "") or "")
        if not path:
            return
        ts = candle.get("ts")
        if not isinstance(ts, datetime):
            return

        rec = {
            "stream": "candle",
            "tf": "1m",
            "ts_utc": ts.astimezone(timezone.utc).isoformat(),
            "ts_ist": _iso_ist(ts),
            "open": float(candle.get("open", 0.0) or 0.0),
            "high": float(candle.get("high", 0.0) or 0.0),
            "low": float(candle.get("low", 0.0) or 0.0),
            "close": float(candle.get("close", 0.0) or 0.0),
            "ticks": int(candle.get("ticks", 0) or 0),
            "total_travel": float(candle.get("total_travel", 0.0) or 0.0),
            "rv2": float(candle.get("rv2", 0.0) or 0.0),
        }

        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, separators=(",", ":")) + "\n")
        except Exception:
            self.log.exception("persist_candle failed")

    def _load_session_candles_locked(self) -> None:
        """Hydrate today's candles from JSONL into self.candles + session extremes.

        Assumes self._lock is held.
        """
        path = str(getattr(self.cfg, "candle_db_file", "") or "")
        if not path or not os.path.exists(path):
            return

        today = datetime.now(IST_TZ).strftime("%Y-%m-%d")
        loaded = 0

        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except Exception:
                        continue

                    ts_ist = str(rec.get("ts_ist") or "")
                    if not ts_ist.startswith(today):
                        continue

                    ts_utc_s = rec.get("ts_utc")
                    if not ts_utc_s:
                        continue
                    try:
                        ts_utc = datetime.fromisoformat(str(ts_utc_s))
                        if ts_utc.tzinfo is None:
                            ts_utc = ts_utc.replace(tzinfo=timezone.utc)
                        ts_utc = ts_utc.astimezone(timezone.utc)
                    except Exception:
                        continue

                    c = {
                        "ts": ts_utc.replace(second=0, microsecond=0),
                        "open": float(rec.get("open", 0.0) or 0.0),
                        "high": float(rec.get("high", 0.0) or 0.0),
                        "low": float(rec.get("low", 0.0) or 0.0),
                        "close": float(rec.get("close", 0.0) or 0.0),
                        "ticks": int(rec.get("ticks", 0) or 0),
                        "total_travel": float(rec.get("total_travel", 0.0) or 0.0),
                        "rv2": float(rec.get("rv2", 0.0) or 0.0),
                        "last_px": float(rec.get("close", 0.0) or 0.0),
                    }

                    # Basic sanity: ignore broken records
                    if not _is_finite_pos(float(c.get("high", 0.0) or 0.0)):
                        continue
                    if not _is_finite_pos(float(c.get("low", 0.0) or 0.0)):
                        continue

                    self.candles.append(c)
                    self._update_session_extremes_locked(c)
                    loaded += 1

        except Exception:
            self.log.exception("load_session_candles read failed")
            return

        if loaded > 0:
            try:
                self.log.info(
                    "hydrated_session_candles=%s session_high=%.2f session_low=%.2f",
                    int(loaded),
                    float(self.session_high),
                    float(self.session_low),
                )
            except Exception:
                pass

    def _get_sr_levels_locked(self, *, lb: Optional[int] = None) -> Tuple[float, float, float, float]:
        """Return (local_support, local_resistance, session_support, session_resistance).

        Assumes self._lock is held.
        """
        lookback = int(lb if lb is not None else int(getattr(self.cfg, "support_lookback", 45)))
        lookback = max(20, lookback)

        if not self.candles:
            p = float(self._last_px or 0.0)
            return p, p, p, p

        recent = list(self.candles)[-lookback:]
        p = float(self._last_px or float(recent[-1].get("close", 0.0) or 0.0))

        try:
            loc_sup = min(float(c.get("low", p) or p) for c in recent)
            loc_res = max(float(c.get("high", p) or p) for c in recent)
        except Exception:
            loc_sup, loc_res = p, p

        sess_sup = float(self.session_low) if math.isfinite(float(self.session_low)) else loc_sup
        sess_res = float(self.session_high) if math.isfinite(float(self.session_high)) else loc_res
        return float(loc_sup), float(loc_res), float(sess_sup), float(sess_res)

    def _apply_metric_state_locked(self, state: Dict[str, Any]) -> None:
        """Apply metric-state updates. Assumes self._lock is held."""
        if not isinstance(state, dict):
            return
        self.bias = state.get("bias")
        self._range_hist = deque(
            [float(x) for x in state.get("range_hist", [])],
            maxlen=self.cfg.squeeze_lookback,
        )
        self._travel_hist = deque(
            [float(x) for x in state.get("travel_hist", [])],
            maxlen=60,
        )
        self._tpt_hist = deque(
            [float(x) for x in state.get("tpt_hist", [])],
            maxlen=60,
        )

    def _classify_regime(self, metrics: Dict[str, Any]) -> str:
        """Classify tape into TREND / CHOP / BREAKOUT_WINDOW.

        BREAKOUT_WINDOW is driven by breakout energy when available; otherwise it can still arm from
        shock/distance-quality fallback for back-compat.
        """
        try:
            path_eff = float(metrics.get("path_eff", 0.0) or 0.0)
            smoothness = float(metrics.get("smoothness", 0.0) or 0.0)
            range_atr = float(metrics.get("range_atr", 0.0) or 0.0)
            dist_abs = abs(float(metrics.get("anchor_dist_atr_signed", 0.0) or 0.0))
            shock = bool(metrics.get("shock", False))
            bo_energy = float(metrics.get("breakout_energy", 0.0) or 0.0)
            chop_low = (path_eff < float(self.cfg.min_path_eff_filter)) and (smoothness < 0.18) and (range_atr < 0.90)

            if bool(getattr(self.cfg, "breakout_enable", True)) and bo_energy >= float(getattr(self.cfg, "breakout_energy_entry_min", 0.62)):
                if chop_low:
                    return "CHOP_BREAKOUT_WINDOW"
                return "BREAKOUT_WINDOW"
            if shock or (dist_abs >= 0.90 and path_eff >= float(self.cfg.entry_path_eff) * 0.85):
                return "BREAKOUT_WINDOW"
            if chop_low:
                return "CHOP"
            return "TREND"
        except Exception:
            return "UNKNOWN"

    @staticmethod
    def _clamp01(x: float) -> float:
        try:
            xx = float(x)
        except Exception:
            return 0.0
        if not math.isfinite(xx):
            return 0.0
        return 0.0 if xx < 0.0 else (1.0 if xx > 1.0 else xx)

    def _candle_features(self, c: Dict[str, Any]) -> Dict[str, float]:
        o = float(c.get("open", 0.0) or 0.0)
        h = float(c.get("high", o) or o)
        l = float(c.get("low", o) or o)
        cl = float(c.get("close", o) or o)
        r = max(self._TICK_SIZE, h - l)
        body_pct = abs(cl - o) / max(_EPS, r)
        upper_wick_pct = max(0.0, h - max(o, cl)) / max(_EPS, r)
        lower_wick_pct = max(0.0, min(o, cl) - l) / max(_EPS, r)
        clv = ((cl - l) - (h - cl)) / max(_EPS, r)
        return {
            "open": o,
            "high": h,
            "low": l,
            "close": cl,
            "range": r,
            "body_pct": body_pct,
            "upper_wick_pct": upper_wick_pct,
            "lower_wick_pct": lower_wick_pct,
            "clv": clv,
        }

    def _price_action_signal_locked(self, m: Dict[str, Any]) -> Dict[str, Any]:
        """3-candle PA detector for dominance / rejection / sweep+reclaim u-turn."""
        if not bool(getattr(self.cfg, "pa_enable", True)) or len(self.candles) < 3:
            return {"pa_side": None, "pa_mode": None, "pa_strength": 0.0, "pa_tags": [], "pa_level": None}

        atr = max(self._TICK_SIZE, float(m.get("atr", self._MICRO_ATR_FALLBACK) or self._MICRO_ATR_FALLBACK))
        c1 = self._candle_features(self.candles[-3])
        c2 = self._candle_features(self.candles[-2])
        c3 = self._candle_features(self.candles[-1])

        tags: List[str] = []
        side: Optional[str] = None
        mode: Optional[str] = None
        level: Optional[float] = None
        strength = 0.0

        dom_body = float(getattr(self.cfg, "pa_dom_body_pct", 0.55))
        dom_clv = float(getattr(self.cfg, "pa_dom_clv", 0.65))
        dom_range_atr = float(getattr(self.cfg, "pa_dom_range_atr", 0.75))
        range_atr = c3["range"] / max(_EPS, atr)
        if c3["body_pct"] >= dom_body and abs(c3["clv"]) >= dom_clv and range_atr >= dom_range_atr:
            mode = "dominance"
            side = "LONG" if c3["clv"] > 0 else "SHORT"
            level = float(c2["high"] if side == "LONG" else c2["low"])
            strength = self._clamp01(
                0.40 * (c3["body_pct"] / max(_EPS, dom_body))
                + 0.35 * (abs(c3["clv"]) / max(_EPS, dom_clv))
                + 0.25 * (range_atr / max(_EPS, dom_range_atr))
            )
            tags.append("dominance")

        rej_w = float(getattr(self.cfg, "pa_rej_wick_pct", 0.45))
        rej_clv_max = float(getattr(self.cfg, "pa_rej_clv_max", 0.25))
        if mode is None:
            if c3["lower_wick_pct"] >= rej_w and c3["clv"] >= -rej_clv_max:
                mode = "rejection"
                side = "LONG"
                level = float(c3["low"])
                strength = self._clamp01(
                    0.55 * (c3["lower_wick_pct"] / max(_EPS, rej_w))
                    + 0.45 * self._clamp01((c3["clv"] + 1.0) / 2.0)
                )
                tags.append("rejection")
            elif c3["upper_wick_pct"] >= rej_w and c3["clv"] <= rej_clv_max:
                mode = "rejection"
                side = "SHORT"
                level = float(c3["high"])
                strength = self._clamp01(
                    0.55 * (c3["upper_wick_pct"] / max(_EPS, rej_w))
                    + 0.45 * self._clamp01((-c3["clv"] + 1.0) / 2.0)
                )
                tags.append("rejection")

        if mode is None:
            sweep_atr = float(getattr(self.cfg, "pa_uturn_sweep_atr", 0.18))
            reclaim_clv = float(getattr(self.cfg, "pa_uturn_reclaim_clv", 0.35))
            prior_low = min(c1["low"], c2["low"])
            prior_high = max(c1["high"], c2["high"])

            sweep_down = c2["low"] < (prior_low - sweep_atr * atr)
            reclaim_up = (c3["close"] > prior_low) and (c3["clv"] >= reclaim_clv)
            if sweep_down and reclaim_up:
                mode = "uturn"
                side = "LONG"
                level = float(c2["low"])
                sweep_depth = (prior_low - c2["low"]) / max(_EPS, atr)
                strength = self._clamp01(
                    0.50 * self._clamp01(sweep_depth / max(_EPS, sweep_atr))
                    + 0.50 * self._clamp01(c3["clv"] / max(_EPS, reclaim_clv))
                )
                tags.extend(["uturn", "sweep_reclaim"])

            sweep_up = c2["high"] > (prior_high + sweep_atr * atr)
            reclaim_dn = (c3["close"] < prior_high) and (c3["clv"] <= -reclaim_clv)
            if mode is None and sweep_up and reclaim_dn:
                mode = "uturn"
                side = "SHORT"
                level = float(c2["high"])
                sweep_depth = (c2["high"] - prior_high) / max(_EPS, atr)
                strength = self._clamp01(
                    0.50 * self._clamp01(sweep_depth / max(_EPS, sweep_atr))
                    + 0.50 * self._clamp01((-c3["clv"]) / max(_EPS, reclaim_clv))
                )
                tags.extend(["uturn", "sweep_reclaim"])

        if strength > 0.0 and strength >= float(getattr(self.cfg, "pa_min_strength", 0.62)):
            tags.append("pa_ok")
        return {"pa_side": side, "pa_mode": mode, "pa_strength": float(strength), "pa_tags": tags, "pa_level": level}

    def _update_breakout_state_locked(self, m: Dict[str, Any]) -> Dict[str, Any]:
        if not bool(getattr(self.cfg, "breakout_enable", True)) or not isinstance(m, dict):
            self._breakout_state = None
            return {"breakout_side": None, "breakout_level": None, "breakout_energy": 0.0, "breakout_origin": None}

        ckey = str(m.get("_candle_ts_utc") or m.get("_candle_ts") or m.get("ts") or "")
        if self._breakout_state is not None and self._breakout_state.last_candle_key == ckey:
            bs = self._breakout_state
            return {"breakout_side": bs.side, "breakout_level": bs.level, "breakout_energy": bs.energy, "breakout_origin": bs.origin}

        atr = max(self._TICK_SIZE, float(m.get("atr", self._MICRO_ATR_FALLBACK) or self._MICRO_ATR_FALLBACK))
        px = float(m.get("px", 0.0) or 0.0)
        bo_side: Optional[str] = None
        bo_level: Optional[float] = None
        origin: Optional[str] = None

        if len(self.candles) >= 2:
            prev = self.candles[-2]
            prev_high = float(prev.get("high", px) or px)
            prev_low = float(prev.get("low", px) or px)
            if px > (prev_high + self._TICK_SIZE):
                bo_side, bo_level, origin = "LONG", prev_high, "bos"
            elif px < (prev_low - self._TICK_SIZE):
                bo_side, bo_level, origin = "SHORT", prev_low, "bos"

        if bool(m.get("shock", False)):
            ss = str(m.get("shock_side", "") or "")
            if ss in ("LONG", "SHORT"):
                bo_side, bo_level, origin = ss, float(m.get("shock_mid", px) or px), "shock"

        pa_mode = str(m.get("pa_mode") or "")
        pa_side = str(m.get("pa_side") or "")
        pa_strength = float(m.get("pa_strength", 0.0) or 0.0)
        if origin is None and pa_mode == "dominance" and pa_side in ("LONG", "SHORT") and pa_strength >= float(getattr(self.cfg, "pa_min_strength", 0.62)):
            bo_side, bo_level, origin = pa_side, float(m.get("pa_level", px) or px), "pa_dom"

        path_eff = self._clamp01(float(m.get("path_eff", 0.0) or 0.0))
        smoothness = self._clamp01(float(m.get("smoothness", 0.0) or 0.0))
        act_w = self._clamp01(float(m.get("activity_w", 0.0) or 0.0))
        clv = float(m.get("clv", 0.0) or 0.0)
        dir_sign = 1.0 if bo_side == "LONG" else (-1.0 if bo_side == "SHORT" else 0.0)
        clv_dir = self._clamp01(abs(clv * dir_sign))
        range_atr = self._clamp01(float(m.get("range_atr", 0.0) or 0.0) / 1.6)
        init_energy = self._clamp01(0.35 * path_eff + 0.25 * smoothness + 0.20 * act_w + 0.20 * clv_dir + 0.10 * range_atr)

        if bo_side in ("LONG", "SHORT") and bo_level is not None:
            self._breakout_state = BreakoutState(
                side=bo_side,
                level=float(bo_level),
                origin=str(origin or "bos"),
                energy=float(max(init_energy, 0.30)),
                best_px=float(px),
                last_candle_key=ckey,
            )
        elif self._breakout_state is None:
            return {"breakout_side": None, "breakout_level": None, "breakout_energy": 0.0, "breakout_origin": None}
        else:
            self._breakout_state.last_candle_key = ckey

        bs = self._breakout_state
        if bs is None:
            return {"breakout_side": None, "breakout_level": None, "breakout_energy": 0.0, "breakout_origin": None}

        bs.best_px = max(float(bs.best_px), px) if bs.side == "LONG" else min(float(bs.best_px), px)
        kill_buf = float(getattr(self.cfg, "breakout_kill_reclaim_atr", 0.35)) * atr
        if bs.side == "LONG" and px < (bs.level - kill_buf):
            self._breakout_state = None
            return {"breakout_side": None, "breakout_level": None, "breakout_energy": 0.0, "breakout_origin": None}
        if bs.side == "SHORT" and px > (bs.level + kill_buf):
            self._breakout_state = None
            return {"breakout_side": None, "breakout_level": None, "breakout_energy": 0.0, "breakout_origin": None}

        base = float(getattr(self.cfg, "breakout_energy_decay_base", 0.74))
        w_eff = float(getattr(self.cfg, "breakout_energy_decay_eff_w", 0.22))
        w_sm = float(getattr(self.cfg, "breakout_energy_decay_smooth_w", 0.10))
        keep = max(0.35, min(0.98, base + w_eff * path_eff + w_sm * smoothness))
        runaway_atr = float(getattr(self.cfg, "breakout_runaway_atr", 1.25))
        dist_atr = abs(px - float(bs.level)) / max(_EPS, atr)
        if dist_atr >= runaway_atr and path_eff < float(self.cfg.entry_path_eff):
            keep *= 0.80
        bs.energy = float(self._clamp01(float(bs.energy) * keep + 0.15 * init_energy))
        if bs.energy < float(getattr(self.cfg, "breakout_energy_keep_min", 0.42)):
            self._breakout_state = None
            return {"breakout_side": None, "breakout_level": None, "breakout_energy": 0.0, "breakout_origin": None}
        return {"breakout_side": bs.side, "breakout_level": bs.level, "breakout_energy": bs.energy, "breakout_origin": bs.origin}

    def _select_tp_mult(self, m: Dict[str, Any], side: str, reason: str) -> float:
        regime = str(m.get("regime", "UNKNOWN") or "UNKNOWN")
        conf = self._clamp01(float(m.get("confidence", 0.0) or 0.0))
        bo_e = self._clamp01(float(m.get("breakout_energy", 0.0) or 0.0))
        pa_s = self._clamp01(float(m.get("pa_strength", 0.0) or 0.0))
        if regime == "CHOP":
            lo, hi = float(self.cfg.tp_mult_chop_min), float(self.cfg.tp_mult_chop_max)
        elif regime in ("BREAKOUT_WINDOW", "CHOP_BREAKOUT_WINDOW"):
            lo, hi = float(self.cfg.tp_mult_break_min), float(self.cfg.tp_mult_break_max)
        else:
            lo, hi = float(self.cfg.tp_mult_trend_min), float(self.cfg.tp_mult_trend_max)
        score = self._clamp01(0.40 * conf + 0.45 * bo_e + 0.15 * pa_s)
        try:
            dist_abs = float(m.get("anchor_dist_atr_abs", 0.0) or 0.0)
            if dist_abs > float(getattr(self.cfg, "late_breakout_anchor_atr", 0.90)):
                score *= 0.85
        except Exception:
            pass
        return float(max(0.10, lo + (hi - lo) * score))

    def _select_runner(self, m: Dict[str, Any]) -> bool:
        regime = str(m.get("regime", "UNKNOWN") or "UNKNOWN")
        conf = self._clamp01(float(m.get("confidence", 0.0) or 0.0))
        bo_e = self._clamp01(float(m.get("breakout_energy", 0.0) or 0.0))
        pa_s = self._clamp01(float(m.get("pa_strength", 0.0) or 0.0))
        if regime in ("BREAKOUT_WINDOW", "CHOP_BREAKOUT_WINDOW"):
            return conf >= float(self.cfg.runner_conf_break) and bo_e >= float(self.cfg.runner_energy_min) and pa_s >= float(self.cfg.runner_pa_min)
        if regime == "TREND":
            return conf >= float(self.cfg.runner_conf_trend) and bo_e >= float(self.cfg.runner_energy_min) and pa_s >= float(self.cfg.runner_pa_min)
        return False

    @staticmethod
    def _quality_and_confidence(metrics: Dict[str, Any]) -> Tuple[str, float]:
        """Produce a compact setup quality grade and normalized confidence [0..1]."""
        try:
            path_eff = max(0.0, min(1.0, float(metrics.get("path_eff", 0.0) or 0.0)))
            activity_w = max(0.0, min(1.0, float(metrics.get("activity_w", 0.0) or 0.0)))
            clv_abs = max(0.0, min(1.0, abs(float(metrics.get("clv", 0.0) or 0.0))))
            smoothness = max(0.0, min(1.0, float(metrics.get("smoothness", 0.0) or 0.0)))
        except Exception:
            return "C", 0.0

        confidence = float(max(0.0, min(1.0, (0.40 * path_eff) + (0.25 * activity_w) + (0.20 * clv_abs) + (0.15 * smoothness))))
        if confidence >= 0.72:
            quality = "A"
        elif confidence >= 0.52:
            quality = "B"
        else:
            quality = "C"
        return quality, confidence

    def _compute_metrics_from_snapshot(
        self,
        candles_snapshot: List[Dict[str, Any]],
        *,
        prev_bias: Optional[str],
        range_hist_snapshot: List[float],
        travel_hist_snapshot: List[float],
        tpt_hist_snapshot: List[float],
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Pure metric compute from snapshots (no shared-state mutation)."""
        state_out: Dict[str, Any] = {
            "bias": prev_bias,
            "range_hist": list(range_hist_snapshot),
            "travel_hist": list(travel_hist_snapshot),
            "tpt_hist": list(tpt_hist_snapshot),
        }
        if len(candles_snapshot) < self.cfg.min_ready_candles:
            return {}, state_out

        df = pd.DataFrame(list(candles_snapshot))
        for col in ("open", "high", "low", "close", "ticks", "total_travel", "rv2", "ts"):
            if col not in df.columns:
                return {}, state_out

        close = pd.to_numeric(df["close"], errors="coerce")
        high = pd.to_numeric(df["high"], errors="coerce")
        low = pd.to_numeric(df["low"], errors="coerce")
        open_ = pd.to_numeric(df["open"], errors="coerce")
        if close.isna().any() or high.isna().any() or low.isna().any() or open_.isna().any():
            return {}, state_out

        hma_series = _hma_series(close, self.cfg.hma_period)
        if hma_series.isna().iloc[-1]:
            return {}, state_out
        hma = float(hma_series.iloc[-1])

        prev_close = close.shift(1).fillna(close)
        tr = np.maximum(
            (high - low).to_numpy(),
            np.maximum(
                np.abs((high - prev_close).to_numpy()),
                np.abs((low - prev_close).to_numpy()),
            ),
        )
        tr_s = pd.Series(tr, index=df.index)
        atr = float(tr_s.rolling(self.cfg.atr_period, min_periods=1).mean().iloc[-1])
        if not _is_finite_pos(atr):
            return {}, state_out

        curr_px = float(close.iloc[-1])
        dist_signed = (curr_px - hma) / atr
        dist_abs = abs(dist_signed)
        upper_band = hma + self.cfg.stretch_limit * atr
        lower_band = hma - self.cfg.stretch_limit * atr

        rng = max(self._TICK_SIZE, float(high.iloc[-1] - low.iloc[-1]))
        clv = float((2.0 * curr_px - float(high.iloc[-1]) - float(low.iloc[-1])) / rng)
        body_pct = float(abs(curr_px - float(open_.iloc[-1])) / rng)

        upper_wick = float(high.iloc[-1] - max(open_.iloc[-1], close.iloc[-1]))
        lower_wick = float(min(open_.iloc[-1], close.iloc[-1]) - low.iloc[-1])
        upper_wick_pct = float(upper_wick / rng)
        lower_wick_pct = float(lower_wick / rng)

        last = candles_snapshot[-1]
        total_travel = float(last.get("total_travel", 0.0))
        ticks = int(last.get("ticks", 0))
        path_eff = float(abs(curr_px - float(last.get("open", curr_px))) / max(_EPS, total_travel))
        travel_per_tick = float(total_travel / max(1, ticks))
        rv2 = float(last.get("rv2", 0.0))
        smoothness = float(abs(curr_px - float(last.get("open", curr_px))) / max(_EPS, math.sqrt(max(_EPS, rv2))))

        travel_hist = deque([float(x) for x in travel_hist_snapshot], maxlen=60)
        travel_hist.append(total_travel)
        baseline_travel = float(np.mean(travel_hist)) if len(travel_hist) >= 10 else max(_EPS, total_travel)
        pav_mult = float(total_travel / max(_EPS, baseline_travel))

        aw_lo = float(self.cfg.activity_w_low)
        aw_hi = float(self.cfg.activity_w_high)
        if pav_mult <= aw_lo:
            activity_w = 0.0
        elif pav_mult >= aw_hi:
            activity_w = 1.0
        else:
            activity_w = float((pav_mult - aw_lo) / max(_EPS, (aw_hi - aw_lo)))

        tpt_hist = deque([float(x) for x in tpt_hist_snapshot], maxlen=60)
        avg_tpt = float(np.mean(tpt_hist)) if len(tpt_hist) >= 1 else travel_per_tick
        tpt_hist.append(travel_per_tick)

        curr_range = float(high.iloc[-1] - low.iloc[-1])
        range_atr = float(curr_range / max(_EPS, atr))
        range_hist = deque([float(x) for x in range_hist_snapshot], maxlen=self.cfg.squeeze_lookback)
        range_hist.append(curr_range)
        is_squeeze = False
        if len(range_hist) >= self.cfg.squeeze_lookback:
            perc = float(np.percentile(np.array(range_hist, dtype=float), self.cfg.squeeze_pct))
            is_squeeze = curr_range <= max(_EPS, perc)

        slope = 0.0
        if len(hma_series) >= 6:
            slope = float((hma_series.iloc[-1] - hma_series.iloc[-6]) / (5.0 * atr))

        def _norm_slope(n_bars: int) -> float:
            if len(hma_series) < (n_bars + 1):
                return 0.0
            try:
                delta = float(hma_series.iloc[-1] - hma_series.iloc[-(n_bars + 1)])
                return float(delta / max(_EPS, float(n_bars) * atr))
            except Exception:
                return 0.0

        htf_slope_15 = _norm_slope(15)
        htf_slope_25 = _norm_slope(25)

        local_res = float(high.iloc[-1])
        try:
            lb_local = int(getattr(self.cfg, "htf_local_lookback", 15) or 15)
        except Exception:
            lb_local = 15
        if lb_local > 0 and len(high) >= 2:
            hist_highs = high.iloc[:-1]
            if not hist_highs.empty:
                local_res = float(hist_highs.tail(lb_local).max())

        new_bias = _update_bias(prev_bias, slope, pos_th=0.05, neg_th=-0.05)

        prev_c = float(close.iloc[-2])
        shock_move = float(curr_px - prev_c)
        shock_thresh = float(max(self.cfg.shock_points, self.cfg.shock_atr_mult * atr))
        is_shock = abs(shock_move) >= shock_thresh
        shock_size = abs(shock_move)
        shock_side = "LONG" if shock_move > 0 else "SHORT"
        shock_mid = float((prev_c + curr_px) / 2.0)

        candle_ts = candles_snapshot[-1].get("ts") if candles_snapshot else None
        candle_ts_utc = candle_ts.astimezone(timezone.utc).isoformat() if isinstance(candle_ts, datetime) else None
        candle_ts_ist = _iso_ist(candle_ts) if isinstance(candle_ts, datetime) else _now_iso_ist()
        metrics = {
            "px": curr_px,
            "ts_utc": candle_ts_utc,
            "ts_ist": candle_ts_ist,
            "ts": candle_ts,
            "hma": hma,
            "atr": atr,
            "upper_band": upper_band,
            "lower_band": lower_band,
            "anchor_dist_atr_signed": dist_signed,
            "anchor_dist_atr_abs": dist_abs,
            "clv": clv,
            "body_pct": body_pct,
            "upper_wick_pct": upper_wick_pct,
            "lower_wick_pct": lower_wick_pct,
            "path_eff": path_eff,
            "travel_per_tick": travel_per_tick,
            "avg_travel_per_tick": avg_tpt,
            "smoothness": smoothness,
            "range_atr": range_atr,
            "pav_mult": pav_mult,
            "activity_w": activity_w,
            "squeeze": is_squeeze,
            "bias": new_bias or "NONE",
            "htf_slope_15": htf_slope_15,
            "htf_slope_25": htf_slope_25,
            "local_res": local_res,
            "shock": is_shock,
            "shock_side": shock_side,
            "shock_thresh": shock_thresh,
            "shock_size": shock_size,
            "tick_overrun": bool(last.get("_tick_overrun", False)),
            "shock_mid": shock_mid,
            "slope": slope,
        }
        regime = self._classify_regime(metrics)
        quality, confidence = self._quality_and_confidence(metrics)
        metrics["regime"] = regime
        metrics["setup_quality"] = quality
        metrics["confidence"] = confidence
        state_out = {
            "bias": new_bias,
            "range_hist": list(range_hist),
            "travel_hist": list(travel_hist),
            "tpt_hist": list(tpt_hist),
        }
        return metrics, state_out

    # ------------------------ Candle Close ------------------------
    def _close_candle_locked(self) -> None:
        """Assumes self._lock is held under process-thread serialization."""
        if not self._process_lock.locked():
            raise RuntimeError("_close_candle_locked requires _process_lock")
        if not self._lock.locked():
            raise RuntimeError("_close_candle_locked requires _lock")
        if self.current_candle is None:
            return

        candle = self.current_candle
        self.candles.append(candle)
        self.current_candle = None
        self._update_session_extremes_locked(candle)
        candles_snapshot = list(self.candles)
        prev_bias = self.bias
        range_hist_snapshot = list(self._range_hist)
        travel_hist_snapshot = list(self._travel_hist)
        tpt_hist_snapshot = list(self._tpt_hist)

        metric_state = {
            "bias": prev_bias,
            "range_hist": range_hist_snapshot,
            "travel_hist": travel_hist_snapshot,
            "tpt_hist": tpt_hist_snapshot,
        }
        metrics: Dict[str, Any] = {}
        lock_released = False
        try:
            self._lock.release()
            lock_released = True
            self._persist_candle_locked(candle)
            metrics, metric_state = self._compute_metrics_from_snapshot(
                candles_snapshot,
                prev_bias=prev_bias,
                range_hist_snapshot=range_hist_snapshot,
                travel_hist_snapshot=travel_hist_snapshot,
                tpt_hist_snapshot=tpt_hist_snapshot,
            )
        except Exception:
            self.log.exception("compute_metrics_from_snapshot failed")
            metrics = {}
        finally:
            if lock_released:
                self._lock.acquire()

        self._apply_metric_state_locked(metric_state)
        # Optional futures flow snapshot (sidecar). Read once per candle close.
        if bool(getattr(self.cfg, "use_fut_flow", False)):
            fut, cached_status = self._get_fut_sidecar_cache()
            fut_flow_status = str(cached_status or "UNSET")
            if isinstance(fut, dict) and isinstance(fut.get("ts"), datetime):
                fut_ts_utc = fut["ts"].astimezone(timezone.utc)
                age = (_now_utc() - fut_ts_utc).total_seconds()
                if 0.0 <= age <= float(getattr(self.cfg, "fut_flow_stale_sec", 180)):
                    cvd_prev = float(getattr(self, "_last_fut_cvd", 0.0) or 0.0)
                    try:
                        cvd_now = float(fut.get("cvd", 0.0) or 0.0)
                    except Exception:
                        cvd_now = cvd_prev
                    if not math.isfinite(cvd_now):
                        cvd_now = cvd_prev
                    cvd_delta = cvd_now - cvd_prev
                    fut["cvd_delta"] = cvd_delta
                    metrics["fut_flow"] = fut
                    metrics["fut_cvd_delta"] = cvd_delta
                    self._last_fut_cvd = cvd_now

                    # carry into tick-level gating (stale-checked again per tick)
                    self._last_fut_flow_snapshot = dict(fut)
                    self._last_fut_flow_ts_utc = fut_ts_utc
                    fut_flow_status = "OK"
                elif age < 0.0:
                    metrics["fut_flow"] = None
                    fut_flow_status = "FUTURE_TS"
                else:
                    metrics["fut_flow"] = None
                    fut_flow_status = "STALE"
            else:
                metrics["fut_flow"] = None

            metrics["fut_flow_status"] = fut_flow_status
            if fut_flow_status != "OK":
                warn_min = _now_utc().strftime("%Y%m%d%H%M")
                if self._fut_flow_warn_min != warn_min:
                    self._fut_flow_warn_min = warn_min
                    fut_path = str(getattr(self.cfg, "fut_sidecar_path", "") or "")
                    self.log.warning("fut_flow_status=%s path=%s", fut_flow_status, fut_path)

        metrics["_candle_ts"] = candle.get("ts")
        cts = candle.get("ts")
        metrics["_candle_ts_utc"] = cts.astimezone(timezone.utc).isoformat() if isinstance(cts, datetime) else None
        metrics["_candle_ts_ist"] = _iso_ist(cts) if isinstance(cts, datetime) else None
        metrics["candle_bucket_id"] = metrics.get("_candle_ts_utc")

        # Price action + breakout energy state at candle close.
        try:
            pa = self._price_action_signal_locked(metrics)
            if isinstance(pa, dict):
                metrics.update(pa)
                self._last_pa = dict(pa)
        except Exception:
            self.log.exception("price_action_signal_failed")
        try:
            br = self._update_breakout_state_locked(metrics)
            if isinstance(br, dict):
                metrics.update(br)
                regime2 = self._classify_regime(metrics)
                metrics["regime"] = regime2
        except Exception:
            self.log.exception("update_breakout_state_failed")

        # Update HTF reversal-proof state (break + hold above local shelf)
        self._update_htf_long_reversal_state_locked(metrics)
        metrics["htf_rev_level"] = self._htf_long_rev_level
        metrics["htf_rev_hold"] = self._htf_long_rev_hold

        # Start-of-minute directional readiness (market-driven; prevents flip-flops)
        ready = self._ready_decision_locked(metrics, self._ema915_ready_state(metrics))
        if isinstance(ready, dict):
            self._write_signal(ready, metrics)

        decision = self._evaluate_strategy(metrics)

        if self.pos is None and metrics and _is_entry_suggestion(decision.get("suggestion")):
            side = "LONG" if "LONG" in str(decision.get("suggestion")) else "SHORT"
            ts = metrics.get("ts")
            if not isinstance(ts, datetime):
                ts = _now_utc()
            px = float(metrics.get("px", 0.0) or 0.0)
            extra = {k: v for k, v in decision.items() if k not in ("suggestion", "reason")}
            intent = EntryIntent(
                engine="candle",
                side=side,
                entry_px=px,
                ts=ts,
                reason=str(decision.get("reason", "")),
                score=1000.0,
                sl_hint=None,
                extra=extra,
            )
            self._commit_entry_intent_locked(intent, metrics)

        self._write_signal(decision, metrics)

        # Deterministic minute-close summary for chart-by-chart audit.
        summary = {
            "suggestion": "MARKET_UPDATE",
            "state": "CLOSE",
            "reason": "candle_close_summary",
            "stream": "signal",
            "channel": "candle",
            "engine": "candle",
            "ts": metrics.get("ts"),
            "ts_utc": metrics.get("ts_utc"),
            "ts_ist": metrics.get("ts_ist"),
            "candle_bucket_id": metrics.get("candle_bucket_id"),
            "open": float(candle.get("open", 0.0) or 0.0),
            "high": float(candle.get("high", 0.0) or 0.0),
            "low": float(candle.get("low", 0.0) or 0.0),
            "close": float(candle.get("close", 0.0) or 0.0),
            "regime": metrics.get("regime"),
            "pa_side": metrics.get("pa_side"),
            "pa_mode": metrics.get("pa_mode"),
            "pa_strength": metrics.get("pa_strength"),
            "pa_tags": metrics.get("pa_tags"),
            "breakout_side": metrics.get("breakout_side"),
            "breakout_level": metrics.get("breakout_level"),
            "breakout_energy": metrics.get("breakout_energy"),
            "breakout_origin": metrics.get("breakout_origin"),
            "final_decision": decision.get("suggestion"),
            "final_reason": decision.get("reason"),
            "final_state": _signal_state(str(decision.get("suggestion") or "")),
            "tp_model": {
                "tp_atr_mult_default": float(self.cfg.tp_atr_mult),
                "tp_mult_chop": [float(self.cfg.tp_mult_chop_min), float(self.cfg.tp_mult_chop_max)],
                "tp_mult_trend": [float(self.cfg.tp_mult_trend_min), float(self.cfg.tp_mult_trend_max)],
                "tp_mult_breakout": [float(self.cfg.tp_mult_break_min), float(self.cfg.tp_mult_break_max)],
            },
            "sl_model": {
                "min_init_sl_points": float(self.cfg.min_init_sl_points),
                "min_init_sl_atr": float(self.cfg.min_init_sl_atr),
                "hard_stop_atr_mult": float(self.cfg.hard_stop_atr_mult),
                "hard_stop_points": float(self.cfg.hard_stop_points),
            },
            "veto_or_override": decision.get("reason"),
        }
        self._write_signal(summary, metrics)

    def _compute_metrics_locked(self) -> Dict[str, Any]:
        """Compute indicators from current state. Assumes self._lock is held."""
        metrics, state = self._compute_metrics_from_snapshot(
            list(self.candles),
            prev_bias=self.bias,
            range_hist_snapshot=list(self._range_hist),
            travel_hist_snapshot=list(self._travel_hist),
            tpt_hist_snapshot=list(self._tpt_hist),
        )
        self._apply_metric_state_locked(state)
        return metrics

    
    def _update_htf_long_reversal_state_locked(self, metrics: Dict[str, Any]) -> None:
        """Track 'break + hold' above local_res for LONGs during bearish HTF.

        This is a candle-close state machine:
        - On first close above local_res (+ buffer), we ARM a shelf level (local_res at break).
        - We then require N consecutive CLOSED candle closes above that shelf (+ buffer) to CONFIRM.
        - If a close falls back below the shelf, we reset.
        """
        if not isinstance(metrics, dict) or not metrics:
            return

        ckey = str(metrics.get("_candle_ts_utc") or metrics.get("_candle_ts") or metrics.get("ts") or "")
        if not ckey or ckey == self._htf_long_rev_key:
            return
        self._htf_long_rev_key = ckey

        try:
            close_px = float(metrics.get("px", 0.0) or 0.0)
        except Exception:
            close_px = 0.0
        if close_px <= 0:
            return

        try:
            local_res = float(metrics.get("local_res", close_px) or close_px)
        except Exception:
            local_res = close_px

        try:
            need = int(getattr(self.cfg, "htf_reversal_hold_bars", 2) or 2)
        except Exception:
            need = 2
        need = max(1, need)

        try:
            buf_ticks = int(getattr(self.cfg, "htf_reversal_buffer_ticks", 1) or 1)
        except Exception:
            buf_ticks = 1
        buf_ticks = max(0, buf_ticks)
        buf = float(buf_ticks) * float(self._TICK_SIZE)

        # If we don't have an active shelf, ARM on a clean close above the current local_res.
        if self._htf_long_rev_level is None:
            if close_px > (local_res + buf):
                self._htf_long_rev_level = local_res
                self._htf_long_rev_hold = 1
            else:
                self._htf_long_rev_hold = 0
            return

        lvl = float(self._htf_long_rev_level)

        # If price makes a NEW structural break above a higher shelf, re-arm to that shelf.
        if close_px > (local_res + buf) and local_res > (lvl + buf):
            self._htf_long_rev_level = local_res
            self._htf_long_rev_hold = 1
            return

        # Hold counting on the ARMED shelf (fixed target, not moving goalposts)
        if close_px > (lvl + buf):
            self._htf_long_rev_hold = min(need, int(self._htf_long_rev_hold) + 1)
        else:
            # Lost the hold -> reset completely
            self._htf_long_rev_level = None
            self._htf_long_rev_hold = 0


# ------------------------ Market-driven setup (READY / CONFLICT) ------------------------
    def _dir_margin_from_metrics(self, m: Dict[str, Any], *, ema915_bias: float = 0.0) -> Dict[str, float]:
        """
        Produce a normalized directional margin in [-1..1] from candle metrics.
        Positive => LONG favored, Negative => SHORT favored.
        Also returns long/short scores in [0..1] for logging.
        """
        clv = float(m.get("clv", 0.0) or 0.0)  # [-1..1]
        dist = float(m.get("anchor_dist_atr_signed", 0.0) or 0.0)  # unbounded-ish
        dist_n = max(-2.0, min(2.0, dist)) / 2.0
        path_eff = float(m.get("path_eff", 0.0) or 0.0)  # [0..1]
        activity_w = float(m.get("activity_w", 0.0) or 0.0)  # [0..1]
        bias = str(m.get("bias", "NONE") or "NONE")
        shock = bool(m.get("shock", False))
        shock_side = str(m.get("shock_side", "") or "")

        s_bias = 1.0 if bias == "LONG" else (-1.0 if bias == "SHORT" else 0.0)
        s_shock = 1.0 if (shock and shock_side == "LONG") else (-1.0 if (shock and shock_side == "SHORT") else 0.0)

        raw = (0.45 * clv) + (0.35 * dist_n) + (0.15 * s_bias) + (0.05 * s_shock) + (0.10 * ema915_bias)  # ~[-1..1]
        try:
            pe_gate = (path_eff - float(self.cfg.min_path_eff_filter)) / max(
                _EPS, (1.0 - float(self.cfg.min_path_eff_filter))
            )
        except Exception:
            pe_gate = path_eff
        pe_gate = max(0.0, min(1.0, pe_gate))
        aw_gate = max(0.0, min(1.0, activity_w))
        strength = (0.5 * pe_gate) + (0.5 * aw_gate)  # [0..1]

        margin = raw * (0.25 + 0.75 * strength)  # damp in weak tape
        margin = max(-1.0, min(1.0, margin))

        long_score = max(0.0, min(1.0, 0.5 + 0.5 * margin))
        short_score = max(0.0, min(1.0, 0.5 - 0.5 * margin))
        return {"margin": margin, "long_score": long_score, "short_score": short_score}

    def _ema915_ready_state(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        out = {"ema915_breach": False, "ema915_side": None, "ema915_age_ms": None, "ema915_bias": 0.0}
        armed = self._ema915_armed
        if not armed or not metrics:
            return out

        side = str(armed.get("side") or "")
        if side not in ("LONG", "SHORT"):
            return out

        lvl = float(armed.get("lvl", 0.0) or 0.0)
        px = float(metrics.get("px", 0.0) or 0.0)
        if side == "LONG":
            breach = px >= (lvl + self._TICK_SIZE)
            bias = 1.0 if breach else 0.0
        else:
            breach = px <= (lvl - self._TICK_SIZE)
            bias = -1.0 if breach else 0.0

        age_ms = None
        if self._last_recv_ms is not None:
            try:
                age_ms = max(0, int(self._last_recv_ms) - int(armed.get("armed_ms", self._last_recv_ms)))
            except Exception:
                age_ms = None

        out.update({"ema915_breach": bool(breach), "ema915_side": side, "ema915_age_ms": age_ms, "ema915_bias": bias})
        return out

    def _ready_decision_locked(
        self, metrics: Dict[str, Any], ema915_state: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Emit a single start-of-minute directional "readiness" decision driven by market evidence.
        Updates internal setup state for hysteresis (prevents flip-flops).
        """
        if not metrics:
            return None
        if self.pos is not None:
            return None

        ema_state = ema915_state if isinstance(ema915_state, dict) else self._ema915_ready_state(metrics)
        ema_bias = float(ema_state.get("ema915_bias", 0.0) or 0.0)
        out = self._dir_margin_from_metrics(metrics, ema915_bias=ema_bias)
        margin = float(out["margin"])
        long_score = float(out["long_score"])
        short_score = float(out["short_score"])
        ema_info = {
            "ema915_breach": bool(ema_state.get("ema915_breach", False)),
            "ema915_side": ema_state.get("ema915_side"),
            "ema915_age_ms": ema_state.get("ema915_age_ms"),
        }

        ckey = str(metrics.get("_candle_ts_utc") or metrics.get("_candle_ts") or metrics.get("ts") or "")
        if self._setup_candle_key == ckey:
            return None
        self._setup_candle_key = ckey

        conflict = abs(margin) < float(self.cfg.conflict_margin)
        ready_long = margin >= float(self.cfg.ready_margin)
        ready_short = margin <= -float(self.cfg.ready_margin)
        flip_long = margin >= float(self.cfg.flip_margin)
        flip_short = margin <= -float(self.cfg.flip_margin)

        if conflict:
            try:
                pa_mode = str(metrics.get("pa_mode") or "")
                pa_side = str(metrics.get("pa_side") or "")
                pa_strength = float(metrics.get("pa_strength", 0.0) or 0.0)
                if pa_mode == "dominance" and pa_side in ("LONG", "SHORT") and pa_strength >= float(getattr(self.cfg, "pa_min_strength", 0.62)):
                    self._setup_side = pa_side
                    self._setup_margin = margin
                    return {
                        "suggestion": f"READY_{pa_side}",
                        "reason": "pa_override_conflict",
                        "margin": margin,
                        "score_long": long_score,
                        "score_short": short_score,
                        "setup_side": pa_side,
                        **ema_info,
                    }
            except Exception:
                pass
            self._setup_margin = margin
            return {
                "suggestion": "HOLD_CONFLICT",
                "reason": "market_indecision",
                "margin": margin,
                "score_long": long_score,
                "score_short": short_score,
                "setup_side": self._setup_side,
                **ema_info,
            }

        # If no setup yet, create one when evidence is strong enough.
        if self._setup_side is None:
            if ready_long:
                self._setup_side = "LONG"
                self._setup_margin = margin
                return {
                    "suggestion": "READY_LONG",
                    "reason": "market_ready",
                    "margin": margin,
                    "score_long": long_score,
                    "score_short": short_score,
                    "setup_side": "LONG",
                    **ema_info,
                }
            if ready_short:
                self._setup_side = "SHORT"
                self._setup_margin = margin
                return {
                    "suggestion": "READY_SHORT",
                    "reason": "market_ready",
                    "margin": margin,
                    "score_long": long_score,
                    "score_short": short_score,
                    "setup_side": "SHORT",
                    **ema_info,
                }
            self._setup_margin = margin
            return {
                "suggestion": "HOLD_WEAK_EDGE",
                "reason": "weak_edge",
                "margin": margin,
                "score_long": long_score,
                "score_short": short_score,
                "setup_side": None,
                **ema_info,
            }

        # Setup exists: keep it unless opposite evidence is very strong (flip_margin).
        if self._setup_side == "LONG":
            if flip_short:
                self._setup_side = "SHORT"
                self._setup_margin = margin
                return {
                    "suggestion": "READY_SHORT",
                    "reason": "flip_proof",
                    "margin": margin,
                    "score_long": long_score,
                    "score_short": short_score,
                    "setup_side": "SHORT",
                    **ema_info,
                }
            self._setup_margin = margin
            if ready_long:
                return {
                    "suggestion": "READY_LONG",
                    "reason": "market_ready",
                    "margin": margin,
                    "score_long": long_score,
                    "score_short": short_score,
                    "setup_side": "LONG",
                    **ema_info,
                }
            return {
                "suggestion": "HOLD_WAIT_CONFIRM",
                "reason": "hold_setup_long",
                "margin": margin,
                "score_long": long_score,
                "score_short": short_score,
                "setup_side": "LONG",
                **ema_info,
            }

        # setup_side == "SHORT"
        if flip_long:
            self._setup_side = "LONG"
            self._setup_margin = margin
            return {
                "suggestion": "READY_LONG",
                "reason": "flip_proof",
                "margin": margin,
                "score_long": long_score,
                "score_short": short_score,
                "setup_side": "LONG",
                **ema_info,
            }
        self._setup_margin = margin
        if ready_short:
            return {
                "suggestion": "READY_SHORT",
                "reason": "market_ready",
                "margin": margin,
                "score_long": long_score,
                "score_short": short_score,
                "setup_side": "SHORT",
                **ema_info,
            }
        return {
            "suggestion": "HOLD_WAIT_CONFIRM",
            "reason": "hold_setup_short",
            "margin": margin,
            "score_long": long_score,
            "score_short": short_score,
            "setup_side": "SHORT",
            **ema_info,
        }

    # ------------------------ Strategy ------------------------
    def _evaluate_strategy(self, m: Dict[str, Any]) -> Dict[str, Any]:
        if not m:
            return {"suggestion": "HOLD", "reason": "warmup"}

        # Guard: during warmup/partial-metrics minutes, avoid KeyError cascades.
        # This can happen when metric dict is truthy but core fields are incomplete.
        core_keys = (
            "px",
            "atr",
            "body_pct",
            "path_eff",
            "clv",
            "pav_mult",
            "upper_wick_pct",
            "lower_wick_pct",
            "upper_band",
            "lower_band",
            "anchor_dist_atr_abs",
        )
        missing = [k for k in core_keys if k not in m]
        if missing:
            return {
                "suggestion": "HOLD",
                "reason": "warmup_missing_core",
                "missing_core": missing[:5],
                "missing_core_n": len(missing),
            }

        # Always keep last diag for tick engine
        self._last_diag = m

        # If in a position: manage state + candle-close exits (intrabar exits handled elsewhere)
        if self.pos is not None:
            self._retest_ctx = None
            return self._manage_open_position(m)

        regime = str(m.get("regime", "UNKNOWN") or "UNKNOWN")
        if regime == "CHOP" and (not bool(getattr(self.cfg, "allow_chop_trades", False))):
            if not bool(getattr(self.cfg, "allow_chop_pa_exceptions", True)):
                return {"suggestion": "NO_TRADE_CHOP", "reason": "regime_chop", "regime": regime}
            bo_e = float(m.get("breakout_energy", 0.0) or 0.0)
            pa_s = float(m.get("pa_strength", 0.0) or 0.0)
            if (bo_e < float(getattr(self.cfg, "breakout_energy_entry_min", 0.62))) and (pa_s < float(getattr(self.cfg, "chop_pa_min_strength", 0.72))):
                return {"suggestion": "NO_TRADE_CHOP", "reason": "regime_chop", "regime": regime}
            m["_chop_restricted"] = True

        # --- FLAT: entry selection ---
        # Global safety: doji / indecision filter
        if m["body_pct"] < self.cfg.min_body_pct:
            return {"suggestion": "HOLD", "reason": "doji_body"}

        # Chop safety (the bullet filter): block only if inefficient and jagged.
        if (m["path_eff"] < self.cfg.min_path_eff_filter) and (float(m.get("smoothness", 0.0) or 0.0) < 0.12):
            return {"suggestion": "HOLD", "reason": "chop_low_path_eff"}

        # Anti-FOMO: if outside bands, block entries in that direction
        # (If price is > upper_band, block longs; if price is < lower_band, block shorts)
        over_up = m["px"] > m["upper_band"]
        over_dn = m["px"] < m["lower_band"]

        # Anti-climax: detect selling climax absorption and block shorts
        anti_climax_block_short = (m["pav_mult"] > 2.5 and m["lower_wick_pct"] > 0.20)
        anti_climax_block_long = (m["pav_mult"] > 2.5 and m["upper_wick_pct"] > 0.20)

        # Shock system: WATCH, then confirm before entry
        shock_decision = self._handle_shock_system(m)
        if shock_decision is not None:
            return shock_decision

        # Price action engine priority: breakout-energy and PA-mode entries.
        pa_mode = str(m.get("pa_mode") or "")
        pa_side = str(m.get("pa_side") or "")
        pa_strength = float(m.get("pa_strength", 0.0) or 0.0)
        pa_level = m.get("pa_level")
        bo_side = str(m.get("breakout_side") or "")
        bo_level = m.get("breakout_level")
        bo_energy = float(m.get("breakout_energy", 0.0) or 0.0)
        margin_now = self._margin_from_metrics_safe(m)
        margin_abs = abs(float(margin_now))
        conf_now = float(m.get("confidence", 0.0) or 0.0)
        path_now = float(m.get("path_eff", 0.0) or 0.0)
        lat_ms = int(m.get("latency_ms", 0) or 0)
        dvwap_atr = self._abs_dvwap_atr(m, float(m.get("px", 0.0) or 0.0))

        def _pa_countertrend_hold() -> Optional[Dict[str, Any]]:
            """Require stronger proof when PA dominance fights active setup-side memory."""
            if pa_mode != "dominance" or pa_side not in ("LONG", "SHORT"):
                return None
            setup_side = str(self._setup_side or "")
            if setup_side not in ("LONG", "SHORT") or setup_side == pa_side:
                return None

            regime_now = str(m.get("regime") or "")
            allow_chop = bool(getattr(self.cfg, "pa_countertrend_allow_chop", False))
            energy_min = float(getattr(self.cfg, "pa_countertrend_min_energy", 0.72))
            margin_min = float(getattr(self.cfg, "pa_countertrend_min_margin", self.cfg.flip_margin) or self.cfg.flip_margin)
            margin_min = max(0.0, min(1.0, margin_min))
            energy_ok = bo_energy >= energy_min

            margin_now = 0.0
            try:
                margin_now = float(self._dir_margin_from_metrics(m, ema915_bias=0.0).get("margin", 0.0))
            except Exception:
                margin_now = 0.0
            margin_ok = margin_now >= margin_min if pa_side == "LONG" else margin_now <= -margin_min
            chop_blocked = (regime_now == "CHOP") and (not allow_chop)

            if energy_ok and margin_ok and (not chop_blocked):
                return None
            return {
                "suggestion": "HOLD_WAIT_CONFIRM",
                "reason": "pa_vs_setup_side",
                "setup_side": setup_side,
                "setup_margin": float(self._setup_margin),
                "countertrend_side": pa_side,
                "countertrend_margin": float(margin_now),
                "countertrend_margin_min": float(margin_min),
                "countertrend_energy": float(bo_energy),
                "countertrend_energy_min": float(energy_min),
                "countertrend_chop_blocked": bool(chop_blocked),
            }

        # Proactive Break Acceptance Lite: catch strong structure even when PA engine is neutral.
        if bool(getattr(self.cfg, "proactive_lite_enable", False)) and bo_side in ("LONG", "SHORT"):
            ext_min = float(getattr(self.cfg, "proactive_lite_extension_atr_min", 1.5) or 1.5)
            req_margin = float(getattr(self.cfg, "proactive_lite_abs_margin_min", 0.25) or 0.25)
            if dvwap_atr >= ext_min:
                req_margin = float(getattr(self.cfg, "proactive_lite_abs_margin_min_ext", req_margin) or req_margin)
            req_margin = max(0.0, min(1.0, req_margin))
            margin_ok = margin_now >= req_margin if bo_side == "LONG" else margin_now <= -req_margin
            lite_ok = (
                bo_energy >= float(getattr(self.cfg, "proactive_lite_energy_min", 0.52))
                and dvwap_atr >= ext_min
                and margin_ok
                and path_now >= float(getattr(self.cfg, "proactive_lite_path_eff_min", 0.08))
                and conf_now >= float(getattr(self.cfg, "proactive_lite_conf_min", 0.30))
            )
            if lite_ok:
                max_lat = int(getattr(self.cfg, "proactive_lite_latency_max_ms", 0) or 0)
                if max_lat > 0 and abs(lat_ms) > max_lat:
                    return {
                        "suggestion": f"ARM_{bo_side}",
                        "reason": "proactive_lite_high_latency",
                        "breakout_energy": bo_energy,
                        "margin": margin_now,
                        "margin_abs": margin_abs,
                        "margin_req": req_margin,
                        "dvwap_atr": dvwap_atr,
                        "latency_ms": lat_ms,
                    }
                if bo_side == "LONG" and (not over_up) and (not anti_climax_block_long):
                    sl_hint = float(bo_level - 2.0 * self._TICK_SIZE) if bo_level is not None else None
                    return self._execute_entry("LONG", m, reason="proactive_break_lite", sl_hint=sl_hint)
                if bo_side == "SHORT" and (not over_dn) and (not anti_climax_block_short):
                    sl_hint = float(bo_level + 2.0 * self._TICK_SIZE) if bo_level is not None else None
                    return self._execute_entry("SHORT", m, reason="proactive_break_lite", sl_hint=sl_hint)

        if bo_side in ("LONG", "SHORT") and bo_level is not None and bo_energy >= float(getattr(self.cfg, "breakout_energy_entry_min", 0.62)):
            if (
                bo_side == "LONG"
                and (not over_up)
                and (not anti_climax_block_long)
                and m.get("bias") == "LONG"
                and float(m.get("path_eff", 0.0) or 0.0) >= float(self.cfg.entry_path_eff)
                and float(m.get("clv", 0.0) or 0.0) >= float(self.cfg.min_clv_confirm)
            ):
                sl_hint = float(bo_level - 2.0 * self._TICK_SIZE)
                return self._execute_entry("LONG", m, reason="breakout_energy", sl_hint=sl_hint)
            if (
                bo_side == "SHORT"
                and (not over_dn)
                and (not anti_climax_block_short)
                and m.get("bias") == "SHORT"
                and float(m.get("path_eff", 0.0) or 0.0) >= float(self.cfg.entry_path_eff)
                and float(m.get("clv", 0.0) or 0.0) <= -float(self.cfg.min_clv_confirm)
            ):
                sl_hint = float(bo_level + 2.0 * self._TICK_SIZE)
                return self._execute_entry("SHORT", m, reason="breakout_energy", sl_hint=sl_hint)

        if pa_mode in ("uturn", "rejection", "dominance") and pa_side in ("LONG", "SHORT") and pa_strength >= float(getattr(self.cfg, "pa_min_strength", 0.62)):
            if pa_mode == "uturn":
                if pa_side == "LONG" and (not over_up) and (not anti_climax_block_long):
                    sl_hint = float(pa_level - 2.0 * self._TICK_SIZE) if pa_level is not None else None
                    return self._execute_entry("LONG", m, reason="pa_uturn", sl_hint=sl_hint)
                if pa_side == "SHORT" and (not over_dn) and (not anti_climax_block_short):
                    sl_hint = float(pa_level + 2.0 * self._TICK_SIZE) if pa_level is not None else None
                    return self._execute_entry("SHORT", m, reason="pa_uturn", sl_hint=sl_hint)

            if pa_mode == "rejection":
                loc_sup, loc_res, _, _ = self._get_sr_levels_locked(lb=int(getattr(self.cfg, "support_lookback", 45)))
                atrx = max(self._TICK_SIZE, float(m.get("atr", self._MICRO_ATR_FALLBACK) or self._MICRO_ATR_FALLBACK))
                sr_buf = float(getattr(self.cfg, "sr_buffer_atr", 0.0) or 0.0) * atrx
                px0 = float(m.get("px", 0.0) or 0.0)
                if pa_side == "LONG" and (not over_up) and (not anti_climax_block_long) and px0 <= (loc_sup + sr_buf):
                    sl_hint = float((pa_level if pa_level is not None else loc_sup) - 2.0 * self._TICK_SIZE)
                    return self._execute_entry("LONG", m, reason="pa_rejection", sl_hint=sl_hint)
                if pa_side == "SHORT" and (not over_dn) and (not anti_climax_block_short) and px0 >= (loc_res - sr_buf):
                    sl_hint = float((pa_level if pa_level is not None else loc_res) + 2.0 * self._TICK_SIZE)
                    return self._execute_entry("SHORT", m, reason="pa_rejection", sl_hint=sl_hint)

            if pa_mode == "dominance":
                countertrend_hold = _pa_countertrend_hold()
                if countertrend_hold is not None:
                    return countertrend_hold
                if pa_side == "LONG" and (not over_up) and (not anti_climax_block_long) and float(m.get("clv", 0.0) or 0.0) >= float(self.cfg.min_clv_confirm):
                    sl_hint = float(pa_level - 2.0 * self._TICK_SIZE) if pa_level is not None else None
                    return self._execute_entry("LONG", m, reason="pa_dominance", sl_hint=sl_hint)
                if pa_side == "SHORT" and (not over_dn) and (not anti_climax_block_short) and float(m.get("clv", 0.0) or 0.0) <= -float(self.cfg.min_clv_confirm):
                    sl_hint = float(pa_level + 2.0 * self._TICK_SIZE) if pa_level is not None else None
                    return self._execute_entry("SHORT", m, reason="pa_dominance", sl_hint=sl_hint)

        if pa_side in ("LONG", "SHORT") and pa_strength >= float(getattr(self.cfg, "pa_veto_strength", 0.78)):
            if pa_side == "LONG" and m.get("bias") == "SHORT":
                return {"suggestion": "HOLD", "reason": "pa_veto_vs_short"}
            if pa_side == "SHORT" and m.get("bias") == "LONG":
                return {"suggestion": "HOLD", "reason": "pa_veto_vs_long"}

        # Dist gate: enter only when still "fresh" (within 1.5 ATR of anchor)
        if m["anchor_dist_atr_abs"] > float(getattr(self.cfg, "max_anchor_dist_atr", 1.5)):
            return {"suggestion": "HOLD", "reason": "too_far_from_anchor"}

        # Patterns: Clean Sweep / Inside Break / Squeeze Engulfing
        # 1) Clean Sweep (institutional-like)
        if m["path_eff"] >= self.cfg.entry_path_eff and m["travel_per_tick"] >= (m["avg_travel_per_tick"] * 1.05):
            if (not over_up) and (not anti_climax_block_long) and m["clv"] >= self.cfg.strong_clv and (m["bias"] == "LONG") and float(m.get("activity_w", 0.0) or 0.0) >= self.cfg.min_activity_w_confirm:
                return self._execute_entry("LONG", m, reason="clean_sweep")
            if (not over_dn) and (not anti_climax_block_short) and m["clv"] <= -self.cfg.strong_clv and (m["bias"] == "SHORT") and float(m.get("activity_w", 0.0) or 0.0) >= self.cfg.min_activity_w_confirm:
                return self._execute_entry("SHORT", m, reason="clean_sweep")

        # 2) Inside Bar Break (fresh break near anchor)
        inside_decision = self._inside_break_entry(m, over_up, over_dn, anti_climax_block_long, anti_climax_block_short)
        if inside_decision is not None:
            return inside_decision

        # 3) Break-retest-hold continuation.
        retest_decision = self._retest_entry(m, over_up, over_dn, anti_climax_block_long, anti_climax_block_short)
        if retest_decision is not None:
            return retest_decision

        # 4) Engulfing after squeeze (expansion scalp)
        engulf_decision = self._squeeze_engulf_entry(m, over_up, over_dn, anti_climax_block_long, anti_climax_block_short)
        if engulf_decision is not None:
            return engulf_decision

        # Default: pressure + bias + confirm
        if (not over_up) and (not anti_climax_block_long) and (m["bias"] == "LONG") and m["clv"] >= self.cfg.min_clv_confirm and float(m.get("activity_w", 0.0) or 0.0) >= self.cfg.min_activity_w_confirm and m["path_eff"] >= float(self.cfg.entry_path_eff):
            return self._execute_entry("LONG", m, reason="bias_pressure")
        if (not over_dn) and (m["bias"] == "SHORT") and m["clv"] <= -self.cfg.min_clv_confirm and float(m.get("activity_w", 0.0) or 0.0) >= self.cfg.min_activity_w_confirm and m["path_eff"] >= float(self.cfg.entry_path_eff) and not anti_climax_block_short:
            return self._execute_entry("SHORT", m, reason="bias_pressure")

        # Anti-climax explicitly explains why we didn't short
        if anti_climax_block_short:
            return {"suggestion": "HOLD", "reason": "anti_climax_block_short"}
        if anti_climax_block_long:
            return {"suggestion": "HOLD", "reason": "anti_climax_block_long"}

        return {"suggestion": "HOLD", "reason": "scanning"}

    def _handle_shock_system(self, m: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Shock confirmation driven by energy, not by fixed candle-count."""
        now = _now_utc()
        max_dist = float(getattr(self.cfg, "max_anchor_dist_atr", 1.5))
        anti_climax_block_short = (
            float(m.get("pav_mult", 0.0) or 0.0) > 2.5 and float(m.get("lower_wick_pct", 0.0) or 0.0) > 0.20
        )
        anti_climax_block_long = (
            float(m.get("pav_mult", 0.0) or 0.0) > 2.5 and float(m.get("upper_wick_pct", 0.0) or 0.0) > 0.20
        )
        ckey = str(m.get("_candle_ts_utc") or m.get("_candle_ts") or m.get("ts") or "")

        if self.pending_shock is not None:
            ps = self.pending_shock
            if ps.expires_at is not None and now > ps.expires_at:
                self.pending_shock = None
                return {"suggestion": "HOLD", "reason": "shock_expired_time"}

            if ps.last_candle_key != ckey:
                ps.last_candle_key = ckey
                path_eff = self._clamp01(float(m.get("path_eff", 0.0) or 0.0))
                act_w = self._clamp01(float(m.get("activity_w", 0.0) or 0.0))
                keep = float(getattr(self.cfg, "shock_energy_decay_base", 0.78))
                keep += float(getattr(self.cfg, "shock_energy_decay_eff_w", 0.18)) * path_eff
                keep += float(getattr(self.cfg, "shock_energy_decay_act_w", 0.12)) * act_w
                keep = max(0.35, min(0.98, keep))
                pxv = float(m.get("px", 0.0) or 0.0)
                if ps.side == "LONG":
                    ps.best_px = max(float(ps.best_px), pxv)
                    if pxv < ps.mid_px:
                        keep *= 0.85
                else:
                    ps.best_px = min(float(ps.best_px), pxv) if ps.best_px > 0 else pxv
                    if pxv > ps.mid_px:
                        keep *= 0.85
                ps.energy = float(self._clamp01(float(ps.energy) * keep))

            if float(ps.energy) < float(getattr(self.cfg, "shock_energy_min", 0.25)):
                self.pending_shock = None
                return {"suggestion": "HOLD", "reason": "shock_expired_energy"}

            if ps.side == "LONG":
                if (
                    m["px"] > ps.mid_px
                    and m["path_eff"] >= self.cfg.entry_path_eff
                    and m["clv"] >= self.cfg.min_clv_confirm
                    and float(m.get("activity_w", 0.0) or 0.0) >= self.cfg.min_activity_w_confirm
                ):
                    if float(m.get("anchor_dist_atr_abs", 0.0) or 0.0) > max_dist:
                        return {"suggestion": "HOLD_WAIT_CONFIRM", "reason": "shock_confirm_too_far_from_anchor"}
                    if bool(m.get("px", 0.0) > m.get("upper_band", float("inf"))):
                        return {"suggestion": "HOLD_WAIT_CONFIRM", "reason": "shock_confirm_overextended_long"}
                    if anti_climax_block_long:
                        return {"suggestion": "HOLD_WAIT_CONFIRM", "reason": "shock_confirm_anti_climax_long"}
                    self.pending_shock = None
                    return {"suggestion": "ENTRY_LONG", "reason": "shock_confirm", "shock_mid": ps.mid_px, "shock_energy": float(ps.energy)}
            else:
                if (
                    m["px"] < ps.mid_px
                    and m["path_eff"] >= self.cfg.entry_path_eff
                    and m["clv"] <= -self.cfg.min_clv_confirm
                    and float(m.get("activity_w", 0.0) or 0.0) >= self.cfg.min_activity_w_confirm
                ):
                    if float(m.get("anchor_dist_atr_abs", 0.0) or 0.0) > max_dist:
                        return {"suggestion": "HOLD_WAIT_CONFIRM", "reason": "shock_confirm_too_far_from_anchor"}
                    if bool(m.get("px", 0.0) < m.get("lower_band", -float("inf"))):
                        return {"suggestion": "HOLD_WAIT_CONFIRM", "reason": "shock_confirm_overextended_short"}
                    if anti_climax_block_short:
                        return {"suggestion": "HOLD_WAIT_CONFIRM", "reason": "shock_confirm_anti_climax_short"}
                    self.pending_shock = None
                    return {"suggestion": "ENTRY_SHORT", "reason": "shock_confirm", "shock_mid": ps.mid_px, "shock_energy": float(ps.energy)}

            return {"suggestion": "HOLD", "reason": "watching_shock", "shock_energy": float(ps.energy)}

        if bool(m.get("shock", False)):
            atr = max(self._TICK_SIZE, float(m.get("atr", self._MICRO_ATR_FALLBACK) or self._MICRO_ATR_FALLBACK))
            shock_size_atr = abs(float(m.get("shock_size", 0.0) or 0.0)) / max(_EPS, atr)
            path_eff = self._clamp01(float(m.get("path_eff", 0.0) or 0.0))
            act_w = self._clamp01(float(m.get("activity_w", 0.0) or 0.0))
            clv_abs = self._clamp01(abs(float(m.get("clv", 0.0) or 0.0)))
            init = self._clamp01(0.30 * self._clamp01(shock_size_atr / 1.0) + 0.30 * path_eff + 0.25 * act_w + 0.15 * clv_abs)
            init = max(0.30, float(init))
            self.pending_shock = PendingShock(
                side=str(m.get("shock_side", "LONG")),
                mid_px=float(m.get("shock_mid", m["px"])),
                energy=float(init),
                best_px=float(m.get("px", 0.0) or 0.0),
                expires_at=now + timedelta(minutes=int(self.cfg.shock_expiry_minutes)),
                last_candle_key=ckey,
            )
            now_ms = int(time.time_ns() // 1_000_000)
            self._shock_lock_side = self.pending_shock.side
            self._shock_lock_until_ms = now_ms + int(self.cfg.shock_lock_ms)
            return {"suggestion": f"WATCH_SHOCK_{self.pending_shock.side}", "reason": "shock_detected", "shock_energy": float(self.pending_shock.energy)}

        return None

    def _inside_break_entry(
        self,
        m: Dict[str, Any],
        over_up: bool,
        over_dn: bool,
        anti_climax_block_long: bool,
        anti_climax_block_short: bool,
    ) -> Optional[Dict[str, Any]]:
        """Break of previous candle high/low, gated within 1 ATR of anchor."""
        if len(self.candles) < 2:
            return None

        prev = self.candles[-2]
        prev_high = float(prev.get("high", m["px"]))
        prev_low = float(prev.get("low", m["px"]))

        # Within 1 ATR of anchor
        if m["anchor_dist_atr_abs"] > 1.0:
            return None

        # Breakout up
        if (not over_up) and (not anti_climax_block_long) and m["px"] > prev_high and m["bias"] == "LONG" and m["path_eff"] >= float(self.cfg.entry_path_eff) and m["clv"] >= self.cfg.min_clv_confirm and float(m.get("activity_w", 0.0) or 0.0) >= self.cfg.min_activity_w_confirm:
            return self._execute_entry("LONG", m, reason="break_prev_high")

        # Breakdown down
        if (not over_dn) and (not anti_climax_block_short) and m["px"] < prev_low and m["bias"] == "SHORT" and m["path_eff"] >= float(self.cfg.entry_path_eff) and m["clv"] <= -self.cfg.min_clv_confirm and float(m.get("activity_w", 0.0) or 0.0) >= self.cfg.min_activity_w_confirm:
            return self._execute_entry("SHORT", m, reason="break_prev_low")

        return None

    def _retest_entry(
        self,
        m: Dict[str, Any],
        over_up: bool,
        over_dn: bool,
        anti_climax_block_long: bool,
        anti_climax_block_short: bool,
    ) -> Optional[Dict[str, Any]]:
        """Break -> retest -> hold continuation entry with runaway-distance expiry."""
        if len(self.candles) < 2:
            self._retest_ctx = None
            return None

        prev = self.candles[-2]
        curr = self.candles[-1]
        prev_high = float(prev.get("high", m["px"]) or m["px"])
        prev_low = float(prev.get("low", m["px"]) or m["px"])
        close_px = float(curr.get("close", m["px"]) or m["px"])
        low_px = float(curr.get("low", close_px) or close_px)
        high_px = float(curr.get("high", close_px) or close_px)
        atr = max(self._TICK_SIZE, float(m.get("atr", self._MICRO_ATR_FALLBACK) or self._MICRO_ATR_FALLBACK))
        act_w = float(m.get("activity_w", 0.0) or 0.0)
        path_eff = float(m.get("path_eff", 0.0) or 0.0)
        touch_ticks = max(0, int(getattr(self.cfg, "retest_touch_ticks", 2) or 2))
        touch_buf = float(touch_ticks) * float(self._TICK_SIZE)
        invalid_atr = float(getattr(self.cfg, "retest_invalid_atr", 0.25) or 0.25)
        rmin = float(getattr(self.cfg, "retest_runaway_atr_min", 0.85) or 0.85)
        rmax = float(getattr(self.cfg, "retest_runaway_atr_max", 1.55) or 1.55)
        pe = self._clamp01((path_eff - float(self.cfg.min_path_eff_filter)) / max(_EPS, (1.0 - float(self.cfg.min_path_eff_filter))))
        runaway_atr = rmax - (rmax - rmin) * pe
        runaway_pts = runaway_atr * atr

        if self._retest_ctx is None:
            if close_px > (prev_high + self._TICK_SIZE) and m.get("bias") == "LONG" and not over_up:
                self._retest_ctx = {"side": "LONG", "level": prev_high, "best_px": close_px, "runaway_pts": runaway_pts}
            elif close_px < (prev_low - self._TICK_SIZE) and m.get("bias") == "SHORT" and not over_dn:
                self._retest_ctx = {"side": "SHORT", "level": prev_low, "best_px": close_px, "runaway_pts": runaway_pts}
            return None

        side = str(self._retest_ctx.get("side", "") or "")
        level = float(self._retest_ctx.get("level", close_px) or close_px)
        best_px = float(self._retest_ctx.get("best_px", close_px) or close_px)
        runaway_pts = float(self._retest_ctx.get("runaway_pts", runaway_pts) or runaway_pts)

        if side == "LONG":
            best_px = max(best_px, close_px)
        elif side == "SHORT":
            best_px = min(best_px, close_px)
        self._retest_ctx["best_px"] = best_px

        if abs(best_px - level) >= runaway_pts:
            self._retest_ctx = None
            return None

        if side == "LONG":
            touched = low_px <= (level + touch_buf)
            held = close_px >= (level + self._TICK_SIZE)
            invalid = close_px < (level - invalid_atr * atr)
            if touched and held and (not over_up) and (not anti_climax_block_long) and m.get("bias") == "LONG" and float(m.get("path_eff", 0.0) or 0.0) >= 0.35 and float(m.get("clv", 0.0) or 0.0) >= float(self.cfg.min_clv_confirm) and act_w >= float(self.cfg.min_activity_w_confirm):
                self._retest_ctx = None
                sl_hint = float(low_px - 2.0 * self._TICK_SIZE)
                return self._execute_entry("LONG", m, reason="retest_hold", sl_hint=sl_hint)
            if invalid:
                self._retest_ctx = None
                return None
        elif side == "SHORT":
            touched = high_px >= (level - touch_buf)
            held = close_px <= (level - self._TICK_SIZE)
            invalid = close_px > (level + invalid_atr * atr)
            if touched and held and (not over_dn) and (not anti_climax_block_short) and m.get("bias") == "SHORT" and float(m.get("path_eff", 0.0) or 0.0) >= 0.35 and float(m.get("clv", 0.0) or 0.0) <= -float(self.cfg.min_clv_confirm) and act_w >= float(self.cfg.min_activity_w_confirm):
                self._retest_ctx = None
                sl_hint = float(high_px + 2.0 * self._TICK_SIZE)
                return self._execute_entry("SHORT", m, reason="retest_hold", sl_hint=sl_hint)
            if invalid:
                self._retest_ctx = None
                return None
        else:
            self._retest_ctx = None
            return None

        return None

    def _squeeze_engulf_entry(
        self,
        m: Dict[str, Any],
        over_up: bool,
        over_dn: bool,
        anti_climax_block_long: bool,
        anti_climax_block_short: bool,
    ) -> Optional[Dict[str, Any]]:
        """Engulfing candle after squeeze -> fast expansion scalp."""
        if len(self.candles) < 3:
            return None

        # We approximate "after squeeze" as: previous candle range was squeezed.
        # Since squeeze is current-candle flag, we infer previous squeeze by range percentile.
        prev = self.candles[-2]
        prev_range = float(prev.get("high", 0.0) - prev.get("low", 0.0))
        if len(self._range_hist) < self.cfg.squeeze_lookback:
            return None
        perc = float(np.percentile(np.array(self._range_hist, dtype=float), self.cfg.squeeze_pct))
        prev_squeeze = prev_range <= max(_EPS, perc)

        if not prev_squeeze:
            return None

        prev_open = float(prev.get("open", m["px"]))
        prev_close = float(prev.get("close", m["px"]))
        curr_open = float(self.candles[-1].get("open", m["px"]))
        curr_close = float(self.candles[-1].get("close", m["px"]))

        # Bullish engulfing
        bullish_engulf = (curr_close > curr_open) and (curr_close > prev_open) and (curr_open < prev_close)
        bearish_engulf = (curr_close < curr_open) and (curr_close < prev_open) and (curr_open > prev_close)

        if bullish_engulf and (not over_up) and (not anti_climax_block_long) and m["bias"] == "LONG" and m["path_eff"] >= 0.50 and m["clv"] >= self.cfg.min_clv_confirm and float(m.get("activity_w", 0.0) or 0.0) >= self.cfg.min_activity_w_confirm:
            return self._execute_entry("LONG", m, reason="squeeze_engulf")

        if bearish_engulf and (not over_dn) and (not anti_climax_block_short) and m["bias"] == "SHORT" and m["path_eff"] >= 0.50 and m["clv"] <= -self.cfg.min_clv_confirm and float(m.get("activity_w", 0.0) or 0.0) >= self.cfg.min_activity_w_confirm:
            return self._execute_entry("SHORT", m, reason="squeeze_engulf")

        return None

    def _manage_open_position(self, m: Dict[str, Any]) -> Dict[str, Any]:
        """Candle-close trade management. Intrabar SL/TP handled in tick engine."""
        p = self.pos
        if p is None:
            return {"suggestion": "HOLD", "reason": "no_pos"}

        atr = float(m.get("atr", 0.0))
        if not _is_finite_pos(atr):
            return {"suggestion": "HOLD", "reason": "bad_atr"}

        ts_exit = _now_utc()

        # Harvest mode: overextension switches to tighter trailing, no panic exit
        dist_abs = float(m.get("anchor_dist_atr_abs", 0.0) or 0.0)
        enter_harvest = float(self.cfg.stretch_limit)
        exit_harvest = float(self.cfg.stretch_limit) * 0.8
        if dist_abs >= enter_harvest:
            p.is_harvest = True
            p.trail_atr_current = self.cfg.harvest_trail_atr
        elif dist_abs <= exit_harvest:
            p.is_harvest = False
            p.trail_atr_current = self.cfg.trail_atr

        # Climax take-profit (captures peak speed before bounce)
        # Only if NOT harvest (harvest is already tight trailing)
        if not p.is_harvest and m["pav_mult"] >= 3.0:
            if p.side == "LONG" and m["upper_wick_pct"] >= self.cfg.wick_threshold and m["clv"] < self.cfg.min_clv_confirm:
                return self._execute_exit("EXIT_CLIMAX", px=m["px"], ts=ts_exit, reason="pav_wick_exhaust")
            if p.side == "SHORT" and m["lower_wick_pct"] >= self.cfg.wick_threshold and m["clv"] > -self.cfg.min_clv_confirm:
                return self._execute_exit("EXIT_CLIMAX", px=m["px"], ts=ts_exit, reason="pav_wick_exhaust")

        # Wick rejection (volatility-aware)
        if p.side == "LONG" and m["upper_wick_pct"] >= self.cfg.wick_threshold and m["clv"] < self.cfg.min_clv_confirm:
            profit_atr = 0.0
            try:
                profit_atr = (float(m["px"]) - float(p.entry_px)) / max(float(m.get("atr", 0.0) or 0.0), 1e-9)
            except Exception:
                profit_atr = 0.0
            if profit_atr >= float(getattr(self.cfg, "wick_exit_min_profit_atr", 0.0)):
                return self._execute_exit("EXIT_WICK", px=m["px"], ts=ts_exit, reason="wick_reject")
        if p.side == "SHORT" and m["lower_wick_pct"] >= self.cfg.wick_threshold and m["clv"] > -self.cfg.min_clv_confirm:
            profit_atr = 0.0
            try:
                profit_atr = (float(p.entry_px) - float(m["px"])) / max(float(m.get("atr", 0.0) or 0.0), 1e-9)
            except Exception:
                profit_atr = 0.0
            if profit_atr >= float(getattr(self.cfg, "wick_exit_min_profit_atr", 0.0)):
                return self._execute_exit("EXIT_WICK", px=m["px"], ts=ts_exit, reason="wick_reject")

        # Harvest mode exit requires CLV flip (lets blow-off continue)
        if p.is_harvest:
            if p.side == "LONG" and m["clv"] <= -self.cfg.min_clv_confirm and float(m.get("activity_w", 0.0) or 0.0) >= self.cfg.min_activity_w_confirm:
                return self._execute_exit("EXIT_HARVEST", px=m["px"], ts=ts_exit, reason="clv_flip")
            if p.side == "SHORT" and m["clv"] >= self.cfg.min_clv_confirm and float(m.get("activity_w", 0.0) or 0.0) >= self.cfg.min_activity_w_confirm:
                return self._execute_exit("EXIT_HARVEST", px=m["px"], ts=ts_exit, reason="clv_flip")

        return {"suggestion": "HOLD", "reason": "manage_pos"}

    # ------------------------ Execution ------------------------
    def _execute_entry(
        self,
        side: str,
        m: Dict[str, Any],
        reason: str = "",
        *,
        tick_ts: Optional[datetime] = None,
        tick_px: Optional[float] = None,
        engine: str = "candle",
        sl_hint: Optional[float] = None,
    ) -> Dict[str, Any]:
        if tick_ts is not None:
            ts = tick_ts
        else:
            m_ts = m.get("ts")
            ts = m_ts if isinstance(m_ts, datetime) else _now_utc()
        px = float(tick_px) if tick_px is not None else float(m.get("px") or self._last_px)

        def _entry_veto_payload(veto_reason: str, *, extra_fields: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
            out = {
                "suggestion": "ENTRY_VETO",
                "state": "CANCEL",
                "stream": "signal",
                "channel": str(engine),
                "engine": str(engine),
                "reason": str(veto_reason),
                "veto_reason": str(veto_reason),
                "entry_side": str(side),
                "px": float(px),
                "entry_px": float(px),
                "ts_utc": ts.astimezone(timezone.utc).isoformat(),
                "ts_ist": _iso_ist(ts),
            }
            if isinstance(extra_fields, dict):
                out.update(extra_fields)
            return out

        # Universal pre-entry gates (applies to both candle and tick engines).
        margin_signed = self._margin_from_metrics_safe(m)
        margin_abs = abs(float(margin_signed))
        min_abs_margin = float(getattr(self.cfg, "min_abs_margin_entry", 0.0) or 0.0)
        if min_abs_margin > 0.0 and margin_abs < min_abs_margin:
            return _entry_veto_payload(
                f"weak_margin_{margin_abs:.2f}",
                extra_fields={"margin": float(margin_signed), "margin_abs": float(margin_abs), "margin_min": float(min_abs_margin)},
            )

        same_side_cd = self._same_side_stopout_block_reason_locked(side, ts, m)
        if same_side_cd:
            return _entry_veto_payload(
                str(same_side_cd),
                extra_fields={"margin": float(margin_signed), "breakout_energy": float(m.get("breakout_energy", 0.0) or 0.0)},
            )

        try:
            guard = self._impulse_reentry_guard_by_side.get(side)
            if isinstance(guard, dict):
                win_end = guard.get("window_end_utc")
                if isinstance(win_end, datetime) and ts.astimezone(timezone.utc) <= win_end:
                    bo_e = float(m.get("breakout_energy", 0.0) or 0.0)
                    need_bo = float(getattr(self.cfg, "impulse_reentry_energy_min", 0.60) or 0.60)
                    need_m = float(getattr(self.cfg, "impulse_reentry_abs_margin_min", 0.15) or 0.15)
                    cnt = int(guard.get("count", 0) or 0)
                    cap = max(0, int(getattr(self.cfg, "impulse_reentry_max", 1) or 1))
                    if cnt >= cap:
                        return _entry_veto_payload(f"impulse_reentry_limit_{cnt}/{cap}", extra_fields={"breakout_energy": bo_e, "margin_abs": margin_abs})
                    if bo_e < need_bo or margin_abs < need_m:
                        return _entry_veto_payload(
                            "impulse_reentry_wait_edge",
                            extra_fields={"breakout_energy": bo_e, "breakout_energy_min": need_bo, "margin_abs": margin_abs, "margin_min": need_m},
                        )
                else:
                    self._impulse_reentry_guard_by_side[side] = None
        except Exception:
            pass

        if self._last_exit_side in ("LONG", "SHORT") and side != self._last_exit_side:
            min_flip_margin = float(getattr(self.cfg, "min_abs_margin_entry_flip", 0.20) or 0.20)
            if margin_abs < min_flip_margin:
                return _entry_veto_payload(
                    f"weak_flip_margin_{margin_abs:.2f}",
                    extra_fields={"margin_abs": margin_abs, "flip_margin_min": min_flip_margin},
                )
            nlock = max(0, int(getattr(self.cfg, "flip_lockout_candles", 0) or 0))
            if nlock > 0 and isinstance(self._last_exit_bucket_utc, datetime):
                curr_bucket = ts.astimezone(timezone.utc).replace(second=0, microsecond=0)
                bars_since = max(0, int((curr_bucket - self._last_exit_bucket_utc).total_seconds() // 60))
                if bars_since < nlock:
                    bo_e = float(m.get("breakout_energy", 0.0) or 0.0)
                    bo_ok = bo_e >= float(getattr(self.cfg, "flip_override_energy_min", 0.60) or 0.60)
                    margin_ok = margin_abs >= float(getattr(self.cfg, "flip_override_abs_margin_min", 0.20) or 0.20)
                    htf_hold = int(m.get("htf_rev_hold", 0) or 0) > 0
                    ema_breach = bool(m.get("ema915_breach", False))
                    ema_age = int(m.get("ema915_age_ms", 0) or 0)
                    ema_ok = ema_breach and ema_age >= int(getattr(self.cfg, "flip_override_ema915_age_ms", 350) or 350)
                    if not (margin_ok and htf_hold and (bo_ok or ema_ok)):
                        return _entry_veto_payload(
                            f"flip_firewall_{bars_since}/{nlock}",
                            extra_fields={
                                "margin_abs": margin_abs,
                                "breakout_energy": bo_e,
                                "htf_rev_hold": int(m.get("htf_rev_hold", 0) or 0),
                                "ema915_breach": bool(ema_breach),
                                "ema915_age_ms": ema_age,
                            },
                        )

        atr = float(m.get("atr", self._MICRO_ATR_FALLBACK) or self._MICRO_ATR_FALLBACK)
        atr = max(self._TICK_SIZE, atr)
        min_sl_distance = max(self._TICK_SIZE, float(self.cfg.min_init_sl_points), float(self.cfg.min_init_sl_atr) * atr)

        hard_min = float(self.cfg.hard_stop_points)
        hard_mult = float(self.cfg.hard_stop_atr_mult)
        hard_pts = max(hard_min, hard_mult * atr, min_sl_distance)
        try:
            tp_mult = float(m.get("tp_mult")) if m.get("tp_mult") is not None else float(self._select_tp_mult(m, side, reason))
        except Exception:
            tp_mult = float(self.cfg.tp_atr_mult)
        tp_mult = float(max(0.10, tp_mult))
        tp_pts = float(tp_mult * atr)
        m["tp_mult"] = tp_mult
        be_trigger_points = float(self.cfg.move_to_be_atr * atr)
        trail_arm_points = float(max(0.0, self.cfg.trail_start_atr) * atr)
        regime = str(m.get("regime", "UNKNOWN") or "UNKNOWN")
        setup_quality = str(m.get("setup_quality", "C") or "C")
        try:
            confidence = float(m.get("confidence", 0.0) or 0.0)
        except Exception:
            confidence = 0.0

        hard_sl_raw = (px - hard_pts) if side == "LONG" else (px + hard_pts)
        hard_sl = _snap_price(hard_sl_raw, self._TICK_SIZE, kind=("stop_long" if side == "LONG" else "stop_short"))
        sl = hard_sl
        forced_runner = m.get("runner", None)
        if forced_runner is None:
            is_runner = bool(self._select_runner(m)) or bool(getattr(self.cfg, "default_runner", False))
            m["runner_selected"] = bool(is_runner)
        else:
            is_runner = bool(forced_runner)
            m["runner_selected"] = bool(is_runner)
        if is_runner:
            tp = None
        else:
            tp_raw = (px + tp_pts) if side == "LONG" else (px - tp_pts)
            tp = _snap_price(tp_raw, self._TICK_SIZE, kind=("tp_long" if side == "LONG" else "tp_short"))

        # Structural stop hint may only tighten if it still respects minimum initial SL floor.
        sl_hint_status = "none"
        sl_hint_distance = None
        if sl_hint is not None:
            try:
                slh = float(sl_hint)
                if math.isfinite(slh):
                    if side == "LONG" and slh < (px - self._TICK_SIZE):
                        dist = float(px - slh)
                        sl_hint_distance = dist
                        if dist < min_sl_distance:
                            sl_hint_status = "rejected_too_tight"
                        else:
                            sl_hint_status = "applied"
                            slh = _snap_price(slh, self._TICK_SIZE, kind="stop_long")
                            sl = max(hard_sl, slh)
                    elif side == "SHORT" and slh > (px + self._TICK_SIZE):
                        dist = float(slh - px)
                        sl_hint_distance = dist
                        if dist < min_sl_distance:
                            sl_hint_status = "rejected_too_tight"
                        else:
                            sl_hint_status = "applied"
                            slh = _snap_price(slh, self._TICK_SIZE, kind="stop_short")
                            sl = min(hard_sl, slh)
                    else:
                        sl_hint_status = "ignored_invalid_side"
                else:
                    sl_hint_status = "ignored_non_finite"
            except Exception:
                sl_hint_status = "error"

        sl_points = abs(float(px) - float(sl))
        sl_atr = float(sl_points / max(_EPS, atr))
        tp_points = abs(float(tp) - float(px)) if tp is not None else None
        rr_est = float(tp_points / max(_EPS, sl_points)) if tp_points is not None else None
        entry_bucket_utc = ts.astimezone(timezone.utc).replace(second=0, microsecond=0)

        # SL sanity hard gate: reject entries with invalid/tiny hint or oversized initial risk.
        veto_reason: Optional[str] = None
        try:
            if bool(getattr(self.cfg, "block_on_sl_hint_rejected", True)) and sl_hint_status == "rejected_too_tight":
                veto_reason = "sl_hint_rejected"
            max_sl_points = float(getattr(self.cfg, "max_sl_points", 0.0) or 0.0)
            max_sl_atr = float(getattr(self.cfg, "max_sl_atr_mult", 0.0) or 0.0)
            if max_sl_points > 0.0 and sl_points > max_sl_points:
                veto_reason = f"sl_points_too_wide_{sl_points:.2f}>{max_sl_points:.2f}"
            if max_sl_atr > 0.0 and sl_atr > max_sl_atr:
                veto_reason = f"sl_atr_too_wide_{sl_atr:.2f}>{max_sl_atr:.2f}"
        except Exception:
            veto_reason = veto_reason or "sl_sanity_check_error"

        if veto_reason:
            try:
                self.log.warning(
                    "entry_veto_sl_sanity side=%s engine=%s reason=%s sl_points=%.2f sl_atr=%.2f",
                    side,
                    str(engine),
                    str(veto_reason),
                    float(sl_points),
                    float(sl_atr),
                )
            except Exception:
                pass
            return {
                "suggestion": "ENTRY_VETO",
                "state": "CANCEL",
                "stream": "signal",
                "channel": str(engine),
                "engine": str(engine),
                "reason": str(veto_reason),
                "veto_reason": str(veto_reason),
                "entry_side": str(side),
                "px": float(px),
                "entry_px": float(px),
                "entry_atr": float(atr),
                "sl": float(sl),
                "tp": float(tp) if tp is not None else None,
                "sl_points": float(sl_points),
                "sl_atr": float(sl_atr),
                "sl_hint_status": str(sl_hint_status),
                "sl_hint_distance": sl_hint_distance,
                "max_sl_points": float(getattr(self.cfg, "max_sl_points", 0.0) or 0.0),
                "max_sl_atr_mult": float(getattr(self.cfg, "max_sl_atr_mult", 0.0) or 0.0),
                "ts_utc": ts.astimezone(timezone.utc).isoformat(),
                "ts_ist": _iso_ist(ts),
            }

        if sl_atr < 0.40:
            self.log.warning(
                "entry_sl_floor_low side=%s engine=%s sl_atr=%.3f sl_points=%.2f floor_pts=%.2f floor_atr=%.2f reason=%s",
                side,
                str(engine),
                float(sl_atr),
                float(sl_points),
                float(min_sl_distance),
                float(self.cfg.min_init_sl_atr),
                str(sl_hint_status),
            )

        if sl_hint_status == "rejected_too_tight":
            self.log.warning(
                "sl_hint_rejected side=%s engine=%s hint_dist=%.2f min_sl_dist=%.2f",
                side,
                str(engine),
                float(sl_hint_distance or 0.0),
                float(min_sl_distance),
            )

        new_pos = Position(
            side=side,
            entry_px=px,
            entry_ts=ts,
            entry_bucket_utc=entry_bucket_utc,
            hard_sl=hard_sl,
            sl=sl,
            sl_init=sl,
            tp=tp,
            best_px=px,
            entry_atr=atr,
            be_trigger_points=be_trigger_points,
            trail_arm_points=trail_arm_points,
            rr_est=rr_est,
            regime=regime,
            setup_quality=setup_quality,
            confidence=confidence,
            is_runner=is_runner,
            engine=engine,
            why=reason or "",
        )

        reserved = {"stream", "engine", "channel", "in_pos", "pos_side"}
        payload = {
            "suggestion": f"ENTRY_{side}",
            "state": "TRIGGER",
            "channel": engine,
            "engine": engine,
            "in_pos": True,
            "px": px,
            "ts_utc": ts.astimezone(timezone.utc).isoformat(),
            "ts_ist": _iso_ist(ts),
            "reason": reason or "entry",
            "runner": new_pos.is_runner,
            "runner_selected": bool(m.get("runner_selected", new_pos.is_runner)),
            "pos_side": side,
            "entry_px": px,
            "entry_atr": atr,
            "tp_mult": float(m.get("tp_mult", self.cfg.tp_atr_mult) or self.cfg.tp_atr_mult),
            "sl": sl,
            "tp": tp,
            "hard_sl": hard_sl,
            "sl_points": sl_points,
            "sl_atr": sl_atr,
            "be_trigger_points": be_trigger_points,
            "trail_arm_points": trail_arm_points,
            "rr_est": rr_est,
            "regime": regime,
            "setup_quality": setup_quality,
            "confidence": confidence,
            "setup_side_snapshot": self._setup_side,
            "setup_margin_snapshot": float(self._setup_margin),
            "candle_bucket_id": entry_bucket_utc.isoformat(),
            "sl_hint_status": sl_hint_status,
            "sl_hint_distance": sl_hint_distance,
            "min_init_sl_points": float(self.cfg.min_init_sl_points),
            "min_init_sl_atr": float(self.cfg.min_init_sl_atr),
            "stream": "signal",
            **{
                k: v
                for k, v in m.items()
                if k not in ("suggestion", "px", "ts_ist", "reason") and k not in reserved
            },
        }

        try:
            with self._write_lock:
                self.jsonl_file.write(json.dumps(_isoize(payload), default=str) + "\n")
                self.jsonl_file.flush()
            try:
                self.log.info(self._fmt_scalper_line(payload))
            except Exception:
                pass
            self.pos = new_pos
            try:
                guard = self._impulse_reentry_guard_by_side.get(side)
                if isinstance(guard, dict):
                    wend = guard.get("window_end_utc")
                    if isinstance(wend, datetime) and ts.astimezone(timezone.utc) <= wend:
                        guard["count"] = int(guard.get("count", 0) or 0) + 1
                    else:
                        self._impulse_reentry_guard_by_side[side] = None
            except Exception:
                pass
            self._maybe_notify_payload(payload)
            return payload
        except Exception:
            self.pos = None
            raise

    def _resolve_entry_intents(self, intents: List[EntryIntent]) -> Optional[EntryIntent]:
        if not intents:
            self._intent_conflict_info = None
            return None

        # Clear previous conflict info unless we set it below.
        self._intent_conflict_info = None

        longs = [i for i in intents if i.side == "LONG"]
        shorts = [i for i in intents if i.side == "SHORT"]

        best_long = max(longs, key=lambda x: float(x.score)) if longs else None
        best_short = max(shorts, key=lambda x: float(x.score)) if shorts else None

        # If both sides are present and too close -> HOLD_CONFLICT (indecision)
        if best_long and best_short:
            delta = abs(float(best_long.score) - float(best_short.score))
            if delta < float(self.cfg.intent_conflict_score_delta):
                # PA dominance can resolve tie-conflicts when strong enough.
                try:
                    pa = self._last_pa if isinstance(self._last_pa, dict) else {}
                    pa_mode = str(pa.get("pa_mode") or "")
                    pa_side = str(pa.get("pa_side") or "")
                    pa_strength = float(pa.get("pa_strength", 0.0) or 0.0)
                    if pa_mode == "dominance" and pa_side in ("LONG", "SHORT") and pa_strength >= float(getattr(self.cfg, "pa_min_strength", 0.62)):
                        if pa_side == "LONG":
                            return best_long
                        return best_short
                except Exception:
                    pass
                self._intent_conflict_info = {
                    "delta": float(delta),
                    "best_long_engine": str(best_long.engine),
                    "best_short_engine": str(best_short.engine),
                    "best_long_score": float(best_long.score),
                    "best_short_score": float(best_short.score),
                }
                return None

            # Setup hysteresis preference: if we already have a setup side, require a bigger delta to flip.
            if self._setup_side in ("LONG", "SHORT"):
                preferred = best_long if self._setup_side == "LONG" else best_short
                other = best_short if self._setup_side == "LONG" else best_long
                if preferred and other:
                    if float(other.score) < float(preferred.score) + float(self.cfg.intent_flip_score_delta):
                        return preferred

        pref = {"ema915": 2, "micro": 1}
        intents.sort(key=lambda x: (x.score, pref.get(x.engine, 0)), reverse=True)
        return intents[0]

    def _recent_candle_streak_locked(self, want: str, *, max_lookback: int = 6) -> int:
        """Count consecutive up (GREEN) or down (RED) closed candles.

        Must be called under self._lock, because it reads self.candles.
        """
        if want not in ("GREEN", "RED"):
            return 0
        try:
            recent = list(self.candles)
        except Exception:
            return 0
        if not recent:
            return 0
        take = recent[-max_lookback:]
        streak = 0
        for c in reversed(take):
            try:
                o = float(c.get("open", 0.0) or 0.0)
                cl = float(c.get("close", 0.0) or 0.0)
            except Exception:
                break
            if want == "GREEN":
                if cl > o:
                    streak += 1
                    continue
                break
            else:
                if cl < o:
                    streak += 1
                    continue
                break
        return int(streak)

    @staticmethod
    def _flow_bias_from_fut(fut_flow: Optional[Dict[str, Any]]) -> float:
        """Compute a signed [-1..1] flow bias from fut_flow fields.

        Positive means buy pressure, negative means sell pressure.
        """
        if not isinstance(fut_flow, dict):
            return 0.0

        def _finite(v: Any, default: float = 0.0) -> float:
            try:
                f = float(v)
                return f if math.isfinite(f) else float(default)
            except Exception:
                return float(default)

        def _clamp(x: float) -> float:
            xx = _finite(x, 0.0)
            return max(-1.0, min(1.0, xx))

        depth_imb = _finite(fut_flow.get("depth_imb", 0.0), 0.0)
        buy_q = _finite(fut_flow.get("buy_qty", 0.0), 0.0)
        sell_q = _finite(fut_flow.get("sell_qty", 0.0), 0.0)
        denom = max(_EPS, buy_q + sell_q)
        qty_imb = (buy_q - sell_q) / denom

        # cvd_delta: if present we use its sign only (normalization differs across feeds)
        cvd_delta = fut_flow.get("cvd_delta")
        cvd_sign = 0.0
        try:
            if cvd_delta is not None:
                cvd_v = _finite(cvd_delta, 0.0)
                cvd_sign = 1.0 if cvd_v > 0 else (-1.0 if cvd_v < 0 else 0.0)
        except Exception:
            cvd_sign = 0.0

        # Weight depth + executed imbalance higher than cvd sign.
        bias = (0.55 * _clamp(depth_imb)) + (0.55 * _clamp(qty_imb)) + (0.25 * cvd_sign)
        return _clamp(bias)

    def _margin_from_metrics_safe(self, m: Dict[str, Any]) -> float:
        try:
            return float(self._dir_margin_from_metrics(m or {}, ema915_bias=0.0).get("margin", 0.0))
        except Exception:
            return 0.0

    def _abs_dvwap_atr(self, m: Dict[str, Any], px_hint: Optional[float] = None) -> float:
        try:
            px = float(px_hint) if px_hint is not None else float(m.get("px", 0.0) or 0.0)
            atr = max(self._TICK_SIZE, float(m.get("atr", self._MICRO_ATR_FALLBACK) or self._MICRO_ATR_FALLBACK))
            fut = m.get("fut_flow")
            if isinstance(fut, dict):
                vwap = fut.get("vwap")
                if vwap is not None:
                    return abs(px - float(vwap)) / max(_EPS, atr)
            vwap0 = m.get("vwap")
            if vwap0 is not None:
                return abs(px - float(vwap0)) / max(_EPS, atr)
        except Exception:
            return 0.0
        return 0.0

    def _edge_score(self, m: Dict[str, Any]) -> float:
        """Compact [0..1] edge score for tiered notifications."""
        try:
            margin_abs = abs(float(m.get("margin", 0.0) or 0.0))
        except Exception:
            margin_abs = 0.0
        if margin_abs <= 0.0:
            margin_abs = abs(self._margin_from_metrics_safe(m))
        bo = max(0.0, min(1.0, float(m.get("breakout_energy", 0.0) or 0.0)))
        pa = max(0.0, min(1.0, float(m.get("pa_strength", 0.0) or 0.0)))
        pe = max(0.0, min(1.0, float(m.get("path_eff", 0.0) or 0.0)))
        ext = self._abs_dvwap_atr(m)
        ext_n = max(0.0, min(1.0, ext / 2.0))
        sc = (0.32 * max(0.0, min(1.0, margin_abs))) + (0.28 * bo) + (0.18 * ext_n) + (0.12 * pa) + (0.10 * pe)
        return max(0.0, min(1.0, float(sc)))

    def _post_notify_webhook(self, evt: Dict[str, Any]) -> None:
        url = str(getattr(self.cfg, "notify_webhook_url", "") or "")
        if not url:
            return
        try:
            body = json.dumps(_isoize(evt), default=str).encode("utf-8")
            req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
            with urllib.request.urlopen(req, timeout=1.5) as _:
                pass
        except (urllib.error.URLError, TimeoutError, ValueError):
            # Notification failures are non-fatal for trading.
            return
        except Exception:
            return

    def _notify_event(self, payload: Dict[str, Any], *, tier: str, side: str, debounce_s: int) -> None:
        if not bool(getattr(self.cfg, "notify_enable", False)):
            return
        now_ms = int(time.time() * 1000)
        key = f"{tier}|{side}|{str(payload.get('suggestion') or '')}"
        last_ms = int(self._notify_last_sent_by_key.get(key, 0) or 0)
        if debounce_s > 0 and (now_ms - last_ms) < (int(debounce_s) * 1000):
            return
        self._notify_last_sent_by_key[key] = now_ms

        edge = self._edge_score(payload)
        evt = {
            "tier": tier,
            "side": side,
            "suggestion": payload.get("suggestion"),
            "reason": payload.get("reason"),
            "engine": payload.get("engine"),
            "channel": payload.get("channel"),
            "ts_utc": payload.get("ts_utc") or payload.get("ts"),
            "ts_ist": payload.get("ts_ist"),
            "px": payload.get("px"),
            "regime": payload.get("regime"),
            "breakout_energy": payload.get("breakout_energy"),
            "margin": payload.get("margin"),
            "edge_score": edge,
            "latency_ms": payload.get("latency_ms"),
        }
        try:
            self.log.info(
                "notify tier=%s side=%s sug=%s reason=%s edge=%.2f",
                str(tier),
                str(side),
                str(payload.get("suggestion", "")),
                str(payload.get("reason", "")),
                float(edge),
            )
        except Exception:
            pass
        self._post_notify_webhook(evt)

    def _maybe_notify_payload(self, payload: Dict[str, Any]) -> None:
        try:
            sug = str(payload.get("suggestion", "") or "")
            if not sug:
                return
            side = str(payload.get("pos_side") or payload.get("pos_side_before") or "")
            if not side:
                side = "LONG" if "LONG" in sug else ("SHORT" if "SHORT" in sug else "")
            side = side or "NONE"
            edge = self._edge_score(payload)
            regime = str(payload.get("regime", "") or "")
            lat = payload.get("latency_ms")
            lat_i = int(lat) if lat is not None else 0

            if sug.startswith("ARM_"):
                if edge >= float(getattr(self.cfg, "notify_arm_edgescore", 0.0) or 0.0):
                    self._notify_event(payload, tier="ARM", side=side, debounce_s=int(getattr(self.cfg, "notify_debounce_arm_s", 60)))
                return

            if sug.startswith("ENTRY_") and sug != "ENTRY_VETO":
                tier = "GO"
                if (lat_i > int(getattr(self.cfg, "notify_latency_downgrade_ms", 0) or 0)) or regime == "CHOP":
                    tier = "ARM"
                min_edge = float(getattr(self.cfg, "notify_go_edgescore", 0.0) or 0.0) if tier == "GO" else float(
                    getattr(self.cfg, "notify_arm_edgescore", 0.0) or 0.0
                )
                if edge >= min_edge:
                    db = int(getattr(self.cfg, "notify_debounce_go_s", 60) or 60) if tier == "GO" else int(
                        getattr(self.cfg, "notify_debounce_arm_s", 60) or 60
                    )
                    self._notify_event(payload, tier=tier, side=side, debounce_s=db)
                return

            if sug.startswith("EXIT_"):
                self._notify_event(payload, tier="MANAGE", side=side, debounce_s=int(getattr(self.cfg, "notify_debounce_manage_s", 20)))
                return

            if sug in ("ENTRY_VETO", "HOLD_WAIT_CONFIRM", "COOLDOWN_ACTIVE"):
                if edge >= float(getattr(self.cfg, "notify_arm_edgescore", 0.0) or 0.0):
                    self._notify_event(payload, tier="ARM", side=side, debounce_s=int(getattr(self.cfg, "notify_debounce_arm_s", 60)))
        except Exception:
            pass

    def _same_side_stopout_block_reason_locked(self, side: str, intent_ts: datetime, m: Dict[str, Any]) -> Optional[str]:
        """Post-stopout same-side cooldown: require bars elapsed and reclaim beyond stopout reference."""
        if side not in ("LONG", "SHORT"):
            return None
        mem = self._stopout_cooldown_by_side.get(side)
        if not isinstance(mem, dict):
            return None

        exit_bucket = mem.get("exit_bucket_utc")
        if not isinstance(exit_bucket, datetime):
            self._stopout_cooldown_by_side[side] = None
            return None

        curr_bucket = intent_ts.astimezone(timezone.utc).replace(second=0, microsecond=0)
        try:
            min_bars = int(mem.get("min_bars", getattr(self.cfg, "cooldown_min_bars", 0)) or 0)
            min_bars = max(0, min_bars)
        except Exception:
            min_bars = 0
        bars_elapsed = max(0, int((curr_bucket - exit_bucket).total_seconds() // 60))

        # Reclaim override: during cooldown, allow same-side re-entry when edge is strong.
        try:
            bo_e = float(m.get("breakout_energy", 0.0) or 0.0)
            margin_abs = abs(self._margin_from_metrics_safe(m))
            bo_ok = bo_e >= float(getattr(self.cfg, "cooldown_reclaim_override_energy_min", 0.55))
            margin_ok = margin_abs >= float(getattr(self.cfg, "cooldown_reclaim_override_abs_margin_min", 0.25))
            if bo_ok and margin_ok:
                self._stopout_cooldown_by_side[side] = None
                return None
        except Exception:
            pass

        if bars_elapsed < min_bars:
            return f"stopout_cooldown_bars_{bars_elapsed}/{min_bars}"

        try:
            reclaim_pts = float(mem.get("reclaim_points", 0.0) or 0.0)
            ref_px = float(mem.get("stopout_px", 0.0) or 0.0)
            px = float(m.get("px", 0.0) or 0.0)
        except Exception:
            return "stopout_cooldown_wait"

        if reclaim_pts > 0.0 and ref_px > 0.0 and px > 0.0:
            if side == "LONG":
                if px < (ref_px + reclaim_pts):
                    return "stopout_cooldown_reclaim_wait"
            else:
                if px > (ref_px - reclaim_pts):
                    return "stopout_cooldown_reclaim_wait"

        # Cooldown satisfied, clear memory for this side.
        self._stopout_cooldown_by_side[side] = None
        return None

    def _entry_context_veto_locked(self, intent: EntryIntent, m: Dict[str, Any]) -> Optional[str]:
        """Dynamic, multi-signal entry veto that avoids fixed one-size-fits-all gates.

        Returns a short veto reason string when the entry should be held (WAIT_CONFIRM).
        Must be called under self._lock.
        """
        side = str(intent.side)
        if side not in ("LONG", "SHORT"):
            return None

        def _veto_exception(gate: str, exc: Exception) -> str:
            try:
                self.log.warning(
                    "entry_veto_gate_exception gate=%s side=%s engine=%s err=%s",
                    gate,
                    side,
                    str(getattr(intent, "engine", "") or ""),
                    str(exc),
                )
            except Exception:
                pass
            return f"veto_exception_{gate}"

        margin_signed = self._margin_from_metrics_safe(m)
        margin_abs = abs(float(margin_signed))
        dir_sign = 1.0 if side == "LONG" else -1.0
        margin_in_favor = margin_signed * dir_sign

        # Global weak-edge guard.
        try:
            min_abs_margin = float(getattr(self.cfg, "min_abs_margin_entry", 0.0) or 0.0)
            if min_abs_margin > 0.0 and margin_abs < min_abs_margin:
                return f"weak_margin_{margin_abs:.2f}"
        except Exception as e:
            return _veto_exception("margin_floor", e)

        # Cooldown after absorption traps (side-specific; prevents immediate re-fire).
        now_ms = int(time.time() * 1000)
        try:
            cd_until = int(getattr(self, "_entry_cooldown_until_ms_by_side", {}).get(side, 0) or 0)
            if now_ms < cd_until:
                return "cooldown_wait"
        except Exception as e:
            return _veto_exception("cooldown_state", e)

        atr = float(m.get("atr", self._MICRO_ATR_FALLBACK) or self._MICRO_ATR_FALLBACK)
        if not _is_finite_pos(atr):
            return None
        atr = max(self._TICK_SIZE, atr)

        # Regime governor: in chop, stand down unless explicitly enabled.
        try:
            if (not bool(getattr(self.cfg, "allow_chop_trades", False))) and str(m.get("regime", "UNKNOWN") or "UNKNOWN") == "CHOP":
                if not bool(getattr(self.cfg, "allow_chop_pa_exceptions", True)):
                    return "regime_chop"
                bo_e = float(m.get("breakout_energy", 0.0) or 0.0)
                pa_s = float(m.get("pa_strength", 0.0) or 0.0)
                if (bo_e < float(getattr(self.cfg, "breakout_energy_entry_min", 0.62))) and (pa_s < float(getattr(self.cfg, "chop_pa_min_strength", 0.72))):
                    return "regime_chop"
        except Exception as e:
            return _veto_exception("regime_guard", e)

        # Post stopout memory: avoid immediate same-side re-entry until reclaim.
        try:
            reason = self._same_side_stopout_block_reason_locked(side, intent.ts, m)
            if reason:
                return reason
        except Exception as e:
            return _veto_exception("stopout_memory", e)

        # Anti-overtrade limiter after impulse exits (TRAIL/TP).
        try:
            guard = self._impulse_reentry_guard_by_side.get(side)
            if isinstance(guard, dict):
                win_end = guard.get("window_end_utc")
                if isinstance(win_end, datetime) and intent.ts.astimezone(timezone.utc) <= win_end:
                    bo_e = float(m.get("breakout_energy", 0.0) or 0.0)
                    need_bo = float(getattr(self.cfg, "impulse_reentry_energy_min", 0.60) or 0.60)
                    need_m = float(getattr(self.cfg, "impulse_reentry_abs_margin_min", 0.15) or 0.15)
                    cnt = int(guard.get("count", 0) or 0)
                    cap = max(0, int(getattr(self.cfg, "impulse_reentry_max", 1) or 1))
                    if cnt >= cap:
                        return f"impulse_reentry_limit_{cnt}/{cap}"
                    if bo_e < need_bo or margin_abs < need_m:
                        return "impulse_reentry_wait_edge"
                else:
                    self._impulse_reentry_guard_by_side[side] = None
        except Exception as e:
            return _veto_exception("impulse_reentry_guard", e)

        # Flip firewall: block opposite-side entries for N candles unless reversal quality is high.
        try:
            if self._last_exit_side in ("LONG", "SHORT") and side != self._last_exit_side:
                min_flip_margin = float(getattr(self.cfg, "min_abs_margin_entry_flip", 0.20) or 0.20)
                if margin_abs < min_flip_margin:
                    return f"weak_flip_margin_{margin_abs:.2f}"
                nlock = max(0, int(getattr(self.cfg, "flip_lockout_candles", 0) or 0))
                if nlock > 0 and isinstance(self._last_exit_bucket_utc, datetime):
                    curr_bucket = intent.ts.astimezone(timezone.utc).replace(second=0, microsecond=0)
                    bars_since = max(0, int((curr_bucket - self._last_exit_bucket_utc).total_seconds() // 60))
                    if bars_since < nlock:
                        bo_e = float(m.get("breakout_energy", 0.0) or 0.0)
                        bo_ok = bo_e >= float(getattr(self.cfg, "flip_override_energy_min", 0.60) or 0.60)
                        margin_ok = margin_abs >= float(getattr(self.cfg, "flip_override_abs_margin_min", 0.20) or 0.20)
                        htf_hold = int(m.get("htf_rev_hold", 0) or 0) > 0
                        ema_breach = bool(m.get("ema915_breach", False))
                        ema_age = int(m.get("ema915_age_ms", 0) or 0)
                        ema_ok = ema_breach and ema_age >= int(getattr(self.cfg, "flip_override_ema915_age_ms", 350) or 350)
                        if not (margin_ok and htf_hold and (bo_ok or ema_ok)):
                            return f"flip_firewall_{bars_since}/{nlock}"
        except Exception as e:
            return _veto_exception("flip_firewall", e)

        px = float(intent.entry_px)
        hma = float(m.get("hma", px) or px)

        # Latency guard (only for tick engines: micro/ema915). 0 disables.
        try:
            max_lat = int(getattr(self.cfg, "entry_latency_max_ms", 0) or 0)
            eng = str(getattr(intent, "engine", "") or "")
            if max_lat > 0 and eng in ("micro", "ema915"):
                lat_raw = m.get("latency_ms", None)
                if lat_raw is not None:
                    lat = int(lat_raw)
                    if abs(lat) > max_lat:
                        return f"high_latency_{lat}ms"
        except Exception as e:
            return _veto_exception("latency_guard", e)

        # Optional strict futures-flow dependency for tick engines.
        try:
            eng = str(getattr(intent, "engine", "") or "")
            if eng in ("micro", "ema915") and bool(getattr(self.cfg, "use_fut_flow", False)):
                ff_mode = str(getattr(self.cfg, "fut_flow_fail_mode", "neutral") or "neutral").lower()
                ff_status = str(m.get("fut_flow_status", "") or "")
                has_flow = isinstance(m.get("fut_flow"), dict)
                if ff_mode == "hold":
                    if (not has_flow) or (ff_status and ff_status != "OK"):
                        return f"fut_flow_{(ff_status or 'missing').lower()}"
        except Exception as e:
            return _veto_exception("fut_flow_guard", e)

        # Squeeze + low efficiency => wait for expansion/confirmation.
        try:
            if bool(m.get("squeeze", False)) and float(m.get("path_eff", 0.0) or 0.0) < float(self.cfg.min_path_eff_filter):
                return "squeeze_chop_wait"
        except Exception as e:
            return _veto_exception("squeeze_guard", e)

        # Absorption trap veto (wick rejection + CLV). If triggered, also arm a short cooldown.
        try:
            clv0 = float(m.get("clv", 0.0) or 0.0)
            lw = float(m.get("lower_wick_pct", 0.0) or 0.0)
            uw = float(m.get("upper_wick_pct", 0.0) or 0.0)
            wick_th = float(getattr(self.cfg, "absorb_wick_pct", 0.0) or 0.0)
            clv_th = float(getattr(self.cfg, "absorb_clv_min", 0.0) or 0.0)
            if wick_th > 0.0:
                if side == "SHORT" and lw >= wick_th and clv0 >= clv_th:
                    cd = int(getattr(self.cfg, "absorb_cooldown_ms", 0) or 0)
                    if cd > 0:
                        self._entry_cooldown_until_ms_by_side["SHORT"] = now_ms + cd
                    return "bullish_absorption_wait"
                if side == "LONG" and uw >= wick_th and clv0 <= (-clv_th):
                    cd = int(getattr(self.cfg, "absorb_cooldown_ms", 0) or 0)
                    if cd > 0:
                        self._entry_cooldown_until_ms_by_side["LONG"] = now_ms + cd
                    return "bearish_absorption_wait"
        except Exception as e:
            return _veto_exception("absorption_guard", e)

        # SR / RR veto (anti-greed guard): don't short the floor / long the ceiling unless there's room.
        try:
            loc_sup, loc_res, sess_sup, sess_res = self._get_sr_levels_locked(lb=int(getattr(self.cfg, "support_lookback", 45)))
            hard_min = float(self.cfg.hard_stop_points)
            hard_mult = float(self.cfg.hard_stop_atr_mult)
            risk_pts = max(hard_min, hard_mult * atr)

            buf = float(getattr(self.cfg, "sr_buffer_atr", 0.0) or 0.0)
            rr_min = float(getattr(self.cfg, "entry_rr_min", 0.0) or 0.0)

            if side == "SHORT":
                floor = loc_sup if px >= loc_sup else sess_sup
                reward = max(0.0, px - floor)
                dist_atr = reward / max(_EPS, atr)
                rr = reward / max(_EPS, risk_pts)
                if buf > 0.0 and dist_atr < buf:
                    return "too_close_to_support"
                if rr_min > 0.0 and rr < rr_min:
                    return f"poor_rr_{rr:.2f}"
            else:
                ceil = loc_res if px <= loc_res else sess_res
                reward = max(0.0, ceil - px)
                dist_atr = reward / max(_EPS, atr)
                rr = reward / max(_EPS, risk_pts)
                if buf > 0.0 and dist_atr < buf:
                    return "too_close_to_resistance"
                if rr_min > 0.0 and rr < rr_min:
                    return f"poor_rr_{rr:.2f}"
        except Exception as e:
            return _veto_exception("sr_rr_guard", e)

        # If the intent didn't carry anchor_dist, compute it.
        anchor_dist_signed = m.get("anchor_dist_atr_signed")
        try:
            if anchor_dist_signed is None:
                anchor_dist_signed = (px - hma) / max(_EPS, atr)
            anchor_dist_signed = float(anchor_dist_signed)
        except Exception as e:
            return _veto_exception("anchor_dist", e)

        dir_sign = 1.0 if side == "LONG" else -1.0
        dist_in_favor = float(anchor_dist_signed) * dir_sign

        # Tick engines should not fight anchor unless tape is exceptionally strong.
        try:
            max_against = float(getattr(self.cfg, "anchor_align_max_against_atr", 0.60) or 0.60)
            eng = str(getattr(intent, "engine", "") or "")
            if max_against > 0.0 and eng in ("micro", "ema915"):
                if dist_in_favor < (-abs(max_against)):
                    return "against_anchor"
        except Exception as e:
            return _veto_exception("anchor_align_guard", e)

        # Avoid taking setups right on top of the anchor/HMA.
        try:
            min_anchor_atr = float(getattr(self.cfg, "entry_anchor_min_atr", 0.0))
            if min_anchor_atr > 0.0 and abs(float(anchor_dist_signed)) < min_anchor_atr:
                return "near_anchor"
        except Exception as e:
            return _veto_exception("near_anchor_guard", e)

        # PA veto on strong opposite candle intent.
        try:
            pa_side = str(m.get("pa_side") or "")
            pa_strength = float(m.get("pa_strength", 0.0) or 0.0)
            veto_th = float(getattr(self.cfg, "pa_veto_strength", 0.78) or 0.78)
            if pa_side in ("LONG", "SHORT") and pa_strength >= veto_th and pa_side != side:
                return "pa_veto"
        except Exception as e:
            return _veto_exception("pa_veto_guard", e)

        # Lip-touch breakout detection against dynamic ATR band around anchor.
        upper = hma + float(self.cfg.stretch_limit) * atr
        lower = hma - float(self.cfg.stretch_limit) * atr
        lip_margin = 0.0
        if side == "LONG":
            lip_margin = (px - upper) / max(_EPS, atr)
        else:
            lip_margin = (lower - px) / max(_EPS, atr)

        fut_flow = m.get("fut_flow")
        flow_bias = self._flow_bias_from_fut(fut_flow)
        flow_support = (flow_bias * dir_sign) >= (-float(self.cfg.flow_bias_veto))
        flow_opposes = (flow_bias * dir_sign) <= (-float(self.cfg.flow_bias_veto))

        clv = float(m.get("clv", 0.0) or 0.0)
        path_eff = float(m.get("path_eff", 0.0) or 0.0)
        momo_ok = (clv * dir_sign) >= float(self.cfg.min_clv_confirm) and path_eff >= float(self.cfg.entry_path_eff)

        # --- FEATURE 1: Micro proactive hardening ---
        # Make "catch it early" trades earn the right: strong candle evidence + strict flow neutrality.
        try:
            r = str(getattr(intent, "reason", "") or "")
            extra = getattr(intent, "extra", None)
            is_proactive = r.startswith("proactive_") or bool((extra or {}).get("is_proactive")) or bool((extra or {}).get("proactive"))
            if str(intent.engine) == "micro" and is_proactive:
                req_clv = float(getattr(self.cfg, "micro_harden_clv", 0.0) or 0.0)
                req_eff = float(getattr(self.cfg, "micro_harden_eff", 0.0) or 0.0)
                req_flow = float(getattr(self.cfg, "micro_flow_veto", 0.0) or 0.0)

                if req_eff > 0.0 and path_eff < req_eff:
                    return f"micro_proactive_weak_eff_{path_eff:.2f}"

                if req_clv > 0.0:
                    if side == "LONG" and clv < req_clv:
                        return f"micro_proactive_weak_clv_{clv:.2f}"
                    if side == "SHORT" and clv > -req_clv:
                        return f"micro_proactive_weak_clv_{clv:.2f}"

                # Strict flow check: even slight opposing flow blocks proactive entries.
                if req_flow > 0.0:
                    if side == "LONG" and flow_bias < -req_flow:
                        return f"micro_proactive_flow_opposes_{flow_bias:.2f}"
                    if side == "SHORT" and flow_bias > req_flow:
                        return f"micro_proactive_flow_opposes_{flow_bias:.2f}"
        except Exception as e:
            return _veto_exception("micro_proactive_guard", e)

        # --- FEATURE 2: HTF bearish filter for LONGs (requires reversal proven) ---
        # If the HTF proxy is bearish, do NOT allow LONG unless:
        # - last closed candle CLOSE broke above the local shelf (break)
        # - current tick is still above that shelf (hold)
        if side == "LONG":
            try:
                s15 = float(m.get("htf_slope_15", 0.0) or 0.0)
                s25 = float(m.get("htf_slope_25", 0.0) or 0.0)
                th = float(getattr(self.cfg, "htf_slope_thresh", 0.0) or 0.0)
                neg_th = -abs(th)

                # Treat as bearish only when both windows are meaningfully negative.
                if th > 0.0 and (s15 < neg_th) and (s25 < neg_th):
                    # Require 'break + hold' above a FIXED shelf level captured at break time.
                    # This avoids a single tick poke above local_res being treated as reversal.
                    try:
                        need = int(getattr(self.cfg, "htf_reversal_hold_bars", 2) or 2)
                    except Exception:
                        need = 2
                    need = max(1, need)
                    try:
                        buf_ticks = int(getattr(self.cfg, "htf_reversal_buffer_ticks", 1) or 1)
                    except Exception:
                        buf_ticks = 1
                    buf_ticks = max(0, buf_ticks)
                    buf = float(buf_ticks) * float(self._TICK_SIZE)

                    lvl = self._htf_long_rev_level
                    hold = int(getattr(self, "_htf_long_rev_hold", 0) or 0)
                    if lvl is None or hold <= 0:
                        return "htf_bear_wait_reversal_break"
                    if hold < need:
                        return f"htf_bear_wait_reversal_hold_{hold}/{need}"
                    if px <= (float(lvl) + buf):
                        return "htf_bear_wait_reversal_hold"
            except Exception as e:
                return _veto_exception("htf_bear_guard", e)

        # Buy/sell climax streak filter (scaled with anchor distance). Only strict when flow isn't confirming.
        streak = self._recent_candle_streak_locked("GREEN" if side == "LONG" else "RED", max_lookback=6)
        if streak >= int(self.cfg.climax_streak) and dist_in_favor >= float(self.cfg.climax_anchor_atr) and (not flow_support):
            return "climax_streak_wait_pullback"

        # Late breakout: when far from anchor, require *some* confirmation (flow or clean momentum).
        if dist_in_favor >= float(self.cfg.late_breakout_anchor_atr) and (not (flow_support or momo_ok)):
            return "late_breakout_no_confirm"

        # Lip-touch breakouts are fragile when flow actively opposes.
        if 0.0 <= lip_margin <= float(self.cfg.lip_breakout_atr) and flow_opposes:
            return "lip_breakout_flow_divergence"

        # Strong flow divergence near the band is a direct veto.
        if flow_opposes and dist_in_favor >= 0.60:
            return "flow_divergence"

        return None

    def _commit_entry_intent_locked(self, intent: EntryIntent, m0: Dict[str, Any]) -> None:
        extra = dict(m0)
        extra.update(intent.extra)
        ckey = str(
            extra.get("_candle_ts_utc")
            or extra.get("_candle_ts")
            or intent.ts.astimezone(timezone.utc).replace(second=0, microsecond=0).isoformat()
        )

        def _emit_hold(reason: str, **more: Any) -> None:
            if not self._should_emit_hold_wait_locked(
                str(reason),
                ckey,
                engine=str(intent.engine),
                want_side=str(intent.side),
            ):
                return
            rs = str(reason)
            sug = "COOLDOWN_ACTIVE" if rs.startswith("stopout_cooldown") or rs.startswith("cooldown_wait") else "HOLD_WAIT_CONFIRM"
            hold = {
                "channel": "tick",
                "engine": intent.engine,
                "ts": intent.ts.astimezone(timezone.utc).isoformat(),
                "suggestion": sug,
                "reason": str(reason),
                "want_side": intent.side,
                "px": float(intent.entry_px),
            }
            hold.update(more)
            self._write_signal(hold, extra)

        def _emit_entry_veto(reason: str, **more: Any) -> None:
            if not self._should_emit_hold_wait_locked(
                str(reason),
                ckey,
                engine=str(intent.engine),
                want_side=str(intent.side),
            ):
                return
            veto_evt = {
                "channel": "tick",
                "engine": intent.engine,
                "ts": intent.ts.astimezone(timezone.utc).isoformat(),
                "suggestion": "ENTRY_VETO",
                "reason": str(reason),
                "veto_reason": str(reason),
                "want_side": intent.side,
                "entry_reason": str(intent.reason or ""),
                "entry_score": float(intent.score),
                "px": float(intent.entry_px),
            }
            veto_evt.update(more)
            self._write_signal(veto_evt, extra)

        if bool(getattr(self.cfg, "respect_bias", False)):
            bias = str(extra.get("bias") or "")
            if bias in ("LONG", "SHORT") and intent.side != bias:
                shock_side = str(extra.get("shock_side") or "")
                if shock_side != intent.side:
                    return

        # Stop flip-flopping: never flip to the opposite side within the same 1m candle bucket.
        try:
            if self._last_exit_bucket_utc is not None and self._last_exit_side in ("LONG", "SHORT"):
                if intent.side != self._last_exit_side:
                    curr_bucket_utc = intent.ts.astimezone(timezone.utc).replace(second=0, microsecond=0)
                    if curr_bucket_utc == self._last_exit_bucket_utc:
                        _emit_hold("flip_cooldown_same_candle")
                        return
        except Exception:
            pass

        # Contextual entry veto (late breakout + flow divergence + streak climax).
        try:
            veto = self._entry_context_veto_locked(intent, extra)
            if veto:
                _emit_hold(str(veto))
                _emit_entry_veto(str(veto))
                return
        except Exception as e:
            # Fail closed for scalper safety when gating logic errors out.
            _emit_hold("veto_check_error", error=str(e)[:120])
            _emit_entry_veto("veto_check_error", error=str(e)[:120])
            return

        # Reversal-proof: if we just exited and the next entry is opposite, require evidence.
        if self._last_exit_side and self._last_exit_px is not None and intent.side != self._last_exit_side:
            atr = float(extra.get("atr", self._MICRO_ATR_FALLBACK) or self._MICRO_ATR_FALLBACK)
            atr = max(self._TICK_SIZE, atr)
            px = float(intent.entry_px)
            move_atr = abs(px - float(self._last_exit_px)) / max(_EPS, atr)

            # Anchor-cross proof (optional): don't reverse while still on the wrong side of the anchor.
            anchor_dist = float(extra.get("anchor_dist_atr_signed", 0.0) or 0.0)
            anchor_ok = True
            if bool(getattr(self.cfg, "reverse_require_anchor_cross", True)):
                if intent.side == "LONG":
                    anchor_ok = anchor_dist > 0.10
                else:
                    anchor_ok = anchor_dist < -0.10

            # Setup margin must be strong enough to flip (market-driven hysteresis).
            setup_strong = abs(float(self._setup_margin)) >= float(self.cfg.flip_margin)

            if (move_atr < float(self.cfg.reverse_min_move_atr)) or (not anchor_ok) or (not setup_strong):
                _emit_hold(
                    "reversal_not_proven",
                    move_atr=float(move_atr),
                    need_move_atr=float(self.cfg.reverse_min_move_atr),
                    anchor_ok=bool(anchor_ok),
                    setup_margin=float(self._setup_margin),
                    setup_side=self._setup_side,
                    blocked_entry_side=intent.side,
                )
                return
        out = self._execute_entry(
            intent.side,
            extra,
            reason=intent.reason,
            tick_ts=intent.ts,
            tick_px=float(intent.entry_px),
            engine=intent.engine,
            sl_hint=intent.sl_hint,
        )
        if not _is_entry_suggestion(str((out or {}).get("suggestion") or "")):
            if isinstance(out, dict):
                self._write_signal(out, extra)
            return
        try:
            self._last_entry_side = intent.side
            if isinstance(intent.extra, dict) and intent.extra.get("recv_ms") is not None:
                self._last_entry_ms = int(intent.extra.get("recv_ms", self._last_entry_ms))
            else:
                self._last_entry_ms = int(intent.ts.timestamp() * 1000)
        except Exception:
            pass

        # Post-commit cleanup to prevent immediate double entries
        if intent.engine == "micro":
            self._armed = None
            self._ema915_armed = None
        elif intent.engine == "ema915":
            self._ema915_armed = None
            self._armed = None

    def _mark_rearm_required(self, exited_engine: str, *, exit_ms: int) -> None:
        for eng in ("micro", "ema915"):
            self._rearm_required[eng] = True
            self._rearm_epoch_ms[eng] = int(exit_ms)
        self._armed = None
        self._ema915_armed = None

    def _mk_tick_metrics(
        self,
        m0: Dict[str, Any],
        ltp: float,
        ts: datetime,
        *,
        recv_ms: Optional[int] = None,
        recv_ns: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Merge slow-truth metrics with live tick edge for immediate emission."""
        out = dict(m0 or {})
        out["px"] = float(ltp)
        out["ts_utc"] = ts.astimezone(timezone.utc).isoformat()
        out["ts_ist"] = _iso_ist(ts)
        candle_bucket = ts.replace(second=0, microsecond=0)
        out["_candle_ts"] = candle_bucket
        out["_candle_ts_utc"] = candle_bucket.astimezone(timezone.utc).isoformat()
        out["_candle_ts_ist"] = _iso_ist(candle_bucket)
        out["candle_bucket_id"] = out["_candle_ts_utc"]

        fut, fut_age = self._get_fut_flow_snapshot(ts)
        if fut is not None:
            out["fut_flow"] = fut
            out["fut_flow_status"] = "OK"
            if fut_age is not None:
                out["fut_flow_age_sec"] = float(fut_age)
        elif bool(getattr(self.cfg, "use_fut_flow", False)):
            if fut_age is None:
                out["fut_flow_status"] = "MISSING"
            elif fut_age < 0:
                out["fut_flow_status"] = "FUTURE_TS"
            else:
                out["fut_flow_status"] = "STALE"
        if recv_ms is not None:
            out["recv_ms"] = int(recv_ms)
            if "latency_ms" not in out:
                try:
                    tick_ms = int(ts.astimezone(timezone.utc).timestamp() * 1000)
                    out["latency_ms"] = int(recv_ms) - tick_ms
                except Exception:
                    pass
        if recv_ns is not None:
            out["recv_ns"] = int(recv_ns)

        if not out.get("regime"):
            out["regime"] = str(getattr(self, "_last_diag", {}).get("regime", "UNKNOWN") or "UNKNOWN")
        try:
            if ("pa_mode" not in out) or (out.get("pa_strength") is None):
                if isinstance(getattr(self, "_last_pa", None), dict):
                    out.update(getattr(self, "_last_pa"))
        except Exception:
            pass
        try:
            if "breakout_energy" not in out:
                bs = getattr(self, "_breakout_state", None)
                if bs is not None:
                    out["breakout_side"] = bs.side
                    out["breakout_level"] = bs.level
                    out["breakout_energy"] = bs.energy
                    out["breakout_origin"] = bs.origin
        except Exception:
            pass
        if not out.get("setup_quality") or out.get("confidence") is None:
            q, c = self._quality_and_confidence(out)
            out.setdefault("setup_quality", q)
            out.setdefault("confidence", c)
        return out

    def _get_fut_flow_snapshot(self, ts: datetime) -> Tuple[Optional[Dict[str, Any]], Optional[float]]:
        """Return the most recent futures-flow snapshot for the given tick timestamp.

        The returned snapshot is stale-checked using cfg.fut_flow_stale_sec. This lets tick-level logic
        reason about microstructure (depth_imb, buy/sell qty, CVD delta) without re-reading the sidecar.
        """
        if not bool(getattr(self.cfg, "use_fut_flow", False)):
            return None, None
        fut = getattr(self, "_last_fut_flow_snapshot", None)
        fut_ts = getattr(self, "_last_fut_flow_ts_utc", None)
        if fut is None or fut_ts is None:
            return None, None

        try:
            ts_utc = ts.astimezone(timezone.utc)
        except Exception:
            return None, None
        try:
            age_sec = (ts_utc - fut_ts).total_seconds()
        except Exception:
            return None, None

        if age_sec < 0:
            return None, age_sec
        if age_sec > float(getattr(self.cfg, "fut_flow_stale_sec", 180)):
            return None, age_sec
        return dict(fut), age_sec

    def _micro_diag_fallback_locked(self) -> Dict[str, Any]:
        """Synthesize minimal diag metrics if slow engine isn't ready."""
        if self._last_diag:
            return dict(self._last_diag)

        atr = self._MICRO_ATR_FALLBACK
        if len(self.candles) >= 2:
            trs = []
            start = max(1, len(self.candles) - 6)
            for i in range(start, len(self.candles)):
                c = self.candles[i]
                p = self.candles[i - 1]
                hi = float(c.get("high", 0.0))
                lo = float(c.get("low", 0.0))
                pc = float(p.get("close", 0.0))
                tr = max(hi - lo, abs(hi - pc), abs(lo - pc))
                if tr > 0:
                    trs.append(tr)
            if trs:
                atr = float(np.median(trs))

        return {
            "atr": max(atr, 20.0 * 0.05),
            "bias": "NONE",
            "squeeze": False,
            "path_eff": 0.0,
            "regime": "UNKNOWN",
            "setup_quality": "C",
            "confidence": 0.0,
        }

    def _micro_vel_acc_locked(self, ms: int, ltp: float) -> Optional[Tuple[float, float]]:
        """Compute vel/acc using arrival time (ms)."""
        self._tbuf.append((int(ms), float(ltp)))
        if len(self._tbuf) < 3:
            return None

        target_min, target_max = 0.4, 0.8
        fallback_max = 5.0

        best = None
        best_err = 1e9

        for ms0, p0 in self._tbuf:
            dt = (ms - ms0) / 1000.0
            if dt <= 0:
                continue
            if target_min <= dt <= target_max:
                err = abs(dt - 0.6)
                if err < best_err:
                    best = (dt, p0)
                    best_err = err

        if best is None:
            for ms0, p0 in self._tbuf:
                dt = (ms - ms0) / 1000.0
                if 0 < dt <= fallback_max:
                    best = (dt, p0)
                    break

        if best is None:
            return None

        dt, p0 = best
        vel = (ltp - p0) / max(_EPS, dt)

        prev_ms = self._last_vel_ms
        dt_eval = (ms - prev_ms) / 1000.0 if isinstance(prev_ms, int) else (self._MS_EVAL_EVERY / 1000.0)
        dt_eval = max(0.001, float(dt_eval))
        acc = (vel - self._last_vel) / dt_eval
        self._last_vel = vel
        self._last_vel_ms = ms

        return vel, acc

    def _micro_intent_locked(
        self,
        ltp: float,
        ts: datetime,
        *,
        recv_ms: int,
        recv_ns: int,
        m0: Dict[str, Any],
        prev: Dict[str, Any],
    ) -> Optional[EntryIntent]:
        _ = recv_ns
        if self.pos:
            return None

        if self._last_ms_eval is not None and (recv_ms - self._last_ms_eval) < self._MS_EVAL_EVERY:
            return None
        self._last_ms_eval = recv_ms

        diag = m0 or self._micro_diag_fallback_locked()
        atr = float(diag.get("atr", self._MICRO_ATR_FALLBACK))

        out = self._micro_vel_acc_locked(recv_ms, ltp)
        if not out:
            return None
        vel, acc = out

        atr_per_sec = atr / 60.0
        vel_z = vel / max(_EPS, atr_per_sec)
        acc_z = acc / max(_EPS, atr_per_sec)
        clip = float(getattr(self.cfg, "micro_z_clip", 10.0))
        vel_z = max(-clip, min(clip, vel_z))
        acc_z = max(-clip, min(clip, acc_z))

        def _accept_ok(side: str) -> bool:
            min_v = float(self.cfg.micro_accept_vel_z)
            max_against = float(self.cfg.micro_accept_acc_z_max_against)
            if side == "LONG":
                return not (vel_z < min_v or acc_z < -max_against)
            return not (vel_z > -min_v or acc_z > max_against)

        prev_h = float(prev.get("high", ltp))
        prev_l = float(prev.get("low", ltp))
        buffer = max(float(self.cfg.micro_arm_buffer_atr) * atr, float(self._TICK_SIZE) * float(self.cfg.micro_arm_buffer_ticks))

        if self._armed is None:
            if vel_z > 2.5 and acc_z > 0.0 and (prev_h - ltp) <= buffer:
                if self._shock_lock_side and recv_ms < self._shock_lock_until_ms and self._shock_lock_side != "LONG":
                    return None
                self._armed = {"mode": "BREAK", "side": "LONG", "lvl": prev_h, "ms": recv_ms}
                if self._rearm_required["micro"] and recv_ms > self._rearm_epoch_ms["micro"]:
                    self._rearm_required["micro"] = False
                self._write_signal({"suggestion": "ARM_LONG", "reason": "micro_arm"}, self._mk_tick_metrics(diag, ltp, ts, recv_ms=recv_ms))
                return None
            if vel_z < -2.5 and acc_z < 0.0 and (ltp - prev_l) <= buffer:
                if self._shock_lock_side and recv_ms < self._shock_lock_until_ms and self._shock_lock_side != "SHORT":
                    return None
                self._armed = {"mode": "BREAK", "side": "SHORT", "lvl": prev_l, "ms": recv_ms}
                if self._rearm_required["micro"] and recv_ms > self._rearm_epoch_ms["micro"]:
                    self._rearm_required["micro"] = False
                self._write_signal({"suggestion": "ARM_SHORT", "reason": "micro_arm"}, self._mk_tick_metrics(diag, ltp, ts, recv_ms=recv_ms))
                return None
            return None

        age_ms = float(recv_ms - int(self._armed.get("ms", recv_ms)))
        if age_ms > self._ARM_TIMEOUT_MS:
            self._armed = None
            return None

        lvl = float(self._armed["lvl"])
        req_hold = self._HOLD_FAST_MS if abs(vel_z) > 4.5 else self._HOLD_SLOW_MS

        if self._armed["mode"] == "BREAK":
            if self._armed["side"] == "LONG":
                if ltp > (lvl + buffer) and age_ms >= req_hold:
                    if self._rearm_required["micro"]:
                        return None
                    if self._shock_lock_side and recv_ms < self._shock_lock_until_ms and self._shock_lock_side != "LONG":
                        return None
                    if self._ema915_armed and self._ema915_armed.get("side") and self._ema915_armed["side"] != "LONG":
                        return None
                    if not _accept_ok("LONG"):
                        return None
                    extra = {"channel": "micro", "vel_z": vel_z, "acc_z": acc_z, "lvl": lvl, "recv_ms": recv_ms}
                    return EntryIntent(
                        engine="micro",
                        side="LONG",
                        entry_px=ltp,
                        ts=ts,
                        reason="proactive_break_accept",
                        score=9000.0 + abs(vel_z) * 10.0,
                        sl_hint=None,
                        extra=extra,
                    )

                if age_ms < req_hold and ltp < (lvl - 1.5 * buffer) and vel_z < -1.0:
                    self._armed = {"mode": "FAIL", "side": "SHORT", "lvl": lvl, "ms": recv_ms}
                    self._write_signal(
                        {"suggestion": "FAIL_LONG_TO_SHORT", "reason": "micro_fail_snapback"},
                        self._mk_tick_metrics(diag, ltp, ts, recv_ms=recv_ms),
                    )
                    return None

            else:
                if ltp < (lvl - buffer) and age_ms >= req_hold:
                    if self._rearm_required["micro"]:
                        return None
                    if self._shock_lock_side and recv_ms < self._shock_lock_until_ms and self._shock_lock_side != "SHORT":
                        return None
                    if self._ema915_armed and self._ema915_armed.get("side") and self._ema915_armed["side"] != "SHORT":
                        return None
                    if not _accept_ok("SHORT"):
                        return None
                    extra = {"channel": "micro", "vel_z": vel_z, "acc_z": acc_z, "lvl": lvl, "recv_ms": recv_ms}
                    return EntryIntent(
                        engine="micro",
                        side="SHORT",
                        entry_px=ltp,
                        ts=ts,
                        reason="proactive_break_accept",
                        score=9000.0 + abs(vel_z) * 10.0,
                        sl_hint=None,
                        extra=extra,
                    )

                if age_ms < req_hold and ltp > (lvl + 1.5 * buffer) and vel_z > 1.0:
                    self._armed = {"mode": "FAIL", "side": "LONG", "lvl": lvl, "ms": recv_ms}
                    self._write_signal(
                        {"suggestion": "FAIL_SHORT_TO_LONG", "reason": "micro_fail_snapback"},
                        self._mk_tick_metrics(diag, ltp, ts, recv_ms=recv_ms),
                    )
                    return None

        if self._armed["mode"] == "FAIL":
            min_fail_hold = 100
            fail_age = age_ms

            if self._armed["side"] == "SHORT":
                if vel_z < -1.5 and acc_z < 0 and fail_age >= min_fail_hold and ltp < (lvl - 0.75 * buffer):
                    if self._rearm_required["micro"]:
                        return None
                    if self._shock_lock_side and recv_ms < self._shock_lock_until_ms and self._shock_lock_side != "SHORT":
                        return None
                    if self._ema915_armed and self._ema915_armed.get("side") and self._ema915_armed["side"] != "SHORT":
                        return None
                    if not _accept_ok("SHORT"):
                        return None
                    extra = {"channel": "micro", "vel_z": vel_z, "acc_z": acc_z, "lvl": lvl, "recv_ms": recv_ms}
                    return EntryIntent(
                        engine="micro",
                        side="SHORT",
                        entry_px=ltp,
                        ts=ts,
                        reason="proactive_trap_reversal",
                        score=9000.0 + abs(vel_z) * 10.0,
                        sl_hint=None,
                        extra=extra,
                    )
            else:
                if vel_z > 1.5 and acc_z > 0 and fail_age >= min_fail_hold and ltp > (lvl + 0.75 * buffer):
                    if self._rearm_required["micro"]:
                        return None
                    if self._shock_lock_side and recv_ms < self._shock_lock_until_ms and self._shock_lock_side != "LONG":
                        return None
                    if self._ema915_armed and self._ema915_armed.get("side") and self._ema915_armed["side"] != "LONG":
                        return None
                    if not _accept_ok("LONG"):
                        return None
                    extra = {"channel": "micro", "vel_z": vel_z, "acc_z": acc_z, "lvl": lvl, "recv_ms": recv_ms}
                    return EntryIntent(
                        engine="micro",
                        side="LONG",
                        entry_px=ltp,
                        ts=ts,
                        reason="proactive_trap_reversal",
                        score=9000.0 + abs(vel_z) * 10.0,
                        sl_hint=None,
                        extra=extra,
                    )

        return None

    def _ema_update(self, prev: Optional[float], x: float, period: int) -> float:
        a = 2.0 / (period + 1.0)
        return x if prev is None else (a * x + (1.0 - a) * prev)

    def _ema915_atr_per_bar_locked(self, m0: Dict[str, Any]) -> float:
        atr = float(m0.get("atr", 0.0) or 0.0)
        sec = max(0.001, float(self._ema915_bar_ms) / 1000.0)
        if atr > 0:
            return max(self._TICK_SIZE, (atr / 60.0) * sec)

        if len(self._ema915_bars) >= 20:
            rs = [float(b["high"] - b["low"]) for b in list(self._ema915_bars)[-20:]]
            rs.sort()
            return max(self._TICK_SIZE, rs[len(rs) // 2])
        return max(self._TICK_SIZE, (self._MICRO_ATR_FALLBACK / 60.0) * sec)

    def _ema915_build_bar_locked(self, ltp: float, ts: datetime, recv_ms: int) -> Optional[Dict[str, Any]]:
        bucket = (recv_ms // self._ema915_bar_ms) * self._ema915_bar_ms

        if self._ema915_curbar is None:
            self._ema915_curbar = {"bucket": bucket, "ts": ts, "open": ltp, "high": ltp, "low": ltp, "close": ltp}
            return None

        cb = self._ema915_curbar
        if bucket == cb["bucket"]:
            cb["high"] = max(float(cb["high"]), ltp)
            cb["low"] = min(float(cb["low"]), ltp)
            cb["close"] = ltp
            cb["ts"] = ts
            return None

        closed = dict(cb)
        self._ema915_bars.append(closed)
        self._ema915_curbar = {"bucket": bucket, "ts": ts, "open": ltp, "high": ltp, "low": ltp, "close": ltp}
        return closed

    def _ema915_strong_candle_locked(self, bar: Dict[str, Any]) -> Dict[str, Union[bool, float]]:
        o = float(bar["open"])
        h = float(bar["high"])
        l = float(bar["low"])
        c = float(bar["close"])
        rng = max(self._TICK_SIZE, h - l)
        body = abs(c - o)
        body_pct = body / rng
        up_w = h - max(o, c)
        dn_w = min(o, c) - l
        up_pct = up_w / rng
        dn_pct = dn_w / rng

        bullish = c > o
        bearish = c < o
        big_body = body_pct >= 0.60
        pin_bull = bullish and dn_pct >= 0.55
        pin_bear = bearish and up_pct >= 0.55

        return {"bull": bullish and (big_body or pin_bull), "bear": bearish and (big_body or pin_bear), "body_pct": body_pct}

    def _ema915_angles_locked(self, atr_per_bar: float) -> Optional[Dict[str, float]]:
        lb = self._ema915_slope_lb
        if len(self._ema9_hist) <= lb or len(self._ema15_hist) <= lb:
            return None

        e9 = float(self._ema9_hist[-1])
        e9p = float(self._ema9_hist[-1 - lb])
        e15 = float(self._ema15_hist[-1])
        e15p = float(self._ema15_hist[-1 - lb])

        s9 = (e9 - e9p) / max(1, lb)
        s15 = (e15 - e15p) / max(1, lb)

        s9n = s9 / max(_EPS, atr_per_bar)
        s15n = s15 / max(_EPS, atr_per_bar)

        a9 = math.degrees(math.atan(abs(s9n)))
        a15 = math.degrees(math.atan(abs(s15n)))

        return {"a9": a9, "a15": a15, "s9": s9, "s15": s15}

    def _ema915_intent_locked(
        self,
        ltp: float,
        ts: datetime,
        *,
        recv_ms: int,
        recv_ns: int,
        m0: Dict[str, Any],
        prev: Dict[str, Any],
    ) -> Optional[EntryIntent]:
        _ = (prev, recv_ns)
        if recv_ms - int(self._ema915_last_eval_ms) < self._ema915_eval_every_ms:
            return None
        self._ema915_last_eval_ms = recv_ms

        closed = self._ema915_build_bar_locked(ltp, ts, recv_ms)
        if closed is not None:
            close_px = float(closed["close"])
            self._ema9 = self._ema_update(self._ema9, close_px, 9)
            self._ema15 = self._ema_update(self._ema15, close_px, 15)
            self._ema9_hist.append(float(self._ema9))
            self._ema15_hist.append(float(self._ema15))

            atr_per_bar = self._ema915_atr_per_bar_locked(m0)
            ang = self._ema915_angles_locked(atr_per_bar)
            if ang is None:
                return None

            a9 = ang["a9"]
            a15 = ang["a15"]
            if a9 > self._ema915_max_angle or a15 > self._ema915_max_angle:
                self._ema915_armed = None
                return None

            e9 = float(self._ema9)
            e15 = float(self._ema15)
            sig = self._ema915_strong_candle_locked(closed)

            o = float(closed["open"])
            h = float(closed["high"])
            l = float(closed["low"])
            pad = self._ema915_touch_pad
            touch = (l - pad) <= e9 <= (h + pad) or (l - pad) <= e15 <= (h + pad)

            if e9 > e15 and ang["s9"] > 0 and ang["s15"] > 0 and a9 >= self._ema915_min_angle and a15 >= self._ema915_min_angle and sig["bull"] and touch:
                if self._shock_lock_side and recv_ms < self._shock_lock_until_ms and self._shock_lock_side != "LONG":
                    return None
                self._ema915_armed = {"side": "LONG", "lvl": h, "sl": l, "armed_ms": recv_ms}
                if self._rearm_required["ema915"] and recv_ms > self._rearm_epoch_ms["ema915"]:
                    self._rearm_required["ema915"] = False
                self._write_signal({"suggestion": "EMA915_ARM_LONG", "reason": "ema915_arm"}, self._mk_tick_metrics(m0, ltp, ts, recv_ms=recv_ms))
            elif e9 < e15 and ang["s9"] < 0 and ang["s15"] < 0 and a9 >= self._ema915_min_angle and a15 >= self._ema915_min_angle and sig["bear"] and touch:
                if self._shock_lock_side and recv_ms < self._shock_lock_until_ms and self._shock_lock_side != "SHORT":
                    return None
                self._ema915_armed = {"side": "SHORT", "lvl": l, "sl": h, "armed_ms": recv_ms}
                if self._rearm_required["ema915"] and recv_ms > self._rearm_epoch_ms["ema915"]:
                    self._rearm_required["ema915"] = False
                self._write_signal({"suggestion": "EMA915_ARM_SHORT", "reason": "ema915_arm"}, self._mk_tick_metrics(m0, ltp, ts, recv_ms=recv_ms))

        if self._ema915_armed is None:
            return None

        age = recv_ms - int(self._ema915_armed["armed_ms"])
        if age < self._ema915_arm_min_age_ms:
            return None
        if age > self._ema915_arm_timeout_ms:
            self._ema915_armed = None
            return None

        side = str(self._ema915_armed["side"])
        lvl = float(self._ema915_armed["lvl"])
        sl_hint = float(self._ema915_armed["sl"])
        atr_for_break = max(self._TICK_SIZE, float(m0.get("atr", self._MICRO_ATR_FALLBACK) or self._MICRO_ATR_FALLBACK))
        break_buffer_pts = max(
            float(self._TICK_SIZE) * float(getattr(self.cfg, "ema915_break_buffer_ticks", 0) or 0),
            float(getattr(self.cfg, "ema915_break_buffer_atr", 0.0) or 0.0) * atr_for_break,
        )

        if side == "LONG" and ltp >= (lvl + break_buffer_pts):
            if self._rearm_required["ema915"]:
                return None
            if self._shock_lock_side and recv_ms < self._shock_lock_until_ms and self._shock_lock_side != "LONG":
                return None
            extra = {
                "channel": "ema915",
                "ema9": float(self._ema9 or 0.0),
                "ema15": float(self._ema15 or 0.0),
                "recv_ms": recv_ms,
                "break_buffer_pts": break_buffer_pts,
            }
            return EntryIntent(
                engine="ema915",
                side="LONG",
                entry_px=ltp,
                ts=ts,
                reason="ema915_break",
                score=8000.0 + age,
                sl_hint=sl_hint,
                extra=extra,
            )

        if side == "SHORT" and ltp <= (lvl - break_buffer_pts):
            if self._rearm_required["ema915"]:
                return None
            if self._shock_lock_side and recv_ms < self._shock_lock_until_ms and self._shock_lock_side != "SHORT":
                return None
            extra = {
                "channel": "ema915",
                "ema9": float(self._ema9 or 0.0),
                "ema15": float(self._ema15 or 0.0),
                "recv_ms": recv_ms,
                "break_buffer_pts": break_buffer_pts,
            }
            return EntryIntent(
                engine="ema915",
                side="SHORT",
                entry_px=ltp,
                ts=ts,
                reason="ema915_break",
                score=8000.0 + age,
                sl_hint=sl_hint,
                extra=extra,
            )

        return None

    def _check_intra_tick_exits(self, ltp: float, ts: datetime, *, recv_ms: int) -> Optional[Dict[str, Any]]:
        """Tick-level hard stop, SL/TP fills, BE + trailing."""
        assert self._lock.locked(), "_check_intra_tick_exits must run under self._lock"
        p = self.pos
        if p is None:
            return None

        def _sl_exit_suggestion() -> str:
            try:
                if getattr(p, "sl_init", None) is not None and float(p.sl) == float(p.sl_init):
                    return "EXIT_INIT_SL"
                if p.is_be and abs(float(p.sl) - float(p.entry_px)) <= float(self.cfg.be_buffer_points) * 2.0:
                    return "EXIT_BE"
            except Exception:
                pass
            return "EXIT_TRAIL"

        def _do_exit(suggestion: str, px: float, reason: str = "") -> Dict[str, Any]:
            _ = recv_ms
            return self._execute_exit(suggestion, px=px, ts=ts, reason=reason or suggestion)

        def _stall_exit_reason() -> Optional[str]:
            try:
                stall_bars = int(getattr(self.cfg, "stall_bars", 0) or 0)
                if stall_bars <= 0:
                    return None
                min_mfe_atr = float(getattr(self.cfg, "stall_min_mfe_atr", 0.0) or 0.0)
                entry_bucket = getattr(p, "entry_bucket_utc", None)
                if not isinstance(entry_bucket, datetime):
                    entry_bucket = p.entry_ts.astimezone(timezone.utc).replace(second=0, microsecond=0)
                curr_bucket = ts.astimezone(timezone.utc).replace(second=0, microsecond=0)
                elapsed_bars = int((curr_bucket - entry_bucket).total_seconds() // 60)
                if elapsed_bars < stall_bars:
                    return None

                if p.side == "LONG":
                    mfe_pts = float(p.best_px - p.entry_px)
                else:
                    mfe_pts = float(p.entry_px - p.best_px)
                mfe_atr = float(mfe_pts / max(_EPS, atr))
                if mfe_atr < min_mfe_atr:
                    return f"stall_kill_{elapsed_bars}bars_mfeatr_{mfe_atr:.2f}"
            except Exception:
                return None
            return None

        # Time-based decay exit (scalper TTL)
        max_hold_s = int(getattr(self.cfg, "max_hold_seconds", 15 * 60))
        entry_ts = getattr(p, "entry_ts", None)
        if isinstance(entry_ts, datetime) and max_hold_s > 0:
            if (ts - entry_ts).total_seconds() >= max_hold_s:
                return _do_exit("EXIT_TIME", float(ltp), reason="max_hold_time")

        atr = float(self._last_diag.get("atr", 0.0))
        if not _is_finite_pos(atr):
            atr = 10.0  # safe fallback; avoids div-by-zero and NaN propagation

        def _should_debounce_trail_hit(side: str, ltp_now: float) -> bool:
            """Ignore tiny trail clips right after the trail first arms.

            Hard stop + init SL + BE exits are never debounced.
            """
            try:
                if not bool(getattr(p, "trail_armed", False)):
                    return False
                arm_ts = getattr(p, "trail_arm_ts", None)
                if not isinstance(arm_ts, datetime):
                    return False
                debounce_ms = int(getattr(self.cfg, "trail_debounce_ms", 0) or 0)
                if debounce_ms <= 0:
                    return False
                age_ms = int((ts - arm_ts).total_seconds() * 1000.0)
                if age_ms >= debounce_ms:
                    return False
                # Only debounce very small breaches (avoid ignoring real reversals)
                tol = 0.15 * float(atr)
                if side == "LONG":
                    return (float(p.sl) - float(ltp_now)) <= tol
                return (float(ltp_now) - float(p.sl)) <= tol
            except Exception:
                return False

        def _trail_profile() -> Tuple[float, Optional[float]]:
            """Return (arm_points, trail_atr_dist). None trail_atr_dist means trailing disabled."""
            try:
                if bool(getattr(p, "is_harvest", False)):
                    trail_arm_pts = float(getattr(p, "trail_arm_points", 0.0) or 0.0)
                    if trail_arm_pts <= 0.0:
                        trail_arm_pts = max(0.0, float(getattr(self.cfg, "trail_start_atr", 0.0) or 0.0) * atr)
                    return trail_arm_pts, float(getattr(self.cfg, "harvest_trail_atr", 0.5) or 0.5)
                if bool(getattr(self.cfg, "regime_trailing_enable", False)):
                    reg = str(getattr(p, "regime", "") or self._last_diag.get("regime", "UNKNOWN") or "UNKNOWN")
                    if reg == "CHOP" and bool(getattr(self.cfg, "trail_disable_in_chop", True)):
                        return 0.0, None
                    if reg == "BREAKOUT_WINDOW":
                        return (
                            max(0.0, float(getattr(self.cfg, "trail_breakout_start_atr", 0.9) or 0.9) * atr),
                            float(getattr(self.cfg, "trail_breakout_dist_atr", 0.9) or 0.9),
                        )
                    if reg == "TREND":
                        return (
                            max(0.0, float(getattr(self.cfg, "trail_trend_start_atr", 0.7) or 0.7) * atr),
                            float(getattr(self.cfg, "trail_trend_dist_atr", 0.7) or 0.7),
                        )
                trail_arm_pts = float(getattr(p, "trail_arm_points", 0.0) or 0.0)
                if trail_arm_pts <= 0.0:
                    trail_arm_pts = max(0.0, float(getattr(self.cfg, "trail_start_atr", 0.0) or 0.0) * atr)
                trail_dist = float(getattr(p, "trail_atr_current", 0.0) or 0.0)
                if trail_dist <= 0.0:
                    trail_dist = float(getattr(self.cfg, "trail_atr", 2.0) or 2.0)
                return trail_arm_pts, trail_dist
            except Exception:
                return 0.0, float(getattr(self.cfg, "trail_atr", 2.0) or 2.0)

        # Update best price for trailing
        if p.side == "LONG":
            p.best_px = max(p.best_px, ltp)

            # Hard stop always enforced
            if ltp <= p.hard_sl:
                return _do_exit("EXIT_HARD_STOP", p.hard_sl)

            # SL
            if ltp <= p.sl:
                sug = _sl_exit_suggestion()
                if sug == "EXIT_TRAIL" and _should_debounce_trail_hit("LONG", float(ltp)):
                    return None
                return _do_exit(sug, p.sl)

            # TP (if not runner)
            if p.tp is not None and ltp >= p.tp:
                return _do_exit("EXIT_TP", p.tp)

            stall_reason = _stall_exit_reason()
            if stall_reason is not None:
                return _do_exit("EXIT_STALL", float(ltp), reason=stall_reason)

            # Move to BE
            be_trigger_pts = float(getattr(p, "be_trigger_points", 0.0) or 0.0)
            if be_trigger_pts <= 0.0:
                be_trigger_pts = float(self.cfg.move_to_be_atr) * atr
            if (not p.is_be) and ((p.best_px - p.entry_px) >= be_trigger_pts):
                be_raw = p.entry_px + self.cfg.be_buffer_points
                be_sl = _snap_price(be_raw, self._TICK_SIZE, kind="stop_long")
                p.sl = max(p.sl, be_sl)
                p.is_be = True

            # Trailing (arm only after a proven move)
            try:
                if not bool(getattr(p, "trail_armed", False)):
                    trail_arm_pts, trail_dist_atr = _trail_profile()
                    if trail_dist_atr is None:
                        trail_arm_pts = 0.0
                    if trail_arm_pts > 0 and ((p.best_px - p.entry_px) >= trail_arm_pts):
                        p.trail_armed = True
                        p.trail_arm_ts = ts
            except Exception:
                pass

            if bool(getattr(p, "trail_armed", False)):
                _, trail_dist_atr = _trail_profile()
                if trail_dist_atr is not None:
                    trail_sl = _snap_price(p.best_px - (float(trail_dist_atr) * atr), self._TICK_SIZE, kind="stop_long")
                    p.sl = max(p.sl, trail_sl)

        else:  # SHORT
            p.best_px = min(p.best_px, ltp)

            if ltp >= p.hard_sl:
                return _do_exit("EXIT_HARD_STOP", p.hard_sl)

            if ltp >= p.sl:
                sug = _sl_exit_suggestion()
                if sug == "EXIT_TRAIL" and _should_debounce_trail_hit("SHORT", float(ltp)):
                    return None
                return _do_exit(sug, p.sl)

            if p.tp is not None and ltp <= p.tp:
                return _do_exit("EXIT_TP", p.tp)

            stall_reason = _stall_exit_reason()
            if stall_reason is not None:
                return _do_exit("EXIT_STALL", float(ltp), reason=stall_reason)

            be_trigger_pts = float(getattr(p, "be_trigger_points", 0.0) or 0.0)
            if be_trigger_pts <= 0.0:
                be_trigger_pts = float(self.cfg.move_to_be_atr) * atr
            if (not p.is_be) and ((p.entry_px - p.best_px) >= be_trigger_pts):
                be_raw = p.entry_px - self.cfg.be_buffer_points
                be_sl = _snap_price(be_raw, self._TICK_SIZE, kind="stop_short")
                p.sl = min(p.sl, be_sl)
                p.is_be = True

            # Trailing (arm only after a proven move)
            try:
                if not bool(getattr(p, "trail_armed", False)):
                    trail_arm_pts, trail_dist_atr = _trail_profile()
                    if trail_dist_atr is None:
                        trail_arm_pts = 0.0
                    if trail_arm_pts > 0 and ((p.entry_px - p.best_px) >= trail_arm_pts):
                        p.trail_armed = True
                        p.trail_arm_ts = ts
            except Exception:
                pass

            if bool(getattr(p, "trail_armed", False)):
                _, trail_dist_atr = _trail_profile()
                if trail_dist_atr is not None:
                    trail_sl = _snap_price(p.best_px + (float(trail_dist_atr) * atr), self._TICK_SIZE, kind="stop_short")
                    p.sl = min(p.sl, trail_sl)

        return None

    def _execute_exit(self, suggestion: str, *, px: float, ts: datetime, reason: str = "") -> Dict[str, Any]:
        p = self.pos
        exit_side = p.side if p is not None else None
        try:
            exit_ms = int(ts.astimezone(timezone.utc).timestamp() * 1000.0)
        except Exception:
            exit_ms = int(time.time() * 1000)
        payload = {
            "suggestion": suggestion,
            "state": "EXIT",
            "stream": "exit",
            "channel": "exit",
            "engine": (p.engine if p else "unknown"),
            "in_pos": False,
            "in_pos_before": bool(p is not None),
            "px": float(px),
            "ts": ts,
            "ts_utc": ts.astimezone(timezone.utc).isoformat(),
            "ts_ist": _iso_ist(ts),
            "exit_candle_bucket_id": ts.astimezone(timezone.utc).replace(second=0, microsecond=0).isoformat(),
            "reason": reason or suggestion,
            "pos_side": None,
            "pos_side_before": exit_side,
        }

        if p is not None:
            payload.update(
                {
                    "entry_px": p.entry_px,
                    "entry_atr": p.entry_atr,
                    "entry_ts_ist": _iso_ist(p.entry_ts),
                    "sl_init": getattr(p, "sl_init", p.sl),
                    "sl_curr": p.sl,
                    "sl_fill_px": float(px),
                    "sl": p.sl,
                    "tp": p.tp,
                    "best_px": p.best_px,
                    "is_be": p.is_be,
                    "is_runner": p.is_runner,
                    "why": p.why,
                    "be_trigger_points": p.be_trigger_points,
                    "trail_arm_points": p.trail_arm_points,
                    "rr_est": p.rr_est,
                    "regime": p.regime,
                    "setup_quality": p.setup_quality,
                    "confidence": p.confidence,
                    "candle_bucket_id": p.entry_bucket_utc.isoformat() if isinstance(p.entry_bucket_utc, datetime) else None,
                }
            )
            try:
                sl_points = abs(float(p.entry_px) - float(getattr(p, "sl_init", p.sl)))
                payload["sl_points"] = sl_points
                payload["sl_atr"] = float(sl_points / max(_EPS, float(p.entry_atr or 0.0)))
            except Exception:
                pass

        try:
            self.log.info(self._fmt_scalper_line(payload))
        except Exception:
            pass

        # Update reversal context + require re-arm after any exit
        self._last_exit_side = exit_side
        self._last_exit_px = float(px)
        self._last_exit_ms = int(exit_ms)
        try:
            self._last_exit_bucket_utc = ts.astimezone(timezone.utc).replace(second=0, microsecond=0)
        except Exception:
            self._last_exit_bucket_utc = None
        try:
            self._mark_rearm_required(payload.get("engine", "unknown"), exit_ms=int(exit_ms))
        except Exception:
            pass

        # Exit-type-aware cooldown memory.
        if exit_side in ("LONG", "SHORT"):
            try:
                bars_map = {
                    "EXIT_INIT_SL": int(getattr(self.cfg, "cooldown_exit_init_sl_bars", 1) or 1),
                    "EXIT_BE": int(getattr(self.cfg, "cooldown_exit_be_bars", 1) or 1),
                    "EXIT_HARD_STOP": int(getattr(self.cfg, "cooldown_exit_hard_stop_bars", 3) or 3),
                    "EXIT_TRAIL": int(getattr(self.cfg, "cooldown_exit_trail_bars", 0) or 0),
                }
                min_bars = max(0, int(bars_map.get(suggestion, 0)))
                if min_bars > 0:
                    atr_ref = float(self._last_diag.get("atr", 0.0) or 0.0)
                    if not _is_finite_pos(atr_ref):
                        atr_ref = float(getattr(p, "entry_atr", self._MICRO_ATR_FALLBACK) or self._MICRO_ATR_FALLBACK)
                    atr_ref = max(self._TICK_SIZE, atr_ref)
                    reclaim_pts = max(self._TICK_SIZE, float(self.cfg.cooldown_reclaim_atr) * atr_ref)
                    self._stopout_cooldown_by_side[exit_side] = {
                        "exit_reason": suggestion,
                        "exit_ts_utc": ts.astimezone(timezone.utc).isoformat(),
                        "exit_bucket_utc": ts.astimezone(timezone.utc).replace(second=0, microsecond=0),
                        "stopout_px": float(px),
                        "atr_ref": float(atr_ref),
                        "reclaim_points": float(reclaim_pts),
                        "min_bars": int(min_bars),
                    }
                else:
                    self._stopout_cooldown_by_side[exit_side] = None
            except Exception:
                self.log.exception("stopout_cooldown_set_failed")

            # Impulse exit anti-overtrade window (same-side re-entries only).
            try:
                if suggestion in ("EXIT_TRAIL", "EXIT_TP"):
                    win_min = max(0, int(getattr(self.cfg, "impulse_reentry_window_min", 0) or 0))
                    if win_min > 0:
                        self._impulse_reentry_guard_by_side[exit_side] = {
                            "start_utc": ts.astimezone(timezone.utc),
                            "window_end_utc": ts.astimezone(timezone.utc) + timedelta(minutes=win_min),
                            "count": 0,
                            "exit_reason": suggestion,
                        }
            except Exception:
                self.log.exception("impulse_reentry_set_failed")

        self.pos = None
        self._maybe_notify_payload(payload)
        return payload

    def _fmt_scalper_line(self, payload: Dict[str, Any]) -> str:
        sug = str(payload.get("suggestion", ""))
        eng = str(payload.get("engine", payload.get("channel", "")) or "")
        reason = str(payload.get("reason", "") or "")

        ts_ist = str(payload.get("ts_ist", "") or payload.get("entry_ts_ist", "") or "")
        hhmmss = ts_ist[11:19] if len(ts_ist) >= 19 else ts_ist

        side = payload.get("pos_side")
        if not side:
            side = payload.get("pos_side_before")
        if not side:
            side = "LONG" if "LONG" in sug else "SHORT" if "SHORT" in sug else ""

        px = payload.get("px")
        entry_px = payload.get("entry_px", px)

        sl = payload.get("sl")
        tp = payload.get("tp")

        sl_pts = None
        tp_pts = None
        try:
            if entry_px is not None and sl is not None:
                sl_pts = abs(float(entry_px) - float(sl))
            if tp is not None and entry_px is not None:
                tp_pts = abs(float(tp) - float(entry_px))
        except Exception:
            sl_pts = None
            tp_pts = None

        vel = payload.get("vel_z")
        acc = payload.get("acc_z")
        ema9 = payload.get("ema9")
        ema15 = payload.get("ema15")
        clv = payload.get("clv")
        wick_up = payload.get("upper_wick_pct")
        path_eff = payload.get("path_eff")

        parts = []
        parts.append(f"{hhmmss} {eng} {side} {reason}".strip())

        try:
            if vel is not None and acc is not None:
                parts.append(f"vel={float(vel):.2f} acc={float(acc):.2f}")
        except Exception:
            pass

        try:
            if ema9 is not None and ema15 is not None:
                parts.append(f"ema9={float(ema9):.2f} ema15={float(ema15):.2f}")
        except Exception:
            pass

        if sl_pts is not None:
            parts.append(f"SL={sl_pts:.2f}")
        if tp_pts is not None:
            parts.append(f"TP={tp_pts:.2f}")

        minute = []
        try:
            if clv is not None:
                minute.append(f"clv={float(clv):.2f}")
            if wick_up is not None:
                minute.append(f"wick_up={float(wick_up):.2f}")
            if path_eff is not None:
                minute.append(f"path_eff={float(path_eff):.2f}")
        except Exception:
            minute = []

        line = " ".join(parts)
        if minute:
            line += " | minute: " + " ".join(minute)
        return line

    def _build_payload(self, decision: Dict[str, Any], metrics: Dict[str, Any]) -> Dict[str, Any]:
        payload: Dict[str, Any] = {}
        suggestion = decision.get("suggestion")
        if suggestion is not None:
            payload["suggestion"] = suggestion
        shock_side = metrics.get("shock_side")
        if shock_side is not None:
            payload["shock_side"] = shock_side
        if "squeeze" in metrics:
            payload["squeeze"] = metrics.get("squeeze")

        reserved = {"stream", "engine", "channel", "in_pos", "pos_side"}
        base_metrics = {k: v for k, v in (metrics or {}).items() if k not in reserved}
        base: Dict[str, Any] = {**base_metrics, **(decision or {})}
        for k, v in base.items():
            if k in payload:
                continue
            payload[k] = v
        if "state" not in payload:
            payload["state"] = _signal_state(str(payload.get("suggestion", "") or ""))
        if "regime" not in payload:
            payload["regime"] = str((metrics or {}).get("regime", "UNKNOWN") or "UNKNOWN")
        if "setup_quality" not in payload or payload.get("confidence") is None:
            q, c = self._quality_and_confidence(payload)
            payload.setdefault("setup_quality", q)
            payload.setdefault("confidence", c)
        payload.setdefault("candle_bucket_id", payload.get("_candle_ts_utc") or payload.get("_candle_ts"))
        return payload

    def _write_signal(self, decision: Dict[str, Any], metrics: Dict[str, Any]) -> None:
        """Write a single JSONL line for signals and (optionally) duplicate ARM to main."""
        try:
            sug0 = str(decision.get("suggestion", "") or "")
            if sug0 == "HOLD_WAIT_CONFIRM":
                rr = str(decision.get("reason", "") or "")
                important_wait = (
                    rr.startswith("stopout_cooldown")
                    or rr.startswith("regime_chop")
                    or rr.startswith("cooldown_wait")
                )
                if not important_wait:
                    return
            # Allow important HOLD_* even when write_hold/write_all are off (scalper needs these)
            important_holds = {"HOLD_CONFLICT", "HOLD_WEAK_EDGE", "HOLD_WAIT_CONFIRM"}
            if (not self.cfg.write_all) and sug0.startswith("HOLD") and sug0 not in important_holds:
                return
            if sug0 == "HOLD" and (not self.cfg.write_hold):
                return
            if sug0.startswith("ENTRY_") and sug0 != "ENTRY_VETO":
                return

            payload = self._build_payload(decision, metrics)
            sug = str(payload.get("suggestion", "") or "")
            if not sug:
                return
            is_arm = "ARM_" in sug
            stream_hint = str(decision.get("stream") or payload.get("stream") or "")
            is_exit = bool(stream_hint == "exit" or sug.startswith("EXIT_") or str(payload.get("channel") or "") == "exit")
            payload["stream"] = "arm" if is_arm else ("exit" if is_exit else "signal")

            if not str(payload.get("engine") or ""):
                if sug.startswith("EMA915_"):
                    payload["engine"] = "ema915"
                elif sug.startswith("ARM_"):
                    payload["engine"] = "micro"
                else:
                    payload["engine"] = str(metrics.get("engine") or decision.get("engine") or "candle")
            if is_exit:
                payload.setdefault("channel", "exit")
            else:
                payload.setdefault("channel", str(payload.get("engine") or ""))
            payload.setdefault("in_pos", bool(self.pos is not None))
            payload.setdefault("pos_side", (self.pos.side if self.pos else None))

            if is_arm:
                eng = str(payload.get("engine", metrics.get("engine", "")) or "")
                cts = payload.get("_candle_ts_utc", payload.get("_candle_ts", metrics.get("_candle_ts")))
                key = (eng, sug, str(cts))
                if key == self._arm_log_key:
                    return
                self._arm_log_key = key

                with self._arm_write_lock:
                    self.arm_jsonl_file.write(json.dumps(_isoize(payload), default=str) + "\n")
                    self.arm_jsonl_file.flush()

                if bool(getattr(self.cfg, "dup_arm_to_main", False)):
                    payload2 = dict(payload)
                    payload2["stream"] = "arm"
                    with self._write_lock:
                        self.jsonl_file.write(json.dumps(_isoize(payload2), default=str) + "\n")
                        self.jsonl_file.flush()
                self._maybe_notify_payload(payload)
            else:
                self._arm_log_key = None
                with self._write_lock:
                    self.jsonl_file.write(json.dumps(_isoize(payload), default=str) + "\n")
                    self.jsonl_file.flush()
                self._maybe_notify_payload(payload)
        except Exception:
            self.log.exception("_write_signal failed")



# ------------------------ Helpers ------------------------

def _tail_last_line(path: str, max_bytes: int = 8192) -> Optional[str]:
    """Return the last non-empty line from a text file. Safe for concurrent writers."""
    if not path:
        return None
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            if size <= 0:
                return None
            f.seek(max(0, size - max_bytes), os.SEEK_SET)
            data = f.read()
        lines = data.splitlines()
        for b in reversed(lines):
            if b and b.strip():
                try:
                    return b.decode("utf-8", errors="ignore")
                except Exception:
                    return None
        return None
    except Exception:
        return None


def _parse_fut_candle_row(line: str) -> Optional[Dict[str, Any]]:
    """Parse one CSV line from futures sidecar candle output (no header)."""
    if not line:
        return None
    parts = [p.strip() for p in line.split(",")]
    if len(parts) < 9:
        return None
    try:
        ts = datetime.fromisoformat(parts[0])
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
    except Exception:
        return None

    def _f(i: int) -> float:
        try:
            v = float(parts[i])
            return v if math.isfinite(v) else 0.0
        except Exception:
            return 0.0

    out: Dict[str, Any] = {
        "ts": ts,
        "o": _f(1),
        "h": _f(2),
        "l": _f(3),
        "c": _f(4),
        "vol": _f(5),
        "ticks": int(float(parts[6] or 0.0)),
        "vwap": _f(7),
        "cvd": _f(8),
    }
    # Optional extended columns appended by enhanced sidecar:
    # 9 sell_qty,10 buy_qty,11 oi,12 oi_high,13 oi_low,14 doi,15 depth_imb,16 spread,17 microprice
    if len(parts) > 9:
        out["sell_qty"] = _f(9)
    if len(parts) > 10:
        out["buy_qty"] = _f(10)
    if len(parts) > 11:
        out["oi"] = _f(11)
    if len(parts) > 12:
        out["oi_high"] = _f(12)
    if len(parts) > 13:
        out["oi_low"] = _f(13)
    if len(parts) > 14:
        out["doi"] = _f(14)
    if len(parts) > 15:
        out["depth_imb"] = _f(15)
    if len(parts) > 16:
        out["spread"] = _f(16)
    if len(parts) > 17:
        out["microprice"] = _f(17)
    return out

def _safe_float(v: Any) -> Optional[float]:
    try:
        if v is None:
            return None
        f = float(v)
        if math.isfinite(f) and f != 0.0:
            return f
        return None
    except Exception:
        return None


def _is_entry_suggestion(s: Optional[str]) -> bool:
    return isinstance(s, str) and s.startswith("ENTRY_") and s != "ENTRY_VETO"


def _signal_state(suggestion: str) -> str:
    s = str(suggestion or "")
    if s == "ENTRY_VETO":
        return "CANCEL"
    if s.startswith("ENTRY_"):
        return "TRIGGER"
    if s.startswith("EXIT_"):
        return "EXIT"
    if "ARM_" in s:
        return "ARM"
    if s.startswith("READY_"):
        return "READY"
    if s.startswith("HOLD") or s.startswith("NO_TRADE") or s.startswith("WATCH_") or s.startswith("FAIL_"):
        return "CANCEL"
    return "STATE"


def _parse_ts(x: Any) -> Optional[datetime]:
    """Accepts ISO strings, epoch seconds/ms, or datetime. Returns UTC-aware datetime."""
    if x is None:
        return None
    if isinstance(x, datetime):
        return x if x.tzinfo else x.replace(tzinfo=timezone.utc)

    if isinstance(x, (int, float)):
        try:
            v = float(x)
            if v > 1e12:
                v = v / 1000.0
            return datetime.fromtimestamp(v, tz=timezone.utc)
        except Exception:
            return None

    if isinstance(x, str):
        s = x.strip()
        if not s:
            return None
        if s.isdigit():
            try:
                v = float(s)
                if v > 1e12:
                    v = v / 1000.0
                return datetime.fromtimestamp(v, tz=timezone.utc)
            except Exception:
                return None
        try:
            if s.endswith("Z"):
                s = s[:-1] + "+00:00"
            dt = datetime.fromisoformat(s)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        except Exception:
            return None

    return None


def _fix_tick_time(
    tick_ts_utc: Optional[datetime],
    recv_ts_utc: datetime,
    *,
    drift_sec: int = 19800,
    drift_tol_sec: int = 120,
    max_abs_latency_ms: int = 300000,
) -> Tuple[Optional[datetime], Optional[int], str]:
    """
    Returns (tick_ts_fixed_utc, latency_ms, latency_status)

    - Corrects +05:30 drift when broker timestamp is IST wall time tagged as UTC.
    - latency_ms is guarded and set to None if absurd.
    """
    if tick_ts_utc is None:
        return None, None, "NO_TICK_TS"

    delta_sec = (tick_ts_utc - recv_ts_utc).total_seconds()

    status = "OK"
    fixed = tick_ts_utc

    if abs(delta_sec - drift_sec) <= drift_tol_sec:
        fixed = tick_ts_utc - timedelta(seconds=drift_sec)
        status = "DRIFT_CORRECTED_MINUS_5H30"
    elif abs(delta_sec + drift_sec) <= drift_tol_sec:
        fixed = tick_ts_utc + timedelta(seconds=drift_sec)
        status = "DRIFT_CORRECTED_PLUS_5H30"

    latency_ms = int((recv_ts_utc - fixed).total_seconds() * 1000)

    if abs(latency_ms) > max_abs_latency_ms:
        return fixed, None, "BAD_TS_DRIFT"

    return fixed, latency_ms, status


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _iso_ist(dt: datetime) -> str:
    """Return ISO-8601 timestamp in IST (+05:30)."""
    try:
        return dt.astimezone(IST_TZ).isoformat()
    except Exception:
        return dt.replace(tzinfo=timezone.utc).isoformat()


def _isoize(obj: Any) -> Any:
    """Recursively convert datetime objects to ISO-8601 strings (stable JSONL schema)."""
    try:
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, dict):
            return {k: _isoize(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_isoize(v) for v in obj]
        return obj
    except Exception:
        return obj


def _now_iso_ist() -> str:
    return _iso_ist(_now_utc())


def _is_finite_pos(x: float) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(float(x)) and float(x) > 0.0


def _snap_price(px: float, tick_size: float, *, kind: str = "nearest") -> float:
    """Snap price to tradable tick based on intent.

    Rules:
    - `stop_long`: round up (toward entry; tighter/more protective).
    - `stop_short`: round down (toward entry; tighter/more protective).
    - `tp_long`: round down (conservative take-profit).
    - `tp_short`: round up (conservative take-profit).
    - `nearest`: standard nearest-tick.
    """
    try:
        p = float(px)
        t = float(tick_size)
    except Exception:
        return px
    if not math.isfinite(p) or not math.isfinite(t) or t <= 0:
        return px

    q = p / t
    if kind == "stop_long":
        out = math.ceil(q - 1e-12) * t
    elif kind == "stop_short":
        out = math.floor(q + 1e-12) * t
    elif kind == "tp_long":
        out = math.floor(q + 1e-12) * t
    elif kind == "tp_short":
        out = math.ceil(q - 1e-12) * t
    else:
        out = round(q) * t
    return float(round(out, 10))


def _hma_series(series: pd.Series, period: int) -> pd.Series:
    """Hull Moving Average series. NaN-safe."""
    period = max(2, int(period))
    half = max(1, period // 2)
    sqrt_p = max(1, int(math.sqrt(period)))

    def _wma(s: pd.Series, p: int) -> pd.Series:
        def _wma_calc(x: np.ndarray) -> float:
            w = np.arange(1, len(x) + 1, dtype=float)
            return float(np.dot(x, w) / float(w.sum()))
        return s.rolling(p, min_periods=1).apply(_wma_calc, raw=True)

    wma_half = _wma(series, half)
    wma_full = _wma(series, period)
    diff = 2.0 * wma_half - wma_full
    return _wma(diff, sqrt_p)


def _update_bias(prev: Optional[str], slope: float, pos_th: float, neg_th: float) -> Optional[str]:
    """Hysteresis bias: switch only on threshold cross; otherwise keep."""
    if slope >= pos_th:
        return "LONG"
    if slope <= neg_th:
        return "SHORT"
    return prev


def _env(key: str, default: str, aliases: Optional[list[str]] = None) -> str:
    v = os.getenv(key)
    if v is not None and str(v).strip() != "":
        return str(v).strip()
    for a in (aliases or []):
        v2 = os.getenv(a)
        if v2 is not None and str(v2).strip() != "":
            return str(v2).strip()
    return default


def _env_bool(key: str, default: bool, aliases: Optional[list[str]] = None) -> bool:
    raw = _env(key, "", aliases).strip().lower()
    if raw == "":
        return default
    if raw in ("1", "true", "yes", "y", "on"):
        return True
    if raw in ("0", "false", "no", "n", "off"):
        return False
    return default


def _parse_level(x: str, default: int = logging.INFO) -> int:
    x = (x or "").strip().upper()
    if not x:
        return default
    if x.isdigit():
        try:
            return int(x)
        except Exception:
            return default
    return int(getattr(logging, x, default))
