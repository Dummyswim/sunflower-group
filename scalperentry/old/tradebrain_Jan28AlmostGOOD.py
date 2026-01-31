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
    min_path_eff_filter: float = float(os.getenv("TB_MIN_PATH_EFF_FILTER", "0.18"))  # stay flat if below
    entry_path_eff: float = float(os.getenv("TB_ENTRY_PATH_EFF", "0.70"))  # trade only if clean
    entry_anchor_min_atr: float = float(os.getenv("TB_ENTRY_ANCHOR_MIN_ATR", "0.55"))  # keep entries off anchor

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
    min_clv_confirm: float = float(os.getenv("TB_MIN_CLV_CONFIRM", "0.05"))  # momentum confirm
    strong_clv: float = float(os.getenv("TB_STRONG_CLV", "0.80"))  # clean sweep pattern
    min_body_pct: float = float(os.getenv("TB_MIN_BODY_PCT", "0.18"))  # avoid doji/indecision

    # Shock system
    shock_atr_mult: float = float(os.getenv("TB_SHOCK_ATR_MULT", "1.5"))
    shock_points: float = float(os.getenv("TB_SHOCK_POINTS", "35.0"))
    shock_confirm_candles: int = int(os.getenv("TB_SHOCK_CONFIRM_CANDLES", "3"))

    # Exits / Trade Mgmt
    hard_stop_points: float = float(os.getenv("TB_HARD_STOP_POINTS", "20.0"))  # intra-tick hard stop vs entry
    tp_atr_mult: float = float(os.getenv("TB_TP_ATR_MULT", "2.5"))  # fixed TP when not runner
    move_to_be_atr: float = float(os.getenv("TB_MOVE_TO_BE_ATR", "0.6"))
    be_buffer_points: float = float(os.getenv("TB_BE_BUFFER_POINTS", "1.0"))
    trail_atr: float = float(os.getenv("TB_TRAIL_ATR", "1.2"))
    harvest_trail_atr: float = float(os.getenv("TB_HARVEST_TRAIL_ATR", "0.5"))
    wick_threshold: float = float(os.getenv("TB_WICK_THRESHOLD", "0.40"))  # volatility-aware default

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

    # Optional: Futures sidecar (Option C). Read once per 1m candle close.
    use_fut_flow: bool = os.getenv("TB_USE_FUT_FLOW", "1") == "1"
    fut_sidecar_path: str = os.getenv("TB_FUT_SIDECAR_PATH", "data/fut_candles.csv")
    fut_flow_stale_sec: int = int(os.getenv("TB_FUT_FLOW_STALE_SEC", "180"))

    # Activity-weighted pressure guard (Option B). Prevent CLV from overconfident signals in low-activity tape.
    activity_w_low: float = float(os.getenv("TB_ACTIVITY_W_LOW", "0.70"))
    activity_w_high: float = float(os.getenv("TB_ACTIVITY_W_HIGH", "1.30"))
    min_activity_w_confirm: float = float(os.getenv("TB_MIN_ACTIVITY_W_CONFIRM", "0.35"))

    # Limits
    max_candles: int = int(os.getenv("TB_MAX_CANDLES", "240"))  # ~4 hours of 1m
    max_ticks_per_candle: int = int(os.getenv("TB_MAX_TICKS_PER_CANDLE", "20000"))  # loop safety
    min_ready_candles: int = int(os.getenv("TB_MIN_READY_CANDLES", "10"))

    # Entry guards
    respect_bias: bool = os.getenv("TB_RESPECT_BIAS", "0") == "1"

    # EMA915
    ema915_arm_min_age_ms: int = int(os.getenv("TB_EMA915_ARM_MIN_AGE_MS", "200"))

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
        if self.hard_stop_points <= 0:
            raise ValueError("hard_stop_points must be > 0")
        if self.tp_atr_mult <= 0:
            raise ValueError("tp_atr_mult must be > 0")
        if self.move_to_be_atr <= 0:
            raise ValueError("move_to_be_atr must be > 0")
        if self.trail_atr <= 0 or self.harvest_trail_atr <= 0:
            raise ValueError("trail_atr and harvest_trail_atr must be > 0")
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

        # Sidecar / activity validation
        if self.fut_flow_stale_sec < 1:
            raise ValueError("TB_FUT_FLOW_STALE_SEC must be >= 1")
        if self.activity_w_high <= self.activity_w_low:
            raise ValueError("TB_ACTIVITY_W_HIGH must be > TB_ACTIVITY_W_LOW")
        if not (0.0 <= self.min_activity_w_confirm <= 1.0):
            raise ValueError("TB_MIN_ACTIVITY_W_CONFIRM must be between 0 and 1")

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
    hard_sl: float
    sl: float
    sl_init: float
    tp: Optional[float]
    best_px: float
    is_be: bool = False
    is_runner: bool = False
    is_harvest: bool = False
    trail_atr_current: float = 0.0
    engine: str = "candle"
    why: str = ""


@dataclass
class PendingShock:
    side: str  # "LONG" or "SHORT"
    mid_px: float
    remaining_candles: int
    expires_at: Optional[datetime] = None


class TradeBrain:
    def __init__(self, cfg: TradeBrainConfig, log: logging.Logger):
        cfg.validate()
        self.cfg = cfg
        self.log = log

        self._lock = threading.Lock()
        self.candles: Deque[Dict[str, Any]] = deque(maxlen=self.cfg.max_candles)
        self.current_candle: Optional[Dict[str, Any]] = None

        self.pos: Optional[Position] = None
        self.bias: Optional[str] = None  # "LONG" or "SHORT" or None
        self.pending_shock: Optional[PendingShock] = None
        self._last_fut_cvd: float = 0.0

        # Market-driven setup memory (prevents flip-flops)
        self._setup_side: Optional[str] = None  # "LONG" / "SHORT" / None
        self._setup_margin: float = 0.0  # last computed margin in [-1..1]
        self._setup_candle_key: Optional[str] = None  # candle key we last updated setup on

        # Reversal context (for reversal-proof)
        self._last_exit_side: Optional[str] = None
        self._last_exit_px: Optional[float] = None
        self._last_exit_ms: int = 0

        # Conflict logging de-dupe (avoid spamming HOLD_CONFLICT)
        self._hold_conflict_key: Optional[str] = None
        self._intent_conflict_info: Optional[Dict[str, Any]] = None

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
        self._MICRO_ATR_FALLBACK = float(getattr(cfg, "micro_atr_fallback", 12.0))
        self._last_px: float = 0.0
        self._TICK_SIZE = float(getattr(cfg, "tick_size", 0.05))
        self._arm_log_key: Optional[Tuple[str, str, str]] = None
        self._last_entry_side: Optional[str] = None
        self._last_entry_ms: int = 0
        self._rearm_required: Dict[str, bool] = {"micro": False, "ema915": False}
        self._rearm_epoch_ms: Dict[str, int] = {"micro": 0, "ema915": 0}
        self._shock_lock_side: Optional[str] = None
        self._shock_lock_until_ms: int = 0
        self._fut_flow_warn_min: Optional[str] = None
        self._last_recv_ms: Optional[int] = None

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

        self._write_lock = threading.Lock()
        self.jsonl_file = open(self.cfg.jsonl_path, "a", encoding="utf-8", buffering=1)
        self._arm_write_lock = threading.Lock()
        self.arm_jsonl_file = open(self.cfg.arm_jsonl_path, "a", encoding="utf-8", buffering=1)

    # ------------------------ Public API ------------------------
    def on_tick(self, tick: Dict[str, Any]) -> None:
        self.process_tick(tick)

    def process_tick(self, tick: Dict[str, Any]) -> None:
        """Main entry point for incoming ticks. Safe under bad packets."""
        try:
            ltp = _safe_float(tick.get("ltp"))
            if ltp is None:
                return

            recv_ns = int(tick.get("recv_ns") or time.time_ns())
            recv_ms = int(tick.get("recv_ms") or (recv_ns // 1_000_000))
            recv_ts_utc = datetime.fromtimestamp(recv_ms / 1000.0, tz=timezone.utc)
            self._last_recv_ms = recv_ms

            raw_tick_ts = tick.get("timestamp")
            tick_ts_utc = _parse_ts(raw_tick_ts) if raw_tick_ts is not None else None
            tick_ts_fixed, latency_ms, latency_status = _fix_tick_time(
                tick_ts_utc,
                recv_ts_utc,
                drift_sec=int(getattr(self.cfg, "ts_drift_sec", 19800)),
                drift_tol_sec=int(getattr(self.cfg, "ts_drift_tolerance_sec", 120)),
                max_abs_latency_ms=int(getattr(self.cfg, "max_abs_latency_ms", 300000)),
            )

            # SINGLE RULE: internal timebase = recv_ts_utc
            ts = recv_ts_utc
            candle_ts = ts.replace(second=0, microsecond=0)

            with self._lock:
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

    # ------------------------ Candle Close ------------------------
    def _close_candle_locked(self) -> None:
        """Assumes self._lock is held."""
        if self.current_candle is None:
            return

        candle = self.current_candle
        self.candles.append(candle)
        self.current_candle = None

        metrics = self._compute_metrics_locked()
        # Optional futures flow snapshot (sidecar). Read once per candle close.
        if bool(getattr(self.cfg, "use_fut_flow", False)):
            fut_flow_status = "MISSING_FILE"
            fut_path = str(getattr(self.cfg, "fut_sidecar_path", "") or "")
            if fut_path and os.path.exists(fut_path):
                line = _tail_last_line(fut_path)
                if not line:
                    fut_flow_status = "EMPTY_FILE"
                    metrics["fut_flow"] = None
                else:
                    fut = _parse_fut_candle_row(line)
                    if isinstance(fut, dict) and isinstance(fut.get("ts"), datetime):
                        fut_ts_utc = fut["ts"].astimezone(timezone.utc)
                        age = abs((_now_utc() - fut_ts_utc).total_seconds())
                        if age <= float(getattr(self.cfg, "fut_flow_stale_sec", 180)):
                            metrics["fut_flow"] = fut
                            cvd_now = float(fut.get("cvd", 0.0) or 0.0)
                            cvd_prev = float(getattr(self, "_last_fut_cvd", 0.0) or 0.0)
                            metrics["fut_cvd_delta"] = cvd_now - cvd_prev
                            self._last_fut_cvd = cvd_now
                            fut_flow_status = "OK"
                        else:
                            metrics["fut_flow"] = None
                            fut_flow_status = "STALE"
                    else:
                        metrics["fut_flow"] = None
                        fut_flow_status = "PARSE_ERROR"
            else:
                metrics["fut_flow"] = None

            metrics["fut_flow_status"] = fut_flow_status
            if fut_flow_status != "OK":
                warn_min = _now_utc().strftime("%Y%m%d%H%M")
                if self._fut_flow_warn_min != warn_min:
                    self._fut_flow_warn_min = warn_min
                    self.log.warning("fut_flow_status=%s path=%s", fut_flow_status, fut_path)

        metrics["_candle_ts"] = candle.get("ts")
        cts = candle.get("ts")
        metrics["_candle_ts_utc"] = cts.astimezone(timezone.utc).isoformat() if isinstance(cts, datetime) else None
        metrics["_candle_ts_ist"] = _iso_ist(cts) if isinstance(cts, datetime) else None

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

    def _compute_metrics_locked(self) -> Dict[str, Any]:
        """Compute indicators from candles. Assumes self._lock is held."""
        if len(self.candles) < self.cfg.min_ready_candles:
            return {}

        df = pd.DataFrame(list(self.candles))
        # Ensure required columns exist
        for col in ("open", "high", "low", "close", "ticks", "total_travel", "rv2", "ts"):
            if col not in df.columns:
                return {}

        close = pd.to_numeric(df["close"], errors="coerce")
        high = pd.to_numeric(df["high"], errors="coerce")
        low = pd.to_numeric(df["low"], errors="coerce")
        open_ = pd.to_numeric(df["open"], errors="coerce")

        if close.isna().any() or high.isna().any() or low.isna().any() or open_.isna().any():
            return {}

        # HMA series + last value
        hma_series = _hma_series(close, self.cfg.hma_period)
        if hma_series.isna().iloc[-1]:
            return {}
        hma = float(hma_series.iloc[-1])

        # ATR
        prev_close = close.shift(1).fillna(close)
        tr = np.maximum((high - low).to_numpy(), np.maximum(np.abs((high - prev_close).to_numpy()), np.abs((low - prev_close).to_numpy())))
        tr_s = pd.Series(tr, index=df.index)
        atr = float(tr_s.rolling(self.cfg.atr_period, min_periods=1).mean().iloc[-1])
        if not _is_finite_pos(atr):
            return {}

        curr_px = float(close.iloc[-1])

        # Signed anchor distance & bands
        dist_signed = (curr_px - hma) / atr
        dist_abs = abs(dist_signed)
        upper_band = hma + self.cfg.stretch_limit * atr
        lower_band = hma - self.cfg.stretch_limit * atr

        # Pressure (CLV) + body%
        rng = max(self._TICK_SIZE, float(high.iloc[-1] - low.iloc[-1]))
        clv = float((2.0 * curr_px - float(high.iloc[-1]) - float(low.iloc[-1])) / rng)
        body_pct = float(abs(curr_px - float(open_.iloc[-1])) / rng)

        # Wick %
        upper_wick = float(high.iloc[-1] - max(open_.iloc[-1], close.iloc[-1]))
        lower_wick = float(min(open_.iloc[-1], close.iloc[-1]) - low.iloc[-1])
        upper_wick_pct = float(upper_wick / rng)
        lower_wick_pct = float(lower_wick / rng)

        # Efficiency metrics (price-only)
        last = self.candles[-1]
        total_travel = float(last.get("total_travel", 0.0))
        ticks = int(last.get("ticks", 0))

        path_eff = float(abs(curr_px - float(last.get("open", curr_px))) / max(_EPS, total_travel))
        travel_per_tick = float(total_travel / max(1, ticks))

        rv2 = float(last.get("rv2", 0.0))
        smoothness = float(abs(curr_px - float(last.get("open", curr_px))) / max(_EPS, math.sqrt(max(_EPS, rv2))))

        # PAV proxy (travel vs baseline)
        self._travel_hist.append(total_travel)
        baseline_travel = float(np.mean(self._travel_hist)) if len(self._travel_hist) >= 10 else max(_EPS, total_travel)
        pav_mult = float(total_travel / max(_EPS, baseline_travel))

        # Activity weight (0..1): map pav_mult to a stable participation proxy.
        # Low tape activity => weight near 0; high activity => weight near 1.
        aw_lo = float(self.cfg.activity_w_low)
        aw_hi = float(self.cfg.activity_w_high)
        if pav_mult <= aw_lo:
            activity_w = 0.0
        elif pav_mult >= aw_hi:
            activity_w = 1.0
        else:
            activity_w = float((pav_mult - aw_lo) / max(_EPS, (aw_hi - aw_lo)))

        self._tpt_hist.append(travel_per_tick)
        avg_tpt = float(np.mean(self._tpt_hist)) if len(self._tpt_hist) >= 10 else travel_per_tick

        # Squeeze: candle range bottom percentile of lookback
        curr_range = float(high.iloc[-1] - low.iloc[-1])
        self._range_hist.append(curr_range)
        is_squeeze = False
        if len(self._range_hist) >= self.cfg.squeeze_lookback:
            perc = float(np.percentile(np.array(self._range_hist, dtype=float), self.cfg.squeeze_pct))
            is_squeeze = curr_range <= max(_EPS, perc)

        # HMA hysteresis: 5-bar normalized slope
        slope = 0.0
        if len(hma_series) >= 6:
            slope = float((hma_series.iloc[-1] - hma_series.iloc[-6]) / (5.0 * atr))

        # Bias state update (hysteresis)
        self.bias = _update_bias(self.bias, slope, pos_th=0.05, neg_th=-0.05)

        # Shock (close - prev_close)
        prev_c = float(close.iloc[-2])
        shock_move = float(curr_px - prev_c)
        shock_thresh = float(max(self.cfg.shock_points, self.cfg.shock_atr_mult * atr))
        is_shock = abs(shock_move) >= shock_thresh
        shock_side = "LONG" if shock_move > 0 else "SHORT"
        shock_mid = float((prev_c + curr_px) / 2.0)

        candle_ts = self.candles[-1].get("ts") if self.candles else None
        return {
            "px": curr_px,
            "ts_ist": _now_iso_ist(),
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
            "pav_mult": pav_mult,
            "activity_w": activity_w,
            "squeeze": is_squeeze,
            "bias": self.bias or "NONE",
            "shock": is_shock,
            "shock_side": shock_side,
            "shock_thresh": shock_thresh,
            "tick_overrun": bool(last.get("_tick_overrun", False)),
            "shock_mid": shock_mid,
        }

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
            self._setup_side = None
            self._setup_margin = margin
            return {
                "suggestion": "HOLD_CONFLICT",
                "reason": "market_indecision",
                "margin": margin,
                "score_long": long_score,
                "score_short": short_score,
                "setup_side": None,
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

        # Always keep last diag for tick engine
        self._last_diag = m

        # If in a position: manage state + candle-close exits (intrabar exits handled elsewhere)
        if self.pos is not None:
            return self._manage_open_position(m)

        # --- FLAT: entry selection ---
        # Global safety: doji / indecision filter
        if m["body_pct"] < self.cfg.min_body_pct:
            return {"suggestion": "HOLD", "reason": "doji_body"}

        # Chop safety (the bullet filter)
        if m["path_eff"] < self.cfg.min_path_eff_filter:
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

        # Dist gate: enter only when still "fresh" (within 1.5 ATR of anchor)
        if m["anchor_dist_atr_abs"] > 1.5:
            return {"suggestion": "HOLD", "reason": "too_far_from_anchor"}
        # Anti-chop: avoid re-entries too close to anchor
        if m["anchor_dist_atr_abs"] < self.cfg.entry_anchor_min_atr:
            return {"suggestion": "HOLD", "reason": "too_close_to_anchor"}

        # Patterns: Clean Sweep / Inside Break / Squeeze Engulfing
        # 1) Clean Sweep (institutional-like)
        if m["path_eff"] >= self.cfg.entry_path_eff and m["travel_per_tick"] >= (m["avg_travel_per_tick"] * 1.05):
            if (not over_up) and (not anti_climax_block_long) and m["clv"] >= self.cfg.strong_clv and (m["bias"] == "LONG"):
                return self._execute_entry("LONG", m, reason="clean_sweep")
            if (not over_dn) and (not anti_climax_block_short) and m["clv"] <= -self.cfg.strong_clv and (m["bias"] == "SHORT"):
                return self._execute_entry("SHORT", m, reason="clean_sweep")

        # 2) Inside Bar Break (fresh break near anchor)
        inside_decision = self._inside_break_entry(m, over_up, over_dn, anti_climax_block_long, anti_climax_block_short)
        if inside_decision is not None:
            return inside_decision

        # 3) Engulfing after squeeze (expansion scalp)
        engulf_decision = self._squeeze_engulf_entry(m, over_up, over_dn, anti_climax_block_long, anti_climax_block_short)
        if engulf_decision is not None:
            return engulf_decision

        # Default: pressure + bias + confirm
        if (not over_up) and (m["bias"] == "LONG") and m["clv"] >= self.cfg.min_clv_confirm and float(m.get("activity_w", 0.0) or 0.0) >= self.cfg.min_activity_w_confirm and m["path_eff"] >= 0.40:
            return self._execute_entry("LONG", m, reason="bias_pressure")
        if (not over_dn) and (m["bias"] == "SHORT") and m["clv"] <= -self.cfg.min_clv_confirm and float(m.get("activity_w", 0.0) or 0.0) >= self.cfg.min_activity_w_confirm and m["path_eff"] >= 0.40 and not anti_climax_block_short:
            return self._execute_entry("SHORT", m, reason="bias_pressure")

        # Anti-climax explicitly explains why we didn't short
        if anti_climax_block_short:
            return {"suggestion": "HOLD", "reason": "anti_climax_block_short"}

        return {"suggestion": "HOLD", "reason": "scanning"}

    def _handle_shock_system(self, m: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """WATCH shock candle, then confirm in same direction with clean efficiency."""
        now = _now_utc()

        if self.pending_shock is not None:
            if self.pending_shock.expires_at is not None and now > self.pending_shock.expires_at:
                self.pending_shock = None
                return {"suggestion": "HOLD", "reason": "shock_expired_time"}

            self.pending_shock.remaining_candles -= 1
            if self.pending_shock.remaining_candles <= 0:
                self.pending_shock = None
                return {"suggestion": "HOLD", "reason": "shock_expired"}

            if self.pending_shock.side == "LONG":
                if m["px"] > self.pending_shock.mid_px and m["path_eff"] >= self.cfg.entry_path_eff and m["clv"] >= self.cfg.min_clv_confirm and float(m.get("activity_w", 0.0) or 0.0) >= self.cfg.min_activity_w_confirm:
                    ps = self.pending_shock
                    self.pending_shock = None
                    return {"suggestion": "ENTRY_LONG", "reason": "shock_confirm", "shock_mid": ps.mid_px}
            else:
                if m["px"] < self.pending_shock.mid_px and m["path_eff"] >= self.cfg.entry_path_eff and m["clv"] <= -self.cfg.min_clv_confirm and float(m.get("activity_w", 0.0) or 0.0) >= self.cfg.min_activity_w_confirm:
                    ps = self.pending_shock
                    self.pending_shock = None
                    return {"suggestion": "ENTRY_SHORT", "reason": "shock_confirm", "shock_mid": ps.mid_px}

            # Decrement after failed confirmation attempt (prevents off-by-one expiry).
            self.pending_shock.remaining_candles = int(self.pending_shock.remaining_candles) - 1
            if self.pending_shock.remaining_candles <= 0:
                self.pending_shock = None
                return {"suggestion": "HOLD", "reason": "shock_expired"}
            return {"suggestion": "HOLD", "reason": "watching_shock"}

        if bool(m.get("shock", False)):
            now = _now_utc()
            self.pending_shock = PendingShock(
                side=str(m.get("shock_side", "LONG")),
                mid_px=float(m.get("shock_mid", m["px"])),
                remaining_candles=self.cfg.shock_confirm_candles,
                expires_at=now + timedelta(minutes=int(getattr(self.cfg, "shock_expiry_minutes", 3))),
            )
            now_ms = int(time.time_ns() // 1_000_000)
            self._shock_lock_side = self.pending_shock.side
            self._shock_lock_until_ms = now_ms + int(getattr(self.cfg, "shock_lock_ms", 60000))
            return {"suggestion": f"WATCH_SHOCK_{self.pending_shock.side}", "reason": "shock_detected"}

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
        if (not over_up) and (not anti_climax_block_long) and m["px"] > prev_high and m["bias"] == "LONG" and m["path_eff"] >= 0.40 and m["clv"] >= self.cfg.min_clv_confirm and float(m.get("activity_w", 0.0) or 0.0) >= self.cfg.min_activity_w_confirm:
            return self._execute_entry("LONG", m, reason="break_prev_high")

        # Breakdown down
        if (not over_dn) and (not anti_climax_block_short) and m["px"] < prev_low and m["bias"] == "SHORT" and m["path_eff"] >= 0.40 and m["clv"] <= -self.cfg.min_clv_confirm and float(m.get("activity_w", 0.0) or 0.0) >= self.cfg.min_activity_w_confirm:
            return self._execute_entry("SHORT", m, reason="break_prev_low")

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
        if m["anchor_dist_atr_abs"] >= self.cfg.stretch_limit:
            p.is_harvest = True
            p.trail_atr_current = self.cfg.harvest_trail_atr
        else:
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
            return self._execute_exit("EXIT_WICK", px=m["px"], ts=ts_exit, reason="wick_reject")
        if p.side == "SHORT" and m["lower_wick_pct"] >= self.cfg.wick_threshold and m["clv"] > -self.cfg.min_clv_confirm:
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
    ) -> Dict[str, Any]:
        if tick_ts is not None:
            ts = tick_ts
        else:
            m_ts = m.get("ts")
            ts = m_ts if isinstance(m_ts, datetime) else _now_utc()
        px = float(tick_px) if tick_px is not None else float(m.get("px") or self._last_px)

        atr = float(m.get("atr", self._MICRO_ATR_FALLBACK) or self._MICRO_ATR_FALLBACK)
        atr = max(self._TICK_SIZE, atr)

        hard_min = float(self.cfg.hard_stop_points)
        hard_mult = float(getattr(self.cfg, "hard_stop_atr_mult", 1.5))
        hard_pts = max(hard_min, hard_mult * atr)
        tp_pts = float(self.cfg.tp_atr_mult * atr)

        hard_sl = (px - hard_pts) if side == "LONG" else (px + hard_pts)
        sl = hard_sl
        is_runner = bool(m.get("runner", getattr(self.cfg, "default_runner", False)))
        tp = None if is_runner else ((px + tp_pts) if side == "LONG" else (px - tp_pts))

        new_pos = Position(
            side=side,
            entry_px=px,
            entry_ts=ts,
            hard_sl=hard_sl,
            sl=sl,
            sl_init=sl,
            tp=tp,
            best_px=px,
            is_runner=is_runner,
            engine=engine,
            why=reason or "",
        )

        reserved = {"stream", "engine", "channel", "in_pos", "pos_side"}
        payload = {
            "suggestion": f"ENTRY_{side}",
            "channel": engine,
            "engine": engine,
            "in_pos": True,
            "px": px,
            "ts_utc": ts.astimezone(timezone.utc).isoformat(),
            "ts_ist": _iso_ist(ts),
            "reason": reason or "entry",
            "runner": new_pos.is_runner,
            "pos_side": side,
            "entry_px": px,
            "sl": sl,
            "tp": tp,
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

    def _commit_entry_intent_locked(self, intent: EntryIntent, m0: Dict[str, Any]) -> None:
        extra = dict(m0)
        extra.update(intent.extra)

        if bool(getattr(self.cfg, "respect_bias", False)):
            bias = str(extra.get("bias") or "")
            if bias in ("LONG", "SHORT") and intent.side != bias:
                shock_side = str(extra.get("shock_side") or "")
                if shock_side != intent.side:
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
                # Emit a single HOLD_WAIT_CONFIRM (not a timer; just "not proven yet")
                self._write_signal(
                    {
                        "suggestion": "HOLD_WAIT_CONFIRM",
                        "reason": "reversal_not_proven",
                        "move_atr": float(move_atr),
                        "need_move_atr": float(self.cfg.reverse_min_move_atr),
                        "anchor_ok": bool(anchor_ok),
                        "setup_margin": float(self._setup_margin),
                        "setup_side": self._setup_side,
                        "blocked_entry_side": intent.side,
                    },
                    extra,
                )
                return
        self._execute_entry(
            intent.side,
            extra,
            reason=intent.reason,
            tick_ts=intent.ts,
            tick_px=float(intent.entry_px),
            engine=intent.engine,
        )
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
        return out

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
            min_v = float(getattr(self.cfg, "micro_accept_vel_z", 1.0))
            max_against = float(getattr(self.cfg, "micro_accept_acc_z_max_against", 3.0))
            if side == "LONG":
                return not (vel_z < min_v or acc_z < -max_against)
            return not (vel_z > -min_v or acc_z > max_against)

        prev_h = float(prev.get("high", ltp))
        prev_l = float(prev.get("low", ltp))
        buffer = max(0.12 * atr, 1.0)

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

        if side == "LONG" and ltp >= (lvl + self._TICK_SIZE):
            if self._rearm_required["ema915"]:
                return None
            if self._shock_lock_side and recv_ms < self._shock_lock_until_ms and self._shock_lock_side != "LONG":
                return None
            extra = {
                "channel": "ema915",
                "ema9": float(self._ema9 or 0.0),
                "ema15": float(self._ema15 or 0.0),
                "recv_ms": recv_ms,
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

        if side == "SHORT" and ltp <= (lvl - self._TICK_SIZE):
            if self._rearm_required["ema915"]:
                return None
            if self._shock_lock_side and recv_ms < self._shock_lock_until_ms and self._shock_lock_side != "SHORT":
                return None
            extra = {
                "channel": "ema915",
                "ema9": float(self._ema9 or 0.0),
                "ema15": float(self._ema15 or 0.0),
                "recv_ms": recv_ms,
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

        # Time-based decay exit (scalper TTL)
        max_hold_s = int(getattr(self.cfg, "max_hold_seconds", 15 * 60))
        entry_ts = getattr(p, "entry_ts", None)
        if isinstance(entry_ts, datetime) and max_hold_s > 0:
            if (ts - entry_ts).total_seconds() >= max_hold_s:
                return _do_exit("EXIT_TIME", float(ltp), reason="max_hold_time")

        atr = float(self._last_diag.get("atr", 0.0))
        if not _is_finite_pos(atr):
            atr = 10.0  # safe fallback; avoids div-by-zero and NaN propagation

        # Update best price for trailing
        if p.side == "LONG":
            p.best_px = max(p.best_px, ltp)

            # Hard stop always enforced
            if ltp <= p.hard_sl:
                return _do_exit("EXIT_HARD_STOP", p.hard_sl)

            # SL
            if ltp <= p.sl:
                return _do_exit(_sl_exit_suggestion(), p.sl)

            # TP (if not runner)
            if p.tp is not None and ltp >= p.tp:
                return _do_exit("EXIT_TP", p.tp)

            # Move to BE
            if (not p.is_be) and ((p.best_px - p.entry_px) >= (self.cfg.move_to_be_atr * atr)):
                p.sl = max(p.sl, p.entry_px + self.cfg.be_buffer_points)
                p.is_be = True

            # Trailing
            trail_atr = p.trail_atr_current if p.trail_atr_current > 0 else self.cfg.trail_atr
            trail_sl = p.best_px - (trail_atr * atr)
            p.sl = max(p.sl, trail_sl)

        else:  # SHORT
            p.best_px = min(p.best_px, ltp)

            if ltp >= p.hard_sl:
                return _do_exit("EXIT_HARD_STOP", p.hard_sl)

            if ltp >= p.sl:
                return _do_exit(_sl_exit_suggestion(), p.sl)

            if p.tp is not None and ltp <= p.tp:
                return _do_exit("EXIT_TP", p.tp)

            if (not p.is_be) and ((p.entry_px - p.best_px) >= (self.cfg.move_to_be_atr * atr)):
                p.sl = min(p.sl, p.entry_px - self.cfg.be_buffer_points)
                p.is_be = True

            trail_atr = p.trail_atr_current if p.trail_atr_current > 0 else self.cfg.trail_atr
            trail_sl = p.best_px + (trail_atr * atr)
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
            "stream": "exit",
            "channel": "exit",
            "engine": (p.engine if p else "unknown"),
            "in_pos": False,
            "in_pos_before": bool(p is not None),
            "px": float(px),
            "ts": ts,
            "ts_utc": ts.astimezone(timezone.utc).isoformat(),
            "ts_ist": _iso_ist(ts),
            "reason": reason or suggestion,
            "pos_side": None,
            "pos_side_before": exit_side,
        }

        if p is not None:
            payload.update(
                {
                    "entry_px": p.entry_px,
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
                }
            )

        try:
            self.log.info(self._fmt_scalper_line(payload))
        except Exception:
            pass

        # Update reversal context + require re-arm after any exit
        self._last_exit_side = exit_side
        self._last_exit_px = float(px)
        self._last_exit_ms = int(exit_ms)
        try:
            self._mark_rearm_required(payload.get("engine", "unknown"), exit_ms=int(exit_ms))
        except Exception:
            pass

        self.pos = None
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
        return payload

    def _write_signal(self, decision: Dict[str, Any], metrics: Dict[str, Any]) -> None:
        """Write a single JSONL line for signals and (optionally) duplicate ARM to main."""
        try:
            sug0 = str(decision.get("suggestion", "") or "")
            # Allow important HOLD_* even when write_hold/write_all are off (scalper needs these)
            important_holds = {"HOLD_CONFLICT", "HOLD_WAIT_CONFIRM", "HOLD_WEAK_EDGE"}
            if (not self.cfg.write_all) and sug0.startswith("HOLD") and sug0 not in important_holds:
                return
            if sug0 == "HOLD" and (not self.cfg.write_hold):
                return
            if sug0.startswith("ENTRY_"):
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
            else:
                self._arm_log_key = None
                with self._write_lock:
                    self.jsonl_file.write(json.dumps(_isoize(payload), default=str) + "\n")
                    self.jsonl_file.flush()
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
    except Exception:
        return None

    def _f(i: int) -> float:
        try:
            return float(parts[i])
        except Exception:
            return float("nan")

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
    return isinstance(s, str) and s.startswith("ENTRY_")


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
