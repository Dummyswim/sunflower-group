"""
scalp_multi_engines.py

Multi-engine 1-minute scalping runner on DhanHQ v2 WebSocket ticks.

What it does
- Reuses your existing dhan.py feed loop + packet decoding (no relay).
- Aggregates quote ticks into 1-minute OHLC candles (bucketed by IST minute).
- Produces one MarketContext per closed candle (regime/session/volatility/bias).
- Runs enabled engines and converts them to standardized TradeSetup candidates.
- Arbitrates to at most one winner per candle, emitting SUGGEST (manual) or TRADE (simulator).
- Logs full lifecycle events (CONTEXT/CANDIDATE/DECISION/ORDER_* /EXIT) in JSONL.

Run
  export DHAN_ACCESS_TOKEN=...; export DHAN_CLIENT_ID=...
  python scalp_multi_engines.py
"""

from __future__ import annotations

import asyncio
from collections import deque
import json
import logging
import os
import re
import time
import urllib.request
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Deque, Dict, List, Optional, Sequence, Union

# --- Reuse your Dhan WS runner + decoder ---
try:
    import dhan  # type: ignore
except Exception as e:  # pragma: no cover
    raise SystemExit(
        "Could not import dhan.py. Put scalp_multi_engines.py next to dhan.py "
        "or ensure dhan.py is on PYTHONPATH. Import error: %r" % (e,)
    )


IST_TZ = getattr(dhan, "IST_TZ", timezone(timedelta(hours=5, minutes=30)))


# ----------------------------
# Small env helpers
# ----------------------------

def _env(key: str, default: str = "") -> str:
    v = os.getenv(key)
    if v is None:
        return default
    s = str(v).strip()
    return s if s != "" else default


def _env_int(key: str, default: int) -> int:
    raw = _env(key, str(default))
    try:
        return int(raw)
    except Exception:
        return default


def _env_float(key: str, default: float) -> float:
    raw = _env(key, str(default))
    try:
        return float(raw)
    except Exception:
        return default


def _env_bool(key: str, default: bool) -> bool:
    raw = _env(key, "")
    if raw == "":
        return default
    raw = raw.strip().lower()
    if raw in ("1", "true", "yes", "y", "on"):
        return True
    if raw in ("0", "false", "no", "n", "off"):
        return False
    return default


def _parse_iso(ts: str) -> datetime:
    s = str(ts or "").strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(s)
    except Exception:
        return datetime.now(timezone.utc)


def _parse_hhmm(value: str, default_h: int = 9, default_m: int = 15) -> tuple[int, int]:
    raw = str(value or "").strip()
    m = re.match(r"^(\d{1,2}):(\d{2})$", raw)
    if not m:
        return default_h, default_m
    hh = int(m.group(1))
    mm = int(m.group(2))
    hh = 0 if hh < 0 else 23 if hh > 23 else hh
    mm = 0 if mm < 0 else 59 if mm > 59 else mm
    return hh, mm


def _feed_ts_to_ist(ts: str) -> datetime:
    """Parse the websocket timestamp and return an IST-aware datetime.

    Keep the feed wall-clock time as-is and mark it as IST.
    This avoids +05:30 drifts when feeds label local wall time with UTC offsets.
    """
    dt = _parse_iso(ts)
    wall = dt.replace(tzinfo=None)
    return wall.replace(tzinfo=IST_TZ)


def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


# ----------------------------
# Data types
# ----------------------------

@dataclass
class Candle1m:
    start_ist: str
    end_ist: str

    open: float
    high: float
    low: float
    close: float

    volume: Optional[int] = None
    is_synthetic: bool = False


@dataclass
class EngineDecision:
    engine: str
    signal: str  # READY_LONG / READY_SHORT / HOLD / FILTER_*
    stop_price: Optional[float] = None
    confidence: Optional[float] = None
    reason: str = ""
    entry_type: str = "stop"  # stop | limit | market
    rationale_tags: List[str] = field(default_factory=list)
    veto_flags: List[str] = field(default_factory=list)


@dataclass
class MarketContext:
    ts: str
    regime: str
    session_phase: str
    vol_state: str
    bias: str
    key_levels: Dict[str, float]
    adx: Optional[float] = None
    plus_di: Optional[float] = None
    minus_di: Optional[float] = None
    atr: Optional[float] = None
    atr_ratio: Optional[float] = None
    slope: Optional[float] = None
    confidence: float = 0.0
    macd_bias: str = "neutral"


@dataclass
class TradeSetup:
    signal_id: str
    ts: str
    engine: str
    direction: str  # LONG | SHORT
    entry_type: str  # stop | limit | market
    entry_price: float

    stop_loss: float
    take_profit_1: float
    take_profit_2: Optional[float]

    max_hold_bars: int
    ttl_bars: int

    quality_score: int
    confidence: float
    risk_reward: float

    regime: str
    compatible_regimes: List[str]

    rationale_tags: List[str]
    veto_flags: List[str]


@dataclass
class ArbiterDecision:
    winner: Optional[TradeSetup]
    reason: str
    suppressed: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class PendingOrder:
    setup: TradeSetup
    armed_candle_idx: int
    expires_candle_idx: int


@dataclass
class OpenPosition:
    setup: TradeSetup
    fill_price: float
    entry_candle_idx: int
    entry_session_phase: str


# ----------------------------
# Candle builder (ticks -> 1m OHLC)
# ----------------------------

class OneMinuteCandleBuilder:
    def __init__(self, gap_fill: bool = False, max_gap_fill_minutes: int = 3):
        self._gap_fill = bool(gap_fill)
        self._max_gap_fill_minutes = max(0, int(max_gap_fill_minutes))
        self._cur_start_ist: Optional[datetime] = None
        self._cur_open: float = 0.0
        self._cur_high: float = 0.0
        self._cur_low: float = 0.0
        self._cur_close: float = 0.0

        self._cur_vol_start: Optional[int] = None
        self._cur_vol_last: Optional[int] = None

        self._last_close: Optional[float] = None

    @staticmethod
    def _minute_bucket_ist(dt_ist: datetime) -> datetime:
        if dt_ist.tzinfo is None:
            dt_ist = dt_ist.replace(tzinfo=IST_TZ)
        else:
            dt_ist = dt_ist.astimezone(IST_TZ)
        return dt_ist.replace(second=0, microsecond=0)

    @staticmethod
    def _iso_ist(dt: datetime) -> str:
        return dt.astimezone(IST_TZ).isoformat()

    def on_tick(self, dt_ist: datetime, price: float, volume: Optional[int]) -> List[Candle1m]:
        closed: List[Candle1m] = []
        bucket = self._minute_bucket_ist(dt_ist)

        if self._cur_start_ist is None:
            self._start_new(bucket, price, volume)
            return closed

        if bucket == self._cur_start_ist:
            self._cur_high = max(self._cur_high, price)
            self._cur_low = min(self._cur_low, price)
            self._cur_close = price
            if volume is not None:
                self._cur_vol_last = int(volume)
            return closed

        closed.extend(self._close_current_and_roll(bucket))

        if self._cur_start_ist == bucket:
            self._cur_high = max(self._cur_high, price)
            self._cur_low = min(self._cur_low, price)
            self._cur_close = price
            if volume is not None:
                if self._cur_vol_start is None:
                    self._cur_vol_start = int(volume)
                self._cur_vol_last = int(volume)

        return closed

    def _start_new(self, start_ist: datetime, price: float, volume: Optional[int]) -> None:
        self._cur_start_ist = start_ist
        self._cur_open = self._cur_high = self._cur_low = self._cur_close = float(price)
        if volume is not None:
            v = int(volume)
            self._cur_vol_start = v
            self._cur_vol_last = v
        else:
            self._cur_vol_start = None
            self._cur_vol_last = None

    def _close_current_and_roll(self, next_bucket_ist: datetime) -> List[Candle1m]:
        assert self._cur_start_ist is not None
        closed: List[Candle1m] = []

        def _emit_candle(start_ist: datetime, o: float, h: float, l: float, c: float,
                         vol: Optional[int], is_synth: bool) -> None:
            end_ist = start_ist + timedelta(minutes=1)
            closed.append(
                Candle1m(
                    start_ist=self._iso_ist(start_ist),
                    end_ist=self._iso_ist(end_ist),
                    open=o,
                    high=h,
                    low=l,
                    close=c,
                    volume=vol,
                    is_synthetic=is_synth,
                )
            )

        vol_delta: Optional[int] = None
        if self._cur_vol_start is not None and self._cur_vol_last is not None:
            vd = int(self._cur_vol_last) - int(self._cur_vol_start)
            vol_delta = vd if vd >= 0 else None

        _emit_candle(
            start_ist=self._cur_start_ist,
            o=self._cur_open,
            h=self._cur_high,
            l=self._cur_low,
            c=self._cur_close,
            vol=vol_delta,
            is_synth=False,
        )

        self._last_close = float(self._cur_close)

        if self._gap_fill and self._last_close is not None:
            cur = self._cur_start_ist + timedelta(minutes=1)
            filled = 0
            while cur < next_bucket_ist:
                if self._max_gap_fill_minutes and filled >= self._max_gap_fill_minutes:
                    break
                _emit_candle(
                    start_ist=cur,
                    o=self._last_close,
                    h=self._last_close,
                    l=self._last_close,
                    c=self._last_close,
                    vol=0,
                    is_synth=True,
                )
                filled += 1
                cur += timedelta(minutes=1)

        seed = self._last_close if self._last_close is not None else self._cur_close
        self._start_new(next_bucket_ist, seed, None)

        return closed


# ----------------------------
# Strategy engine interface
# ----------------------------

EngineOutput = Union[EngineDecision, Sequence[EngineDecision], None]


class StrategyEngine:
    name: str = "base"

    def on_candle(self, candle: Candle1m, context: Optional[MarketContext] = None) -> EngineOutput:
        raise NotImplementedError


# ----------------------------
# Indicator helpers
# ----------------------------

def _crossover(prev_a: Optional[float], a: float, prev_b: Optional[float], b: float) -> bool:
    if prev_a is None or prev_b is None:
        return False
    return float(a) > float(b) and float(prev_a) <= float(prev_b)


def _crossunder(prev_a: Optional[float], a: float, prev_b: Optional[float], b: float) -> bool:
    if prev_a is None or prev_b is None:
        return False
    return float(a) < float(b) and float(prev_a) >= float(prev_b)


class EMA:
    def __init__(self, length: int):
        self.length = max(1, int(length))
        self.alpha = 2.0 / (self.length + 1.0)
        self.value: Optional[float] = None

    def update(self, x: float) -> float:
        px = float(x)
        if self.value is None:
            self.value = px
        else:
            self.value = (self.alpha * px) + ((1.0 - self.alpha) * self.value)
        return float(self.value)


class SMA:
    def __init__(self, length: int):
        self.length = max(1, int(length))
        self.q: Deque[float] = deque(maxlen=self.length)
        self.sum: float = 0.0

    def update(self, x: float) -> Optional[float]:
        fx = float(x)
        if len(self.q) == self.length:
            self.sum -= self.q[0]
        self.q.append(fx)
        self.sum += fx
        if len(self.q) < self.length:
            return None
        return self.sum / float(self.length)


class RMA:
    def __init__(self, length: int):
        self.length = max(1, int(length))
        self.value: Optional[float] = None
        self._warm_count = 0
        self._warm_sum = 0.0

    def update(self, x: float) -> Optional[float]:
        fx = float(x)
        if self.value is None:
            self._warm_sum += fx
            self._warm_count += 1
            if self._warm_count < self.length:
                return None
            self.value = self._warm_sum / float(self.length)
            return float(self.value)

        self.value = ((self.value * float(self.length - 1)) + fx) / float(self.length)
        return float(self.value)


class ATR:
    def __init__(self, length: int):
        self.length = max(1, int(length))
        self.prev_close: Optional[float] = None
        self.rma = RMA(self.length)

    def update(self, high: float, low: float, close: float) -> Optional[float]:
        h = float(high)
        l = float(low)
        c = float(close)
        if self.prev_close is None:
            tr = h - l
        else:
            tr = max(h - l, abs(h - self.prev_close), abs(l - self.prev_close))
        self.prev_close = c
        return self.rma.update(tr)


class ADX:
    def __init__(self, length: int = 14):
        self.length = max(1, int(length))
        self.prev_high: Optional[float] = None
        self.prev_low: Optional[float] = None
        self.prev_close: Optional[float] = None

        self.tr_rma = RMA(self.length)
        self.plus_dm_rma = RMA(self.length)
        self.minus_dm_rma = RMA(self.length)
        self.dx_rma = RMA(self.length)

    def update(self, high: float, low: float, close: float) -> tuple[Optional[float], Optional[float], Optional[float]]:
        h = float(high)
        l = float(low)
        c = float(close)

        if self.prev_high is None or self.prev_low is None or self.prev_close is None:
            self.prev_high = h
            self.prev_low = l
            self.prev_close = c
            return None, None, None

        up_move = h - self.prev_high
        down_move = self.prev_low - l
        plus_dm = up_move if (up_move > down_move and up_move > 0.0) else 0.0
        minus_dm = down_move if (down_move > up_move and down_move > 0.0) else 0.0

        tr = max(h - l, abs(h - self.prev_close), abs(l - self.prev_close))
        atr = self.tr_rma.update(tr)
        plus_sm = self.plus_dm_rma.update(plus_dm)
        minus_sm = self.minus_dm_rma.update(minus_dm)

        self.prev_high = h
        self.prev_low = l
        self.prev_close = c

        if atr is None or plus_sm is None or minus_sm is None or atr <= 0.0:
            return None, None, None

        plus_di = 100.0 * (plus_sm / atr)
        minus_di = 100.0 * (minus_sm / atr)
        denom = plus_di + minus_di
        dx = (100.0 * abs(plus_di - minus_di) / denom) if denom > 0.0 else 0.0
        adx = self.dx_rma.update(dx)

        return adx, plus_di, minus_di


# ----------------------------
# Context detector (Forecaster)
# ----------------------------

class MarketContextDetector:
    def __init__(self):
        # Faster ADX default for 1m scalping.
        self.adx_len = _env_int("REGIME_ADX_LENGTH", 9)
        self.slope_len = _env_int("REGIME_SLOPE_EMA", 20)
        self.regime_confirm_bars = max(1, _env_int("REGIME_CONFIRM_BARS", 2))
        self.trend_adx_threshold = _env_float("REGIME_TREND_ADX", 25.0)
        self.range_adx_threshold = _env_float("REGIME_RANGE_ADX", 20.0)
        self.volatile_atr_ratio = _env_float("REGIME_VOLATILE_ATR_RATIO", 1.5)
        self.slope_atr_threshold = _env_float("REGIME_SLOPE_ATR_FRAC", 0.15)

        self.adx = ADX(self.adx_len)
        self.atr14 = ATR(14)
        self.atr_sma20 = SMA(20)
        self.ema = EMA(self.slope_len)
        self.prev_ema: Optional[float] = None

        self.macd_fast = EMA(_env_int("MACD_FAST", 12))
        self.macd_slow = EMA(_env_int("MACD_SLOW", 26))
        self.macd_signal = EMA(_env_int("MACD_SIGNAL", 9))

        self.regime: str = "UNCLEAR"
        self.pending_regime: Optional[str] = None
        self.pending_count: int = 0
        self._last_ctx: Optional[MarketContext] = None

        self.recent_highs: Deque[float] = deque(maxlen=20)
        self.recent_lows: Deque[float] = deque(maxlen=20)

    def _session_phase(self, candle: Candle1m) -> str:
        dt_ist = _parse_iso(candle.start_ist).astimezone(IST_TZ)
        mins = dt_ist.hour * 60 + dt_ist.minute
        if (9 * 60 + 15) <= mins < (10 * 60 + 45):
            return "OPEN_GO"
        if (11 * 60 + 30) <= mins < (14 * 60):
            return "MIDDAY"
        return "LATE"

    def _raw_regime(
        self,
        adx_v: Optional[float],
        plus_di: Optional[float],
        minus_di: Optional[float],
        atr_ratio: Optional[float],
        slope_norm: Optional[float],
    ) -> str:
        slope_mag = abs(slope_norm) if slope_norm is not None else 0.0
        if (
            adx_v is not None
            and adx_v >= self.trend_adx_threshold
            and slope_mag >= self.slope_atr_threshold
            and plus_di is not None
            and minus_di is not None
        ):
            return "TREND_UP" if plus_di >= minus_di else "TREND_DOWN"

        is_volatile_chop = (
            atr_ratio is not None
            and atr_ratio >= self.volatile_atr_ratio
            and (adx_v is None or adx_v < self.trend_adx_threshold)
        )
        if is_volatile_chop:
            return "VOLATILE_CHOP"

        is_range = (
            adx_v is not None
            and adx_v <= self.range_adx_threshold
            and slope_mag < self.slope_atr_threshold
        )
        if is_range:
            return "RANGE"

        return "UNCLEAR"

    def _apply_hysteresis(self, raw_regime: str) -> str:
        if raw_regime == self.regime:
            self.pending_regime = None
            self.pending_count = 0
            return self.regime

        if self.pending_regime != raw_regime:
            self.pending_regime = raw_regime
            self.pending_count = 1
            return self.regime

        self.pending_count += 1
        if self.pending_count >= self.regime_confirm_bars:
            self.regime = raw_regime
            self.pending_regime = None
            self.pending_count = 0
        return self.regime

    def on_candle(self, candle: Candle1m) -> MarketContext:
        # Synthetic candles should not update regime indicators.
        if getattr(candle, "is_synthetic", False) and self._last_ctx is not None:
            last = self._last_ctx
            ctx = MarketContext(
                ts=str(candle.end_ist),
                regime=last.regime,
                session_phase=self._session_phase(candle),
                vol_state=last.vol_state,
                bias=last.bias,
                key_levels=dict(last.key_levels),
                adx=last.adx,
                plus_di=last.plus_di,
                minus_di=last.minus_di,
                atr=last.atr,
                atr_ratio=last.atr_ratio,
                slope=last.slope,
                confidence=max(0.0, float(last.confidence) * 0.85),
                macd_bias=last.macd_bias,
            )
            self._last_ctx = ctx
            return ctx

        adx_v, plus_di, minus_di = self.adx.update(candle.high, candle.low, candle.close)
        atr_v = self.atr14.update(candle.high, candle.low, candle.close)
        atr_mean = self.atr_sma20.update(atr_v if atr_v is not None else 0.0)
        atr_ratio = (atr_v / atr_mean) if (atr_v is not None and atr_mean not in (None, 0.0)) else None

        ema_v = self.ema.update(candle.close)
        slope = None if self.prev_ema is None else (ema_v - self.prev_ema)
        self.prev_ema = ema_v

        slope_norm = None
        if slope is not None:
            base = atr_v if (atr_v is not None and atr_v > 0.0) else max(abs(candle.close), 1.0)
            slope_norm = slope / base

        raw_regime = self._raw_regime(adx_v, plus_di, minus_di, atr_ratio, slope_norm)
        regime = self._apply_hysteresis(raw_regime)

        vol_state = "normal"
        if atr_ratio is not None and atr_ratio >= 1.25:
            vol_state = "expanded"
        elif atr_ratio is not None and atr_ratio <= 0.80:
            vol_state = "contracted"

        macd_line = self.macd_fast.update(candle.close) - self.macd_slow.update(candle.close)
        signal_line = self.macd_signal.update(macd_line)
        delta = macd_line - signal_line
        macd_bias = "neutral"
        macd_eps = (atr_v or (abs(candle.close) * 0.001)) * 0.02
        if delta > macd_eps:
            macd_bias = "long"
        elif delta < -macd_eps:
            macd_bias = "short"

        bias = "neutral"
        if regime == "TREND_UP":
            bias = "long"
        elif regime == "TREND_DOWN":
            bias = "short"

        confidence = 0.0
        if adx_v is not None:
            confidence += 0.5 * _clamp((adx_v - 15.0) / 20.0, 0.0, 1.0)
        if atr_ratio is not None:
            confidence += 0.2 * _clamp(abs(atr_ratio - 1.0), 0.0, 1.0)
        if slope_norm is not None:
            confidence += 0.3 * _clamp(abs(slope_norm) / max(self.slope_atr_threshold, 1e-6), 0.0, 1.0)

        self.recent_highs.append(float(candle.high))
        self.recent_lows.append(float(candle.low))
        key_levels = {
            "swing_high": max(self.recent_highs) if self.recent_highs else float(candle.high),
            "swing_low": min(self.recent_lows) if self.recent_lows else float(candle.low),
            "last_close": float(candle.close),
        }

        ctx = MarketContext(
            ts=str(candle.end_ist),
            regime=regime,
            session_phase=self._session_phase(candle),
            vol_state=vol_state,
            bias=bias,
            key_levels=key_levels,
            adx=adx_v,
            plus_di=plus_di,
            minus_di=minus_di,
            atr=atr_v,
            atr_ratio=atr_ratio,
            slope=slope,
            confidence=_clamp(confidence, 0.0, 1.0),
            macd_bias=macd_bias,
        )
        self._last_ctx = ctx
        return ctx


# ----------------------------
# Engines
# ----------------------------

class MomentumEngine(StrategyEngine):
    name = "momentum"

    def __init__(self, length: int = 12, tick_size: float = 0.05):
        self.length = max(1, int(length))
        self.tick_size = float(tick_size)
        self._closes: List[float] = []
        self._mom0_prev: Optional[float] = None
        self.min_strength_atr_frac = _env_float("MOM_MIN_STRENGTH_ATR_FRAC", 0.15)
        self.close_acceptance = _clamp(_env_float("MOM_CLOSE_ACCEPTANCE", 0.60), 0.5, 0.95)

    def on_candle(self, candle: Candle1m, context: Optional[MarketContext] = None) -> EngineDecision:
        c = float(candle.close)
        self._closes.append(c)

        if len(self._closes) <= self.length:
            return EngineDecision(engine=self.name, signal="HOLD", reason="warmup_close_series")

        mom0 = c - float(self._closes[-1 - self.length])

        if self._mom0_prev is None:
            self._mom0_prev = mom0
            return EngineDecision(engine=self.name, signal="HOLD", reason="warmup_mom1")

        mom1 = mom0 - float(self._mom0_prev)
        self._mom0_prev = mom0

        if context is not None and context.regime not in ("TREND_UP", "TREND_DOWN"):
            return EngineDecision(engine=self.name, signal="HOLD", reason="blocked_non_trend_regime")

        bar_range = max(float(candle.high) - float(candle.low), self.tick_size)
        atr = context.atr if (context is not None and context.atr is not None) else bar_range
        min_strength = max(self.min_strength_atr_frac * atr, self.tick_size)

        close_pos = (float(candle.close) - float(candle.low)) / bar_range
        long_close_ok = close_pos >= self.close_acceptance
        short_close_ok = close_pos <= (1.0 - self.close_acceptance)

        strength = max(abs(mom0), abs(mom1))
        conf = _clamp(strength / max(atr * 2.0, self.tick_size), 0.0, 1.0)

        if mom0 > min_strength and mom1 > 0.0 and long_close_ok:
            if context is not None and context.macd_bias == "short":
                return EngineDecision(engine=self.name, signal="HOLD", reason="blocked_macd_countertrend")
            return EngineDecision(
                engine=self.name,
                signal="READY_LONG",
                stop_price=float(candle.high) + self.tick_size,
                confidence=conf,
                reason=f"mom0={mom0:.4f} mom1={mom1:.4f} atr={atr:.4f}",
                entry_type="stop",
                rationale_tags=["trend_aligned", "strong_momentum", "close_near_high"],
            )

        if mom0 < -min_strength and mom1 < 0.0 and short_close_ok:
            if context is not None and context.macd_bias == "long":
                return EngineDecision(engine=self.name, signal="HOLD", reason="blocked_macd_countertrend")
            return EngineDecision(
                engine=self.name,
                signal="READY_SHORT",
                stop_price=float(candle.low) - self.tick_size,
                confidence=conf,
                reason=f"mom0={mom0:.4f} mom1={mom1:.4f} atr={atr:.4f}",
                entry_type="stop",
                rationale_tags=["trend_aligned", "strong_momentum", "close_near_low"],
            )

        return EngineDecision(engine=self.name, signal="HOLD", reason=f"no_setup mom0={mom0:.4f} mom1={mom1:.4f}")


class KeltnerChannelsEngine(StrategyEngine):
    name = "keltner"

    def __init__(
        self,
        length: int = 20,
        mult: float = 2.0,
        use_exp: bool = True,
        bands_style: str = "Average True Range",
        atr_length: int = 10,
        tick_size: float = 0.05,
    ):
        self.length = max(1, int(length))
        self.mult = float(mult)
        self.use_exp = bool(use_exp)
        self.bands_style = (bands_style or "Average True Range").strip()
        if self.bands_style not in ("Average True Range", "True Range", "Range"):
            self.bands_style = "Average True Range"
        self.atr_length = max(1, int(atr_length))
        self.tick_size = float(tick_size)

        self.src_ema = EMA(self.length)
        self.src_sma = SMA(self.length)
        self.atr = ATR(self.atr_length)
        self.range_rma = RMA(self.length)

        self.prev_src: Optional[float] = None
        self.prev_upper: Optional[float] = None
        self.prev_lower: Optional[float] = None
        self.prev_close: Optional[float] = None

    def on_candle(self, candle: Candle1m, context: Optional[MarketContext] = None) -> EngineOutput:
        src = float(candle.close)
        ma = self.src_ema.update(src) if self.use_exp else self.src_sma.update(src)
        if ma is None:
            self.prev_close = src
            return EngineDecision(engine=self.name, signal="HOLD", reason="warmup_ma")

        rangema: Optional[float]
        if self.bands_style == "Average True Range":
            rangema = self.atr.update(candle.high, candle.low, candle.close)
            if rangema is None:
                self.prev_close = src
                return EngineDecision(engine=self.name, signal="HOLD", reason="warmup_atr")
        elif self.bands_style == "True Range":
            h = float(candle.high)
            l = float(candle.low)
            if self.prev_close is None:
                rangema = h - l
            else:
                rangema = max(h - l, abs(h - self.prev_close), abs(l - self.prev_close))
        else:
            rangema = self.range_rma.update(float(candle.high) - float(candle.low))
            if rangema is None:
                self.prev_close = src
                return EngineDecision(engine=self.name, signal="HOLD", reason="warmup_range_rma")

        upper = float(ma) + (float(rangema) * self.mult)
        lower = float(ma) - (float(rangema) * self.mult)

        cross_upper = _crossover(self.prev_src, src, self.prev_upper, upper)
        cross_lower = _crossunder(self.prev_src, src, self.prev_lower, lower)

        self.prev_src = src
        self.prev_upper = upper
        self.prev_lower = lower
        self.prev_close = src

        if cross_upper:
            return EngineDecision(
                engine=self.name,
                signal="READY_LONG",
                stop_price=float(candle.high) + self.tick_size,
                reason=f"cross_upper ma={ma:.2f} upper={upper:.2f}",
                entry_type="stop",
                rationale_tags=["band_breakout"],
            )
        if cross_lower:
            return EngineDecision(
                engine=self.name,
                signal="READY_SHORT",
                stop_price=float(candle.low) - self.tick_size,
                reason=f"cross_lower ma={ma:.2f} lower={lower:.2f}",
                entry_type="stop",
                rationale_tags=["band_breakout"],
            )

        return EngineDecision(engine=self.name, signal="HOLD", reason=f"no_cross ma={ma:.2f} upper={upper:.2f} lower={lower:.2f}")


class MACDEngine(StrategyEngine):
    name = "macd"

    def __init__(self, fast_length: int = 12, slow_length: int = 26, macd_length: int = 9):
        self.fast_length = max(1, int(fast_length))
        self.slow_length = max(self.fast_length + 1, int(slow_length))
        self.macd_length = max(1, int(macd_length))

        self.fast_ema = EMA(self.fast_length)
        self.slow_ema = EMA(self.slow_length)
        self.signal_ema = EMA(self.macd_length)

    def on_candle(self, candle: Candle1m, context: Optional[MarketContext] = None) -> EngineDecision:
        close = float(candle.close)
        macd = self.fast_ema.update(close) - self.slow_ema.update(close)
        a_macd = self.signal_ema.update(macd)
        delta = macd - a_macd

        eps = ((context.atr if context and context.atr is not None else close * 0.001) * 0.02)
        if delta > eps:
            return EngineDecision(engine=self.name, signal="FILTER_BULL", confidence=_clamp(abs(delta), 0.0, 1.0), reason=f"delta={delta:.6f}")
        if delta < -eps:
            return EngineDecision(engine=self.name, signal="FILTER_BEAR", confidence=_clamp(abs(delta), 0.0, 1.0), reason=f"delta={delta:.6f}")
        return EngineDecision(engine=self.name, signal="FILTER_NEUTRAL", confidence=0.5, reason=f"delta={delta:.6f}")


class ConsecutiveUpDownEngine(StrategyEngine):
    name = "consecutive"

    def __init__(self, consecutive_bars_up: int = 3, consecutive_bars_down: int = 3):
        self.consecutive_bars_up = max(1, int(consecutive_bars_up))
        self.consecutive_bars_down = max(1, int(consecutive_bars_down))
        self.prev_close: Optional[float] = None
        self.ups = 0
        self.dns = 0

    def on_candle(self, candle: Candle1m, context: Optional[MarketContext] = None) -> EngineDecision:
        price = float(candle.close)
        if self.prev_close is None:
            self.prev_close = price
            return EngineDecision(engine=self.name, signal="HOLD", reason="warmup_close")

        self.ups = self.ups + 1 if price > self.prev_close else 0
        self.dns = self.dns + 1 if price < self.prev_close else 0
        self.prev_close = price

        if self.ups >= self.consecutive_bars_up:
            return EngineDecision(
                engine=self.name,
                signal="READY_SHORT",
                reason=f"exhaustion ups={self.ups}",
                entry_type="market",
                rationale_tags=["exhaustion", "counter_move"],
            )
        if self.dns >= self.consecutive_bars_down:
            return EngineDecision(
                engine=self.name,
                signal="READY_LONG",
                reason=f"exhaustion dns={self.dns}",
                entry_type="market",
                rationale_tags=["exhaustion", "counter_move"],
            )
        return EngineDecision(engine=self.name, signal="HOLD", reason=f"ups={self.ups} dns={self.dns}")


class ChannelBreakOutEngine(StrategyEngine):
    name = "channel_breakout"

    def __init__(self, length: int = 5, tick_size: float = 0.05):
        self.length = max(2, int(length))
        self.tick_size = float(tick_size)
        self.highs: Deque[float] = deque(maxlen=self.length)
        self.lows: Deque[float] = deque(maxlen=self.length)
        self.ranges: Deque[float] = deque(maxlen=self.length)
        self.compression_factor = _env_float("CHBRK_COMPRESSION", 0.9)

    def on_candle(self, candle: Candle1m, context: Optional[MarketContext] = None) -> EngineDecision:
        if len(self.highs) < self.length or len(self.lows) < self.length:
            self.highs.append(float(candle.high))
            self.lows.append(float(candle.low))
            self.ranges.append(float(candle.high) - float(candle.low))
            return EngineDecision(engine=self.name, signal="HOLD", reason="warmup_channel")

        up_bound = max(self.highs)
        down_bound = min(self.lows)
        avg_range = (sum(self.ranges) / float(len(self.ranges))) if self.ranges else 0.0
        curr_range = max(float(candle.high) - float(candle.low), self.tick_size)
        compressed = avg_range > 0.0 and curr_range <= avg_range * self.compression_factor

        dec = EngineDecision(engine=self.name, signal="HOLD", reason="no_setup")

        regime = context.regime if context is not None else "UNCLEAR"
        if regime in ("TREND_UP", "TREND_DOWN"):
            if regime == "TREND_UP" and float(candle.close) > up_bound and compressed:
                dec = EngineDecision(
                    engine=self.name,
                    signal="READY_LONG",
                    stop_price=up_bound + self.tick_size,
                    reason=f"trend_breakout up={up_bound:.2f} compressed={compressed}",
                    entry_type="stop",
                    rationale_tags=["trend_aligned", "breakout", "post_compression"],
                )
            elif regime == "TREND_DOWN" and float(candle.close) < down_bound and compressed:
                dec = EngineDecision(
                    engine=self.name,
                    signal="READY_SHORT",
                    stop_price=down_bound - self.tick_size,
                    reason=f"trend_breakout dn={down_bound:.2f} compressed={compressed}",
                    entry_type="stop",
                    rationale_tags=["trend_aligned", "breakout", "post_compression"],
                )
            else:
                dec = EngineDecision(engine=self.name, signal="HOLD", reason=f"trend_no_break compressed={compressed}")

        elif regime in ("RANGE", "VOLATILE_CHOP"):
            if float(candle.high) >= up_bound:
                rng = max(float(candle.high) - float(candle.low), self.tick_size)
                wick_reject = (float(candle.high) - float(candle.close)) / rng
                touch_depth = max(0.0, (float(candle.high) - up_bound) / rng)
                conf = _clamp(0.45 + 0.35 * wick_reject + 0.20 * touch_depth, 0.35, 0.95)
                dec = EngineDecision(
                    engine=self.name,
                    signal="READY_SHORT",
                    stop_price=up_bound,
                    reason=f"range_fade_upper={up_bound:.2f}",
                    entry_type="limit",
                    confidence=conf,
                    rationale_tags=["range_fade", "upper_extreme"],
                )
            elif float(candle.low) <= down_bound:
                rng = max(float(candle.high) - float(candle.low), self.tick_size)
                wick_reject = (float(candle.close) - float(candle.low)) / rng
                touch_depth = max(0.0, (down_bound - float(candle.low)) / rng)
                conf = _clamp(0.45 + 0.35 * wick_reject + 0.20 * touch_depth, 0.35, 0.95)
                dec = EngineDecision(
                    engine=self.name,
                    signal="READY_LONG",
                    stop_price=down_bound,
                    reason=f"range_fade_lower={down_bound:.2f}",
                    entry_type="limit",
                    confidence=conf,
                    rationale_tags=["range_fade", "lower_extreme"],
                )
            else:
                dec = EngineDecision(engine=self.name, signal="HOLD", reason="range_no_touch")

        self.highs.append(float(candle.high))
        self.lows.append(float(candle.low))
        self.ranges.append(curr_range)
        return dec


class PivotExtensionEngine(StrategyEngine):
    name = "pivot_extension"

    def __init__(self, left_bars: int = 4, right_bars: int = 2):
        self.left_bars = max(1, int(left_bars))
        self.right_bars = max(1, int(right_bars))
        self.highs: List[float] = []
        self.lows: List[float] = []

    def on_candle(self, candle: Candle1m, context: Optional[MarketContext] = None) -> EngineOutput:
        self.highs.append(float(candle.high))
        self.lows.append(float(candle.low))

        needed = self.left_bars + self.right_bars + 1
        if len(self.highs) < needed:
            return EngineDecision(engine=self.name, signal="HOLD", reason="warmup_pivot")

        if context is not None and context.regime not in ("RANGE", "VOLATILE_CHOP"):
            return EngineDecision(engine=self.name, signal="HOLD", reason="blocked_non_range_regime")

        pivot_idx = len(self.highs) - 1 - self.right_bars
        left_start = pivot_idx - self.left_bars
        right_end = pivot_idx + self.right_bars

        pivot_high = self.highs[pivot_idx]
        left_highs = self.highs[left_start:pivot_idx]
        right_highs = self.highs[pivot_idx + 1:right_end + 1]
        is_ph = all(pivot_high > h for h in left_highs) and all(pivot_high > h for h in right_highs)

        pivot_low = self.lows[pivot_idx]
        left_lows = self.lows[left_start:pivot_idx]
        right_lows = self.lows[pivot_idx + 1:right_end + 1]
        is_pl = all(pivot_low < l for l in left_lows) and all(pivot_low < l for l in right_lows)

        if is_pl:
            return EngineDecision(
                engine=self.name,
                signal="READY_LONG",
                stop_price=pivot_low,
                reason=f"pivot_low={pivot_low:.2f} at_offset=-{self.right_bars}",
                entry_type="limit",
                rationale_tags=["range_specialist", "pivot_low"],
            )
        if is_ph:
            return EngineDecision(
                engine=self.name,
                signal="READY_SHORT",
                stop_price=pivot_high,
                reason=f"pivot_high={pivot_high:.2f} at_offset=-{self.right_bars}",
                entry_type="limit",
                rationale_tags=["range_specialist", "pivot_high"],
            )

        return EngineDecision(engine=self.name, signal="HOLD", reason="no_pivot")


class PriceActionEngine(StrategyEngine):
    name = "price_action"

    def __init__(self, tick_size: float = 0.05):
        self.tick_size = float(tick_size)
        self.history: Deque[Candle1m] = deque(maxlen=3)

        self.engulf_body_mult = _env_float("PA_ENGULF_BODY_MULT", 1.05)
        self.wick_body_mult = _env_float("PA_WICK_BODY_MULT", 2.2)
        self.opp_wick_max_mult = _env_float("PA_OPP_WICK_MAX_MULT", 0.6)
        self.doji_body_frac = _env_float("PA_DOJI_BODY_FRAC", 0.40)
        self.level_prox_atr = _env_float("PA_LEVEL_PROX_ATR", 0.35)
        self.close_accept = _clamp(_env_float("PA_CLOSE_ACCEPT", 0.65), 0.5, 0.95)

    @staticmethod
    def _body_wicks(c: Candle1m) -> tuple[float, float, float]:
        body = abs(float(c.close) - float(c.open))
        upper = float(c.high) - max(float(c.open), float(c.close))
        lower = min(float(c.open), float(c.close)) - float(c.low)
        return body, max(0.0, upper), max(0.0, lower)

    def on_candle(self, candle: Candle1m, context: Optional[MarketContext] = None) -> EngineOutput:
        if getattr(candle, "is_synthetic", False):
            return EngineDecision(engine=self.name, signal="HOLD", reason="synthetic_candle")

        self.history.append(candle)
        if len(self.history) < 3:
            return EngineDecision(engine=self.name, signal="HOLD", reason="warming_up")

        c1, c2, c3 = list(self.history)
        b1, uw1, lw1 = self._body_wicks(c1)
        b2, _, _ = self._body_wicks(c2)
        b3, uw3, lw3 = self._body_wicks(c3)

        def bull(c: Candle1m) -> bool:
            return float(c.close) > float(c.open)

        def bear(c: Candle1m) -> bool:
            return float(c.close) < float(c.open)

        bar_range = max(float(c3.high) - float(c3.low), self.tick_size)
        close_pos = (float(c3.close) - float(c3.low)) / bar_range

        atr = float(context.atr) if (context is not None and context.atr is not None) else None
        swing_high_raw = context.key_levels.get("swing_high") if context else None
        swing_low_raw = context.key_levels.get("swing_low") if context else None
        swing_high = float(swing_high_raw) if swing_high_raw is not None else None
        swing_low = float(swing_low_raw) if swing_low_raw is not None else None

        # 1) Dominance (engulfing)
        tol = self.tick_size * 2.0
        if bear(c2) and bull(c3) and b3 >= (b2 * self.engulf_body_mult):
            engulf = (float(c3.open) <= float(c2.close) + tol) and (float(c3.close) >= float(c2.open) - tol)
            if engulf and close_pos >= self.close_accept:
                return EngineDecision(
                    engine=self.name,
                    signal="READY_LONG",
                    stop_price=float(c3.high) + self.tick_size,
                    confidence=0.78,
                    entry_type="stop",
                    rationale_tags=["dominance_bullish"],
                )

        if bull(c2) and bear(c3) and b3 >= (b2 * self.engulf_body_mult):
            engulf = (float(c3.open) >= float(c2.close) - tol) and (float(c3.close) <= float(c2.open) + tol)
            if engulf and close_pos <= (1.0 - self.close_accept):
                return EngineDecision(
                    engine=self.name,
                    signal="READY_SHORT",
                    stop_price=float(c3.low) - self.tick_size,
                    confidence=0.78,
                    entry_type="stop",
                    rationale_tags=["dominance_bearish"],
                )

        # 2) Rejection (pin bar), prefer near swing levels.
        base = max(b3, self.tick_size)
        near_low = bool(
            atr is not None
            and swing_low is not None
            and abs(float(c3.low) - swing_low) <= self.level_prox_atr * atr
        )
        near_high = bool(
            atr is not None
            and swing_high is not None
            and abs(float(c3.high) - swing_high) <= self.level_prox_atr * atr
        )

        if lw3 >= self.wick_body_mult * base and uw3 <= self.opp_wick_max_mult * base and close_pos >= self.close_accept:
            conf = 0.66 + (0.10 if near_low else 0.0)
            return EngineDecision(
                engine=self.name,
                signal="READY_LONG",
                stop_price=float(c3.high) + self.tick_size,
                confidence=_clamp(conf, 0.0, 1.0),
                entry_type="stop",
                rationale_tags=["rejection_bullish"] + (["near_swing_low"] if near_low else []),
            )

        if uw3 >= self.wick_body_mult * base and lw3 <= self.opp_wick_max_mult * base and close_pos <= (1.0 - self.close_accept):
            conf = 0.66 + (0.10 if near_high else 0.0)
            return EngineDecision(
                engine=self.name,
                signal="READY_SHORT",
                stop_price=float(c3.low) - self.tick_size,
                confidence=_clamp(conf, 0.0, 1.0),
                entry_type="stop",
                rationale_tags=["rejection_bearish"] + (["near_swing_high"] if near_high else []),
            )

        # 3) U-turn (morning/evening star simplified).
        c1_body = max(b1, self.tick_size)
        is_doji2 = b2 <= self.doji_body_frac * c1_body
        c1_mid = (float(c1.open) + float(c1.close)) / 2.0

        if bear(c1) and is_doji2 and bull(c3) and float(c3.close) >= c1_mid:
            return EngineDecision(
                engine=self.name,
                signal="READY_LONG",
                stop_price=float(c3.high) + self.tick_size,
                confidence=0.90,
                entry_type="stop",
                rationale_tags=["u_turn_bullish"],
            )

        if bull(c1) and is_doji2 and bear(c3) and float(c3.close) <= c1_mid:
            return EngineDecision(
                engine=self.name,
                signal="READY_SHORT",
                stop_price=float(c3.low) - self.tick_size,
                confidence=0.90,
                entry_type="stop",
                rationale_tags=["u_turn_bearish"],
            )

        return EngineDecision(engine=self.name, signal="HOLD", reason="no_pattern")


# ----------------------------
# Arbitration and risk governor
# ----------------------------

class SignalArbiter:
    def __init__(self):
        self.min_score = _env_float("ARBITER_MIN_SCORE", 70.0)
        self.conflict_band = _env_float("ARBITER_CONFLICT_BAND", 5.0)

        # Weights sum to 100.
        self.w_trend = 25.0
        self.w_regime = 20.0
        self.w_time = 15.0
        self.w_momentum = 15.0
        self.w_rr = 15.0
        self.w_levels = 10.0

        # Session-time quality knobs.
        self.time_q_open_go = _clamp(_env_float("ARBITER_TIME_Q_OPEN_GO", 1.0), 0.0, 1.0)
        self.time_q_late = _clamp(_env_float("ARBITER_TIME_Q_LATE", 0.70), 0.0, 1.0)
        self.time_q_midday = _clamp(_env_float("ARBITER_TIME_Q_MIDDAY", 0.35), 0.0, 1.0)
        self.time_q_other = _clamp(_env_float("ARBITER_TIME_Q_OTHER", 0.50), 0.0, 1.0)

        # Risk-reward normalization knobs.
        self.rr_target_default = max(0.10, _env_float("ARBITER_RR_TARGET_DEFAULT", 2.0))
        self.rr_target_trend = max(0.10, _env_float("ARBITER_RR_TARGET_TREND", 2.5))
        self.rr_target_range = max(0.10, _env_float("ARBITER_RR_TARGET_RANGE", 1.30))
        self.rr_target_range_fade = max(0.10, _env_float("ARBITER_RR_TARGET_RANGE_FADE", 1.12))
        self.rr_target_range_specialist = max(0.10, _env_float("ARBITER_RR_TARGET_RANGE_SPECIALIST", 1.25))
        self.rr_target_chop = max(0.10, _env_float("ARBITER_RR_TARGET_CHOP", 1.20))
        self.use_tp2_for_trend_rr = _env_bool("ARBITER_USE_TP2_FOR_TREND_RR", True)

        # Optional floor for counter-trend reversal patterns.
        self.countertrend_pattern_floor = _clamp(_env_float("ARBITER_COUNTERTREND_PATTERN_FLOOR", 0.0), 0.0, 0.75)

    def _time_quality(self, session_phase: str) -> float:
        if session_phase == "OPEN_GO":
            return self.time_q_open_go
        if session_phase == "LATE":
            return self.time_q_late
        if session_phase == "MIDDAY":
            return self.time_q_midday
        return self.time_q_other

    @staticmethod
    def _is_countertrend_pattern(setup: TradeSetup) -> bool:
        tags = setup.rationale_tags or []
        return any(("u_turn" in t) or ("rejection" in t) or ("dominance" in t) for t in tags)

    def _trend_alignment(self, setup: TradeSetup, ctx: MarketContext) -> float:
        if ctx.bias == "neutral":
            return 0.5
        if (ctx.bias == "long" and setup.direction == "LONG") or (ctx.bias == "short" and setup.direction == "SHORT"):
            return 1.0
        if self.countertrend_pattern_floor > 0.0 and self._is_countertrend_pattern(setup):
            return self.countertrend_pattern_floor
        return 0.0

    def _regime_compat(self, setup: TradeSetup, ctx: MarketContext) -> float:
        return 1.0 if ctx.regime in setup.compatible_regimes else 0.0

    def _effective_rr(self, setup: TradeSetup, ctx: MarketContext) -> float:
        rr = float(setup.risk_reward)
        if (
            self.use_tp2_for_trend_rr
            and ctx.regime in ("TREND_UP", "TREND_DOWN")
            and setup.take_profit_2 is not None
        ):
            risk = max(abs(float(setup.entry_price) - float(setup.stop_loss)), 1e-6)
            rr = abs(float(setup.take_profit_2) - float(setup.entry_price)) / risk
        return max(0.0, rr)

    def _rr_target(self, setup: TradeSetup, ctx: MarketContext) -> float:
        if ctx.regime in ("TREND_UP", "TREND_DOWN"):
            return self.rr_target_trend
        if ctx.regime == "RANGE":
            tags = setup.rationale_tags or []
            if any("range_fade" in t for t in tags):
                return self.rr_target_range_fade
            if any(("range_specialist" in t) or ("counter_move" in t) for t in tags):
                return self.rr_target_range_specialist
            return self.rr_target_range
        if ctx.regime == "VOLATILE_CHOP":
            return self.rr_target_chop
        return self.rr_target_default

    def _levels_quality(self, setup: TradeSetup) -> float:
        if any(v.startswith("near_") for v in setup.veto_flags):
            return 0.25
        return 1.0

    def score(self, setup: TradeSetup, ctx: MarketContext) -> float:
        trend = self._trend_alignment(setup, ctx)
        regime = self._regime_compat(setup, ctx)
        tqual = self._time_quality(ctx.session_phase)
        mqual = _clamp(setup.confidence, 0.0, 1.0)
        rr_target = self._rr_target(setup, ctx)
        rrqual = _clamp(self._effective_rr(setup, ctx) / max(rr_target, 1e-6), 0.0, 1.0)
        lqual = self._levels_quality(setup)

        score = (
            self.w_trend * trend
            + self.w_regime * regime
            + self.w_time * tqual
            + self.w_momentum * mqual
            + self.w_rr * rrqual
            + self.w_levels * lqual
        )

        tags = setup.rationale_tags or []
        bonus = 0.0
        if any("u_turn" in t for t in tags):
            bonus += 10.0
        elif any("rejection" in t for t in tags):
            bonus += 6.0
        elif any("dominance" in t for t in tags):
            bonus += 4.0

        # Soft penalty for each veto.
        score -= 5.0 * float(len(setup.veto_flags))
        score += bonus
        return _clamp(score, 0.0, 100.0)

    def choose(self, setups: List[TradeSetup], ctx: MarketContext) -> ArbiterDecision:
        if not setups:
            return ArbiterDecision(winner=None, reason="no_candidates")

        scored: List[tuple[float, TradeSetup]] = []
        for s in setups:
            score = self.score(s, ctx)
            s.quality_score = int(round(score))
            scored.append((score, s))

        scored.sort(key=lambda x: x[0], reverse=True)
        top_score, top_setup = scored[0]

        if top_score < self.min_score:
            return ArbiterDecision(
                winner=None,
                reason=f"low_quality score={top_score:.1f}<min={self.min_score:.1f}",
                suppressed=[{"signal_id": s.signal_id, "engine": s.engine, "direction": s.direction, "score": sc} for sc, s in scored],
            )

        if len(scored) >= 2:
            second_score, second_setup = scored[1]
            opposite = top_setup.direction != second_setup.direction
            close_scores = abs(top_score - second_score) <= self.conflict_band
            strong_bias = ctx.bias in ("long", "short") and (
                (ctx.bias == "long" and top_setup.direction == "LONG")
                or (ctx.bias == "short" and top_setup.direction == "SHORT")
            )
            if opposite and close_scores and not strong_bias:
                return ArbiterDecision(
                    winner=None,
                    reason=f"conflict top={top_score:.1f} second={second_score:.1f}",
                    suppressed=[{"signal_id": s.signal_id, "engine": s.engine, "direction": s.direction, "score": sc} for sc, s in scored],
                )

        suppressed = [
            {"signal_id": s.signal_id, "engine": s.engine, "direction": s.direction, "score": sc}
            for sc, s in scored
            if s.signal_id != top_setup.signal_id
        ]
        return ArbiterDecision(winner=top_setup, reason="selected", suppressed=suppressed)


@dataclass
class GateResult:
    allowed: bool
    reason: str


class RiskManager:
    def __init__(self):
        self.max_daily_loss_points = _env_float("RISK_MAX_DAILY_LOSS_POINTS", 150.0)
        self.max_consecutive_losses = _env_int("RISK_MAX_CONSECUTIVE_LOSSES", 3)
        self.cooldown_bars = _env_int("RISK_COOLDOWN_BARS", 5)
        self.max_trades_per_session = _env_int("RISK_MAX_TRADES_PER_SESSION", 12)
        self.block_volatile_chop = _env_bool("RISK_BLOCK_VOLATILE_CHOP", False)
        self.volatile_guard_ratio = _env_float("RISK_VOL_SPIKE_ATR_RATIO", 2.2)

        self.daily_pnl_points = 0.0
        self.consecutive_losses = 0
        self.cooldown_until_candle = 0
        self.trades_today = 0
        self.current_day: Optional[str] = None

    def on_new_candle(self, candle: Candle1m) -> None:
        day = _parse_iso(candle.start_ist).astimezone(IST_TZ).date().isoformat()
        if self.current_day != day:
            self.current_day = day
            self.daily_pnl_points = 0.0
            self.consecutive_losses = 0
            self.cooldown_until_candle = 0
            self.trades_today = 0

    def gate(self, candle_idx: int, ctx: MarketContext, has_open_position: bool) -> GateResult:
        if has_open_position:
            return GateResult(False, "open_position_active")

        if self.daily_pnl_points <= -abs(self.max_daily_loss_points):
            return GateResult(False, "daily_loss_limit")

        if self.trades_today >= max(1, self.max_trades_per_session):
            return GateResult(False, "trade_count_cap")

        if candle_idx < self.cooldown_until_candle:
            return GateResult(False, f"cooldown_until={self.cooldown_until_candle}")

        if self.block_volatile_chop and ctx.regime == "VOLATILE_CHOP":
            return GateResult(False, "volatile_chop_guard")

        if ctx.atr_ratio is not None and ctx.atr_ratio >= self.volatile_guard_ratio:
            return GateResult(False, "volatility_spike_guard")

        return GateResult(True, "ok")

    def on_entry(self) -> None:
        self.trades_today += 1

    def on_exit(self, pnl_points: float, candle_idx: int) -> None:
        self.daily_pnl_points += float(pnl_points)
        if pnl_points < 0.0:
            self.consecutive_losses += 1
            if self.consecutive_losses >= self.max_consecutive_losses:
                self.cooldown_until_candle = candle_idx + max(1, self.cooldown_bars)
        else:
            self.consecutive_losses = 0


# ----------------------------
# Engine manager + JSONL writer
# ----------------------------

class JsonlAppender:
    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    def append(self, row: Dict[str, Any]) -> None:
        line = json.dumps(row, default=str, separators=(",", ":"))
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(line + "\n")


class WinStatsStore:
    def __init__(self):
        self.path = _env("NOTIFY_STATS_PATH", "logs/win_stats.json")
        self.data: Dict[str, Dict[str, int]] = {}
        try:
            if os.path.exists(self.path):
                with open(self.path, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                    if isinstance(raw, dict):
                        for k, v in raw.items():
                            if isinstance(k, str) and isinstance(v, dict):
                                wins = int(v.get("wins", 0))
                                total = int(v.get("total", 0))
                                self.data[k] = {"wins": max(0, wins), "total": max(0, total)}
        except Exception:
            self.data = {}

    @staticmethod
    def key(engine: str, regime: str, session_phase: str) -> str:
        return f"{engine}:{regime}:{session_phase}"

    def _flush(self) -> None:
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, separators=(",", ":"))

    def record(self, setup: TradeSetup, session_phase: str, win: bool) -> None:
        k = self.key(setup.engine, setup.regime, session_phase)
        rec = self.data.get(k, {"wins": 0, "total": 0})
        rec["total"] += 1
        if win:
            rec["wins"] += 1
        self.data[k] = rec
        try:
            self._flush()
        except Exception:
            pass

    def winrate(self, setup: TradeSetup, ctx: MarketContext) -> float:
        k = self.key(setup.engine, ctx.regime, ctx.session_phase)
        rec = self.data.get(k)
        if not rec:
            return 0.0
        total = int(rec.get("total", 0))
        wins = int(rec.get("wins", 0))
        if total < 10:
            return 0.0
        return float(wins) / float(max(1, total))


class Notifier:
    def __init__(self):
        self.enabled = _env_bool("NOTIFY_ENABLED", False)
        self.webhook = _env("NOTIFY_WEBHOOK_URL", "")
        self.min_score = _env_float("NOTIFY_MIN_SCORE", 75.0)
        self.min_winrate = _env_float("NOTIFY_MIN_WINRATE", 0.55)
        self.cooldown_sec = max(0, _env_int("NOTIFY_COOLDOWN_SEC", 30))
        self._last_sent = 0.0

    def should_send(self, setup: TradeSetup, winrate: float) -> bool:
        if not self.enabled or not self.webhook:
            return False
        if float(setup.quality_score) < float(self.min_score):
            return False
        if float(winrate) < float(self.min_winrate):
            return False
        now = time.time()
        if (now - self._last_sent) < float(self.cooldown_sec):
            return False
        self._last_sent = now
        return True

    def send(self, setup: TradeSetup, ctx: MarketContext, winrate: float) -> None:
        payload = {
            "ts": setup.ts,
            "engine": setup.engine,
            "direction": setup.direction,
            "entry": setup.entry_price,
            "sl": setup.stop_loss,
            "tp1": setup.take_profit_1,
            "score": setup.quality_score,
            "regime": ctx.regime,
            "session": ctx.session_phase,
            "winrate": winrate,
        }
        try:
            req = urllib.request.Request(
                self.webhook,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )
            urllib.request.urlopen(req, timeout=2.0)
        except Exception:
            pass


class EngineManager:
    def __init__(
        self,
        engines: Sequence[StrategyEngine],
        jsonl_path: str,
        log: logging.Logger,
        write_hold: bool = False,
        write_candles: bool = False,
    ):
        self.engines = list(engines)
        self.log = log
        self.jsonl = JsonlAppender(jsonl_path)
        self.write_hold = bool(write_hold)
        self.write_candles = bool(write_candles)

        # Manual scalping mode: emit suggestions only (no order lifecycle simulation).
        # Set STRAT_MODE=sim to restore the previous "armed/fill/exit" simulator.
        self.mode = _env("STRAT_MODE", "manual").strip().lower()
        self.suggest_only = self.mode in ("manual", "suggest", "suggest_only")

        # Skip the first N minutes from market open.
        self.open_warmup_minutes = max(0, _env_int("STRAT_OPEN_WARMUP_MINUTES", 15))
        self.market_open_hhmm = _env("MARKET_OPEN_IST", "09:15")
        self.market_open_h, self.market_open_m = _parse_hhmm(self.market_open_hhmm, default_h=9, default_m=15)

        self.context_detector = MarketContextDetector()
        self.arbiter = SignalArbiter()
        self.risk = RiskManager()

        self.tick_size = _env_float("TICK_SIZE", 0.05)
        self.default_ttl_bars = max(1, _env_int("SETUP_TTL_BARS", 2))
        self.default_max_hold_bars = max(1, _env_int("SETUP_MAX_HOLD_BARS", 3))
        self.tp1_r = _env_float("SETUP_TP1_R", 1.2)
        self.tp2_r = _env_float("SETUP_TP2_R", 2.8)

        self.tp1_r_trend = _env_float("SETUP_TP1_R_TREND", 1.2)
        self.tp2_r_trend = _env_float("SETUP_TP2_R_TREND", 2.8)
        self.ttl_trend = max(1, _env_int("SETUP_TTL_BARS_TREND", 3))
        self.hold_trend = max(1, _env_int("SETUP_MAX_HOLD_BARS_TREND", 6))

        self.tp1_r_range = _env_float("SETUP_TP1_R_RANGE", 1.0)
        self.tp2_r_range = _env_float("SETUP_TP2_R_RANGE", 1.6)
        self.ttl_range = max(1, _env_int("SETUP_TTL_BARS_RANGE", 2))
        self.hold_range = max(1, _env_int("SETUP_MAX_HOLD_BARS_RANGE", 3))

        self.tp1_r_chop = _env_float("SETUP_TP1_R_CHOP", 1.0)
        self.tp2_r_chop = _env_float("SETUP_TP2_R_CHOP", 1.3)
        self.ttl_chop = max(1, _env_int("SETUP_TTL_BARS_CHOP", 2))
        self.hold_chop = max(1, _env_int("SETUP_MAX_HOLD_BARS_CHOP", 2))

        self.sl_atr_mult = _env_float("SETUP_SL_ATR_MULT", 1.35)
        self.sl_buffer_atr_frac = _env_float("SETUP_SL_BUFFER_ATR_FRAC", 0.25)
        self.pa_sl_buffer_atr_frac = _env_float("PA_SL_BUFFER_ATR_FRAC", 0.10)
        self.regime_override_conf = _clamp(_env_float("SETUP_REGIME_OVERRIDE_CONF", 0.90), 0.5, 0.99)

        self.pending: Optional[PendingOrder] = None
        self.open_pos: Optional[OpenPosition] = None
        self.candle_idx = 0
        self.signal_seq = 0

        # Simulator realism knobs.
        self.sim_slippage_ticks = max(0, _env_int("SIM_SLIPPAGE_TICKS", 1))
        self.sim_spread_ticks = max(0, _env_int("SIM_SPREAD_TICKS", 0))
        self.sim_intrabar_model = _env("SIM_INTRABAR_MODEL", "ohlc_path").strip().lower()
        if self.sim_intrabar_model not in ("ohlc_path", "conservative", "optimistic"):
            self.sim_intrabar_model = "ohlc_path"

        # Optional suggestion notifier + online win stats.
        self.notifier = Notifier()
        self.win_stats = WinStatsStore()

    @staticmethod
    def _normalize_decisions(result: EngineOutput) -> List[EngineDecision]:
        if result is None:
            return []
        if isinstance(result, EngineDecision):
            return [result]
        return [d for d in result if isinstance(d, EngineDecision)]

    def _log_event(self, kind: str, candle: Candle1m, **payload: Any) -> None:
        meta = {
            "kind": kind,
            "event_ts": datetime.now(IST_TZ).isoformat(),
            "candle_start_ist": candle.start_ist,
            "candle_end_ist": candle.end_ist,
        }

        # Keep key order intentional for high-frequency rows in JSONL.
        if kind == "DECISION":
            priority = ("decision", "mode", "reason", "candidate_count", "suppressed")
            row: Dict[str, Any] = {}
            for k in priority:
                if k in payload:
                    row[k] = payload[k]
            row.update(meta)
            for k, v in payload.items():
                if k not in row:
                    row[k] = v
        elif kind == "ENGINE_SIGNAL":
            priority = ("engine", "signal", "stop_price", "confidence", "reason", "entry_type")
            row = {}
            for k in priority:
                if k in payload:
                    row[k] = payload[k]
            row.update(meta)
            for k, v in payload.items():
                if k not in row:
                    row[k] = v
        else:
            row = dict(meta)
            row.update(payload)
        self.jsonl.append(row)

    def _compatible_regimes(self, engine: str) -> List[str]:
        mapping = {
            "momentum": ["TREND_UP", "TREND_DOWN"],
            "channel_breakout": ["TREND_UP", "TREND_DOWN", "RANGE", "VOLATILE_CHOP"],
            "pivot_extension": ["RANGE", "VOLATILE_CHOP"],
            "keltner": ["TREND_UP", "TREND_DOWN"],
            "consecutive": ["RANGE", "VOLATILE_CHOP"],
            "price_action": ["TREND_UP", "TREND_DOWN", "RANGE", "VOLATILE_CHOP", "UNCLEAR"],
        }
        return mapping.get(engine, ["TREND_UP", "TREND_DOWN", "RANGE", "VOLATILE_CHOP", "UNCLEAR"])

    def _rr_profile(self, ctx: MarketContext) -> tuple[float, Optional[float], int, int]:
        if ctx.regime in ("TREND_UP", "TREND_DOWN"):
            return self.tp1_r_trend, self.tp2_r_trend, self.hold_trend, self.ttl_trend
        if ctx.regime == "RANGE":
            return self.tp1_r_range, self.tp2_r_range, self.hold_range, self.ttl_range
        if ctx.regime == "VOLATILE_CHOP":
            tp2 = self.tp2_r_chop if self.tp2_r_chop > self.tp1_r_chop else None
            return self.tp1_r_chop, tp2, self.hold_chop, self.ttl_chop
        return self.tp1_r, self.tp2_r, self.default_max_hold_bars, self.default_ttl_bars

    def _build_setup(self, dec: EngineDecision, candle: Candle1m, ctx: MarketContext) -> Optional[TradeSetup]:
        if dec.signal not in ("READY_LONG", "READY_SHORT"):
            return None

        direction = "LONG" if dec.signal.endswith("LONG") else "SHORT"
        compat = self._compatible_regimes(dec.engine)
        tags = list(dec.rationale_tags) if dec.rationale_tags else ["engine_signal", "regime_compatible"]

        if ctx.regime not in compat:
            conf_override = float(dec.confidence or 0.0)
            is_trend_engine = dec.engine in ("momentum", "keltner", "channel_breakout")
            looks_like_breakout = any(("trend" in t) or ("breakout" in t) or ("momentum" in t) for t in tags)
            if not (
                is_trend_engine
                and looks_like_breakout
                and conf_override >= self.regime_override_conf
                and ctx.regime in ("UNCLEAR", "RANGE")
            ):
                return None

        self.signal_seq += 1
        signal_id = f"{dec.engine}_{self.candle_idx}_{self.signal_seq}"

        entry_type = dec.entry_type if dec.entry_type in ("stop", "limit", "market") else "stop"
        if entry_type == "market":
            entry_price = float(candle.close)
        else:
            entry_price = float(dec.stop_price) if isinstance(dec.stop_price, (int, float)) else float(candle.close)

        tp1_r, tp2_r, max_hold_bars, ttl_bars = self._rr_profile(ctx)
        bar_range = max(float(candle.high) - float(candle.low), self.tick_size)
        atr = ctx.atr if (ctx.atr is not None and ctx.atr > 0.0) else bar_range
        atr_base = float(atr) if atr else bar_range

        is_price_action = (dec.engine == "price_action") or any(("u_turn" in t) or ("rejection" in t) or ("dominance" in t) for t in tags)
        buf_frac = self.pa_sl_buffer_atr_frac if is_price_action else self.sl_buffer_atr_frac
        buf = max(self.tick_size, atr_base * float(buf_frac))

        if entry_type == "limit":
            lim_buf = max(self.tick_size, self.sl_atr_mult * atr_base)
            stop_loss = (entry_price - lim_buf) if direction == "LONG" else (entry_price + lim_buf)
        else:
            stop_loss = (float(candle.low) - buf) if direction == "LONG" else (float(candle.high) + buf)

        risk = max(abs(float(entry_price) - float(stop_loss)), self.tick_size)
        if direction == "LONG":
            tp1 = entry_price + float(tp1_r) * risk
            tp2 = (entry_price + float(tp2_r) * risk) if tp2_r is not None else None
        else:
            tp1 = entry_price - float(tp1_r) * risk
            tp2 = (entry_price - float(tp2_r) * risk) if tp2_r is not None else None

        rr = abs(tp1 - entry_price) / max(abs(entry_price - stop_loss), self.tick_size)
        confidence = dec.confidence if dec.confidence is not None else _clamp(ctx.confidence, 0.0, 1.0)

        return TradeSetup(
            signal_id=signal_id,
            ts=str(candle.end_ist),
            engine=dec.engine,
            direction=direction,
            entry_type=entry_type,
            entry_price=float(entry_price),
            stop_loss=float(stop_loss),
            take_profit_1=float(tp1),
            take_profit_2=float(tp2) if tp2 is not None else None,
            max_hold_bars=int(max_hold_bars),
            ttl_bars=int(ttl_bars),
            quality_score=0,
            confidence=float(_clamp(confidence, 0.0, 1.0)),
            risk_reward=float(rr),
            regime=ctx.regime,
            compatible_regimes=compat,
            rationale_tags=tags,
            veto_flags=list(dec.veto_flags),
        )

    def _order_filled(self, setup: TradeSetup, candle: Candle1m) -> bool:
        h = float(candle.high)
        l = float(candle.low)
        e = float(setup.entry_price)

        if setup.entry_type == "market":
            return True

        if setup.direction == "LONG":
            if setup.entry_type == "stop":
                return h >= e
            return l <= e  # limit

        if setup.entry_type == "stop":
            return l <= e
        return h >= e  # limit

    def _execution_price(self, raw_price: float, direction: str, is_entry: bool) -> float:
        """Apply simple spread+slippage model for simulator fills/exits."""
        price = float(raw_price)
        buffer_ticks = float(self.sim_slippage_ticks) + (0.5 * float(self.sim_spread_ticks))
        if buffer_ticks <= 0.0:
            return price
        impact = max(0.0, buffer_ticks) * max(self.tick_size, 1e-6)

        if is_entry:
            return (price + impact) if direction == "LONG" else (price - impact)
        return (price - impact) if direction == "LONG" else (price + impact)

    def _intrabar_path(self, candle: Candle1m) -> List[float]:
        o = float(candle.open)
        h = float(candle.high)
        l = float(candle.low)
        c = float(candle.close)
        if c >= o:
            return [o, l, h, c]
        return [o, h, l, c]

    @staticmethod
    def _first_touch(path: List[float], level: float) -> Optional[int]:
        if len(path) < 2:
            return None
        for i in range(len(path) - 1):
            a = path[i]
            b = path[i + 1]
            lo = a if a < b else b
            hi = b if b > a else a
            if lo <= level <= hi:
                return i
        return None

    def _resolve_ambiguous_exit(self, s: TradeSetup, candle: Candle1m, target: float, target_label: str) -> tuple[str, float]:
        if self.sim_intrabar_model == "optimistic":
            return f"{target_label}_ambiguous", float(target)
        if self.sim_intrabar_model == "conservative":
            return "stop_loss_ambiguous", float(s.stop_loss)

        path = self._intrabar_path(candle)
        tp_touch = self._first_touch(path, float(target))
        sl_touch = self._first_touch(path, float(s.stop_loss))

        if tp_touch is not None and sl_touch is not None:
            if tp_touch < sl_touch:
                return f"{target_label}_ambiguous", float(target)
            return "stop_loss_ambiguous", float(s.stop_loss)
        if tp_touch is not None:
            return f"{target_label}_ambiguous", float(target)
        return "stop_loss_ambiguous", float(s.stop_loss)

    def _evaluate_exit(self, pos: OpenPosition, candle: Candle1m) -> Optional[Dict[str, Any]]:
        s = pos.setup
        h = float(candle.high)
        l = float(candle.low)
        c = float(candle.close)
        target = float(pos.setup.take_profit_2) if (pos.setup.take_profit_2 is not None and pos.setup.regime in ("TREND_UP", "TREND_DOWN")) else float(pos.setup.take_profit_1)
        target_label = "take_profit_2" if (pos.setup.take_profit_2 is not None and pos.setup.regime in ("TREND_UP", "TREND_DOWN")) else "take_profit_1"

        hit_stop = False
        hit_tp1 = False

        if s.direction == "LONG":
            hit_stop = l <= float(s.stop_loss)
            hit_tp1 = h >= target
        else:
            hit_stop = h >= float(s.stop_loss)
            hit_tp1 = l <= target

        bars_open = self.candle_idx - pos.entry_candle_idx + 1

        reason = None
        exit_price_raw = None
        if hit_stop and hit_tp1:
            reason, exit_price_raw = self._resolve_ambiguous_exit(s, candle, target, target_label)
        elif hit_stop:
            reason = "stop_loss"
            exit_price_raw = float(s.stop_loss)
        elif hit_tp1:
            reason = target_label
            exit_price_raw = target
        elif bars_open >= max(1, s.max_hold_bars):
            reason = "max_hold_bars"
            exit_price_raw = c

        if reason is None or exit_price_raw is None:
            return None

        exit_price = self._execution_price(float(exit_price_raw), s.direction, is_entry=False)
        pnl_points = (exit_price - pos.fill_price) if s.direction == "LONG" else (pos.fill_price - exit_price)
        return {
            "reason": reason,
            "exit_price": exit_price,
            "bars_open": bars_open,
            "pnl_points": pnl_points,
        }

    def _advance_lifecycle(self, candle: Candle1m, ctx: MarketContext) -> None:
        # Pending order path.
        if self.pending is not None:
            pending = self.pending
            if self.candle_idx > pending.expires_candle_idx:
                self._log_event(
                    "ORDER_CANCELED",
                    candle,
                    signal_id=pending.setup.signal_id,
                    engine=pending.setup.engine,
                    reason="ttl_expired",
                )
                self.pending = None
            elif self._order_filled(pending.setup, candle):
                self.open_pos = OpenPosition(
                    setup=pending.setup,
                    fill_price=self._execution_price(float(pending.setup.entry_price), pending.setup.direction, is_entry=True),
                    entry_candle_idx=self.candle_idx,
                    entry_session_phase=ctx.session_phase,
                )
                self.pending = None
                self.risk.on_entry()
                self._log_event(
                    "ORDER_FILLED",
                    candle,
                    signal_id=self.open_pos.setup.signal_id,
                    engine=self.open_pos.setup.engine,
                    direction=self.open_pos.setup.direction,
                    fill_price=self.open_pos.fill_price,
                )

        # Open position path.
        if self.open_pos is not None:
            info = self._evaluate_exit(self.open_pos, candle)
            if info is not None:
                closed = self.open_pos
                self.open_pos = None
                self.risk.on_exit(float(info["pnl_points"]), self.candle_idx)
                self.win_stats.record(closed.setup, closed.entry_session_phase, win=(float(info["pnl_points"]) > 0.0))
                self._log_event(
                    "EXIT",
                    candle,
                    signal_id=closed.setup.signal_id,
                    engine=closed.setup.engine,
                    direction=closed.setup.direction,
                    entry_price=closed.fill_price,
                    exit_price=info["exit_price"],
                    reason=info["reason"],
                    bars_open=info["bars_open"],
                    pnl_points=info["pnl_points"],
                )

    def _in_open_warmup(self, candle: Candle1m) -> bool:
        if self.open_warmup_minutes <= 0:
            return False
        try:
            dt_ist = _parse_iso(candle.start_ist).astimezone(IST_TZ)
        except Exception:
            return False
        open_dt = dt_ist.replace(hour=self.market_open_h, minute=self.market_open_m, second=0, microsecond=0)
        allow_dt = open_dt + timedelta(minutes=self.open_warmup_minutes)
        return dt_ist < allow_dt

    def on_candle(self, candle: Candle1m) -> None:
        self.candle_idx += 1

        if self.write_candles:
            self._log_event("CANDLE_CLOSE", candle, **asdict(candle))

        ctx = self.context_detector.on_candle(candle)
        self.risk.on_new_candle(candle)
        if not self.suggest_only:
            self._advance_lifecycle(candle, ctx)

        self._log_event("CONTEXT", candle, context=asdict(ctx))
        if bool(getattr(candle, "is_synthetic", False)):
            decision = "NO_SUGGEST" if self.suggest_only else "NO_TRADE"
            self._log_event("DECISION", candle, decision=decision, mode=self.mode, reason="synthetic_candle", candidate_count=0)
            self.log.info("[DECISION] %s reason=synthetic_candle regime=%s", decision, ctx.regime)
            return

        raw_candidates: List[TradeSetup] = []
        for eng in self.engines:
            try:
                try:
                    decisions = self._normalize_decisions(eng.on_candle(candle, ctx))
                except TypeError:
                    decisions = self._normalize_decisions(eng.on_candle(candle))
            except Exception as e:
                decisions = [
                    EngineDecision(
                        engine=getattr(eng, "name", "unknown"),
                        signal="ERROR",
                        reason=f"{type(e).__name__}: {e}",
                    )
                ]

            for dec in decisions:
                if dec.signal.startswith("FILTER_"):
                    if self.write_hold:
                        self._log_event("ENGINE_SIGNAL", candle, engine=dec.engine, signal=dec.signal, reason=dec.reason)
                    continue

                if (dec.signal == "HOLD") and (not self.write_hold):
                    continue

                self._log_event(
                    "ENGINE_SIGNAL",
                    candle,
                    engine=dec.engine,
                    signal=dec.signal,
                    stop_price=dec.stop_price,
                    confidence=dec.confidence,
                    reason=dec.reason,
                    entry_type=dec.entry_type,
                )

                setup = self._build_setup(dec, candle, ctx)
                if setup is None:
                    continue
                raw_candidates.append(setup)
                self._log_event("CANDIDATE", candle, candidate=asdict(setup))

        # Open warmup: engines can build state, but no suggestions.
        if self._in_open_warmup(candle):
            self._log_event(
                "DECISION",
                candle,
                decision="NO_SUGGEST",
                mode=self.mode,
                reason=f"open_warmup_{self.open_warmup_minutes}m",
                candidate_count=len(raw_candidates),
            )
            self.log.info(
                "[DECISION] NO_SUGGEST reason=open_warmup_%sm regime=%s",
                self.open_warmup_minutes,
                ctx.regime,
            )
            return

        # Manual mode: always produce best-effort suggestions.
        if self.suggest_only:
            arb = self.arbiter.choose(raw_candidates, ctx)
            if arb.winner is None:
                self._log_event(
                    "DECISION",
                    candle,
                    decision="NO_SUGGEST",
                    mode=self.mode,
                    reason=arb.reason,
                    candidate_count=len(raw_candidates),
                    suppressed=arb.suppressed,
                )
                self.log.info("[DECISION] NO_SUGGEST reason=%s regime=%s", arb.reason, ctx.regime)
                return

            winner = arb.winner
            wr = self.win_stats.winrate(winner, ctx)
            if self.notifier.should_send(winner, wr):
                self.notifier.send(winner, ctx, wr)
            self._log_event(
                "DECISION",
                candle,
                decision="SUGGEST",
                mode=self.mode,
                reason=arb.reason,
                winner=asdict(winner),
                suppressed=arb.suppressed,
            )
            self.log.info(
                "[DECISION] SUGGEST %s %s %s entry=%.2f sl=%.2f tp1=%.2f q=%s regime=%s",
                winner.signal_id,
                winner.engine,
                winner.direction,
                winner.entry_price,
                winner.stop_loss,
                winner.take_profit_1,
                winner.quality_score,
                ctx.regime,
            )
            return

        # Simulator mode: retain risk gate + order lifecycle behavior.
        gate = self.risk.gate(self.candle_idx, ctx, has_open_position=(self.open_pos is not None))
        if not gate.allowed:
            self._log_event(
                "DECISION",
                candle,
                decision="NO_TRADE",
                mode=self.mode,
                reason=gate.reason,
                candidate_count=len(raw_candidates),
            )
            self.log.info("[DECISION] NO_TRADE reason=%s regime=%s", gate.reason, ctx.regime)
            return

        arb = self.arbiter.choose(raw_candidates, ctx)
        if arb.winner is None:
            self._log_event(
                "DECISION",
                candle,
                decision="NO_TRADE",
                mode=self.mode,
                reason=arb.reason,
                candidate_count=len(raw_candidates),
                suppressed=arb.suppressed,
            )
            self.log.info("[DECISION] NO_TRADE reason=%s regime=%s", arb.reason, ctx.regime)
            return

        winner = arb.winner
        if self.pending is not None:
            self._log_event(
                "ORDER_CANCELED",
                candle,
                signal_id=self.pending.setup.signal_id,
                engine=self.pending.setup.engine,
                reason="replaced_by_new_winner",
            )

        self.pending = PendingOrder(
            setup=winner,
            armed_candle_idx=self.candle_idx,
            expires_candle_idx=self.candle_idx + max(1, int(winner.ttl_bars)),
        )

        self._log_event(
            "DECISION",
            candle,
            decision="TRADE",
            mode=self.mode,
            reason=arb.reason,
            winner=asdict(winner),
            suppressed=arb.suppressed,
        )
        self._log_event(
            "ORDER_ARMED",
            candle,
            signal_id=winner.signal_id,
            engine=winner.engine,
            direction=winner.direction,
            entry_type=winner.entry_type,
            entry_price=winner.entry_price,
            stop_loss=winner.stop_loss,
            take_profit_1=winner.take_profit_1,
            take_profit_2=winner.take_profit_2,
            ttl_bars=winner.ttl_bars,
            max_hold_bars=winner.max_hold_bars,
            quality_score=winner.quality_score,
            confidence=winner.confidence,
            rationale_tags=winner.rationale_tags,
        )

        self.log.info(
            "[DECISION] TRADE %s %s %s entry=%.2f sl=%.2f tp1=%.2f q=%s regime=%s",
            winner.signal_id,
            winner.engine,
            winner.direction,
            winner.entry_price,
            winner.stop_loss,
            winner.take_profit_1,
            winner.quality_score,
            ctx.regime,
        )


# ----------------------------
# Relay object (plugs into dhan.run_quote_feed as "tb")
# ----------------------------

class MultiEngineRelay:
    def __init__(self, mgr: EngineManager, candle_builder: OneMinuteCandleBuilder, log: logging.Logger):
        self.mgr = mgr
        self.cb = candle_builder
        self.log = log
        self._tick_seen = 0

    def on_tick(self, tick: Dict[str, Any]) -> None:
        try:
            if not isinstance(tick, dict):
                return
            if tick.get("kind") != "dhan_quote_packet":
                return

            ts = str(tick.get("timestamp") or "")
            dt_ist = _feed_ts_to_ist(ts)

            price = float(tick.get("ltp") or 0.0)
            vol_raw = tick.get("volume")
            volume = int(vol_raw) if vol_raw is not None else None

            self._tick_seen += 1
            closed = self.cb.on_tick(dt_ist, price, volume)

            for candle in closed:
                self.mgr.on_candle(candle)

        except Exception as e:
            self.log.warning("[RELAY] tick handling error: %s", e)


# ----------------------------
# Orchestrator
# ----------------------------

def _build_engines(log: Optional[logging.Logger] = None) -> List[StrategyEngine]:
    tick_size = _env_float("TICK_SIZE", 0.05)
    builders: Dict[str, Any] = {
        "momentum": lambda: MomentumEngine(length=_env_int("MOM_LENGTH", 12), tick_size=tick_size),
        "keltner": lambda: KeltnerChannelsEngine(
            length=_env_int("KELTNER_LENGTH", 20),
            mult=_env_float("KELTNER_MULT", 2.0),
            use_exp=_env_bool("KELTNER_USE_EXP", True),
            bands_style=_env("KELTNER_BANDS_STYLE", "Average True Range"),
            atr_length=_env_int("KELTNER_ATR_LENGTH", 10),
            tick_size=tick_size,
        ),
        "macd": lambda: MACDEngine(
            fast_length=_env_int("MACD_FAST", 12),
            slow_length=_env_int("MACD_SLOW", 26),
            macd_length=_env_int("MACD_SIGNAL", 9),
        ),
        "consecutive": lambda: ConsecutiveUpDownEngine(
            consecutive_bars_up=_env_int("CONS_BARS_UP", 3),
            consecutive_bars_down=_env_int("CONS_BARS_DOWN", 3),
        ),
        "channel_breakout": lambda: ChannelBreakOutEngine(
            length=_env_int("CHBRK_LENGTH", 5),
            tick_size=tick_size,
        ),
        "pivot_extension": lambda: PivotExtensionEngine(
            left_bars=_env_int("PIVOT_LEFT", 4),
            right_bars=_env_int("PIVOT_RIGHT", 2),
        ),
        "price_action": lambda: PriceActionEngine(tick_size=tick_size),
    }

    if _env_bool("LIST_ENGINES", False):
        print("Available engines:", ", ".join(sorted(builders.keys())))
        raise SystemExit(0)

    enabled = [s.strip().lower() for s in _env("ENABLE_ENGINES", "momentum").split(",") if s.strip()]
    unknown = [name for name in enabled if name not in builders]
    if unknown and log is not None:
        log.warning("[SYS] unknown engines requested (ignored): %s", unknown)
        log.warning("[SYS] available engines: %s", sorted(builders.keys()))

    engines: List[StrategyEngine] = []
    for name in enabled:
        builder = builders.get(name)
        if builder is not None:
            engines.append(builder())
    return engines


async def run_system() -> None:
    cfg = dhan.DhanConfig(
        exchange_segment=getattr(dhan, "_env", lambda k, d, aliases=None: _env(k, d))("EXCHANGE_SEGMENT", "IDX_I", aliases=["FULL_EXCHANGE_SEGMENT"]) or "IDX_I",
        security_id=getattr(dhan, "_env_int", lambda k, d, aliases=None: _env_int(k, d))("SECURITY_ID", 13, aliases=["FULL_SECURITY_ID"]),
        ws_ping_interval=getattr(dhan, "_env_int", lambda k, d, aliases=None: _env_int(k, d))("WS_PING_INTERVAL", 30, aliases=["FULL_WS_PING_INTERVAL"]),
        ws_ping_timeout=getattr(dhan, "_env_int", lambda k, d, aliases=None: _env_int(k, d))("WS_PING_TIMEOUT", 20, aliases=["FULL_WS_PING_TIMEOUT"]),
        reconnect_delay_base=getattr(dhan, "_env_float", lambda k, d, aliases=None: _env_float(k, d))("RECONNECT_BASE", 2.0, aliases=["FULL_RECONNECT_BASE"]),
        reconnect_delay_cap=getattr(dhan, "_env_float", lambda k, d, aliases=None: _env_float(k, d))("RECONNECT_CAP", 60.0, aliases=["FULL_RECONNECT_CAP"]),
        max_reconnect_attempts=getattr(dhan, "_env_int", lambda k, d, aliases=None: _env_int(k, d))("MAX_RECONNECTS", 0, aliases=["FULL_MAX_RECONNECTS"]),
        log_every_n=getattr(dhan, "_env_int", lambda k, d, aliases=None: _env_int(k, d))("LOG_EVERY_N", 500, aliases=["FULL_LOG_EVERY_N"]),
        log_ticks=getattr(dhan, "_env_bool", lambda k, d, aliases=None: _env_bool(k, d))("LOG_TICKS", False, aliases=["FULL_LOG_TICKS"]),
        log_tick_summary_sec=getattr(dhan, "_env_int", lambda k, d, aliases=None: _env_int(k, d))("LOG_TICK_SUMMARY_SEC", 600, aliases=["FULL_LOG_TICK_SUMMARY_SEC"]),
    )

    log_file = getattr(dhan, "_env", lambda k, d, aliases=None: _env(k, d))("LOG_FILE", "logs/scalp_engines.log", aliases=["FULL_LOG_FILE"])
    log_level = getattr(dhan, "_parse_log_level", lambda d=logging.INFO: logging.INFO)(logging.INFO)
    log_to_console = getattr(dhan, "_env_bool", lambda k, d, aliases=None: _env_bool(k, d))("LOG_TO_CONSOLE", False, aliases=["FULL_LOG_TO_CONSOLE"])
    log = dhan.setup_logging(log_file=log_file, level=log_level, also_console=log_to_console)

    engines = _build_engines(log=log)
    if not engines:
        raise SystemExit("No engines enabled. Set ENABLE_ENGINES=... or LIST_ENGINES=1")

    jsonl_path = _env("STRAT_JSONL", "logs/strategy_signals.jsonl")
    write_hold = _env_bool("STRAT_WRITE_HOLD", False)
    write_candles = _env_bool("STRAT_WRITE_CANDLES", False)
    gap_fill = _env_bool("STRAT_GAP_FILL", False)
    max_gap_fill = _env_int("STRAT_GAP_FILL_MAX_MINUTES", 3)

    log.info("[SYS] engines=%s", [getattr(e, "name", "unknown") for e in engines])
    log.info("[SYS] jsonl=%s write_hold=%s write_candles=%s gap_fill=%s", jsonl_path, write_hold, write_candles, gap_fill)
    if not write_candles:
        log.warning("[SYS] STRAT_WRITE_CANDLES is disabled; TP/SL outcome validation from JSONL will be limited.")

    mgr = EngineManager(
        engines=engines,
        jsonl_path=jsonl_path,
        log=log,
        write_hold=write_hold,
        write_candles=write_candles,
    )
    relay = MultiEngineRelay(
        mgr=mgr,
        candle_builder=OneMinuteCandleBuilder(gap_fill=gap_fill, max_gap_fill_minutes=max_gap_fill),
        log=log,
    )

    await dhan.run_quote_feed(cfg, relay, log)


if __name__ == "__main__":
    try:
        asyncio.run(run_system())
    except KeyboardInterrupt:
        pass
