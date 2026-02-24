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
import math
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


def _is_finite(x: Optional[float]) -> bool:
    return x is not None and isinstance(x, (int, float)) and math.isfinite(float(x))


def _safe_float(x: Optional[float], default: float = 0.0) -> float:
    return float(x) if _is_finite(x) else float(default)


def _safe_div(num: float, den: float, default: float = 0.0) -> float:
    d = float(den)
    if not math.isfinite(d) or abs(d) <= 1e-12:
        return float(default)
    n = float(num)
    if not math.isfinite(n):
        return float(default)
    return n / d


def _sigmoid(x: float) -> float:
    z = _safe_float(x, 0.0)
    if z >= 35.0:
        return 1.0
    if z <= -35.0:
        return 0.0
    return 1.0 / (1.0 + math.exp(-z))


def _softmax(values: Sequence[float]) -> List[float]:
    if not values:
        return []
    finite_vals = [_safe_float(v, 0.0) for v in values]
    vmax = max(finite_vals)
    exps = [math.exp(v - vmax) for v in finite_vals]
    total = sum(exps)
    if total <= 0.0 or not math.isfinite(total):
        n = float(len(exps))
        return [1.0 / n for _ in exps]
    return [v / total for v in exps]


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
    regime_probs: Dict[str, float] = field(default_factory=dict)
    trend_strength: float = 0.0
    transition_risk: float = 1.0
    trend_age: int = 0
    exhaustion_bull: float = 0.0
    exhaustion_bear: float = 0.0


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
        self.adx_len = _env_int("REGIME_ADX_LENGTH", 9)
        self.regime_unclear_threshold = _clamp(_env_float("REGIME_UNCLEAR_THRESHOLD", 0.45), 0.30, 0.70)
        self.state_stay_prob = _clamp(_env_float("REGIME_STATE_STAY_PROB", 0.84), 0.55, 0.97)
        self.state_flip_prob = _clamp(_env_float("REGIME_STATE_FLIP_PROB", 0.03), 0.01, 0.20)
        self.regime_ema_fast_len = max(2, _env_int("REGIME_EMA_FAST", 8))
        self.regime_ema_mid_len = max(3, _env_int("REGIME_EMA_MID", 20))
        self.regime_ema_slow_len = max(5, _env_int("REGIME_EMA_SLOW", 50))
        self.exhaust_mom_len = max(2, _env_int("EXH_MOM_LENGTH", 6))

        self.adx = ADX(self.adx_len)
        self.atr14 = ATR(14)
        self.atr_sma20 = SMA(20)
        self.ema_fast = EMA(self.regime_ema_fast_len)
        self.ema_mid = EMA(self.regime_ema_mid_len)
        self.ema_slow = EMA(self.regime_ema_slow_len)
        self.ema20 = EMA(20)
        self.prev_ema_fast: Optional[float] = None
        self.prev_ema_mid: Optional[float] = None
        self.prev_ema_slow: Optional[float] = None

        self.macd_fast = EMA(_env_int("MACD_FAST", 12))
        self.macd_slow = EMA(_env_int("MACD_SLOW", 26))
        self.macd_signal = EMA(_env_int("MACD_SIGNAL", 9))

        self._last_ctx: Optional[MarketContext] = None
        self._states: Sequence[str] = ("TREND_UP", "TREND_DOWN", "RANGE", "VOLATILE_CHOP")
        self.posterior: Dict[str, float] = {
            "TREND_UP": 0.25,
            "TREND_DOWN": 0.25,
            "RANGE": 0.25,
            "VOLATILE_CHOP": 0.25,
        }
        self.prev_di_norm: float = 0.0
        self.prev_close_for_run: Optional[float] = None
        self.prev_mom0: Optional[float] = None
        self.run_up = 0
        self.run_down = 0
        self.trend_age = 0
        self.last_trend_label = "UNCLEAR"

        self.recent_highs: Deque[float] = deque(maxlen=20)
        self.recent_lows: Deque[float] = deque(maxlen=20)
        self.recent_bars: Deque[Candle1m] = deque(maxlen=24)
        self.close_hist: Deque[float] = deque(maxlen=128)

    def _session_phase(self, candle: Candle1m) -> str:
        dt_ist = _parse_iso(candle.start_ist).astimezone(IST_TZ)
        mins = dt_ist.hour * 60 + dt_ist.minute
        if (9 * 60 + 15) <= mins < (10 * 60 + 45):
            return "OPEN_GO"
        if (11 * 60 + 30) <= mins < (14 * 60):
            return "MIDDAY"
        return "LATE"

    def _structure_scores(self) -> tuple[float, float]:
        if len(self.recent_bars) < 4:
            return 0.0, 0.0
        hhhl = 0
        lhll = 0
        pairs = 0
        bars = list(self.recent_bars)
        for i in range(1, len(bars)):
            prev = bars[i - 1]
            cur = bars[i]
            pairs += 1
            if float(cur.high) > float(prev.high) and float(cur.low) > float(prev.low):
                hhhl += 1
            if float(cur.high) < float(prev.high) and float(cur.low) < float(prev.low):
                lhll += 1
        if pairs <= 0:
            return 0.0, 0.0
        return float(hhhl) / float(pairs), float(lhll) / float(pairs)

    def _transition_matrix(self) -> Dict[str, Dict[str, float]]:
        stay = self.state_stay_prob
        flip = self.state_flip_prob
        residual = max(0.0, 1.0 - stay - flip)
        to_range = residual * 0.55
        to_chop = residual * 0.45
        trend_residual = max(0.0, 1.0 - stay)
        return {
            "TREND_UP": {
                "TREND_UP": stay,
                "TREND_DOWN": flip,
                "RANGE": to_range,
                "VOLATILE_CHOP": to_chop,
            },
            "TREND_DOWN": {
                "TREND_UP": flip,
                "TREND_DOWN": stay,
                "RANGE": to_range,
                "VOLATILE_CHOP": to_chop,
            },
            "RANGE": {
                "TREND_UP": trend_residual * 0.35,
                "TREND_DOWN": trend_residual * 0.35,
                "RANGE": stay,
                "VOLATILE_CHOP": trend_residual * 0.30,
            },
            "VOLATILE_CHOP": {
                "TREND_UP": trend_residual * 0.30,
                "TREND_DOWN": trend_residual * 0.30,
                "RANGE": trend_residual * 0.40,
                "VOLATILE_CHOP": stay,
            },
        }

    def _emission_probs(
        self,
        adx_v: Optional[float],
        plus_di: Optional[float],
        minus_di: Optional[float],
        atr_ratio: Optional[float],
        s_fast: float,
        s_mid: float,
        s_slow: float,
    ) -> Dict[str, float]:
        adx_n = _clamp(_safe_div(_safe_float(adx_v, 0.0) - 15.0, 25.0, 0.0), 0.0, 1.0)
        plus = _safe_float(plus_di, 0.0)
        minus = _safe_float(minus_di, 0.0)
        di_norm = _safe_div(plus - minus, plus + minus + 1e-6, 0.0)
        di_mag = abs(di_norm)
        vexp = _clamp(_safe_div(_safe_float(atr_ratio, 1.0) - 1.0, 0.8, 0.0), 0.0, 1.0)
        vctr = _clamp(_safe_div(1.0 - _safe_float(atr_ratio, 1.0), 0.4, 0.0), 0.0, 1.0)

        salign = 1.0 if ((s_fast >= 0 and s_mid >= 0 and s_slow >= 0) or (s_fast <= 0 and s_mid <= 0 and s_slow <= 0)) else 0.0
        hhhl, lhll = self._structure_scores()
        trend_power = (0.85 * adx_n) + (0.35 * salign) + (0.30 * di_mag)

        up_logit = -0.20 + trend_power + (0.85 * max(di_norm, 0.0)) + (0.35 * max(s_fast, 0.0)) + (0.25 * hhhl) - (0.20 * lhll)
        dn_logit = -0.20 + trend_power + (0.85 * max(-di_norm, 0.0)) + (0.35 * max(-s_fast, 0.0)) + (0.25 * lhll) - (0.20 * hhhl)
        range_logit = -0.10 + (0.95 * (1.0 - adx_n)) + (0.45 * vctr) + (0.30 * (1.0 - min(1.0, abs(s_mid)))) + (0.25 * (1.0 - di_mag))
        chop_logit = -0.25 + (0.90 * vexp) + (0.40 * (1.0 - adx_n)) + (0.20 * (1.0 - salign))

        probs = _softmax([up_logit, dn_logit, range_logit, chop_logit])
        return {
            "TREND_UP": probs[0],
            "TREND_DOWN": probs[1],
            "RANGE": probs[2],
            "VOLATILE_CHOP": probs[3],
        }

    def _posterior_probs(self, emission: Dict[str, float]) -> Dict[str, float]:
        trans = self._transition_matrix()
        states = self._states
        pred: Dict[str, float] = {}
        for s_to in states:
            pred_val = 0.0
            for s_from in states:
                pred_val += _safe_float(self.posterior.get(s_from), 0.25) * _safe_float(trans.get(s_from, {}).get(s_to), 0.0)
            pred[s_to] = pred_val

        raw: Dict[str, float] = {}
        total = 0.0
        for s in states:
            val = max(0.0, pred.get(s, 0.0) * emission.get(s, 0.0))
            raw[s] = val
            total += val
        if total <= 0.0 or not math.isfinite(total):
            return {s: 1.0 / float(len(states)) for s in states}
        return {s: raw[s] / total for s in states}

    def _decide_regime(self, posterior: Dict[str, float], di_norm: float) -> str:
        top_state = max(self._states, key=lambda s: posterior.get(s, 0.0))
        top_prob = _safe_float(posterior.get(top_state), 0.0)
        if top_prob < self.regime_unclear_threshold:
            return "UNCLEAR"
        if top_state == "TREND_UP" and di_norm <= 0.0:
            return "UNCLEAR"
        if top_state == "TREND_DOWN" and di_norm >= 0.0:
            return "UNCLEAR"
        return top_state

    def _update_exhaustion(
        self,
        candle: Candle1m,
        atr_v: float,
        di_norm: float,
        close_ema20: float,
    ) -> tuple[float, float]:
        close = float(candle.close)
        open_ = float(candle.open)
        high = float(candle.high)
        low = float(candle.low)
        bar_range = max(high - low, 1e-6)
        atr = max(atr_v, 1e-6)

        if self.prev_close_for_run is None:
            self.prev_close_for_run = close
        if close > self.prev_close_for_run:
            self.run_up += 1
            self.run_down = 0
        elif close < self.prev_close_for_run:
            self.run_down += 1
            self.run_up = 0
        self.prev_close_for_run = close

        self.close_hist.append(close)
        mom0 = 0.0
        if len(self.close_hist) > self.exhaust_mom_len:
            mom0 = close - self.close_hist[-1 - self.exhaust_mom_len]
        mom1 = 0.0
        if self.prev_mom0 is not None:
            mom1 = mom0 - self.prev_mom0
        self.prev_mom0 = mom0

        ext = _safe_div(close - close_ema20, atr, 0.0)
        decel_up = _clamp(_safe_div(-mom1, atr, 0.0), 0.0, 3.0)
        decel_down = _clamp(_safe_div(mom1, atr, 0.0), 0.0, 3.0)
        di_comp = _clamp(abs(self.prev_di_norm) - abs(di_norm), 0.0, 1.0)
        self.prev_di_norm = di_norm

        upper_wick = _clamp(_safe_div(high - max(open_, close), bar_range, 0.0), 0.0, 1.0)
        lower_wick = _clamp(_safe_div(min(open_, close) - low, bar_range, 0.0), 0.0, 1.0)
        run_up_n = _clamp(_safe_div(float(self.run_up), 5.0, 0.0), 0.0, 1.0)
        run_dn_n = _clamp(_safe_div(float(self.run_down), 5.0, 0.0), 0.0, 1.0)

        fail_up = 0
        fail_dn = 0
        bars = list(self.recent_bars)
        for i in range(1, len(bars)):
            prev = bars[i - 1]
            cur = bars[i]
            rng = max(float(cur.high) - float(cur.low), 1e-6)
            cpos = _safe_div(float(cur.close) - float(cur.low), rng, 0.5)
            if float(cur.high) > float(prev.high) and cpos < 0.75:
                fail_up += 1
            if float(cur.low) < float(prev.low) and cpos > 0.25:
                fail_dn += 1
        fail_up_n = _clamp(_safe_div(float(fail_up), 5.0, 0.0), 0.0, 1.0)
        fail_dn_n = _clamp(_safe_div(float(fail_dn), 5.0, 0.0), 0.0, 1.0)

        z_bear = -2.2 + (0.90 * max(ext, 0.0)) + (0.80 * run_up_n) + (0.70 * decel_up) + (0.50 * di_comp) + (0.60 * upper_wick) + (0.60 * fail_up_n)
        z_bull = -2.2 + (0.90 * max(-ext, 0.0)) + (0.80 * run_dn_n) + (0.70 * decel_down) + (0.50 * di_comp) + (0.60 * lower_wick) + (0.60 * fail_dn_n)
        return _sigmoid(z_bull), _sigmoid(z_bear)

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
                regime_probs=dict(last.regime_probs),
                trend_strength=last.trend_strength,
                transition_risk=min(1.0, last.transition_risk + 0.08),
                trend_age=last.trend_age,
                exhaustion_bull=last.exhaustion_bull,
                exhaustion_bear=last.exhaustion_bear,
            )
            self._last_ctx = ctx
            return ctx

        self.recent_bars.append(candle)
        adx_v, plus_di, minus_di = self.adx.update(candle.high, candle.low, candle.close)
        atr_v = self.atr14.update(candle.high, candle.low, candle.close)
        atr_mean = self.atr_sma20.update(_safe_float(atr_v, 0.0))
        atr_ratio = _safe_div(_safe_float(atr_v, 0.0), _safe_float(atr_mean, 0.0), 1.0) if _is_finite(atr_v) and _is_finite(atr_mean) else None

        ema_fast = self.ema_fast.update(candle.close)
        ema_mid = self.ema_mid.update(candle.close)
        ema_slow = self.ema_slow.update(candle.close)
        ema20 = self.ema20.update(candle.close)

        s_fast = _safe_div(ema_fast - _safe_float(self.prev_ema_fast, ema_fast), max(_safe_float(atr_v, 0.0), 1e-6), 0.0)
        s_mid = _safe_div(ema_mid - _safe_float(self.prev_ema_mid, ema_mid), max(_safe_float(atr_v, 0.0), 1e-6), 0.0)
        s_slow = _safe_div(ema_slow - _safe_float(self.prev_ema_slow, ema_slow), max(_safe_float(atr_v, 0.0), 1e-6), 0.0)
        self.prev_ema_fast = ema_fast
        self.prev_ema_mid = ema_mid
        self.prev_ema_slow = ema_slow

        plus = _safe_float(plus_di, 0.0)
        minus = _safe_float(minus_di, 0.0)
        di_norm = _safe_div(plus - minus, plus + minus + 1e-6, 0.0)
        emission = self._emission_probs(adx_v, plus_di, minus_di, atr_ratio, s_fast, s_mid, s_slow)
        posterior = self._posterior_probs(emission)
        self.posterior = posterior
        regime = self._decide_regime(posterior, di_norm)

        vol_state = "normal"
        if atr_ratio is not None and atr_ratio >= 1.25:
            vol_state = "expanded"
        elif atr_ratio is not None and atr_ratio <= 0.80:
            vol_state = "contracted"

        macd_line = self.macd_fast.update(candle.close) - self.macd_slow.update(candle.close)
        signal_line = self.macd_signal.update(macd_line)
        delta = macd_line - signal_line
        macd_bias = "neutral"
        macd_eps = max(_safe_float(atr_v, abs(float(candle.close)) * 0.001), 1e-6) * 0.02
        if delta > macd_eps:
            macd_bias = "long"
        elif delta < -macd_eps:
            macd_bias = "short"

        p_up = _safe_float(posterior.get("TREND_UP"), 0.0)
        p_dn = _safe_float(posterior.get("TREND_DOWN"), 0.0)
        trend_strength = max(p_up, p_dn)
        transition_risk = 1.0 - max(_safe_float(posterior.get(s), 0.0) for s in self._states)

        bias = "neutral"
        if p_up - p_dn > 0.12:
            bias = "long"
        elif p_dn - p_up > 0.12:
            bias = "short"

        if regime in ("TREND_UP", "TREND_DOWN"):
            if self.last_trend_label == regime:
                self.trend_age += 1
            else:
                self.trend_age = 1
            self.last_trend_label = regime
        else:
            self.trend_age = 0
            self.last_trend_label = "UNCLEAR"

        confidence = _clamp(
            (0.55 * trend_strength)
            + (0.25 * (1.0 - transition_risk))
            + (0.20 * min(1.0, abs(di_norm))),
            0.0,
            1.0,
        )

        exhaustion_bull, exhaustion_bear = self._update_exhaustion(
            candle=candle,
            atr_v=max(_safe_float(atr_v, 0.0), 1e-6),
            di_norm=di_norm,
            close_ema20=ema20,
        )

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
            slope=s_mid,
            confidence=_clamp(confidence, 0.0, 1.0),
            macd_bias=macd_bias,
            regime_probs=dict(posterior),
            trend_strength=_clamp(trend_strength, 0.0, 1.0),
            transition_risk=_clamp(transition_risk, 0.0, 1.0),
            trend_age=max(0, self.trend_age),
            exhaustion_bull=_clamp(exhaustion_bull, 0.0, 1.0),
            exhaustion_bear=_clamp(exhaustion_bear, 0.0, 1.0),
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
        self.min_strength_atr_frac = _env_float("MOM_MIN_STRENGTH_ATR_FRAC", 0.12)
        self.close_acceptance = _clamp(_env_float("MOM_CLOSE_ACCEPTANCE", 0.57), 0.5, 0.95)
        self.decel_tolerance_frac = _clamp(_env_float("MOM_DECEL_TOL_FRAC", 0.25), 0.0, 1.0)

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
        decel_tol = self.decel_tolerance_frac * min_strength

        if mom0 > min_strength and mom1 > (-decel_tol) and long_close_ok:
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

        if mom0 < -min_strength and mom1 < decel_tol and short_close_ok:
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
        self.u_turn_min_exhaust = _clamp(_env_float("PA_U_TURN_MIN_EXHAUST", 0.65), 0.30, 0.95)
        self.u_turn_max_trend_strength = _clamp(_env_float("PA_U_TURN_MAX_TREND_STRENGTH", 0.75), 0.30, 0.98)
        self.u_turn_min_transition_risk = _clamp(_env_float("PA_U_TURN_MIN_TRANSITION_RISK", 0.30), 0.05, 0.95)

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
        ex_bull = _safe_float(context.exhaustion_bull if context else None, 0.0)
        ex_bear = _safe_float(context.exhaustion_bear if context else None, 0.0)
        trisk = _safe_float(context.transition_risk if context else None, 1.0)
        tstr = _safe_float(context.trend_strength if context else None, 0.0)
        u_turn_context_ok = trisk >= self.u_turn_min_transition_risk and tstr <= self.u_turn_max_trend_strength

        if bear(c1) and is_doji2 and bull(c3) and float(c3.close) >= c1_mid and ex_bull >= self.u_turn_min_exhaust and u_turn_context_ok and near_low:
            return EngineDecision(
                engine=self.name,
                signal="READY_LONG",
                stop_price=float(c3.high) + self.tick_size,
                confidence=_clamp(0.62 + (0.20 * ex_bull) + (0.10 * trisk), 0.0, 1.0),
                entry_type="stop",
                rationale_tags=["u_turn_bullish"],
            )

        if bull(c1) and is_doji2 and bear(c3) and float(c3.close) <= c1_mid and ex_bear >= self.u_turn_min_exhaust and u_turn_context_ok and near_high:
            return EngineDecision(
                engine=self.name,
                signal="READY_SHORT",
                stop_price=float(c3.low) - self.tick_size,
                confidence=_clamp(0.62 + (0.20 * ex_bear) + (0.10 * trisk), 0.0, 1.0),
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
        self.min_score_midday = max(0.0, _env_float("ARBITER_MIN_SCORE_MIDDAY", 74.0))
        self.min_score_unclear = max(0.0, _env_float("ARBITER_MIN_SCORE_UNCLEAR", 72.0))
        self.conflict_band = _env_float("ARBITER_CONFLICT_BAND", 5.0)
        self.min_pwin = _clamp(_env_float("ARBITER_MIN_P_WIN", 0.55), 0.30, 0.90)
        self.min_ev = _env_float("ARBITER_MIN_EV", 0.10)
        self.hard_countertrend_block_strength = _clamp(_env_float("ARBITER_CT_BLOCK_TREND_STRENGTH", 0.70), 0.45, 0.95)
        self.hard_countertrend_exhaust = _clamp(_env_float("ARBITER_CT_MIN_EXHAUST", 0.75), 0.50, 0.98)
        self.u_turn_swing_atr_max = _clamp(_env_float("ARBITER_U_TURN_SWING_ATR_MAX", 0.35), 0.10, 1.50)
        self.reliability_lookup: Optional[Any] = None

    def set_reliability_lookup(self, fn: Optional[Any]) -> None:
        self.reliability_lookup = fn

    def _effective_min_score(self, ctx: MarketContext) -> float:
        floor = float(self.min_score)
        if ctx.session_phase == "MIDDAY":
            floor = max(floor, float(self.min_score_midday))
        if ctx.regime == "UNCLEAR":
            floor = max(floor, float(self.min_score_unclear))
        return floor

    @staticmethod
    def _is_countertrend_pattern(setup: TradeSetup) -> bool:
        tags = setup.rationale_tags or []
        return any(("u_turn" in t) or ("rejection" in t) or ("dominance" in t) for t in tags)

    @staticmethod
    def _dir_align(setup: TradeSetup, ctx: MarketContext) -> int:
        if ctx.bias == "neutral":
            return 0
        if (ctx.bias == "long" and setup.direction == "LONG") or (ctx.bias == "short" and setup.direction == "SHORT"):
            return 1
        return -1

    @staticmethod
    def _directional_regime_prob(setup: TradeSetup, ctx: MarketContext) -> float:
        probs = ctx.regime_probs or {}
        if setup.direction == "LONG":
            return _safe_float(probs.get("TREND_UP"), 0.0)
        return _safe_float(probs.get("TREND_DOWN"), 0.0)

    @staticmethod
    def _dist_swing_atr(setup: TradeSetup, ctx: MarketContext) -> float:
        atr = max(_safe_float(ctx.atr, 0.0), 1e-6)
        if setup.direction == "LONG":
            anchor = _safe_float(ctx.key_levels.get("swing_low"), setup.entry_price)
        else:
            anchor = _safe_float(ctx.key_levels.get("swing_high"), setup.entry_price)
        return abs(float(setup.entry_price) - anchor) / atr

    def _reliability_prior(self, setup: TradeSetup, ctx: MarketContext) -> float:
        if self.reliability_lookup is None:
            return 0.50
        try:
            val = float(self.reliability_lookup(setup, ctx))
            if not math.isfinite(val):
                return 0.50
            return _clamp(val, 0.05, 0.95)
        except Exception:
            return 0.50

    def _p_win(self, setup: TradeSetup, ctx: MarketContext) -> float:
        dir_align = self._dir_align(setup, ctx)
        dir_prob = self._directional_regime_prob(setup, ctx)
        transition_risk = _safe_float(ctx.transition_risk, 1.0)
        trend_strength = _safe_float(ctx.trend_strength, 0.0)
        dist_swing_atr = self._dist_swing_atr(setup, ctx)
        near_swing = 1.0 if dist_swing_atr <= self.u_turn_swing_atr_max else 0.0
        ex = _safe_float(ctx.exhaustion_bull if setup.direction == "LONG" else ctx.exhaustion_bear, 0.0)
        rel = self._reliability_prior(setup, ctx)
        tags = setup.rationale_tags or []
        is_u_turn = 1.0 if any("u_turn" in t for t in tags) else 0.0
        is_rej = 1.0 if any("rejection" in t for t in tags) else 0.0
        is_dom = 1.0 if any("dominance" in t for t in tags) else 0.0

        z = -0.40
        z += 1.10 * _safe_float(setup.confidence, 0.0)
        z += 0.65 * float(dir_align)
        z += 0.85 * dir_prob
        z += 0.60 * (rel - 0.50)
        z += 0.45 * ex
        z += 0.30 * (1.0 - transition_risk)
        z += 0.25 * is_dom * max(0.0, float(dir_align))
        z += 0.15 * is_rej * near_swing
        z += 0.30 * is_u_turn * near_swing
        z += 0.20 * is_u_turn * transition_risk
        z -= 0.35 * is_u_turn * (1.0 - near_swing)
        z -= 0.30 * max(0.0, trend_strength - 0.80) * (1.0 if dir_align < 0 else 0.0)
        return _sigmoid(z)

    def _ev(self, setup: TradeSetup, p_win: float) -> float:
        rr = max(float(setup.risk_reward), 0.0)
        return (p_win * rr) - (1.0 - p_win)

    def _hard_block_reason(self, setup: TradeSetup, ctx: MarketContext) -> Optional[str]:
        tags = setup.rationale_tags or []
        dir_align = self._dir_align(setup, ctx)
        ex = _safe_float(ctx.exhaustion_bull if setup.direction == "LONG" else ctx.exhaustion_bear, 0.0)
        trend_strength = _safe_float(ctx.trend_strength, 0.0)
        transition_risk = _safe_float(ctx.transition_risk, 1.0)
        dist_swing_atr = self._dist_swing_atr(setup, ctx)

        if (
            dir_align < 0
            and self._is_countertrend_pattern(setup)
            and trend_strength >= self.hard_countertrend_block_strength
            and (ex < self.hard_countertrend_exhaust or transition_risk < 0.25)
        ):
            return "blocked_countertrend_strong_trend"

        if any("u_turn" in t for t in tags) and dist_swing_atr > self.u_turn_swing_atr_max:
            return "blocked_u_turn_far_from_swing"
        return None

    def score(self, setup: TradeSetup, ctx: MarketContext) -> tuple[float, float, float, Optional[str]]:
        hard_block = self._hard_block_reason(setup, ctx)
        if hard_block is not None:
            return 0.0, 0.0, -1.0, hard_block

        p_win = self._p_win(setup, ctx)
        ev = self._ev(setup, p_win)
        ev_norm = _clamp(_safe_div(ev + 1.0, 2.0, 0.0), 0.0, 1.0)
        score = 100.0 * ((0.60 * p_win) + (0.40 * ev_norm))
        return _clamp(score, 0.0, 100.0), p_win, ev, None

    def choose(self, setups: List[TradeSetup], ctx: MarketContext) -> ArbiterDecision:
        if not setups:
            return ArbiterDecision(winner=None, reason="no_candidates")

        scored: List[tuple[float, float, float, TradeSetup]] = []
        suppressed: List[Dict[str, Any]] = []
        for s in setups:
            score, p_win, ev, hard_block = self.score(s, ctx)
            s.quality_score = int(round(score))
            if hard_block is not None:
                suppressed.append(
                    {
                        "signal_id": s.signal_id,
                        "engine": s.engine,
                        "direction": s.direction,
                        "score": score,
                        "p_win": p_win,
                        "ev": ev,
                        "reason": hard_block,
                    }
                )
                continue
            if p_win < self.min_pwin or ev < self.min_ev:
                suppressed.append(
                    {
                        "signal_id": s.signal_id,
                        "engine": s.engine,
                        "direction": s.direction,
                        "score": score,
                        "p_win": p_win,
                        "ev": ev,
                        "reason": "low_pwin_or_ev",
                    }
                )
                continue
            scored.append((score, p_win, ev, s))

        if not scored:
            return ArbiterDecision(winner=None, reason="no_viable_candidates", suppressed=suppressed)

        scored.sort(key=lambda x: x[0], reverse=True)
        top_score, top_pwin, top_ev, top_setup = scored[0]
        min_required = self._effective_min_score(ctx)

        if top_score < min_required:
            return ArbiterDecision(
                winner=None,
                reason=f"low_quality score={top_score:.1f}<min={min_required:.1f}",
                suppressed=(
                    suppressed
                    + [
                        {"signal_id": s.signal_id, "engine": s.engine, "direction": s.direction, "score": sc, "p_win": pw, "ev": ev}
                        for sc, pw, ev, s in scored
                    ]
                ),
            )

        if len(scored) >= 2:
            second_score, _, _, second_setup = scored[1]
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
                    suppressed=(
                        suppressed
                        + [
                            {"signal_id": s.signal_id, "engine": s.engine, "direction": s.direction, "score": sc, "p_win": pw, "ev": ev}
                            for sc, pw, ev, s in scored
                        ]
                    ),
                )

        kept = [
            {"signal_id": s.signal_id, "engine": s.engine, "direction": s.direction, "score": sc, "p_win": pw, "ev": ev}
            for sc, pw, ev, s in scored
            if s.signal_id != top_setup.signal_id
        ]
        return ArbiterDecision(winner=top_setup, reason=f"selected p_win={top_pwin:.2f} ev={top_ev:.2f}", suppressed=(suppressed + kept))


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


class TickRecorder:
    def __init__(self, path: str, log: logging.Logger, strict: bool = False):
        self.path = path
        self.log = log
        self.strict = bool(strict)
        self.seq = 0
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    def append(self, tick: Dict[str, Any]) -> None:
        self.seq += 1
        row = {
            "seq": self.seq,
            "recv_ts_ist": datetime.now(IST_TZ).isoformat(),
            "tick": tick,
        }
        try:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, separators=(",", ":")) + "\n")
        except Exception as e:
            self.log.error("[TICK_RECORDER] write failed seq=%s path=%s err=%s", self.seq, self.path, e)
            if self.strict:
                raise


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

    @staticmethod
    def tag_key(engine: str, tag: str, regime: str, session_phase: str) -> str:
        return f"{engine}|{tag}:{regime}:{session_phase}"

    @staticmethod
    def global_key(engine: str, tag: str) -> str:
        return f"{engine}|{tag}:ALL:ALL"

    def _flush(self) -> None:
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, separators=(",", ":"))

    def _record_key(self, k: str, win: bool) -> None:
        rec = self.data.get(k, {"wins": 0, "total": 0})
        rec["total"] += 1
        if win:
            rec["wins"] += 1
        self.data[k] = rec

    def record(self, setup: TradeSetup, session_phase: str, win: bool) -> None:
        k = self.key(setup.engine, setup.regime, session_phase)
        self._record_key(k, win)
        for tag in (setup.rationale_tags or []):
            self._record_key(self.tag_key(setup.engine, str(tag), setup.regime, session_phase), win)
            self._record_key(self.global_key(setup.engine, str(tag)), win)
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

    def bayesian_reliability(self, setup: TradeSetup, ctx: MarketContext, alpha: float = 3.0, beta: float = 3.0) -> float:
        tags = setup.rationale_tags or []
        if not tags:
            return 0.50

        priors: List[float] = []
        for tag in tags:
            local_key = self.tag_key(setup.engine, str(tag), ctx.regime, ctx.session_phase)
            global_key = self.global_key(setup.engine, str(tag))
            local = self.data.get(local_key, {"wins": 0, "total": 0})
            glob = self.data.get(global_key, {"wins": 0, "total": 0})

            lw = float(local.get("wins", 0))
            lt = float(local.get("total", 0))
            gw = float(glob.get("wins", 0))
            gt = float(glob.get("total", 0))

            local_rel = _safe_div(lw + alpha, lt + alpha + beta, 0.50)
            global_rel = _safe_div(gw + alpha, gt + alpha + beta, 0.50)
            lam = _clamp(_safe_div(lt, 30.0, 0.0), 0.0, 1.0)
            priors.append((lam * local_rel) + ((1.0 - lam) * global_rel))

        if not priors:
            return 0.50
        return _clamp(sum(priors) / float(len(priors)), 0.05, 0.95)


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
        self.win_stats = WinStatsStore()
        self.arbiter = SignalArbiter()
        self.arbiter.set_reliability_lookup(self.win_stats.bayesian_reliability)
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

        # Optional suggestion notifier.
        self.notifier = Notifier()

        # Human-friendly scalper action logging.
        self.scalper_friendly_log = _env_bool("SCALPER_FRIENDLY_LOG", True)

        # Selective relaxation for consecutive signals in UNCLEAR regimes.
        self.consec_allow_unclear = _env_bool("SETUP_CONSEC_ALLOW_UNCLEAR", True)
        self.consec_unclear_max_adx = _env_float("SETUP_CONSEC_UNCLEAR_MAX_ADX", 24.0)
        self.consec_unclear_block_midday = _env_bool("SETUP_CONSEC_UNCLEAR_BLOCK_MIDDAY", True)

        # Optional one-bar-early lead alert (notify-only guidance, not an auto-trade).
        self.lead_alert_enabled = _env_bool("LEAD_ALERT_ENABLED", True)
        self.lead_alert_cooldown_bars = max(0, _env_int("LEAD_ALERT_COOLDOWN_BARS", 1))
        self.lead_min_adx = _env_float("LEAD_MIN_ADX", 24.0)
        self.lead_min_di_spread = _env_float("LEAD_MIN_DI_SPREAD", 10.0)
        self.lead_min_conf = _env_float("LEAD_MIN_CONF", 0.45)
        self.lead_block_midday = _env_bool("LEAD_BLOCK_MIDDAY", True)
        self._last_lead_candle_idx = -10_000_000
        self.refire_cooldown_bars = max(0, _env_int("SETUP_REFIRE_COOLDOWN_BARS", 2))
        self.refire_pullback_atr = _env_float("SETUP_REFIRE_PULLBACK_ATR", 0.60)
        self.last_winner_direction: Optional[str] = None
        self.last_winner_entry: Optional[float] = None
        self.last_winner_candle_idx: int = -10_000_000

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

    @staticmethod
    def _scalper_action(entry_type: str, direction: str) -> str:
        side = "BUY" if direction == "LONG" else "SELL"
        if entry_type == "market":
            return f"{side} NOW at market"
        if entry_type == "limit":
            return f"PLACE {side} LIMIT at entry"
        return f"PLACE {side} STOP at entry"

    def _log_scalper_suggestion(self, setup: TradeSetup, ctx: MarketContext, decision: str) -> None:
        if not self.scalper_friendly_log:
            return
        tp2_txt = "-" if setup.take_profit_2 is None else f"{float(setup.take_profit_2):.2f}"
        action = self._scalper_action(setup.entry_type, setup.direction)
        self.log.info(
            "[SCALPER] %s | %s NIFTY50 | entry=%.2f | sl=%.2f | tp1=%.2f | tp2=%s | q=%s | regime=%s | hold<=%sb",
            decision,
            action,
            setup.entry_price,
            setup.stop_loss,
            setup.take_profit_1,
            tp2_txt,
            setup.quality_score,
            ctx.regime,
            setup.max_hold_bars,
        )

    def _log_scalper_wait(self, reason: str, ctx: MarketContext) -> None:
        if not self.scalper_friendly_log:
            return
        self.log.info(
            "[SCALPER] WAIT | no trade setup | reason=%s | regime=%s | bias=%s | conf=%.2f",
            reason,
            ctx.regime,
            ctx.bias,
            ctx.confidence,
        )

    def _lead_alert_signal(self, ctx: MarketContext) -> Optional[tuple[str, float, str]]:
        if not self.lead_alert_enabled:
            return None
        if ctx.regime != "UNCLEAR":
            return None
        if self.lead_block_midday and ctx.session_phase == "MIDDAY":
            return None
        if self.candle_idx <= (self._last_lead_candle_idx + self.lead_alert_cooldown_bars):
            return None
        if (
            ctx.adx is None
            or ctx.plus_di is None
            or ctx.minus_di is None
            or ctx.confidence is None
            or ctx.macd_bias not in ("long", "short")
        ):
            return None
        if float(ctx.adx) < float(self.lead_min_adx):
            return None
        if float(ctx.confidence) < float(self.lead_min_conf):
            return None

        if ctx.macd_bias == "long":
            spread = float(ctx.plus_di) - float(ctx.minus_di)
            if spread >= float(self.lead_min_di_spread):
                return "LONG", spread, "pretrend_long"
        else:
            spread = float(ctx.minus_di) - float(ctx.plus_di)
            if spread >= float(self.lead_min_di_spread):
                return "SHORT", spread, "pretrend_short"
        return None

    def _emit_lead_alert(self, candle: Candle1m, ctx: MarketContext) -> None:
        lead = self._lead_alert_signal(ctx)
        if lead is None:
            return
        direction, di_spread, reason = lead
        self._last_lead_candle_idx = self.candle_idx
        self._log_event(
            "LEAD_ALERT",
            candle,
            direction=direction,
            reason=reason,
            di_spread=di_spread,
            confidence=ctx.confidence,
            adx=ctx.adx,
            regime=ctx.regime,
            session_phase=ctx.session_phase,
            entry_hint=float(candle.close),
        )
        if self.scalper_friendly_log:
            self.log.info(
                "[SCALPER] LEAD_ALERT | %s BIAS building | watch breakout next 1m | last=%.2f | di_spread=%.2f | adx=%.2f | conf=%.2f",
                direction,
                float(candle.close),
                di_spread,
                float(ctx.adx or 0.0),
                float(ctx.confidence),
            )

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

        if dec.engine == "consecutive" and ctx.regime == "UNCLEAR" and self.consec_allow_unclear:
            adx_ok = (ctx.adx is None) or (float(ctx.adx) <= float(self.consec_unclear_max_adx))
            session_ok = (not self.consec_unclear_block_midday) or (ctx.session_phase != "MIDDAY")
            if adx_ok and session_ok:
                if "UNCLEAR" not in compat:
                    compat = list(compat) + ["UNCLEAR"]
                if "unclear_range_like" not in tags:
                    tags.append("unclear_range_like")

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

        bar_range = max(float(candle.high) - float(candle.low), self.tick_size)
        atr = ctx.atr if (ctx.atr is not None and ctx.atr > 0.0) else bar_range
        atr_base = float(atr) if atr else bar_range
        trend_strength = _safe_float(ctx.trend_strength, 0.0)
        transition_risk = _safe_float(ctx.transition_risk, 1.0)
        phase_factor = 1.0 if ctx.session_phase == "OPEN_GO" else (-1.0 if ctx.session_phase == "MIDDAY" else 0.0)

        if (
            self.last_winner_direction is not None
            and self.last_winner_entry is not None
            and direction == self.last_winner_direction
            and (self.candle_idx - self.last_winner_candle_idx) <= self.refire_cooldown_bars
        ):
            pullback_depth = (
                _safe_div(self.last_winner_entry - float(candle.low), atr_base, 0.0)
                if direction == "LONG"
                else _safe_div(float(candle.high) - self.last_winner_entry, atr_base, 0.0)
            )
            if pullback_depth < float(self.refire_pullback_atr):
                return None

        self.signal_seq += 1
        signal_id = f"{dec.engine}_{self.candle_idx}_{self.signal_seq}"

        entry_type = dec.entry_type if dec.entry_type in ("stop", "limit", "market") else "stop"
        if entry_type != "market":
            continuation_score = (0.55 * trend_strength) + (0.30 * (1.0 - transition_risk))
            continuation_score += 0.20 * (_safe_float(dec.confidence, 0.0))
            continuation_score += 0.10 if any(("breakout" in t) or ("momentum" in t) for t in tags) else 0.0

            pullback_score = (0.45 * transition_risk) + (0.25 * _safe_float(ctx.exhaustion_bull if direction == "LONG" else ctx.exhaustion_bear, 0.0))
            pullback_score += 0.15 * (1.0 if any(("rejection" in t) or ("u_turn" in t) for t in tags) else 0.0)
            pullback_score += 0.10 * max(0.0, _safe_float(ctx.atr_ratio, 1.0) - 1.0)
            if (pullback_score - continuation_score) >= 0.15:
                entry_type = "limit"
            elif (continuation_score - pullback_score) >= 0.15:
                entry_type = "stop"

        if entry_type == "market":
            entry_price = float(candle.close)
        elif entry_type == "limit":
            retrace = max(self.tick_size, 0.25 * atr_base)
            entry_price = float(candle.close) - retrace if direction == "LONG" else float(candle.close) + retrace
        else:
            entry_price = float(dec.stop_price) if isinstance(dec.stop_price, (int, float)) else float(candle.close)

        tp1_r, tp2_r, max_hold_bars_profile, ttl_bars_profile = self._rr_profile(ctx)
        ttl_dyn = int(round(2.0 + (2.5 * trend_strength) - (2.0 * transition_risk) + phase_factor - max(0.0, _safe_float(ctx.atr_ratio, 1.0) - 1.4)))
        hold_dyn = int(round(4.0 + (4.0 * trend_strength) + (2.0 * max(0.0, abs(_safe_float(ctx.slope, 0.0)))) - max(0.0, _safe_float(ctx.atr_ratio, 1.0) - 1.5)))
        ttl_bars = int(_clamp(float(ttl_dyn), 2.0, 8.0))
        max_hold_bars = int(_clamp(float(hold_dyn), 3.0, 12.0))
        ttl_bars = max(int(ttl_bars_profile), ttl_bars)
        max_hold_bars = max(int(max_hold_bars_profile), max_hold_bars)

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
            self._log_scalper_wait("synthetic_candle", ctx)
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
            warmup_reason = f"open_warmup_{self.open_warmup_minutes}m"
            self._log_event(
                "DECISION",
                candle,
                decision="NO_SUGGEST",
                mode=self.mode,
                reason=warmup_reason,
                candidate_count=len(raw_candidates),
            )
            self.log.info(
                "[DECISION] NO_SUGGEST reason=open_warmup_%sm regime=%s",
                self.open_warmup_minutes,
                ctx.regime,
            )
            self._log_scalper_wait(warmup_reason, ctx)
            return

        # Manual mode: always produce best-effort suggestions.
        if self.suggest_only:
            arb = self.arbiter.choose(raw_candidates, ctx)
            if arb.winner is None:
                self._emit_lead_alert(candle, ctx)
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
                self._log_scalper_wait(arb.reason, ctx)
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
            self._log_scalper_suggestion(winner, ctx, decision="SUGGEST")
            self.last_winner_direction = winner.direction
            self.last_winner_entry = float(winner.entry_price)
            self.last_winner_candle_idx = int(self.candle_idx)
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
            self._log_scalper_wait(gate.reason, ctx)
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
            self._log_scalper_wait(arb.reason, ctx)
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
        self._log_scalper_suggestion(winner, ctx, decision="TRADE")
        self.last_winner_direction = winner.direction
        self.last_winner_entry = float(winner.entry_price)
        self.last_winner_candle_idx = int(self.candle_idx)


# ----------------------------
# Relay object (plugs into dhan.run_quote_feed as "tb")
# ----------------------------

class MultiEngineRelay:
    def __init__(self, mgr: EngineManager, candle_builder: OneMinuteCandleBuilder, log: logging.Logger, tick_recorder: Optional[TickRecorder] = None):
        self.mgr = mgr
        self.cb = candle_builder
        self.log = log
        self.tick_recorder = tick_recorder
        self._tick_seen = 0

    def on_tick(self, tick: Dict[str, Any]) -> None:
        try:
            if not isinstance(tick, dict):
                return
            if tick.get("kind") != "dhan_quote_packet":
                return

            if self.tick_recorder is not None:
                self.tick_recorder.append(tick)

            ts = str(tick.get("timestamp") or "")
            dt_ist = _feed_ts_to_ist(ts)

            price = float(tick.get("ltp") or 0.0)
            if not math.isfinite(price) or price <= 0.0:
                self.log.warning("[RELAY] dropped tick with invalid ltp=%r ts=%s", tick.get("ltp"), ts)
                return
            vol_raw = tick.get("volume")
            try:
                volume = int(vol_raw) if vol_raw is not None else None
            except Exception:
                self.log.warning("[RELAY] volume parse failed volume=%r ts=%s", vol_raw, ts)
                volume = None

            self._tick_seen += 1
            closed = self.cb.on_tick(dt_ist, price, volume)

            for candle in closed:
                self.mgr.on_candle(candle)

        except Exception as e:
            self.log.warning("[RELAY] tick handling error: %s", e)


# ----------------------------
# Orchestrator
# ----------------------------

def _validate_runtime_config(log: logging.Logger) -> None:
    errors: List[str] = []
    warnings: List[str] = []

    tick_size = _env_float("TICK_SIZE", 0.05)
    if tick_size <= 0.0:
        errors.append("TICK_SIZE must be > 0")

    if _env_int("SETUP_MAX_HOLD_BARS", 3) < 1:
        errors.append("SETUP_MAX_HOLD_BARS must be >= 1")
    if _env_int("SETUP_TTL_BARS", 2) < 1:
        errors.append("SETUP_TTL_BARS must be >= 1")

    min_score = _env_float("ARBITER_MIN_SCORE", 70.0)
    min_score_midday = _env_float("ARBITER_MIN_SCORE_MIDDAY", 74.0)
    min_score_unclear = _env_float("ARBITER_MIN_SCORE_UNCLEAR", 72.0)
    for k, v in (("ARBITER_MIN_SCORE", min_score), ("ARBITER_MIN_SCORE_MIDDAY", min_score_midday), ("ARBITER_MIN_SCORE_UNCLEAR", min_score_unclear)):
        if not (0.0 <= v <= 100.0):
            errors.append(f"{k} must be in [0,100]")

    min_p_win = _env_float("ARBITER_MIN_P_WIN", 0.55)
    if not (0.0 <= min_p_win <= 1.0):
        errors.append("ARBITER_MIN_P_WIN must be in [0,1]")

    state_stay = _env_float("REGIME_STATE_STAY_PROB", 0.84)
    state_flip = _env_float("REGIME_STATE_FLIP_PROB", 0.03)
    if state_stay + state_flip >= 1.0:
        errors.append("REGIME_STATE_STAY_PROB + REGIME_STATE_FLIP_PROB must be < 1")

    u_turn_swing = _env_float("ARBITER_U_TURN_SWING_ATR_MAX", 0.35)
    if u_turn_swing <= 0.0:
        errors.append("ARBITER_U_TURN_SWING_ATR_MAX must be > 0")

    warmup = _env_int("STRAT_OPEN_WARMUP_MINUTES", 15)
    if warmup < 0:
        errors.append("STRAT_OPEN_WARMUP_MINUTES must be >= 0")
    if warmup > 120:
        warnings.append("STRAT_OPEN_WARMUP_MINUTES is very high; no signals may appear for long.")

    if errors:
        for msg in errors:
            log.error("[CONFIG] %s", msg)
        raise SystemExit("Invalid configuration. Fix [CONFIG] errors and rerun.")
    for msg in warnings:
        log.warning("[CONFIG] %s", msg)


def _build_engines(log: Optional[logging.Logger] = None) -> List[StrategyEngine]:
    tick_size = _env_float("TICK_SIZE", 0.05)
    builders: Dict[str, Any] = {
        "momentum": lambda: MomentumEngine(length=_env_int("MOM_LENGTH", 10), tick_size=tick_size),
        "keltner": lambda: KeltnerChannelsEngine(
            length=_env_int("KELTNER_LENGTH", 20),
            mult=_env_float("KELTNER_MULT", 1.6),
            use_exp=_env_bool("KELTNER_USE_EXP", True),
            bands_style=_env("KELTNER_BANDS_STYLE", "Average True Range"),
            atr_length=_env_int("KELTNER_ATR_LENGTH", 8),
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
    _validate_runtime_config(log)

    engines = _build_engines(log=log)
    if not engines:
        raise SystemExit("No engines enabled. Set ENABLE_ENGINES=... or LIST_ENGINES=1")

    jsonl_path = _env("STRAT_JSONL", "logs/strategy_signals.jsonl")
    write_hold = _env_bool("STRAT_WRITE_HOLD", True)
    write_candles = _env_bool("STRAT_WRITE_CANDLES", True)
    gap_fill = _env_bool("STRAT_GAP_FILL", False)
    max_gap_fill = _env_int("STRAT_GAP_FILL_MAX_MINUTES", 3)

    log.info("[SYS] engines=%s", [getattr(e, "name", "unknown") for e in engines])
    log.info("[SYS] jsonl=%s write_hold=%s write_candles=%s gap_fill=%s", jsonl_path, write_hold, write_candles, gap_fill)
    if not write_candles:
        log.warning("[SYS] STRAT_WRITE_CANDLES is disabled; TP/SL outcome validation from JSONL will be limited.")

    tick_recorder: Optional[TickRecorder] = None
    if _env_bool("STRAT_TICK_LOG_ENABLED", False):
        tick_path_tmpl = _env("STRAT_TICK_LOG_PATH", "logs/ticks_{date}.ndjson")
        date_token = datetime.now(IST_TZ).strftime("%Y-%m-%d")
        tick_path = tick_path_tmpl.replace("{date}", date_token)
        tick_recorder = TickRecorder(
            path=tick_path,
            log=log,
            strict=_env_bool("STRAT_TICK_LOG_STRICT", False),
        )
        log.info("[SYS] tick_recorder enabled path=%s", tick_path)

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
        tick_recorder=tick_recorder,
    )

    await dhan.run_quote_feed(cfg, relay, log)


if __name__ == "__main__":
    try:
        asyncio.run(run_system())
    except KeyboardInterrupt:
        pass
