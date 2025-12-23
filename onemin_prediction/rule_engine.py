#!/usr/bin/env python3
"""Rule engine shared by live and training (single source of truth)."""
from __future__ import annotations

import logging
import os
from collections import Counter, deque
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def _safe_getenv_bool(key: str, default: bool = False) -> bool:
    val = os.getenv(key)
    if val is None:
        return bool(default)
    return str(val).strip().lower() in ("1", "true", "yes", "y", "on")


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        if np.isfinite(v):
            return v
    except Exception:
        pass
    return float(default)


def compute_flow_signal(
    features: Mapping[str, Any],
    *,
    log: bool = False,
    mutate: bool = True,
) -> Tuple[float, int, float, float, float, int]:
    """Compute flow score/side from micro imbalance + futures CVD + VWAP deviation."""
    def _get(k: str, default: float = 0.0) -> float:
        try:
            v = float(features.get(k, default))
            return v if np.isfinite(v) else float(default)
        except Exception:
            return float(default)

    micro_imb = _get('micro_imbalance', _get('imbalance', 0.0))
    fut_cvd = _get('fut_cvd_delta', 0.0)
    fut_vwap = _get('fut_vwap_dev', 0.0)
    fut_vol = max(0.0, _get('fut_vol_delta', 0.0))

    try:
        vwap_ext = float(os.getenv('FLOW_VWAP_EXT', '0.0020') or '0.0020')
    except Exception:
        vwap_ext = 0.0020
    try:
        cvd_min = float(os.getenv('FLOW_CVD_MIN', '2.5e-05') or '2.5e-05')
    except Exception:
        cvd_min = 2.5e-05

    if fut_vwap > +vwap_ext:
        vwap_side = 1
    elif fut_vwap < -vwap_ext:
        vwap_side = -1
    else:
        vwap_side = 0

    strong_bear_regime = (vwap_side == -1) and (micro_imb <= 0.0) and (fut_cvd <= +cvd_min)
    strong_bull_regime = (vwap_side == +1) and (micro_imb >= 0.0) and (fut_cvd >= -cvd_min)

    cvd_term = 0.0
    if cvd_min > 0:
        cvd_term = float(np.tanh(fut_cvd / cvd_min))

    vwap_term = 0.0
    if vwap_ext > 0:
        vwap_term = float(np.tanh(fut_vwap / vwap_ext))

    if strong_bear_regime:
        flow_score = -1.0 + min(0.0, micro_imb) + 0.25 * cvd_term
        flow_side = -1
        regime = 'BEAR_LOCK'
    elif strong_bull_regime:
        flow_score = +1.0 + max(0.0, micro_imb) + 0.25 * cvd_term
        flow_side = +1
        regime = 'BULL_LOCK'
    else:
        flow_score = (0.55 * vwap_term) + (0.30 * cvd_term) + (0.15 * micro_imb)
        amp = 1.0 + min(0.75, fut_vol)
        flow_score *= amp
        if flow_score > 0:
            flow_side = 1
        elif flow_score < 0:
            flow_side = -1
        else:
            flow_side = 0
        regime = 'BLEND'

    if log:
        logger.debug(
            "[FLOW] micro_imb=%.3f fut_cvd=%.6f fut_vwap_dev=%.4f vwap_side=%+d fut_vol=%.3f "
            "regime=%s → score=%.3f side=%+d",
            micro_imb, fut_cvd, fut_vwap, vwap_side, fut_vol, regime, flow_score, flow_side
        )

    if mutate and isinstance(features, dict):
        features['flow_score'] = float(flow_score)
        features['flow_side'] = int(flow_side)
        features['flow_micro_imb'] = float(micro_imb)
        features['flow_fut_cvd'] = float(fut_cvd)
        features['flow_fut_vwap_dev'] = float(fut_vwap)
        features['flow_vwap_side'] = int(vwap_side)
        features['flow_fut_vol'] = float(fut_vol)
        features['flow_regime'] = str(regime)

    return float(flow_score), int(flow_side), float(micro_imb), float(fut_cvd), float(fut_vwap), int(vwap_side)


def compute_structure_score(features: Mapping[str, Any]) -> Tuple[float, int]:
    """Score structure bias from pivots/FVG/OB features."""
    ps_up = bool(features.get("struct_pivot_swipe_up", 0))
    ps_dn = bool(features.get("struct_pivot_swipe_down", 0))
    fvg_up = bool(features.get("struct_fvg_up_present", 0))
    fvg_dn = bool(features.get("struct_fvg_down_present", 0))
    ob_bull = bool(features.get("struct_ob_bull_present", 0))
    ob_bear = bool(features.get("struct_ob_bear_present", 0))

    bull = ps_up or fvg_up or ob_bull
    bear = ps_dn or fvg_dn or ob_bear

    if bull and not bear:
        return 1.0, 1
    if bear and not bull:
        return -1.0, -1
    return 0.0, 0


def compute_ta_rule_signal(feats: Mapping[str, float]) -> float:
    """
    EMA-trend strict TA rule signal in [-1, +1].
    """
    try:
        ema_fast = float(feats.get("ema_8", 0.0))
        ema_slow = float(feats.get("ema_21", 0.0))
        ema_50 = float(feats.get("ema_50", ema_slow))
        last_px = float(feats.get("last_price", 0.0))

        micro_slope = float(feats.get("micro_slope", 0.0))
        fut_vwap_dev = float(feats.get("fut_vwap_dev", 0.0))

        rsi = float(feats.get("ta_rsi14", 50.0))
        macd_hist = float(feats.get("ta_macd_hist", 0.0))
        bb_bw = float(feats.get("ta_bb_bw", 0.0))
        bb_pctb = float(feats.get("ta_bb_pctb", 0.5))
    except Exception:
        return 0.0

    if not np.isfinite(last_px) or last_px <= 0.0:
        return 0.0

    ema_trend = 0
    if ema_fast > ema_slow:
        ema_trend = 1
    elif ema_fast < ema_slow:
        ema_trend = -1

    score = 0.0

    try:
        slope_min = float(os.getenv("TA_SLOPE_MIN", "0.15"))
    except ValueError:
        slope_min = 0.15

    try:
        vwap_min = float(os.getenv("TA_VWAP_DEV_MIN", "0.0005"))
    except ValueError:
        vwap_min = 0.0005

    if ema_trend > 0 and last_px > max(ema_fast, ema_slow, ema_50):
        score += 0.5
    elif ema_trend < 0 and last_px < min(ema_fast, ema_slow, ema_50):
        score -= 0.5

    if micro_slope > slope_min:
        score += 0.25
    elif micro_slope < -slope_min:
        score -= 0.25

    if fut_vwap_dev > vwap_min:
        score += 0.25
    elif fut_vwap_dev < -vwap_min:
        score -= 0.25

    try:
        rsi_bull = float(os.getenv("TA_RSI_BULL", "60.0"))
        rsi_bear = float(os.getenv("TA_RSI_BEAR", "40.0"))
    except ValueError:
        rsi_bull, rsi_bear = 60.0, 40.0

    if rsi >= rsi_bull:
        score += 0.15
    elif rsi <= rsi_bear:
        score -= 0.15

    if macd_hist > 0.0:
        score += 0.10
    elif macd_hist < 0.0:
        score -= 0.10

    if bb_bw > 0.05:
        if bb_pctb > 0.6:
            score += 0.10
        elif bb_pctb < 0.4:
            score -= 0.10

    if ema_trend > 0 and fut_vwap_dev < 0:
        score *= 0.6
    elif ema_trend < 0 and fut_vwap_dev > 0:
        score *= 0.6

    return float(np.clip(score, -1.0, 1.0))


def compute_rule_hierarchy(
    *,
    name: str,
    rule_sig: float,
    features_raw: Mapping[str, Any],
    mtf: Optional[Mapping[str, Any]],
    is_bull_setup: bool,
    is_bear_setup: bool,
    any_setup: bool,
    ambiguous_setup: bool,
) -> Tuple[str, str, str, List[str]]:
    """Flow > HTF trend > EMA trend > structure > pattern."""
    try:
        rule_min_sig = float(os.getenv("RULE_MIN_SIG", "0.20") or "0.20")
    except Exception:
        rule_min_sig = 0.20

    if not np.isfinite(rule_sig) or abs(rule_sig) < rule_min_sig:
        base_dir = "NA"
        base_side = 0
    else:
        base_dir = "BUY" if rule_sig > 0 else "SELL"
        base_side = 1 if base_dir == "BUY" else -1

    conflict_level = "none"
    conflict_reasons: List[str] = []

    flow_score, flow_side, micro_imb, fut_cvd, fut_vwap, vwap_side = compute_flow_signal(features_raw)

    try:
        mtf_cons = float(mtf.get("mtf_consensus", 0.0)) if mtf else 0.0
        if not np.isfinite(mtf_cons):
            mtf_cons = 0.0
    except Exception:
        mtf_cons = 0.0

    try:
        htf_min = float(os.getenv("HTF_MIN_CONS", "0.10") or "0.10")
    except Exception:
        htf_min = 0.10

    if mtf_cons > htf_min:
        htf_side = 1
    elif mtf_cons < -htf_min:
        htf_side = -1
    else:
        htf_side = 0

    ema_trend = 0
    try:
        ema_fast = float(features_raw.get("ema_9", features_raw.get("ema_8", 0.0)))
        ema_slow = float(features_raw.get("ema_21", 0.0))
        if np.isfinite(ema_fast) and np.isfinite(ema_slow):
            if ema_fast > ema_slow:
                ema_trend = 1
            elif ema_fast < ema_slow:
                ema_trend = -1
    except Exception:
        ema_trend = 0

    trend_side = 1 if ema_trend > 0 else (-1 if ema_trend < 0 else 0)

    if ambiguous_setup:
        struct_side = 0
    else:
        if is_bull_setup and not is_bear_setup:
            struct_side = 1
        elif is_bear_setup and not is_bull_setup:
            struct_side = -1
        else:
            struct_side = 0

    ctx_flow_htf = flow_side or htf_side
    ctx_trend_struct = trend_side or struct_side

    final_side = ctx_flow_htf or ctx_trend_struct or base_side or 0

    if ctx_flow_htf != 0 and ctx_trend_struct != 0 and ctx_flow_htf != ctx_trend_struct:
        conflict_reasons.append("flow_htf_vs_trend_struct")
        align_trend_htf = (trend_side != 0 and htf_side != 0 and trend_side == htf_side)
        weak_counter_flow = (abs(float(flow_score)) < 0.35)

        if align_trend_htf and weak_counter_flow and abs(float(mtf_cons)) >= 0.35:
            conflict_level = "yellow"
        else:
            conflict_level = "red"

        final_side = trend_side or htf_side or flow_side or struct_side or base_side or 0

    if final_side > 0 and not (flow_side > 0 or htf_side > 0):
        conflict_reasons.append("no_flow_htf_approval")
        conflict_level = "red" if conflict_level == "none" else conflict_level
    if final_side < 0 and not (flow_side < 0 or htf_side < 0):
        conflict_reasons.append("no_flow_htf_approval")
        conflict_level = "red" if conflict_level == "none" else conflict_level

    if not any_setup and ctx_flow_htf == 0 and base_side != 0:
        conflict_reasons.append("rule_sig_only_no_setup")
        conflict_level = "red" if conflict_level == "none" else conflict_level

    if final_side > 0:
        rule_dir = "BUY"
    elif final_side < 0:
        rule_dir = "SELL"
    else:
        rule_dir = "FLAT"

    if rule_dir == "BUY" and vwap_side < 0 and flow_side <= 0:
        conflict_reasons.append("vwap_veto_buy")
        conflict_level = "red" if conflict_level == "none" else conflict_level
    if rule_dir == "SELL" and vwap_side > 0 and flow_side >= 0:
        conflict_reasons.append("vwap_veto_sell")
        conflict_level = "red" if conflict_level == "none" else conflict_level

    logger.info(
        "[%s] [RULE-HIER] rule_sig=%.3f base_dir=%s flow_side=%+d htf_side=%+d "
        "trend_side=%+d struct_side=%+d any_setup=%s ambiguous=%s conflict_level=%s reasons=%s → rule_dir=%s",
        name,
        rule_sig,
        base_dir,
        flow_side,
        htf_side,
        trend_side,
        struct_side,
        bool(any_setup),
        bool(ambiguous_setup),
        str(conflict_level),
        ",".join(conflict_reasons) if conflict_reasons else "none",
        rule_dir,
    )

    return rule_dir, base_dir, conflict_level, conflict_reasons


@dataclass
class DecisionSnapshot:
    lane: str
    intent: str
    tradeable: bool
    size_mult: float
    score: float
    edge_required: float
    edge_quantile: float
    trend_strength: float
    trend_signals: int
    conflict: str
    hard_veto: list
    soft_penalties: dict
    muted_only_soft: bool


class DecisionState:
    """Per-connection decision state (rolling edge history + hysteresis + counters)."""

    def __init__(self, name: str, edge_window: int = 300, edge_pctl: float = 0.85, hyst_bars: int = 3, summary_every: int = 50):
        self.name = str(name)
        self.edge_window = max(30, int(edge_window))
        self.edge_pctl = float(edge_pctl)
        self.hyst_bars = max(0, int(hyst_bars))
        self.summary_every = max(10, int(summary_every))

        self._edge = {
            1: deque(maxlen=self.edge_window),
            -1: deque(maxlen=self.edge_window),
        }
        self._trend_hold = 0
        self._trend_side = 0
        self._last_lane = 'NONE'

        self._counts = Counter()
        self._decisions = 0
        self._session_date = None

    def update_edge(self, side: int, strength: float) -> None:
        if side not in (1, -1):
            return
        try:
            m = float(strength)
        except Exception:
            return
        if not (m >= 0.0):
            return
        self._edge[side].append(m)

    def edge_threshold(self, side: int, *, min_edge: float) -> tuple:
        """Return (required_edge, quantile_edge)."""
        min_edge = float(min_edge)
        hist = self._edge.get(side, None)
        if not hist or len(hist) < 30:
            return max(min_edge, 0.0), 0.0
        try:
            arr = sorted(float(x) for x in hist if x is not None)
            if len(arr) < 30:
                return max(min_edge, 0.0), 0.0
            q = min(0.99, max(0.50, float(self.edge_pctl)))
            k = int(round(q * (len(arr) - 1)))
            k = max(0, min(len(arr) - 1, k))
            thr = float(arr[k])
            return max(min_edge, thr), thr
        except Exception:
            return max(min_edge, 0.0), 0.0

    def bump_counts(self, snap: DecisionSnapshot) -> None:
        self._decisions += 1
        self._counts[f"lane:{snap.lane}"] += 1
        self._counts[f"intent:{snap.intent}"] += 1
        self._counts[f"tradeable:{int(bool(snap.tradeable))}"] += 1
        if bool(getattr(snap, 'muted_only_soft', False)):
            self._counts['muted_only_soft'] += 1
        for r in (snap.hard_veto or []):
            self._counts[f"hard:{r}"] += 1
        for r in (snap.soft_penalties or {}).keys():
            self._counts[f"soft:{r}"] += 1

    def maybe_log_summary(self, ts: 'datetime') -> None:
        try:
            d = ts.date()
        except Exception:
            return
        if self._session_date is None:
            self._session_date = d
        if d != self._session_date:
            self.log_summary(final=True)
            self._counts.clear()
            self._decisions = 0
            self._session_date = d

        if self._decisions > 0 and (self._decisions % self.summary_every == 0):
            self.log_summary(final=False)

    def log_summary(self, *, final: bool = False) -> None:
        tag = 'EOD' if final else 'MID'
        most = self._counts.most_common(12)
        msg = ", ".join([f"{k}={v}" for k, v in most]) if most else "no_counts"
        logger.info("[%s] [DECISION-%s] n=%d | %s", self.name, tag, self._decisions, msg)


def decide_trade(
    *,
    state: DecisionState,
    cfg: Any,
    features_raw: Dict[str, float],
    mtf: Dict[str, float],
    rule_dir: str,
    conflict_level: str = 'none',
    conflict_reasons: Optional[List[str]] = None,
    teacher_strength: float,
    is_bull_setup: bool,
    is_bear_setup: bool,
    safe_df: Any,
) -> Dict[str, Any]:
    """Hard/soft veto + lane selection (SETUP vs TREND) + rolling edge gating."""
    hard: list[str] = []
    soft: dict[str, float] = {}

    o = _safe_float(features_raw.get('open', float('nan')), float('nan'))
    h = _safe_float(features_raw.get('high', float('nan')), float('nan'))
    l = _safe_float(features_raw.get('low', float('nan')), float('nan'))
    c = _safe_float(features_raw.get('close', float('nan')), float('nan'))
    if not (np.isfinite(o) and np.isfinite(h) and np.isfinite(l) and np.isfinite(c)):
        hard.append('data_invalid')

    if 'data_invalid' in hard:
        return {
            'lane': 'NONE',
            'intent': 'FLAT',
            'tradeable': False,
            'size_mult': 0.0,
            'score': 0.0,
            'edge_required': 0.0,
            'edge_quantile': 0.0,
            'trend_strength': 0.0,
            'trend_signals': 0,
            'conflict': 'na',
            'tape_conflict_level': str(conflict_level or 'none'),
            'tape_conflict_reasons': list(conflict_reasons or []),
            'hard_veto': list(hard),
            'soft_penalties': {},
            'gate_reasons': list(hard),
        }

    intent = str(rule_dir or 'NA').upper()
    if intent not in ('BUY', 'SELL'):
        return {
            'lane': 'NONE',
            'intent': 'FLAT',
            'tradeable': False,
            'size_mult': 0.0,
            'score': 0.0,
            'edge_required': 0.0,
            'edge_quantile': 0.0,
            'trend_strength': 0.0,
            'trend_signals': 0,
            'conflict': 'na',
            'tape_conflict_level': str(conflict_level or 'none'),
            'tape_conflict_reasons': list(conflict_reasons or []),
            'hard_veto': hard + ['no_direction'],
            'soft_penalties': soft,
            'gate_reasons': hard + ['no_direction'],
        }

    side = 1 if intent == 'BUY' else -1

    if safe_df is None:
        try:
            min_edge = float(os.getenv('BACKFILL_MIN_SIG', '0.10') or '0.10')
        except Exception:
            min_edge = 0.10
        tape_conflict = str(conflict_level or 'none').lower()
        backfill_allow = {'no_flow_htf_approval', 'rule_sig_only_no_setup'}
        if tape_conflict == 'red' and set(conflict_reasons or []).issubset(backfill_allow):
            tape_conflict = 'none'
        tradeable = bool(abs(float(teacher_strength)) >= min_edge and tape_conflict != 'red')
        lane = 'SETUP' if (is_bull_setup or is_bear_setup) else 'TREND'
        gate_reasons = []
        if tape_conflict == 'red':
            gate_reasons.append('tape_conflict_red')
        if not tradeable:
            gate_reasons.append('min_teacher_strength')
        return {
            'lane': lane,
            'intent': intent,
            'tradeable': bool(tradeable),
            'size_mult': 1.0 if tradeable else 0.0,
            'score': float(abs(float(teacher_strength))),
            'edge_required': float(min_edge),
            'edge_quantile': 0.0,
            'trend_strength': 0.0,
            'trend_signals': 0,
            'tape_conflict_level': str(tape_conflict),
            'tape_conflict_reasons': list(conflict_reasons or []),
            'conflict': str(tape_conflict),
            'trigger': False,
            'trigger_name': 'backfill',
            'raw_strength': float(abs(float(teacher_strength))),
            'centered_strength': float(abs(float(teacher_strength))),
            'would_trade_without_soft': bool(tradeable),
            'hard_veto': [],
            'soft_penalties': {},
            'gate_reasons': gate_reasons,
        }

    tape_conflict = str(conflict_level or 'none').lower()
    if tape_conflict == 'red':
        hard.append('tape_conflict_red')
    elif tape_conflict == 'yellow':
        try:
            soft['tape_conflict_yellow'] = float(os.getenv('PEN_TAPE_YELLOW', '0.01') or '0.01')
        except Exception:
            soft['tape_conflict_yellow'] = 0.01
    conflict = str(tape_conflict)

    try:
        raw_strength = float(abs(teacher_strength))
    except Exception:
        raw_strength = 0.0
    raw_strength = float(np.clip(raw_strength, 0.0, 1.0))
    centered_strength = raw_strength

    state.update_edge(side, raw_strength)

    if getattr(state, '_trend_hold', 0) > 0:
        try:
            state._trend_hold = max(0, int(state._trend_hold) - 1)
        except Exception:
            state._trend_hold = 0

    flow_score, flow_side, micro_imb, fut_cvd, fut_vwap, vwap_side = compute_flow_signal(features_raw)
    flow_regime = str(features_raw.get('flow_regime', '') or '').upper()
    ema_chop = bool(_safe_float(features_raw.get('ema_regime_chop_5t', 0.0)) > 0.5)
    ema_bias = int(round(_safe_float(features_raw.get('ema_bias_5t', 0.0))))
    ema_break = int(round(_safe_float(features_raw.get('ema15_break_veto', 0.0))))
    entry_tag = int(round(_safe_float(features_raw.get('ema_entry_tag', 0.0))))
    entry_side = int(round(_safe_float(features_raw.get('ema_entry_side', 0.0))))
    ema15 = _safe_float(features_raw.get('ema_15', features_raw.get('ema15', 0.0)))
    struct_side = int(round(_safe_float(features_raw.get('struct_side', 0.0))))
    indicator_score = _safe_float(features_raw.get('indicator_score', 0.0))
    fast_setup_ready = bool(_safe_float(features_raw.get('fast_setup_ready', 0.0)) > 0.5)


    trend_signals = 0
    if ema_bias == side:
        trend_signals += 1
    if vwap_side == side:
        trend_signals += 1
    if flow_side == side and abs(float(flow_score)) >= 0.35:
        trend_signals += 1
    try:
        mtf_cons = float(mtf.get('mtf_consensus', 0.0)) if mtf else 0.0
    except Exception:
        mtf_cons = 0.0
    if (mtf_cons > 0.35 and side == 1) or (mtf_cons < -0.35 and side == -1):
        trend_signals += 1

    trend_strength = float(min(1.0, trend_signals / 3.0))

    trigger = False
    trigger_name = 'none'
    if entry_side == side and entry_tag in (1, 2, 3):
        trigger = True
        trigger_name = {1: 'pullback', 2: 'retest', 3: 'xover_conf'}.get(entry_tag, 'ema_tag')

    if not trigger and ema_break == side:
        trigger = True
        trigger_name = 'ema15_break'

    try:
        breakout_mtf = float(os.getenv('BREAKOUT_MTF_MIN', '0.5') or '0.5')
    except Exception:
        breakout_mtf = 0.5
    if side == 1:
        breakout_ok = (ema_break == 1 and mtf_cons >= breakout_mtf and flow_side == 1 and struct_side >= 0)
    else:
        breakout_ok = (ema_break == -1 and mtf_cons <= -breakout_mtf and flow_side == -1 and struct_side <= 0)

    try:
        mom_min = float(os.getenv('MOMO_IND_MIN', '0.35') or '0.35')
    except Exception:
        mom_min = 0.35
    momentum_override = bool(ema_break == side and flow_side == side and abs(float(indicator_score)) >= mom_min)

    ambiguous_setup = bool(is_bull_setup and is_bear_setup)
    setup_lane = False
    if not ambiguous_setup:
        if side == 1 and is_bull_setup:
            setup_lane = True
        if side == -1 and is_bear_setup:
            setup_lane = True
    if not setup_lane and fast_setup_ready:
        setup_lane = True

    lane = 'NONE'
    continuation = False
    breakout_lane = False
    if breakout_ok or momentum_override:
        lane = 'BREAKOUT'
        breakout_lane = True
    elif setup_lane:
        lane = 'SETUP'
    else:
        try:
            min_sig = int(os.getenv('TREND_MIN_SIGNALS', '2') or '2')
        except Exception:
            min_sig = 2
        if trend_signals >= min_sig and trigger:
            lane = 'TREND'
        else:
            flow_lock = flow_regime in ('BEAR_LOCK', 'BULL_LOCK')
            ema_align = (side == 1 and c >= ema15) or (side == -1 and c <= ema15)
            if flow_lock and vwap_side == side and ema_align:
                lane = 'TREND'
                continuation = True
                soft['trend_continuation'] = float(os.getenv('PEN_TREND_CONT', '0.01') or '0.01')

    if lane == 'NONE' and getattr(state, '_trend_hold', 0) > 0 and getattr(state, '_trend_side', 0) == side and trend_signals >= 2:
        lane = 'TREND'
        soft['hysteresis_hold'] = float(os.getenv('PEN_HYST', '0.005') or '0.005')

    # Probabilistic lane selection to reduce hard no-lane waits.
    setup_score = 0.0
    if setup_lane:
        setup_score += 0.60
    if fast_setup_ready:
        setup_score += 0.15
    if flow_side == side:
        setup_score += 0.10
    if vwap_side == side:
        setup_score += 0.10
    if (mtf_cons >= 0.35 and side == 1) or (mtf_cons <= -0.35 and side == -1):
        setup_score += 0.10
    setup_score = float(np.clip(setup_score, 0.0, 1.0))

    trend_score = 0.0
    trend_score += 0.20 * min(3, max(0, trend_signals))
    if trigger:
        trend_score += 0.20
    if ema_bias == side:
        trend_score += 0.10
    if flow_side == side:
        trend_score += 0.10
    trend_score = float(np.clip(trend_score, 0.0, 1.0))

    lane_score = max(setup_score, trend_score)
    try:
        lane_min = float(os.getenv('LANE_SCORE_MIN', '0.50') or '0.50')
    except Exception:
        lane_min = 0.50
    if lane == 'NONE' and lane_score >= lane_min:
        lane = 'SETUP' if setup_score >= trend_score else 'TREND'
        soft['lane_score_override'] = float(os.getenv('PEN_LANE_SCORE', '0.01') or '0.01')

    if tape_conflict == 'yellow' and lane == 'TREND' and not trigger:
        hard.append('tape_yellow_need_trigger')

    if lane == 'NONE':
        hard.append('lane_score_low')

    if side == 1 and (vwap_side < 0 and flow_side <= 0 and abs(fut_vwap) >= float(os.getenv('FLOW_VWAP_EXT', '0.0020') or '0.0020')):
        hard.append('structure_conflict')
    if side == -1 and (vwap_side > 0 and flow_side >= 0 and abs(fut_vwap) >= float(os.getenv('FLOW_VWAP_EXT', '0.0020') or '0.0020')):
        hard.append('structure_conflict')

    if ema_chop:
        trend_flow_agree = (ema_bias == side and flow_side == side)
        try:
            ema_chop_hard_min = float(os.getenv('EMA_CHOP_HARD_MIN', '0.55') or '0.55')
        except Exception:
            ema_chop_hard_min = 0.55
        if (not trend_flow_agree) and (not breakout_lane) and (lane_score < ema_chop_hard_min):
            hard.append('ema_chop_hard')
        else:
            soft['ema_chop_veto'] = float(os.getenv('PEN_CHOP', '0.02') or '0.02')

    require_setup = _safe_getenv_bool('REQUIRE_SETUP', default=True)
    if require_setup and lane == 'TREND':
        soft['require_setup_no_setup'] = float(os.getenv('PEN_NO_SETUP', '0.01') or '0.01')

    try:
        min_edge = float(os.getenv('GATE_MARGIN_THR', '0.06') or '0.06')
    except Exception:
        min_edge = 0.06

    edge_required, edge_q = state.edge_threshold(side, min_edge=min_edge)
    if lane == 'BREAKOUT':
        try:
            edge_required += float(os.getenv('BREAKOUT_EDGE_BOOST', '0.03') or '0.03')
        except Exception:
            edge_required += 0.03

    score = float(raw_strength)
    if lane == 'SETUP':
        score += float(os.getenv('BONUS_SETUP', '0.02') or '0.02')
    if lane == 'TREND':
        score += float(os.getenv('BONUS_TREND', '0.01') or '0.01')
    if lane == 'BREAKOUT':
        score += float(os.getenv('BONUS_BREAKOUT', '0.005') or '0.005')
    if flow_side == side:
        score += float(os.getenv('BONUS_FLOW', '0.01') or '0.01')
    if vwap_side == side:
        score += float(os.getenv('BONUS_VWAP', '0.005') or '0.005')
    score -= float(sum(float(v) for v in soft.values()))

    would_trade_wo_soft = (len(hard) == 0) and (lane != 'NONE') and (raw_strength >= edge_required)
    tradeable = would_trade_wo_soft and (score >= edge_required)

    if lane == 'TREND':
        try:
            state._trend_hold = int(getattr(state, 'hyst_bars', 0) or 0)
        except Exception:
            state._trend_hold = 0
        state._trend_side = side
    state._last_lane = lane

    if lane == 'SETUP':
        size_mult = 1.0
    elif lane == 'TREND':
        try:
            size_mult = float(os.getenv('TREND_LANE_SIZE_MULT', '0.5') or '0.5')
        except Exception:
            size_mult = 0.5
    elif lane == 'BREAKOUT':
        try:
            size_mult = float(os.getenv('BREAKOUT_LANE_SIZE_MULT', '0.30') or '0.30')
        except Exception:
            size_mult = 0.30
        if momentum_override and not breakout_ok:
            try:
                size_mult *= float(os.getenv('MOMO_SIZE_MULT', '0.70') or '0.70')
            except Exception:
                size_mult *= 0.70
    else:
        size_mult = 0.0

    if lane == 'TREND' and continuation:
        try:
            cont_mult = float(os.getenv('TREND_CONT_SIZE_MULT', '0.35') or '0.35')
        except Exception:
            cont_mult = 0.35
        size_mult *= float(np.clip(cont_mult, 0.05, 1.0))

    if lane == 'TREND':
        try:
            state._trend_hold = int(getattr(state, 'hyst_bars', 0) or 0)
        except Exception:
            state._trend_hold = 0
        state._trend_side = int(side)
    else:
        state._trend_hold = 0
        state._trend_side = 0

    if tape_conflict == 'yellow':
        try:
            size_mult *= float(os.getenv('TAPE_YELLOW_SIZE_MULT', '0.60') or '0.60')
        except Exception:
            size_mult *= 0.60
    elif tape_conflict == 'red':
        size_mult = 0.0

    if ema_break == -side:
        soft['ema15_break_veto'] = float(os.getenv('PEN_EMA_BREAK', '0.02') or '0.02')
        if lane != 'SETUP' and raw_strength < (edge_required + 0.05):
            hard.append('ema15_break_against')
            tradeable = False

    gate_reasons = []
    gate_reasons.extend(hard)
    gate_reasons.extend(list(soft.keys()))
    if lane == 'TREND':
        gate_reasons.append('lane_trend')
    elif lane == 'SETUP':
        gate_reasons.append('lane_setup')
    elif lane == 'BREAKOUT':
        gate_reasons.append('lane_breakout')

    try:
        state._last_lane = lane
    except Exception:
        pass
    if lane == 'TREND' and trigger and trend_signals >= 2:
        try:
            state._trend_side = side
            state._trend_hold = max(int(getattr(state, 'hyst_bars', 0) or 0), int(getattr(state, '_trend_hold', 0) or 0))
        except Exception:
            pass

    snap = DecisionSnapshot(
        lane=lane,
        intent=intent,
        tradeable=bool(tradeable),
        size_mult=float(size_mult),
        score=float(score),
        edge_required=float(edge_required),
        edge_quantile=float(edge_q),
        trend_strength=float(trend_strength),
        trend_signals=int(trend_signals),
        conflict=str(conflict),
        hard_veto=list(hard),
        soft_penalties=dict(soft),
        muted_only_soft=bool(would_trade_wo_soft and (not tradeable) and (len(soft) > 0) and (len(hard) == 0)),
    )
    state.bump_counts(snap)

    return {
        'lane': lane,
        'intent': intent,
        'tradeable': bool(tradeable),
        'size_mult': float(size_mult),
        'score': float(score),
        'edge_required': float(edge_required),
        'edge_quantile': float(edge_q),
        'trend_strength': float(trend_strength),
        'trend_signals': int(trend_signals),
        'tape_conflict_level': str(tape_conflict),
        'tape_conflict_reasons': list(conflict_reasons or []),
        'conflict': str(conflict),
        'trigger': bool(trigger),
        'trigger_name': str(trigger_name),
        'raw_strength': float(raw_strength),
        'centered_strength': float(centered_strength),
        'would_trade_without_soft': bool(would_trade_wo_soft),
        'hard_veto': list(hard),
        'soft_penalties': dict(soft),
        'gate_reasons': gate_reasons,
    }
