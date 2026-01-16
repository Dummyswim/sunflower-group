#!/usr/bin/env python3
"""Rule engine shared by live and training (single source of truth)."""
from __future__ import annotations

import logging
import os
from collections import Counter, deque
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        v = int(x)
        return v
    except Exception:
        return int(default)


def _sign(x: float) -> int:
    try:
        v = float(x)
    except Exception:
        return 0
    if v > 0:
        return 1
    if v < 0:
        return -1
    return 0


class RegimeStateMachine:
    """Simple regime state with hysteresis + volatility band."""

    def __init__(self, *, hold_bars: int = 3, vol_ema_alpha: float = 0.15) -> None:
        self.hold_bars = max(1, int(hold_bars))
        self.vol_ema_alpha = float(vol_ema_alpha)
        self._regime = "SIDEWAYS"
        self._age = 0
        self._candidate = None
        self._cand_age = 0
        self._atr_ema = None
        self._vol_band = "NORMAL"

    def _update_vol_band(self, atr_1t: float) -> str:
        if not np.isfinite(atr_1t) or atr_1t <= 0:
            return self._vol_band
        if self._atr_ema is None:
            self._atr_ema = float(atr_1t)
        else:
            self._atr_ema = (self.vol_ema_alpha * float(atr_1t)) + ((1.0 - self.vol_ema_alpha) * self._atr_ema)
        low_mult = _safe_float(os.getenv("VOL_BAND_LOW_MULT", "0.85"), 0.85)
        high_mult = _safe_float(os.getenv("VOL_BAND_HIGH_MULT", "1.15"), 1.15)
        if self._atr_ema and atr_1t <= (self._atr_ema * low_mult):
            self._vol_band = "LOW"
        elif self._atr_ema and atr_1t >= (self._atr_ema * high_mult):
            self._vol_band = "HIGH"
        else:
            self._vol_band = "NORMAL"
        return self._vol_band

    def update(self, candidate: str, atr_1t: float) -> dict:
        if candidate == self._regime:
            self._age += 1
            self._candidate = None
            self._cand_age = 0
        else:
            if candidate != self._candidate:
                self._candidate = candidate
                self._cand_age = 1
            else:
                self._cand_age += 1
            if self._cand_age >= self.hold_bars:
                self._regime = candidate
                self._age = 0
                self._candidate = None
                self._cand_age = 0
        vol_band = self._update_vol_band(float(atr_1t))
        return {
            "regime": self._regime,
            "regime_age": int(self._age),
            "vol_band": vol_band,
        }


class DynamicThresholds:
    """Dynamic thresholds with bounded EMA updates."""

    def __init__(self) -> None:
        self.enabled = _safe_getenv_bool("DYNAMIC_THRESHOLDS", default=True)
        self.alpha = _safe_float(os.getenv("DYN_THRESH_EMA_ALPHA", "0.20"), 0.20)
        self.update_every = max(1, _safe_int(os.getenv("DYN_THRESH_UPDATE_EVERY", "1"), 1))
        self._tick = 0
        self.base = {}
        self.curr = {}
        self.bounds = {}
        self._init_defaults()

    def _init_defaults(self) -> None:
        defaults = {
            "FLOW_STRONG_MIN": 0.50,
            "HTF_STRONG_VETO_MIN": 0.70,
            "LANE_SCORE_MIN": 0.50,
            "EMA_CHOP_HARD_MIN": 0.55,
            "GATE_MARGIN_THR": 0.06,
            "FLOW_VWAP_EXT": 0.0020,
            "BB_BW_PCTL_CHOP_MAX": 0.30,
            "DI_SPREAD_MIN": 9.0,
        }
        for k, dflt in defaults.items():
            raw = os.getenv(k, str(dflt))
            base = _safe_float(raw, dflt)
            self.base[k] = base
            self.curr[k] = base
            lo = _safe_float(os.getenv(f"DYN_{k}_MIN", str(base * 0.7)), base * 0.7)
            hi = _safe_float(os.getenv(f"DYN_{k}_MAX", str(base * 1.5)), base * 1.5)
            self.bounds[k] = (min(lo, hi), max(lo, hi))

    def _mult_for_regime(self, key: str, regime: str) -> float:
        env_key = f"DYN_{key}_{regime}_MULT"
        if os.getenv(env_key) is not None:
            return _safe_float(os.getenv(env_key), 1.0)
        if key in ("FLOW_STRONG_MIN", "LANE_SCORE_MIN", "GATE_MARGIN_THR", "FLOW_VWAP_EXT", "HTF_STRONG_VETO_MIN", "EMA_CHOP_HARD_MIN"):
            if regime.startswith("TREND"):
                return 0.90
            if regime == "CHOP":
                return 1.10
            if regime == "SIDEWAYS":
                return 1.05
            if regime == "REVERSAL_RISK":
                return 1.00
        if key == "BB_BW_PCTL_CHOP_MAX":
            if regime.startswith("TREND"):
                return 0.80
            if regime == "CHOP":
                return 1.15
            if regime == "SIDEWAYS":
                return 1.05
        if key == "DI_SPREAD_MIN":
            if regime.startswith("TREND"):
                return 0.90
            if regime == "CHOP":
                return 1.10
        return 1.0

    def _mult_for_vol(self, key: str, vol_band: str) -> float:
        env_key = f"DYN_{key}_VOL_{vol_band}_MULT"
        if os.getenv(env_key) is not None:
            return _safe_float(os.getenv(env_key), 1.0)
        if key in ("FLOW_STRONG_MIN", "LANE_SCORE_MIN", "GATE_MARGIN_THR"):
            if vol_band == "HIGH":
                return 0.95
            if vol_band == "LOW":
                return 1.05
        return 1.0

    def update(self, *, regime: str, vol_band: str, regime_age: int) -> dict:
        self._tick += 1
        if not self.enabled or (self._tick % self.update_every != 0):
            return self.curr
        if regime_age < max(1, _safe_int(os.getenv("REGIME_HOLD_BARS", "3"), 3)):
            return self.curr
        for key, base in self.base.items():
            target = base * self._mult_for_regime(key, regime) * self._mult_for_vol(key, vol_band)
            lo, hi = self.bounds.get(key, (base * 0.7, base * 1.5))
            target = float(np.clip(target, lo, hi))
            prev = float(self.curr.get(key, base))
            self.curr[key] = (self.alpha * target) + ((1.0 - self.alpha) * prev)
        return self.curr

    def get(self, key: str, default: float) -> float:
        if not self.enabled:
            return float(default)
        return float(self.curr.get(key, default))


def classify_regime(
    *,
    features_raw: Mapping[str, Any],
) -> dict:
    """Return candidate regime and reversal-risk modifiers."""
    flow_score = _safe_float(features_raw.get("flow_score", 0.0), 0.0)
    vwap_dev = _safe_float(features_raw.get("flow_fut_vwap_dev", 0.0), 0.0)
    vwap_side = int(round(_safe_float(features_raw.get("flow_vwap_side", 0.0), 0.0)))
    ema_chop = bool(_safe_float(features_raw.get("ema_regime_chop_5t", 0.0), 0.0) > 0.5)
    ema_bias = int(round(_safe_float(features_raw.get("ema_bias_5t", 0.0), 0.0)))
    bb_bw_pct = _safe_float(features_raw.get("ta_bb_bw_pct", 1.0), 1.0)
    di_spread = _safe_float(features_raw.get("ta_di_spread", 0.0), 0.0)
    st_flip = bool(_safe_float(features_raw.get("ta_supertrend_flip", 0.0), 0.0) > 0.5)
    micro_imb = _safe_float(features_raw.get("flow_micro_imb", 0.0), 0.0)
    micro_slope = _safe_float(features_raw.get("flow_micro_slope", features_raw.get("micro_slope", 0.0)), 0.0)
    fut_cvd = _safe_float(features_raw.get("flow_fut_cvd", 0.0), 0.0)
    trap_long = bool(_safe_float(features_raw.get("trap_long", 0.0), 0.0) > 0.5)
    trap_short = bool(_safe_float(features_raw.get("trap_short", 0.0), 0.0) > 0.5)

    flow_trend_min = _safe_float(os.getenv("REGIME_FLOW_TREND_MIN", "0.35"), 0.35)
    flow_chop_max = _safe_float(os.getenv("REGIME_FLOW_CHOP_MAX", "0.20"), 0.20)
    vwap_trend_min = _safe_float(os.getenv("REGIME_VWAP_TREND_MIN", "0.0006"), 0.0006)
    vwap_chop_max = _safe_float(os.getenv("REGIME_VWAP_CHOP_MAX", "0.0003"), 0.0003)
    bb_chop_max = _safe_float(os.getenv("BB_BW_PCTL_CHOP_MAX", "0.30"), 0.30)
    di_trend_min = _safe_float(os.getenv("DI_SPREAD_MIN", "9.0"), 9.0)

    candidate = "SIDEWAYS"
    bb_chop = (bb_bw_pct <= bb_chop_max) and (abs(di_spread) < di_trend_min)
    if ema_chop or (abs(flow_score) <= flow_chop_max and abs(vwap_dev) <= vwap_chop_max) or bb_chop:
        candidate = "CHOP"
    elif abs(flow_score) >= flow_trend_min and abs(vwap_dev) >= vwap_trend_min:
        dir_bias = ema_bias if ema_bias != 0 else _sign(flow_score)
        if dir_bias > 0:
            candidate = "TREND_UP"
        elif dir_bias < 0:
            candidate = "TREND_DN"
        else:
            candidate = "SIDEWAYS"

    rev_imb = _safe_float(os.getenv("REVERSAL_IMB_MIN", "0.05"), 0.05)
    rev_cvd = _safe_float(os.getenv("REVERSAL_CVD_MIN", "0.0"), 0.0)
    vwap_rev_min = _safe_float(os.getenv("REVERSAL_VWAP_MIN", "0.0004"), 0.0004)
    rev_slope = _safe_float(os.getenv("REVERSAL_SLOPE_MIN", "0.10"), 0.10)
    reversal_risk = False
    flow_dir = _sign(flow_score)
    if candidate.startswith("TREND") and flow_dir != 0:
        if flow_dir < 0:
            flip_ok = (micro_imb >= rev_imb) or (micro_slope >= rev_slope)
            reversal_risk = (trap_long or (flip_ok and fut_cvd >= rev_cvd)) and (vwap_side <= 0) and (abs(vwap_dev) >= vwap_rev_min)
        else:
            flip_ok = (micro_imb <= -rev_imb) or (micro_slope <= -rev_slope)
            reversal_risk = (trap_short or (flip_ok and fut_cvd <= -rev_cvd)) and (vwap_side >= 0) and (abs(vwap_dev) >= vwap_rev_min)
    if candidate.startswith("TREND") and st_flip:
        reversal_risk = True

    return {
        "candidate": candidate,
        "reversal_risk": bool(reversal_risk),
    }


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
    vwap_ext_override: Optional[float] = None,
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
    if vwap_ext_override is not None and np.isfinite(float(vwap_ext_override)):
        vwap_ext = float(vwap_ext_override)
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

    try:
        lock_imb = float(os.getenv("FLOW_LOCK_IMB_MIN", "0.05") or "0.05")
    except Exception:
        lock_imb = 0.05
    try:
        lock_cvd = float(os.getenv("FLOW_LOCK_CVD_MIN", str(cvd_min)) or str(cvd_min))
    except Exception:
        lock_cvd = float(cvd_min)
    lock_imb = abs(lock_imb)
    lock_cvd = abs(lock_cvd)

    # Lock regimes should require stronger evidence to avoid sticky flow.
    strong_bear_regime = (vwap_side == -1) and (micro_imb <= -lock_imb) and (fut_cvd <= -lock_cvd)
    strong_bull_regime = (vwap_side == +1) and (micro_imb >= +lock_imb) and (fut_cvd >= +lock_cvd)

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

    ob_bull_valid = bool(features.get("struct_ob_bull_valid", 0))
    ob_bear_valid = bool(features.get("struct_ob_bear_valid", 0))
    if not (ob_bull_valid or ob_bear_valid):
        try:
            last_px = _safe_float(features.get("close", features.get("last_price", 0.0)), 0.0)
            atr = _safe_float(features.get("atr_1t", features.get("atr_3t", 0.0)), 0.0)
            max_pct = float(os.getenv("OB_RELEVANCE_MAX_PCT", "0.0020") or "0.0020")
            max_atr_mult = float(os.getenv("OB_RELEVANCE_ATR_MULT", "0.35") or "0.35")
        except Exception:
            last_px = 0.0
            atr = 0.0
            max_pct = 0.0020
            max_atr_mult = 0.35
        max_rel = max_pct
        if last_px > 0.0 and atr > 0.0:
            max_rel = min(max_pct, (max_atr_mult * atr) / last_px)
        bull_dist = abs(_safe_float(features.get("struct_ob_bull_dist", 0.0), 0.0))
        bear_dist = abs(_safe_float(features.get("struct_ob_bear_dist", 0.0), 0.0))
        ob_bull_valid = bool(ob_bull and (bull_dist <= max_rel) and (ps_up or fvg_up))
        ob_bear_valid = bool(ob_bear and (bear_dist <= max_rel) and (ps_dn or fvg_dn))

    bull = ps_up or fvg_up or ob_bull_valid
    bear = ps_dn or fvg_dn or ob_bear_valid

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
        bb_bw_pct = float(feats.get("ta_bb_bw_pct", 0.5))
        di_spread = float(feats.get("ta_di_spread", 0.0))
        st_dir = int(round(float(feats.get("ta_supertrend_dir", 0.0))))
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
    try:
        di_spread_min = float(os.getenv("DI_SPREAD_MIN", "9.0"))
    except ValueError:
        di_spread_min = 8.0
    if abs(di_spread) >= di_spread_min:
        score += 0.10 if di_spread > 0 else -0.10
    if st_dir != 0:
        score += 0.10 if st_dir > 0 else -0.10

    try:
        bb_bw_chop = float(os.getenv("BB_BW_PCTL_CHOP_MAX", "0.30"))
    except ValueError:
        bb_bw_chop = 0.25
    if bb_bw_pct <= bb_bw_chop:
        score *= 0.70

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
    dynamic_thresholds: Optional[Mapping[str, float]] = None,
    regime: Optional[str] = None,
    reversal_risk: bool = False,
) -> Tuple[str, str, str, List[str]]:
    """Flow trigger with HTF veto and structure context."""
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

    vwap_ext_override = None
    if dynamic_thresholds and "FLOW_VWAP_EXT" in dynamic_thresholds:
        vwap_ext_override = float(dynamic_thresholds.get("FLOW_VWAP_EXT", 0.0))
    flow_score, flow_side, micro_imb, fut_cvd, fut_vwap, vwap_side = compute_flow_signal(
        features_raw,
        vwap_ext_override=vwap_ext_override,
    )
    di_spread = _safe_float(features_raw.get("ta_di_spread", 0.0), 0.0)
    st_dir = int(round(_safe_float(features_raw.get("ta_supertrend_dir", 0.0), 0.0)))

    try:
        mtf_cons = float(mtf.get("mtf_consensus", 0.0)) if mtf else 0.0
        if not np.isfinite(mtf_cons):
            mtf_cons = 0.0
    except Exception:
        mtf_cons = 0.0

    try:
        flow_strong_min = float(os.getenv("FLOW_STRONG_MIN", "0.50") or "0.50")
    except Exception:
        flow_strong_min = 0.50
    if dynamic_thresholds and "FLOW_STRONG_MIN" in dynamic_thresholds:
        flow_strong_min = float(dynamic_thresholds.get("FLOW_STRONG_MIN", flow_strong_min))
    try:
        htf_neutral_min = float(os.getenv("HTF_NEUTRAL_MIN", "0.35") or "0.35")
    except Exception:
        htf_neutral_min = 0.35
    try:
        htf_strong_veto_min = float(os.getenv("HTF_STRONG_VETO_MIN", "0.70") or "0.70")
    except Exception:
        htf_strong_veto_min = 0.60
    if dynamic_thresholds and "HTF_STRONG_VETO_MIN" in dynamic_thresholds:
        htf_strong_veto_min = float(dynamic_thresholds.get("HTF_STRONG_VETO_MIN", htf_strong_veto_min))
    if regime == "CHOP":
        htf_strong_veto_min = min(0.95, float(htf_strong_veto_min) * 1.15)

    flow_dir = 0
    if abs(float(flow_score)) >= flow_strong_min:
        flow_dir = 1 if flow_score > 0 else -1
    allow_rule_fallback = _safe_getenv_bool("ALLOW_RULE_SIG_FALLBACK", default=False)
    if flow_dir == 0 and allow_rule_fallback and base_side != 0:
        flow_dir = base_side
        conflict_reasons.append("flow_not_strong_rule_sig_fallback")

    htf_side = 0
    htf_veto = 0
    if abs(float(mtf_cons)) >= htf_strong_veto_min:
        htf_side = 1 if mtf_cons > 0 else -1
        htf_veto = htf_side
    elif abs(float(mtf_cons)) >= htf_neutral_min:
        htf_side = 1 if mtf_cons > 0 else -1

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

    if flow_dir == 0:
        rule_dir = "FLAT"
        conflict_reasons.append("flow_not_strong")
    else:
        if htf_veto != 0 and htf_veto != flow_dir:
            veto_mode = str(os.getenv("HTF_VETO_MODE", "hard") or "hard").lower().strip()
            regime_txt = (regime or "na").upper()
            allow_soft = (regime_txt.startswith("TREND") and not reversal_risk)
            if veto_mode == "soft" and allow_soft:
                rule_dir = "BUY" if flow_dir > 0 else "SELL"
                conflict_level = "yellow"
                conflict_reasons.append("htf_veto_soft")
            elif veto_mode == "conditional" and allow_soft:
                try:
                    soft_flow_min = float(os.getenv("HTF_VETO_SOFT_FLOW_MIN", "0.85") or "0.85")
                except Exception:
                    soft_flow_min = 0.85
                vwap_ext = vwap_ext_override
                if vwap_ext is None:
                    try:
                        vwap_ext = float(os.getenv("FLOW_VWAP_EXT", "0.0020") or "0.0020")
                    except Exception:
                        vwap_ext = 0.0020
                try:
                    di_spread_min = float(os.getenv("DI_SPREAD_MIN", "9.0") or "9.0")
                except Exception:
                    di_spread_min = 8.0
                if dynamic_thresholds and "DI_SPREAD_MIN" in dynamic_thresholds:
                    di_spread_min = float(dynamic_thresholds.get("DI_SPREAD_MIN", di_spread_min))
                ema_bias = int(round(_safe_float(features_raw.get("ema_bias_5t", 0.0), 0.0)))
                ema_chop = bool(_safe_float(features_raw.get("ema_regime_chop_5t", 0.0), 0.0) > 0.5)
                override_ok = (not ema_chop) and (abs(float(flow_score)) >= soft_flow_min) and (vwap_side == flow_dir) \
                    and (abs(float(fut_vwap)) >= float(vwap_ext)) and (ema_bias == flow_dir)
                st_align = (st_dir == flow_dir) and (st_dir != 0)
                di_align = (abs(di_spread) >= di_spread_min) and (_sign(di_spread) == flow_dir)
                override_ok = override_ok or (st_align and di_align and (abs(float(flow_score)) >= soft_flow_min * 0.9))
                if override_ok:
                    rule_dir = "BUY" if flow_dir > 0 else "SELL"
                    conflict_level = "yellow"
                    conflict_reasons.append("htf_veto_soft_override")
                else:
                    rule_dir = "FLAT"
                    conflict_level = "red"
                    conflict_reasons.append("htf_veto")
            else:
                rule_dir = "FLAT"
                conflict_level = "red"
                conflict_reasons.append("htf_veto")
        elif struct_side != 0 and struct_side != flow_dir:
            try:
                struct_flow_min = float(os.getenv("STRUCT_OPPOSE_FLOW_MIN", "0.70") or "0.70")
            except Exception:
                struct_flow_min = 0.70
            try:
                vwap_ext = float(os.getenv("FLOW_VWAP_EXT", "0.0020") or "0.0020")
            except Exception:
                vwap_ext = 0.0020
            if vwap_ext_override is not None:
                vwap_ext = float(vwap_ext_override)
            try:
                di_spread_min = float(os.getenv("DI_SPREAD_MIN", "9.0") or "9.0")
            except Exception:
                di_spread_min = 8.0
            if dynamic_thresholds and "DI_SPREAD_MIN" in dynamic_thresholds:
                di_spread_min = float(dynamic_thresholds.get("DI_SPREAD_MIN", di_spread_min))
            st_align = (st_dir == flow_dir) and (st_dir != 0)
            di_align = (abs(di_spread) >= di_spread_min) and (_sign(di_spread) == flow_dir)
            strong_flow_override = (abs(float(flow_score)) >= struct_flow_min) and (abs(float(fut_vwap)) >= vwap_ext)
            if st_align and di_align and (abs(float(flow_score)) >= struct_flow_min * 0.9):
                strong_flow_override = True
            if strong_flow_override:
                rule_dir = "BUY" if flow_dir > 0 else "SELL"
                conflict_level = "yellow"
                conflict_reasons.append("struct_opposition_override")
            else:
                rule_dir = "FLAT"
                conflict_level = "red"
                conflict_reasons.append("struct_opposition")
        else:
            rule_dir = "BUY" if flow_dir > 0 else "SELL"
            if htf_side != 0 and htf_side != flow_dir:
                conflict_level = "yellow"
                conflict_reasons.append("htf_against")

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
        self._ambig_bars = 0

        self._counts = Counter()
        self._decisions = 0
        self._session_date = None
        self.regime_state = RegimeStateMachine(hold_bars=_safe_int(os.getenv("REGIME_HOLD_BARS", "3"), 3))
        self.dynamic_thresholds = DynamicThresholds()

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
    tape_valid: bool = True,
    teacher_strength: float,
    is_bull_setup: bool,
    is_bear_setup: bool,
    safe_df: Any,
    dynamic_thresholds: Optional[Mapping[str, float]] = None,
    regime: Optional[str] = None,
    reversal_risk: bool = False,
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
            'tape_valid': bool(tape_valid),
            'hard_veto': list(hard),
            'soft_penalties': {},
            'gate_reasons': list(hard),
        }

    if _safe_getenv_bool('TAPE_REQUIRED_FOR_TRADING', default=False) and (not tape_valid):
        hard.append('tape_required_invalid')

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
    if not tape_valid:
        tape_conflict = 'invalid'
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
            'tape_valid': bool(tape_valid),
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
    if not tape_valid:
        tape_conflict = 'invalid'
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
    bb_bw_pct = _safe_float(features_raw.get('ta_bb_bw_pct', 0.5), 0.5)
    di_spread = _safe_float(features_raw.get('ta_di_spread', 0.0), 0.0)
    st_dir = int(round(_safe_float(features_raw.get('ta_supertrend_dir', 0.0))))
    st_flip = bool(_safe_float(features_raw.get('ta_supertrend_flip', 0.0)) > 0.5)


    try:
        flow_strong_min = float(os.getenv("FLOW_STRONG_MIN", "0.50") or "0.50")
    except Exception:
        flow_strong_min = 0.50
    if dynamic_thresholds and "FLOW_STRONG_MIN" in dynamic_thresholds:
        flow_strong_min = float(dynamic_thresholds.get("FLOW_STRONG_MIN", flow_strong_min))
    try:
        di_spread_min = float(os.getenv("DI_SPREAD_MIN", "9.0") or "9.0")
    except Exception:
        di_spread_min = 8.0
    if dynamic_thresholds and "DI_SPREAD_MIN" in dynamic_thresholds:
        di_spread_min = float(dynamic_thresholds.get("DI_SPREAD_MIN", di_spread_min))
    try:
        bb_chop_max = float(os.getenv("BB_BW_PCTL_CHOP_MAX", "0.30") or "0.30")
    except Exception:
        bb_chop_max = 0.25
    if dynamic_thresholds and "BB_BW_PCTL_CHOP_MAX" in dynamic_thresholds:
        bb_chop_max = float(dynamic_thresholds.get("BB_BW_PCTL_CHOP_MAX", bb_chop_max))
    bb_squeeze = bool(bb_bw_pct <= bb_chop_max)

    trend_signals = 0
    if ema_bias == side:
        trend_signals += 1
    if vwap_side == side:
        trend_signals += 1
    if flow_side == side and abs(float(flow_score)) >= float(flow_strong_min):
        trend_signals += 1
    di_dir = _sign(di_spread)
    if di_dir == side and abs(float(di_spread)) >= float(di_spread_min):
        trend_signals += 1
    if st_dir == side:
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
    try:
        ambig_persist = int(os.getenv('AMBIG_PERSIST_BARS', '2') or '2')
    except Exception:
        ambig_persist = 2
    if ambiguous_setup:
        try:
            state._ambig_bars = max(0, int(getattr(state, '_ambig_bars', 0))) + 1
        except Exception:
            state._ambig_bars = 1
    else:
        state._ambig_bars = 0
    if state._ambig_bars >= ambig_persist:
        hard.append('setup_ambiguous')
    elif ambiguous_setup:
        soft['setup_ambiguous'] = float(os.getenv('PEN_AMBIG', '0.02') or '0.02')
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
    if di_dir == side and abs(float(di_spread)) >= float(di_spread_min):
        trend_score += 0.10
    if st_dir == side:
        trend_score += 0.10
    trend_score = float(np.clip(trend_score, 0.0, 1.0))

    lane_score = max(setup_score, trend_score)
    try:
        lane_min = float(os.getenv('LANE_SCORE_MIN', '0.50') or '0.50')
    except Exception:
        lane_min = 0.50
    if dynamic_thresholds and "LANE_SCORE_MIN" in dynamic_thresholds:
        lane_min = float(dynamic_thresholds.get("LANE_SCORE_MIN", lane_min))
    if trend_signals >= 2:
        lane_min *= 0.90
    if bb_squeeze and lane == 'NONE':
        lane_min *= 1.10
    if lane == 'NONE' and lane_score >= lane_min:
        lane = 'SETUP' if setup_score >= trend_score else 'TREND'
        soft['lane_score_override'] = float(os.getenv('PEN_LANE_SCORE', '0.01') or '0.01')

    if tape_conflict == 'yellow' and lane == 'TREND' and not trigger:
        hard.append('tape_yellow_need_trigger')
    if bb_squeeze and lane in ('TREND', 'BREAKOUT') and not trigger:
        hard.append('bb_squeeze_chop')

    if lane == 'NONE':
        hard.append('lane_score_low')

    rv_10 = _safe_float(features_raw.get('rv_10', 0.0), 0.0)
    atr_1t = _safe_float(features_raw.get('atr_1t', features_raw.get('atr_3t', 0.0)), 0.0)
    try:
        rv_atr_min = float(os.getenv('RV_ATR_MIN', '1e-05') or '1e-05')
    except Exception:
        rv_atr_min = 1e-05
    rv_atr = abs(rv_10) / atr_1t if atr_1t > 0 else 0.0
    if rv_atr_min > 0.0 and (lane in ('TREND', 'BREAKOUT')) and (not setup_lane):
        if rv_atr < rv_atr_min:
            hard.append('rv_atr_low')

    vwap_ext_dyn = None
    if dynamic_thresholds and "FLOW_VWAP_EXT" in dynamic_thresholds:
        vwap_ext_dyn = float(dynamic_thresholds.get("FLOW_VWAP_EXT", 0.0))
    try:
        vwap_ext_gate = float(os.getenv('FLOW_VWAP_EXT', '0.0020') or '0.0020')
    except Exception:
        vwap_ext_gate = 0.0020
    if vwap_ext_dyn is not None:
        vwap_ext_gate = float(vwap_ext_dyn)
    if side == 1 and (vwap_side < 0 and flow_side <= 0 and abs(fut_vwap) >= vwap_ext_gate):
        hard.append('structure_conflict')
    if side == -1 and (vwap_side > 0 and flow_side >= 0 and abs(fut_vwap) >= vwap_ext_gate):
        hard.append('structure_conflict')

    if ema_chop or bb_squeeze:
        trend_flow_agree = (ema_bias == side and flow_side == side)
        try:
            ema_chop_hard_min = float(os.getenv('EMA_CHOP_HARD_MIN', '0.55') or '0.55')
        except Exception:
            ema_chop_hard_min = 0.55
        if dynamic_thresholds and "EMA_CHOP_HARD_MIN" in dynamic_thresholds:
            ema_chop_hard_min = float(dynamic_thresholds.get("EMA_CHOP_HARD_MIN", ema_chop_hard_min))
        if trend_signals >= 2 and trend_flow_agree:
            ema_chop_hard_min *= 0.90
        if (not trend_flow_agree) and (not breakout_lane) and (lane_score < ema_chop_hard_min):
            hard.append('ema_chop_hard')
        else:
            soft['ema_chop_veto'] = float(os.getenv('PEN_CHOP', '0.02') or '0.02')

    require_setup = _safe_getenv_bool('REQUIRE_SETUP', default=True)
    if require_setup and lane == 'TREND':
        if trend_signals < 2:
            soft['require_setup_no_setup'] = float(os.getenv('PEN_NO_SETUP', '0.01') or '0.01')

    try:
        min_edge = float(os.getenv('GATE_MARGIN_THR', '0.06') or '0.06')
    except Exception:
        min_edge = 0.06
    if dynamic_thresholds and "GATE_MARGIN_THR" in dynamic_thresholds:
        min_edge = float(dynamic_thresholds.get("GATE_MARGIN_THR", min_edge))

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

    if reversal_risk and lane in ("TREND", "BREAKOUT") and (not trigger):
        hard.append("short_reversal_risk" if side == -1 else "long_reversal_risk")
    elif reversal_risk:
        soft["reversal_risk"] = float(os.getenv("PEN_REVERSAL_RISK", "0.02") or "0.02")
    if st_flip and lane in ("TREND", "BREAKOUT") and (not trigger):
        hard.append("supertrend_flip")
    elif st_dir != 0 and st_dir != side and lane in ("TREND", "BREAKOUT"):
        soft["supertrend_conflict"] = float(os.getenv("PEN_SUPERTREND_CONFLICT", "0.01") or "0.01")

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
        'tape_valid': bool(tape_valid),
        'trigger': bool(trigger),
        'trigger_name': str(trigger_name),
        'raw_strength': float(raw_strength),
        'centered_strength': float(centered_strength),
        'would_trade_without_soft': bool(would_trade_wo_soft),
        'hard_veto': list(hard),
        'soft_penalties': dict(soft),
        'gate_reasons': gate_reasons,
    }
