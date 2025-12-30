# main_event_loop.py
import asyncio
import base64
import json
import logging
import os
import time
from collections import deque
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple, Mapping
from contextlib import suppress

import numpy as np
import pandas as pd
import websockets

from core_handler import UnifiedWebSocketHandler as WSHandler
from feature_pipeline import FeaturePipeline, TA
from logging_setup import log_every
from policy_pipeline import PolicyPipeline
from signal_context import build_signal_context, compose_policy_features
from signal_log_utils import append_jsonl, validate_or_quarantine
from rule_engine import (
    compute_flow_signal,
    compute_structure_score,
    compute_rule_hierarchy,
    compute_ta_rule_signal,
    classify_regime,
    DecisionState,
    decide_trade,
)


logger = logging.getLogger(__name__)
IST = timezone(timedelta(hours=5, minutes=30))

LABEL_SKIP = "SKIP"


def _safe_getenv_bool(key: str, default: bool = False) -> bool:
    val = os.getenv(key)
    if val is None:
        return bool(default)
    return str(val).strip().lower() in ("1", "true", "yes", "y", "on")


def _log_env_audit(keys: List[str]) -> None:
    try:
        parts = []
        for k in keys:
            v = os.getenv(k)
            parts.append(f"{k}={v if v is not None else '<unset>'}")
        logger.info("[ENV] %s", " ".join(parts))
    except Exception:
        pass

def _make_trade_outcome_label_live(
    df: pd.DataFrame,
    idx: int,
    horizon_bars: int,
    tp_pct: float,
    sl_pct: float,
    side: str,
) -> str:
    """
    Compute a simple TP/SL outcome label for a trade opened at df.iloc[idx]
    and held for up to `horizon_bars` subsequent candles.

    horizon_bars is specified in *bars*, not minutes, so this function
    works for any candle interval (1-min, 5-min, etc.).
    """
    try:
        n = int(len(df))
    except Exception:
        return "invalid"

    if n == 0 or idx < 0 or idx >= n:
        return "invalid"

    try:
        highs = df["high"].to_numpy(dtype=float)
        lows = df["low"].to_numpy(dtype=float)
        closes = df["close"].to_numpy(dtype=float)
    except Exception:
        return "invalid"

    if highs.size != n or lows.size != n or closes.size != n:
        return "invalid"

    entry = closes[idx]
    if not np.isfinite(entry) or entry <= 0.0:
        return "invalid"

    try:
        horizon_bars = max(1, int(horizon_bars))
    except Exception:
        horizon_bars = 1

    lo_idx = idx + 1
    if lo_idx >= n:
        return "no_hit"

    hi_idx = min(idx + horizon_bars, n - 1)

    side_norm = str(side).upper()
    if side_norm not in ("BUY", "SELL"):
        return "invalid"

    if side_norm == "BUY":
        tp_price = entry * (1.0 + float(tp_pct))
        sl_price = entry * (1.0 - float(sl_pct))
        for j in range(lo_idx, hi_idx + 1):
            h = highs[j]
            l = lows[j]
            if np.isfinite(h) and h >= tp_price:
                return "hit_tp"
            if np.isfinite(l) and l <= sl_price:
                return "hit_sl"
    else:
        tp_price = entry * (1.0 - float(tp_pct))
        sl_price = entry * (1.0 + float(sl_pct))
        for j in range(lo_idx, hi_idx + 1):
            h = highs[j]
            l = lows[j]
            if np.isfinite(l) and l <= tp_price:
                return "hit_tp"
            if np.isfinite(h) and h >= sl_price:
                return "hit_sl"

    return "no_hit"


def compute_trade_window_label(
    df: pd.DataFrame,
    idx: int,
    horizon: int = 2,
    tp_ret: float = 0.0005,
    sl_ret: float = 0.0005,
    fallback_atr_mult: float = 0.20,
) -> Optional[str]:
    """
    Return BUY/SELL/FLAT or LABEL_SKIP if we cannot label.

    Step 1: Path-based TP/SL race over the next `horizon` bars:
        - If only upside TP is hit → BUY
        - If only downside TP is hit → SELL
        - If both hit or neither hits → go to fallback

    Step 2 (fallback): If no clean TP/SL winner:
        - Compute simple ATR over the last ~14 bars.
        - If end-of-window return magnitude > fallback_atr_mult * ATR,
          assign direction by sign.
        - Otherwise label FLAT.
    """
    try:
        n = int(len(df))
    except Exception:
        return LABEL_SKIP

    if n == 0 or idx < 0:
        return LABEL_SKIP

    try:
        horizon = max(1, int(horizon))
    except Exception:
        horizon = 1

    end_idx = idx + horizon
    if end_idx >= n:
        return LABEL_SKIP

    try:
        close = df["close"].astype(float)
        high = df["high"].astype(float)
        low = df["low"].astype(float)
    except Exception:
        try:
            ret = df["close"].iloc[end_idx] / df["close"].iloc[idx] - 1.0
        except Exception:
            return LABEL_SKIP
        if ret > tp_ret:
            return "BUY"
        if ret < -sl_ret:
            return "SELL"
        return "FLAT"

    entry = float(close.iloc[idx])
    if not np.isfinite(entry) or entry <= 0.0:
        return LABEL_SKIP

    window = df.iloc[idx + 1 : end_idx + 1]
    if window.empty:
        return LABEL_SKIP

    up_ret = window["high"].astype(float) / entry - 1.0
    dn_ret = window["low"].astype(float) / entry - 1.0

    try:
        tp_thr = float(tp_ret)
        sl_thr = float(sl_ret)
    except Exception:
        tp_thr = 0.0006
        sl_thr = 0.0006

    up_hit = bool((up_ret >= tp_thr).any())
    dn_hit = bool((dn_ret <= -sl_thr).any())

    if up_hit and not dn_hit:
        return "BUY"
    if dn_hit and not up_hit:
        return "SELL"

    try:
        end_price = float(close.iloc[end_idx])
        end_ret = end_price / entry - 1.0
    except Exception:
        return "FLAT"

    try:
        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        atr = tr.rolling(14, min_periods=1).mean()
        atr_val = float(atr.iloc[idx])
        atr_ret = atr_val / entry if entry > 0 else 0.0
    except Exception:
        atr_ret = 0.0

    if not np.isfinite(atr_ret) or atr_ret <= 0.0:
        if end_ret > tp_thr:
            return "BUY"
        if end_ret < -sl_thr:
            return "SELL"
        return "FLAT"

    k = float(fallback_atr_mult)
    try:
        min_ret = float(os.getenv("TRAIN_FALLBACK_ATR_MIN_RET", "0.00025") or "0.00025")
    except Exception:
        min_ret = 0.00025
    thr = max(k * atr_ret, min_ret)

    if end_ret > thr:
        return "BUY"
    if end_ret < -thr:
        return "SELL"

    return "FLAT"


def compute_tp_sl_direction_label(
    df: pd.DataFrame,
    idx: int,
    horizon_bars: int,
    base_tp_pct: float,
    base_sl_pct: float,
    rv10: Optional[float] = None,
    atr1: Optional[float] = None,
    vol_k: float = 0.0,
) -> Optional[str]:
    """
    TP/SL path-based BUY/SELL/FLAT label for a trade-window starting at df.iloc[idx].

    Uses _make_trade_outcome_label_live for both BUY and SELL and returns:
      - "BUY"  if BUY TP is hit first and SELL TP is not.
      - "SELL" if SELL TP is hit first and BUY TP is not.
      - "FLAT" for ambiguous / no-edge windows (both/no TP or symmetric outcomes).

    Returns LABEL_SKIP on any hard failure so callers can cleanly skip the row.
    """
    try:
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            return LABEL_SKIP
        n = int(len(df))
        if idx < 0 or idx >= n:
            return LABEL_SKIP
        hb = max(1, int(horizon_bars))
    except Exception:
        return LABEL_SKIP

    # Volatility-aware adjustment: derive a single proxy from rv_10 / atr_1t
    vol_proxy = 0.0
    for v in (rv10, atr1):
        try:
            if v is not None:
                v = float(v)
                if np.isfinite(v):
                    vol_proxy = max(vol_proxy, abs(v))
        except Exception:
            continue

    try:
        vol_k = float(vol_k)
    except Exception:
        vol_k = 0.0

    # Start from config TP/SL; inflate with volatility if requested
    eff_tp = float(base_tp_pct)
    eff_sl = float(base_sl_pct)
    if vol_k > 0.0 and vol_proxy > 0.0:
        adj = float(vol_k * vol_proxy)
        # Guard rails to avoid insane thresholds (<= 2%)
        adj = float(np.clip(adj, 0.0, 0.02))
        eff_tp = max(eff_tp, adj)
        eff_sl = max(eff_sl, adj)

    buy_out = _make_trade_outcome_label_live(df, idx, hb, eff_tp, eff_sl, "BUY")
    sell_out = _make_trade_outcome_label_live(df, idx, hb, eff_tp, eff_sl, "SELL")

    try:
        ts_str = str(df.index[idx])
    except Exception:
        ts_str = str(idx)

    logger.debug(
        "[LABEL-TP/SL] ts=%s idx=%d hb=%d base_tp=%.5f base_sl=%.5f eff_tp=%.5f eff_sl=%.5f "
        "vol_proxy=%.6f vol_k=%.3f buy_out=%s sell_out=%s",
        ts_str,
        idx,
        hb,
        float(base_tp_pct),
        float(base_sl_pct),
        eff_tp,
        eff_sl,
        vol_proxy,
        vol_k,
        buy_out,
        sell_out,
    )

    # Directional decision:
    # 1) TP wins uniquely
    if buy_out == "hit_tp" and sell_out != "hit_tp":
        return "BUY"
    if sell_out == "hit_tp" and buy_out != "hit_tp":
        return "SELL"

    # 2) SL-only outcomes: opposite direction
    if buy_out == "hit_sl" and sell_out not in ("hit_tp", "hit_sl"):
        return "SELL"
    if sell_out == "hit_sl" and buy_out not in ("hit_tp", "hit_sl"):
        return "BUY"

    # 3) Fallback: net drift over the horizon
    try:
        ent = float(df["close"].iloc[idx])
        n = len(df)
        hb_eff = min(hb, max(1, n - idx - 1))
        close_h = float(df["close"].iloc[idx + hb_eff])
        ret = (close_h - ent) / max(abs(ent), 1e-6)
    except Exception:
        ret = 0.0

    drift_thr = max(eff_tp * 0.5, 0.0003)

    if ret >= drift_thr:
        return "BUY"
    if ret <= -drift_thr:
        return "SELL"

    return "FLAT"


# ---------------------------------------------------------------------------
# FLOW / STRUCTURE / TRAP HELPERS
# ---------------------------------------------------------------------------

def _detect_trap_patterns(
    df: pd.DataFrame,
    idx_ref: int,
    name: str,
    break_bps: float = 2.0,
) -> Tuple[int, int]:
    """
    Simple failed-breakout / trap heuristic.

    - Long trap (trap_short=1): current bar wicks above recent highs and
      closes back below the prior swing high.
    - Short trap (trap_long=1): current bar wicks below recent lows and
      closes back above the prior swing low.

    Returns:
        trap_long, trap_short flags as 0/1.
    """
    try:
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            return 0, 0
        n = int(len(df))
        if idx_ref <= 2 or idx_ref >= n:
            return 0, 0

        # Use last 3 completed bars as context
        start = max(0, idx_ref - 3)
        prev_slice = df.iloc[start:idx_ref]
        if prev_slice.empty:
            return 0, 0

        prev_high = float(prev_slice["high"].max())
        prev_low = float(prev_slice["low"].min())
        cur = df.iloc[idx_ref]
        cur_high = float(cur["high"])
        cur_low = float(cur["low"])
        cur_close = float(cur["close"])

        if not all(np.isfinite(v) for v in [prev_high, prev_low, cur_high, cur_low, cur_close]):
            return 0, 0

        thr = (break_bps * 1e-4) * max(cur_close, 1e-6)

        trap_long = 0
        trap_short = 0

        # Long trap: poke below lows, close back inside/over
        if cur_low < prev_low - thr and cur_close >= prev_low:
            trap_long = 1

        # Short trap: poke above highs, close back inside/under
        if cur_high > prev_high + thr and cur_close <= prev_high:
            trap_short = 1

        if trap_long or trap_short:
            logger.info(
                "[%s] [TRAP] idx=%s trap_long=%s trap_short=%s "
                "prev_high=%.2f prev_low=%.2f cur_high=%.2f cur_low=%.2f close=%.2f",
                name,
                str(df.index[idx_ref]),
                bool(trap_long),
                bool(trap_short),
                prev_high,
                prev_low,
                cur_high,
                cur_low,
                cur_close,
            )

        return trap_long, trap_short
    except Exception:
        return 0, 0



def compute_setup_conditional_label(
    df: pd.DataFrame,
    idx_ref: int,
    idx_ts: datetime,
    horizon_bars: int,
    features_for_log: Mapping[str, Any],
    name: str,
) -> Tuple[Optional[str], Dict[str, float]]:
    """
    Setup-conditional label for live training.

    Scalper intent:
      - Only label when the *market is actually saying something*:
          * one-sided structure (bull XOR bear)
          * tape/flow not fighting the setup (no hard VWAP lock against it)
          * at least 2-of-3 evidence agree (setup, micro_imb, flow/VWAP)

    Output:
      - ("BUY" / "SELL" / "FLAT" / None, aux_info dict)

    Notes:
      - This function is for *high-quality* labels. If it returns None,
        caller may fall back to a looser trade-window label with lower weight.
    """
    try:
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            return None, {}
        n = int(len(df))
        if idx_ref < 0 or idx_ref >= n:
            return None, {}

        # --- Structure setup flags (one-sided only) ---
        is_bull = bool(float(features_for_log.get("is_bull_setup", 0.0)) > 0.5)
        is_bear = bool(float(features_for_log.get("is_bear_setup", 0.0)) > 0.5)
        ambiguous = bool(is_bull and is_bear)
        if (not is_bull and not is_bear) or ambiguous:
            return None, {}
        setup_side = 1 if is_bull else -1

        # --- Flow / tape context (VWAP + CVD + micro imbalance) ---
        flow_score, flow_side, micro_imb, fut_cvd, fut_vwap, vwap_side = compute_flow_signal(
            features_for_log,
            mutate=False,
        )

        # --- Evidence thresholds (env-tunable) ---
        try:
            imb_min = float(os.getenv("SETUP_LABEL_IMB_MIN_LIVE", os.getenv("SETUP_LABEL_IMB_MIN", "0.04")) or "0.04")
        except Exception:
            imb_min = 0.04

        # micro evidence: sign + magnitude
        if setup_side == 1:
            imb_ok = (micro_imb >= +imb_min)
        else:
            imb_ok = (micro_imb <= -imb_min)

        # VWAP evidence: allow if not extended against you
        # (vwap_side comes from FLOW_VWAP_EXT, default 20 bps)
        if setup_side == 1:
            vwap_ok = (vwap_side >= 0)
        else:
            vwap_ok = (vwap_side <= 0)

        # Flow evidence: if flow has a side, it must match setup; if flow is neutral, don't veto.
        flow_ok = True
        if flow_side != 0:
            flow_ok = (flow_side == setup_side)

        # Hard veto: if flow regime is locked against setup, skip label.
        # (we infer lock from vwap_side + flow_side mismatch and strong magnitude)
        hard_veto = (flow_side != 0 and flow_side != setup_side and abs(flow_score) >= 0.85 and abs(fut_vwap) >= float(os.getenv("FLOW_VWAP_EXT", "0.0020") or "0.0020"))
        if hard_veto:
            return None, {}

        # 2-of-3 evidence rule (setup is assumed true already)
        evidence_votes = int(imb_ok) + int(vwap_ok) + int(flow_ok)
        clear_setup = (evidence_votes >= 2)

        if not clear_setup:
            logger.debug(
                "[%s] [LABEL-SETUP] skip @%s ref=%s votes=%d (imb_ok=%s vwap_ok=%s flow_ok=%s) micro_imb=%.3f thr=%.3f flow=%.3f vwap_dev=%.4f",
                name,
                idx_ts.strftime("%H:%M:%S") if isinstance(idx_ts, datetime) else str(idx_ts),
                str(df.index[idx_ref]),
                evidence_votes,
                bool(imb_ok), bool(vwap_ok), bool(flow_ok),
                float(micro_imb), float(imb_min),
                float(flow_score), float(fut_vwap),
            )
            return None, {}

        # Trap detection (context only)
        trap_long, trap_short = _detect_trap_patterns(
            df=df,
            idx_ref=idx_ref,
            name=name,
            break_bps=float(os.getenv("TRAP_BREAK_BPS", "2.0") or "2.0"),
        )

        # Horizons
        try:
            short_bars = int(os.getenv("SETUP_LABEL_SHORT_BARS", "1") or "1")
        except Exception:
            short_bars = 1
        short_bars = max(1, short_bars)

        hb_main = max(1, int(horizon_bars))
        hb_short = min(short_bars, hb_main)

        # Entry price
        ent = float(df["close"].iloc[idx_ref])
        if not np.isfinite(ent) or ent <= 0:
            return None, {}

        def _safe_ret(hb: int) -> float:
            try:
                hb_eff = min(hb, max(1, n - idx_ref - 1))
                close_h = float(df["close"].iloc[idx_ref + hb_eff])
                if not np.isfinite(close_h):
                    return 0.0
                return (close_h - ent) / max(abs(ent), 1e-6)
            except Exception:
                return 0.0

        ret_main = _safe_ret(hb_main)
        ret_short = _safe_ret(hb_short)

        # Volatility-aware epsilon
        try:
            eps_bp = float(os.getenv("SETUP_LABEL_EPS_BP", "5") or "5")
        except Exception:
            eps_bp = 5.0
        bps_eps = eps_bp * 1e-4

        try:
            atr_1t = float(features_for_log.get("atr_1t", features_for_log.get("atr_3t", 0.0)))
            if not np.isfinite(atr_1t):
                atr_1t = 0.0
        except Exception:
            atr_1t = 0.0

        try:
            atr_mult = float(os.getenv("SETUP_LABEL_ATR_EPS_MULT", "0.5") or "0.5")
        except Exception:
            atr_mult = 0.5

        eps_atr = atr_mult * (atr_1t / max(abs(ent), 1e-6)) if (atr_1t > 0.0) else 0.0
        eps = max(bps_eps, eps_atr, 0.00025)

        if ret_main > eps:
            label = "BUY"
        elif ret_main < -eps:
            label = "SELL"
        else:
            label = "FLAT"

        # Short-horizon directional tag for analysis only
        if ret_short > eps:
            short_dir = 1
        elif ret_short < -eps:
            short_dir = -1
        else:
            short_dir = 0

        aux_info = {
            "aux_ret_main": float(ret_main),
            "aux_ret_short": float(ret_short),
            "aux_label_short": float(short_dir),
            "trap_long": float(trap_long),
            "trap_short": float(trap_short),
        }

        logger.info(
            "[%s] [LABEL-SETUP] hb=%d ref=%s tgt=%s ret=%.6f eps=%.6f label=%s setup_side=%+d votes=%d micro_imb=%.3f flow=%.3f vwap_dev=%.4f traps(L/S)=%s/%s",
            name,
            hb_main,
            str(df.index[idx_ref]),
            str(df.index[min(idx_ref + hb_main, n - 1)]),
            float(ret_main),
            float(eps),
            label,
            int(setup_side),
            int(evidence_votes),
            float(micro_imb),
            float(flow_score),
            float(fut_vwap),
            bool(trap_long),
            bool(trap_short),
        )
        return label, aux_info
    except Exception as e:
        logger.error(f"[{name}] [LABEL-SETUP] Failed: {e}", exc_info=True)
        return None, {}

# ========== FUTURES SIDECAR FEATURE INGEST ==========

_FUT_CACHE = {"mtime": None, "last_row": None, "prev_row": None}

def _read_latest_fut_features(path: str, spot_last_px: float) -> Dict[str, float]:
    """
    Read latest futures VWAP/CVD/volume features from sidecar CSV.
    Returns bounded, NaN-safe regime/orderflow proxies.
    """
    feats: Dict[str, float] = {}
    try:
        if not path or not os.path.exists(path):
            return feats

        mtime = os.path.getmtime(path)
        if _FUT_CACHE["mtime"] != mtime:
            df = pd.read_csv(path)
            if df is None or df.empty:
                return feats

            # Headerless sidecar fallback: assign expected columns when missing
            if "session_vwap" not in df.columns:
                expected = ["ts", "open", "high", "low", "close", "volume", "tick_count", "session_vwap", "cvd", "cum_volume"]
                cols = expected[: len(df.columns)]
                cols += [f"col_{i}" for i in range(len(df.columns) - len(cols))]
                df.columns = cols

            if "cum_volume" not in df.columns:
                if "volume" in df.columns:
                    df["cum_volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0.0).cumsum()
                elif "tick_count" in df.columns:
                    df["cum_volume"] = pd.to_numeric(df["tick_count"], errors="coerce").fillna(0.0).cumsum()
                else:
                    df["cum_volume"] = 0.0
            if "cvd" not in df.columns:
                df["cvd"] = 0.0

            df = df.tail(2).copy()
            _FUT_CACHE["mtime"] = mtime
            _FUT_CACHE["prev_row"] = df.iloc[0].to_dict() if len(df) > 1 else None
            _FUT_CACHE["last_row"] = df.iloc[-1].to_dict()

        last = _FUT_CACHE["last_row"] or {}
        prev = _FUT_CACHE["prev_row"] or {}

        cur_vwap = float(last.get("session_vwap", 0.0) or 0.0)
        cur_cvd = float(last.get("cvd", 0.0) or 0.0)
        cur_vol = float(last.get("cum_volume", 0.0) or 0.0)

        prev_cvd = float(prev.get("cvd", cur_cvd) or cur_cvd)
        prev_vol = float(prev.get("cum_volume", cur_vol) or cur_vol)

        cvd_delta = cur_cvd - prev_cvd
        vol_delta = cur_vol - prev_vol

        # Bounded order-flow proxies
        cvd_norm = float(np.tanh(cvd_delta / max(1.0, cur_vol)))
        vol_norm = float(np.tanh(vol_delta / 10000.0))

        if cur_vwap > 0.0 and spot_last_px > 0.0:
            vwap_dev = (spot_last_px - cur_vwap) / max(1e-9, cur_vwap)
        else:
            vwap_dev = 0.0
        vwap_dev = float(np.clip(vwap_dev, -0.01, 0.01))

        feats.update({
            "fut_session_vwap": cur_vwap,
            "fut_vwap_dev": vwap_dev,
            "fut_cvd_delta": cvd_norm,
            "fut_vol_delta": vol_norm,
        })
    except Exception as e:
        logger.debug(f"[FUT] read_latest_fut_features failed: {e}", exc_info=True)
    return feats


def _compute_vol_features(candle_df: pd.DataFrame) -> Dict[str, float]:
    feats: Dict[str, float] = {}
    try:
        if candle_df is None or candle_df.empty:
            return feats
        df = candle_df.tail(30).copy()

        hi = df["high"].astype(float).values
        lo = df["low"].astype(float).values
        cl = df["close"].astype(float).values

        rng = np.maximum(0.0, hi - lo)
        feats["atr_1t"] = float(np.mean(rng[-5:])) if len(rng) >= 5 else float(np.mean(rng))
        feats["atr_3t"] = float(np.mean(rng[-15:])) if len(rng) >= 15 else feats["atr_1t"]

        if len(cl) >= 10:
            rets = np.diff(cl) / np.maximum(1e-9, cl[:-1])
            feats["rv_10"] = float(np.std(rets[-10:]))
        else:
            feats["rv_10"] = 0.0
    except Exception as e:
        logger.debug(f"[VOL] compute_vol_features failed: {e}", exc_info=True)
    return feats


def _time_of_day_features(ts: datetime) -> Dict[str, float]:
    """
    Encode minutes since 09:15 IST as sin/cos.
    """
    feats: Dict[str, float] = {}
    try:
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=IST)

        open_ts = ts.replace(hour=9, minute=15, second=0, microsecond=0)
        mins = max(0.0, (ts - open_ts).total_seconds() / 60.0)
        ang = 2.0 * np.pi * (mins / 375.0)  # ~6.25h session
        feats["tod_sin"] = float(np.sin(ang))
        feats["tod_cos"] = float(np.cos(ang))
    except Exception:
        feats["tod_sin"] = 0.0
        feats["tod_cos"] = 0.0
    return feats


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        if np.isfinite(v):
            return v
    except Exception:
        pass
    return float(default)


def _scalper_playbook_line(
    *,
    features_raw: Mapping[str, Any],
    dir_overall: str,
    tradeable: bool,
    gate_reasons: List[str],
    rule_dir: str,
    teacher_strength: float,
    p_success: Optional[float],
    conf_bucket: str,
    struct_setup_side: Optional[int] = None,
) -> str:
    """Return a human/scalper friendly one-liner for logs."""
    o = _safe_float(features_raw.get("open", 0.0))
    h = _safe_float(features_raw.get("high", 0.0))
    l = _safe_float(features_raw.get("low", 0.0))
    close = _safe_float(features_raw.get("close", features_raw.get("last_close", 0.0)))
    ema_fast = _safe_float(features_raw.get("ema_9", features_raw.get("ema_8", features_raw.get("ema8", 0.0))))
    ema_mid = _safe_float(features_raw.get("ema_9", features_raw.get("ema_8", 0.0)))
    ema_slow = _safe_float(features_raw.get("ema_21", features_raw.get("ema21", 0.0)))
    vwap_dev = _safe_float(features_raw.get("fut_vwap_dev", features_raw.get("vwap_dev", 0.0)))

    flow_score, flow_side, micro_imb, fut_cvd, fut_vwap, vwap_side = compute_flow_signal(features_raw)
    struct_score, struct_side = compute_structure_score(features_raw)
    if struct_setup_side is None:
        struct_setup_side = struct_side

    
    # EMA module fields (computed on candle-synced series)
    ema_chop = bool(_safe_float(features_raw.get("ema_regime_chop_5t", 0.0)) > 0.5)
    ema_bias = int(round(_safe_float(features_raw.get("ema_bias_5t", 0.0))))
    ema_break = int(round(_safe_float(features_raw.get("ema15_break_veto", 0.0))))
    tag_code = int(round(_safe_float(features_raw.get("ema_entry_tag", 0.0))))
    tag_map = {0: "NONE", 1: "PULLBACK", 2: "RETEST", 3: "XOVER_CONF"}
    entry_tag = tag_map.get(tag_code, "NONE")

    # Decision-trend tags (decision TF)
    uptrend = (ema_mid >= ema_slow) and (close >= ema_mid)
    downtrend = (ema_mid <= ema_slow) and (close <= ema_mid)

    # 'chop' is primarily HTF chop veto; fall back to local undecided state
    chop = ema_chop or ((not uptrend) and (not downtrend))

    # Extract flow regime (used for conditional planning logic)
    flow_regime = str(features_raw.get("flow_regime", "") or "").upper()

    ema15_break_dn = (ema_break < 0)
    ema15_break_up = (ema_break > 0)

    # Optional breakdown flag: prefer EMA15 breakdown; allow explicit structural flag if present
    breakdown = bool(ema15_break_dn) or ( _safe_float(features_raw.get("struct_breakdown", 0.0)) > 0.5 ) or ( _safe_float(features_raw.get("breakdown", 0.0)) > 0.5 )
# Translate into a scalper-style sentence

    # Translate into a scalper-style sentence (action-consistent)
    if not tradeable:
        if "ema15_break_against" in gate_reasons:
            plan = "EMA15 break against: wait for retest/hold before committing"
        elif "ema15_break_veto" in gate_reasons or ema15_break_dn or ema15_break_up:
            if ema15_break_dn:
                plan = "EMA15 breakdown veto: bearish momentum shift; avoid longs. Wait for retest/reclaim before acting"
            elif ema15_break_up:
                plan = "EMA15 breakout veto: momentum pop; avoid fading. Wait for pullback/hold before acting"
            else:
                plan = "EMA15 veto: wait for cleaner alignment"
        elif "ema_chop_veto" in gate_reasons or chop:
            if flow_regime == "BEAR_LOCK":
                plan = "Flow locked bearish; only consider shorts on pullback into EMA9/15 or breakdown-retest. Avoid chasing"
            elif flow_regime == "BULL_LOCK":
                plan = "Flow locked bullish; only consider longs on pullback into EMA9/15 or breakout-retest. Avoid chasing"
            else:
                plan = "HTF chop: no edge. Wait for trend + structure + flow alignment (A+ only)"
        elif breakdown:
            plan = "Breakdown: wait for reclaim/hold before longs; shorts only if flow+structure confirm"
        else:
            plan = "WAIT: need confirmation (flow/structure alignment) before committing"
    else:
        # Tradeable: keep plan aligned with chosen direction
        if ema15_break_dn and dir_overall == "BUY":
            plan = "Long is tradeable but EMA15 breakdown is fresh: consider skipping or demand extra confirmation"
        elif ema15_break_up and dir_overall == "SELL":
            plan = "Short is tradeable but EMA15 breakout is fresh: consider skipping or demand extra confirmation"
        elif entry_tag != "NONE":
            if entry_tag == "PULLBACK":
                plan = "Playbook: pullback/bounce near EMA15/EMA21. Take only if flow+structure agree; SL beyond rejection wick"
            elif entry_tag == "RETEST":
                plan = "Playbook: break & retest with EMA confluence. Enter on rejection; SL beyond retest extreme"
            else:
                plan = "Playbook: crossover-confirmed. Avoid if EMAs flat/tangled; exit if EMAs reconverge"
        elif downtrend and flow_side <= 0:
            plan = "Downtrend: prefer shorts on weak bounces; avoid longs until reclaim above EMA15/VWAP"
        elif uptrend and flow_side >= 0:
            plan = "Uptrend: take pullbacks, don’t chase. Stay with trend while structure remains clean"
        else:
            plan = "Edge: trade small unless flow+structure align strongly"



    # Compact context string
    reg_txt = "CHOP" if ema_chop else ("TREND_UP" if ema_bias > 0 else ("TREND_DN" if ema_bias < 0 else "MIXED"))
    ctx = (
        f"ctx(flow={flow_side:+d},structBias={struct_side:+d},structSetup={struct_setup_side:+d},"
        f"vwap={vwap_side:+d},vwapDev={vwap_dev:+.4f},emaReg={reg_txt},emaBias={ema_bias:+d},tag={entry_tag})"
    )
    if p_success is None:
        probs = "p_success=NA"
    else:
        probs = f"p_success={p_success*100:.0f}%"
    trad = "TRADE" if tradeable else "WAIT"
    if tradeable:
        decision = f"dir={dir_overall} (teacher={rule_dir}) strength={teacher_strength:.3f} {conf_bucket}"
    else:
        decision = f"intent={dir_overall} (teacher={rule_dir}) strength={teacher_strength:.3f} {conf_bucket}"


    # Gate suffix for logging (kept short)
    if gate_reasons:
        max_show = 4
        shown = ",".join(gate_reasons[:max_show])
        if len(gate_reasons) > max_show:
            shown += f",+{len(gate_reasons) - max_show}"
        gate_txt = f" | gate={shown}"
    else:
        gate_txt = ""

    return f"[SCALPER] {trad}: {plan} | {decision} | {probs} | {ctx}{gate_txt}"


def _scalper_action_line(
    *,
    intent: str,
    tradeable: bool,
    policy_authorized: bool,
    reversal_risk: bool,
    prev_intent: Optional[str],
    prev_tradeable: bool,
    gate_reasons: List[str],
) -> str:
    """Short, chart-friendly scalper action line."""
    intent = str(intent or "FLAT").upper()
    if reversal_risk and intent in ("BUY", "SELL"):
        action = "AVOID_REVERSAL"
    elif tradeable and policy_authorized and intent in ("BUY", "SELL"):
        action = f"ENTER_{intent}"
    elif prev_tradeable and prev_intent == intent and intent in ("BUY", "SELL"):
        action = f"HOLD_IF_IN_{intent}"
    elif intent in ("BUY", "SELL"):
        action = "WAIT_CONFIRM"
    else:
        action = "NO_TRADE"
    gate_hint = "n/a"
    if gate_reasons:
        gate_hint = ",".join(gate_reasons[:3])
    return f"[SCALPER] ACTION: {action} | gates={gate_hint}"


# ========== HEARTBEAT / WATCHDOG / WS UTILITIES ==========

async def _websocket_heartbeat(name: str, ws_handler: WSHandler, interval_sec: int = 30):
    while True:
        try:
            ticks = getattr(ws_handler, "tick_count", 0)
            last_ts = getattr(ws_handler, "last_packet_time", None)
            if last_ts and hasattr(last_ts, "timestamp"):
                age = max(0.0, time.time() - last_ts.timestamp())
                logger.info("[%s] Websocket is active. ticks=%d, last_packet_age=%.1fs", name, ticks, age)
            else:
                logger.info("[%s] Websocket is active. ticks=%d, awaiting first packet...", name, ticks)
        except asyncio.CancelledError:
            logger.info(f"[{name}] Heartbeat cancelled")
            break
        except Exception as e:
            logger.debug(f"[{name}] Heartbeat error (ignored): {e}")
        finally:
            await asyncio.sleep(interval_sec)

def _build_dhan_ws_url(cfg: Any) -> Optional[str]:
    try:
        tok_b64 = getattr(cfg, "dhan_access_token_b64", "") or ""
        cid_b64 = getattr(cfg, "dhan_client_id_b64", "") or ""
        if not tok_b64 or not cid_b64:
            return None, {}
        access_token = base64.b64decode(tok_b64).decode("utf-8")
        client_id = base64.b64decode(cid_b64).decode("utf-8")
        return ("wss://api-feed.dhan.co"
                f"?version=2&token={access_token}&clientId={client_id}&authType=2")
    except Exception as e:
        logger.error(f"Failed to build Dhan WS URL: {e}")
        return None

def _subscription_payload(cfg: Any) -> Dict[str, Any]:
    return {
        "RequestCode": 15,
        "InstrumentCount": 1,
        "InstrumentList": [{
            "ExchangeSegment": getattr(cfg, "nifty_exchange_segment", "IDX_I"),
            "SecurityId": str(getattr(cfg, "nifty_security_id", "")),
        }]
    }

async def _data_stall_watchdog(name: str, ws_handler: WSHandler, resubscribe_cb, reconnect_cb, stall_secs: int, reconnect_secs: int):
    did_resubscribe = False
    last_sub_time: Optional[datetime] = None

    def set_last_sub_time(ts: datetime):
        nonlocal last_sub_time
        last_sub_time = ts

    setattr(reconnect_cb, "_set_last_sub_time", set_last_sub_time)

    while True:
        try:
            await asyncio.sleep(1)
            now = datetime.now(IST)
            last_pkt = getattr(ws_handler, "last_packet_time", None)
            if last_pkt is None:
                if last_sub_time:
                    since_sub = (now - last_sub_time).total_seconds()
                    logger.info(f"[{name}] Watchdog: {since_sub:.1f}s since subscribe, no packets yet")
                    if since_sub >= stall_secs and not did_resubscribe:
                        logger.warning(f"[{name}] No data for {stall_secs}s after subscribe — re-subscribing")
                        try:
                            await resubscribe_cb()
                            did_resubscribe = True
                            logger.info(f"[{name}] Resubscribe issued at {now.strftime('%H:%M:%S')}")
                        except Exception as e:
                            logger.error(f"[{name}] Resubscribe failed: {e}")
                    if since_sub >= reconnect_secs:
                        logger.warning(f"[{name}] No data for {reconnect_secs}s — reconnecting")
                        await reconnect_cb()
                        break
                continue
            else:
                did_resubscribe = False
        except asyncio.CancelledError:
            logger.debug(f"[{name}] Watchdog cancelled")
            break
        except Exception as e:
            logger.error(f"[{name}] Watchdog error: {e}", exc_info=True)
            await asyncio.sleep(2)

async def _ws_connect_and_stream(name: str, cfg: Any, ws_handler: WSHandler, stop_event: asyncio.Event):
    backoff_base = int(getattr(cfg, "reconnect_delay_base", 2)) or 2
    max_attempts_cfg = getattr(cfg, "max_reconnect_attempts", 5)
    try:
        max_attempts = int(max_attempts_cfg)
    except Exception:
        max_attempts = 5
    infinite = max_attempts <= 0

    while not stop_event.is_set():
        ws_url = _build_dhan_ws_url(cfg)
        if not ws_url:
            logger.critical(f"[{name}] Missing or invalid credentials; cannot build WS URL")
            return

        attempt = 0
        while (infinite or attempt < max_attempts) and not stop_event.is_set():
            attempt += 1
            backoff = min(backoff_base * (2 ** (attempt - 1)), 60)
            try:
                logger.info(f"[{name}] Connecting to wss://api-feed.dhan.co?[masked] (attempt {attempt}/{max_attempts if not infinite else '∞'})")
                ping_interval = int(getattr(cfg, "ws_ping_interval", 30)) or 30
                ping_timeout = int(getattr(cfg, "ws_ping_timeout", 10)) or 10
                async with websockets.connect(
                    ws_url,
                    ping_interval=ping_interval,
                    ping_timeout=ping_timeout,
                    max_size=10 * 1024 * 1024,
                    compression=None,
                    open_timeout=30,
                    close_timeout=10
                ) as ws:
                    logger.info(f"[{name}] WebSocket connected")
                    sub = _subscription_payload(cfg)
                    last_sub_time = datetime.now(IST)
                    await ws.send(json.dumps(sub))
                    logger.info(f"[{name}] Subscription sent at {last_sub_time.strftime('%H:%M:%S')}")

                    stall_secs = int(getattr(cfg, "data_stall_seconds", 15)) or 15
                    reconn_secs = int(getattr(cfg, "data_stall_reconnect_seconds", 30)) or 30

                    async def resubscribe():
                        nonlocal last_sub_time
                        await ws.send(json.dumps(sub))
                        last_sub_time = datetime.now(IST)
                        logger.info(f"[{name}] Resubscription sent at {last_sub_time.strftime('%H:%M:%S')}")

                    async def reconnect():
                        try:
                            await ws.close()
                        except Exception:
                            pass

                    watchdog_task = asyncio.create_task(
                        _data_stall_watchdog(name, ws_handler, resubscribe, reconnect, stall_secs, reconn_secs)
                    )
                    setter = getattr(reconnect, "_set_last_sub_time", None)
                    if callable(setter):
                        setter(last_sub_time)

                    try:
                        async for message in ws:
                            if stop_event.is_set():
                                break
                            if isinstance(message, bytes):
                                tick_data = None
                                try:
                                    if len(message) == 16 and message and message[0] == ws_handler.TICKER_PACKET:
                                        if hasattr(ws_handler, "_parse_ticker_packet"):
                                            tick_data = ws_handler._parse_ticker_packet(message)
                                except Exception as e:
                                    logger.error(f"[{name}] Parse error: {e}", exc_info=True)
                                if tick_data:
                                    await ws_handler._process_tick(tick_data)
                            else:
                                try:
                                    data = json.loads(message)
                                    code = data.get("ResponseCode")
                                    if code:
                                        logger.info(f"[{name}] Control: code={code} msg={data.get('ResponseMessage', '')}")
                                except Exception:
                                    logger.debug(f"[{name}] Text message: {str(message)[:200]}")
                            if ws_handler.last_packet_time is None and callable(setter):
                                setter(last_sub_time)
                    except asyncio.CancelledError:
                        logger.info(f"[{name}] Message loop cancelled")
                        raise
                    except websockets.exceptions.ConnectionClosed as e:
                        logger.warning(f"[{name}] WS connection closed: {e}")
                    except Exception as e:
                        logger.error(f"[{name}] Fatal message loop error: {e}", exc_info=True)
                    finally:
                        try:
                            watchdog_task.cancel()
                            await asyncio.wait_for(watchdog_task, timeout=3.0)
                        except Exception:
                            pass

                if stop_event.is_set():
                    return
                logger.warning(f"[{name}] Message loop ended; reconnecting after {backoff}s")
                await asyncio.sleep(backoff)
            except asyncio.CancelledError:
                logger.info(f"[{name}] Connector cancelled")
                return
            except Exception as e:
                level = logger.warning if "no close frame" in str(e).lower() else logger.error
                level(f"[{name}] Connect attempt failed: {e}")
                if infinite or attempt < max_attempts:
                    logger.info(f"[{name}] Retrying in {backoff}s")
                    await asyncio.sleep(backoff)
                else:
                    logger.critical(f"[{name}] Failed to establish WS after {max_attempts} attempts")
                    return
        if stop_event.is_set():
            return
        await asyncio.sleep(1)


# ========== HELPERS ==========

def _extract_best_and_mid_from_tick(tick: Dict[str, Any]) -> Tuple[float, float]:
    try:
        ltp = float(tick.get('ltp', 0.0))
    except Exception:
        ltp = 0.0
    bid = ask = ltp
    try:
        md = tick.get('market_depth')
        if isinstance(md, list) and md:
            best = md[0]
            bid = float(best.get('bid_price', bid))
            ask = float(best.get('ask_price', ask))
    except Exception:
        pass
    mid = (bid + ask) / 2.0 if (bid > 0 and ask > 0) else ltp
    best_price = ltp if ltp > 0 else mid
    return best_price, mid


# ========== MAIN LOOP (PREDICTIVE-ONLY, PRE-CLOSE TRIGGER) ==========

async def main_loop(config, policy_pipe: PolicyPipeline, train_features, token_b64, chat_id):
    logger.info("=" * 60)
    logger.info("STARTING RULE-AS-TEACHER MAIN EVENT LOOP")
    logger.info("=" * 60)
    _log_env_audit([
        "HTF_MIN_CONS", "RULE_MIN_SIG", "FLOW_VWAP_EXT", "FLOW_STRONG_MIN", "STRUCT_OPPOSE_FLOW_MIN",
        "POLICY_MIN_SUCCESS", "POLICY_BUY_PATH", "POLICY_SELL_PATH",
        "POLICY_VETO_STRICT", "POLICY_MIN_SIZE_MULT",
        "CALIB_BUY_PATH", "CALIB_SELL_PATH",
        "FEATURE_SCHEMA_COLS_PATH", "POLICY_SCHEMA_COLS_PATH",
        "TREND_MIN_SIGNALS", "TREND_LANE_SIZE_MULT",
        "LANE_SCORE_MIN", "EMA_CHOP_HARD_MIN", "PEN_NO_SETUP",
        "TAPE_YELLOW_SIZE_MULT", "REQUIRE_SETUP", "LEGACY_TRAIN_CSV",
        "ENABLE_INDICATOR_MODULATION", "IND_MOD_SCALE",
        "ENABLE_PATTERN_MODULATION", "PAT_MOD_SCALE",
        "REGIME_HOLD_BARS", "DYNAMIC_THRESHOLDS", "DYN_THRESH_EMA_ALPHA", "DYN_THRESH_UPDATE_EVERY",
        "REGIME_FLOW_TREND_MIN", "REGIME_FLOW_CHOP_MAX", "REGIME_VWAP_TREND_MIN", "REGIME_VWAP_CHOP_MAX",
        "VOL_BAND_LOW_MULT", "VOL_BAND_HIGH_MULT",
        "HTF_VETO_MODE", "HTF_VETO_SOFT_FLOW_MIN",
        "REVERSAL_IMB_MIN", "REVERSAL_CVD_MIN", "REVERSAL_VWAP_MIN", "REVERSAL_SLOPE_MIN", "PEN_REVERSAL_RISK",
        "USE_MOVE_HEAD", "MOVE_HEAD_MODE", "MOVE_HEAD_FALLBACK_PROXY",
        "MOVE_EDGE_MIN", "MOVE_EDGE_TREND_ONLY", "MOVE_PROXY_ATR_MULT", "POLICY_MOVE_PATH",
        "RV_ATR_MIN", "TRAIN_FALLBACK_ATR_MIN_RET",
        "FLOW_LOCK_IMB_MIN", "FLOW_LOCK_CVD_MIN",
    ])

    # Normalize connections (single connection default)
    connections: List[Tuple[str, Any]] = []
    try:
        if hasattr(config, "connections") and isinstance(config.connections, dict):
            for name, cfg in config.connections.items():
                connections.append((str(name), cfg))
        elif hasattr(config, "connections") and hasattr(config, "__iter__"):
            for i, cfg in enumerate(config.connections, start=1):
                sec = getattr(cfg, "nifty_security_id", getattr(cfg, "security_id", "NA"))
                seg = getattr(cfg, "nifty_exchange_segment", getattr(cfg, "exchange_segment", "NA"))
                connections.append((f"conn{i}:{seg}:{sec}", cfg))
        else:
            sec = getattr(config, "nifty_security_id", getattr(config, "security_id", "NA"))
            seg = getattr(config, "nifty_exchange_segment", getattr(config, "exchange_segment", "NA"))
            connections.append((f"primary:{seg}:{sec}", config))
    except Exception:
        connections.append(("primary", config))

    logger.info(f"Total connections configured: {len(connections)}")

    # Shared components
    feat_pipe = FeaturePipeline(train_features=train_features)
    model_pipe = policy_pipe
    if model_pipe is None:
        raise RuntimeError("policy_pipe is required for rule-as-teacher mode")
    logger.info("Global components initialized successfully")
    # Paths
    feature_log_path = getattr(config, 'feature_log_path', 'feature_log.csv')
    logger.info(f"Feature log path: {feature_log_path}")

    # NEW: dedicated training log (avoid schema mixing with legacy feature_log.csv)
    train_log_path = getattr(config, "train_log_path", None) or os.getenv("TRAIN_LOG_PATH", "").strip()
    if not train_log_path:
        logger.error("TRAIN_LOG_PATH missing; training will be disabled.")
    else:
        logger.info(f"Training log path: {train_log_path}")

    # Background online trainer (policy)
    trainer_task = None
    try:
        from online_trainer_regen_v2_bundle import background_trainer_loop
        buy_out = os.getenv("POLICY_BUY_PATH", "trained_models/production/policy_buy.json")
        sell_out = os.getenv("POLICY_SELL_PATH", "trained_models/production/policy_sell.json")
        Path(Path(buy_out).parent or ".").mkdir(parents=True, exist_ok=True)
        Path(Path(sell_out).parent or ".").mkdir(parents=True, exist_ok=True)
        if train_log_path:
            trainer_task = asyncio.create_task(background_trainer_loop(
                feature_log_path=train_log_path,
                buy_out_path=buy_out,
                sell_out_path=sell_out,
                pipeline_ref=model_pipe,
                interval_sec=int(getattr(config, "trainer_interval_sec", 300)) if hasattr(config, "trainer_interval_sec") else 300,
                min_rows=int(getattr(config, "trainer_min_rows", 100)) if hasattr(config, "trainer_min_rows") else 100
            ))
            logger.info("Online trainer task started")
        else:
            logger.error("Online trainer not started: TRAIN_LOG_PATH missing")
    except Exception as e:
        logger.warning(f"Online trainer not started: {e}")

    # Background Platt calibrator (policy)
    calib_task = None
    try:
        from calibrator import background_calibrator_loop
        calib_buy_out = os.getenv("CALIB_BUY_PATH", "trained_models/production/calib_buy.json")
        calib_sell_out = os.getenv("CALIB_SELL_PATH", "trained_models/production/calib_sell.json")
        if train_log_path:
            try:
                calib_max_bytes = int(os.getenv("CALIB_MAX_BYTES", "6000000") or "6000000")
            except Exception:
                calib_max_bytes = 6_000_000
            try:
                calib_max_rows = int(os.getenv("CALIB_MAX_ROWS", "50000") or "50000")
            except Exception:
                calib_max_rows = 50_000
            calib_task = asyncio.create_task(background_calibrator_loop(
                feature_log_path=train_log_path,
                calib_buy_out_path=calib_buy_out,
                calib_sell_out_path=calib_sell_out,
                interval_sec=int(getattr(config, "calib_interval_sec", 1200)) if hasattr(config, "calib_interval_sec") else 1200,
                min_dir_rows=int(getattr(config, "calib_min_rows", 120)) if hasattr(config, "calib_min_rows") else 120,
                max_bytes=calib_max_bytes,
                max_rows=calib_max_rows,
            ))
            logger.info(f"[CALIB] Calibrator task started → buy={calib_buy_out} sell={calib_sell_out}")
        else:
            logger.error("[CALIB] Calibrator not started: TRAIN_LOG_PATH missing")
    except Exception as e:
        logger.warning(f"[CALIB] Calibrator not started: {e}")

    # Drift baseline refresher (optional)
    async def _maybe_refresh_drift_baseline():
        try:
            path = Path(feature_log_path)
            if not path.exists():
                return
            df = pd.read_csv(
                path,
                header=None,
                names=["ts", "decision", "label", "score", "alpha", "tradeable", "is_flat", "ticks", "rest"],
                on_bad_lines='skip',
            )
            keep = ["ema_8","ema_21","spread","micro_slope","micro_imbalance","mean_drift_pct","last_zscore","last_price"]
            baselines = {k: [] for k in keep}
            sample_n = 600
            for row in df.tail(sample_n).itertuples(index=False):
                try:
                    toks = str(row.rest).split(",")
                    for t in toks:
                        if "=" not in t:
                            continue
                        k, v = t.split("=", 1)
                        if k in baselines:
                            baselines[k].append(float(v))
                except Exception:
                    continue
            baselines = {k: v for k, v in baselines.items() if v}
            if baselines:
                feat_pipe.train_features = baselines
                logger.info(f"[DRIFT] Baseline refreshed from feature_log: keys={list(baselines.keys())} samples~{min(len(next(iter(baselines.values()))), sample_n)}")
        except Exception as e:
            logger.debug(f"[DRIFT] Baseline refresh failed (ignored): {e}")

    async def drift_baseline_refresh_loop(interval_sec: int = 600):
        logger.info(f"[DRIFT] Baseline refresh loop started (every {interval_sec}s)")
        while True:
            try:
                await _maybe_refresh_drift_baseline()
            except asyncio.CancelledError:
                logger.info("[DRIFT] Baseline refresh loop cancelled")
                break
            except Exception as e:
                logger.debug(f"[DRIFT] Periodic refresh failed (ignored): {e}")
            await asyncio.sleep(interval_sec)

    baseline_task = asyncio.create_task(drift_baseline_refresh_loop(600))

    # FIX #3: Load rule signal weights from environment
    rule_weight_ind = float(os.getenv("RULE_WEIGHT_IND", "0.50"))
    rule_weight_mtf = float(os.getenv("RULE_WEIGHT_MTF", "0.35"))
    rule_weight_pat = float(os.getenv("RULE_WEIGHT_PAT", "0.15"))
    rule_weight_ta  = float(os.getenv("RULE_WEIGHT_TA",  "0.00"))

    # Normalize to sum to 1.0 (include TA weight if provided)
    total_weight = rule_weight_ind + rule_weight_mtf + rule_weight_pat + rule_weight_ta
    if total_weight > 0:
        rule_weight_ind /= total_weight
        rule_weight_mtf /= total_weight
        rule_weight_pat /= total_weight
        rule_weight_ta  /= total_weight
    logger.info(
        f"Rule weights: IND={rule_weight_ind:.3f} MTF={rule_weight_mtf:.3f} PAT={rule_weight_pat:.3f} TA={rule_weight_ta:.3f}"
    )

    # Per-connection tasks and state
    stop_events: Dict[str, asyncio.Event] = {}
    ws_tasks: List[Tuple[str, WSHandler, asyncio.Task]] = []
    hb_tasks: List[Tuple[str, asyncio.Task]] = []

    for name, cfg in connections:
        try:
            ws_handler = WSHandler(cfg)

            ob_ring = deque(maxlen=10)
            staged_map: Dict[datetime, Dict[str, Any]] = {}   # features staged per reference candle start (2-min horizon)

            decision_state = DecisionState(
                name=name,
                edge_window=int(os.getenv('EDGE_WINDOW', '300') or '300'),
                edge_pctl=float(os.getenv('EDGE_PCTL', '0.85') or '0.85'),
                hyst_bars=int(os.getenv('TREND_HYST_BARS', '3') or '3'),
                summary_every=int(os.getenv('DECISION_SUMMARY_EVERY', '50') or '50'),
            )
            setup_ready_at: Optional[datetime] = None  # setup READY timestamp for delayed entry

            async def _on_tick_cb(tick: Dict[str, Any]):
                try:
                    best_px, mid_px = _extract_best_and_mid_from_tick(tick)
                    ob_snapshot = {
                        'bid_price': best_px if best_px <= mid_px else mid_px,
                        'ask_price': best_px if best_px >= mid_px else mid_px
                    }
                    ob_ring.append(ob_snapshot)
                except Exception as e:
                    logger.debug(f"[{name}] on_tick callback error (ignored): {e}")

            # ---------- PRE-CLOSE: STAGE FEATURES AT CURRENT CANDLE START (2-MIN HORIZON) ----------
            async def _on_preclose_cb(preview_df: pd.DataFrame, full_df: pd.DataFrame):
                try:
                    nonlocal setup_ready_at
                    if preview_df is None or preview_df.empty:
                        return

                    # Current bucket start time t (candle being closed now)
                    current_bucket_start = preview_df.index[-1]

                    interval_sec = int(getattr(cfg, 'candle_interval_seconds', 60))
                    # Trade horizon in minutes (bars); defaults to env TRADE_HORIZON_MIN
                    horizon_min = int(getattr(cfg, "trade_horizon_min",
                                              int(os.getenv("TRADE_HORIZON_MIN", "10") or "10")))
                    horizon_min = max(1, horizon_min)

                    # Reference time for label & training row
                    ref_start = current_bucket_start
                    # Target close candle start (for logging only)
                    target_start = current_bucket_start + timedelta(minutes=horizon_min)
                    # --- Candle-synced EMA/TA base (no tick-EMA) -----------------
                    # We compute EMAs on the SAME candle series whose open/close is used later,
                    # to avoid tick-vs-candle mismatches.
                    interval_sec = int(getattr(cfg, 'candle_interval_seconds', 60))
                    base_min = max(1, int(interval_sec // 60))

                    # Determine timeframes (overrideable by env)
                    # Note: base candles are built at cfg.candle_interval_seconds (default 300s = 5T in run_main.py),
                    # so the safe default is decision_tf == base TF. If you want a 1T decision TF, set
                    # CANDLE_INTERVAL_SECONDS=60 so the candle builder actually produces 1T bars.
                    def _tf_to_min(_tf: str) -> int:
                        try:
                            s = (_tf or "").strip().lower()
                            if not s:
                                return 0
                            if s.endswith("t"):
                                return int(s[:-1])
                            if s.endswith("min"):
                                return int(s[:-3])
                            return int(s)
                        except Exception:
                            return 0

                    decision_tf = (os.getenv("EMA_DECISION_TF", "") or "").strip()
                    filter_tf = (os.getenv("EMA_FILTER_TF", "") or "").strip()

                    if not decision_tf:
                        decision_tf = f"{base_min}T"
                    if not filter_tf:
                        # If base is 1T, default filter is 5T, else filter==decision (stable)
                        filter_tf = "5T" if base_min == 1 else decision_tf

                    # Safety clamps: do not "upsample" EMAs to a smaller TF than the base candle builder.
                    # Resampling 5T candles to 1T does NOT create new information and can confuse break/entry logic.
                    dec_min = _tf_to_min(decision_tf)
                    fil_min = _tf_to_min(filter_tf)
                    if dec_min and dec_min < base_min:
                        logger.warning(f"[{name}] EMA_DECISION_TF={decision_tf} < base TF {base_min}T; clamping to {base_min}T")
                        decision_tf = f"{base_min}T"
                        dec_min = base_min
                    if fil_min and fil_min < dec_min:
                        logger.warning(f"[{name}] EMA_FILTER_TF={filter_tf} < decision TF {decision_tf}; clamping to {decision_tf}")
                        filter_tf = decision_tf


                    # Merge preview candle into full candle stream (preclose preview is near-final)
                    base_df = full_df.tail(1200).copy() if isinstance(full_df, pd.DataFrame) else pd.DataFrame()
                    if isinstance(preview_df, pd.DataFrame) and not preview_df.empty:
                        try:
                            idx = preview_df.index[-1]
                            row = preview_df.iloc[-1]
                            for col in ("open", "high", "low", "close", "volume", "tick_count"):
                                if col in preview_df.columns:
                                    base_df.loc[idx, col] = row.get(col)
                        except Exception:
                            pass

                    # Decision TF candles (same series used for body/wick logic)
                    decision_df = base_df if decision_tf == f"{base_min}T" else FeaturePipeline.resample_ohlc(base_df, decision_tf)
                    filter_df = base_df if filter_tf == decision_tf else FeaturePipeline.resample_ohlc(base_df, filter_tf)

                    if decision_df is None or decision_df.empty or "close" not in decision_df.columns:
                        return

                    # --- Price series used by microstructure features ---
                    try:
                        prices = pd.to_numeric(decision_df["close"], errors="coerce").dropna().astype(float).tolist()
                    except Exception:
                        prices = []

                    if not prices:
                        return  # nothing to compute safely

                    last_px = float(prices[-1])
# Recent candles for MTF patterns and SR bundle
                    safe_df = decision_df.tail(500) if isinstance(decision_df, pd.DataFrame) and not decision_df.empty else pd.DataFrame()

                    # TA/EMA
                    ema_feats = FeaturePipeline.compute_emas(prices[-200:])
                    try:
                        candle_df_ta = safe_df if not safe_df.empty else decision_df.tail(200)
                        ta = TA.compute_ta_bundle(candle_df_ta, ema_feats)
                    except Exception:
                        ta = {}

                    tf_list = getattr(cfg, "pattern_timeframes", ["1T", "3T", "5T"])
                    mtf = FeaturePipeline.compute_mtf_pattern_consensus(
                        candle_df=safe_df,
                        timeframes=tf_list,
                        rvol_window=int(getattr(cfg, 'pattern_rvol_window', 5)),
                        rvol_thresh=float(getattr(cfg, 'pattern_rvol_threshold', 1.2)),
                        min_winrate=float(getattr(cfg, 'pattern_min_winrate', 0.55))
                    ) or {}
                    pattern_features = FeaturePipeline.compute_candlestick_patterns(
                        candles=safe_df.tail(max(3, int(getattr(cfg, 'pattern_rvol_window', 5)))),
                        rvol_window=int(getattr(cfg, 'pattern_rvol_window', 5)),
                        rvol_thresh=float(getattr(cfg, 'pattern_rvol_threshold', 1.2)),
                        min_winrate=float(getattr(cfg, 'pattern_min_winrate', 0.55))
                    ) or {}
                    sr = FeaturePipeline.compute_sr_features(
                        candle_df=safe_df,
                        timeframes=tf_list,
                    ) if isinstance(safe_df, pd.DataFrame) and not safe_df.empty else {
                        "sr_1T_hi_dist": 0.0, "sr_1T_lo_dist": 0.0,
                        "sr_3T_hi_dist": 0.0, "sr_3T_lo_dist": 0.0,
                        "sr_5T_hi_dist": 0.0, "sr_5T_lo_dist": 0.0,
                        "sr_breakout_up": 0.0, "sr_breakout_dn": 0.0
                    }

                    # NEW: structure bundle (pivot swipe / FVG / order-block)
                    try:
                        structure_feats: Dict[str, float] = FeaturePipeline.compute_structure_bundle(
                            safe_df.tail(40) if isinstance(safe_df, pd.DataFrame) else pd.DataFrame()
                        )
                    except Exception:
                        structure_feats = {}

                    # Order-flow and microstructure
                    try:
                        order_books = list(ob_ring)
                    except Exception:
                        order_books = []
                    ofd = FeaturePipeline.order_flow_dynamics(order_books)
                    try:
                        micro_ctx = ws_handler.get_micro_features()
                        px_arr = np.asarray(prices[-64:], dtype=float)
                        last_px = float(px_arr[-1]) if px_arr.size else last_px
                        try:
                            vol_short = float(micro_ctx.get('std_dltp_short', 0.0))
                            denom = max(1e-6, vol_short)
                            micro_slope_normed = float(micro_ctx.get('slope', 0.0)) / denom
                        except Exception:
                            micro_slope_normed = float(micro_ctx.get('slope', 0.0))
                        if float(micro_ctx.get('price_range_tightness', 0.0)) >= 0.995:
                            micro_slope_normed = float(np.tanh(micro_slope_normed))
                    except Exception:
                        micro_ctx = {}
                        micro_slope_normed = 0.0

                    indicator_features = {
                        'ema_trend': 1.0 if float(ema_feats.get('ema_8', 0.0)) > float(ema_feats.get('ema_21', 0.0))
                                     else (-1.0 if float(ema_feats.get('ema_8', 0.0)) < float(ema_feats.get('ema_21', 0.0)) else 0.0),
                        'micro_slope': micro_slope_normed,
                        'imbalance': float(micro_ctx.get('imbalance', 0.0)),
                        'mean_drift': float(micro_ctx.get('mean_drift_pct', 0.0)) / 100.0,
                        'std_dltp_short': float(micro_ctx.get('std_dltp_short', 0.0)),
                        'price_range_tightness': float(micro_ctx.get('price_range_tightness', 0.0)),
                    }
                    try:
                        indicator_score = model_pipe.compute_indicator_score(indicator_features)
                        logger.debug(f"[{name}] Indicator score: {indicator_score:.3f}")
                    except Exception:
                        indicator_score = 0.0

                    # Normalization scale
                    try:
                        px = np.array(prices[-64:], dtype=float)
                        scale = float(np.std(np.diff(px))) if px.size >= 3 else 1.0
                        scale = max(1e-6, scale)
                    except Exception:
                        scale = 1.0

                    # --- NEW: vol/regime/time-of-day features ---
                    vol_feats = _compute_vol_features(safe_df)
                    tod_ts = current_bucket_start if isinstance(current_bucket_start, datetime) else datetime.now(IST)
                    tod_feats = _time_of_day_features(tod_ts)

                    # --- NEW: futures sidecar features ---
                    fut_path = os.getenv("FUT_SIDECAR_PATH", "trained_models/production/fut_candles_vwap_cvd.csv")
                    fut_feats = _read_latest_fut_features(fut_path, spot_last_px=last_px)

                    if fut_feats:
                        logger.debug(f"[FUT] injected futures features: {fut_feats}")

                    # last_zscore
                    try:
                        last_zscore = 0.0
                        if len(prices) >= 2:
                            px_arr64 = np.array(prices[-64:], dtype=float)
                            if px_arr64.size >= 2:
                                px_last = float(px_arr64[-1])
                                px_mean32 = float(np.mean(px_arr64[-32:])) if px_arr64.size >= 32 else float(np.mean(px_arr64))
                                px_std32 = float(np.std(px_arr64[-32:])) if px_arr64.size >= 32 else float(np.std(px_arr64))
                                last_zscore = (px_last - px_mean32) / max(1e-9, px_std32)
                    except Exception:
                        last_zscore = 0.0

                    # --- Reversal / regime flags (wick extremes, VWAP reversion, CVD divergence)
                    try:
                        if isinstance(safe_df, pd.DataFrame) and not safe_df.empty:
                            last_candle = safe_df.iloc[-1]
                            prev_candle = safe_df.iloc[-2] if len(safe_df) >= 2 else last_candle
                        else:
                            # Build a minimal struct for wick calc from prices
                            if prices and len(prices) >= 2:
                                last_candle = {"open": prices[-2], "high": prices[-1], "low": prices[-1], "close": prices[-1]}
                                prev_candle = {"open": prices[-3] if len(prices) >= 3 else prices[-2], "high": prices[-2], "low": prices[-2], "close": prices[-2]}
                            else:
                                last_candle = None
                                prev_candle = None
                    except Exception:
                        last_candle = None
                        prev_candle = None

                    try:
                        wick_up, wick_down = FeaturePipeline._compute_wick_extremes(last_candle) if last_candle is not None else (0.0, 0.0)
                    except Exception:
                        wick_up, wick_down = 0.0, 0.0

                    # VWAP reversion flag – prefer futures session vwap if present
                    vwap_val = None
                    if "fut_session_vwap" in fut_feats:
                        try:
                            vwap_val = float(fut_feats.get("fut_session_vwap", None))
                        except Exception:
                            vwap_val = None

                    try:
                        px_hist_df = safe_df if (isinstance(safe_df, pd.DataFrame) and not safe_df.empty) else None
                        vwap_rev = FeaturePipeline._compute_vwap_reversion_flag(px_hist_df, vwap_val) if px_hist_df is not None else 0.0
                    except Exception:
                        vwap_rev = 0.0

                    # CVD divergence using futures sidecar delta if available
                    try:
                        fut_cvd_delta = fut_feats.get("fut_cvd_delta", None)
                        last_close_val = float(prices[-1]) if prices else 0.0
                        prev_close_val = float(prices[-2]) if len(prices) >= 2 else last_close_val
                        px_change = last_close_val - prev_close_val
                        cvd_div = FeaturePipeline._compute_cvd_divergence(px_change, fut_cvd_delta)
                    except Exception:
                        cvd_div = 0.0

                    rev_cross_feats: Dict[str, float] = FeaturePipeline.compute_reversal_cross_features(
                        safe_df.tail(20) if isinstance(safe_df, pd.DataFrame) else pd.DataFrame(),
                        {
                            "wick_extreme_up": float(wick_up),
                            "wick_extreme_down": float(wick_down),
                            "cvd_divergence": float(cvd_div),
                            "vwap_reversion_flag": float(vwap_rev),
                        },
                    )
                    # --- EMA module (regime/bias/entry-tag/veto) ---------------
                    ema_module_feats, ema_meta = FeaturePipeline.compute_ema_module(
                        decision_df=decision_df,
                        filter_df=filter_df,
                        decision_tf=decision_tf,
                        filter_tf=filter_tf,
                    )


                    ta_feats = ta or {}
                    pat_feats = pattern_features or {}
                    mtf_feats = mtf or {}

                    try:
                        last_candle_dict = last_candle if isinstance(last_candle, dict) else (last_candle.to_dict() if last_candle is not None else {})
                    except Exception:
                        last_candle_dict = {}

                    # Backward-compatible alias (older code used `last`)
                    last = last_candle_dict

                    features_raw = {
                        **ema_feats,
                        **(ema_module_feats or {}),
                        **ta_feats,
                        **pat_feats,
                        **mtf_feats,
                        **sr,
                        **structure_feats,  # NEW: pivot/FVG/order-block structure
                        **indicator_features,
                        **ofd,
                        **vol_feats,
                        **tod_feats,
                        **fut_feats,
                        'micro_slope': micro_slope_normed,
                        'micro_imbalance': float(micro_ctx.get('imbalance', 0.0)),
                        'mean_drift_pct': float(micro_ctx.get('mean_drift_pct', 0.0)),
                        'last_price': float(last_px),
                        'last_zscore': float(last_zscore),
                        'std_dltp_short': float(micro_ctx.get('std_dltp_short', 0.0)),
                        'price_range_tightness': float(micro_ctx.get('price_range_tightness', 0.0)),
                        "indicator_score": float(indicator_score),
                        # Reversal / regime flags
                        "wick_extreme_up": float(wick_up),
                        "wick_extreme_down": float(wick_down),
                        "vwap_reversion_flag": float(vwap_rev),
                        "cvd_divergence": float(cvd_div),
                        "open": float(last_candle_dict.get("open", np.nan)) if "open" in last_candle_dict else float(np.nan),
                        "high": float(last_candle_dict.get("high", np.nan)) if "high" in last_candle_dict else float(np.nan),
                        "low": float(last_candle_dict.get("low", np.nan)) if "low" in last_candle_dict else float(np.nan),
                        "close": float(last_candle_dict.get("close", np.nan)) if "close" in last_candle_dict else float(np.nan),
                    }
                    # Setup flags (pivot/FVG/OB)
                    is_pivot_swipe_up = bool(features_raw.get("struct_pivot_swipe_up", 0))
                    is_pivot_swipe_down = bool(features_raw.get("struct_pivot_swipe_down", 0))
                    is_fvg_up = bool(features_raw.get("struct_fvg_up_present", 0))
                    is_fvg_down = bool(features_raw.get("struct_fvg_down_present", 0))
                    is_ob_bull = bool(features_raw.get("struct_ob_bull_present", 0))
                    is_ob_bear = bool(features_raw.get("struct_ob_bear_present", 0))

                    try:
                        ob_max_pct = float(os.getenv("OB_RELEVANCE_MAX_PCT", "0.0020") or "0.0020")
                        ob_max_atr_mult = float(os.getenv("OB_RELEVANCE_ATR_MULT", "0.35") or "0.35")
                    except Exception:
                        ob_max_pct = 0.0020
                        ob_max_atr_mult = 0.35
                    try:
                        last_close = float(features_raw.get("close", features_raw.get("last_price", 0.0)) or 0.0)
                        atr = float(features_raw.get("atr_1t", features_raw.get("atr_3t", 0.0)) or 0.0)
                    except Exception:
                        last_close = 0.0
                        atr = 0.0
                    max_rel = ob_max_pct
                    if last_close > 0.0 and atr > 0.0:
                        max_rel = min(ob_max_pct, (ob_max_atr_mult * atr) / last_close)
                    bull_dist = abs(float(features_raw.get("struct_ob_bull_dist", 0.0) or 0.0))
                    bear_dist = abs(float(features_raw.get("struct_ob_bear_dist", 0.0) or 0.0))
                    ob_bull_valid = bool(is_ob_bull and (bull_dist <= max_rel) and (is_pivot_swipe_up or is_fvg_up))
                    ob_bear_valid = bool(is_ob_bear and (bear_dist <= max_rel) and (is_pivot_swipe_down or is_fvg_down))
                    try:
                        ob_decay = int(os.getenv("OB_DECAY_BARS", "8") or "8")
                    except Exception:
                        ob_decay = 8
                    if ob_bull_valid:
                        decision_state._ob_bull_age = 0
                    elif is_ob_bull:
                        decision_state._ob_bull_age = int(getattr(decision_state, "_ob_bull_age", 0)) + 1
                    else:
                        decision_state._ob_bull_age = 0
                    if ob_bear_valid:
                        decision_state._ob_bear_age = 0
                    elif is_ob_bear:
                        decision_state._ob_bear_age = int(getattr(decision_state, "_ob_bear_age", 0)) + 1
                    else:
                        decision_state._ob_bear_age = 0
                    if int(getattr(decision_state, "_ob_bull_age", 0)) >= ob_decay:
                        ob_bull_valid = False
                    if int(getattr(decision_state, "_ob_bear_age", 0)) >= ob_decay:
                        ob_bear_valid = False
                    features_raw["struct_ob_bull_valid"] = int(ob_bull_valid)
                    features_raw["struct_ob_bear_valid"] = int(ob_bear_valid)

                    is_bull_setup = is_pivot_swipe_up or is_fvg_up or ob_bull_valid
                    is_bear_setup = is_pivot_swipe_down or is_fvg_down or ob_bear_valid

                    # --- scalper filter: structure becomes *directional setup* only if flow+VWAP allow it ---
                    flow_score0, flow_side0, micro0, fut_cvd0, fut_vwap0, vwap_side0 = compute_flow_signal(features_raw)

                    # if structure says bull but we are extended below VWAP with sell/neutral flow → downgrade
                    if is_bull_setup and (vwap_side0 < 0) and (flow_side0 <= 0):
                        logger.info(
                            "[%s] [STRUCT-FILTER] Downgrade bull_setup (level-only): vwap_side=%+d flow_side=%+d fut_vwap_dev=%.4f micro_imb=%.3f fut_cvd=%.6f",
                            name, vwap_side0, flow_side0, fut_vwap0, micro0, fut_cvd0
                        )
                        is_bull_setup = False

                    # symmetric for bear setup
                    if is_bear_setup and (vwap_side0 > 0) and (flow_side0 >= 0):
                        logger.info(
                            "[%s] [STRUCT-FILTER] Downgrade bear_setup (level-only): vwap_side=%+d flow_side=%+d fut_vwap_dev=%.4f micro_imb=%.3f fut_cvd=%.6f",
                            name, vwap_side0, flow_side0, fut_vwap0, micro0, fut_cvd0
                        )
                        is_bear_setup = False

                    ambiguous_setup = is_bull_setup and is_bear_setup
                    is_any_setup = is_bull_setup or is_bear_setup
                    fast_setup_ready = bool(
                        (is_bull_setup and flow_side0 > 0) or (is_bear_setup and flow_side0 < 0)
                    )

                    # Structure diagnostics: explicit log line for pivot/FVG/OB
                    try:
                        logger.info(
                            "[%s] [STRUCT] pivot_swipe_up=%d pivot_swipe_down=%d "
                            "fvg_up=%d fvg_down=%d ob_bull=%d ob_bear=%d "
                            "bull_setup=%d bear_setup=%d",
                            name,
                            int(is_pivot_swipe_up),
                            int(is_pivot_swipe_down),
                            int(is_fvg_up),
                            int(is_fvg_down),
                            int(is_ob_bull),
                            int(is_ob_bear),
                            int(is_bull_setup),
                            int(is_bear_setup),
                        )
                    except Exception:
                        pass

                    features_raw["is_bull_setup"] = int(is_bull_setup)
                    features_raw["is_bear_setup"] = int(is_bear_setup)
                    features_raw["is_any_setup"] = int(is_any_setup)
                    features_raw["fast_setup_ready"] = int(fast_setup_ready)
                    try:
                        tape_valid = bool(float(features_raw.get("volume", 0.0) or 0.0) > 0.0)
                    except Exception:
                        tape_valid = False
                    if (not tape_valid) and _safe_getenv_bool("TAPE_VALID_USE_TICKS", default=False):
                        try:
                            tape_valid = bool(float(features_raw.get("tick_count", 0.0) or 0.0) > 0.0)
                        except Exception:
                            tape_valid = False
                    if (not tape_valid) and _safe_getenv_bool("TAPE_VALID_USE_FUTURES", default=False):
                        try:
                            fut_vol_delta = float(features_raw.get("fut_vol_delta", 0.0) or 0.0)
                            fut_cvd_delta = float(features_raw.get("fut_cvd_delta", 0.0) or 0.0)
                            tape_valid = bool(abs(fut_vol_delta) > 0.0 or abs(fut_cvd_delta) > 0.0)
                        except Exception:
                            tape_valid = False
                    features_raw["tape_valid"] = int(tape_valid)

                    dyn_thresholds = None
                    if hasattr(decision_state, "dynamic_thresholds"):
                        dyn_thresholds = getattr(decision_state, "dynamic_thresholds")
                    dyn_values = dyn_thresholds.curr if getattr(dyn_thresholds, "enabled", False) else None
                    vwap_ext_override = None
                    if isinstance(dyn_values, dict) and "FLOW_VWAP_EXT" in dyn_values:
                        vwap_ext_override = float(dyn_values.get("FLOW_VWAP_EXT", 0.0))

                    # Compute flow once (and log once). Downstream modules reuse cached values.
                    compute_flow_signal(features_raw, log=True, vwap_ext_override=vwap_ext_override)
                    features_raw.update(rev_cross_feats)
                    features = FeaturePipeline.normalize_features(features_raw, scale=scale)

                    # Build a fused EMA/MTF + pattern rule signal (primary dir source)
                    try:
                        ind_score = float(indicator_score) if indicator_score is not None else 0.0
                    except Exception:
                        ind_score = 0.0

                    try:
                        mtf_cons = float(mtf.get("mtf_consensus", 0.0)) if mtf else 0.0
                    except Exception:
                        mtf_cons = 0.0

                    pat_adj = 0.0
                    if pattern_features:
                        try:
                            pat_adj = float(pattern_features.get("probability_adjustment", 0.0))
                        except Exception:
                            pat_adj = 0.0

                    ta_rule = compute_ta_rule_signal(features_raw)

                    # Combined rule signal: indicators + MTF + patterns + TA
                    rule_sig = (rule_weight_ind * ind_score) + (rule_weight_mtf * mtf_cons) + (rule_weight_pat * pat_adj) + (rule_weight_ta * ta_rule)

                    struct_score, struct_side = compute_structure_score(features_raw)
                    features_raw["struct_side"] = int(struct_side)
                    features_raw["struct_score"] = float(struct_score)

                    regime_candidate = "SIDEWAYS"
                    reversal_risk = False
                    trap_flag = bool(float(features_raw.get("trap_long", 0.0)) > 0.5 or float(features_raw.get("trap_short", 0.0)) > 0.5)
                    try:
                        regime_info = classify_regime(features_raw=features_raw)
                        regime_candidate = str(regime_info.get("candidate") or "SIDEWAYS")
                        reversal_risk = bool(regime_info.get("reversal_risk", False))
                    except Exception:
                        regime_candidate = "SIDEWAYS"
                        reversal_risk = False

                    base_regime = regime_candidate
                    if regime_candidate.startswith("TREND") and reversal_risk:
                        regime_candidate = "REVERSAL_RISK"

                    regime_state = {"regime": "SIDEWAYS", "regime_age": 0, "vol_band": "NORMAL"}
                    if hasattr(decision_state, "regime_state"):
                        try:
                            regime_state = decision_state.regime_state.update(
                                candidate=regime_candidate,
                                atr_1t=float(features_raw.get("atr_1t", 0.0) or 0.0),
                            )
                        except Exception:
                            regime_state = {"regime": "SIDEWAYS", "regime_age": 0, "vol_band": "NORMAL"}

                    regime = str(regime_state.get("regime", "SIDEWAYS"))
                    regime_age = int(regime_state.get("regime_age", 0))
                    vol_band = str(regime_state.get("vol_band", "NORMAL"))

                    if dyn_thresholds is not None:
                        try:
                            prev_dyn = dict(dyn_thresholds.curr)
                        except Exception:
                            prev_dyn = {}
                        dyn_values = dyn_thresholds.update(regime=regime, vol_band=vol_band, regime_age=regime_age)
                        if isinstance(dyn_values, dict) and prev_dyn:
                            changed = []
                            for k, v in dyn_values.items():
                                try:
                                    if k in prev_dyn and abs(float(v) - float(prev_dyn.get(k))) > 1e-9:
                                        changed.append(f"{k}={prev_dyn.get(k):.4f}->{v:.4f}")
                                except Exception:
                                    continue
                            if changed:
                                logger.info("[%s] [DYN-THRESH] regime=%s age=%d vol=%s | %s", name, regime, regime_age, vol_band, ", ".join(changed))

                    rule_signals = {
                        "rule_sig": float(rule_sig),
                        "flow_side": float(features_raw.get("flow_side", 0.0)),
                        "flow_score": float(features_raw.get("flow_score", 0.0)),
                        "vwap_side": float(features_raw.get("flow_vwap_side", 0.0)),
                        "struct_side": float(struct_side),
                        "struct_score": float(struct_score),
                        "trend_side": float(features_raw.get("ema_bias_5t", 0.0)),
                        "mtf_consensus": float(mtf_cons),
                        "indicator_score": float(ind_score),
                        "pattern_adj": float(pat_adj),
                        "ta_rule": float(ta_rule),
                        "regime": str(regime),
                        "regime_age": int(regime_age),
                        "base_regime": str(base_regime),
                        "vol_band": str(vol_band),
                        "reversal_risk": bool(reversal_risk),
                        "trap_flag": bool(trap_flag),
                    }

                    # Hierarchy-based rule direction (flow/HTF > EMA trend > structure)
                    is_bull_setup = bool(features_raw.get("is_bull_setup", 0))
                    is_bear_setup = bool(features_raw.get("is_bear_setup", 0))
                    ambiguous_setup = is_bull_setup and is_bear_setup
                    any_setup = is_bull_setup or is_bear_setup

                    rule_dir, base_dir, tape_conflict_level, tape_conflict_reasons = compute_rule_hierarchy(
                        name=name,
                        rule_sig=rule_sig,
                        features_raw=features_raw,
                        mtf=mtf,
                        is_bull_setup=is_bull_setup,
                        is_bear_setup=is_bear_setup,
                        any_setup=any_setup,
                        ambiguous_setup=ambiguous_setup,
                        dynamic_thresholds=dyn_values if isinstance(dyn_values, dict) else None,
                        regime=regime,
                        reversal_risk=bool(reversal_risk),
                    )
                    rule_signals["flow_side"] = float(features_raw.get("flow_side", 0.0))
                    rule_signals["flow_score"] = float(features_raw.get("flow_score", 0.0))
                    rule_signals["vwap_side"] = float(features_raw.get("flow_vwap_side", 0.0))
                    if not tape_valid:
                        tape_conflict_level = "invalid"
                        tape_conflict_reasons = list(tape_conflict_reasons or []) + ["tape_invalid_data"]
                    teacher_strength = float(abs(rule_sig)) if np.isfinite(rule_sig) else 0.0
                    decision = decide_trade(
                        state=decision_state,
                        cfg=cfg,
                        features_raw=features_raw,
                        mtf=mtf,
                        rule_dir=rule_dir,
                        conflict_level=tape_conflict_level,
                        conflict_reasons=tape_conflict_reasons,
                        tape_valid=bool(tape_valid),
                        teacher_strength=teacher_strength,
                        is_bull_setup=bool(is_bull_setup),
                        is_bear_setup=bool(is_bear_setup),
                        safe_df=safe_df,
                        dynamic_thresholds=dyn_values if isinstance(dyn_values, dict) else None,
                        regime=regime,
                        reversal_risk=bool(reversal_risk),
                    )

                    teacher_tradeable = bool(decision.get('tradeable', False))
                    tradeable_flag = bool(teacher_tradeable)
                    gate_reasons = list(decision.get('gate_reasons', []))
                    gate_reasons_teacher = list(gate_reasons)
                    lane = str(decision.get('lane', 'NONE'))
                    size_mult = float(decision.get('size_mult', 0.0))
                    gates = {
                        "lane": lane,
                        "tape_conflict_level": str(decision.get('tape_conflict_level', tape_conflict_level)),
                        "tape_conflict_reasons": list(decision.get('tape_conflict_reasons', tape_conflict_reasons or [])),
                        "gate_reasons": list(gate_reasons_teacher),
                    }

                    # Policy success probability (BUY/SELL model) refines teacher decision
                    policy_min = float(os.getenv("POLICY_MIN_SUCCESS", "0.52") or "0.52")
                    policy_min = float(np.clip(policy_min, 0.0, 0.99))
                    try:
                        policy_override_ratio = float(os.getenv("OVERRIDE_MIN_SUCCESS_RATIO", "0.92") or "0.92")
                    except Exception:
                        policy_override_ratio = 0.92
                    try:
                        policy_max_micro = float(os.getenv("MAX_MICRO_SIZE", "0.25") or "0.25")
                    except Exception:
                        policy_max_micro = 0.25
                    p_success_raw = None
                    p_success_cal = None
                    p_move = None
                    p_edge = None
                    policy_features = None
                    if isinstance(model_pipe, PolicyPipeline) and rule_dir in ("BUY", "SELL"):
                        policy_features = compose_policy_features(
                            features=features,
                            rule_signals=rule_signals,
                            gates=gates,
                            teacher_strength=teacher_strength,
                        )
                        feat_names = list(policy_features.keys())
                        feat_vals = [policy_features[k] for k in feat_names]
                        p_success_raw, p_success_cal = model_pipe.predict_success(
                            feature_names=feat_names,
                            feature_values=feat_vals,
                            teacher_dir=rule_dir,
                        )
                    use_move_head = _safe_getenv_bool("USE_MOVE_HEAD", default=False)
                    if use_move_head:
                        mode = str(os.getenv("MOVE_HEAD_MODE", "proxy") or "proxy").lower().strip()
                        fallback_proxy = _safe_getenv_bool("MOVE_HEAD_FALLBACK_PROXY", default=True)
                        if mode == "model":
                            if policy_features is None:
                                policy_features = compose_policy_features(
                                    features=features,
                                    rule_signals=rule_signals,
                                    gates=gates,
                                    teacher_strength=teacher_strength,
                                )
                            feat_names = list(policy_features.keys())
                            feat_vals = [policy_features[k] for k in feat_names]
                            try:
                                p_move = model_pipe.predict_move(
                                    feature_names=feat_names,
                                    feature_values=feat_vals,
                                )
                            except Exception:
                                p_move = None
                            if p_move is None and fallback_proxy:
                                mode = "proxy"
                        if mode == "proxy":
                            try:
                                rv10 = float(features_raw.get("rv_10", 0.0) or 0.0)
                                atr1 = float(features_raw.get("atr_1t", 0.0) or 0.0)
                            except Exception:
                                rv10 = 0.0
                                atr1 = 0.0
                            try:
                                move_k = float(os.getenv("MOVE_PROXY_ATR_MULT", "0.35") or "0.35")
                            except Exception:
                                move_k = 0.35
                            if atr1 > 0 and move_k > 0:
                                p_move = min(1.0, abs(rv10) / (move_k * atr1))

                    policy_authorized = False
                    override_reason = "NONE"
                    if teacher_tradeable:
                        if p_success_cal is None:
                            tradeable_flag = False
                            gate_reasons.append("policy_no_score")
                        else:
                            if float(p_success_cal) >= policy_min:
                                if use_move_head and p_move is not None:
                                    try:
                                        edge_min = float(os.getenv("MOVE_EDGE_MIN", "0.30") or "0.30")
                                    except Exception:
                                        edge_min = 0.30
                                    p_edge = float(p_success_cal) * float(p_move)
                                    move_edge_trend_only = _safe_getenv_bool("MOVE_EDGE_TREND_ONLY", default=True)
                                    apply_move_gate = True
                                    if move_edge_trend_only and lane not in ("TREND", "BREAKOUT"):
                                        apply_move_gate = False
                                    if apply_move_gate and p_edge < edge_min:
                                        tradeable_flag = False
                                        gate_reasons.append("move_edge_veto")
                                        policy_authorized = False
                                        override_reason = "move_edge_veto"
                                    else:
                                        denom = max(1e-6, 1.0 - policy_min)
                                        scale = float(np.clip((float(p_success_cal) - policy_min) / denom, 0.0, 1.0))
                                        size_mult *= scale
                                        policy_authorized = True
                                        tradeable_flag = True
                                else:
                                    denom = max(1e-6, 1.0 - policy_min)
                                    scale = float(np.clip((float(p_success_cal) - policy_min) / denom, 0.0, 1.0))
                                    size_mult *= scale
                                    policy_authorized = True
                                    tradeable_flag = True
                            else:
                                override_min = float(policy_min) * float(policy_override_ratio)
                                flow_side = int(round(float(rule_signals.get("flow_side", 0.0))))
                                try:
                                    mtf_cons = float(rule_signals.get("mtf_consensus", 0.0))
                                except Exception:
                                    mtf_cons = 0.0
                                try:
                                    htf_neutral = float(os.getenv("HTF_NEUTRAL_MIN", "0.35") or "0.35")
                                    htf_strong_veto = float(os.getenv("HTF_STRONG_VETO_MIN", "0.70") or "0.70")
                                except Exception:
                                    htf_neutral = 0.35
                                    htf_strong_veto = 0.60
                                side = 1 if rule_dir == "BUY" else -1
                                flow_align = (flow_side == side)
                                htf_align = (mtf_cons >= htf_neutral and side == 1) or (mtf_cons <= -htf_neutral and side == -1)
                                htf_veto = (mtf_cons >= htf_strong_veto and side == -1) or (mtf_cons <= -htf_strong_veto and side == 1)
                                hard_veto = list(decision.get("hard_veto", []))
                                override_ok = bool(flow_align and htf_align and (not htf_veto) and (len(hard_veto) == 0) and (float(p_success_cal) >= override_min))
                                if override_ok:
                                    policy_authorized = True
                                    tradeable_flag = True
                                    size_mult = min(float(size_mult), float(policy_max_micro))
                                    override_reason = "policy_micro_override"
                                    gate_reasons.append("policy_micro_override")
                                else:
                                    tradeable_flag = False
                                    gate_reasons.append("policy_veto")
                    else:
                        tradeable_flag = False
                        gate_reasons.append("teacher_ineligible")
                    # Structured decision log (machine-readable)
                    logger.info(
                        "[%s] [DECISION] lane=%s intent=%s tradeable=%s teacher_eligible=%s policy_auth=%s override=%s "
                        "size=%.2f score=%.3f edge_req=%.3f edge_q=%.3f str_center=%.3f str_raw=%.3f "
                        "trend_sig=%d trend_str=%.2f trigger=%s(%s) tape=%s conflict=%s would_trade_wo_soft=%s "
                        "regime=%s reg_age=%d vol=%s rev_risk=%s "
                        "hard=%s soft=%s",
                        name,
                        lane,
                        str(decision.get('intent', 'NA')),
                        bool(tradeable_flag),
                        bool(teacher_tradeable),
                        bool(policy_authorized),
                        str(override_reason),
                        float(size_mult),
                        float(decision.get('score', 0.0)),
                        float(decision.get('edge_required', 0.0)),
                        float(decision.get('edge_quantile', 0.0)),
                        float(decision.get('centered_strength', 0.0)),
                        float(decision.get('raw_strength', 0.0)),
                        int(decision.get('trend_signals', 0)),
                        float(decision.get('trend_strength', 0.0)),
                        bool(decision.get('trigger', False)),
                        str(decision.get('trigger_name', 'none')),
                        str(decision.get('tape_conflict_level', 'none')),
                        str(decision.get('conflict', 'na')),
                        bool(decision.get('would_trade_without_soft', False)),
                        str(regime),
                        int(regime_age),
                        str(vol_band),
                        bool(reversal_risk),
                        list(decision.get('hard_veto', [])),
                        list((decision.get('soft_penalties', {}) or {}).keys()),
                    )

                    try:
                        decision_state.maybe_log_summary(current_bucket_start)
                    except Exception:
                        pass

                    # --- Setup state and optional delayed entry ---
                    setup_state = "NONE"
                    if lane == "SETUP":
                        setup_state = "READY" if tradeable_flag else "BUILDING"
                    elif is_any_setup and lane != "SETUP":
                        # structure present but we are using non-setup logic (or blocked)
                        setup_state = "BUILDING" if not tradeable_flag else "READY"
                    elif lane == "TREND":
                        setup_state = "BYPASS"

                    entry_delay_min = int(os.getenv("SETUP_ENTRY_DELAY_MIN", "0") or "0")
                    if not is_any_setup:
                        setup_ready_at = None
                    else:
                        if setup_state == "READY" and setup_ready_at is None:
                            setup_ready_at = current_bucket_start
                        if setup_ready_at is not None and entry_delay_min > 0:
                            ready_after = setup_ready_at + timedelta(minutes=entry_delay_min)
                            if current_bucket_start < ready_after:
                                tradeable_flag = False
                                gate_reasons.append("entry_delay")
                            elif current_bucket_start >= ready_after:
                                setup_ready_at = None  # consumed delay window

                    logger.info(
                        f"[{name}] [SETUP] state={setup_state} "
                        f"teacher_dir={rule_dir or 'NA'} strength={teacher_strength:.3f} "
                        f"is_bull_setup={is_bull_setup} is_bear_setup={is_bear_setup} "
                        f"tradeable={tradeable_flag} teacher_eligible={teacher_tradeable} policy_auth={policy_authorized}"
                    )

                    # --- Final direction: teacher defines direction; policy only filters entries ---
                    intent = str(decision.get('intent', 'FLAT')) if isinstance(decision, dict) else 'FLAT'
                    dir_overall = intent if intent in ("BUY", "SELL") else "FLAT"
                    if dir_overall in ("NEUTRAL", "FLAT"):
                        tradeable_flag = False
                        gate_reasons.append("teacher_flat")

                    # Signal record: prediction for close at t+2*interval
                    suggest_tradeable = bool(tradeable_flag)
                    sig = {
                        "pred_for": target_start.isoformat(), # where target_start = t + 2*interval
                        "decision": "USER",
                        "teacher_dir": str(rule_dir),
                        "teacher_tradeable": bool(teacher_tradeable),
                        "exec_tradeable": bool(tradeable_flag),
                        "policy_authorized": bool(policy_authorized),
                        "override_reason": str(override_reason),
                        "teacher_strength": float(teacher_strength),
                        "policy_success_raw": float(p_success_raw) if p_success_raw is not None else None,
                        "policy_success_calib": float(p_success_cal) if p_success_cal is not None else None,
                        "policy_move_p": float(p_move) if p_move is not None else None,
                        "policy_edge_p": float(p_edge) if p_edge is not None else None,
                        "policy_min_success": float(policy_min),
                        "tape_valid": bool(tape_valid),
                        "ema_decision_tf": (ema_meta or {}).get("decision_tf"),
                        "ema_filter_tf": (ema_meta or {}).get("filter_tf"),
                        "ema_regime": (ema_meta or {}).get("regime"),
                        "ema_bias": (ema_meta or {}).get("bias"),
                        "ema_entry_tag": (ema_meta or {}).get("entry_tag"),
                        "ema_entry_side": (ema_meta or {}).get("entry_side"),
                        "ema15_break": (ema_meta or {}).get("ema15_break"),
                        "mtf_consensus": float(mtf.get("mtf_consensus", 0.0)) if mtf else 0.0,
                        "indicator_score": float(indicator_score) if indicator_score is not None else None,
                        "pattern_adj": float(pattern_features.get('probability_adjustment', 0.0)) if pattern_features else 0.0,
                        "regime": str(regime),
                        "regime_age": int(regime_age),
                        "base_regime": str(base_regime),
                        "vol_band": str(vol_band),
                        "reversal_risk": bool(reversal_risk),
                        "trap_flag": bool(trap_flag),
                        "suggest_tradeable": suggest_tradeable,
                        "rule_signal": float(rule_sig) if rule_sig is not None else None,
                        "rule_direction": rule_dir,
                        "direction": dir_overall,
                        "tradeable": tradeable_flag,
                        "lane": lane,
                        "size_mult": float(size_mult),
                        "edge_required": float(decision.get('edge_required', 0.0)) if isinstance(decision, dict) else None,
                        "score": float(decision.get('score', 0.0)) if isinstance(decision, dict) else None,
                        "gate_reasons": list(gate_reasons),
                        "is_bull_setup": bool(is_bull_setup),
                        "is_bear_setup": bool(is_bear_setup),
                        "is_any_setup": bool(is_any_setup),
                        "setup_state": setup_state,
                        "setup_ready_at": setup_ready_at.isoformat() if setup_ready_at else None,
                    }

                    spath = getattr(cfg, "signals_path", "trained_models/production/signals.jsonl")
                    Path(Path(spath).parent).mkdir(parents=True, exist_ok=True)
                    with open(spath, "a", encoding="utf-8") as f:
                        f.write(json.dumps(sig) + "\n")

                    p_success = p_success_cal if p_success_cal is not None else p_success_raw
                    if p_success is None:
                        conf_bucket = "NA"
                    elif p_success >= 0.70:
                        conf_bucket = "HIGH"
                    elif p_success >= 0.60:
                        conf_bucket = "MED"
                    else:
                        conf_bucket = "LOW"

                    rule_dir_display = rule_dir or "n/a"
                    if tradeable_flag and not gate_reasons:
                        gate_reasons.append('pass')
                    gate_reason_str = ",".join(gate_reasons) if gate_reasons else "n/a"
                    p_success_txt = f"{p_success*100:.1f}%" if p_success is not None else "NA"

                    interval_min = max(
                        1,
                        int(getattr(cfg, "candle_interval_seconds", 60) // 60),
                    )

                    scalper_line = _scalper_playbook_line(
                        features_raw=features_raw,
                        dir_overall=dir_overall,
                        tradeable=tradeable_flag,
                        gate_reasons=gate_reasons,
                        rule_dir=rule_dir_display,
                        teacher_strength=teacher_strength,
                        p_success=p_success,
                        conf_bucket=conf_bucket,
                        struct_setup_side=(
                            1 if (is_bull_setup and not is_bear_setup)
                            else (-1 if (is_bear_setup and not is_bull_setup) else 0)
                        ),
                    )
                    logger.info(f"[{name}] {interval_min}-min {scalper_line}")
                    prev_intent = getattr(decision_state, "_last_intent", None)
                    prev_tradeable = bool(getattr(decision_state, "_last_tradeable", False))
                    action_line = _scalper_action_line(
                        intent=dir_overall,
                        tradeable=tradeable_flag,
                        policy_authorized=policy_authorized,
                        reversal_risk=bool(reversal_risk),
                        prev_intent=prev_intent,
                        prev_tradeable=prev_tradeable,
                        gate_reasons=gate_reasons,
                    )
                    logger.info(f"[{name}] {interval_min}-min {action_line}")
                    if tradeable_flag and policy_authorized and dir_overall in ("BUY", "SELL"):
                        logger.info(
                            f"[{name}] {interval_min}-min [SCALPER ALERT] ENTER_{dir_overall} NOW | "
                            f"p_success={p_success_txt} | strength={teacher_strength:.3f} | gate={gate_reason_str}"
                        )
                    decision_state._last_intent = dir_overall
                    decision_state._last_tradeable = bool(tradeable_flag)


                    logger.info(
                        f"[{name}] {interval_min}-min view: "
                        f"p_success={p_success_txt} | "
                        f"dir={dir_overall} (teacher_dir={rule_dir_display}) | strength={teacher_strength:.3f} ({conf_bucket}) | "
                        f"tradeable={tradeable_flag} "
                        f"| gate={gate_reason_str}"
                    )
                    logger.info(
                        f"[{name}] [SIGNAL] {horizon_min}m horizon: start={current_bucket_start.strftime('%H:%M:%S')} "
                        f"target={target_start.strftime('%H:%M:%S')} "
                        f"(suggest_tradeable={suggest_tradeable}) → {spath}"
                    )

                    # Stage features for training when t+2 candle closes
                    features_for_log = dict(features)
                    # FIX #1: Cleanup old staged entries (memory leak prevention)
                    if len(staged_map) > 100:
                        cutoff = datetime.now(IST) - timedelta(minutes=60)
                        stale_keys = [k for k in staged_map.keys() if k < cutoff]
                        for k in stale_keys:
                            staged_map.pop(k, None)
                        if stale_keys:
                            logger.debug(f"[{name}] Cleaned {len(stale_keys)} stale staged entries")

                    staged_map[ref_start] = {
                        "features": features_for_log,
                        "teacher_dir": str(rule_dir),
                        "teacher_tradeable": bool(teacher_tradeable),
                        "teacher_strength": float(teacher_strength),
                        "rule_signals": dict(rule_signals) if isinstance(rule_signals, dict) else {},
                        "gates": {
                            "lane": lane,
                            "tape_conflict_level": str(tape_conflict_level),
                            "tape_conflict_reasons": list(tape_conflict_reasons or []),
                            "gate_reasons": list(gate_reasons),
                        },
                        "tradeable": bool(tradeable_flag),
                        "lane": lane,
                        "size_mult": float(size_mult),
                        "edge_required": float(decision.get('edge_required', 0.0)) if isinstance(decision, dict) else None,
                        "score": float(decision.get('score', 0.0)) if isinstance(decision, dict) else None,
                        "policy_success_raw": float(p_success_raw) if p_success_raw is not None else None,
                        "policy_success_calib": float(p_success_cal) if p_success_cal is not None else None,
                    }

                except Exception as e:
                    logger.error(f"[{name}] on_preclose error: {e}", exc_info=True)

            # ---------- CANDLE CLOSE: 2-MIN LABEL (close_{t+2} vs close_t) ----------
            async def _on_candle_cb(candle_df: pd.DataFrame, full_df: pd.DataFrame):
                try:
                    if not isinstance(candle_df, pd.DataFrame) or candle_df.empty:
                        return

                    idx_ts = candle_df.index[-1]
                    row_t2 = candle_df.iloc[-1]

                    logger.info(
                        f"[{name}] Candle closed at {idx_ts.strftime('%H:%M:%S')} | "
                        f"O:{float(row_t2.get('open', 0.0)):.2f} "
                        f"H:{float(row_t2.get('high', 0.0)):.2f} "
                        f"L:{float(row_t2.get('low', 0.0)):.2f} "
                        f"C:{float(row_t2.get('close', 0.0)):.2f} "
                        f"Vol:{int(row_t2.get('volume', 0)):,} "
                        f"Ticks:{int(row_t2.get('tick_count', 0))}"
                    )

                    # Trade-window label: horizon_min (minutes) -> bars using candle interval
                    horizon_min = int(getattr(cfg, "trade_horizon_min",
                                              int(os.getenv("TRADE_HORIZON_MIN", "10") or "10")))
                    horizon_min = max(1, horizon_min)

                    interval_sec = int(getattr(cfg, "candle_interval_seconds", 60))
                    candle_interval_min = max(1.0, interval_sec / 60.0)
                    horizon_bars = max(1, int(round(horizon_min / candle_interval_min)))

                    # Align reference timestamp to candle grid using horizon_bars
                    ref_minutes = int(round(horizon_bars * candle_interval_min))
                    ref_ts = idx_ts - timedelta(minutes=ref_minutes)

                    if not isinstance(full_df, pd.DataFrame) or full_df.empty:
                        logger.info(
                            f"[{name}] [TRAIN] Skipping trade-window label for {idx_ts.strftime('%H:%M:%S')} "
                            "(full_df empty)"
                        )
                        return

                    idx_ref = None

                    # Prefer stepping back from the actual idx_ts position to stay on-grid.
                    try:
                        idx_ts_loc = full_df.index.get_loc(idx_ts)
                        if isinstance(idx_ts_loc, np.ndarray):
                            idx_ts_loc = int(idx_ts_loc[0])
                        idx_ts_loc = int(idx_ts_loc)
                        idx_ref = idx_ts_loc - horizon_bars
                    except Exception:
                        idx_ref = None

                    # Fallback: try locating ref_ts directly (may fail if timestamps are missing)
                    if idx_ref is None:
                        try:
                            idx_ref = full_df.index.get_loc(ref_ts)
                            if isinstance(idx_ref, np.ndarray):
                                idx_ref = int(idx_ref[0])
                            idx_ref = int(idx_ref)
                        except Exception:
                            idx_ref = len(full_df) - horizon_bars - 1

                    if idx_ref is None or idx_ref < 0:
                        logger.info(
                            f"[{name}] [TRAIN] Skipping trade-window label for {idx_ts.strftime('%H:%M:%S')} "
                            f"(cannot resolve ref index {ref_ts})"
                        )
                        return

                    ref_start = full_df.index[idx_ref]

                    # Retrieve staged features for the reference time t (for vol + training row)
                    staged = staged_map.pop(ref_start, None)

                    if not staged:
                        logger.info(
                            f"[{name}] [TRAIN] Skipping train record for {idx_ts.strftime('%H:%M:%S')} "
                            f"(missing staged context @ {ref_start})"
                        )
                        return

                    teacher_dir = str((staged or {}).get("teacher_dir", "FLAT"))
                    teacher_tradeable = bool((staged or {}).get("teacher_tradeable", False))
                    teacher_strength = float((staged or {}).get("teacher_strength", 0.0))
                    features_for_log = dict((staged or {}).get("features", {}))
                    rule_signals = dict((staged or {}).get("rule_signals", {}))
                    gates = dict((staged or {}).get("gates", {}))
                    policy_success_raw = (staged or {}).get("policy_success_raw")
                    policy_success_calib = (staged or {}).get("policy_success_calib")

                    # Vol proxies from features (optional)
                    rv10 = features_for_log.get("rv_10")
                    atr1 = features_for_log.get("atr_1t")

                    # Setup-conditional label:
                    #   - only label when clear one-sided structure + strong imbalance
                    #   - inside that subset, direction is sign-of-return with eps band
                    
                    label_source = "setup"
                    label, aux_info = compute_setup_conditional_label(
                        df=full_df,
                        idx_ref=idx_ref,
                        idx_ts=idx_ts,
                        horizon_bars=horizon_bars,
                        features_for_log=features_for_log,
                        name=name,
                    )

                    # Fallback: if no clear setup, still label the price path,
                    # but with lower weight so we don't drown the model in chop.
                    if label in (None, LABEL_SKIP):
                        label_source = "window"
                        try:
                            tp_ret = float(getattr(cfg, "trade_tp_pct", float(os.getenv("TRADE_TP_PCT", "0.0005") or "0.0005")))
                        except Exception:
                            tp_ret = 0.0005
                        try:
                            sl_ret = float(getattr(cfg, "trade_sl_pct", float(os.getenv("TRADE_SL_PCT", "0.0005") or "0.0005")))
                        except Exception:
                            sl_ret = 0.0005
                        try:
                            fb_atr = float(os.getenv("TRAIN_FALLBACK_ATR_MULT", "0.14") or "0.14")
                        except Exception:
                            fb_atr = 0.14

                        label = compute_trade_window_label(
                            df=full_df,
                            idx=idx_ref,
                            horizon=horizon_bars,
                            tp_ret=tp_ret,
                            sl_ret=sl_ret,
                            fallback_atr_mult=fb_atr,
                        )

                    if label in (None, LABEL_SKIP):
                        logger.info(
                            f"[{name}] [TRAIN] Label skipped for {idx_ts.strftime('%H:%M:%S')} (no setup + cannot label window)"
                        )
                        return

                        logger.info(
                            f"[{name}] [TRAIN] Using fallback window label for {idx_ts.strftime('%H:%M:%S')} "
                            f"(ref={pd.to_datetime(ref_start).strftime('%H:%M:%S')}) label={label}"
                        )

                    # High-visibility label log (setup-conditional context)
                    def _fmt_float(x):
                        try:
                            x = float(x)
                            return f"{x:.6f}" if np.isfinite(x) else "nan"
                        except Exception:
                            return "nan"

                    logger.info(
                        "[LABEL] [SETUP] horizon=%dm (~%d x %.0fm bars) @ %s → label=%s "
                        "rv_10=%s atr_1t=%s teacher_tradeable=%s p_success=%s",
                        horizon_min,
                        horizon_bars,
                        candle_interval_min,
                        idx_ts.strftime("%H:%M:%S"),
                        label,
                        _fmt_float(rv10),
                        _fmt_float(atr1),
                        teacher_tradeable,
                        f"{policy_success_calib:.4f}" if policy_success_calib is not None else "NA",
                    )

                    
                    # Sample weight: teach the model harder on clean setups, softer on generic window labels.
                    if label_source == "setup":
                        label_weight = 1.0 if label in ("BUY", "SELL") else 0.40
                        label_from_clear_setup = 1.0
                    else:
                        label_weight = 0.35 if label in ("BUY", "SELL") else 0.15
                        label_from_clear_setup = 0.0

                    # Write training record (SignalContext) with validation.
                    if not train_log_path:
                        logger.error("[%s] [TRAIN] TRAIN_LOG_PATH missing; record not written", name)
                        return

                    schema_cols: List[str] = []
                    try:
                        sp = os.getenv("FEATURE_SCHEMA_COLS_PATH", "").strip()
                        if sp:
                            obj = json.loads(open(sp, "r", encoding="utf-8").read())
                            schema_cols = obj["columns"] if isinstance(obj, dict) else list(obj)
                    except Exception:
                        schema_cols = []

                    if not schema_cols:
                        logger.error("[%s] [TRAIN] Missing schema cols; record skipped", name)
                        return

                    aux_info = aux_info or {}
                    provenance = {
                        "record_source": "online",
                        "scored": policy_success_raw is not None,
                        "aux_ret_main": aux_info.get("aux_ret_main"),
                        "aux_ret_short": aux_info.get("aux_ret_short"),
                        "aux_label_short": aux_info.get("aux_label_short"),
                        "trap_long": aux_info.get("trap_long"),
                        "trap_short": aux_info.get("trap_short"),
                        "label_from_clear_setup": float(label_from_clear_setup),
                        "label_source_flag": 1.0 if label_source == "setup" else 0.0,
                        "tick_count": int(row_t2.get("tick_count", 0)),
                    }

                    schema_version = os.getenv("SCHEMA_VERSION", "schema_v4")
                    label_version = os.getenv("LABEL_VERSION", "label_v4")
                    pipeline_version = os.getenv("PIPELINE_VERSION", "pipeline_v4")

                    bar_min = int(max(1, round(candle_interval_min)))

                    record, errors = build_signal_context(
                        schema_cols=schema_cols,
                        schema_version=schema_version,
                        label_version=label_version,
                        pipeline_version=pipeline_version,
                        symbol=str(name),
                        bar_min=bar_min,
                        horizon_min=horizon_min,
                        ts_ref_start=ref_start,
                        ts_target_close=idx_ts,
                        label=str(label),
                        label_source=str(label_source),
                        label_weight=float(label_weight),
                        features=features_for_log,
                        rule_signals=rule_signals,
                        gates=gates,
                        teacher_dir=teacher_dir,
                        teacher_tradeable=teacher_tradeable,
                        teacher_strength=teacher_strength,
                        provenance=provenance,
                        model={
                            "p_success_raw": policy_success_raw,
                            "p_success_calib": policy_success_calib,
                        },
                    )

                    if errors or record is None:
                        qrec = {
                            "record_version": "sc_v1",
                            "errors": errors,
                            "record": {
                                "symbol": str(name),
                                "ts_target_close": str(idx_ts),
                                "label": str(label),
                                "features": features_for_log,
                                "provenance": provenance,
                            },
                        }
                        append_jsonl(train_log_path.replace(".jsonl", "_quarantine.jsonl"), qrec)
                        logger.warning("[%s] [TRAIN] Record quarantined: %s", name, ",".join(errors))
                        return

                    if validate_or_quarantine(record, schema_cols=schema_cols, train_log_path=train_log_path):
                        append_jsonl(train_log_path, record)
                    else:
                        logger.warning("[%s] [TRAIN] Record failed validation; quarantined", name)

                    logger.info(
                        f"[{name}] [TRAIN] Logged features for ref={pd.to_datetime(ref_start).strftime('%H:%M:%S')} "
                        f"-> tgt={idx_ts.strftime('%H:%M:%S')} label={label} teacher_tradeable={teacher_tradeable}"
                    )

                except Exception as e:
                    logger.error(f"[{name}] on_candle callback error: {e}", exc_info=True)

            # Wire callbacks
            ws_handler.on_tick = _on_tick_cb
            ws_handler.on_preclose = _on_preclose_cb
            ws_handler.on_candle = _on_candle_cb

            # Start connector and heartbeat
            stop_event = asyncio.Event()
            stop_events[name] = stop_event
            ws_task = asyncio.create_task(_ws_connect_and_stream(name, cfg, ws_handler, stop_event))
            hb_task = asyncio.create_task(_websocket_heartbeat(name, ws_handler, interval_sec=30))
            ws_tasks.append((name, ws_handler, ws_task))
            hb_tasks.append((name, hb_task))
            logger.info(f"[{name}] Connector and heartbeat started")

        except Exception as e:
            logger.error(f"[{name}] Failed to initialize connection: {e}", exc_info=True)

    if not ws_tasks:
        logger.critical("No connections could be initialized. Exiting.")
        try:
            if trainer_task:
                trainer_task.cancel()
                await asyncio.wait_for(trainer_task, timeout=5.0)
        except Exception:
            pass
        return

    # Keep main alive until session guard requests stop
    stop_main_event = asyncio.Event()

    async def _session_guard():
        try:
            def _today_ist(h, m):
                now = datetime.now(IST)
                return now.replace(hour=h, minute=m, second=0, microsecond=0)

            roll_ts = _today_ist(15, 10)
            exit_ts = _today_ist(15, 15)

            now = datetime.now(IST)
            if now > exit_ts:
                roll_ts = roll_ts + timedelta(days=1)
                exit_ts = exit_ts + timedelta(days=1)

            await asyncio.sleep(max(0.0, (roll_ts - datetime.now(IST)).total_seconds()))
            try:
                if not _safe_getenv_bool("ENABLE_EOD_ROLL", default=False):
                    logger.info("[EOD] Feature-log roll disabled (set ENABLE_EOD_ROLL=1 to enable).")
                else:
                    hist_path = os.getenv("FEATURE_LOG_HIST", "trained_models/production/feature_log_hist.csv")
                    _roll_feature_logs(
                        daily_path=getattr(config, 'feature_log_path', 'feature_log.csv'),
                        hist_path=hist_path,
                        cap_rows=2000
                    )
                    logger.info("[EOD] Rolled daily feature log into historical and capped to 2000 rows")
            except Exception as e:
                logger.warning(f"[EOD] Roll failed: {e}")

            await asyncio.sleep(max(0.0, (exit_ts - datetime.now(IST)).total_seconds()))
            logger.info("[EOD] Session end reached (15:15 IST). Requesting shutdown.")
            stop_main_event.set()
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.warning(f"[EOD] Session guard error: {e}")

    def _roll_feature_logs(daily_path: str, hist_path: str, cap_rows: int = 2000) -> None:
        try:
            import pandas as pd
            dfs = []
            if os.path.exists(hist_path) and os.path.getsize(hist_path) > 0:
                dfs.append(pd.read_csv(hist_path, engine="python", on_bad_lines="skip"))
            if os.path.exists(daily_path) and os.path.getsize(daily_path) > 0:
                dfs.append(pd.read_csv(daily_path, engine="python", on_bad_lines="skip"))

            if not dfs:
                return

            df = pd.concat(dfs, ignore_index=True)
            if "ts" in df.columns:
                df = df.drop_duplicates(subset=["ts"], keep="last")
            df = df.tail(int(cap_rows))

            os.makedirs(os.path.dirname(hist_path) or ".", exist_ok=True)
            df.to_csv(hist_path, index=False)

            # truncate daily
            try:
                open(daily_path, "w", encoding="utf-8").write("")
            except Exception:
                pass
        except Exception as e:
            logger.warning(f"[EOD] Failed rolling feature logs: {e}")

    session_task = asyncio.create_task(_session_guard())

    try:
        while not stop_main_event.is_set():
            await asyncio.sleep(5)
    except asyncio.CancelledError:
        logger.info("Main event loop cancelled")
    except Exception as e:
        logger.error(f"Fatal error in main_loop: {e}", exc_info=True)
    finally:
        logger.info("Shutting down main loop")

        for name, _handler, _task in ws_tasks:
            try:
                stop_events[name].set()
            except Exception:
                pass

        for name, _handler, ws_task in ws_tasks:
            ws_task.cancel()
            with suppress(asyncio.CancelledError, asyncio.TimeoutError):
                await asyncio.wait_for(ws_task, timeout=5.0)

        for name, hb_task in hb_tasks:
            hb_task.cancel()
            with suppress(asyncio.CancelledError, asyncio.TimeoutError):
                await asyncio.wait_for(hb_task, timeout=3.0)

        if trainer_task:
            trainer_task.cancel()
            with suppress(asyncio.CancelledError, asyncio.TimeoutError):
                await asyncio.wait_for(trainer_task, timeout=5.0)

        if baseline_task:
            baseline_task.cancel()
            with suppress(asyncio.CancelledError, asyncio.TimeoutError):
                await asyncio.wait_for(baseline_task, timeout=5.0)

        if calib_task:
            calib_task.cancel()
            with suppress(asyncio.CancelledError, asyncio.TimeoutError):
                await asyncio.wait_for(calib_task, timeout=5.0)

        if session_task:
            session_task.cancel()
            with suppress(asyncio.CancelledError, asyncio.TimeoutError):
                await asyncio.wait_for(session_task, timeout=5.0)

        logger.info("=" * 60)
        logger.info("MAIN EVENT LOOP SHUTDOWN COMPLETE")
        logger.info("=" * 60)
