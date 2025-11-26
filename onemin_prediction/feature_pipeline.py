# feature_pipeline.py
"""
Unified feature engineering and drift detection (probabilities-only).
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging

try:
    from scipy.stats import ks_2samp  # type: ignore
except Exception:
    ks_2samp = None

logger = logging.getLogger(__name__)

def _safe_series(arr, min_len=1):
    try:
        s = pd.Series(arr, dtype="float64")
        if s.size < min_len:
            return pd.Series([0.0]*min_len, dtype="float64")
        return s
    except Exception:
        return pd.Series([0.0]*min_len, dtype="float64")

class TA:
    @staticmethod
    def rsi(prices: List[float], period: int = 14) -> float:
        try:
            s = _safe_series(prices, min_len=period+2)
            delta = s.diff()
            up = delta.clip(lower=0).ewm(alpha=1/period, adjust=False).mean()
            down = -delta.clip(upper=0).ewm(alpha=1/period, adjust=False).mean()
            rs = up / np.maximum(1e-12, down)
            rsi = 100 - (100 / (1 + rs.iloc[-1]))
            rsi = float(np.nan_to_num(rsi))
            return float(np.clip(rsi, 0.0, 100.0))
        except Exception:
            return 50.0

    @staticmethod
    def macd(prices: List[float], fast: int = 12, slow: int = 26, signal: int = 9):
        try:
            s = _safe_series(prices, min_len=slow+signal+2)
            ema_fast = s.ewm(span=min(fast, len(s)), adjust=False).mean()
            ema_slow = s.ewm(span=min(slow, len(s)), adjust=False).mean()
            macd_line = ema_fast - ema_slow
            signal_line = macd_line.ewm(span=min(signal, len(macd_line)), adjust=False).mean()
            hist = macd_line - signal_line
            return float(macd_line.iloc[-1]), float(signal_line.iloc[-1]), float(hist.iloc[-1])
        except Exception:
            return 0.0, 0.0, 0.0

    @staticmethod
    def bollinger(prices: List[float], period: int = 20, nbdev: float = 2.0):
        try:
            s = _safe_series(prices, min_len=period+1)
            ma = s.rolling(window=min(period, len(s))).mean()
            sd = s.rolling(window=min(period, len(s))).std(ddof=0)
            mid = ma.iloc[-1]
            upper = float(mid + nbdev * (sd.iloc[-1] if np.isfinite(sd.iloc[-1]) else 0.0))
            lower = float(mid - nbdev * (sd.iloc[-1] if np.isfinite(sd.iloc[-1]) else 0.0))
            px = float(s.iloc[-1])
            denom = max(1e-12, upper - lower)
            pctb = (px - lower) / denom
            bw = denom / max(1e-12, mid if np.isfinite(mid) and abs(mid) > 1e-12 else 1.0)
            return float(upper), float(mid), float(lower), float(np.clip(pctb, 0.0, 1.0)), float(np.nan_to_num(bw))
        except Exception:
            return 0.0, 0.0, 0.0, 0.5, 0.0

    @staticmethod
    def compute_ta_bundle(prices: List[float]) -> Dict[str, float]:
        rsi14 = TA.rsi(prices, period=14)
        macd, macd_sig, macd_hist = TA.macd(prices)
        bb_u, bb_m, bb_l, bb_pctb, bb_bw = TA.bollinger(prices)
        return {
            "ta_rsi14": float(rsi14),
            "ta_macd": float(macd),
            "ta_macd_signal": float(macd_sig),
            "ta_macd_hist": float(macd_hist),
            "ta_bb_upper": float(bb_u),
            "ta_bb_mid": float(bb_m),
            "ta_bb_lower": float(bb_l),
            "ta_bb_pctb": float(bb_pctb),
            "ta_bb_bw": float(bb_bw),
        }

class FeaturePipeline:
    def __init__(self, train_features: Dict):
        self.train_features = train_features
        logger.info("Feature pipeline initialized (probabilities-only)")

    @staticmethod
    def _to_pandas_freq(tf: str) -> str:
        try:
            s = str(tf).strip()
            if s.lower().endswith("t"):
                return s[:-1] + "min"
            return s
        except Exception:
            return str(tf)

    @staticmethod
    def compute_emas(prices: List[float], periods: Optional[List[int]] = None) -> Dict[str, float]:
        if periods is None:
            periods = [8, 21, 50]
        s = pd.Series(prices, dtype='float64') if prices else pd.Series([], dtype='float64')
        out = {}
        for p in periods:
            try:
                out[f'ema_{p}'] = float(s.ewm(span=min(p, max(1, len(s))), adjust=False).mean().iloc[-1]) if len(s) else 0.0
            except Exception:
                out[f'ema_{p}'] = 0.0
        logger.debug(f"EMAs computed (len={len(prices)}): {out}")
        return out

    @staticmethod
    def order_flow_dynamics(order_books: List[Dict], window: int = 5) -> Dict[str, float]:
        try:
            ob_df = pd.DataFrame(order_books[-window:])
            if ob_df.empty:
                return {'spread': 0.0}
            spread = (ob_df['ask_price'].astype(float).mean() -
                      ob_df['bid_price'].astype(float).mean()) if 'ask_price' in ob_df and 'bid_price' in ob_df else 0.0
            return {'spread': float(spread)}
        except Exception:
            return {'spread': 0.0}

    @staticmethod
    def _is_bounded_key(k: str) -> bool:
        if k.startswith("pat_is") or k.startswith("mtf_tf_"):
            return True
        bounded_keys = {
            "pat_rvol", "probability_adjustment", "mtf_adj", "mtf_consensus",
            "price_range_tightness", "ta_bb_pctb", "ta_rsi14",
        }
        bounded_keys.update({
            "fut_vwap_dev",
            "fut_cvd_delta",
            "fut_vol_delta",
            "rv_10",
            "atr_1t",
            "atr_3t",
            "tod_sin",
            "tod_cos",
        })
        return k in bounded_keys


    @staticmethod
    def compute_micro_trend(px_hist) -> dict:
        """
        Compute simple micro-trend features from a price history.

        Inputs:
            px_hist: list/array of recent closes (oldest -> newest)

        Returns dict with:
            - micro_slope: average per-bar price change (normalised)
            - micro_imbalance: up vs down bar imbalance in [-1, 1]
            - mean_drift_pct: total drift from first->last in percent
            - last_zscore: z-score of last price vs recent mean
        """
        try:
            arr = np.asarray(px_hist, dtype=float)
        except Exception:
            return {
                "micro_slope": 0.0,
                "micro_imbalance": 0.0,
                "mean_drift_pct": 0.0,
                "last_zscore": 0.0,
            }

        n = arr.size
        if n < 3 or not np.all(np.isfinite(arr)):
            return {
                "micro_slope": 0.0,
                "micro_imbalance": 0.0,
                "mean_drift_pct": 0.0,
                "last_zscore": 0.0,
            }

        # Basic differences
        diffs = np.diff(arr)
        diffs = diffs[np.isfinite(diffs)]
        if diffs.size == 0:
            micro_slope = 0.0
        else:
            # normalise by last price for scale invariance
            last_px = float(arr[-1])
            denom = max(abs(last_px), 1e-6)
            micro_slope = float(diffs[-min(5, diffs.size) :].mean() / denom)

        # Up/down bar imbalance
        up = float((diffs > 0).sum())
        down = float((diffs < 0).sum())
        total = up + down
        if total > 0:
            micro_imbalance = (up - down) / total  # in [-1, 1]
        else:
            micro_imbalance = 0.0

        # Overall drift from first -> last in pct
        first_px = float(arr[0])
        denom_first = max(abs(first_px), 1e-6)
        mean_drift_pct = float((last_px - first_px) / denom_first)

        # Last z-score vs recent window
        try:
            if n >= 32:
                window = arr[-32:]
            else:
                window = arr
            mu = float(window.mean())
            sigma = float(window.std())
            last_zscore = (last_px - mu) / max(sigma, 1e-6)
        except Exception:
            last_zscore = 0.0

        return {
            "micro_slope": micro_slope,
            "micro_imbalance": micro_imbalance,
            "mean_drift_pct": mean_drift_pct,
            "last_zscore": last_zscore,
        }


    @staticmethod
    def _compute_wick_extremes(last: pd.Series) -> tuple[float, float]:
        """
        Compute upper/lower wick ratios for the last candle.

        Returns (wick_extreme_up, wick_extreme_down) in [-1, 1].
        """
        try:
            o = float(last["open"])
            h = float(last["high"])
            l = float(last["low"])
            c = float(last["close"])
        except Exception:
            return 0.0, 0.0

        body = max(abs(c - o), 1e-6)
        upper_wick = max(h - max(o, c), 0.0) / body
        lower_wick = max(min(o, c) - l, 0.0) / body

        # Clip to a sane range
        upper_wick = float(max(-1.0, min(upper_wick, 5.0)))
        lower_wick = float(max(-1.0, min(lower_wick, 5.0)))

        # Normalise to roughly [-1, 1] with soft cap
        upper_norm = max(-1.0, min(upper_wick / 3.0, 1.0))
        lower_norm = max(-1.0, min(lower_wick / 3.0, 1.0))
        return upper_norm, lower_norm

    @staticmethod
    def _compute_vwap_reversion_flag(px_hist: pd.DataFrame, vwap: float | None) -> float:
        """
        Flag when price pierces VWAP and closes back inside (mean-reversion flavour).

        Returns value in [-1, 1]: +1 for bearish reversion from above,
        -1 for bullish reversion from below, 0 otherwise.
        """
        if vwap is None:
            return 0.0
        try:
            last = px_hist.iloc[-1]
            prev = px_hist.iloc[-2] if len(px_hist) >= 2 else last
            c_last = float(last["close"])
            c_prev = float(prev["close"])
        except Exception:
            return 0.0

        above_prev = c_prev > vwap
        below_prev = c_prev < vwap
        above_last = c_last > vwap
        below_last = c_last < vwap

        # From above back toward VWAP or below → bearish reversion
        if above_prev and (not above_last):
            return 1.0
        # From below back toward VWAP or above → bullish reversion
        if below_prev and (not below_last):
            return -1.0
        return 0.0

    @staticmethod
    def _compute_cvd_divergence(price_change: float, cvd_delta: float | None) -> float:
        """
        Simple divergence flag between price change and futures CVD.

        Returns in [-1, 1]: +1 when price up but CVD down (bearish divergence),
        -1 when price down but CVD up (bullish divergence), 0 otherwise.
        """
        if cvd_delta is None:
            return 0.0

        try:
            p_chg = float(price_change)
            c_chg = float(cvd_delta)
        except Exception:
            return 0.0

        if abs(p_chg) < 1e-6 or abs(c_chg) < 1e-9:
            return 0.0

        price_sign = 1.0 if p_chg > 0 else -1.0
        cvd_sign = 1.0 if c_chg > 0 else -1.0

        if price_sign == 1.0 and cvd_sign == -1.0:
            return 1.0
        if price_sign == -1.0 and cvd_sign == 1.0:
            return -1.0
        return 0.0

    @staticmethod
    def compute_reversal_cross_features(tail: pd.DataFrame, features: Dict[str, float]) -> Dict[str, float]:
        """
        Crossed reversal flags using wick extremes, VWAP reversion, and CVD divergence context.
        """
        out: Dict[str, float] = {}
        try:
            wick_up = float(features.get("wick_extreme_up", 0.0) or 0.0)
            wick_dn = float(features.get("wick_extreme_down", 0.0) or 0.0)
            cvd_div = float(features.get("cvd_divergence", 0.0) or 0.0)
            vwap_rev = float(features.get("vwap_reversion_flag", 0.0) or 0.0)

            closes = (
                tail["close"].astype(float)
                if isinstance(tail, pd.DataFrame) and (not tail.empty) and ("close" in tail.columns)
                else pd.Series(dtype=float)
            )

            if len(closes) >= 10:
                recent = closes.tail(10)
                last_close = float(recent.iloc[-1])
                hi_10 = float(recent.max())
                lo_10 = float(recent.min())
                span = max(1.0, hi_10 - lo_10)
                near_high = 1.0 if (hi_10 - last_close) / span <= 0.25 else 0.0
                near_low = 1.0 if (last_close - lo_10) / span <= 0.25 else 0.0
            else:
                near_high = near_low = 0.0

            out["rev_cross_upper_wick_cvd"] = float(
                wick_up >= 0.7 and cvd_div < -0.2 and near_high > 0.0
            )
            out["rev_cross_upper_wick_vwap"] = float(
                wick_up >= 0.7 and vwap_rev < 0.0 and near_high > 0.0
            )
            out["rev_cross_lower_wick_cvd"] = float(
                wick_dn >= 0.7 and cvd_div > 0.2 and near_low > 0.0
            )
            out["rev_cross_lower_wick_vwap"] = float(
                wick_dn >= 0.7 and vwap_rev > 0.0 and near_low > 0.0
            )
        except Exception as exc:
            logger.debug("[REV] cross-feature computation failed: %s", exc)
        return out

    @staticmethod
    def _clip_feature_value(k: str, v: float) -> float:
        try:
            if k == "ta_rsi14":
                return float(np.clip(v, 0.0, 100.0))
            if k in ("ta_bb_pctb", "price_range_tightness"):
                return float(np.clip(v, 0.0, 1.0))
            if k == "micro_slope":
                return float(np.clip(v, -3.0, +3.0))
            if k == "last_zscore":
                return float(np.clip(v, -6.0, +6.0))
            if k == "pat_rvol":
                return float(np.clip(v, 0.0, 5.0))
            return float(v)
        except Exception:
            return float(v) if isinstance(v, (int, float)) else 0.0

    @staticmethod
    def normalize_features(features: Dict, scale: float = 1.0) -> Dict[str, float]:
        out: Dict[str, float] = {}
        try:
            denom = float(scale) if (isinstance(scale, (int, float)) and scale > 0) else 1.0
        except Exception:
            denom = 1.0

        scale_by_retstd = {
            "ema_8", "ema_21",
            "ta_macd", "ta_macd_signal", "ta_macd_hist",
            "ta_bb_bw",
        }

        for k, v in features.items():
            try:
                fv = float(v)
            except Exception:
                continue

            if FeaturePipeline._is_bounded_key(k):
                out[k] = FeaturePipeline._clip_feature_value(k, fv)
                continue

            if k == "micro_slope":
                out[k] = FeaturePipeline._clip_feature_value(k, fv)
                continue

            if k in ("last_zscore", "pat_rvol"):
                fv = FeaturePipeline._clip_feature_value(k, fv)

            if k in scale_by_retstd and denom != 1.0:
                out[k] = float(fv / max(1e-6, denom))
            else:
                out[k] = float(fv)

        try:
            logger.debug(f"[NORM] scale={denom:.6g} | keys={len(out)}")
        except Exception:
            pass
        return out

    @staticmethod
    def compute_candlestick_patterns(
        candles,
        volume: Optional[List[float]] = None,
        support_resistance: Optional[List[float]] = None,
        winrates: Optional[Dict[str, float]] = None,
        rvol_window: int = 5,
        rvol_thresh: float = 1.2,
        min_winrate: float = 0.55
    ) -> Dict[str, float]:
        # (Use your existing implementation from the current file: unchanged)
        # from feature_pipeline import FeaturePipeline as _FP  # self-ref to reuse your existing full method
        # This is a thin wrapper to keep the full method below in the same file.
        return FeaturePipeline._compute_candlestick_patterns_impl(
            candles=candles,
            winrates=winrates,
            rvol_window=rvol_window,
            rvol_thresh=rvol_thresh,
            min_winrate=min_winrate
        )

    @staticmethod
    def _compute_candlestick_patterns_impl(candles, winrates=None, rvol_window=5, rvol_thresh=1.2, min_winrate=0.55) -> Dict[str, float]:
        # Paste the full body of your current compute_candlestick_patterns here (unchanged)
        # For brevity, reusing your provided implementation 1:1
        # BEGIN original body
        out = {}
        try:
            # Pre-initialize defaults...
            flags = {
                'pat_is_hammer': 0.0, 'pat_is_inverted_hammer': 0.0, 'pat_is_shooting_star': 0.0,
                'pat_is_bullish_engulfing': 0.0, 'pat_is_bearish_engulfing': 0.0,
                'pat_is_doji': 0.0, 'pat_is_inside_bar': 0.0, 'pat_is_outside_bar': 0.0,
                'pat_is_morning_star': 0.0, 'pat_is_evening_star': 0.0,
                'pat_is_harami_bullish': 0.0, 'pat_is_harami_bearish': 0.0,
                'pat_is_piercing_line': 0.0, 'pat_is_dark_cloud': 0.0,
                'pat_is_three_white_soldiers': 0.0, 'pat_is_three_black_crows': 0.0,
                'pat_is_thestrat_2u2u_cont': 0.0, 'pat_is_thestrat_2d2d_cont': 0.0,
                'pat_is_thestrat_2d_1_2u_rev': 0.0, 'pat_is_thestrat_2u_1_2d_rev': 0.0,
            }
            winrate_out = {f'pat_winrate_{k}': 0.0 for k in [
                'hammer','inverted_hammer','shooting_star','bullish_engulfing','bearish_engulfing',
                'doji','inside_bar','outside_bar','morning_star','evening_star',
                'harami_bullish','harami_bearish','piercing_line','dark_cloud',
                'three_white_soldiers','three_black_crows',
                'thestrat_2u2u_cont','thestrat_2d2d_cont','thestrat_2d_1_2u_rev','thestrat_2u_1_2d_rev'
            ]}
            out = {}
            out.update(flags)
            out.update(winrate_out)
            out['pat_rvol'] = 0.0
            out['probability_adjustment'] = 0.0
            out['pat_confirmed_by_rvol'] = 0.0
            if hasattr(candles, "iloc"):
                df = candles.copy()
            else:
                try:
                    df = pd.DataFrame(candles)
                    if 'timestamp' in df.columns:
                        df = df.set_index('timestamp')
                except Exception:
                    return out
            cols = {'open', 'high', 'low', 'close'}
            if not cols.issubset(set(df.columns)):
                return out
            tail = df.tail(max(3, rvol_window)).copy()
            if tail.empty or len(tail) < 1:
                return out
            if 'tick_count' not in tail.columns:
                tail['tick_count'] = 0
            def row_at(i_from_end: int):
                try:
                    return tail.iloc[-i_from_end]
                except Exception:
                    return None
            c0 = row_at(1); c1 = row_at(2); c2 = row_at(3)
            def body(o, c): return abs(float(c) - float(o))
            def range_(h, l): return max(1e-9, float(h) - float(l))
            def upper_wick(o, h, c): return float(h) - max(float(o), float(c))
            def lower_wick(o, l, c): return min(float(o), float(c)) - float(l)
            def dir_(o, c):
                c = float(c); o = float(o)
                return 1 if c > o else (-1 if c < o else 0)
            def is_inside(h_child, l_child, h_parent, l_parent) -> bool:
                return (float(h_child) < float(h_parent)) and (float(l_child) > float(l_parent))
            def is_outside(h_child, l_child, h_parent, l_parent) -> bool:
                return (float(h_child) > float(h_parent)) and (float(l_child) < float(l_parent))
            def is_2_up(h_child, l_child, h_parent, l_parent) -> bool:
                inside = is_inside(h_child, l_child, h_parent, l_parent)
                outside = is_outside(h_child, l_child, h_parent, l_parent)
                return (not inside) and (not outside) and (float(h_child) > float(h_parent))
            def is_2_down(h_child, l_child, h_parent, l_parent) -> bool:
                inside = is_inside(h_child, l_child, h_parent, l_parent)
                outside = is_outside(h_child, l_child, h_parent, l_parent)
                return (not inside) and (not outside) and (float(l_child) < float(l_parent))
            default_winrates = {
                'hammer': 0.62, 'inverted_hammer': 0.56, 'shooting_star': 0.58,
                'bullish_engulfing': 0.62, 'bearish_engulfing': 0.62, 'doji': 0.52,
                'inside_bar': 0.55, 'outside_bar': 0.56, 'morning_star': 0.64, 'evening_star': 0.64,
                'harami_bullish': 0.57, 'harami_bearish': 0.57, 'piercing_line': 0.60, 'dark_cloud': 0.60,
                'three_white_soldiers': 0.64, 'three_black_crows': 0.64,
                'thestrat_2u2u_cont': 0.58, 'thestrat_2d2d_cont': 0.58, 'thestrat_2d_1_2u_rev': 0.60, 'thestrat_2u_1_2d_rev': 0.60,
            }
            wr = dict(default_winrates)
            if isinstance(winrates, dict):
                wr.update({k: float(v) for k, v in winrates.items() if k in wr})
            winrate_out = {f'pat_winrate_{k}': float(wr.get(k, 0.0)) for k in default_winrates.keys()}
            def downtrend():
                try:
                    closes = tail['close'].astype(float).values[-min(4, len(tail)):]
                    if len(closes) < 3: return False
                    return (closes[-1] < closes[-2]) and (closes[-2] < closes[-3])
                except Exception:
                    return False
            def uptrend():
                try:
                    closes = tail['close'].astype(float).values[-min(4, len(tail)):]
                    if len(closes) < 3: return False
                    return (closes[-1] > closes[-2]) and (closes[-2] > closes[-3])
                except Exception:
                    return False
                     

            # Robust RVOL baseline: trimmed median, drop very low-tick minutes (e.g., startup)
            try:
                recent_ticks = tail['tick_count'].astype(float)
                prev = recent_ticks.iloc[:-1]
                prev_clean = prev[prev >= 60]  # ignore startup/partial bars
                if len(prev_clean) >= max(1, rvol_window - 1):
                    baseline = float(prev_clean.tail(rvol_window - 1).median())
                else:
                    baseline = float(prev_clean.median()) if len(prev_clean) else float(prev.tail(max(1, rvol_window - 1)).mean())
                baseline = max(1e-9, baseline)
                rvol = float(recent_ticks.iloc[-1] / baseline) if baseline > 0 else 0.0
            except Exception:
                rvol = 0.0


                
            if c0 is not None:
                o0, h0, l0, c0c = float(c0['open']), float(c0['high']), float(c0['low']), float(c0['close'])
                b0 = body(o0, c0c)
                r0 = range_(h0, l0)
                uw0 = upper_wick(o0, h0, c0c)
                lw0 = lower_wick(o0, l0, c0c)
                body_ratio0 = b0 / r0
                if body_ratio0 <= 0.1:
                    flags['pat_is_doji'] = 1.0
                if downtrend() and lw0 >= 2.0 * b0 and uw0 <= 1.0 * b0 and body_ratio0 <= 0.35:
                    flags['pat_is_hammer'] = 1.0
                if downtrend() and uw0 >= 2.0 * b0 and lw0 <= 1.0 * b0 and body_ratio0 <= 0.35:
                    flags['pat_is_inverted_hammer'] = 1.0
                if uptrend() and uw0 >= 2.0 * b0 and lw0 <= 1.0 * b0 and body_ratio0 <= 0.35:
                    flags['pat_is_shooting_star'] = 1.0
            if c1 is not None and c0 is not None:
                o1, h1, l1, c1c = float(c1['open']), float(c1['high']), float(c1['low']), float(c1['close'])
                o0, h0, l0, c0c = float(c0['open']), float(c0['high']), float(c0['low']), float(c0['close'])
                if (c1c < o1) and (c0c > o0) and (min(o0, c0c) <= min(o1, c1c)) and (max(o0, c0c) >= max(o1, c1c)):
                    flags['pat_is_bullish_engulfing'] = 1.0
                if (c1c > o1) and (c0c < o0) and (max(o0, c0c) >= max(o1, c1c)) and (min(o0, c0c) <= min(o1, c1c)):
                    flags['pat_is_bearish_engulfing'] = 1.0
                if (float(h0) < float(h1)) and (float(l0) > float(l1)):
                    flags['pat_is_inside_bar'] = 1.0
                if (float(h0) > float(h1)) and (float(l0) < float(l1)):
                    flags['pat_is_outside_bar'] = 1.0
                small_inside_body = (max(o0, c0c) < max(o1, c1c)) and (min(o0, c0c) > min(o1, c1c))
                if small_inside_body and (c1c < o1) and (c0c >= o0):
                    flags['pat_is_harami_bullish'] = 1.0
                if small_inside_body and (c1c > o1) and (c0c <= o0):
                    flags['pat_is_harami_bearish'] = 1.0
                if downtrend() and (c1c < o1) and (c0c > o0):
                    midpoint_1 = (o1 + c1c) / 2.0
                    if (c0c >= midpoint_1) and (c0c < o1):
                        flags['pat_is_piercing_line'] = 1.0
                if uptrend() and (c1c > o1) and (c0c < o0):
                    midpoint_1 = (o1 + c1c) / 2.0
                    if (c0c <= midpoint_1) and (c0c > c1c):
                        flags['pat_is_dark_cloud'] = 1.0
            if c2 is not None and c1 is not None and c0 is not None:
                o2, h2, l2, c2c = float(c2['open']), float(c2['high']), float(c2['low']), float(c2['close'])
                o1, h1, l1, c1c = float(c1['open']), float(c1['high']), float(c1['low']), float(c1['close'])
                o0, h0, l0, c0c = float(c0['open']), float(c0['high']), float(c0['low']), float(c0['close'])
                b2 = body(o2, c2c); r2 = range_(h2, l2)
                body_ratio2 = b2 / r2
                b1 = body(o1, c1c); r1 = range_(h1, l1)
                body_ratio1 = b1 / r1 if r1 > 0 else 1.0
                b0 = body(o0, c0c); r0 = range_(h0, l0)
                body_ratio0 = b0 / r0 if r0 > 0 else 1.0
                cond_ms = (downtrend()
                           and (c2c < o2)
                           and (body_ratio1 <= 0.35)
                           and (c0c > o0)
                           and (c0c >= (o2 + c2c) / 2.0))
                if cond_ms:
                    flags['pat_is_morning_star'] = 1.0
                cond_es = (uptrend()
                           and (c2c > o2)
                           and (body_ratio1 <= 0.35)
                           and (c0c < o0)
                           and (c0c <= (o2 + c2c) / 2.0))
                if cond_es:
                    flags['pat_is_evening_star'] = 1.0
                if (dir_(o2, c2c) > 0) and (dir_(o1, c1c) > 0) and (dir_(o0, c0c) > 0):
                    if (c1c > c2c) and (c0c > c1c) and (body_ratio2 >= 0.5) and (body_ratio1 >= 0.5) and (body_ratio0 >= 0.5):
                        flags['pat_is_three_white_soldiers'] = 1.0
                if (dir_(o2, c2c) < 0) and (dir_(o1, c1c) < 0) and (dir_(o0, c0c) < 0):
                    if (c1c < c2c) and (c0c < c1c) and (body_ratio2 >= 0.5) and (body_ratio1 >= 0.5) and (body_ratio0 >= 0.5):
                        flags['pat_is_three_black_crows'] = 1.0
                if is_2_up(h1, l1, h2, l2) and is_2_up(h0, l0, h1, l1):
                    flags['pat_is_thestrat_2u2u_cont'] = 1.0
                if is_2_down(h1, l1, h2, l2) and is_2_down(h0, l0, h1, l1):
                    flags['pat_is_thestrat_2d2d_cont'] = 1.0
                if (float(h1) < float(h2)) and (float(l1) > float(l2)) and is_2_up(h0, l0, h1, l1):
                    if downtrend() or (dir_(o2, c2c) < 0):
                        flags['pat_is_thestrat_2d_1_2u_rev'] = 1.0
                if (float(h1) < float(h2)) and (float(l1) > float(l2)) and is_2_down(h0, l0, h1, l1):
                    if uptrend() or (dir_(o2, c2c) > 0):
                        flags['pat_is_thestrat_2u_1_2d_rev'] = 1.0
            confirmed = (rvol >= rvol_thresh)
            base_adj = {
                'pat_is_hammer': 0.08, 'pat_is_inverted_hammer': 0.05, 'pat_is_shooting_star': 0.06,
                'pat_is_bullish_engulfing': 0.10, 'pat_is_bearish_engulfing': 0.10,
                'pat_is_inside_bar': 0.05, 'pat_is_outside_bar': 0.06,
                'pat_is_morning_star': 0.12, 'pat_is_evening_star': 0.12, 'pat_is_doji': 0.03,
                'pat_is_harami_bullish': 0.06, 'pat_is_harami_bearish': 0.06,
                'pat_is_piercing_line': 0.10, 'pat_is_dark_cloud': 0.10,
                'pat_is_three_white_soldiers': 0.13, 'pat_is_three_black_crows': 0.13,
                'pat_is_thestrat_2u2u_cont': 0.06, 'pat_is_thestrat_2d2d_cont': 0.06,
                'pat_is_thestrat_2d_1_2u_rev': 0.08, 'pat_is_thestrat_2u_1_2d_rev': 0.08,
            }
            bullish_keys = [
                'pat_is_hammer', 'pat_is_inverted_hammer',
                'pat_is_bullish_engulfing', 'pat_is_morning_star',
                'pat_is_piercing_line', 'pat_is_three_white_soldiers',
                'pat_is_thestrat_2u2u_cont', 'pat_is_thestrat_2d_1_2u_rev',
                'pat_is_harami_bullish'
            ]
            bearish_keys = [
                'pat_is_shooting_star', 'pat_is_bearish_engulfing',
                'pat_is_evening_star', 'pat_is_dark_cloud',
                'pat_is_three_black_crows', 'pat_is_thestrat_2d2d_cont',
                'pat_is_thestrat_2u_1_2d_rev', 'pat_is_harami_bearish'
            ]
            neutral_keys = ['pat_is_inside_bar', 'pat_is_outside_bar', 'pat_is_doji']
            def adj_for(flag_key: str) -> float:
                map_key = flag_key.replace('pat_is_', '')
                wr_key = map_key
                # use defaults
                wr_val = {'hammer':0.62,'inverted_hammer':0.56,'shooting_star':0.58,'bullish_engulfing':0.62,
                          'bearish_engulfing':0.62,'doji':0.52,'inside_bar':0.55,'outside_bar':0.56,'morning_star':0.64,'evening_star':0.64,
                          'harami_bullish':0.57,'harami_bearish':0.57,'piercing_line':0.60,'dark_cloud':0.60,
                          'three_white_soldiers':0.64,'three_black_crows':0.64,'thestrat_2u2u_cont':0.58,'thestrat_2d2d_cont':0.58,
                          'thestrat_2d_1_2u_rev':0.60,'thestrat_2u_1_2d_rev':0.60}.get(wr_key, 0.0)
                if wr_val < min_winrate:
                    return 0.0
                return float(base_adj.get(flag_key, 0.0))
            bullish_adj = sum(adj_for(k) for k in bullish_keys if flags.get(k, 0.0) > 0.5)
            bearish_adj = sum(adj_for(k) for k in bearish_keys if flags.get(k, 0.0) > 0.5)
            neutral_adj = 0.5 * sum(adj_for(k) for k in neutral_keys if flags.get(k, 0.0) > 0.5)
            raw_adj = (bullish_adj - bearish_adj) + neutral_adj
            if not confirmed and abs(raw_adj) > 0.03:
                raw_adj = 0.03 * np.sign(raw_adj)
            probability_adjustment = float(np.clip(raw_adj, -0.08, 0.08))
            out = {}
            out.update(flags)
            out.update(winrate_out)
            out['pat_rvol'] = float(rvol)
            out['probability_adjustment'] = probability_adjustment
            out['pat_confirmed_by_rvol'] = 1.0 if confirmed else 0.0
            numeric_out = {}
            for k, v in out.items():
                try:
                    numeric_out[k] = float(v)
                except Exception:
                    continue
            return numeric_out
        except Exception as e:
            logger.debug(f"Pattern detection error: {e}")
            return out
        # END original body

    @staticmethod
    def compute_mtf_pattern_consensus(
        candle_df: pd.DataFrame,
        timeframes: Optional[List[str]] = None,
        rvol_window: int = 5,
        rvol_thresh: float = 1.2,
        min_winrate: float = 0.55
    ) -> Dict[str, float]:
        # Keep your existing implementation (unchanged)
        # from feature_pipeline import FeaturePipeline as _FP
        return FeaturePipeline._compute_mtf_pattern_consensus_impl(candle_df, timeframes, rvol_window, rvol_thresh, min_winrate)

    @staticmethod
    def _compute_mtf_pattern_consensus_impl(candle_df, timeframes=None, rvol_window=5, rvol_thresh=1.2, min_winrate=0.55) -> Dict[str, float]:
        # Paste your current method body unchanged
        # BEGIN original body
        out = {}
        try:
            if timeframes is None:
                timeframes = ["1T", "3T", "5T"]
            out_defaults = {"mtf_consensus": 0.0, "mtf_adj": 0.0}
            for tf0 in (timeframes or []):
                key_tf = tf0.replace("min", "T") if tf0.endswith("min") else tf0
                out_defaults[f"mtf_tf_{key_tf}"] = 0.0
                out_defaults[f"mtf_tf_{key_tf}_adj"] = 0.0
            out = dict(out_defaults)
            if not isinstance(candle_df, pd.DataFrame) or candle_df.empty:
                return out
            votes = []
            adjs = []
            for tf in timeframes:
                try:
                    tf_res = FeaturePipeline._to_pandas_freq(tf)
                    is_one_minute = tf in ("1T", "1min")
                    if is_one_minute:
                        recent = candle_df.tail(max(3, rvol_window))
                    else:
                        df = candle_df.copy()
                        ohlc = df[["open", "high", "low", "close"]].resample(tf_res, label="left", closed="left").agg({
                            "open": "first", "high": "max", "low": "min", "close": "last"
                        })
                        ticks = df[["tick_count"]].resample(tf_res, label="left", closed="left").sum()
                        recent = pd.concat([ohlc, ticks], axis=1).dropna(subset=["open","high","low","close"], how="any")
                        if recent.empty:
                            continue
                        recent = recent.tail(max(3, rvol_window))
                    pat = FeaturePipeline.compute_candlestick_patterns(
                        recent, rvol_window=rvol_window, rvol_thresh=rvol_thresh, min_winrate=min_winrate
                    )
                    adj = float(pat.get("probability_adjustment", 0.0))
                    vote = 1.0 if adj > 1e-6 else (-1.0 if adj < -1e-6 else 0.0)
                    votes.append(vote)
                    adjs.append(adj)
                    key_tf = tf.replace("min", "T") if tf.endswith("min") else tf
                    out[f"mtf_tf_{key_tf}"] = float(vote)
                    out[f"mtf_tf_{key_tf}_adj"] = float(adj)
                except Exception:
                    continue
            if not votes:
                return out
            cons = float(np.clip(np.mean(votes), -1.0, 1.0))
            weights = []
            ordered = []
            for tf in ["1T", "3T", "5T"]:
                if f"mtf_tf_{tf}_adj" in out:
                    ordered.append(out[f"mtf_tf_{tf}_adj"])
                    weights.append(1.0 if tf == "1T" else (0.6 if tf == "3T" else 0.4))
            if ordered and weights:
                w = np.asarray(weights, dtype=float)
                a = np.asarray(ordered, dtype=float)
                mtf_adj = float(np.clip(np.dot(w, a) / max(1e-9, np.sum(w)), -0.08, 0.08))
            else:
                mtf_adj = 0.0
            out["mtf_consensus"] = cons
            out["mtf_adj"] = mtf_adj
            return out
        except Exception as e:
            logger.debug(f"MTF consensus error: {e}")
            return out
        # END original body

    @staticmethod
    def compute_sr_features(candle_df: pd.DataFrame) -> Dict[str, float]:
        """
        Compute simple support/resistance features over 1T/3T/5T windows:
        - distances from last close to rolling high/low
        - recent breakout flags
        """
        out = {
            "sr_1T_hi_dist": 0.0, "sr_1T_lo_dist": 0.0,
            "sr_3T_hi_dist": 0.0, "sr_3T_lo_dist": 0.0,
            "sr_5T_hi_dist": 0.0, "sr_5T_lo_dist": 0.0,
            "sr_breakout_up": 0.0, "sr_breakout_dn": 0.0
        }
        try:
            if not isinstance(candle_df, pd.DataFrame) or candle_df.empty:
                return out
            df = candle_df.copy()
            last_close = float(df["close"].astype(float).iloc[-1])
            def _dist(window: int) -> tuple:
                sub = df.tail(window)
                if sub.empty:
                    return 0.0, 0.0
                hi = float(sub["high"].astype(float).max())
                lo = float(sub["low"].astype(float).min())
                hi_dist = (hi - last_close) / max(1e-9, last_close)
                lo_dist = (last_close - lo) / max(1e-9, last_close)
                return hi_dist, lo_dist
            out["sr_1T_hi_dist"], out["sr_1T_lo_dist"] = _dist(1)
            out["sr_3T_hi_dist"], out["sr_3T_lo_dist"] = _dist(3)
            out["sr_5T_hi_dist"], out["sr_5T_lo_dist"] = _dist(5)
            # breakout flags vs recent 5T band
            sub5 = df.tail(5)
            if not sub5.empty:
                hi5 = float(sub5["high"].astype(float).max())
                lo5 = float(sub5["low"].astype(float).min())
                out["sr_breakout_up"] = 1.0 if last_close >= hi5 else 0.0
                out["sr_breakout_dn"] = 1.0 if last_close <= lo5 else 0.0
            return out
        except Exception:
            return out

    def detect_drift(self, live_features: Dict) -> Dict[str, Dict[str, float]]:
        drift_stats = {}
        if ks_2samp is None:
            logger.debug("SciPy not available; skipping KS-based drift detection")
            return drift_stats
        for feat in self.train_features:
            try:
                base = np.asarray(self.train_features[feat], dtype=float)
                live_val = live_features.get(feat, None)
                if live_val is None:
                    continue
                live_arr = np.asarray(live_val if hasattr(live_val, '__len__') else [live_val], dtype=float)
                if base.size == 0 or live_arr.size == 0:
                    continue
                stat, pval = ks_2samp(base, live_arr, alternative='two-sided', mode='auto')
                drift_stats[feat] = {'ks_stat': float(stat), 'p_value': float(pval)}
            except Exception:
                continue
        return drift_stats
