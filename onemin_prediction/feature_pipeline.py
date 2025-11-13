"""
Unified feature engineering and model utilities.
Consolidates: feature_engineering.py, drift_detector.py, smart_order_executor.py
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging

try: 
    from scipy.stats import ks_2samp # type: ignore
except Exception: 
    ks_2samp = None

logger = logging.getLogger(__name__)




def _safe_series(arr, min_len=1):
    """Convert array to pandas Series with NaN safety."""
    try:
        s = pd.Series(arr, dtype="float64")
        if s.size < min_len:
            return pd.Series([0.0]*min_len, dtype="float64")
        return s
    except Exception:
        return pd.Series([0.0]*min_len, dtype="float64")


class TA:
    """Technical Analysis indicators without volume dependency."""
    

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
            return float(np.clip(rsi, 0.0, 100.0))  # ← NEW: clip to natural bounds
        except Exception:
            return 50.0


    @staticmethod
    def macd(prices: List[float], fast: int = 12, slow: int = 26, signal: int = 9):
        """Calculate MACD indicator."""
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
        """Calculate Bollinger Bands."""
        try:
            s = _safe_series(prices, min_len=period+1)
            ma = s.rolling(window=min(period, len(s))).mean()
            sd = s.rolling(window=min(period, len(s))).std(ddof=0)
            mid = ma.iloc[-1]
            upper = float(mid + nbdev * (sd.iloc[-1] if np.isfinite(sd.iloc[-1]) else 0.0))
            lower = float(mid - nbdev * (sd.iloc[-1] if np.isfinite(sd.iloc[-1]) else 0.0))
            px = float(s.iloc[-1])
            # %B and bandwidth
            denom = max(1e-12, upper - lower)
            pctb = (px - lower) / denom
            bw = denom / max(1e-12, mid if np.isfinite(mid) and abs(mid) > 1e-12 else 1.0)
            return float(upper), float(mid), float(lower), float(np.clip(pctb, 0.0, 1.0)), float(np.nan_to_num(bw))
        except Exception:
            return 0.0, 0.0, 0.0, 0.5, 0.0
    
    @staticmethod
    def compute_ta_bundle(prices: List[float]) -> Dict[str, float]:
        """Compute all TA indicators in one call."""
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
    """
    All-in-one feature engineering, drift detection, and order execution logic.
    Eliminates redundant file separation.
    """
    
    def __init__(self, train_features: Dict, rl_agent):
        self.train_features = train_features
        self.rl_agent = rl_agent
        logger.info("Feature pipeline initialized")



    @staticmethod
    def _to_pandas_freq(tf: str) -> str:
        """
        Convert timeframe aliases to pandas resample-friendly formats.
        'T' → 'min' (e.g., '1T' -> '1min'), leaving others unchanged.
        """
        try:
            s = str(tf).strip()
            if s.lower().endswith("t"):
                return s[:-1] + "min"
            return s
        except Exception:
            return str(tf)


    # ========== FEATURE ENGINEERING ==========
    @staticmethod
    def compute_emas(prices: List[float], periods: Optional[List[int]] = None) -> Dict[str, float]:
        """Compute EMAs for given periods."""

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
        """Compute spread from order books (volume-free)."""
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
    def volatility_regime_tag(rng_value: float, rng_history: List[float]) -> str:
        """Tag volatility regime using price range percentiles."""
        try:
            prc = np.percentile(rng_history, [33, 66]) if len(rng_history) >= 3 else [rng_value, rng_value]
            if rng_value < prc[0]:
                return 'low'
            elif rng_value < prc[1]:
                return 'medium'
            else:
                return 'high'
        except Exception:
            return 'unknown'


    @staticmethod
    def _is_bounded_key(k: str) -> bool:
        """Pattern flags, mtf votes/adj, bounded indicators and tightness/%B"""
        if k.startswith("pat_is") or k.startswith("mtf_tf_"):
            return True
        return k in {
            "pat_rvol", "probability_adjustment", "mtf_adj", "mtf_consensus",
            "price_range_tightness", "ta_bb_pctb", "ta_rsi14"
        }

    @staticmethod
    def _clip_feature_value(k: str, v: float) -> float:
        """Clip features to their natural bounds"""
        try:
            if k == "ta_rsi14":
                return float(np.clip(v, 0.0, 100.0))
            if k == "ta_bb_pctb" or k == "price_range_tightness":
                return float(np.clip(v, 0.0, 1.0))
            if k == "micro_slope":
                # micro_slope_normed is provided by main_event_loop;
                # clip robustly to limit outliers in flat regimes
                return float(np.clip(v, -3.0, +3.0))
            if k == "last_zscore":
                return float(np.clip(v, -6.0, +6.0))
            if k == "pat_rvol":
                # cap extreme RVOL spikes from tiny baselines
                return float(np.clip(v, 0.0, 5.0))
            return float(v)
        except Exception:
            return float(v) if isinstance(v, (int, float)) else 0.0



    @staticmethod
    def normalize_features(features: Dict, scale: float = 1.0) -> Dict[str, float]:
        """
        Schema-aware normalization:
        - Do NOT divide bounded/dimensionless features by ret_std.
        - Do NOT re-scale already-normalized micro_slope.
        - Only apply ret_std scaling to selected unbounded features.
        """
        out: Dict[str, float] = {}
        try:
            denom = float(scale) if (isinstance(scale, (int, float)) and scale > 0) else 1.0
        except Exception:
            denom = 1.0

        # Only these benefit from ret_std scaling (unbounded, magnitude-sensitive)
        scale_by_retstd = {
            "ema_8", "ema_21",            # raw magnitudes
            "ta_macd", "ta_macd_signal", "ta_macd_hist",
            "ta_bb_bw",                   # unbounded width proxy
            # Note: we intentionally skip: last_price, last_zscore, mean_drift_pct,
            # std_dltp_short, micro_imbalance, spread → keep consistent scales
        }

        for k, v in features.items():
            try:
                fv = float(v)
            except Exception:
                continue

            # Keep bounded keys untouched by ret_std; clip to natural bounds
            if FeaturePipeline._is_bounded_key(k):
                out[k] = FeaturePipeline._clip_feature_value(k, fv)
                continue

            # micro_slope already normalized upstream; no second scaling
            if k == "micro_slope":
                out[k] = FeaturePipeline._clip_feature_value(k, fv)
                continue

            # Optional clip for a few unstable but unbounded keys
            if k in ("last_zscore", "pat_rvol"):
                fv = FeaturePipeline._clip_feature_value(k, fv)

            if k in scale_by_retstd and denom != 1.0:
                out[k] = float(fv / max(1e-6, denom))
            else:
                out[k] = float(fv)

        # Visibility: log a one-liner occasionally for debugging
        try:
            logger.debug(f"[NORM] scale={denom:.6g} | keys={len(out)} "
                        f"| scaled_keys={len(scale_by_retstd & set(features.keys()))}")
        except Exception:
            pass

        return out



    # ========== DRIFT DETECTION ==========

    def detect_drift(self, live_features: Dict) -> Dict[str, Dict[str, float]]:
        """
        Detect concept drift using KS-statistics.
        Integrated from drift_detector.py.
        """
        drift_stats = {}
        
        # Guard when SciPy is unavailable
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



    # ========== SMART ORDER EXECUTION ==========
    def place_order(self, price: float, fill_prob: float, time_waited: float, 
                   get_mid_price_func) -> float:
        """
        Adaptive limit order execution with RL feedback.
        Integrated from smart_order_executor.py.
        """
        try:
            price = float(price) if price is not None else 0.0
            fp = float(fill_prob) if fill_prob is not None else 0.0
            tw = float(time_waited) if time_waited is not None else 0.0
        except Exception:
            price, fp, tw = 0.0, 0.0, 0.0

        try:
            threshold = float(getattr(self.rl_agent, 'threshold', 0.5))
        except Exception:
            threshold = 0.5

        if fp < threshold and tw > 5:
            try:
                mid = float(get_mid_price_func()) if callable(get_mid_price_func) else float(get_mid_price_func)
            except Exception:
                mid = price
            price = self._adjust_limit_towards_mid(price, mid, fraction=0.25)
        
        return price

    @staticmethod
    def _adjust_limit_towards_mid(price: float, mid_price: float, fraction: float) -> float:
        """Adjust limit price towards mid."""
        return price + fraction * (mid_price - price)


            
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
        """
        Detects classic candlestick patterns using the last 1–3 closed candles.
        Returns numeric-only dict; keys always present with zero defaults when history is insufficient.
        """
        try:
            # Pre-initialize defaults to keep schema stable on early returns
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
            # Pre-init winrates to 0.0; trainer/backtests may fill actuals later
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

            # Normalize input to DataFrame
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

            # Use last up to 5 rows to compute RVOL on tick_count
            tail = df.tail(max(3, rvol_window)).copy()
            if tail.empty or len(tail) < 1:
                return out
        

            # Safe tick_count (activity proxy)
            if 'tick_count' not in tail.columns:
                tail['tick_count'] = 0
            
            # Candle aliases
            # c0: last closed candle, c1: previous, c2: previous of previous
            def row_at(i_from_end: int):
                try:
                    return tail.iloc[-i_from_end]
                except Exception:
                    return None
            
            c0 = row_at(1)
            c1 = row_at(2)
            c2 = row_at(3)
            
            def body(o, c): 
                return abs(float(c) - float(o))
            def range_(h, l): 
                return max(1e-9, float(h) - float(l))
            def upper_wick(o, h, c): 
                return float(h) - max(float(o), float(c))
            def lower_wick(o, l, c): 
                return min(float(o), float(c)) - float(l)
            def dir_(o, c):
                c = float(c); o = float(o)
                return 1 if c > o else (-1 if c < o else 0)
            
            # Strict inside/outside helpers (avoid double-tagging)
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
            
            flags = {
                # existing
                'pat_is_hammer': 0.0,
                'pat_is_inverted_hammer': 0.0,
                'pat_is_shooting_star': 0.0,
                'pat_is_bullish_engulfing': 0.0,
                'pat_is_bearish_engulfing': 0.0,
                'pat_is_doji': 0.0,
                'pat_is_inside_bar': 0.0,
                'pat_is_outside_bar': 0.0,
                'pat_is_morning_star': 0.0,
                'pat_is_evening_star': 0.0,
                # new (two-candle)
                'pat_is_harami_bullish': 0.0,
                'pat_is_harami_bearish': 0.0,
                'pat_is_piercing_line': 0.0,
                'pat_is_dark_cloud': 0.0,
                # new (three-candle)
                'pat_is_three_white_soldiers': 0.0,
                'pat_is_three_black_crows': 0.0,
                # new (TheStrat sequences)
                'pat_is_thestrat_2u2u_cont': 0.0,
                'pat_is_thestrat_2d2d_cont': 0.0,
                'pat_is_thestrat_2d_1_2u_rev': 0.0,
                'pat_is_thestrat_2u_1_2d_rev': 0.0,
            }
            
            # Default winrates (placeholders; tune via backtests/logs)
            default_winrates = {
                # existing
                'hammer': 0.62,
                'inverted_hammer': 0.56,
                'shooting_star': 0.58,
                'bullish_engulfing': 0.62,
                'bearish_engulfing': 0.62,
                'doji': 0.52,
                'inside_bar': 0.55,
                'outside_bar': 0.56,
                'morning_star': 0.64,
                'evening_star': 0.64,
                # new (two-candle)
                'harami_bullish': 0.57,
                'harami_bearish': 0.57,
                'piercing_line': 0.60,
                'dark_cloud': 0.60,
                # new (three-candle)
                'three_white_soldiers': 0.64,
                'three_black_crows': 0.64,
                # new (TheStrat)
                'thestrat_2u2u_cont': 0.58,
                'thestrat_2d2d_cont': 0.58,
                'thestrat_2d_1_2u_rev': 0.60,
                'thestrat_2u_1_2d_rev': 0.60,
            }
            wr = dict(default_winrates)
            if isinstance(winrates, dict):
                # Only update known keys to keep schema stable
                wr.update({k: float(v) for k, v in winrates.items() if k in wr})
            winrate_out = {f'pat_winrate_{k}': float(wr.get(k, 0.0)) for k in default_winrates.keys()}
            
            # Trend helper (last 3 closes slope)
            def downtrend():
                try:
                    closes = tail['close'].astype(float).values[-min(4, len(tail)):]
                    if len(closes) < 3:
                        return False
                    return (closes[-1] < closes[-2]) and (closes[-2] < closes[-3])
                except Exception:
                    return False
            def uptrend():
                try:
                    closes = tail['close'].astype(float).values[-min(4, len(tail)):]
                    if len(closes) < 3:
                        return False
                    return (closes[-1] > closes[-2]) and (closes[-2] > closes[-3])
                except Exception:
                    return False
            
            # RVOL via tick_count
            try:
                recent_ticks = tail['tick_count'].astype(float)
                if len(recent_ticks) >= rvol_window:
                    baseline = max(1e-9, recent_ticks.iloc[:-1].tail(rvol_window - 1).mean())
                else:
                    baseline = max(1e-9, recent_ticks.iloc[:-1].mean()) if len(recent_ticks) > 1 else 1.0
                rvol = float(recent_ticks.iloc[-1] / baseline) if baseline > 0 else 0.0
            except Exception:
                rvol = 0.0
            
            # Single-candle patterns on c0
            if c0 is not None:
                o0, h0, l0, c0c = float(c0['open']), float(c0['high']), float(c0['low']), float(c0['close'])
                b0 = body(o0, c0c)
                r0 = range_(h0, l0)
                uw0 = upper_wick(o0, h0, c0c)
                lw0 = lower_wick(o0, l0, c0c)
                body_ratio0 = b0 / r0
                
                # Doji
                if body_ratio0 <= 0.1:
                    flags['pat_is_doji'] = 1.0
                
                # Hammer (after downtrend): small body near top, long lower shadow
                if downtrend() and lw0 >= 2.0 * b0 and uw0 <= 1.0 * b0 and body_ratio0 <= 0.35:
                    flags['pat_is_hammer'] = 1.0
                
                # Inverted Hammer (after downtrend): long upper wick
                if downtrend() and uw0 >= 2.0 * b0 and lw0 <= 1.0 * b0 and body_ratio0 <= 0.35:
                    flags['pat_is_inverted_hammer'] = 1.0
                
                # Shooting Star (after uptrend)
                if uptrend() and uw0 >= 2.0 * b0 and lw0 <= 1.0 * b0 and body_ratio0 <= 0.35:
                    flags['pat_is_shooting_star'] = 1.0
            
            # Two-candle patterns using c1, c0
            if c1 is not None and c0 is not None:
                o1, h1, l1, c1c = float(c1['open']), float(c1['high']), float(c1['low']), float(c1['close'])
                o0, h0, l0, c0c = float(c0['open']), float(c0['high']), float(c0['low']), float(c0['close'])
                
                # Engulfing (body-only engulf)
                if (c1c < o1) and (c0c > o0) and (min(o0, c0c) <= min(o1, c1c)) and (max(o0, c0c) >= max(o1, c1c)):
                    flags['pat_is_bullish_engulfing'] = 1.0
                if (c1c > o1) and (c0c < o0) and (max(o0, c0c) >= max(o1, c1c)) and (min(o0, c0c) <= min(o1, c1c)):
                    flags['pat_is_bearish_engulfing'] = 1.0
                
                # Inside bar (strict contraction)
                if is_inside(h0, l0, h1, l1):
                    flags['pat_is_inside_bar'] = 1.0
                
                # Outside bar (strict expansion)
                if is_outside(h0, l0, h1, l1):
                    flags['pat_is_outside_bar'] = 1.0
                
                # Harami (body inside previous body)
                small_inside_body = (max(o0, c0c) < max(o1, c1c)) and (min(o0, c0c) > min(o1, c1c))
                if small_inside_body and (c1c < o1) and (c0c >= o0):   # bullish harami
                    flags['pat_is_harami_bullish'] = 1.0
                if small_inside_body and (c1c > o1) and (c0c <= o0):   # bearish harami
                    flags['pat_is_harami_bearish'] = 1.0
                
                # Piercing Line (downtrend, bearish then bullish, close > 50% into prior body, below prior open)
                if downtrend() and (c1c < o1) and (c0c > o0):
                    midpoint_1 = (o1 + c1c) / 2.0
                    if (c0c >= midpoint_1) and (c0c < o1):
                        flags['pat_is_piercing_line'] = 1.0
                
                # Dark Cloud Cover (uptrend, bullish then bearish, close < 50% into prior body, above prior close)
                if uptrend() and (c1c > o1) and (c0c < o0):
                    midpoint_1 = (o1 + c1c) / 2.0
                    if (c0c <= midpoint_1) and (c0c > c1c):
                        flags['pat_is_dark_cloud'] = 1.0
            
            # Three-candle patterns using c2, c1, c0
            if c2 is not None and c1 is not None and c0 is not None:
                o2, h2, l2, c2c = float(c2['open']), float(c2['high']), float(c2['low']), float(c2['close'])
                o1, h1, l1, c1c = float(c1['open']), float(c1['high']), float(c1['low']), float(c1['close'])
                o0, h0, l0, c0c = float(c0['open']), float(c0['high']), float(c0['low']), float(c0['close'])
                
                b2 = body(o2, c2c)
                r2 = range_(h2, l2)
                body_ratio2 = b2 / r2
                
                b1 = body(o1, c1c)
                r1 = range_(h1, l1)
                body_ratio1 = b1 / r1 if r1 > 0 else 1.0
                
                b0 = body(o0, c0c)
                r0 = range_(h0, l0)
                body_ratio0 = b0 / r0 if r0 > 0 else 1.0
                
                # Morning Star (downtrend: bearish, small, bullish close > midpoint of first)
                cond_ms = (downtrend() 
                        and (c2c < o2)  # first bearish
                        and (body_ratio1 <= 0.35)  # small middle
                        and (c0c > o0)  # last bullish
                        and (c0c >= (o2 + c2c) / 2.0))
                if cond_ms:
                    flags['pat_is_morning_star'] = 1.0
                
                # Evening Star (uptrend)
                cond_es = (uptrend()
                        and (c2c > o2)  # first bullish
                        and (body_ratio1 <= 0.35)  # small middle
                        and (c0c < o0)  # last bearish
                        and (c0c <= (o2 + c2c) / 2.0))
                if cond_es:
                    flags['pat_is_evening_star'] = 1.0
                
                # Three White Soldiers: 3 strong bullish, rising closes
                if (dir_(o2, c2c) > 0) and (dir_(o1, c1c) > 0) and (dir_(o0, c0c) > 0):
                    if (c1c > c2c) and (c0c > c1c) and (body_ratio2 >= 0.5) and (body_ratio1 >= 0.5) and (body_ratio0 >= 0.5):
                        flags['pat_is_three_white_soldiers'] = 1.0
                
                # Three Black Crows: 3 strong bearish, falling closes
                if (dir_(o2, c2c) < 0) and (dir_(o1, c1c) < 0) and (dir_(o0, c0c) < 0):
                    if (c1c < c2c) and (c0c < c1c) and (body_ratio2 >= 0.5) and (body_ratio1 >= 0.5) and (body_ratio0 >= 0.5):
                        flags['pat_is_three_black_crows'] = 1.0
                
                # TheStrat sequences we can detect with (c2,c1,c0) window:
                # 2-2 continuation up/down
                if is_2_up(h1, l1, h2, l2) and is_2_up(h0, l0, h1, l1):
                    flags['pat_is_thestrat_2u2u_cont'] = 1.0
                if is_2_down(h1, l1, h2, l2) and is_2_down(h0, l0, h1, l1):
                    flags['pat_is_thestrat_2d2d_cont'] = 1.0
                
                # 2-1-2 reversal: we require c1 inside c2 and c0 2 in breakout direction.
                # Up reversal variant (common: prior 2d then inside then 2u). We relax the "prior 2d" to trend or c2 bearish.
                if is_inside(h1, l1, h2, l2) and is_2_up(h0, l0, h1, l1):
                    if downtrend() or (dir_(o2, c2c) < 0):
                        flags['pat_is_thestrat_2d_1_2u_rev'] = 1.0
                # Down reversal variant
                if is_inside(h1, l1, h2, l2) and is_2_down(h0, l0, h1, l1):
                    if uptrend() or (dir_(o2, c2c) > 0):
                        flags['pat_is_thestrat_2u_1_2d_rev'] = 1.0


            # Volume confirmation via tick_count RVOL
            confirmed = (rvol >= rvol_thresh)

            # Probability adjustment logic (reduced caps)
            base_adj = {
                # existing
                'pat_is_hammer': 0.08,
                'pat_is_inverted_hammer': 0.05,
                'pat_is_shooting_star': 0.06,
                'pat_is_bullish_engulfing': 0.10,
                'pat_is_bearish_engulfing': 0.10,
                'pat_is_inside_bar': 0.05,
                'pat_is_outside_bar': 0.06,
                'pat_is_morning_star': 0.12,
                'pat_is_evening_star': 0.12,
                'pat_is_doji': 0.03,
                # new (two-candle)
                'pat_is_harami_bullish': 0.06,
                'pat_is_harami_bearish': 0.06,
                'pat_is_piercing_line': 0.10,
                'pat_is_dark_cloud': 0.10,
                # new (three-candle)
                'pat_is_three_white_soldiers': 0.13,
                'pat_is_three_black_crows': 0.13,
                # new (TheStrat)
                'pat_is_thestrat_2u2u_cont': 0.06,
                'pat_is_thestrat_2d2d_cont': 0.06,
                'pat_is_thestrat_2d_1_2u_rev': 0.08,
                'pat_is_thestrat_2u_1_2d_rev': 0.08,
            }

            # Identify bullish vs bearish effects
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
                # Apply winrate threshold screen
                map_key = flag_key.replace('pat_is_', '')
                wr_key = map_key
                wr_val = float(wr.get(wr_key, 0.0))
                if wr_val < min_winrate:
                    return 0.0
                return float(base_adj.get(flag_key, 0.0))

            bullish_adj = sum(adj_for(k) for k in bullish_keys if flags.get(k, 0.0) > 0.5)
            bearish_adj = sum(adj_for(k) for k in bearish_keys if flags.get(k, 0.0) > 0.5)
            neutral_adj = 0.5 * sum(adj_for(k) for k in neutral_keys if flags.get(k, 0.0) > 0.5)

            raw_adj = (bullish_adj - bearish_adj) + neutral_adj

            # Require RVOL confirmation for adjustments > 0.03
            if not confirmed and abs(raw_adj) > 0.03:
                raw_adj = 0.03 * np.sign(raw_adj)

            probability_adjustment = float(np.clip(raw_adj, -0.08, 0.08))

            out = {}
            out.update(flags)
            out.update(winrate_out)
            out['pat_rvol'] = float(rvol)
            out['probability_adjustment'] = probability_adjustment
            out['pat_confirmed_by_rvol'] = 1.0 if confirmed else 0.0


            
            
            
            
            
            # Only numeric values (safe for logging/normalization)
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


    @staticmethod
    def compute_mtf_pattern_consensus(
        candle_df: pd.DataFrame,
        timeframes: Optional[List[str]] = None,
        rvol_window: int = 5,
        rvol_thresh: float = 1.2,
        min_winrate: float = 0.55
    ) -> Dict[str, float]:
        """
        Compute 1m/3m/5m pattern signals and a signed consensus in [-1,+1].

        Returns:
            {
            'mtf_consensus': signed float [-1,+1],
            'mtf_adj': probability adjustment in [-0.08, 0.08],
            'mtf_tf_1T': signed vote [-1,0,1], ... for each TF,
            'mtf_tf_1T_adj': adj in [-0.08,0.08], ... for each TF
            }
        """
        out = {}
        try:
            if timeframes is None:
                timeframes = ["1T", "3T", "5T"]
            
            # Pre-initialize defaults before any early return
            out_defaults = {
                "mtf_consensus": 0.0,
                "mtf_adj": 0.0,
            }
            for tf0 in (timeframes or []):
                key_tf = tf0.replace("min", "T") if tf0.endswith("min") else tf0
                out_defaults[f"mtf_tf_{key_tf}"] = 0.0
                out_defaults[f"mtf_tf_{key_tf}_adj"] = 0.0
            out = dict(out_defaults)

            if not isinstance(candle_df, pd.DataFrame) or candle_df.empty:
                return out
            # continue with existing logic...
            
            votes = []
            adjs = []
            for tf in timeframes:
                try:
                    tf_res = FeaturePipeline._to_pandas_freq(tf)
                    is_one_minute = tf in ("1T", "1min")

                    if is_one_minute:
                        # Use last up to 5 rows as-is (no resample)
                        recent = candle_df.tail(max(3, rvol_window))
                    else:
                        # Resample safely to TF with OHLC; preserve tick_count sum for RVOL proxy
                        df = candle_df.copy()
                        ohlc = df[["open", "high", "low", "close"]].resample(tf_res, label="left", closed="left").agg({
                            "open": "first",
                            "high": "max",
                            "low": "min",
                            "close": "last"
                        })
                        ticks = df[["tick_count"]].resample(tf_res, label="left", closed="left").sum()
                        recent = pd.concat([ohlc, ticks], axis=1)
                        recent = recent.dropna(subset=["open", "high", "low", "close"], how="any")
                        if recent.empty:
                            continue
                        recent = recent.tail(max(3, rvol_window))

                    pat = FeaturePipeline.compute_candlestick_patterns(
                        recent, rvol_window=rvol_window, rvol_thresh=rvol_thresh, min_winrate=min_winrate
                    )
                    adj = float(pat.get("probability_adjustment", 0.0))

                    # Map adjustment to a directional vote
                    vote = 1.0 if adj > 1e-6 else (-1.0 if adj < -1e-6 else 0.0)
                    votes.append(vote)
                    adjs.append(adj)

                    # Keep legacy key names stable: turn '1min' → '1T', '3min' → '3T', etc.
                    if tf.endswith("min"):
                        key_tf = tf.replace("min", "T")
                    else:
                        key_tf = tf

                    out[f"mtf_tf_{key_tf}"] = float(vote)
                    out[f"mtf_tf_{key_tf}_adj"] = float(adj)
                except Exception:
                    continue

            if not votes:
                return out
            
            # Consensus as average vote, bounded
            cons = float(np.clip(np.mean(votes), -1.0, 1.0))
            # Combine adjustments with diminishing weights (1T > 3T > 5T)
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
