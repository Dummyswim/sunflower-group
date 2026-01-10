# feature_pipeline.py
"""
Unified feature engineering and drift detection (probabilities-only).
"""
import pandas as pd
import numpy as np
import os
from typing import Any, Dict, List, Optional, Tuple, cast
import logging

try:
    from scipy.stats import ks_2samp  # type: ignore
except Exception:
    ks_2samp = None

logger = logging.getLogger(__name__)

def _coerce_float(x: Any) -> Optional[float]:
    try:
        v = float(cast(Any, x))
    except Exception:
        return None
    return v if np.isfinite(v) else None

def _ks_2samp_stats(base: np.ndarray, live_arr: np.ndarray) -> Optional[Tuple[float, float]]:
    if ks_2samp is None:
        return None
    try:
        ks = cast(Any, ks_2samp)
        result = ks(base, live_arr, alternative='two-sided', mode='auto')
    except TypeError:
        try:
            result = ks_2samp(base, live_arr, alternative='two-sided')
        except Exception:
            return None
    except Exception:
        return None

    if hasattr(result, "statistic") and hasattr(result, "pvalue"):
        stat = getattr(result, "statistic")
        pval = getattr(result, "pvalue")
    elif isinstance(result, tuple) and len(result) >= 2:
        stat, pval = result[0], result[1]
    else:
        return None

    if isinstance(stat, tuple) or isinstance(pval, tuple):
        return None
    stat_f = _coerce_float(stat)
    pval_f = _coerce_float(pval)
    if stat_f is None or pval_f is None:
        return None
    return stat_f, pval_f

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
    def stoch_kd(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        k_window: int = 14,
        d_window: int = 3,
    ) -> Tuple[pd.Series, pd.Series]:
        """Stochastic %K / %D, 0–100, NaN-safe."""
        try:
            highest = high.rolling(k_window, min_periods=1).max()
            lowest = low.rolling(k_window, min_periods=1).min()
            denom = (highest - lowest).replace(0.0, np.nan)
            k = 100.0 * (close - lowest) / denom
            k = k.bfill().fillna(50.0)
            d = k.rolling(d_window, min_periods=1).mean()
            return k, d
        except Exception:
            return pd.Series(dtype=float), pd.Series(dtype=float)

    @staticmethod
    def cci(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        window: int = 20,
    ) -> pd.Series:
        """Commodity Channel Index."""
        try:
            tp = (high + low + close) / 3.0
            ma = tp.rolling(window, min_periods=1).mean()
            md = (tp - ma).abs().rolling(window, min_periods=1).mean()
            md = md.replace(0.0, np.nan)
            cci = (tp - ma) / (0.015 * md)
            return cci.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        except Exception:
            return pd.Series(dtype=float)

    @staticmethod
    def adx(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        window: int = 14,
    ) -> pd.Series:
        """Average Directional Index (0–100)."""
        try:
            up_move = high.diff()
            down_move = -low.diff()
            plus_dm = np.where(
                (up_move > down_move) & (up_move > 0), up_move, 0.0
            )
            minus_dm = np.where(
                (down_move > up_move) & (down_move > 0), down_move, 0.0
            )

            tr1 = high - low
            tr2 = (high - close.shift()).abs()
            tr3 = (low - close.shift()).abs()
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

            tr_s = tr.rolling(window, min_periods=1).sum().replace(0.0, np.nan)
            plus_di = 100.0 * pd.Series(plus_dm).rolling(window, min_periods=1).sum() / tr_s
            minus_di = 100.0 * pd.Series(minus_dm).rolling(window, min_periods=1).sum() / tr_s

            dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan)
            adx = dx.rolling(window, min_periods=1).mean()
            return adx.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        except Exception:
            return pd.Series(dtype=float)

    @staticmethod
    def mfi(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        volume: pd.Series,
        window: int = 14,
    ) -> pd.Series:
        """Money Flow Index (0–100)."""
        try:
            tp = (high + low + close) / 3.0
            mf = tp * volume
            pos_mf = mf.where(tp >= tp.shift(), 0.0)
            neg_mf = mf.where(tp < tp.shift(), 0.0).abs()

            pos_sum = pos_mf.rolling(window, min_periods=1).sum()
            neg_sum = neg_mf.rolling(window, min_periods=1).sum().replace(0.0, np.nan)
            mr = pos_sum / neg_sum
            mfi = 100.0 - (100.0 / (1.0 + mr))
            return mfi.replace([np.inf, -np.inf], np.nan).fillna(50.0)
        except Exception:
            return pd.Series(dtype=float)

    @staticmethod
    def momentum(close: pd.Series, window: int = 14) -> pd.Series:
        """Simple close-to-close momentum."""
        try:
            mom = close.diff(window)
            return mom.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        except Exception:
            return pd.Series(dtype=float)

    @staticmethod
    def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
        """On-Balance Volume; we’ll z-score it later."""
        try:
            direction = pd.Series(
                np.sign(close.diff().fillna(0.0).to_numpy()),
                index=close.index,
                dtype=float,
            )
            obv = (direction * volume).cumsum()
            return obv.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        except Exception:
            return pd.Series(dtype=float)

    @staticmethod
    def _zscore_last(series: pd.Series, window: int = 100) -> float:
        """Return last value as clipped z-score (NaN-safe)."""
        if series is None or series.empty:
            return 0.0
        tail = series.iloc[-window:]
        try:
            m = float(tail.mean())
            s = float(tail.std(ddof=0))
            if not np.isfinite(s) or s <= 0.0:
                return 0.0
            z = (float(tail.iloc[-1]) - m) / s
            return float(np.clip(z, -5.0, 5.0))
        except Exception:
            return 0.0

    @staticmethod
    def compute_ta_bundle(
        candle_df: pd.DataFrame,
        ema_feats: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Compute a compact TA bundle used by both offline and online pipelines.

        All outputs are NaN-safe and either naturally bounded or softly clipped
        so that downstream normalization does not explode.
        """
        try:
            close = candle_df["close"].astype(float)
        except Exception:
            return {}
        close_list = close.tolist()

        out: Dict[str, float] = {}

        # --- Base series ----------------------------------------------------
        high = candle_df.get("high")
        low = candle_df.get("low")
        vol = candle_df.get("volume", candle_df.get("vol"))
        if vol is None:
            vol = candle_df.get("tick_count")

        if high is not None:
            high = high.astype(float)
        if low is not None:
            low = low.astype(float)
        if vol is not None:
            vol = vol.astype(float).fillna(0.0)

        # --- RSI14 ----------------------------------------------------------
        try:
            rsi = TA.rsi(close_list, period=14)
            out["ta_rsi14"] = float(rsi)
        except Exception:
            out["ta_rsi14"] = 50.0

        # --- MACD (line / signal / hist) -----------------------------------
        try:
            macd_line, macd_signal, macd_hist = TA.macd(close_list)
            if len(close_list):
                out["ta_macd_line"] = float(macd_line)
                out["ta_macd_signal"] = float(macd_signal)
                out["ta_macd_hist"] = float(np.clip(macd_hist, -5.0, 5.0))
        except Exception:
            pass

        # --- Bollinger Bands (+ %B, bandwidth) ------------------------------
        try:
            bb_up, bb_mid, bb_low, bb_pctb, bb_bw = TA.bollinger(prices=close_list)
            if len(close_list):
                out["ta_bb_mid"] = float(bb_mid)
                out["ta_bb_pctb"] = float(bb_pctb)  # 0–1
                out["ta_bb_bw"] = float(np.clip(bb_bw, 0.0, 5.0))
        except Exception:
            pass

        # --- Stoch / CCI / ADX / MFI (need H/L[/V]) -------------------------
        if high is not None and low is not None:
            try:
                stoch_k, stoch_d = TA.stoch_kd(high, low, close)
                if len(stoch_k):
                    out["ta_stoch_k"] = float(stoch_k.iloc[-1])  # 0–100
                if len(stoch_d):
                    out["ta_stoch_d"] = float(stoch_d.iloc[-1])
            except Exception:
                pass

            try:
                cci = TA.cci(high, low, close)
                if len(cci):
                    out["ta_cci"] = float(np.clip(cci.iloc[-1] / 200.0, -3.0, 3.0))
            except Exception:
                pass

            try:
                adx = TA.adx(high, low, close)
                if len(adx):
                    out["ta_adx"] = float(np.clip(adx.iloc[-1], 0.0, 100.0))
            except Exception:
                pass

        if high is not None and low is not None and vol is not None:
            try:
                mfi = TA.mfi(high, low, close, vol)
                if len(mfi):
                    out["ta_mfi"] = float(mfi.iloc[-1])  # 0–100
            except Exception:
                pass

        # --- Momentum & OBV-based oscillator --------------------------------
        try:
            mom14 = TA.momentum(close, window=14)
            if len(mom14):
                out["ta_mom14"] = float(
                    np.clip(mom14.iloc[-1] / max(1.0, float(close.iloc[-1])), -0.05, 0.05)
                )
        except Exception:
            pass

        if vol is not None:
            try:
                obv = TA.obv(close, vol)
                z = TA._zscore_last(obv, window=100)
                out["ta_obv_z"] = float(z)
            except Exception:
                pass

        return out

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
    def compute_emas(prices: List[float], periods: Optional[List[int]] = None, *, log: bool = False) -> Dict[str, float]:
        if periods is None:
            periods = [8, 9, 15, 21, 50]
        s = pd.Series(prices, dtype='float64') if prices else pd.Series([], dtype='float64')
        out = {}
        for p in periods:
            try:
                if len(s):
                    ema_span = max(1, int(p))
                    ema_series = s.ewm(span=ema_span, adjust=False, min_periods=1).mean()
                    out[f"ema_{p}"] = float(ema_series.iloc[-1])
                else:
                    out[f'ema_{p}'] = float('nan')
            except Exception:
                out[f'ema_{p}'] = float('nan')
        if log or str(os.getenv('DEBUG_EMA_LOG','0')).lower() in ('1','true','yes'):
            logger.debug(f"EMAs computed on series (len={len(prices)}): {out}")
        return out


    @staticmethod
    def _atr_from_ohlc(df: pd.DataFrame, period: int = 14) -> float:
        """Compute ATR(period) from OHLC dataframe (NaN-safe)."""
        try:
            if df is None or df.empty:
                return 0.0
            h = df["high"].astype(float)
            l = df["low"].astype(float)
            c = df["close"].astype(float)
            prev_c = c.shift(1)
            tr = pd.concat(
                [
                    (h - l).abs(),
                    (h - prev_c).abs(),
                    (l - prev_c).abs(),
                ],
                axis=1,
            ).max(axis=1)
            atr = tr.rolling(window=max(2, int(period)), min_periods=2).mean()
            v = float(atr.iloc[-1]) if len(atr) else 0.0
            return float(np.nan_to_num(v))
        except Exception:
            return 0.0

    @staticmethod
    def resample_ohlc(
        candle_df: pd.DataFrame,
        tf: str,
    ) -> pd.DataFrame:
        """Resample an OHLC(V) dataframe to timeframe `tf` (e.g., '5T')."""
        try:
            if candle_df is None or candle_df.empty:
                return pd.DataFrame()
            df = candle_df.copy()
            if not isinstance(df.index, pd.DatetimeIndex):
                try:
                    df.index = pd.to_datetime(df.index)
                except Exception:
                    return pd.DataFrame()
            # Normalize columns
            need = ["open", "high", "low", "close"]
            if any(c not in df.columns for c in need):
                return pd.DataFrame()

            agg = {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
            }
            if "volume" in df.columns:
                agg["volume"] = "sum"
            elif "vol" in df.columns:
                # keep compatible naming
                df = df.rename(columns={"vol": "volume"})
                agg["volume"] = "sum"
            if "tick_count" in df.columns:
                agg["tick_count"] = "sum"

            tf_res = tf
            try:
                # Pandas is deprecating the alias "T"; normalize to "min" to silence warnings.
                if isinstance(tf_res, str) and tf_res.endswith("T"):
                    tf_res = tf_res[:-1] + "min"
            except Exception:
                tf_res = tf
            out = df.resample(tf_res, label="left", closed="left").agg(cast(Any, agg))
            out = out.dropna(subset=["open", "high", "low", "close"], how="any")
            return out
        except Exception:
            return pd.DataFrame()

    @staticmethod
    def compute_ema_module(
        decision_df: pd.DataFrame,
        filter_df: pd.DataFrame,
        decision_tf: str,
        filter_tf: str,
    ) -> Tuple[Dict[str, float], Dict[str, Any]]:
        """
        Compute EMA-based regime/bias/entry-tag and EMA15 break veto on candle-synced series.

        Returns:
          ema_feats_numeric: float-only fields safe to merge into features_raw
          ema_meta: human-readable fields for logs/signals
        """
        ema_feats: Dict[str, float] = {}
        meta: Dict[str, Any] = {
            "decision_tf": decision_tf,
            "filter_tf": filter_tf,
            "regime": "na",
            "bias": 0,
            "entry_tag": "NONE",
            "entry_side": 0,
            "ema15_break": "NONE",
        }
        try:
            # --- Filter TF regime/bias (EMA21 vs EMA50) --------------------
            fdf = filter_df.tail(200) if isinstance(filter_df, pd.DataFrame) else pd.DataFrame()
            if not fdf.empty and all(c in fdf.columns for c in ("open","high","low","close")):
                closes_f = fdf["close"].astype(float).tolist()
                emas_f = FeaturePipeline.compute_emas(closes_f, periods=[21, 50], log=False)
                ema21_f = float(emas_f.get("ema_21", 0.0))
                ema50_f = float(emas_f.get("ema_50", 0.0))
                # slope approximation using last two EMA points
                if len(closes_f) >= 3:
                    emas_f_prev = FeaturePipeline.compute_emas(closes_f[:-1], periods=[21, 50], log=False)
                    ema21_prev = float(emas_f_prev.get("ema_21", ema21_f))
                    ema50_prev = float(emas_f_prev.get("ema_50", ema50_f))
                else:
                    ema21_prev, ema50_prev = ema21_f, ema50_f

                slope21 = ema21_f - ema21_prev
                slope50 = ema50_f - ema50_prev
                atr_f = FeaturePipeline._atr_from_ohlc(fdf, period=14)
                denom = max(1e-9, atr_f)

                sep_norm = abs(ema21_f - ema50_f) / denom
                slope21_norm = abs(slope21) / denom

                flat_thr = float(os.getenv("EMA_FLAT_SLOPE_ATR", "0.06"))
                tangle_thr = float(os.getenv("EMA_TANGLE_SEP_ATR", "0.20"))

                is_flat = slope21_norm <= flat_thr
                is_tangled = sep_norm <= tangle_thr

                try:
                    early_slope_thr = float(os.getenv("EMA_EARLY_SLOPE_ATR", "0.04"))
                    early_sep_thr = float(os.getenv("EMA_EARLY_SEP_ATR", "0.15"))
                except Exception:
                    early_slope_thr = 0.04
                    early_sep_thr = 0.15

                if is_flat and is_tangled:
                    regime = "CHOP"
                    bias = 0
                else:
                    if ema21_f > ema50_f and slope21 > 0 and slope50 >= 0:
                        if slope21_norm >= early_slope_thr and sep_norm >= early_sep_thr:
                            regime = "TREND_UP_EARLY" if sep_norm < tangle_thr else "TREND_UP"
                            bias = 1
                        else:
                            regime = "MIXED"
                            bias = 0
                    elif ema21_f < ema50_f and slope21 < 0 and slope50 <= 0:
                        if slope21_norm >= early_slope_thr and sep_norm >= early_sep_thr:
                            regime = "TREND_DN_EARLY" if sep_norm < tangle_thr else "TREND_DN"
                            bias = -1
                        else:
                            regime = "MIXED"
                            bias = 0
                    else:
                        regime = "MIXED"
                        bias = 0

                meta["regime"] = regime
                meta["bias"] = int(bias)
                ema_feats["ema_regime_chop_5t"] = 1.0 if regime == "CHOP" else 0.0
                ema_feats["ema_bias_5t"] = float(bias)

            # --- Decision TF entry tagging + EMA15-break veto ---------------
            ddf = decision_df.tail(250) if isinstance(decision_df, pd.DataFrame) else pd.DataFrame()
            if not ddf.empty and all(c in ddf.columns for c in ("open","high","low","close")):
                closes_d = ddf["close"].astype(float).tolist()
                emas_d = FeaturePipeline.compute_emas(closes_d, periods=[9, 15, 21], log=False)
                ema9 = float(emas_d.get("ema_9", 0.0))
                ema15 = float(emas_d.get("ema_15", 0.0))
                ema21 = float(emas_d.get("ema_21", 0.0))
                # last candle
                last = ddf.iloc[-1]
                prev = ddf.iloc[-2] if len(ddf) >= 2 else last
                o, h, l, c = float(last["open"]), float(last["high"]), float(last["low"]), float(last["close"])
                o0, c0 = float(prev["open"]), float(prev["close"])

                rng = max(1e-9, abs(h - l))
                body = abs(c - o)
                body_ratio = body / rng

                atr_d = FeaturePipeline._atr_from_ohlc(ddf.tail(50), period=14)
                atrd = max(1e-9, atr_d)

                # EMA15 break veto using candle body vs wick logic
                break_dn = (o >= ema15) and (c < ema15) and (body_ratio >= float(os.getenv("EMA_BREAK_MIN_BODYR", "0.60")))
                break_up = (o <= ema15) and (c > ema15) and (body_ratio >= float(os.getenv("EMA_BREAK_MIN_BODYR", "0.60")))
                if break_dn:
                    meta["ema15_break"] = "BREAKDOWN"
                    ema_feats["ema15_break_veto"] = -1.0
                elif break_up:
                    meta["ema15_break"] = "BREAKOUT"
                    ema_feats["ema15_break_veto"] = 1.0
                else:
                    ema_feats["ema15_break_veto"] = 0.0

                # Entry tags (do NOT force direction)
                entry_tag = "NONE"
                entry_side = 0

                # Pullback bounce near EMA15/EMA21 with rejection
                dist15 = abs(c - ema15)
                dist21 = abs(c - ema21)
                pb_thr = float(os.getenv("EMA_PULLBACK_DIST_ATR", "0.25")) * atrd
                wick_up = (h - max(o, c)) / rng
                wick_dn = (min(o, c) - l) / rng
                rej_dn = wick_dn >= float(os.getenv("EMA_REJ_WICK_MIN", "0.35"))
                rej_up = wick_up >= float(os.getenv("EMA_REJ_WICK_MIN", "0.35"))

                # Trend requires at least 2 decision candles; otherwise treat as warmup (no trend bias)
                trend_up = False
                trend_dn = False
                if len(closes_d) >= 2:
                    try:
                        ema15_prev = float(FeaturePipeline.compute_emas(closes_d[:-1], periods=[15], log=False).get("ema_15", ema15))
                    except Exception:
                        ema15_prev = ema15
                    slope15 = float(ema15 - ema15_prev)
                    # Use strict slope to avoid false "trend" when series is tiny
                    trend_up = (ema15 >= ema21) and (slope15 > 0.0)
                    trend_dn = (ema15 <= ema21) and (slope15 < 0.0)

                if trend_up and (dist15 <= pb_thr or dist21 <= pb_thr) and rej_dn and c >= ema15:
                    entry_tag = "PULLBACK"
                    entry_side = 1
                elif trend_dn and (dist15 <= pb_thr or dist21 <= pb_thr) and rej_up and c <= ema15:
                    entry_tag = "PULLBACK"
                    entry_side = -1

                # 9/15 crossover with confirmation
                try:
                    if len(closes_d) >= 3:
                        emas_prev = FeaturePipeline.compute_emas(closes_d[:-1], periods=[9,15], log=False)
                        ema9_prev = float(emas_prev.get("ema_9", ema9))
                        ema15_prev = float(emas_prev.get("ema_15", ema15))
                        x_up = (ema9_prev <= ema15_prev) and (ema9 > ema15) and (c > ema9) and (c > ema15)
                        x_dn = (ema9_prev >= ema15_prev) and (ema9 < ema15) and (c < ema9) and (c < ema15)
                        if entry_tag == "NONE":
                            if x_up:
                                entry_tag = "XOVER_CONF"
                                entry_side = 1
                            elif x_dn:
                                entry_tag = "XOVER_CONF"
                                entry_side = -1
                except Exception:
                    pass

                # Break & retest (simple, recent swing level + EMA15 confluence)
                try:
                    if entry_tag == "NONE" and len(ddf) >= 25:
                        lookback = int(os.getenv("EMA_RETEST_LOOKBACK", "20"))
                        window = ddf.iloc[-(lookback+2):-2]
                        if not window.empty:
                            lvl_hi = float(window["high"].max())
                            lvl_lo = float(window["low"].min())
                            # breakout occurred on previous candle close
                            prev_close = float(prev["close"])
                            prev_open = float(prev["open"])
                            prev_rng = max(1e-9, float(prev["high"]) - float(prev["low"]))
                            prev_bodyr = abs(prev_close - prev_open) / prev_rng
                            disp_thr = float(os.getenv("EMA_RETEST_DISP_BODYR", "0.55"))
                            # long retest
                            if prev_close > lvl_hi and prev_bodyr >= disp_thr:
                                # current candle retests level and holds, also near EMA15
                                if (l <= lvl_hi + 0.20*atrd) and (c >= lvl_hi) and (abs(lvl_hi - ema15) <= 0.35*atrd):
                                    entry_tag = "RETEST"
                                    entry_side = 1
                            # short retest
                            if prev_close < lvl_lo and prev_bodyr >= disp_thr:
                                if (h >= lvl_lo - 0.20*atrd) and (c <= lvl_lo) and (abs(lvl_lo - ema15) <= 0.35*atrd):
                                    entry_tag = "RETEST"
                                    entry_side = -1
                except Exception:
                    pass

                meta["entry_tag"] = entry_tag
                meta["entry_side"] = int(entry_side)

                tag_map = {"NONE": 0.0, "PULLBACK": 1.0, "RETEST": 2.0, "XOVER_CONF": 3.0}
                ema_feats["ema_entry_tag"] = float(tag_map.get(entry_tag, 0.0))
                ema_feats["ema_entry_side"] = float(entry_side)

        except Exception:
            # fail-safe: keep defaults
            pass
        return ema_feats, meta

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
        bounded_keys = {
            # Existing bounded features
            "indicator_score",
            "struct_pivot_swipe_up",
            "struct_pivot_swipe_down",
            "struct_fvg_up_present",
            "struct_fvg_down_present",
            "struct_ob_bull_present",
            "struct_ob_bear_present",
            "ta_rsi14",
            "ta_bb_pctb",
            "ta_bb_bw",
            "mtf_consensus",
            "pattern_prob_adjustment",
            "vwap_reversion_flag",
            "cvd_divergence",
            # New TA features
            "ta_macd_line",
            "ta_macd_signal",
            "ta_macd_hist",
            "ta_stoch_k",
            "ta_stoch_d",
            "ta_cci",
            "ta_adx",
            "ta_mfi",
            "ta_mom14",
            "ta_obv_z",
        }
        if k in bounded_keys:
            return True
        return k.startswith("struct_")


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
        # --- Fast regression slope on last N bars -------------------------
        if diffs.size == 0:
            micro_slope = 0.0
        else:
            last_px = float(arr[-1])
            denom = max(abs(last_px), 1e-6)

            window_n = min(5, arr.size)
            y = arr[-window_n:]
            x = np.arange(window_n, dtype=float)

            try:
                x_mean = float(x.mean())
                y_mean = float(y.mean())
                num = float(np.sum((x - x_mean) * (y - y_mean)))
                den = float(np.sum((x - x_mean) ** 2)) or 1e-6
                slope = num / den
            except Exception:
                slope = 0.0

            micro_slope = float(slope / denom)

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
    def compute_pivot_swipe_features(df: pd.DataFrame, lookback: int = 20) -> dict:
        """
        Approximate pivot swing + swipe behaviour on the last candles.

        Returns keys:
          - struct_pivot_is_swing_high / struct_pivot_is_swing_low ∈ {0,1}
          - struct_pivot_swipe_up / struct_pivot_swipe_down ∈ {0,1}
          - struct_pivot_dist_from_high / struct_pivot_dist_from_low  (normalised distance)
        """
        feats = {
            "struct_pivot_is_swing_high": 0.0,
            "struct_pivot_is_swing_low": 0.0,
            "struct_pivot_swipe_up": 0.0,
            "struct_pivot_swipe_down": 0.0,
            "struct_pivot_dist_from_high": 0.0,
            "struct_pivot_dist_from_low": 0.0,
        }
        if not isinstance(df, pd.DataFrame) or df.empty:
            return feats

        df = df.tail(max(3, lookback)).copy()
        if len(df) < 3:
            return feats

        highs = df["high"].astype(float)
        lows = df["low"].astype(float)
        closes = df["close"].astype(float)

        last = df.iloc[-1]
        last_close = float(last["close"])
        if not np.isfinite(last_close) or last_close <= 0.0:
            last_close = 0.0

        pivot = df.iloc[-2]
        pivot_high = float(pivot["high"])
        pivot_low = float(pivot["low"])

        try:
            swing_high = float(highs.max())
            swing_low = float(lows.min())
        except Exception:
            swing_high = pivot_high
            swing_low = pivot_low

        if np.isfinite(pivot_high) and pivot_high >= swing_high:
            feats["struct_pivot_is_swing_high"] = 1.0
        if np.isfinite(pivot_low) and pivot_low <= swing_low:
            feats["struct_pivot_is_swing_low"] = 1.0

        if last_close > 0.0:
            if np.isfinite(swing_high):
                feats["struct_pivot_dist_from_high"] = float((last_close - swing_high) / last_close)
            if np.isfinite(swing_low):
                feats["struct_pivot_dist_from_low"] = float((last_close - swing_low) / last_close)

        prev = df.iloc[-2]
        prev_high = float(prev["high"])
        prev_low = float(prev["low"])
        last_high = float(last["high"])
        last_low = float(last["low"])

        if np.isfinite(prev_high) and np.isfinite(last_high) and np.isfinite(last_close):
            if (last_high > prev_high) and (last_close < prev_high):
                feats["struct_pivot_swipe_up"] = 1.0

        if np.isfinite(prev_low) and np.isfinite(last_low) and np.isfinite(last_close):
            if (last_low < prev_low) and (last_close > prev_low):
                feats["struct_pivot_swipe_down"] = 1.0

        return feats

    @staticmethod
    def compute_fvg_features(df: pd.DataFrame) -> dict:
        """
        Simple 3-candle Fair Value Gap (FVG) approximation around last bars.
        """
        feats = {
            "struct_fvg_up_present": 0.0,
            "struct_fvg_down_present": 0.0,
            "struct_fvg_up_size": 0.0,
            "struct_fvg_down_size": 0.0,
        }
        if not isinstance(df, pd.DataFrame) or len(df) < 3:
            return feats

        df = df.tail(3).copy()
        a, b, c = df.iloc[0], df.iloc[1], df.iloc[2]

        try:
            high_a = float(a["high"])
            low_a = float(a["low"])
            low_b = float(b["low"])
            high_b = float(b["high"])
            low_c = float(c["low"])
        except Exception:
            return feats

        px_ref = float(c.get("close", high_b))
        if not np.isfinite(px_ref) or px_ref <= 0.0:
            px_ref = 1.0

        if np.isfinite(low_b) and np.isfinite(high_a) and (low_b > high_a):
            gap = float(low_b - high_a)
            if gap > 0.0:
                feats["struct_fvg_up_present"] = 1.0
                feats["struct_fvg_up_size"] = gap / px_ref

        if np.isfinite(high_b) and np.isfinite(low_a) and (high_b < low_a):
            gap = float(low_a - high_b)
            if gap > 0.0:
                feats["struct_fvg_down_present"] = 1.0
                feats["struct_fvg_down_size"] = gap / px_ref

        return feats

    @staticmethod
    def compute_orderblock_features(df: pd.DataFrame, lookback: int = 20) -> dict:
        """
        Very lightweight order-block proxy:

        - Bullish OB ≈ lowest low within lookback (support zone)
        - Bearish OB ≈ highest high within lookback (resistance zone)
        """
        feats = {
            "struct_ob_bull_present": 0.0,
            "struct_ob_bear_present": 0.0,
            "struct_ob_bull_dist": 0.0,
            "struct_ob_bear_dist": 0.0,
        }
        if not isinstance(df, pd.DataFrame) or df.empty:
            return feats

        df = df.tail(max(3, lookback)).copy()
        closes = df["close"].astype(float).to_numpy()
        highs = df["high"].astype(float).to_numpy()
        lows = df["low"].astype(float).to_numpy()

        if closes.size == 0 or highs.size == 0 or lows.size == 0:
            return feats

        last_close = float(closes[-1])
        if not np.isfinite(last_close) or last_close <= 0.0:
            last_close = 0.0

        idx_low = int(np.nanargmin(lows))
        ob_low = float(lows[idx_low])
        if np.isfinite(ob_low) and last_close > 0.0:
            feats["struct_ob_bull_present"] = 1.0
            feats["struct_ob_bull_dist"] = (last_close - ob_low) / last_close

        idx_high = int(np.nanargmax(highs))
        ob_high = float(highs[idx_high])
        if np.isfinite(ob_high) and last_close > 0.0:
            feats["struct_ob_bear_present"] = 1.0
            feats["struct_ob_bear_dist"] = (last_close - ob_high) / last_close

        # --- Conflict resolution: do NOT allow both bull & bear OB at once ---
        bull = bool(feats["struct_ob_bull_present"])
        bear = bool(feats["struct_ob_bear_present"])
        if bull and bear:
            bull_dist = float(feats.get("struct_ob_bull_dist", 0.0))
            bear_dist = float(feats.get("struct_ob_bear_dist", 0.0))

            bull_mag = abs(bull_dist)
            bear_mag = abs(bear_dist)

            max_rel_dist = 0.01
            if bull_mag > max_rel_dist and bear_mag > max_rel_dist:
                feats["struct_ob_bull_present"] = 0.0
                feats["struct_ob_bear_present"] = 0.0
                feats["struct_ob_bull_dist"] = 0.0
                feats["struct_ob_bear_dist"] = 0.0
            elif bull_mag <= bear_mag:
                feats["struct_ob_bear_present"] = 0.0
                feats["struct_ob_bear_dist"] = 0.0
            else:
                feats["struct_ob_bull_present"] = 0.0
                feats["struct_ob_bull_dist"] = 0.0

        return feats

    @staticmethod
    def compute_structure_bundle(df: pd.DataFrame) -> dict:
        """
        Unified structure signal bundle:
          - pivot swipe / swing distance
          - FVG presence & size
          - order-block presence & distance
        """
        if not isinstance(df, pd.DataFrame) or df.empty:
            return {
                "struct_pivot_is_swing_high": 0.0,
                "struct_pivot_is_swing_low": 0.0,
                "struct_pivot_swipe_up": 0.0,
                "struct_pivot_swipe_down": 0.0,
                "struct_pivot_dist_from_high": 0.0,
                "struct_pivot_dist_from_low": 0.0,
                "struct_fvg_up_present": 0.0,
                "struct_fvg_down_present": 0.0,
                "struct_fvg_up_size": 0.0,
                "struct_fvg_down_size": 0.0,
                "struct_ob_bull_present": 0.0,
                "struct_ob_bear_present": 0.0,
                "struct_ob_bull_dist": 0.0,
                "struct_ob_bear_dist": 0.0,
            }

        feats = {}
        feats.update(FeaturePipeline.compute_pivot_swipe_features(df))
        feats.update(FeaturePipeline.compute_fvg_features(df))
        feats.update(FeaturePipeline.compute_orderblock_features(df))
        return feats

    @staticmethod
    def compute_pattern_features(df: pd.DataFrame) -> Dict[str, float]:
        try:
            if not isinstance(df, pd.DataFrame) or df.empty:
                return {}
            return FeaturePipeline.compute_candlestick_patterns(
                candles=df.tail(max(3, 5)),
                rvol_window=5,
                rvol_thresh=1.2,
                min_winrate=0.55,
            )
        except Exception:
            return {}

    @staticmethod
    def compute_mtf_pattern_features(
        df: pd.DataFrame,
        base_tf: str = "1T",
        higher_tfs: Optional[List[str]] = None,
        rvol_window: int = 5,
        rvol_thresh: float = 1.2,
        min_winrate: float = 0.55,
    ) -> Dict[str, float]:
        if higher_tfs is None:
            higher_tfs = ["3T", "5T"]
        try:
            return FeaturePipeline.compute_mtf_pattern_consensus(
                candle_df=df,
                timeframes=[base_tf] + list(higher_tfs),
                rvol_window=rvol_window,
                rvol_thresh=rvol_thresh,
                min_winrate=min_winrate,
            )
        except Exception:
            return {}

    @staticmethod
    def compute_rvol(volume: np.ndarray, window: int = 10) -> float:
        try:
            v = np.asarray(volume, dtype=float)
            if v.size < window + 1:
                return 0.0
            recent = v[-1]
            base = np.nanmedian(v[-window - 1:-1])
            return float(recent / max(1e-9, base))
        except Exception:
            return 0.0

    @staticmethod
    def compute_atr_from_candles(df: pd.DataFrame, window: int = 14) -> float:
        try:
            if df is None or df.empty:
                return 0.0
            hi = pd.to_numeric(df["high"], errors="coerce")
            lo = pd.to_numeric(df["low"], errors="coerce")
            tr = (hi - lo).abs().tail(max(1, window))
            return float(np.nanmean(tr))
        except Exception:
            return 0.0

    @staticmethod
    def compute_atr(df: pd.DataFrame, window: int = 14) -> float:
        return FeaturePipeline.compute_atr_from_candles(df, window=window)

    @staticmethod
    def compute_realised_vol(px_hist: List[float], window: int = 10) -> float:
        try:
            arr = np.asarray(px_hist, dtype=float)
            if arr.size < window + 1:
                return 0.0
            diff = np.diff(arr[-(window + 1):])
            return float(np.nanstd(diff))
        except Exception:
            return 0.0

    @staticmethod
    def time_of_day_sin_cos(ts) -> tuple[float, float]:
        try:
            if not isinstance(ts, (pd.Timestamp,)):
                ts = pd.to_datetime(ts)
            minute_of_day = ts.hour * 60 + ts.minute
            angle = 2.0 * np.pi * (minute_of_day / (24.0 * 60.0))
            return float(np.sin(angle)), float(np.cos(angle))
        except Exception:
            return 0.0, 0.0

    @staticmethod
    def compute_tod_features(ts) -> Dict[str, float]:
        s, c = FeaturePipeline.time_of_day_sin_cos(ts)
        return {"tod_sin": s, "tod_cos": c}

    @staticmethod
    def compute_wick_extremes(last: pd.Series) -> tuple[float, float]:
        return FeaturePipeline._compute_wick_extremes(last)

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
                        ohlc_agg = cast(Any, {
                            "open": "first",
                            "high": "max",
                            "low": "min",
                            "close": "last",
                        })
                        ohlc = df[["open", "high", "low", "close"]].resample(tf_res, label="left", closed="left").agg(ohlc_agg)
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
    def compute_sr_features(candle_df: pd.DataFrame, timeframes: Optional[List[str]] = None) -> Dict[str, float]:
        """
        Compute simple support/resistance features over 1T/3T/5T windows:
        - distances from last close to rolling high/low
        - recent breakout flags
        """
        if timeframes is None:
            timeframes = ["1T", "3T", "5T"]
        out = {"sr_breakout_up": 0.0, "sr_breakout_dn": 0.0}
        for tf in timeframes:
            key = str(tf).replace("min", "T") if str(tf).endswith("min") else str(tf)
            out[f"sr_{key}_hi_dist"] = 0.0
            out[f"sr_{key}_lo_dist"] = 0.0
        try:
            if not isinstance(candle_df, pd.DataFrame) or candle_df.empty:
                return out
            df = candle_df.copy()
            last_close = float(df["close"].astype(float).iloc[-1])
            def _dist(window: int) -> tuple:
                sub = df.tail(max(1, window))
                if sub.empty:
                    return 0.0, 0.0
                hi = float(sub["high"].astype(float).max())
                lo = float(sub["low"].astype(float).min())
                hi_dist = (hi - last_close) / max(1e-9, last_close)
                lo_dist = (last_close - lo) / max(1e-9, last_close)
                return hi_dist, lo_dist

            breakout_window = 0
            for tf in timeframes:
                try:
                    tf_str = str(tf)
                    tf_clean = tf_str.replace("T", "").replace("min", "")
                    bars = int(float(tf_clean)) if tf_clean else 0
                except Exception:
                    bars = 0
                bars = max(1, bars)
                key = tf_str.replace("min", "T") if tf_str.endswith("min") else tf_str
                hi_d, lo_d = _dist(bars)
                out[f"sr_{key}_hi_dist"] = hi_d
                out[f"sr_{key}_lo_dist"] = lo_d
                breakout_window = max(breakout_window, bars)

            sub5 = df.tail(max(5, breakout_window))
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
                res = _ks_2samp_stats(base, live_arr)
                if res is None:
                    continue
                stat, pval = res
                drift_stats[feat] = {'ks_stat': stat, 'p_value': pval}
            except Exception:
                continue
        return drift_stats
