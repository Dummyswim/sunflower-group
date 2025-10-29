"""
Optimized signal analyzer - VOLUME INDICATORS REMOVED
Fixed all validation issues and improved signal generation
"""
"""
Enhanced Signal Analyzer with Price Action Validation and Multi-Timeframe Support
"""
import pandas as pd
import numpy as np
import logging
from typing import Dict, Optional, List, Tuple, Any
from datetime import datetime, timedelta
from collections import deque as _deque

from pattern_detector import CandlestickPatternDetector, ResistanceDetector
from logging_setup import log_span

try:
    import talib
except ImportError:
    talib = None

logger = logging.getLogger(__name__)


# ==========================================================================================
# PRICE ACTION VALIDATOR
# ==========================================================================================

class PriceActionValidator:
    """Validates signals against current price action."""
    
    def __init__(self, config):
        self.config = config
        logger.info("PriceActionValidator initialized")



    def validate_against_price_action(
            self, 
            signal_type: str, 
            df: pd.DataFrame,
            signal_result: Dict,  # NEW: Pass signal_result
            session_info: Dict,   # NEW: Pass session_info
            lookback: int = 3
        ) -> Tuple[bool, str]:
            """
            Validate signal against recent price action.
            Returns (is_valid, reason)
            """
            try:
                logger.info("=" * 50)

                log_span("PRICE ACTION VALIDATION START")

                logger.debug(f"signal_type={signal_type}, df_len={len(df)}, lookback={lookback}")

                if df.empty or len(df) < lookback:
                    logger.warning("Insufficient data for validation")
                    return True, "Insufficient data for validation"
                
                # Get last N candles
                recent_candles = df.tail(lookback)
                logger.debug(f"Recent candles: {recent_candles[['open', 'high', 'low', 'close']].to_dict('records')}")
                
                # Analyze candle patterns
                candle_analysis = self._analyze_candles(recent_candles)
                logger.info(f"Candle analysis: bullish={candle_analysis['bullish_count']}, "
                            f"strong_bullish={candle_analysis['strong_bullish_count']}, "
                            f"bearish={candle_analysis['bearish_count']}, "
                            f"strong_bearish={candle_analysis['strong_bearish_count']}, "
                            f"momentum={candle_analysis['momentum']:.2f}, "
                            f"higher_lows={candle_analysis['higher_lows']}, "
                            f"lower_highs={candle_analysis['lower_highs']}")



                # Factor ignored pattern for nuance in ranging (boost if supportive) - FIXED undefined vars
                ignored_pattern = {}  # Safe default
                session_ranging = False  # Safe default
                try:
                    # Get ignored_pattern from signal_result (now passed as param)
                    if not isinstance(signal_result, dict):
                        raise ValueError("signal_result not a dict")
                    contributions = signal_result.get('contributions', {})
                    logger.debug(f"Contributions keys: {list(contributions.keys())}")
                    ignored_pattern = contributions.get('pattern_ignored', False)
                    logger.debug(f"Ignored pattern from signal_result: {ignored_pattern}")
                    
                    # Get session from session_info (now passed as param)
                    if not isinstance(session_info, dict):
                        raise ValueError("session_info not a dict")
                    session_val = session_info.get('session', 'normal')
                    session_ranging = session_val == 'ranging'
                    logger.debug(f"Session from session_info: val={session_val}, ranging={session_ranging}")
                except Exception as e:
                    logger.debug(f"Var resolution error (ignored): {e}")


                if session_ranging and ignored_pattern:
                    pattern_signal = str(ignored_pattern.get('signal', 'neutral')).lower()
                    # Supportive means the ignored pattern agrees with our side in ranging (SELL+short, BUY+long)
                    if (signal_type == 'SELL' and pattern_signal in ('short', 'sell')) or (signal_type == 'BUY' and pattern_signal in ('long', 'buy')):
                        logger.debug(f"Ignored pattern nuance: {ignored_pattern.get('name', 'unknown')} supportive in ranging — overriding veto")
                        return True, "Ignored pattern nuance override in ranging (supportive)"

                logger.debug(f"PA nuance check: ignored_pattern={ignored_pattern}, session_ranging={session_ranging}")

                # Check for contradictions
                if "BUY" in signal_type:
                    if candle_analysis['strong_bearish_count'] >= 2:
                        return False, f"Strong bearish candles detected ({candle_analysis['strong_bearish_count']}/{lookback})"
                    
                    if candle_analysis['bearish_count'] == lookback:
                        if candle_analysis.get('higher_lows', False): 
                            logger.debug("All recent candles bearish but higher lows forming — allowing BUY") 
                        else: 
                            return False, "All recent candles are bearish"
                    
                    if candle_analysis['momentum'] < -0.5:
                        return False, f"Strong downward momentum ({candle_analysis['momentum']:.2f})"
                    if candle_analysis.get('lower_highs', False):
                        return False, "Lower highs forming — avoid buying into pressure"
                elif "SELL" in signal_type:
                    if candle_analysis['strong_bullish_count'] >= 2:
                        return False, f"Strong bullish candles detected ({candle_analysis['strong_bullish_count']}/{lookback})"
                    
                    if candle_analysis['bullish_count'] == lookback:
                        if candle_analysis.get('lower_highs', False):
                            logger.debug("All recent candles bullish but lower highs forming — allowing SELL")
                        else: 
                            return False, "All recent candles are bullish"
                        
                    if candle_analysis['momentum'] > 0.5:
                        return False, f"Strong upward momentum ({candle_analysis['momentum']:.2f})"
                    if candle_analysis.get('higher_lows', False):
                        return False, "Higher lows forming — avoid shorting rising structure"
                    
                # Additional validation for current candle — context-aware

                # Narrow allowance for trend‑pullback BUYs (strict; uses local structure + momentum) 

                try: 
                    contrib = signal_result.get('contributions', {}) if isinstance(signal_result, dict) else {} 
                    rsi_up = bool(contrib.get('rsi', {}).get('rsi_cross_up', False)) 
                    macd_slope = float(contrib.get('macd', {}).get('hist_slope', 0.0)) 
                    if ("BUY" in signal_type and candle_analysis.get('higher_lows', False) and rsi_up and macd_slope > 0 and session_info.get('session','normal') in ('trending','normal')): 
                        logger.info("PA allowance: trend‑pullback BUY → higher‑lows + RSI‑50 cross up + MACD slope>0") 
                        return True, "Trend‑pullback BUY allowance" 
                except Exception: 
                    pass 
                


                # Symmetric allowance for trend‑pullback SELLs (strict; local structure + momentum)
                try:
                    contrib = signal_result.get('contributions', {}) if isinstance(signal_result, dict) else {}
                    rsi_dn = bool(contrib.get('rsi', {}).get('rsi_cross_down', False))
                    macd_slope = float(contrib.get('macd', {}).get('hist_slope', 0.0))
                    if ("SELL" in signal_type and candle_analysis.get('lower_highs', False) and rsi_dn and macd_slope < 0 
                        and session_info.get('session','normal') in ('trending','normal')):
                        logger.info("PA allowance: trend‑pullback SELL → lower‑highs + RSI‑50 cross down + MACD slope<0")
                        return True, "Trend‑pullback SELL allowance"
                except Exception:
                    pass


                                
                current_candle = recent_candles.iloc[-1]
                current_direction = "bullish" if current_candle['close'] > current_candle['open'] else "bearish"
                body_ratio = self._calculate_body_ratio(current_candle)

                # Use a momentum gate for vetoes (percent units; validator computed *100)
                mom_gate = float(getattr(self.config, 'price_action_momentum_gate', 0.35))

                # Log context used for the decision
                logger.debug(
                    "PA current-candle check → dir=%s, body_ratio=%.2f, "
                    "momentum=%.2f, higher_lows=%s, lower_highs=%s, gate=%.2f",
                    current_direction,
                    body_ratio,
                    candle_analysis.get('momentum', 0.0),
                    candle_analysis.get('higher_lows', False),
                    candle_analysis.get('lower_highs', False),
                    mom_gate,
                )

                # Strong BUY veto only if current candle strongly bearish AND short-term context supports down move               
                if "STRONG_BUY" in signal_type and current_direction == "bearish":
                    if body_ratio > self.config.price_action_min_body_ratio:
                        mom = candle_analysis.get('momentum', 0.0)
                        has_higher_lows = candle_analysis.get('higher_lows', False)
                        if mom <= 0 or not has_higher_lows:
                            return False, "Veto STRONG_BUY: red candle without higher-lows and positive momentum"
                        logger.info("PA override: red candle but higher-lows and positive momentum — allowing STRONG_BUY")

                if "STRONG_SELL" in signal_type and current_direction == "bullish":
                    if body_ratio > self.config.price_action_min_body_ratio:
                        mom = candle_analysis.get('momentum', 0.0)
                        has_lower_highs = candle_analysis.get('lower_highs', False)
                        if mom >= 0 or not has_lower_highs:
                            return False, "Veto STRONG_SELL: green candle without lower-highs and negative momentum"
                        logger.info("PA override: green candle but lower-highs and negative momentum — allowing STRONG_SELL")


                logger.debug(f"PA Validation: signal_type={signal_type}, "
                            f"last_candle={'green' if df['close'].iloc[-1] > df['open'].iloc[-1] else 'red'}, "
                            f"last_close={df['close'].iloc[-1]:.2f}")

                

                logger.debug(f"Price action validation passed for {signal_type}")
                logger.info("PRICE ACTION VALIDATION COMPLETE")
                logger.info("=" * 50)
                
                

                try:
                    contrib = signal_result.get('contributions', {}) or {}
                    ps = contrib.get('pivot_swipe', {})
                    imb = contrib.get('imbalance', {})
                    sig = str(signal_result.get('composite_signal','NEUTRAL')).upper()
                    
                    # Hard veto: counter-swipe at PDH/PDL in limited SR room
                    sr_room = str(signal_result.get('mtf_analysis', {}).get('sr_room','UNKNOWN') if 'mtf_analysis' in signal_result else getattr(self, '_last_sr_room', 'UNKNOWN'))
                    if ps and sr_room == 'LIMITED':
                        if (sig.startswith('BUY') and ps.get('direction') == 'SHORT') or \
                           (sig.startswith('SELL') and ps.get('direction') == 'LONG'):
                            return False, f"Counter pivot swipe at {ps.get('name')} in LIMITED room"
                    
                    # Allowance: aligned swipe in ranging session
                    if ps and session_info.get('session') == 'ranging':
                        if (sig.startswith('BUY') and ps.get('direction') == 'LONG') or \
                           (sig.startswith('SELL') and ps.get('direction') == 'SHORT'):
                            return True, "Aligned pivot swipe in range"
                    
                    # Imbalance caution: if imbalance suggests retrace against signal, veto strong calls
                    if imb:
                        if (sig.startswith('STRONG_BUY') and imb.get('direction') == 'SHORT') or \
                           (sig.startswith('STRONG_SELL') and imb.get('direction') == 'LONG'):
                            return False, "Imbalance suggests adverse retrace risk"
                except Exception:
                    pass
                

                
                return True, "Price action aligned"
                
            except Exception as e:
                logger.error(f"Price action validation error: {e}")
                return True, "Validation error - allowing signal"



    
    def _analyze_candles(self, candles: pd.DataFrame) -> Dict:
        """Analyze candle patterns and characteristics."""
        try:
            
            analysis = {
                'bullish_count': 0,
                'bearish_count': 0,
                'strong_bullish_count': 0,
                'strong_bearish_count': 0,
                'momentum': 0.0,
                'avg_body_ratio': 0.0,
                'higher_lows': False,
                'lower_highs': False
            }
                    
            body_ratios = []
            price_changes = []
            
            for idx, candle in candles.iterrows():
                # Determine direction
                is_bullish = candle['close'] > candle['open']
                
                if is_bullish:
                    analysis['bullish_count'] += 1
                else:
                    analysis['bearish_count'] += 1
                
                # Calculate body ratio
                body_ratio = self._calculate_body_ratio(candle)
                body_ratios.append(body_ratio)
                
                # Check for strong candles
                if body_ratio > self.config.price_action_min_body_ratio:
                    if is_bullish:
                        analysis['strong_bullish_count'] += 1
                    else:
                        analysis['strong_bearish_count'] += 1
                
                # Calculate price change
                price_change = (candle['close'] - candle['open']) / candle['open'] if candle['open'] > 0 else 0
                price_changes.append(price_change)
            
            # Calculate momentum (weighted average of price changes)
            weights = np.linspace(0.5, 1.0, len(price_changes))  # Recent candles weighted more
            analysis['momentum'] = np.average(price_changes, weights=weights) * 100
            analysis['avg_body_ratio'] = np.mean(body_ratios)
            
            logger.debug(f"Candle analysis: Bullish={analysis['bullish_count']}, "
                        f"Bearish={analysis['bearish_count']}, "
                        f"Momentum={analysis['momentum']:.2f}")

            # Sequence structure over last 3 candles

            if len(candles) >= 3: 
                lows = np.array(candles['low'].tail(3).values, dtype=float) 
                highs = np.array(candles['high'].tail(3).values, dtype=float) 
                analysis['higher_lows'] = bool(np.all(np.diff(lows) > 0)) 
                analysis['lower_highs'] = bool(np.all(np.diff(highs) < 0))

            return analysis
            
        except Exception as e:
            logger.error(f"Candle analysis error: {e}")
            return {
                'bullish_count': 0,
                'bearish_count': 0,
                'strong_bullish_count': 0,
                'strong_bearish_count': 0,
                'momentum': 0.0,
                'avg_body_ratio': 0.0
            }
    
    def _calculate_body_ratio(self, candle: pd.Series) -> float:
        """Calculate candle body to total range ratio."""
        try:
            body = abs(candle['close'] - candle['open'])
            total_range = candle['high'] - candle['low']
            
            if total_range > 0:
                return body / total_range
            return 0.0
            
        except Exception as e:
            logger.error(f"Body ratio calculation error: {e}")
            return 0.0


# ==========================================================================================
# MULTI-TIMEFRAME ANALYZER
# ==========================================================================================

class MultiTimeframeAnalyzer:
    """Analyzes multiple timeframes for alignment."""
    
    def __init__(self, config):
        self.config = config
        logger.info("MultiTimeframeAnalyzer initialized")




    def check_timeframe_alignment_quiet(
        self,
        signal_5m: Dict,
        indicators_15m: Dict,
        df_15m: pd.DataFrame,
        session_info: Dict
    ) -> Tuple[bool, float, str]:
        prev = logger.level
        try:
            logger.setLevel(logging.WARNING)
            return self.check_timeframe_alignment(signal_5m, indicators_15m, df_15m, session_info)
        finally:
            logger.setLevel(prev)



    def check_timeframe_alignment(
        self,
        signal_5m: Dict,
        indicators_15m: Dict,
        df_15m: pd.DataFrame,
        session_info: Dict
    ) -> Tuple[bool, float, str]:
        """
        Check if 5-min signal aligns with 15-min trend (sanitized).
        Returns (is_aligned, alignment_score, description)
        """
        try:
            logger.info("=" * 50)

            log_span("MTF ALIGNMENT CHECK STARTING")

            logger.debug(
                f"MTF inputs: 5m={signal_5m.get('composite_signal','NEUTRAL')}, "
                f"15m_len={len(df_15m) if df_15m is not None else 0}, "
                f"15m_keys={list(indicators_15m.keys()) if indicators_15m else []}"
            )

            # Config gate
            if not self.config.multi_timeframe_alignment:
                logger.info("MTF alignment disabled in config")
                return True, 1.0, "MTF alignment disabled"

            # Data guard
            if not indicators_15m or df_15m is None or df_15m.empty:
                logger.warning("No 15-min data for MTF analysis")
                return True, 0.5, "No 15-min data available"

            # 5m signal direction
            sig = str(signal_5m.get('composite_signal', '')).upper()
            if "BUY" in sig:
                signal_direction, signal_name = 1, "BUY"
            elif "SELL" in sig:
                signal_direction, signal_name = -1, "SELL"
            else:
                signal_direction, signal_name = 0, "NEUTRAL"
            logger.info(f"5m Signal: {signal_name} (direction={signal_direction})")

            # 15m trend analysis snapshot
            trend_15m = self._analyze_higher_timeframe_trend(indicators_15m, df_15m)
            logger.info("15m Trend Analysis:")
            logger.info(f"  Direction: {trend_15m['direction']} (-1=bear, 0=neutral, 1=bull)")
            logger.info(f"  Strength: {trend_15m['strength']:.1f}")
            logger.info(f"  Momentum Aligned: {trend_15m['momentum_aligned']}")
            logger.info(f"  S/R Room: {trend_15m['sr_aligned']}")

            # Initialize score and breakdown
            alignment_score = 0.0
            score_breakdown: List[str] = []
            price_position = 0.5  # safe default until computed

            # 1) Direction (40%)
            if signal_direction == 0:
                alignment_score += 0.2
                score_breakdown.append("5m NEUTRAL (+0.2)")
            elif trend_15m['direction'] == signal_direction:
                alignment_score += 0.4
                score_breakdown.append("Direction MATCH (+0.4)")
            elif trend_15m['direction'] == 0:
                alignment_score += 0.2
                score_breakdown.append("Direction NEUTRAL (+0.2)")
            else:
                score_breakdown.append("Direction MISMATCH (+0.0)")



            # 2) Momentum (30%) — align relative to 5m trade direction (defer scoring until price_position computed)
            momentum_aligned_trade = False
            try:
                rsi_val_15 = float(indicators_15m.get('rsi', {}).get('value', 50))
                macd_sig_15 = str(
                    indicators_15m.get('macd', {}).get('signal_type',
                    indicators_15m.get('macd', {}).get('signal', 'neutral'))
                ).lower()
                if signal_direction == 1 and rsi_val_15 > 50 and 'bearish' not in macd_sig_15:
                    momentum_aligned_trade = True
                elif signal_direction == -1 and rsi_val_15 < 50 and 'bullish' not in macd_sig_15:
                    momentum_aligned_trade = True
            except Exception:
                pass
            momentum_pts = 0.3 if momentum_aligned_trade else 0.0
            momentum_note = "Momentum ALIGNED with 5m (+0.3)" if momentum_aligned_trade else "Momentum NOT aligned with 5m (+0.0)"





            # 3) S/R room (30%) — room in 5m trade direction
            sr_aligned_trade = False
            try:
                if df_15m is not None and not df_15m.empty and len(df_15m) >= 20 and signal_direction != 0:
                    current_price = float(df_15m['close'].iloc[-1])
                    recent_high = float(df_15m['high'].tail(20).max())
                    recent_low = float(df_15m['low'].tail(20).min())
                    rng = (recent_high - recent_low)
                    price_pos = (current_price - recent_low) / rng if rng > 0 else 0.5
                    price_position = price_pos  # expose to extreme haircut

                    if signal_direction == 1 and price_pos < 0.7:
                        sr_aligned_trade = True
                    elif signal_direction == -1 and price_pos > 0.3:
                        sr_aligned_trade = True
            except Exception:
                pass

            if sr_aligned_trade:
                alignment_score += 0.3
                score_breakdown.append("S/R room AVAILABLE for 5m (+0.3)")
            else:
                score_breakdown.append("S/R room LIMITED for 5m (+0.0)")




            # Momentum downgrade near extremes if 5m closed slope opposes trade
            try:
                # 5m closed slope from 5m signal contributions (defensive)
                slope5 = float((signal_5m.get('contributions', {}) or {}).get('macd', {}).get('hist_slope', 0.0))
            except Exception:
                slope5 = 0.0
            # Trend-aware top threshold (0.90 on strong trend days)
            top_hi = float(getattr(self.config, 'extreme_price_pos_hi', 0.95))
            try:
                if session_info.get('session', 'normal') == 'trending' and float(trend_15m.get('strength', 0.0)) >= 0.60:
                    top_hi = min(top_hi, 0.90)
            except Exception:
                pass
            bot_lo = float(getattr(self.config, 'extreme_price_pos_lo', 0.05))

            downgraded = False
            if signal_direction == 1 and price_position >= top_hi and slope5 < 0 and momentum_pts > 0.0:
                momentum_pts = 0.1
                downgraded = True
                logger.info("[MTF] Momentum downgraded at top-of-range: opposing 5m slope (slope5=%+.3f, pos=%.2f, thr=%.2f) → +0.1", slope5, price_position, top_hi)
            if signal_direction == -1 and price_position <= bot_lo and slope5 > 0 and momentum_pts > 0.0:
                momentum_pts = 0.1
                downgraded = True
                logger.info("[MTF] Momentum downgraded at bottom-of-range: opposing 5m slope (slope5=%+.3f, pos=%.2f, thr=%.2f) → +0.1", slope5, price_position, bot_lo)
            # Apply momentum points now (after downgrade check)
            alignment_score += momentum_pts
            if downgraded:
                score_breakdown.append("Momentum ALIGNED (downgraded near extreme, +0.1)")
            else:
                score_breakdown.append(momentum_note)


            # Extreme-context MTF haircut (after price_position computed)
            try:

                # Use 0.90 threshold on trend days (strength≥0.60), else config extreme_price_pos_hi
                top_hi = float(getattr(self.config, 'extreme_price_pos_hi', 0.95))
                try:
                    if session_info.get('session', 'normal') == 'trending' and float(trend_15m.get('strength', 0.0)) >= 0.60:
                        top_hi = min(top_hi, 0.90)
                except Exception:
                    pass
                if signal_direction == 1 and price_position >= top_hi and not sr_aligned_trade:
                    alignment_score = max(0.0, alignment_score - 0.15)
                    score_breakdown.append(f"Top-of-range BUY haircut (-0.15 @thr={top_hi:.2f})")
                    logger.info("MTF haircut applied: Top-of-range BUY (-0.15 @thr=%.2f)", top_hi)

                    
                    
                if signal_direction == -1 and price_position <= self.config.extreme_price_pos_lo and not sr_aligned_trade:
                    alignment_score = max(0.0, alignment_score - 0.15)
                    score_breakdown.append("Bottom-of-range SELL haircut (-0.15)")
                    logger.info("MTF haircut applied: Bottom-of-range SELL (-0.15)")
            except Exception:
                pass


            # --- Adaptive Threshold by SR Room and 15m Consistency (unified, single source of truth) ---

            # Derive an SR-room label from the boolean sr_aligned_trade computed above
            sr_room = 'AVAILABLE' if sr_aligned_trade else 'LIMITED'

            # Session label used only for logging/context
            session = session_info.get('session', 'normal') if isinstance(session_info, dict) else 'normal'

            # Base threshold from config driven by SR-room
            base_thr = self.config.mtf_threshold_available if sr_aligned_trade else self.config.mtf_threshold_limited
            logger.info(f"[Adaptive Threshold] Base by SR-room: {base_thr:.2f} (session={session}, sr_room={sr_room})")

            # Expose for callers (without changing return signature)
            try:
                self._last_sr_room = sr_room
            except Exception:
                pass



            # Compute 15m Supertrend direction consistency over the last N bars (window in config)
            cons_adj = 0.0
            st_15m_consistency = None
            try:
                st_series = indicators_15m.get('supertrend', {}).get('direction_series', None)
                if st_series is not None and len(st_series) >= int(getattr(self.config, 'mtf_consistency_window', 5)):
                    win = int(getattr(self.config, 'mtf_consistency_window', 5))
                    last_vals = st_series.tail(win).astype(float).tolist()
                    target = trend_15m['direction']  # from earlier HTF analysis
                    
                    if target != 0: 
                        matches = sum(1 for v in last_vals if v == target)
                        total = sum(1 for v in last_vals if v != 0)
                        st_15m_consistency = (matches / max(1, total)) if total > 0 else None
                        
                        if st_15m_consistency is not None:
                            # Slide threshold downward if highly consistent; upward if weak
                            if st_15m_consistency >= 0.80:
                                cons_adj = -abs(float(getattr(self.config, 'mtf_consistency_adjust', 0.05)))
                            elif st_15m_consistency <= 0.50:
                                cons_adj = +abs(float(getattr(self.config, 'mtf_consistency_adjust', 0.05)))
            except Exception as e:
                logger.debug(f"[Adaptive Threshold] Consistency calc skipped: {e}")

            if st_15m_consistency is None:
                logger.info("[Adaptive Threshold] 15m Supertrend Consistency: N/A (no adjustment)")
            else:
                logger.info(f"[Adaptive Threshold] 15m Supertrend Consistency: {st_15m_consistency:.2f}, Adjustment: {cons_adj:+.2f}")

            # Dynamic MTF threshold adjustments (market-aware, bounded)
            dyn_adj = 0.0
            if getattr(self.config, 'mtf_dynamic_enable', True):
                try:
                    session = session_info.get('session', 'normal') if isinstance(session_info, dict) else 'normal'
                    trend_dir = trend_15m.get('direction', 0)
                    trend_strength = float(trend_15m.get('strength', 0.0))
                                                
                                                
                    rsi_15 = float(indicators_15m.get('rsi', {}).get('value', 50.0))
                    bb_bw = float(indicators_15m.get('bollinger', {}).get('bandwidth', 0.0))
                    last_t = df_15m.index[-1].time() if (df_15m is not None and not df_15m.empty) else None


                    # Gate relax nudges during open/close AND optional early‑expansion window
                    from datetime import datetime as _dt
                    open_close = False
                    if last_t is not None:
                        open_close = (
                            (_dt.strptime('09:15','%H:%M').time() <= last_t < _dt.strptime('09:45','%H:%M').time()) or
                            (_dt.strptime('14:45','%H:%M').time() <= last_t <= _dt.strptime('15:15','%H:%M').time())
                        )
                    skip_relax = open_close
                    try:
                        # If an expansion flip window is active, skip relaxes until it expires
                        now_ist = df_15m.index[-1].to_pydatetime() if (df_15m is not None and not df_15m.empty) else datetime.now()
                        if getattr(self, "_skip_relax_until", None) and now_ist <= self._skip_relax_until:
                            skip_relax = True
                            logger.info(f"[MTF-DYN] Relax skipped (early expansion window until {self._skip_relax_until.strftime('%H:%M:%S')})")
                        elif getattr(self, "_skip_relax_until", None) and now_ist > self._skip_relax_until:
                            self._skip_relax_until = None
                    except Exception:
                        pass



                    # 1) Trend agree + strong + room AVAILABLE → relax a bit (skip relax at open/close or early-expansion window)
                    if (not skip_relax) and trend_dir != 0 and trend_dir == signal_direction and trend_strength >= 0.60 and sr_aligned_trade:
                        dyn_adj += float(getattr(self.config, 'mtf_adj_trend_agree', -0.05))
                        logger.info(f"[MTF-DYN] Trend agree + strong → {dyn_adj:+.2f}")
                    elif sr_aligned_trade and trend_dir != 0 and trend_dir == signal_direction and trend_strength >= 0.60:
                        logger.info("[MTF-DYN] Trend-agree relax skipped (open/close or early expansion window)")



                    # 2) Ranging + LIMITED room → tighten (always)
                    if session == 'ranging' and not sr_aligned_trade:
                        dyn_adj += float(getattr(self.config, 'mtf_adj_ranging_no_room', +0.05))
                        logger.info(f"[MTF-DYN] Ranging + LIMITED room → {dyn_adj:+.2f}")


                    # 3) 15m squeeze + room AVAILABLE → relax (skip relax at open/close AND require momentum aligned)
                    
                    if (not skip_relax) and sr_aligned_trade and momentum_aligned_trade and bb_bw > 0 and bb_bw <= float(getattr(self.config, 'mtf_squeeze_bandwidth', 8.0)):


                        dyn_adj += float(getattr(self.config, 'mtf_adj_squeeze', -0.05))
                        logger.info(f"[MTF-DYN] 15m squeeze (momentum-aligned) → {dyn_adj:+.2f}")
                    else:

                        if (not skip_relax) and sr_aligned_trade and bb_bw > 0 and bb_bw <= float(getattr(self.config, 'mtf_squeeze_bandwidth', 8.0)) and not momentum_aligned_trade:

                            logger.info(f"[MTF-DYN] Squeeze relax skipped (15m momentum not aligned)")



                    # 4) RSI extremes at range edge → tighten (always)
                    if signal_direction == 1 and rsi_15 >= float(getattr(self.config, 'mtf_rsi_extreme_buy', 75.0)) and price_position >= self.config.extreme_price_pos_hi:
                        dyn_adj += float(getattr(self.config, 'mtf_adj_extreme_rsi', +0.05))
                        logger.info(f"[MTF-DYN] BUY at top + RSI15 extreme → {dyn_adj:+.2f}")
                    if signal_direction == -1 and rsi_15 <= float(getattr(self.config, 'mtf_rsi_extreme_sell', 25.0)) and price_position <= self.config.extreme_price_pos_lo:
                        dyn_adj += float(getattr(self.config, 'mtf_adj_extreme_rsi', +0.05))
                        logger.info(f"[MTF-DYN] SELL at bottom + RSI15 extreme → {dyn_adj:+.2f}")

                    # 5) Tighten near open/close windows (always)
                    if open_close:
                        dyn_adj += float(getattr(self.config, 'mtf_adj_open_close', +0.03))
                        logger.info(f"[MTF-DYN] Open/Close tighten → {dyn_adj:+.2f}")


                            
                except Exception as e:
                    logger.debug(f"[MTF-DYN] Adjust skipped: {e}")


            thr_min = float(getattr(self.config, 'mtf_dynamic_min', 0.40))
            thr_max = float(getattr(self.config, 'mtf_dynamic_max', 0.70))
            
            



            # Regime-adaptive threshold selection
            regime_adaptive = getattr(self.config, 'mtf_regime_detection', True)
            if regime_adaptive:
                # Detect if this is a pullback or reversal trade
                is_pullback = self._detect_pullback_trade(signal_5m, trend_15m, indicators_15m)
                
                if is_pullback:
                    base_thr_regime = float(getattr(self.config, 'mtf_threshold_pullback', 0.40))
                    logger.info(f"[MTF-REGIME] 🔄 PULLBACK trade detected → threshold={base_thr_regime:.2f}")
                else:
                    base_thr_regime = float(getattr(self.config, 'mtf_threshold_reversal', 0.60))
                    logger.info(f"[MTF-REGIME] ↩️ REVERSAL trade detected → threshold={base_thr_regime:.2f}")
                
                # Use regime-specific base, then apply adjustments
                base_thr = base_thr_regime

            thr_eff = min(thr_max, max(thr_min, base_thr + cons_adj + dyn_adj))
            logger.info(f"[Adaptive Threshold] Final effective threshold: {thr_eff:.2f} (clamped [{thr_min:.2f},{thr_max:.2f}])")
            
            
            

            # Alignment verdict using effective threshold
            is_aligned = alignment_score >= thr_eff
            description = (
                f"✅ ALIGNED with 15m (score: {alignment_score:.2f} >= {thr_eff:.2f})"
                if is_aligned else
                f"❌ NOT ALIGNED with 15m (score: {alignment_score:.2f} < {thr_eff:.2f})"
            )
            logger.info(f"[Adaptive Threshold] Alignment verdict: {description}")



            # Final one-shot breakdown and result
            logger.info("MTF Score Breakdown:")
            for item in score_breakdown:
                logger.info(f"  - {item}")
            logger.info(f"TOTAL SCORE: {alignment_score:.2f}")
            logger.info(f"MTF RESULT: {description}")
            logger.info("=" * 50)
            return is_aligned, alignment_score, description

        except Exception as e:
            logger.error(f"MTF alignment check error: {e}", exc_info=True)
            return True, 0.5, "MTF check error"







    def _detect_pullback_trade(self, signal_5m: Dict, trend_15m: Dict, indicators_15m: Dict) -> bool:
        """
        Detect if this is a pullback trade (trading with HTF trend) vs reversal (against HTF).
        Returns True if pullback, False if reversal.
        """
        try:
            sig_5m = str(signal_5m.get('composite_signal', '')).upper()
            htf_dir = trend_15m.get('direction', 0)
            
            # Pullback: 5m signal aligns with 15m trend direction
            if "BUY" in sig_5m and htf_dir == 1:
                logger.info("[MTF-REGIME] Pullback: BUY with 15m uptrend")
                return True
            elif "SELL" in sig_5m and htf_dir == -1:
                logger.info("[MTF-REGIME] Pullback: SELL with 15m downtrend")
                return True
            
            # Check if 15m is in early reversal (trend weakening but not flipped)
            rsi_15m = float(indicators_15m.get('rsi', {}).get('value', 50))
            macd_15m = str(indicators_15m.get('macd', {}).get('signal_type', 'neutral')).lower()
            
            if "BUY" in sig_5m and htf_dir == -1:
                # Potential reversal from downtrend
                if rsi_15m > 40 and 'bullish' in macd_15m:
                    logger.info("[MTF-REGIME] Early reversal: BUY with 15m showing turn")
                    return True  # Treat as pullback (early reversal has HTF support forming)
            
            elif "SELL" in sig_5m and htf_dir == 1:
                # Potential reversal from uptrend
                if rsi_15m < 60 and 'bearish' in macd_15m:
                    logger.info("[MTF-REGIME] Early reversal: SELL with 15m showing turn")
                    return True  # Treat as pullback (early reversal has HTF support forming)
            
            logger.info("[MTF-REGIME] Counter-trend reversal trade")
            return False  # True reversal (counter-trend)
            
        except Exception as e:
            logger.debug(f"[MTF-REGIME] Detection error: {e}")
            return False  # Default to reversal (stricter threshold)





    def _analyze_higher_timeframe_trend(self, indicators: Dict, df: pd.DataFrame) -> Dict:
        """
        Analyze trend on a higher timeframe with detailed, high-verbosity logging.
        
        Returns a dict with:
        - direction: int  (-1: bearish, 0: neutral, 1: bullish)
        - strength: float in [0.0, 1.0] (aggregated confidence score)
        - momentum_aligned: bool
        - sr_aligned: bool

        Notes:
        - This method is defensive against missing/None/non-numeric indicator values.
        - It logs detailed diagnostics to help troubleshoot instability and data issues.
        """
        # Local helpers (no external deps added)
        def _as_float(val, default=0.0) -> float:
            try:
                if val is None:
                    return float(default)
                # Handle strings like "nan", "None", etc., gracefully
                return float(val)
            except Exception:
                return float(default)

        def _has_cols(frame: pd.DataFrame, cols) -> bool:
            try:
                return frame is not None and not frame.empty and all(c in frame.columns for c in cols)
            except Exception:
                return False

        try:
            # Ensure indicators is a dict to avoid attribute errors
            if not isinstance(indicators, dict):
                logger.warning("indicators is not a dict; proceeding with empty indicators.")
                indicators = {}

            timeframe_hint = str(indicators.get('timeframe', '15m'))
            logger.info(f"Analyzing {timeframe_hint} timeframe indicators...")

            # Initialize analysis output
            analysis = {
                'direction': 0,          # -1: bearish, 0: neutral, 1: bullish
                'strength': 0.0,
                'momentum_aligned': False,
                'sr_aligned': False
            }

            # Extract indicator sub-dicts defensively
            ema_data = indicators.get('ema', {}) or {}
            st_data = indicators.get('supertrend', {}) or {}
            rsi_data = indicators.get('rsi', {}) or {}
            macd_data = indicators.get('macd', {}) or {}

            # Collect debug info for verbose logging
            debug = {
                'ema': {},
                'supertrend': {},
                'rsi': {},
                'macd': {},
                'price': {},
                'conflict': {}
            }

            # --- 1) EMA trend ---
            ema_direction = 0
            ema_signal = str(ema_data.get('signal', 'neutral')).lower()

            short_ema = _as_float(ema_data.get('short', 0))
            medium_ema = _as_float(ema_data.get('medium', 0))
            long_ema = _as_float(ema_data.get('long', 0))

            debug['ema'] = {
                'signal': ema_signal,
                'short': short_ema,
                'medium': medium_ema,
                'long': long_ema
            }

            try:
                logger.info(f"  EMA Values: Short={short_ema:.2f}, Med={medium_ema:.2f}, Long={long_ema:.2f}")
            except Exception:
                logger.info(f"  EMA Values: Short={short_ema}, Med={medium_ema}, Long={long_ema}")
            logger.info(f"  EMA Signal: {ema_signal}")

            if ema_signal in ('bullish', 'golden_cross', 'above'):
                ema_direction = 1
                analysis['direction'] = 1
                logger.info("  EMA indicates BULLISH")
            elif ema_signal in ('bearish', 'death_cross', 'below'):
                ema_direction = -1
                analysis['direction'] = -1
                logger.info("  EMA indicates BEARISH")
            else:
                logger.info("  EMA is NEUTRAL")

            # --- 2) Supertrend ---
            st_direction = 0
            st_trend = str(st_data.get('trend', 'neutral')).lower()
            st_value = _as_float(st_data.get('value', 0))

            debug['supertrend'] = {'trend': st_trend, 'value': st_value}
            try:
                logger.info(f"  Supertrend: {st_trend} (value={st_value:.2f})")
            except Exception:
                logger.info(f"  Supertrend: {st_trend} (value={st_value})")

            if st_trend == 'bullish':
                st_direction = 1
                if analysis['direction'] >= 0:
                    analysis['direction'] = 1
                    analysis['strength'] += 0.5
                    logger.info("  Supertrend confirms BULLISH")
            elif st_trend == 'bearish':
                st_direction = -1
                if analysis['direction'] <= 0:
                    analysis['direction'] = -1
                    analysis['strength'] += 0.5
                    logger.info("  Supertrend confirms BEARISH")

            # Resolve conflicts between EMA and Supertrend — require MACD + price/RSI context to trust EMA
                        
            # --- Symmetric EMA↔Supertrend Conflict Resolution ---

            if ema_direction != 0 and st_direction != 0 and ema_direction != st_direction:
                rsi_value = _as_float(rsi_data.get('value', 50), 50)
                macd_sig = str(macd_data.get('signal_type', 'neutral')).lower()
                has_price_cols = _has_cols(df, ['close'])
                last_close = _as_float(df['close'].iloc[-1]) if has_price_cols else 0.0

                price_above_med = last_close > medium_ema
                price_below_med = last_close < medium_ema
                macd_bull = 'bullish' in macd_sig
                macd_bear = 'bearish' in macd_sig

                debug['conflict'] = {
                    'ema_direction': ema_direction,
                    'st_direction': st_direction,
                    'rsi_value': rsi_value,
                    'macd_signal_type': macd_sig,
                    'last_close': last_close,
                    'medium_ema': medium_ema,
                    'price_above_medium_ema': price_above_med,
                    'price_below_medium_ema': price_below_med
                }

                ema_supported_up = (ema_direction == 1 and price_above_med and rsi_value >= 55 and macd_bull)
                ema_supported_dn = (ema_direction == -1 and price_below_med and rsi_value <= 45 and macd_bear)
                st_supported_up  = (st_direction  == 1 and price_above_med and rsi_value >= 55 and macd_bull)
                st_supported_dn  = (st_direction  == -1 and price_below_med and rsi_value <= 45 and macd_bear)

                if ema_supported_up or ema_supported_dn:
                    analysis['direction'] = ema_direction
                    analysis['strength']  = max(analysis.get('strength', 0.0), 0.5) + 0.3
                    logger.info("  CONFLICT RESOLUTION: trust EMA (MACD+RSI+price support EMA)")
                elif st_supported_up or st_supported_dn:
                    analysis['direction'] = st_direction
                    analysis['strength']  = max(analysis.get('strength', 0.0), 0.5) + 0.2
                    logger.info("  CONFLICT RESOLUTION: trust ST (MACD+RSI+price support ST)")
                else:
                    analysis['direction'] = 0
                    analysis['strength']  = 0.25
                    logger.info("  CONFLICT RESOLUTION: insufficient support → NEUTRAL")


            # --- 3) RSI momentum ---
            if isinstance(rsi_data, dict) and rsi_data:
                rsi_value = _as_float(rsi_data.get('value', 50), 50)
                debug['rsi'] = {'value': rsi_value}
                try:
                    logger.info(f"  RSI: {rsi_value:.2f}")
                except Exception:
                    logger.info(f"  RSI: {rsi_value}")

                # Align RSI thresholds with conflict logic for consistency (55/45)
                if analysis['direction'] == 1 and rsi_value >= 55:
                    analysis['momentum_aligned'] = True
                    analysis['strength'] += 0.1
                    logger.info("  RSI supports BULLISH momentum")
                elif analysis['direction'] == -1 and rsi_value <= 45:
                    analysis['momentum_aligned'] = True
                    analysis['strength'] += 0.1
                    logger.info("  RSI supports BEARISH momentum")
                elif analysis['direction'] == 0 and 40 <= rsi_value <= 60:
                    analysis['momentum_aligned'] = True
                    logger.info("  RSI is NEUTRAL")
                else:
                    logger.info("  RSI does NOT align with trend")

            # --- 4) Support/Resistance room ---
            if _has_cols(df, ['close', 'high', 'low']) and len(df) >= 20:
                # Use numeric coercion + dropna to avoid NaN issues
                try:
                    current_price = _as_float(df['close'].iloc[-1])
                    highs = pd.to_numeric(df['high'].tail(20), errors='coerce').dropna()
                    lows = pd.to_numeric(df['low'].tail(20), errors='coerce').dropna()

                    if not highs.empty and not lows.empty:
                        recent_high = float(highs.max())
                        recent_low = float(lows.min())
                        price_range = recent_high - recent_low

                        if price_range > 0:
                            price_position = (current_price - recent_low) / price_range
                        else:
                            price_position = 0.5  # flat range fallback

                        debug['price'] = {
                            'current': current_price,
                            'recent_low': recent_low,
                            'recent_high': recent_high,
                            'range': price_range,
                            'position_0to1': price_position
                        }

                        try:
                            logger.info(f"  Price Position: {price_position:.2f} (0=low, 1=high)")
                            logger.info(f"  Range: {recent_low:.2f} - {recent_high:.2f}")
                        except Exception:
                            logger.info(f"  Price Position: {price_position} (0=low, 1=high)")
                            logger.info(f"  Range: {recent_low} - {recent_high}")

                        # Check if there's room to move
                        if analysis['direction'] == 1 and price_position < 0.8:
                            analysis['sr_aligned'] = True
                            analysis['strength'] += 0.1
                            logger.info("  S/R: Room to move UP")
                        elif analysis['direction'] == -1 and price_position > 0.2:
                            analysis['sr_aligned'] = True
                            analysis['strength'] += 0.1
                            logger.info("  S/R: Room to move DOWN")
                        elif analysis['direction'] == 0 and 0.3 <= price_position <= 0.7:
                            analysis['sr_aligned'] = True
                            logger.info("  S/R: In middle of range")
                        else:
                            logger.info("  S/R: Limited room for movement")
                    else:
                        logger.info("  S/R: Insufficient numeric highs/lows in last 20 bars; skipping S/R assessment")
                except Exception as e_sr:
                    logger.warning(f"  S/R computation skipped due to error: {e_sr}", exc_info=True)
            else:
                logger.info("  S/R: Dataframe missing required columns or insufficient length (need >= 20 rows); skipping S/R")

            # Clamp strength to [0, 1] for stability
            if not isinstance(analysis['strength'], (int, float)):
                analysis['strength'] = 0.0
            analysis['strength'] = max(0.0, min(1.0, float(analysis['strength'])))

            # Final summary + debug dump
            logger.info(
                f"{timeframe_hint} Summary: "
                f"Direction={analysis['direction']}, "
                f"Strength={analysis['strength']:.2f}, "
                f"Momentum={analysis['momentum_aligned']}, "
                f"S/R={analysis['sr_aligned']}"
            )
            logger.debug(f"Diagnostics: {debug}")

            return analysis

        except Exception as e:
            logger.error(f"Higher timeframe analysis error: {e}", exc_info=True)
            return {
                'direction': 0,
                'strength': 0.0,
                'momentum_aligned': False,
                'sr_aligned': False
            }





class SignalValidator:
    """Validates trading signals."""
    def __init__(self, config):
        self.config = config

class SignalDurationPredictor:
    """Predicts signal duration."""
    def __init__(self, config):
        self.config = config
    
    def predict(self, indicators: Dict, df: pd.DataFrame, signal: Dict) -> Dict:
        """Predict signal duration."""
        return {
            'estimated_minutes': 10,
            'confidence': 'medium',
            'factors': []
        }

class EnhancedIndicatorConsensus:
    """Analyzes indicator consensus."""
    def __init__(self, config):
        self.config = config
    
    def calculate_group_consensus(self, indicators: Dict) -> Dict:
        """Calculate consensus among indicator groups."""
        try:
            groups = {}
            for group_name, indicator_list in self.config.indicator_groups.items():
                bullish_count = 0
                bearish_count = 0    
                
                for ind_name in indicator_list: 
                    if ind_name in indicators: 
                        # Prefer semantic signals when available (e.g., MACD uses 'signal_type') 
                        if ind_name == 'macd': 
                            sig_val = indicators[ind_name].get('signal_type', indicators[ind_name].get('signal', 'neutral')) 
                        else: 
                            sig_val = indicators[ind_name].get('signal', 'neutral')
                    
                        if 'bullish' in str(sig_val).lower() or 'buy' in str(sig_val).lower():
                            bullish_count += 1
                        elif 'bearish' in str(sig_val).lower() or 'sell' in str(sig_val).lower():
                            bearish_count += 1

                
                
                total = bullish_count + bearish_count
                if total > 0:
                    agreement = max(bullish_count, bearish_count) / total
                else:
                    agreement = 0
                
                groups[group_name] = {
                    'bullish': bullish_count,
                    'bearish': bearish_count,
                    'agreement': agreement,
                    'indicator_count': len(indicator_list)
                }
                
            logger.debug(f"Group consensus → " f"leading={groups.get('leading', {}).get('agreement', 0):.2f}, " f"lagging={groups.get('lagging', {}).get('agreement', 0):.2f}, " f"volatility={groups.get('volatility', {}).get('agreement', 0):.2f}") 
            return groups
        
        except Exception as e:
            logger.error(f"Group consensus error: {e}")
            return {}

class TrendAlignmentAnalyzer:
    """Analyzes trend alignment."""
    def __init__(self, config):
        self.config = config
    
    def analyze_trend(self, df: pd.DataFrame, indicators: Dict) -> Dict:
        """Analyze trend alignment."""
        try:
            if df.empty or len(df) < 20:
                return {'trend': 'neutral', 'aligned': False}
            
            # Simple trend using price MA
            ma20 = df['close'].rolling(20).mean()
            current_price = df['close'].iloc[-1]
            
            if current_price > ma20.iloc[-1]:
                trend = 'bullish'
            elif current_price < ma20.iloc[-1]:
                trend = 'bearish'
            else:
                trend = 'neutral'
            
            return {
                'trend': trend,
                'aligned': True,
                'strength': abs(current_price - ma20.iloc[-1]) / ma20.iloc[-1] * 100
            }
        except Exception as e:
            logger.error(f"Trend analysis error: {e}")
            return {'trend': 'neutral', 'aligned': False}
        
    

class AdaptiveThresholdManager:
    def __init__(self):
        self.recent_accuracy = _deque(maxlen=50)
        self.threshold_adjustments = {}
        logger.info("[LEARN] AdaptiveThresholdManager initialized")
    
    def learn_from_session(self, hitrate_data: list[dict]):
        try:
            for s in (hitrate_data or []):
                if s.get('correct') and s.get('rejection_reason'):
                    rr = str(s.get('rejection_reason'))
                    self.threshold_adjustments[rr] = self.threshold_adjustments.get(rr, 0) - 0.01
                elif (s.get('correct') is False) and (not s.get('rejection_reason')):
                    self.threshold_adjustments['passed_wrong'] = self.threshold_adjustments.get('passed_wrong', 0) + 0.01
            logger.info(f"[LEARN] Suggested adjustments: {self.threshold_adjustments}")
        except Exception as e:
            logger.debug(f"[LEARN] learn_from_session skipped: {e}")

def apply_learned_adjustments(base_conf: float, ctx_key: str, manager: Optional[AdaptiveThresholdManager]) -> float:
    try:
        if not manager:
            return base_conf
        adj = float(manager.threshold_adjustments.get(ctx_key, 0.0))
        new_conf = max(0.0, min(100.0, base_conf + adj * 100))
        logger.info(f"[LEARN] Confidence adjusted by {adj:+.3f} for {ctx_key}: {base_conf:.1f}→{new_conf:.1f}")
        return new_conf
    except Exception:
        return base_conf
        
        
# ==========================================================================================
# ENHANCED SIGNAL ANALYZER WITH ALL IMPROVEMENTS
# ==========================================================================================

class ConsolidatedSignalAnalyzer:
    """Enhanced signal analyzer with all new features."""
    
    def __init__(self, config, technical_analysis=None):
        self.config = config
        self.technical = technical_analysis
        
        self.pattern_detector = CandlestickPatternDetector(self.config) if 'CandlestickPatternDetector' in globals() else None
        self.resistance_detector = ResistanceDetector() if 'ResistanceDetector' in globals() else None
        logger.info("[PATTERN] Detector initialized (TA‑Lib layer=%s)", "ON" if self.pattern_detector else "OFF")
        

                
        
        # Initialize validators and analyzers
        self.price_action_validator = PriceActionValidator(config)
        self.mtf_analyzer = MultiTimeframeAnalyzer(config)
        
        # Previous components (updated)
        # self.validator = SignalValidator(config)
        self.predictor = SignalDurationPredictor(config)
        self.consensus_analyzer = EnhancedIndicatorConsensus(config)
        self.trend_analyzer = TrendAlignmentAnalyzer(config)
        
        # History tracking
        # self.signal_history = deque(maxlen=100)
        self.signal_history = _deque(maxlen=100)
        self.last_alert_time = None
        self._last_reject = None  # Add this line
        self.current_df = None 
        self._nm_std_ref = None
        
 
        self._nm_last_side = 0
        self._nm_side_streak = 0
        
        # Online micro-only next-1m model
        self.nm_model = None
        self._nm_pending = {}  # next_bar_time -> {'x': np.array, 'p_ml': float}
        try:
            if getattr(self.config, 'enable_next_minute_ml', True):
                from next_minute_model import OnlineLogit
                # We use 10 features (see _build_nm_features)
                self.nm_model = OnlineLogit(n_features=10,
                                            lr=float(getattr(self.config,'nm_ml_lr',0.03)),
                                            l2=float(getattr(self.config,'nm_ml_l2',0.0005)))
                logger.info("[NM-ML] OnlineLogit initialized (n=10, lr=%.3f, l2=%.5f)",
                            float(getattr(self.config,'nm_ml_lr',0.03)),
                            float(getattr(self.config,'nm_ml_l2',0.0005)))
        except Exception as e:
            logger.warning("[NM-ML] init skipped: %s", e)
            self.nm_model = None
        
        logger.info("Enhanced ConsolidatedSignalAnalyzer initialized")




    def _detect_momentum_exhaustion(self, indicators: Dict, df: pd.DataFrame, signal_direction: str) -> Tuple[bool, str]:
        """
        Detect if momentum is exhausted (lagging indicators predict past, not future).
        Returns (is_exhausted, reason)
        """
        try:
            logger.info("=" * 50)
            logger.info("[MOMENTUM-EXHAUSTION] Detection started")
            
            if not getattr(self.config, 'enable_momentum_exhaustion', True):
                logger.info("[MOMENTUM-EXHAUSTION] Detection disabled")
                return False, "Detection disabled"
            
            rsi_val = float(indicators.get('rsi', {}).get('value', 50))
            
            macd_hist_series = indicators.get('macd', {}).get('histogram_series', pd.Series(dtype=float))
            
            logger.info(f"[MOMENTUM-EXHAUSTION] Signal={signal_direction}, RSI={rsi_val:.2f}")
            
            # Check RSI extremes
            if "BUY" in signal_direction:
                rsi_threshold = float(getattr(self.config, 'momentum_exhaustion_rsi_threshold', 75.0))
                if rsi_val >= rsi_threshold:
                    logger.warning(f"[MOMENTUM-EXHAUSTION] ❌ BUY rejected: RSI {rsi_val:.2f} >= {rsi_threshold}")
                    return True, f"RSI overbought ({rsi_val:.2f})"
            
            elif "SELL" in signal_direction:
                rsi_low = float(getattr(self.config, 'momentum_exhaustion_rsi_low', 25.0))
                if rsi_val <= rsi_low:
                    logger.warning(f"[MOMENTUM-EXHAUSTION] ❌ SELL rejected: RSI {rsi_val:.2f} <= {rsi_low}")
                    return True, f"RSI oversold ({rsi_val:.2f})"
            
            # Check MACD histogram weakening (last N bars)
            if not macd_hist_series.empty and len(macd_hist_series) >= 4:
                bars_to_check = int(getattr(self.config, 'momentum_exhaustion_macd_bars', 3))
                recent_hist = macd_hist_series.tail(bars_to_check).values
                
                if "BUY" in signal_direction:
                    # Check if MACD histogram is decreasing despite BUY signal
                    if all(recent_hist[i] > recent_hist[i+1] for i in range(len(recent_hist)-1)):
                        logger.warning(f"[MOMENTUM-EXHAUSTION] ❌ BUY rejected: MACD histogram weakening")
                        logger.info(f"[MOMENTUM-EXHAUSTION] Histogram values: {recent_hist}")
                        return True, "MACD momentum weakening"
                
                elif "SELL" in signal_direction:
                    # Check if MACD histogram is increasing despite SELL signal
                    if all(recent_hist[i] < recent_hist[i+1] for i in range(len(recent_hist)-1)):
                        logger.warning(f"[MOMENTUM-EXHAUSTION] ❌ SELL rejected: MACD histogram strengthening against SELL")
                        logger.info(f"[MOMENTUM-EXHAUSTION] Histogram values: {recent_hist}")
                        return True, "MACD momentum strengthening (wrong direction)"
            
            # Check price-RSI divergence if enabled
            if getattr(self.config, 'momentum_exhaustion_divergence_check', True):
                rsi_divergence = indicators.get('rsi', {}).get('divergence', 'none')
                if "BUY" in signal_direction and rsi_divergence == 'bearish_divergence':
                    logger.warning(f"[MOMENTUM-EXHAUSTION] ❌ BUY rejected: Bearish divergence detected")
                    return True, "Bearish RSI divergence"
                elif "SELL" in signal_direction and rsi_divergence == 'bullish_divergence':
                    logger.warning(f"[MOMENTUM-EXHAUSTION] ❌ SELL rejected: Bullish divergence detected")
                    return True, "Bullish RSI divergence"
            
            logger.info("[MOMENTUM-EXHAUSTION] ✅ No exhaustion detected")
            logger.info("=" * 50)
            return False, "No exhaustion"
            
        except Exception as e:
            logger.error(f"[MOMENTUM-EXHAUSTION] Error: {e}", exc_info=True)
            return False, "Error in detection"




    def _get_value(self, d: dict, *path, default=None):
        try:
            for k in path:
                if isinstance(d, dict):
                    d = d.get(k, {})
                else:
                    return default
            return d if d not in (None, {}, []) else default
        except Exception:
            return default



    def _compose_explanation_line(self, ind: dict, contrib: dict, signal_result: dict) -> str:
        """Compose explainer sentence for the next 5m candle, including historical win-rate [recent/long] if available."""
        try:
            side = str(signal_result.get('composite_signal','NEUTRAL'))
            side_simple = "BUY" if "BUY" in side else "SELL" if "SELL" in side else "NEUTRAL"
            rsi_v = self._get_value(ind, 'rsi', 'value', default=50.0)
            macd_h = self._get_value(ind, 'macd', 'histogram', default=0.0)
            macd_s_form = self._get_value(ind, 'macd', 'hist_slope_forming', default=self._get_value(contrib, 'macd','hist_slope', default=0.0))
            macd_s_closed = self._get_value(ind, 'macd', 'hist_slope_closed', default=macd_s_form)
            ema_sig = self._get_value(ind, 'ema', 'signal', default='neutral')
            bb_sig = self._get_value(ind, 'bollinger', 'signal', default='neutral')
            bb_pos = self._get_value(ind, 'bollinger', 'position', default=0.5)
            kc_sig = self._get_value(ind, 'keltner', 'signal', default='within')
            st_tr = self._get_value(ind, 'supertrend', 'trend', default='neutral')
            imp = self._get_value(ind, 'impulse', 'state', default='blue')
            oi_ctx = self._get_value(ind, 'oi', 'signal', default='neutral')

            # Price Action quick summary
            pa = "neutral"
            try:
                df = getattr(self, 'current_df', None)
                if df is not None and len(df) >= 3:
                    last3 = df.tail(3)
                    lows = pd.to_numeric(last3['low'], errors='coerce').to_numpy(dtype=float)
                    highs = pd.to_numeric(last3['high'], errors='coerce').to_numpy(dtype=float)
                    if np.all(np.diff(lows) > 0):
                        pa = "higher-lows"
                    elif np.all(np.diff(highs) < 0):
                        pa = "lower-highs"
            except Exception:
                pass

            # Pattern
            pat = contrib.get('pattern_top', {}) if isinstance(contrib, dict) else {}
            pat_name = str(pat.get('name','NONE')).replace('CDL','').replace('_',' ').title()
            pat_sig  = str(pat.get('signal','NEUTRAL')).title()

            # Evidence (explicit tag [recent/long])
            ev = signal_result.get('evidence', {}) or {}
            hw = ev.get('historical_winrate', {}) if isinstance(ev, dict) else {}
            p_long = hw.get('p_long');  n_long = int(hw.get('n_long', 0))
            p_recent = hw.get('p_recent'); n_recent = int(hw.get('n_recent', 0))

            hist_suffix = ""
            # Prefer long if well-sampled; else use recent if available
            if isinstance(p_long, (int, float)) and n_long >= 100:
                hist_suffix = f" and historical win-rate {float(p_long):.0f}% (n={n_long}) [long]"
            elif isinstance(p_recent, (int, float)) and n_recent > 0:
                hist_suffix = f" and historical win-rate {float(p_recent):.0f}% (n={n_recent}) [recent]"

            return (
                f"Next 5m can be {side_simple} because RSI is {float(rsi_v):.1f}, "
                f"MACD hist={float(macd_h):+.4f} (slope_form={float(macd_s_form):+.3f}, slope_closed={float(macd_s_closed):+.3f}), "
                f"EMA stack is {str(ema_sig)}, Bollinger is {str(bb_sig)} (pos={float(bb_pos):.2f}), "
                f"Keltner is {str(kc_sig)}, Supertrend is {str(st_tr)}, Impulse is {str(imp)}, "
                f"OI context is {str(oi_ctx)}, Price Action is {str(pa)}, previous pattern is {pat_name} ({pat_sig})"
                f"{hist_suffix}"
            ).strip()
        except Exception:
            return "Next 5m explainer unavailable (insufficient context)."





    def _prev_day_hilo(self, df: pd.DataFrame) -> tuple[Optional[float], Optional[float]]:
        try:
            if df is None or df.empty:
                return (None, None)
            last_day = df.index[-1].date()
            prev = df[df.index.date < last_day]
            if prev.empty:
                return (None, None)
            day = prev.index[-1].date()
            prev_day = prev[prev.index.date == day]
            return (float(prev_day['high'].max()), float(prev_day['low'].min()))
        except Exception:
            return (None, None)

    def _last_swing_levels(self, df: pd.DataFrame, lookback: int = 20) -> tuple[Optional[float], Optional[float]]:
        try:
            if df is None or len(df) < 3: return (None, None)
            w = df.tail(lookback)
            hi = float(w['high'].iloc[-3:-1].max())
            lo = float(w['low'].iloc[-3:-1].min())
            return (hi, lo)
        except Exception:
            return (None, None)

    def _bps(self, a: float, b: float) -> float:
        try:
            return abs((a - b) / max(1e-9, b)) * 10_000.0
        except Exception:
            return float('inf')



    def detect_pivot_swipe(self, df: pd.DataFrame, indicators: dict, cfg) -> dict:
        """Detect last closed bar swipe & reclaim vs PDH/PDL or last swing."""
        out = {'detected': False}
        try:
            if df is None or len(df) < 5 or not getattr(cfg, 'enable_pivot_swipe', True):
                return out
            bps_tol = float(getattr(cfg, 'pivot_swipe_bps_tolerance', 6.0))
            levels = []
            pdh, pdl = self._prev_day_hilo(df)
            if 'PDH' in getattr(cfg, 'pivot_swipe_levels', []) and pdh:
                levels.append(('PDH', pdh))
            if 'PDL' in getattr(cfg, 'pivot_swipe_levels', []) and pdl:
                levels.append(('PDL', pdl))
            if 'SWING_5m' in getattr(cfg, 'pivot_swipe_levels', []):
                sh, sl = self._last_swing_levels(df, 20)
                if sh: levels.append(('SWING_H', sh))
                if sl: levels.append(('SWING_L', sl))
            
            if not levels: return out

            last = df.iloc[-1]
            for name, lvl in levels:
                # Swipe up: high pierces level by tol bps, but close < lvl (reclaim inside)
                if last['high'] > lvl and self._bps(last['high'], lvl) <= bps_tol and last['close'] < lvl:
                    wick = float(last['high'] - max(last['open'], last['close']))
                    out = {'detected': True, 'direction': 'SHORT', 'level_name': name, 'level': float(lvl),
                        'quality': {'wick_size': wick}}
                    logger.info("[SWIPE] SHORT @ %s level=%.2f wick=%.2f", name, lvl, wick)
                    break
                # Swipe down: low pierces level by tol bps, but close > lvl
                if last['low'] < lvl and self._bps(last['low'], lvl) <= bps_tol and last['close'] > lvl:
                    wick = float(min(last['open'], last['close']) - last['low'])
                    out = {'detected': True, 'direction': 'LONG', 'level_name': name, 'level': float(lvl),
                        'quality': {'wick_size': wick}}
                    logger.info("[SWIPE] LONG @ %s level=%.2f wick=%.2f", name, lvl, wick)
                    break
            return out
        except Exception:
            return out



    def detect_imbalance_structure(self, df: pd.DataFrame, indicators: dict, cfg) -> dict:
        """
        3-candle imbalance (C1,C2,C3) + EMA(20/50) widening with reason logging on miss.
        Returns:
        - {'detected': True, 'direction': 'LONG'|'SHORT', 'quality': {...}}
        - {'detected': False, 'why': '<reason>'} when not detected
        """
        out = {'detected': False}
        try:
            # Guards
            if df is None or len(df) < 3:
                out['why'] = "insufficient bars (<3)"
                logger.info("[IMB][MISS] %s", out['why'])
                return out
            if not getattr(cfg, 'enable_imbalance_structure', True):
                out['why'] = "disabled in config"
                logger.info("[IMB][MISS] %s", out['why'])
                return out

            # Last three closed 5m bars
            c1, c2, c3 = df.iloc[-3], df.iloc[-2], df.iloc[-1]

            # C2 body strength
            body = abs(float(c2['close']) - float(c2['open']))
            total = max(1e-9, float(c2['high']) - float(c2['low']))
            body_ratio = float(body / total)
            body_ratio_min = float(getattr(cfg, 'imbalance_c2_min_body_ratio', 0.60))
            if body_ratio < body_ratio_min:
                out['why'] = f"C2 body {body_ratio:.2f} < {body_ratio_min:.2f}"
                logger.info("[IMB][MISS] %s", out['why'])
                return out

            # C1–C3 displacement (gap)
            bull = float(c1['high']) < float(c3['low'])
            bear = float(c1['low']) > float(c3['high'])
            if not (bull or bear):
                out['why'] = "no C1–C3 displacement (gap)"
                logger.info("[IMB][MISS] %s", out['why'])
                return out

            # EMA 20/50 widening (bps)
            widening_ok = True
            spread_bps = None
            try:
                p = pd.to_numeric(df['close'], errors='coerce').ffill()
                pair = getattr(cfg, 'ema_widen_pair', (20, 50))
                ema20 = p.ewm(span=int(pair[0]), adjust=False).mean()
                ema50 = p.ewm(span=int(pair[1]), adjust=False).mean()
                a, b = float(ema20.iloc[-1]), float(ema50.iloc[-1])
                # basis points relative to EMA50 for stability
                spread_bps = self._bps(a, b)
                widen_min = float(getattr(cfg, 'ema_widen_min_bps', 8.0))
                widening_ok = spread_bps >= widen_min
                if not widening_ok:
                    out['why'] = f"EMA widen {spread_bps:.1f} bps < {widen_min:.1f} bps"
            except Exception:
                # If EMA calc fails, don't block detection just because of EMA
                widening_ok = True

            if not widening_ok:
                logger.info("[IMB][MISS] %s", out.get('why', 'EMA widening below minimum'))
                return out

            # Detected
            direction = 'LONG' if bull else 'SHORT'
            out = {
                'detected': True,
                'direction': direction,
                'quality': {
                    'c2_body_ratio': body_ratio,
                    'ema_widen_bps': float(spread_bps) if spread_bps is not None else None
                }
            }
            logger.info("[IMB] %s c2_body=%.2f widen_bps=%s",
                        direction, body_ratio,
                        f"{spread_bps:.1f}" if spread_bps is not None else "N/A")
            return out

        except Exception as e:
            out['why'] = f"error: {e}"
            logger.debug("[IMB][MISS] %s", out['why'])
            return out



    def _build_nm_features(self, micro: Dict[str, float], imb_blend: float, slp_avg: float, z_sig: float) -> np.ndarray:
        """
        Construct micro-only feature vector (length 10), NaN/Inf safe:
        [imb, imbS, slp_avg, slpS, momΔ, stdΔ, drift, nL/100, nS/50, z_sig]
        """
        try:
            imbS = float(micro.get('imbalance_short', 0.0) or 0.0)
            slpS = float(micro.get('slope_short', 0.0) or 0.0)
            std_d = float(micro.get('std_dltp', 0.0) or 0.0)
            drift = float(micro.get('vwap_drift_pct', 0.0) or 0.0) / 100.0
            nL = float(micro.get('n', 0) or 0.0)
            nS = float(micro.get('n_short', 0) or 0.0)
            mom = float(micro.get('momentum_delta', 0.0) or 0.0)
            x = np.array([imb_blend, imbS, slp_avg, slpS, mom, std_d, drift, nL/100.0, nS/50.0, float(z_sig)], dtype=float)
            x[~np.isfinite(x)] = 0.0
            return x
        except Exception:
            return np.zeros(10, dtype=float)

    def remember_nm_example(self, next_bar_time, x: np.ndarray, p_ml: float):
        try:
            self._nm_pending[next_bar_time] = {'x': np.asarray(x, dtype=float), 'p_ml': float(p_ml)}
            logger.info("[NM-ML] stored example for %s | p_ml=%.3f", str(next_bar_time), float(p_ml))
        except Exception as e:
            logger.debug("[NM-ML] remember skipped: %s", e)



    def analyze_next_minute(
        self, 
        indicators_5m: Dict, 
        indicators_15m: Optional[Dict], 
        micro: Dict[str, float], 
        micro_context_score: float, 
        session: str
    ) -> Dict:
        """
        Predict next 1m candle using dual micro windows (short+long) and 5m context.
        High-visibility and sanitized; includes:
        - micro conflict arbitration (boundary flip, slope dominates),
        - volatility shock visibility and binomial significance,
        - adaptive threshold with sign-aware extremes,
        - sign-preserving reversal penalty,
        - drift guard (display neutralization only),
        - Gate‑1 with magnitude OR significance, plus edge/slope/accel assists,
        - final classification with clear OUT logs.
        """
        try:
            import numpy as np
            import pandas as pd

            def _flt(x, d=0.0):
                try:
                    v = float(x)
                    return v if np.isfinite(v) else float(d)
                except Exception:
                    return float(d)

            def _clip(v, lo, hi):
                return float(max(lo, min(hi, float(v))))

            # --- micro (dual window) ---
            imbL = _flt(micro.get('imbalance', 0.0))
            imbS = _flt(micro.get('imbalance_short', 0.0))
            slpL = _flt(micro.get('slope', 0.0))
            slpS = _flt(micro.get('slope_short', 0.0))
            nL   = int(micro.get('n', 0) or 0)
            nS   = int(micro.get('n_short', 0) or 0)
            std_d = _flt(micro.get('std_dltp', 0.0))
            drift = _flt(micro.get('vwap_drift_pct', 0.0))
            ctx   = _flt(micro_context_score, 0.5)

            # 5m context (for priors/guards only)
            macd_closed = _flt(self._get_value(indicators_5m, 'macd', 'hist_slope_closed', default=0.0))
            rsi5        = _flt(self._get_value(indicators_5m, 'rsi', 'value', default=50.0), 50.0)
            bb_pos      = _flt(self._get_value(indicators_5m, 'bollinger', 'position', default=0.5), 0.5)

            # Weights (picked from config)
            w_macd = _flt(getattr(self.config, 'next_minute_macd_prior_weight', 2.5), 2.5)
            w_imb  = _flt(getattr(self.config, 'next_minute_imb_weight', 55.0), 55.0)
            w_slp  = _flt(getattr(self.config, 'next_minute_slope_weight', 10.0), 10.0)
            w_rsi  = _flt(getattr(self.config, 'next_minute_rsi_prior_weight', 0.05), 0.05)

            # Blend windows
            wS = _flt(getattr(self.config, 'next_minute_imb_short_weight', 0.60), 0.60) if nS >= int(getattr(self.config, 'micro_short_min_ticks_1m', 12)) else 0.0
            wL = _flt(getattr(self.config, 'next_minute_imb_long_weight', 0.40), 0.40)
            w_sum = max(1e-9, wS + wL)

            imb = (wS*imbS + wL*imbL) / w_sum
            slp_avg = ((wS*slpS + wL*slpL) / w_sum)
            slp_aligned = np.sign(imb) * max(0.0, np.sign(imb) * slp_avg)

            # Fresh-ticks guard
            minL = int(getattr(self.config, 'micro_min_ticks_1m', 30))
            minS = int(getattr(self.config, 'micro_short_min_ticks_1m', 12))
            if nL < minL:
                logger.info("[NEXT-1m][SAN] NEUTRAL: insufficient long-window ticks (n=%d<%d)", nL, minL)
                return {'composite_signal':'NEUTRAL','confidence':30.0,'forecast_color':'NEUTRAL','forecast_prob':50.0,'why':"Insufficient long-window ticks"}

            logger.info("[NEXT-1m][SAN][CTX] imbL=%+.2f imbS=%+.2f → imb*=%+.2f | slpL=%+.3f slpS=%+.3f → slp_avg=%+.3f | stdΔ=%.5f drift=%+.3f%% | bb=%.2f ctx5m=%.2f",
                        imbL, imbS, imb, slpL, slpS, slp_avg, std_d, drift, bb_pos, ctx)

            # --- MICRO CONFLICTS ---

            # 1) Boundary flip: short window dominates when it clearly flips vs long
            try:
                mom_delta = _flt(micro.get('momentum_delta', 0.0), 0.0)
                flip_ratio = _flt(getattr(self.config, 'next_minute_boundary_flip_ratio', 1.40), 1.40)
                flip_mom   = _flt(getattr(self.config, 'next_minute_boundary_flip_mom_delta', 0.25), 0.25)
                if np.sign(imbS) != 0 and np.sign(imbL) != 0 and np.sign(imbS) != np.sign(imbL):
                    if (abs(imbS) >= flip_ratio*abs(imbL)) or (abs(mom_delta) >= flip_mom):
                        old = imb
                        imb = imbS
                        slp_avg = slpS if wS > 0 else slp_avg
                        slp_aligned = np.sign(imb) * max(0.0, np.sign(imb)*slp_avg)
                        logger.info("[NEXT-1m][SAN][BFLIP] short dominates (|imbS|≥%.2fx|imbL| or |momΔ|≥%.2f) → imb*: %.2f→%.2f",
                                    flip_ratio, flip_mom, old, imb)
            except Exception as _e:
                logger.debug("[NEXT-1m][SAN][BFLIP] skipped: %s", _e)

            # 2) Slope dominates weak, opposing count-imbalance
            try:
                smin = _flt(getattr(self.config, 'next_minute_slope_conflict_min', 0.45), 0.45)
                icap = _flt(getattr(self.config, 'next_minute_imb_conflict_cap', 0.20), 0.20)
                repl = _flt(getattr(self.config, 'next_minute_imb_conflict_replacement', 0.15), 0.15)
                slope_strong = (abs(slpL) >= smin and abs(slpS) >= smin and np.sign(slpL) == np.sign(slpS))
                signs_opp = (np.sign(imb) != 0) and (np.sign(imb) != np.sign(slp_avg))
                weak_imb = abs(imb) <= icap
                if slope_strong and signs_opp and weak_imb:
                    old = imb
                    imb = np.sign(slp_avg) * min(repl, max(abs(old), 0.12))
                    slp_aligned = np.sign(imb) * max(0.0, np.sign(imb) * slp_avg)
                    logger.info("[NEXT-1m][SAN][CONFLICT] slope dominates weak opposing imb → imb*: %.2f→%.2f | slp_avg=%.3f", old, imb, slp_avg)
            except Exception as _e:
                logger.debug("[NEXT-1m][SAN][CONFLICT] skipped: %s", _e)

            # 3) Volatility shock visibility
            vol_ref = (self._nm_std_ref or std_d) if std_d > 0 else 0.0
            shock_mult = _flt(getattr(self.config, 'next_minute_vol_shock_mult', 1.60), 1.60)
            vol_shock = (std_d > 0.0 and vol_ref > 0.0 and std_d >= shock_mult * vol_ref)
            if vol_shock:
                logger.info("[NEXT-1m][SAN][VSHOCK] stdΔ shock: current=%.5f ref=%.5f (x%.2f)", std_d, vol_ref, (std_d/vol_ref))

            # 4) Significance of imbalance (binomial z)
            moves_total = int(micro.get('moves_total', 0) or 0)
            if moves_total <= 0:
                moves_total = max(10, int(0.8 * nL))
            z_sig = abs(imb) * (moves_total ** 0.5)
            z_min_base = _flt(getattr(self.config, 'next_minute_signif_z_min_base', 0.90), 0.90)
            z_min_trend = _flt(getattr(self.config, 'next_minute_signif_z_min_trend', 0.80), 0.80)
            z_min = z_min_trend if ctx >= 0.80 else z_min_base
            logger.info("[NEXT-1m][SAN][SIGNIF] z=%.2f | moves=%d | z_min=%.2f | vol_shock=%s", z_sig, moves_total, z_min, vol_shock)


            
            # --- micro-only model probability (optional) ---
            p_ml = None
            x_ml = None
            try:
                if self.nm_model is not None:
                    x_ml = self._build_nm_features(micro, imb, slp_avg, z_sig)
                    p_ml = float(self.nm_model.predict_proba(x_ml))
                    logger.info("[NM-ML] p_ml=%.3f | features=%s", p_ml, np.round(x_ml, 3))
            except Exception as e:
                logger.debug("[NM-ML] predict skipped: %s", e)
                p_ml = None


            # --- adaptive threshold (sign-aware extremes) ---
            base_th   = _flt(getattr(self.config, 'micro_imbalance_min', 0.30), 0.30)
            noise_k   = _flt(getattr(self.config, 'micro_noise_sigma_mult', 2.0), 2.0)
            need_chk  = int(getattr(self.config, 'micro_persistence_min_checks', 1))

            imb_th = base_th

            # Noise-aware adjust with EMA ref
            if std_d > 0.0:
                if self._nm_std_ref is None:
                    self._nm_std_ref = std_d
                else:
                    self._nm_std_ref = 0.9*self._nm_std_ref + 0.1*std_d
                if std_d <= 0.90 * self._nm_std_ref:
                    imb_th = max(0.08, imb_th - 0.02)
                elif std_d >= 1.25 * self._nm_std_ref:
                    imb_th = min(0.45, imb_th + 0.02)

            # Sign-aware extreme adjustment
            hi = _flt(getattr(self.config, 'next_minute_mtf_extreme_pos_hi', 0.95), 0.95)
            lo = _flt(getattr(self.config, 'next_minute_mtf_extreme_pos_lo', 0.05), 0.05)
            relax_pp  = _flt(getattr(self.config, 'next_minute_extreme_relax_pp', 0.12), 0.12)
            harden_pp = _flt(getattr(self.config, 'next_minute_extreme_harden_pp', 0.05), 0.05)

            near_top = (bb_pos >= hi)
            near_bot = (bb_pos <= lo)
            imb_sign = np.sign(imb) if imb != 0 else np.sign(slp_aligned)

            if near_top or near_bot:
                old_th = imb_th
                if near_top and imb_sign < 0:
                    imb_th = max(0.08, imb_th - relax_pp)
                    logger.info("[NEXT-1m][SAN][ADAPT] extreme adjust: top→SELL relax %.2f→%.2f", old_th, imb_th)
                elif near_bot and imb_sign > 0:
                    imb_th = max(0.08, imb_th - relax_pp)
                    logger.info("[NEXT-1m][SAN][ADAPT] extreme adjust: bottom→BUY relax %.2f→%.2f", old_th, imb_th)
                else:
                    imb_th = max(imb_th, base_th + harden_pp)
                    logger.info("[NEXT-1m][SAN][ADAPT] extreme adjust: into extreme harden %.2f→%.2f", old_th, imb_th)

            logger.info("[NEXT-1m][SAN][ADAPT] imb_th=%.2f (base=%.2f, stdΔ=%.5f ref=%.5f, ctx=%.2f, bb=%.2f)",
                        imb_th, base_th, std_d, (self._nm_std_ref or 0.0), ctx, bb_pos)

            # --- forecast-only prior (for probability display) ---
            prior = 50.0
            prior += w_macd * _clip(macd_closed / 0.20, -1.0, 1.0)
            prior += w_rsi  * (rsi5 - 50.0)
            prior += w_imb  * _clip(imb, -1.0, 1.0)
            prior += w_slp  * _clip(slp_aligned, -1.0, 1.0)



            # Short/long disagreement penalty and decel (sign‑preserving, lighter; require both)
            try:
                if nS >= minS:
                    disagree = (np.sign(imbS) != 0 and np.sign(imbL) != 0 and np.sign(imbS) != np.sign(imbL)) \
                            or (np.sign(slpS) != 0 and np.sign(slpL) != 0 and np.sign(slpS) != np.sign(slpL))
                    decel = (float(micro.get('momentum_delta', 0.0) or 0.0) < -0.25)
                    if disagree and decel:
                        delta = abs(prior - 50.0)
                        # Lighter, multiplicative trim; no large fixed subtractor
                        shrink = float(getattr(self.config, 'rev_penalty_shrink_factor', 0.60))  # 60% of delta
                        mag = delta * shrink
                        sign = 1.0 if imb >= 0 else -1.0
                        
                        # If statistically significant, don't let display collapse to 50. Keep a small floor away from 50.
                        z_floor_pp = 0.0
                        try:
                            if z_sig >= z_min:
                                z_floor_pp = float(getattr(self.config, 'rev_penalty_min_floor_pp', 2.0))  # keep ≥2 pp off 50
                        except Exception:
                            z_floor_pp = 0.0
                        
                        old_prior = prior
                        # Respect the sign; limit to at least the floor if significant
                        prior = 50.0 + sign * max(mag, z_floor_pp)
                        
                        logger.info("[NEXT-1m][SANREV2] disagree&decel → trim Δ=%.1f shrink=%.2f → mag=%.1fpp (floor=%.1fpp) sign=%s old=%.1f new=%.1f",
                                    delta, shrink, mag, z_floor_pp, ("BUY" if sign>0 else "SELL"), old_prior, prior)
            except Exception as _e:
                logger.debug("[NEXT-1m][SANREV2] skip: %s", _e)

            color_prob = _clip(prior, 10.0, 90.0)



            # Blend: rule prob vs model prob (display only; direction remains micro-led)
            try:
                if p_ml is not None and getattr(self.config, 'enable_next_minute_ml', True):
                    alpha = float(getattr(self.config, 'nm_ml_blend_alpha', 0.70))
                    p_rule = float(color_prob) / 100.0
                    p_final = float(alpha * p_rule + (1.0 - alpha) * p_ml)
                    # Optional assist (disabled by default)
                    if getattr(self.config, 'nm_ml_assist_enabled', False):
                        conf = float(getattr(self.config, 'nm_ml_assist_confident', 0.62))
                        pp = float(getattr(self.config, 'nm_ml_assist_pp', 8.0))
                        if p_ml >= conf:
                            color_prob = min(90.0, 100.0 * p_final + pp)
                        elif p_ml <= 1.0 - conf:
                            color_prob = max(10.0, 100.0 * p_final - pp)
                        else:
                            color_prob = 100.0 * p_final
                    else:
                        color_prob = 100.0 * p_final
                    logger.info("[NM-ML] blended prob=%.1f%% (rule=%.1f%%, ml=%.1f%%, α=%.2f)", color_prob, p_rule*100.0, (p_ml*100.0), alpha)
            except Exception as e:
                logger.debug("[NM-ML] blend skipped: %s", e)


            # Dynamic neutral band: base from config
            base_nb = _flt(getattr(self.config, 'next_minute_forecast_neutral_band_pp',
                                getattr(self.config, 'next_minute_forecast_neutral_band', 0.02)), 0.02)

            # Dynamic band: if z is clearly above threshold, shrink/disable neutral clamp
            if z_sig >= 1.30 * z_min:
                nb_pp = 0.00
            elif z_sig >= 1.10 * z_min:
                nb_pp = max(0.005, base_nb * 0.5)
            else:
                nb_pp = base_nb

            forecast_dir = "BUY" if color_prob >= 50.0 else "SELL"
            if nb_pp > 0.0 and abs(color_prob - 50.0) < (nb_pp * 100.0):
                logger.info("[NEXT-1m][SAN][FORECAST] within dynamic neutral band ±%.0fpp (z=%.2f, z_min=%.2f) → display NEUTRAL",
                            nb_pp*100.0, z_sig, z_min)
                forecast_dir = "NEUTRAL"

            logger.info("[NEXT-1m][SAN][FORECAST] dir=%s prob=%.1f%% | priors: macd=%+.3f(w=%.1f) rsi=%.1f(w=%.2f) imb=%+.2f(w=%.1f) slp=%+.3f(w=%.1f)",
                        forecast_dir, color_prob, macd_closed, w_macd, rsi5, w_rsi, imb, w_imb, slp_aligned, w_slp)

            

            # Drift guard (display-only neutralization of forecast)
            drift_guard = _flt(getattr(self.config, 'next_minute_drift_guard_pct', 0.01), 0.01)
            if forecast_dir == "SELL" and drift >= drift_guard and rsi5 >= 50.0:
                logger.info("[NEXT-1m][SAN][GUARD] Neutralize SELL (up-drift %.3f%%, rsi5=%.1f≥50)", drift, rsi5)
                return {'composite_signal': 'NEUTRAL', 'confidence': 30.0,
                        'forecast_color': 'NEUTRAL', 'forecast_prob': color_prob,
                        'why': "Drift guard neutralized SELL in up-drift"}
            if forecast_dir == "BUY" and drift <= -drift_guard and rsi5 <= 50.0:
                logger.info("[NEXT-1m][SAN][GUARD] Neutralize BUY (down-drift %.3f%%, rsi5=%.1f≤50)", drift, rsi5)
                return {'composite_signal': 'NEUTRAL', 'confidence': 30.0,
                        'forecast_color': 'NEUTRAL', 'forecast_prob': color_prob,
                        'why': "Drift guard neutralized BUY in down-drift"}

            # --- Gate‑1: persistence with magnitude OR significance; then assists ---
            self._nm_last_side = 0 if not hasattr(self, '_nm_last_side') else self._nm_last_side
            self._nm_side_streak = 0 if not hasattr(self, '_nm_side_streak') else self._nm_side_streak

            side_now = 1 if imb > 0 else (-1 if imb < 0 else 0)
            noise_ok = True
            if std_d > 0.0:
                noise_ok = std_d <= (noise_k * (self._nm_std_ref or std_d))

            if self._nm_side_streak < need_chk:
                imb_pass = (abs(imb) >= imb_th and side_now != 0)
                signif_pass = (z_sig >= z_min and side_now != 0)
                if noise_ok and (imb_pass or signif_pass):
                    if signif_pass and not imb_pass:
                        logger.info("[NEXT-1m][SAN][G1] Passed by significance (z=%.2f ≥ %.2f)", z_sig, z_min)
                    if self._nm_last_side == side_now:
                        self._nm_side_streak += 1
                    else:
                        self._nm_last_side = side_now
                        self._nm_side_streak = 1
                else:
                    self._nm_last_side = 0
                    self._nm_side_streak = 0

            # Edge override (extreme fade using short window)
            edge_imb = _flt(getattr(self.config, 'next_minute_edge_override_short_imb', 0.20), 0.20)
            edge_slp = _flt(getattr(self.config, 'next_minute_edge_override_short_slp', 0.35), 0.35)
            if self._nm_side_streak < need_chk and nS >= minS:
                if near_top and (imbS <= -edge_imb) and (slpS <= -edge_slp):
                    self._nm_last_side = -1
                    self._nm_side_streak = need_chk
                    logger.info("[NEXT-1m][SAN][G1] Edge override: TOP fade → SELL (imbS=%.2f slpS=%.3f)", imbS, slpS)
                elif near_bot and (imbS >= +edge_imb) and (slpS >= +edge_slp):
                    self._nm_last_side = +1
                    self._nm_side_streak = need_chk
                    logger.info("[NEXT-1m][SAN][G1] Edge override: BOTTOM fade → BUY (imbS=%.2f slpS=%.3f)", imbS, slpS)

            # Slope‑assist near threshold
            if self._nm_side_streak < need_chk:
                try:
                    slope_assist_min = _flt(getattr(self.config, 'next_minute_slope_assist_min', 0.35), 0.35)
                    assist = (
                        noise_ok and
                        np.sign(imb) != 0 and
                        np.sign(imb) == np.sign(slp_aligned) and
                        abs(slp_aligned) >= slope_assist_min and
                        abs(imb) >= max(0.60*imb_th, 0.12)
                    )
                    if assist:
                        self._nm_last_side = 1 if imb > 0 else -1
                        self._nm_side_streak = need_chk
                        logger.info("[NEXT-1m][SAN][G1] Allowed: slope‑assist (imb=%.2f th=%.2f slp=%.3f)", imb, imb_th, slp_aligned)
                except Exception as e:
                    logger.debug("[NEXT-1m][SAN][G1] slope-assist skipped: %s", e)

            if self._nm_side_streak < need_chk:
                logger.info("[NEXT-1m][SAN][G1] NEUTRAL: micro not persistent (imb=%.2f<th=%.2f or streak=%d/%d, noise_ok=%s)",
                            imb, imb_th, self._nm_side_streak, need_chk, noise_ok)
                return {'composite_signal': 'NEUTRAL', 'confidence': 30.0,
                        'forecast_color': 'NEUTRAL', 'forecast_prob': color_prob,
                        'why': "Next 1m NEUTRAL: micro not persistent"}

            # Final classification (direction from micro, probability from prior)
            direction = "BUY" if imb > 0 else "SELL"
            conf = min(85.0, 40.0 + 30.0 * min(1.0, abs(imb)) + 15.0 * min(1.0, abs(slp_aligned)))
            if not noise_ok:
                conf = 35.0

            why = (f"micro: imb={imb:+.2f}, slope={slp_aligned:+.3f}, stdΔ={std_d:.5f}, drift={drift:+.2f}%; "
                f"gates: imb_th={imb_th:.2f}")
            logger.info("[NEXT-1m][SAN][OUT] side=%s | conf=%.1f%% | prob=%.1f%% | %s",
                        direction, conf, color_prob, why)

      


            return {
                'composite_signal': direction,
                'confidence': conf,
                'forecast_color': forecast_dir,
                'forecast_prob': color_prob,
                'why': why,
                'nm_features': x_ml.tolist() if isinstance(x_ml, np.ndarray) else None,
                'model_p': float(p_ml) if p_ml is not None else None
            }


        except Exception as e:
            logger.error("[NEXT-1m][SAN] error: %s", e, exc_info=True)
            return {'composite_signal': 'NEUTRAL', 'confidence': 0.0,
                    'forecast_color': 'NEUTRAL', 'forecast_prob': 50.0, 'why': 'error'}





    def _apply_oi_sd_nudges(self, weighted_score: float, indicators: dict, contributions: dict, scalping_signals: list) -> float:
        try:
            # OI context (confirmation-only; bounded)
            if getattr(self.config, 'enable_oi_integration', True):
                oi_ind = indicators.get('oi', {}) if isinstance(indicators, dict) else {}
                oi_sig = str(oi_ind.get('signal', 'neutral')).lower()
                oi_pct = float(oi_ind.get('oi_change_pct', 0.0) or 0.0)
                if not np.isfinite(oi_pct):
                    oi_pct = 0.0
                min_pct = float(getattr(self.config, 'oi_min_change_pct', 0.10))
                ctx_boost = float(getattr(self.config, 'oi_context_boost', 0.03))
                if abs(oi_pct) >= min_pct:
                    if oi_sig in ('long_build_up', 'short_covering'):
                        old = weighted_score
                        bump = ctx_boost if oi_sig == 'long_build_up' else (ctx_boost * 0.5)
                        weighted_score += bump
                        scalping_signals.append(f"OI context (+): {oi_sig} (ΔOI%={oi_pct:.2f})")
                        logger.info(f"[OI] Bullish context → score {old:+.3f} → {weighted_score:+.3f} (ΔOI%={oi_pct:.2f})")
                        
                    elif oi_sig in ('short_build_up', 'long_unwinding'):
                        old = weighted_score
                        bump = ctx_boost if oi_sig == 'short_build_up' else (ctx_boost * 0.5)
                        weighted_score -= bump
                        scalping_signals.append(f"OI context (-): {oi_sig} (ΔOI%={oi_pct:.2f})")
                        logger.info(f"[OI] Bearish context → score {old:+.3f} → {weighted_score:+.3f} (ΔOI%={oi_pct:.2f})")
                        
                contributions['oi'] = {
                    'signal': oi_ind.get('signal', 'neutral'),
                    'oi': oi_ind.get('oi', 0),
                    'oi_change': oi_ind.get('oi_change', 0),
                    'oi_change_pct': oi_pct,
                    'contribution': 0.0
                }
            # Supply/Demand tiny nudge/dampener
            if getattr(self.config, 'enable_supply_demand_integration', True):
                sd = contributions.get('supply_demand', {}) if isinstance(contributions, dict) else {}
                at_supply = bool(sd.get('at_supply', False))
                at_demand = bool(sd.get('at_demand', False))
                sd_boost = float(getattr(self.config, 'sd_context_boost', 0.03))
                if at_demand and weighted_score > 0:
                    old = weighted_score
                    weighted_score += sd_boost
                    scalping_signals.append("S/D context (+): at demand")
                    logger.info("[S/D] Demand zone boost → %.3f→%.3f", old, weighted_score)
                    
                elif at_supply and weighted_score < 0:
                    old = weighted_score
                    weighted_score -= sd_boost
                    scalping_signals.append("S/D context (-): at supply")
                    logger.info("[S/D] Supply zone boost → %.3f→%.3f", old, weighted_score)
                    
                else:
                    if at_supply and weighted_score > 0:
                        old = weighted_score
                        weighted_score -= (sd_boost * 0.5)
                        scalping_signals.append("S/D dampener: BUY into supply")
                        logger.info("[S/D] Dampener: BUY into supply → %.3f→%.3f", old, weighted_score)
                        
                    if at_demand and weighted_score < 0:
                        old = weighted_score
                        weighted_score += (sd_boost * 0.5)
                        scalping_signals.append("S/D dampener: SELL into demand")
                        logger.info("[S/D] Dampener: SELL into demand → %.3f→%.3f", old, weighted_score)
                        
        except Exception as e:
            logger.debug(f"[CTX] OI/S-D nudges skipped: {e}")
        return weighted_score

    def _apply_micro_nudge(self, weighted_score: float, contributions: dict, scalping_signals: list, session: str = "mid") -> float:
        try:
            if not getattr(self.config, 'enable_microstructure_nudge', True):
                return weighted_score
            # Only in borderline zone
            if abs(weighted_score) >= 0.06:
                return weighted_score
            # Session multiplier
            muls = getattr(self.config, 'micro_session_multipliers', {"open":0.5,"mid":1.0,"close":0.75})
            m = float(muls.get(session, 1.0))
            base = float(getattr(self.config, 'micro_base_nudge', 0.02))
            imb_th = float(getattr(self.config, 'micro_imbalance_thresh', 0.60))
            noise_k = float(getattr(self.config, 'micro_noise_sigma_mult', 1.5))
            # Pull snapshot if provider present
            snap = {}
            try:
                if hasattr(self, 'get_micro_features') and callable(self.get_micro_features):
                    snap = self.get_micro_features() or {}
            except Exception:
                snap = {}
            imb = float(snap.get('imbalance', 0.0) or 0.0)
            std_d = float(snap.get('std_dltp', 0.0) or 0.0)
            # Simple dynamic noise guard: compare to running median proxy if available in contributions
            noise_ok = True
            if std_d > 0.0:
                # Store simple EMA of std in contributions to estimate typical noise
                med = float(contributions.get('_micro_std_ref', std_d))
                contributions['_micro_std_ref'] = (0.9 * med + 0.1 * std_d) if np.isfinite(med) else std_d
                noise_ok = std_d <= (noise_k * contributions['_micro_std_ref'])
            # Apply nudge only when imbalance decisive and noise acceptable
            if noise_ok and abs(imb) >= imb_th:
                sign = np.sign(weighted_score) if weighted_score != 0 else (1 if imb > 0 else -1)
                nudge = base * m * sign
                old = weighted_score
                weighted_score += nudge
                scalping_signals.append(f"Micro tie-break: imb={imb:.2f}, stdΔ={std_d:.5f} (m={m:.2f})")
                logger.info("[MICRO] nudge=%+.3f → %.3f→%.3f | imb=%.2f stdΔ=%.5f (m=%.2f, sess=%s)",
                            nudge, old, weighted_score, imb, std_d, m, session)
                
        except Exception as e:
            logger.debug(f"[MICRO] nudge skipped: {e}")
        return weighted_score




    def _apply_evidence_nudge_and_why(self, signal_result: dict, indicators_5m: dict) -> None:
        """
        Bounded evidence-based confidence nudge + WHY string (with explicit historical win-rate [recent/long]).
        Order: compute evidence → update signal_result['evidence'] → compose WHY including the evidence.
        """
        try:
            st = getattr(self, 'setup_stats', None)
            contrib = signal_result.get('contributions', {}) or {}

            # Build minimal record for key
            rec = {
                "direction": signal_result.get('composite_signal', 'NEUTRAL'),
                "mtf_score": float(signal_result.get('mtf_score', 0.0)),
                "breadth": int(signal_result.get('active_indicators', 0)),
                "weighted_score": float(signal_result.get('weighted_score', 0.0)),
                "macd_hist_slope": float(self._get_value(contrib, 'macd', 'hist_slope', default=0.0) or 0.0),
                "rsi_value": float(self._get_value(contrib, 'rsi', 'rsi_value', default=50.0) or 50.0),
                "rsi_cross_up": bool(self._get_value(contrib, 'rsi', 'rsi_cross_up', default=False)),
                "rsi_cross_down": bool(self._get_value(contrib, 'rsi', 'rsi_cross_down', default=False)),
                "sr_room": str(self._get_value(signal_result, 'mtf_analysis', 'sr_room', default='UNKNOWN') or 'UNKNOWN'),
                "oi_signal": str(self._get_value(contrib, 'oi', 'signal', default='neutral') or 'neutral'),
                "tod": signal_result.get('session', 'mid')
            }

            # Defaults if no setup_stats
            p_recent = n_recent = p_long = n_long = 0
            if st:
                from setup_statistics import make_setup_key
                key = make_setup_key(rec)

                # Evidence params
                n_min = int(getattr(self.config, 'evidence_min_n', 30))
                n_mid = int(getattr(self.config, 'evidence_mid_n', 50))
                n_str = int(getattr(self.config, 'evidence_strong_n', 100))
                max_pp = float(getattr(self.config, 'evidence_max_nudge_pp', 5.0))
                delta_conf = float(getattr(self.config, 'evidence_conflict_delta_pp', 12.0))

                # Pull tables
                p_recent, lo_r, hi_r, n_recent = st.get(key, window="recent")
                p_long,  lo_l, hi_l, n_long  = st.get(key, window="long")

                # Nudge (bounded)
                use_recent = n_recent >= n_min
                conflict = use_recent and (n_long >= n_min) and (abs(p_recent - p_long) > (delta_conf/100.0))
                target = p_recent if use_recent else (p_long if n_long >= n_str else None)
                n = n_recent if use_recent else (n_long if n_long >= n_str else 0)

                if target is not None and n >= n_min:
                    conf = float(signal_result.get('confidence', 0.0))
                    raw_delta = (target*100.0 - 50.0)
                    scale = min(1.0, max(0.5, (n / max(1.0, n_mid))))
                    if conflict and n_recent < n_mid:
                        scale *= 0.3
                    delta_pp = max(-max_pp, min(max_pp, raw_delta * 0.10 * scale))
                    signal_result['confidence'] = max(0.0, min(100.0, conf + delta_pp))
                    logger.info("[EVIDENCE] p_recent=%.1f%%(n=%d), p_long=%.1f%%(n=%d) → conf %.1f→%.1f (Δ=%.1f pp)",
                                p_recent*100.0, n_recent, p_long*100.0, n_long, conf, signal_result['confidence'], delta_pp)
                    

                # Store evidence for WHY composition
                signal_result.setdefault('evidence', {})['historical_winrate'] = {
                    'p_recent': round(p_recent*100.0, 1), 'n_recent': int(n_recent),
                    'p_long': round(p_long*100.0, 1), 'n_long': int(n_long)
                }

            # Compose WHY AFTER evidence is attached so we can include [recent/long] explicitly
            why = self._compose_explanation_line(indicators_5m, contrib, signal_result)
            signal_result['why'] = why
            logger.info(why)
            

        except Exception as e:
            logger.debug(f"[EVIDENCE] nudge/why skipped: {e}")




    def _cap_confidence_sr_limited(self, signal_result: dict, indicators_5m: dict):
        try:
            mtf = signal_result.get('mtf_analysis', {}) or {}
            sr_room = str(mtf.get('sr_room', 'UNKNOWN'))
            if sr_room != 'LIMITED':
                return
            atr = float(self._get_value(indicators_5m, 'atr', 'value', default=0.0) or 0.0)
            price = float(indicators_5m.get('price', 0.0) or 0.0)
            sd = signal_result.get('contributions', {}).get('supply_demand', {}) or {}
            up = sd.get('nearest_up'); dn = sd.get('nearest_dn')
            dist = None
            if up and price>0:
                dist = abs(float(up) - price)
            if dn and price>0:
                d2 = abs(price - float(dn))
                dist = min(dist, d2) if dist is not None else d2
            if atr <= 0 or dist is None:
                return
            cap = float(getattr(self.config, 'sr_confidence_cap_limited', 0.65))
            override = float(getattr(self.config, 'sr_confidence_cap_override_pp', 0.70))
            conf = float(signal_result.get('confidence', 0.0))
            if dist < 0.5 * atr:
                if 'evidence' in signal_result and signal_result['evidence'].get('historical_winrate',{}).get('p_recent',0)/100.0 >= 0.72 and \
                   signal_result['evidence']['historical_winrate'].get('n_recent',0) >= 50:
                    signal_result['confidence'] = min(signal_result['confidence'], override*100.0)
                else:
                    old = conf; signal_result['confidence'] = min(conf, cap*100.0)
                logger.info("[CAP] LIMITED room near SR: conf %.1f→%.1f (ATR=%.2f, dist=%.2f)",
                            conf, signal_result['confidence'], atr, dist)
                
        except Exception as e:
            logger.debug(f"[CAP] LIMITED room cap skipped: {e}")




    async def analyze_and_generate_signal(
        self, 
        indicators_5m: Dict, 
        df_5m: pd.DataFrame,
        indicators_15m: Optional[Dict] = None,
        df_15m: Optional[pd.DataFrame] = None
    ) -> Optional[Dict]:
        """
        Generate trading signal with all enhancements.
        
        This method performs a comprehensive analysis of the given indicators and dataframes
        to produce a trading signal. It includes various checks and validations to ensure
        the signal's reliability.
        """
        
        try:
            # Input validation
            if not indicators_5m or df_5m.empty:
                logger.warning("Invalid input: empty indicators or dataframe")
                return None
            
            logger.info("=" * 50)
            log_span("Starting enhanced signal analysis...")


            # Reset last rejection for this bar
            self._last_reject = None 
            # Store dataframe for market regime detection
            self.current_df = df_5m 
            
            # Detect market session characteristics
            session_info = self.detect_session_characteristics(df_5m)

            # Cache session for scorer internals to avoid re-detection/logging
            try:
                self._cached_session_info = session_info
            except Exception:
                pass


            logger.debug(f"Session: {session_info.get('session','unknown')} | Strategy: {session_info.get('strategy','standard')}")



            # Expansion flip handling → skip relax nudges briefly
            try:

                exp_flag, exp_flip = self._detect_expansion_lite(df_5m, indicators_5m)
                # Cache expansion flag to reuse in scorer (avoids a second call and "unused" warning)
                try:
                    self._last_expansion_flag = exp_flag
                except Exception:
                    pass
                if exp_flip:
                    bars = 3
                    # Use df_5m index to preserve timezone and avoid naive/aware compare issues downstream
                    ts_ref = df_5m.index[-1] if (df_5m is not None and not df_5m.empty) else datetime.now()
                    until = ts_ref + timedelta(minutes=5 * bars)
                    setattr(self.mtf_analyzer, "_skip_relax_until", until)
                    logger.info(f"[EXP] Flip detected → skip relax nudges for next {bars} bars (until {until.strftime('%H:%M:%S')})")


                                    
            except Exception:
                pass




            # Make HTF MACD signal available to the scorer (for ST softening gate)
            try:
                self._htf_macd_sig = str(
                    indicators_15m.get('macd', {}).get('signal_type',
                    indicators_15m.get('macd', {}).get('signal', 'neutral'))
                ).lower() if indicators_15m else None
            except Exception as e:
                logger.debug(f"Error getting HTF MACD signal: {e}")
                self._htf_macd_sig = None
            logger.debug(f"HTF MACD for softening gate: {self._htf_macd_sig}")


            # Compute HTF price position and extreme flags for guards
            self._htf_price_pos = None
            self._htf_sr_room = True
            self._extreme_top = False
            self._extreme_bottom = False
            try:
                if df_15m is not None and not df_15m.empty and len(df_15m) >= 20:
                    cur = float(df_15m['close'].iloc[-1])
                    hi = float(df_15m['high'].tail(20).max())
                    lo = float(df_15m['low'].tail(20).min())
                    rng = (hi - lo) or 1e-6
                    self._htf_price_pos = max(0.0, min(1.0, (cur - lo) / rng))
                    self._extreme_top = self._htf_price_pos >= self.config.extreme_price_pos_hi
                    self._extreme_bottom = self._htf_price_pos <= self.config.extreme_price_pos_lo
                    self._htf_sr_room = (self._extreme_top is False) and (self._extreme_bottom is False)
                    logger.debug(f"HTF guards: price_pos={self._htf_price_pos}, top={self._extreme_top}, bottom={self._extreme_bottom}, sr_room={self._htf_sr_room}")
            except Exception as e:
                logger.debug(f"HTF price_position calc skipped: {e}")

            rsi5 = float(indicators_5m.get('rsi', {}).get('value', 50.0))
            rsi15 = float(indicators_15m.get('rsi', {}).get('value', 50.0)) if indicators_15m else 50.0
            self._extreme_overbought = (rsi5 >= self.config.extreme_rsi_5m) or (rsi15 >= self.config.extreme_rsi_15m)
            self._extreme_oversold = (rsi5 <= (100 - self.config.extreme_rsi_5m)) or (rsi15 <= (100 - self.config.extreme_rsi_15m))
            logger.debug(f"RSI extremes: rsi5={rsi5:.2f}, rsi15={rsi15:.2f}, overbought={self._extreme_overbought}, oversold={self._extreme_oversold}")


            # 1. Calculate weighted signal
            signal_result = self._calculate_weighted_signal(indicators_5m)



            # ONLY microstructure tie-break; OI/S-D already applied inside scorer to avoid double-nudging
            try:
                sess = session_info.get('session', 'mid')
                ws_before = float(signal_result['weighted_score'])
                weighted_score = self._apply_micro_nudge(ws_before, signal_result['contributions'], signal_result['scalping_signals'], session=sess)
                signal_result['weighted_score'] = weighted_score
                if abs(weighted_score - ws_before) > 1e-9:
                    logger.info("[CTX] Micro nudge applied post-scorer: %.3f→%.3f", ws_before, weighted_score)
            except Exception as e:
                logger.debug(f"[CTX] micro nudge skipped: {e}")




            # Nuance: in ranging sessions, apply a small directional haircut if a counter-pattern was ignored
            try:
                if session_info.get('session') == 'ranging':
                    contrib = signal_result.get('contributions', {})
                    ignored = contrib.get('pattern_ignored', {})
                    if ignored:
                        pat_sig = str(ignored.get('signal','neutral')).upper()
                        ws = float(signal_result.get('weighted_score', 0.0))
                        mr = signal_result.get('market_regime', 'NORMAL')
                        # Only haircut when the ignored pattern contradicts the current bias and we are not in a strong-trend regime
                        if ws > 0 and pat_sig == 'SHORT' and mr == 'NORMAL':
                            old = ws
                            ws -= 0.04
                            signal_result['weighted_score'] = ws
                            signal_result.setdefault('scalping_signals', []).append("Ranging nuance: SHORT pattern ignored → BUY haircut")
                            logger.info(f"Ranging nuance applied: SHORT pattern ignored → BUY haircut (score {old:.3f} → {ws:.3f})")
                        elif ws < 0 and pat_sig == 'LONG' and mr == 'NORMAL':
                            old = ws
                            ws += 0.04
                            signal_result['weighted_score'] = ws
                            signal_result.setdefault('scalping_signals', []).append("Ranging nuance: LONG pattern ignored → SELL haircut")
                            logger.info(f"Ranging nuance applied: LONG pattern ignored → SELL haircut (score {old:.3f} → {ws:.3f})")
                        # Reclassify after haircut using same thresholds as scorer
                        ws = signal_result['weighted_score']
                        if mr == "STRONG_UPTREND":
                            buy_th, sell_th = 0.05, -0.25
                        elif mr == "STRONG_DOWNTREND":
                            buy_th, sell_th = 0.25, -0.05
                        else:
                            buy_th, sell_th = 0.10, -0.10
                        prev_sig = signal_result['composite_signal']
                        if ws < (sell_th - 0.1):
                            signal_result['composite_signal'] = 'STRONG_SELL'
                        elif ws < sell_th:
                            signal_result['composite_signal'] = 'SELL'
                        elif ws > (buy_th + 0.1):
                            signal_result['composite_signal'] = 'STRONG_BUY'
                        elif ws > buy_th:
                            signal_result['composite_signal'] = 'BUY'
                        else:
                            signal_result['composite_signal'] = 'NEUTRAL'
                        if signal_result['composite_signal'] != prev_sig:
                            logger.debug(f"Reclassified after range nuance: {prev_sig} → {signal_result['composite_signal']} (score={ws:.3f})")
            except Exception as e:
                logger.debug(f"Ranging nuance skipped: {e}")



            # Apply confidence adjustment based on session (MOVED HERE - AFTER signal_result creation)
            if 'confidence_adjustment' in session_info:
                signal_result['confidence'] *= session_info['confidence_adjustment']
                logger.debug(f"Session-adjusted confidence: {signal_result['confidence']:.1f}%")

            # 3. Price Action Validation
            if self.config.price_action_validation:
                # High-visibility: log params before call
                logger.debug(f"PA Validation Call: signal_type={signal_result['composite_signal']}, "
                            f"df_len={len(df_5m)}, lookback={self.config.price_action_lookback}, "
                            f"signal_result_keys={list(signal_result.keys())}, "
                            f"session_info={session_info}")

                pa_valid, pa_reason = self.price_action_validator.validate_against_price_action(
                    signal_result['composite_signal'],
                    df_5m,
                    signal_result,   # Pass signal_result
                    session_info,    # Pass session_info
                    self.config.price_action_lookback
                )
                   
                if not pa_valid:
                    logger.info(f"❌ Signal rejected by price action: {pa_reason}")
                    signal_result['rejection_reason'] = "pa_veto"
                    self._last_reject = {'stage': 'PA', 'reason': pa_reason}
                    return None


                 
                logger.info(f"✓ Price action validation passed: {pa_reason}")


                # Check momentum exhaustion (critical gate; high-visibility)
                try:
                    logger.info("=" * 60)
                    logger.info("[MOMENTUM-EXHAUSTION] Gate: entering after PA validation")
                    exhausted, exhaustion_reason = self._detect_momentum_exhaustion(
                        indicators_5m,
                        df_5m,
                        signal_result.get('composite_signal', 'NEUTRAL')
                    )
                    if exhausted:
                        # Normalize and persist rejection context for pre-close candidate logging
                        logger.warning(f"[MOMENTUM-EXHAUSTION] ❌ Signal blocked: {exhaustion_reason}")
                        # Populate normalized reason so upstream callers can map properly even if signal is None
                        signal_result['rejection_reason'] = "momentum_exhaustion"
                        self._last_reject = {'stage': 'MOMENTUM', 'reason': exhaustion_reason}
                        return None
                    logger.info("[MOMENTUM-EXHAUSTION] ✅ No exhaustion detected - proceeding")
                    logger.info("=" * 60)
                except Exception as e:
                    logger.error(f"[MOMENTUM-EXHAUSTION] Gate error: {e}", exc_info=True)




            # Ensure ATR and price are present in contributions/top-level for alert gating
            try:
                contrib = signal_result.setdefault('contributions', {})
                if isinstance(indicators_5m, dict):
                    atr_blk = indicators_5m.get('atr', {})
                    if atr_blk:
                        contrib['atr'] = {'value': float(atr_blk.get('value', 0.0) or 0.0), 'period': int(atr_blk.get('period', 14))}
                # Current 5m close as price
                if df_5m is not None and not df_5m.empty:
                    signal_result['price'] = float(df_5m['close'].iloc[-1])
            except Exception:
                pass




            # 4. Multi-Timeframe Alignment Check
            mtf_aligned = True
            mtf_score = 1.0
            mtf_description = "MTF not checked"
            
            if self.config.mtf_enabled and indicators_15m and df_15m is not None:
                
                mtf_aligned, mtf_score, mtf_description = self.mtf_analyzer.check_timeframe_alignment( 
                    signal_result,
                    indicators_15m,
                    df_15m, 
                    session_info )
                mtf_val = float(mtf_score)
                
                
                
                
                
                
                
                                
                if (
                    self.config.mtf_alignment_required
                    and str(signal_result.get('composite_signal', 'NEUTRAL')).upper() != 'NEUTRAL'
                    and not mtf_aligned
                ):
                    mtf_val = float(mtf_score)
                    comp = str(signal_result.get('composite_signal','')).upper()
                    dir_up = ('BUY' in comp)
                    slope = float(signal_result.get('contributions', {}).get('macd', {}).get('hist_slope', 0.0))
                    breadth = int(signal_result.get('active_indicators', 0))
                    slope_ok = (slope > 0 and dir_up) or (slope < 0 and not dir_up)
                    
                    lb = float(getattr(self.config, 'mtf_borderline_min', 0.55))
                    ub = float(getattr(self.config, 'mtf_borderline_max', 0.60))
                    penalty = float(getattr(self.config, 'mtf_borderline_conf_penalty', 12.0))
                    enable_soft = bool(getattr(self.config, 'enable_mtf_borderline_soft_allow', True))
                    
        
                    mag_ok = abs(slope) >= float(getattr(self.config, 'slope_soft_allow_min_mag', 0.15))
                    price_above_short = bool(signal_result.get('contributions', {}).get('ema', {}).get('price_above_short', False))
                    ema_ok = (price_above_short if dir_up else (not price_above_short))
                    if enable_soft and (lb <= mtf_val < ub) and slope_ok and mag_ok and ema_ok and breadth >= 4:

                        
                        old_conf = float(signal_result.get('confidence', 0.0))
                        signal_result['confidence'] = max(0.0, old_conf - penalty)
                        signal_result.setdefault('scalping_signals', []).append("Borderline MTF soft‑allow (confidence demoted)")
                        logger.info("[MTF] Soft-allowed borderline band: mtf=%.2f slope=%+.6f |mag_ok=%s| ema_ok=%s breadth=%d → conf %.1f→%.1f",
                                    mtf_val, slope, mag_ok, ema_ok, breadth, old_conf, float(signal_result['confidence']))

                        

                    else:
                        logger.info(f"❌ Signal rejected: {mtf_description}")
                        signal_result['rejection_reason'] = "mtf_not_aligned"
                        self._last_reject = {'stage': 'MTF', 'reason': mtf_description}
                        return None

                




                # BUY guard in bearish HTF when MTF is below strong threshold
                try:
                    actionable = str(signal_result.get('composite_signal', 'NEUTRAL')).upper() in ('BUY','STRONG_BUY','SELL','STRONG_SELL')
                    comp = str(signal_result.get('composite_signal','')).upper()
                    dir_up = ('BUY' in comp)
                    guard_on = bool(getattr(self.config, 'buy_guard_htf_enabled', True))
                    guard_thr = float(getattr(self.config, 'buy_guard_mtf_threshold', 0.70))
                    if guard_on and actionable and dir_up and (mtf_val < guard_thr):
                        # Use 15m MACD signal and 15m short EMA reclaim check
                        macd_15_sig = None
                        try:
                            macd_15_sig = str(indicators_15m.get('macd', {}).get('signal_type', 
                                            indicators_15m.get('macd', {}).get('signal','neutral'))).lower() if indicators_15m else 'neutral'
                        except Exception:
                            macd_15_sig = 'neutral'
                        price_above_15m_short = False
                        try:
                            if indicators_15m and df_15m is not None and not df_15m.empty:
                                last_15_close = float(df_15m['close'].iloc[-1])
                                ema_short_15 = float(indicators_15m.get('ema', {}).get('short', 0.0))
                                price_above_15m_short = last_15_close > ema_short_15 > 0
                        except Exception:
                            price_above_15m_short = False
                        
                        hostile_macd = ('bearish_strengthening' in (macd_15_sig or ''))
                        if hostile_macd and not price_above_15m_short:
                            logger.info("[GUARD] ❌ BUY blocked by HTF: mtf=%.2f<%.2f, 15m MACD=%s, price_above_15m_short=%s",
                                        mtf_val, guard_thr, macd_15_sig, price_above_15m_short)
                            signal_result['rejection_reason'] = "buy_guard_htf"
                            self._last_reject = {'stage': 'MTF', 'reason': 'BUY guard (HTF bearish_strengthening & below 15m short EMA)'}
                            return None
                except Exception as _e:
                    logger.debug(f"BUY guard skipped: {_e}")

                # SELL guard in bullish HTF when MTF is below strong threshold (symmetry with BUY guard)
                try:
                    actionable = str(signal_result.get('composite_signal','NEUTRAL')).upper() in ('BUY','STRONG_BUY','SELL','STRONG_SELL')
                    comp = str(signal_result.get('composite_signal','')).upper()
                    dir_down = ('SELL' in comp)
                    guard_on = bool(getattr(self.config, 'sell_guard_htf_enabled', True))
                    guard_thr = float(getattr(self.config, 'sell_guard_mtf_threshold', 0.70))
                    
                    if guard_on and actionable and dir_down and (mtf_val < guard_thr):
                        macd_15_sig = None
                        try:
                            macd_15_sig = str(indicators_15m.get('macd', {}).get('signal_type',
                                            indicators_15m.get('macd', {}).get('signal','neutral'))).lower() if indicators_15m else 'neutral'
                        except Exception:
                            macd_15_sig = 'neutral'
                        
                        price_below_15m_short = False
                        try:
                            if indicators_15m and df_15m is not None and not df_15m.empty:
                                last_15_close = float(df_15m['close'].iloc[-1])
                                ema_short_15 = float(indicators_15m.get('ema', {}).get('short', 0.0))
                                price_below_15m_short = (last_15_close < ema_short_15) if ema_short_15 > 0 else False
                        except Exception:
                            price_below_15m_short = False
                        
                        hostile_macd = ('bullish_strengthening' in (macd_15_sig or ''))
                        if hostile_macd and not price_below_15m_short:
                            logger.info("[GUARD] ❌ SELL blocked by HTF: mtf=%.2f<%.2f, 15m MACD=%s, price_below_15m_short=%s",
                                        mtf_val, guard_thr, macd_15_sig, price_below_15m_short)
                            signal_result['rejection_reason'] = "sell_guard_htf"
                            self._last_reject = {'stage': 'MTF', 'reason': 'SELL guard (HTF bullish_strengthening & above 15m short EMA)'}
                            return None
                except Exception as _e:
                    logger.debug(f"SELL guard skipped: {_e}")





                # Require higher breadth for actionable; allow 3/6 conditionally
                # actionable = str(signal_result.get('composite_signal', 'NEUTRAL')).upper() in ('BUY','SELL','STRONG_BUY','STRONG_SELL')
                contributions = signal_result.get('contributions', {})


                # Direction + MACD slope for guards (define BEFORE using it in hard-neutral)
                comp_up = str(signal_result.get('composite_signal','')).upper()
                direction = 'BUY' if 'BUY' in comp_up else ('SELL' if 'SELL' in comp_up else 'NEUTRAL')


                macd_slope = float(contributions.get('macd', {}).get('hist_slope', 0.0))
                if pd.isna(macd_slope) or np.isinf(macd_slope):
                    macd_slope = 0.0


                # [HARD-NEUTRAL] LIMITED S/R + weak MTF + opposing slope → reject early
                try:
                    sr_room = getattr(self.mtf_analyzer, '_last_sr_room', 'UNKNOWN')
                    if (direction in ('BUY','SELL') and sr_room == 'LIMITED' and mtf_val < 0.65 and
                        ((direction == 'BUY' and macd_slope <= 0) or (direction == 'SELL' and macd_slope >= 0))):
                        log_span("[HARD-NEUTRAL] Worst cluster blocked")
                        logger.info("sr_room=%s | mtf=%.2f | dir=%s | slope=%.6f", sr_room, mtf_val, direction, macd_slope)
                        signal_result['rejection_reason'] = "hard_neutral_worst_cluster"
                        self._last_reject = {'stage': 'MTF', 'reason': 'LIMITED+weakMTF+opposingSlope'}
                        
                        return None
                except Exception as e:
                    logger.debug("Hard-neutral check skipped: %s", e)


                # Opposing-slope adjustment in weak MTF (nudge toward neutral before gates)
                try:
                    ws = float(signal_result.get('weighted_score', 0.0))
                    mr = str(signal_result.get('market_regime', 'NORMAL'))
                    penalty = 0.06
                    if int(signal_result.get('active_indicators', 0)) < 4:
                        penalty += 0.02  # slightly stronger with low breadth

                    # NEW: add banded boost for 0.50–0.65 MTF
                    try:
                        if 0.50 <= mtf_val < 0.65:
                            penalty += 0.01  # gentle extra bite in the borderline band
                    except Exception:
                        pass

                    # Optional extra penalty controlled by config
                    try:
                        extra = float(getattr(self.config, 'weak_mtf_band_extra_penalty', 0.0))
                        if 0.50 <= mtf_val < 0.65 and extra > 0:
                            penalty += min(0.03, extra)
                            logger.info(f"[ADJ] Weak‑MTF band extra penalty applied: +{extra:.3f}")
                            
                            logger.info("[ADJ] Weak-MTF opposing-slope adjustment in effect (mtf=%.2f, slope=%.6f, score_before=%+.3f)",
                                        mtf_val, macd_slope, float(signal_result.get('weighted_score', 0.0)))
                            


                            
                    except Exception:
                        pass


                    changed = False
                    if mtf_val < 0.65 and ws > 0 and macd_slope <= 0:
                        ws_old = ws; ws = ws - penalty; changed = True
                        logger.info(f"[ADJ] Weak MTF BUY vs slope<=0 → score {ws_old:+.3f} → {ws:+.3f}")
                    elif mtf_val < 0.65 and ws < 0 and macd_slope >= 0:
                        ws_old = ws; ws = ws + penalty; changed = True
                        logger.info(f"[ADJ] Weak MTF SELL vs slope>=0 → score {ws_old:+.3f} → {ws:+.3f}")

                    if changed:
                        signal_result['weighted_score'] = ws
                        # Reclassify using the same thresholds as scorer
                        if mr == "STRONG_UPTREND":
                            buy_th, sell_th = 0.05, -0.25
                        elif mr == "STRONG_DOWNTREND":
                            buy_th, sell_th = 0.25, -0.05
                        else:
                            buy_th, sell_th = 0.10, -0.10

                        prev_sig = str(signal_result.get('composite_signal','NEUTRAL'))
                        if ws < (sell_th - 0.1):
                            signal_result['composite_signal'] = 'STRONG_SELL'
                        elif ws < sell_th:
                            signal_result['composite_signal'] = 'SELL'
                        elif ws > (buy_th + 0.1):
                            signal_result['composite_signal'] = 'STRONG_BUY'
                        elif ws > buy_th:
                            signal_result['composite_signal'] = 'BUY'
                        else:
                            signal_result['composite_signal'] = 'NEUTRAL'
                        if signal_result['composite_signal'] != prev_sig:
                            logger.info(f"[ADJ] Reclassified: {prev_sig} → {signal_result['composite_signal']} (mtf={mtf_val:.2f})")


                        # Pass optional context to text builder for display-only damping
                        try:
                            contributions['_ctx_mtf_score'] = mtf_val
                            contributions['_ctx_sr_room'] = getattr(self.mtf_analyzer, '_last_sr_room', None)
                        except Exception:
                            pass



                        # Rebuild dynamic prediction text
                        text, meta = self._build_dynamic_prediction(
                            composite_signal=signal_result['composite_signal'],
                            weighted_score=ws,
                            active_count=int(signal_result.get('active_indicators', 0)),
                            contributions=contributions,
                            market_regime=mr
                        )
                        signal_result['next_candle_prediction'] = text
                        signal_result['prediction_meta'] = meta
                except Exception as e:
                    logger.debug(f"Weak-MTF opposing-slope adjustment skipped: {e}")




                # Breadth gate
                active_inds = int(signal_result.get('active_indicators', 0))
                min_alert = int(getattr(self.config, 'min_active_indicators_for_alert', 4))
                if actionable and active_inds < min_alert:
                    allow_three = (
                        bool(getattr(self.config, 'enable_conditional_breadth', True)) and
                        active_inds == 3 and
                        mtf_val >= float(getattr(self.config, 'conditional_breadth_mtf_min', 0.65)) and
                        ((direction == 'BUY' and macd_slope > 0) or (direction == 'SELL' and macd_slope < 0))
                    )
                    if allow_three:
                        logger.info(f"[BREADTH] ✓ 3/6 allowed (mtf={mtf_val:.2f}, slope={macd_slope:+.6f})")
                    else:
                        signal_result['rejection_reason'] = "breadth_3_lt_4"
                        logger.info(f"[BREADTH] ❌ Reject: active_indicators {active_inds}/6 < {min_alert}")
                        return None  # reject


                # Momentum-slope guard for borderline MTF
                if getattr(self.config, 'enable_momentum_slope_guard', True) and mtf_val < 0.65 and actionable:
                    if direction == 'BUY' and macd_slope <= 0:
                        signal_result['rejection_reason'] = "slope_guard"
                        logger.info(f"[SLOPE] ❌ BUY rejected: slope={macd_slope:+.6f}, mtf={mtf_val:.2f}")
                        return None
                    if direction == 'SELL' and macd_slope >= 0:
                        signal_result['rejection_reason'] = "slope_guard"
                        logger.info(f"[SLOPE] ❌ SELL rejected: slope={macd_slope:+.6f}, mtf={mtf_val:.2f}")
                        return None



                if signal_result.get('composite_signal') == 'NEUTRAL' and not mtf_aligned: 
                    logger.debug(f"MTF not aligned (score {mtf_score:.2f}) but 5m is NEUTRAL — continuing.")
            
            # Demote strong calls when HTF momentum disagrees or breadth is low
            try:
                comp = str(signal_result.get('composite_signal', '')).upper()
                if comp.startswith('STRONG_'):
                    macd_sig_15 = str(indicators_15m.get('macd', {}).get('signal_type', 
                                    indicators_15m.get('macd', {}).get('signal','neutral'))).lower() if indicators_15m else 'neutral'
                    low_breadth = signal_result.get('active_indicators', 0) < 4
                    weak_mtf = mtf_score < 0.80
                    htf_contra = (('buy' in comp.lower() and 'bearish' in macd_sig_15) or 
                                ('sell' in comp.lower() and 'bullish' in macd_sig_15))
                    
                    if low_breadth or weak_mtf or htf_contra:
                        old = signal_result['composite_signal']
                        signal_result['composite_signal'] = 'BUY' if 'BUY' in old else 'SELL'
                        logger.info(f"Demotion: {old} → {signal_result['composite_signal']} "
                                f"(breadth={signal_result.get('active_indicators',0)}, "
                                f"mtf={mtf_score:.2f}, 15m MACD={macd_sig_15})")
            except Exception as e:
                logger.debug(f"Strong demotion check skipped: {e}")

            # Candidate layer for NEUTRAL results (heads-up only; no alerts)
            candidate_signal = None
            candidate_confidence = 0.0
            try:
                if signal_result.get('composite_signal') == 'NEUTRAL':
                    contrib = signal_result.get('contributions', {})
                    ema_sig = str(contrib.get('ema', {}).get('signal', 'neutral')).lower()
                    st_sig = str(contrib.get('supertrend', {}).get('signal', 'neutral')).lower()
                    price_above_short = bool(contrib.get('ema', {}).get('price_above_short', False))
                    macd_slope = float(contrib.get('macd', {}).get('hist_slope', 0.0))
                    rsi_val = float(contrib.get('rsi', {}).get('rsi_value', 50.0))
                    ws = float(signal_result.get('weighted_score', 0.0))
                    session = session_info.get('session', 'normal')

                    # Trend and momentum shorthand
                    trend_bull = price_above_short or any(x in ema_sig for x in ['bullish', 'golden_cross', 'above']) or ('bullish' in st_sig)
                    trend_bear = (not price_above_short) or any(x in ema_sig for x in ['bearish', 'death_cross', 'below']) or ('bearish' in st_sig)
                    mom_bull = (macd_slope > 0) or (rsi_val > 50.0)
                    mom_bear = (macd_slope < 0) or (rsi_val < 50.0)

                    # RSI 50 cross checks using series (if available)
                    rsi_series = None
                    try:
                        rsi_series = indicators_5m.get('rsi', {}).get('rsi_series')
                    except Exception:
                        rsi_series = None
                    rsi_cross_up = False
                    rsi_cross_down = False
                    try:
                        if rsi_series is not None and len(rsi_series) >= 2:
                            prev_rsi = float(rsi_series.iloc[-2])
                            curr_rsi = float(rsi_series.iloc[-1])
                            rsi_cross_up = (prev_rsi <= 50.0 and curr_rsi > 50.0)
                            rsi_cross_down = (prev_rsi >= 50.0 and curr_rsi < 50.0)
                    except Exception:
                        pass

                    # Band rebound approximation (from contributions or bollinger signal)
                    bb_pos = float(contrib.get('bollinger', {}).get('position', 0.5))
                    bb_sig_5m = str(indicators_5m.get('bollinger', {}).get('signal', 'neutral')).lower()
                    rebound_low = (bb_pos <= 0.35) or (bb_sig_5m in ('near_lower', 'oversold'))
                    rebound_high = (bb_pos >= 0.65) or (bb_sig_5m in ('near_upper', 'overbought'))

                    # Candidate MTF tolerance: allow when >= 0.50 or MTF disabled/no data
                    mtf_ok = (not self.config.mtf_enabled) or (mtf_score >= 0.50) or (indicators_15m is None or df_15m is None)

                    # Session-aware ws thresholds (ranging is more permissive)
                    ws_buy_th = 0.03 if session == 'ranging' else 0.05
                    ws_sell_th = -0.03 if session == 'ranging' else -0.05

                    # Price action supportive (reuse validator analysis)
                    pa = self.price_action_validator._analyze_candles(df_5m.tail(3))
                    pa_ok_buy = not pa.get('lower_highs', False)
                    pa_ok_sell = not pa.get('higher_lows', False)

                    # BUY_CANDIDATE (heads-up only; never actionable here)
                    if (ws > ws_buy_th and trend_bull and mom_bull and rsi_cross_up and rebound_low and pa_ok_buy and mtf_ok):
                        candidate_signal = 'BUY_CANDIDATE'
                        candidate_confidence = min(45.0, max(30.0, abs(ws) * 100))
                        logger.info(f"Heads-up NEUTRAL→BUY_CANDIDATE (session={session}, ws={ws:.3f}, rsi_cross_up={rsi_cross_up}, rebound_low={rebound_low}, mtf_ok={mtf_ok})")

                    # SELL_CANDIDATE (heads-up only; never actionable here)
                    elif (ws < ws_sell_th and trend_bear and mom_bear and rsi_cross_down and rebound_high and pa_ok_sell and mtf_ok):
                        candidate_signal = 'SELL_CANDIDATE'
                        candidate_confidence = min(45.0, max(30.0, abs(ws) * 100))
                        logger.info(f"Heads-up NEUTRAL→SELL_CANDIDATE (session={session}, ws={ws:.3f}, rsi_cross_down={rsi_cross_down}, rebound_high={rebound_high}, mtf_ok={mtf_ok})")

            except Exception as e:
                logger.debug(f"Candidate evaluation skipped: {e}")

            # 5. Calculate group consensus
            signal_result['mtf_score'] = float(mtf_score) # already persisted above;
             
            group_consensus = self.consensus_analyzer.calculate_group_consensus(indicators_5m)
            
            # 6. Analyze trend alignment
            trend_analysis = self.trend_analyzer.analyze_trend(df_5m, indicators_5m)
            

            # Session‑aware min_strength override for safe, aligned contexts
            try:
                session = session_info.get('session', 'normal')
                exp_flag = bool(getattr(self, "_last_expansion_flag", False))
                contributions = signal_result.get('contributions', {})
                slope = float(contributions.get('macd', {}).get('hist_slope', 0.0))
                comp_up = str(signal_result.get('composite_signal','')).upper()
                dir_up = ('BUY' in comp_up)
                slope_ok = (slope > 0 and dir_up) or (slope < 0 and not dir_up)
                mtf_safe = float(signal_result.get('mtf_score', 0.0)) 

                rsi_up = bool(contributions.get('rsi', {}).get('rsi_cross_up', False))
                rsi_dn = bool(contributions.get('rsi', {}).get('rsi_cross_down', False))
                cross_ok = (rsi_up and dir_up) or (rsi_dn and not dir_up)

                if (session == 'ranging') and exp_flag and slope_ok and (mtf_safe >= 0.65 or (mtf_safe >= 0.50 and cross_ok)):
                    base_min = float(getattr(self.config, 'min_signal_strength', 0.10))
                    delta = float(getattr(self.config, 'session_ranging_strength_delta', 0.02))
                    # Stronger relief when we rely on the soft MTF path (>=0.50 with RSI cross)
                    if mtf_safe < 0.65:
                        delta = max(delta, 0.04)
                    override = max(0.0, base_min - delta)
                    signal_result['min_strength_override'] = override
                    logger.info("[SESSION] min_strength override (ranging+expansion): %.3f → %.3f (mtf=%.2f, slope=%+.6f, cross_ok=%s, dir=%s)", base_min, override, mtf_safe, slope, cross_ok, 'BUY' if dir_up else 'SELL')
                    

                
            except Exception as _e:
                logger.debug(f"Session min_strength override skipped: {_e}")

            

            # Pattern confirmation gate (toggleable): require qualified pattern in aligned context for neutral→actionable promotions
            try:
                if getattr(self.config, 'require_pattern_confirmation', False):
                    comp = str(signal_result.get('composite_signal','NEUTRAL')).upper()
                    actionable = comp in ('BUY','SELL','STRONG_BUY','STRONG_SELL')
                    
                    if actionable:
                        contrib = signal_result.get('contributions', {}) or {}
                        pat = contrib.get('pattern_top', {}) or {}
                        pat_sig = str(pat.get('signal', 'NEUTRAL')).upper()
                        pat_conf = int(pat.get('confidence', 0))
                        bb_pos = float(contrib.get('bollinger', {}).get('position', 0.5))
                        
                        # Location-quality: reversal patterns near extremes
                        loc_ok = True
                        if getattr(self.config, 'enable_pattern_location_quality', True):
                            if 'BUY' in comp:
                                loc_ok = (bb_pos <= 0.35)
                            elif 'SELL' in comp:
                                loc_ok = (bb_pos >= 0.65)
                        
                        dir_ok = (('BUY' in comp and pat_sig == 'LONG') or ('SELL' in comp and pat_sig == 'SHORT'))
                        
                        if not (dir_ok and pat_conf >= int(getattr(self.config, 'pattern_min_strength', 50)) and loc_ok):
                            signal_result['rejection_reason'] = "pattern_required"
                            logger.info("[PATTERN-GATE] ❌ Reject: pattern_required (dir_ok=%s, conf=%d, loc_ok=%s, bb_pos=%.2f)", dir_ok, pat_conf, loc_ok, bb_pos)
                            return None
                        else:
                            logger.info("[PATTERN-GATE] ✓ Passed: %s (conf=%d, loc_ok=%s, bb_pos=%.2f)", str(pat.get('name','')).upper(), pat_conf, loc_ok, bb_pos)
                            
            except Exception as e:
                logger.debug(f"Pattern gate skipped: {e}")


            
            
            # 7. Enhanced validation
            is_valid = self._enhanced_validation(
                signal_result, 
                group_consensus, 
                trend_analysis,
                df_5m
            )
        
            if not is_valid: 
                logger.info(f"[VALIDATION] ❌ Reject '{signal_result.get('composite_signal','NEUTRAL')}': reason={signal_result.get('rejection_reason','unknown')} | mtf={signal_result.get('mtf_score',0.0):.2f} | active={signal_result.get('active_indicators',0)}/6 | strength={signal_result.get('weighted_score',0.0):+.3f}")

                self._last_reject = {'stage': 'VALIDATION', 'reason': 'Enhanced validation failed'} 
                return None

            # 8. Check cooldown
            if not self._check_cooldown(signal_result):
                logger.info("⏰ Signal in cooldown period")
                return None
            
            # 9. Calculate final confidence (with MTF boost) 
            final_confidence = self._calculate_final_confidence_with_mtf( signal_result, group_consensus, trend_analysis, mtf_score )
                        
            # Defensive: ensure final_confidence is numeric
            try:
                final_confidence = float(final_confidence)
                if np.isnan(final_confidence) or np.isinf(final_confidence):
                    raise ValueError("final_confidence NaN/Inf")
            except Exception:
                logger.error("Final confidence invalid (None/NaN/Inf) — falling back to base confidence")
                final_confidence = float(signal_result.get('confidence', 0.0))
            
            
            # 10. Final confidence checks (relocated hard floor)
            if final_confidence < self.config.confidence_hard_floor:
                logger.info(f"❌ Final confidence {final_confidence:.1f}% < hard floor {self.config.confidence_hard_floor}%")
                signal_result['rejection_reason'] = "confidence_floor"
                self._last_reject = {'stage': 'CONFIDENCE', 'reason': f'Below hard floor ({self.config.confidence_hard_floor}%)'}
                return None


            # 11. Predict duration
            duration_prediction = self.predictor.predict(
                indicators_5m, df_5m, signal_result
            )
            
            # 12. Calculate entry/exit levels
            entry_exit = self._calculate_entry_exit_levels(
                df_5m, indicators_5m, signal_result
            )

            # Enforce Risk/Reward floor for actionable signals — respect entry_exit decision and tapered floor
            try:
                rr_val = float(entry_exit.get('risk_reward', 0))
                comp = str(signal_result.get('composite_signal', 'NEUTRAL')).upper()

                # If entry_exit already rejected on rr_floor, honor it
                if entry_exit.get('rejection_reason') == 'rr_floor':
                    signal_result['rejection_reason'] = "rr_floor"
                    log_span("[RISK] R:R Enforcement")

                    logger.info("x" * 60)
                    logger.info(f"❌ Rejected by R:R (from entry/exit): {rr_val:.2f}")
                    logger.info("x" * 60)
                    self._last_reject = {'stage': 'RR', 'reason': f'R:R {rr_val:.2f} below tapered floor'}
                    return None

                base_floor = float(getattr(self.config, 'min_risk_reward_floor', 1.0))
                use_taper = bool(getattr(self.config, 'enable_rr_taper', True))
                burst_strength = float(getattr(self.config, 'rr_taper_burst_strength', 0.30))

                # Normalize conf_limit
                conf_limit_raw = float(getattr(self.config, 'rr_taper_confidence_min', 0.75))
                conf_limit = (conf_limit_raw / 100.0) if conf_limit_raw > 1.0 else conf_limit_raw
                if conf_limit_raw > 1.0:
                    logger.info("[RISK] Normalized rr_taper_confidence_min from %.2f to %.2f (fraction)", conf_limit_raw, conf_limit)

                conf_pct = float(signal_result.get('confidence', 0.0)) / 100.0
                is_burst = (abs(float(signal_result.get('weighted_score', 0.0))) >= burst_strength)



                rr_floor = base_floor
                if use_taper and is_burst and conf_pct >= conf_limit:
                    rr_floor = float(getattr(self.config, 'rr_taper_floor', 0.80))
                    logger.info(f"[RISK] Tapered R:R floor active {rr_floor:.2f} (burst+conf); base={base_floor:.2f}")

                if comp in ('BUY', 'STRONG_BUY', 'SELL', 'STRONG_SELL') and rr_val < rr_floor:
                    logger.info("x" * 60)
                    logger.info(f"❌ Rejected by R:R filter → {rr_val:.2f} < floor {rr_floor:.2f}")
                    logger.info("x" * 60)
                    signal_result['rejection_reason'] = "rr_floor"
                    self._last_reject = {'stage': 'RR', 'reason': f'R:R {rr_val:.2f} below floor {rr_floor:.2f}'}
                    return None
                else:
                    logger.debug(f"R:R check passed → {rr_val:.2f} ≥ floor {rr_floor:.2f}")
            except Exception as e:
                logger.debug(f"R:R enforcement skipped due to error: {e}")

            
            
            # 13. Get accuracy metrics
            accuracy_metrics = self._calculate_accuracy_metrics()
            
            # 14. Get market structure
            market_structure = self._analyze_market_structure(df_5m, indicators_5m)

            # Our 5m index holds the START time; derive the true close time 
            bar_start_time = df_5m.index[-1] 
            bar_close_time = bar_start_time + timedelta(seconds=self.config.candle_interval_seconds)





            final_signal = { 
                    **signal_result, 
                    'confidence': final_confidence, 
                    'group_consensus': group_consensus, 
                    'trend_analysis': trend_analysis,
                    'mtf_analysis': {
                        'aligned': mtf_aligned,
                        'score': mtf_score,
                        'description': mtf_description,
                        'sr_room': getattr(self.mtf_analyzer, '_last_sr_room', None)
                    },
                    'duration_prediction': duration_prediction, 
                    'entry_exit': entry_exit, 
                    'entry_price': entry_exit.get('entry_price', 0), 
                    'stop_loss': entry_exit.get('stop_loss', 0), 
                    'take_profit': entry_exit.get('take_profit', 0), 
                    'risk_reward': entry_exit.get('risk_reward', 0), 
                    'accuracy_metrics': accuracy_metrics, 
                    'market_structure': market_structure, 
                    'timestamp': datetime.now(), 
                    'timeframe': '5m', 
                    'bar_close_time': str(bar_close_time),
                    'bar_start_time': str(bar_start_time),
                    'higher_timeframe': '15m' if (df_15m is not None and indicators_15m is not None) else None, 
                    'action': self._get_action_from_signal(signal_result['composite_signal']), 
                    # Heads-up fields for Neutral promotion (no alerts sent on these) 
                    'candidate_signal': candidate_signal, 
                    'candidate_confidence': candidate_confidence 
                }

            logger.info(f"Bar times → start={bar_start_time.strftime('%H:%M:%S')}, close={bar_close_time.strftime('%H:%M:%S')}")

            # Update history (do NOT update last_alert_time here — alert time is set when alert is actually sent)
            self.signal_history.append(final_signal)

            logger.info("=" * 50)
            logger.info(f"✅ SIGNAL GENERATED [5m]: {signal_result['composite_signal']}")
            logger.info(f"   Confidence: {final_confidence:.1f}%")
            logger.info(f"   Score: {signal_result['weighted_score']:.3f}")
            logger.info(f"   MTF: {mtf_description}")
            logger.info(f"   Duration: {duration_prediction['estimated_minutes']} mins")
            logger.info("=" * 50)
       

            # Debug mode for detailed signal analysis
            if hasattr(self.config, 'debug_mode') and self.config.debug_mode:
                logger.info("=" * 60)
                logger.info("DEBUG: SIGNAL ANALYSIS DETAILS")
                logger.info("=" * 60)
                
                # Log all indicator signals
                logger.info("5M INDICATORS:")
                for name, data in indicators_5m.items():
                    if isinstance(data, dict):
                        if name == 'macd':
                            signal = data.get('signal_type', data.get('signal', 'N/A'))
                            value = data.get('histogram', 'N/A')
                        else:
                            signal = data.get('signal', 'N/A')
                            value = data.get('value', 'N/A')
                        logger.info(f"  {name}: signal={signal}, value={value}")

                if indicators_15m:
                    logger.info("\n15M INDICATORS:")

                    for name, data in indicators_15m.items():
                        if isinstance(data, dict): 
                            if name == 'macd': 
                                signal = data.get('signal_type', data.get('signal', 'N/A')) 
                                value = data.get('histogram', 'N/A')
                            else: 
                                signal = data.get('signal', 'N/A') 
                                value = data.get('value', 'N/A') 
                            logger.info(f" {name}: signal={signal}, value={value}")
                
                logger.info("=" * 60)


            try:
                # Historical evidence and WHY line (display + small bounded confidence nudge)
                self._apply_evidence_nudge_and_why(final_signal, indicators_5m)
                # LIMITED SR cap with ATR distance
                self._cap_confidence_sr_limited(final_signal, indicators_5m)
            except Exception as e:
                logger.debug(f"[POST] evidence/why/cap skipped: {e}")


            return final_signal
            
        except Exception as e:
            logger.error(f"Signal analysis error: {e}", exc_info=True)
            return None


    def _detect_breakout_evidence(self, df: pd.DataFrame, indicators: Dict) -> bool:
        """
        Lightweight breakout evidence: last close within 0.10% of recent high and MACD histogram strengthening or BB position > 0.90.
        """
        try:
            if df is None or df.empty or len(df) < 20:
                return False
            recent = df.tail(20)
            close = float(recent['close'].iloc[-1])
            hi = float(recent['high'].max())
            if hi <= 0:
                return False
            near_high = (hi - close) / max(1e-9, hi) <= 0.001  # 0.10%
            macd_hist = float((indicators or {}).get('macd', {}).get('histogram', 0.0))
            macd_series = (indicators or {}).get('macd', {}).get('histogram_series')
            strengthening = False
            try:
                if macd_series is not None and len(macd_series) >= 2:
                    strengthening = float(macd_series.iloc[-1]) > float(macd_series.iloc[-2])
            except Exception:
                strengthening = False
            bb_pos = float((indicators or {}).get('bollinger', {}).get('position', 0.5))
            return bool(near_high and (strengthening or bb_pos >= 0.90))
        except Exception:
            return False

    def _detect_breakdown_evidence(self, df: pd.DataFrame, indicators: Dict) -> bool:
        """
        Lightweight breakdown evidence: last close within 0.10% of recent low and MACD histogram weakening or BB position < 0.10.
        """
        try:
            if df is None or df.empty or len(df) < 20:
                return False
            recent = df.tail(20)
            close = float(recent['close'].iloc[-1])
            lo = float(recent['low'].min())
            if lo <= 0:
                return False
            near_low = (close - lo) / max(1e-9, lo) <= 0.001  # 0.10%
            macd_hist = float((indicators or {}).get('macd', {}).get('histogram', 0.0))
            macd_series = (indicators or {}).get('macd', {}).get('histogram_series')
            weakening = False
            try:
                if macd_series is not None and len(macd_series) >= 2:
                    weakening = float(macd_series.iloc[-1]) < float(macd_series.iloc[-2])
            except Exception:
                weakening = False
            bb_pos = float((indicators or {}).get('bollinger', {}).get('position', 0.5))
            return bool(near_low and (weakening or bb_pos <= 0.10))
        except Exception:
            return False




    def should_generate_rapid_scalping_signal(self, df: pd.DataFrame, last_signal_time: datetime) -> bool:
        """
        Generate more frequent signals for scalping.

        This method checks for rapid scalping opportunities based on various patterns and conditions.

        Parameters:
        df (pd.DataFrame): DataFrame containing market data.
        last_signal_time (datetime): Timestamp of the last signal generated.

        Returns:
        bool: True if a rapid scalping signal should be generated, False otherwise.
        """

        try:
            # Protect against None value
            if last_signal_time is None:
                logger.info("Last signal time is None, returning False")
                return False

            # Allow rapid signals if strong momentum detected
            time_diff = (datetime.now() - last_signal_time).total_seconds()
            logger.debug(f"Time since last signal: {time_diff:.2f} seconds")
            if time_diff < 30:
                logger.info("Returning False due to cooldown period")
                return False

            # Check for rapid scalping opportunities
            last_3 = df.tail(3)
            logger.debug(f"Last 3 rows of DataFrame:\n{last_3}")

            # Pattern 1: Three consecutive same-direction candles
            all_green = all(last_3['close'] > last_3['open'])
            all_red = all(last_3['close'] < last_3['open'])
            logger.debug(f"All green: {all_green}, All red: {all_red}")

            # Pattern 2: Price acceleration
            if len(df) > 10:
                recent_move = abs(df['close'].iloc[-1] - df['close'].iloc[-3])
                avg_move = abs(df['close'].diff()).tail(10).mean()
                accelerating = recent_move > avg_move * 2
                logger.debug(f"Recent move: {recent_move:.2f}, Average move: {avg_move:.2f}, Accelerating: {accelerating}")

                if (all_green or all_red) and accelerating:
                    logger.info("Rapid scalping signal generated due to acceleration pattern")
                    return True

            # Pattern 3: Support/Resistance break
            if hasattr(self, 'resistance_detector') and self.resistance_detector is not None:
                levels = self.resistance_detector.detect_levels(df)
                current = df['close'].iloc[-1]
                logger.debug(f"Nearest resistance: {levels['nearest_resistance']:.2f}, Nearest support: {levels['nearest_support']:.2f}")

                # Breaking resistance
                if current > levels['nearest_resistance'] * 0.999:
                    logger.info("Rapid scalping signal generated due to resistance break")
                    return True
                # Bouncing from support
                if current < levels['nearest_support'] * 1.001:
                    logger.info("Rapid scalping signal generated due to support bounce")
                    return True

            logger.info("No rapid scalping signal generated")
            return False

        except Exception as e:
            logger.error(f"Error generating rapid scalping signal: {e}", exc_info=True)
            return False


    def detect_session_characteristics(self, df: pd.DataFrame) -> Dict:
        """
        Auto-detect session type based on market behavior, not time.

        This method analyzes the given DataFrame to determine the current session characteristics,
        including session type, strategy, and confidence adjustment.

        Parameters:
        df (pd.DataFrame): DataFrame containing market data.

        Returns:
        Dict: Dictionary containing session characteristics.
        """

        try:
            # Check if DataFrame has less than 20 rows

            if len(df) < 20: 
                logger.info("Insufficient data for session detection, returning 'unknown' session") 
                return { 'session': 'unknown', 'strategy': 'standard', 'confidence_adjustment': 1.0, 'characteristics': {} }
            

            # Analyze last 20 candles
            recent = df.tail(20)
            logger.debug(f"Analyzing last 20 rows of DataFrame:\n{recent}")

            # Calculate metrics
            volatility = recent['high'].std() / recent['close'].mean() * 100
            logger.debug(f"Volatility: {volatility:.2f}%")

            avg_range = (recent['high'] - recent['low']).mean()
            logger.debug(f"Average range: {avg_range:.2f}")

            trend_strength = abs(recent['close'].iloc[-1] - recent['close'].iloc[0]) / recent['close'].iloc[0] * 100
            logger.debug(f"Trend strength: {trend_strength:.2f}%")

            current_hour = datetime.now().hour
            logger.debug(f"Current hour: {current_hour}")

            # Detect opening patterns (high volatility, wide ranges)
            if volatility > 0.5 and avg_range > recent['close'].mean() * 0.003:
                session_type = 'opening_volatile'
                strategy = 'fade_extremes'
                confidence_adjustment = 0.8  # Lower confidence in volatile periods
                logger.info(f"Detected 'opening_volatile' session with strategy '{strategy}'")

            # Detect trending session (steady movement, lower volatility)
            elif trend_strength > 0.3 and volatility < 0.3:
                session_type = 'trending'
                strategy = 'follow_trend'
                confidence_adjustment = 1.2  # Higher confidence in trends
                logger.info(f"Detected 'trending' session with strategy '{strategy}'")

            # Detect ranging/consolidation (low volatility, small ranges)
            elif volatility < 0.2 and avg_range < recent['close'].mean() * 0.002:
                session_type = 'ranging'
                strategy = 'mean_reversion'
                confidence_adjustment = 0.9
                logger.info(f"Detected 'ranging' session with strategy '{strategy}'")

            # Detect closing session (decreasing volatility)
            elif len(df) > 60:
                early_vol = df.iloc[-60:-40]['high'].std() / df.iloc[-60:-40]['close'].mean() * 100
                late_vol = recent['high'].std() / recent['close'].mean() * 100

                if late_vol < early_vol * 0.7:  # Volatility decreased by 30%+
                    session_type = 'closing'
                    strategy = 'book_profits'
                    confidence_adjustment = 0.7
                    logger.info(f"Detected 'closing' session with strategy '{strategy}'")
                else:
                    session_type = 'normal'
                    strategy = 'standard'
                    confidence_adjustment = 1.0
                    logger.info(f"Detected 'normal' session with strategy '{strategy}'")
            else:
                session_type = 'normal'
                strategy = 'standard'
                confidence_adjustment = 1.0
                logger.info(f"Detected 'normal' session with strategy '{strategy}'")

            # Detect specific patterns
            patterns = {
                'gap_detected': False,
                'breakout_potential': False,
                'reversal_potential': False
            }

            # Gap detection
            if len(df) > 1:
                gap = abs(df['open'].iloc[-1] - df['close'].iloc[-2]) / df['close'].iloc[-2] * 100
                patterns['gap_detected'] = gap > 0.3
                logger.debug(f"Gap detected: {patterns['gap_detected']}")

            # Breakout detection (price near recent high/low)
            recent_high = recent['high'].max()
            recent_low = recent['low'].min()
            current = df['close'].iloc[-1]

            if (recent_high - current) / current < 0.002:
                patterns['breakout_potential'] = True
                logger.debug(f"Breakout potential: {patterns['breakout_potential']}")
            elif (current - recent_low) / current < 0.002:
                patterns['reversal_potential'] = True
                logger.debug(f"Reversal potential: {patterns['reversal_potential']}")

            logger.info(f"Session Auto-Detected: {session_type} | Strategy: {strategy}")
            logger.info(f"Volatility: {volatility:.2f}% | Trend: {trend_strength:.2f}%")

            return {
                'session': session_type,
                'strategy': strategy,
                'confidence_adjustment': confidence_adjustment,
                'characteristics': {
                    'volatility': volatility,
                    'avg_range': avg_range,
                    'trend_strength': trend_strength,
                    'patterns': patterns
                }
            }

        except Exception as e:
            logger.error(f"Session detection error: {e}", exc_info=True)
            return {'session': 'unknown', 'strategy': 'standard', 'confidence_adjustment': 1.0}



    def _detect_expansion_lite(self, df: pd.DataFrame, indicators_5m: Dict) -> tuple[bool, bool]:
        """
        Lite expansion detector (no volume): expansion=True when ≥2 of the following are true over the recent window:
        - Bollinger bandwidth rising,
        - HL-range acceleration positive (mean range increasing),
        - Absolute MACD histogram change rising.
        Returns (expansion, flip_from_compression).
        """
        try:
            if df is None or df.empty or len(df) < 10:
                return False, False

            # Window sizes (small to focus on near-term)
            w = 5
            d = 3

            # 1) Bollinger bandwidth rising
            try:
                bb_bw = float((indicators_5m or {}).get('bollinger', {}).get('bandwidth', 0.0))
                bb_series = (indicators_5m or {}).get('bollinger', {}).get('middle_series', None)
                # If we have a series, reconstruct bandwidth from upper/lower where possible for robustness
                upper_s = (indicators_5m or {}).get('bollinger', {}).get('upper_series', None)
                lower_s = (indicators_5m or {}).get('bollinger', {}).get('lower_series', None)
                bw_rising = False
                if upper_s is not None and lower_s is not None:
                    try:
                        u = pd.to_numeric(upper_s, errors='coerce').tail(w).replace([np.inf, -np.inf], np.nan)
                        l = pd.to_numeric(lower_s, errors='coerce').tail(w).replace([np.inf, -np.inf], np.nan)
                        m = (u + l) / 2.0
                        bw = ((u - l) / m).replace([np.inf, -np.inf], np.nan).dropna()
                        if len(bw) >= d:
                            bw_rising = bool(bw.diff().tail(d).mean() > 0)
                    except Exception:
                        bw_rising = False
                else:
                    # Fallback: single-point bandwidth snapshot cannot show rising; treat as neutral
                    bw_rising = False
            except Exception:
                bw_rising = False

            # 2) HL-range acceleration (mean high-low increasing)
            try:
                rng = (pd.to_numeric(df['high'], errors='coerce') - pd.to_numeric(df['low'], errors='coerce')).replace([np.inf, -np.inf], np.nan)
                rng_ma = rng.rolling(window=max(3, w)).mean().dropna().tail(d+1)
                hl_accel = False
                if len(rng_ma) >= d+1:
                    # Compare last 2 deltas to see if average range is increasing
                    deltas = rng_ma.diff().dropna().tail(d)
                    hl_accel = bool(deltas.mean() > 0)
                else:
                    hl_accel = False
            except Exception:
                hl_accel = False

            # 3) Absolute MACD histogram change rising
            try:
                macd_hist_s = (indicators_5m or {}).get('macd', {}).get('histogram_series', None)
                macd_abs_rising = False
                if macd_hist_s is not None and len(macd_hist_s) >= (d+2):
                    hs = pd.to_numeric(macd_hist_s, errors='coerce').replace([np.inf, -np.inf], np.nan).dropna().tail(d+2)
                    if len(hs) >= (d+2):
                        abs_diff = hs.diff().abs().dropna().tail(d)
                        macd_abs_rising = bool(abs_diff.mean() > 0 and abs_diff.iloc[-1] >= abs_diff.iloc[0])
            except Exception:
                macd_abs_rising = False

            signals = [bw_rising, hl_accel, macd_abs_rising]
            expansion_now = sum(1 for s in signals if s) >= 2

            # Flip from compression detection (track previous state)
            last_state = bool(getattr(self, "_was_expanded", False))
            flip = (not last_state) and expansion_now
            try:
                self._was_expanded = expansion_now
            except Exception:
                pass

            logger.info("[EXP] Lite expansion → bb_rising=%s hl_accel=%s macd_abs_rising=%s => expansion=%s flip=%s",
                        bw_rising, hl_accel, macd_abs_rising, expansion_now, flip)
            
            return expansion_now, flip
        except Exception as e:
            logger.debug(f"[EXP] Lite expansion calc error: {e}")
            return False, False



    def _calculate_final_confidence_with_mtf(
        self, 
        signal_result: Dict, 
        group_consensus: Dict, 
        trend_analysis: Dict, 
        mtf_score: float 
    ) -> float:
        """
        Calculate final confidence including MTF boost.

        This method calculates the final confidence of a signal by combining various factors,
        including group consensus, trend alignment, MTF alignment, and indicator agreement.

        Parameters:
        signal_result (Dict): Dictionary containing signal result.
        group_consensus (Dict): Dictionary containing group consensus.
        trend_analysis (Dict): Dictionary containing trend analysis.
        mtf_score (float): MTF score.

        Returns:
        float: Final confidence value.
        """

        try:
            # Start with base confidence (ensure numeric)
            base_confidence = float(signal_result.get('confidence', 0.0))
            logger.debug(f"Base confidence: {base_confidence:.1f}")

            # None-safe MTF score
            try:
                mtf_score = float(mtf_score) if mtf_score is not None else 0.0
            except Exception:
                mtf_score = 0.0
            logger.debug(f"MTF score: {mtf_score:.2f}")

            # Factor 1: Group consensus bonus (up to +15%)
            consensus_bonus = 0.0
            if group_consensus:
                valid_groups = [g for g in group_consensus.values() if g.get('indicator_count', 0) > 0]
                if valid_groups:
                    avg_agreement = float(np.mean([g.get('agreement', 0) for g in valid_groups]))
                    consensus_bonus = avg_agreement * 15.0
            logger.debug(f"Consensus bonus: {consensus_bonus:.1f}")

            # Factor 2: Trend alignment bonus (up to +10%) — skip for NEUTRAL
            trend_bonus = 0.0
            sig_str = str(signal_result.get('composite_signal', '')).upper()
            if trend_analysis.get('aligned', False) and sig_str not in ('NEUTRAL', ''):
                signal_direction = 1 if "BUY" in sig_str else -1 if "SELL" in sig_str else 0
                trend_direction = 1 if trend_analysis.get('trend') == 'bullish' else -1 if trend_analysis.get('trend') == 'bearish' else 0
                if signal_direction == trend_direction:
                    trend_bonus = 10.0
            logger.debug(f"Trend bonus: {trend_bonus:.1f}")

            # Factor 3: MTF alignment bonus (up to +20%) — skip for NEUTRAL
            mtf_bonus = (mtf_score * 20.0) if (self.config.mtf_enabled and sig_str not in ('NEUTRAL', '')) else 0.0
            logger.debug(f"MTF bonus: {mtf_bonus:.1f}")

            # Factor 4: Indicator agreement — count and cross‑domain breadth (unified)
            agreement_bonus = 0.0
            try:
                active_ratio = signal_result['active_indicators'] / max(1, len(self.config.indicator_weights))
                contributions = signal_result.get('contributions', {})
                groups = getattr(self.config, 'indicator_groups', {
                    'leading': ['rsi', 'macd', 'bollinger'],
                    'lagging': ['ema', 'supertrend'],
                    'volatility': ['bollinger', 'keltner'],
                })

                def group_active(names: list) -> bool:
                    return any(abs(contributions.get(n, {}).get('contribution', 0.0)) >= 0.03 for n in names)

                trend_active = group_active(groups.get('lagging', []))
                momentum_active = group_active(groups.get('leading', []))
                vol_active = group_active(groups.get('volatility', []))
                group_coverage = (int(trend_active) + int(momentum_active) + int(vol_active)) / 3.0

                if active_ratio >= 0.7:
                    agreement_bonus += 5.0
                if group_coverage >= 0.67:
                    agreement_bonus += 5.0
                if active_ratio < 0.33 or group_coverage == 0:
                    agreement_bonus -= 10.0

                logger.debug(f"Agreement breadth → active_ratio={active_ratio:.2f}, "
                            f"group_coverage={group_coverage:.2f}, bonus={agreement_bonus:+.1f}")
            except Exception:
                # Fallback to count‑only if anything goes wrong
                active_ratio = signal_result['active_indicators'] / max(1, len(self.config.indicator_weights))
                if active_ratio >= 0.7:
                    agreement_bonus = 10.0
                elif active_ratio >= 0.5:
                    agreement_bonus = 5.0
                elif active_ratio < 0.3:
                    agreement_bonus = -10.0
                else:
                    agreement_bonus = 0.0

            # Small bonus when EMA and MACD agree with the signal (stacked momentum)
            try:
                contributions = signal_result.get('contributions', {})
                ema_sig = str(contributions.get('ema', {}).get('signal', 'neutral')).lower()
                macd_sig = str(contributions.get('macd', {}).get('signal_type', 'neutral')).lower()
                is_buy = 'buy' in sig_str
                is_sell = 'sell' in sig_str

                if is_buy and any(x in ema_sig for x in ['bullish','golden_cross','above']) and 'bullish' in macd_sig:
                    trend_bonus += 5.0
                if is_sell and any(x in ema_sig for x in ['bearish','death_cross','below']) and 'bearish' in macd_sig:
                    trend_bonus += 5.0
            except Exception:
                pass

            # Small bonus when EMA and MACD agree with the signal (stacked momentum)
            try:
                contributions = signal_result.get('contributions', {}) or {}
                ema_sig = str(contributions.get('ema', {}).get('signal', 'neutral')).lower()
                macd_sig = str(contributions.get('macd', {}).get('signal_type', contributions.get('macd', {}).get('signal', 'neutral'))).lower()
                is_buy = 'buy' in sig_str
                is_sell = 'sell' in sig_str
                both_bull = ('bull' in ema_sig or 'above' in ema_sig) and ('bull' in macd_sig)
                both_bear = ('bear' in ema_sig or 'below' in ema_sig) and ('bear' in macd_sig)
                if (is_buy and both_bull) or (is_sell and both_bear):
                    agreement_bonus += 2.5
                    logger.debug(f"EMA+MACD agreement bonus applied (+2.5): ema={ema_sig}, macd={macd_sig}, sig={sig_str}")
            except Exception:
                pass

            # Combine
            final_confidence = base_confidence + consensus_bonus + trend_bonus + mtf_bonus + agreement_bonus
            final_confidence = max(0.0, min(100.0, final_confidence))
            logger.debug(f"Final confidence computed: {final_confidence:.1f}")

            return final_confidence
            


        except Exception as e:
            logger.error(f"Error calculating final confidence: {e}", exc_info=True)
            # Fallback to base confidence if something went wrong
            return float(signal_result.get('confidence', 0.0))

    def _calculate_weighted_signal(self, indicators: Dict) -> Dict:
        """Calculate weighted signal for INDEX SCALPING with market regime awareness."""

        # IMPORTANT: initialize locals used throughout (prevents UnboundLocalError)
        weighted_score: float = 0.0
        composite_signal: str = 'NEUTRAL'

        # Optional visibility for initialization (consistent with high-verbosity style)
        logger.debug("[SCORER] initialized locals: weighted_score=%.3f, composite_signal=%s",
                    weighted_score, composite_signal)
        

        
        try:
            # Always build a local result dict; never reference 'signal_result' in this scope
            result: Dict[str, Any] = {
                'weighted_score': 0.0,
                'contributions': {},
                'active_indicators': 0,
                'composite_signal': 'NEUTRAL',
                'confidence': 0.0,
                'market_regime': 'NORMAL',
            }
            contributions = result['contributions']
            active_count = 0
            scalping_signals: List[str] = []
            

            # CRITICAL: Detect market regime first
            market_regime = self.detect_market_regime(self.current_df) if hasattr(self, 'current_df') else "NORMAL"
            result['market_regime'] = market_regime
            logger.debug("[SCORER] market_regime=%s", market_regime)
            

            
            for ind_name, weight in self.config.indicator_weights.items():
                if ind_name not in indicators:
                    continue

                indicator = indicators[ind_name]
                signal = str(indicator.get('signal', 'neutral')).lower()
                value = indicator.get('value', 0)
                contribution = 0.0
                prediction = ""
                extras = {}

       
                                        

                # ADJUST SIGNALS BASED ON MARKET REGIME
                if ind_name == 'rsi':
                    # Always capture micro features and neutralize on fallback before regime logic
                    extras['rsi_value'] = indicator.get('value', 50)
                    try:
                        fallback_flag = bool(indicator.get('fallback', False))
                        extras['fallback'] = fallback_flag
                        if fallback_flag:
                            logger.info("[RSI] Fallback detected → contribution neutralized (all regimes)")
                            contribution = 0.0  # continue to record micro-features but avoid directional contribution
                        
                        rsi_series = indicator.get('rsi_series')
                        prev_rsi = float(rsi_series.iloc[-2]) if rsi_series is not None and len(rsi_series) >= 2 else None
                        curr_rsi = float(indicator.get('value', 50))
                        rsi_cross_up = bool(prev_rsi is not None and prev_rsi <= 50.0 and curr_rsi > 50.0)
                        rsi_cross_down = bool(prev_rsi is not None and prev_rsi >= 50.0 and curr_rsi < 50.0)
                        extras['rsi_value'] = curr_rsi
                        extras['rsi_cross_up'] = rsi_cross_up
                        extras['rsi_cross_down'] = rsi_cross_down
                        logger.debug(f"[RSI] value={curr_rsi:.1f} prev={prev_rsi} upX={rsi_cross_up} dnX={rsi_cross_down}")
                    except Exception:
                        extras['rsi_value'] = indicator.get('value', 50)
                    
                    # Now apply regime adjustments ONLY if not fallback
                    if not extras.get('fallback', False):
                        if market_regime == "STRONG_UPTREND":
                            # In strong uptrends, RSI can stay overbought
                            if value > 85:
                                contribution = -weight * 0.5
                                active_count += 1
                            elif value > 70:
                                contribution = weight * 0.3
                                active_count += 1
                            elif value < 30:
                                contribution = weight * 1.5
                                active_count += 1
                            elif value > 50:
                                contribution = weight * 0.2
                                active_count += 1
                        else:
                            # Normal market conditions (your existing logic)
                            if signal == 'overbought' or value > 75:
                                contribution = -weight
                                active_count += 1
                                if value > 80:
                                    prediction = "RSI overbought → Pullback likely"
                                    scalping_signals.append(prediction)
                            elif signal == 'oversold' or value < 25:
                                contribution = weight
                                active_count += 1
                                if value < 20:
                                    prediction = "RSI oversold → Bounce likely"
                                    scalping_signals.append(prediction)
                            elif signal != 'neutral':
                                active_count += 1
                            # Micro-features already captured above for RSI; no duplication here
       
                            
                elif ind_name == 'bollinger':
                    position = indicator.get('position', 0.5)
                    if market_regime == "STRONG_UPTREND":
                        # In uptrend, upper band touches are continuation
                        if position > 0.95:
                            contribution = -weight * 0.3  # Mild caution only
                            active_count += 1
                        elif position < 0.5:  # Mid-band is support in uptrend
                            contribution = weight
                            active_count += 1
                            prediction = "At Bollinger support → Buy opportunity"
                            scalping_signals.append(prediction)
                    else:
                        # Normal conditions
                        if signal in ['overbought', 'near_upper'] or position > 0.9:
                            contribution = -weight
                            active_count += 1
                            prediction = "At Bollinger upper → Resistance"
                            scalping_signals.append(prediction)
                        elif signal in ['oversold', 'near_lower'] or position < 0.1:
                            contribution = weight
                            active_count += 1
                            prediction = "At Bollinger lower → Support"
                            scalping_signals.append(prediction)
                        elif signal not in ['neutral', 'within']:
                            active_count += 1
                            
                        # Store Bollinger position for later quality checks
                        extras['position'] = indicator.get('position', 0.5)


                            
                elif ind_name == 'keltner':
                    if signal == 'above_upper':
                        if market_regime == "STRONG_UPTREND":
                            contribution = weight * 0.2  # Breakout continuation
                        else:
                            contribution = -weight
                        active_count += 1
                    elif signal == 'below_lower':
                        if market_regime == "STRONG_DOWNTREND":
                            contribution = -weight * 0.2
                        else:
                            contribution = weight
                        active_count += 1
                    elif signal != 'within':
                        active_count += 1


                elif ind_name == 'macd':
                    hist = indicator.get('histogram', 0)
                    signal_type = indicator.get('signal_type', 'neutral')
                    
                    

                    # CLOSED-only slope from indicator (hist_slope_closed is precomputed in technical_indicators)
                    hist_slope = float(indicator.get('hist_slope_closed', 0.0) or 0.0)
                    if pd.isna(hist_slope) or np.isinf(hist_slope):
                        hist_slope = 0.0


                        
                        
                        
                    if hist > 0 or 'bullish' in signal_type:
                        # Reduce bullish contribution if histogram is positive but weakening
                        weakening = bool(hist_slope < 0)
                        contribution = weight * (0.6 if weakening else 1.0)
                        active_count += 1
                        if not weakening and hist > 2:
                            prediction = "MACD bullish momentum"
                            scalping_signals.append(prediction)
                    elif hist < 0 or 'bearish' in signal_type:
                        # Reduce bearish penalty if histogram is rising (less negative)
                        easing = bool(hist_slope > 0)
                        contribution = -weight * (0.5 if easing else 1.0)
                        active_count += 1
                        if not easing and hist < -2:
                            prediction = "MACD bearish divergence"
                            scalping_signals.append(prediction)
                    elif signal_type != 'neutral':
                        active_count += 1
                        
                    
                    # Expose slope and macd signal_type for quality checks and consensus 
                    extras['hist_slope'] = hist_slope 
                    extras['signal_type'] = signal_type                       


                elif ind_name == 'ema':
                    short_series = indicator.get('short_series')
                    price_above_short = False
                    try:
                        if short_series is not None and len(short_series) > 0 and hasattr(self, 'current_df'):
                            price_above_short = float(self.current_df['close'].iloc[-1]) > float(short_series.iloc[-1])
                    except Exception:
                        price_above_short = False

                    if signal in ['bullish', 'golden_cross', 'above']:
                        # Soften bullish if price hasn’t reclaimed the short EMA
                        contribution = weight * (0.6 if not price_above_short else 1.0)
                        active_count += 1
                    elif signal in ['bearish', 'death_cross', 'below']:
                        # Soften bearish if price is above short EMA (early reversal tell)
                        contribution = -weight * (0.6 if price_above_short else 1.0)
                        active_count += 1
                    elif signal != 'neutral':
                        active_count += 1

                    # Expose price_above_short for quality checks
                    extras['price_above_short'] = price_above_short


                elif ind_name == 'supertrend':
                    st_signal = str(indicator.get('signal', indicator.get('trend', 'neutral'))).lower()
                    # Early-flip softening: only if 5m reversal context is present AND HTF MACD is not bearish
                    ema_sig_local = str(contributions.get('ema', {}).get('signal', 'neutral')).lower()
                    price_above_short = bool(contributions.get('ema', {}).get('price_above_short', False))
                    macd_slope_local = float(contributions.get('macd', {}).get('hist_slope', 0.0))
                    htf_macd_sig = getattr(self, '_htf_macd_sig', None)

                    if st_signal == 'bullish':
                        contribution = weight
                        active_count += 1
                    elif st_signal == 'bearish':
                        softening_ctx = any(x in ema_sig_local for x in ['bullish','golden_cross','above']) and price_above_short and macd_slope_local >= 0
                        htf_hostile = bool(htf_macd_sig and ('bearish' in htf_macd_sig))
                       
                       

                        top_guard = bool(getattr(self, '_extreme_top', False))
                        bottom_guard = bool(getattr(self, '_extreme_bottom', False))
                        if softening_ctx and not htf_hostile and not top_guard and not bottom_guard:
                            contribution = -weight * 0.3
                            active_count += 1
                            scalping_signals.append("ST lagging vs EMA/MACD → penalty softened")
                            logger.info("ST bearish but 5m EMA/MACD/price show turn; HTF MACD not bearish → softened ST penalty")
                        else:
                            contribution = -weight
                            active_count += 1
                            if softening_ctx and (htf_hostile or top_guard or bottom_guard):
                                why = "HTF MACD bearish" if htf_hostile else f"HTF price_position extreme (top={top_guard}, bottom={bottom_guard})"
                                logger.info(f"ST softening blocked: {why} → full penalty respected")
                            scalping_signals.append("Supertrend bearish → Trend change")
                            
                    elif st_signal != 'neutral':
                        active_count += 1


                weighted_score += contribution
                

                val = value
                if ind_name == 'bollinger':
                    val = indicator.get('position', 0.5)
                    
                contributions[ind_name] = {
                    'signal': indicator.get('signal', 'neutral'),
                    'value': val,
                    'weight': weight,
                    'contribution': contribution
                }
                if extras:
                    contributions[ind_name].update(extras)


            # Microstructure: adjust score slightly for last-3-candle thrust before classification
            try:
                if hasattr(self, 'current_df') and not self.current_df.empty and len(self.current_df) >= 3:
                    recent = self.current_df.tail(3).copy()
                    rng = (recent['high'] - recent['low']).replace(0, np.nan)
                    body = (recent['close'] - recent['open']).abs()
                    body_ratio = (body / rng).fillna(0)
                    strong = body_ratio > self.config.price_action_min_body_ratio
                    strong_bull = int(((recent['close'] > recent['open']) & strong).sum())
                    strong_bear = int(((recent['close'] < recent['open']) & strong).sum())
                    if strong_bull >= 2:
                        weighted_score += 0.05
                        scalping_signals.append("PA microstructure: 2/3 strong green")
                    elif strong_bear >= 2:
                        weighted_score -= 0.05
                        scalping_signals.append("PA microstructure: 2/3 strong red")
            except Exception:
                pass

            # Structure bias (last 3 bars): penalize BUY on lower-highs, SELL on higher-lows
            try:
                if hasattr(self, 'current_df') and not self.current_df.empty and len(self.current_df) >= 3:
                    last3 = self.current_df.tail(3)
                    highs = last3['high'].to_numpy(dtype=float)
                    lows = last3['low'].to_numpy(dtype=float)
                    lower_highs = bool(np.all(np.diff(highs) < 0))
                    higher_lows = bool(np.all(np.diff(lows) > 0))
                    if lower_highs and weighted_score > 0:
                        weighted_score -= 0.08
                        scalping_signals.append("Structure bias: lower-highs")
                    elif higher_lows and weighted_score < 0:
                        weighted_score += 0.08
                        scalping_signals.append("Structure bias: higher-lows")
            except Exception:
                pass




            # Supply/Demand zone context (swing-based; confirmation-only)
            try:
                if getattr(self.config, 'enable_supply_demand_integration', True) and hasattr(self, 'current_df') and self.current_df is not None and len(self.current_df) >= 30 and self.resistance_detector:
                    df_sd = self.current_df.tail(150).copy()
                    levels_info = self.resistance_detector.detect_levels(df_sd, lookback=min(100, len(df_sd)))
                    curr = float(levels_info.get('current_price', float(df_sd['close'].iloc[-1])))
                    lv = levels_info.get('levels', {}) or {}
                    
                    strong_res = [float(x) for x in lv.get('strong_resistance', []) if np.isfinite(x)]
                    mod_res = [float(x) for x in lv.get('moderate_resistance', []) if np.isfinite(x)]
                    strong_sup = [float(x) for x in lv.get('strong_support', []) if np.isfinite(x)]
                    mod_sup = [float(x) for x in lv.get('moderate_support', []) if np.isfinite(x)]

                    def _bps(a, b):
                        try:
                            denom = max(abs(b), 1e-9)
                            return abs((a - b) / denom) * 10_000.0
                        except Exception:
                            return float('inf')

                    up_levels = sorted([x for x in (strong_res + mod_res) if x >= curr])
                    dn_levels = sorted([x for x in (strong_sup + mod_sup) if x <= curr], reverse=True)

                    nearest_up = up_levels[0] if up_levels else None
                    nearest_dn = dn_levels[0] if dn_levels else None
                    up_bps = _bps(nearest_up, curr) if nearest_up is not None else float('inf')
                    dn_bps = _bps(nearest_dn, curr) if nearest_dn is not None else float('inf')

                    dist_bps = float(getattr(self.config, 'sd_zone_distance_bps', 8.0))
                    at_supply = bool(nearest_up is not None and up_bps <= dist_bps)
                    at_demand = bool(nearest_dn is not None and dn_bps <= dist_bps)

                    # Expose for telemetry/alerts
                    contributions['supply_demand'] = {
                        'at_supply': at_supply,
                        'at_demand': at_demand,
                        'nearest_up': nearest_up,
                        'nearest_dn': nearest_dn,
                        'up_bps': float(up_bps) if np.isfinite(up_bps) else None,
                        'dn_bps': float(dn_bps) if np.isfinite(dn_bps) else None,
                        'contribution': 0.0
                    }

                    # Tiny, bounded nudge/dampener
                    sd_boost = float(getattr(self.config, 'sd_context_boost', 0.03))
                    if at_demand and weighted_score > 0:
                        old = weighted_score
                        weighted_score += sd_boost
                        scalping_signals.append(f"S/D context (+): at demand (≤{dist_bps:.1f} bps)")
                        logger.info("[S/D] Demand zone boost: price≈%.2f within %.1f bps of support → %.3f→%.3f",
                                    curr, dn_bps, old, weighted_score)
                        
                    elif at_supply and weighted_score < 0:
                        old = weighted_score
                        weighted_score -= sd_boost
                        scalping_signals.append(f"S/D context (-): at supply (≤{dist_bps:.1f} bps)")
                        logger.info("[S/D] Supply zone boost: price≈%.2f within %.1f bps of resistance → %.3f→%.3f",
                                    curr, up_bps, old, weighted_score)
                        
                    else:
                        if at_supply and weighted_score > 0:
                            old = weighted_score
                            weighted_score -= (sd_boost * 0.5)
                            scalping_signals.append("S/D dampener: BUY into supply")
                            logger.info("[S/D] Dampener: BUY into supply → %.3f→%.3f", old, weighted_score)
                            
                        if at_demand and weighted_score < 0:
                            old = weighted_score
                            weighted_score += (sd_boost * 0.5)
                            scalping_signals.append("S/D dampener: SELL into demand")
                            logger.info("[S/D] Dampener: SELL into demand → %.3f→%.3f", old, weighted_score)
                            
            except Exception as e:
                logger.debug(f"[S/D] Context skipped: {e}")




            # S/R proximity penalty to align prelim with HTF room
            try:
                bb_pos = float(contributions.get('bollinger', {}).get('position', 0.5))
                if weighted_score > 0 and bb_pos >= 0.90:
                    weighted_score -= 0.06
                    scalping_signals.append("S/R proximity penalty: near upper while BUY")
                    logger.info(f"S/R penalty applied (BUY): bb_pos={bb_pos:.2f} → score={weighted_score:.3f}")
                elif weighted_score < 0 and bb_pos <= 0.10:
                    weighted_score += 0.06
                    scalping_signals.append("S/R proximity penalty: near lower while SELL")
                    logger.info(f"S/R penalty applied (SELL): bb_pos={bb_pos:.2f} → score={weighted_score:.3f}")
            except Exception:
                pass


            # Extra SR+RSI nuance to avoid buying top/selling bottom in ranges (context-aware)
            try:
                rsi_val = float(contributions.get('rsi', {}).get('rsi_value', 50.0))
                bb_pos = float(contributions.get('bollinger', {}).get('position', 0.5))
                sr_room = str(contributions.get('_ctx_sr_room', getattr(self.mtf_analyzer, '_last_sr_room', 'UNKNOWN')) or 'UNKNOWN').upper()



                macd_slope = float(self._get_value(indicators, 'macd', 'hist_slope_closed', default=self._get_value(contributions, 'macd', 'hist_slope', default=0.0)) or 0.0)
                if pd.isna(macd_slope) or np.isinf(macd_slope):
                    macd_slope = 0.0
                logger.info("[SLOPE] MACD closed slope used for gating: %+.6f", macd_slope)


                
                if pd.isna(macd_slope) or np.isinf(macd_slope):
                    macd_slope = 0.0

                # BUY near top: stronger haircut when LIMITED room or momentum not supportive
                if weighted_score > 0 and (bb_pos >= 0.90 or rsi_val >= 70):
                    old = weighted_score
                    penalty = 0.025
                    if sr_room == 'LIMITED':
                        penalty += 0.035
                    if macd_slope <= 0:
                        penalty += 0.020
                    weighted_score -= penalty
                    scalping_signals.append("SR+RSI nuance: BUY near upper → haircut (ctx-aware)")
                    logger.info(f"SR+RSI BUY haircut: bb_pos={bb_pos:.2f}, rsi={rsi_val:.1f}, sr={sr_room}, slope={macd_slope:+.6f} → {old:+.3f}→{weighted_score:+.3f}")

                # SELL near bottom: stronger haircut when LIMITED room or slope ≥ 0 (bounce risk)
                elif weighted_score < 0 and (bb_pos <= 0.10 or rsi_val <= 30):
                    old = weighted_score
                    boost = 0.025
                    if sr_room == 'LIMITED':
                        boost += 0.035
                    if macd_slope >= 0:
                        boost += 0.020
                    if rsi_val <= 35 and sr_room == 'LIMITED':
                        boost += 0.020
                    weighted_score += boost
                    scalping_signals.append("SR+RSI nuance: SELL near lower → haircut (ctx-aware)")
                    logger.info(f"SR+RSI SELL haircut: bb_pos={bb_pos:.2f}, rsi={rsi_val:.1f}, sr={sr_room}, slope={macd_slope:+.6f} → {old:+.3f}→{weighted_score:+.3f}")
            except Exception:
                pass




            # HTF top/bottom haircut to align prelim with range room (before pattern boost)
            try:
                if weighted_score > 0 and getattr(self, '_extreme_top', False):
                    weighted_score -= 0.12
                    scalping_signals.append("HTF guard haircut: 15m top-of-range (BUY)")
                    logger.info(f"HTF top-range guard applied → score={weighted_score:.3f}")
                elif weighted_score < 0 and getattr(self, '_extreme_bottom', False):
                    weighted_score += 0.12
                    scalping_signals.append("HTF guard haircut: 15m bottom-of-range (SELL)")
                    logger.info(f"HTF bottom-range guard applied → score={weighted_score:.3f}")
            except Exception:
                pass



            # Low-breadth attenuation: if <3 indicators active, shrink the score
            # if active_count < 3:
            #     weighted_score *= 0.8

            # Low-breadth quality gate and attenuation
            if active_count < 3:
                bull_bias = weighted_score > 0

                macd_slope = float(contributions.get('macd', {}).get('hist_slope', 0.0))
                if pd.isna(macd_slope) or np.isinf(macd_slope):
                    macd_slope = 0.0
                                
                price_above_short = bool(contributions.get('ema', {}).get('price_above_short', False))
                rsi_val = float(contributions.get('rsi', {}).get('rsi_value', 50.0))
                bb_pos = float(contributions.get('bollinger', {}).get('position', 0.5))
                quality_penalty = 0.0

                if bull_bias:
                    if macd_slope <= 0:
                        quality_penalty += 0.06
                    if not price_above_short:
                        quality_penalty += 0.06
                    if rsi_val <= 50:
                        quality_penalty += 0.04
                    if bb_pos < 0.5:
                        quality_penalty += 0.03
                    if quality_penalty > 0:
                        weighted_score -= quality_penalty
                        scalping_signals.append("Low breadth BUY quality penalty")
                        logger.debug(f"Low-breadth BUY penalty applied: {quality_penalty:.3f}")
                else:
                    if macd_slope >= 0:
                        quality_penalty += 0.06
                    if price_above_short:
                        quality_penalty += 0.06
                    if rsi_val >= 50:
                        quality_penalty += 0.04
                    if bb_pos > 0.5:
                        quality_penalty += 0.03
                    if quality_penalty > 0:
                        weighted_score += quality_penalty
                        scalping_signals.append("Low breadth SELL quality penalty")
                        logger.debug(f"Low-breadth SELL penalty applied: {quality_penalty:.3f}")

                # Stronger attenuation when only 1–2 indicators contribute
                weighted_score *= 0.7


            # OI context boost/penalty (confirmation-only; bounded and NaN-safe)
            try:
                oi_ind = indicators.get('oi', {}) if isinstance(indicators, dict) else {}
                oi_sig = str(oi_ind.get('signal', 'neutral')).lower()
                oi_pct = float(oi_ind.get('oi_change_pct', 0.0))
                if pd.isna(oi_pct) or np.isinf(oi_pct):
                    oi_pct = 0.0
                
                if getattr(self.config, 'enable_oi_integration', True):
                    ctx_boost = float(getattr(self.config, 'oi_context_boost', 0.03))
                    min_pct = float(getattr(self.config, 'oi_min_change_pct', 0.10))  # percent
                    
                    if abs(oi_pct) >= min_pct:
                        if oi_sig in ('long_build_up', 'short_covering'):
                            old = weighted_score
                            bump = ctx_boost if oi_sig == 'long_build_up' else (ctx_boost * 0.5)
                            weighted_score += bump
                            scalping_signals.append(f"OI context (+): {oi_sig} (ΔOI%={oi_pct:.2f})")
                            logger.info(f"[OI] Bullish context → score {old:+.3f} → {weighted_score:+.3f} (ΔOI%={oi_pct:.2f})")
                            
                        elif oi_sig in ('short_build_up', 'long_unwinding'):
                            old = weighted_score
                            bump = ctx_boost if oi_sig == 'short_build_up' else (ctx_boost * 0.5)
                            weighted_score -= bump
                            scalping_signals.append(f"OI context (-): {oi_sig} (ΔOI%={oi_pct:.2f})")
                            logger.info(f"[OI] Bearish context → score {old:+.3f} → {weighted_score:+.3f} (ΔOI%={oi_pct:.2f})")
                            
                    
                    # Expose in contributions for telemetry/alerts
                    contributions['oi'] = {
                        'signal': oi_ind.get('signal', 'neutral'),
                        'oi': oi_ind.get('oi', 0),
                        'oi_change': oi_ind.get('oi_change', 0),
                        'oi_change_pct': oi_pct,
                        'contribution': 0.0
                    }
            except Exception:
                pass



            # Micro‑range neutralizer: low score, tight bands, and weak/contrary momentum → force NEUTRAL
            try:
                bb_bw = float(indicators.get('bollinger', {}).get('bandwidth', 0.0)) if isinstance(indicators, dict) else 0.0
                macd_hist = float(indicators.get('macd', {}).get('histogram', 0.0)) if isinstance(indicators, dict) else 0.0

                macd_slope = float(contributions.get('macd', {}).get('hist_slope', 0.0))
                if pd.isna(macd_slope) or np.isinf(macd_slope):
                    macd_slope = 0.0                

                # Expansion-aware neutralizer: detect once and annotate contributions

                try:
                    # Prefer the cached flag set earlier in analyze_and_generate_signal
                    exp_flag = getattr(self, "_last_expansion_flag", None)
                    if exp_flag is None:
                        exp_flag, _ = self._detect_expansion_lite(self.current_df, indicators)
                except Exception:
                    exp_flag = False
                if isinstance(contributions, dict):
                    contributions['_ctx_expansion'] = exp_flag



                                        
                # Expansion-aware micro-range handling: allow momentum+RSI-confirmed rebounds to avoid over-neutralizing
                rsi_val = float(contributions.get('rsi', {}).get('rsi_value', 50.0))
                rsi_up = bool(contributions.get('rsi', {}).get('rsi_cross_up', False))
                rsi_dn = bool(contributions.get('rsi', {}).get('rsi_cross_down', False))

                if abs(weighted_score) < 0.12 and bb_bw <= 8.0:
                    # BUY-side allowance: slightly negative score but momentum turns up with RSI support
                    if (weighted_score < 0 and macd_slope > 0 and (rsi_up or rsi_val >= 50.0)):
                        old = weighted_score
                        bump = 0.04 if exp_flag else 0.02
                        weighted_score = min(0.12, weighted_score + bump)
                        scalping_signals.append("Micro-range allowance: momentum+RSI (up) confirmation")
                        logger.info("[NEUTRALIZER] Skipped collapse (BUY-side allowance, expansion=%s) → %+.3f→%+.3f", exp_flag, old, weighted_score)
                        
                    # SELL-side allowance: slightly positive score but momentum turns down with RSI support
                    elif (weighted_score > 0 and macd_slope < 0 and (rsi_dn or rsi_val <= 50.0)):
                        old = weighted_score
                        bump = 0.04 if exp_flag else 0.02
                        weighted_score = max(-0.12, weighted_score - bump)
                        scalping_signals.append("Micro-range allowance: momentum+RSI (down) confirmation")
                        logger.info("[NEUTRALIZER] Skipped collapse (SELL-side allowance, expansion=%s) → %+.3f→%+.3f", exp_flag, old, weighted_score)
                        
                    # Otherwise apply the usual collapse
                    elif (abs(macd_hist) < 0.005 or (weighted_score > 0 and macd_slope <= 0) or (weighted_score < 0 and macd_slope >= 0)):
                        old = weighted_score
                        collapse = 0.35 if exp_flag else 0.20
                        weighted_score *= collapse
                        scalping_signals.append(f"Micro-range neutralizer ({'exp' if exp_flag else 'compr'}): tight BB + tiny/contrary MACD")
                        logger.info(f"[NEUTRALIZER] Micro-range → score {old:+.3f} → {weighted_score:+.3f} (bb_bw={bb_bw:.2f}, macd_hist={macd_hist:+.4f}, slope={macd_slope:+.6f}, expansion={exp_flag})")
                        

                                    
                    
            except Exception:
                pass




            # Oversold-bounce guard: below lower KC + positive MACD momentum + sub-35 RSI + low BB position
            try:
                kc_sig = str(indicators.get('keltner', {}).get('signal', 'within')).lower() if isinstance(indicators, dict) else 'within'
                macd_hist = float(indicators.get('macd', {}).get('histogram', 0.0)) if isinstance(indicators, dict) else 0.0
                macd_slope = float(contributions.get('macd', {}).get('hist_slope', 0.0))
                rsi_val = float(contributions.get('rsi', {}).get('rsi_value', 50.0))
                bb_pos = float(contributions.get('bollinger', {}).get('position', 0.5))
                
                if (weighted_score < 0 and 
                    kc_sig == 'below_lower' and 
                    macd_hist > 0 and 
                    macd_slope > 0 and 
                    rsi_val <= 35.0 and 
                    bb_pos <= 0.40):
                    
                    old = weighted_score
                    weighted_score = max(weighted_score, -0.08)  # soften into borderline SELL
                    scalping_signals.append("Bounce guard: below KC + MACD↑ + RSI<=35 → SELL softened")
                    logger.info("Bounce guard applied → %.3f→%.3f (KC=below_lower, MACD_hist=%.4f, slope=%+.6f, RSI=%.1f, BBpos=%.2f)",
                                old, weighted_score, macd_hist, macd_slope, rsi_val, bb_pos)
            except Exception as e:
                logger.debug(f"Bounce guard skipped: {e}")



    
            # ADJUST THRESHOLDS BASED ON MARKET REGIME
            if market_regime == "STRONG_UPTREND":
                buy_threshold = 0.05  # Lower threshold for buys
                sell_threshold = -0.25  # Higher threshold for sells
            elif market_regime == "STRONG_DOWNTREND":
                buy_threshold = 0.25
                sell_threshold = -0.05
            else:
                buy_threshold = 0.1
                sell_threshold = -0.1



            # Classify direction only (do NOT build static text here)
            if weighted_score < (sell_threshold - 0.1):
                composite_signal = 'STRONG_SELL'
            elif weighted_score < sell_threshold:
                composite_signal = 'SELL'
            elif weighted_score > (buy_threshold + 0.1):
                composite_signal = 'STRONG_BUY'
            elif weighted_score > buy_threshold:
                composite_signal = 'BUY'
            else:
                composite_signal = 'NEUTRAL'

            # Build dynamic, volatility-aware prediction text + meta
            next_candle_prediction, prediction_meta = self._build_dynamic_prediction(
                composite_signal=composite_signal,
                weighted_score=weighted_score,
                active_count=active_count,
                contributions=contributions,
                market_regime=market_regime
            )

  
            # Guard against NaN/Inf before downstream usage
            if pd.isna(weighted_score) or np.isinf(weighted_score):
                logger.error("Invalid weighted_score detected: %.3f, resetting to 0", weighted_score)
                weighted_score = 0.0

            
            # Include pattern detection in signal (HTF/SR-aware gating)
            if self.pattern_detector and hasattr(self, 'current_df') and not self.current_df.empty:
                try:
                    pattern_result = self.pattern_detector.detect_patterns(self.current_df)


                    # Attach and log the top pattern explicitly for audit and alerts (no re-bind)
                    if pattern_result:
                        contributions['pattern_top'] = {
                            'name': str(pattern_result.get('name', '')).upper(),
                            'signal': str(pattern_result.get('signal', 'NEUTRAL')).upper(),
                            'confidence': int(pattern_result.get('confidence', 0))
                        }
                        logger.info("[PATTERN] Recognized: %s (%s, %d%%)",
                                    contributions['pattern_top']['name'],
                                    contributions['pattern_top']['signal'],
                                    contributions['pattern_top']['confidence'])
                        



                    # Apply pattern boost or ignore based on context
                    if pattern_result and pattern_result.get('confidence', 0) > 0:
                        pattern_signal = pattern_result.get('signal', 'neutral')
                        htf_macd_sig = str(getattr(self, '_htf_macd_sig', '') or '').lower()
                        bb_pos = float(contributions.get('bollinger', {}).get('position', 0.5))

                        top_guard = bool(getattr(self, '_extreme_top', False))
                        bottom_guard = bool(getattr(self, '_extreme_bottom', False))
                        hostile_buy_ctx = ('bearish' in htf_macd_sig) or (bb_pos >= 0.90) or top_guard
                        hostile_sell_ctx = ('bullish' in htf_macd_sig) or (bb_pos <= 0.10) or bottom_guard

                        if pattern_signal == 'LONG' and weighted_score > 0:
                            if hostile_buy_ctx:
                                logger.info("Pattern detected but ignored (HTF hostile or near upper band): %s | htf_macd=%s, bb_pos=%.2f",
                                            str(pattern_result.get('name', '')).upper(), htf_macd_sig, bb_pos)
                                
                            else:
                               
                                                                
                                # Location-quality and OI synergy
                                pat_name = str(pattern_result.get('name', '')).upper()
                                pat_dir = str(pattern_result.get('signal', 'NEUTRAL')).upper()
                                oi_ctx = contributions.get('oi', {}) if isinstance(contributions, dict) else {}
                                oi_sig = str(oi_ctx.get('signal', 'neutral')).lower()

                                loc_ok = True
                                if getattr(self.config, 'enable_pattern_location_quality', True):
                                    # Reversal: prefer extremes; Continuation: prefer mid-band with trend alignment
                                    if pat_dir == 'LONG':
                                        loc_ok = (bb_pos <= 0.35)
                                    elif pat_dir == 'SHORT':
                                        loc_ok = (bb_pos >= 0.65)

                                # OI synergy (light)
                                oi_ok = (pat_dir == 'LONG' and oi_sig in ('long_build_up','short_covering')) or (pat_dir == 'SHORT' and oi_sig in ('short_build_up','long_unwinding'))

                                boost = 0.05
                                if not loc_ok:
                                    boost *= 0.5
                                if oi_ok:
                                    boost *= 1.2  # small bump with OI confirmation



                                # Location-aware confirmation: skip pattern boost when location is poor
                                try:
                                    bb_pos = float(indicators.get('bollinger', {}).get('position', 0.5)) if isinstance(indicators, dict) else 0.5
                                except Exception:
                                    bb_pos = 0.5
                                # Resolve last SR-room from MTF for this pass (if exposed)
                                try:
                                    sr_room = str(getattr(self.mtf_analyzer, '_last_sr_room', 'UNKNOWN'))
                                except Exception:
                                    sr_room = 'UNKNOWN'
                                pattern_allowed = bool(loc_ok and not (sr_room == 'LIMITED' and bb_pos >= 0.65))
                                if not pattern_allowed:
                                    logger.info("[PATTERN] boost skipped: poor location (loc_ok=%s, bb_pos=%.2f, sr=%s)", loc_ok, bb_pos, sr_room)
                                else:
                                    logger.info("[PATTERN] boost applied: %s (boost=%.3f, loc_ok=%s, oi_ok=%s)", pat_name, boost, loc_ok, oi_ok)
                                    weighted_score += float(boost)
                                    signal_result['pattern_boost_applied'] = True


                                

                                # Mark applied for post-hoc analysis
                                contributions['pattern_boost_applied'] = True
                                
                                
                                
                                
                        elif pattern_signal == 'SHORT' and weighted_score < 0:
                            if hostile_sell_ctx:
                                logger.info("Pattern detected but ignored (HTF hostile or near lower band): %s | htf_macd=%s, bb_pos=%.2f",
                                            str(pattern_result.get('name', '')).upper(), htf_macd_sig, bb_pos)
                                
                            else:
                                


                                pat_name = str(pattern_result.get('name', '')).upper()
                                pat_dir = str(pattern_result.get('signal', 'NEUTRAL')).upper()
                                oi_ctx = contributions.get('oi', {}) if isinstance(contributions, dict) else {}
                                oi_sig = str(oi_ctx.get('signal', 'neutral')).lower()
                                bb_pos = float(contributions.get('bollinger', {}).get('position', 0.5))

                                loc_ok = True
                                if getattr(self.config, 'enable_pattern_location_quality', True):
                                    loc_ok = (bb_pos >= 0.65)

                                oi_ok = (oi_sig in ('short_build_up','long_unwinding'))

                                boost = 0.05
                                if not loc_ok:
                                    boost *= 0.5
                                if oi_ok:
                                    boost *= 1.2

                                weighted_score -= boost
                                scalping_signals.append(f"Pattern applied: {pat_name} (loc={'ok' if loc_ok else 'weak'}, oi={'ok' if oi_ok else 'neutral'})")
                                logger.info("Pattern boost applied: %s (boost=%.3f, loc_ok=%s, oi_ok=%s)", pat_name, boost, loc_ok, oi_ok)
                                

                                contributions['pattern_boost_applied'] = True

                                
                                
                        else:
                            
                            logger.info("[PATTERN] ignored: contradicts 5m bias → %s", str(pattern_result.get('name', '')).upper())
                            
                            # Record ignored counter-pattern for nuanced adjustments
                            contributions['pattern_ignored'] = {
                                'name': pattern_result.get('name'),
                                'signal': pattern_signal,
                                'confidence': pattern_result.get('confidence', 0)
                            }

                except Exception as e:
                    logger.debug(f"Pattern detection skipped: {e}")


            # Re-classify after pattern/quality adjustments to keep mapping consistent
            prev_signal = composite_signal
            if weighted_score < (sell_threshold - 0.1):
                composite_signal = 'STRONG_SELL'
            elif weighted_score < sell_threshold:
                composite_signal = 'SELL'
            elif weighted_score > (buy_threshold + 0.1):
                composite_signal = 'STRONG_BUY'
            elif weighted_score > buy_threshold:
                composite_signal = 'BUY'
            else:
                composite_signal = 'NEUTRAL'

            logger.info("[SETUP] Detectors executing (post-pattern block): score=%+.3f sig=%s", weighted_score, composite_signal)



            # Pivot Swipe Structure
            try:
                ps = self.detect_pivot_swipe(self.current_df, indicators, self.config)
                if ps.get('detected'):
                    contributions['pivot_swipe'] = { 'direction': ps['direction'], 'level': ps['level'], 'name': ps['level_name'], 'contribution': 0.0 }
                    w = float(getattr(self.config, 'pivot_swipe_weight', 0.06))
                    if ('LONG' in ps['direction'] and weighted_score > 0) or ('SHORT' in ps['direction'] and weighted_score < 0):
                        old = weighted_score; weighted_score += (w if weighted_score > 0 else -w)
                        scalping_signals.append(f"Pivot swipe @ {ps['level_name']} ({ps['direction']})")
                        logger.info("[SWIPE] %+0.3f → %+0.3f (%s @ %s)", old, weighted_score, ps['direction'], ps['level_name'])
                    else:

                        try:
                            sess_info = getattr(self, '_cached_session_info', None)
                            sess_name = str(sess_info.get('session', 'unknown')) if isinstance(sess_info, dict) else 'unknown'
                        except Exception:
                            sess_name = 'unknown'
                        if sess_name == 'ranging':
                               
                            old = weighted_score; weighted_score *= 0.9
                            scalping_signals.append("Swipe contradiction in range → haircut")
                            logger.info("[SWIPE] Contradict (range) → %.3f→%.3f", old, weighted_score)
            except Exception:
                pass

            # Imbalance Structure
            try:
                imb = self.detect_imbalance_structure(self.current_df, indicators, self.config)
                if imb.get('detected'):
                    contributions['imbalance'] = { 'direction': imb['direction'], 'c2_body_ratio': imb['quality'].get('c2_body_ratio'), 'ema_widen_bps': imb['quality'].get('ema_widen_bps'), 'contribution': 0.0 }
                    w = float(getattr(self.config, 'imbalance_weight', 0.07))
                    if ('LONG' in imb['direction'] and weighted_score > 0) or ('SHORT' in imb['direction'] and weighted_score < 0):
                        old = weighted_score; weighted_score += (w if weighted_score > 0 else -w)
                        scalping_signals.append(f"Imbalance structure ({imb['direction']})")
                        logger.info("[IMB] %+0.3f → %+0.3f (%s)", old, weighted_score, imb['direction'])
                    else:
                        old = weighted_score; weighted_score *= 0.9
                        scalping_signals.append("Imbalance adverse → haircut")
                        logger.info("[IMB] Adverse haircut → %.3f→%.3f", old, weighted_score)
            except Exception:
                pass



            # CHECK IF COPIED CORRECTLY
            
            
            # Reclassify after setup nudges so classification matches the updated score
            try:
                if market_regime == "STRONG_UPTREND":
                    buy_threshold, sell_threshold = 0.05, -0.25
                elif market_regime == "STRONG_DOWNTREND":
                    buy_threshold, sell_threshold = 0.25, -0.05
                else:
                    buy_threshold, sell_threshold = 0.10, -0.10

                prev_sig = composite_signal
                if weighted_score < (sell_threshold - 0.1):
                    composite_signal = 'STRONG_SELL'
                elif weighted_score < sell_threshold:
                    composite_signal = 'SELL'
                elif weighted_score > (buy_threshold + 0.1):
                    composite_signal = 'STRONG_BUY'
                elif weighted_score > buy_threshold:
                    composite_signal = 'BUY'
                else:
                    composite_signal = 'NEUTRAL'
                if composite_signal != prev_sig:
                    logger.debug("Reclassified after setups: %s → %s (score=%.3f)", prev_sig, composite_signal, weighted_score)
            except Exception as e:
                logger.debug(f"Reclassify after setups skipped: {e}")


            # Rebuild dynamic prediction after reclassification
            try:
                contributions['_ctx_mtf_score'] = contributions.get('_ctx_mtf_score', None)  # leave if already set
                if contributions['_ctx_mtf_score'] is None:  # if MTF was not computed earlier, leave None
                    pass
                contributions['_ctx_sr_room'] = contributions.get('_ctx_sr_room', getattr(self.mtf_analyzer, '_last_sr_room', None))
            except Exception:
                pass

            next_candle_prediction, prediction_meta = self._build_dynamic_prediction(
                composite_signal=composite_signal,
                weighted_score=weighted_score,
                active_count=active_count,
                contributions=contributions,
                market_regime=market_regime
            )

            
            # Confidence calculation with market regime boost
            base_confidence = abs(weighted_score) * 100
           
            if active_count >= 4:
                # Protect against division by zero
                # denominator = active_count * 0.166
                denominator = max(0.001, active_count * 0.166)  # Prevent zero
                
                if denominator > 0:
                    agreement_ratio = abs(weighted_score) / denominator
                else:
                    agreement_ratio = 0.0

                if agreement_ratio > 0.6:
                    base_confidence += 25
                elif agreement_ratio > 0.4:
                    base_confidence += 15
            
            # Market regime confidence boost
            if market_regime in ["STRONG_UPTREND", "STRONG_DOWNTREND"]:
                if (market_regime == "STRONG_UPTREND" and weighted_score > 0) or \
                (market_regime == "STRONG_DOWNTREND" and weighted_score < 0):
                    base_confidence += 15  # Boost for trend-aligned signals
                    next_candle_prediction += f" [TREND DAY - {market_regime}]"

            confidence = min(100, max(base_confidence, 20))
            if pd.isna(confidence) or np.isinf(confidence):
                logger.error(f"Invalid confidence detected: {confidence}, resetting to 0")
                confidence = 0.0
                        
            logger.info(f"Market Regime: {market_regime}")
            logger.info(f"Weighted Score: {weighted_score:.3f}")
            logger.info(f"Active Indicators: {active_count}/{len(self.config.indicator_weights)}")
            logger.info(f"Confidence: {confidence:.1f}%")
            logger.info(f"Preliminary prediction (subject to PA/MTF): {next_candle_prediction}")




            # Pattern already attached earlier in this scorer pass — skip duplicate work
            logger.debug("[PATTERN] Duplicate attach skipped (already attached earlier in scorer)")
            


            # Note: result[] dict writes skipped — returning explicit dict below
            logger.debug("[SCORER] finalize: using explicit return dict (result[] unchanged)")
            

                            

            return {
                'weighted_score': weighted_score,
                'composite_signal': composite_signal,
                'confidence': confidence,
                'active_indicators': active_count,
                'contributions': contributions,
                'next_candle_prediction': next_candle_prediction,
                'prediction_meta': prediction_meta,  # NEW
                'scalping_signals': scalping_signals,
                'action_recommendation': self._get_scalping_action(weighted_score),
                'market_regime': market_regime,
                'extreme_context': {
                    'htf_price_pos': getattr(self, '_htf_price_pos', None),
                    'top_extreme': getattr(self, '_extreme_top', False),
                    'bottom_extreme': getattr(self, '_extreme_bottom', False),
                    'overbought': getattr(self, '_extreme_overbought', False),
                    'oversold': getattr(self, '_extreme_oversold', False),
                    'breakout_evidence': self._detect_breakout_evidence(self.current_df, indicators),
                    'breakdown_evidence': self._detect_breakdown_evidence(self.current_df, indicators),
                    
                }
            }



            
        except Exception as e:
            logger.error(f"Weighted signal calculation error: {e}")
            return {
                'weighted_score': 0,
                'composite_signal': 'NEUTRAL',
                'confidence': 0,
                'active_indicators': 0,
                'contributions': {},
                'next_candle_prediction': "Error in calculation",
                'scalping_signals': [],
                'market_regime': 'UNKNOWN'
            }



    def _build_dynamic_prediction(
        self,
        composite_signal: str,
        weighted_score: float,
        active_count: int,
        contributions: Dict,
        market_regime: str
    ) -> tuple[str, Dict[str, Any]]:
        """
        Build a probabilistic, volatility-aware next-candle prediction.
        Returns (text, meta_dict).
        """
            
            
        try:
            # 1) Directional probability from score, breadth, and momentum features
            # Score → base prob via smooth squashing; gentler base to reduce saturation on 5m
            ws = float(weighted_score)
            score_mag = float(min(1.0, max(0.0, abs(ws))))
            base = 0.5 + 0.35 * np.tanh(ws / 0.18)

            # Momentum adds a small, sign-aware tilt
            macd_slope = float(contributions.get('macd', {}).get('hist_slope', 0.0))

            if pd.isna(macd_slope) or np.isinf(macd_slope):
                macd_slope = 0.0
                                
            rsi_val = float(contributions.get('rsi', {}).get('rsi_value', 50.0))
            rsi_up = bool(contributions.get('rsi', {}).get('rsi_cross_up', False))
            rsi_dn = bool(contributions.get('rsi', {}).get('rsi_cross_down', False))

            mom_bonus = 0.0
            if 'BUY' in composite_signal:
                if (macd_slope > 0) or (rsi_val > 50.0) or rsi_up:
                    mom_bonus += 0.05
                if macd_slope > 0:
                    mom_bonus += 0.03
                if rsi_up:
                    mom_bonus += 0.02
            elif 'SELL' in composite_signal:
                if (macd_slope < 0) or (rsi_val < 50.0) or rsi_dn:
                    mom_bonus += 0.05
                if macd_slope < 0:
                    mom_bonus += 0.03
                if rsi_dn:
                    mom_bonus += 0.02

            # Breadth (active indicators) → reliability tilt
            breadth_bonus = min(0.10, max(0.0, (int(active_count) - 3) * 0.03))  # 0@<=3, up to +0.10

            # Regime tilt
            regime_bonus = 0.0
            if market_regime == "STRONG_UPTREND" and 'BUY' in composite_signal:
                regime_bonus = 0.03
            elif market_regime == "STRONG_DOWNTREND" and 'SELL' in composite_signal:
                regime_bonus = 0.03

            # Compose probability of GREEN (prob_up) canonically
            prob_up = float(base)
            if 'BUY' in composite_signal:
                prob_up = base + mom_bonus + breadth_bonus + regime_bonus
            elif 'SELL' in composite_signal:
                # For sell, probability of red = 1 - prob_up with a sign-aware tilt
                prob_red = (1.0 - base) + mom_bonus + breadth_bonus + regime_bonus
                prob_up = 1.0 - prob_red
            else:
                # Neutral — snap back to 0.5 with tiny band
                prob_up = 0.5 + 0.05 * np.tanh(ws / 0.10)
                

            # Raw probability for meta
            prob_up_raw = max(0.0, min(1.0, float(prob_up)))

            # PA tilt: small, structure-aware nudge for next candle (last 3 bars)
            try:
                if hasattr(self, 'current_df') and self.current_df is not None and len(self.current_df) >= 3:
                    last3 = self.current_df.tail(3)
                    highs = pd.to_numeric(last3['high'], errors='coerce').to_numpy(dtype=float)
                    lows = pd.to_numeric(last3['low'], errors='coerce').to_numpy(dtype=float)
                    bodies = (pd.to_numeric(last3['close'], errors='coerce') - pd.to_numeric(last3['open'], errors='coerce')).to_numpy(dtype=float)

                    higher_lows = bool(np.all(np.diff(lows) > 0))
                    lower_highs = bool(np.all(np.diff(highs) < 0))
                    green = int(np.sum(bodies > 0))
                    red = int(np.sum(bodies < 0))

                    pa_bull = (higher_lows or green >= 2)
                    pa_bear = (lower_highs or red >= 2)

                    # Only tilt neutral displays; keep it small and momentum‑aware
                    if 'NEUTRAL' in composite_signal:
                        if pa_bull and macd_slope > 0:
                            prob_up_raw = min(0.85, prob_up_raw + 0.05)
                        elif pa_bear and macd_slope < 0:
                            prob_up_raw = max(0.15, prob_up_raw - 0.05)
            except Exception:
                pass

            # Soft clamp for display
            prob_up_text = float(max(0.20, min(0.80, prob_up_raw)))


            
            # Display-only damping in weak contexts (optional ctx fields)
            try:
                ctx_mtf = contributions.get('_ctx_mtf_score', None)
                ctx_sr = str(contributions.get('_ctx_sr_room', '') or '').upper()
                if ctx_mtf is not None:
                    # Cap text at 60% when mtf<0.50 or SR room LIMITED
                    if float(ctx_mtf) < 0.50 or ctx_sr == 'LIMITED':
                        if prob_up_text > 0.60:
                            logger.info(f"[PRED-TEXT] Damped display prob due to weak ctx (mtf={ctx_mtf}, sr={ctx_sr}): {prob_up_text*100:.0f}% → 60%")
                        prob_up_text = min(prob_up_text, 0.60)
            except Exception:
                pass


            prob_red_text = float(1.0 - prob_up_text)
            prob_up = prob_up_text  # used only for text below (percent)
            prob_red = prob_red_text


            # 2) Expected move (pts) from live volatility
            try:
                vol_range = float(self._calculate_volatility_range(self.current_df)) if getattr(self, 'current_df', None) is not None else 0.0
            except Exception:
                vol_range = 0.0

            # 3) Scale expected move by score magnitude and breadth
            strength_scale = 0.6 + 0.6 * score_mag  # 0.6..1.2
            breadth_scale = 0.85 + 0.05 * max(0, int(active_count) - 3)  # ~0.85..1.1
            exp_move = max(2.0, vol_range * strength_scale * breadth_scale)  # floor to 2 pts to avoid "0 pt" messages


            # 4) Build text (you likely already have this below; keep only one copy)
            if 'BUY' in composite_signal:
                lo = int(round(exp_move * 0.6))
                hi = int(round(exp_move * 1.2))
                text = f"Next 5m: {int(round(prob_up*100))}% GREEN | exp +{lo} to +{hi} pts"
            elif 'SELL' in composite_signal:
                lo = int(round(exp_move * 0.6))
                hi = int(round(exp_move * 1.2))
                text = f"Next 5m: {int(round(prob_red*100))}% RED | exp -{lo} to -{hi} pts"
            else:
                lo = int(round(exp_move * 0.4))
                hi = int(round(exp_move * 0.8))
                text = f"Next 5m: Range likely | ±{lo} to ±{hi} pts"

            meta = {
                'prob_up': round(prob_up_raw, 4),
                'prob_red': round(1.0 - prob_up_raw, 4),
                'exp_move_pts': {'lo': lo, 'hi': hi},
                'score_mag': round(score_mag, 3),
                'active_indicators': int(active_count),
                'macd_slope': round(macd_slope, 6),
                'rsi': round(rsi_val, 1),
                'rsi_cross_up': bool(rsi_up),
                'rsi_cross_down': bool(rsi_dn),
                'market_regime': market_regime
            }
            return text, meta

        except Exception as e:
            logger.debug(f"Dynamic prediction failed, fallback to static: {e}")
            # Safe fallback to a minimal neutral text
            return "Next 5m: Uncertain | ±5 to ±10 pts", {
                'prob_up': 0.5, 'prob_red': 0.5, 'exp_move_pts': {'lo': 5, 'hi': 10}
            }



    def _get_scalping_action(self, score: float) -> str:
        """Get specific scalping action based on score."""
        if score < -0.15:
            return "SHORT NOW with 15-point stop, target -25 points"
        elif score < -0.05:
            return "SHORT with tight 10-point stop, target -20 points"
        elif score > 0.15:
            return "LONG NOW with 15-point stop, target +25 points"
        elif score > 0.05:
            return "LONG with tight 10-point stop, target +20 points"
        else:
            return "WAIT - No clear scalping opportunity"


    def detect_market_regime(self, df: pd.DataFrame) -> str:
        """Detect if market is trending or ranging - CRITICAL for index trading."""
        try:
            if len(df) < 50:
                return "NORMAL"
            
            # Calculate multiple timeframe MAs
            ma_20 = df['close'].rolling(20).mean()
            ma_50 = df['close'].rolling(50).mean() if len(df) >= 50 else ma_20
            
            price = df['close'].iloc[-1]
            
            # Calculate trend strength
            if len(ma_20) > 10:
                ma_slope = (ma_20.iloc[-1] - ma_20.iloc[-10]) / ma_20.iloc[-10] * 100
                
                # Strong uptrend criteria (like today's chart)
                if ma_20.iloc[-1] > ma_50.iloc[-1] and price > ma_20.iloc[-1]:
                    if ma_slope > 0.1:  
                        logger.info(f"Market Regime: STRONG_UPTREND (slope: {ma_slope:.2f}%)")
                        return "STRONG_UPTREND"
                
                # Strong downtrend
                elif ma_20.iloc[-1] < ma_50.iloc[-1] and price < ma_20.iloc[-1]:
                    if ma_slope < -0.2:
                        logger.info(f"Market Regime: STRONG_DOWNTREND (slope: {ma_slope:.2f}%)")
                        return "STRONG_DOWNTREND"
            
            logger.info("Market Regime: NORMAL (ranging/mild trend)")
            return "NORMAL"
            
        except Exception as e:
            logger.error(f"Market regime detection error: {e}")
            return "NORMAL"




    def _enhanced_validation(
        self, 
        signal_result: Dict, 
        group_consensus: Dict, 
        trend_analysis: Dict,
        df: pd.DataFrame
    ) -> bool:
        """
        Enhanced validation with multiple quality gates.
        Returns True if signal passes all validation checks.
        """
        try:
            logger.info("=" * 60)
            logger.info("[VALIDATION] Starting enhanced validation")
            logger.info("=" * 60)
            
            signal_type = signal_result.get('composite_signal', 'NEUTRAL')
            weighted_score = signal_result.get('weighted_score', 0)
            active = signal_result.get('active_indicators', 0)
            
            logger.info(f"[VALIDATION] Signal: {signal_type}, Score: {weighted_score:+.3f}, Active: {active}/6")
            
            # Determine if signal is actionable (for strict breadth requirement)
            is_actionable = signal_type in ('BUY', 'STRONG_BUY', 'SELL', 'STRONG_SELL')
            
            # Gate 1: Minimum Activity (STRICT 4/6 for actionable)
            min_active_strict = int(getattr(self.config, 'min_active_indicators', 4))
            min_active_calib = int(getattr(self.config, 'min_active_indicators_calibration', 2))
            
            threshold_to_use = min_active_strict if is_actionable else min_active_calib
            
            if active < threshold_to_use:
                logger.warning(f"[VAL] ❌ Rejected: active={active} < required={threshold_to_use}")
                logger.info(f"[VAL] Breadth gate: Need {threshold_to_use}/6 indicators active")
                signal_result['rejection_reason'] = "min_active"
                return False
            
            logger.info(f"[VAL] ✅ Breadth OK: {active}/6 indicators active (threshold={threshold_to_use})")
            
            # Gate 2: Minimum Strength
            min_strength = float(getattr(self.config, 'min_signal_strength', 0.10))
            
            # Check for session-specific override
            override = signal_result.get('min_strength_override')
            if override is not None:
                min_strength = float(override)
                logger.info(f"[VAL] Using override min_strength: {min_strength:.3f}")
            
            if abs(weighted_score) < min_strength:
                logger.warning(f"[VAL] ❌ Rejected: |score|={abs(weighted_score):.3f} < min={min_strength:.3f}")
                signal_result['rejection_reason'] = "min_strength"
                return False
            
            logger.info(f"[VAL] ✅ Strength OK: |{weighted_score:.3f}| >= {min_strength:.3f}")
            
            # Gate 3: Breadth for Alerts (actionable signals)
            if is_actionable:
                if active < 4:
                    logger.warning(f"[VAL] ❌ Rejected: actionable signal needs 4+ active, got {active}")
                    signal_result['rejection_reason'] = "breadth_3_lt_4"
                    return False
                logger.info(f"[VAL] ✅ Alert breadth OK: {active}/6 >= 4")
            
            # Gate 4: Core Confirmations (EMA, MACD, Supertrend)
            contributions = signal_result.get('contributions', {})
            
            ema_sig = str(contributions.get('ema', {}).get('signal', 'neutral')).lower()
            macd_sig = str(contributions.get('macd', {}).get('signal_type', 'neutral')).lower()
            st_sig = str(contributions.get('supertrend', {}).get('signal', 'neutral')).lower()
            
            # Count bullish/bearish among core indicators
            core_bullish = sum([
                any(x in ema_sig for x in ['bullish', 'golden_cross', 'above']),
                'bullish' in macd_sig,
                'bullish' in st_sig
            ])
            
            core_bearish = sum([
                any(x in ema_sig for x in ['bearish', 'death_cross', 'below']),
                'bearish' in macd_sig,
                'bearish' in st_sig
            ])
            
            logger.info(f"[VAL] Core indicators: {core_bullish} bullish, {core_bearish} bearish")
            
            # MTF score for relaxed confirmation
            mtf_score = float(signal_result.get('mtf_score', 0.0))
            
            # Check if EMA+MACD agree (can substitute for 2/3 requirement)
            ema_macd_agree_bull = (any(x in ema_sig for x in ['bullish', 'golden_cross', 'above']) and 
                                   'bullish' in macd_sig)
            ema_macd_agree_bear = (any(x in ema_sig for x in ['bearish', 'death_cross', 'below']) and 
                                   'bearish' in macd_sig)
            
            if "BUY" in signal_type:
                # Need 2/3 bullish OR (1/3 with strong MTF) OR (EMA+MACD agree)
                if not (core_bullish >= 2 or 
                       (core_bullish >= 1 and mtf_score >= 0.7) or 
                       ema_macd_agree_bull):
                    logger.warning(f"[VAL] ❌ BUY rejected: insufficient core confirmations")
                    logger.info(f"[VAL] Core: {core_bullish}/3 bullish, MTF={mtf_score:.2f}, EMA+MACD={ema_macd_agree_bull}")
                    signal_result['rejection_reason'] = "core_confirms"
                    return False
                logger.info(f"[VAL] ✅ BUY core confirmations OK")
                
            elif "SELL" in signal_type:
                # Need 2/3 bearish OR (1/3 with strong MTF) OR (EMA+MACD agree)
                if not (core_bearish >= 2 or 
                       (core_bearish >= 1 and mtf_score >= 0.7) or 
                       ema_macd_agree_bear):
                    logger.warning(f"[VAL] ❌ SELL rejected: insufficient core confirmations")
                    logger.info(f"[VAL] Core: {core_bearish}/3 bearish, MTF={mtf_score:.2f}, EMA+MACD={ema_macd_agree_bear}")
                    signal_result['rejection_reason'] = "core_confirms"
                    return False
                logger.info(f"[VAL] ✅ SELL core confirmations OK")
            
            # Gate 5: Structure Guard
            # SELL: need low break OR price below short EMA OR negative slope
            # BUY: need high break OR price above short EMA OR positive slope
            if is_actionable and len(df) >= 3:
                last_3 = df.tail(3)
                recent_high = float(last_3['high'].max())
                recent_low = float(last_3['low'].min())
                current_high = float(df['high'].iloc[-1])
                current_low = float(df['low'].iloc[-1])
                
                price_above_short = bool(contributions.get('ema', {}).get('price_above_short', False))
                macd_slope = float(contributions.get('macd', {}).get('hist_slope', 0.0))
                
                if "SELL" in signal_type:
                    low_break = current_low < recent_low
                    structure_ok = low_break or (not price_above_short) or (macd_slope < 0)
                    
                    if not structure_ok:
                        logger.warning(f"[VAL] ❌ SELL structure guard: no low break, price above EMA, slope positive")
                        signal_result['rejection_reason'] = "structure_guard_sell"
                        return False
                    logger.info(f"[VAL] ✅ SELL structure OK (low_break={low_break}, price_below={not price_above_short}, slope<0={macd_slope<0})")
                    
                elif "BUY" in signal_type:
                    high_break = current_high > recent_high
                    structure_ok = high_break or price_above_short or (macd_slope > 0)
                    
                    if not structure_ok:
                        logger.warning(f"[VAL] ❌ BUY structure guard: no high break, price below EMA, slope negative")
                        signal_result['rejection_reason'] = "structure_guard_buy"
                        return False
                    logger.info(f"[VAL] ✅ BUY structure OK (high_break={high_break}, price_above={price_above_short}, slope>0={macd_slope>0})")
            
            logger.info("=" * 60)
            logger.info("[VALIDATION] ✅ All gates passed")
            logger.info("=" * 60)
            return True
            
        except Exception as e:
            logger.error(f"[VALIDATION] Error: {e}", exc_info=True)
            # Default to allowing signal if validation error
            return True


    
    def _check_cooldown(self, signal_result: Dict) -> bool:
        """Check if signal is in cooldown period."""
        try:
            if not self.last_alert_time:
                return True
            
            elapsed = (datetime.now() - self.last_alert_time).total_seconds()
            
            # Dynamic cooldown based on signal strength
            if 'STRONG' in signal_result['composite_signal']:
                cooldown = self.config.base_cooldown_seconds * self.config.strong_signal_cooldown_factor
            else:
                cooldown = self.config.base_cooldown_seconds
            
            return elapsed >= cooldown
            
        except Exception as e:
            logger.error(f"Cooldown check error: {e}")
            return True
            


    def _should_apply_rr_taper(self, confidence_pct: float, weighted_score: float, mtf_score: float, df: pd.DataFrame) -> bool:
        """
        Decide if we can relax the R:R floor (taper) based on confidence, momentum magnitude, MTF, and time-of-day.
        Uses configurable thresholds; emits high-visibility logs for audit.
        """
        try:
            ws_mag = abs(float(weighted_score))

            conf_soft_raw = float(getattr(self.config, 'rr_taper_confidence_soft_min',
                                        getattr(self.config, 'rr_taper_confidence_min', 0.75)))
            # Normalize to 0–1 if config provided percent (e.g., 64.0)
            conf_soft = (conf_soft_raw / 100.0) if conf_soft_raw > 1.0 else conf_soft_raw
            if conf_soft_raw > 1.0:
                logger.info("[RISK] Normalized rr_taper_confidence_soft_min from %.2f to %.2f (fraction)", conf_soft_raw, conf_soft)

            
            burst = float(getattr(self.config, 'rr_taper_burst_strength', 0.30))
            mtf_min_strong = float(getattr(self.config, 'rr_taper_mtf_min_strong', 0.70))
            # Base pathway: slightly relaxed conf gate + momentum magnitude
            base_condition = (float(confidence_pct) >= conf_soft and ws_mag >= (burst - 0.05))
            # Late-day pathway
            late_day = False
            try:
                ts = df.index[-1] if (df is not None and not df.empty) else None
                late_day = bool(ts and (ts.hour > 14 or (ts.hour == 14 and ts.minute >= 45)))
            except Exception:
                pass
            late_day_condition = late_day and (float(mtf_score) >= 0.70)
            # Strong-alignment pathway: allow taper with lower momentum when MTF is strong
            strong_alignment = (float(mtf_score) >= mtf_min_strong and ws_mag >= 0.20)
            ok = bool(base_condition or late_day_condition or strong_alignment)  # FIXED

            logger.info(
                "[RISK] RR-taper eligibility: base=%s late=%s strong=%s → %s "
                "(conf=%.2f, ws=%.3f, mtf=%.2f, conf_soft=%.2f, burst=%.2f, mtf_min=%.2f)",
                base_condition, late_day_condition, strong_alignment, ok,
                float(confidence_pct), float(weighted_score), float(mtf_score),
                conf_soft, burst, mtf_min_strong
            )
            return ok
        except Exception as e:
            logger.debug(f"[RISK] RR-taper eligibility check error: {e}", exc_info=True)
            return False


    def _rr_floor_by_regime(self, signal_result: dict, indicators: dict) -> float:
        """
        Compute regime-adaptive R:R floor. Uses config flags and a conservative 'tight-range' proxy from
        Bollinger bandwidth (<0.40) and market_regime. Falls back to min_risk_reward_floor.
        """
        try:
            if not getattr(self.config, 'rr_floor_use_adaptive', False):
                return float(getattr(self.config, 'min_risk_reward_floor', 1.0))

            base = float(getattr(self.config, 'min_risk_reward_floor', 1.0))
            rr_ranging = float(getattr(self.config, 'rr_floor_ranging', 0.60))
            rr_trending = float(getattr(self.config, 'rr_floor_trending', 1.00))

            # Tight-range proxy
            bb_bw = 999.0
            try:
                bb_bw = float((indicators or {}).get('bollinger', {}).get('bandwidth', 999.0))
            except Exception:
                pass

            regime = str(signal_result.get('market_regime', 'NORMAL')).upper()
            tight = (bb_bw < 0.40)  # your log slice shows 0.26–0.33, so 0.40 is a safe cutoff

            if tight and regime in ('NORMAL',):  # treat NORMAL+very tight bands as 'ranging'
                logger.info("[RISK] Tight-range detected (BB bw=%.2f) → adaptive R:R floor=%.2f", bb_bw, rr_ranging)
                return rr_ranging

            logger.info("[RISK] Regime floor (regime=%s, BB bw=%.2f) → floor=%.2f", regime, bb_bw, rr_trending)
            return rr_trending
        except Exception:
            return float(getattr(self.config, 'min_risk_reward_floor', 1.0))









    def _calculate_entry_exit_levels(
        self, 
        df: pd.DataFrame, 
        indicators: Dict, 
        signal_result: Dict
    ) -> Dict:
        """Calculate entry/exit levels using volatility instead of ATR."""


        # Micro-ATR for bursts (fallback if TA-Lib not present)
        def _micro_atr(frame: pd.DataFrame, period: int = 14) -> float:
            try:
                if frame is None or len(frame) < period + 1:
                    return 0.0
                high = pd.to_numeric(frame['high'].tail(period+1), errors='coerce')
                low = pd.to_numeric(frame['low'].tail(period+1), errors='coerce')
                close = pd.to_numeric(frame['close'].tail(period+1), errors='coerce')
                prev_close = close.shift(1).fillna(close.iloc[0])
                tr = pd.concat([
                    (high - low),
                    (high - prev_close).abs(),
                    (prev_close - low).abs()
                ], axis=1).max(axis=1)
                return float(tr.tail(period).mean())
            except Exception:
                return 0.0

            
        try:
            if df.empty:
                logger.warning("Empty dataframe for entry/exit calculation")
                return {
                    'entry_price': 0,
                    'stop_loss': 0,
                    'take_profit': 0,
                    'risk_reward': 0
                }
            
            current_price = float(df['close'].iloc[-1])
            logger.debug(f"Current price: {current_price:.2f}")
            
            # Get volatility-based range
            volatility_range = self._calculate_volatility_range(df)
            logger.debug(f"Volatility range: {volatility_range:.2f}")
            
            # Use indicator-informed bounds as raw anchors 
            support = current_price - volatility_range 
            resistance = current_price + volatility_range
            
            if 'bollinger' in indicators and isinstance(indicators['bollinger'], dict): 
                try: 
                    bb_lower = float(indicators['bollinger'].get('lower', support)) 
                    bb_upper = float(indicators['bollinger'].get('upper', resistance)) 
                    if np.isfinite(bb_lower) and bb_lower > 0: 
                        prev = support 
                        support = max(support, bb_lower) 
                        logger.debug(f"Bollinger lower applied: {bb_lower:.2f} (support {prev:.2f}→{support:.2f})") 
                    if np.isfinite(bb_upper) and bb_upper > 0: 
                        prev = resistance 
                        resistance = min(resistance, bb_upper) 
                        logger.debug(f"Bollinger upper applied: {bb_upper:.2f} (resistance {prev:.2f}→{resistance:.2f})") 
                except Exception as e: 
                    logger.debug(f"Bollinger bounds skipped: {e}")
            
            if 'keltner' in indicators and isinstance(indicators['keltner'], dict):
                try:
                    kc_lower = float(indicators['keltner'].get('lower', support))
                    kc_upper = float(indicators['keltner'].get('upper', resistance))
                    if np.isfinite(kc_lower) and kc_lower > 0:
                        prev = support
                        support = max(support, kc_lower)
                        logger.debug(f"Keltner lower applied: {kc_lower:.2f} (support {prev:.2f}→{support:.2f})")
                    if np.isfinite(kc_upper) and kc_upper > 0:
                        prev = resistance
                        resistance = min(resistance, kc_upper)
                        logger.debug(f"Keltner upper applied: {kc_upper:.2f} (resistance {prev:.2f}→{resistance:.2f})")
                except Exception as e:
                    logger.debug(f"Keltner bounds skipped: {e}")

            # Build directional nearest levels
            upper_candidates = [resistance, current_price + volatility_range]
            lower_candidates = [support, current_price - volatility_range]

            # Add debug logging for level selection
            logger.debug(f"Raw candidates: upper={upper_candidates}, lower={lower_candidates}")


            try:
                # Keep only valid floats
                upper_candidates = [float(x) for x in upper_candidates if np.isfinite(x)]
                lower_candidates = [float(x) for x in lower_candidates if np.isfinite(x)]
            except Exception:
                pass

            # Nearest levels in correct directions
            resistance_above = min([lvl for lvl in upper_candidates if lvl > current_price], default=current_price + max(1.0, volatility_range * 0.5))
            support_below = max([lvl for lvl in lower_candidates if lvl < current_price], default=current_price - max(1.0, volatility_range * 0.5))


            # Add debug logging for chosen bounds
            logger.debug(f"Chosen bounds: resistance_above={resistance_above:.2f}, support_below={support_below:.2f}")

            signal_type = str(signal_result.get('composite_signal', '')).upper()
            entry = current_price
            adjust_step = max(1.0, volatility_range * 0.25)  # sane fallback distance





            # Initialize floor with base; can be replaced by adaptive/taper later
            min_rr = float(getattr(self.config, 'min_risk_reward_floor', 1.0))
            rr_floor_note = "base"
            # --- Burst Sizing and Tapered R:R Floor Logic ---
            
            


            min_rr_floor = float(getattr(self.config, 'min_risk_reward_floor', 1.0))
            use_taper = bool(getattr(self.config, 'enable_rr_taper', True))
            burst_strength = float(getattr(self.config, 'rr_taper_burst_strength', 0.30))

            # Normalize conf_limit (MUST be before burst check)
            conf_limit_raw = float(getattr(self.config, 'rr_taper_confidence_min', 0.75))
            # Normalize to 0–1 if config provided percent (e.g., 64.0)
            conf_limit = (conf_limit_raw / 100.0) if conf_limit_raw > 1.0 else conf_limit_raw
            if conf_limit_raw > 1.0:
                logger.info("[RISK] Normalized rr_taper_confidence_min from %.2f to %.2f (fraction)", conf_limit_raw, conf_limit)

            confidence_pct = float(signal_result.get('confidence', 0.0)) / 100.0
            is_burst = (abs(float(signal_result.get('weighted_score', 0.0))) >= burst_strength)
            atr = _micro_atr(df, int(getattr(self.config, 'atr_period', 14)))



            
            if use_taper and is_burst and confidence_pct >= conf_limit and atr > 0:
                # Burst: ATR-based distances, keep min_rr as base for now (may taper later)
                sl_dist = float(getattr(self.config, 'atr_multiplier_sl', 0.6)) * atr
                tp_dist = float(getattr(self.config, 'atr_multiplier_tp', 1.2)) * atr
                rr_floor_note = "burst/ATR"
                logger.info(f"[RISK] Burst ATR sizing used: ATR={atr:.2f} SLd={sl_dist:.2f} TPd={tp_dist:.2f} (was vol_range={volatility_range:.2f})")
            else:
                # Non-burst: volatility-capped distances
                sl_dist = volatility_range * float(getattr(self.config, 'sl_volatility_cap_multiple', 1.0))
                tp_dist = volatility_range * float(getattr(self.config, 'tp_volatility_cap_multiple', 1.8))
                
                
                # Adaptive R:R floor (regime- and tight-range-aware)
                try:
                    rr_floor = self._rr_floor_by_regime(signal_result, indicators)
                    min_rr = float(rr_floor)  # apply adaptive floor now
                    rr_floor_note = "adaptive"
                    logger.info("[RISK] Regime-adaptive R:R floor=%.2f (base=%.2f)", min_rr, min_rr_floor)
                except Exception as e:
                    logger.debug(f"[RISK] Adaptive floor calc failed: {e}")
                    min_rr = min_rr_floor
                    rr_floor_note = "base"
                # Tight-range booster: extend TP when bands are compressed
                try:
                    bb_bandwidth = float((indicators or {}).get('bollinger', {}).get('bandwidth', 999.0))
                    if bb_bandwidth < 0.40:
                        tp_dist *= 1.50
                        logger.info("[RISK] Tight range (BB=%.2f) → TP extended by 50%%: %.2f", bb_bandwidth, tp_dist)
                except Exception as e:
                    logger.debug(f"[RISK] TP extension skipped: {e}")



            if "BUY" in signal_type:
                sl_uncapped = max(support_below, entry * (1 - self.config.stop_loss_percentage / 100.0))
                tp_uncapped = max(resistance_above, entry * (1 + self.config.take_profit_percentage / 100.0))
                

                # Optional ATR-based distances in tight ranges (non-burst path)
                try:
                    bb_bandwidth = float((indicators or {}).get('bollinger', {}).get('bandwidth', 999.0))
                    if bb_bandwidth < 0.40:
                        atr_val = float((indicators or {}).get('atr', {}).get('value', 0.0))
                        if atr_val > 0:
                            sl_dist = atr_val * 0.80
                            tp_dist = atr_val * 1.20
                            logger.info("[RISK] Ranging (BB tight) with ATR: SLd=%.2f TPd=%.2f", sl_dist, tp_dist)
                except Exception as e:
                    logger.debug(f"[RISK] Ranging ATR sizing skipped: {e}")

           
                
                stop_loss = max(entry - sl_dist, sl_uncapped)
                take_profit = min(entry + tp_dist, tp_uncapped)
                if stop_loss >= entry:
                    logger.info(f"Monotonicity fix (BUY): SL {stop_loss:.2f} >= entry {entry:.2f} → adjusting")
                    stop_loss = entry - adjust_step
                if take_profit <= entry:
                    logger.info(f"Monotonicity fix (BUY): TP {take_profit:.2f} <= entry {entry:.2f} → adjusting")
                    take_profit = entry + adjust_step
      
                    
                    
            elif "SELL" in signal_type:
                sl_uncapped = min(resistance_above, entry * (1 + self.config.stop_loss_percentage / 100.0))
                tp_uncapped = max(support_below, entry * (1 - self.config.take_profit_percentage / 100.0))
                

                # Optional ATR-based distances in tight ranges (non-burst path; SELL symmetry)
                try:
                    bb_bandwidth = float((indicators or {}).get('bollinger', {}).get('bandwidth', 999.0))
                    if bb_bandwidth < 0.40:
                        atr_val = float((indicators or {}).get('atr', {}).get('value', 0.0))
                        if atr_val > 0:
                            sl_dist = atr_val * 0.80
                            tp_dist = atr_val * 1.20
                            logger.info("[RISK] Ranging (BB tight) with ATR (SELL): SLd=%.2f TPd=%.2f", sl_dist, tp_dist)
                except Exception as e:
                    logger.debug(f"[RISK] Ranging ATR sizing skipped (SELL): {e}")

                
                stop_loss = min(entry + sl_dist, sl_uncapped)
                take_profit = max(entry - tp_dist, tp_uncapped)
                if stop_loss <= entry:
                    logger.info(f"Monotonicity fix (SELL): SL {stop_loss:.2f} <= entry {entry:.2f} → adjusting")
                    stop_loss = entry + adjust_step
                if take_profit >= entry:
                    logger.info(f"Monotonicity fix (SELL): TP {take_profit:.2f} >= entry {entry:.2f} → adjusting")
                    take_profit = entry - adjust_step
            else:
                stop_loss = entry - min(sl_dist, volatility_range)
                take_profit = entry + min(tp_dist, volatility_range)


            # Compute risk/reward
            risk = abs(entry - stop_loss)
            reward = abs(take_profit - entry)
            risk_reward = (reward / risk) if risk > 0 else 0.0
            # Taper eligibility (can override floor for bursts only)
            try:
                exp_flag = bool(signal_result.get('contributions', {}).get('_ctx_expansion', False))
                logger.info("[RISK] Taper precheck: is_burst=%s, conf_pct=%.2f, ws=%+.3f, mtf=%.2f, expansion=%s",
                            bool(is_burst), float(confidence_pct),
                            float(signal_result.get('weighted_score', 0.0)),
                            float(signal_result.get('mtf_score', 0.0)), exp_flag)
                if use_taper and is_burst:
                    taper_ok = self._should_apply_rr_taper(confidence_pct,
                                                        float(signal_result.get('weighted_score', 0.0)),
                                                        float(signal_result.get('mtf_score', 0.0)),
                                                        df)
                    if taper_ok:
                        min_rr = float(getattr(self.config, 'rr_taper_floor', 0.80))
                        rr_floor_note = "tapered"
                        logger.info(f"[RISK] Tapered R:R floor active: {min_rr:.2f} (eligibility met)")
            except Exception as e:
                logger.debug(f"[RISK] Taper check skipped: {e}")
            logger.info(f"Entry/Exit: Entry={entry:.2f}, SL={stop_loss:.2f}, TP={take_profit:.2f}, R:R={risk_reward:.2f} (floor={min_rr:.2f}, note={rr_floor_note})")
            # Final enforcement
            if risk_reward < min_rr:
                logger.info(f"❌ R:R {risk_reward:.2f} below floor {min_rr:.2f} — rejecting signal")
                return {
                    'entry_price': round(entry, 2),
                    'stop_loss': round(stop_loss, 2),
                    'take_profit': round(take_profit, 2),
                    'risk_reward': round(risk_reward, 2),
                    'size_multiplier': round(max(0.25, min(1.0, (risk_reward / max(1.0, min_rr)))) * max(0.25, confidence_pct), 2),
                    'rejection_reason': 'rr_floor'
                }


            
             # Show raw candidates for traceability (high-visibility DEBUG) 
            logger.debug(f"Entry/Exit candidates: upper={upper_candidates} lower={lower_candidates}")

            logger.info(f"Entry/Exit (directional): res_above={resistance_above:.2f}, "
                        f"sup_below={support_below:.2f}, entry={entry:.2f}, "
                        f"SL={stop_loss:.2f}, TP={take_profit:.2f}")
            
            # Compute risk/reward; if risk=0, nudge SL slightly to avoid degenerate R:R
            risk = abs(entry - stop_loss)
            if risk == 0:
                logger.info("SL equals entry; nudging SL by adjust_step to maintain valid R:R")
                if "BUY" in signal_type:
                    stop_loss = entry - adjust_step
                elif "SELL" in signal_type:
                    stop_loss = entry + adjust_step

          
                risk = abs(entry - stop_loss)
                
                logger.debug(f"Nudged SL to preserve R:R: entry={entry:.2f}, SL={stop_loss:.2f}, risk={risk:.2f}")
                          
            reward = abs(take_profit - entry)
            risk_reward = (reward / risk) if risk > 0 else 0.0

                
            
            
            # Position size multiplier (bounded), consistent with rejection path
            size_multiplier = max(0.25, min(1.0, (risk_reward / max(1.0, min_rr)))) * max(0.25, confidence_pct)

            
            logger.info(f"Entry/Exit: Entry={entry:.2f}, SL={stop_loss:.2f}, TP={take_profit:.2f}, R:R={risk_reward:.2f}")
        
            return {
                'entry_price': round(entry, 2),
                'stop_loss': round(stop_loss, 2),
                'take_profit': round(take_profit, 2),
                'risk_reward': round(risk_reward, 2),
                'size_multiplier': round(size_multiplier, 2)
            }

            
        except Exception as e:
            logger.error(f"Error calculating entry/exit levels: {e}")
            return {
                'entry_price': 0,
                'stop_loss': 0,
                'take_profit': 0,
                'risk_reward': 0
            }



    def _calculate_volatility_range(self, df: pd.DataFrame, window: int = 10) -> float:
        """
        NaN/Inf-safe volatility proxy: median of high-low over last window bars.
        Returns a small positive float (points).
        Falls back to 6.0 if unavailable.
        """
        try:
            if df is None or df.empty:
                logger.debug("[VOL] _calculate_volatility_range: empty df → fallback 6.0")
                return 6.0
            w = max(3, int(window))
            hl = (pd.to_numeric(df['high'], errors='coerce') - pd.to_numeric(df['low'], errors='coerce')).tail(w)
            hl = hl.replace([np.inf, -np.inf], np.nan).dropna()
            if hl.empty:
                logger.debug("[VOL] _calculate_volatility_range: no finite ranges → fallback 6.0")
                return 6.0
            # Use median to reduce outlier impact; clamp to reasonable bounds
            vol = float(np.median(hl))
            if not np.isfinite(vol) or vol <= 0:
                logger.debug("[VOL] _calculate_volatility_range: non-finite or <=0 → fallback 6.0")
                return 6.0
            vol = float(np.clip(vol, 2.0, 50.0))
            logger.debug(f"[VOL] proxy={vol:.2f} (window={w})")
            return vol
        except Exception as e:
            logger.debug(f"[VOL] calc error → fallback 6.0: {e}")
            return 6.0

    
    def _calculate_accuracy_metrics(self) -> Dict:
        """Calculate historical accuracy metrics."""
        try:
            if not self.signal_history:
                return {
                    'signal_accuracy': 0.0,
                    'win_rate': 0.0,
                    'confidence_sustain': 0.0,
                    'total_trades': 0,
                    'avg_profit': 0.0,
                    'sharpe_ratio': 0.0
                }
            
            recent_signals = list(self.signal_history)[-50:]  # Last 50 signals
            
            # Mock calculations (would need actual P&L tracking)
            total = len(recent_signals)
            
            # Estimate based on confidence levels
            high_confidence = sum(1 for s in recent_signals if s.get('confidence', 0) > 70)
            win_rate = (high_confidence / total * 100) if total > 0 else 0
            
            avg_confidence = np.mean([s.get('confidence', 0) for s in recent_signals])
            
            return {
                'signal_accuracy': win_rate,
                'win_rate': win_rate,
                'confidence_sustain': avg_confidence,
                'total_trades': total,
                'avg_profit': 0.0,  # Would need actual tracking
                'sharpe_ratio': 0.0  # Would need actual tracking
            }
            
        except Exception as e:
            logger.error(f"Accuracy metrics calculation error: {e}")
            return {
                'signal_accuracy': 0.0,
                'win_rate': 0.0,
                'confidence_sustain': 0.0,
                'total_trades': 0,
                'avg_profit': 0.0,
                'sharpe_ratio': 0.0
            }
    
    def _analyze_market_structure(self, df: pd.DataFrame, indicators: Dict) -> Dict:
        """Analyze market structure for context."""
        try:
            if df.empty or len(df) < 20:
                return {
                    'trend': 'unknown',
                    'trend_strength': 0,
                    'support': 0,
                    'resistance': 0,
                    'pivot': 0
                }
            
            current_price = float(df['close'].iloc[-1])
            
            # Calculate pivot points
            high = float(df['high'].iloc[-1])
            low = float(df['low'].iloc[-1])
            close = float(df['close'].iloc[-1])
            
            pivot = (high + low + close) / 3
            resistance = 2 * pivot - low
            support = 2 * pivot - high
            
            # Get trend from indicators
            trend = 'neutral'
            trend_strength = 0
            
            if 'supertrend' in indicators:
                st_trend = indicators['supertrend'].get('trend', 'neutral')
                if st_trend == 'bullish':
                    trend = 'bullish'
                    trend_strength = 60
                elif st_trend == 'bearish':
                    trend = 'bearish'
                    trend_strength = 60
            
            if 'ema' in indicators:
                ema_signal = indicators['ema'].get('signal', 'neutral')
                if ema_signal in ['bullish', 'golden_cross']:
                    trend = 'bullish'
                    trend_strength = max(trend_strength, 70)
                elif ema_signal in ['bearish', 'death_cross']:
                    trend = 'bearish'
                    trend_strength = max(trend_strength, 70)
            
            return {
                'trend': trend,
                'trend_strength': trend_strength,
                'support': round(support, 2),
                'resistance': round(resistance, 2),
                'pivot': round(pivot, 2)
            }
            
        except Exception as e:
            logger.error(f"Market structure analysis error: {e}")
            return {
                'trend': 'unknown',
                'trend_strength': 0,
                'support': 0,
                'resistance': 0,
                'pivot': 0
            }
    
    def _get_action_from_signal(self, signal: str) -> str:
        """Convert signal to action."""
        if 'BUY' in signal:
            return 'BUY'
        elif 'SELL' in signal:
            return 'SELL'
        else:
            return 'HOLD'
