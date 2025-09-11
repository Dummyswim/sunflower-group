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
from collections import deque
from pattern_detector import CandlestickPatternDetector, ResistanceDetector
from datetime import datetime, timedelta

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
                logger.info("PRICE ACTION VALIDATION START")
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
                    pattern_signal = ignored_pattern.get('signal', 'neutral').lower()
                    if (signal_type == 'SELL' and pattern_signal == 'long') or (signal_type == 'BUY' and pattern_signal == 'short'):
                        logger.debug(f"Ignored pattern nuance: {ignored_pattern.get('name', 'unknown')} supportive in ranging — overriding veto")
                        return True, "Ignored pattern nuance override in ranging"
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


    def check_timeframe_alignment(self, signal_5m: Dict, indicators_15m: Dict, df_15m: pd.DataFrame) -> Tuple[bool, float, str]:
        """
        Check if 5-min signal aligns with 15-min trend - FIXED VERSION.
        Returns (is_aligned, alignment_score, description)
        """
        try:

            logger.info("=" * 50) 
            logger.info("MTF ALIGNMENT CHECK STARTING") 
            logger.debug(f"MTF inputs: 5m={signal_5m.get('composite_signal','NEUTRAL')}, " f"15m_len={len(df_15m) if df_15m is not None else 0}, " f"15m_keys={list(indicators_15m.keys()) if indicators_15m else []}")
            
            
            if not self.config.multi_timeframe_alignment:
                logger.info("MTF alignment disabled in config")
                return True, 1.0, "MTF alignment disabled"
            
            if not indicators_15m or df_15m is None or df_15m.empty:
                logger.warning("No 15-min data for MTF analysis")
                return True, 0.5, "No 15-min data available"
            
            # Get 5m signal direction
            sig = str(signal_5m.get('composite_signal', '')).upper()
            if "BUY" in sig:
                signal_direction = 1
                signal_name = "BUY"
            elif "SELL" in sig:
                signal_direction = -1
                signal_name = "SELL"
            else:
                signal_direction = 0
                signal_name = "NEUTRAL"
            
            logger.info(f"5m Signal: {signal_name} (direction={signal_direction})")
            
            # Analyze 15m trend with detailed logging
            trend_15m = self._analyze_higher_timeframe_trend(indicators_15m, df_15m)
            
            logger.info(f"15m Trend Analysis:")
            logger.info(f"  Direction: {trend_15m['direction']} (-1=bear, 0=neutral, 1=bull)")
            logger.info(f"  Strength: {trend_15m['strength']:.1f}")
            logger.info(f"  Momentum Aligned: {trend_15m['momentum_aligned']}")
            logger.info(f"  S/R Room: {trend_15m['sr_aligned']}")
            
            # Calculate alignment score with detailed breakdown
            alignment_score = 0.0
            score_breakdown = []
            
            # 1) Direction (40%) — cap credit when 5m is NEUTRAL 
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


            # Extreme-context MTF haircut: prevent borderline passes at range edges
            try:
                if signal_direction == 1 and 'price_position' in locals():
                    if price_position >= self.config.extreme_price_pos_hi and not sr_aligned_trade:
                        alignment_score = max(0.0, alignment_score - 0.15)
                        score_breakdown.append("Top-of-range BUY haircut (−0.15)")
                if signal_direction == -1 and 'price_position' in locals():
                    if price_position <= self.config.extreme_price_pos_lo and not sr_aligned_trade:
                        alignment_score = max(0.0, alignment_score - 0.15)
                        score_breakdown.append("Bottom-of-range SELL haircut (−0.15)")
            except Exception:
                pass



            # 2) Momentum (30%) — align relative to 5m trade direction
                        
            momentum_aligned_trade = False
            try:
                rsi_val_15 = float(indicators_15m.get('rsi', {}).get('value', 50))
                macd_sig_15 = str(indicators_15m.get('macd', {}).get('signal_type', 
                                indicators_15m.get('macd', {}).get('signal','neutral'))).lower()
                if signal_direction == 1 and rsi_val_15 > 50 and 'bearish' not in macd_sig_15:
                    momentum_aligned_trade = True
                elif signal_direction == -1 and rsi_val_15 < 50 and 'bullish' not in macd_sig_15:
                    momentum_aligned_trade = True
            except Exception:
                pass


            
            
            
            if momentum_aligned_trade:
                alignment_score += 0.3
                score_breakdown.append("Momentum ALIGNED with 5m (+0.3)")
            else:
                score_breakdown.append("Momentum NOT aligned with 5m (+0.0)")
            
            # 3) S/R room (30%) — room in 5m trade direction
            sr_aligned_trade = False
            try:
                if df_15m is not None and not df_15m.empty and len(df_15m) >= 20 and signal_direction != 0:
                    current_price = float(df_15m['close'].iloc[-1])
                    recent_high = float(df_15m['high'].tail(20).max())
                    recent_low = float(df_15m['low'].tail(20).min())
                    rng = (recent_high - recent_low)
                    price_pos = (current_price - recent_low) / rng if rng > 0 else 0.5
                    if signal_direction == 1 and price_pos < 0.7:
                        sr_aligned_trade = True
                    elif signal_direction == -1 and price_pos > 0.3:
                        sr_aligned_trade = True
                    # Make price_position available for extreme haircut
                    price_position = price_pos
            except Exception:
                pass
            if sr_aligned_trade:
                alignment_score += 0.3
                score_breakdown.append("S/R room AVAILABLE for 5m (+0.3)")
            else:
                score_breakdown.append("S/R room LIMITED for 5m (+0.0)")




            # logger.debug(f"MTF components (5m_dir={signal_direction}) → " f"momentum_trade={momentum_aligned_trade}, sr_trade={sr_aligned_trade}")
            
            # # Pattern boost — only for directional 5m and HTF not hostile with ≥1 supportive component
            # pattern_boost = False
            # if signal_direction != 0 and signal_5m.get('weighted_score', 0) >= 0.10:
            #     patterns_str = " ".join(signal_5m.get('scalping_signals', [])).lower()
            #     if ("pattern" in patterns_str) and (trend_15m['direction'] in (0, signal_direction)) and (momentum_aligned_trade or sr_aligned_trade):
            #         target = max(alignment_score, self.config.trend_alignment_threshold)
            #         if target > alignment_score:
            #             alignment_score = target
            #             pattern_boost = True
            #             logger.info(f"PATTERN BOOST: 5m pattern with HTF supportive → score raised to {alignment_score:.2f}")

            # # Log the scoring breakdown
            # logger.info("MTF Score Breakdown:")
            # for item in score_breakdown:
            #     logger.info(f"  - {item}")
            # logger.info(f"TOTAL SCORE: {alignment_score:.2f}")
            
            # # Special case: Strong 5m pattern with neutral/compatible 15m
            # pattern_boost = False
            # if signal_5m.get('weighted_score', 0) >= 0.10:
            #     patterns = signal_5m.get('scalping_signals', [])
            #     if any('pattern' in str(p).lower() for p in patterns):
            #         if trend_15m['direction'] in (0, signal_direction):
            #             alignment_score = max(alignment_score, 0.6)
            #             pattern_boost = True
            #             logger.info("PATTERN BOOST: Strong 5m pattern detected, boosting score to 0.6")
                        
                        
                        
            logger.debug(f"MTF components (5m_dir={signal_direction}) → momentum_trade={momentum_aligned_trade}, sr_trade={sr_aligned_trade}")            
        # Pattern boost — only for directional 5m and HTF not hostile with ≥1 supportive component
            pattern_boost = False
            if signal_direction != 0 and signal_5m.get('weighted_score', 0) >= 0.10:
                patterns_str = " ".join(signal_5m.get('scalping_signals', [])).lower()
                if ("pattern" in patterns_str) and (trend_15m['direction'] in (0, signal_direction)) and (momentum_aligned_trade or sr_aligned_trade):
                    target = max(alignment_score, self.config.trend_alignment_threshold)
                    if target > alignment_score:
                        alignment_score = target
                        pattern_boost = True
                        logger.info(f"PATTERN BOOST: 5m pattern with HTF supportive → score raised to {alignment_score:.2f}")
            
            # Log the scoring breakdown after any boost
            logger.info("MTF Score Breakdown:")
            for item in score_breakdown:
                logger.info(f"  - {item}")
            logger.info(f"TOTAL SCORE: {alignment_score:.2f}")

                            
                        
                        
            
            # # Determine if aligned
            # threshold = self.config.trend_alignment_threshold
            
            session = session_info.get('session', 'normal') if 'session_info' in locals() else 'normal'  # Safe session lookup
            threshold = self.config.trend_alignment_threshold
            if session == 'ranging':
                threshold -= 0.10  # Less strict in ranging (e.g., 0.60 -> 0.50)
                logger.info(f"Dynamic MTF threshold for ranging: {threshold:.2f}")
            else:
                logger.debug(f"MTF threshold: standard {threshold:.2f}")


            
            is_aligned = alignment_score >= threshold
            
            # Create description
            if is_aligned:
                description = f"✅ ALIGNED with 15m (score: {alignment_score:.2f} >= {threshold:.2f})"
            else:
                description = f"❌ NOT ALIGNED with 15m (score: {alignment_score:.2f} < {threshold:.2f})"
            
            logger.info(f"MTF RESULT: {description}")
            logger.info("=" * 50)
            
            return is_aligned, alignment_score, description
            
        except Exception as e:
            logger.error(f"MTF alignment check error: {e}", exc_info=True)
            return True, 0.5, "MTF check error"


    def _analyze_higher_timeframe_trend(self, indicators: Dict, df: pd.DataFrame) -> Dict:
        """Analyze trend on higher timeframe with detailed logging."""
        try:
            logger.info("Analyzing 15m timeframe indicators...")
            
            analysis = {
                'direction': 0,  # -1: bearish, 0: neutral, 1: bullish
                'strength': 0.0,
                'momentum_aligned': False,
                'sr_aligned': False
            }
            
            # 1. Check EMA trend
            ema_direction = 0
            if 'ema' in indicators:
                ema_data = indicators['ema']
                ema_signal = ema_data.get('signal', 'neutral')
                
                # Get actual EMA values for logging
                short_ema = ema_data.get('short', 0)
                medium_ema = ema_data.get('medium', 0)
                long_ema = ema_data.get('long', 0)
                
                logger.info(f"  EMA Values: Short={short_ema:.2f}, Med={medium_ema:.2f}, Long={long_ema:.2f}")
                logger.info(f"  EMA Signal: {ema_signal}")
                
                if ema_signal in ['bullish', 'golden_cross', 'above']:
                    ema_direction = 1
                    analysis['direction'] = 1
                    logger.info("  EMA indicates BULLISH")
                elif ema_signal in ['bearish', 'death_cross', 'below']:
                    ema_direction = -1
                    analysis['direction'] = -1
                    logger.info("  EMA indicates BEARISH")
                else:
                    logger.info("  EMA is NEUTRAL")
            
            # 2. Check Supertrend
            st_direction = 0
            if 'supertrend' in indicators:
                st_data = indicators['supertrend']
                st_trend = st_data.get('trend', 'neutral')
                st_value = st_data.get('value', 0)
                
                logger.info(f"  Supertrend: {st_trend} (value={st_value:.2f})")
                
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



                # Resolve conflicts between EMA and Supertrend — require MACD support to trust EMA
                if ema_direction != 0 and st_direction != 0 and ema_direction != st_direction:
                    rsi_value = float(indicators.get('rsi', {}).get('value', 50))
                    macd_sig = str(indicators.get('macd', {}).get('signal_type', 'neutral')).lower()
                    medium_ema = float(ema_data.get('medium', 0))
                    price = float(df['close'].iloc[-1]) if not df.empty else 0.0
                    price_above_med = price > medium_ema
                    price_below_med = price < medium_ema
                    macd_bull = 'bullish' in macd_sig
                    macd_bear = 'bearish' in macd_sig
                    
                    if ema_direction == 1 and price_above_med and rsi_value >= 55 and macd_bull:
                        analysis['direction'] = 1
                        analysis['strength'] = max(analysis['strength'], 0.5) + 0.3
                        logger.info("  CONFLICT: EMA=1, ST=-1 → RESOLVED: trust EMA (MACD supports uptrend)")
                    elif ema_direction == -1 and price_below_med and rsi_value <= 45 and macd_bear:
                        analysis['direction'] = -1
                        analysis['strength'] = max(analysis['strength'], 0.5) + 0.3
                        logger.info("  CONFLICT: EMA=-1, ST=1 → RESOLVED: trust EMA (MACD supports downtrend)")
                    else:
                        analysis['direction'] = 0
                        analysis['strength'] = 0.25
                        logger.info(f"  CONFLICT: EMA={ema_direction}, ST={st_direction}, MACD={macd_sig} → Using NEUTRAL")


            
            
            # 3. Check RSI momentum
            if 'rsi' in indicators:
                rsi_value = indicators['rsi'].get('value', 50)
                logger.info(f"  RSI: {rsi_value:.2f}")
                
                # Check if momentum aligns with direction
                if analysis['direction'] == 1 and rsi_value > 45:  # Bullish with supportive RSI
                    analysis['momentum_aligned'] = True
                    logger.info("  RSI supports BULLISH momentum")
                elif analysis['direction'] == -1 and rsi_value < 55:  # Bearish with supportive RSI
                    analysis['momentum_aligned'] = True
                    logger.info("  RSI supports BEARISH momentum")
                elif analysis['direction'] == 0 and 40 <= rsi_value <= 60:  # Neutral
                    analysis['momentum_aligned'] = True
                    logger.info("  RSI is NEUTRAL")
                else:
                    logger.info("  RSI does NOT align with trend")
            
            # 4. Check support/resistance room
            if not df.empty and len(df) >= 20:
                current_price = float(df['close'].iloc[-1])
                recent_high = float(df['high'].tail(20).max())
                recent_low = float(df['low'].tail(20).min())
                
                price_range = recent_high - recent_low
                if price_range > 0:
                    price_position = (current_price - recent_low) / price_range
                else:
                    price_position = 0.5
                
                logger.info(f"  Price Position: {price_position:.2f} (0=low, 1=high)")
                logger.info(f"  Range: {recent_low:.2f} - {recent_high:.2f}")
                
                # Check if there's room to move
                if analysis['direction'] == 1 and price_position < 0.8:  # Bullish with room up
                    analysis['sr_aligned'] = True
                    logger.info("  S/R: Room to move UP")
                elif analysis['direction'] == -1 and price_position > 0.2:  # Bearish with room down
                    analysis['sr_aligned'] = True
                    logger.info("  S/R: Room to move DOWN")
                elif analysis['direction'] == 0 and 0.3 <= price_position <= 0.7:  # Neutral in middle
                    analysis['sr_aligned'] = True
                    logger.info("  S/R: In middle of range")
                else:
                    logger.info("  S/R: Limited room for movement")
            
            logger.info(f"15m Summary: Direction={analysis['direction']}, "
                    f"Strength={analysis['strength']:.1f}, "
                    f"Momentum={analysis['momentum_aligned']}, "
                    f"S/R={analysis['sr_aligned']}")
            
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
# ==========================================================================================
# ENHANCED SIGNAL ANALYZER WITH ALL IMPROVEMENTS
# ==========================================================================================

class ConsolidatedSignalAnalyzer:
    """Enhanced signal analyzer with all new features."""
    
    def __init__(self, config, technical_analysis=None):
        self.config = config
        self.technical = technical_analysis
        self.pattern_detector = CandlestickPatternDetector()
        self.resistance_detector = ResistanceDetector()
        
        # Initialize validators and analyzers
        self.price_action_validator = PriceActionValidator(config)
        self.mtf_analyzer = MultiTimeframeAnalyzer(config)
        
        # Previous components (updated)
        # self.validator = SignalValidator(config)
        self.predictor = SignalDurationPredictor(config)
        self.consensus_analyzer = EnhancedIndicatorConsensus(config)
        self.trend_analyzer = TrendAlignmentAnalyzer(config)
        
        # History tracking
        self.signal_history = deque(maxlen=100)
        self.last_alert_time = None
        self._last_reject = None  # Add this line
        self.current_df = None 
        
        logger.info("Enhanced ConsolidatedSignalAnalyzer initialized")
    
    async def analyze_and_generate_signal(
        self, 
        indicators_5m: Dict, 
        df_5m: pd.DataFrame,
        indicators_15m: Optional[Dict] = None,
        df_15m: Optional[pd.DataFrame] = None
    ) -> Optional[Dict]:
        """Generate trading signal with all enhancements."""
        
        try:
            # Input validation
            if not indicators_5m or df_5m.empty:
                logger.warning("Invalid input: empty indicators or dataframe")
                return None
            
            logger.info("=" * 50)
            logger.info("Starting enhanced signal analysis...") 
            self._last_reject = None # reset last rejection for this bar 
            self.current_df = df_5m # Store for market regime detection


            # Detect market session characteristics
            session_info = self.detect_session_characteristics(df_5m)
            logger.info(f"Session: {session_info['session']} | Strategy: {session_info['strategy']}")

            # Make HTF MACD signal available to the scorer (for ST softening gate)
            try:
                self._htf_macd_sig = str(
                    indicators_15m.get('macd', {}).get('signal_type',
                    indicators_15m.get('macd', {}).get('signal', 'neutral'))
                ).lower() if indicators_15m else None
            except Exception:
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
                    self._last_reject = {'stage': 'PA', 'reason': pa_reason} 
                    return None
                    
                logger.info(f"✓ Price action validation passed: {pa_reason}")

            
            
            
            # 4. Multi-Timeframe Alignment Check
            mtf_aligned = True
            mtf_score = 1.0
            mtf_description = "MTF not checked"
            
            if self.config.mtf_enabled and indicators_15m and df_15m is not None:
                mtf_aligned, mtf_score, mtf_description = self.mtf_analyzer.check_timeframe_alignment(
                    signal_result,
                    indicators_15m,
                    df_15m
                )
                
                
                # if self.config.mtf_alignment_required and not mtf_aligned: 
                #     logger.info(f"❌ Signal rejected: {mtf_description}") 
                #     self._last_reject = {'stage': 'MTF', 'reason': mtf_description} 
                #     return None
                
                # Enforce MTF only for directional calls; skip veto for NEUTRAL (non-actionable) 
                if (self.config.mtf_alignment_required and signal_result.get('composite_signal') != 'NEUTRAL' and not mtf_aligned): 
                    logger.info(f"❌ Signal rejected: {mtf_description}") 
                    self._last_reject = {'stage': 'MTF', 'reason': mtf_description} 
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
            signal_result['mtf_score'] = mtf_score 
            group_consensus = self.consensus_analyzer.calculate_group_consensus(indicators_5m)
            
            # 6. Analyze trend alignment
            trend_analysis = self.trend_analyzer.analyze_trend(df_5m, indicators_5m)
            
            # 7. Enhanced validation
            is_valid = self._enhanced_validation(
                signal_result, 
                group_consensus, 
                trend_analysis,
                df_5m
            )
      
            if not is_valid: 
                logger.info("❌ Signal failed enhanced validation") 
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
            
        
            # Enforce Risk/Reward floor for actionable signals 
            try: 
                rr_val = float(entry_exit.get('risk_reward', 0)) 
                rr_floor = float(getattr(self.config, 'min_risk_reward_floor', 1.0)) 
                comp = str(signal_result.get('composite_signal', 'NEUTRAL')).upper()
                if comp in ('BUY', 'STRONG_BUY', 'SELL', 'STRONG_SELL') and rr_val < rr_floor: 
                    logger.info(f"❌ Rejected by R:R filter → {rr_val:.2f} < floor {rr_floor:.2f}") 
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
                    'mtf_analysis': { 'aligned': mtf_aligned, 'score': mtf_score, 'description': mtf_description }, 
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

            return final_signal
            
        except Exception as e:
            logger.error(f"Signal analysis error: {e}", exc_info=True)
            return None


    def should_generate_rapid_scalping_signal(self, df: pd.DataFrame, last_signal_time: datetime) -> bool:
        """Generate more frequent signals for scalping."""

        # Protect against None value
        if last_signal_time is None:
            return False
                
        # Allow rapid signals if strong momentum detected
        if (datetime.now() - last_signal_time).total_seconds() < 30:
            return False
            
        # Check for rapid scalping opportunities
        last_3 = df.tail(3)
        
        # Pattern 1: Three consecutive same-direction candles
        all_green = all(last_3['close'] > last_3['open'])
        all_red = all(last_3['close'] < last_3['open'])
        
        # Pattern 2: Price acceleration
        if len(df) > 10:
            recent_move = abs(df['close'].iloc[-1] - df['close'].iloc[-3])
            avg_move = abs(df['close'].diff()).tail(10).mean()
            accelerating = recent_move > avg_move * 2
            
            if (all_green or all_red) and accelerating:
                return True
        
        # Pattern 3: Support/Resistance break
        if hasattr(self, 'resistance_detector') and self.resistance_detector is not None:
            levels = self.resistance_detector.detect_levels(df)
            current = df['close'].iloc[-1]
            
            # Breaking resistance
            if current > levels['nearest_resistance'] * 0.999:
                return True
            # Bouncing from support
            if current < levels['nearest_support'] * 1.001:
                return True
        
        return False




    def detect_session_characteristics(self, df: pd.DataFrame) -> Dict:
        """Auto-detect session type based on market behavior, not time."""
        try:
            if len(df) < 20:
                return {'session': 'unknown', 'characteristics': {}}
            
            # Analyze last 20 candles
            recent = df.tail(20)
            current_hour = datetime.now().hour
            
            # Calculate metrics
            volatility = recent['high'].std() / recent['close'].mean() * 100
            avg_range = (recent['high'] - recent['low']).mean()
            trend_strength = abs(recent['close'].iloc[-1] - recent['close'].iloc[0]) / recent['close'].iloc[0] * 100
            
            # Detect opening patterns (high volatility, wide ranges)
            if volatility > 0.5 and avg_range > recent['close'].mean() * 0.003:
                session_type = 'opening_volatile'
                strategy = 'fade_extremes'
                confidence_adjustment = 0.8  # Lower confidence in volatile periods
                
            # Detect trending session (steady movement, lower volatility)
            elif trend_strength > 0.3 and volatility < 0.3:
                session_type = 'trending'
                strategy = 'follow_trend'
                confidence_adjustment = 1.2  # Higher confidence in trends
                
            # Detect ranging/consolidation (low volatility, small ranges)
            elif volatility < 0.2 and avg_range < recent['close'].mean() * 0.002:
                session_type = 'ranging'
                strategy = 'mean_reversion'
                confidence_adjustment = 0.9
                
            # Detect closing session (decreasing volatility)
            elif len(df) > 60:
                early_vol = df.iloc[-60:-40]['high'].std() / df.iloc[-60:-40]['close'].mean() * 100
                late_vol = recent['high'].std() / recent['close'].mean() * 100
                
                if late_vol < early_vol * 0.7:  # Volatility decreased by 30%+
                    session_type = 'closing'
                    strategy = 'book_profits'
                    confidence_adjustment = 0.7
                else:
                    session_type = 'normal'
                    strategy = 'standard'
                    confidence_adjustment = 1.0
            else:
                session_type = 'normal'
                strategy = 'standard'
                confidence_adjustment = 1.0
            
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
            
            # Breakout detection (price near recent high/low)
            recent_high = recent['high'].max()
            recent_low = recent['low'].min()
            current = df['close'].iloc[-1]
            
            if (recent_high - current) / current < 0.002:
                patterns['breakout_potential'] = True
            elif (current - recent_low) / current < 0.002:
                patterns['reversal_potential'] = True
            
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
            logger.error(f"Session detection error: {e}")
            return {'session': 'unknown', 'strategy': 'standard', 'confidence_adjustment': 1.0}


    def _calculate_final_confidence_with_mtf( self, signal_result: Dict, group_consensus: Dict, trend_analysis: Dict, mtf_score: float ) -> float: 
        """Calculate final confidence including MTF boost.""" 
        try: 
            # Start with base confidence (ensure numeric) 
            base_confidence = float(signal_result.get('confidence', 0.0))

            # None-safe MTF score
            try:
                mtf_score = float(mtf_score) if mtf_score is not None else 0.0
            except Exception:
                mtf_score = 0.0
            
            # Factor 1: Group consensus bonus (up to +15%)
            consensus_bonus = 0.0
            if group_consensus:
                valid_groups = [g for g in group_consensus.values() if g.get('indicator_count', 0) > 0]
                if valid_groups:
                    avg_agreement = float(np.mean([g.get('agreement', 0) for g in valid_groups]))
                    consensus_bonus = avg_agreement * 15.0
           
            # Factor 2: Trend alignment bonus (up to +10%) — skip for NEUTRAL 
            trend_bonus = 0.0 
            sig_str = str(signal_result.get('composite_signal', '')).upper() 
            if trend_analysis.get('aligned', False) and sig_str not in ('NEUTRAL', ''): 
                signal_direction = 1 if "BUY" in sig_str else -1 if "SELL" in sig_str else 0 
                trend_direction = 1 if trend_analysis.get('trend') == 'bullish' else -1 if trend_analysis.get('trend') == 'bearish' else 0 
                if signal_direction == trend_direction: 
                    trend_bonus = 10.0
         
            # Factor 3: MTF alignment bonus (up to +20%) — skip for NEUTRAL
            mtf_bonus = (mtf_score * 20.0) if (self.config.mtf_enabled and sig_str not in ('NEUTRAL', '')) else 0.0

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


            # Score magnitude scaling (0..1) and capped additive bonuses 
            try: 
                ws_mag = min(1.0, abs(float(signal_result.get('weighted_score', 0.0))))
            except Exception: 
                ws_mag = 0.0

            consensus_bonus *= (0.5 + 0.5 * ws_mag)  # 50–100% scaling
            trend_bonus *= ws_mag
            mtf_bonus *= ws_mag
            agreement_bonus *= (0.5 + 0.5 * ws_mag)

            total_bonus = consensus_bonus + trend_bonus + mtf_bonus + agreement_bonus
            total_bonus = min(30.0, max(-15.0, total_bonus))
            final_confidence = base_confidence + total_bonus
            final_confidence = max(0.0, min(100.0, float(final_confidence)))


            logger.debug(f"Confidence calculation: Base={base_confidence:.1f}, "
                        f"Consensus=+{consensus_bonus:.1f}, Trend=+{trend_bonus:.1f}, "
                        f"MTF=+{mtf_bonus:.1f}, Agreement={agreement_bonus:+.1f}, "
                        f"TotalBonus={total_bonus:+.1f}, Final={final_confidence:.1f}")

            if sig_str in ('NEUTRAL', ''):
                final_confidence = min(final_confidence, 45.0)
                
                
            # Confidence cap in extreme context without breakout evidence
            try:
                ec = signal_result.get('extreme_context', {})
                sig_str_up = str(signal_result.get('composite_signal','')).upper()
                if ec:
                    cap = float(getattr(self.config, 'confidence_cap_at_extreme', 80.0))
                    if 'BUY' in sig_str_up and ec.get('top_extreme') and not ec.get('breakout_evidence'):
                        if final_confidence > cap:
                            logger.info(f"Confidence capped at extreme (top, no breakout): {final_confidence:.1f}% → {cap:.1f}%")
                            final_confidence = cap
                    if 'SELL' in sig_str_up and ec.get('bottom_extreme') and not ec.get('breakdown_evidence'):
                        if final_confidence > cap:
                            logger.info(f"Confidence capped at extreme (bottom, no breakdown): {final_confidence:.1f}% → {cap:.1f}%")
                            final_confidence = cap
            except Exception:
                pass


            return final_confidence



        except Exception as e:
            logger.error(f"Error calculating final confidence: {e}", exc_info=True)
            # Fallback to base confidence if something went wrong
            return float(signal_result.get('confidence', 0.0))









    def _calculate_weighted_signal(self, indicators: Dict) -> Dict:
        """Calculate weighted signal for INDEX SCALPING with market regime awareness."""
        try:
            weighted_score = 0.0
            contributions = {}
            active_count = 0
            scalping_signals = []
            
            # CRITICAL: Detect market regime first
            if hasattr(self, 'current_df'):
                market_regime = self.detect_market_regime(self.current_df)
            else:
                market_regime = "NORMAL"
            
            


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
                    if market_regime == "STRONG_UPTREND":
                        # In strong uptrends, RSI can stay overbought
                        if value > 85:  # Only extreme overbought is bearish
                            contribution = -weight * 0.5  # Reduced weight
                            active_count += 1
                        elif value > 70:  # Normal overbought is still bullish in trend
                            contribution = weight * 0.3
                            active_count += 1
                        elif value < 30:  # Oversold in uptrend = strong buy
                            contribution = weight * 1.5
                            active_count += 1
                        elif value > 50:
                            contribution = weight * 0.2
                            active_count += 1
                    else:
                        # Normal market conditions
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
                            
                            
                        # Store RSI micro-feature for later quality checks
                        extras['rsi_value'] = indicator.get('value', 50)



                            
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
                    hist_series = indicator.get('histogram_series')
                    hist_slope = 0.0
                    try:
                        if hist_series is not None and len(hist_series) > 1:
                            hist_slope = float(hist_series.iloc[-1] - hist_series.iloc[-2])
                    except Exception:
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


            # CHECK IF THIS SECTION OF CODE IS CORRECTLY INTEGRATED 
            # Low-breadth quality gate and attenuation
            if active_count < 3:
                bull_bias = weighted_score > 0
                macd_slope = float(contributions.get('macd', {}).get('hist_slope', 0.0))
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
            
            # Signal determination with regime-adjusted thresholds
            if weighted_score < (sell_threshold - 0.1):
                composite_signal = 'STRONG_SELL'
                next_candle_prediction = "📉 Next 5-15min: RED candle expected (-20 to -30 points)"
            elif weighted_score < sell_threshold:
                composite_signal = 'SELL'
                next_candle_prediction = "📉 Next 5min: Likely RED candle (-15 to -20 points)"
            elif weighted_score > (buy_threshold + 0.1):
                composite_signal = 'STRONG_BUY'
                next_candle_prediction = "📈 Next 5-15min: GREEN candle expected (+20 to +30 points)"
            elif weighted_score > buy_threshold:
                composite_signal = 'BUY'
                next_candle_prediction = "📈 Next 5min: Likely GREEN candle (+15 to +20 points)"
            else:
                composite_signal = 'NEUTRAL'
                next_candle_prediction = "➡️ Consolidation expected, no clear direction"
                        

            if pd.isna(weighted_score) or np.isinf(weighted_score):
                logger.error(f"Invalid weighted_score detected: {weighted_score}, resetting to 0")
                weighted_score = 0.0
           
            # Include pattern detection in signal (HTF/SR-aware gating)
            if self.pattern_detector and hasattr(self, 'current_df') and not self.current_df.empty:
                try:
                    pattern_result = self.pattern_detector.detect_patterns(self.current_df)
                    if pattern_result and pattern_result.get('confidence', 0) > 0:
                        pattern_signal = pattern_result.get('signal', 'neutral')
                        htf_macd_sig = str(getattr(self, '_htf_macd_sig', '') or '').lower()
                        bb_pos = float(contributions.get('bollinger', {}).get('position', 0.5))

                        top_guard = bool(getattr(self, '_extreme_top', False))
                        bottom_guard = bool(getattr(self, '_extreme_bottom', False))
                        hostile_buy_ctx = ('bearish' in htf_macd_sig) or (bb_pos >= 0.90) or top_guard
                        hostile_sell_ctx = ('bullish' in htf_macd_sig) or (bb_pos <= 0.10) or bottom_guard

                        breakout_evidence = bool(pattern_result.get('name') == 'breakout_potential' and pattern_signal == 'LONG')
                        breakdown_evidence = bool(pattern_result.get('name') == 'breakdown_potential' and pattern_signal == 'SHORT')

                                                

                        if pattern_signal == 'LONG' and weighted_score > 0:
                            if hostile_buy_ctx:
                                logger.info(f"Pattern detected but ignored (HTF hostile or near upper band): {pattern_result['name']} | htf_macd={htf_macd_sig}, bb_pos={bb_pos:.2f}")
                            else:
                                weighted_score += 0.05
                                scalping_signals.append(f"Pattern applied: {pattern_result['name']}")
                                logger.info(f"Pattern boost applied: {pattern_result['name']}")
                        elif pattern_signal == 'SHORT' and weighted_score < 0:
                            if hostile_sell_ctx:
                                logger.info(f"Pattern detected but ignored (HTF hostile or near lower band): {pattern_result['name']} | htf_macd={htf_macd_sig}, bb_pos={bb_pos:.2f}")
                            else:
                                weighted_score -= 0.05
                                scalping_signals.append(f"Pattern applied: {pattern_result['name']}")
                                logger.info(f"Pattern boost applied: {pattern_result['name']}")
                        else:
                            logger.info(f"Pattern detected but ignored (contradicts 5m bias): {pattern_result.get('name')}")
                except Exception as e:
                    logger.debug(f"Pattern detection skipped: {e}")




            # Re-classify after pattern/quality adjustments to keep mapping consistent 
            prev_signal = composite_signal 
            if weighted_score < (sell_threshold - 0.1): 
                composite_signal = 'STRONG_SELL' 
                next_candle_prediction = "📉 Next 5-15min: RED candle expected (-20 to -30 points)"
            elif weighted_score < sell_threshold: 
                composite_signal = 'SELL' 
                next_candle_prediction = "📉 Next 5min: Likely RED candle (-15 to -20 points)" 
            elif weighted_score > (buy_threshold + 0.1): 
                composite_signal = 'STRONG_BUY' 
                next_candle_prediction = "📈 Next 5-15min: GREEN candle expected (+20 to +30 points)" 
            elif weighted_score > buy_threshold: 
                composite_signal = 'BUY' 
                next_candle_prediction = "📈 Next 5min: Likely GREEN candle (+15 to +20 points)" 
            else: 
                composite_signal = 'NEUTRAL' 
                next_candle_prediction = "➡️ Consolidation expected, no clear direction" 
            
            if composite_signal != prev_signal: 
                logger.debug(f"Reclassified after adjustments: {prev_signal} → {composite_signal} (score={weighted_score:.3f})")
            
            
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
            
            logger.info(f"Market Regime: {market_regime}")
            logger.info(f"Weighted Score: {weighted_score:.3f}")
            logger.info(f"Active Indicators: {active_count}/{len(self.config.indicator_weights)}")
            logger.info(f"Confidence: {confidence:.1f}%")
            logger.info(f"Preliminary prediction (subject to PA/MTF): {next_candle_prediction}")



            return {
                'weighted_score': weighted_score,
                'composite_signal': composite_signal,
                'confidence': confidence,
                'active_indicators': active_count,
                'contributions': contributions,
                'next_candle_prediction': next_candle_prediction,
                'scalping_signals': scalping_signals,
                'action_recommendation': self._get_scalping_action(weighted_score),
                'market_regime': market_regime,
                'extreme_context': {
                    'htf_price_pos': getattr(self, '_htf_price_pos', None),
                    'top_extreme': getattr(self, '_extreme_top', False),
                    'bottom_extreme': getattr(self, '_extreme_bottom', False),
                    'overbought': getattr(self, '_extreme_overbought', False),
                    'oversold': getattr(self, '_extreme_oversold', False),
                    'breakout_evidence': bool(locals().get('breakout_evidence', False)),
                    'breakdown_evidence': bool(locals().get('breakdown_evidence', False)),
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


    def _enhanced_validation(self, signal_result: Dict, group_consensus: Dict, trend_analysis: Dict, df: pd.DataFrame) -> bool: 
        """Enhanced signal validation with directional confirmations and structure checks.""" 
        try: 
            # Minimum activity and raw strength 
            if signal_result['active_indicators'] < self.config.min_active_indicators: 
                logger.debug(f"Too few active indicators: {signal_result['active_indicators']}") 
                return False 
            

            if abs(signal_result['weighted_score']) < self.config.min_signal_strength:
                logger.debug(f"Weak signal strength: {signal_result['weighted_score']}")
                return False

            # Require higher breadth for directional alerts (filter weak 3/6 cases)
            actionable = signal_result.get('composite_signal', 'NEUTRAL') in ('BUY','SELL','STRONG_BUY','STRONG_SELL')
            if actionable and signal_result['active_indicators'] < getattr(self.config, 'min_active_indicators_for_alert', 4):
                logger.info(f"❌ Actionable rejected: active_indicators {signal_result['active_indicators']}/6 < min_for_alert {self.config.min_active_indicators_for_alert}")
                return False


            contributions = signal_result.get('contributions', {})
            composite = signal_result.get('composite_signal', 'NEUTRAL')
            mtf_val = float(signal_result.get('mtf_score', 0.0))


            def sig(name: str) -> str: 
                if name == 'macd':
                    return str(contributions.get('macd', {}).get('signal_type', contributions.get('macd', {}).get('signal', 'neutral'))).lower() 
                return str(contributions.get(name, {}).get('signal', 'neutral')).lower()


            # Count confirmations among core trend indicators
            bull_conf = 0
            bear_conf = 0
            core = ['ema', 'macd', 'supertrend']
            for k in core:
                s = sig(k)
                if any(x in s for x in ['bullish', 'golden_cross', 'above']):
                    bull_conf += 1
                if any(x in s for x in ['bearish', 'death_cross', 'below']):
                    bear_conf += 1

            # EMA+MACD agreement helpers
            ema_sig = sig('ema')
            macd_sig = str(contributions.get('macd', {}).get('signal_type', 'neutral')).lower()
            ema_bull = any(x in ema_sig for x in ['bullish','golden_cross','above'])
            ema_bear = any(x in ema_sig for x in ['bearish','death_cross','below'])
            macd_bull = 'bullish' in macd_sig
            macd_bear = 'bearish' in macd_sig

            # Require confirmations; relax to 1 when MTF is strong (>=0.7) or EMA+MACD agree
            if 'BUY' in composite:
                if bull_conf < 2:
                    if mtf_val >= 0.7 and (bull_conf >= 1 or (ema_bull and macd_bull)):
                        logger.debug("BUY allowed with 1 confirmation due to strong MTF or EMA+MACD agreement")
                    else:
                        logger.debug(f"BUY rejected: bullish confirmations {bull_conf}/2")
                        return False

            if 'SELL' in composite:
                if bear_conf < 2:
                    if mtf_val >= 0.7 and (bear_conf >= 1 or (ema_bear and macd_bear)):
                        logger.debug("SELL allowed with 1 confirmation due to strong MTF or EMA+MACD agreement")
                    else:
                        logger.debug(f"SELL rejected: bearish confirmations {bear_conf}/2")
                        return False

            # Structure guard for shorts: avoid shorting without a recent low break or EMA< conditions
            if 'SELL' in composite and not df.empty:
                close = float(df['close'].iloc[-1])
                recent_low = float(df['low'].tail(5).min())
                ema_bearish = any(x in ema_sig for x in ['bearish', 'death_cross', 'below'])
                if close > recent_low and not ema_bearish:
                    logger.debug("SELL rejected: no 5-candle low break and EMA not bearish/below")
                    return False

            return True
        except Exception as e:
            logger.error(f"Enhanced validation error: {e}")
            return False

    
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
        


    def _calculate_entry_exit_levels(
        self, 
        df: pd.DataFrame, 
        indicators: Dict, 
        signal_result: Dict
    ) -> Dict:
        """Calculate entry/exit levels using volatility instead of ATR."""
        
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





            # Volatility caps
            tp_cap_dist = self.config.tp_volatility_cap_multiple * volatility_range
            sl_cap_dist = self.config.sl_volatility_cap_multiple * volatility_range

            if "BUY" in signal_type:
                # Uncapped directional candidates
                sl_uncapped = max(support_below, entry * (1 - self.config.stop_loss_percentage / 100.0))
                tp_uncapped = max(resistance_above, entry * (1 + self.config.take_profit_percentage / 100.0))

                # Apply volatility caps (keep sides correct)
                stop_loss = max(entry - sl_cap_dist, sl_uncapped)     # cap SL distance (not deeper than cap)
                take_profit = min(entry + tp_cap_dist, tp_uncapped)   # cap TP distance (not farther than cap)

                # Enforce monotonicity with robust fallback
                if stop_loss >= entry:
                    logger.info(f"Monotonicity fix (BUY): SL {stop_loss:.2f} >= entry {entry:.2f} → adjusting")
                    stop_loss = entry - adjust_step
                if take_profit <= entry:
                    logger.info(f"Monotonicity fix (BUY): TP {take_profit:.2f} <= entry {entry:.2f} → adjusting")
                    take_profit = entry + adjust_step

                # Log if caps were applied
                if stop_loss > sl_uncapped:
                    logger.debug(f"SL cap applied (BUY): uncapped={sl_uncapped:.2f}, capped={stop_loss:.2f}, cap_dist={sl_cap_dist:.2f}")
                if take_profit < tp_uncapped:
                    logger.debug(f"TP cap applied (BUY): uncapped={tp_uncapped:.2f}, capped={take_profit:.2f}, cap_dist={tp_cap_dist:.2f}")

            elif "SELL" in signal_type:
                # Uncapped directional candidates
                sl_uncapped = min(resistance_above, entry * (1 + self.config.stop_loss_percentage / 100.0))
                tp_uncapped = max(support_below, entry * (1 - self.config.take_profit_percentage / 100.0))

                # Apply volatility caps (keep sides correct)
                stop_loss = min(entry + sl_cap_dist, sl_uncapped)     # cap SL distance (not farther than cap)
                take_profit = max(entry - tp_cap_dist, tp_uncapped)   # cap TP distance (not farther than cap)

                # Enforce monotonicity with robust fallback
                if stop_loss <= entry:
                    logger.info(f"Monotonicity fix (SELL): SL {stop_loss:.2f} <= entry {entry:.2f} → adjusting")
                    stop_loss = entry + adjust_step
                if take_profit >= entry:
                    logger.info(f"Monotonicity fix (SELL): TP {take_profit:.2f} >= entry {entry:.2f} → adjusting")
                    take_profit = entry - adjust_step

                # Log if caps were applied
                if stop_loss < sl_uncapped:
                    logger.debug(f"SL cap applied (SELL): uncapped={sl_uncapped:.2f}, capped={stop_loss:.2f}, cap_dist={sl_cap_dist:.2f}")
                if take_profit > tp_uncapped:
                    logger.debug(f"TP cap applied (SELL): uncapped={tp_uncapped:.2f}, capped={take_profit:.2f}, cap_dist={tp_cap_dist:.2f}")

            else:
                # Neutral fallback: symmetric range
                stop_loss = entry - min(sl_cap_dist, volatility_range)
                take_profit = entry + min(tp_cap_dist, volatility_range)


            
             # Show raw candidates for traceability (high-visibility DEBUG) 
            logger.debug(f"Entry/Exit candidates: upper={upper_candidates} lower={lower_candidates}")

            logger.debug(f"Entry/Exit (directional): res_above={resistance_above:.2f}, "
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

                
            
            
            
            logger.info(f"Entry/Exit: Entry={entry:.2f}, SL={stop_loss:.2f}, TP={take_profit:.2f}, R:R={risk_reward:.2f}")
            
            return {
                'entry_price': round(entry, 2),
                'stop_loss': round(stop_loss, 2),
                'take_profit': round(take_profit, 2),
                'risk_reward': round(risk_reward, 2)
            }
            
        except Exception as e:
            logger.error(f"Error calculating entry/exit levels: {e}")
            return {
                'entry_price': 0,
                'stop_loss': 0,
                'take_profit': 0,
                'risk_reward': 0
            }


    def _calculate_volatility_range(self, df: pd.DataFrame) -> float:
        """Calculate price range based on volatility."""
        try:
            if len(df) < 20:
                # Use high-low range for short data
                return float((df['high'] - df['low']).mean())
            
            # Calculate using standard deviation
            returns = df['close'].pct_change().dropna()
            volatility = returns.std()
            current_price = float(df['close'].iloc[-1])
            
            
            # Convert to price range (2 standard deviations)
            volatility_range = current_price * volatility * 2

            # Ensure minimum range (configurable for 5m scalps)
            min_pct = float(getattr(self.config, 'min_volatility_range_pct', 0.002))  # default 0.20%
            min_range = current_price * min_pct

            logger.debug(f"Volatility range (raw)={volatility_range:.2f}, min={min_range:.2f} ({min_pct*100:.2f}%)")
            return max(volatility_range, min_range)
            
        except Exception as e:
            logger.error(f"Volatility range calculation error: {e}")
            return float(df['close'].iloc[-1] * 0.01)  # 1% fallback
    
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
