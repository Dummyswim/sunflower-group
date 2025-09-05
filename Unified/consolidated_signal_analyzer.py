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
        lookback: int = 3
    ) -> Tuple[bool, str]:
        """
        Validate signal against recent price action.
        Returns (is_valid, reason)
        """
        try:
            if df.empty or len(df) < lookback:
                return True, "Insufficient data for validation"
            
            # Get last N candles
            recent_candles = df.tail(lookback)
            
            # Analyze candle patterns
            candle_analysis = self._analyze_candles(recent_candles)

            # Check for contradictions
            if "BUY" in signal_type:
                if candle_analysis['strong_bearish_count'] >= 2:
                # Don't buy on strong bearish candles                    
                    return False, f"Strong bearish candles detected ({candle_analysis['strong_bearish_count']}/{lookback})"
                
                # Don't buy if all recent candles are red
                if candle_analysis['bearish_count'] == lookback:
                    if candle_analysis.get('higher_lows', False): 
                        logger.debug("All recent candles bearish but higher lows forming — allowing BUY") 
                    else: 
                        return False, "All recent candles are bearish"


                
                # Don't buy on downward momentum
                if candle_analysis['momentum'] < -0.5:
                    return False, f"Strong downward momentum ({candle_analysis['momentum']:.2f})"
                if candle_analysis.get('lower_highs', False):
                    return False, "Lower highs forming — avoid buying into pressure"
            elif "SELL" in signal_type:
                # Don't sell on strong bullish candles
                if candle_analysis['strong_bullish_count'] >= 2:
                    return False, f"Strong bullish candles detected ({candle_analysis['strong_bullish_count']}/{lookback})"
                
                # Don't sell if all recent candles are green
                if candle_analysis['bullish_count'] == lookback:
                    if candle_analysis.get('lower_highs', False):
                        logger.debug("All recent candles bullish but lower highs forming — allowing SELL")
                    else: 
                        return False, "All recent candles are bullish"
                    
                # Don't sell on upward momentum
                if candle_analysis['momentum'] > 0.5:
                    return False, f"Strong upward momentum ({candle_analysis['momentum']:.2f})"
                if candle_analysis.get('higher_lows', False):
                    return False, "Higher lows forming — avoid shorting rising structure"

            
            # Additional validation for current candle
            current_candle = recent_candles.iloc[-1]
            current_direction = "bullish" if current_candle['close'] > current_candle['open'] else "bearish"
            
            # Strong signal-candle contradiction check
            if "STRONG_BUY" in signal_type and current_direction == "bearish":
                body_ratio = self._calculate_body_ratio(current_candle)
                if body_ratio > self.config.price_action_min_body_ratio:
                    return False, "Current candle is strongly bearish"
                    
            elif "STRONG_SELL" in signal_type and current_direction == "bullish":
                body_ratio = self._calculate_body_ratio(current_candle)
                if body_ratio > self.config.price_action_min_body_ratio:
                    return False, "Current candle is strongly bullish"
            
            logger.debug(f"Price action validation passed for {signal_type}")
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
        Check if 5-min signal aligns with 15-min trend.
        Returns (is_aligned, alignment_score, description)
        """
        try: 
            if not self.config.multi_timeframe_alignment: 
                return True, 1.0, "MTF alignment disabled" 
            
            if not indicators_15m or df_15m.empty: 
                logger.warning("No 15-min data for MTF analysis") 
                return True, 0.5, "No 15-min data available"
            
            signal_direction = 1 if "BUY" in signal_5m.get('composite_signal', '') else -1
            trend_15m = self._analyze_higher_timeframe_trend(indicators_15m, df_15m)

            # Start score
            alignment_score = 0.0

            # 1) Direction (40%)
            if trend_15m['direction'] == signal_direction:
                alignment_score += 0.4
            elif trend_15m['direction'] == 0:
                alignment_score += 0.2

            # 2) Momentum (30%)
            if trend_15m['momentum_aligned']:
                alignment_score += 0.3

            # 3) S/R room (30%)
            if trend_15m['sr_aligned']:
                alignment_score += 0.3

            # Regime/pattern leniency: strong 5m case with 15m neutral or easing
            regime = signal_5m.get('market_regime', 'NORMAL')
            strong_5m = (signal_5m.get('weighted_score', 0) >= 0.10 and 'three_white_soldiers' in ' '.join(signal_5m.get('scalping_signals', [])).lower())
            ema_slope_up = False
            try:
                ema_med = indicators_15m.get('ema', {}).get('medium_series')
                ema_slope_up = bool(ema_med is not None and len(ema_med) > 1 and ema_med.iloc[-1] > ema_med.iloc[-2])
            except Exception:
                pass

            if (regime == 'STRONG_UPTREND' and trend_15m['direction'] in (1, 0)) or (strong_5m and (trend_15m['direction'] in (0,) or ema_slope_up)):
                alignment_score = max(alignment_score, self.config.trend_alignment_threshold)
            elif (regime == 'STRONG_DOWNTREND' and trend_15m['direction'] in (-1, 0)):
                alignment_score = max(alignment_score, self.config.trend_alignment_threshold)


            # # Regime leniency for strong 5m trend when 15m is neutral/compatible
            # regime = signal_5m.get('market_regime', 'NORMAL')
            
            # if regime == 'STRONG_UPTREND' and trend_15m['direction'] in (1, 0):
            #     alignment_score = max(alignment_score, self.config.trend_alignment_threshold)
            # elif regime == 'STRONG_DOWNTREND' and trend_15m['direction'] in (-1, 0):
            #     alignment_score = max(alignment_score, self.config.trend_alignment_threshold)

            is_aligned = alignment_score >= self.config.trend_alignment_threshold
            description = f"{'Aligned' if is_aligned else 'Not aligned'} with 15-min trend (score: {alignment_score:.2f})"


            logger.info(f"MTF Analysis: {description}")
            if not is_aligned:
                logger.info(f"15m details → direction: {trend_15m['direction']}, "
                            f"momentum_aligned: {trend_15m['momentum_aligned']}, "
                            f"sr_aligned: {trend_15m['sr_aligned']}")
            else:
                logger.debug(f"15m trend details: {trend_15m}")
            return is_aligned, alignment_score, description

        except Exception as e:
            logger.error(f"MTF alignment check error: {e}")
            return True, 0.5, "MTF check error"


    
    def _analyze_higher_timeframe_trend(self, indicators: Dict, df: pd.DataFrame) -> Dict:
        """Analyze trend on higher timeframe."""
        try:
            analysis = {
                'direction': 0,  # -1: bearish, 0: neutral, 1: bullish
                'strength': 0.0,
                'momentum_aligned': False,
                'sr_aligned': False
            }
            
            # 1. Check EMA trend
            if 'ema' in indicators:
                ema_signal = indicators['ema'].get('signal', 'neutral')
                if ema_signal in ['bullish', 'golden_cross']:
                    analysis['direction'] = 1
                elif ema_signal in ['bearish', 'death_cross']:
                    analysis['direction'] = -1
            
            # 2. Check Supertrend
            if 'supertrend' in indicators:
                st_trend = indicators['supertrend'].get('trend', 'neutral')
                if st_trend == 'bullish':
                    if analysis['direction'] >= 0:
                        analysis['direction'] = 1
                        analysis['strength'] += 0.5
                elif st_trend == 'bearish':
                    if analysis['direction'] <= 0:
                        analysis['direction'] = -1
                        analysis['strength'] += 0.5
            
            # 3. Check RSI momentum
            if 'rsi' in indicators:
                rsi_value = indicators['rsi'].get('value', 50)
                if (analysis['direction'] == 1 and rsi_value > 50) or \
                   (analysis['direction'] == -1 and rsi_value < 50):
                    analysis['momentum_aligned'] = True
            
            # 4. Check support/resistance
            if not df.empty and len(df) >= 20:
                current_price = df['close'].iloc[-1]
                recent_high = df['high'].tail(20).max()
                recent_low = df['low'].tail(20).min()
                
                price_position = (current_price - recent_low) / (recent_high - recent_low) if recent_high > recent_low else 0.5
                
                if analysis['direction'] == 1 and price_position < 0.7:  # Room to go up
                    analysis['sr_aligned'] = True
                elif analysis['direction'] == -1 and price_position > 0.3:  # Room to go down
                    analysis['sr_aligned'] = True
            
            return analysis
            
        except Exception as e:
            logger.error(f"Higher timeframe analysis error: {e}")
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
                        signal = indicators[ind_name].get('signal', 'neutral')
                        if 'bullish' in str(signal).lower() or 'buy' in str(signal).lower():
                            bullish_count += 1
                        elif 'bearish' in str(signal).lower() or 'sell' in str(signal).lower():
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
        self.validator = SignalValidator(config)
        self.predictor = SignalDurationPredictor(config)
        self.consensus_analyzer = EnhancedIndicatorConsensus(config)
        self.trend_analyzer = TrendAlignmentAnalyzer(config)
        
        # History tracking
        self.signal_history = deque(maxlen=100)
        self.last_alert_time = None
        
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
            self.current_df = df_5m  # Store for market regime detection

            # Detect market session characteristics
            session_info = self.detect_session_characteristics(df_5m)
            logger.info(f"Session: {session_info['session']} | Strategy: {session_info['strategy']}")
            
            # 1. Calculate weighted signal
            signal_result = self._calculate_weighted_signal(indicators_5m)
            
            # Apply confidence adjustment based on session (MOVED HERE - AFTER signal_result creation)
            if 'confidence_adjustment' in session_info:
                signal_result['confidence'] *= session_info['confidence_adjustment']
                logger.debug(f"Session-adjusted confidence: {signal_result['confidence']:.1f}%")

                            
            # # 2. HARD FLOOR CHECK - Reject signals below 50% confidence
            # if signal_result['confidence'] < self.config.confidence_hard_floor:
            #     logger.info(f"❌ Signal rejected: Confidence {signal_result['confidence']:.1f}% "
            #                f"< hard floor {self.config.confidence_hard_floor}%")
            #     return None
            
            # 3. Price Action Validation
            if self.config.price_action_validation:
                pa_valid, pa_reason = self.price_action_validator.validate_against_price_action(
                    signal_result['composite_signal'],
                    df_5m,
                    self.config.price_action_lookback
                )
                
                if not pa_valid:
                    logger.info(f"❌ Signal rejected by price action: {pa_reason}")
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
                
                if self.config.mtf_alignment_required and not mtf_aligned:
                    logger.info(f"❌ Signal rejected: {mtf_description}")
                    return None
            
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
                return None
            
            # 8. Check cooldown
            if not self._check_cooldown(signal_result):
                logger.info("⏰ Signal in cooldown period")
                return None
            
            # 9. Calculate final confidence (with MTF boost)
            final_confidence = self._calculate_final_confidence_with_mtf(
                signal_result,
                group_consensus,
                trend_analysis,
                mtf_score
            )
            
            
            # 10. Final confidence checks (relocated hard floor)
            if final_confidence < self.config.confidence_hard_floor:
                logger.info(f"❌ Final confidence {final_confidence:.1f}% < hard floor {self.config.confidence_hard_floor}%")
                return None

            if final_confidence < self.config.min_confidence:
                logger.info(f"❌ Final confidence {final_confidence:.1f}% < minimum {self.config.min_confidence}%")
                return None
                        
            # 11. Predict duration
            duration_prediction = self.predictor.predict(
                indicators_5m, df_5m, signal_result
            )
            
            # 12. Calculate entry/exit levels
            entry_exit = self._calculate_entry_exit_levels(
                df_5m, indicators_5m, signal_result
            )
            
            # 13. Get accuracy metrics
            accuracy_metrics = self._calculate_accuracy_metrics()
            
            # 14. Get market structure
            market_structure = self._analyze_market_structure(df_5m, indicators_5m)


            # Flatten entry_exit into top-level fields for consumers (main, charts, telegram)
            final_signal = { 
                **signal_result, 
                'confidence': final_confidence, 
                'group_consensus': group_consensus, 
                'trend_analysis': trend_analysis, 
                'mtf_analysis': { 
                    'aligned': mtf_aligned, 
                    'score': mtf_score, 
                    'description': mtf_description 
                }, 
                'duration_prediction': duration_prediction, 
                'entry_exit': entry_exit, # keep nested copy 
                # Top-level flattened keys expected by charting and alert modules: 
                'entry_price': entry_exit.get('entry_price', 0), 
                'stop_loss': entry_exit.get('stop_loss', 0), 
                'take_profit': entry_exit.get('take_profit', 0), 
                'risk_reward': entry_exit.get('risk_reward', 0), 
                'accuracy_metrics': accuracy_metrics, 
                'market_structure': market_structure, 
                'timestamp': datetime.now(), 
                'timeframe': '5m', 
                'higher_timeframe': '15m' if (df_15m is not None and indicators_15m is not None) else None, 
                'action': self._get_action_from_signal(signal_result['composite_signal']) 
                }
                            

            # Update history (do NOT update last_alert_time here — alert time is set when alert is actually sent)
            self.signal_history.append(final_signal)

            logger.info("=" * 50)
            logger.info(f"✅ SIGNAL GENERATED [5m]: {signal_result['composite_signal']}")
            logger.info(f"   Confidence: {final_confidence:.1f}%")
            logger.info(f"   Score: {signal_result['weighted_score']:.3f}")
            logger.info(f"   MTF: {mtf_description}")
            logger.info(f"   Duration: {duration_prediction['estimated_minutes']} mins")
            logger.info("=" * 50)

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


    def _calculate_final_confidence_with_mtf(
        self,
        signal_result: Dict,
        group_consensus: Dict,
        trend_analysis: Dict,
        mtf_score: float
    ) -> float:
        """Calculate final confidence including MTF boost."""
        try:
            # Start with base confidence
            base_confidence = signal_result['confidence']
            
            # Factor 1: Group consensus bonus (up to +15%)
            consensus_bonus = 0
            if group_consensus:
                valid_groups = [g for g in group_consensus.values() if g.get('indicator_count', 0) > 0]
                if valid_groups:
                    avg_agreement = np.mean([g.get('agreement', 0) for g in valid_groups])
                    consensus_bonus = avg_agreement * 15
            
            # Factor 2: Trend alignment bonus (up to +10%)
            trend_bonus = 0
            if trend_analysis.get('aligned', False):
                signal_direction = 1 if 'BUY' in signal_result['composite_signal'] else -1
                trend_direction = 1 if trend_analysis['trend'] == 'bullish' else -1 if trend_analysis['trend'] == 'bearish' else 0
                if signal_direction == trend_direction:
                    trend_bonus = 10
            
            
            # Factor 3: MTF alignment bonus (up to +20%) 
            mtf_bonus = mtf_score * 20 if self.config.mtf_enabled else 0
            
            # Factor 4: Indicator agreement
            active_ratio = signal_result['active_indicators'] / len(self.config.indicator_weights)
            if active_ratio >= 0.7:
                agreement_bonus = 10
            elif active_ratio >= 0.5:
                agreement_bonus = 5
            elif active_ratio < 0.3:
                agreement_bonus = -10
            else:
                agreement_bonus = 0

            # Small bonus when EMA and MACD agree with the signal (stacked momentum)
            try:
                contributions = signal_result.get('contributions', {})
                ema_sig = str(contributions.get('ema', {}).get('signal', 'neutral')).lower()
                macd_sig = str(contributions.get('macd', {}).get('signal_type', 'neutral')).lower()
                is_buy = 'buy' in signal_result.get('composite_signal','').lower()
                is_sell = 'sell' in signal_result.get('composite_signal','').lower()

                if is_buy and any(x in ema_sig for x in ['bullish','golden_cross','above']) and 'bullish' in macd_sig:
                    trend_bonus += 5
                if is_sell and any(x in ema_sig for x in ['bearish','death_cross','below']) and 'bearish' in macd_sig:
                    trend_bonus += 5
            except Exception:
                pass

            
            # Calculate final confidence
            final_confidence = base_confidence + consensus_bonus + trend_bonus + mtf_bonus + agreement_bonus
            
            
            # Cap between 0 and 100
            final_confidence = max(0, min(100, final_confidence))
            
            logger.debug(f"Confidence calculation: Base={base_confidence:.1f}, "
                        f"Consensus=+{consensus_bonus:.1f}, Trend=+{trend_bonus:.1f}, "
                        f"MTF=+{mtf_bonus:.1f}, Agreement={agreement_bonus:+.1f}, "
                        f"Final={final_confidence:.1f}")
            
            return final_confidence
            
        except Exception as e:
            logger.error(f"Error calculating final confidence: {e}")
            return signal_result.get('confidence', 0)
    

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

                    # Expose slope for quality checks
                    extras['hist_slope'] = hist_slope                        


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
                    if signal == 'bullish':
                        contribution = weight
                        active_count += 1
                    elif signal == 'bearish':
                        contribution = -weight
                        active_count += 1
                        prediction = "Supertrend bearish → Trend change"
                        scalping_signals.append(prediction)
                    elif signal != 'neutral':
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
                            
            # Include pattern detection in signal
            pattern_bonus = 0
            if self.pattern_detector and hasattr(self, 'current_df') and not self.current_df.empty:
                try:
                    pattern_result = self.pattern_detector.detect_patterns(self.current_df)
                    
                    if pattern_result and pattern_result.get('confidence', 0) > 0:
                        pattern_signal = pattern_result.get('signal', 'neutral')
                        if pattern_signal == 'LONG' and weighted_score > 0:
                            pattern_bonus = 0.05
                            weighted_score += pattern_bonus
                            scalping_signals.append(f"Pattern applied: {pattern_result['name']}")
                            logger.info(f"Pattern boost applied: {pattern_result['name']}")
                        elif pattern_signal == 'SHORT' and weighted_score < 0:
                            pattern_bonus = -0.05
                            weighted_score += pattern_bonus
                            scalping_signals.append(f"Pattern applied: {pattern_result['name']}")
                            logger.info(f"Pattern boost applied: {pattern_result['name']}")
                        else:
                            logger.info(f"Pattern detected but ignored (contradicts bias): {pattern_result['name']}")

                except Exception as e:
                    logger.debug(f"Pattern detection skipped: {e}")

            # Confidence calculation with market regime boost
            base_confidence = abs(weighted_score) * 100
           
            if active_count >= 4:
                # Protect against division by zero
                denominator = active_count * 0.166
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
            logger.info(f"Prediction: {next_candle_prediction}")
            
            return {
                'weighted_score': weighted_score,
                'composite_signal': composite_signal,
                'confidence': confidence,
                'active_indicators': active_count,
                'contributions': contributions,
                'next_candle_prediction': next_candle_prediction,
                'scalping_signals': scalping_signals,
                'action_recommendation': self._get_scalping_action(weighted_score),
                'market_regime': market_regime
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
     
            contributions = signal_result.get('contributions', {})
            composite = signal_result.get('composite_signal', 'NEUTRAL')
            mtf_val = float(signal_result.get('mtf_score', 0.0))

            def sig(name: str) -> str:
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
            
            # Use support/resistance from indicators
            support = current_price - volatility_range
            resistance = current_price + volatility_range
            
            if 'bollinger' in indicators and isinstance(indicators['bollinger'], dict):
                support = max(support, float(indicators['bollinger'].get('lower', support)))
                resistance = min(resistance, float(indicators['bollinger'].get('upper', resistance)))
            
            if 'keltner' in indicators and isinstance(indicators['keltner'], dict):
                support = max(support, float(indicators['keltner'].get('lower', support)))
                resistance = min(resistance, float(indicators['keltner'].get('upper', resistance)))
            
            signal_type = signal_result.get('composite_signal', '')
            
            # Calculate levels based on signal
            if "BUY" in signal_type:
                entry = current_price
                
                stop_loss = max(
                    support,
                    current_price * (1 - self.config.stop_loss_percentage / 100.0)
                )
                take_profit = min(
                    resistance,
                    current_price * (1 + self.config.take_profit_percentage / 100.0)
                )

            elif "SELL" in signal_type:
                entry = current_price
                
                stop_loss = min(
                    resistance,
                    current_price * (1 + self.config.stop_loss_percentage / 100.0)
                )
                take_profit = max(
                    support,
                    current_price * (1 - self.config.take_profit_percentage / 100.0)
                )

            else:
                entry = current_price
                stop_loss = current_price - volatility_range
                take_profit = current_price + volatility_range
            
            # Calculate risk/reward ratio
            risk = abs(entry - stop_loss)
            reward = abs(take_profit - entry)
            risk_reward = (reward / risk) if risk > 0 else 0
            
            logger.info(f"Entry/Exit: Entry={entry:.2f}, SL={stop_loss:.2f}, "
                       f"TP={take_profit:.2f}, R:R={risk_reward:.2f}")
            
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
            
            # Ensure minimum range
            min_range = current_price * 0.005  # 0.5% minimum
            
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
