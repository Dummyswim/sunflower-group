"""
Integrated alert system with pattern recognition and prediction.
Orchestrates all components for live trading and backtesting.
"""
import logging
import os
import time
from datetime import datetime, timezone, timedelta
import pytz  # if you prefer pytz
from typing import Optional, Dict, List, Tuple
import pandas as pd

# Configuration and components
import config
from pattern_engine import PatternEngine
from advanced_indicators import AdvancedIndicators
from prediction_engine import PredictionEngine
from ohlc_builder import OHLCBuilder
from telegram_bot import TelegramBot
from chart_utils import ChartGenerator
from replay_engine import ReplayEngine
from signal_analyzer import SignalQualityAnalyzer

logger = logging.getLogger(__name__)

class AlertSystem:
    """
    Main alert system orchestrating all components.
    Handles both live WebSocket data and CSV replay for backtesting.
    """
    
    def __init__(self, mode: str = "live", replay_csv: Optional[str] = None):
        """
        Initialize alert system.
        
        Args:
            mode: "live" for real-time WebSocket or "replay" for CSV backtesting
            replay_csv: Path to CSV file for replay mode (required if mode="replay")
        """
        self.mode = mode
        self.replay_csv = replay_csv
        
        # Validate inputs
        if mode == "replay" and not replay_csv:
            raise ValueError("CSV path required for replay mode")
        
        # Initialize core components
        self._initialize_components()
        
        # Initialize tracking
        self._initialize_tracking()
        
        # Setup directories
        os.makedirs(config.CHART_DIR, exist_ok=True)
        
        logger.info(f"AlertSystem initialized in {mode} mode")
    
    def _initialize_components(self):
        """Initialize all system components."""
        from enhanced_patterns import EnhancedPatternRecognition
        self.enhanced_patterns = EnhancedPatternRecognition()
        
        self.ohlc_builder = OHLCBuilder(window=config.OHLC_WINDOW,
            timeframe_minutes=5  # Changed from default 1 to 5
        )
            
        self.pattern_engine = PatternEngine(
            window=config.PATTERN_WINDOW,
            default_prob=config.DEFAULT_PATTERN_PROB,
            history_file=config.PATTERN_HISTORY_FILE
        )
        self.prediction_engine = PredictionEngine()
        self.indicators = AdvancedIndicators()
        self.chart_generator = ChartGenerator()
        self.signal_analyzer = SignalQualityAnalyzer()

        # Initialize Telegram bot (optional)
        self.telegram = None
        if config.TELEGRAM_BOT_TOKEN_B64 and config.TELEGRAM_CHAT_ID:
            try:
                self.telegram = TelegramBot(
                    config.TELEGRAM_BOT_TOKEN_B64,
                    config.TELEGRAM_CHAT_ID
                )
                logger.info("Telegram bot initialized")
            except Exception as e:
                logger.warning(f"Telegram bot initialization failed: {e}")
    
    def _initialize_tracking(self):
        """Initialize tracking variables."""
        self.last_alert_time = None
        self.last_prediction = None
        self.prediction_history = []
        
        # Statistics
        self.total_patterns_detected = 0
        self.total_alerts_sent = 0
        self.session_start = datetime.now()
        
        # Performance metrics
        self.performance_metrics = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'max_confidence_correct': 0,
            'min_confidence_wrong': 1.0
        }
    
    def process_tick(self, timestamp: datetime, price: float, volume: int = 0) -> bool:
        """
        Process a single tick of market data.
        
        Args:
            timestamp: Tick timestamp
            price: Tick price (LTP)
            volume: Tick volume (optional)
            
        Returns:
            True if a new minute candle was created
        """
        # Validate price
        if not self._validate_price(price):
            return False
        
        # Add tick to OHLC builder
        new_candle = self.ohlc_builder.add_tick(timestamp, price, volume)
        
        # Analyze completed candle when new minute starts
        if new_candle and self.ohlc_builder.tick_count > 1:
            self._analyze_completed_candle()
        
        return new_candle
    
    def process_candle(self, candle: Dict):
        """
        Process a complete OHLC candle (used in replay mode).
        
        Args:
            candle: Dictionary with keys: timestamp, open, high, low, close, volume
        """
        # Validate candle data
        if not self._validate_candle(candle):
            return
        
        timestamp = candle['timestamp']
        
        # Simulate tick data from OHLC
        self.ohlc_builder.add_tick(timestamp, candle['open'], candle.get('volume', 0))
        
        if candle['high'] != candle['open']:
            self.ohlc_builder.add_tick(timestamp, candle['high'])
        
        if candle['low'] not in [candle['open'], candle['high']]:
            self.ohlc_builder.add_tick(timestamp, candle['low'])
        
        self.ohlc_builder.add_tick(timestamp, candle['close'])
        
        # Analyze the completed candle
        self._analyze_completed_candle()
    
    def _analyze_completed_candle(self):
        """Analyze patterns and generate predictions for completed candles."""
        df = self.ohlc_builder.get_completed_candles()
        
        if len(df) < config.MIN_CANDLES_FOR_ANALYSIS:
            logger.debug(f"Insufficient data: {len(df)}/{config.MIN_CANDLES_FOR_ANALYSIS} candles")
            return
        
        # Evaluate previous prediction
        if self.last_prediction:
            self._evaluate_last_prediction(df)
        
        # Perform analysis
        analysis_result = self._perform_analysis(df)
        
        if analysis_result:
            patterns, prediction, indicators_data, signal_metrics = analysis_result
            
            # Store current state
            self.last_prediction = {
                "timestamp": df.index[-1],
                "price": df['close'].iloc[-1],
                "patterns": patterns,
                "prediction": prediction,
                "indicators": indicators_data,
                "signal_metrics": signal_metrics
            } 
            
            # Check alert conditions
            if self._should_send_alert(patterns, prediction):
                self._send_alert(df, patterns, prediction, indicators_data)
            
            # Log summary
            self._log_analysis_summary(df, patterns, prediction)
    
    def _perform_analysis(self, df: pd.DataFrame) -> Optional[Tuple]:
        """
        Perform complete technical and pattern analysis.
        
        Returns:
            Tuple of (patterns, prediction, indicators_data) or None
        """
        try:

            # Detect all patterns
            talib_patterns = self.pattern_engine.detect_patterns(df)
            enhanced_patterns = self.enhanced_patterns.detect_all_patterns(df) if hasattr(self, 'enhanced_patterns') else []
            all_patterns = talib_patterns + enhanced_patterns


            for pattern in all_patterns:
            
                logger.info(f"Pattern: {pattern['name']} | Direction: {pattern['direction']} | Confidence: {pattern.get('confidence', 0):.2%}")
                                        

            # Calculate indicators
            indicators_data = {
                'atr': self.indicators.calculate_atr(df, config.ATR_PERIOD),
                'momentum': self.indicators.calculate_momentum(df, config.MOMENTUM_LOOKBACK),
                'volume_profile': self.indicators.calculate_volume_profile(df),
                'support_resistance': self.indicators.calculate_support_resistance(df)
            }
            
            # Calculate volatility ratio
            atr_ratio = self.indicators.calculate_volatility_ratio(
                df, indicators_data['atr']
            )
            
            # Generate prediction
            prediction = self.prediction_engine.predict(
                all_patterns,
                indicators_data['momentum'],
                indicators_data['volume_profile'],
                atr_ratio,
                indicators_data['support_resistance']
            )
            
            # Calculate signal quality metrics
            signal_metrics = self.signal_analyzer.calculate_signal_metrics(
                all_patterns, df, prediction, indicators_data
            )
            
            return all_patterns, prediction, indicators_data, signal_metrics

        except Exception as e:
            logger.error(f"Analysis failed: {e}", exc_info=True)
            return None
    
    def _evaluate_last_prediction(self, df: pd.DataFrame):
        """Evaluate the accuracy of the last prediction."""
        if len(df) < 2:
            return
        
        # Calculate actual movement
        prev_close = df['close'].iloc[-2]
        current_close = df['close'].iloc[-1]
        actual_move = current_close - prev_close
        
        # Determine directions
        actual_direction = "bullish" if actual_move > 0 else "bearish" if actual_move < 0 else "neutral"
        predicted_direction = self.last_prediction["prediction"]["direction"]
        confidence = self.last_prediction["prediction"]["confidence"]
        
        was_correct = (predicted_direction == actual_direction)
        
        # Update pattern performance
        for pattern in self.last_prediction.get("patterns", []):
            self.pattern_engine.update_pattern_performance(pattern["name"], was_correct)
        
        # Save pattern history
        self.pattern_engine.save_history()
        
        # Update performance metrics
        self._update_performance_metrics(was_correct, confidence)
        
        # Track prediction
        self.prediction_history.append({
            "timestamp": self.last_prediction["timestamp"],
            "predicted": predicted_direction,
            "actual": actual_direction,
            "correct": was_correct,
            "confidence": confidence,
            "price_change": actual_move,
            "price_change_pct": (actual_move / prev_close) * 100
        })
        
        # Limit history size
        if len(self.prediction_history) > 1000:
            self.prediction_history = self.prediction_history[-1000:]
        
        # Log evaluation
        accuracy = self._calculate_recent_accuracy()
        result_symbol = "[OK]" if was_correct else "[X]"
        logger.info(
            f"Prediction: {predicted_direction} vs {actual_direction} "
            f"{result_symbol} | "
            f"Confidence: {confidence:.1%} | "
            f"Accuracy: {accuracy:.1%}"
        )
    
    def _update_performance_metrics(self, was_correct: bool, confidence: float):
        """Update performance tracking metrics."""
        self.performance_metrics['total_trades'] += 1
        
        if was_correct:
            self.performance_metrics['winning_trades'] += 1
            if confidence > self.performance_metrics['max_confidence_correct']:
                self.performance_metrics['max_confidence_correct'] = confidence
        else:
            self.performance_metrics['losing_trades'] += 1
            if confidence < self.performance_metrics['min_confidence_wrong']:
                self.performance_metrics['min_confidence_wrong'] = confidence
    
    def _should_send_alert(self, patterns: List[Dict], prediction: Dict) -> bool:
        """Determine if an alert should be sent."""
        # Check cooldown
        if self.last_alert_time:
            elapsed = (datetime.now() - self.last_alert_time).total_seconds()
            if elapsed < config.COOLDOWN_SECONDS:
                logger.debug(f"Alert suppressed - cooldown ({elapsed:.0f}s/{config.COOLDOWN_SECONDS}s)")
                return False
        
        # Check pattern count
        if len(patterns) < config.MIN_PATTERNS_FOR_ALERT:
            return False
        
        # Check confidence
        if prediction["confidence"] < config.PATTERN_CONFIDENCE_THRESHOLD:
            return False
        
        # Check direction
        if prediction["direction"] == "neutral":
            return False
        
        # Check strength
        if prediction["strength"] not in ["strong", "moderate"]:
            return False
        
        return True
    
    def _send_alert(self, df: pd.DataFrame, patterns: List[Dict], 
                   prediction: Dict, indicators_data: Dict):
        """Send alert with chart via Telegram."""
        try:
            timestamp = int(time.time())
            chart_path = os.path.join(config.CHART_DIR, f"alert_{timestamp}.png")
            
            # Generate chart
            chart_created = self.chart_generator.create_pattern_chart(
                df, patterns, prediction, indicators_data, chart_path
            )
            
            # Format message
            message = self._format_alert_message(
                df, patterns, prediction, indicators_data
            )
            
            # Send alert
            if self.telegram:
                text_sent = self.telegram.send_message(message)
                
                chart_sent = False
                if chart_created and os.path.exists(chart_path):
                    chart_sent = self.telegram.send_chart(
                        "Pattern Analysis Chart", chart_path
                    )
                
                if text_sent or chart_sent:
                    self.total_alerts_sent += 1
                    self.last_alert_time = datetime.now()
                    logger.info(f"Alert #{self.total_alerts_sent} sent")
            else:
                logger.info(f"Alert triggered: {message[:100]}...")
                self.last_alert_time = datetime.now()
            
        except Exception as e:
            logger.error(f"Alert sending failed: {e}", exc_info=True)

    def _format_alert_message(self, df: pd.DataFrame, patterns: List[Dict],
                            prediction: Dict, indicators_data: Dict) -> str:
        """Format comprehensive alert message."""
        price = df['close'].iloc[-1]
        timestamp = df.index[-1]

        # Ensure timestamp is timezone-aware (assume UTC if not set)
        ist = pytz.timezone("Asia/Kolkata")
        if timestamp.tzinfo is None:
            timestamp = timestamp.tz_localize("UTC")
        timestamp_ist = timestamp.astimezone(ist)

        # Format supporting details
        pattern_text = self._format_pattern_summary(patterns)
        accuracy = self._calculate_recent_accuracy()
        win_rate = self._calculate_win_rate()

        # Safe ATR formatting
        atr_value = indicators_data.get("atr")
        atr_str = f"{atr_value:.2f}" if atr_value is not None else "N/A"

        # Safe momentum
        momentum_str = f"{indicators_data.get('momentum', 0):.2%}"

        # Safe volume trend
        volume_trend = indicators_data.get("volume_profile", {}).get("volume_trend", "N/A")

        message = f"""
<b>[PATTERN ALERT]</b>
{'-' * 25}
<b>Price:</b> {price:,.2f}
<b>Time:</b> {timestamp_ist.strftime('%H:%M:%S IST')}

<b>Patterns ({len(patterns)}):</b>
{pattern_text}

<b>Prediction:</b>
  - Direction: {prediction.get('direction', 'UNKNOWN').upper()}
  - Confidence: {prediction.get('confidence', 0):.1%}
  - Strength: {prediction.get('strength', 'N/A').upper()}

<b>Indicators:</b>
  - ATR: {atr_str}
  - Momentum: {momentum_str}
  - Volume: {volume_trend}

<b>Performance:</b>
  - Accuracy: {accuracy:.1%}
  - Win Rate: {win_rate:.1%}
  - Alerts: {self.total_alerts_sent}
{'-' * 25}
"""
        return message.strip()
    
#     def _format_alert_message(self, df: pd.DataFrame, patterns: List[Dict],
#                              prediction: Dict, indicators_data: Dict) -> str:
#         """Format comprehensive alert message."""
#         price = df['close'].iloc[-1]
#         timestamp = df.index[-1]
        
#         pattern_text = self._format_pattern_summary(patterns)
#         accuracy = self._calculate_recent_accuracy()
#         win_rate = self._calculate_win_rate()
#         # Convert timestamp to IST
#         ist = pytz.timezone("Asia/Kolkata")
#         timestamp_ist = timestamp.astimezone(ist)
        
#         message = f"""
# <b>[PATTERN ALERT]</b>
# {'-' * 25}
# <b>Price:</b> {price:,.2f}
# <b>Time:</b> {timestamp_ist.strftime('%H:%M:%S IST')}
        

# <b>Patterns ({len(patterns)}):</b>
# {pattern_text}

# <b>Prediction:</b>
#   - Direction: {prediction['direction'].upper()}
#   - Confidence: {prediction['confidence']:.1%}
#   - Strength: {prediction['strength'].upper()}

# <b>Indicators:</b>
#   - ATR: {indicators_data['atr']:.2f if indicators_data['atr'] else 'N/A'}
#   - Momentum: {indicators_data['momentum']:.2%}
#   - Volume: {indicators_data['volume_profile']['volume_trend']}

# <b>Performance:</b>
#   - Accuracy: {accuracy:.1%}
#   - Win Rate: {win_rate:.1%}
#   - Alerts: {self.total_alerts_sent}
# {'-' * 25}
# """
#         return message
    
    def _format_pattern_summary(self, patterns: List[Dict]) -> str:
        """Format pattern summary for alert message."""
        if not patterns:
            return "  - No patterns"
        
        top_patterns = sorted(
            patterns,
            key=lambda p: p.get('confidence', 0) * p.get('strength', 0),
            reverse=True
        )[:3]
        
        lines = []
        for p in top_patterns:
            name = p['name'].replace('CDL', '')
            symbol = '^' if p['direction'] == 'bullish' else 'v'
            lines.append(f"  - {name} {symbol} ({p['confidence']:.0%})")
        
        return "\n".join(lines)
    
    def _log_analysis_summary(self, df: pd.DataFrame, patterns: List[Dict],
                             prediction: Dict):
        """Log analysis summary."""
        if patterns or prediction['direction'] != 'neutral':
            price = df['close'].iloc[-1]
            pattern_names = [p['name'].replace('CDL', '') for p in patterns[:3]]
            
            logger.info(
                f"Analysis: Price={price:.2f} | "
                f"Patterns={len(patterns)} ({', '.join(pattern_names)}) | "
                f"Prediction={prediction['direction']} ({prediction['confidence']:.1%})"
            )
    
    def _validate_price(self, price: float) -> bool:
        """Validate price data."""
        if price <= 0:
            logger.warning(f"Invalid price: {price}")
            return False
        
        # Add sanity checks if needed
        if hasattr(config, 'PRICE_SANITY_MIN') and hasattr(config, 'PRICE_SANITY_MAX'):
            if not (config.PRICE_SANITY_MIN < price < config.PRICE_SANITY_MAX):
                logger.warning(f"Price outside sanity range: {price}")
                return False
        
        return True
    
    def _validate_candle(self, candle: Dict) -> bool:
        """Validate candle data."""
        required_keys = ['timestamp', 'open', 'high', 'low', 'close']
        
        for key in required_keys:
            if key not in candle:
                logger.warning(f"Missing required key in candle: {key}")
                return False
        
        # Validate OHLC relationships
        if not (candle['low'] <= candle['open'] <= candle['high'] and
                candle['low'] <= candle['close'] <= candle['high']):
            logger.warning("Invalid OHLC relationships in candle")
            return False
        
        return True
    
    def _calculate_recent_accuracy(self, last_n: int = 100) -> float:
        """Calculate accuracy over recent predictions."""
        if not self.prediction_history:
            return 0.0
        
        recent = self.prediction_history[-last_n:] if len(self.prediction_history) > last_n \
                 else self.prediction_history
        
        if not recent:
            return 0.0
        
        correct = sum(1 for p in recent if p["correct"])
        return correct / len(recent)
    
    def _calculate_win_rate(self) -> float:
        """Calculate overall win rate."""
        total = self.performance_metrics['total_trades']
        if total == 0:
            return 0.0
        
        wins = self.performance_metrics['winning_trades']
        return wins / total
    
    def _format_session_duration(self) -> str:
        """Format session duration."""
        duration = datetime.now() - self.session_start
        hours = duration.total_seconds() / 3600
        
        if hours < 1:
            return f"{int(duration.total_seconds() / 60)} minutes"
        else:
            return f"{hours:.1f} hours"
    
    def get_statistics(self) -> Dict:
        """Get comprehensive session statistics."""
        stats = {
            "mode": self.mode,
            "session_duration": self._format_session_duration(),
            "total_patterns_detected": self.total_patterns_detected,
            "total_alerts_sent": self.total_alerts_sent,
            "total_predictions": len(self.prediction_history),
            "candles_processed": len(self.ohlc_builder.candles),
            "prediction_accuracy": self._calculate_recent_accuracy(),
            "win_rate": self._calculate_win_rate(),
            "performance_metrics": self.performance_metrics
        }
        
        # Add pattern performance
        if hasattr(self.pattern_engine, 'pattern_history'):
            pattern_stats = {}
            for name, history in self.pattern_engine.pattern_history.items():
                if history:
                    pattern_stats[name] = {
                        "hit_rate": sum(history) / len(history),
                        "samples": len(history)
                    }
            stats["pattern_performance"] = pattern_stats
        
        return stats
    
    def run_replay(self, speed: float = 0.0):
        """Run system in replay mode."""
        if not self.replay_csv:
            raise ValueError("CSV path required for replay mode")
        
        logger.info(f"Starting replay: {self.replay_csv}")
        
        replay = ReplayEngine(self.replay_csv)
        stats = replay.get_statistics()
        logger.info(f"Replay data: {stats}")
        
        # Process candles
        replay.replay(self.process_candle, speed=speed)
        
        # Display results
        self._display_replay_results()
    
    def _display_replay_results(self):
        """Display comprehensive replay results."""
        stats = self.get_statistics()
        
        logger.info("=" * 50)
        logger.info("REPLAY COMPLETE")
        logger.info("=" * 50)
        
        for key, value in stats.items():
            if key not in ["pattern_performance", "performance_metrics"]:
                logger.info(f"{key}: {value}")
        
        # Performance metrics
        if "performance_metrics" in stats:
            logger.info("\nPerformance Metrics:")
            for key, value in stats["performance_metrics"].items():
                logger.info(f"  {key}: {value}")
        
        # Pattern performance
        if "pattern_performance" in stats:
            logger.info("\nTop Patterns:")
            patterns = stats["pattern_performance"]
            sorted_patterns = sorted(
                patterns.items(),
                key=lambda x: x[1]["hit_rate"] if x[1]["samples"] > 5 else 0,
                reverse=True
            )[:10]
            
            for name, perf in sorted_patterns:
                if perf["samples"] > 5:
                    logger.info(f"  {name}: {perf['hit_rate']:.1%} ({perf['samples']} samples)")
    
    def shutdown(self):
        """Gracefully shutdown the system."""
        logger.info("Shutting down alert system...")
        
        # Save pattern history
        if hasattr(self.pattern_engine, 'save_history'):
            self.pattern_engine.save_history()
        
        # Send final notification
        if self.telegram:
            stats = self.get_statistics()
            message = (
                f"System Stopped\n"
                f"Duration: {stats['session_duration']}\n"
                f"Alerts: {stats['total_alerts_sent']}\n"
                f"Accuracy: {stats['prediction_accuracy']:.1%}"
            )
            self.telegram.send_message(message)
        
        logger.info("Shutdown complete")
