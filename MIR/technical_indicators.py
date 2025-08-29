"""
Enhanced technical indicators module with improved signal generation and real accuracy tracking.

Signal Generators:
- TechnicalIndicators: Static methods for indicator calculations
- SignalGenerator: Basic weighted signal generation for simple strategies  
- EnhancedSignalGenerator: Advanced predictive signals with market structure analysis
"""
import pandas as pd
import numpy as np
import logging
from typing import Dict, Optional, Tuple, List, Any
from scipy import stats
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

@dataclass
class TradingSignal:
    """Trading signal with all metrics."""
    signal_type: str
    action: str
    confidence: float
    strength: float
    entry_price: float
    stop_loss: float
    take_profit: float
    risk_reward: float

@dataclass
class SignalRecord:
    """Record of a signal for accuracy tracking."""
    timestamp: datetime
    signal_type: str
    action: str
    price: float
    confidence: float
    predicted_duration: int
    actual_outcome: Optional[str] = None
    actual_duration: Optional[int] = None
    profit_loss: Optional[float] = None

class SignalPerformanceTracker:
    """Track real signal performance for accuracy calculation."""
    
    def __init__(self, lookback_period: int = 100):
        self.signals = deque(maxlen=lookback_period)
        self.completed_trades = deque(maxlen=lookback_period)
        self.current_position = None
        self.entry_price = None
        self.entry_time = None
        logger.info(f"SignalPerformanceTracker initialized with lookback={lookback_period}")
    
    def record_signal(self, signal_type: str, action: str, price: float, 
                     confidence: float, predicted_duration: int):
        """Record a new signal."""
        record = SignalRecord(
            timestamp=datetime.now(),
            signal_type=signal_type,
            action=action,
            price=price,
            confidence=confidence,
            predicted_duration=predicted_duration
        )
        logger.debug(
           f"Signal recorded -> type={signal_type}, action={action}, price={price:.2f}, "
            f"confidence={confidence:.1f}, duration={predicted_duration}m"
)
        self.signals.append(record)
        
        # Track position entry
        if action in ['buy', 'sell'] and self.current_position is None:
            self.current_position = action
            self.entry_price = price
            self.entry_time = datetime.now()
            logger.debug(f"Position entered: {action} at {price:.2f}")
    
    def update_position(self, current_price: float):
        """Update current position with market price."""
        if self.current_position and self.entry_price:
            duration = int((datetime.now() - self.entry_time).total_seconds() / 60)
            
            # Calculate profit/loss
            if self.current_position == 'buy':
                profit_loss = ((current_price - self.entry_price) / self.entry_price) * 100
            else:  # sell
                profit_loss = ((self.entry_price - current_price) / self.entry_price) * 100
            
            # Check if we should close position (simplified logic)
            if abs(profit_loss) > 0.5:  # 0.5% move
                self._close_position(current_price, profit_loss, duration)
    
    def _close_position(self, exit_price: float, profit_loss: float, duration: int):
        """Close current position and record outcome."""
        if self.current_position:
            outcome = 'win' if profit_loss > 0 else 'loss'
            
            # Find the corresponding signal
            for signal in reversed(self.signals):
                if signal.action == self.current_position and signal.actual_outcome is None:
                    signal.actual_outcome = outcome
                    signal.actual_duration = duration
                    signal.profit_loss = profit_loss
                    break
            
            # Record completed trade
            self.completed_trades.append({
                'entry_time': self.entry_time,
                'exit_time': datetime.now(),
                'position': self.current_position,
                'entry_price': self.entry_price,
                'exit_price': exit_price,
                'profit_loss': profit_loss,
                'duration': duration,
                'outcome': outcome
            })
            
            logger.info(f"Position closed: {self.current_position} - {outcome} ({profit_loss:.2f}%)")
            
            # Reset position
            self.current_position = None
            self.entry_price = None
            self.entry_time = None
    
    def calculate_metrics(self) -> Dict:
        """Calculate real accuracy metrics from trading history."""
        try:
            if len(self.completed_trades) < 3:
                logger.debug("Insufficient trade history for metrics - returning defaults")
                return {
                    "signal_accuracy": 50.0,
                    "confidence_sustain": 50.0,
                    "win_rate": 50.0,
                    "avg_profit": 0.0,
                    "sharpe_ratio": 0.0,
                    "total_trades": len(self.completed_trades)
                }
            
            # Calculate win rate
            wins = sum(1 for trade in self.completed_trades if trade['outcome'] == 'win')
            total = len(self.completed_trades)
            win_rate = (wins / total) * 100 if total > 0 else 50.0
            
            # Calculate average profit
            profits = [trade['profit_loss'] for trade in self.completed_trades]
            avg_profit = np.mean(profits) if profits else 0.0
            
            # Calculate Sharpe ratio (simplified)
            if len(profits) > 1:
                sharpe_ratio = np.mean(profits) / (np.std(profits) + 0.0001)
            else:
                sharpe_ratio = 0.0
            
            # Calculate signal accuracy (signals that led to profitable trades)
            accurate_signals = 0
            total_signals = 0
            
            for signal in self.signals:
                if signal.actual_outcome:
                    if signal.actual_outcome == 'win':
                        accurate_signals += 1
                    total_signals += 1
            
            signal_accuracy = (accurate_signals / total_signals * 100) if total_signals > 0 else 50.0
            
            # Calculate confidence correlation (how well confidence predicts success)
            if total_signals >= 5:
                confidences = []
                outcomes = []
                
                for signal in self.signals:
                    if signal.actual_outcome:
                        confidences.append(signal.confidence)
                        outcomes.append(1 if signal.actual_outcome == 'win' else 0)
                
                if len(confidences) > 1:
                    try:
                        # Sanitize arrays before correlation
                        confidences = np.array(confidences)
                        outcomes = np.array(outcomes)
                        
                        # Remove NaN and Inf values
                        valid_mask = ~(np.isnan(confidences) | np.isinf(confidences) | 
                                     np.isnan(outcomes) | np.isinf(outcomes))
                        
                        if np.sum(valid_mask) > 1:
                            valid_conf = confidences[valid_mask]
                            valid_out = outcomes[valid_mask]
                            correlation = np.corrcoef(valid_conf, valid_out)[0, 1]
                            confidence_sustain = abs(correlation) * 100 if not np.isnan(correlation) else 50.0
                        else:
                            confidence_sustain = 50.0
                    except Exception as e:
                        logger.warning(f"Confidence correlation calculation failed: {e}")
                        confidence_sustain = 50.0
                else:
                    confidence_sustain = 50.0
            else:
                confidence_sustain = 50.0
            
            metrics = {
                "signal_accuracy": round(signal_accuracy, 1),
                "confidence_sustain": round(confidence_sustain, 1),
                "win_rate": round(win_rate, 1),
                "avg_profit": round(avg_profit, 2),
                "sharpe_ratio": round(sharpe_ratio, 2),
                "total_trades": total
            }
            
            logger.info(f"Performance metrics calculated: Win Rate={win_rate:.1f}%, Accuracy={signal_accuracy:.1f}%")
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating accuracy metrics: {e}")
            logger.debug("Returning default metrics due to calculation error")
            return {
                "signal_accuracy": 50.0,
                "confidence_sustain": 50.0,
                "win_rate": 50.0,
                "avg_profit": 0.0,
                "sharpe_ratio": 0.0,
                "total_trades": 0
            }

class MomentumTracker:
    """Track momentum across multiple timeframes."""
    
    def __init__(self):
        self.momentum_history = deque(maxlen=100)
        logger.debug("MomentumTracker initialized")
    
    def update(self, momentum_data: Dict):
        """Update momentum history."""
        self.momentum_history.append({
            "timestamp": pd.Timestamp.now(),
            "data": momentum_data
        })
        logger.debug(f"Momentum updated: {momentum_data.get('macd', {}).get('current', 0):.3f}")
    
    def get_trend(self) -> str:
        """Get overall momentum trend."""
        if len(self.momentum_history) < 5:
            logger.debug("Insufficient momentum history - returning neutral")
            return "neutral"
        
        recent = list(self.momentum_history)[-5:]
        increasing = sum(1 for i in range(1, len(recent)) 
                        if recent[i]['data'].get('macd', {}).get('current', 0) > 
                           recent[i-1]['data'].get('macd', {}).get('current', 0))
        
        if increasing >= 3:
            trend = "increasing"
        elif increasing <= 1:
            trend = "decreasing"
        else:
            trend = "stable"
            
        logger.debug(f"Momentum trend: {trend}")
        return trend

class EnhancedSignalGenerator:
    """
    Advanced signal generator with predictive capabilities and market structure analysis.
    Use this for sophisticated trading strategies that consider market context.
    """
    
    def __init__(self):
        """Initialize enhanced signal generator."""
        self.signal_history = deque(maxlen=200)
        self.trade_history = deque(maxlen=100)
        self.momentum_tracker = MomentumTracker()
        self.performance_tracker = SignalPerformanceTracker()
        logger.info("Enhanced SignalGenerator initialized with performance tracking")
    
    def generate_trading_signal(self, indicators: Dict, ohlcv_df: pd.DataFrame, weights: Dict) -> Dict:
        """Generate predictive trading signals with real accuracy tracking."""
        try:
            logger.info("=== Starting Enhanced Signal Generation ===")
            logger.debug(f"Indicators received: {list(indicators.keys())}")
            logger.debug(f"OHLCV rows: {len(ohlcv_df)}, Weights: {weights}")
            
            # Update position with current price
            if len(ohlcv_df) > 0:
                current_price = ohlcv_df['close'].iloc[-1]
                self.performance_tracker.update_position(current_price)
            
            # Identify market structure
            market_structure = self._identify_market_structure(ohlcv_df)
            logger.info(f"Market structure identified: {market_structure['trend']} "
                       f"(strength: {market_structure['trend_strength']:.1f}%)")
            
            # Get leading indicators
            leading_signals = self._get_leading_signals(indicators, ohlcv_df)
            logger.info(f"Leading signals: momentum={leading_signals['momentum']:.3f}, "
                       f"overbought={leading_signals['overbought']}, "
                       f"oversold={leading_signals['oversold']}")
            
            # Calculate entry points
            entry_points = self._calculate_entry_points(ohlcv_df, market_structure)
            logger.debug(f"Entry points: near_support={entry_points['near_support']}, "
                        f"near_resistance={entry_points['near_resistance']}")
            
            # Generate signal based on confluence
            signal_type, confidence = self._determine_signal(
                market_structure, leading_signals, entry_points
            )
            
            # Calculate stop loss and targets using unified ATR
            current_price = ohlcv_df['close'].iloc[-1]
            atr = TechnicalIndicators.calculate_atr(
                ohlcv_df['high'], 
                ohlcv_df['low'], 
                ohlcv_df['close']
            ).iloc[-1]
            
            if pd.isna(atr) or atr <= 0:
                logger.warning("ATR calculation failed, using price std as fallback")
                atr = ohlcv_df['close'].std()
            
            if "BUY" in signal_type:
                stop_loss = entry_points['support'] - (atr * 0.5)
                take_profit = entry_points['resistance']
                action = "buy"
            elif "SELL" in signal_type:
                stop_loss = entry_points['resistance'] + (atr * 0.5)
                take_profit = entry_points['support']
                action = "sell"
            else:
                stop_loss = current_price - atr
                take_profit = current_price + atr
                action = "hold"
            
            # Predict duration
            duration_prediction = self._predict_duration(leading_signals['momentum'], indicators)
            
            # Record signal for tracking
            self.performance_tracker.record_signal(
                signal_type, action, current_price, confidence,
                duration_prediction.get('estimated_minutes', 10)
            )
            
            # Get real accuracy metrics
            accuracy_metrics = self.performance_tracker.calculate_metrics()
            
            # Add to signal history
            self.signal_history.append({
                "timestamp": datetime.now(),
                "signal": signal_type,
                "action": action,
                "price": current_price,
                "confidence": confidence
            })
            
            result = {
                "composite_signal": signal_type,
                "action": action,
                "confidence": min(confidence, 100),
                "weighted_score": leading_signals['momentum'],
                "entry_price": current_price,
                "stop_loss": round(stop_loss, 2),
                "take_profit": round(take_profit, 2),
                "risk_reward": abs(take_profit - current_price) / abs(current_price - stop_loss) if stop_loss != current_price else 0,
                "market_structure": market_structure,
                "entry_points": entry_points,
                "accuracy_metrics": accuracy_metrics,
                "duration_prediction": duration_prediction,
                "contributions": self._get_indicator_contributions(indicators, weights)
            }
            
            logger.info(f"=== Signal Generated: {signal_type} ===")
            logger.debug(f"Indicators received: {list(indicators.keys())}")
            logger.debug(f"OHLCV rows: {len(ohlcv_df)}, Weights: {weights}")
            logger.info(f"Confidence: {confidence:.1f}%, Accuracy: {accuracy_metrics['signal_accuracy']:.1f}%")
            logger.info(f"Action: {action}, Entry: {current_price:.2f}, SL: {stop_loss:.2f}, TP: {take_profit:.2f}")
            logger.info(f"Signal generated: {signal_type} with {confidence:.1f}% confidence "
                       f"(Accuracy: {accuracy_metrics['signal_accuracy']:.1f}%)")
            
            return result
            
        except Exception as e:
            logger.error(f"Signal generation error: {e}", exc_info=True)
            logger.warning("Returning neutral signal due to generation error")
            return self._get_neutral_signal()
    
    def _determine_signal(self, market_structure: Dict, leading_signals: Dict, 
                         entry_points: Dict) -> Tuple[str, float]:
        """Determine signal type and confidence based on market conditions."""
        logger.debug("Determining signal from market conditions")
        
        if market_structure['trend'] == 'uptrend':
            if leading_signals['momentum'] > 0.3 and entry_points['near_support']:
                logger.info("Strong BUY signal detected - uptrend with support bounce")
                return "STRONG_BUY", 80 + (leading_signals['strength'] * 15)
            elif leading_signals['momentum'] > 0 and entry_points['risk_reward_favorable']:
                logger.info("BUY signal detected - uptrend with favorable R:R")
                return "BUY", 65 + (leading_signals['strength'] * 10)
            elif leading_signals['momentum'] < -0.5 and entry_points['near_resistance']:
                logger.debug("Weak SELL in uptrend - near resistance")
                return "WEAK_SELL", 50
            else:
                return "HOLD", 40
                
        elif market_structure['trend'] == 'downtrend':
            if leading_signals['momentum'] < -0.3 and entry_points['near_resistance']:
                logger.info("Strong SELL signal detected - downtrend with resistance rejection")
                return "STRONG_SELL", 80 + (abs(leading_signals['strength']) * 15)
            elif leading_signals['momentum'] < 0 and not entry_points['risk_reward_favorable']:
                logger.info("SELL signal detected - downtrend continuation")
                return "SELL", 65 + (abs(leading_signals['strength']) * 10)
            elif leading_signals['momentum'] > 0.5 and entry_points['near_support']:
                logger.debug("Weak BUY in downtrend - near support")
                return "WEAK_BUY", 50
            else:
                return "HOLD", 40
                
        else:  # Ranging market
            if entry_points['near_support'] and leading_signals['oversold']:
                logger.info("BUY signal in range - oversold at support")
                return "BUY", 60
            elif entry_points['near_resistance'] and leading_signals['overbought']:
                logger.info("SELL signal in range - overbought at resistance")
                return "SELL", 60
            else:
                return "NEUTRAL", 30
    
    def _get_indicator_contributions(self, indicators: Dict, weights: Dict) -> Dict:
        """Calculate individual indicator contributions."""
        contributions = {}
        
        for name, data in indicators.items():
            if name in weights and data:
                signal = data.get('signal', 'neutral')
                weight = weights.get(name, 0)
                
                # Map signal to value
                signal_values = {
                    'strong_buy': 1.0, 'buy': 0.75, 'bullish': 0.5,
                    'oversold': 0.5, 'weak_buy': 0.25, 'neutral': 0,
                    'weak_sell': -0.25, 'bearish': -0.5, 'overbought': -0.5,
                    'sell': -0.75, 'strong_sell': -1.0
                }
                
                value = signal_values.get(signal, 0)
                contribution = value * weight
                
                contributions[name] = {
                    'signal': signal,
                    'value': value,
                    'weight': weight,
                    'contribution': round(contribution, 3)
                }
        
        return contributions
    
    def _identify_market_structure(self, ohlcv_df: pd.DataFrame) -> Dict:
        """Identify market structure (trend, support, resistance)."""
        try:
            closes = ohlcv_df['close'].values
            highs = ohlcv_df['high'].values
            lows = ohlcv_df['low'].values
            
            logger.debug(f"Analyzing market structure with {len(closes)} candles")
            
            # Calculate trend using multiple timeframes
            sma_20 = np.mean(closes[-20:]) if len(closes) >= 20 else np.mean(closes)
            sma_50 = np.mean(closes[-50:]) if len(closes) >= 50 else sma_20
            
            # Identify trend
            if closes[-1] > sma_20 > sma_50:
                trend = "uptrend"
            elif closes[-1] < sma_20 < sma_50:
                trend = "downtrend"
            else:
                trend = "ranging"
            
            # Find support and resistance
            recent_highs = highs[-20:] if len(highs) >= 20 else highs
            recent_lows = lows[-20:] if len(lows) >= 20 else lows
            
            resistance = np.max(recent_highs)
            support = np.min(recent_lows)
            pivot = (resistance + support + closes[-1]) / 3
            
            structure = {
                "trend": trend,
                "resistance": resistance,
                "support": support,
                "pivot": pivot,
                "trend_strength": abs(closes[-1] - sma_50) / sma_50 * 100
            }
            
            logger.debug(f"Market trend: {trend}, Support: {support:.2f}, Resistance: {resistance:.2f}")
            return structure
            
        except Exception as e:
            logger.error(f"Market structure error: {e}")
            logger.warning("Returning neutral market structure due to error")
            return {"trend": "neutral", "resistance": 0, "support": 0, "pivot": 0, "trend_strength": 0}

    def _get_leading_signals(self, indicators: Dict, ohlcv_df: pd.DataFrame) -> Dict:
        """Get leading indicator signals that predict future movement."""
        try:
            signals = {
                "momentum": 0,
                "strength": 0,
                "overbought": False,
                "oversold": False
            }
            
            logger.debug("Calculating leading signals from indicators")
            
            # MACD momentum (leading)
            if indicators.get('macd') and 'histogram' in indicators['macd']:
                hist = indicators['macd']['histogram']
                if hist > 0:
                    signals['momentum'] += 0.3
                    if indicators['macd'].get('histogram_series') is not None:
                        hist_series = indicators['macd']['histogram_series'].tail(3)
                        if len(hist_series) >= 3 and hist_series.iloc[-1] > hist_series.iloc[-2]:
                            signals['momentum'] += 0.2
                            logger.debug("MACD momentum increasing - bullish")
                else:
                    signals['momentum'] -= 0.3
                    logger.debug("MACD histogram negative - bearish")
            
            # RSI divergence (leading)
            if indicators.get('rsi'):
                rsi_val = indicators['rsi'].get('rsi', 50)
                if rsi_val < 30:
                    signals['oversold'] = True
                    signals['momentum'] += 0.2
                    logger.debug(f"RSI oversold: {rsi_val:.1f} - potential bounce")
                elif rsi_val > 70:
                    signals['overbought'] = True
                    signals['momentum'] -= 0.2
                    logger.debug(f"RSI overbought: {rsi_val:.1f} - potential reversal")
            
            # Supertrend (trend confirmation)
            if indicators.get('supertrend'):
                if indicators['supertrend'].get('trend') == 'bullish':
                    signals['momentum'] += 0.1
                    logger.debug("Supertrend bullish - trend confirmation")
                else:
                    signals['momentum'] -= 0.1
                    logger.debug("Supertrend bearish - trend confirmation")
            
            signals['strength'] = abs(signals['momentum'])
            
            logger.debug(f"Leading signals calculated: momentum={signals['momentum']:.3f}, strength={signals['strength']:.3f}")
            return signals
            
        except Exception as e:
            logger.error(f"Leading signals error: {e}")
            logger.warning("Returning neutral leading signals due to error")
            return {"momentum": 0, "strength": 0, "overbought": False, "oversold": False}

    def _calculate_entry_points(self, ohlcv_df: pd.DataFrame, market_structure: Dict) -> Dict:
        """Calculate optimal entry points."""
        try:
            current_price = ohlcv_df['close'].iloc[-1]
            
            # Distance from support/resistance
            dist_to_support = (current_price - market_structure['support']) / current_price * 100
            dist_to_resistance = (market_structure['resistance'] - current_price) / current_price * 100
            
            entry_points = {
                "support": market_structure['support'],
                "resistance": market_structure['resistance'],
                "near_support": dist_to_support < 0.5,
                "near_resistance": dist_to_resistance < 0.5,
                "risk_reward_favorable": dist_to_resistance > (dist_to_support * 2)
            }
            
            logger.debug(f"Entry analysis: Price={current_price:.2f}, "
                        f"Support dist={dist_to_support:.2f}%, "
                        f"Resistance dist={dist_to_resistance:.2f}%")
            
            return entry_points
            
        except Exception as e:
            logger.error(f"Entry points error: {e}")
            logger.warning("Returning default entry points due to error")
            return {"support": 0, "resistance": 0, "near_support": False, 
                   "near_resistance": False, "risk_reward_favorable": False}

    def _calculate_atr(self, ohlcv_df: pd.DataFrame) -> float:
        """Calculate ATR for stop loss calculation."""
        try:
            if len(ohlcv_df) < 14:
                return ohlcv_df['close'].std()
            
            high = ohlcv_df['high']
            low = ohlcv_df['low']
            close = ohlcv_df['close']
            
            tr1 = high - low
            tr2 = abs(high - close.shift())
            tr3 = abs(low - close.shift())
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            
            atr = tr.rolling(window=14).mean().iloc[-1]
            logger.debug(f"ATR calculated: {atr:.2f}")
            
            return atr
            
        except Exception as e:
            logger.error(f"ATR calculation error: {e}")
            return 0

    def _predict_duration(self, momentum: float, indicators: Dict) -> Dict:
        """Predict signal duration based on momentum and indicators."""
        try:
            # Use the existing static method
            duration_data = TechnicalIndicators.predict_signal_duration(
                indicators, 
                pd.DataFrame()
            )
            
            # Adjust based on momentum
            if abs(momentum) > 0.5:
                duration_data['estimated_minutes'] = int(duration_data.get('estimated_minutes', 10) * 1.5)
                logger.debug(f"Duration adjusted for high momentum: {duration_data['estimated_minutes']} mins")
            
            return duration_data
            
        except Exception as e:
            logger.error(f"Duration prediction error: {e}")
            logger.warning("Returning default duration prediction due to error")
            return {
                "estimated_minutes": 10,
                "confidence": "low",
                "strength_trend": "stable",
                "key_levels": {}
            }

    def _get_neutral_signal(self) -> Dict:
        """Return neutral signal when error occurs."""
        logger.warning("Returning NEUTRAL signal due to error or undefined conditions")
        return {
            "composite_signal": "NEUTRAL",
            "action": "hold",
            "confidence": 0,
            "weighted_score": 0,
            "entry_price": 0,
            "stop_loss": 0,
            "take_profit": 0,
            "risk_reward": 0,
            "accuracy_metrics": {"signal_accuracy": 50, "confidence_sustain": 50, "win_rate": 50},
            "duration_prediction": {"estimated_minutes": 0, "confidence": "low"},
            "contributions": {},
            "market_structure": {"trend": "neutral", "resistance": 0, "support": 0, "pivot": 0},
            "entry_points": {"near_support": False, "near_resistance": False}
        }

# Keep existing TechnicalIndicators class below (no changes needed)
class TechnicalIndicators:
    """
    Static technical indicator calculations.
    Use these methods directly for indicator values without signal generation.
    """

    @staticmethod
    def sanitize_array(arr: np.ndarray) -> np.ndarray:
        """Sanitize numpy array by removing NaN and Inf values."""
        if len(arr) == 0:
            return arr
        
        # Replace inf with large finite values
        arr = np.where(np.isinf(arr), np.nan, arr)
        
        # Forward fill NaN values
        if np.all(np.isnan(arr)):
            return np.zeros_like(arr)
        
        # Use pandas for forward/backward fill (updated syntax)
        series = pd.Series(arr)
        series = series.ffill().bfill().fillna(0)  # Updated to use ffill() and bfill()
        
        return series.values

    
    # @staticmethod
    # def sanitize_array(arr: np.ndarray) -> np.ndarray:
    #     """Sanitize numpy array by removing NaN and Inf values."""
    #     if len(arr) == 0:
    #         return arr
        
    #     # Replace inf with large finite values
    #     arr = np.where(np.isinf(arr), np.nan, arr)
        
    #     # Forward fill NaN values
    #     if np.all(np.isnan(arr)):
    #         return np.zeros_like(arr)
        
    #     # Use pandas for forward fill
    #     series = pd.Series(arr)
    #     series = series.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
    #     return series.values
    
    @staticmethod
    def calculate_ema(prices: pd.Series, period: int) -> pd.Series:
        """Calculate Exponential Moving Average."""
        try:
            if len(prices) < period:
                logger.debug(f"Insufficient data for EMA({period}): {len(prices)} < {period}")
                return pd.Series(index=prices.index)
            ema = prices.ewm(span=period, adjust=False).mean()
            logger.debug(f"EMA({period}) calculated: {ema.iloc[-1]:.2f}")
            return ema
        except Exception as e:
            logger.error(f"EMA calculation error: {e}")
            return pd.Series(index=prices.index)
    
    @staticmethod
    def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Average True Range - Single unified implementation."""
        try:
            if len(high) < period:
                logger.debug(f"Insufficient data for ATR: {len(high)} < {period}")
                return pd.Series(index=close.index)
                
            tr1 = high - low
            tr2 = abs(high - close.shift())
            tr3 = abs(low - close.shift())
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(window=period).mean()
            
            return atr
            
        except Exception as e:
            logger.error(f"ATR calculation error: {e}")
            return pd.Series(index=close.index)
    
    @staticmethod
    def calculate_macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict:
        """Calculate MACD indicator."""
        try:
            if len(prices) < slow + signal:
                logger.warning(f"Insufficient data for MACD: {len(prices)} < {slow + signal}")
                return {"signal": "neutral"}
            
            ema_fast = TechnicalIndicators.calculate_ema(prices, fast)
            ema_slow = TechnicalIndicators.calculate_ema(prices, slow)
            macd_line = ema_fast - ema_slow
            signal_line = macd_line.ewm(span=signal, adjust=False).mean()
            histogram = macd_line - signal_line
            
            # Determine signal
            current_hist = histogram.iloc[-1]
            prev_hist = histogram.iloc[-2]
            
            if current_hist > 0 and prev_hist <= 0:
                signal_type = "buy"
                logger.debug("MACD bullish crossover detected")
            elif current_hist < 0 and prev_hist >= 0:
                signal_type = "sell"
                logger.debug("MACD bearish crossover detected")
            elif current_hist > 0:
                signal_type = "bullish"
            elif current_hist < 0:
                signal_type = "bearish"
            else:
                signal_type = "neutral"
            
            result = {
                "macd": macd_line.iloc[-1],
                "signal": signal_line.iloc[-1],
                "histogram": histogram.iloc[-1],
                "signal": signal_type,
                "macd_series": macd_line,
                "signal_series": signal_line,
                "histogram_series": histogram
            }
            
            logger.debug(f"MACD: {result['macd']:.2f}, Signal: {result['signal']:.2f}, Hist: {result['histogram']:.2f}")
            return result
            
        except Exception as e:
            logger.error(f"MACD calculation error: {e}")
            return {"signal": "neutral"}
    
    @staticmethod
    def calculate_rsi(prices: pd.Series, period: int = 14) -> Dict:
        """Calculate RSI indicator."""
        try:
            if len(prices) < period + 1:
                logger.warning(f"Insufficient data for RSI: {len(prices)} < {period + 1}")
                return {"signal": "neutral"}
            
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            current_rsi = rsi.iloc[-1]
            
            if current_rsi >= 70:
                signal_type = "overbought"
                logger.debug(f"RSI overbought: {current_rsi:.1f}")
            elif current_rsi <= 30:
                signal_type = "oversold"
                logger.debug(f"RSI oversold: {current_rsi:.1f}")
            elif current_rsi >= 60:
                signal_type = "bullish"
            elif current_rsi <= 40:
                signal_type = "bearish"
            else:
                signal_type = "neutral"
            
            return {
                "rsi": current_rsi,
                "signal": signal_type,
                "rsi_series": rsi
            }
            
        except Exception as e:
            logger.error(f"RSI calculation error: {e}")
            return {"signal": "neutral"}
    
    @staticmethod
    def calculate_vwap(prices: pd.Series, volumes: pd.Series, period: int = 20) -> Dict:
        """Calculate VWAP indicator."""
        try:
            if len(prices) < period:
                logger.warning(f"Insufficient data for VWAP: {len(prices)} < {period}")
                return {"signal": "neutral"}
            
            typical_price = prices
            cumulative_tpv = (typical_price * volumes).rolling(window=period).sum()
            cumulative_volume = volumes.rolling(window=period).sum()
            
            # Avoid division by zero
            cumulative_volume = cumulative_volume.replace(0, 1)
            vwap = cumulative_tpv / cumulative_volume
            
            current_price = prices.iloc[-1]
            current_vwap = vwap.iloc[-1]
            
            if np.isnan(current_vwap):
                logger.warning("VWAP is NaN, using price as fallback")
                current_vwap = current_price
            
            deviation = ((current_price - current_vwap) / current_vwap) * 100
            
            if deviation > 2:
                signal_type = "overbought"
            elif deviation < -2:
                signal_type = "oversold"
            elif deviation > 0:
                signal_type = "bullish"
            elif deviation < 0:
                signal_type = "bearish"
            else:
                signal_type = "neutral"
            
            logger.debug(f"VWAP: {current_vwap:.2f}, Deviation: {deviation:.2f}%")
            
            return {
                "vwap": current_vwap,
                "deviation": deviation,
                "signal": signal_type,
                "vwap_series": vwap
            }
            
        except Exception as e:
            logger.error(f"VWAP calculation error: {e}")
            return {"signal": "neutral"}
    
    @staticmethod
    def calculate_keltner_channels(prices: pd.Series, high: pd.Series, low: pd.Series, 
                                   period: int = 20, multiplier: float = 2.0) -> Dict:
        """Calculate Keltner Channels."""
        try:
            if len(prices) < period:
                logger.warning(f"Insufficient data for Keltner: {len(prices)} < {period}")
                return {"signal": "neutral"}
            
            middle = prices.rolling(window=period).mean()
            atr = TechnicalIndicators.calculate_atr(high, low, prices, period)
            upper = middle + (multiplier * atr)
            lower = middle - (multiplier * atr)
            
            current_price = prices.iloc[-1]
            current_upper = upper.iloc[-1]
            current_lower = lower.iloc[-1]
            current_middle = middle.iloc[-1]
            
            if current_price > current_upper:
                signal_type = "overbought"
                position = "above_upper"
            elif current_price < current_lower:
                signal_type = "oversold"
                position = "below_lower"
            elif current_price > current_middle:
                signal_type = "bullish"
                position = "above_middle"
            elif current_price < current_middle:
                signal_type = "bearish"
                position = "below_middle"
            else:
                signal_type = "neutral"
                position = "at_middle"
            
            logger.debug(f"Keltner position: {position}")
            
            return {
                "upper": current_upper,
                "middle": current_middle,
                "lower": current_lower,
                "position": position,
                "signal": signal_type,
                "upper_series": upper,
                "middle_series": middle,
                "lower_series": lower
            }
            
        except Exception as e:
            logger.error(f"Keltner Channels calculation error: {e}")
            return {"signal": "neutral"}
    
    @staticmethod
    def calculate_supertrend(prices: pd.Series, high: pd.Series, low: pd.Series,
                            period: int = 10, multiplier: float = 3.0) -> Dict:
        """Calculate Supertrend indicator."""
        try:
            if len(prices) < period:
                logger.warning(f"Insufficient data for Supertrend: {len(prices)} < {period}")
                return {"signal": "neutral"}
            
            atr = TechnicalIndicators.calculate_atr(high, low, prices, period)
            hl_avg = (high + low) / 2
            
            upper_band = hl_avg + (multiplier * atr)
            lower_band = hl_avg - (multiplier * atr)
            
            supertrend = pd.Series(index=prices.index, dtype=float)
            direction = pd.Series(index=prices.index, dtype=float)
            
            for i in range(period, len(prices)):
                if prices.iloc[i] <= upper_band.iloc[i]:
                    supertrend.iloc[i] = upper_band.iloc[i]
                    direction.iloc[i] = -1
                else:
                    supertrend.iloc[i] = lower_band.iloc[i]
                    direction.iloc[i] = 1
            
            current_direction = direction.iloc[-1]
            prev_direction = direction.iloc[-2] if len(direction) > 1 else current_direction
            
            if current_direction == 1 and prev_direction == -1:
                signal_type = "buy"
                trend = "bullish"
                logger.debug("Supertrend bullish signal")
            elif current_direction == -1 and prev_direction == 1:
                signal_type = "sell"
                trend = "bearish"
                logger.debug("Supertrend bearish signal")
            elif current_direction == 1:
                signal_type = "bullish"
                trend = "bullish"
            elif current_direction == -1:
                signal_type = "bearish"
                trend = "bearish"
            else:
                signal_type = "neutral"
                trend = "neutral"
            
            return {
                "supertrend": supertrend.iloc[-1],
                "direction": current_direction,
                "trend": trend,
                "signal": signal_type,
                "supertrend_series": supertrend,
                "direction_series": direction
            }
            
        except Exception as e:
            logger.error(f"Supertrend calculation error: {e}")
            return {"signal": "neutral"}
    
    @staticmethod
    def calculate_impulse_macd(prices: pd.Series) -> Dict:
        """Calculate Impulse MACD system."""
        try:
            if len(prices) < 35:
                logger.warning(f"Insufficient data for Impulse MACD: {len(prices)} < 35")
                return {"signal": "neutral"}
            
            # Calculate MACD
            macd_result = TechnicalIndicators.calculate_macd(prices)
            
            # Calculate EMA
            ema_13 = TechnicalIndicators.calculate_ema(prices, 13)
            
            # Determine impulse state
            macd_histogram = macd_result.get("histogram", 0)
            ema_slope = ema_13.iloc[-1] - ema_13.iloc[-2] if len(ema_13) > 1 else 0
            
            if macd_histogram > 0 and ema_slope > 0:
                state = "green"
                signal_type = "buy"
                logger.debug("Impulse MACD: Green (Buy)")
            elif macd_histogram < 0 and ema_slope < 0:
                state = "red"
                signal_type = "sell"
                logger.debug("Impulse MACD: Red (Sell)")
            else:
                state = "blue"
                signal_type = "neutral"
            
            return {
                "state": state,
                "signal": signal_type,
                "macd_histogram": macd_histogram,
                "ema_slope": ema_slope
            }
            
        except Exception as e:
            logger.error(f"Impulse MACD calculation error: {e}")
            return {"signal": "neutral"}

    @staticmethod
    def calculate_signal_momentum(indicator_series: pd.Series, lookback: int = 10) -> Dict:
        """Calculate momentum metrics for signal strength."""
        try:
            if len(indicator_series) < lookback:
                logger.debug(f"Insufficient data for momentum: {len(indicator_series)} < {lookback}")
                return {"momentum": 0, "acceleration": 0, "trend_consistency": 0, "slope": 0}
            
            recent = indicator_series.tail(lookback)
            recent = recent.dropna()
            
            if len(recent) < 2:
                logger.debug("Too few non-null values for momentum calculation")
                return {"momentum": 0, "acceleration": 0, "trend_consistency": 0, "slope": 0}
            
            # Sanitize the array
            recent_values = TechnicalIndicators.sanitize_array(recent.values)
            
            # Calculate momentum
            momentum = (recent_values[-1] - recent_values[0]) / max(lookback, 1)
            
            # Calculate acceleration
            acceleration = 0
            if len(recent_values) >= 3:
                mid_point = len(recent_values) // 2
                first_half_momentum = (recent_values[mid_point] - recent_values[0]) / max(mid_point, 1)
                second_half_momentum = (recent_values[-1] - recent_values[mid_point]) / max(mid_point, 1)
                acceleration = second_half_momentum - first_half_momentum
                
                if np.isnan(acceleration):
                    acceleration = 0
            
            # Calculate trend consistency
            trend_consistency = 0
            slope = 0
            
            if len(recent_values) >= 3:
                try:
                    x = np.arange(len(recent_values))
                    slope, intercept, r_value, p_value, std_err = stats.linregress(x, recent_values)
                    trend_consistency = r_value ** 2
                except Exception as e:
                    logger.debug(f"Linear regression failed: {e}")
            
            return {
                "momentum": float(momentum),
                "acceleration": float(acceleration),
                "trend_consistency": float(trend_consistency),
                "slope": float(slope)
            }
            
        except Exception as e:
            logger.error(f"Momentum calculation error: {e}")
            return {"momentum": 0, "acceleration": 0, "trend_consistency": 0, "slope": 0}
    
    @staticmethod
    def predict_signal_duration(indicators: Dict, ohlcv_df: pd.DataFrame) -> Dict:
        """Predict signal duration based on momentum analysis."""
        try:
            logger.debug("Calculating signal duration prediction")
            
            predictions = {
                "estimated_minutes": 0,
                "estimated_candles": 0,
                "confidence": "low",
                "strength_trend": "neutral",
                "key_levels": {},
                "momentum_status": {}
            }
            
            # Analyze MACD momentum
            if indicators.get("macd") and "histogram_series" in indicators["macd"]:
                macd_momentum = TechnicalIndicators.calculate_signal_momentum(
                    indicators["macd"]["histogram_series"]
                )
                predictions["momentum_status"]["macd"] = macd_momentum
            
            # Analyze RSI momentum  
            if indicators.get("rsi") and "rsi_series" in indicators["rsi"]:
                rsi_momentum = TechnicalIndicators.calculate_signal_momentum(
                    indicators["rsi"]["rsi_series"]
                )
                predictions["momentum_status"]["rsi"] = rsi_momentum
            
            # Count stable Supertrend candles
            supertrend_stable_candles = 0
            if indicators.get("supertrend") and "direction_series" in indicators["supertrend"]:
                direction_series = indicators["supertrend"]["direction_series"].dropna()
                if len(direction_series) > 0:
                    current_direction = direction_series.iloc[-1]
                    
                    for i in range(len(direction_series) - 1, -1, -1):
                        if direction_series.iloc[i] == current_direction:
                            supertrend_stable_candles += 1
                        else:
                            break
            
            # Calculate average momentum with NaN checks
            total_momentum = 0
            total_consistency = 0
            indicator_count = 0
            
            for indicator_name, momentum_data in predictions["momentum_status"].items():
                if momentum_data:
                    momentum_val = momentum_data.get("momentum", 0)
                    consistency_val = momentum_data.get("trend_consistency", 0)
                    
                    if not np.isnan(momentum_val):
                        total_momentum += abs(momentum_val)
                    if not np.isnan(consistency_val):
                        total_consistency += consistency_val
                        indicator_count += 1
            
            if indicator_count > 0:
                avg_momentum = total_momentum / indicator_count
                avg_consistency = total_consistency / indicator_count
                
                if np.isnan(avg_momentum):
                    avg_momentum = 0
                if np.isnan(avg_consistency):
                    avg_consistency = 0
                
                # Estimate duration
                base_duration = 5
                
                if avg_momentum > 0.5:
                    momentum_multiplier = 3
                elif avg_momentum > 0.3:
                    momentum_multiplier = 2
                elif avg_momentum > 0.1:
                    momentum_multiplier = 1.5
                else:
                    momentum_multiplier = 1
                
                consistency_multiplier = max(0.5, min(2.5, 1 + avg_consistency))
                stability_bonus = max(0.5, min(supertrend_stable_candles / 10, 2))
                
                try:
                    product = base_duration * momentum_multiplier * consistency_multiplier * stability_bonus
                    
                    if np.isnan(product) or np.isinf(product):
                        estimated_candles = 10
                    else:
                        estimated_candles = int(product)
                        estimated_candles = max(1, min(estimated_candles, 30))
                        
                except (ValueError, OverflowError) as e:
                    logger.warning(f"Error calculating estimated candles: {e}")
                    estimated_candles = 10
                
                predictions["estimated_candles"] = estimated_candles
                predictions["estimated_minutes"] = estimated_candles
                
                # Determine confidence
                if avg_consistency > 0.7 and supertrend_stable_candles > 5:
                    predictions["confidence"] = "high"
                elif avg_consistency > 0.5 and supertrend_stable_candles > 3:
                    predictions["confidence"] = "medium"
                else:
                    predictions["confidence"] = "low"
                
                # Determine strength trend
                avg_acceleration = 0
                accel_count = 0
                for m in predictions["momentum_status"].values():
                    accel_val = m.get("acceleration", 0)
                    if not np.isnan(accel_val):
                        avg_acceleration += accel_val
                        accel_count += 1
                
                if accel_count > 0:
                    avg_acceleration = avg_acceleration / accel_count
                
                if avg_acceleration > 0.01:
                    predictions["strength_trend"] = "strengthening"
                elif avg_acceleration < -0.01:
                    predictions["strength_trend"] = "weakening"
                else:
                    predictions["strength_trend"] = "stable"
            else:
                predictions["estimated_candles"] = 10
                predictions["estimated_minutes"] = 10
            
            # Calculate key levels
            if len(ohlcv_df) >= 20:
                recent_highs = ohlcv_df['high'].tail(20)
                recent_lows = ohlcv_df['low'].tail(20)
                
                resistance = float(recent_highs.max())
                support = float(recent_lows.min())
                pivot = float((resistance + support + ohlcv_df['close'].iloc[-1]) / 3)
                
                predictions["key_levels"] = {
                    "resistance": round(resistance, 2),
                    "support": round(support, 2),
                    "pivot": round(pivot, 2)
                }
            
            logger.debug(f"Duration prediction: {predictions['estimated_minutes']} mins "
                        f"(confidence: {predictions['confidence']})")
            return predictions
            
        except Exception as e:
            logger.error(f"Duration prediction error: {e}", exc_info=True)
            return {
                "estimated_minutes": 10,
                "estimated_candles": 10,
                "confidence": "low",
                "strength_trend": "neutral",
                "key_levels": {}
            }

class SignalGenerator:
    """
    Basic signal generator for simple weighted signal strategies.
    Use this for straightforward indicator-based trading without market context.
    """
    
    def __init__(self):
        """Initialize signal generator."""
        self.accuracy_tracker = SignalAccuracyTracker()
        self.performance_tracker = SignalPerformanceTracker()
        logger.info("SignalGenerator initialized with performance tracking")
    
    def calculate_weighted_signal_with_duration(
        self, 
        indicators: Dict, 
        weights: Dict, 
        ohlcv_df: pd.DataFrame
    ) -> Dict:
        """Calculate weighted signal with duration prediction."""
        try:
            logger.debug("Calculating weighted signal with duration")
            
            # Get base signal
            signal_result = self._calculate_weighted_signal(indicators, weights)
            
            # Add duration prediction
            duration_prediction = TechnicalIndicators.predict_signal_duration(indicators, ohlcv_df)
            signal_result["duration_prediction"] = duration_prediction
            
            # Calculate real accuracy metrics
            current_price = ohlcv_df['close'].iloc[-1] if len(ohlcv_df) > 0 else 0
            
            # Record signal for tracking
            self.performance_tracker.record_signal(
                signal_result["composite_signal"],
                signal_result["action"],
                current_price,
                signal_result["confidence"],
                duration_prediction.get("estimated_minutes", 10)
            )
            
            # Update position if needed
            self.performance_tracker.update_position(current_price)
            
            # Get real accuracy metrics
            accuracy_metrics = self.performance_tracker.calculate_metrics()
            signal_result["accuracy_metrics"] = accuracy_metrics
            
            # Track signal
            self.accuracy_tracker.add_signal({
                "timestamp": pd.Timestamp.now(),
                "signal": signal_result["composite_signal"],
                "action": signal_result["action"],
                "confidence": signal_result["confidence"],
                "predicted_duration": duration_prediction.get("estimated_candles", 0),
                "price": current_price
            })
            
            logger.info(f"Signal: {signal_result['composite_signal']} "
                       f"(Accuracy: {accuracy_metrics['signal_accuracy']:.1f}%, "
                       f"Win Rate: {accuracy_metrics['win_rate']:.1f}%)")
            
            return signal_result
            
        except Exception as e:
            logger.error(f"Error calculating signal: {e}", exc_info=True)
            return self._get_default_signal_result()
    
    def _calculate_weighted_signal(self, indicators: Dict, weights: Dict) -> Dict:
        """Calculate weighted signal from indicators."""
        try:
            signal_values = {
                "strong_buy": 1.0,
                "buy": 0.75,
                "bullish": 0.5,
                "weak_buy": 0.25,
                "neutral": 0.0,
                "weak_sell": -0.25,
                "bearish": -0.5,
                "sell": -0.75,
                "strong_sell": -1.0,
                "overbought": -0.5,
                "oversold": 0.5
            }
            
            weighted_sum = 0
            contributions = {}
            active_indicators = 0
            
            for indicator_name, indicator_data in indicators.items():
                if indicator_name in weights and indicator_data:
                    signal = indicator_data.get("signal", "neutral")
                    signal_value = signal_values.get(signal, 0)
                    weight = weights.get(indicator_name, 0)
                    contribution = signal_value * weight
                    
                    weighted_sum += contribution
                    contributions[indicator_name] = {
                        "signal": signal,
                        "value": signal_value,
                        "weight": weight,
                        "contribution": round(contribution, 3)
                    }
                    
                    if signal != "neutral":
                        active_indicators += 1
            
            # Determine composite signal
            if weighted_sum >= 0.75:
                composite_signal = "STRONG_BUY"
                action = "buy"
            elif weighted_sum >= 0.5:
                composite_signal = "BUY"
                action = "buy"
            elif weighted_sum >= 0.25:
                composite_signal = "WEAK_BUY"
                action = "hold"
            elif weighted_sum <= -0.75:
                composite_signal = "STRONG_SELL"
                action = "sell"
            elif weighted_sum <= -0.5:
                composite_signal = "SELL"
                action = "sell"
            elif weighted_sum <= -0.25:
                composite_signal = "WEAK_SELL"
                action = "hold"
            else:
                composite_signal = "NEUTRAL"
                action = "hold"
            
            # Calculate confidence
            confidence = min(abs(weighted_sum) * 100, 100)
            
            result = {
                "composite_signal": composite_signal,
                "action": action,
                "weighted_score": round(weighted_sum, 3),
                "confidence": round(confidence, 1),
                "active_indicators": active_indicators,
                "contributions": contributions
            }
            
            logger.debug(f"Weighted signal: {composite_signal} (score: {weighted_sum:.3f})")
            return result
            
        except Exception as e:
            logger.error(f"Error in signal calculation: {e}")
            return self._get_default_signal_result()
    
    def _get_default_signal_result(self) -> Dict:
        """Get default signal result."""
        return {
            "composite_signal": "NEUTRAL",
            "action": "hold",
            "weighted_score": 0.0,
            "confidence": 0.0,
            "active_indicators": 0,
            "contributions": {},
            "accuracy_metrics": {
                "signal_accuracy": 50.0,
                "confidence_sustain": 50.0,
                "win_rate": 50.0,
                "avg_duration_accuracy": 50.0
            },
            "duration_prediction": {
                "estimated_minutes": 0,
                "estimated_candles": 0,
                "confidence": "low",
                "strength_trend": "neutral"
            }
        }

class SignalAccuracyTracker:
    """Track and calculate signal accuracy metrics."""
    
    def __init__(self, window_size: int = 100):
        """Initialize accuracy tracker."""
        self.window_size = window_size
        self.signal_history = deque(maxlen=window_size)
        self.prediction_history = deque(maxlen=window_size)
        logger.debug(f"SignalAccuracyTracker initialized with window_size={window_size}")
    
    def add_signal(self, signal_data: Dict):
        """Add a signal to history for tracking."""
        self.signal_history.append(signal_data)
        logger.debug(f"Signal added: {signal_data.get('signal', 'unknown')}")
    
    def add_prediction(self, prediction_data: Dict):
        """Add a prediction for validation."""
        self.prediction_history.append(prediction_data)
        logger.debug("Prediction added to history")
    
    def calculate_accuracy_metrics(self) -> Dict:
        """Calculate accuracy metrics based on historical signals."""
        try:
            if len(self.signal_history) < 10:
                logger.debug("Insufficient history for accuracy calculation")
                return {
                    "signal_accuracy": 50.0,
                    "confidence_sustain": 50.0,
                    "win_rate": 50.0,
                    "avg_duration_accuracy": 50.0
                }
            
            # Calculate win rate
            correct_signals = 0
            total_signals = 0
            
            for i in range(1, len(self.signal_history)):
                prev_signal = self.signal_history[i-1]
                curr_signal = self.signal_history[i]
                
                if prev_signal.get('action') == 'buy':
                    if curr_signal.get('price', 0) > prev_signal.get('price', 0):
                        correct_signals += 1
                    total_signals += 1
                elif prev_signal.get('action') == 'sell':
                    if curr_signal.get('price', 0) < prev_signal.get('price', 0):
                        correct_signals += 1
                    total_signals += 1
            
            win_rate = (correct_signals / max(total_signals, 1)) * 100
            
            # Calculate confidence correlation
            confidence_correlation = self._calculate_confidence_correlation()
            
            # Calculate duration accuracy
            duration_accuracy = self._calculate_duration_accuracy()
            
            # Overall signal accuracy
            signal_accuracy = (win_rate * 0.5 + confidence_correlation * 0.3 + duration_accuracy * 0.2)
            
            metrics = {
                "signal_accuracy": round(signal_accuracy, 1),
                "confidence_sustain": round(confidence_correlation, 1),
                "win_rate": round(win_rate, 1),
                "avg_duration_accuracy": round(duration_accuracy, 1)
            }
            
            logger.debug(f"Accuracy metrics calculated: {metrics}")
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating accuracy metrics: {e}")
            return {
                "signal_accuracy": 50.0,
                "confidence_sustain": 50.0,
                "win_rate": 50.0,
                "avg_duration_accuracy": 50.0
            }
    
    def _calculate_confidence_correlation(self) -> float:
        """Calculate confidence correlation."""
        try:
            if len(self.signal_history) < 5:
                return 50.0
            
            confidences = []
            outcomes = []
            
            for i in range(1, len(self.signal_history)):
                prev_signal = self.signal_history[i-1]
                curr_signal = self.signal_history[i]
                
                confidence = prev_signal.get('confidence', 50)
                confidences.append(confidence)
                
                if prev_signal.get('action') == 'buy':
                    outcome = 1 if curr_signal.get('price', 0) > prev_signal.get('price', 0) else 0
                elif prev_signal.get('action') == 'sell':
                    outcome = 1 if curr_signal.get('price', 0) < prev_signal.get('price', 0) else 0
                else:
                    outcome = 0.5
                
                outcomes.append(outcome)
            
            if len(confidences) > 1 and len(outcomes) > 1:
                correlation = np.corrcoef(confidences, outcomes)[0, 1]
                if not np.isnan(correlation):
                    return abs(correlation) * 100
            
            return 50.0
            
        except Exception as e:
            logger.error(f"Error calculating confidence correlation: {e}")
            return 50.0
    
    def _calculate_duration_accuracy(self) -> float:
        """Calculate duration prediction accuracy."""
        try:
            if len(self.prediction_history) < 5:
                return 50.0
            
            accurate_predictions = 0
            total_predictions = 0
            
            for prediction in self.prediction_history:
                predicted_duration = prediction.get('predicted_duration', 0)
                actual_duration = prediction.get('actual_duration', 0)
                
                if predicted_duration > 0 and actual_duration > 0:
                    error_pct = abs(predicted_duration - actual_duration) / predicted_duration
                    if error_pct < 0.3:
                        accurate_predictions += 1
                    total_predictions += 1
            
            if total_predictions > 0:
                return (accurate_predictions / total_predictions) * 100
            
            return 50.0
            
        except Exception as e:
            logger.error(f"Error calculating duration accuracy: {e}")
            return 50.0
