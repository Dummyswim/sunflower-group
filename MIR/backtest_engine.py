"""
Backtesting engine for strategy validation.
"""
import pandas as pd
import numpy as np
import logging
import json
from typing import Dict, List, Any
from datetime import datetime
from technical_indicators import TechnicalIndicators, SignalGenerator

logger = logging.getLogger(__name__)

class BacktestEngine:
    """Backtesting engine for signal validation."""
    
    def __init__(self, initial_capital: float = 1000000):
        """Initialize backtest engine."""
        self.initial_capital = initial_capital
        self.signal_generator = SignalGenerator()
        logger.info(f"BacktestEngine initialized with capital: {initial_capital}")
    
    def run_backtest(self, ohlcv_data: pd.DataFrame, weights: Dict, min_data_points: int) -> Dict:
        """Run backtest on historical data."""
        try:
            logger.info(f"Running backtest on {len(ohlcv_data)} candles")
            
            trades = []
            signals = []
            capital = self.initial_capital
            position = None
            
            for i in range(min_data_points, len(ohlcv_data)):
                # Get data slice
                data_slice = ohlcv_data.iloc[:i+1]
                prices = pd.Series(data_slice['close'].values, index=data_slice.index)
                volumes = pd.Series(data_slice['volume'].values, index=data_slice.index)
                highs = pd.Series(data_slice['high'].values, index=data_slice.index)
                lows = pd.Series(data_slice['low'].values, index=data_slice.index)
                
                # Calculate indicators
                indicators = {
                    "macd": TechnicalIndicators.calculate_macd(prices),
                    "rsi": TechnicalIndicators.calculate_rsi(prices),
                    "vwap": TechnicalIndicators.calculate_vwap(prices, volumes),
                    "keltner": TechnicalIndicators.calculate_keltner_channels(prices, highs, lows),
                    "supertrend": TechnicalIndicators.calculate_supertrend(prices, highs, lows),
                    "impulse": TechnicalIndicators.calculate_impulse_macd(prices)
                }
                
                # Generate signal
                signal_result = self.signal_generator.calculate_weighted_signal_with_duration(
                    indicators, weights, data_slice
                )
                
                signals.append({
                    "timestamp": data_slice.index[-1],
                    "price": prices.iloc[-1],
                    "signal": signal_result
                })
                
                # Execute trades
                if signal_result['action'] == 'buy' and position is None:
                    position = {
                        "entry_price": prices.iloc[-1],
                        "entry_time": data_slice.index[-1],
                        "quantity": int(capital / prices.iloc[-1])
                    }
                elif signal_result['action'] == 'sell' and position is not None:
                    exit_price = prices.iloc[-1]
                    pnl = (exit_price - position['entry_price']) * position['quantity']
                    capital += pnl
                    
                    trades.append({
                        "entry_time": position['entry_time'],
                        "exit_time": data_slice.index[-1],
                        "entry_price": position['entry_price'],
                        "exit_price": exit_price,
                        "quantity": position['quantity'],
                        "pnl": pnl,
                        "return_pct": (exit_price / position['entry_price'] - 1) * 100
                    })
                    position = None
            
            # Calculate metrics
            total_trades = len(trades)
            winning_trades = sum(1 for t in trades if t['pnl'] > 0)
            losing_trades = sum(1 for t in trades if t['pnl'] < 0)
            
            if total_trades > 0:
                win_rate = (winning_trades / total_trades) * 100
                avg_win = np.mean([t['pnl'] for t in trades if t['pnl'] > 0]) if winning_trades > 0 else 0
                avg_loss = np.mean([t['pnl'] for t in trades if t['pnl'] < 0]) if losing_trades > 0 else 0
                total_pnl = sum(t['pnl'] for t in trades)
                roi = ((capital - self.initial_capital) / self.initial_capital) * 100
            else:
                win_rate = avg_win = avg_loss = total_pnl = roi = 0
            
            # Test duration predictions
            duration_accuracy = self._test_duration_predictions(signals)
            
            results = {
                "initial_capital": self.initial_capital,
                "final_capital": capital,
                "total_trades": total_trades,
                "winning_trades": winning_trades,
                "losing_trades": losing_trades,
                "win_rate": win_rate,
                "avg_win": avg_win,
                "avg_loss": avg_loss,
                "total_pnl": total_pnl,
                "roi": roi,
                "trades": trades,
                "duration_accuracy": duration_accuracy
            }
            
            logger.info(f"Backtest complete: ROI={roi:.2f}%, Win Rate={win_rate:.2f}%")
            return results
            
        except Exception as e:
            logger.error(f"Backtest error: {e}", exc_info=True)
            return {}
    
    def _test_duration_predictions(self, signals: List[Dict]) -> Dict:
        """Test accuracy of duration predictions."""
        try:
            predictions_analyzed = 0
            correct_predictions = 0
            total_error = 0
            high_confidence_correct = 0
            high_confidence_total = 0
            
            for i in range(len(signals) - 1):
                current_signal = signals[i]['signal']
                duration_pred = current_signal.get('duration_prediction', {})
                
                if duration_pred.get('estimated_candles', 0) > 0:
                    predicted_duration = duration_pred['estimated_candles']
                    
                    # Check actual duration
                    actual_duration = 0
                    for j in range(i + 1, min(i + predicted_duration + 10, len(signals))):
                        if signals[j]['signal']['composite_signal'] == current_signal['composite_signal']:
                            actual_duration += 1
                        else:
                            break
                    
                    predictions_analyzed += 1
                    error = abs(predicted_duration - actual_duration)
                    total_error += error
                    
                    if error <= 3:  # Within 3 candles
                        correct_predictions += 1
                    
                    if duration_pred.get('confidence') == 'high':
                        high_confidence_total += 1
                        if error <= 3:
                            high_confidence_correct += 1
            
            if predictions_analyzed > 0:
                accuracy = (correct_predictions / predictions_analyzed) * 100
                avg_error = total_error / predictions_analyzed
                high_conf_accuracy = (high_confidence_correct / high_confidence_total * 100) if high_confidence_total > 0 else 0
            else:
                accuracy = avg_error = high_conf_accuracy = 0
            
            return {
                "predictions_analyzed": predictions_analyzed,
                "accuracy": accuracy,
                "average_error": avg_error,
                "high_confidence_accuracy": high_conf_accuracy
            }
            
        except Exception as e:
            logger.error(f"Duration test error: {e}")
            return {}
    
    def generate_report(self, results: Dict, output_file: str) -> Dict:
        """Generate backtest report."""
        try:
            # Create summary
            summary = {
                "performance": {
                    "initial_capital": results.get("initial_capital", 0),
                    "final_capital": results.get("final_capital", 0),
                    "total_pnl": results.get("total_pnl", 0),
                    "roi": results.get("roi", 0)
                },
                "trading_stats": {
                    "total_trades": results.get("total_trades", 0),
                    "winning_trades": results.get("winning_trades", 0),
                    "losing_trades": results.get("losing_trades", 0),
                    "win_rate": results.get("win_rate", 0)
                },
                "risk_metrics": {
                    "avg_win": results.get("avg_win", 0),
                    "avg_loss": results.get("avg_loss", 0),
                    "profit_factor": abs(results.get("avg_win", 0) / results.get("avg_loss", 1)) if results.get("avg_loss", 0) != 0 else 0
                },
                "signal_accuracy": results.get("duration_accuracy", {})
            }
            
            # Save to file
            with open(output_file, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            
            # Print summary
            print("\n" + "="*60)
            print("BACKTEST RESULTS SUMMARY")
            print("="*60)
            print(f"Initial Capital: Rs {summary['performance']['initial_capital']:,.2f}")
            print(f"Final Capital: Rs {summary['performance']['final_capital']:,.2f}")
            print(f"Total PnL: Rs {summary['performance']['total_pnl']:,.2f}")
            print(f"ROI: {summary['performance']['roi']:.2f}%")
            print("-"*60)
            print(f"Total Trades: {summary['trading_stats']['total_trades']}")
            print(f"Win Rate: {summary['trading_stats']['win_rate']:.2f}%")
            print(f"Profit Factor: {summary['risk_metrics']['profit_factor']:.2f}")
            print("="*60)
            
            logger.info(f"Report saved to {output_file}")
            return summary
            
        except Exception as e:
            logger.error(f"Report generation error: {e}")
            return {}
