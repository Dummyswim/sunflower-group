"""
Backtesting engine for strategy validation.
"""
import pandas as pd
import numpy as np
from typing import Dict, List
import logging
from alert_system import AlertSystem

logger = logging.getLogger(__name__)

class BacktestingEngine:
    """Engine for backtesting trading strategies."""
    
    def __init__(self, initial_capital: float = 1000000):
        """Initialize backtesting engine."""
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.positions = []
        self.trades = []
        self.alert_system = AlertSystem(mode="replay")
        
    def run_backtest(self, data: pd.DataFrame) -> Dict:
        """
        Run backtest on historical data.
        
        Args:
            data: DataFrame with OHLC data
            
        Returns:
            Dictionary with backtest results
        """
        results = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_return': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'win_rate': 0.0
        }
        
        # Process each candle
        for i in range(100, len(data)):  # Start after warm-up period
            # Get current window
            window = data.iloc[i-100:i+1]
            
            # Perform analysis
            analysis = self.alert_system._perform_analysis(window)
            
            if analysis:
                patterns, prediction, indicators, metrics = analysis
                
                # Check for trade signal
                if self._should_trade(prediction, metrics):
                    self._execute_trade(
                        data.iloc[i],
                        prediction,
                        metrics
                    )
        
        # Calculate results
        results = self._calculate_results()
        return results
    
    def _should_trade(self, prediction: Dict, metrics: Dict) -> bool:
        """Determine if trade should be executed."""
        return (
            prediction['confidence'] > 0.6 and
            metrics['risk_reward_ratio'] > 1.5 and
            prediction['direction'] != 'neutral'
        )
    
    def _execute_trade(self, candle: pd.Series, prediction: Dict, metrics: Dict):
        """Execute a trade based on signals."""
        trade = {
            'timestamp': candle.name,
            'entry_price': candle['close'],
            'direction': prediction['direction'],
            'stop_loss': metrics['stop_loss'],
            'take_profit': metrics['take_profit'],
            'position_size': metrics.get('position_size', 1)
        }
        self.trades.append(trade)
    
    def _calculate_results(self) -> Dict:
        """Calculate backtest performance metrics."""
        if not self.trades:
            return {'error': 'No trades executed'}
        
        # Calculate trade outcomes
        returns = []
        for trade in self.trades:
            # Simplified P&L calculation
            if trade['direction'] == 'bullish':
                pnl = (trade['take_profit'] - trade['entry_price']) * trade['position_size']
            else:
                pnl = (trade['entry_price'] - trade['take_profit']) * trade['position_size']
            returns.append(pnl)
        
        returns = np.array(returns)
        winning_trades = sum(returns > 0)
        
        return {
            'total_trades': len(self.trades),
            'winning_trades': winning_trades,
            'losing_trades': len(self.trades) - winning_trades,
            'total_return': sum(returns),
            'average_return': np.mean(returns),
            'win_rate': winning_trades / len(self.trades) if self.trades else 0,
            'sharpe_ratio': np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
        }
