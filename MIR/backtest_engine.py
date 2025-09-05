"""
Enhanced backtesting engine with JSON/CSV support and real performance tracking.
"""
import pandas as pd
import numpy as np
import logging
import json
from typing import Dict, List, Any
from datetime import datetime
from pathlib import Path
from technical_indicators import TechnicalIndicators, SignalGenerator, EnhancedSignalGenerator
import config

logger = logging.getLogger(__name__)

class BacktestEngine:
    """Enhanced backtesting engine for comprehensive strategy validation."""
    
    def __init__(self, initial_capital: float = 1000000):
        """Initialize backtest engine."""
        self.initial_capital = initial_capital
        self.signal_generator = SignalGenerator()
        self.enhanced_generator = EnhancedSignalGenerator()
        self.use_enhanced = True
        logger.info(f"BacktestEngine initialized with capital: {initial_capital}")
    
    def load_data(self, filepath: str) -> pd.DataFrame:
        """Load data from JSON or CSV file."""
        try:
            file_ext = Path(filepath).suffix.lower()
            
            if file_ext == '.json':
                return self._load_json_data(filepath)
            elif file_ext == '.csv':
                return self._load_csv_data(filepath)
            else:
                raise ValueError(f"Unsupported file type: {file_ext}")
                
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def _load_json_data(self, filepath: str) -> pd.DataFrame:
        """Load and preprocess JSON data."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        df = pd.DataFrame(data)
        
        # Handle missing close prices - use (H+L+O)/3
        if 'close' not in df.columns:
            logger.warning("Close prices missing - using typical price")
            df['close'] = (df['high'] + df['low'] + df['open']) / 3
        
        # Handle missing volume
        if 'volume' not in df.columns:
            logger.warning("Volume missing - using estimated values")
            # Estimate based on price volatility
            price_range = df['high'] - df['low']
            avg_range = price_range.mean()
            df['volume'] = 50000 * (1 + (price_range / avg_range))
            df['volume'] = df['volume'].astype(int)
        
        # Ensure numeric types
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df = df.dropna()
        
        # Create datetime index
        df.index = pd.date_range(start='2024-01-01 09:15:00', 
                                periods=len(df), freq='5min')
        
        logger.info(f"Loaded {len(df)} candles from JSON")
        return df
    
    def _load_csv_data(self, filepath: str) -> pd.DataFrame:
        """Load and preprocess CSV data."""
        df = pd.read_csv(filepath)
        
        # Clean column names
        df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
        
        # Parse date
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        
        # Rename columns to standard format
        column_mapping = {
            'open_': 'open',
            'high_': 'high', 
            'low_': 'low',
            'close_': 'close',
            'shares_traded': 'volume',
            'turnover_(₹_cr)': 'turnover'
        }
        df.rename(columns=column_mapping, inplace=True)
        
        # Ensure numeric types
        for col in ['open', 'high', 'low', 'close']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        if 'volume' in df.columns:
            df['volume'] = pd.to_numeric(df['volume'], errors='coerce').fillna(0).astype(int)
        else:
            df['volume'] = 1000000  # Default for daily data
        
        df = df.dropna(subset=['open', 'high', 'low', 'close'])
        
        logger.info(f"Loaded {len(df)} candles from CSV")
        return df
    
    def run_comprehensive_backtest(self, data: pd.DataFrame) -> Dict:
        """Run comprehensive backtest with detailed metrics."""
        
        # VALIDATE DATA FIRST
        data_quality = self.validate_data_completeness(data)
        if not data_quality['usable']:
            logger.error(f"Data quality too poor: {data_quality['quality_score']}%")
            logger.error(f"Issues: {', '.join(data_quality['issues'])}")
            return {
                'error': 'Data quality insufficient',
                'quality_score': data_quality['quality_score'],
                'issues': data_quality['issues']
            }
        
        logger.info(f"Data quality check passed: {data_quality['quality_score']}%")
        logger.info(f"Running backtest on {len(data)} candles")
        
        trades = []
        signals = []
        positions = []
        capital = self.initial_capital
        position = None
        
        # Performance tracking
        equity_curve = [self.initial_capital]
        max_drawdown = 0
        peak_capital = self.initial_capital
        consecutive_wins = 0
        consecutive_losses = 0
        max_consecutive_wins = 0
        max_consecutive_losses = 0
        
        # Skip warmup period
        min_data = config.MIN_DATA_POINTS
        
        for i in range(min_data, len(data)):
            current_data = data.iloc[:i+1]
            current_price = current_data['close'].iloc[-1]
            current_time = current_data.index[-1]
            
            # Calculate indicators
            indicators = self._calculate_indicators(current_data)
            
            # Generate signal
            if self.use_enhanced:
                signal_result = self.enhanced_generator.generate_trading_signal(
                    indicators, current_data, config.INDICATOR_WEIGHTS
                )
            else:
                signal_result = self.signal_generator.calculate_weighted_signal_with_duration(
                    indicators, config.INDICATOR_WEIGHTS, current_data
                )
            
            # Store signal
            signals.append({
                'timestamp': current_time,
                'price': current_price,
                'signal': signal_result['composite_signal'],
                'action': signal_result['action'],
                'confidence': signal_result.get('confidence', 0),
                'score': signal_result.get('weighted_score', 0)
            })
            
            # Trading logic
            if position is None:
                # Check entry conditions
                if signal_result['action'] == 'buy' and signal_result['confidence'] > 50:
                    position = {
                        'type': 'long',
                        'entry_price': current_price,
                        'entry_time': current_time,
                        'quantity': int(capital * 0.95 / current_price),
                        'stop_loss': signal_result.get('stop_loss', current_price * 0.98),
                        'take_profit': signal_result.get('take_profit', current_price * 1.02)
                    }
                    positions.append(position.copy())
                    logger.debug(f"Entered LONG at {current_price:.2f}")
                    
            else:
                # Check exit conditions
                exit_trade = False
                exit_reason = ""
                
                if position['type'] == 'long':
                    if current_price <= position['stop_loss']:
                        exit_trade = True
                        exit_reason = "Stop Loss"
                    elif current_price >= position['take_profit']:
                        exit_trade = True
                        exit_reason = "Take Profit"
                    elif signal_result['action'] == 'sell' and signal_result['confidence'] > 60:
                        exit_trade = True
                        exit_reason = "Signal Reversal"
                
                if exit_trade:
                    # Calculate P&L
                    pnl = (current_price - position['entry_price']) * position['quantity']
                    returns = (current_price / position['entry_price'] - 1) * 100
                    capital += pnl
                    
                    # Track consecutive wins/losses
                    if pnl > 0:
                        consecutive_wins += 1
                        consecutive_losses = 0
                        max_consecutive_wins = max(max_consecutive_wins, consecutive_wins)
                    else:
                        consecutive_losses += 1
                        consecutive_wins = 0
                        max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
                    
                    trade = {
                        'entry_time': position['entry_time'],
                        'exit_time': current_time,
                        'type': position['type'],
                        'entry_price': position['entry_price'],
                        'exit_price': current_price,
                        'quantity': position['quantity'],
                        'pnl': pnl,
                        'returns': returns,
                        'exit_reason': exit_reason
                    }
                    trades.append(trade)
                    position = None
                    
                    logger.debug(f"Exited at {current_price:.2f} - P&L: {pnl:.2f}")
            
            # Update equity curve
            current_equity = capital
            if position:
                unrealized_pnl = (current_price - position['entry_price']) * position['quantity']
                current_equity = capital + unrealized_pnl
            
            equity_curve.append(current_equity)
            
            # Track drawdown
            if current_equity > peak_capital:
                peak_capital = current_equity
            drawdown = (peak_capital - current_equity) / peak_capital * 100
            max_drawdown = max(max_drawdown, drawdown)
        
        # Calculate final metrics
        return self._calculate_comprehensive_metrics(
            trades, signals, capital, self.initial_capital, 
            max_drawdown, equity_curve, positions,
            max_consecutive_wins, max_consecutive_losses
        )
    
    def _calculate_indicators(self, df: pd.DataFrame) -> Dict:
        """Calculate all technical indicators."""
        prices = pd.Series(df['close'].values, index=df.index)
        volumes = pd.Series(df['volume'].values, index=df.index)
        highs = pd.Series(df['high'].values, index=df.index)
        lows = pd.Series(df['low'].values, index=df.index)
        
        return {
            "macd": TechnicalIndicators.calculate_macd(prices),
            "rsi": TechnicalIndicators.calculate_rsi(prices),
            "vwap": TechnicalIndicators.calculate_vwap(prices, volumes),
            "keltner": TechnicalIndicators.calculate_keltner_channels(prices, highs, lows),
            "supertrend": TechnicalIndicators.calculate_supertrend(prices, highs, lows),
            "impulse": TechnicalIndicators.calculate_impulse_macd(prices)
        }
    
    def _calculate_comprehensive_metrics(self, trades, signals, final_capital, 
                                        initial_capital, max_drawdown, equity_curve,
                                        positions, max_wins, max_losses) -> Dict:
        """Calculate comprehensive performance metrics."""
        
        if not trades:
            return {
                'initial_capital': initial_capital,
                'final_capital': final_capital,
                'total_trades': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'roi': 0
            }
        
        # Basic metrics
        winning_trades = [t for t in trades if t['pnl'] > 0]
        losing_trades = [t for t in trades if t['pnl'] <= 0]
        
        total_trades = len(trades)
        num_winners = len(winning_trades)
        num_losers = len(losing_trades)
        
        win_rate = (num_winners / total_trades * 100) if total_trades > 0 else 0
        
        avg_win = np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0
        avg_loss = abs(np.mean([t['pnl'] for t in losing_trades])) if losing_trades else 0
        
        gross_profit = sum([t['pnl'] for t in winning_trades]) if winning_trades else 0
        gross_loss = abs(sum([t['pnl'] for t in losing_trades])) if losing_trades else 0
        
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else 0
        
        total_pnl = final_capital - initial_capital
        roi = (total_pnl / initial_capital * 100)
        
        # Calculate Sharpe Ratio
        returns = [t['returns'] for t in trades]
        if len(returns) > 1:
            sharpe_ratio = np.mean(returns) / (np.std(returns) + 0.0001) * np.sqrt(252)
        else:
            sharpe_ratio = 0
        
        # Signal analysis
        signal_distribution = {}
        for s in signals:
            sig_type = s['signal']
            signal_distribution[sig_type] = signal_distribution.get(sig_type, 0) + 1
        
        return {
            'initial_capital': initial_capital,
            'final_capital': round(final_capital, 2),
            'total_pnl': round(total_pnl, 2),
            'roi': round(roi, 2),
            'total_trades': total_trades,
            'winning_trades': num_winners,
            'losing_trades': num_losers,
            'win_rate': round(win_rate, 2),
            'avg_win': round(avg_win, 2),
            'avg_loss': round(avg_loss, 2),
            'profit_factor': round(profit_factor, 2),
            'max_drawdown': round(max_drawdown, 2),
            'sharpe_ratio': round(sharpe_ratio, 2),
            'max_consecutive_wins': max_wins,
            'max_consecutive_losses': max_losses,
            'total_signals': len(signals),
            'signal_distribution': signal_distribution
        }
    
    def generate_report(self, results: Dict, output_file: str = "backtest_report.json"):
        """Generate detailed backtest report."""
        try:
            # Save to JSON
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            # Print summary
            print("\n" + "="*70)
            print(" "*20 + "BACKTEST RESULTS SUMMARY")
            print("="*70)
            print(f"Initial Capital:     Rs {results['initial_capital']:,.2f}")
            print(f"Final Capital:       Rs {results['final_capital']:,.2f}")
            print(f"Total P&L:          Rs {results['total_pnl']:,.2f}")
            print(f"ROI:                {results['roi']:.2f}%")
            print("-"*70)
            print(f"Total Trades:       {results['total_trades']}")
            print(f"Winning Trades:     {results['winning_trades']}")
            print(f"Losing Trades:      {results['losing_trades']}")
            print(f"Win Rate:           {results['win_rate']:.2f}%")
            print(f"Profit Factor:      {results['profit_factor']:.2f}")
            print("-"*70)
            print(f"Average Win:        Rs {results['avg_win']:,.2f}")
            print(f"Average Loss:       Rs {results['avg_loss']:,.2f}")
            print(f"Max Drawdown:       {results['max_drawdown']:.2f}%")
            print(f"Sharpe Ratio:       {results['sharpe_ratio']:.2f}")
            print("-"*70)
            print("Signal Distribution:")
            for signal, count in results.get('signal_distribution', {}).items():
                print(f"  {signal:15} {count:5}")
            print("="*70)
            
            logger.info(f"Report saved to {output_file}")
            return results
            
        except Exception as e:
            logger.error(f"Report generation error: {e}")
            return {}
    def validate_data_completeness(self, df: pd.DataFrame) -> Dict:
        """Check data quality before backtesting."""
        missing_close = 'close' not in df.columns
        missing_volume = 'volume' not in df.columns or df['volume'].sum() == 0
        
        quality_score = 100
        issues = []
        
        if missing_close:
            quality_score -= 30
            issues.append("Missing close prices")
        if missing_volume:
            quality_score -= 20
            issues.append("Missing volume data")
        
        # Check for data gaps
        if len(df) > 0:
            time_diff = df.index.to_series().diff()
            expected_freq = pd.Timedelta(minutes=5)
            gaps = time_diff[time_diff > expected_freq * 2]
            if len(gaps) > 0:
                quality_score -= 10
                issues.append(f"{len(gaps)} time gaps detected")
        
        logger.info(f"Data quality score: {quality_score}% - Issues: {', '.join(issues) if issues else 'None'}")
        
        return {
            'quality_score': quality_score,
            'missing_close': missing_close,
            'missing_volume': missing_volume,
            'issues': issues,
            'usable': quality_score >= 50
        }


# Run simulation
if __name__ == "__main__":
    engine = BacktestEngine()
    
    # Test with JSON data
    json_data = engine.load_data("intra.json")
    json_results = engine.run_comprehensive_backtest(json_data)
    engine.generate_report(json_results, "json_backtest.json")
    
    # Test with CSV data
    csv_data = engine.load_data("NIFTY 50-02-06-2025-to-29-08-2025.csv")
    csv_results = engine.run_comprehensive_backtest(csv_data)
    engine.generate_report(csv_results, "csv_backtest.json")
