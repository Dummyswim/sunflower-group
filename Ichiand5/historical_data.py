"""
Historical data fetcher for backtesting with DhanHQ API.
Supports both daily and intraday timeframes.
"""
import base64
import logging
import json
import argparse
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
import requests
from pathlib import Path

from technical_indicators import TechnicalIndicators, SignalGenerator
from telegram_bot import TelegramBot
import config

logger = logging.getLogger(__name__)

class DhanHistoricalData:
    """Fetch and process historical data from DhanHQ API."""
    
    BASE_URL = "https://api.dhan.co"
    
    # Timeframe mappings
    TIMEFRAME_MAP = {
        "1MIN": "1",
        "5MIN": "5", 
        "15MIN": "15",
        "25MIN": "25",
        "60MIN": "60",
        "1HOUR": "60",
        "DAILY": "DAILY",
        "1DAY": "DAILY",
        "DAY": "DAILY",
        "WEEKLY": "WEEKLY",
        "MONTHLY": "MONTHLY"
    }
    
    def __init__(self, access_token_b64: str, client_id_b64: str):
        """Initialize with Dhan credentials."""
        try:
            self.access_token = base64.b64decode(access_token_b64).decode("utf-8")
            self.client_id = base64.b64decode(client_id_b64).decode("utf-8")
        except Exception as e:
            logger.error(f"Failed to decode credentials: {e}")
            raise ValueError("Invalid base64 encoded credentials")
        
        self.headers = {
            "access-token": self.access_token,
            "Content-Type": "application/json"
        }
        
        logger.info("DhanHistoricalData initialized")
    
    def fetch_historical_data(self, 
                            security_id: str,
                            exchange_segment: str,
                            instrument_type: str,
                            from_date: str,
                            to_date: str,
                            timeframe: str = "DAILY",
                            expiry_code: int = 0) -> pd.DataFrame:
        """
        Fetch historical data from DhanHQ API.
        
        Args:
            security_id: Security ID of the instrument
            exchange_segment: Exchange segment (NSE_EQ, NSE_FNO, etc.)
            instrument_type: Type of instrument (EQUITY, FUTIDX, OPTIDX, etc.)
            from_date: Start date (YYYY-MM-DD)
            to_date: End date (YYYY-MM-DD)
            timeframe: Data timeframe (DAILY, 1MIN, 5MIN, etc.)
            expiry_code: Expiry code for derivatives (0 for equity)
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            # Map timeframe
            api_timeframe = self.TIMEFRAME_MAP.get(timeframe.upper(), timeframe)
            
            # Determine endpoint based on timeframe
            if api_timeframe in ["1", "5", "15", "25", "60"]:
                # Intraday data endpoint
                endpoint = f"{self.BASE_URL}/v2/charts/intraday"
                
                # Check date range (max 5 days for intraday)
                from_dt = datetime.strptime(from_date, "%Y-%m-%d")
                to_dt = datetime.strptime(to_date, "%Y-%m-%d")
                if (to_dt - from_dt).days > 5:
                    logger.warning("Intraday data limited to 5 days. Adjusting date range.")
                    from_date = (to_dt - timedelta(days=4)).strftime("%Y-%m-%d")
                
                payload = {
                    "securityId": security_id,
                    "exchangeSegment": exchange_segment,
                    "instrument": instrument_type,
                    "expiryCode": expiry_code,
                    "fromDate": from_date,
                    "toDate": to_date,
                    "interval": api_timeframe
                }
            else:
                # Daily/Weekly/Monthly data endpoint
                endpoint = f"{self.BASE_URL}/v2/charts/historical"
                
                payload = {
                    "securityId": security_id,
                    "exchangeSegment": exchange_segment,
                    "instrument": instrument_type,
                    "expiryCode": expiry_code,
                    "fromDate": from_date,
                    "toDate": to_date
                }
            
            logger.info(f"Fetching {timeframe} data from {from_date} to {to_date}")
            
            # Make API request
            response = requests.post(
                endpoint,
                headers=self.headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                
                if "data" in data and data["data"]:
                    # Parse data based on response format
                    df = self._parse_historical_response(data["data"], api_timeframe)
                    logger.info(f"Fetched {len(df)} data points")
                    return df
                else:
                    logger.warning("No data received from API")
                    return pd.DataFrame()
            else:
                logger.error(f"API request failed: {response.status_code} - {response.text}")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error fetching historical data: {e}")
            return pd.DataFrame()
    
    def _parse_historical_response(self, data: List, timeframe: str) -> pd.DataFrame:
        """Parse historical data response into DataFrame."""
        try:
            records = []
            
            for item in data:
                # Handle different response formats
                if isinstance(item, dict):
                    record = {
                        "timestamp": pd.to_datetime(item.get("timestamp", item.get("time"))),
                        "open": float(item.get("open", 0)),
                        "high": float(item.get("high", 0)),
                        "low": float(item.get("low", 0)),
                        "close": float(item.get("close", 0)),
                        "volume": int(item.get("volume", 0))
                    }
                elif isinstance(item, list) and len(item) >= 6:
                    # Array format: [timestamp, open, high, low, close, volume]
                    record = {
                        "timestamp": pd.to_datetime(item[0], unit='s' if isinstance(item[0], int) else None),
                        "open": float(item[1]),
                        "high": float(item[2]),
                        "low": float(item[3]),
                        "close": float(item[4]),
                        "volume": int(item[5])
                    }
                else:
                    continue
                
                records.append(record)
            
            if records:
                df = pd.DataFrame(records)
                df.set_index("timestamp", inplace=True)
                df.sort_index(inplace=True)
                return df
            
            return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Error parsing historical response: {e}")
            return pd.DataFrame()
    
    def fetch_nifty_historical(self, from_date: str, to_date: str, 
                              timeframe: str = "DAILY") -> pd.DataFrame:
        """
        Convenience method to fetch Nifty50 historical data.
        
        Args:
            from_date: Start date (YYYY-MM-DD)
            to_date: End date (YYYY-MM-DD)
            timeframe: Data timeframe
            
        Returns:
            DataFrame with Nifty50 OHLCV data
        """
        return self.fetch_historical_data(
            security_id=str(config.NIFTY_SECURITY_ID),
            exchange_segment="IDX_I",  # Index segment
            instrument_type="INDEX",
            from_date=from_date,
            to_date=to_date,
            timeframe=timeframe
        )


class BacktestEngine:
    """Backtesting engine for historical data analysis."""
    
    def __init__(self, telegram_bot: Optional[TelegramBot] = None):
        """Initialize backtesting engine."""
        self.signal_generator = SignalGenerator()
        self.telegram_bot = telegram_bot
        self.results = []
        
    def run_backtest(self, df: pd.DataFrame, 
                     start_capital: float = 1000000) -> Dict:
        """
        Run backtest on historical data with all 6 indicators.
        
        Args:
            df: DataFrame with OHLCV data
            start_capital: Starting capital for backtest
            
        Returns:
            Dictionary with backtest results
        """
        try:
            if df.empty or len(df) < config.MIN_DATA_POINTS:
                logger.error("Insufficient data for backtesting")
                return {}
            
            logger.info(f"Starting backtest with {len(df)} data points")
            
            # Initialize tracking variables
            capital = start_capital
            position = 0
            trades = []
            signals = []
            equity_curve = []
            
            # Process data in rolling windows
            for i in range(config.MIN_DATA_POINTS, len(df)):
                # Get window of data
                window_df = df.iloc[i-config.MIN_DATA_POINTS:i].copy()
                current_price = df['close'].iloc[i]
                current_time = df.index[i]
                
                # Calculate all 6 indicators
                indicators = self._calculate_all_indicators(window_df)
                
                # Generate signal
                signal_result = self.signal_generator.calculate_weighted_signal(
                    window_df, indicators, config.INDICATOR_WEIGHTS
                )
                
                signals.append({
                    'timestamp': current_time,
                    'signal': signal_result['composite_signal'],
                    'score': signal_result['weighted_score'],
                    'confidence': signal_result['confidence']
                })
                
                # Execute trades based on signals
                trade = self._execute_trade_logic(
                    signal_result,
                    current_price,
                    current_time,
                    capital,
                    position
                )
                
                if trade:
                    trades.append(trade)
                    capital = trade['capital']
                    position = trade['position']
                
                # Track equity
                equity = capital + (position * current_price)
                equity_curve.append({
                    'timestamp': current_time,
                    'equity': equity,
                    'capital': capital,
                    'position_value': position * current_price
                })
                
                # Log progress every 100 bars
                if i % 100 == 0:
                    logger.debug(f"Processed {i}/{len(df)} bars")
            
            # Calculate performance metrics
            results = self._calculate_performance_metrics(
                trades, 
                equity_curve, 
                start_capital,
                df
            )
            
            # Add signal statistics
            results['signals'] = self._analyze_signals(signals)
            results['trades'] = trades
            results['equity_curve'] = pd.DataFrame(equity_curve)
            
            logger.info(f"Backtest complete: {len(trades)} trades executed")
            
            return results
            
        except Exception as e:
            logger.error(f"Backtest error: {e}", exc_info=True)
            return {}
    
    def _calculate_all_indicators(self, df: pd.DataFrame) -> Dict:
        """Calculate all 6 technical indicators."""
        try:
            indicators = {}
            
            # 1. Ichimoku Cloud
            indicators['ichimoku'] = TechnicalIndicators.calculate_ichimoku(
                df, config.ICHIMOKU_PARAMS
            )
            
            # 2. Stochastic Oscillator
            indicators['stochastic'] = TechnicalIndicators.calculate_stochastic(
                df, config.STOCHASTIC_PARAMS
            )
            
            # 3. On-Balance Volume
            indicators['obv'] = TechnicalIndicators.calculate_obv(
                df, config.OBV_PARAMS
            )
            
            # 4. Bollinger Bands
            indicators['bollinger'] = TechnicalIndicators.calculate_bollinger_bands(
                df, config.BOLLINGER_PARAMS
            )
            
            # 5. ADX
            indicators['adx'] = TechnicalIndicators.calculate_adx(
                df, config.ADX_PARAMS
            )
            
            # 6. ATR
            indicators['atr'] = TechnicalIndicators.calculate_atr(
                df, config.ATR_PARAMS
            )
            
            return indicators
            
        except Exception as e:
            logger.error(f"Indicator calculation error: {e}")
            return {}
    
    def _execute_trade_logic(self, signal_result: Dict, price: float, 
                            timestamp: datetime, capital: float, 
                            position: float) -> Optional[Dict]:
        """Execute trade based on signal."""
        try:
            # Only trade on strong signals
            if abs(signal_result['weighted_score']) < config.MIN_SIGNAL_STRENGTH:
                return None
            
            if signal_result['confidence'] < config.MIN_CONFIDENCE:
                return None
            
            trade = None
            
            # Strong buy signal
            if signal_result['composite_signal'] in ['STRONG BUY', 'BUY']:
                if position <= 0:  # Not long or short
                    # Calculate position size (risk 2% per trade)
                    risk_amount = capital * 0.02
                    shares = int(risk_amount / price)
                    
                    if shares > 0:
                        cost = shares * price
                        if cost <= capital:
                            trade = {
                                'timestamp': timestamp,
                                'type': 'BUY',
                                'price': price,
                                'shares': shares,
                                'capital': capital - cost,
                                'position': position + shares,
                                'signal_score': signal_result['weighted_score']
                            }
            
            # Strong sell signal
            elif signal_result['composite_signal'] in ['STRONG SELL', 'SELL']:
                if position > 0:  # Long position exists
                    # Close long position
                    proceeds = position * price
                    trade = {
                        'timestamp': timestamp,
                        'type': 'SELL',
                        'price': price,
                        'shares': position,
                        'capital': capital + proceeds,
                        'position': 0,
                        'signal_score': signal_result['weighted_score']
                    }
            
            return trade
            
        except Exception as e:
            logger.error(f"Trade execution error: {e}")
            return None
    
    def _calculate_performance_metrics(self, trades: List[Dict], 
                                      equity_curve: List[Dict],
                                      start_capital: float,
                                      price_data: pd.DataFrame) -> Dict:
        """Calculate comprehensive performance metrics."""
        try:
            if not equity_curve:
                return {}
            
            equity_df = pd.DataFrame(equity_curve)
            final_equity = equity_df['equity'].iloc[-1]
            
            # Basic metrics
            total_return = (final_equity - start_capital) / start_capital
            
            # Calculate daily returns
            equity_df['returns'] = equity_df['equity'].pct_change()
            
            # Sharpe ratio (assuming 252 trading days)
            if len(equity_df) > 1:
                sharpe = (equity_df['returns'].mean() / equity_df['returns'].std()) * np.sqrt(252)
            else:
                sharpe = 0
            
            # Maximum drawdown
            equity_df['cummax'] = equity_df['equity'].cummax()
            equity_df['drawdown'] = (equity_df['equity'] - equity_df['cummax']) / equity_df['cummax']
            max_drawdown = equity_df['drawdown'].min()
            
            # Win rate
            winning_trades = [t for t in trades if t['type'] == 'SELL']
            if winning_trades:
                profitable = sum(1 for t in winning_trades if t.get('pnl', 0) > 0)
                win_rate = profitable / len(winning_trades)
            else:
                win_rate = 0
            
            metrics = {
                'start_capital': start_capital,
                'final_equity': final_equity,
                'total_return': total_return * 100,
                'sharpe_ratio': sharpe,
                'max_drawdown': max_drawdown * 100,
                'total_trades': len(trades),
                'win_rate': win_rate * 100,
                'avg_daily_return': equity_df['returns'].mean() * 100,
                'volatility': equity_df['returns'].std() * 100,
                'best_day': equity_df['returns'].max() * 100,
                'worst_day': equity_df['returns'].min() * 100
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Metrics calculation error: {e}")
            return {}
    
    def _analyze_signals(self, signals: List[Dict]) -> Dict:
        """Analyze signal distribution and effectiveness."""
        try:
            signal_df = pd.DataFrame(signals)
            
            # Count signals by type
            signal_counts = signal_df['signal'].value_counts().to_dict()
            
            # Average confidence by signal type
            avg_confidence = signal_df.groupby('signal')['confidence'].mean().to_dict()
            
            # Signal strength distribution
            strong_signals = sum(1 for s in signals if abs(s['score']) >= 0.7)
            moderate_signals = sum(1 for s in signals if 0.4 <= abs(s['score']) < 0.7)
            weak_signals = sum(1 for s in signals if abs(s['score']) < 0.4)
            
            return {
                'signal_counts': signal_counts,
                'avg_confidence': avg_confidence,
                'strong_signals': strong_signals,
                'moderate_signals': moderate_signals,
                'weak_signals': weak_signals,
                'total_signals': len(signals)
            }
            
        except Exception as e:
            logger.error(f"Signal analysis error: {e}")
            return {}
    
    def generate_backtest_report(self, results: Dict, output_file: str = None):
        """Generate comprehensive backtest report."""
        try:
            if not results:
                logger.error("No results to report")
                return
            
            report = []
            report.append("=" * 60)
            report.append("BACKTEST REPORT - 6 INDICATOR STRATEGY")
            report.append("=" * 60)
            
            # Performance metrics
            report.append("\n📊 PERFORMANCE METRICS:")
            report.append(f"Initial Capital: ₹{results['start_capital']:,.2f}")
            report.append(f"Final Equity: ₹{results['final_equity']:,.2f}")
            report.append(f"Total Return: {results['total_return']:.2f}%")
            report.append(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
            report.append(f"Max Drawdown: {results['max_drawdown']:.2f}%")
            report.append(f"Volatility: {results['volatility']:.2f}%")
            
            # Trading statistics
            report.append("\n📈 TRADING STATISTICS:")
            report.append(f"Total Trades: {results['total_trades']}")
            report.append(f"Win Rate: {results['win_rate']:.2f}%")
            report.append(f"Avg Daily Return: {results['avg_daily_return']:.2f}%")
            report.append(f"Best Day: {results['best_day']:.2f}%")
            report.append(f"Worst Day: {results['worst_day']:.2f}%")
            
            # Signal analysis
            if 'signals' in results:
                report.append("\n🎯 SIGNAL ANALYSIS:")
                signals = results['signals']
                report.append(f"Total Signals: {signals['total_signals']}")
                report.append(f"Strong Signals: {signals['strong_signals']}")
                report.append(f"Moderate Signals: {signals['moderate_signals']}")
                report.append(f"Weak Signals: {signals['weak_signals']}")
                
                if 'signal_counts' in signals:
                    report.append("\nSignal Distribution:")
                    for signal_type, count in signals['signal_counts'].items():
                        report.append(f"  {signal_type}: {count}")
            
            # Indicator weights used
            report.append("\n⚙️ INDICATOR CONFIGURATION:")
            for indicator, weight in config.INDICATOR_WEIGHTS.items():
                report.append(f"  {indicator.upper()}: {weight:.1%}")
            
            report.append("\n" + "=" * 60)
            
            # Join report
            report_text = "\n".join(report)
            
            # Print to console
            print(report_text)
            
            # Save to file if specified
            if output_file:
                Path(output_file).parent.mkdir(parents=True, exist_ok=True)
                with open(output_file, 'w') as f:
                    f.write(report_text)
                logger.info(f"Report saved to {output_file}")
            
            # Send to Telegram if available
            if self.telegram_bot:
                # Truncate for Telegram (max 4096 chars)
                telegram_msg = report_text[:4000] if len(report_text) > 4000 else report_text
                self.telegram_bot.send_message(f"<pre>{telegram_msg}</pre>")
            
            # Save detailed results to CSV
            if 'equity_curve' in results:
                csv_file = output_file.replace('.txt', '_equity.csv') if output_file else 'backtest_equity.csv'
                results['equity_curve'].to_csv(csv_file)
                logger.info(f"Equity curve saved to {csv_file}")
            
            if 'trades' in results and results['trades']:
                trades_file = output_file.replace('.txt', '_trades.csv') if output_file else 'backtest_trades.csv'
                pd.DataFrame(results['trades']).to_csv(trades_file)
                logger.info(f"Trades saved to {trades_file}")
                
        except Exception as e:
            logger.error(f"Report generation error: {e}")


def parse_arguments():
    """Parse command-line arguments for historical data retrieval."""
    parser = argparse.ArgumentParser(
        description="Fetch historical data from DhanHQ for backtesting",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Get daily data for last month
  python historical_data.py --timeframe daily --from 2024-01-01 --to 2024-01-31
  
  # Get minute data for last 5 days
  python historical_data.py --timeframe 1min --from 2024-01-25 --to 2024-01-30
  
  # Run backtest on historical data
  python historical_data.py --timeframe daily --from 2023-01-01 --to 2024-01-01 --backtest
  
  # Save data to CSV
  python historical_data.py --timeframe 5min --from 2024-01-25 --to 2024-01-30 --output data.csv
        """
    )
    
    parser.add_argument(
        '--timeframe', '-t',
        type=str,
        required=True,
        choices=['1min', '5min', '15min', '25min', '60min', '1hour', 
                'daily', '1day', 'day', 'weekly', 'monthly'],
        help='Timeframe for historical data'
    )
    
    parser.add_argument(
        '--from', '-f',
        dest='from_date',
        type=str,
        required=True,
        help='Start date (YYYY-MM-DD)'
    )
    
    parser.add_argument(
        '--to', '-t2',
        dest='to_date',
        type=str,
        required=True,
        help='End date (YYYY-MM-DD)'
    )
    
    parser.add_argument(
        '--symbol', '-s',
        type=str,
        default='NIFTY',
        help='Symbol to fetch (default: NIFTY)'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Output CSV file path'
    )
    
    parser.add_argument(
        '--backtest', '-b',
        action='store_true',
        help='Run backtest on fetched data'
    )
    
    parser.add_argument(
        '--telegram', '-tg',
        action='store_true',
        help='Send results to Telegram'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    return parser.parse_args()


def validate_dates(from_date: str, to_date: str) -> Tuple[bool, str]:
    """Validate date inputs."""
    try:
        from_dt = datetime.strptime(from_date, "%Y-%m-%d")
        to_dt = datetime.strptime(to_date, "%Y-%m-%d")
        
        if from_dt > to_dt:
            return False, "From date must be before to date"
        
        if to_dt > datetime.now():
            return False, "To date cannot be in the future"
        
        return True, "Valid"
        
    except ValueError as e:
        return False, f"Invalid date format: {e}"


def main():
    """Main entry point for historical data fetcher."""
    # Parse arguments
    args = parse_arguments()
    
    # Setup logging
    from logging_setup import setup_logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging("logs/historical_data.log", log_level)
    
    logger.info("=" * 60)
    logger.info("DHAN HISTORICAL DATA FETCHER")
    logger.info("=" * 60)
    
    # Validate dates
    valid, msg = validate_dates(args.from_date, args.to_date)
    if not valid:
        logger.error(f"Date validation failed: {msg}")
        sys.exit(1)
    
    # Validate environment
    if not config.DHAN_ACCESS_TOKEN_B64 or not config.DHAN_CLIENT_ID_B64:
        logger.error("Missing Dhan credentials in environment")
        sys.exit(1)
    
    try:
        # Initialize components
        telegram_bot = None
        if args.telegram and config.TELEGRAM_BOT_TOKEN_B64:
            telegram_bot = TelegramBot(
                config.TELEGRAM_BOT_TOKEN_B64,
                config.TELEGRAM_CHAT_ID
            )
            telegram_bot.send_message(
                f"📊 Fetching {args.timeframe} data\n"
                f"📅 From: {args.from_date}\n"
                f"📅 To: {args.to_date}"
            )
        
        # Initialize data fetcher
        fetcher = DhanHistoricalData(
            config.DHAN_ACCESS_TOKEN_B64,
            config.DHAN_CLIENT_ID_B64
        )
        
        # Fetch data
        logger.info(f"Fetching {args.timeframe} data from {args.from_date} to {args.to_date}")
        
        if args.symbol.upper() == 'NIFTY':
            df = fetcher.fetch_nifty_historical(
                args.from_date,
                args.to_date,
                args.timeframe.upper()
            )
        else:
            # For other symbols, you'd need to map to security_id
            logger.error(f"Symbol {args.symbol} not yet implemented")
            sys.exit(1)
        
        if df.empty:
            logger.error("No data received")
            sys.exit(1)
        
        logger.info(f"Fetched {len(df)} data points")
        
        # Save to CSV if requested
        if args.output:
            df.to_csv(args.output)
            logger.info(f"Data saved to {args.output}")
        
        # Display sample data
        print("\nSample Data (first 5 rows):")
        print(df.head())
        print(f"\nTotal rows: {len(df)}")
        print(f"Date range: {df.index[0]} to {df.index[-1]}")
        
        # Run backtest if requested
        if args.backtest:
            logger.info("Starting backtest...")
            engine = BacktestEngine(telegram_bot)
            results = engine.run_backtest(df)
            
            if results:
                report_file = args.output.replace('.csv', '_report.txt') if args.output else 'backtest_report.txt'
                engine.generate_backtest_report(results, report_file)
            else:
                logger.error("Backtest failed")
        
        # Send summary to Telegram
        if telegram_bot:
            telegram_bot.send_message(
                f"✅ <b>Data Fetch Complete</b>\n"
                f"📊 Timeframe: {args.timeframe}\n"
                f"📈 Data Points: {len(df)}\n"
                f"📅 Range: {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}\n"
                f"💾 Output: {args.output or 'Not saved'}"
            )
        
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        if telegram_bot:
            telegram_bot.send_message(f"❌ Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
