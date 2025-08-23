#!/usr/bin/env python3
"""
NSE Historical Data Backtesting Runner
Integrates NSE data fetching with the pattern recognition alert system.
"""
import argparse
import logging
import sys
import os
from datetime import datetime, timedelta
import pandas as pd
import json

# Import our modules
from logging_setup import setup_logging
from nse_data_fetcher import NSEDataFetcher
from alert_system import AlertSystem
from replay_engine import ReplayEngine

logger = logging.getLogger(__name__)

class BacktestRunner:
    """
    Orchestrates backtesting with NSE historical data.
    """
    
    def __init__(self, 
                 symbol: str,
                 start_date: datetime,
                 end_date: datetime,
                 interval: str = "1d",
                 data_source: str = "auto"):
        """
        Initialize backtest runner.
        
        Args:
            symbol: NSE symbol to backtest
            start_date: Backtest start date
            end_date: Backtest end date  
            interval: Data interval (1m, 5m, 15m, 1h, 1d)
            data_source: Preferred data source (auto, nsepy, yfinance, jugaad)
        """
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.interval = interval
        self.data_source = data_source
        
        # Initialize components
        self.fetcher = NSEDataFetcher()
        self.alert_system = None
        self.results = {
            "symbol": symbol,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "interval": interval,
            "patterns_detected": 0,
            "alerts_triggered": 0,
            "predictions": [],
            "accuracy": 0.0
        }
        
    def fetch_data(self) -> pd.DataFrame:
        """Fetch historical data for backtesting."""
        logger.info(f"Fetching {self.symbol} data from {self.start_date} to {self.end_date}")
        
        # Determine if it's an index or equity
        if self.symbol in ["NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY"]:
            df = self.fetcher.fetch_index_history(
                self.symbol, self.start_date, self.end_date, self.interval
            )
        else:
            df = self.fetcher.fetch_equity_history(
                self.symbol, self.start_date, self.end_date, self.interval
            )
        
        if df is None or df.empty:
            logger.error(f"Failed to fetch data for {self.symbol}")
            # Try to generate synthetic data for demonstration
            logger.info("Using synthetic data for demonstration")
            df = self.fetcher._generate_synthetic_data(self.start_date, self.end_date)
        
        logger.info(f"Fetched {len(df)} data points")
        return df
    
    def prepare_csv(self, df: pd.DataFrame, output_file: str) -> str:
        """
        Prepare data in CSV format for replay.
        
        Args:
            df: DataFrame with OHLCV data
            output_file: Path to save CSV file
            
        Returns:
            Path to saved CSV file
        """
        # Ensure correct column order and format
        csv_df = df[['time', 'open', 'high', 'low', 'close', 'volume']].copy()
        
        # Convert time to string format if needed
        if pd.api.types.is_datetime64_any_dtype(csv_df['time']):
            csv_df['time'] = csv_df['time'].astype(str)
        
        # Save to CSV
        csv_df.to_csv(output_file, index=False)
        logger.info(f"Saved data to {output_file}")
        
        return output_file
    
    def run_backtest(self, csv_file: str):
        """
        Run backtest using the alert system.
        
        Args:
            csv_file: Path to CSV file with historical data
        """
        logger.info("Starting backtest...")
        
        # Initialize alert system in replay mode
        self.alert_system = AlertSystem(mode="replay", replay_csv=csv_file)
        
        # Run the replay
        self.alert_system.run_replay(speed=0)  # Speed=0 for instant processing
        
        # Collect results
        stats = self.alert_system.get_statistics()
        self.results.update({
            "patterns_detected": stats.get("total_patterns_detected", 0),
            "alerts_triggered": stats.get("total_alerts_sent", 0),
            "accuracy": stats.get("prediction_accuracy", 0.0),
            "total_predictions": stats.get("total_predictions", 0),
            "candles_processed": stats.get("candles_processed", 0)
        })
        
        # Extract pattern performance
        if "pattern_performance" in stats:
            self.results["pattern_performance"] = stats["pattern_performance"]
        
        logger.info("Backtest completed")
    
    def generate_report(self) -> Dict:
        """Generate comprehensive backtest report."""
        report = self.results.copy()
        
        # Add performance metrics
        if self.alert_system:
            # Calculate additional metrics
            predictions = self.alert_system.prediction_history
            if predictions:
                # Win rate
                wins = sum(1 for p in predictions if p["correct"])
                report["win_rate"] = wins / len(predictions) if predictions else 0
                
                # Confidence analysis
                high_conf = [p for p in predictions if p["confidence"] > 0.7]
                if high_conf:
                    high_conf_accuracy = sum(1 for p in high_conf if p["correct"]) / len(high_conf)
                    report["high_confidence_accuracy"] = high_conf_accuracy
                
                # Direction breakdown
                bullish = [p for p in predictions if p["predicted"] == "bullish"]
                bearish = [p for p in predictions if p["predicted"] == "bearish"]
                
                if bullish:
                    report["bullish_accuracy"] = sum(1 for p in bullish if p["correct"]) / len(bullish)
                if bearish:
                    report["bearish_accuracy"] = sum(1 for p in bearish if p["correct"]) / len(bearish)
        
        return report
    
    def save_report(self, filepath: str):
        """Save backtest report to file."""
        report = self.generate_report()
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Report saved to {filepath}")
    
    def print_summary(self):
        """Print backtest summary to console."""
        report = self.generate_report()
        
        print("\n" + "="*60)
        print(f"BACKTEST RESULTS - {self.symbol}")
        print("="*60)
        print(f"Period: {self.start_date.date()} to {self.end_date.date()}")
        print(f"Interval: {self.interval}")
        print(f"Candles Processed: {report.get('candles_processed', 0)}")
        print(f"Patterns Detected: {report.get('patterns_detected', 0)}")
        print(f"Alerts Triggered: {report.get('alerts_triggered', 0)}")
        print(f"Total Predictions: {report.get('total_predictions', 0)}")
        print(f"Overall Accuracy: {report.get('accuracy', 0):.1%}")
        
        if "high_confidence_accuracy" in report:
            print(f"High Confidence Accuracy: {report['high_confidence_accuracy']:.1%}")
        
        if "bullish_accuracy" in report:
            print(f"Bullish Predictions Accuracy: {report['bullish_accuracy']:.1%}")
        
        if "bearish_accuracy" in report:
            print(f"Bearish Predictions Accuracy: {report['bearish_accuracy']:.1%}")
        
        # Pattern performance
        if "pattern_performance" in report:
            print("\nTop Performing Patterns:")
            patterns = report["pattern_performance"]
            sorted_patterns = sorted(
                patterns.items(),
                key=lambda x: x[1]["hit_rate"] if x[1]["samples"] > 5 else 0,
                reverse=True
            )[:5]
            
            for pattern_name, perf in sorted_patterns:
                if perf["samples"] > 5:
                    print(f"  {pattern_name}: {perf['hit_rate']:.1%} ({perf['samples']} samples)")
        
        print("="*60 + "\n")

def main():
    parser = argparse.ArgumentParser(
        description="Backtest pattern recognition system with NSE historical data"
    )
    
    # Required arguments
    parser.add_argument(
        "symbol",
        help="NSE symbol to backtest (e.g., NIFTY, RELIANCE, TCS)"
    )
    
    # Optional arguments
    parser.add_argument(
        "--start-date",
        type=str,
        help="Start date (YYYY-MM-DD). Default: 30 days ago"
    )
    
    parser.add_argument(
        "--end-date",
        type=str,
        help="End date (YYYY-MM-DD). Default: today"
    )
    
    parser.add_argument(
        "--interval",
        choices=["1m", "5m", "15m", "30m", "1h", "1d"],
        default="1d",
        help="Data interval. Default: 1d"
    )
    
    parser.add_argument(
        "--output-dir",
        default="backtest_results",
        help="Directory to save results. Default: backtest_results"
    )
    
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Don't use cached data"
    )
    
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(
        logfile=f"backtest_{args.symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
        console_level=getattr(logging, args.log_level)
    )
    
    # Parse dates
    if args.end_date:
        end_date = datetime.strptime(args.end_date, "%Y-%m-%d")
    else:
        end_date = datetime.now()
    
    if args.start_date:
        start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
    else:
        # Default to 30 days for daily, 7 days for intraday
        if args.interval == "1d":
            start_date = end_date - timedelta(days=30)
        else:
            start_date = end_date - timedelta(days=7)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        # Initialize runner
        runner = BacktestRunner(
            symbol=args.symbol.upper(),
            start_date=start_date,
            end_date=end_date,
            interval=args.interval
        )
        
        # Fetch data
        df = runner.fetch_data()
        
        if df.empty:
            logger.error("No data available for backtesting")
            sys.exit(1)
        
        # Prepare CSV for replay
        csv_file = os.path.join(
            args.output_dir,
            f"{args.symbol}_{args.interval}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.csv"
        )
        runner.prepare_csv(df, csv_file)
        
        # Run backtest
        runner.run_backtest(csv_file)
        
        # Generate and save report
        report_file = os.path.join(
            args.output_dir,
            f"report_{args.symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        runner.save_report(report_file)
        
        # Print summary
        runner.print_summary()
        
        logger.info(f"Backtest complete. Results saved to {args.output_dir}")
        
    except Exception as e:
        logger.error(f"Backtest failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
