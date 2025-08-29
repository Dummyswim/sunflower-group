#!/usr/bin/env python3
"""
Enhanced main script with historical data support and backtesting.
"""
import argparse
import sys
import time
import signal
import logging
from datetime import datetime, timedelta

import config
from logging_setup import setup_logging
from telegram_bot import TelegramBot
from dhan_ws_client import DhanWebSocketClient
from historical_data import DhanHistoricalData
from backtest_engine import BacktestEngine

logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Nifty50 Trading Alert System")
    
    parser.add_argument("--mode", choices=["live", "backtest", "fetch"],
                       default="live",
                       help="Operation mode: live trading, backtest, or fetch data")
    
    parser.add_argument("--timeframe", choices=["minute", "daily"],
                       default="minute",
                       help="Timeframe for historical data")
    
    parser.add_argument("--from-date", type=str,
                       help="Start date (YYYY-MM-DD)")
    
    parser.add_argument("--to-date", type=str,
                       help="End date (YYYY-MM-DD)")
    
    parser.add_argument("--security-id", type=str, default="13",
                       help="Security ID (default: 13 for Nifty50)")
    
    parser.add_argument("--exchange", type=str, default="IDX_I",
                       help="Exchange segment (default: IDX_I)")
    
    parser.add_argument("--initial-capital", type=float, default=1000000,
                       help="Initial capital for backtesting")
    
    parser.add_argument("--output", type=str, default="backtest_report.json",
                       help="Output file for backtest results")
    
    return parser.parse_args()

def signal_handler(signum, frame):
    """Handle system signals."""
    logger.info(f"Received signal {signum}")
    sys.exit(0)

def run_live_trading():
    """Run live trading with WebSocket connection."""
    try:
        # Initialize Telegram bot
        telegram_bot = TelegramBot(
            config.TELEGRAM_BOT_TOKEN_B64,
            config.TELEGRAM_CHAT_ID
        )
        
        # Test Telegram connection
        if not telegram_bot.send_message("🚀 Nifty50 Alert System Started - Live Mode"):
            logger.warning("Failed to send startup message to Telegram")
        
        # Initialize Dhan WebSocket client
        dhan_client = DhanWebSocketClient(
            config.DHAN_ACCESS_TOKEN_B64,
            config.DHAN_CLIENT_ID_B64,
            telegram_bot
        )
        
        # Connect and run
        dhan_client.connect()
        
        logger.info("Live trading system running - Press Ctrl+C to stop")
        
        # Keep running
        while True:
            time.sleep(60)
            if not dhan_client.connected:
                logger.warning("WebSocket disconnected, attempting reconnection")
                dhan_client.connect()
                
    except KeyboardInterrupt:
        logger.info("Shutting down live trading")
        if dhan_client:
            dhan_client.disconnect()
        if telegram_bot:
            telegram_bot.send_message("🛑 Nifty50 Alert System Stopped")
    except Exception as e:
        logger.error(f"Live trading error: {e}")
        sys.exit(1)

def run_backtest(args):
    """Run backtesting on historical data."""
    try:
        if not args.from_date or not args.to_date:
            print("Error: --from-date and --to-date required for backtesting")
            sys.exit(1)
        
        print(f"\nFetching historical data from {args.from_date} to {args.to_date}...")
        
        # Initialize data fetcher
        data_fetcher = DhanHistoricalData(
            config.DHAN_ACCESS_TOKEN_B64,
            config.DHAN_CLIENT_ID_B64
        )
        
        # Fetch historical data
        if args.timeframe == "minute":
            ohlcv_data = data_fetcher.fetch_data_in_chunks(
                args.from_date, args.to_date, timeframe="minute"
            )
        else:
            ohlcv_data = data_fetcher.get_daily_data(
                args.from_date, args.to_date,
                args.security_id, args.exchange
            )
        
        if ohlcv_data is None or ohlcv_data.empty:
            print("Error: No data retrieved")
            sys.exit(1)
        
        print(f"Retrieved {len(ohlcv_data)} candles")
        print(f"Date range: {ohlcv_data.index[0]} to {ohlcv_data.index[-1]}")
        
        # Initialize backtest engine
        backtest_engine = BacktestEngine(initial_capital=args.initial_capital)
        
        print("\nRunning backtest...")
        
        # Run backtest
        results = backtest_engine.run_backtest(
            ohlcv_data,
            config.INDICATOR_WEIGHTS,
            config.MIN_DATA_POINTS
        )
        
        # Generate report
        report = backtest_engine.generate_report(results, args.output)
        
        # Test signal duration predictions
        if results.get("duration_accuracy"):
            accuracy = results["duration_accuracy"]
            print("\n" + "="*60)
            print("SIGNAL DURATION PREDICTION ANALYSIS")
            print("="*60)
            print(f"Predictions Analyzed: {accuracy.get('predictions_analyzed', 0)}")
            print(f"Overall Accuracy: {accuracy.get('accuracy', 0):.2f}%")
            print(f"Average Error: {accuracy.get('average_error', 0):.1f} candles")
            print(f"High Confidence Accuracy: {accuracy.get('high_confidence_accuracy', 0):.2f}%")
            print("="*60)
        
    except Exception as e:
        logger.error(f"Backtest error: {e}")
        sys.exit(1)

def fetch_and_display_data(args):
    """Fetch and display historical data."""
    try:
        if not args.from_date or not args.to_date:
            print("Error: --from-date and --to-date required for data fetching")
            sys.exit(1)
        
        # Initialize data fetcher
        data_fetcher = DhanHistoricalData(
            config.DHAN_ACCESS_TOKEN_B64,
            config.DHAN_CLIENT_ID_B64
        )
        
        print(f"\nFetching {args.timeframe} data from {args.from_date} to {args.to_date}...")
        
        # Fetch data
        if args.timeframe == "minute":
            data = data_fetcher.get_intraday_data(
                args.from_date, args.to_date,
                args.security_id, args.exchange
            )
        else:
            data = data_fetcher.get_daily_data(
                args.from_date, args.to_date,
                args.security_id, args.exchange
            )
        
        if data is not None and not data.empty:
            print(f"\nRetrieved {len(data)} candles")
            print("\nFirst 5 candles:")
            print(data.head())
            print("\nLast 5 candles:")
            print(data.tail())
            print("\nData Summary:")
            print(data.describe())
            
            # Save to CSV
            filename = f"data_{args.timeframe}_{args.from_date}_{args.to_date}.csv"
            data.to_csv(filename)
            print(f"\nData saved to {filename}")
        else:
            print("No data retrieved")
            
    except Exception as e:
        logger.error(f"Data fetch error: {e}")
        sys.exit(1)

def main():
    """Main entry point."""
    # Parse arguments
    args = parse_arguments()
    
    # Setup logging
    setup_logging(config.LOG_FILE)
    
    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Print banner
    print("=" * 60)
    print("    NIFTY50 TRADING SYSTEM")
    print(f"    Mode: {args.mode.upper()}")
    print(f"    Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # Validate config
    try:
        config.validate_config()
    except ValueError as e:
        print(f"Configuration error: {e}")
        sys.exit(1)
    
    # Execute based on mode
    if args.mode == "live":
        run_live_trading()
    elif args.mode == "backtest":
        run_backtest(args)
    elif args.mode == "fetch":
        fetch_and_display_data(args)

if __name__ == "__main__":
    main()