"""
Main execution file for the enhanced trading system.
"""
import pandas as pd
from alert_system import AlertSystem
from backtesting_engine import BacktestingEngine
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def main():
    """Main execution function."""
    # Load historical data
    df = pd.read_csv('NIFTY 50-02-06-2025-to-29-08-2025.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    
    # Initialize backtesting
    backtest = BacktestingEngine(initial_capital=1000000)
    
    # Run backtest
    results = backtest.run_backtest(df)
    
    # Print results
    print("\n=== Backtest Results ===")
    for key, value in results.items():
        print(f"{key}: {value}")
    
    # Initialize live system (if needed)
    # alert_system = AlertSystem(mode="live")
    # alert_system.run()

if __name__ == "__main__":
    main()
