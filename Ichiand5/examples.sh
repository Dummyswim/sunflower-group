#!/bin/bash
# Usage examples for historical data fetcher

echo "DhanHQ Historical Data Fetcher - Usage Examples"
echo "==============================================="

# Example 1: Get daily data for backtesting
echo "1. Fetching daily data for last year:"
python historical_data.py \
    --timeframe daily \
    --from 2023-01-01 \
    --to 2024-01-01 \
    --output data/nifty_daily_2023.csv \
    --verbose

# Example 2: Get minute data for last 5 days
echo "2. Fetching 1-minute data for last 5 days:"
python historical_data.py \
    --timeframe 1min \
    --from 2024-01-25 \
    --to 2024-01-30 \
    --output data/nifty_1min_recent.csv

# Example 3: Run backtest on daily data
echo "3. Running backtest on 6 months of daily data:"
python historical_data.py \
    --timeframe daily \
    --from 2023-07-01 \
    --to 2024-01-01 \
    --backtest \
    --output reports/backtest_6months.csv \
    --telegram

# Example 4: Get 5-minute data for intraday analysis
echo "4. Fetching 5-minute data for intraday analysis:"
python historical_data.py \
    --timeframe 5min \
    --from 2024-01-29 \
    --to 2024-01-30 \
    --output data/nifty_5min_intraday.csv \
    --verbose

# Example 5: Get hourly data
echo "5. Fetching hourly data:"
python historical_data.py \
    --timeframe 60min \
    --from 2024-01-15 \
    --to 2024-01-30 \
    --output data/nifty_hourly.csv

# Example 6: Full backtest with report
echo "6. Complete backtest with all features:"
python historical_data.py \
    --timeframe daily \
    --from 2022-01-01 \
    --to 2024-01-01 \
    --backtest \
    --telegram \
    --output reports/full_backtest_2years.csv \
    --verbose
