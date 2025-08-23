"""
NSE Historical Data Fetcher for Backtesting
Fetches historical data from NSE using multiple sources.
"""
import logging
import os
import time
from datetime import datetime, timedelta
from typing import Optional, Dict, List
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# Try importing data sources
try:
    from nsepy import get_history
    HAS_NSEPY = True
except ImportError:
    HAS_NSEPY = False
    logger.warning("nsepy not installed. Install with: pip install nsepy")

try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False
    logger.warning("yfinance not installed. Install with: pip install yfinance")

try:
    from jugaad_data import nse
    HAS_JUGAAD = True
except ImportError:
    HAS_JUGAAD = False
    logger.warning("jugaad-data not installed. Install with: pip install jugaad-data")

class NSEDataFetcher:
    """
    Fetches historical data from NSE using multiple sources.
    Falls back to alternative sources if primary fails.
    """
    
    # NSE symbol to Yahoo Finance symbol mapping
    YAHOO_SYMBOL_MAP = {
        "NIFTY": "^NSEI",
        "BANKNIFTY": "^NSEBANK",
        "RELIANCE": "RELIANCE.NS",
        "TCS": "TCS.NS",
        "INFY": "INFOSYS.NS",
        "HDFCBANK": "HDFCBANK.NS",
        "ICICIBANK": "ICICIBANK.NS",
        "SBIN": "SBIN.NS",
        "HDFC": "HDFC.NS",
        "ITC": "ITC.NS",
        "BHARTIARTL": "BHARTIARTL.NS",
        "KOTAKBANK": "KOTAKBANK.NS",
        "LT": "LT.NS",
        "AXISBANK": "AXISBANK.NS",
        "ASIANPAINT": "ASIANPAINT.NS",
    }
    
    def __init__(self, cache_dir: str = "nse_data_cache"):
        """
        Initialize NSE data fetcher.
        
        Args:
            cache_dir: Directory to cache downloaded data
        """
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
        # Determine available data sources
        self.sources = []
        if HAS_NSEPY:
            self.sources.append("nsepy")
        if HAS_YFINANCE:
            self.sources.append("yfinance")
        if HAS_JUGAAD:
            self.sources.append("jugaad")
            
        if not self.sources:
            logger.warning("No data sources available. Please install nsepy, yfinance, or jugaad-data")
        
        logger.info(f"NSEDataFetcher initialized with sources: {self.sources}")
    
    def fetch_equity_history(self, 
                           symbol: str,
                           start_date: datetime,
                           end_date: datetime,
                           interval: str = "1m") -> Optional[pd.DataFrame]:
        """
        Fetch historical data for an equity symbol.
        
        Args:
            symbol: NSE symbol (e.g., "RELIANCE", "TCS")
            start_date: Start date for historical data
            end_date: End date for historical data
            interval: Time interval (1m, 5m, 15m, 1h, 1d)
            
        Returns:
            DataFrame with columns: time, open, high, low, close, volume
        """
        # Check cache first
        cache_file = self._get_cache_filename(symbol, start_date, end_date, interval)
        if os.path.exists(cache_file):
            logger.info(f"Loading cached data from {cache_file}")
            return pd.read_csv(cache_file, parse_dates=['time'])
        
        # Try each data source
        df = None
        for source in self.sources:
            try:
                if source == "nsepy":
                    df = self._fetch_nsepy(symbol, start_date, end_date, interval)
                elif source == "yfinance":
                    df = self._fetch_yfinance(symbol, start_date, end_date, interval)
                elif source == "jugaad":
                    df = self._fetch_jugaad(symbol, start_date, end_date, interval)
                
                if df is not None and not df.empty:
                    logger.info(f"Successfully fetched data using {source}")
                    break
                    
            except Exception as e:
                logger.error(f"Failed to fetch data using {source}: {e}")
                continue
        
        if df is not None and not df.empty:
            # Standardize column names
            df = self._standardize_dataframe(df)
            
            # Cache the data
            df.to_csv(cache_file, index=False)
            logger.info(f"Cached data to {cache_file}")
            
        return df
    
    def fetch_index_history(self,
                          index_name: str,
                          start_date: datetime,
                          end_date: datetime,
                          interval: str = "1m") -> Optional[pd.DataFrame]:
        """
        Fetch historical data for an index.
        
        Args:
            index_name: Index name (NIFTY, BANKNIFTY, etc.)
            start_date: Start date
            end_date: End date
            interval: Time interval
            
        Returns:
            DataFrame with OHLCV data
        """
        # Indices are handled similarly to equities
        return self.fetch_equity_history(index_name, start_date, end_date, interval)
    
    def _fetch_nsepy(self, symbol: str, start_date: datetime, 
                    end_date: datetime, interval: str) -> Optional[pd.DataFrame]:
        """Fetch data using nsepy library."""
        if not HAS_NSEPY:
            return None
        
        try:
            # nsepy primarily supports daily data
            if interval not in ["1d", "1D"]:
                logger.warning("nsepy only supports daily data")
                return None
            
            # Check if it's an index
            if symbol in ["NIFTY", "BANKNIFTY"]:
                df = get_history(
                    symbol=symbol,
                    start=start_date,
                    end=end_date,
                    index=True
                )
            else:
                df = get_history(
                    symbol=symbol,
                    start=start_date,
                    end=end_date
                )
            
            if df.empty:
                return None
            
            # Reset index to get Date as a column
            df = df.reset_index()
            df.rename(columns={'Date': 'time'}, inplace=True)
            
            return df
            
        except Exception as e:
            logger.error(f"nsepy fetch failed: {e}")
            return None
    
    def _fetch_yfinance(self, symbol: str, start_date: datetime,
                       end_date: datetime, interval: str) -> Optional[pd.DataFrame]:
        """Fetch data using yfinance library."""
        if not HAS_YFINANCE:
            return None
        
        try:
            # Convert symbol to Yahoo format
            yahoo_symbol = self.YAHOO_SYMBOL_MAP.get(symbol, f"{symbol}.NS")
            
            # Map interval to yfinance format
            interval_map = {
                "1m": "1m", "5m": "5m", "15m": "15m", "30m": "30m",
                "1h": "1h", "1d": "1d", "1D": "1d"
            }
            yf_interval = interval_map.get(interval, "1d")
            
            # Note: yfinance has limitations on intraday data (only last 7-60 days)
            ticker = yf.Ticker(yahoo_symbol)
            
            # For intraday, use download with specific period
            if yf_interval in ["1m", "5m", "15m", "30m"]:
                # yfinance limits: 1m data for last 7 days, 5m for 60 days
                days_limit = 7 if yf_interval == "1m" else 60
                actual_start = max(start_date, datetime.now() - timedelta(days=days_limit))
                
                df = ticker.history(
                    start=actual_start,
                    end=end_date,
                    interval=yf_interval
                )
            else:
                df = ticker.history(
                    start=start_date,
                    end=end_date,
                    interval=yf_interval
                )
            
            if df.empty:
                return None
            
            # Reset index to get Datetime as column
            df = df.reset_index()
            df.rename(columns={'Datetime': 'time', 'Date': 'time'}, inplace=True)
            
            return df
            
        except Exception as e:
            logger.error(f"yfinance fetch failed: {e}")
            return None
    
    def _fetch_jugaad(self, symbol: str, start_date: datetime,
                     end_date: datetime, interval: str) -> Optional[pd.DataFrame]:
        """Fetch data using jugaad-data library."""
        if not HAS_JUGAAD:
            return None
        
        try:
            # jugaad-data supports various intervals
            n = nse.NSELive()
            
            # For indices
            if symbol in ["NIFTY", "BANKNIFTY"]:
                # jugaad might need different approach for indices
                stock_data = n.stock_quote(symbol)
                # This typically returns current quote, not historical
                logger.warning("jugaad-data historical fetch not fully implemented")
                return None
            
            # For stocks - this is a simplified example
            # jugaad-data API might differ, check documentation
            from jugaad_data.nse import stock_df
            df = stock_df(
                symbol=symbol,
                from_date=start_date,
                to_date=end_date,
                series="EQ"
            )
            
            if df.empty:
                return None
                
            df.rename(columns={'DATE': 'time'}, inplace=True)
            return df
            
        except Exception as e:
            logger.error(f"jugaad fetch failed: {e}")
            return None
    
    def _standardize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize DataFrame columns to expected format.
        
        Expected columns: time, open, high, low, close, volume
        """
        # Common column mappings
        column_map = {
            'Time': 'time', 'Datetime': 'time', 'Date': 'time', 'DATE': 'time',
            'Open': 'open', 'OPEN': 'open',
            'High': 'high', 'HIGH': 'high', 
            'Low': 'low', 'LOW': 'low',
            'Close': 'close', 'CLOSE': 'close', 'Last': 'close',
            'Volume': 'volume', 'VOLUME': 'volume', 'Total Trade Quantity': 'volume',
            'No. of Trades': 'trades', 'Trades': 'trades'
        }
        
        # Rename columns
        df.rename(columns=column_map, inplace=True)
        
        # Ensure required columns exist
        required = ['time', 'open', 'high', 'low', 'close']
        for col in required:
            if col not in df.columns:
                logger.warning(f"Missing required column: {col}")
                return pd.DataFrame()
        
        # Add volume if missing
        if 'volume' not in df.columns:
            df['volume'] = 0
        
        # Select only required columns
        df = df[['time', 'open', 'high', 'low', 'close', 'volume']]
        
        # Ensure time is datetime
        if not pd.api.types.is_datetime64_any_dtype(df['time']):
            df['time'] = pd.to_datetime(df['time'])
        
        # Sort by time
        df.sort_values('time', inplace=True)
        
        # Remove duplicates
        df.drop_duplicates(subset=['time'], inplace=True)
        
        # Reset index
        df.reset_index(drop=True, inplace=True)
        
        return df
    
    def _get_cache_filename(self, symbol: str, start_date: datetime,
                          end_date: datetime, interval: str) -> str:
        """Generate cache filename."""
        start_str = start_date.strftime("%Y%m%d")
        end_str = end_date.strftime("%Y%m%d")
        filename = f"{symbol}_{interval}_{start_str}_{end_str}.csv"
        return os.path.join(self.cache_dir, filename)
    
    def get_sample_data(self, symbol: str = "NIFTY", days: int = 30) -> pd.DataFrame:
        """
        Get sample data for testing.
        
        Args:
            symbol: Symbol to fetch
            days: Number of days of historical data
            
        Returns:
            DataFrame with OHLCV data
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Try daily data first (most reliable)
        df = self.fetch_equity_history(symbol, start_date, end_date, "1d")
        
        if df is None or df.empty:
            logger.warning(f"Could not fetch sample data for {symbol}")
            # Generate synthetic data for testing
            df = self._generate_synthetic_data(start_date, end_date)
        
        return df
    
    def _generate_synthetic_data(self, start_date: datetime, 
                                end_date: datetime) -> pd.DataFrame:
        """Generate synthetic OHLCV data for testing."""
        logger.info("Generating synthetic data for testing")
        
        # Generate minute-level data
        dates = pd.date_range(start=start_date, end=end_date, freq='1min')
        
        # Filter for market hours (9:15 AM to 3:30 PM IST)
        dates = [d for d in dates if 9 <= d.hour < 16]
        
        n = len(dates)
        if n == 0:
            return pd.DataFrame()
        
        # Generate realistic price movement
        np.random.seed(42)
        base_price = 25000  # Base NIFTY level
        returns = np.random.normal(0, 0.001, n)  # 0.1% volatility
        prices = base_price * np.exp(np.cumsum(returns))
        
        # Create OHLC from prices
        df = pd.DataFrame({
            'time': dates,
            'open': prices * (1 + np.random.uniform(-0.001, 0.001, n)),
            'high': prices * (1 + np.random.uniform(0, 0.002, n)),
            'low': prices * (1 + np.random.uniform(-0.002, 0, n)),
            'close': prices,
            'volume': np.random.randint(1000, 10000, n)
        })
        
        # Ensure high >= close, open and low <= close, open
        df['high'] = df[['open', 'high', 'close']].max(axis=1)
        df['low'] = df[['open', 'low', 'close']].min(axis=1)
        
        return df
