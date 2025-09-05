"""
Data Persistence Module - Rolling Window with Persistent Storage
Handles data storage, retrieval, and historical data fetching
"""
import pandas as pd
import numpy as np
import json
import pickle
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple, List
import requests
import base64
import time
import hashlib

logger = logging.getLogger(__name__)

class DataPersistenceManager:
    """Manages persistent rolling window data storage and retrieval."""
    
    def __init__(self, config):
        self.config = config
        self.data_dir = Path("data")
        self.data_dir.mkdir(exist_ok=True)
        
        # File paths
        self.rolling_file_5m = self.data_dir / "nifty50_rolling_5m.pkl"
        self.rolling_file_15m = self.data_dir / "nifty50_rolling_15m.pkl"
        self.metadata_file = self.data_dir / "metadata.json"
        
        # Data storage
        self.data_5m = pd.DataFrame()
        self.data_15m = pd.DataFrame()
        self.metadata = {}
        
        # Rolling window parameters
        self.max_days_5m = 30  # Keep 30 days of 5-min data
        self.max_days_15m = 60  # Keep 60 days of 15-min data
        self.min_required_candles = 200  # Minimum candles needed for indicators
        
        # API configuration for historical data
        self.api_base_url = "https://api.dhan.co/v2"
        
        logger.info("DataPersistenceManager initialized")
        logger.info(f"Data directory: {self.data_dir}")
        
    def initialize(self) -> bool:
        """Initialize data storage - load from persistent file or fetch from API."""
        try:
            logger.info("=" * 60)
            logger.info("INITIALIZING DATA PERSISTENCE")
            logger.info("=" * 60)
            
            # Try to load from persistent storage
            loaded_5m = self._load_from_file("5m")
            loaded_15m = self._load_from_file("15m")
            
            # Check if data is fresh and sufficient
            needs_update_5m = self._check_data_freshness("5m", self.data_5m)
            needs_update_15m = self._check_data_freshness("15m", self.data_15m)
            
            # Fetch from API if needed
            if not loaded_5m or needs_update_5m:
                logger.info("Fetching 5-min data from API...")
                self.data_5m = self._fetch_historical_data("5")
                self._save_to_file("5m", self.data_5m)
                
            if not loaded_15m or needs_update_15m:
                logger.info("Fetching 15-min data from API...")
                self.data_15m = self._fetch_historical_data("15")
                self._save_to_file("15m", self.data_15m)
            
            # Log status
            logger.info(f"✅ Data loaded successfully:")
            logger.info(f"   5-min candles: {len(self.data_5m)}")
            logger.info(f"   15-min candles: {len(self.data_15m)}")
            
            if not self.data_5m.empty:
                logger.info(f"   5-min range: {self.data_5m.index[0]} to {self.data_5m.index[-1]}")
            if not self.data_15m.empty:
                logger.info(f"   15-min range: {self.data_15m.index[0]} to {self.data_15m.index[-1]}")
                
            return True
            
        except Exception as e:
            logger.error(f"Data initialization failed: {e}")
            return False
    
    def _load_from_file(self, timeframe: str) -> bool:
        """Load data from persistent file."""
        try:
            if timeframe == "5m":
                file_path = self.rolling_file_5m
            else:
                file_path = self.rolling_file_15m
            
            if not file_path.exists():
                logger.info(f"No existing {timeframe} data file found")
                return False
            
            # Load data
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            
            # Validate data
            if not isinstance(data, pd.DataFrame) or data.empty:
                logger.warning(f"Invalid or empty {timeframe} data in file")
                return False
            
            # Verify data integrity
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in data.columns for col in required_columns):
                logger.warning(f"Missing required columns in {timeframe} data")
                return False
            
            # Store data
            if timeframe == "5m":
                self.data_5m = data
            else:
                self.data_15m = data
            
            logger.info(f"✅ Loaded {len(data)} {timeframe} candles from persistent storage")
            
            # Load metadata
            self._load_metadata()
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading {timeframe} data: {e}")
            return False
    
    def _save_to_file(self, timeframe: str, data: pd.DataFrame) -> bool:
        """Save data to persistent file."""
        try:
            if data.empty:
                logger.warning(f"Cannot save empty {timeframe} data")
                return False

            if timeframe == "5m":
                file_path = self.rolling_file_5m
                max_candles = int(self.max_days_5m * 12 * 6.5)
            else:
                file_path = self.rolling_file_15m
                max_candles = int(self.max_days_15m * 4 * 6.5)

            
            # Trim to rolling window size
            if len(data) > max_candles:
                data = data.tail(int(max_candles))
            
            # Save data
            with open(file_path, 'wb') as f:
                pickle.dump(data, f)
            
            # Update metadata
            self._update_metadata(timeframe, data)
            
            logger.debug(f"Saved {len(data)} {timeframe} candles to {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving {timeframe} data: {e}")
            return False
    
    def _check_data_freshness(self, timeframe: str, data: pd.DataFrame) -> bool:
        """Check if data needs updating."""
        try:
            if data.empty:
                logger.info(f"No {timeframe} data available, need to fetch")
                return True
            
            # Check if we have enough data
            min_candles = self.min_required_candles
            if len(data) < min_candles:
                logger.info(f"Insufficient {timeframe} data: {len(data)} < {min_candles}")
                return True
            
            import pytz # at top if not present ... 
            tz = pytz.timezone('Asia/Kolkata') 
            current_time = datetime.now(tz) 
            last_ts = data.index[-1] 
            if getattr(last_ts, 'tzinfo', None) is None: 
                last_ts = tz.localize(last_ts) 
            else: 
                last_ts = last_ts.astimezone(tz) 
            time_diff = (current_time - last_ts).total_seconds() / 60
            

       
            
            # if time_diff > max_age_minutes:
            #     logger.info(f"{timeframe} data is stale: {time_diff:.0f} minutes old")
            #     return True
            
            # logger.info(f"{timeframe} data is fresh: {time_diff:.0f} minutes old")
            return False
            
        except Exception as e:
            logger.error(f"Error checking data freshness: {e}")
            return True
        
    def _fetch_historical_data(self, interval: str) -> pd.DataFrame:
        """Fetch historical data from Dhan API."""
        try:
            # Decode access token
            access_token = base64.b64decode(self.config.dhan_access_token_b64.strip()).decode("utf-8").strip()
            
            # Calculate date range
            to_date = datetime.now()
            if interval == "5":
                from_date = to_date - timedelta(days=30)
            else:  # 15 min
                from_date = to_date - timedelta(days=60)
            
            # Format dates for API
            from_str = from_date.strftime("%Y-%m-%d 09:15:00")
            to_str = to_date.strftime("%Y-%m-%d 15:15:00")
            
            logger.info(f"Fetching {interval}-min data from {from_str} to {to_str}")
            
            # API request
            headers = {
                'Accept': 'application/json',
                'Content-Type': 'application/json',
                'access-token': access_token
            }
            
            payload = {
                "securityId": str(self.config.nifty_security_id),
                "exchangeSegment": self.config.nifty_exchange_segment,
                "instrument": "INDEX",
                "interval": interval,
                "fromDate": from_str,
                "toDate": to_str,
                "oi": False
            }
            
            logger.info(f"API Request Payload: {payload}")
            
            response = requests.post(
                f"{self.api_base_url}/charts/intraday",
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code != 200:
                logger.error(f"API error: {response.status_code} - {response.text}")
                return pd.DataFrame()
            
            data = response.json()
            logger.info(f"API Response received with status {response.status_code}")
            
            if not data:
                logger.error("Empty response from API")
                return pd.DataFrame()
            
            # The response format is directly the OHLCV arrays (as shown in tick.json)
            # Check if we have the expected keys
            if 'open' in data and isinstance(data['open'], list):
                logger.info("Processing data in array format (direct OHLCV arrays)")
                
                open_prices = data.get('open', [])
                high_prices = data.get('high', [])
                low_prices = data.get('low', [])
                close_prices = data.get('close', [])
                volumes = data.get('volume', [])
                timestamps = data.get('timestamp', [])
                
                if not open_prices:
                    logger.error("No price data in API response")
                    return pd.DataFrame()
                
                logger.info(f"Received {len(open_prices)} data points")
                
                # Create DataFrame from arrays
                candles = []
                num_candles = len(open_prices)
                
                for i in range(num_candles):
                    try:
                        # Get timestamp - it's in Unix timestamp format
                        if timestamps and i < len(timestamps):
                            # Convert Unix timestamp to datetime
                            
                            timestamp = pd.to_datetime(timestamps[i], unit='s', utc=True).tz_convert('Asia/Kolkata')
                        else:
                            # Calculate timestamp based on interval
                            base_time = pd.to_datetime(from_str).tz_localize('Asia/Kolkata')
                            if interval == "5":
                                timestamp = base_time + timedelta(minutes=5*i)
                            else:  # 15 min
                                timestamp = base_time + timedelta(minutes=15*i)
                        
                        # Skip non-market hours
                        if timestamp.hour < 9 or (timestamp.hour == 9 and timestamp.minute < 15):
                            continue
                        if timestamp.hour > 15 or (timestamp.hour == 15 and timestamp.minute > 30):
                            continue
                        
                        # Skip weekends
                        if timestamp.weekday() >= 5:
                            continue
                        
                        # Get OHLC values - all should be at the same index
                        open_val = float(open_prices[i]) if i < len(open_prices) else 0
                        high_val = float(high_prices[i]) if i < len(high_prices) else 0
                        low_val = float(low_prices[i]) if i < len(low_prices) else 0
                        close_val = float(close_prices[i]) if i < len(close_prices) else 0
                        
                        # Skip invalid data
                        if open_val == 0 or close_val == 0:
                            continue
                        
                        # Get or create synthetic volume for index
                        if volumes and i < len(volumes) and volumes[i] > 0:
                            volume = int(volumes[i])
                        else:
                            # Create synthetic volume for index
                            price_range = high_val - low_val if high_val > low_val else 0.01
                            volatility = (price_range / close_val) * 100 if close_val > 0 else 0.1
                            volume = int(10000 + volatility * 5000)
                        
                        candles.append({
                            'timestamp': timestamp,
                            'open': open_val,
                            'high': high_val,
                            'low': low_val,
                            'close': close_val,
                            'volume': volume
                        })
                        
                    except Exception as e:
                        logger.debug(f"Skipping candle {i}: {e}")
                        continue
                        
            elif 'data' in data:
                # Check if it's nested under 'data' key
                api_data = data['data']
                
                if 'open' in api_data and isinstance(api_data['open'], list):
                    logger.info("Processing nested data format")
                    
                    open_prices = api_data.get('open', [])
                    high_prices = api_data.get('high', [])
                    low_prices = api_data.get('low', [])
                    close_prices = api_data.get('close', [])
                    volumes = api_data.get('volume', [])
                    timestamps = api_data.get('timestamp', [])
                    
                    if not open_prices:
                        logger.error("No price data in nested API response")
                        return pd.DataFrame()
                    
                    logger.info(f"Received {len(open_prices)} data points from nested structure")
                    
                    # Process same as above
                    candles = []
                    for i in range(len(open_prices)):
                        try:
                            # Get timestamp
                            if timestamps and i < len(timestamps):
                                # Treat API epoch as UTC then convert to IST
                                timestamp = pd.to_datetime(timestamps[i], unit='s', utc=True).tz_convert('Asia/Kolkata')
                            else:
                                # Build base time explicitly in IST
                                base_time = pd.to_datetime(from_str).tz_localize('Asia/Kolkata')
                                if interval == "5":
                                    timestamp = base_time + timedelta(minutes=5*i)
                                else:
                                    timestamp = base_time + timedelta(minutes=15*i)

                            # Skip non-market hours and weekends
                            if timestamp.hour < 9 or (timestamp.hour == 9 and timestamp.minute < 15):
                                continue
                            if timestamp.hour > 15 or (timestamp.hour == 15 and timestamp.minute > 30):
                                continue
                            if timestamp.weekday() >= 5:
                                continue
                            
                            open_val = float(open_prices[i]) if i < len(open_prices) else 0
                            high_val = float(high_prices[i]) if i < len(high_prices) else 0
                            low_val = float(low_prices[i]) if i < len(low_prices) else 0
                            close_val = float(close_prices[i]) if i < len(close_prices) else 0
                            
                            if open_val == 0 or close_val == 0:
                                continue
                            
                            # Volume handling
                            if volumes and i < len(volumes) and volumes[i] > 0:
                                volume = int(volumes[i])
                            else:
                                price_range = high_val - low_val
                                volatility = (price_range / close_val) * 100 if close_val > 0 else 0
                                volume = int(10000 + volatility * 5000)
                            
                            candles.append({
                                'timestamp': timestamp,
                                'open': open_val,
                                'high': high_val,
                                'low': low_val,
                                'close': close_val,
                                'volume': volume
                            })
                        except Exception as e:
                            logger.debug(f"Skipping candle {i}: {e}")
                            continue
                            
                elif 'candles' in api_data:
                    # Old format with candles array (fallback)
                    logger.info("Processing old candle format")
                    for candle in api_data.get('candles', []):
                        if len(candle) >= 6:
                            timestamp_str, open_val, high, low, close, volume = candle[:6]
                            timestamp = pd.to_datetime(timestamp_str)
                            
                            if volume == 0 or volume is None:
                                price_range = high - low
                                volatility = (price_range / close) * 100 if close > 0 else 0
                                volume = int(10000 + volatility * 5000)
                            
                            candles.append({
                                'timestamp': timestamp,
                                'open': float(open_val),
                                'high': float(high),
                                'low': float(low),
                                'close': float(close),
                                'volume': int(volume)
                            })
                else:
                    logger.error(f"Unknown nested data format. Keys: {list(api_data.keys())[:10]}")
                    return pd.DataFrame()
            else:
                logger.error(f"Unknown response format. Top-level keys: {list(data.keys())}")
                return pd.DataFrame()
            
            if not candles:
                logger.warning("No valid candles parsed from API response")
                return pd.DataFrame()
            
            # Create DataFrame
            df = pd.DataFrame(candles).set_index('timestamp')
            df = df.sort_index()
            
            # Remove any duplicate timestamps
            df = df[~df.index.duplicated(keep='last')]
            
            logger.info(f"✅ Fetched {len(df)} candles from API")
            if len(df) > 0:
                logger.info(f"   Date range: {df.index[0]} to {df.index[-1]}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching historical data: {e}", exc_info=True)
            return pd.DataFrame()

    
    def update_candle(self, candle: pd.DataFrame, timeframe: str):
        """Update data with new candle and save incrementally."""
        try:
            if candle.empty:
                logger.warning(f"Empty candle provided for update- {candle.empty}")
                return

            if timeframe == "5m":
                current_data = self.data_5m
                max_candles = int(self.max_days_5m * 12 * 6.5)
            else:
                current_data = self.data_15m
                max_candles = int(self.max_days_15m * 4 * 6.5)

            # Append or update candle
            if current_data is None or current_data.empty:
                updated_data = candle.copy()
            else:
                candle_time = candle.index[0]
                if candle_time in current_data.index:
                    current_data.loc[candle_time] = candle.iloc[0]
                    updated_data = current_data
                else:
                    updated_data = pd.concat([current_data, candle])

            
            # Trim to rolling window
            if len(updated_data) > max_candles:
                updated_data = updated_data.tail(max_candles)
            
            # Update in memory
            if timeframe == "5m":
                self.data_5m = updated_data
            else:
                self.data_15m = updated_data
            
            # Save to file (incremental save)
            self._save_to_file(timeframe, updated_data)
            
            logger.debug(f"Updated {timeframe} data with new candle at {candle.index[0]}")
            
        except Exception as e:
            logger.error(f"Error updating {timeframe} candle: {e}")
    
    def get_data(self, timeframe: str, num_candles: Optional[int] = None) -> pd.DataFrame:
        """Get data for analysis."""
        try:
            if timeframe == "5m":
                data = self.data_5m
            elif timeframe == "15m":
                data = self.data_15m
            else:
                logger.error(f"Invalid timeframe: {timeframe}")
                return pd.DataFrame()
            
            if data.empty:
                logger.warning(f"No {timeframe} data available")
                return pd.DataFrame()
            
            if num_candles:
                return data.tail(num_candles).copy()
            else:
                return data.copy()
                
        except Exception as e:
            logger.error(f"Error getting {timeframe} data: {e}")
            return pd.DataFrame()
    
    def _load_metadata(self):
        """Load metadata about stored data."""
        try:
            if self.metadata_file.exists():
                with open(self.metadata_file, 'r') as f:
                    self.metadata = json.load(f)
                logger.debug("Metadata loaded successfully")
        except Exception as e:
            logger.error(f"Error loading metadata: {e}")
            self.metadata = {}
    
    def _update_metadata(self, timeframe: str, data: pd.DataFrame):
        """Update metadata about stored data."""
        try:
            if data.empty:
                return
            
            self.metadata[timeframe] = {
                'last_update': datetime.now().isoformat(),
                'candle_count': len(data),
                'first_candle': str(data.index[0]),
                'last_candle': str(data.index[-1]),
                'file_size': self.rolling_file_5m.stat().st_size if timeframe == "5m" else self.rolling_file_15m.stat().st_size
            }
            
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error updating metadata: {e}")
    
    def get_combined_analysis_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Get both 5m and 15m data for multi-timeframe analysis."""
        return self.data_5m.copy(), self.data_15m.copy()
    
    def cleanup_old_data(self):
        """Clean up old data beyond rolling window."""
        try:
            # Clean 5m data
            if len(self.data_5m) > 0:
                max_candles_5m = int(self.max_days_5m * 12 * 6.5)
                if len(self.data_5m) > max_candles_5m:
                    self.data_5m = self.data_5m.tail(max_candles_5m)
                    self._save_to_file("5m", self.data_5m)
                    logger.info(f"Cleaned up 5m data to {len(self.data_5m)} candles")
            
            # Clean 15m data
            if len(self.data_15m) > 0:
                max_candles_15m = int(self.max_days_15m * 4 * 6.5)
                if len(self.data_15m) > max_candles_15m:
                    self.data_15m = self.data_15m.tail(max_candles_15m)
                    self._save_to_file("15m", self.data_15m)
                    logger.info(f"Cleaned up 15m data to {len(self.data_15m)} candles")
                    
        except Exception as e:
            logger.error(f"Error cleaning up old data: {e}")
