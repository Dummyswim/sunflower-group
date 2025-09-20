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
        """Initialize DataPersistenceManager with robust error handling and logging."""
        logger.info("=" * 60)
        logger.info("INITIALIZING DATA PERSISTENCE")
        logger.info("=" * 60)

         
        
        self.config = config
        self.data_dir = Path("data")
        self.data_dir.mkdir(exist_ok=True)
        
        # File paths
        self.rolling_file_5m = self.data_dir / "sensex_rolling_5m.pkl"
        self.rolling_file_15m = self.data_dir / "sensex_rolling_15m.pkl"
        self.metadata_file = self.data_dir / "metadata.json"
        
        # Initialize with proper timezone-aware empty DataFrames
        import pytz
        self.ist = pytz.timezone('Asia/Kolkata')
        
        # Create properly initialized DataFrames - NEVER None
        self.data_5m = self._create_empty_dataframe()
        self.data_15m = self._create_empty_dataframe()
        self.metadata = {}
        
        # Rolling window parameters
        self.max_days_5m = 30 # Keep 30 days of 5-min data
        self.max_days_15m = 60  # Keep 60 days of 15-min data
        self.min_required_candles = 200 # Minimum candles needed for indicators
        
        # API configuration
        self.api_base_url = "https://api.dhan.co/v2"
        
        logger.info("DataPersistenceManager initialized")
        logger.info(f"✅ Data directory: {self.data_dir}")
        logger.info(f"✅ 5m file: {self.rolling_file_5m}")
        logger.info(f"✅ 15m file: {self.rolling_file_15m}")
        logger.info(f"✅ DataFrames initialized (5m: {type(self.data_5m)}, 15m: {type(self.data_15m)})")


    def _create_empty_dataframe(self) -> pd.DataFrame:
        """Create a properly initialized empty DataFrame with timezone-aware index."""
        logger.debug("Creating empty DataFrame with IST timezone")
        
        df = pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])
        df.index = pd.DatetimeIndex([], tz=self.ist, name='timestamp')
        
        logger.debug(f"Created DataFrame: columns={df.columns.tolist()}, index_tz={df.index.tz}")
        return df


    def _ensure_dataframes_valid(self) -> None:
        """Ensure all DataFrames are valid and never None."""
        logger.debug("Validating DataFrames...")
        
        # Check and fix data_5m
        if self.data_5m is None:
            logger.warning("⚠️ data_5m was None, creating new DataFrame")
            self.data_5m = self._create_empty_dataframe()
        elif not isinstance(self.data_5m, pd.DataFrame):
            logger.warning(f"⚠️ data_5m was {type(self.data_5m)}, creating new DataFrame")
            self.data_5m = self._create_empty_dataframe()
        
        # Check and fix data_15m
        if self.data_15m is None:
            logger.warning("⚠️ data_15m was None, creating new DataFrame")
            self.data_15m = self._create_empty_dataframe()
        elif not isinstance(self.data_15m, pd.DataFrame):
            logger.warning(f"⚠️ data_15m was {type(self.data_15m)}, creating new DataFrame")
            self.data_15m = self._create_empty_dataframe()
        
        logger.debug(f"✅ DataFrames valid - 5m: {len(self.data_5m)} rows, 15m: {len(self.data_15m)} rows")



    def _normalize_index_tz(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure DateTimeIndex is tz-aware Asia/Kolkata and sorted. Always returns a DataFrame."""
        logger.debug(f"Normalizing index timezone - Input type: {type(df)}")
        
        try:
            # Handle None input - return empty DataFrame
            if df is None:
                logger.warning("Input is None, returning empty DataFrame")
                return self._create_empty_dataframe()
            
            # Handle non-DataFrame input
            if not isinstance(df, pd.DataFrame):
                logger.warning(f"Input is {type(df)}, returning empty DataFrame")
                return self._create_empty_dataframe()
            
            # Handle empty DataFrame
            if df.empty:
                logger.debug("Input DataFrame is empty, returning as-is")
                return df
            
            # Make a copy to avoid modifying original
            out = df.copy()
            
            # Log current index state
            logger.debug(f"Current index: type={type(out.index)}, tz={getattr(out.index, 'tz', 'None')}, len={len(out)}")
            
            # Ensure we have a DatetimeIndex
            if not isinstance(out.index, pd.DatetimeIndex):
                logger.info("Converting index to DatetimeIndex")
                out.index = pd.to_datetime(out.index, errors='coerce')
            
            # Handle timezone
            if out.index.tz is None:
                logger.info("Localizing naive index to Asia/Kolkata")
                out.index = out.index.tz_localize(self.ist)
            elif out.index.tz != self.ist:
                logger.info(f"Converting index from {out.index.tz} to Asia/Kolkata")
                out.index = out.index.tz_convert(self.ist)
            else:
                logger.debug("Index already in Asia/Kolkata timezone")
            
            # Sort by index and standardize index name 
            out = out.sort_index()
            
            logger.debug(f"✅ Normalized: tz={out.index.tz}, len={len(out)}")
            return out
            
        except Exception as e:
            logger.error(f"❌ Error normalizing index: {e}", exc_info=True)
            logger.info("Returning empty DataFrame due to error")
            return self._create_empty_dataframe()



    def initialize(self) -> bool:
        """Initialize data storage - load from persistent file or fetch from API."""
        try:
            logger.info("=" * 60)
            logger.info("INITIALIZING DATA PERSISTENCE")
            logger.info("=" * 60)
            
            # Ensure DataFrames are valid before starting
            self._ensure_dataframes_valid()
            logger.info("✅ DataFrames validated")
            
            # Try to load from persistent storage
            logger.info("Loading from persistent storage...")
            loaded_5m = self._load_from_file("5m")
            loaded_15m = self._load_from_file("15m")
            
            logger.info(f"Load results: 5m={loaded_5m}, 15m={loaded_15m}")
            
            # Ensure DataFrames are still valid after loading
            self._ensure_dataframes_valid()
            
            # Check if data is fresh and sufficient
            needs_update_5m = self._check_data_freshness("5m", self.data_5m)
            needs_update_15m = self._check_data_freshness("15m", self.data_15m)
            
            logger.info(f"Freshness check: 5m needs update={needs_update_5m}, 15m needs update={needs_update_15m}")
            
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
                
            try: 
                if not self.data_5m.empty: 
                    logger.debug(f"5m tz={self.data_5m.index.tz}, last={self.data_5m.index[-1]}")
                    
                    if not self.data_15m.empty: 
                        logger.debug(f"15m tz={self.data_15m.index.tz}, last={self.data_15m.index[-1]}") 
            except Exception: 
                pass
            
            return True
            
        except Exception as e:
            logger.error(f"Data initialization failed: {e}")
            return False
    
        
    def _load_from_file(self, timeframe: str) -> bool:
        """Load data from persistent file with enhanced error handling."""
        try:
            if timeframe == "5m":
                file_path = self.rolling_file_5m
            else:
                file_path = self.rolling_file_15m
            
            logger.info(f"Attempting to load {timeframe} data from {file_path}")
            
            if not file_path.exists():
                logger.info(f"No existing {timeframe} data file found at {file_path}")
                return False
            
            # Load data
            logger.debug(f"Opening file: {file_path}")
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            
            logger.info(f"Loaded object type: {type(data)}")
            
            # Validate loaded data
            if data is None:
                logger.warning(f"Loaded data is None for {timeframe}")
                return False
            
            if not isinstance(data, pd.DataFrame):
                logger.warning(f"Loaded data is not a DataFrame: {type(data)}")
                return False
            
            if data.empty:
                logger.warning(f"Loaded DataFrame is empty for {timeframe}")
                return False
            
            # Verify required columns
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                logger.warning(f"Missing columns in {timeframe} data: {missing_columns}")
                return False
            
            logger.info(f"Data validation passed: {len(data)} rows, columns={data.columns.tolist()}")
            
            # Normalize index timezone
            data = self._normalize_index_tz(data)
            
            # Ensure normalization didn't return None
            if data is None or not isinstance(data, pd.DataFrame):
                logger.error(f"Normalization failed for {timeframe} data")
                return False
            
            # Store the data
            if timeframe == "5m":
                self.data_5m = data
            else:
                self.data_15m = data
            
            logger.info(f"✅ Successfully loaded {len(data)} {timeframe} candles from storage")
            logger.debug(f"Date range: {data.index[0]} to {data.index[-1]}")
            
            # Load metadata
            self._load_metadata()
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error loading {timeframe} data: {e}", exc_info=True)
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

            # Ensure numeric columns are finite
            try:
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    if col in data.columns:
                        data[col] = pd.to_numeric(data[col], errors='coerce').replace([np.inf, -np.inf], np.nan).fillna(0.0)
            except Exception:
                pass

                        
            # Normalize and standardize index (defensive)
            data = self._normalize_index_tz(data)
            try:
                data.index.name = 'timestamp'
            except Exception:
                pass

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
        """Check if data needs updating with robust None handling."""
        try:
            logger.debug(f"Checking freshness for {timeframe} data")
            
            # Handle None or invalid data
            if data is None:
                logger.info(f"Data is None for {timeframe}, need to fetch")
                return True
            
            if not isinstance(data, pd.DataFrame):
                logger.info(f"Data is not DataFrame ({type(data)}) for {timeframe}, need to fetch")
                return True
            
            if data.empty:
                logger.info(f"No {timeframe} data available (empty DataFrame), need to fetch")
                return True
            
            # Check if we have enough data
            min_candles = self.min_required_candles
            current_candles = len(data)
            
            if current_candles < min_candles:
                logger.info(f"Insufficient {timeframe} data: {current_candles} < {min_candles}")
                return True
            
            # Check data age
            try:
                import pytz
                tz = pytz.timezone('Asia/Kolkata')
                current_time = datetime.now(tz)
                
                last_ts = data.index[-1]
                if getattr(last_ts, 'tzinfo', None) is None:
                    last_ts = tz.localize(last_ts)
                else:
                    last_ts = last_ts.astimezone(tz)
                
                time_diff = (current_time - last_ts).total_seconds() / 60
                
                logger.info(f"{timeframe} data age: {time_diff:.0f} minutes old (last: {last_ts})")
                
            except Exception as e:
                logger.warning(f"Could not check data age: {e}")
            
            logger.info(f"{timeframe} data is sufficient: {current_candles} candles available")
            return False
            
        except Exception as e:
            logger.error(f"❌ Error checking data freshness: {e}", exc_info=True)
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
                "securityId": str(self.config.sensex_security_id),
                "exchangeSegment": self.config.sensex_exchange_segment,
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
                            
                        # logger.debug(f"Old-format candle ts normalized to IST: {timestamp} (tz={getattr(timestamp, 'tzinfo', None)})")
                        
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
                            
                            # Localize to IST (API old format may be naive) 
                            timestamp = pd.to_datetime(timestamp_str) 
                            if getattr(timestamp, 'tzinfo', None) is None: 
                                timestamp = timestamp.tz_localize('Asia/Kolkata') 
                            else: 
                                timestamp = timestamp.tz_convert('Asia/Kolkata')
                            
                            
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
        """Update data with new candle and save incrementally with enhanced logging."""
        try:
            logger.debug(f"update_candle called for {timeframe}")
            
            # Validate input
            if candle is None:
                logger.warning(f"Candle is None for {timeframe}, skipping update")
                return
            
            if not isinstance(candle, pd.DataFrame):
                logger.warning(f"Candle is not DataFrame ({type(candle)}), skipping update")
                return
            
            if candle.empty:
                logger.warning(f"Empty candle provided for {timeframe}, skipping update")
                return
            
            logger.debug(f"Candle info: shape={candle.shape}, index={candle.index[0] if not candle.empty else 'empty'}")
            
            # Ensure DataFrames are valid
            self._ensure_dataframes_valid()
            
            # Get current data and parameters
            if timeframe == "5m":
                current_data = self.data_5m
                max_candles = int(self.max_days_5m * 12 * 6.5)
            else:
                current_data = self.data_15m
                max_candles = int(self.max_days_15m * 4 * 6.5)
            
            logger.debug(f"Current {timeframe} data: {len(current_data)} rows, max={max_candles}")
            
            # Normalize the incoming candle
            candle = self._normalize_index_tz(candle)
            
            # Ensure current data is normalized
            if not current_data.empty:
                current_data = self._normalize_index_tz(current_data)
            
            # Append or update candle
            candle_time = candle.index[0]
            logger.debug(f"Processing candle at {candle_time}")
            
            if current_data.empty:
                logger.info(f"First candle for {timeframe}: {candle_time}")
                updated_data = candle.copy()
            else:
                if candle_time in current_data.index:
                    logger.debug(f"Updating existing candle at {candle_time}")
                    current_data.loc[candle_time] = candle.iloc[0]
                    updated_data = current_data
                else:
                    logger.debug(f"Appending new candle at {candle_time}")
                    updated_data = pd.concat([current_data, candle], axis=0)
            
            # Remove duplicates if any
            updated_data = updated_data[~updated_data.index.duplicated(keep='last')]
            
            # Trim to rolling window
            if len(updated_data) > max_candles:
                old_len = len(updated_data)
                updated_data = updated_data.tail(max_candles)
                logger.debug(f"Trimmed {timeframe} data from {old_len} to {len(updated_data)} rows")
            
            # Final sort
            updated_data = updated_data.sort_index()
            
            # Update in memory
            if timeframe == "5m":
                self.data_5m = updated_data
            else:
                self.data_15m = updated_data
            
            # Save to file
            self._save_to_file(timeframe, updated_data)
            
            logger.info(f"✅ Updated {timeframe} data: {len(updated_data)} total candles, last={candle_time}")
            
        except Exception as e:
            logger.error(f"❌ Error updating {timeframe} candle: {e}", exc_info=True)

    

    def get_data(self, timeframe: str, num_candles: Optional[int] = None) -> pd.DataFrame:
        """Get data for analysis with comprehensive error handling and logging."""
        try:
            logger.debug(f"get_data called: timeframe={timeframe}, num_candles={num_candles}")
            
            # Ensure DataFrames are valid
            self._ensure_dataframes_valid()
            
            # Get the appropriate data
            if timeframe == "5m":
                data = self.data_5m
            elif timeframe == "15m":
                data = self.data_15m
            else:
                logger.error(f"Invalid timeframe: {timeframe}")
                return self._create_empty_dataframe()
            
            # Validate data
            if data is None:
                logger.warning(f"Data is None for {timeframe}, returning empty DataFrame")
                return self._create_empty_dataframe()
            
            if not isinstance(data, pd.DataFrame):
                logger.warning(f"Data is not DataFrame ({type(data)}) for {timeframe}")
                return self._create_empty_dataframe()
            
            if data.empty:
                logger.debug(f"No {timeframe} data available (empty DataFrame)")
                return data.copy()
            
            # Normalize timezone
            data = self._normalize_index_tz(data)
            
            # Return requested number of candles
            if num_candles and num_candles > 0:
                if len(data) > num_candles:
                    result = data.tail(num_candles).copy()
                    logger.debug(f"Returning last {num_candles} candles of {len(data)} total")
                else:
                    result = data.copy()
                    logger.debug(f"Requested {num_candles} but only {len(data)} available")
            else:
                result = data.copy()
                logger.debug(f"Returning all {len(data)} candles")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Error getting {timeframe} data: {e}", exc_info=True)
            return self._create_empty_dataframe()


    
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
