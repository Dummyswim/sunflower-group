"""
Historical data fetcher for Dhan API with backtesting support.
"""
import requests
import pandas as pd
import numpy as np
import base64
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional, List, Any
import time

logger = logging.getLogger(__name__)

class DhanHistoricalData:
    """Fetch and process historical data from Dhan API."""
    
    BASE_URL = "https://api.dhan.co"
    
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
    
    def get_historical_data(self, 
                          security_id: str,
                          exchange_segment: str,
                          instrument: str,
                          expiry_code: int,
                          from_date: str,
                          to_date: str,
                          resolution: str = "1") -> Optional[pd.DataFrame]:
        """
        Fetch historical OHLC data from Dhan API.
        
        Args:
            security_id: Security ID of the instrument
            exchange_segment: Exchange segment (e.g., "IDX_I", "NSE_EQ")
            instrument: Instrument type (e.g., "INDEX", "EQUITY")
            expiry_code: Expiry code (0 for non-derivatives)
            from_date: Start date in YYYY-MM-DD format
            to_date: End date in YYYY-MM-DD format
            resolution: Time resolution ("1" for 1 minute, "D" for daily)
            
        Returns:
            DataFrame with OHLC data or None if error
        """
        try:
            endpoint = f"{self.BASE_URL}/v2/charts/historical"
            
            # Convert dates to required format
            from_dt = datetime.strptime(from_date, "%Y-%m-%d")
            to_dt = datetime.strptime(to_date, "%Y-%m-%d")
            
            # Adjust for IST timezone
            from_timestamp = from_dt.strftime("%Y-%m-%d %H:%M:%S")
            to_timestamp = to_dt.strftime("%Y-%m-%d 15:30:00")  # Market close time
            
            payload = {
                "securityId": security_id,
                "exchangeSegment": exchange_segment,
                "instrument": instrument,
                "expiryCode": expiry_code,
                "fromDate": from_timestamp,
                "toDate": to_timestamp,
                "resolution": resolution
            }
            
            logger.info(f"Fetching historical data from {from_date} to {to_date}")
            
            response = requests.post(endpoint, json=payload, headers=self.headers)
            
            if response.status_code == 200:
                data = response.json()
                
                if "data" in data and "candles" in data["data"]:
                    candles = data["data"]["candles"]
                    
                    if not candles:
                        logger.warning("No data received for the specified period")
                        return None
                    
                    # Convert to DataFrame
                    df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 
                                                        'low', 'close', 'volume'])
                    
                    # Convert timestamp to datetime
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
                    df.set_index('timestamp', inplace=True)
                    
                    # Sort by timestamp
                    df.sort_index(inplace=True)
                    
                    logger.info(f"Retrieved {len(df)} candles")
                    return df
                else:
                    logger.error(f"Invalid response structure: {data}")
                    return None
            else:
                logger.error(f"API request failed: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"Error fetching historical data: {e}")
            return None
    
    def get_intraday_data(self, from_date: str, to_date: str, 
                         security_id: str = "13", 
                         exchange_segment: str = "IDX_I") -> Optional[pd.DataFrame]:
        """
        Fetch intraday (1-minute) OHLC data.
        """
        return self.get_historical_data(
            security_id=security_id,
            exchange_segment=exchange_segment,
            instrument="INDEX",
            expiry_code=0,
            from_date=from_date,
            to_date=to_date,
            resolution="1"  # 1 minute
        )
    
    def get_daily_data(self, from_date: str, to_date: str,
                      security_id: str = "13",
                      exchange_segment: str = "IDX_I") -> Optional[pd.DataFrame]:
        """
        Fetch daily OHLC data.
        """
        return self.get_historical_data(
            security_id=security_id,
            exchange_segment=exchange_segment,
            instrument="INDEX",
            expiry_code=0,
            from_date=from_date,
            to_date=to_date,
            resolution="D"  # Daily
        )
    
    def fetch_data_in_chunks(self, from_date: str, to_date: str, 
                           timeframe: str = "minute",
                           chunk_days: int = 5) -> Optional[pd.DataFrame]:
        """
        Fetch data in chunks to handle large date ranges.
        
        Args:
            from_date: Start date
            to_date: End date
            timeframe: "minute" or "daily"
            chunk_days: Days per chunk for minute data
            
        Returns:
            Combined DataFrame
        """
        try:
            from_dt = datetime.strptime(from_date, "%Y-%m-%d")
            to_dt = datetime.strptime(to_date, "%Y-%m-%d")
            
            all_data = []
            current_date = from_dt
            
            while current_date < to_dt:
                chunk_end = min(current_date + timedelta(days=chunk_days), to_dt)
                
                logger.info(f"Fetching chunk: {current_date.strftime('%Y-%m-%d')} to "
                          f"{chunk_end.strftime('%Y-%m-%d')}")
                
                if timeframe == "minute":
                    chunk_data = self.get_intraday_data(
                        current_date.strftime("%Y-%m-%d"),
                        chunk_end.strftime("%Y-%m-%d")
                    )
                else:
                    chunk_data = self.get_daily_data(
                        current_date.strftime("%Y-%m-%d"),
                        chunk_end.strftime("%Y-%m-%d")
                    )
                
                if chunk_data is not None and not chunk_data.empty:
                    all_data.append(chunk_data)
                
                current_date = chunk_end + timedelta(days=1)
                
                # Rate limiting
                time.sleep(0.5)
            
            if all_data:
                combined_df = pd.concat(all_data)
                combined_df = combined_df[~combined_df.index.duplicated(keep='first')]
                combined_df.sort_index(inplace=True)
                return combined_df
            
            return None
            
        except Exception as e:
            logger.error(f"Error fetching data in chunks: {e}")
            return None
