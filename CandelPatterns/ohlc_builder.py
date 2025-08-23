"""
Enhanced OHLC candle builder with tick aggregation.
"""
import collections
import logging
from datetime import datetime, timezone
from typing import Dict, Optional
import pandas as pd

logger = logging.getLogger(__name__)

class OHLCBuilder:
    """
    Builds OHLC candles from tick data with proper time alignment.
    """
    
    def __init__(self, window: int = 300):
        """
        Initialize OHLC builder.
        
        Args:
            window: Maximum number of candles to keep in memory
        """
        self.window = window
        self.candles = collections.OrderedDict()
        self.current_minute = None
        self.tick_count = 0
        
    def add_tick(self, timestamp: datetime, price: float, 
                 volume: Optional[int] = None) -> bool:
        """
        Add a tick and update the corresponding minute candle.
        
        Returns:
            True if a new candle was created
        """
        # Normalize to minute
        minute_key = timestamp.replace(second=0, microsecond=0)
        new_candle = False
        
        if minute_key not in self.candles:
            # New candle
            self.candles[minute_key] = {
                "open": price,
                "high": price,
                "low": price,
                "close": price,
                "volume": volume or 0,
                "tick_count": 1,
                "timestamp": minute_key
            }
            new_candle = True
            
            # Maintain window size
            while len(self.candles) > self.window:
                self.candles.popitem(last=False)
            
            logger.debug(f"New candle at {minute_key}: {price}")
        else:
            # Update existing candle
            candle = self.candles[minute_key]
            candle["high"] = max(candle["high"], price)
            candle["low"] = min(candle["low"], price)
            candle["close"] = price
            candle["volume"] += volume or 0
            candle["tick_count"] += 1
        
        self.tick_count += 1
        self.current_minute = minute_key
        
        return new_candle
    
    def get_dataframe(self, exclude_current: bool = False) -> pd.DataFrame:
        """
        Get OHLC data as DataFrame.
        
        Args:
            exclude_current: If True, exclude the current (incomplete) candle
        """
        if not self.candles:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        
        data = list(self.candles.values())
        
        if exclude_current and data:
            # Remove last candle if it's the current minute
            now_minute = datetime.now(timezone.utc).replace(second=0, microsecond=0)
            if data[-1]["timestamp"] >= now_minute:
                data = data[:-1]
        
        if not data:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        
        df = pd.DataFrame(data)
        df.set_index("timestamp", inplace=True)
        df = df[["open", "high", "low", "close", "volume", "tick_count"]]
        
        return df
    
    def get_completed_candles(self) -> pd.DataFrame:
        """Get only completed candles (excluding current minute)."""
        return self.get_dataframe(exclude_current=True)
    
    def get_latest_candle(self) -> Optional[Dict]:
        """Get the most recent candle."""
        if not self.candles:
            return None
        return list(self.candles.values())[-1]
    
    def clear(self):
        """Clear all candles."""
        self.candles.clear()
        self.tick_count = 0
        logger.info("OHLC builder cleared")
