"""
OHLC candle builder from tick data with configurable timeframe.
"""
import logging
from datetime import datetime, timedelta
from typing import Optional, Tuple
import pandas as pd

logger = logging.getLogger(__name__)

class OHLCBuilder:
    """
    Build OHLC candles from tick data with configurable timeframe.
    """
    
    def __init__(self, window: int = 100, timeframe_minutes: int = 5):
        """
        Initialize OHLC builder.
        
        Args:
            window: Maximum number of candles to keep
            timeframe_minutes: Candlestick timeframe in minutes (default: 5)
        """
        self.window = window
        self.timeframe_minutes = timeframe_minutes
        self.candles = []
        self.current_candle = None
        self.tick_count = 0
        
        logger.info(f"OHLCBuilder initialized with {timeframe_minutes}-minute candles")
    
    def add_tick(self, timestamp: datetime, price: float, volume: int = 0) -> bool:
        """
        Add a tick and check if a new candle is created.
        
        Returns:
            True if a new candle was started (previous completed)
        """
        self.tick_count += 1
        
        # Round timestamp to the timeframe boundary
        candle_time = self._get_candle_time(timestamp)
        
        # Check if we need to start a new candle
        if self.current_candle is None or self.current_candle['time'] != candle_time:
            # Complete previous candle if exists
            if self.current_candle is not None:
                self._complete_candle()
            
            # Start new candle
            self.current_candle = {
                'time': candle_time,
                'open': price,
                'high': price,
                'low': price,
                'close': price,
                'volume': volume,
                'ticks': 1
            }
            
            logger.debug(f"New {self.timeframe_minutes}-min candle at {candle_time}: {price}")
            return True
        else:
            # Update current candle
            self.current_candle['high'] = max(self.current_candle['high'], price)
            self.current_candle['low'] = min(self.current_candle['low'], price)
            self.current_candle['close'] = price
            self.current_candle['volume'] += volume
            self.current_candle['ticks'] += 1
            
            return False
    
    def _get_candle_time(self, timestamp: datetime) -> datetime:
        """
        Round timestamp down to the nearest timeframe boundary.
        
        For 5-minute candles:
        - 10:21:45 -> 10:20:00
        - 10:24:59 -> 10:20:00
        - 10:25:00 -> 10:25:00
        """
        # Remove seconds and microseconds
        timestamp = timestamp.replace(second=0, microsecond=0)
        
        # Round down to nearest timeframe boundary
        minutes = timestamp.minute
        rounded_minutes = (minutes // self.timeframe_minutes) * self.timeframe_minutes
        
        return timestamp.replace(minute=rounded_minutes)
    
    def _complete_candle(self):
        """Complete and store the current candle."""
        if self.current_candle:
            self.candles.append(self.current_candle.copy())
            
            # Maintain window size
            if len(self.candles) > self.window:
                self.candles.pop(0)
            
            logger.debug(f"Completed {self.timeframe_minutes}-min candle: "
                       f"O={self.current_candle['open']:.2f} "
                       f"H={self.current_candle['high']:.2f} "
                       f"L={self.current_candle['low']:.2f} "
                       f"C={self.current_candle['close']:.2f} "
                       f"V={self.current_candle['volume']} "
                       f"T={self.current_candle['ticks']}")
    
    def get_completed_candles(self) -> pd.DataFrame:
        """
        Get all completed candles as DataFrame.
        
        Returns:
            DataFrame with OHLCV data
        """
        if not self.candles:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.candles)
        df.set_index('time', inplace=True)
        
        # Ensure required columns
        for col in ['open', 'high', 'low', 'close']:
            if col in df.columns:
                df[col] = df[col].astype(float)
        
        if 'volume' not in df.columns:
            df['volume'] = 0
        
        return df[['open', 'high', 'low', 'close', 'volume']]
    
    def get_latest_candle(self) -> Optional[dict]:
        """Get the most recent completed candle."""
        return self.candles[-1] if self.candles else None
    
    def get_current_candle(self) -> Optional[dict]:
        """Get the current (incomplete) candle."""
        return self.current_candle.copy() if self.current_candle else None
    
    def reset(self):
        """Reset the builder."""
        self.candles = []
        self.current_candle = None
        self.tick_count = 0
        logger.info("OHLCBuilder reset")
