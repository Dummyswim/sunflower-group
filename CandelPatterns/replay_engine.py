"""
CSV replay engine for backtesting strategies.
"""
import logging
import pandas as pd
from datetime import datetime
from typing import Callable, Optional, Dict

logger = logging.getLogger(__name__)

class ReplayEngine:
    """
    Replay historical data from CSV for backtesting.
    """
    
    def __init__(self, csv_path: str):
        """
        Initialize replay engine with CSV data.
        
        CSV should have columns: time, open, high, low, close, volume (optional)
        """
        self.csv_path = csv_path
        self.data = None
        self.current_index = 0
        self.load_data()
        
    def load_data(self):
        """Load and validate CSV data."""
        try:
            self.data = pd.read_csv(self.csv_path)
            
            # Validate columns
            required = ['time', 'open', 'high', 'low', 'close']
            missing = [col for col in required if col not in self.data.columns]
            if missing:
                raise ValueError(f"Missing required columns: {missing}")
            
            # Parse time column
            try:
                self.data['time'] = pd.to_datetime(self.data['time'], utc=True)
            except:
                # Try parsing as epoch seconds
                self.data['time'] = pd.to_datetime(self.data['time'], unit='s', utc=True)
            
            # Add volume if missing
            if 'volume' not in self.data.columns:
                self.data['volume'] = 0
            
            # Sort by time
            self.data.sort_values('time', inplace=True)
            self.data.reset_index(drop=True, inplace=True)
            
            logger.info(f"Loaded {len(self.data)} candles from {self.csv_path}")
            logger.info(f"Date range: {self.data['time'].min()} to {self.data['time'].max()}")
            
        except Exception as e:
            logger.error(f"Failed to load CSV: {e}")
            raise
    
    def replay(self, callback: Callable, speed: float = 0.0):
        """
        Replay the data through a callback function.
        
        Args:
            callback: Function to call for each candle (receives row as dict)
            speed: Delay between candles in seconds (0 for instant)
        """
        logger.info("Starting replay...")
        
        for idx, row in self.data.iterrows():
            self.current_index = idx
            
            # Convert row to dict
            candle = {
                'timestamp': row['time'].to_pydatetime(),
                'open': float(row['open']),
                'high': float(row['high']),
                'low': float(row['low']),
                'close': float(row['close']),
                'volume': int(row.get('volume', 0))
            }
            
            # Call the callback
            try:
                callback(candle)
            except Exception as e:
                logger.error(f"Callback error at index {idx}: {e}")
            
            # Progress update every 100 candles
            if idx % 100 == 0 and idx > 0:
                progress = (idx / len(self.data)) * 100
                logger.info(f"Replay progress: {progress:.1f}% ({idx}/{len(self.data)})")
            
            # Optional delay
            if speed > 0:
                import time
                time.sleep(speed)
        
        logger.info("Replay completed")
    
    def get_statistics(self) -> Dict:
        """Get statistics about the loaded data."""
        if self.data is None or self.data.empty:
            return {}
        
        return {
            "total_candles": len(self.data),
            "date_range": f"{self.data['time'].min()} to {self.data['time'].max()}",
            "price_range": f"{self.data['low'].min():.2f} to {self.data['high'].max():.2f}",
            "avg_volume": self.data['volume'].mean(),
            "total_volume": self.data['volume'].sum()
        }
