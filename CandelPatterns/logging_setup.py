"""
Logging configuration module.
Sets up file and console logging with proper formatting.
"""
import logging
import sys
from datetime import datetime

def setup_logging(logfile="logs/nifty_alerts.log", console_level=logging.INFO):
    """
    Configure logging for the application.
    
    Args:
        logfile: Path to log file
        console_level: Logging level for console output
    """
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # File handler
    file_handler = logging.FileHandler(logfile)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    console_handler.setFormatter(formatter)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    logging.info(f"Logging initialized - Log file: {logfile}")
