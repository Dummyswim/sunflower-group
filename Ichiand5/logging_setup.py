"""
Logging configuration module with enhanced error handling.
Sets up file and console logging with proper formatting.
"""
import logging
import logging.handlers
import sys
import os
from datetime import datetime
from pathlib import Path

def setup_logging(logfile="logs/nifty_alerts.log", console_level=logging.INFO):
    """
    Configure logging for the application with enhanced error handling.
    
    Args:
        logfile: Path to log file
        console_level: Logging level for console output
    """
    try:
        # Ensure log directory exists
        log_dir = Path(logfile).parent
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # File handler with rotation
        file_handler = logging.handlers.RotatingFileHandler(
            logfile,
            maxBytes=10485760,  # 10MB
            backupCount=5
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(console_level)
        console_handler.setFormatter(formatter)
        
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)
        
        # Remove existing handlers to avoid duplicates
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
            
        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)
        
        # Log successful initialization
        logging.info(f"Logging initialized - Log file: {logfile}, Console level: {logging.getLevelName(console_level)}")
        
    except PermissionError as e:
        print(f"ERROR: Permission denied creating log file: {e}", file=sys.stderr)
        raise
    except Exception as e:
        print(f"ERROR: Failed to setup logging: {e}", file=sys.stderr)
        raise

def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the given name.
    
    Args:
        name: Logger name (usually __name__)
    
    Returns:
        Logger instance
    """
    return logging.getLogger(name)
