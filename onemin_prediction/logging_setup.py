"""
Enhanced logging configuration with IST timezone support.
Provides colored console output, file rotation, and performance tracking.
"""
import logging
import logging.handlers
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

# IST timezone constant
IST = timezone(timedelta(hours=5, minutes=30))


class ISTFormatter(logging.Formatter):
    """Custom formatter to display log timestamps in IST."""
    
    def formatTime(self, record, datefmt=None):
        """Override formatTime to use IST timezone."""
        dt = datetime.fromtimestamp(record.created, IST)
        if datefmt:
            return dt.strftime(datefmt)
        return dt.strftime('%Y-%m-%d %H:%M:%S')


class ColoredFormatter(ISTFormatter):
    """Formatter with ANSI color codes for console output."""
    
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
        'RESET': '\033[0m'       # Reset
    }
    
    def format(self, record):
        """Add color to levelname for console output."""
        levelname = record.levelname
        if levelname in self.COLORS:
            colored_levelname = f"{self.COLORS[levelname]}{levelname}{self.COLORS['RESET']}"
            record.levelname = colored_levelname
        
        formatted = super().format(record)
        record.levelname = levelname  # Reset for other handlers
        return formatted


def setup_logging(
    logfile: str = "logs/unified_trading.log",
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG,
    enable_colors: bool = True,
    max_bytes: int = 10485760,  # 10MB
    backup_count: int = 5
) -> None:
    """
    Configure comprehensive logging with IST timezone.
    
    Args:
        logfile: Path to log file
        console_level: Logging level for console (default: INFO)
        file_level: Logging level for file (default: DEBUG)
        enable_colors: Enable colored console output (default: True)
        max_bytes: Maximum log file size before rotation (default: 10MB)
        backup_count: Number of backup files to keep (default: 5)
    """
    try:
        # Create log directory
        log_dir = Path(logfile).parent
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)
        
        # Remove existing handlers to avoid duplicates
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        # ========== CONSOLE HANDLER ==========
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(console_level)
        

        if enable_colors:
            console_formatter = ColoredFormatter(
                '%(asctime)s | %(levelname)-8s | %(name)s | [%(funcName)s:%(lineno)d] | %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
        else:
            console_formatter = ISTFormatter(
                '%(asctime)s | %(levelname)-8s | %(name)s | [%(funcName)s:%(lineno)d] | %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )


        
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)
        
        # ========== FILE HANDLER ==========
        file_handler = logging.handlers.RotatingFileHandler(
            logfile,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(file_level)
        
        file_formatter = ISTFormatter(
            '%(asctime)s | %(levelname)-8s | %(name)s | [%(filename)s:%(funcName)s:%(lineno)d] | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
        
        # ========== SUPPRESS NOISY LIBRARIES ==========
        logging.getLogger('websockets').setLevel(logging.WARNING)
        logging.getLogger('urllib3').setLevel(logging.WARNING)
        logging.getLogger('urllib3.connectionpool').setLevel(logging.INFO)
        logging.getLogger('matplotlib').setLevel(logging.WARNING)
        logging.getLogger('asyncio').setLevel(logging.INFO)
        
        

        # ========== ENABLE DEBUG FOR NEW MODULES (OPTIONAL) ==========
        logging.getLogger('online_trainer').setLevel(logging.DEBUG)
        logging.getLogger('feature_pipeline').setLevel(logging.DEBUG)
        logging.getLogger('model_pipeline').setLevel(logging.INFO)

        
        # ========== LOG INITIALIZATION ==========
        logging.info("=" * 60)
        logging.info("LOGGING SYSTEM INITIALIZED (IST TIMEZONE)")
        logging.info(f"Log file: {logfile}")
        logging.info(f"Console level: {logging.getLevelName(console_level)}")
        logging.info(f"File level: {logging.getLevelName(file_level)}")
        logging.info(f"Colors: {'Enabled' if enable_colors else 'Disabled'}")
        logging.info(f"Timezone: IST (UTC+5:30)")
        logging.info("=" * 60)
        
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
        name: Logger name (typically __name__)
    
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


def log_span(title: str, logger: Optional[logging.Logger] = None):
    """
    Log a visual separator with title.
    
    Args:
        title: Title to display
        logger: Logger instance (uses root if None)
    """
    if logger is None:
        logger = logging.getLogger()
    
    logger.info("=" * 60)
    logger.info(title)
    logger.info("=" * 60)
