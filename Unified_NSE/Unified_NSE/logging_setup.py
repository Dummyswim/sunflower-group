"""
Enhanced logging configuration with comprehensive features - FIXED VERSION
Includes colored console output, performance metrics, and error tracking.
"""
import logging
import logging.handlers
import sys
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional, Dict, Any
import traceback


# Define IST timezone
IST = timezone(timedelta(hours=5, minutes=30))




class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for console output."""
    
    # Color codes
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green  
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
        'RESET': '\033[0m'       # Reset
    }
    
    def format(self, record):
        # Add color to level name for console
        if hasattr(record, 'no_color') and record.no_color:
            return super().format(record)
        
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = f"{self.COLORS[levelname]}{levelname}{self.COLORS['RESET']}"
        
        formatted = super().format(record)
        record.levelname = levelname  # Reset for file handler
        return formatted

    
class ISTFormatter(logging.Formatter):
    """Custom formatter to show log timestamps in IST."""
    
    def formatTime(self, record, datefmt=None):
        dt = datetime.fromtimestamp(record.created, IST)
        if datefmt:
            return dt.strftime(datefmt)
        return dt.isoformat()

class SafePerformanceFilter(logging.Filter):
    """Filter to safely add performance metrics to log records."""
    
    def __init__(self):
        super().__init__()
        self.start_time = datetime.now()
        self.log_counts = {
            'DEBUG': 0,
            'INFO': 0,
            'WARNING': 0,
            'ERROR': 0,
            'CRITICAL': 0
        }
    
    def filter(self, record):
        try:
            # Always add runtime to record to prevent KeyError
            record.runtime = (datetime.now() - self.start_time).total_seconds()
            
            # Track log counts
            if record.levelname in self.log_counts:
                self.log_counts[record.levelname] += 1
            
            # Add module context
            record.module_context = f"{record.funcName}:{record.lineno}"
            
        except Exception as e:
            # If anything fails, ensure runtime exists with default value
            record.runtime = 0.0
            record.module_context = f"{record.funcName}:{record.lineno}"
            
        return True

class StandardFilter(logging.Filter):
    """Filter for handlers that don't need runtime field."""
    
    def filter(self, record):
        # Add module_context if not present
        if not hasattr(record, 'module_context'):
            record.module_context = f"{record.funcName}:{record.lineno}"
        return True

class ErrorTracker:
    """Track and aggregate errors for reporting."""
    
    def __init__(self, max_errors: int = 100):
        self.errors = []
        self.max_errors = max_errors
        self.error_counts = {}
    
    def track_error(self, record: logging.LogRecord):
        """Track an error record."""
        if record.levelno >= logging.ERROR:
            error_info = {
                'timestamp': datetime.fromtimestamp(record.created, IST),
                'module': record.module,
                'function': record.funcName,
                'line': record.lineno,
                'message': record.getMessage(),
                'traceback': record.exc_info
            }
            
            self.errors.append(error_info)
            if len(self.errors) > self.max_errors:
                self.errors.pop(0)
            
            # Count by module
            module_key = f"{record.funcName}:{record.lineno}"
            self.error_counts[module_key] = self.error_counts.get(module_key, 0) + 1
    
    def get_summary(self) -> Dict[str, Any]:
        """Get error summary."""
        return {
            'total_errors': len(self.errors),
            'recent_errors': self.errors[-10:],
            'error_by_module': self.error_counts,
            'most_common': max(self.error_counts.items(), key=lambda x: x[1]) if self.error_counts else None
        }

# Global error tracker
error_tracker = ErrorTracker()

class ErrorTrackingHandler(logging.Handler):
    """Handler to track errors."""
    
    def emit(self, record):
        error_tracker.track_error(record)

def setup_logging(
    logfile: str = "logs/unified_trading.log",
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG,
    enable_colors: bool = True,
    max_bytes: int = 10485760,  # 10MB
    backup_count: int = 5,
    enable_performance: bool = True,
    enable_error_tracking: bool = True
) -> None:
    """
    Configure comprehensive logging for the application - FIXED VERSION.
    
    Args:
        logfile: Path to log file
        console_level: Logging level for console output
        file_level: Logging level for file output
        enable_colors: Enable colored console output
        max_bytes: Maximum size of log file before rotation
        backup_count: Number of backup files to keep
        enable_performance: Enable performance metrics
        enable_error_tracking: Enable error tracking
    """
    try:
        # Ensure log directory exists
        log_dir = Path(logfile).parent
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure root logger first
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)
        
        # Remove existing handlers
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        # ========== CONSOLE HANDLER ==========
        # Create formatters for console (without runtime field)
        
        console_formatter = ISTFormatter(
            '%(asctime)s - %(levelname)-8s - [%(module_context)s] - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        # if enable_colors:
        #     console_formatter = ColoredFormatter(
        #         '%(asctime)s - %(levelname)-8s - [%(module_context)s] - %(message)s',
        #         datefmt='%Y-%m-%d %H:%M:%S'
        #     )
        # else:
        #     console_formatter = ISTFormatter(
        #         '%(asctime)s - %(levelname)-8s - [%(module_context)s] - %(message)s',
        #         datefmt='%Y-%m-%d %H:%M:%S'
        #     )
        
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(console_level)
        console_handler.setFormatter(console_formatter)
        
        # Add standard filter to console handler
        console_filter = StandardFilter()
        console_handler.addFilter(console_filter)
        
        # ========== FILE HANDLER ==========
        # File formatter (without runtime field)
        file_formatter = ISTFormatter(
            '%(asctime)s - %(levelname)-8s - [%(filename)s:%(lineno)d] - '
            '%(funcName)s() - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        
        file_handler = logging.handlers.RotatingFileHandler(
            logfile,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(file_level)
        file_handler.setFormatter(file_formatter)
        
        # Add standard filter to file handler
        file_filter = StandardFilter()
        file_handler.addFilter(file_filter)
        
        # ========== PERFORMANCE HANDLER (if enabled) ==========
        perf_handler = None
        if enable_performance:
            perf_handler = logging.handlers.RotatingFileHandler(
                log_dir / "performance.log",
                maxBytes=max_bytes,
                backupCount=2
            )
            perf_handler.setLevel(logging.DEBUG)
            
            # Performance formatter with runtime field
            perf_formatter = logging.Formatter(
                '%(asctime)s,%(runtime).2f,%(levelname)s,%(module)s,%(funcName)s,%(message)s'
            )
            perf_handler.setFormatter(perf_formatter)
            
            # Add performance filter ONLY to performance handler
            perf_filter = SafePerformanceFilter()
            perf_handler.addFilter(perf_filter)
        
        # ========== ERROR TRACKING HANDLER ==========
        if enable_error_tracking:
            error_handler = ErrorTrackingHandler()
            error_handler.setLevel(logging.ERROR)
            root_logger.addHandler(error_handler)
        
        # ========== ADD HANDLERS TO ROOT LOGGER ==========
        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)
        
        # Only add performance handler if enabled
        if enable_performance and perf_handler:
            root_logger.addHandler(perf_handler)
        
        # ========== SUPPRESS NOISY LIBRARIES ==========
        logging.getLogger('websockets').setLevel(logging.WARNING)
        logging.getLogger('urllib3').setLevel(logging.WARNING)
        logging.getLogger('matplotlib').setLevel(logging.WARNING)
        logging.getLogger("urllib3.connectionpool").setLevel(logging.INFO)

        
        # ========== LOG INITIALIZATION SUCCESS ==========
        logging.info("=" * 60)
        logging.info("LOGGING SYSTEM INITIALIZED")
        logging.info(f"Log file: {logfile}")
        logging.info(f"Console level: {logging.getLevelName(console_level)}")
        logging.info(f"File level: {logging.getLevelName(file_level)}")
        logging.info(f"Colors: {'Enabled' if enable_colors else 'Disabled'}")
        logging.info(f"Performance tracking: {'Enabled' if enable_performance else 'Disabled'}")
        logging.info(f"Error tracking: {'Enabled' if enable_error_tracking else 'Disabled'}")
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
        name: Logger name (usually __name__)
    
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


# Performance logging helpers
def log_performance(logger: logging.Logger, operation: str, duration: float, 
                    details: Optional[Dict] = None):
    """Log performance metrics - SAFE VERSION."""
    try:
        msg = f"PERF: {operation} took {duration:.3f}s"
        if details:
            msg += f" | {json.dumps(details)}"
        
        # Create a LogRecord manually to ensure runtime field exists
        record = logger.makeRecord(
            logger.name,
            logging.INFO,
            "(unknown file)",
            0,
            msg,
            (),
            None
        )
        
        # Ensure runtime field exists
        if not hasattr(record, 'runtime'):
            record.runtime = 0.0
        
        logger.handle(record)
    except Exception as e:
        # Fallback to simple logging if performance logging fails
        logger.info(f"PERF: {operation} took {duration:.3f}s")
        logger.info(f"x" * 60)



# Add this to logging_setup.py, after logger is set up (bottom of file is fine)
def log_span(title: str):
    logger = logging.getLogger()  # Use root logger for span logs
    logger.info("=" * 60)
    logger.info(title)
    logger.info("=" * 60)
