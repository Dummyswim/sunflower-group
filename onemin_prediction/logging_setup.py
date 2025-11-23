"""
Logging setup module for AR-NMS system.
Provides structured logging with IST timezone, rate limiting, and dynamic level control.
All color code functionality has been removed for clean file output.
"""



import logging
import logging.handlers
import os
import sys
import time
import json
from datetime import datetime, timezone, timedelta

from pathlib import Path
from typing import Optional, Dict, Callable, Any
from threading import Lock, Thread



# IST timezone constant
IST = timezone(timedelta(hours=5, minutes=30))

# Global state for log_every rate limiting
_log_every_state: Dict[str, float] = {}
_log_every_lock = Lock()

# Global state for dynamic level watcher
_level_watcher_task: Optional[Thread] = None
_level_watcher_stop = False


class ISTFormatter(logging.Formatter):
    """
    Custom formatter that uses IST timezone for timestamps.
    No color codes - plain text output only.
    """
    
    def __init__(self, fmt=None, datefmt=None, style='%'):
        super().__init__(fmt=fmt, datefmt=datefmt, style=style)
    
    def formatTime(self, record, datefmt=None):
        """Override to use IST timezone."""
        dt = datetime.fromtimestamp(record.created, tz=IST)
        if datefmt:
            return dt.strftime(datefmt)
        return dt.isoformat()
    
    def format(self, record):
        """Format the log record without any color codes."""
        # Store original values
        original_levelname = record.levelname
        
        # Format the message
        result = super().format(record)
        
        # Restore original values
        record.levelname = original_levelname
        
        return result


def log_every(
    key: str,
    interval_sec: float,
    log_func: Callable,
    message: str,
    *args,
    **kwargs
) -> None:
    """
    Rate-limited logging: only logs if interval_sec has passed since last call with this key.
    
    Args:
        key: Unique identifier for this log message
        interval_sec: Minimum seconds between logs for this key
        log_func: Logging function to call (e.g., logger.info)
        message: Log message format string
        *args: Positional arguments for message formatting
        **kwargs: Keyword arguments for message formatting
    
    Example:
        log_every("my-key", 60, logger.info, "Status: %s", status_value)
    """
    try:
        now = time.time()
        with _log_every_lock:
            last_time = _log_every_state.get(key, 0.0)
            if (now - last_time) >= interval_sec:
                _log_every_state[key] = now
                should_log = True
            else:
                should_log = False
        
        if should_log:
            log_func(message, *args, **kwargs)
    except Exception as e:
        # Fallback: always log if rate limiting fails
        try:
            log_func(message, *args, **kwargs)
        except Exception:
            pass


def setup_logging2(
    logfile: str = "logs/unified_trading.log",
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG,
    enable_colors_console: bool = False,  # Ignored - kept for API compatibility
    enable_colors_file: bool = False,      # Ignored - kept for API compatibility
    max_bytes: int = 10_485_760,
    backup_count: int = 5,
    heartbeat_cooldown_sec: float = 30.0,
    heartbeat_cooldown_console_sec: float = 0.0,
    telegram_alerts: bool = False,
    telegram_min_level: int = logging.ERROR
) -> None:
    """
    Set up logging with IST timestamps, rotating file handler, and optional rate limiting.
    
    Args:
        logfile: Path to log file
        console_level: Logging level for console output
        file_level: Logging level for file output
        enable_colors_console: Ignored (kept for compatibility)
        enable_colors_file: Ignored (kept for compatibility)
        max_bytes: Max bytes per log file before rotation
        backup_count: Number of backup log files to keep
        heartbeat_cooldown_sec: Rate limit for file handler (0 = no limit)
        heartbeat_cooldown_console_sec: Rate limit for console handler (0 = no limit)
        telegram_alerts: Enable telegram alerts (placeholder)
        telegram_min_level: Minimum level for telegram alerts
    """
    
    # Create logs directory if needed
    try:
        Path(logfile).parent.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"Warning: Could not create log directory: {e}", file=sys.stderr)
    
    # Get root logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)  # Capture everything; handlers filter
    
    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Define format strings (no color codes)
    detailed_format = (
        '%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s'
    )
    simple_format = (
        '%(asctime)s | %(levelname)-8s | %(message)s'
    )
    
    date_format = '%Y-%m-%d %H:%M:%S'
    
    # Console handler (plain text)
    try:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(console_level)
        console_formatter = ISTFormatter(
            fmt=simple_format,
            datefmt=date_format
        )
        console_handler.setFormatter(console_formatter)
        
        # Add rate limiting filter if requested
        if heartbeat_cooldown_console_sec > 0:
            console_handler.addFilter(
                RateLimitFilter(cooldown_sec=heartbeat_cooldown_console_sec)
            )
        
        logger.addHandler(console_handler)
    except Exception as e:
        print(f"Warning: Could not set up console handler: {e}", file=sys.stderr)
    
    # File handler (rotating, plain text)
    try:
        file_handler = logging.handlers.RotatingFileHandler(
            logfile,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(file_level)
        file_formatter = ISTFormatter(
            fmt=detailed_format,
            datefmt=date_format
        )
        file_handler.setFormatter(file_formatter)
        
        # Add rate limiting filter if requested
        if heartbeat_cooldown_sec > 0:
            file_handler.addFilter(
                RateLimitFilter(cooldown_sec=heartbeat_cooldown_sec)
            )
        
        logger.addHandler(file_handler)
    except Exception as e:
        print(f"Warning: Could not set up file handler: {e}", file=sys.stderr)
    
    # Log successful initialization
    logger.info("Logging system initialized (IST timezone, no color codes)")
    logger.info(f"Console level: {logging.getLevelName(console_level)}, "
                f"File level: {logging.getLevelName(file_level)}")
    logger.info(f"Log file: {logfile} (max {max_bytes:,} bytes, "
                f"{backup_count} backups)")


class RateLimitFilter(logging.Filter):
    """
    Logging filter that rate-limits repetitive messages.
    Deduplicates identical messages within a cooldown window.
    """
    
    def __init__(self, cooldown_sec: float = 30.0):
        super().__init__()
        self.cooldown_sec = cooldown_sec
        self._last_log_times: Dict[str, float] = {}
        self._lock = Lock()
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Return True if record should be logged."""
        try:
            # Create a key from the log message
            key = f"{record.name}:{record.levelno}:{record.getMessage()}"
            now = time.time()
            
            with self._lock:
                last_time = self._last_log_times.get(key, 0.0)
                if (now - last_time) >= self.cooldown_sec:
                    self._last_log_times[key] = now
                    return True
                return False
        except Exception:
            # On error, allow the log through
            return True


def start_dynamic_level_watcher(
    config_path: str = "logs/log_level.json",
    poll_sec: float = 2.0
) -> None:
    """
    Start a background thread that watches a JSON config file for log level changes.
    
    Config file format:
    {
        "root": "INFO",
        "websockets": "WARNING",
        "asyncio": "WARNING"
    }
    
    Args:
        config_path: Path to JSON config file
        poll_sec: Polling interval in seconds
    """
    global _level_watcher_task, _level_watcher_stop
    
    if _level_watcher_task is not None:
        return  # Already running
    
    _level_watcher_stop = False
    
    def _watcher_loop():
        """Background thread that polls config file."""
        last_mtime = 0.0
        
        while not _level_watcher_stop:
            try:
                # Check if file exists and has been modified
                if not os.path.exists(config_path):
                    time.sleep(poll_sec)
                    continue
                
                mtime = os.path.getmtime(config_path)
                if mtime <= last_mtime:
                    time.sleep(poll_sec)
                    continue
                
                last_mtime = mtime
                
                # Read and apply config
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                
                for logger_name, level_val in config.items():
                    try:
                        # Support nested mapping under a "loggers" key or similar
                        if isinstance(level_val, dict):
                            for sub_logger, sub_level in level_val.items():
                                if not isinstance(sub_level, str):
                                    continue
                                lvl = getattr(logging, sub_level.upper(), None)
                                if not isinstance(lvl, int):
                                    continue
                                (logging.getLogger(sub_logger) if sub_logger != "root" else logging.getLogger()).setLevel(lvl)
                                logging.info("[LEVEL] Updated %s to %s", sub_logger, sub_level)
                            continue

                        # Only strings are valid top-level values
                        if not isinstance(level_val, str):
                            logging.debug("[LEVEL] Skipping non-string level for %s", logger_name)
                            continue
                        lvl = getattr(logging, level_val.upper(), None)
                        if not isinstance(lvl, int):
                            logging.warning("[LEVEL] Invalid level name for %s: %r", logger_name, level_val)
                            continue
                        (logging.getLogger(logger_name) if logger_name != "root" else logging.getLogger()).setLevel(lvl)
                        logging.info("[LEVEL] Updated %s to %s", logger_name, level_val)
                    except Exception as e:
                        logging.warning("[LEVEL] Failed to update %s: %s", logger_name, e)
                
            except Exception as e:
                logging.debug(f"[LEVEL] Watcher error: {e}")
            
            time.sleep(poll_sec)
    
    _level_watcher_task = Thread(target=_watcher_loop, daemon=True)
    _level_watcher_task.start()


def stop_dynamic_level_watcher() -> None:
    """Stop the dynamic level watcher thread."""
    global _level_watcher_task, _level_watcher_stop
    
    if _level_watcher_task is None:
        return
    
    _level_watcher_stop = True
    try:
        _level_watcher_task.join(timeout=5.0)
    except Exception:
        pass
    
    _level_watcher_task = None


# Utility functions for common logging patterns

def log_exception(logger: logging.Logger, message: str, exc: Exception) -> None:
    """
    Log an exception with full traceback.
    
    Args:
        logger: Logger instance
        message: Descriptive message
        exc: Exception to log
    """
    logger.error(f"{message}: {exc}", exc_info=True)


def log_dict(logger: logging.Logger, level: int, prefix: str, data: Dict[str, Any]) -> None:
    """
    Log a dictionary in a readable format.
    
    Args:
        logger: Logger instance
        level: Logging level (e.g., logging.INFO)
        prefix: Message prefix
        data: Dictionary to log
    """
    try:
        formatted = json.dumps(data, indent=2)
        logger.log(level, f"{prefix}:\n{formatted}")
    except Exception:
        logger.log(level, f"{prefix}: {data}")


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the given name.
    
    Args:
        name: Logger name (typically __name__)
    
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


# Module-level logger for this file
_module_logger = logging.getLogger(__name__)
