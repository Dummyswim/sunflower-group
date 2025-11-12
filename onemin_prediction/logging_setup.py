# logging_setup.py
import logging
import os
from logging.handlers import RotatingFileHandler

class RateLimitFilter(logging.Filter):
    """
    Simple cooldown-based rate limiter per logger+level+msg signature.
    Allows one log per key per cooldown_seconds.
    """
    def __init__(self, cooldown_seconds: float = 30.0):
        super().__init__()
        self.cooldown = float(cooldown_seconds)
        self._last = {}

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            key = (record.name, record.levelno, getattr(record, "msg", ""))
            now = getattr(record, "_rl_now", None)
            if now is None:
                # lazy import to avoid global deps
                import time
                now = time.time()
            last = self._last.get(key, 0.0)
            if now - last >= self.cooldown:
                self._last[key] = now
                return True
            # Suppress duplicates within cooldown window
            return False
        except Exception:
            # fail-open for safety
            return True


def setup_logging2(
    logfile: str,
    console_level: int = logging.INFO,
    file_level: int = logging.INFO,
    enable_colors_console: bool = True,
    enable_colors_file: bool = False,
    max_bytes: int = 10_485_760,  # 10MB
    backup_count: int = 5,
    debug_sample_n: int = 1,
    heartbeat_cooldown_sec: float = 0.0  # set >0 to de-duplicate heartbeats
):
    """
    Install console and rotating file handlers with independent color toggles
    and UTF-8 file encoding. Keeps console colors; removes ANSI in files.
    """
    logger = logging.getLogger()
    logger.setLevel(min(console_level, file_level))

    # Remove existing handlers to avoid duplicates on reload
    for h in list(logger.handlers):
        logger.removeHandler(h)

    # Console handler (optional colors)
    try:
        if enable_colors_console:
            try:
                # Use colorlog if present for pretty console output
                import colorlog  # type: ignore
                ch = logging.StreamHandler()
                ch.setLevel(console_level)
                fmt = "%(log_color)s%(asctime)s | %(levelname)s | %(name)s | [%(funcName)s:%(lineno)d] | %(message)s"
                datefmt = "%Y-%m-%d %H:%M:%S"
                ch.setFormatter(colorlog.ColoredFormatter(
                    fmt=fmt,
                    datefmt=datefmt,
                    log_colors={
                        "DEBUG": "cyan",
                        "INFO": "green",
                        "WARNING": "yellow",
                        "ERROR": "red",
                        "CRITICAL": "bold_red",
                    },
                ))
            except Exception:
                # Fallback: plain console
                ch = logging.StreamHandler()
                ch.setLevel(console_level)
                ch.setFormatter(logging.Formatter(
                    "%(asctime)s | %(levelname)s | %(name)s | [%(funcName)s:%(lineno)d] | %(message)s"
                ))
        else:
            ch = logging.StreamHandler()
            ch.setLevel(console_level)
            ch.setFormatter(logging.Formatter(
                "%(asctime)s | %(levelname)s | %(name)s | [%(funcName)s:%(lineno)d] | %(message)s"
            ))
        logger.addHandler(ch)
    except Exception:
        pass

    # File handler (UTF-8, no ANSI)
    try:
        os.makedirs(os.path.dirname(logfile) or ".", exist_ok=True)
    except Exception:
        pass

    fh = RotatingFileHandler(
        logfile,
        mode="a",
        maxBytes=int(max_bytes),
        backupCount=int(backup_count),
        encoding="utf-8",
        delay=False
    )
    fh.setLevel(file_level)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | [%(funcName)s:%(lineno)d] | %(message)s"
    ))

    # Optional: rate-limit heartbeats globally in file to keep files tidy
    if heartbeat_cooldown_sec and heartbeat_cooldown_sec > 0:
        fh.addFilter(RateLimitFilter(cooldown_seconds=float(heartbeat_cooldown_sec)))

    logger.addHandler(fh)

    # Reduce third-party verbosity defaults (you can still override later)
    logging.getLogger("websockets").setLevel(logging.INFO)
    logging.getLogger("asyncio").setLevel(logging.INFO)

    return logger
