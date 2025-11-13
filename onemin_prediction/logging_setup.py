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
    heartbeat_cooldown_sec: float = 0.0,  # file rate-limit heartbeat
    heartbeat_cooldown_console_sec: float = 0.0  # NEW: console rate-limit (optional)
):
    """
    Install console and rotating file handlers with independent color toggles.
    - Root logger is forced to DEBUG to always pass logs; handler levels control visibility.
    - Optional RateLimitFilter can be applied to file and/or console to dedupe repeating messages.
    """
    logger = logging.getLogger()
    # Always allow DEBUG; handlers decide what shows
    logger.setLevel(logging.DEBUG)

    # Remove existing handlers to avoid duplicates on reload
    for h in list(logger.handlers):
        logger.removeHandler(h)

    # Console handler (colors optional)
    try:
        if enable_colors_console:
            try:
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

        if heartbeat_cooldown_console_sec and heartbeat_cooldown_console_sec > 0:
            ch.addFilter(RateLimitFilter(cooldown_seconds=float(heartbeat_cooldown_console_sec)))

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
    if heartbeat_cooldown_sec and heartbeat_cooldown_sec > 0:
        fh.addFilter(RateLimitFilter(cooldown_seconds=float(heartbeat_cooldown_sec)))
    logger.addHandler(fh)

    # Reduce third-party verbosity defaults (overridable later)
    logging.getLogger("websockets").setLevel(logging.INFO)
    logging.getLogger("asyncio").setLevel(logging.INFO)

    return logger


def update_global_verbosity(high: bool = True) -> None:
    """
    Dynamically raise/lower handler levels across all handlers.
    high=True -> DEBUG, high=False -> INFO
    """
    root = logging.getLogger()
    new = logging.DEBUG if high else logging.INFO
    for h in root.handlers:
        try:
            h.setLevel(new)
        except Exception:
            continue
    root.info("Global verbosity updated → %s", logging.getLevelName(new))
