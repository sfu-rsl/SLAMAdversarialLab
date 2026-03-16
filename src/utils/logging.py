"""Centralized logging configuration for SLAMAdverserialLab."""

import logging
import os
import sys
from typing import Optional


# ANSI color codes
class Colors:
    """ANSI escape codes for terminal colors."""

    RESET = "\033[0m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    BOLD = "\033[1m"
    # Bright variants
    BRIGHT_RED = "\033[91m"
    BRIGHT_YELLOW = "\033[93m"


class ColoredFormatter(logging.Formatter):
    """Formatter that adds colors to log levels."""

    LEVEL_COLORS = {
        logging.DEBUG: Colors.CYAN,
        logging.INFO: Colors.GREEN,
        logging.WARNING: Colors.YELLOW,
        logging.ERROR: Colors.RED,
        logging.CRITICAL: Colors.BOLD + Colors.BRIGHT_RED,
    }

    def __init__(self, fmt: str, datefmt: str = None, use_colors: bool = True):
        super().__init__(fmt, datefmt)
        self.use_colors = use_colors

    def format(self, record: logging.LogRecord) -> str:
        if self.use_colors and sys.stdout.isatty():
            # Color the level name
            color = self.LEVEL_COLORS.get(record.levelno, Colors.RESET)
            record.levelname = f"{color}{record.levelname}{Colors.RESET}"
            # Color the entire message for ERROR and CRITICAL
            if record.levelno >= logging.ERROR:
                record.msg = f"{color}{record.msg}{Colors.RESET}"
        return super().format(record)


def _package_root(name: str) -> str:
    """Return top-level logger namespace for a module logger name."""
    if not name:
        return "slamadverseriallab"
    return name.split(".", 1)[0]


def _resolve_level(level: Optional[str]) -> int:
    """Parse configured log level with INFO fallback."""
    raw_level = level or os.environ.get("SLAMWEATHER_LOG_LEVEL", "INFO")
    try:
        return getattr(logging, raw_level.upper())
    except AttributeError:
        return logging.INFO


def _configure_root_logger(root_logger: logging.Logger, level: int) -> None:
    """Configure handlers on the root package logger exactly once."""
    if root_logger.handlers:
        return

    root_logger.setLevel(level)
    # Prevent double emission through ancestor/root loggers.
    root_logger.propagate = False

    console_formatter = ColoredFormatter(
        "[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        use_colors=True,
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    log_file = os.environ.get("SLAMWEATHER_LOG_FILE")
    if log_file:
        try:
            log_dir = os.path.dirname(log_file)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir, exist_ok=True)

            file_formatter = logging.Formatter(
                "[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(level)
            file_handler.setFormatter(file_formatter)
            root_logger.addHandler(file_handler)
            root_logger.info(f"Logging to file: {log_file}")
        except Exception as e:
            root_logger.error(f"Failed to create log file handler: {e}")


def get_logger(name: str, level: Optional[str] = None) -> logging.Logger:
    """
    Get a configured logger instance.

    Args:
        name: Logger name (usually __name__ from the calling module)
        level: Optional log level override (DEBUG, INFO, WARNING, ERROR, CRITICAL)

    Returns:
        Configured logger instance

    Environment Variables:
        SLAMWEATHER_LOG_FILE: If set, logs will also be written to this file
        SLAMWEATHER_LOG_LEVEL: Override default log level (default: INFO)
    """
    logger = logging.getLogger(name)
    root_name = _package_root(name)
    root_logger = logging.getLogger(root_name)

    resolved_level = _resolve_level(level)
    _configure_root_logger(root_logger, resolved_level)

    # Child loggers should not have handlers; they propagate to package root.
    if name != root_name:
        if logger.handlers:
            logger.handlers.clear()
        logger.propagate = True
        logger.setLevel(_resolve_level(level) if level is not None else logging.NOTSET)
        return logger

    # Root logger path.
    logger.setLevel(resolved_level)
    for handler in logger.handlers:
        handler.setLevel(resolved_level)
    return logger
