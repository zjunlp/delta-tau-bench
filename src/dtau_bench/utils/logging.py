"""
Unified logging configuration for dtau_bench.

Provides:
- setup_logging(): Configure root logger with console + file handlers
- get_logger(): Get a named logger under the 'dtau_bench' hierarchy
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Union


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

VERBOSE_FORMAT = (
    "%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d | %(message)s"
)


# ─────────────────────────────────────────────────────────────────────────────
# Setup
# ─────────────────────────────────────────────────────────────────────────────

def setup_logging(
    level: Union[str, int] = "INFO",
    log_file: Optional[Union[str, Path]] = None,
    verbose: bool = False,
    quiet: bool = False,
) -> logging.Logger:
    """Configure the root 'dtau_bench' logger.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
            Overridden by verbose (→DEBUG) or quiet (→WARNING).
        log_file: Optional path to write logs to file.
        verbose: If True, set level to DEBUG and use verbose format.
        quiet: If True, set level to WARNING.
    
    Returns:
        The configured 'dtau_bench' root logger.
    """
    # Determine effective level
    if verbose:
        effective_level = logging.DEBUG
    elif quiet:
        effective_level = logging.WARNING
    else:
        if isinstance(level, str):
            effective_level = getattr(logging, level.upper(), logging.INFO)
        else:
            effective_level = level

    # Choose format
    fmt = VERBOSE_FORMAT if verbose else DEFAULT_FORMAT
    formatter = logging.Formatter(fmt, datefmt=DEFAULT_DATE_FORMAT)

    # Configure the dtau_bench hierarchy logger
    root_logger = logging.getLogger("dtau_bench")
    root_logger.setLevel(effective_level)
    
    # Clear existing handlers to avoid duplicates on repeated calls
    root_logger.handlers.clear()

    # Console handler (stderr)
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(effective_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)  # File always captures all levels
        file_handler.setFormatter(logging.Formatter(VERBOSE_FORMAT, datefmt=DEFAULT_DATE_FORMAT))
        root_logger.addHandler(file_handler)

    # Suppress noisy third-party loggers
    logging.getLogger("litellm").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    return root_logger


def get_logger(name: str) -> logging.Logger:
    """Get a logger under the 'dtau_bench' hierarchy.
    
    Args:
        name: Logger name (will be prefixed with 'dtau_bench.').
            Use module-relative names like 'rollout.engine' or 'strategy.rag'.
    
    Returns:
        Named logger instance.
    
    Example:
        >>> logger = get_logger("rollout.engine")
        >>> logger.info("Starting rollout...")
    """
    if name.startswith("dtau_bench."):
        return logging.getLogger(name)
    return logging.getLogger(f"dtau_bench.{name}")
