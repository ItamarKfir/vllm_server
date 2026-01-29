"""
Logging utility with date and time timestamps.
"""

import logging
import sys
from datetime import datetime


def setup_logger(name: str = "vllm_server", level: int = logging.INFO) -> logging.Logger:
    """Setup logger with date/time format."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger


def log_with_timestamp(message: str, level: str = "INFO") -> None:
    """Quick log with timestamp (date and time)."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    level_str = level.upper().ljust(8)
    print(f"{timestamp} | {level_str} | {message}", flush=True)
