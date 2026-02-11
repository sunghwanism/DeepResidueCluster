
import logging
import sys
from typing import Optional
from pathlib import Path

_initialized = False

def setup_logging(log_file: Optional[str] = None, level: int = logging.INFO):
    """
    Configures the root logger with console and optional file handlers.
    Should be called once at the application entry point.
    """
    global _initialized
    if _initialized:
        return

    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Remove existing handlers to avoid duplication/conflict
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    # Define Formatters
    console_fmt = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_fmt = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Console Handler (stdout)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_fmt)
    root_logger.addHandler(console_handler)

    # File Handler
    if log_file:
        try:
            Path(log_file).parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(file_fmt)
            root_logger.addHandler(file_handler)
        except Exception as e:
            print(f"Failed to create log file handler: {e}", file=sys.stderr)

    _initialized = True

def get_logger(name: str) -> logging.Logger:
    """
    Returns a logger with the specified name. 
    Auto-initializes logging configuration if not already done.
    """
    if not _initialized:
        setup_logging() 
    return logging.getLogger(name)
