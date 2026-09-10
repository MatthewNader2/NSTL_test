"""
NSTL Centralized Logging Configuration
Provides structured logging with consistent format, file + console output,
and per-component loggers for easy debugging.
"""
import logging
import os
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler

# Default log directory
_LOG_DIR = os.environ.get("NSTL_LOGS_DIR", str(Path(__file__).resolve().parent.parent / "logs"))

_INITIALIZED = False


def setup_logging(level: int = logging.INFO, log_file: str = "nstl.log") -> None:
    """
    Initialize the NSTL logging system.
    Call once at application startup (e.g., in main.py).
    
    Creates:
      - Console handler (stdout) with colored level names
      - File handler (logs/nstl.log) with full timestamps
    """
    global _INITIALIZED
    if _INITIALIZED:
        return
    _INITIALIZED = True

    os.makedirs(_LOG_DIR, exist_ok=True)
    log_path = os.path.join(_LOG_DIR, log_file)

    # Root format
    fmt = "%(asctime)s | %(name)-22s | %(levelname)-7s | %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    # Root logger
    root = logging.getLogger()
    root.setLevel(level)

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    console.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
    root.addHandler(console)

    # Rotating file handler (10MB max, 5 backups)
    file_handler = RotatingFileHandler(log_path, maxBytes=10*1024*1024, backupCount=5, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)  # Always capture DEBUG to file
    file_handler.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
    root.addHandler(file_handler)

    # Silence noisy third-party libraries
    for noisy in [
        "uvicorn.access",
        "uvicorn.error",
        "httpx",
        "httpcore",
        "sentence_transformers",
        "transformers",
        "filelock",
        "urllib3",
    ]:
        logging.getLogger(noisy).setLevel(logging.WARNING)

    logging.getLogger("nstl").info(f"NSTL logging initialized. File: {log_path}")


def get_logger(name: str) -> logging.Logger:
    """
    Returns a namespaced NSTL logger.
    Usage: logger = get_logger("router")  -> creates logger "nstl.router"
    """
    return logging.getLogger(f"nstl.{name}")
