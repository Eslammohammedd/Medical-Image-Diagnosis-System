"""
utils/logger.py
───────────────
Configures a project-wide logger that writes to both the console and a
rotating file (logs/app.log).  Import and call get_logger() from any module.
"""
import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path


_LOG_DIR  = Path(__file__).resolve().parent.parent / "logs"
_LOG_FILE = _LOG_DIR / "app.log"
_LOG_DIR.mkdir(parents=True, exist_ok=True)

_FMT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FMT = "%Y-%m-%d %H:%M:%S"


def get_logger(name: str = "medical_ai", level: int = logging.INFO) -> logging.Logger:
    """
    Returns a configured logger.  Calling with the same name returns the
    same instance (standard Python behaviour).

    Usage
    -----
    >>> from utils.logger import get_logger
    >>> log = get_logger(__name__)
    >>> log.info("Model loaded successfully.")
    """
    logger = logging.getLogger(name)

    if logger.handlers:          # already configured – return as-is
        return logger

    logger.setLevel(level)

    formatter = logging.Formatter(_FMT, datefmt=_DATE_FMT)

    # ── Console handler ────────────────────────────────────────────────────────
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)

    # ── Rotating file handler (max 5 MB × 3 backups) ─────────────────────────
    file_handler = RotatingFileHandler(
        _LOG_FILE,
        maxBytes=5 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger
