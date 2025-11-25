# tools/logger.py
"""
Project-wide logging configuration.

Usage:
    from tools.logger import get_logger
    logger = get_logger(__name__)
"""

import logging
import os
from typing import Optional


def _ensure_log_dir(path: str = "outputs/logs") -> None:
    os.makedirs(path, exist_ok=True)


def get_logger(name: Optional[str] = None) -> logging.Logger:
    _ensure_log_dir()
    logger = logging.getLogger(name if name else "data_scientist_agentic_ai")

    if not logger.handlers:
        logger.setLevel(logging.INFO)

        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch_fmt = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        ch.setFormatter(ch_fmt)
        logger.addHandler(ch)

        # File handler
        fh = logging.FileHandler("outputs/logs/app.log", encoding="utf-8")
        fh.setLevel(logging.INFO)
        fh_fmt = logging.Formatter(
            "%(asctime)s\t%(levelname)s\t%(name)s\t%(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        fh.setFormatter(fh_fmt)
        logger.addHandler(fh)

    return logger
