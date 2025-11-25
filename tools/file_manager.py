# tools/file_manager.py
"""
File and path utilities for the project.
"""

from __future__ import annotations
import os
from typing import Optional


def ensure_dir(path: str) -> None:
    """Ensure a directory exists."""
    os.makedirs(path, exist_ok=True)


def join_path(*parts: str) -> str:
    """Safe os.path.join wrapper."""
    return os.path.join(*parts)


def default_processed_path(filename: str = "cleaned_data.csv") -> str:
    """Return default path for processed data."""
    ensure_dir("data/processed")
    return os.path.join("data", "processed", filename)


def default_model_path(filename: str = "saved_model.pkl") -> str:
    """Return default path for saved models."""
    ensure_dir("models")
    return os.path.join("models", filename)


def default_eda_dir() -> str:
    """Return default directory for EDA outputs."""
    ensure_dir("outputs/eda")
    return "outputs/eda"
