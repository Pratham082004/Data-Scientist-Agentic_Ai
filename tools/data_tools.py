# tools/data_tools.py
"""
Data tools: loading, cleaning, encoding, scaling, outlier detection.
Used by DataCleanerAgent.
"""

from __future__ import annotations
from typing import Optional

import os
import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .file_manager import ensure_dir
from .logger import get_logger

logger = get_logger(__name__)


# ---------- I/O ----------

def load_csv(path: str) -> pd.DataFrame:
    logger.info("Loading CSV from %s", path)
    df = pd.read_csv(path)
    return df


def save_dataframe(df: pd.DataFrame, path: Optional[str] = None) -> str:
    """
    FIXED VERSION:
    - Always saves to a FILE, never a folder.
    - Generates a unique timestamp filename by default.
    - Ensures the 'data/processed' directory exists.
    """

    # Ensure processed folder exists
    processed_dir = "data/processed"
    ensure_dir(processed_dir)

    # Auto-generate a unique timestamped file if no path is given
    if path is None:
        ts = int(time.time() * 1000)
        path = f"{processed_dir}/cleaned_{ts}.csv"

    logger.info("Saving dataframe to %s", path)

    # Save file (path is always a file, NOT a folder)
    df.to_csv(path, index=False)

    return path


# ---------- Cleaning helpers ----------

def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    before = df.shape[0]
    df = df.drop_duplicates()
    after = df.shape[0]
    logger.info("Dropped %d duplicate rows.", before - after)
    return df


def impute_numerics(df: pd.DataFrame, strategy: str = "mean") -> pd.DataFrame:
    num_cols = df.select_dtypes(include=[np.number]).columns
    if strategy == "mean":
        df[num_cols] = df[num_cols].fillna(df[num_cols].mean())
    elif strategy == "median":
        df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    else:
        raise ValueError(f"Unsupported numeric imputation strategy: {strategy}")
    logger.info("Imputed numeric columns using %s.", strategy)
    return df


def impute_categoricals(df: pd.DataFrame, strategy: str = "mode") -> pd.DataFrame:
    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    if strategy == "mode":
        for col in cat_cols:
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].mode().iloc[0])
    else:
        raise ValueError(f"Unsupported categorical imputation strategy: {strategy}")
    logger.info("Imputed categorical columns using %s.", strategy)
    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    for col in cat_cols:
        df[col] = df[col].astype("category").cat.codes
    logger.info("Encoded %d categorical columns with category codes.", len(cat_cols))
    return df


def scale_numerics(df: pd.DataFrame, method: str = "standard") -> pd.DataFrame:
    num_cols = df.select_dtypes(include=[np.number]).columns
    if method == "standard" and len(num_cols) > 0:
        scaler = StandardScaler()
        df[num_cols] = scaler.fit_transform(df[num_cols])
        logger.info("Scaled %d numeric columns with StandardScaler.", len(num_cols))
    else:
        logger.info("No scaling performed (method=%s, num_cols=%d).", method, len(num_cols))
    return df


def detect_outliers(df: pd.DataFrame, method: str = "iqr") -> pd.Series:
    """
    Returns a boolean Series marking rows considered outliers.
    Simple IQR-based detection applied per numeric column.
    """
    num_cols = df.select_dtypes(include=[np.number]).columns
    if method != "iqr" or len(num_cols) == 0:
        logger.info("Outlier detection skipped (method=%s, num_cols=%d).", method, len(num_cols))
        return pd.Series([False] * df.shape[0], index=df.index)

    is_outlier = pd.Series(False, index=df.index)
    for col in num_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        col_outliers = (df[col] < lower) | (df[col] > upper)
        is_outlier |= col_outliers

    logger.info("Detected %d outlier rows using IQR.", is_outlier.sum())
    return is_outlier
