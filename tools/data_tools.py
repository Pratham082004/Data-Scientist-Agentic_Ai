# tools/data_tools.py
"""
Data tools: loading, cleaning, encoding, scaling, outlier detection.

These functions are called by DataCleanerAgent (Agent A).
"""

from __future__ import annotations
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .file_manager import ensure_dir, default_processed_path
from .logger import get_logger

logger = get_logger(__name__)

# Allow agents to reference this as default path
default_processed_path = default_processed_path()


# ---------- I/O ----------

def load_csv(path: str) -> pd.DataFrame:
    logger.info("Loading CSV from %s", path)
    df = pd.read_csv(path)
    return df


def save_dataframe(df: pd.DataFrame, path: Optional[str] = None) -> str:
    if path is None:
        path = default_processed_path
    ensure_dir(path.rsplit("/", 1)[0])
    logger.info("Saving dataframe to %s", path)
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
