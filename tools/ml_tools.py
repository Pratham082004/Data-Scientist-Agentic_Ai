# tools/ml_tools.py
"""
ML tools: feature preparation, splitting, evaluation, dataset profiling, saving.

AutoML-Ready Version — used by MLAgent
------------------------------------------------------------------
✔ No model construction here  (MLAgent builds models)
✔ No usage of model.estimators_ (avoids RF/XGB crashes)
✔ Only handles:
    - Data loading
    - Feature/label preparation
    - Dataset profiling (NEW)
    - Evaluation metrics (expanded)
    - Train/test split
    - Model saving
"""

from __future__ import annotations
from typing import Tuple, Dict, Any, Optional

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split as sk_train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    r2_score,
    mean_squared_error,
    mean_absolute_error,
)

from .logger import get_logger
from .file_manager import default_model_path

logger = get_logger(__name__)


# --------------------------------------------------------------------------
# DATA LOADING
# --------------------------------------------------------------------------

def load_dataframe_from_path(path: str) -> pd.DataFrame:
    logger.info("Loading dataframe from %s", path)
    return pd.read_csv(path)


# --------------------------------------------------------------------------
# FEATURE PREPARATION
# --------------------------------------------------------------------------

def prepare_features_labels(df: pd.DataFrame, target: str) -> Tuple[pd.DataFrame, pd.Series]:
    if target not in df.columns:
        raise KeyError(f"Target column '{target}' not found in dataframe.")

    y = df[target]
    X = df.drop(columns=[target])

    # Drop NA rows
    mask = ~(X.isnull().any(axis=1) | y.isnull())
    X = X[mask]
    y = y[mask]

    logger.info("Prepared features X shape=%s, y length=%d", X.shape, len(y))
    return X, y


# --------------------------------------------------------------------------
# TRAIN/TEST SPLIT
# --------------------------------------------------------------------------

def train_test_split(X, y, test_size: float = 0.2, random_state: int = 42):
    logger.info("Splitting data: test_size=%.2f, random_state=%d", test_size, random_state)
    return sk_train_test_split(X, y, test_size=test_size, random_state=random_state)


# --------------------------------------------------------------------------
# EVALUATION (EXPANDED FOR AUTOML)
# --------------------------------------------------------------------------

def evaluate_model(model, X_test, y_test, problem_type: str = "classification") -> Dict[str, float]:
    preds = model.predict(X_test)

    # -------------------- CLASSIFICATION --------------------
    if problem_type == "classification":
        try:
            proba = model.predict_proba(X_test)[:, 1] if len(np.unique(y_test)) == 2 else None
        except Exception:
            proba = None

        metrics = {
            "accuracy": float(accuracy_score(y_test, preds)),
            "precision": float(precision_score(y_test, preds, average="weighted", zero_division=0)),
            "recall": float(recall_score(y_test, preds, average="weighted", zero_division=0)),
            "f1": float(f1_score(y_test, preds, average="weighted")),
        }

        # Optional ROC-AUC (only for binary)
        if proba is not None:
            try:
                metrics["roc_auc"] = float(roc_auc_score(y_test, proba))
            except Exception:
                metrics["roc_auc"] = None
        else:
            metrics["roc_auc"] = None

    # -------------------- REGRESSION ------------------------
    else:
        mse = mean_squared_error(y_test, preds)
        metrics = {
            "r2": float(r2_score(y_test, preds)),
            "mse": float(mse),
            "rmse": float(np.sqrt(mse)),
            "mae": float(mean_absolute_error(y_test, preds)),
        }

    logger.info("Evaluation metrics: %s", metrics)
    return metrics


# --------------------------------------------------------------------------
# DATASET PROFILING HELPERS (USED BY AUTOML)
# --------------------------------------------------------------------------

def get_dataset_profile(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Returns key dataset statistics used by AutoML to decide model selection.
    """
    profile = {
        "rows": df.shape[0],
        "cols": df.shape[1],
        "num_features": len(df.select_dtypes(include=[np.number]).columns),
        "cat_features": len(df.select_dtypes(include=["object", "category"]).columns),
        "missing_pct": float(df.isnull().mean().mean()),  # avg missing %
        "memory_mb": round(df.memory_usage(deep=True).sum() / (1024 * 1024), 3),
        "skewed_features": _count_skewed(df),
        "cardinality": _column_cardinality(df),
    }

    return profile


def _count_skewed(df: pd.DataFrame, threshold: float = 1.0) -> int:
    """
    Count number of numerical columns with high skewness.
    """
    num_cols = df.select_dtypes(include=[np.number])
    return int((num_cols.skew().abs() > threshold).sum())


def _column_cardinality(df: pd.DataFrame) -> Dict[str, int]:
    """
    Count unique values per column (used in target inference & model choice).
    """
    return {col: int(df[col].nunique()) for col in df.columns}


# --------------------------------------------------------------------------
# CLASS IMBALANCE DETECTION
# --------------------------------------------------------------------------

def detect_imbalance(y: pd.Series) -> Dict[str, Any]:
    """
    Used in AutoML to decide:
    - whether to apply class weights
    - whether to pick tree models over linear
    """
    counts = y.value_counts(normalize=True)
    imbalance = float(counts.max() - counts.min()) if len(counts) > 1 else 0.0

    return {
        "imbalance_score": imbalance,
        "distribution": counts.to_dict(),
        "is_imbalanced": imbalance > 0.5,
    }


# --------------------------------------------------------------------------
# MODEL SAVING
# --------------------------------------------------------------------------

def save_model(model, path: Optional[str] = None) -> str:
    if path is None:
        path = default_model_path()
    joblib.dump(model, path)
    logger.info("Saved model to %s", path)
    return path
