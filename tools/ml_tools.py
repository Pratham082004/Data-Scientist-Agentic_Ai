# tools/ml_tools.py
"""
ML tools: feature preparation, splitting, model training, evaluation, saving.

Used by MLAgent (Agent C).
"""

from __future__ import annotations
from typing import Tuple, Dict, Any, List, Optional

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split as sk_train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    r2_score,
    mean_squared_error,
)

from .file_manager import default_model_path
from .logger import get_logger

logger = get_logger(__name__)


# ---------- Loading helper (reused by Analyst/ML) ----------

def load_dataframe_from_path(path: str) -> pd.DataFrame:
    logger.info("Loading dataframe from %s", path)
    return pd.read_csv(path)


# ---------- Feature preparation ----------

def prepare_features_labels(df: pd.DataFrame, target: str) -> Tuple[pd.DataFrame, pd.Series]:
    if target not in df.columns:
        raise KeyError(f"Target column '{target}' not found in dataframe.")

    y = df[target]
    X = df.drop(columns=[target])

    # Basic safety: drop rows with NA in either X or y
    mask = ~(X.isnull().any(axis=1) | y.isnull())
    X = X[mask]
    y = y[mask]

    logger.info("Prepared features X shape=%s, y length=%d", X.shape, len(y))
    return X, y


# ---------- Train-test split ----------

def train_test_split(X, y, test_size: float = 0.2, random_state: int = 42):
    logger.info("Splitting data: test_size=%.2f, random_state=%d", test_size, random_state)
    return sk_train_test_split(X, y, test_size=test_size, random_state=random_state)


# ---------- Baseline training ----------

def train_baseline_model(X_train, y_train, problem_type: str = "classification"):
    if problem_type == "classification":
        model = LogisticRegression(max_iter=200)
    else:
        model = LinearRegression()
    model.fit(X_train, y_train)
    logger.info("Trained baseline model: %s", type(model).__name__)
    return model


# ---------- Advanced models ----------

def try_advanced_models(
    X_train,
    y_train,
    X_test,
    y_test,
    problem_type: str = "classification",
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []

    if problem_type == "classification":
        candidates = [
            ("LogisticRegression", LogisticRegression(max_iter=200)),
            ("RandomForestClassifier", RandomForestClassifier(n_estimators=100)),
        ]
    else:
        candidates = [
            ("LinearRegression", LinearRegression()),
            ("RandomForestRegressor", RandomForestRegressor(n_estimators=100)),
        ]

    for name, model in candidates:
        model.fit(X_train, y_train)
        if problem_type == "classification":
            preds = model.predict(X_test)
            metrics = {
                "accuracy": float(accuracy_score(y_test, preds)),
                "f1": float(f1_score(y_test, preds, average="weighted")),
            }
        else:
            preds = model.predict(X_test)
            metrics = {
                "r2": float(r2_score(y_test, preds)),
                "mse": float(mean_squared_error(y_test, preds)),
            }
        results.append({"name": name, "model": model, "metrics": metrics})
        logger.info("Advanced model %s metrics: %s", name, metrics)

    return results


# ---------- Evaluation ----------

def evaluate_model(model, X_test, y_test, problem_type: str = "classification") -> Dict[str, float]:
    preds = model.predict(X_test)
    if problem_type == "classification":
        metrics = {
            "accuracy": float(accuracy_score(y_test, preds)),
            "f1": float(f1_score(y_test, preds, average="weighted")),
        }
    else:
        metrics = {
            "r2": float(r2_score(y_test, preds)),
            "mse": float(mean_squared_error(y_test, preds)),
        }
    logger.info("Evaluation metrics: %s", metrics)
    return metrics


# ---------- Model saving ----------

def save_model(model, path: Optional[str] = None) -> str:
    if path is None:
        path = default_model_path()
    joblib.dump(model, path)
    logger.info("Saved model to %s", path)
    return path
