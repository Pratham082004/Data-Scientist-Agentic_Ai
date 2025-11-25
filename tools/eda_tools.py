# tools/eda_tools.py
"""
EDA tools: summary statistics, correlation heatmap, distribution plots, relationships.

Used by AnalystAgent (Agent B).
"""

from __future__ import annotations
from typing import Dict, List, Optional

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import seaborn as sns  # optional
    _HAS_SEABORN = True
except ImportError:
    _HAS_SEABORN = False

from .file_manager import ensure_dir, default_eda_dir
from .logger import get_logger

logger = get_logger(__name__)


# ---------- Load helper ----------

def load_dataframe_from_path(path: str) -> pd.DataFrame:
    logger.info("Loading dataframe from %s", path)
    return pd.read_csv(path)


# ---------- Summary ----------

def summary_statistics(df: pd.DataFrame) -> Dict:
    """
    Returns basic summary stats as a nested dict (good for JSON serialization).
    """
    desc = df.describe(include="all").transpose()
    summary = desc.fillna("").to_dict(orient="index")
    logger.info("Generated summary statistics for %d columns.", len(summary))
    return summary


# ---------- Correlation heatmap ----------

def correlation_heatmap(df: pd.DataFrame, out_path: Optional[str] = None) -> str:
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] == 0:
        logger.warning("No numeric columns for correlation heatmap.")
        return ""

    if out_path is None or os.path.isdir(out_path):
        ensure_dir(default_eda_dir())
        out_path = os.path.join(default_eda_dir(), "correlation_heatmap.png")

    plt.figure(figsize=(8, 6))
    corr = numeric_df.corr()
    if _HAS_SEABORN:
        sns.heatmap(corr, annot=False)
    else:
        plt.imshow(corr, aspect="auto")
        plt.colorbar()
        plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
        plt.yticks(range(len(corr.columns)), corr.columns)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    logger.info("Saved correlation heatmap to %s", out_path)
    return out_path


# ---------- Distribution plots ----------

def plot_distributions(df: pd.DataFrame, out_dir: Optional[str] = None, columns: Optional[List[str]] = None) -> List[str]:
    if out_dir is None:
        out_dir = default_eda_dir()
    ensure_dir(out_dir)

    if columns is None:
        columns = df.columns.tolist()

    file_paths: List[str] = []

    for col in columns:
        series = df[col]
        plt.figure(figsize=(6, 4))

        if np.issubdtype(series.dtype, np.number):
            # numeric histogram
            plt.hist(series.dropna(), bins=30)
            plt.xlabel(col)
            plt.ylabel("Frequency")
            plt.title(f"Distribution of {col}")
        else:
            # categorical bar plot
            value_counts = series.value_counts().head(20)
            plt.bar(value_counts.index.astype(str), value_counts.values)
            plt.xticks(rotation=45, ha="right")
            plt.ylabel("Count")
            plt.title(f"Top categories of {col}")

        plt.tight_layout()
        filename = f"dist_{col}.png".replace(" ", "_")
        path = os.path.join(out_dir, filename)
        plt.savefig(path)
        plt.close()
        file_paths.append(path)

    logger.info("Saved %d distribution plots to %s", len(file_paths), out_dir)
    return file_paths


# ---------- Feature relationships ----------

def compute_feature_relationships(df: pd.DataFrame, top_k: int = 5) -> Dict:
    """
    Simple correlation-based relationships on numeric columns.
    Returns a dict with sorted pairwise correlations.
    """
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] < 2:
        logger.warning("Not enough numeric columns for relationships.")
        return {}

    corr = numeric_df.corr().abs()
    # get upper triangle without diagonal
    pairs = []
    cols = corr.columns
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            pairs.append(((cols[i], cols[j]), corr.iloc[i, j]))
    pairs_sorted = sorted(pairs, key=lambda x: x[1], reverse=True)[:top_k]

    relationships = [
        {"feature_1": a, "feature_2": b, "correlation": float(val)}
        for (a, b), val in pairs_sorted
    ]
    logger.info("Computed top %d feature relationships.", len(relationships))
    return {"top_relationships": relationships}
