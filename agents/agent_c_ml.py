"""
MLAgent — Pure Python AutoML v2 (Balanced Auto-Mode)
------------------------------------------------------

✔ Deterministic AutoML (Mode A)
✔ Balanced speed vs accuracy (Option B)
✔ Auto model selection based on dataset size
✔ Auto CV folds (5 / 3 / 2 / none)
✔ Downsampled CV for very large datasets
✔ Handles missing-class folds
✔ Fallback scoring when CV fails
✔ Feature importance extraction (tree models)
✔ Sample prediction extraction
✔ Full ML Insight JSON report
✔ Saves JSON to: outputs/ml/ml_insights_report.json
✔ No estimators_ usage anywhere
"""

from __future__ import annotations
import logging
import os
import json
import time
from typing import Any, Dict, Optional, List

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.linear_model import LogisticRegression, LinearRegression, SGDRegressor
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, r2_score

from .llm_agent_base import LLMAgentBase, AgentResult

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# Optional gradient-boosting libraries
try:
    import lightgbm as lgb  # type: ignore
    HAS_LGB = True
except Exception:
    HAS_LGB = False

try:
    import xgboost as xgb  # type: ignore
    HAS_XGB = True
except Exception:
    HAS_XGB = False


# =====================================================================
#  MLAgent
# =====================================================================
class MLAgent(LLMAgentBase):
    def __init__(self, name: str = "MLAgent", use_llm_plan: bool = False, **kwargs):
        super().__init__(name=name, **kwargs)
        self.use_llm_plan = use_llm_plan

    # ================================================================
    # PUBLIC run() — Mode A = deterministic AutoML
    # ================================================================
    def run(self, context: Dict[str, Any], tools: Any = None) -> AgentResult:
        if self.mode == "A":
            try:
                return self._run_deterministic(context, tools)
            except Exception as e:
                logger.exception("MLAgent failed: %s", e)
                return AgentResult(False, error=str(e), messages=[f"ml_error: {e}"])
        else:
            # Fallback to full THINK/PLAN/ACT if you ever enable Mode B
            return super().run(context, tools)

    # =====================================================================
    # Feature Importance Helper
    # =====================================================================
    def _extract_feature_importance(self, model, X_cols: List[str]):
        """
        Generic feature_importances_ reader.
        Only works for tree-based models (RF, LGBM, XGB, etc.).
        """
        try:
            if hasattr(model, "feature_importances_"):
                values = getattr(model, "feature_importances_", None)
                if values is None:
                    return None
                return sorted(
                    list(zip(X_cols, values)),
                    key=lambda x: float(x[1]),
                    reverse=True,
                )
        except Exception:
            pass
        return None

    # =====================================================================
    # Decide CV folds (Balanced Auto-Mode)
    # =====================================================================
    def _decide_cv_splits(self, n_samples: int) -> int:
        """
        Balanced strategy:
            - < 15k  → 5 folds
            - 15–40k → 3 folds
            - 40–80k → 2 folds
            - > 80k  → no CV (0 → use fallback scoring only)
        """
        if n_samples < 15000:
            return 5
        elif n_samples < 40000:
            return 3
        elif n_samples < 80000:
            return 2
        else:
            return 0  # no CV, only train/test scoring

    def _make_cv_splitter(
        self,
        y: pd.Series,
        problem_type: str,
        n_splits: int,
    ):
        if n_splits < 2:
            return None

        if problem_type == "classification":
            try:
                return StratifiedKFold(
                    n_splits=n_splits, shuffle=True, random_state=42
                )
            except Exception:
                pass

        return KFold(n_splits=n_splits, shuffle=True, random_state=42)

    # =====================================================================
    # Decide model suite based on dataset size
    # =====================================================================
    def _build_model_suite(
        self, problem_type: str, n_rows: int
    ) -> Dict[str, Any]:
        """
        Balanced strategy:
            - Small (<20k): richer models (RF with more trees, XGB/LGB if available)
            - Medium (20–80k): moderate RF/LGB, maybe XGB
            - Large (>80k): speed-friendly configs (fewer trees, SGD, LGB)
        """
        models: Dict[str, Any] = {}

        if problem_type == "classification":
            # Always include a linear baseline
            models["LogReg"] = LogisticRegression(max_iter=300)

            # RandomForest tuned by dataset size
            if n_rows < 20000:
                models["RF"] = RandomForestClassifier(
                    n_estimators=300, max_depth=None, random_state=42
                )
            elif n_rows < 80000:
                models["RF"] = RandomForestClassifier(
                    n_estimators=150, max_depth=None, random_state=42
                )
            else:
                models["RF"] = RandomForestClassifier(
                    n_estimators=80, max_depth=15, random_state=42
                )

            # LightGBM if available
            if HAS_LGB:
                if n_rows < 50000:
                    models["LGBM"] = lgb.LGBMClassifier(
                        n_estimators=200,
                        learning_rate=0.1,
                        num_leaves=31,
                        random_state=42,
                    )
                else:
                    models["LGBM"] = lgb.LGBMClassifier(
                        n_estimators=120,
                        learning_rate=0.1,
                        num_leaves=31,
                        random_state=42,
                    )

            # XGBoost only for not-too-large datasets
            if HAS_XGB and n_rows < 50000:
                models["XGB"] = xgb.XGBClassifier(
                    n_estimators=200,
                    max_depth=4,
                    learning_rate=0.1,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    tree_method="hist",
                    random_state=42,
                )

            # SGD for very large data
            if n_rows > 80000:
                models["SGDClassifier"] = SGDClassifier(
                    loss="log_loss",
                    max_iter=1000,
                    random_state=42,
                )

        else:
            # Regression
            models["LinReg"] = LinearRegression()

            if n_rows < 20000:
                models["RFReg"] = RandomForestRegressor(
                    n_estimators=300, max_depth=None, random_state=42
                )
            elif n_rows < 80000:
                models["RFReg"] = RandomForestRegressor(
                    n_estimators=150, max_depth=None, random_state=42
                )
            else:
                models["RFReg"] = RandomForestRegressor(
                    n_estimators=80, max_depth=15, random_state=42
                )

            if HAS_LGB:
                if n_rows < 50000:
                    models["LGBMReg"] = lgb.LGBMRegressor(
                        n_estimators=200,
                        learning_rate=0.1,
                        num_leaves=31,
                        random_state=42,
                    )
                else:
                    models["LGBMReg"] = lgb.LGBMRegressor(
                        n_estimators=120,
                        learning_rate=0.1,
                        num_leaves=31,
                        random_state=42,
                    )

            if HAS_XGB and n_rows < 50000:
                models["XGBReg"] = xgb.XGBRegressor(
                    n_estimators=200,
                    max_depth=4,
                    learning_rate=0.1,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    tree_method="hist",
                    random_state=42,
                )

            if n_rows > 80000:
                models["SGDRegressor"] = SGDRegressor(
                    max_iter=1000,
                    random_state=42,
                )

        return models

    # =====================================================================
    # Deterministic AutoML Pipeline (Balanced)
    # =====================================================================
    def _run_deterministic(self, context: Dict[str, Any], tools: Any) -> AgentResult:
        messages: List[str] = []

        # -------------------------------------
        # Load dataframe
        # -------------------------------------
        df: Optional[pd.DataFrame] = context.get("dataframe")
        data_path = context.get("data_path")
        target = context.get("target_column")

        if df is None:
            if not data_path:
                raise ValueError("MLAgent: No dataframe in context and no data_path provided.")
            df = tools.load_dataframe_from_path(data_path)
            messages.append(f"Loaded dataframe {df.shape} from {data_path}")
        else:
            messages.append(f"Using dataframe from context {df.shape}")

        if df.empty:
            raise ValueError("Empty dataframe — cannot train.")

        n_rows_total = int(df.shape[0])

        # -------------------------------------
        # Detect / validate target
        # -------------------------------------
        if not target:
            cols = [c for c in df.columns if c != "_is_outlier"]
            if not cols:
                raise ValueError("No valid target column found.")
            target = cols[-1]
            messages.append(f"Inferred target='{target}'")
        else:
            messages.append(f"Using target: {target}")

        if target not in df.columns:
            raise ValueError(f"Target '{target}' not found in dataframe.")

        # -------------------------------------
        # X, y preparation
        # -------------------------------------
        if hasattr(tools, "prepare_features_labels"):
            X, y = tools.prepare_features_labels(df, target)
        else:
            X = df.drop(columns=[target])
            y = df[target]

        messages.append(f"Prepared X={X.shape}, y={len(y)}")

        # -------------------------------------
        # Problem type inference
        # -------------------------------------
        if y.nunique() <= 20 and not np.issubdtype(y.dtype, np.floating):
            problem_type = "classification"
        else:
            problem_type = "regression"

        messages.append(f"Problem type: {problem_type}")

        # Regression → numeric target
        if problem_type == "regression":
            y = pd.to_numeric(y, errors="coerce")
            mask = y.notna()
            X, y = X[mask], y[mask]
            messages.append(f"Coerced numeric target → samples={len(y)}")

        if len(y) < 5:
            raise ValueError("Too few samples (min=5).")

        # -------------------------------------
        # Train/Test split
        # -------------------------------------
        X_train, X_test, y_train, y_test = tools.train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        messages.append(
            f"Split → X_train={X_train.shape}, X_test={X_test.shape}, "
            f"y_train={len(y_train)}, y_test={len(y_test)}"
        )

        # -------------------------------------
        # Build model suite (Balanced auto-mode)
        # -------------------------------------
        models = self._build_model_suite(problem_type, n_rows_total)
        messages.append(f"Models attempted: {list(models.keys())}")

        # -------------------------------------
        # Decide CV strategy & CV dataset
        # -------------------------------------
        n_train = len(y_train)
        n_splits = self._decide_cv_splits(n_train)

        cv_scores: Dict[str, float] = {}

        # Optionally downsample for CV if extremely large
        if n_splits >= 2 and n_train > 100_000:
            cv_sample_size = min(25_000, n_train)
            rng = np.random.default_rng(seed=42)
            sample_idx = rng.choice(n_train, size=cv_sample_size, replace=False)
            X_train_cv = X_train.iloc[sample_idx]
            y_train_cv = y_train.iloc[sample_idx]
            messages.append(
                f"Downsampled for CV: {cv_sample_size} of {n_train} rows "
                f"(splits={n_splits})"
            )
        else:
            X_train_cv = X_train
            y_train_cv = y_train

        cv_splitter = self._make_cv_splitter(y_train_cv, problem_type, n_splits)

        # -------------------------------------
        # Cross-validation (if enabled)
        # -------------------------------------
        if cv_splitter is not None:
            for name, model in models.items():
                fold_scores: List[float] = []

                for fold, (tr, val) in enumerate(cv_splitter.split(X_train_cv, y_train_cv)):
                    X_tr, X_val = X_train_cv.iloc[tr], X_train_cv.iloc[val]
                    y_tr, y_val = y_train_cv.iloc[tr], y_train_cv.iloc[val]

                    try:
                        m = clone(model)
                        m.fit(X_tr, y_tr)
                        preds = m.predict(X_val)

                        if problem_type == "classification":
                            fold_scores.append(float(accuracy_score(y_val, preds)))
                        else:
                            fold_scores.append(float(r2_score(y_val, preds)))

                    except Exception as e:
                        messages.append(f"Fold {fold} failed for {name}: {e}")

                if fold_scores:
                    cv_scores[name] = float(np.mean(fold_scores))
                    messages.append(
                        f"CV {name} (n_splits={n_splits}): {cv_scores[name]:.4f}"
                    )

        # -------------------------------------
        # Fallback single-shot scoring (or large datasets w/ no CV)
        # -------------------------------------
        if not cv_scores:
            messages.append("⚠ No valid CV scores → using train/test scoring fallback.")

            for name, model in models.items():
                try:
                    m = clone(model)
                    m.fit(X_train, y_train)
                    preds = m.predict(X_test)

                    if problem_type == "classification":
                        score = float(accuracy_score(y_test, preds))
                    else:
                        score = float(r2_score(y_test, preds))

                    cv_scores[name] = score
                    messages.append(f"Fallback {name}: {score:.4f}")

                except Exception as e:
                    messages.append(f"Fallback failed for {name}: {e}")

        if not cv_scores:
            raise RuntimeError("No valid scores from CV or fallback!")

        # -------------------------------------
        # Best model
        # -------------------------------------
        best_name = max(cv_scores, key=cv_scores.get)
        best_model = clone(models[best_name])
        best_model.fit(X_train, y_train)

        messages.append(f"Best model → {best_name} (score={cv_scores[best_name]:.4f})")

        # -------------------------------------
        # ML INSIGHT JSON REPORT
        # -------------------------------------
        feature_imp = self._extract_feature_importance(best_model, list(X.columns))

        # sample predictions (up to 10)
        sample_predictions: List[Dict[str, float]] = []
        try:
            n_samples = min(10, len(X_test))
            if n_samples > 0:
                preds = best_model.predict(X_test[:n_samples])
                for true, pred in zip(y_test[:n_samples], preds):
                    try:
                        sample_predictions.append(
                            {"true": float(true), "pred": float(pred)}
                        )
                    except Exception:
                        continue
        except Exception as e:
            messages.append(f"Sample prediction extraction failed: {e}")

        ml_insights: Dict[str, Any] = {
            "dataset_shape": [int(df.shape[0]), int(df.shape[1])],
            "train_shape": [int(X_train.shape[0]), int(X_train.shape[1])],
            "test_shape": [int(X_test.shape[0]), int(X_test.shape[1])],
            "target_column": str(target),
            "problem_type": problem_type,
            "models_attempted": list(models.keys()),
            "cv_strategy": {
                "n_train": n_train,
                "n_splits": n_splits,
                "cv_downsampled": bool(n_splits >= 2 and n_train > 100_000),
            },
            "cv_scores": {k: float(v) for k, v in cv_scores.items()},
            "best_model": {
                "name": best_name,
                "score": float(cv_scores[best_name]),
            },
            "feature_importances": (
                [
                    {"feature": str(f), "importance": float(imp)}
                    for (f, imp) in feature_imp
                ]
                if feature_imp
                else None
            ),
            "sample_predictions": sample_predictions,
            "warnings": [m for m in messages if "failed" in m.lower() or "⚠" in m],
        }

        # -------------------------------------
        # Save ML insight JSON → outputs/ml/ml_insights_report.json
        # -------------------------------------
        ml_insights_path: Optional[str] = None
        try:
            os.makedirs("outputs/ml", exist_ok=True)
            # You can switch to timestamped files if you want:
            # ts = int(time.time() * 1000)
            # filename = f"ml_insights_{ts}.json"
            filename = "ml_insights_report.json"
            ml_insights_path = os.path.join("outputs", "ml", filename)

            with open(ml_insights_path, "w", encoding="utf-8") as f:
                json.dump(ml_insights, f, indent=2, default=float)

            messages.append(f"ML insights JSON saved → {ml_insights_path}")
        except Exception as e:
            messages.append(f"Failed to save ML insights JSON: {e}")

        # -------------------------------------
        # Save model
        # -------------------------------------
        saved_path: Optional[str] = None
        try:
            saved_path = tools.save_model(best_model)
            messages.append(f"Model saved → {saved_path}")
        except Exception as e:
            messages.append(f"Save failed: {e}")

        # -------------------------------------
        # Return
        # -------------------------------------
        return AgentResult(
            success=True,
            outputs={
                "target": target,
                "problem_type": problem_type,
                "scores": cv_scores,
                "best_model_name": best_name,
                "best_score": cv_scores[best_name],
                "saved_path": saved_path,
                "ml_insights": ml_insights,
                "ml_insights_path": ml_insights_path,
            },
            messages=messages,
        )

    # --------------------------------------------------------------
    # Mode B stubs
    # --------------------------------------------------------------
    def think(self, ctx):
        return "Mode B disabled"

    def plan(self, thought):
        return {}

    def act(self, plan, tools):
        return self._run_deterministic(plan.get("context", {}), tools)

    def reflect(self, result):
        return None if result.success else {"retry": False}
