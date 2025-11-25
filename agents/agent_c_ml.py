# agents/agent_c_ml.py
"""
Machine Learning Agent.

Responsibilities:
- Determine problem type (classification/regression/clustering)
- Prepare features & labels (including optional target extraction)
- Run model selection (simple baseline + optional advanced models via tools.ml_tools)
- Tune hyperparameters (optionally)
- Evaluate and return best model, metrics, and saved model path

This agent uses tools.ml_tools to run heavy lifting. It expects sklearn-compatible objects or
tools wrappers that implement the needed API.
"""

from __future__ import annotations
import logging
from dataclasses import asdict
from typing import Any, Dict, Optional

import pandas as pd
from .llm_agent_base import LLMAgentBase, AgentResult

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class MLAgent(LLMAgentBase):
    def __init__(self, name: str = "MLAgent", **kwargs):
        super().__init__(name=name, **kwargs)

    def think(self, context: Dict[str, Any]) -> str:
        # Determine problem nature: look for target and basic stats
        df: Optional[pd.DataFrame] = context.get("dataframe")
        target = context.get("target_column")
        if df is None:
            return "No dataframe provided to MLAgent. Will request dataset_path load."
        if not target:
            # heuristics for target
            guess = None
            # look for common target names
            for col in ["target", "label", "y", "outcome"]:
                if col in df.columns:
                    guess = col
                    break
            if not guess:
                # fallback: choose last column
                guess = df.columns[-1]
            return f"Detected potential target '{guess}'. Rows={df.shape[0]}, cols={df.shape[1]}."
        return f"Target provided: {target}. Rows={df.shape[0]}, cols={df.shape[1]}."

    def plan(self, thought: str) -> Dict[str, Any]:
        # default plan; could be replaced by LLM call
        plan = {
            "steps": [
                {"action": "extract_features_labels"},
                {"action": "train_test_split"},
                {"action": "baseline_model_train"},
                {"action": "evaluate_baseline"},
                {"action": "optional_advanced_models"},
            ],
            "save_model_path": "models/saved_model.pkl",
            "test_size": 0.2,
            "random_state": 42,
        }
        return plan

    def act(self, plan: Dict[str, Any], tools: Any) -> AgentResult:
        """
        tools expected to provide:
         - prepare_features_labels(df, target) -> (X, y)
         - train_baseline_model(X_train, y_train, problem_type) -> model
         - evaluate_model(model, X_test, y_test) -> dict metrics
         - try_advanced_models(X_train, y_train, X_test, y_test) -> list of candidate results
         - save_model(model, path) -> saved_path
        """
        try:
            df = plan.get("context", {}).get("dataframe")
            if df is None:
                if tools is None or not hasattr(tools, "load_dataframe_from_path"):
                    raise RuntimeError("tools.load_dataframe_from_path required for MLAgent")
                ds_path = plan.get("dataset_path") or plan.get("context", {}).get("dataset_path")
                df = tools.load_dataframe_from_path(ds_path)

            target = plan.get("target") or plan.get("context", {}).get("target_column")
            if not target:
                # try heuristics: common target names or last column
                for c in ["target", "label", "y", "outcome"]:
                    if c in df.columns:
                        target = c
                        break
                if not target:
                    target = df.columns[-1]  # fallback
            messages = [f"Using target column: {target}"]

            # prepare X, y
            if hasattr(tools, "prepare_features_labels"):
                X, y = tools.prepare_features_labels(df, target)
                messages.append("Prepared features and labels.")
            else:
                # naive preparation: drop target, take everything else, drop NA rows
                y = df[target]
                X = df.drop(columns=[target])
                messages.append("Used naive feature extraction (drop target).")

            # split
            test_size = plan.get("test_size", 0.2)
            random_state = plan.get("random_state", 42)
            if hasattr(tools, "train_test_split"):
                X_train, X_test, y_train, y_test = tools.train_test_split(X, y, test_size=test_size, random_state=random_state)
            else:
                # fallback to sklearn
                from sklearn.model_selection import train_test_split
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
            messages.append(f"Train/test split done: test_size={test_size}")

            # determine problem type
            problem_type = "classification" if y.nunique() <= 20 and y.dtype != "float" else "regression"
            messages.append(f"Detected problem type: {problem_type}")

            # baseline training
            baseline_model = None
            if hasattr(tools, "train_baseline_model"):
                baseline_model = tools.train_baseline_model(X_train, y_train, problem_type)
                messages.append("Trained baseline model via tools.")
            else:
                # fallback simple models
                if problem_type == "classification":
                    from sklearn.linear_model import LogisticRegression
                    baseline_model = LogisticRegression(max_iter=200)
                else:
                    from sklearn.linear_model import LinearRegression
                    baseline_model = LinearRegression()
                baseline_model.fit(X_train, y_train)
                messages.append("Trained fallback sklearn baseline model.")

            # evaluate
            if hasattr(tools, "evaluate_model"):
                metrics = tools.evaluate_model(baseline_model, X_test, y_test, problem_type)
            else:
                # fallback basic evaluation
                if problem_type == "classification":
                    from sklearn.metrics import accuracy_score, f1_score
                    preds = baseline_model.predict(X_test)
                    metrics = {"accuracy": float(accuracy_score(y_test, preds)), "f1": float(f1_score(y_test, preds, average="weighted"))}
                else:
                    from sklearn.metrics import r2_score, mean_squared_error
                    preds = baseline_model.predict(X_test)
                    metrics = {"r2": float(r2_score(y_test, preds)), "mse": float(mean_squared_error(y_test, preds))}

            messages.append(f"Evaluation metrics: {metrics}")

            saved_path = None
            if hasattr(tools, "save_model"):
                saved_path = tools.save_model(baseline_model, plan.get("save_model_path"))
                messages.append(f"Saved model via tools to {saved_path}")
            else:
                # fallback to joblib
                try:
                    import joblib
                    joblib.dump(baseline_model, plan.get("save_model_path", "models/saved_model.pkl"))
                    saved_path = plan.get("save_model_path", "models/saved_model.pkl")
                    messages.append(f"Saved model to {saved_path} via joblib.")
                except Exception:
                    messages.append("Model saving skipped (joblib not available or failed).")

            outputs = {
                "problem_type": problem_type,
                "model": baseline_model,
                "metrics": metrics,
                "saved_path": saved_path,
            }

            return AgentResult(success=True, outputs=outputs, messages=messages, metadata={"target": target})

        except Exception as e:
            logger.exception("MLAgent.act failed: %s", e)
            return AgentResult(success=False, error=str(e), messages=[f"act_error: {e}"])

    def reflect(self, result: AgentResult) -> Optional[Dict[str, Any]]:
        if not result.success:
            return {"retry": True, "reason": "ML training failed; try with simpler preprocessing."}
        # If metrics poor, optionally suggest running advanced models
        metrics = result.outputs.get("metrics", {})
        # crude heuristic: if classification accuracy < 0.6 request advanced models
        if result.outputs.get("problem_type") == "classification" and metrics.get("accuracy", 1.0) < 0.6:
            return {"retry": False, "suggestion": "Consider trying ensemble or tuning; coordinator may trigger advanced models."}
        return None
