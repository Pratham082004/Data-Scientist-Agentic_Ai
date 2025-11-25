# agents/agent_a_data_cleaner.py
"""
Data Acquisition & Cleaning Agent.

Responsibilities:
- Load data (delegates to tools.data_tools.load_csv or load_from_source)
- Inspect schema and statistics
- Decide cleaning strategy via think->plan using the LLM base
- Execute cleaning actions using tools.data_tools helpers
- Return cleaned dataframe path or object and a standardized report

This agent extends LLMAgentBase and implements the lifecycle methods.
"""

from __future__ import annotations
import logging
from dataclasses import asdict
from typing import Any, Dict, Optional

import pandas as pd

from .llm_agent_base import LLMAgentBase, AgentResult

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class DataCleanerAgent(LLMAgentBase):
    """
    Agent A: handles loading and cleaning datasets.
    Expected keys in context:
      - 'dataset_path' (str): path to a CSV file OR
      - 'dataframe' (pd.DataFrame): an in-memory DataFrame
      - 'target_column' (Optional[str])
      - 'config' (dict) optional agent configs
    """

    def __init__(self, name: str = "DataCleaner", **kwargs):
        super().__init__(name=name, **kwargs)

    # ---- Think: short summary of dataset and objectives ----
    def think(self, context: Dict[str, Any]) -> str:
        # Build a short dataset summary to feed LLM if needed
        if "dataframe" in context and isinstance(context["dataframe"], pd.DataFrame):
            df = context["dataframe"]
        elif "dataset_path" in context:
            try:
                df = pd.read_csv(context["dataset_path"], nrows=100)  # sample to summarize
            except Exception:
                # minimal fallback
                return "Unable to sample dataset — will request the full file load."
        else:
            return "No dataset provided."

        # quick heuristics summary
        num_rows, num_cols = df.shape
        dtypes = df.dtypes.apply(lambda t: t.name).to_dict()
        missing = df.isnull().sum().to_dict()

        thought = (
            f"Dataset sample: rows={num_rows}, cols={num_cols}. "
            f"dtypes={dtypes}. missing_counts_sample={missing}."
        )
        return thought

    # ---- Plan: decide cleaning steps ----
    def plan(self, thought: str) -> Dict[str, Any]:
        # We keep plan simple here; in production you might call the LLM:
        # prompt = f"<SYSTEM ROLE> You are DataCleaner. Given: {thought}. Produce a stepwise cleaning plan."
        # response = self.call_llm(prompt)
        # parse plan from response
        # For now, produce a conservative default plan that can be overridden by LLM later.
        plan = {
            "steps": [
                {"action": "load_full"},
                {"action": "drop_duplicates"},
                {"action": "impute_numeric_mean"},
                {"action": "impute_categorical_mode"},
                {"action": "encode_categoricals"},
                {"action": "scale_numerics_optional"},
                {"action": "detect_outliers_mark"},
            ],
            "notes": "Default conservative plan. Replace by LLM suggested plan if available.",
        }
        return plan

    # ---- Act: execute cleaning steps using tools.data_tools ----
    def act(self, plan: Dict[str, Any], tools: Any) -> AgentResult:
        """
        tools expected to provide:
          - load_csv(path) -> pd.DataFrame
          - save_dataframe(df, path)
          - drop_duplicates(df)
          - impute_numerics(df, strategy="mean")
          - impute_categoricals(df, strategy="mode")
          - encode_categoricals(df)
          - scale_numerics(df, method="standard")   # optional
          - detect_outliers(df, method="iqr")
        """
        try:
            # load dataset
            df = None
            if "dataframe" in plan.get("context", {}):
                df = plan["context"]["dataframe"]
            # fallback to context-driven load (tools should be provided)
            if df is None:
                if tools is None or not hasattr(tools, "load_csv"):
                    raise RuntimeError("tools.load_csv required but not provided to DataCleanerAgent.act")
                df = tools.load_csv(plan.get("dataset_path") or plan.get("context", {}).get("dataset_path"))

            original_shape = df.shape
            messages = [f"Loaded dataset with shape {original_shape}"]

            # Execute the sequence of steps defined in plan["steps"]
            for step in plan.get("steps", []):
                action = step.get("action")
                if action == "drop_duplicates":
                    if hasattr(tools, "drop_duplicates"):
                        df = tools.drop_duplicates(df)
                        messages.append("Dropped duplicates.")
                elif action == "impute_numeric_mean":
                    if hasattr(tools, "impute_numerics"):
                        df = tools.impute_numerics(df, strategy="mean")
                        messages.append("Imputed numeric columns with mean.")
                elif action == "impute_categorical_mode":
                    if hasattr(tools, "impute_categoricals"):
                        df = tools.impute_categoricals(df, strategy="mode")
                        messages.append("Imputed categorical columns with mode.")
                elif action == "encode_categoricals":
                    if hasattr(tools, "encode_categoricals"):
                        df = tools.encode_categoricals(df)
                        messages.append("Encoded categorical columns.")
                elif action == "scale_numerics_optional":
                    if hasattr(tools, "scale_numerics"):
                        # scaling is optional and controlled by config in plan
                        if plan.get("apply_scaling", True):
                            df = tools.scale_numerics(df, method=plan.get("scaler", "standard"))
                            messages.append("Scaled numeric columns.")
                elif action == "detect_outliers_mark":
                    if hasattr(tools, "detect_outliers"):
                        outlier_mask = tools.detect_outliers(df)
                        messages.append(f"Detected outliers; mask length={len(outlier_mask)}")
                        # optionally annotate
                        df["_is_outlier"] = outlier_mask
                elif action == "load_full":
                    # already loaded above
                    continue
                else:
                    messages.append(f"Unknown action '{action}' skipped.")

            # Save processed dataframe if tools provide save capability
            saved_path = None
            if tools and hasattr(tools, "save_dataframe"):
                # choose a default path or use plan/context
                saved_path = plan.get("save_path") or getattr(tools, "default_processed_path", None)
                if saved_path:
                    tools.save_dataframe(df, saved_path)
                    messages.append(f"Saved cleaned dataframe to {saved_path}")

            result = AgentResult(
                success=True,
                outputs={"dataframe": df, "saved_path": saved_path},
                messages=messages,
                metadata={"original_shape": original_shape, "cleaned_shape": df.shape},
            )
            return result

        except Exception as e:
            logger.exception("DataCleanerAgent.act failed: %s", e)
            return AgentResult(success=False, error=str(e), messages=[f"act_error: {e}"])

    # ---- Reflect: quick validation and possible retries ----
    def reflect(self, result: AgentResult) -> Optional[Dict[str, Any]]:
        if not result.success:
            # try to provide an automatic mitigation
            return {"retry": True, "reason": "cleaning failed; will attempt simpler imputation."}
        # Basic validation: ensure no empty dataset
        df = result.outputs.get("dataframe")
        if df is None or getattr(df, "shape", (0, 0))[0] == 0:
            return {"retry": True, "reason": "cleaned dataframe empty."}
        return None
