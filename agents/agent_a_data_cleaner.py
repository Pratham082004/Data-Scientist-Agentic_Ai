# agents/agent_a_data_cleaner.py
"""
Data Acquisition & Cleaning Agent (LLM-Enhanced, Mode-B)

Features:
✔ LLM-based plan generation (Mode B)
✔ Fallback to safe default plan
✔ Fully automatic CSV generation with timestamp
✔ Clean lifecycle: THINK → LLM PLAN → ACT → REFLECT
"""

from __future__ import annotations
import logging
import time
import json
from typing import Any, Dict, Optional

import pandas as pd

from .llm_agent_base import LLMAgentBase, AgentResult

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class DataCleanerAgent(LLMAgentBase):
    def __init__(self, name="DataCleaner", use_llm_plan=True, **kwargs):
        super().__init__(name=name, **kwargs)
        self.use_llm_plan = use_llm_plan   # 🔥 toggle Mode B

    # ----------------------------------------------------------
    # THINK
    # ----------------------------------------------------------
    def think(self, context: Dict[str, Any]) -> str:
        df = context.get("dataframe")
        data_path = context.get("data_path")

        if isinstance(df, pd.DataFrame):
            sample = df
        elif data_path:
            try:
                sample = pd.read_csv(data_path, nrows=80)
            except Exception:
                return "Could not sample CSV; will load fully."
        else:
            return "No dataset found."

        rows, cols = sample.shape
        dtypes = sample.dtypes.astype(str).to_dict()
        miss = sample.isnull().sum().to_dict()

        return (
            "Dataset summary:\n"
            f"- rows={rows}, cols={cols}\n"
            f"- dtypes={dtypes}\n"
            f"- missing={miss}\n"
            "The LLM must generate an optimal cleaning plan."
        )

    # ----------------------------------------------------------
    # PLAN (LLM-MODE)
    # ----------------------------------------------------------
    def plan(self, thought: str) -> Dict[str, Any]:

        if not self.use_llm_plan:
            # DEFAULT CLEANING PLAN
            return {
                "steps": [
                    {"action": "load"},
                    {"action": "drop_duplicates"},
                    {"action": "impute_numeric"},
                    {"action": "impute_categorical"},
                    {"action": "encode_categorical"},
                    {"action": "scale"},
                    {"action": "detect_outliers"},
                    {"action": "save"},
                ]
            }

        # ---------- LLM Mode B ----------
        prompt = f"""
You are an expert Data Cleaning Agent.

Given this dataset summary:

{thought}

Generate an OPTIMAL CLEANING PLAN as a JSON list of steps.
Each step must be in the EXACT format:

{{
  "action": "<one of: drop_duplicates, impute_numeric, impute_categorical, 
              encode_categorical, scale, detect_outliers, save>"
}}

DO NOT include explanations. Return ONLY valid JSON.
"""

        try:
            llm_response = self.call_llm(prompt)

            # Extract JSON
            text = llm_response.get("content") or llm_response.get("response") or ""
            steps = json.loads(text)

            return {"steps": steps}

        except Exception as e:
            logger.warning("LLM plan failed → using fallback: %s", e)

            return {
                "steps": [
                    {"action": "load"},
                    {"action": "drop_duplicates"},
                    {"action": "impute_numeric"},
                    {"action": "impute_categorical"},
                    {"action": "encode_categorical"},
                    {"action": "scale"},
                    {"action": "detect_outliers"},
                    {"action": "save"},
                ]
            }

    # ----------------------------------------------------------
    # ACT
    # ----------------------------------------------------------
    def act(self, plan: Dict[str, Any], tools: Any) -> AgentResult:
        try:
            ctx = plan.get("context", {})
            df = ctx.get("dataframe")
            data_path = ctx.get("data_path")
            messages = []

            # LOAD
            if df is None:
                if not data_path:
                    raise RuntimeError("Cleaner requires 'data_path'")
                df = tools.load_csv(data_path)
                messages.append(f"Loaded dataset from {data_path}, shape={df.shape}")

            original_shape = df.shape

            save_path = None

            # EXECUTE PLAN
            for step in plan["steps"]:
                action = step["action"]

                if action == "drop_duplicates":
                    df = tools.drop_duplicates(df)
                    messages.append("Dropped duplicates.")

                elif action == "impute_numeric":
                    df = tools.impute_numerics(df)
                    messages.append("Imputed numeric columns.")

                elif action == "impute_categorical":
                    df = tools.impute_categoricals(df)
                    messages.append("Imputed categorical columns.")

                elif action == "encode_categorical":
                    df = tools.encode_categoricals(df)
                    messages.append("Encoded categoricals.")

                elif action == "scale":
                    df = tools.scale_numerics(df)
                    messages.append("Scaled numeric columns.")

                elif action == "detect_outliers":
                    mask = tools.detect_outliers(df)
                    df["_is_outlier"] = mask
                    messages.append(f"Outliers detected: {mask.sum()} rows.")

                elif action == "save":
                    timestamp = int(time.time() * 1000)
                    save_path = f"data/processed/cleaned.csv"
                    tools.save_dataframe(df, save_path)
                    messages.append(f"Saved cleaned data → {save_path}")

                elif action == "load":
                    continue

            return AgentResult(
                success=True,
                outputs={"dataframe": df, "saved_path": save_path},
                messages=messages,
                metadata={"original_shape": original_shape, "cleaned_shape": df.shape},
            )

        except Exception as e:
            logger.exception("Cleaner failed: %s", e)
            return AgentResult(success=False, error=str(e), messages=[f"act_error: {e}"])

    # ----------------------------------------------------------
    # REFLECT
    # ----------------------------------------------------------
    def reflect(self, result: AgentResult):
        if not result.success:
            return {"retry": True, "reason": "Cleaning error → retry."}

        df = result.outputs.get("dataframe")
        if df is None or df.empty:
            return {"retry": True, "reason": "Dataframe empty after cleaning."}

        return None
