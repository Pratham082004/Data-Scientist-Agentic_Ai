# agents/agent_b_analyst.py
"""
Data Analyst & Visualization Agent.

Responsibilities:
- Receive cleaned dataframe
- Produce summary statistics
- Produce correlation matrix and EDA plots via tools.eda_tools
- Highlight feature-target relationships
- Return structured insights and file paths to generated plots
"""

from __future__ import annotations
import logging
from typing import Any, Dict, Optional, List

import pandas as pd

from .llm_agent_base import LLMAgentBase, AgentResult

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class AnalystAgent(LLMAgentBase):
    def __init__(self, name: str = "Analyst", **kwargs):
        super().__init__(name=name, **kwargs)

    def think(self, context: Dict[str, Any]) -> str:
        # Produce a short observation of what's expected
        df: Optional[pd.DataFrame] = context.get("dataframe")
        if df is None and "dataset_path" in context:
            return "No dataframe provided in context; will request load from dataset_path."
        if df is None:
            return "No data provided."

        num_rows, num_cols = df.shape
        num_missing = int(df.isnull().sum().sum())
        thought = (
            f"Dataframe ready: rows={num_rows}, cols={num_cols}, total_missing={num_missing}. "
            "Plan EDA focusing on distributions, correlations, and top feature-target relationships."
        )
        return thought

    def plan(self, thought: str) -> Dict[str, Any]:
        # Default plan; LLM can override by calling call_llm if desired
        plan = {
            "steps": [
                {"action": "summary_stats"},
                {"action": "correlation_heatmap"},
                {"action": "plot_distributions"},
                {"action": "top_feature_relationships"},
            ],
            "plot_dir": "outputs/eda",
            "top_k_relationships": 5,
        }
        return plan

    def act(self, plan: Dict[str, Any], tools: Any) -> AgentResult:
        """
        tools expected to have:
         - ensure_dataframe_loaded(context or path) -> pd.DataFrame
         - summary_statistics(df) -> dict (or DataFrame)
         - save_plot(fig, path) or plotting helpers returning file paths
         - correlation_heatmap(df, out_path)
         - plot_distributions(df, out_dir, columns=None)
         - compute_feature_importance_insights(df, target, k)
        """
        try:
            df = plan.get("context", {}).get("dataframe")
            if df is None:
                if tools is None or not hasattr(tools, "load_dataframe_from_path"):
                    raise RuntimeError("tools.load_dataframe_from_path required for AnalystAgent")
                ds_path = plan.get("dataset_path") or plan.get("context", {}).get("dataset_path")
                df = tools.load_dataframe_from_path(ds_path)

            messages: List[str] = []

            # summary statistics
            summary = None
            if hasattr(tools, "summary_statistics"):
                summary = tools.summary_statistics(df)
                messages.append("Generated summary statistics.")
            else:
                summary = df.describe(include="all").to_dict()
                messages.append("Fallback summary via pandas.describe")

            # correlation heatmap
            heatmap_path = None
            if hasattr(tools, "correlation_heatmap"):
                heatmap_path = tools.correlation_heatmap(df, out_path=plan.get("plot_dir"))
                messages.append(f"Saved correlation heatmap to {heatmap_path}")

            # distributions
            distribution_paths = []
            if hasattr(tools, "plot_distributions"):
                distribution_paths = tools.plot_distributions(df, out_dir=plan.get("plot_dir"))
                messages.append(f"Plotted distributions, count={len(distribution_paths)}")
            else:
                messages.append("No distribution plotting tools available; skipped.")

            # top relationships
            relationships = None
            if hasattr(tools, "compute_feature_relationships"):
                relationships = tools.compute_feature_relationships(df, top_k=plan.get("top_k_relationships", 5))
                messages.append("Computed top feature relationships.")
            else:
                messages.append("No compute_feature_relationships in tools; skipped.")

            outputs = {
                "summary": summary,
                "heatmap_path": heatmap_path,
                "distribution_paths": distribution_paths,
                "relationships": relationships,
            }
            return AgentResult(success=True, outputs=outputs, messages=messages)

        except Exception as e:
            logger.exception("AnalystAgent.act failed: %s", e)
            return AgentResult(success=False, error=str(e), messages=[f"act_error: {e}"])

    def reflect(self, result: AgentResult) -> Optional[Dict[str, Any]]:
        if not result.success:
            return {"retry": True, "reason": "EDA tools failed; retry with smaller sample."}
        # If no heatmap was produced, perhaps regenerate with numeric-only columns
        if not result.outputs.get("heatmap_path"):
            return {"retry": True, "reason": "Heatmap not produced; attempt numeric-only heatmap."}
        return None
