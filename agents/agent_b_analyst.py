"""
Analyst Agent — LLM Enhanced (Mode-B)

NEW FEATURES:
✔ Generates full INSIGHT REPORT (JSON)
✔ Saves insight report to outputs/insights/*.json
✔ LLM-generated EDA plan (mode B)
✔ Summary, heatmap, distributions, relationships
✔ Coordinator-friendly clean interface
✔ Report is machine-readable so AI can answer queries
"""

from __future__ import annotations
import logging
import json
from typing import Any, Dict, Optional, List

import pandas as pd
import os
from datetime import datetime

from .llm_agent_base import LLMAgentBase, AgentResult

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class AnalystAgent(LLMAgentBase):
    def __init__(self, name="Analyst", use_llm_plan=True, **kwargs):
        super().__init__(name=name, **kwargs)
        self.use_llm_plan = use_llm_plan


    # ----------------------------------------------------------
    # THINK
    # ----------------------------------------------------------
    def think(self, context: Dict[str, Any]) -> str:
        df: Optional[pd.DataFrame] = context.get("dataframe")
        data_path = context.get("data_path")

        if isinstance(df, pd.DataFrame):
            rows, cols = df.shape
            missing = int(df.isnull().sum().sum())
            return (
                f"Cleaned dataframe available.\n"
                f"rows={rows}, cols={cols}, missing={missing}\n"
                "Generate the best EDA plan."
            )

        if data_path:
            try:
                sample = pd.read_csv(data_path, nrows=80)
                rows, cols = sample.shape
                missing = int(sample.isnull().sum().sum())
                return (
                    f"Sample loaded from {data_path}. "
                    f"rows={rows}, cols={cols}, missing={missing}. "
                    "Generate EDA plan."
                )
            except Exception:
                return "Unable to preview dataset; generate generic EDA plan."

        return "No dataframe available."


    # ----------------------------------------------------------
    # PLAN — LLM MODE
    # ----------------------------------------------------------
    def plan(self, thought: str) -> Dict[str, Any]:
        if self.use_llm_plan:
            prompt = f"""
You are an expert EDA planner.

Dataset summary:
{thought}

Choose ONLY from these valid actions:
- summary_stats
- correlation_heatmap
- plot_distributions
- top_feature_relationships

Return JSON list only:
[
  {{"action": "summary_stats"}},
  {{"action": "plot_distributions"}}
]
"""

            try:
                llm_resp = self.call_llm(prompt)
                content = llm_resp.get("content") or llm_resp.get("response") or ""

                steps = json.loads(content)

                allowed = {
                    "summary_stats",
                    "correlation_heatmap",
                    "plot_distributions",
                    "top_feature_relationships",
                }
                steps = [s for s in steps if s.get("action") in allowed]

                if not steps:
                    raise ValueError("LLM returned no valid steps")

                return {
                    "steps": steps,
                    "plot_dir": "outputs/eda",
                    "top_k_relationships": 5,
                }

            except Exception as e:
                logger.warning("LLM plan failed → fallback: %s", e)

        # Fallback
        return {
            "steps": [
                {"action": "summary_stats"},
                {"action": "correlation_heatmap"},
                {"action": "plot_distributions"},
                {"action": "top_feature_relationships"},
            ],
            "plot_dir": "outputs/eda",
            "top_k_relationships": 5,
        }


    # ----------------------------------------------------------
    # INSIGHT REPORT BUILDER
    # ----------------------------------------------------------
    def _build_insight_report(
        self,
        df: pd.DataFrame,
        summary: Any,
        heatmap_path: str,
        distribution_paths: List[str],
        relationships: Any
    ) -> Dict[str, Any]:

        return {
            "overview": {
                "rows": df.shape[0],
                "columns": df.shape[1],
                "missing_values": int(df.isnull().sum().sum()),
                "columns_list": list(df.columns),
                "dtypes": df.dtypes.astype(str).to_dict(),
            },
            "summary_statistics": summary,
            "correlation_heatmap": heatmap_path,
            "distribution_plots": distribution_paths,
            "top_feature_relationships": relationships,
            "natural_language_summary": (
                f"The dataset has {df.shape[0]} rows and {df.shape[1]} columns. "
                f"{int(df.isnull().sum().sum())} values are missing. "
                f'Key variables include: {", ".join(df.columns[:5])} ...'
            )
        }


    # ----------------------------------------------------------
    # ACT
    # ----------------------------------------------------------
    def act(self, plan: Dict[str, Any], tools: Any) -> AgentResult:
        try:
            messages: List[str] = []

            # Load dataframe
            df = plan.get("context", {}).get("dataframe")
            if df is None:
                data_path = plan.get("context", {}).get("data_path") or plan.get("data_path")
                if not data_path:
                    raise RuntimeError("No dataframe or path provided.")

                df = tools.load_dataframe_from_path(data_path)
                messages.append(f"Loaded dataframe ({df.shape}).")

            plot_dir = plan.get("plot_dir", "outputs/eda")

            summary = None
            heatmap_path = None
            distribution_paths = []
            relationships = None

            # Execute EDA steps
            for step in plan["steps"]:
                action = step["action"]

                if action == "summary_stats":
                    summary = tools.summary_statistics(df)
                    messages.append("Summary statistics generated.")

                elif action == "correlation_heatmap":
                    heatmap_path = tools.correlation_heatmap(df, out_path=plot_dir)
                    messages.append(f"Correlation heatmap saved → {heatmap_path}")

                elif action == "plot_distributions":
                    distribution_paths = tools.plot_distributions(df, out_dir=plot_dir)
                    messages.append(f"Generated {len(distribution_paths)} distribution plots.")

                elif action == "top_feature_relationships":
                    relationships = tools.compute_feature_relationships(
                        df, top_k=plan.get("top_k_relationships", 5)
                    )
                    messages.append("Computed feature relationships.")

            # -------- Build insight report --------
            insight_report = self._build_insight_report(
                df=df,
                summary=summary,
                heatmap_path=heatmap_path,
                distribution_paths=distribution_paths,
                relationships=relationships
            )

            # -------- Save insight JSON file --------
            insight_dir = "outputs/insights"
            os.makedirs(insight_dir, exist_ok=True)

            timestamp = int(datetime.now().timestamp() * 1000)
            insight_path = os.path.join(insight_dir, f"insights_report.json")

            with open(insight_path, "w", encoding="utf-8") as f:
                json.dump(insight_report, f, indent=4)

            messages.append(f"Insight report saved → {insight_path}")

            # Return results
            return AgentResult(
                success=True,
                outputs={
                    "summary": summary,
                    "heatmap_path": heatmap_path,
                    "distribution_paths": distribution_paths,
                    "relationships": relationships,
                    "insights": insight_report,
                    "insights_path": insight_path,
                },
                messages=messages,
            )

        except Exception as e:
            logger.exception("AnalystAgent.act failed: %s", e)
            return AgentResult(
                success=False,
                error=str(e),
                messages=[f"act_error: {e}"],
            )


    # ----------------------------------------------------------
    # REFLECT
    # ----------------------------------------------------------
    def reflect(self, result: AgentResult) -> Optional[Dict[str, Any]]:
        if not result.success:
            return {"retry": False}
        return None
