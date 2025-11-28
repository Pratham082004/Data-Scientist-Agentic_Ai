"""
QAAgent — Hybrid EDA + ML Q&A (Mode C)
--------------------------------------

Reads:
  - outputs/ml/ml_insights_report.json
  - outputs/insights/insights_report.json

Capabilities:
  ✔ Loads EDA + ML JSON insight reports
  ✔ Pure-Python factual answers for:
      - best model, scores, target, shapes
      - CV scores, feature importances, sample preds
      - dataset overview, summary statistics
      - top feature relationships & correlations
  ✔ LLM / Hybrid Mode (if llm_client is configured):
      - High-level "why / how / explain / interpret" questions
      - Combines raw facts + natural language answer
  ✔ Safe: if JSON missing or incomplete, responds gracefully
"""

from __future__ import annotations
import json
import logging
import os
from typing import Any, Dict, Optional, List

from .llm_agent_base import LLMAgentBase, AgentResult

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class QAAgent(LLMAgentBase):
    def __init__(self, name: str = "QAAgent", **kwargs):
        """
        kwargs are passed to LLMAgentBase:
            - mode: "A" (python only) or "B" (hybrid with LLM)
            - llm_client: OpenRouterClient or None
            - max_retries, retry_delay, ...
        """
        super().__init__(name=name, **kwargs)

    # ============================================================
    # Public entrypoint (override run)
    # ============================================================
    def run(self, context: Dict[str, Any], tools: Any = None) -> AgentResult:
        """
        context:
          {
            "question": "<user question>",
            # optional overrides:
            "ml_path": "path/to/ml_insights_report.json",
            "eda_path": "path/to/insights_report.json",
            "ml_insights": {...},   # preloaded ML json
            "eda_insights": {...},  # preloaded EDA json
          }
        """
        messages: List[str] = []

        question: Optional[str] = context.get("question")
        if not question:
            return AgentResult(
                success=False,
                error="QAAgent: 'question' missing from context.",
                messages=["No question provided to QAAgent."],
            )

        # 1) Load reports
        ml_insights, eda_insights, ml_path, eda_path = self._load_reports(context, messages)

        # 2) Try pure-Python answer
        answer, used_mode = self._answer_python(question, ml_insights, eda_insights)

        # 3) If needed and possible, use LLM / hybrid refinement
        if self.llm_client:
            # If no answer at all → full LLM answer
            if answer is None:
                llm_answer = self._answer_with_llm(question, ml_insights, eda_insights)
                if llm_answer:
                    answer = llm_answer
                    used_mode = "llm"
            # If we already have a factual answer and in mode B → hybrid refinement
            elif self.mode == "B":
                refined = self._refine_with_llm(question, answer, ml_insights, eda_insights)
                if refined:
                    answer = refined
                    used_mode = "hybrid"

        if answer is None:
            answer = (
                "I couldn't derive a clear answer from the current EDA and ML insight reports. "
                "You may need to regenerate the insights or clarify your question."
            )
            used_mode = used_mode or "unknown"

        messages.append(f"QA used mode: {used_mode}")

        return AgentResult(
            success=True,
            outputs={
                "answer": answer,
                "used_mode": used_mode,
                "sources": {
                    "ml_path": ml_path,
                    "eda_path": eda_path,
                },
            },
            messages=messages,
        )

    # ============================================================
    # JSON loading
    # ============================================================
    def _load_reports(
        self,
        context: Dict[str, Any],
        messages: List[str],
    ):
        # Defaults
        default_ml_path = os.path.join("outputs", "ml", "ml_insights_report.json")
        default_eda_path = os.path.join("outputs", "insights", "insights_report.json")

        ml_insights = context.get("ml_insights")
        eda_insights = context.get("eda_insights")

        ml_path = context.get("ml_path") or default_ml_path
        eda_path = context.get("eda_path") or default_eda_path

        # Load ML insights
        if ml_insights is None:
            try:
                with open(ml_path, "r", encoding="utf-8") as f:
                    ml_insights = json.load(f)
                messages.append(f"Loaded ML insights from {ml_path}")
            except Exception as e:
                messages.append(f"Failed to load ML insights from {ml_path}: {e}")
                ml_insights = {}

        # Load EDA insights
        if eda_insights is None:
            try:
                with open(eda_path, "r", encoding="utf-8") as f:
                    eda_insights = json.load(f)
                messages.append(f"Loaded EDA insights from {eda_path}")
            except Exception as e:
                messages.append(f"Failed to load EDA insights from {eda_path}: {e}")
                eda_insights = {}

        return ml_insights, eda_insights, ml_path, eda_path

    # ============================================================
    # Pure-Python QA routing
    # ============================================================
    def _answer_python(
        self,
        question: str,
        ml: Dict[str, Any],
        eda: Dict[str, Any],
    ):
        """
        Returns (answer_str or None, used_mode_str or None)
        """
        q = question.lower()

        # ---------- 1. ML-focused questions ----------
        ml_ins = ml if isinstance(ml, dict) else {}
        ml_target = ml_ins.get("target") or ml_ins.get("target_column")
        ml_problem_type = ml_ins.get("problem_type")
        ml_scores = ml_ins.get("scores") or ml_ins.get("cv_scores")
        ml_best = ml_ins.get("best_model") or {
            "name": ml_ins.get("best_model_name"),
            "score": ml_ins.get("best_score"),
        }

        # best model
        if "best model" in q or "which model" in q:
            if ml_best and ml_best.get("name") is not None:
                name = ml_best.get("name")
                score = ml_best.get("score") or ml_ins.get("best_score")
                return (
                    f"The best model selected was **{name}** with a score of **{score:.4f}**."
                    if isinstance(score, (int, float))
                    else f"The best model selected was **{name}**."
                ), "python"

        # cv scores
        if ("cv" in q or "cross validation" in q or "cross-validation" in q) and ("score" in q or "scores" in q):
            if ml_scores:
                lines = ["Cross-validation scores by model:"]
                for mname, s in ml_scores.items():
                    lines.append(f" - {mname}: {s:.4f}" if isinstance(s, (int, float)) else f" - {mname}: {s}")
                return "\n".join(lines), "python"

        # target
        if "target" in q:
            if ml_target:
                return f"The target column used for modeling is **{ml_target}**.", "python"

        # problem type
        if "classification" in q or "regression" in q or "problem type" in q:
            if ml_problem_type:
                return f"The problem was inferred as a **{ml_problem_type}** task.", "python"

        # feature importances
        if ("feature importance" in q) or ("important features" in q) or ("top features" in q):
            fi = ml_ins.get("feature_importances")
            if isinstance(fi, list) and fi:
                top_k = fi[:10]
                lines = ["Top features by importance:"]
                for feat, val in top_k:
                    if isinstance(val, (int, float)):
                        lines.append(f" - {feat}: {val:.4f}")
                    else:
                        lines.append(f" - {feat}: {val}")
                return "\n".join(lines), "python"

        # sample predictions
        if "sample prediction" in q or "example prediction" in q or "predicted vs actual" in q:
            sp = ml_ins.get("sample_predictions")
            if isinstance(sp, list) and sp:
                lines = ["Example predictions (true vs predicted):"]
                for entry in sp[:10]:
                    true = entry.get("true")
                    pred = entry.get("pred")
                    lines.append(f" - true: {true}, predicted: {pred}")
                return "\n".join(lines), "python"

        # ---------- 2. EDA-focused questions ----------
        eda_ins = eda if isinstance(eda, dict) else {}
        overview = eda_ins.get("overview", {})
        rows = overview.get("rows")
        cols = overview.get("columns")
        dtypes = overview.get("dtypes")
        missing_values = overview.get("missing_values")
        nat_summary = eda_ins.get("natural_language_summary")

        # dataset shape / rows / columns
        if "shape" in q or ("how many" in q and "row" in q) or ("how many" in q and "record" in q):
            if rows is not None and cols is not None:
                return f"The dataset contains **{rows} rows** and **{cols} columns**.", "python"

        if "column" in q and ("type" in q or "dtype" in q):
            if dtypes:
                lines = ["Column data types:"]
                for c, t in dtypes.items():
                    lines.append(f" - {c}: {t}")
                return "\n".join(lines), "python"

        if "missing" in q or "null" in q or "na" in q:
            if missing_values is not None:
                return f"The dataset has **{missing_values} missing values** in total.", "python"

        # summary / overview request
        if "summary" in q or "overview" in q or "describe" in q:
            if nat_summary:
                return nat_summary, "python"

        # top relationships / correlations
        if "relationship" in q or "correlation" in q:
            rel_block = eda_ins.get("top_feature_relationships") or eda_ins.get("top_relationships")
            # we expect {'top_relationships': [ ... ]} OR a list
            top_rels = []
            if isinstance(rel_block, dict) and "top_relationships" in rel_block:
                top_rels = rel_block["top_relationships"]
            elif isinstance(rel_block, list):
                top_rels = rel_block

            if isinstance(top_rels, list) and top_rels:
                # if question mentions two features by name
                q_lower = q.lower()
                for rel in top_rels:
                    f1 = str(rel.get("feature_1", "")).lower()
                    f2 = str(rel.get("feature_2", "")).lower()
                    corr = rel.get("correlation")
                    if f1 and f2 and f1 in q_lower and f2 in q_lower and isinstance(corr, (int, float)):
                        return (
                            f"The correlation between **{rel['feature_1']}** and **{rel['feature_2']}** "
                            f"is approximately **{corr:.4f}**.",
                            "python",
                        )

                # otherwise just list top relationships
                lines = ["Top feature relationships:"]
                for rel in top_rels[:5]:
                    f1 = rel.get("feature_1")
                    f2 = rel.get("feature_2")
                    corr = rel.get("correlation")
                    if f1 and f2 and isinstance(corr, (int, float)):
                        lines.append(f" - {f1} vs {f2}: correlation={corr:.4f}")
                if len(lines) > 1:
                    return "\n".join(lines), "python"

        # If nothing matched
        return None, None

    # ============================================================
    # LLM helpers
    # ============================================================
    def _answer_with_llm(
        self,
        question: str,
        ml: Dict[str, Any],
        eda: Dict[str, Any],
    ) -> Optional[str]:
        """
        Use the LLM to answer directly from JSON insights.
        """
        if not self.llm_client:
            return None

        try:
            ml_snip = json.dumps(ml or {}, default=str)[:5000]
            eda_snip = json.dumps(eda or {}, default=str)[:5000]

            prompt = f"""
You are a data scientist assistant.

You are given:
1) ML insight JSON (model performance, feature importances, etc.)
2) EDA insight JSON (overview, summary statistics, correlations, etc.)

Use ONLY these insights to answer the user's question.
Be concise, factual, and explain reasoning in simple English.

ML_INSIGHTS_JSON:
{ml_snip}

EDA_INSIGHTS_JSON:
{eda_snip}

USER_QUESTION:
{question}

Return ONLY the answer, no JSON, no extra labels.
"""

            resp = self.call_llm(prompt, max_tokens=400, temperature=0.2)
            content = resp["choices"][0]["message"]["content"].strip()
            return content or None
        except Exception as e:
            logger.warning("QAAgent LLM answer failed: %s", e)
            return None

    def _refine_with_llm(
        self,
        question: str,
        base_answer: str,
        ml: Dict[str, Any],
        eda: Dict[str, Any],
    ) -> Optional[str]:
        """
        Hybrid mode: give the LLM a base factual answer and ask it to refine / expand.
        """
        if not self.llm_client:
            return None

        try:
            prompt = f"""
You are a data scientist assistant.

We already extracted a factual answer from structured ML + EDA insights.
Your job is to polish and slightly expand that answer:
- keep it correct and grounded in data
- keep it concise and user-friendly
- you MAY rephrase and lightly elaborate, but do not invent new metrics

USER QUESTION:
{question}

BASE FACTUAL ANSWER:
{base_answer}

Return the improved answer only.
"""

            resp = self.call_llm(prompt, max_tokens=300, temperature=0.3)
            content = resp["choices"][0]["message"]["content"].strip()
            return content or None
        except Exception as e:
            logger.warning("QAAgent LLM refinement failed: %s", e)
            return None

    # ============================================================
    # THINK / PLAN / ACT / REFLECT (not used in Mode C)
    # ============================================================
    def think(self, context: Dict[str, Any]) -> str:
        return "QAAgent uses a direct run(context) hybrid flow; THINK is not used."

    def plan(self, thought: str) -> Dict[str, Any]:
        return {}

    def act(self, plan: Dict[str, Any], tools: Any) -> AgentResult:
        ctx = plan.get("context", {})
        return self.run(ctx, tools)

    def reflect(self, result: AgentResult):
        if not result.success:
            return {"retry": False, "reason": result.error}
        return None
