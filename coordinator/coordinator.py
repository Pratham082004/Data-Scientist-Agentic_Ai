"""
Coordinator Agent (Hybrid Mode)
---------------------------------------------
Uses LLM only for global pipeline planning.
Agents run in deterministic (mode A) operation,
except QAAgent which can run in hybrid LLM mode.
"""

from __future__ import annotations
import os
import yaml
import logging
from typing import Any, Dict, Optional
from dotenv import load_dotenv

from agents import (
    DataCleanerAgent,
    AnalystAgent,
    MLAgent,
)
from agents.qa_agent import QAAgent  # <-- NEW

from agents.memory import EpisodicMemory, ReflexionMemory
from llm.openrouter_client import OpenRouterClient


load_dotenv()
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


# ----------------------------------------------------------
# YAML Loader
# ----------------------------------------------------------
def load_yaml(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config missing: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ----------------------------------------------------------
# Coordinator
# ----------------------------------------------------------
class Coordinator:
    def __init__(
        self,
        settings_path: str = "config/settings.yaml",
        openrouter_path: str = "config/openrouter.yaml",
    ):

        # -----------------------------
        # Load Settings
        # -----------------------------
        settings = load_yaml(settings_path)
        self.project_settings = settings.get("project", {})
        self.agent_settings = settings.get("agents", {})
        self.memory_settings = settings.get("memory", {})

        openrouter_cfg = load_yaml(openrouter_path).get("openrouter", {})

        # -----------------------------
        # LLM Client (used for pipeline planning + QAAgent)
        # -----------------------------
        self.llm_client = OpenRouterClient(
            model=os.getenv("OPENROUTER_MODEL") or openrouter_cfg.get("model"),
            base_url=openrouter_cfg.get(
                "endpoint",
                "https://openrouter.ai/api/v1/chat/completions"
            ),
        )

        # -----------------------------
        # Memory
        # -----------------------------
        self.episodic_memory = EpisodicMemory(
            max_length=self.memory_settings["episodic"]["max_length"]
        )
        self.reflexion_memory = ReflexionMemory(
            memory_path=self.memory_settings["reflexion"]["storage_path"]
        )

        # -----------------------------
        # Agents (Hybrid Mode = deterministic + QA hybrid)
        # -----------------------------
        self.agent_a = DataCleanerAgent(
            name="DataCleanerAgent",
            mode="A",
            use_llm_plan=False,
            llm_client=None,
            max_retries=self.agent_settings["common"]["max_retries"],
            retry_delay=self.agent_settings["common"]["retry_delay"],
        )

        self.agent_b = AnalystAgent(
            name="AnalystAgent",
            mode="A",
            use_llm_plan=False,
            llm_client=None,
            max_retries=self.agent_settings["common"]["max_retries"],
            retry_delay=self.agent_settings["common"]["retry_delay"],
        )

        self.agent_c = MLAgent(
            name="MLAgent",
            use_llm_plan=False,
            mode="A",
            llm_client=None,
            max_retries=self.agent_settings["common"]["max_retries"],
            retry_delay=self.agent_settings["common"]["retry_delay"],
        )

        # NEW: QA Agent (Mode B hybrid by default)
        self.agent_q = QAAgent(
            name="QAAgent",
            mode="B",                  # enable hybrid LLM refinement
            llm_client=self.llm_client,
            max_retries=self.agent_settings["common"]["max_retries"],
            retry_delay=self.agent_settings["common"]["retry_delay"],
        )

        logger.info("Coordinator initialized (Hybrid Mode).")

    # ----------------------------------------------------------
    # LLM Global Pipeline Planner
    # ----------------------------------------------------------
    def _plan_pipeline(self, request: str) -> Dict[str, Any]:

        memory_block = self.episodic_memory.get_context_block()
        lessons = self.reflexion_memory.get_recent_lessons()

        with open("llm/prompts/coordinator_prompt.txt", "r", encoding="utf-8") as f:
            system_prompt = f.read()

        prompt = f"""
User Request:
{request}

Recent Episodic Memory:
{memory_block}

Reflexion Lessons:
{lessons}

Return ONLY valid JSON:
{{
   "steps":[
       {{"agent":"cleaner"}},
       {{"agent":"analyst"}},
       {{"agent":"ml"}}
   ]
}}
"""

        try:
            resp = self.llm_client.chat_completion(
                system_prompt=system_prompt,
                prompt=prompt,
                max_tokens=300,
                temperature=0.2
            )
            content = resp["choices"][0]["message"]["content"]

            import json
            plan = json.loads(content)

            if "steps" not in plan:
                raise ValueError("Invalid LLM plan")

            return plan

        except Exception as e:
            logger.warning("LLM planning failed → using fallback. Error=%s", e)
            return {
                "steps": [
                    {"agent": "cleaner"},
                    {"agent": "analyst"},
                    {"agent": "ml"},
                ]
            }

    # ----------------------------------------------------------
    # Main Execution Flow
    # ----------------------------------------------------------
    def run(self, request: str, dataset_path: str, target_column: Optional[str] = None):

        logger.info("Coordinator: starting workflow")

        context = {
            "data_path": dataset_path,
            "target_column": target_column
        }

        plan = self._plan_pipeline(request)
        steps = plan.get("steps", [])

        final_outputs = {}
        df_cache = None

        for step in steps:
            agent_key = step["agent"]

            # ---------------- CLEANER ----------------
            if agent_key == "cleaner":
                res = self.agent_a.run(context, tools=self._load_data_tools())
                final_outputs["cleaner"] = res
                self._record_memory("DataCleanerAgent", res)

                if res.success:
                    df_cache = res.outputs.get("dataframe")
                    context["dataframe"] = df_cache

            # ---------------- ANALYST ----------------
            elif agent_key == "analyst":
                if df_cache is not None:
                    context["dataframe"] = df_cache

                res = self.agent_b.run(context, tools=self._load_eda_tools())
                final_outputs["analyst"] = res
                self._record_memory("AnalystAgent", res)

            # ---------------- ML ----------------
            elif agent_key == "ml":
                if df_cache is not None:
                    context["dataframe"] = df_cache

                res = self.agent_c.run(context, tools=self._load_ml_tools())
                final_outputs["ml"] = res
                self._record_memory("MLAgent", res)

        logger.info("Coordinator: workflow complete.")
        return final_outputs

    # ----------------------------------------------------------
    # Q&A Entry Point (NEW)
    # ----------------------------------------------------------
    def answer_question(self, question: str):
        """
        Uses QAAgent over the already-generated insight JSONs.
        Does NOT rerun the full pipeline.
        """
        logger.info("Coordinator: QAAgent answering question.")
        ctx = {"question": question}
        res = self.agent_q.run(ctx, tools=None)
        # We don't record QA into episodic memory by default, but you could.
        return res

    # ----------------------------------------------------------
    # Memory Recorder
    # ----------------------------------------------------------
    def _record_memory(self, agent_name: str, result):
        self.episodic_memory.add_episode(
            agent_name=agent_name,
            thought=result.messages[0] if result.messages else "",
            plan=None,
            action_summary="; ".join(result.messages),
            success=result.success,
        )

        if not result.success:
            self.reflexion_memory.add_entry(
                agent_name,
                {
                    "success": False,
                    "error": result.error,
                    "lesson": "Fix preprocessing / planning in next iteration.",
                },
            )

    # ----------------------------------------------------------
    # Tool Loaders
    # ----------------------------------------------------------
    def _load_data_tools(self):
        from tools import data_tools
        return data_tools

    def _load_eda_tools(self):
        from tools import eda_tools
        return eda_tools

    def _load_ml_tools(self):
        from tools import ml_tools
        return ml_tools
