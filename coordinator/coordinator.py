# coordinator/coordinator.py
"""
Coordinator Agent
------------------
The central orchestrator for the multi-agent Data Scientist Agentic AI system.

Key features:
- Loads config from YAML + .env (with env overrides)
- Uses OpenRouter model specified in .env or openrouter.yaml
- Delegates tasks to: DataCleanerAgent → AnalystAgent → MLAgent
- Tracks episodic memory & reflexion memory
- Optional LLM-driven pipeline planning
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
from agents.memory import EpisodicMemory, ReflexionMemory

from llm.openrouter_client import OpenRouterClient


# Load .env early
load_dotenv()

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


# ----------------------------------------------------------------------
# Helper: Load YAML config
# ----------------------------------------------------------------------
def load_yaml(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class Coordinator:
    def __init__(
        self,
        settings_path: str = "config/settings.yaml",
        openrouter_path: str = "config/openrouter.yaml",
    ):

        # ------------------------------------------------------------------
        # Load YAML configs
        # ------------------------------------------------------------------
        self.settings = load_yaml(settings_path).get("project", {})
        self.agent_settings = load_yaml(settings_path).get("agents", {})
        self.memory_settings = load_yaml(settings_path).get("memory", {})

        openrouter_cfg = load_yaml(openrouter_path).get("openrouter", {})

        # ------------------------------------------------------------------
        # Initialize LLM Client (OpenRouter)
        # ------------------------------------------------------------------
        self.llm_client = OpenRouterClient(
            model=os.getenv("OPENROUTER_MODEL") or openrouter_cfg.get("model"),
            base_url=openrouter_cfg.get("endpoint", "https://openrouter.ai/api/v1/chat/completions"),
        )

        # ------------------------------------------------------------------
        # Setup Memory Systems
        # ------------------------------------------------------------------
        self.episodic_memory = EpisodicMemory(
            max_length=self.memory_settings["episodic"]["max_length"]
        )

        self.reflexion_memory = ReflexionMemory(
            memory_path=self.memory_settings["reflexion"]["storage_path"]
        )

        # ------------------------------------------------------------------
        # Initialize Agents with shared LLM client
        # ------------------------------------------------------------------
        self.agent_a = DataCleanerAgent(
            llm_client=self.llm_client,
            max_retries=self.agent_settings["common"]["max_retries"],
            retry_delay=self.agent_settings["common"]["retry_delay"],
        )

        self.agent_b = AnalystAgent(
            llm_client=self.llm_client,
            max_retries=self.agent_settings["common"]["max_retries"],
            retry_delay=self.agent_settings["common"]["retry_delay"],
        )

        self.agent_c = MLAgent(
            llm_client=self.llm_client,
            max_retries=self.agent_settings["common"]["max_retries"],
            retry_delay=self.agent_settings["common"]["retry_delay"],
        )

        logger.info("Coordinator initialized successfully.")

    # ----------------------------------------------------------------------
    # Optional: LLM-based pipeline planning
    # ----------------------------------------------------------------------
    def _plan_pipeline(self, request: str) -> Dict[str, Any]:
        """
        Uses LLM to determine workflow pipeline.
        Returns a JSON dict with "steps": [...]
        """

        memory_block = self.episodic_memory.get_context_block()
        lessons = self.reflexion_memory.get_recent_lessons()

        # Use prompt file from llm/prompts
        with open("llm/prompts/coordinator_prompt.txt", "r", encoding="utf-8") as f:
            system_prompt = f.read()

        prompt = f"""
User Request:
{request}

Recent Episodic Memory:
{memory_block}

Reflexion Lessons:
{lessons}

Respond with ONLY valid JSON.
"""

        try:
            resp = self.llm_client.chat_completion(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=0.2,
                max_tokens=300,
            )
            content = resp["choices"][0]["message"]["content"]
            import json
            plan = json.loads(content)

            if "steps" not in plan:
                raise ValueError("Invalid LLM response: missing 'steps'")
            return plan

        except Exception as e:
            logger.warning("LLM planning failed (%s). Using fallback order.", e)
            return {
                "steps": [
                    {"agent": "cleaner"},
                    {"agent": "analyst"},
                    {"agent": "ml"},
                ]
            }

    # ----------------------------------------------------------------------
    # Main Pipeline Runner
    # ----------------------------------------------------------------------
    def run(self, request: str, dataset_path: str, target_column: Optional[str] = None):
        """
        Execute multi-agent workflow.
        """

        logger.info("Coordinator starting workflow")
        context = {
            "dataset_path": dataset_path,
            "target_column": target_column,
        }

        # LLM-based or fallback plan
        plan = self._plan_pipeline(request)
        steps = plan.get("steps", [])

        final_outputs = {}
        df_cache = None

        # --------------------------------------------------------------
        # Execute agents in LLM-determined order
        # --------------------------------------------------------------
        for step in steps:
            agent_key = step.get("agent")

            if agent_key == "cleaner":
                logger.info("Running DataCleanerAgent")
                res = self.agent_a.run(context, tools=self._load_data_tools())
                self._record_memory("DataCleanerAgent", res)
                if res.success:
                    df_cache = res.outputs.get("dataframe")
                    context["dataframe"] = df_cache
                final_outputs["cleaner"] = res

            elif agent_key == "analyst":
                logger.info("Running AnalystAgent")
                if df_cache is not None:
                    context["dataframe"] = df_cache
                res = self.agent_b.run(context, tools=self._load_eda_tools())
                self._record_memory("AnalystAgent", res)
                final_outputs["analyst"] = res

            elif agent_key == "ml":
                logger.info("Running MLAgent")
                if df_cache is not None:
                    context["dataframe"] = df_cache
                res = self.agent_c.run(context, tools=self._load_ml_tools())
                self._record_memory("MLAgent", res)
                final_outputs["ml"] = res

        logger.info("Workflow completed.")
        return final_outputs

    # ----------------------------------------------------------------------
    # Memory Recorder
    # ----------------------------------------------------------------------
    def _record_memory(self, agent_name: str, result):
        # Episodic Memory
        self.episodic_memory.add_episode(
            agent_name=agent_name,
            thought=result.messages[0] if result.messages else "",
            plan=None,
            action_summary="; ".join(result.messages),
            success=result.success,
        )

        # Reflexion Memory
        if not result.success:
            self.reflexion_memory.add_entry(
                agent_name,
                {
                    "success": False,
                    "error": result.error,
                    "lesson": "Consider adjusting cleaning or preprocessing strategy.",
                },
            )

    # ----------------------------------------------------------------------
    # Lazy Tool Loaders
    # ----------------------------------------------------------------------
    def _load_data_tools(self):
        from tools import data_tools
        return data_tools

    def _load_eda_tools(self):
        from tools import eda_tools
        return eda_tools

    def _load_ml_tools(self):
        from tools import ml_tools
        return ml_tools
