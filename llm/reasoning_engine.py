# llm/reasoning_engine.py
"""
Reasoning Engine
----------------

Utility helpers for LLM-based:
- THOUGHT generation
- PLAN creation
- ACTION suggestion

This is optional because each agent already has its own think/plan/act cycle,
but you can use this module when you want a central reasoning pipeline
or need more complex meta-planning.
"""

from __future__ import annotations
from typing import Dict, Any, Optional
import json
import logging

from .openrouter_client import OpenRouterClient

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class ReasoningEngine:
    def __init__(self, llm_client: Optional[OpenRouterClient] = None, system_prompt: Optional[str] = None):
        self.llm_client = llm_client or OpenRouterClient()
        self.system_prompt = system_prompt or (
            "You are a helpful reasoning engine that breaks problems into steps, "
            "creates clear plans, and suggests safe tool actions in JSON."
        )

    def think_and_plan(self, task_description: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Produces a JSON plan for given task description.
        The plan structure is left flexible on purpose; typical format:

        {
          "thought": "...",
          "plan": [
            {"step": 1, "action": "clean_data", "details": "..."},
            {"step": 2, "action": "run_eda"},
            {"step": 3, "action": "train_model"}
          ]
        }
        """
        context_text = json.dumps(context, indent=2) if context else "{}"

        prompt = f"""
Task:
{task_description}

Context:
{context_text}

You MUST:
1. Think step-by-step.
2. Output a valid JSON object with keys: "thought" and "plan".
3. "plan" must be a list of steps with "action" and optional "details".

Respond with JSON only, nothing else.
"""

        try:
            resp = self.llm_client.chat_completion(
                prompt=prompt,
                system_prompt=self.system_prompt,
                max_tokens=400,
                temperature=0.2,
            )
            content = resp["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error("ReasoningEngine LLM error: %s", e)
            # minimal fallback
            return {
                "thought": "Falling back to default plan due to LLM error.",
                "plan": [
                    {"step": 1, "action": "clean_data"},
                    {"step": 2, "action": "run_eda"},
                    {"step": 3, "action": "train_model"},
                ],
            }

        try:
            plan_json = json.loads(content)
            return plan_json
        except Exception as e:
            logger.error("Failed to parse ReasoningEngine JSON: %s | content: %s", e, content)
            return {
                "thought": "LLM returned invalid JSON. Using default fallback.",
                "plan": [
                    {"step": 1, "action": "clean_data"},
                    {"step": 2, "action": "run_eda"},
                    {"step": 3, "action": "train_model"},
                ],
            }
