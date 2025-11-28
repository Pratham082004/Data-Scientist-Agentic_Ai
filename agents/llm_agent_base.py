"""
LLM-backed Agent base class.

Upgraded capabilities:
- Supports mode "A" (deterministic) and mode "B" (LLM-aware agents)
- Provides THINK → PLAN → ACT → REFLECT lifecycle
- Common LLM helper (call_llm) for agents like MLAgent
- Safe error handling and retry logic
- ❗ IMPORTANT: We NO LONGER merge context keys into the plan.
  Each agent reads context only from plan["context"].
"""

from __future__ import annotations
import time
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, List

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


# ==========================================================
# RESULT WRAPPER
# ==========================================================
@dataclass
class AgentResult:
    success: bool
    outputs: Dict[str, Any] = field(default_factory=dict)
    messages: List[str] = field(default_factory=list)
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


# ==========================================================
# BASE AGENT
# ==========================================================
class LLMAgentBase:
    """
    Base class for any agent (with or without LLM help).

    Subclasses must implement:
      - think(self, context) -> str
      - plan(self, thought) -> Dict[str, Any]
      - act(self, plan, tools) -> AgentResult
      - reflect(self, result) -> Optional[Dict[str, Any]]
    """

    def __init__(
        self,
        name: str,
        prompt_template: Optional[str] = None,
        mode: str = "A",                # "A" = deterministic, "B" = LLM-aware
        max_retries: int = 2,
        retry_delay: float = 1.0,
        llm_client: Optional[Any] = None,
    ):
        self.name = name
        self.mode = (mode or "A").upper()
        if self.mode not in ("A", "B"):
            self.mode = "A"

        self.prompt_template = prompt_template or ""
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.llm_client = llm_client  # e.g., OpenRouterClient

        logger.info(
            "[%s] Initialized (mode=%s, max_retries=%d)",
            self.name,
            self.mode,
            self.max_retries,
        )

    # ------------------------------------------------------
    # LLM CALL HELPER (used by MLAgent, etc.)
    # ------------------------------------------------------
    def call_llm(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """
        Thin wrapper around self.llm_client.chat_completion(...).

        Expected response format (OpenAI / OpenRouter style):
        {
          "choices": [
            { "message": { "content": "..." } }
          ]
        }
        """
        if not self.llm_client:
            raise RuntimeError(f"LLM client not configured for agent '{self.name}'.")

        last_exc: Optional[Exception] = None

        for attempt in range(1, self.max_retries + 1):
            try:
                logger.debug(
                    "[%s] LLM call attempt %d/%d. Prompt preview: %s",
                    self.name,
                    attempt,
                    self.max_retries,
                    prompt[:200],
                )

                # Delegate to OpenRouterClient (or similar)
                resp = self.llm_client.chat_completion(
                    prompt=prompt,
                    **kwargs,
                )

                if "choices" not in resp:
                    raise RuntimeError("Invalid LLM response: missing 'choices' key.")

                return resp

            except Exception as e:
                last_exc = e
                logger.warning(
                    "[%s] LLM call failed on attempt %d/%d: %s",
                    self.name,
                    attempt,
                    self.max_retries,
                    e,
                )
                time.sleep(self.retry_delay * attempt)

        raise RuntimeError(
            f"[{self.name}] LLM call failed after {self.max_retries} attempts: {last_exc}"
        )

    # ------------------------------------------------------
    # ABSTRACT HOOKS — MUST BE OVERRIDDEN
    # ------------------------------------------------------
    def think(self, context: Dict[str, Any]) -> str:
        raise NotImplementedError()

    def plan(self, thought: str) -> Dict[str, Any]:
        raise NotImplementedError()

    def act(self, plan: Dict[str, Any], tools: Any) -> AgentResult:
        raise NotImplementedError()

    def reflect(self, result: AgentResult) -> Optional[Dict[str, Any]]:
        """
        Optional: subclasses can request retries, e.g.
        return {"retry": True, "reason": "..."}
        """
        return None

    # ------------------------------------------------------
    # MAIN ORCHESTRATION LOOP
    # ------------------------------------------------------
    def run(self, context: Dict[str, Any], tools: Any = None) -> AgentResult:
        """
        THINK → PLAN → ACT → REFLECT (+ optional retries).

        context: arbitrary dict containing things like:
                 { "data_path": "...", "dataframe": df, "target_column": "..." }

        tools:   module or object providing helper functions
                 (e.g., tools.data_tools, tools.eda_tools, tools.ml_tools)
        """
        logger.info("[%s] run() started (mode=%s)", self.name, self.mode)

        last_result: Optional[AgentResult] = None

        for attempt in range(1, self.max_retries + 1):
            try:
                # ---------------- THINK ----------------
                logger.debug("[%s] THINK — using context keys: %s", self.name, list(context.keys()))
                thought = self.think(context)
                logger.debug("[%s] Thought: %s", self.name, thought)

                # ---------------- PLAN -----------------
                plan = self.plan(thought) or {}
                if not isinstance(plan, dict):
                    raise TypeError(f"[{self.name}] plan() must return a dict, got {type(plan)}")

                # ❗ IMPORTANT: we only attach the full context under a SINGLE key.
                # We DO NOT merge context keys into top-level plan to avoid
                # collisions and weird bugs (especially for MLAgent).
                plan["context"] = context

                logger.debug("[%s] Final plan before act(): %s", self.name, plan)

                # ---------------- ACT ------------------
                result = self.act(plan, tools)
                logger.info("[%s] act() finished. success=%s", self.name, result.success)

                # ---------------- REFLECT --------------
                reflection = self.reflect(result)
                if (
                    reflection
                    and reflection.get("retry")
                    and attempt < self.max_retries
                ):
                    logger.info(
                        "[%s] Reflection requested retry (%d/%d). Reason: %s",
                        self.name,
                        attempt,
                        self.max_retries,
                        reflection.get("reason", ""),
                    )
                    last_result = result
                    time.sleep(self.retry_delay)
                    continue  # retry loop

                # No retry requested → return result
                return result

            except Exception as e:
                logger.exception(
                    "[%s] Exception in run() attempt %d: %s",
                    self.name,
                    attempt,
                    e,
                )
                last_result = AgentResult(
                    success=False,
                    error=str(e),
                    messages=[f"exception: {e}"],
                )
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay)
                    continue

        # If we exhausted retries, return the last result or a generic failure
        return last_result or AgentResult(
            success=False,
            error=f"[{self.name}] Agent failed with unknown error.",
            messages=[],
        )
