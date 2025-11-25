# agents/llm_agent_base.py
"""
LLM-backed Agent base class.

Responsibilities:
- Provide a consistent interface for LLM calls (call_llm)
- Provide a THINK -> PLAN -> ACT -> REFLECT lifecycle
- Provide structured logging and safe error handling
- Provide a placeholder integration with llm.openrouter_client API wrapper

Note: An implementation of llm.openrouter_client.openrouter_chat_completion(...) is expected
to be available in the project and will be used by call_llm().
"""

from __future__ import annotations
import time
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, List

# Use Python's logging (project can swap with loguru in tools/logger)
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


@dataclass
class AgentResult:
    success: bool
    outputs: Dict[str, Any] = field(default_factory=dict)
    messages: List[str] = field(default_factory=list)
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class LLMAgentBase:
    """
    Base class for any LLM-backed agent in the system.

    Subclasses should implement:
      - think(self, context) -> str
      - plan(self, thought) -> Dict
      - act(self, plan, tools) -> AgentResult
      - reflect(self, result) -> Optional[Dict]

    This base class provides `run(context)` which orchestrates the lifecycle.
    """

    def __init__(
        self,
        name: str,
        prompt_template: Optional[str] = None,
        max_retries: int = 2,
        retry_delay: float = 1.0,
        llm_client: Optional[Any] = None,
    ):
        self.name = name
        self.prompt_template = prompt_template or ""
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.llm_client = llm_client  # expected to be an instance/wrapper for OpenRouter or other LLM client
        logger.debug("Initialized agent '%s' with max_retries=%d", name, max_retries)

    # ----- LLM calling helper -----
    def call_llm(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """
        Wrapper to call the configured LLM client.

        Expects llm_client to expose a method `chat_completion(prompt, **kwargs)`
        or `openrouter_chat_completion(...)`. Implementations can adapt as needed.
        """
        if not self.llm_client:
            raise RuntimeError("LLM client not configured for agent '%s'." % self.name)

        last_exc = None
        for attempt in range(1, self.max_retries + 1):
            try:
                logger.debug("[%s] Calling LLM (attempt %d). Prompt length=%d", self.name, attempt, len(prompt))
                # standard expected interface; adapt if your client differs
                if hasattr(self.llm_client, "chat_completion"):
                    resp = self.llm_client.chat_completion(prompt=prompt, **kwargs)
                elif hasattr(self.llm_client, "openrouter_chat_completion"):
                    resp = self.llm_client.openrouter_chat_completion(prompt=prompt, **kwargs)
                else:
                    # fallback: try generic call
                    resp = self.llm_client(prompt=prompt, **kwargs)  # type: ignore
                logger.debug("[%s] LLM call succeeded on attempt %d", self.name, attempt)
                return resp
            except Exception as e:
                last_exc = e
                logger.warning("[%s] LLM call failed attempt %d/%d: %s", self.name, attempt, self.max_retries, e)
                time.sleep(self.retry_delay * attempt)

        logger.error("[%s] All LLM attempts failed. Raising last exception.", self.name)
        raise last_exc  # propagate final exception

    # ----- Lifecycle hooks intended for override -----
    def think(self, context: Dict[str, Any]) -> str:
        """
        Produce a short 'thought' describing the agent's internal reasoning about the task.
        Should be implemented by subclass.
        """
        raise NotImplementedError()

    def plan(self, thought: str) -> Dict[str, Any]:
        """
        Given the thought, produce a structured plan (steps, tools to invoke, parameters).
        """
        raise NotImplementedError()

    def act(self, plan: Dict[str, Any], tools: Any) -> AgentResult:
        """
        Execute the plan using provided tools (a module or object containing helper functions).
        Return AgentResult with outputs and any metadata.
        """
        raise NotImplementedError()

    def reflect(self, result: AgentResult) -> Optional[Dict[str, Any]]:
        """
        Optional reflection step: inspect results and decide whether to re-plan/retry.
        Return a dict with actions to take (e.g., {'retry': True, 'reason': '...'}), or None.
        """
        return None

    # ----- Orchestrator -----
    def run(self, context: Dict[str, Any], tools: Any = None) -> AgentResult:
        """
        Full run: think -> plan -> act -> reflect -> maybe retry.

        context: free-form dict that includes dataset, config, params, etc.
        tools: reference to tools module or object (data_tools, eda_tools, ml_tools)
        """
        logger.info("[%s] run() started", self.name)
        attempt = 0
        last_result: Optional[AgentResult] = None

        while attempt <= self.max_retries:
            attempt += 1
            try:
                thought = self.think(context)
                logger.debug("[%s] Thought: %s", self.name, thought)

                plan = self.plan(thought)
                logger.debug("[%s] Plan: %s", self.name, plan)

                result = self.act(plan, tools)
                logger.info("[%s] Action completed. success=%s", self.name, result.success)

                reflection = self.reflect(result)
                if reflection:
                    logger.debug("[%s] Reflection suggests: %s", self.name, reflection)
                    # simple reflexion model: if reflection requests retry, allow it
                    if reflection.get("retry") and attempt <= self.max_retries:
                        logger.info("[%s] Reflection requested retry (attempt %d).", self.name, attempt)
                        last_result = result
                        time.sleep(self.retry_delay)
                        continue

                # final return if no retry requested
                return result

            except Exception as e:
                logger.exception("[%s] Exception during run (attempt %d): %s", self.name, attempt, e)
                last_result = AgentResult(success=False, error=str(e), messages=[f"exception: {e}"])
                if attempt <= self.max_retries:
                    logger.info("[%s] Retrying after exception (attempt %d).", self.name, attempt)
                    time.sleep(self.retry_delay * attempt)
                    continue
                break

        # if we exit loop without successful result, return last_result or a failure
        if last_result:
            return last_result
        return AgentResult(success=False, error="Unknown error in agent run", messages=[])
