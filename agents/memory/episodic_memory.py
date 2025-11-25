# agents/memory/episodic_memory.py
"""
EpisodicMemory
---------------
Lightweight memory that stores recent events, thoughts, plans, and actions.

Purpose:
- Provide LLM agents with short-term history
- Improve multi-step reasoning
- Maintain continuity across multi-agent pipelines

This is NOT long-term memory; old episodes expire automatically.
"""

from __future__ import annotations
from typing import List, Dict, Any, Optional
import datetime


class EpisodicMemory:
    def __init__(self, max_length: int = 20):
        self.max_length = max_length
        self.episodes: List[Dict[str, Any]] = []

    def add_episode(
        self,
        agent_name: str,
        thought: Optional[str] = None,
        plan: Optional[Dict[str, Any]] = None,
        action_summary: Optional[str] = None,
        success: bool = True,
    ):
        episode = {
            "timestamp": datetime.datetime.now().isoformat(),
            "agent": agent_name,
            "thought": thought,
            "plan": plan,
            "action": action_summary,
            "success": success,
        }
        self.episodes.append(episode)

        # enforce memory size
        if len(self.episodes) > self.max_length:
            self.episodes.pop(0)

    def get_recent(self, limit: int = 10) -> List[Dict[str, Any]]:
        return self.episodes[-limit:]

    def get_context_block(self) -> str:
        """
        Returns a formatted text block of recent episodes
        that can be inserted into LLM prompts.
        """
        block = ""
        for ep in self.episodes[-self.max_length:]:
            block += (
                f"[{ep['timestamp']}] Agent={ep['agent']} | "
                f"Thought={ep.get('thought')} | "
                f"Plan={ep.get('plan')} | "
                f"Success={ep.get('success')}\n"
            )
        return block.strip()
