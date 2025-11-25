# agents/memory/reflexion_memory.py
"""
ReflexionMemory
----------------
Stores agent self-improvement data.

Purpose:
- Capture insights from failures and successes.
- Allow agents to adjust future planning.
- Provide LLM with prior lessons when generating new plans.

Used inside LLMAgentBase.reflect() to store:
- error causes
- reasoning improvements
- suggestions for next attempts
"""

from __future__ import annotations
from typing import List, Dict, Any
import datetime
import json
import os


class ReflexionMemory:
    def __init__(self, memory_path: str = "outputs/logs/reflexion_memory.json"):
        self.memory_path = memory_path
        self.data: List[Dict[str, Any]] = []

        # Load existing memory if available
        if os.path.exists(self.memory_path):
            try:
                with open(self.memory_path, "r", encoding="utf-8") as f:
                    self.data = json.load(f)
            except Exception:
                self.data = []

    def add_entry(self, agent_name: str, event: Dict[str, Any]):
        """
        event structure example:
        {
            "timestamp": "2025-11-25T14:00",
            "success": False,
            "error": "Missing column 'Age'",
            "lesson": "Check dataframe columns before planning model training.",
            "context_summary": "df_has_columns=['A','B'], missing_target"
        }
        """
        entry = {
            "agent": agent_name,
            "timestamp": datetime.datetime.now().isoformat(),
            **event
        }
        self.data.append(entry)
        self._save()

    def get_recent_lessons(self, limit: int = 5) -> List[Dict[str, Any]]:
        return self.data[-limit:]

    def get_all_lessons(self) -> List[Dict[str, Any]]:
        return self.data

    def _save(self):
        os.makedirs(os.path.dirname(self.memory_path), exist_ok=True)
        with open(self.memory_path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=4)
