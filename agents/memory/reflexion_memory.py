# agents/memory/reflexion_memory.py

"""
ReflexionMemory (Patched)
-------------------------
✔ Never stores un-serializable objects (models, dataframes, numpy, scikit-learn)
✔ Safely strips objects before JSON dump
✔ Stores ONLY safe text summaries
"""

from __future__ import annotations
from typing import List, Dict, Any
import datetime
import json
import os


def _safe(obj: Any):
    """
    Convert ANY Python object into a JSON-safe representation.
    - Models → class name
    - Dataframes → shape string
    - Exceptions → str()
    - Everything else → safe string
    """
    try:
        # Primitive types = keep as-is
        if isinstance(obj, (str, int, float, bool, type(None))):
            return obj

        # Dict → recursively clean
        if isinstance(obj, dict):
            return {k: _safe(v) for k, v in obj.items()}

        # List or tuple
        if isinstance(obj, (list, tuple)):
            return [_safe(i) for i in obj]

        # Pandas DataFrame
        if "pandas" in str(type(obj)):
            return f"DataFrame(shape={obj.shape})"

        # Sklearn / XGBoost / LightGBM models
        if "sklearn" in str(type(obj)) or "xgboost" in str(type(obj)).lower() or "lightgbm" in str(type(obj)).lower():
            return f"MLModel({type(obj).__name__})"

        # Everything else
        return str(obj)

    except Exception:
        return "<unserializable>"


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
        Stores safe, serializable memory entries.
        Automatically cleans the event to avoid errors.
        """
        safe_event = _safe(event)

        entry = {
            "agent": agent_name,
            "timestamp": datetime.datetime.now().isoformat(),
            **safe_event,
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
