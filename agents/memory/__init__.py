# agents/memory/__init__.py
"""
Memory system for Agentic AI.

Includes:
- ReflexionMemory: used for self-improvement and analysis of past failures.
- EpisodicMemory: used for storing short & medium-term agent experiences.
"""

from .reflexion_memory import ReflexionMemory
from .episodic_memory import EpisodicMemory

__all__ = [
    "ReflexionMemory",
    "EpisodicMemory",
]
