"""Skill memory subsystem for SkillRL.

Ported from SkillRL's ``agent_system/memory/`` (the verl-free layer authored by
the SkillRL / verl-agent team). These modules depend only on numpy /
sentence-transformers / openai — no framework coupling.
"""

from .base import BaseMemory
from .memory import SimpleMemory, SearchMemory
from .skills_only_memory import SkillsOnlyMemory
from .skill_updater import SkillUpdater

# RetrievalMemory needs faiss-cpu + sentence-transformers, which the AReaL NPU
# env may not install (the default skills-only template path does not use them).
# Guard the import so the memory package -- and thus SkillEnvWorkflow, which
# only needs SkillsOnlyMemory -- stays importable without those deps.
# RetrievalMemory is None when they are unavailable. (The original
# agent_system/memory/__init__.py imports it eagerly because that env has faiss.)
try:
    from .retrieval_memory import RetrievalMemory
except ImportError:
    RetrievalMemory = None  # type: ignore[assignment]

__all__ = [
    "BaseMemory",
    "SimpleMemory",
    "SearchMemory",
    "RetrievalMemory",
    "SkillsOnlyMemory",
    "SkillUpdater",
]
