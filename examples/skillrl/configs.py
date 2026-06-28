"""Config for SkillRL on AReaL.

Extends ``GRPOConfig`` with the env + skill-memory knobs that SkillRL needs,
without touching ``areal/api/cli_args.py`` (per CLAUDE.md, modifying the core
config structures requires "Ask First"). This mirrors the
``MultiTurnGRPOConfig(GRPOConfig)`` pattern in ``examples/multi_turn_math``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from areal.api.cli_args import GRPOConfig


@dataclass
class SkillRLConfig(GRPOConfig):
    """GRPO config + SkillRL env / skill-memory configuration."""

    # Env-side config consumed by SkillEnvWorkflow.
    env: dict[str, Any] = field(
        default_factory=dict,
        metadata={
            "help": "Env config: search_url, topk, timeout, max_steps, "
            "log_requests, etc. Passed to build_search_envs."
        },
    )

    # Skill memory config (SkillRL pillar B/C).
    skills_only_memory: dict[str, Any] = field(
        default_factory=dict,
        metadata={
            "help": "SkillsOnlyMemory config: skills_json_path, retrieval_mode, "
            "embedding_model_path, top_k, task_specific_top_k, "
            "enable_dynamic_update, update_threshold, max_new_skills, "
            "skill_update_freq, embedding_device."
        },
    )
