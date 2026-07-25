"""Config for SkillRL on AReaL.

Extends ``GRPOConfig`` with the env + skill-memory knobs that SkillRL needs,
without touching ``areal/api/cli_args.py`` (per CLAUDE.md, modifying the core
config structures requires "Ask First"). This mirrors the
``MultiTurnGRPOConfig(GRPOConfig)`` pattern in ``examples/multi_turn_math``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from areal.api.cli_args import GRPOConfig, NormConfig


@dataclass
class SkillGigpoNormConfig(NormConfig):
    """``NormConfig`` subclass that carries GiGPO params to the actor worker.

    AReaL serializes the actor config to remote workers via
    ``SerializedDataclass`` (``areal/infra/rpc/serialization.py``), which
    enumerates ``dataclasses.fields()`` and rebuilds via the class's import
    path. A plain extra attribute on ``PPOActorConfig`` would be dropped, but a
    ``NormConfig`` *subclass* with extra declared fields survives (its fields
    are enumerated and it is rebuilt as the subclass). So train.py installs this
    on ``config.actor.reward_norm`` and ``SkillFSDPPPOActor.compute_advantages``
    reads it back on the worker.

    With ``mean_level=None`` / ``std_level=None`` (set by train.py), the stock
    ``Normalization`` built from this config is a no-op, so it does not
    re-normalize the pre-computed GiGPO advantage -- it is purely a param
    carrier. ``group_size`` (inherited) = ``n_samples * max_steps`` (the GiGPO
    episode group / prompt block size).
    """

    enable: bool = field(default=False, metadata={"help": "Enable GiGPO advantage."})
    step_advantage_w: float = field(
        default=1.0, metadata={"help": "ω in Eq. 8 (default 1.0)."}
    )
    mode: str = field(
        default="mean_std_norm",
        metadata={"help": "mean_std_norm | mean_norm", "choices": ["mean_std_norm", "mean_norm"]},
    )
    gamma: float = field(default=0.95, metadata={"help": "Eq. 5 discount."})
    max_steps: int = field(
        default=0, metadata={"help": "env max_steps (trajectory length within a block)."}
    )
    anchor_mode: str = field(
        default="text_exact",
        metadata={
            "help": "How the Eq.6 step-group anchor is sourced + clustered. "
            "'text_exact' (default): extract anchor text from the rendered "
            "prompt (char-index [c0,c1]) and cluster by string equality -- "
            "universal, works for any env. 'text_similarity': same text "
            "extraction, cluster by SequenceMatcher.ratio()>=thresh (search). "
            "'hash': blake2b(current_obs)->int64 + equality (vectorized, no "
            "decode) -- exact-match optimization for alfworld/webshop. "
            "text_exact and hash are semantically equivalent (hash is the "
            "fast path). See github/doc/SkillRL_GiGPO_anchor从prompt界定方法.md.",
            "choices": ["text_exact", "text_similarity", "hash"],
        },
    )
    similarity_thresh: float = field(
        default=0.95,
        metadata={"help": "SequenceMatcher.ratio() threshold for anchor_mode=text_similarity."},
    )


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

    # GiGPO (Group-in-Group) advantage config (top-level, read by train.py to
    # build the SkillGigpoNormConfig installed on config.actor.reward_norm).
    # See github/doc/SkillRL_GiGPO_适配AReaL方案.md.
    gigpo: dict[str, Any] = field(
        default_factory=dict,
        metadata={
            "help": "GiGPO config: enable, step_advantage_w, mode "
            "(mean_std_norm|mean_norm), gamma, anchor_mode "
            "(text_exact|text_similarity|hash), similarity_thresh."
        },
    )
