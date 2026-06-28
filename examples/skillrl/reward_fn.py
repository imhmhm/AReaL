"""Reward functions for SkillRL tasks.

In SkillRL's env-driven design the reward comes from the env itself (e.g.
SearchEnv computes exact-match score at episode end). AReaL's rollout workflow
already returns the scalar ``rewards`` field per trajectory, so a separate
reward_fn is mostly a pass-through / hook for extra shaping.

Keeping this module + a string-path reference (``reward_fn=``) mirrors the
``examples/multi_turn_math`` convention so the trainer wiring stays uniform.
"""

from __future__ import annotations

from typing import Any


def search_reward_fn(
    prompt: str,
    completions: str,
    prompt_ids: list[int],
    completion_ids: list[int],
    **kwargs: Any,
) -> float:
    """Pass-through reward for the Search env.

    The actual reward (exact-match) is produced by ``SearchEnv`` inside the
    workflow and placed on ``rewards``. This function exists so the config can
    point ``reward_fn=`` at a stable string path and apply optional shaping.

    Parameters mirror AReaL's reward_fn signature.
    """
    # kwargs may carry the env reward under various keys; default 0.0 means
    # "rely on the env reward already on the trajectory".
    return float(kwargs.get("episode_reward", 0.0))
