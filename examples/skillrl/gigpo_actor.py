"""FSDP PPO actor subclass with a GiGPO advantage branch.

AReaL has no advantage-estimator registry -- ``compute_advantages`` is a single
hardwired GAE in ``PPOActor`` (``areal/trainer/ppo/actor.py:128``), and the
actor class is hard-instantiated inside ``FSDPPPOActor.__init__``
(``areal/engine/fsdp_engine.py:1633``), so subclassing ``PPOActor`` alone is not
enough. The clean, core-fork-free injection (verified against AReaL's dispatch):
subclass ``FSDPPPOActor`` and override ``compute_advantages``. ``PPOActorController``
dispatches by method name (``train_controller._custom_function_call`` ->
``async_call_engine(worker.id, method, ...)``), so the worker subclass's override
is picked up automatically. ``SkillPPOTrainer._create_actor`` (see
``gigpo_trainer.py``) instantiates this class.

How GiGPO params reach the worker: AReaL serializes the actor config to workers
via ``SerializedDataclass`` (``areal/infra/rpc/serialization.py``), which uses
``dataclasses.fields()`` + the class's import path -- so a plain extra attribute
on ``PPOActorConfig`` would be DROPPED, but a ``NormConfig`` *subclass* with
extra declared fields survives (its fields are enumerated and it is rebuilt via
its own ``class_path``). So train.py sets ``config.actor.reward_norm`` to a
``SkillGigpoNormConfig`` (subclass) carrying the gigpo params, and we read them
here from ``self.actor.config.reward_norm``. With ``mean_level=None`` /
``std_level=None`` the stock ``Normalization`` is a no-op (returns ``x``
unchanged -- verified in ``areal/utils/data.py:1182``), so super's
``reward_norm(adv_row)`` passes the pre-computed advantage through untouched.

GiGPO branch: compute the per-row advantage ``A^E + ω·A^S`` from the full batch
(``rewards`` / ``step_rewards`` / ``anchor_hash`` -- all emitted by
``SkillEnvWorkflow``), write it into ``data["rewards"]``, and delegate to
``super().compute_advantages``. AReaL's GAE path (``discount=gae_lambda=1``,
``values=0``, ``reward_norm`` no-op, ``kl_ctl=0``, ``reward_scaling=1``,
``reward_bias=0``, ``reward_clip`` high -- all set in ``train.py``) then
broadcasts it uniformly to every response token of the row -- exactly GiGPO's
per-step outcome advantage (GiGPO is outcome-based: no GAE, no value). The raw
episode reward is restored to ``data["rewards"]`` afterwards (``ppo_update``
reads it for correct/incorrect-seq logging).
"""

from __future__ import annotations

from typing import Any

import torch

from areal.engine.fsdp_engine import FSDPPPOActor

from .configs import SkillGigpoNormConfig
from .gigpo_advantage import compute_gigpo_per_row_advantage


class SkillFSDPPPOActor(FSDPPPOActor):
    """FSDPPPOActor with an optional GiGPO advantage branch.

    When ``config.reward_norm`` is not an enabled ``SkillGigpoNormConfig`` (the
    GRPO path), ``compute_advantages`` falls through to the stock AReaL GAE path
    -- so this is a safe drop-in.
    """

    @torch.no_grad()
    def compute_advantages(self, data: dict[str, Any]) -> dict[str, Any]:
        rn_cfg = self.actor.config.reward_norm
        if not (
            isinstance(rn_cfg, SkillGigpoNormConfig) and rn_cfg.enable
        ):
            # GRPO path (stock AReaL): unchanged.
            return super().compute_advantages(data)

        for key in ("step_rewards", "anchor_hash"):
            if key not in data:
                raise KeyError(
                    f"GiGPO is enabled but the rollout batch is missing "
                    f"{key!r}. SkillEnvWorkflow emits step_rewards / "
                    f"anchor_hash per step-row; ensure the active workflow is "
                    f"SkillEnvWorkflow (or a subclass), not a stock one."
                )

        # GiGPO: per-row advantage over the full batch.
        adv_row = compute_gigpo_per_row_advantage(
            rewards=data["rewards"],
            step_rewards=data["step_rewards"],
            anchor_hash=data["anchor_hash"],
            group_size=rn_cfg.group_size,
            max_steps=rn_cfg.max_steps,
            step_advantage_w=rn_cfg.step_advantage_w,
            mode=rn_cfg.mode,
            gamma=rn_cfg.gamma,
        )

        # Transport: stash the pre-computed advantage into `rewards` so the
        # stock GAE path (discount=gae_lambda=1, values=0; reward_norm is a
        # no-op via mean/std_level=None; kl_ctl=0) broadcasts it to every
        # response token of each row. super() reads `rewards` into a LOCAL
        # reward_score (it does not mutate data["rewards"]).
        original_rewards = data["rewards"]
        data["rewards"] = adv_row.to(
            dtype=original_rewards.dtype, device=original_rewards.device
        )
        result = super().compute_advantages(data)
        # Restore the raw episode reward: ppo_update reads data["rewards"] for
        # correct/incorrect-seq stats (expects raw 0/10, not the advantage).
        data["rewards"] = original_rewards
        return result
