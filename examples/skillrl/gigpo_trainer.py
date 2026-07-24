"""PPOTrainer subclass that wires the GiGPO-capable actor.

AReaL's ``PPOTrainer._create_actor`` (``areal/trainer/rl_trainer.py:633``)
hard-selects the actor class by ``train_backend`` (fsdp -> ``FSDPPPOActor``,
hard-imported); the actor class is not config-driven. We override
``_create_actor`` to swap in ``SkillFSDPPPOActor`` (whose ``compute_advantages``
has the GiGPO branch) when GiGPO is enabled on the fsdp backend. When GiGPO is
disabled, or on non-fsdp backends, we fall through to the stock path -- so this
is a safe drop-in and existing GRPO runs are unaffected.

Only the fsdp train backend is supported for GiGPO (megatron/archon engine
wrappers have parallel ``compute_advantages`` delegators at
``megatron_engine.py:1706`` / ``archon_engine.py:1258`` and would need parallel
subclasses; alfworld runs on fsdp).
"""

from __future__ import annotations

from areal.api.cli_args import PPOActorConfig
from areal.trainer.rl_trainer import PPOTrainer
from areal.utils.environ import is_single_controller

from .configs import SkillGigpoNormConfig


class SkillPPOTrainer(PPOTrainer):
    """PPOTrainer that uses SkillFSDPPPOActor when GiGPO is enabled (fsdp)."""

    def _create_actor(self, actor_config: PPOActorConfig):
        rn = getattr(actor_config, "reward_norm", None)
        gigpo_enabled = isinstance(rn, SkillGigpoNormConfig) and getattr(
            rn, "enable", False
        )

        if self.allocation_mode.train_backend != "fsdp":
            if gigpo_enabled:
                raise NotImplementedError(
                    f"GiGPO on AReaL only supports the fsdp train_backend "
                    f"(got {self.allocation_mode.train_backend!r}). megatron/"
                    f"archon would need parallel engine subclasses."
                )
            return super()._create_actor(actor_config)

        # fsdp: use the GiGPO-capable actor (falls back to stock GRPO via super
        # when reward_norm is not an enabled SkillGigpoNormConfig).
        from .gigpo_actor import SkillFSDPPPOActor

        if is_single_controller():
            actor = SkillFSDPPPOActor.as_controller(actor_config, self.scheduler)
        else:
            actor = SkillFSDPPPOActor(config=actor_config)
        actor.create_process_group(parallel_strategy=self.allocation_mode.train)
        return actor
