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
from .gigpo_advantage import (
    build_step_group,
    cluster_by_equality,
    cluster_by_similarity,
    compute_gigpo_per_row_advantage,
)


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

        if "step_rewards" not in data:
            raise KeyError(
                "GiGPO is enabled but the rollout batch is missing "
                "'step_rewards'. SkillEnvWorkflow emits step_rewards per "
                "step-row; ensure the active workflow is SkillEnvWorkflow "
                "(or a subclass), not a stock one."
            )

        # Eq. 6 (step group): cluster by anchor, dispatched by anchor_mode.
        #   hash            -> int64 equality (vectorized, no decode)
        #   text_exact      -> decode anchor text, string equality
        #   text_similarity -> decode anchor text, SequenceMatcher>=thresh
        anchor_mode = rn_cfg.anchor_mode
        group_size = rn_cfg.group_size
        if anchor_mode == "hash":
            if "anchor_hash" not in data:
                raise KeyError(
                    "anchor_mode=hash but the rollout batch is missing "
                    "'anchor_hash'. SkillEnvWorkflow emits anchor_hash per "
                    "step-row when anchor_mode=hash."
                )
            step_group_id = build_step_group(data["anchor_hash"], group_size)
        else:  # text_exact | text_similarity
            for key in ("anchor_c0", "anchor_c1"):
                if key not in data:
                    raise KeyError(
                        f"anchor_mode={anchor_mode!r} but the rollout batch is "
                        f"missing {key!r}. SkillEnvWorkflow emits anchor_c0 / "
                        f"anchor_c1 per step-row when anchor_mode is a text "
                        f"mode."
                    )
            anchor_texts = self._decode_anchor_texts(data)
            if anchor_mode == "text_exact":
                step_group_id = cluster_by_equality(anchor_texts, group_size)
            else:  # text_similarity
                step_group_id = cluster_by_similarity(
                    anchor_texts, rn_cfg.similarity_thresh, group_size
                )

        # GiGPO: per-row advantage over the full batch (Eq. 8 = A^E + ω·A^S).
        adv_row = compute_gigpo_per_row_advantage(
            rewards=data["rewards"],
            step_rewards=data["step_rewards"],
            step_group_id=step_group_id,
            group_size=group_size,
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

    def _decode_anchor_texts(self, data: dict[str, Any]) -> list[str | None]:
        """Recover each row's anchor text from ``input_ids`` + char span [c0,c1].

        text_exact / text_similarity path. The prompt portion (``loss_mask==0``
        within ``attention_mask``) decodes to the rendered prompt string;
        slicing ``[c0:c1]`` yields the bare anchor (``current_obs``) -- see
        ``SkillRL_GiGPO_anchor从prompt界定方法.md`` §3. Padding rows (``c0<0``)
        -> ``None`` (size-1 cluster).

        ``skip_special_tokens=False`` is REQUIRED: the decoded string must
        align 1:1 with the ``rendered`` the spans were computed on, so special
        tokens like ``<|im_start|>`` must be preserved (stripping them would
        shift every c0/c1).
        """
        input_ids = data["input_ids"]
        attn = data["attention_mask"].bool()
        loss_mask = data["loss_mask"]
        c0s = data["anchor_c0"]
        c1s = data["anchor_c1"]
        texts: list[str | None] = []
        for i in range(input_ids.shape[0]):
            c0 = int(c0s[i])
            if c0 < 0:
                texts.append(None)
                continue
            c1 = int(c1s[i])
            # Row layout is left-aligned [prompt|response|pad]; prompt tokens =
            # valid (attention) AND prompt (loss_mask==0).
            prompt_len = int((attn[i] & (loss_mask[i] == 0)).sum().item())
            decoded = self.tokenizer.decode(
                input_ids[i, :prompt_len], skip_special_tokens=False
            )
            texts.append(decoded[c0:c1])
        return texts
