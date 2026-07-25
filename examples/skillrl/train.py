"""SkillRL training entry point on AReaL (search / alfworld / webshop).

Self-contained per AReaL examples convention (see ``examples/multi_turn_math``):
- task-specific workflow / reward / dataset live under ``examples/skillrl``
- depends only on AReaL's stable public API
- workflow & reward_fn referenced by string path

Env dispatch is driven by ``env.env_name`` in the config
(``search`` / ``alfworld`` / ``webshop``); each maps to a
(workflow class, dataset loader, reward_fn path) triple.

Usage:
    python -m examples.skillrl.train --config examples/skillrl/config_search.yaml
    python -m examples.skillrl.train --config examples/skillrl/config_alfworld.yaml
    python -m examples.skillrl.train --config examples/skillrl/config_webshop.yaml
"""

from __future__ import annotations

import sys

from areal import PPOTrainer
from areal.api.cli_args import load_expr_config
from areal.utils.hf_utils import load_hf_tokenizer

from .configs import SkillGigpoNormConfig, SkillRLConfig
from .dataset import (
    build_alfworld_rl_dataset,
    build_search_rl_dataset,
    build_webshop_rl_dataset,
)
from .gigpo_trainer import SkillPPOTrainer
from .skill_env_workflow import AlfworldEnvWorkflow, SkillEnvWorkflow, WebShopEnvWorkflow


# env_name -> (workflow class, dataset loader, reward_fn string path)
ENV_REGISTRY = {
    "search": (SkillEnvWorkflow, build_search_rl_dataset, "examples.skillrl.reward_fn.search_reward_fn"),
    "alfworld": (AlfworldEnvWorkflow, build_alfworld_rl_dataset, "examples.skillrl.reward_fn.alfworld_reward_fn"),
    "webshop": (WebShopEnvWorkflow, build_webshop_rl_dataset, "examples.skillrl.reward_fn.webshop_reward_fn"),
}


def _build_memory(skills_cfg: dict):
    """Construct a SkillsOnlyMemory from config kwargs.

    Kept for reference; on AReaL the workflow builds its own read-only memory on
    each worker from the `skills_only_memory` config (see main()), since a built
    SkillsOnlyMemory is not JSON-serializable across the rollout-worker boundary.
    """
    from .memory import SkillsOnlyMemory

    return SkillsOnlyMemory(
        skills_json_path=skills_cfg["skills_json_path"],
        retrieval_mode=skills_cfg.get("retrieval_mode", "template"),
        embedding_model_path=skills_cfg.get("embedding_model_path"),
        task_specific_top_k=skills_cfg.get("task_specific_top_k"),
        embedding_device=skills_cfg.get("embedding_device", "cpu"),
    )


def main(args):
    config, _ = load_expr_config(args, SkillRLConfig)
    tokenizer = load_hf_tokenizer(config.tokenizer_path)

    skills_cfg = dict(config.skills_only_memory)
    env_cfg = dict(config.env)
    # GiGPO config (read here so anchor_mode can flow into workflow_kwargs
    # below; also reused by the GiGPO advantage block later).
    gigpo_cfg = dict(config.gigpo) if config.gigpo else {}
    gigpo_enabled = bool(gigpo_cfg.get("enable", False))
    env_name = env_cfg.get("env_name", "search")
    if env_name not in ENV_REGISTRY:
        raise ValueError(
            f"Unknown env.env_name={env_name!r}; expected one of {list(ENV_REGISTRY)}."
        )
    workflow_cls, dataset_builder, reward_fn_str = ENV_REGISTRY[env_name]

    # num_episodes for the env-driven counter datasets (alfworld/webshop) lives
    # on the free-form `env` config as train_num_episodes / val_num_episodes,
    # because TrainDatasetConfig is a strict dataclass with no num_episodes key.
    # search ignores the kwarg (its builder has **kwargs).
    train_dataset = dataset_builder(
        split="train",
        dataset_config=config.train_dataset,
        num_episodes=env_cfg.get("train_num_episodes"),
    )
    valid_dataset = dataset_builder(
        split="test",
        dataset_config=config.valid_dataset,
        num_episodes=env_cfg.get("val_num_episodes"),
    )

    # ---- Skill memory wiring -----------------------------------------------
    # AReaL runs the RolloutWorkflow on rollout WORKERS, instantiated from a
    # string path + JSON-serialized kwargs. So workflow_kwargs must be entirely
    # JSON-serializable: a built SkillsOnlyMemory / SkillEvolutionController
    # object CANNOT be passed here (it raises "Object of type SkillsOnlyMemory
    # is not JSON serializable" at submit time).
    #
    # Pillar B (retrieval): supported. The workflow rebuilds a read-only memory
    # on each worker from the `skills_only_memory` config dict (its __init__
    # `elif skills_only_memory is not None` branch). Train and eval workflows
    # build separate instances, so eval can't be inflated by train-side writes.
    #
    # Pillar C (evolution): NOT yet supported on AReaL. Evolution needs shared
    # mutable state (failure buffer + skill memory) aggregated across all
    # trajectories on the driver, but both touchpoints -- the workflow's
    # evolution_controller object and the dynamic_filter_fn closure -- are
    # non-JSON-serializable, and worker-side state would not round-trip back.
    enable_evolution = skills_cfg.get("enable_dynamic_update", False)
    if enable_evolution:
        raise NotImplementedError(
            "SkillRL skill evolution (skills_only_memory.enable_dynamic_update=true) "
            "is not yet supported on AReaL: the evolution controller and its shared "
            "memory cannot be JSON-serialized to rollout workers, and worker-side "
            "state would not round-trip back to the driver. Set "
            "enable_dynamic_update=false to run retrieval-only (pillar B)."
        )

    base_workflow_kwargs = dict(
        reward_fn=reward_fn_str,
        gconfig=config.gconfig,
        tokenizer=config.tokenizer_path,
        env_config=env_cfg,
        max_steps=env_cfg.get("max_steps", 8),
        skills_only_memory=skills_cfg if skills_cfg else None,
        # GiGPO Eq.6 anchor mode for the workflow: 'none' when GiGPO is off
        # (GRPO) so the workflow skips anchor computation entirely; otherwise
        # the configured mode (hash | text_exact | text_similarity).
        anchor_mode=(gigpo_cfg.get("anchor_mode", "text_exact") if gigpo_enabled else "none"),
    )

    # NOTE: do NOT add `memory` or `evolution_controller` here -- they are live
    # objects and not JSON-serializable. The workflow constructs its read-only
    # memory from `skills_only_memory` (config) on the worker.
    workflow_kwargs = dict(base_workflow_kwargs)
    workflow_kwargs["is_train"] = True

    eval_workflow_kwargs = dict(base_workflow_kwargs)
    eval_workflow_kwargs["gconfig"] = config.gconfig.new(temperature=0.0, n_samples=1)
    eval_workflow_kwargs["is_train"] = False

    # ---- Per-step rollout group sizing + advantage estimator --------------
    # skill_env_workflow emits `max_steps` training rows per trajectory (one
    # row per env step, see SkillEnvWorkflow._run -- faithful to SkillRL's
    # per-step gather_rollout_data, NOT a concatenated trajectory). AReaL's
    # reward normalization groups rows into contiguous `group_size` chunks for
    # the GRPO advantage, so one group must span ALL per-step rows of the
    # n_samples trajectories that share a prompt:
    #     group_size = n_samples * max_steps
    # (GroupedRolloutWorkflow's own group_size stays n_samples -- that is the
    # *rollout* concurrency, a separate axis.)
    max_steps = env_cfg.get("max_steps", 8)
    group_size = config.gconfig.n_samples * max_steps

    if gigpo_enabled:
        # GiGPO (A^E + ω·A^S, port of gigpo/core_gigpo.py). Install a
        # SkillGigpoNormConfig on config.actor.reward_norm: a NormConfig
        # *subclass* survives AReaL's fields()-based worker serialization (a
        # plain extra attr would be dropped), carrying the GiGPO params to the
        # remote actor. With mean/std_level=None the stock Normalization is a
        # no-op, so super's reward_norm(adv_row) passes the pre-computed
        # advantage through untouched. See SkillFSDPPPOActor.compute_advantages.
        config.actor.reward_norm = SkillGigpoNormConfig(
            mean_level=None,
            std_level=None,
            group_size=group_size,
            enable=True,
            step_advantage_w=gigpo_cfg.get("step_advantage_w", 1.0),
            mode=gigpo_cfg.get("mode", "mean_std_norm"),
            gamma=gigpo_cfg.get("gamma", 0.95),
            max_steps=max_steps,
            anchor_mode=gigpo_cfg.get("anchor_mode", "text_exact"),
            similarity_thresh=gigpo_cfg.get("similarity_thresh", 0.95),
        )
        # GiGPO does its own two-level normalization; neutralize the stock
        # reward shaping / advantage norm so the GAE transport (discount=
        # gae_lambda=1, values=0) carries the pre-computed advantage unaltered.
        config.actor.adv_norm = None
        config.actor.kl_ctl = 0.0
        config.actor.reward_scaling = 1.0
        config.actor.reward_bias = 0.0
        config.actor.reward_clip = 1000.0
        trainer_cls = SkillPPOTrainer
        rn = config.actor.reward_norm
        print(
            f"[skillrl] GiGPO enabled: group_size={group_size} "
            f"(n_samples={config.gconfig.n_samples} * max_steps={max_steps}), "
            f"mode={rn.mode}, gamma={rn.gamma}, "
            f"step_advantage_w={rn.step_advantage_w}, anchor_mode={rn.anchor_mode}"
        )
    else:
        # GRPO: with the yaml default group_size=n_samples, the framework would
        # group `n_samples` *steps* instead of `n_samples` *trajectories*,
        # making the advantage wrong -- so overwrite to the per-step product.
        if config.actor.reward_norm is not None:
            config.actor.reward_norm.group_size = group_size
            print(
                f"[skillrl] per-step rollout: reward_norm.group_size set to "
                f"{group_size} (n_samples * max_steps)"
            )
        trainer_cls = PPOTrainer

    with trainer_cls(
        config,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
    ) as trainer:
        trainer.train(
            workflow=workflow_cls,
            workflow_kwargs=workflow_kwargs,
            eval_workflow=workflow_cls,
            eval_workflow_kwargs=eval_workflow_kwargs,
        )


if __name__ == "__main__":
    main(sys.argv[1:])
