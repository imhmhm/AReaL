"""SkillRL training entry point (Search task) on AReaL.

Self-contained per AReaL examples convention (see ``examples/multi_turn_math``):
- task-specific workflow / reward / dataset live under ``examples/skillrl``
- depends only on AReaL's stable public API
- workflow & reward_fn referenced by string path

Usage:
    python -m examples.skillrl.train --config recipes/search_skill.yaml
"""

from __future__ import annotations

import sys

from areal import PPOTrainer
from areal.api.cli_args import load_expr_config
from areal.utils.hf_utils import load_hf_tokenizer

from .configs import SkillRLConfig
from .dataset import build_search_rl_dataset  # local, self-contained
from .memory import SkillsOnlyMemory
from .skill_env_workflow import SkillEnvWorkflow
from .skill_hooks import SkillEvolutionController


def _build_memory(skills_cfg: dict):
    """Construct a SkillsOnlyMemory from config kwargs."""
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

    train_dataset = build_search_rl_dataset(
        split="train",
        dataset_config=config.train_dataset,
    )
    valid_dataset = build_search_rl_dataset(
        split="test",
        dataset_config=config.valid_dataset,
    )

    skills_cfg = dict(config.skills_only_memory)
    env_cfg = dict(config.env)

    # ★ Pillar B/C: build a SHARED skill memory + evolution controller.
    # Train and eval get separate memory instances so new skills (written only
    # into the train memory) cannot inflate validation scores (anti-leakage).
    enable_evolution = skills_cfg.get("enable_dynamic_update", False)
    train_memory = _build_memory(skills_cfg) if skills_cfg else None
    eval_memory = _build_memory(skills_cfg) if skills_cfg else None

    evolution_controller = None
    dynamic_filter_fn = None
    if enable_evolution and train_memory is not None:
        evolution_controller = SkillEvolutionController(
            memory=train_memory,
            update_threshold=skills_cfg.get("update_threshold", 0.4),
            max_new_skills=skills_cfg.get("max_new_skills", 3),
            skill_update_freq=skills_cfg.get("skill_update_freq", 5),
            save_dir=str(config.cluster.fileroot) + "/skill_evolution",
        )
        # AReaL calls this once per trajectory — side effect: drive evolution.
        dynamic_filter_fn = evolution_controller.make_should_accept_fn()

    # Inject the (possibly evolved) shared memory + controller into the train
    # workflow. Eval workflow gets its own memory, no controller.
    base_workflow_kwargs = dict(
        reward_fn="examples.skillrl.reward_fn.search_reward_fn",
        gconfig=config.gconfig,
        tokenizer=config.tokenizer_path,
        env_config=env_cfg,
        max_steps=env_cfg.get("max_steps", 8),
        skills_only_memory=skills_cfg if skills_cfg else None,
    )

    workflow_kwargs = dict(base_workflow_kwargs)
    workflow_kwargs["memory"] = train_memory
    workflow_kwargs["evolution_controller"] = evolution_controller

    eval_workflow_kwargs = dict(base_workflow_kwargs)
    eval_workflow_kwargs["gconfig"] = config.gconfig.new(temperature=0.0, n_samples=1)
    eval_workflow_kwargs["memory"] = eval_memory
    eval_workflow_kwargs["evolution_controller"] = None

    with PPOTrainer(
        config,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
    ) as trainer:
        trainer.train(
            workflow=SkillEnvWorkflow,
            workflow_kwargs=workflow_kwargs,
            eval_workflow=SkillEnvWorkflow,
            eval_workflow_kwargs=eval_workflow_kwargs,
            dynamic_filter_fn=dynamic_filter_fn,
        )


if __name__ == "__main__":
    main(sys.argv[1:])
