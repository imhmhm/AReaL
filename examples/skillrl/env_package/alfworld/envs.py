# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
# Adapted for AReaL: single-trajectory (no Ray) adapter for SkillRL.
#
# Licensed under the Apache License, Version 2.0 (the "License");
"""ALFWorld single-env adapter for AReaL.

SkillRL's original ``AlfworldEnvs`` wrapped N textworld envs in Ray actors
(verl-agent vectorized batch loop). AReaL runs one trajectory per
``arun_episode``, so we drop Ray entirely and drive a single
``AlfredTWEnv.init_env(batch_size=1)`` textworld gym env directly.

The vendored ``alfworld/`` package uses top-level ``alfworld.*`` imports, so we
insert this package's parent (``env_package/alfworld/``) at the FRONT of
``sys.path`` - this makes ``import alfworld`` resolve to the vendored copy
(which carries the ``use_expert=False`` RL pin) even if a pip ``alfworld``
(0.5.0 fork) is installed.
"""

import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
# env_package/alfworld/ holds the vendored alfworld/ package -> front of path.
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

import yaml  # noqa: E402


def load_config_file(path):
    assert os.path.exists(path), f"Invalid config file: {path}"
    with open(path) as reader:
        config = yaml.safe_load(reader)
    return config


def compute_reward(info, multi_modal=False):
    """Reward shaping: 10 * won (textworld); +goal_condition_success_rate (thor)."""
    if multi_modal:
        return 10.0 * float(info["won"]) + float(info.get("goal_condition_success_rate", 0.0))
    return 10.0 * float(info["won"])


class AlfworldSingleEnvAdapter:
    """Single-trajectory adapter over an ``AlfredTWEnv`` textworld gym env.

    Contract consumed by ``SkillEnvWorkflow._run``:
        reset(task_kwargs) -> (obs_text, info)
        step(action)       -> (obs_text, reward, done, info)
        close()
        chat_history       -> list (unused by alfworld; history via traj_steps)

    ``info`` carries ``won`` (bool), ``admissible_commands`` (List[str]) and
    ``extra.gamefile`` (str) - exactly what the textworld env exposes.
    """

    def __init__(self, env_config, seed: int = 0, is_train: bool = True):
        from alfworld.agents.environment import get_environment

        alf_cfg = env_config.get("alfworld", {}) or {}
        # Resolve config_tw.yaml (default: the vendored one next to this file).
        alf_config_path = alf_cfg.get("alf_config_path") or os.path.join(
            _THIS_DIR, "configs", "config_tw.yaml"
        )
        eval_dataset = alf_cfg.get("eval_dataset", "eval_in_distribution")

        config = load_config_file(alf_config_path)
        env_type = config["env"]["type"]  # 'AlfredTWEnv'
        base_env = get_environment(env_type)(
            config, train_eval="train" if is_train else eval_dataset
        )
        self.multi_modal = env_type == "AlfredThorEnv"
        self.env = base_env.init_env(batch_size=1)
        try:
            self.env.seed(seed)
        except Exception:  # noqa: BLE001
            pass
        self._closed = False

    @staticmethod
    def _unwrap_infos(infos: dict) -> dict:
        """textworld batches info values (batch_size=1); unwrap each with [0]."""
        return {k: v[0] for k, v in infos.items()}

    def reset(self, task_kwargs: dict) -> tuple[str, dict]:
        # ALFWorld picks its own game on reset; task_kwargs is not consumed
        # (the dataset row is just an episode counter). The task description
        # is parsed from the textworld observation by _extract_task.
        obs, infos = self.env.reset()
        info = self._unwrap_infos(infos)
        obs_text = obs[0] if isinstance(obs, list) else obs
        return obs_text, info

    def step(self, action: str) -> tuple[str, float, bool, dict]:
        obs, scores, dones, infos = self.env.step([action])
        info = self._unwrap_infos(infos)
        obs_text = obs[0] if isinstance(obs, list) else obs
        reward = compute_reward(info, self.multi_modal)
        done = bool(dones[0] if isinstance(dones, list) else dones)
        return obs_text, reward, done, info

    @property
    def chat_history(self) -> list:
        # ALFWorld has no chat_history; the workflow renders history from
        # traj_steps (action_history) instead.
        return []

    def close(self):
        if self._closed:
            return
        try:
            self.env.close()
        except Exception:  # noqa: BLE001
            pass
        self._closed = True
