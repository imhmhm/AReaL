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
import threading

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


class _AlfworldEnvPool:
    """Process-level pool of textworld gym envs over ONE shared ``AlfredTWEnv``.

    ``AlfredTWEnv.__init__`` runs ``collect_game_files`` -- a ~20s scan of all
    8810 game files (the tqdm bar at alfred_tw_env.py:154). AReaL re-instantiates
    the workflow per rollout task and runs ``group_size`` episodes concurrently,
    so building a fresh ``AlfredTWEnv`` per episode re-scans every time. We build
    the ``AlfredTWEnv`` ONCE per worker process (keyed by config path + train/eval
    split) and hand out cheap ``init_env(batch_size=1)`` gym envs from a pool,
    recycling them across episodes (``reset()`` picks a new game each call). This
    mirrors original SkillRL's ``build_alfworld_envs`` (build a pool once, reuse).
    """

    def __init__(self, alf_config_path: str, train_eval: str):
        from alfworld.agents.environment import get_environment

        config = load_config_file(alf_config_path)
        env_type = config["env"]["type"]  # 'AlfredTWEnv'
        self.base_env = get_environment(env_type)(config, train_eval=train_eval)
        self.multi_modal = env_type == "AlfredThorEnv"
        self._free: list = []
        self._lock = threading.Lock()

    def acquire(self):
        # Pop a recycled gym env if available; else create one. init_env is cheap
        # (no file rescan -- game_files already collected), and creating it under
        # the lock serializes gym-env registration safely. Once the pool warms up
        # to group_size, acquires are just a pop.
        with self._lock:
            if self._free:
                return self._free.pop()
            return self.base_env.init_env(batch_size=1)

    def release(self, env):
        with self._lock:
            self._free.append(env)


_POOLS: dict[tuple[str, str], _AlfworldEnvPool] = {}
_POOLS_LOCK = threading.Lock()


def _get_pool(alf_config_path: str, train_eval: str) -> _AlfworldEnvPool:
    """Get (building once per process) the pool for this config + train/eval split."""
    key = (alf_config_path, train_eval)
    with _POOLS_LOCK:
        if key not in _POOLS:
            _POOLS[key] = _AlfworldEnvPool(alf_config_path, train_eval)
        return _POOLS[key]


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
        alf_cfg = env_config.get("alfworld", {}) or {}
        # Resolve config_tw.yaml (default: the vendored one next to this file).
        alf_config_path = alf_cfg.get("alf_config_path") or os.path.join(
            _THIS_DIR, "configs", "config_tw.yaml"
        )
        eval_dataset = alf_cfg.get("eval_dataset", "eval_in_distribution")
        train_eval = "train" if is_train else eval_dataset

        # Borrow a (recycled) gym env from the process-level pool. The expensive
        # AlfredTWEnv (20s game-file scan) is built once per process inside the
        # pool; each adapter only gets a cheap init_env gym env and returns it on
        # close() for reuse by a later episode.
        self._pool = _get_pool(alf_config_path, train_eval)
        self.multi_modal = self._pool.multi_modal
        self.env = self._pool.acquire()
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
        # textworld.gym batches by batch_size=1; obs is a list OR tuple `(obs_str,)`
        # depending on the textworld/gym version -- accept both.
        obs_text = obs[0] if isinstance(obs, (list, tuple)) else obs
        return obs_text, info

    def step(self, action: str) -> tuple[str, float, bool, dict]:
        obs, scores, dones, infos = self.env.step([action])
        info = self._unwrap_infos(infos)
        obs_text = obs[0] if isinstance(obs, (list, tuple)) else obs
        reward = compute_reward(info, self.multi_modal)
        done = bool(dones[0] if isinstance(dones, (list, tuple)) else dones)
        return obs_text, reward, done, info

    @property
    def chat_history(self) -> list:
        # ALFWorld has no chat_history; the workflow renders history from
        # traj_steps (action_history) instead.
        return []

    def close(self):
        if self._closed:
            return
        # Return the gym env to the pool for reuse by a later episode. Do NOT
        # close it: the shared AlfredTWEnv + textworld envs live for the process
        # (cleaned up at exit), and reset() re-initializes per-episode state.
        try:
            if self.env is not None:
                self._pool.release(self.env)
        except Exception:  # noqa: BLE001
            pass
        self.env = None
        self._closed = True
