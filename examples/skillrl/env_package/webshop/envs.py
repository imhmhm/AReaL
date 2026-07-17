# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
# Adapted for AReaL: single-trajectory (no Ray) adapter for SkillRL.
#
# Licensed under the Apache License, Version 2.0 (the "License");
"""WebShop single-env adapter for AReaL.

SkillRL's original ``WebshopMultiProcessEnv`` wrapped N ``WebAgentTextEnv``
gym envs in Ray actors. AReaL runs one trajectory per ``arun_episode``, so we
drop Ray and drive a single ``WebAgentTextEnv`` directly.

The env is built ONCE (loading products + building the Lucene search index is
expensive) and ``reset(session=idx)`` is called per episode with a fresh goal
index (train: goals >= 500, val: goals < 500, matching SkillRL's split).

The vendored ``webshop/`` package uses top-level ``web_agent_site.*`` imports,
so we insert ``env_package/webshop/webshop/`` on ``sys.path`` (mirrors the
original ``WebshopWorker`` sys.path append).

External runtime deps (NOT vendored, must be prepared via webshop/setup.sh):
  - data/items_shuffle[_1000].json, items_ins_v2[_1000].json, items_human_ins.json
  - search_engine/indexes/ (Lucene index, built by convert_product_file_format.py)
  - spacy en_core_web_sm, pyserini (+ Java), thefuzz, flask, bs4
"""

import os
import random
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_WEBSHOP_ROOT = os.path.join(_THIS_DIR, "webshop")
if _WEBSHOP_ROOT not in sys.path:
    sys.path.insert(0, _WEBSHOP_ROOT)

_DATA_DIR = os.path.join(_WEBSHOP_ROOT, "data")


class WebshopSingleEnvAdapter:
    """Single-trajectory adapter over a ``WebAgentTextEnv`` gym env.

    Contract consumed by ``SkillEnvWorkflow._run``:
        reset(task_kwargs) -> (obs_text, info)
        step(action)       -> (obs_text, reward, done, info)
        close()
        chat_history       -> list (unused; history via traj_steps)

    Reward shaping mirrors SkillRL's ``WebshopWorker``: 10.0 on a winning
    terminal step (``done and reward == 1.0``), else 0. ``info['task_score']``
    keeps the raw WebShop 0-1 reward for metrics.
    """

    def __init__(self, env_config, seed: int = 0, is_train: bool = True):
        import gym

        webshop_cfg = env_config.get("webshop", {}) or {}
        use_small = webshop_cfg.get("use_small", True)
        human_goals = webshop_cfg.get("human_goals", False)

        if use_small:
            file_path = os.path.join(_DATA_DIR, "items_shuffle_1000.json")
            attr_path = os.path.join(_DATA_DIR, "items_ins_v2_1000.json")
        else:
            file_path = os.path.join(_DATA_DIR, "items_shuffle.json")
            attr_path = os.path.join(_DATA_DIR, "items_ins_v2.json")

        env_kwargs = {
            "observation_mode": "text",
            "num_products": None,
            "human_goals": human_goals,
            "file_path": file_path,
            "attr_path": attr_path,
            "seed": seed,
        }
        # Triggers gym.register via web_agent_site/envs/__init__.py.
        from web_agent_site.envs import WebAgentTextEnv  # noqa: F401

        self.env = gym.make("WebAgentTextEnv-v0", **env_kwargs)
        self.is_train = is_train
        self._rng = random.Random(seed)
        try:
            self._goals = self.env.server.goals
        except Exception:  # noqa: BLE001
            self._goals = []
        self._closed = False

    def _pick_goal_idx(self) -> int:
        n = len(self._goals)
        if n == 0:
            return 0
        # SkillRL split: train uses goals[500:], val uses goals[:500].
        lo, hi = (500, n) if self.is_train and n > 500 else (0, min(500, n))
        if hi <= lo:
            lo, hi = 0, n
        return self._rng.randint(lo, hi - 1)

    def reset(self, task_kwargs: dict) -> tuple[str, dict]:
        idx = self._pick_goal_idx()
        obs, info = self.env.reset(session=idx)
        info = dict(info or {})
        info["available_actions"] = self.env.get_available_actions()
        info["won"] = False
        return obs, info

    def step(self, action: str) -> tuple[str, float, bool, dict]:
        obs, reward, done, info = self.env.step(action)
        info = dict(info or {})
        info["available_actions"] = self.env.get_available_actions()
        info["task_score"] = reward
        # Rule-based reward: win for 10, lose for 0 (matches SkillRL WebshopWorker).
        if done and reward == 1.0:
            info["won"] = True
            reward = 10.0
        else:
            info["won"] = False
            reward = 0.0
        return obs, float(reward), bool(done), info

    @property
    def chat_history(self) -> list:
        # WebShop has no chat_history; history rendered from traj_steps.
        return []

    def close(self):
        if self._closed:
            return
        try:
            self.env.close()
        except Exception:  # noqa: BLE001
            pass
        self._closed = True
