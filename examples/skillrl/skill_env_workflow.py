"""Env-driven multi-turn rollout workflow for SkillRL on AReaL.

This is the AReaL adaptation of SkillRL's ``agent_system/multi_turn_rollout``
loop. The original verl-agent loop ran on the driver process and cross-called
the Ray worker every step (with pad/unpad). AReaL's :class:`RolloutWorkflow`
abstraction lets us run the whole env-driven loop *inside* ``arun_episode``,
calling ``engine.agenerate`` directly per step — no pad/unpad, no cross-process
round-trips.

Key design points (see docs/SkillRL_迁移AReaL方案.md §4.2):
- ``arun_episode`` handles a SINGLE trajectory. GRPO group / padding / async
  concurrency are all handled by the framework (``GroupedRolloutWorkflow`` +
  ``concat_padded_tensors``).
- Skill retrieval happens once at reset; skill injection happens every step
  in ``build_text_obs`` (via ``SkillsOnlyMemory.format_for_prompt``).
- Returns a dict with ``attention_mask`` (so the framework can pad) and a 1-D
  ``rewards`` scalar (so the framework does NOT pad it).
- ``success_rate`` / ``task_type`` / ``prompt_str`` are stashed for the skill
  evolution controller (non-tensor fields; not consumed by the framework).
"""

from __future__ import annotations

import logging
from typing import Any
from uuid import uuid4

import torch
from omegaconf import OmegaConf
from transformers import PreTrainedTokenizerFast

from areal.api import InferenceEngine, ModelRequest, RolloutWorkflow
from areal.api.cli_args import GenerationHyperparameters

from .env_package.prompts.search import (
    SEARCH_TEMPLATE,
    SEARCH_TEMPLATE_NO_HIS,
    SEARCH_TEMPLATE_WITH_MEMORY,
)
from .env_package.search import build_search_envs, search_projection
from .memory import SkillsOnlyMemory

logger = logging.getLogger("SkillEnvWorkflow")


class SingleEnvAdapter:
    """Wrap a single sub-env of SkillRL's vectorized ``SearchMultiProcessEnv``.

    SkillRL's envs are vectorized (``reset(kwargs: List[dict])`` /
    ``step(actions: List[str])`` operate on ``env_num * group_n`` sub-envs at
    once). AReaL runs one trajectory per ``arun_episode``, so we build the
    vectorized env with ``env_num=1, group_n=1`` and expose a single-element
    interface here — no need to rewrite the env itself.
    """

    def __init__(self, env_config, seed: int = 0):
        self.env_config = env_config
        self.seed = seed
        self._vec_env = build_search_envs(
            seed=seed,
            env_num=1,
            group_n=1,
            is_train=True,
            env_config=env_config,
        )
        # The single underlying SearchEnv instance.
        self._env = self._vec_env.envs[0]
        self._vec_env.reset([])  # no-op padding-free warmup
        self._vec_env.close()

    def reset(self, task_kwargs: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        """Reset the single env with one task's kwargs.

        ``task_kwargs`` carries the per-task data from the dataset
        (``question``, ``ground_truth``, ``data_source``).
        """
        self._vec_env = build_search_envs(
            seed=self.seed,
            env_num=1,
            group_n=1,
            is_train=True,
            env_config=self.env_config,
        )
        self._env = self._vec_env.envs[0]
        # SearchMultiProcessEnv._sync_reset expects these keys.
        extras = {
            "ground_truth": task_kwargs["ground_truth"],
            "max_turns": task_kwargs.get("max_turns", self.env_config.max_steps),
            "data_source": task_kwargs.get("data_source", "unknown"),
        }
        self._env.reset(extras)
        # The initial "observation" is the question itself (no tool feedback yet).
        obs = task_kwargs["question"]
        info = {"data_source": extras["data_source"]}
        return obs, info

    def step(self, action: str) -> tuple[list, float, bool, dict[str, Any]]:
        out = self._env.step(action)
        obs = out["observations"]
        obs = [] if len(obs) == 0 else obs[0]["content"].strip()
        reward = float(out["reward"])
        done = bool(out["done"])
        info = dict(out.get("metadata", {}))
        info["postprocessed_action"] = out.get("postprocessed_action")
        info["won"] = bool(done and reward >= 1.0)
        return obs, reward, done, info

    @property
    def turns(self) -> int:
        return getattr(self._env, "turns", 0)

    @property
    def chat_history(self) -> list:
        return getattr(self._env, "chat_history", [])

    def close(self):
        try:
            self._vec_env.close()
        except Exception:  # noqa: BLE001
            pass


class SkillEnvWorkflow(RolloutWorkflow):
    """Skill-augmented multi-turn rollout over a gym env.

    Mirrors ``areal/workflow/multi_turn.py``'s ``MultiTurnWorkflow`` but drives
    an env state machine instead of retrying until reward>0.
    """

    def __init__(
        self,
        reward_fn,  # unused: env provides the reward; kept for API symmetry
        gconfig: GenerationHyperparameters,
        tokenizer: PreTrainedTokenizerFast | str,
        env_config: dict[str, Any],
        max_steps: int,
        skills_only_memory: dict[str, Any] | None = None,
        memory: SkillsOnlyMemory | None = None,
        seed: int = 0,
        evolution_controller: Any = None,
    ):
        from areal.utils.hf_utils import load_hf_tokenizer

        if isinstance(tokenizer, str):
            tokenizer = load_hf_tokenizer(tokenizer)
        self.tokenizer = tokenizer
        self.gconfig = gconfig.new_with_stop_and_pad_token_ids(tokenizer)
        self.max_steps = max_steps
        self.seed = seed

        # env_config as OmegaConf (build_search_envs reads nested attrs).
        self.env_config = (
            env_config if isinstance(env_config, OmegaConf) else OmegaConf.create(env_config)
        )

        # Skill memory (SkillRL's verl-free layer). Either passed in (shared
        # across rollouts for skill evolution) or built from kwargs.
        if memory is not None:
            self.memory = memory
        elif skills_only_memory is not None:
            self.memory = SkillsOnlyMemory(
                skills_json_path=skills_only_memory["skills_json_path"],
                retrieval_mode=skills_only_memory.get("retrieval_mode", "template"),
                embedding_model_path=skills_only_memory.get("embedding_model_path"),
                task_specific_top_k=skills_only_memory.get("task_specific_top_k"),
                embedding_device=skills_only_memory.get("embedding_device", "cpu"),
            )
            self._skills_cfg = skills_only_memory
        else:
            self.memory = None
            self._skills_cfg = None
        if skills_only_memory is not None and not hasattr(self, "_skills_cfg"):
            self._skills_cfg = skills_only_memory

        # Pool of envs (re)created per episode — see arun_episode.
        self._env_adapter: SingleEnvAdapter | None = None

        # Skill evolution controller (pillar C). When set, the workflow records
        # failed trajectories (per-step {action, observation}) into it; the
        # controller's should_accept_fn drives the actual evolution.
        self.evolution_controller = evolution_controller

    # ------------------------------------------------------------------ #
    # Prompt building (skill injection)                                   #
    # ------------------------------------------------------------------ #

    def _build_text_obs(
        self,
        task_description: str,
        retrieved_memories: dict | None,
        history: str,
        step_count: int,
        init: bool,
    ) -> str:
        """Build the text observation the LLM sees, with skills injected."""
        if init:
            return SEARCH_TEMPLATE_NO_HIS.format(task_description=task_description)

        use_skills = retrieved_memories is not None and retrieved_memories.get(
            "general_skills"
        ) or retrieved_memories.get("task_specific_skills")

        if use_skills:
            memory_context = self.memory.format_for_prompt(retrieved_memories)
            return SEARCH_TEMPLATE_WITH_MEMORY.format(
                task_description=task_description,
                retrieved_memories=memory_context,
                step_count=step_count,
                memory_context=history,
            )
        return SEARCH_TEMPLATE.format(
            task_description=task_description,
            step_count=step_count,
            memory_context=history,
        )

    @staticmethod
    def _build_history(chat_history: list) -> str:
        """Render the env chat history as the ``{memory_context}`` string."""
        lines = []
        for msg in chat_history:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "assistant":
                lines.append(f"<search>{content}</search>")
            elif role == "user":
                lines.append(f"<information>{content}</information>")
        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    # RolloutWorkflow interface                                           #
    # ------------------------------------------------------------------ #

    async def arun_episode(self, engine: InferenceEngine, data: dict) -> dict | None:
        """Run a single env-driven trajectory with skill injection.

        Returns a dict of [1, ...] tensors consumable by
        ``concat_padded_tensors`` (must contain ``attention_mask``).
        """
        env = SingleEnvAdapter(self.env_config, seed=self.seed)
        try:
            return await self._run(env, engine, data)
        finally:
            env.close()

    async def _run(self, env: SingleEnvAdapter, engine: InferenceEngine, data: dict):
        task_kwargs = data  # dataset row: question / ground_truth / data_source
        obs_text, info = env.reset(task_kwargs)
        task_description = task_kwargs["question"]

        # ★ Skill retrieval — once per trajectory (SkillRL's pillar B).
        retrieved = None
        if self.memory is not None:
            top_k = (self._skills_cfg or {}).get("top_k", 6)
            retrieved = self.memory.retrieve(
                task_description=task_description, top_k=top_k
            )

        seq: list[int] = []
        logprobs: list[float] = []
        loss_mask: list[int] = []
        versions: list[int] = []
        episode_reward = 0.0
        done = False

        # The first user turn is the task itself.
        history = ""
        # Track the prompt token ids used for the FIRST generation so we can
        # compute input_len correctly across turns.
        prev_prompt_ids: list[int] = []
        # Per-step trajectory for skill-evolution failure analysis (pillar C).
        traj_steps: list[dict[str, str]] = []

        for step in range(self.max_steps):
            text_obs = self._build_text_obs(
                task_description=task_description,
                retrieved_memories=retrieved,
                history=history,
                step_count=step,
                init=(step == 0),
            )

            # Build the chat input for this turn.
            messages = [{"role": "user", "content": text_obs}]
            input_ids = list(
                self.tokenizer.apply_chat_template(
                    messages, tokenize=True, add_generation_prompt=True
                )
            )

            req = ModelRequest(
                rid=uuid4().hex,
                input_ids=input_ids,
                gconfig=self.gconfig.new(n_samples=1),
                tokenizer=self.tokenizer,
            )
            resp = await engine.agenerate(req)

            output_tokens = resp.output_tokens
            output_logprobs = resp.output_logprobs
            output_versions = getattr(resp, "output_versions", [-1] * len(output_tokens))

            # Decode the model action and project to an env action.
            action_str = self.tokenizer.decode(output_tokens, skip_special_tokens=True)
            results, valids = search_projection([action_str])
            env_action = results[0] if results else ""

            # Accumulate training tokens: prompt (mask=0) + response (mask=1).
            # Across turns we keep the full conversation as one sequence, but
            # only newly-generated response tokens are trainable. For the
            # multi-turn concat we follow MultiTurnWorkflow: append the
            # incremental prompt tail + the response.
            input_len = len(resp.input_tokens) - len(seq)
            seq = list(resp.input_tokens)
            seq += output_tokens
            logprobs += [0.0] * input_len + output_logprobs
            loss_mask += [0] * input_len + [1] * len(output_tokens)
            versions += [-1] * input_len + list(output_versions)
            prev_prompt_ids = input_ids

            if not env_action:
                # Invalid action — feed empty observation, let the model retry.
                env_obs, reward, done, step_info = "", 0.0, False, {}
            else:
                env_obs, reward, done, step_info = env.step(env_action)

            episode_reward = max(episode_reward, reward)  # SearchEnv: reward only at done

            # Record per-step (action, observation) for skill-evolution analysis.
            traj_steps.append(
                {"action": action_str[:1500], "observation": (env_obs or "")[:800]}
            )

            history = self._build_history(env.chat_history)

            if done:
                break

        # success metric for skill evolution (SkillRL pillar C trigger).
        won = episode_reward >= 1.0
        task_type = (
            self.memory._detect_task_type(task_description)
            if self.memory is not None
            else "search"
        )

        # ★ Pillar C: record failed trajectory for the evolution controller.
        # The controller is shared (thread-safe); non-tensor trajectory data
        # bypasses concat_padded_tensors (which only handles tensors).
        if not won and self.evolution_controller is not None:
            self.evolution_controller.record_failure(
                task=task_description,
                trajectory=traj_steps,
                task_type=task_type,
            )

        # NOTE: returned dict must contain ONLY tensors (concat_padded_tensors
        # calls torch.cat per key). Non-tensor success/prompt data is carried
        # via the evolution_controller side-channel above.
        res = {
            "input_ids": torch.tensor(seq, dtype=torch.int32).unsqueeze(0),
            "logprobs": torch.tensor(logprobs, dtype=torch.float32).unsqueeze(0),
            "loss_mask": torch.tensor(loss_mask, dtype=torch.int32).unsqueeze(0),
            "versions": torch.tensor(versions, dtype=torch.int32).unsqueeze(0),
            "rewards": torch.tensor([episode_reward], dtype=torch.float32),  # 1-D scalar
            "attention_mask": torch.ones(len(seq), dtype=torch.bool).unsqueeze(0),
        }
        return res
