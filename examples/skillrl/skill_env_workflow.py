"""Env-driven multi-turn rollout workflow for SkillRL on AReaL.

This is the AReaL adaptation of SkillRL's ``agent_system/multi_turn_rollout``
loop. The original verl-agent loop ran on the driver process and cross-called
the Ray worker every step (with pad/unpad). AReaL's :class:`RolloutWorkflow`
abstraction lets us run the whole env-driven loop *inside* ``arun_episode``,
calling ``engine.agenerate`` directly per step - no pad/unpad, no cross-process
round-trips.

Key design points (see docs/SkillRL_迁移AReaL方案.md §4.2):
- ``arun_episode`` handles a SINGLE trajectory. GRPO group / padding / async
  concurrency are all handled by the framework (``GroupedRolloutWorkflow`` +
  ``concat_padded_tensors``).
- Skill retrieval happens once at reset; skill injection happens every step
  in ``build_text_obs`` (via ``SkillsOnlyMemory.format_for_prompt``).
- Returns a dict with ``attention_mask`` (so the framework can pad) and a 1-D
  ``rewards`` scalar (so the framework does NOT pad it).

Env dispatch (template method):
- ``SkillEnvWorkflow`` is the base + search default. The env-specific bits
  (env adapter factory, projection fn, prompt building, history rendering,
  task extraction, task_type default) are overridable hooks. Search keeps its
  original behaviour as the default, so the search task is unchanged.
- ``AlfworldEnvWorkflow`` / ``WebShopEnvWorkflow`` subclass it and override the
  hooks for their env (lazy-importing their env_package so search runs without
  alfworld/webshop deps installed).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
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


@dataclass
class TurnContext:
    """Everything an env-specific ``_build_text_obs`` may need for one turn.

    Each env uses the fields it needs; the base (search) uses
    ``task_description``/``history``/``retrieved_memories``, alfworld/webshop
    also use ``current_obs``/``info``/``traj_steps``.
    """

    task_description: str
    current_obs: str
    info: dict[str, Any]
    retrieved_memories: dict | None
    history: str
    traj_steps: list[dict[str, str]] = field(default_factory=list)
    step_count: int = 0
    init: bool = False


class SingleEnvAdapter:
    """Wrap a single sub-env of SkillRL's vectorized ``SearchMultiProcessEnv``.

    SkillRL's envs are vectorized (``reset(kwargs: List[dict])`` /
    ``step(actions: List[str])`` operate on ``env_num * group_n`` sub-envs at
    once). AReaL runs one trajectory per ``arun_episode``, so we build the
    vectorized env with ``env_num=1, group_n=1`` and expose a single-element
    interface here - no need to rewrite the env itself.
    """

    def __init__(self, env_config, seed: int = 0, is_train: bool = True):
        self.env_config = env_config
        self.seed = seed
        self.is_train = is_train
        self._vec_env = build_search_envs(
            seed=seed,
            env_num=1,
            group_n=1,
            is_train=is_train,
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
            is_train=self.is_train,
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
    """Skill-augmented multi-turn rollout over a gym env (base + search default).

    Mirrors ``areal/workflow/multi_turn.py``'s ``MultiTurnWorkflow`` but drives
    an env state machine instead of retrying until reward>0.

    Subclasses (AlfworldEnvWorkflow / WebShopEnvWorkflow) override the env hooks
    ``_make_env_adapter`` / ``projection_f`` / ``_build_text_obs`` /
    ``_build_history`` / ``_extract_task`` / ``task_type_default``.
    """

    # Env-specific hooks (overridden by subclasses). Set in __init__.
    task_type_default: str = "search"

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
        is_train: bool = True,
        evolution_controller: Any = None,
    ):
        from areal.utils.hf_utils import load_hf_tokenizer

        if isinstance(tokenizer, str):
            tokenizer = load_hf_tokenizer(tokenizer)
        self.tokenizer = tokenizer
        self.gconfig = gconfig.new_with_stop_and_pad_token_ids(tokenizer)
        self.max_steps = max_steps
        self.seed = seed
        self.is_train = is_train

        # env_config as OmegaConf (build_*_envs read nested attrs).
        self.env_config = (
            env_config if isinstance(env_config, OmegaConf) else OmegaConf.create(env_config)
        )
        # history_length: alfworld/webshop render the last N (obs, action) pairs;
        # search does not use it (defaults to 0 -> no history block).
        self.history_length = self.env_config.get("history_length", 0)

        # Env-specific projection fn (default: search). Subclasses override.
        self.projection_f = search_projection

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

        # Skill evolution controller (pillar C). When set, the workflow records
        # failed trajectories (per-step {action, observation}) into it; the
        # controller's should_accept_fn drives the actual evolution.
        self.evolution_controller = evolution_controller

    # ------------------------------------------------------------------ #
    # Env-specific hooks (override in subclasses)                         #
    # ------------------------------------------------------------------ #

    def _make_env_adapter(self, seed: int) -> SingleEnvAdapter:
        """Build a fresh single-env adapter for one trajectory (default: search)."""
        return SingleEnvAdapter(self.env_config, seed=seed, is_train=self.is_train)

    def _extract_task(self, obs_text: str, task_kwargs: dict[str, Any]) -> str:
        """Extract the task description used for skill retrieval + task_type.

        Default (search): the dataset row's ``question`` IS the task.
        Alfworld parses ``'Your task is to: '`` from the textworld obs;
        WebShop splits the obs on ``' [SEP] '``.
        """
        return task_kwargs["question"]

    def _build_text_obs(self, ctx: TurnContext) -> str:
        """Build the text observation the LLM sees, with skills injected (search)."""
        if ctx.init:
            return SEARCH_TEMPLATE_NO_HIS.format(task_description=ctx.task_description)

        use_skills = retrieved_has_skills(ctx.retrieved_memories)

        if use_skills:
            memory_context = self.memory.format_for_prompt(ctx.retrieved_memories)
            return SEARCH_TEMPLATE_WITH_MEMORY.format(
                task_description=ctx.task_description,
                retrieved_memories=memory_context,
                step_count=ctx.step_count,
                memory_context=ctx.history,
            )
        return SEARCH_TEMPLATE.format(
            task_description=ctx.task_description,
            step_count=ctx.step_count,
            memory_context=ctx.history,
        )

    def _build_history(self, chat_history: list, traj_steps: list[dict[str, str]]) -> str:
        """Render history for the prompt's history slot (default: search tags)."""
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
        env = self._make_env_adapter(seed=self.seed)
        try:
            return await self._run(env, engine, data)
        finally:
            env.close()

    async def _run(self, env, engine: InferenceEngine, data: dict):
        task_kwargs = data  # dataset row: question / ground_truth / data_source
        obs_text, info = env.reset(task_kwargs)
        task_description = self._extract_task(obs_text, task_kwargs)

        # ★ Skill retrieval - once per trajectory (SkillRL's pillar B).
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
        won = False

        # The first user turn is the task itself.
        history = ""
        # Per-step trajectory for skill-evolution failure analysis (pillar C)
        # and for action_history rendering (alfworld/webshop).
        traj_steps: list[dict[str, str]] = []
        current_obs = obs_text

        for step in range(self.max_steps):
            ctx = TurnContext(
                task_description=task_description,
                current_obs=current_obs,
                info=info,
                retrieved_memories=retrieved,
                history=history,
                traj_steps=traj_steps,
                step_count=step,
                init=(step == 0),
            )
            text_obs = self._build_text_obs(ctx)

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
            results, valids = self.projection_f([action_str])
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

            if not env_action:
                # Invalid action - feed empty observation, let the model retry.
                env_obs, reward, done, step_info = "", 0.0, False, {}
            else:
                env_obs, reward, done, step_info = env.step(env_action)

            episode_reward = max(episode_reward, reward)
            won = won or bool(step_info.get("won", False))

            # Record per-step (action, observation) for skill-evolution analysis.
            traj_steps.append(
                {"action": action_str[:1500], "observation": (env_obs or "")[:800]}
            )

            history = self._build_history(env.chat_history, traj_steps)
            current_obs = env_obs if env_obs else current_obs
            info = step_info

            if done:
                break

        # success metric for skill evolution (SkillRL pillar C trigger).
        # reward shaping is 0/1 (search) or 0/10 (alfworld/webshop's 10*won),
        # so >= 1.0 captures a win for all envs.
        won = won or episode_reward >= 1.0
        task_type = (
            self.memory._detect_task_type(task_description)
            if self.memory is not None
            else self.task_type_default
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


def retrieved_has_skills(retrieved_memories: dict | None) -> bool:
    """Whether the retrieved memory carries injectable skills."""
    return bool(
        retrieved_memories is not None
        and (retrieved_memories.get("general_skills") or retrieved_memories.get("task_specific_skills"))
    )


# ---------------------------------------------------------------------- #
# Alfworld / WebShop workflows (lazy-import their env_package so that    #
# the search task runs without alfworld/webshop deps installed).          #
# ---------------------------------------------------------------------- #


class AlfworldEnvWorkflow(SkillEnvWorkflow):
    """SkillEnvWorkflow for ALFWorld (textworld) tasks.

    Overrides: alfworld projection, single-env adapter (AlfredTWEnv),
    text-obs builder (admissible_actions + action_history), task extraction
    (parses ``'Your task is to: '``), task_type default.
    """

    task_type_default = "alfworld"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Lazy import so search doesn't require alfworld deps.
        from .env_package.alfworld import alfworld_projection

        self.projection_f = alfworld_projection

    def _make_env_adapter(self, seed: int):
        from .env_package.alfworld.envs import AlfworldSingleEnvAdapter

        return AlfworldSingleEnvAdapter(
            self.env_config, seed=seed, is_train=self.is_train
        )

    def _extract_task(self, obs_text: str, task_kwargs: dict[str, Any]) -> str:
        task_start = obs_text.find("Your task is to: ")
        if task_start != -1:
            return obs_text[task_start + len("Your task is to: "):].strip()
        return obs_text  # fallback: whole obs

    def _build_text_obs(self, ctx: TurnContext) -> str:
        from .env_package.prompts.alfworld import (
            ALFWORLD_TEMPLATE,
            ALFWORLD_TEMPLATE_NO_HIS,
            ALFWORLD_TEMPLATE_WITH_MEMORY,
        )

        admissible = ctx.info.get("admissible_commands", []) or []
        admissible_str = "\n ".join(f"'{s}'" for s in admissible if s != "help")
        n_steps = len(ctx.traj_steps)
        hl = min(self.history_length, n_steps) if self.history_length > 0 else 0

        if ctx.init or self.history_length <= 0:
            return ALFWORLD_TEMPLATE_NO_HIS.format(
                current_observation=ctx.current_obs,
                admissible_actions=admissible_str,
            )
        use_skills = retrieved_has_skills(ctx.retrieved_memories)
        if use_skills:
            memory_context = self.memory.format_for_prompt(ctx.retrieved_memories)
            return ALFWORLD_TEMPLATE_WITH_MEMORY.format(
                task_description=ctx.task_description,
                retrieved_memories=memory_context,
                step_count=n_steps,
                history_length=hl,
                action_history=ctx.history,
                current_step=n_steps + 1,
                current_observation=ctx.current_obs,
                admissible_actions=admissible_str,
            )
        return ALFWORLD_TEMPLATE.format(
            task_description=ctx.task_description,
            step_count=n_steps,
            history_length=hl,
            action_history=ctx.history,
            current_step=n_steps + 1,
            current_observation=ctx.current_obs,
            admissible_actions=admissible_str,
        )

    def _build_history(self, chat_history: list, traj_steps: list[dict[str, str]]) -> str:
        """Render the last ``history_length`` (observation, action) pairs."""
        if self.history_length <= 0 or not traj_steps:
            return ""
        recent = traj_steps[-self.history_length:]
        lines = []
        for s in recent:
            lines.append(f"Observation: {s.get('observation', '')}")
            lines.append(f"Action: {s.get('action', '')}")
        return "\n".join(lines)


class WebShopEnvWorkflow(SkillEnvWorkflow):
    """SkillEnvWorkflow for WebShop tasks.

    Overrides: webshop projection, single-env adapter (WebAgentTextEnv),
    text-obs builder (available_actions + action_history), task extraction
    (splits obs on ``' [SEP] '``), task_type default.
    """

    task_type_default = "webshop"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        from .env_package.webshop import webshop_projection

        self.projection_f = webshop_projection

    def _make_env_adapter(self, seed: int):
        from .env_package.webshop.envs import WebshopSingleEnvAdapter

        return WebshopSingleEnvAdapter(
            self.env_config, seed=seed, is_train=self.is_train
        )

    def _extract_task(self, obs_text: str, task_kwargs: dict[str, Any]) -> str:
        # WebShop obs format: "<obs> [SEP] Instruction: [SEP] <instr> [SEP] <obs>"
        # (matches WebshopEnvironmentManager.extract_task: parts[1]=='Instruction:',
        #  parts[2] is the instruction).
        parts = obs_text.split(" [SEP] ")
        for i, p in enumerate(parts):
            if p.strip() == "Instruction:" and i + 1 < len(parts):
                return parts[i + 1].strip()
        # fallback: dataset question if present
        return task_kwargs.get("question", obs_text)

    def _build_text_obs(self, ctx: TurnContext) -> str:
        from .env_package.prompts.webshop import (
            WEBSHOP_TEMPLATE,
            WEBSHOP_TEMPLATE_NO_HIS,
            WEBSHOP_TEMPLATE_WITH_MEMORY,
        )

        avail = ctx.info.get("available_actions", []) or []
        n_steps = len(ctx.traj_steps)
        hl = min(self.history_length, n_steps) if self.history_length > 0 else 0

        if ctx.init or self.history_length <= 0:
            return WEBSHOP_TEMPLATE_NO_HIS.format(
                task_description=ctx.task_description,
                current_observation=ctx.current_obs,
                available_actions=avail,
            )
        use_skills = retrieved_has_skills(ctx.retrieved_memories)
        if use_skills:
            memory_context = self.memory.format_for_prompt(ctx.retrieved_memories)
            return WEBSHOP_TEMPLATE_WITH_MEMORY.format(
                task_description=ctx.task_description,
                retrieved_memories=memory_context,
                step_count=n_steps,
                history_length=hl,
                action_history=ctx.history,
                current_step=n_steps + 1,
                current_observation=ctx.current_obs,
                available_actions=avail,
            )
        return WEBSHOP_TEMPLATE.format(
            task_description=ctx.task_description,
            step_count=n_steps,
            history_length=hl,
            action_history=ctx.history,
            current_step=n_steps + 1,
            current_observation=ctx.current_obs,
            available_actions=avail,
        )

    def _build_history(self, chat_history: list, traj_steps: list[dict[str, str]]) -> str:
        if self.history_length <= 0 or not traj_steps:
            return ""
        recent = traj_steps[-self.history_length:]
        lines = []
        for s in recent:
            lines.append(f"Observation: {s.get('observation', '')}")
            lines.append(f"Action: {s.get('action', '')}")
        return "\n".join(lines)
