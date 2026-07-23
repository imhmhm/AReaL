"""VehicleControlWorkflow - aReaL RolloutWorkflow 实现 (车控 RL 环境).

仿 TIRWorkflow 结构, 把 aReaL token 级推理和 VehicleToolRunner 环境执行拼起来.
不调 LLM, 纯环境 + 共享格式层. LLM 推理由 InferenceEngine (token 级) 完成.

数据流:
    runner = build_tool_runner(config)
    runner.reset(scenario)
    prompt_text, tools = runner.get_prompt_inputs()  # 共享格式层
    input_ids = tokenizer.apply_chat_template(messages, tools=tools)
    while not terminal:
        resp = engine.agenerate(ModelRequest(input_ids))  # token 级推理
        tool_calls = parse_tool_calls_from_text(decode(resp))  # 共享解析层
        result = runner.execute(tool_call)  # 环境执行
        obs_ids = tokenizer.encode(result.observation)  # observation mask=0
    reward = reward_fn(prompt, completions, prompt_ids, completion_ids, **data)
"""
from __future__ import annotations

import copy
import json
import re
import sys
import uuid
from typing import Any

import torch
from transformers import PreTrainedTokenizerFast

from areal import workflow_context
from areal.api import (
    AsyncRewardWrapper,
    InferenceEngine,
    ModelRequest,
    ModelResponse,
    RolloutWorkflow,
)
from areal.api.cli_args import (
    GenerationHyperparameters,
    GRPOConfig,
    dataclass,
    field,
)
from areal.utils import logging, stats_tracker

logger = logging.getLogger("VehicleControl workflow")


@dataclass
class VCConfig:
    """车控 RL workflow 配置."""
    max_turns: int = field(default=8)
    max_length: int = field(default=4096)
    harness_root: str = field(default="d:/Code/GitHub/vehicle_control")


@dataclass
class VCGRPOConfig(GRPOConfig):
    vc: VCConfig = field(default_factory=VCConfig)


class VehicleControlWorkflow(RolloutWorkflow):
    """车控 RL rollout workflow.

    在 arun_episode 内:
    1. 从 data 重建 Scenario
    2. build_tool_runner + reset (初始化车辆状态)
    3. get_prompt_inputs -> messages -> tokenize (共享格式层)
    4. 循环: agenerate -> parse_tool_calls_from_text -> execute (共享解析 + 环境)
    5. loss_mask: LLM 生成=1, observation=0
    6. reward_fn -> tensor dict
    """

    def __init__(
        self,
        reward_fn,
        gconfig: GenerationHyperparameters,
        tokenizer: PreTrainedTokenizerFast | str,
        vc_config: VCConfig,
        enable_thinking: bool = False,
    ):
        super().__init__()
        if isinstance(tokenizer, str):
            from areal.utils.hf_utils import load_hf_tokenizer
            tokenizer = load_hf_tokenizer(tokenizer)
        if isinstance(reward_fn, str):
            from areal.utils.dynamic_import import import_from_string
            reward_fn = import_from_string(reward_fn)

        # 注入 vehicle_control 到 sys.path (agent_harness 依赖)
        harness_root = vc_config.harness_root
        if harness_root and harness_root not in sys.path:
            sys.path.insert(0, harness_root)

        from agent_harness.rl.tool_runner import build_tool_runner
        from agent_harness.eval.scenario import Scenario
        from agent_harness.format import parse_tool_calls_from_text
        from agent_harness.config import HarnessConfig

        self.reward_fn = reward_fn
        self.gconfig = gconfig.new_with_stop_and_pad_token_ids(tokenizer)
        self.tokenizer = tokenizer
        self.vc_config = vc_config
        self.max_turns = vc_config.max_turns
        self.max_length = vc_config.max_length
        self.enable_thinking = enable_thinking
        self.harness_config = HarnessConfig()
        self.async_reward_fn = AsyncRewardWrapper(reward_fn)

        self._build_tool_runner = build_tool_runner
        self._Scenario = Scenario
        self._parse_tool_calls_from_text = parse_tool_calls_from_text
        logger.info("VehicleControl workflow initialized")

    async def arun_episode(
        self, engine: InferenceEngine, data: dict[str, Any]
    ) -> dict[str, Any]:
        """Run a single vehicle control episode."""
        # 1. 从 data 重建 Scenario
        scenario = self._Scenario.from_workflow_data(data)

        # 2. build_tool_runner + reset
        runner = self._build_tool_runner(self.harness_config)
        runner.reset(scenario)

        # 3. get_prompt_inputs -> messages -> tokenize
        system_prompt, tools = runner.get_prompt_inputs()
        messages = list(data.get("messages", []))
        if messages and messages[0]["role"] == "system":
            messages[0]["content"] = system_prompt
        else:
            messages.insert(0, {"role": "system", "content": system_prompt})

        input_ids = self.tokenizer.apply_chat_template(
            messages,
            tools=tools,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=self.enable_thinking,
        )

        # 4. 多轮循环
        return await self._multi_round_response(engine, input_ids, data, runner)

    async def _multi_round_response(self, engine, prompt_ids, data, runner):
        prompt_str = self.tokenizer.decode(prompt_ids)
        completions_str = ""
        tool_call_count = 0
        tool_success_count = 0
        turn = 0
        max_len = self.max_length

        # initialize seq, logprobs, loss_mask, versions
        context_ids = copy.deepcopy(prompt_ids)
        seq = copy.deepcopy(prompt_ids)
        logprobs = [0.0] * len(context_ids)
        loss_mask = [0] * len(context_ids)
        versions = [-1] * len(context_ids)
        output_ids = []

        while turn < self.max_turns and len(context_ids) < max_len:
            # Generate response
            gconfig = self.gconfig.new(n_samples=1)
            req = ModelRequest(
                rid=uuid.uuid4().hex,
                input_ids=context_ids,
                gconfig=gconfig,
                tokenizer=self.tokenizer,
            )
            resp = await engine.agenerate(req)

            context_ids.extend(resp.output_tokens)
            seq.extend(resp.output_tokens)
            logprobs.extend(resp.output_logprobs)
            loss_mask.extend([1] * resp.output_len)
            versions.extend(resp.output_versions)

            cur_text = self.tokenizer.decode(resp.output_tokens)
            completions_str += cur_text
            output_ids.extend(resp.output_tokens)

            # EOS / length 截断
            if resp.stop_reason in ("length", "abort"):
                break
            if len(context_ids) > 0 and context_ids[-1] in [
                self.tokenizer.pad_token_id,
                self.tokenizer.eos_token_id,
            ]:
                break

            # 解析 tool_calls (共享格式层)
            tool_calls = self._parse_tool_calls_from_text(cur_text)
            if not tool_calls:
                break  # 无工具调用 = 终态 (纯文本回复)

            # 执行工具
            terminal = False
            for tc in tool_calls:
                turn += 1
                tool_call_count += 1
                result = runner.execute(tc)
                if result.success:
                    tool_success_count += 1

                # observation -> token ids (mask=0)
                obs_ids = self._encode_observation(
                    result.observation, tc.get("id", "")
                )
                context_ids.extend(obs_ids)
                seq.extend(obs_ids)
                logprobs.extend([0.0] * len(obs_ids))
                loss_mask.extend([0] * len(obs_ids))
                versions.extend([-1] * len(obs_ids))
                completions_str += result.observation

                if result.is_terminal:
                    terminal = True
                    break

            if terminal:
                break

        # reward
        reward = await self.async_reward_fn(
            prompt_str,
            completions_str,
            prompt_ids,
            output_ids,
            trajectory=runner.get_trajectory(),
            **data,
        )

        # stats
        stats_tracker.get(workflow_context.stat_scope()).scalar(
            tool_call_count=tool_call_count,
            tool_success_count=tool_success_count,
        )

        res = dict(
            input_ids=torch.tensor(seq[:max_len]).unsqueeze(0),
            logprobs=torch.tensor(logprobs[:max_len]).unsqueeze(0),
            loss_mask=torch.tensor(loss_mask[:max_len]).unsqueeze(0),
            versions=torch.tensor(versions[:max_len]).unsqueeze(0),
            attention_mask=torch.ones(len(seq[:max_len]), dtype=torch.bool).unsqueeze(0),
            rewards=torch.tensor([float(reward)]),
        )
        return res

    def _encode_observation(self, observation: str, tool_call_id: str) -> list[int]:
        """把 observation 文本编码为 tool 角色的 token ids.

        尝试用 chat template 构造 (含特殊 token), 失败则退化为直接 encode.
        observation 段 loss_mask=0, 不参与训练.
        """
        # 尝试 ChatML 格式 (Qwen/GLM4 等)
        try:
            tool_text = (
                f"<|im_start|>tool\n{observation}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )
            ids = self.tokenizer.encode(tool_text, add_special_tokens=False)
            if len(ids) > 0:
                return ids
        except Exception:
            pass
        # 退化为直接 encode
        return self.tokenizer.encode(observation, add_special_tokens=False)
