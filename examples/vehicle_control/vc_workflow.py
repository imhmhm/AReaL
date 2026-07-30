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
import sys
import threading
import uuid
from typing import Any

import torch
from transformers import PreTrainedTokenizerFast

from areal import workflow_context
from areal.api import (
    AsyncRewardWrapper,
    InferenceEngine,
    ModelRequest,
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
    # VehicleToolRunner 池预热数量。池跨 step 持久 (模块级单例); 每个并发 episode
    # 借一个 runner (独立 coordinator/registry -> 状态隔离)。设为 >= max_concurrent_rollouts
    # 时所有 episode 零排队、零 on-loop 建池阻塞; 小于则池空时 on-demand 增长 (~30ms/个)。
    pool_warm_size: int = field(default=8)


@dataclass
class VCGRPOConfig(GRPOConfig):
    vc: VCConfig = field(default_factory=VCConfig)


# ====================================================================
# 进程级 VehicleToolRunner 池 (跨 step 持久, 隔离 + 复用)
# ====================================================================
# AReaL 每个 training step 重建 workflow 实例, 但池挂在模块级 dict 上, 跨 step 复用。
# 每个 episode 从池里借一个 runner (含独立 coordinator/registry -> 状态隔离,
# 见 vehicle_control/tests/test_mock_instance_isolation.py), 用完归还; reset(scenario)
# 清状态后可被下一个 episode 复用。
#
# 重活只做一次: 第一个 runner 建自己的 TfidfRetriever (~300ms, jieba+tfidf fit),
# 之后抽出来作为 shared retriever 注入后续 runner (build_harness overrides["retriever"]),
# 后续 build 只剩 coordinator 实例化 (类已缓存 ~1ms) + registry 反射 (~27ms) ≈ 30ms。
# build_openai_tools 只取 ToolSpec 的文本字段 (tool_id/description/param_schema),
# 不碰 callable_obj -> 共享 retriever 不影响各 runner 自己的 executor 路由。
_RUNNER_POOLS: dict[tuple, _VehicleRunnerPool] = {}
_RUNNER_POOLS_LOCK = threading.Lock()


class _VehicleRunnerPool:
    """进程内 VehicleToolRunner 池。"""

    def __init__(self, harness_config, build_fn, warm_size: int, logger):
        self._cfg = harness_config
        self._build = build_fn
        self._logger = logger
        self._free: list = []
        self._lock = threading.Lock()
        self._shared_retriever = None
        # eager: 第一个 runner 建自己的 retriever -> 抽出 shared; 再预热 warm_size-1
        # 个 (复用 shared retriever, 每个 ~30ms)。全在 workflow __init__ 调用栈里做
        # (off asyncio loop), 不阻塞 rollout event loop。
        first = self._build(self._cfg)
        self._shared_retriever = first.retriever
        self._free.append(first)
        for _ in range(max(0, warm_size - 1)):
            self._free.append(
                self._build(self._cfg, overrides={"retriever": self._shared_retriever})
            )
        logger.info(
            f"[VC pool] warmed {len(self._free)} runners, shared retriever captured"
        )

    def acquire(self):
        with self._lock:
            if self._free:
                return self._free.pop()
        # 池空: on-demand 建一个 (复用 shared retriever, ~30ms, 阻塞 loop 极短)。
        return self._build(self._cfg, overrides={"retriever": self._shared_retriever})

    def release(self, runner):
        with self._lock:
            self._free.append(runner)


def _get_runner_pool(harness_root, harness_config, build_fn, warm_size, logger):
    """取 (或首次建) 进程级 runner 池, key=(harness_root, HarnessConfig 全字段)。"""
    key = (harness_root, tuple(sorted(harness_config.__dict__.items())))
    with _RUNNER_POOLS_LOCK:
        p = _RUNNER_POOLS.get(key)
        if p is None:
            p = _VehicleRunnerPool(harness_config, build_fn, warm_size, logger)
            _RUNNER_POOLS[key] = p
        return p


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

        from agent_harness.config import HarnessConfig
        from agent_harness.eval.scenario import Scenario
        from agent_harness.format import parse_tool_calls_from_text
        from agent_harness.rl.tool_runner import build_tool_runner

        self.reward_fn = reward_fn
        self.gconfig = gconfig.new_with_stop_and_pad_token_ids(tokenizer)
        self.tokenizer = tokenizer
        self.vc_config = vc_config
        self.max_turns = vc_config.max_turns
        self.max_length = vc_config.max_length
        self.enable_thinking = enable_thinking
        self.harness_config = HarnessConfig()
        self.async_reward_fn = AsyncRewardWrapper(reward_fn)

        self._Scenario = Scenario
        self._parse_tool_calls_from_text = parse_tool_calls_from_text
        # 进程级 runner 池: 跨 step 持久, 每 episode 借还, 独立 coordinator 隔离状态。
        self._pool = _get_runner_pool(
            harness_root,
            self.harness_config,
            build_tool_runner,
            vc_config.pool_warm_size,
            logger,
        )
        logger.info("VehicleControl workflow initialized")

    async def arun_episode(
        self, engine: InferenceEngine, data: dict[str, Any]
    ) -> dict[str, Any]:
        """Run a single vehicle control episode."""
        # 1. 从 data 重建 Scenario
        scenario = self._Scenario.from_workflow_data(data)

        # 2. 从池借 runner + reset (reset 清状态, 与上一个 episode 隔离)
        runner = self._pool.acquire()
        try:
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
        finally:
            # 5. 归还 runner (其状态由下一次 reset 清空, 供别的 episode 复用)
            self._pool.release(runner)

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
                obs_ids = self._encode_observation(result.observation, tc.get("id", ""))
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
            attention_mask=torch.ones(len(seq[:max_len]), dtype=torch.bool).unsqueeze(
                0
            ),
            rewards=torch.tensor([float(reward)]),
        )
        return res

    def _encode_observation(self, observation: str, tool_call_id: str) -> list[int]:
        """把 observation 编码为 tool 角色消息 + 下一轮 assistant 开场的 token ids.

        用 tokenizer.apply_chat_template 渲染 (通用, 不再硬编码 ChatML 特殊 token),
        失败则退化为直接 encode observation。这段 loss_mask=0, 不参与训练。

        add_generation_prompt=True 会在 tool 消息后补上下一轮 assistant 的开场
        (如 ChatML 的 `<|im_start|>assistant\n`), 使紧接的 agenerate 能续写。
        """
        msg = {"role": "tool", "content": observation}
        if tool_call_id:
            msg["tool_call_id"] = tool_call_id
        try:
            text = self.tokenizer.apply_chat_template(
                [msg],
                tokenize=False,
                add_generation_prompt=True,
            )
            ids = self.tokenizer.encode(text, add_special_tokens=False)
            if ids:
                return ids
        except Exception:
            pass
        # 退化为直接 encode observation (无 tool 角色包装; loss_mask=0 仍正确)
        return self.tokenizer.encode(observation, add_special_tokens=False)
