"""Vehicle Control RL 训练入口 (仿 train_tir.py)."""
import sys

from areal import PPOTrainer
from areal.api.cli_args import load_expr_config
from areal.dataset import get_custom_dataset
from areal.utils import logging
from areal.utils.hf_utils import load_hf_tokenizer

from vc_workflow import VCGRPOConfig  # isort: skip

logger = logging.getLogger("VehicleControl Training")


def vc_reward_fn(prompt, completions, prompt_ids, completion_ids,
                 trajectory=None, **data) -> float:
    """车控 reward 函数.

    基于 scenario.expected 三层冗余 + assertions 算分.
    trajectory 由 VehicleToolRunner.get_trajectory() 提供.
    """
    if trajectory is None:
        return 0.0

    expected = data.get("expected", {})
    assertions = data.get("assertions", [])
    reward_config = data.get("reward_config", {})
    scenario_id = data.get("scenario_id", "unknown")

    w_tool = reward_config.get("tool_selection", 0.3)
    w_arg = reward_config.get("arg_correctness", 0.3)
    w_state = reward_config.get("state_achieved", 0.3)
    w_format = reward_config.get("format", 0.1)
    w_eff = reward_config.get("efficiency_penalty", -0.05)

    score = 0.0

    # 1. 工具选择正确性
    expected_tools = expected.get("tools", [])
    expected_names = [t["name"] for t in expected_tools]
    actual_steps = []
    if trajectory and trajectory.turns:
        for turn in trajectory.turns:
            actual_steps.extend(turn.react_steps)

    actual_names = [s.tool_call for s in actual_steps if s.tool_call]
    matched = 0
    for en in expected_names:
        if en in actual_names:
            matched += 1
    if expected_names:
        score += w_tool * (matched / len(expected_names))

    # 2. 参数正确性
    arg_correct = 0
    arg_total = 0
    for exp_tc in expected_tools:
        exp_name = exp_tc["name"]
        exp_args = exp_tc.get("arguments", {})
        for step in actual_steps:
            if step.tool_call == exp_name and step.args:
                arg_total += 1
                if all(step.args.get(k) == v for k, v in exp_args.items()):
                    arg_correct += 1
                break
    if arg_total > 0:
        score += w_arg * (arg_correct / arg_total)

    # 3. 无错误 (format + no_error)
    has_error = any(
        step.result and isinstance(step.result, dict) and not step.result.get("success", True)
        for step in actual_steps
    )
    if not has_error:
        score += w_format

    # 4. 终态路由正确
    exp_route = expected.get("route")
    if exp_route:
        last_step = actual_steps[-1] if actual_steps else None
        if last_step and hasattr(last_step, "result") and isinstance(last_step.result, dict):
            route_sig = last_step.result.get("route_signal", "")
            if route_sig == exp_route.lower() or route_sig == exp_route:
                score += 0.1

    # 5. 效率惩罚 (超步数)
    max_turns_assert = None
    for a in assertions:
        if a.get("type") == "max_turns":
            max_turns_assert = a.get("value", 8)
    if max_turns_assert and len(actual_steps) > max_turns_assert:
        score += w_eff * (len(actual_steps) - max_turns_assert)

    # 6. 断言硬检查 (额外加分)
    for a in assertions:
        atype = a.get("type")
        if atype == "must_call":
            target = a.get("target")
            if target and target in actual_names:
                score += 0.05
            else:
                score -= 0.05
        elif atype == "must_not_call":
            target = a.get("target")
            if target and target in actual_names:
                score -= 0.1
        elif atype == "finish_called":
            if "finish" in actual_names:
                score += 0.05
            else:
                score -= 0.05

    return max(0.0, min(1.0, score))


def main(args):
    config, _ = load_expr_config(args, VCGRPOConfig)

    logger.info("Starting VehicleControl training")
    logger.info(f"Configuration: {config.experiment_name}")
    logger.info(f"Model: {config.actor.path}")

    tokenizer = load_hf_tokenizer(config.tokenizer_path)

    train_dataset = get_custom_dataset(
        split="train", dataset_config=config.train_dataset, tokenizer=tokenizer
    )
    valid_dataset = get_custom_dataset(
        split="test", dataset_config=config.valid_dataset, tokenizer=tokenizer
    )

    workflow_kwargs = dict(
        reward_fn="examples.vehicle_control.train_vc.vc_reward_fn",
        gconfig=config.gconfig,
        tokenizer=config.tokenizer_path,
        vc_config=config.vc,
        enable_thinking=False,
    )
    eval_workflow_kwargs = workflow_kwargs.copy()
    eval_workflow_kwargs["gconfig"] = config.gconfig.new(temperature=0.6)

    with PPOTrainer(config, train_dataset, valid_dataset) as trainer:
        trainer.train(
            workflow="examples.vehicle_control.vc_workflow.VehicleControlWorkflow",
            workflow_kwargs=workflow_kwargs,
            eval_workflow="examples.vehicle_control.vc_workflow.VehicleControlWorkflow",
            eval_workflow_kwargs=eval_workflow_kwargs,
        )


if __name__ == "__main__":
    main(sys.argv[1:])
