"""instruction following reward function for instruction following tasks."""

import ast
import json
import re

from areal.utils import logging

logger = logging.getLogger("InstructionFollowingReward")

# Pattern for thinking section removal
_THINKING_START_TAG = "<think>"
_THINKING_END_TAG = "</think>"
_ANSWER_START_TAG = "<answer>"
_ANSWER_END_TAG = "</answer>"


def if_reward_fn(
    prompt: str,
    completions: str,
    prompt_ids: list[int],
    completion_ids: list[int],
    ground_truth: str | None = None,
    **kwargs,
) -> float:
    """
    Compute reward for instruction following tasks.

    This function verifies whether the model's output follows the specified
    instruction constraints (e.g., keyword presence, format requirements,
    length constraints, etc.).

    Args:
        prompt: The original prompt/instruction
        completions: Model output string
        prompt_ids: Tokenized prompt IDs (unused for instruction following)
        completion_ids: Tokenized completion IDs (unused for instruction following)
        ground_truth: JSON string containing instruction_id and kwargs.
            Format: '[{"instruction_id": ["keywords:existence"], "kwargs": [{"keywords": ["AI"]}]}]'
        **kwargs: Additional data from dataset

    Returns:
        Reward value (0.0 to 1.0), computed as average across all constraints
    """
    if ground_truth is None:
        logger.warning("No ground_truth provided for IF reward")
        return 0.0

    if len(completions) == 0:
        logger.warning("Empty completion received for IF reward")
        return 0.0

    try:
        # Parse ground truth constraints
        constraint_dict = _parse_ground_truth(ground_truth)

        # Extract answer (remove thinking section and answer tags)
        answer = _remove_thinking_section(completions)

        if len(answer) == 0:
            logger.warning("Empty answer after removing thinking section")
            return 0.0

        # Get instruction registry
        from areal.reward.IFEvalG import INSTRUCTION_DICT

        instruction_keys = constraint_dict.get("instruction_id", [])
        args_list = constraint_dict.get("kwargs", [])

        if len(instruction_keys) == 0:
            logger.warning("Empty instruction_id list in ground_truth")
            return 0.0

        if len(instruction_keys) != len(args_list):
            logger.warning(
                f"Mismatch between instruction_id count ({len(instruction_keys)}) "
                f"and kwargs count ({len(args_list)})"
            )

        rewards = []
        for i, instruction_key in enumerate(instruction_keys):
            args = args_list[i] if i < len(args_list) else {}
            if args is None:
                args = {}
            args = {k: v for k, v in args.items() if v is not None}

            if instruction_key not in INSTRUCTION_DICT:
                logger.warning(f"Unknown instruction: {instruction_key}")
                rewards.append(0.0)
                continue

            try:
                instruction_cls = INSTRUCTION_DICT[instruction_key]
                instruction_instance = instruction_cls(instruction_key)
                instruction_instance.build_description(**args)

                if instruction_instance.check_following(answer):
                    rewards.append(1.0)
                else:
                    rewards.append(0.0)
            except Exception as e:
                logger.warning(
                    f"Error checking instruction {instruction_key}: {e}",
                    exc_info=True,
                )
                rewards.append(0.0)

        return sum(rewards) / max(len(rewards), 1)

    except Exception:
        logger.warning("Exception in if_reward_fn", exc_info=True)
        return 0.0


def _parse_ground_truth(ground_truth: str) -> dict:
    """
    Parse ground truth constraints from string format.

    Handles multiple formats:
    1. JSON strings with 'null' values (e.g., from Nemotron format)
    2. Python dict strings with 'None' values (e.g., from IF_multi format)
    3. Nested/double-encoded JSON strings

    Args:
        ground_truth: JSON string or Python dict string

    Returns:
        Dictionary with instruction_id and kwargs
    """
    constraint_dict = None

    # Try json.loads first (handles 'null' values from JSON)
    try:
        constraint_dict = json.loads(ground_truth)
    except (json.JSONDecodeError, TypeError):
        pass

    # Fallback to ast.literal_eval (handles Python dict strings with 'None')
    if constraint_dict is None:
        try:
            constraint_dict = ast.literal_eval(ground_truth)
        except (ValueError, SyntaxError):
            pass

    # Handle nested JSON string (double-encoded case)
    # e.g., '"{"instruction_id": ["test"]}"' or '"[{...}]"'
    if isinstance(constraint_dict, str):
        try:
            constraint_dict = json.loads(constraint_dict)
        except (json.JSONDecodeError, TypeError):
            pass

    # Handle list format: take first element
    if isinstance(constraint_dict, list):
        if len(constraint_dict) == 0:
            return {"instruction_id": [], "kwargs": []}
        constraint_dict = constraint_dict[0]

    if constraint_dict is None:
        logger.warning(f"Failed to parse ground_truth: {ground_truth[:100]}")
        return {"instruction_id": [], "kwargs": []}

    return constraint_dict


def _remove_thinking_section(prediction: str) -> str:
    """
    Remove thinking section and answer tags from prediction.

    This handles the common format where models output:
    <think>...analysis...</think>
    <answer>...final answer...</answer>

    Args:
        prediction: Raw model output

    Returns:
        Cleaned answer string
    """

    # split on thinking end tag and take everything after it
    if _THINKING_END_TAG in prediction:
        prediction = prediction.split(_THINKING_END_TAG, 1)[-1]
    # The thinking section is truncated
    elif _THINKING_START_TAG in prediction:
        return ""

    prediction = prediction.replace(_THINKING_START_TAG, "")
    # Remove answer tags
    prediction = prediction.replace(_ANSWER_START_TAG, "").replace(_ANSWER_END_TAG, "")

    return prediction.strip()


__all__ = ["if_reward_fn"]