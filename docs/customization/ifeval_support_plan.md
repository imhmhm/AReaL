# AReaL IFEval Support Implementation Plan

## Background

AReaL currently lacks support for Instruction Following (IFEval) tasks. The existing reward functions only cover:

- `gsm8k_reward_fn` - Math answer verification
- `geometry3k_reward_fn` - Geometry answer verification
- `clevr_count_70k_reward_fn` - Counting verification
- `MathVerifyWorker` - General math verification

This document outlines the plan to add IFEval support to AReaL, enabling training on instruction following tasks like those in the Tulu3 RLVR pipeline.

---

## Implementation Overview

| Component | File Location | Function |
|-----------|---------------|----------|
| Reward function | `areal/reward/ifeval.py` | IFEval constraint verification |
| IFEval registry | `areal/reward/ifeval_registry.py` | Instruction checkers (reuse IFEvalG) |
| Dataset processing | `areal/dataset/ifeval.py` | Data loading and format conversion |
| Registration update | `areal/reward/__init__.py` | Register new reward |
| Example script | `examples/ifeval/ifeval_grpo.py` | Training example |
| Unit tests | `tests/test_ifeval_reward.py` | Test cases |

---

## Step 1: Create Reward Function

**File**: `areal/reward/ifeval.py`

```python
"""IFEval reward function for instruction following tasks."""

import ast
import json

from areal.utils import logging

logger = logging.getLogger("IFEvalReward")


def ifeval_reward_fn(
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
        prompt_ids: Tokenized prompt IDs (unused for IFEval)
        completion_ids: Tokenized completion IDs (unused for IFEval)
        ground_truth: JSON string containing instruction_id and kwargs.
            Format: '[{"instruction_id": ["keywords:existence"], "kwargs": [{"keywords": ["AI"]}]}]'
        **kwargs: Additional data from dataset

    Returns:
        Reward value (0.0 to 1.0), computed as average across all constraints

    Example:
        >>> ifeval_reward_fn(
        ...     prompt="Write about AI",
        ...     completions="AI is transforming the world.",
        ...     prompt_ids=None,
        ...     completion_ids=None,
        ...     ground_truth='[{"instruction_id": ["keywords:existence"], "kwargs": [{"keywords": ["AI"]}]}]',
        ... )
        1.0
    """
    if ground_truth is None:
        logger.warning("No ground_truth provided for IFEval reward")
        return 0.0

    if len(completions) == 0:
        logger.warning("Empty completion received for IFEval reward")
        return 0.0

    try:
        # Parse ground truth constraints
        constraint_dict = ast.literal_eval(ground_truth)
        if isinstance(constraint_dict, str):
            constraint_dict = json.loads(constraint_dict)

        # Handle list format: take first element
        if isinstance(constraint_dict, list):
            constraint_dict = constraint_dict[0]

        # Extract answer (remove thinking section and answer tags)
        answer = _remove_thinking_section(completions)

        if len(answer) == 0:
            logger.warning("Empty answer after removing thinking section")
            return 0.0

        # Get instruction registry
        from areal.reward.ifeval_registry import INSTRUCTION_DICT

        instruction_keys = constraint_dict.get("instruction_id", [])
        args_list = constraint_dict.get("kwargs", [])

        if len(instruction_keys) == 0:
            logger.warning("Empty instruction_id list in ground_truth")
            return 0.0

        rewards = []
        for instruction_key, args in zip(instruction_keys, args_list):
            if args is None:
                args = {}
            args = {k: v for k, v in args.items() if v is not None}

            if instruction_key not in INSTRUCTION_DICT:
                logger.warning(f"Unknown instruction: {instruction_key}")
                rewards.append(0.0)
                continue

            instruction_cls = INSTRUCTION_DICT[instruction_key]
            instruction_instance = instruction_cls(instruction_key)
            instruction_instance.build_description(**args)

            if instruction_instance.check_following(answer):
                rewards.append(1.0)
            else:
                rewards.append(0.0)

        return sum(rewards) / max(len(rewards), 1)

    except Exception:
        logger.warning("Exception in ifeval_reward_fn", exc_info=True)
        return 0.0


def _remove_thinking_section(prediction: str) -> str:
    """
    Remove thinking section and answer tags from prediction.

    This handles the common format where models output:
    <thinking>...analysis...</thinking>
    <answer>...final answer...</answer>

    Args:
        prediction: Raw model output

    Returns:
        Cleaned answer string
    """
    prediction = prediction.replace("</thinking>", "").strip()
    # Remove thinking section (everything before last </thinking>)
    prediction = prediction.split("<thinking>")[-1]
    # Remove answer tags
    prediction = prediction.replace("<answer>", "").replace("</answer>", "")
    return prediction.strip()
```

---

## Step 2: Create IFEval Registry

**File**: `areal/reward/ifeval_registry.py`

There are two options:

### Option A: Reuse IFEvalG from open-instruct (Recommended)

Copy the following files from `open-instruct` to `areal/reward/IFEvalG/`:
- `open_instruct/IFEvalG/instructions_registry.py`
- `open_instruct/IFEvalG/instructions.py`
- `open_instruct/IFEvalG/instructions_util.py`

Then create a simple wrapper:

```python
"""IFEval instruction registry - wrapper over IFEvalG."""

from areal.reward.IFEvalG.instructions_registry import INSTRUCTION_DICT

__all__ = ["INSTRUCTION_DICT"]
```

### Option B: Create Simplified Registry

If you prefer a minimal implementation without all 60+ constraints:

```python
"""Simplified IFEval instruction registry with core constraints."""

import re
from collections.abc import Sequence

import langdetect

# Instruction key prefixes
_KEYWORD = "keywords:"
_LENGTH = "length_constraints:"
_FORMAT = "detectable_format:"
_CHANGE_CASES = "change_case:"
_STARTEND = "startend:"
_PUNCTUATION = "punctuation:"


class Instruction:
    """Base class for instruction checkers."""

    def __init__(self, instruction_id: str):
        self.id = instruction_id

    def build_description(self, **kwargs):
        raise NotImplementedError

    def check_following(self, value: str) -> bool:
        raise NotImplementedError


class KeywordChecker(Instruction):
    """Check if specified keywords exist in response."""

    def build_description(self, *, keywords: Sequence[str] | None = None):
        self._keywords = keywords or []

    def check_following(self, value: str) -> bool:
        value_lower = value.lower()
        return all(kw.lower() in value_lower for kw in self._keywords)


class KeywordFrequencyChecker(Instruction):
    """Check if a word appears exactly N times."""

    def build_description(self, *, word: str | None = None, frequency: int | None = None, N: int | None = None):
        self._word = word or ""
        self._frequency = frequency or N or 1

    def check_following(self, value: str) -> bool:
        words = re.findall(r"\b\w+\b", value.lower())
        count = sum(1 for w in words if w == self._word.lower())
        return count == self._frequency


class ForbiddenWords(Instruction):
    """Check that forbidden words are not present."""

    def build_description(self, *, forbidden_words: Sequence[str] | None = None):
        self._forbidden_words = forbidden_words or []

    def check_following(self, value: str) -> bool:
        value_lower = value.lower()
        return not any(fw.lower() in value_lower for fw in self._forbidden_words)


class NumberOfWords(Instruction):
    """Check word count constraint."""

    def build_description(self, *, num_words: int | None = None, relation: str | None = None, N: int | None = None):
        self._num_words = num_words or N or 100
        self._relation = relation or "at least"  # "at least", "at most", "around"

    def check_following(self, value: str) -> bool:
        words = value.strip().split()
        count = len(words)
        if self._relation == "at least":
            return count >= self._num_words
        elif self._relation == "at most":
            return count <= self._num_words
        elif self._relation == "around":
            tolerance = max(round(self._num_words * 0.1), 1)
            return abs(count - self._num_words) <= tolerance
        return False


class CapitalLettersEnglishChecker(Instruction):
    """Check entire response is uppercase."""

    def build_description(self, **kwargs):
        pass

    def check_following(self, value: str) -> bool:
        return value == value.upper()


class LowercaseLettersEnglishChecker(Instruction):
    """Check entire response is lowercase."""

    def build_description(self, **kwargs):
        pass

    def check_following(self, value: str) -> bool:
        return value == value.lower()


class EndChecker(Instruction):
    """Check response ends with specified phrase."""

    def build_description(self, *, end_phrase: str | None = None):
        self._end_phrase = end_phrase or ""

    def check_following(self, value: str) -> bool:
        return value.strip().endswith(self._end_phrase)


class QuotationChecker(Instruction):
    """Check entire response is wrapped in double quotes."""

    def build_description(self, **kwargs):
        pass

    def check_following(self, value: str) -> bool:
        return value.startswith('"') and value.endswith('"')


class CommaChecker(Instruction):
    """Check no commas in response."""

    def build_description(self, **kwargs):
        pass

    def check_following(self, value: str) -> bool:
        return "," not in value


class JsonFormat(Instruction):
    """Check response is valid JSON."""

    def build_description(self, **kwargs):
        pass

    def check_following(self, value: str) -> bool:
        import json
        try:
            json.loads(value)
            return True
        except json.JSONDecodeError:
            return False


class BulletListChecker(Instruction):
    """Check number of bullet points."""

    def build_description(self, *, num_bullets: int | None = None, N: int | None = None):
        self._num_bullets = num_bullets or N or 1

    def check_following(self, value: str) -> bool:
        lines = value.split("\n")
        bullets = [line for line in lines if line.strip().startswith(("*", "-"))]
        return len(bullets) == self._num_bullets


# Registry mapping instruction keys to checker classes
INSTRUCTION_DICT = {
    _KEYWORD + "existence": KeywordChecker,
    _KEYWORD + "frequency": KeywordFrequencyChecker,
    _KEYWORD + "forbidden_words": ForbiddenWords,
    _LENGTH + "number_words": NumberOfWords,
    _CHANGE_CASES + "english_capital": CapitalLettersEnglishChecker,
    _CHANGE_CASES + "english_lowercase": LowercaseLettersEnglishChecker,
    _STARTEND + "end_checker": EndChecker,
    _STARTEND + "quotation": QuotationChecker,
    _PUNCTUATION + "no_comma": CommaChecker,
    _FORMAT + "json_format": JsonFormat,
    _FORMAT + "number_bullet_lists": BulletListChecker,
}

__all__ = ["INSTRUCTION_DICT", "Instruction"]
```

**Recommendation**: Use Option A (reuse IFEvalG) for full constraint coverage (60+ constraints) and compatibility with existing datasets.

---

## Step 3: Register Reward Function

**File**: `areal/reward/__init__.py`

Update the existing file:

```python
from math_verify.metric import math_metric
from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig

from areal.utils import logging

logger = logging.getLogger("RewardUtils")

# Add "ifeval" to the list
VALID_REWARD_FN = ["clevr_count_70k", "geometry3k", "gsm8k", "ifeval"]


def get_custom_reward_fn(path: str, **kwargs):
    if "clevr_count_70k" in path:
        from .clevr_count_70k import clevr_count_70k_reward_fn
        return clevr_count_70k_reward_fn
    elif "geometry3k" in path:
        from .geometry3k import geometry3k_reward_fn
        return geometry3k_reward_fn
    elif "ifeval" in path:  # Add this branch
        from .ifeval import ifeval_reward_fn
        return ifeval_reward_fn
    else:
        raise ValueError(
            f"Reward function {path} is not supported. "
            f"Supported reward functions are: {VALID_REWARD_FN}. "
        )


# ... existing MathVerifyWorker code ...


__all__ = [
    "VALID_REWARD_FN",
    "get_custom_reward_fn",
    "MathVerifyWorker",
    "get_math_verify_worker",
    "gsm8k_reward_fn",
    "geometry3k_reward_fn",
    "clevr_count_70k_reward_fn",
    "ifeval_reward_fn",  # Add this
]


_LAZY_IMPORTS = {
    "gsm8k_reward_fn": "areal.reward.gsm8k",
    "geometry3k_reward_fn": "areal.reward.geometry3k",
    "clevr_count_70k_reward_fn": "areal.reward.clevr_count_70k",
    "ifeval_reward_fn": "areal.reward.ifeval",  # Add this
}


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        import importlib
        module = importlib.import_module(_LAZY_IMPORTS[name])
        val = getattr(module, name)
        globals()[name] = val
        return val
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return list(__all__)
```

---

## Step 4: Create Dataset Processing Function

**File**: `areal/dataset/ifeval.py`

```python
"""IFEval dataset processing for RL training."""

from datasets import load_dataset
from transformers import PreTrainedTokenizerFast


def get_ifeval_rl_dataset(
    path: str,
    split: str,
    tokenizer: PreTrainedTokenizerFast | None = None,
    max_length: int | None = None,
    ground_truth_key: str = "ground_truth",
):
    """
    Load and process IFEval dataset for RL training.

    The dataset should have the following fields:
    - messages: List of chat messages (OpenAI format)
    - ground_truth: Constraint information (JSON string)

    Alternatively, if the dataset has different field names:
    - prompt/question: The user instruction
    - ground_truth/constraint: The constraint information

    Args:
        path: HuggingFace dataset path or local path
        split: Dataset split (e.g., "train", "test")
        tokenizer: Tokenizer for length filtering (optional)
        max_length: Maximum prompt length in tokens (optional)
        ground_truth_key: Key name for ground truth field

    Returns:
        Processed HuggingFace Dataset

    Example:
        >>> from transformers import AutoTokenizer
        >>> tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B")
        >>> dataset = get_ifeval_rl_dataset(
        ...     path="allenai/IF_multi_constraints_upto5",
        ...     split="train",
        ...     tokenizer=tokenizer,
        ...     max_length=2048,
        ... )
    """
    dataset = load_dataset(path=path, split=split)

    def process(sample):
        # Handle different input formats
        if "messages" in sample:
            messages = sample["messages"]
        elif "prompt" in sample:
            messages = [{"role": "user", "content": sample["prompt"]}]
        elif "question" in sample:
            messages = [{"role": "user", "content": sample["question"]}]
        else:
            raise ValueError(f"Dataset must have 'messages', 'prompt', or 'question' field")

        # Handle different ground truth field names
        ground_truth = sample.get(ground_truth_key) or sample.get("constraint")
        if ground_truth is None:
            raise ValueError(f"Dataset must have '{ground_truth_key}' or 'constraint' field")

        return {
            "messages": messages,
            "ground_truth": ground_truth,
        }

    # Apply processing
    columns_to_remove = [c for c in dataset.column_names if c not in ["messages", "ground_truth"]]
    dataset = dataset.map(process, remove_columns=columns_to_remove)

    # Filter by length if tokenizer and max_length are provided
    if tokenizer is not None and max_length is not None:
        def filter_length(sample):
            # Get the last user message
            messages = sample["messages"]
            user_content = ""
            for msg in reversed(messages):
                if msg["role"] == "user":
                    user_content = msg["content"]
                    break

            # Tokenize and check length
            tokens = tokenizer.encode(user_content, add_special_tokens=False)
            return len(tokens) <= max_length

        dataset = dataset.filter(filter_length)

    return dataset


def get_ifeval_rl_dataset_with_dataset_field(
    path: str,
    split: str,
    tokenizer: PreTrainedTokenizerFast | None = None,
    max_length: int | None = None,
    ground_truth_key: str = "ground_truth",
    dataset_key: str = "dataset",
):
    """
    Load IFEval dataset with verifier type field for multi-verifier support.

    This is useful for mixed datasets where different samples use different verifiers.
    The 'dataset' field specifies which verifier to use (e.g., "ifeval", "math", etc.)

    Args:
        path: HuggingFace dataset path
        split: Dataset split
        tokenizer: Tokenizer for length filtering
        max_length: Maximum prompt length
        ground_truth_key: Key for ground truth field
        dataset_key: Key for verifier type field

    Returns:
        Processed Dataset with 'dataset' field
    """
    dataset = load_dataset(path=path, split=split)

    def process(sample):
        if "messages" in sample:
            messages = sample["messages"]
        elif "prompt" in sample:
            messages = [{"role": "user", "content": sample["prompt"]}]
        else:
            raise ValueError("Dataset must have 'messages' or 'prompt' field")

        ground_truth = sample.get(ground_truth_key) or sample.get("constraint")
        verifier_type = sample.get(dataset_key, "ifeval")  # Default to ifeval

        return {
            "messages": messages,
            "ground_truth": ground_truth,
            "dataset": verifier_type,  # For multi-verifier selection
        }

    columns_to_remove = [c for c in dataset.column_names
                         if c not in ["messages", "ground_truth", "dataset"]]
    dataset = dataset.map(process, remove_columns=columns_to_remove)

    if tokenizer is not None and max_length is not None:
        def filter_length(sample):
            messages = sample["messages"]
            user_content = ""
            for msg in reversed(messages):
                if msg["role"] == "user":
                    user_content = msg["content"]
                    break
            tokens = tokenizer.encode(user_content, add_special_tokens=False)
            return len(tokens) <= max_length

        dataset = dataset.filter(filter_length)

    return dataset


__all__ = ["get_ifeval_rl_dataset", "get_ifeval_rl_dataset_with_dataset_field"]
```

---

## Step 5: Create Training Example

**File**: `examples/ifeval/ifeval_grpo.py`

```python
"""
GRPO training example for IFEval (Instruction Following) tasks.

This script demonstrates how to train a model using GRPO on IFEval datasets,
where the reward is computed based on whether the model follows specified
instruction constraints.
"""

import hydra
from omegaconf import DictConfig

from areal.api.cli_args import GenerationHyperparameters
from areal.dataset.ifeval import get_ifeval_rl_dataset
from areal.reward import ifeval_reward_fn
from areal.trainer.rl_trainer import RLTrainer
from areal.utils.hf_utils import load_hf_tokenizer
from areal.workflow.rlvr import RLVRWorkflow


@hydra.main(config_path="../conf", config_name="grpo_config", version_base="1.1")
def main(config: DictConfig):
    """
    Main training function.

    Args:
        config: Hydra configuration containing:
            - model.name_or_path: Model path
            - model.tokenizer_path: Tokenizer path
            - dataset.path: Dataset path (e.g., "allenai/IF_multi_constraints_upto5")
            - dataset.split: Dataset split
            - dataset.max_length: Maximum prompt length
            - generation: Generation hyperparameters
            - training: Training hyperparameters
    """
    # Load tokenizer
    tokenizer = load_hf_tokenizer(
        config.model.tokenizer_path or config.model.name_or_path
    )

    # Load dataset
    train_dataset = get_ifeval_rl_dataset(
        path=config.dataset.path,
        split=config.dataset.split,
        tokenizer=tokenizer,
        max_length=config.dataset.max_length,
    )

    print(f"Loaded {len(train_dataset)} samples for training")

    # Create generation config
    gconfig = GenerationHyperparameters(
        max_tokens=config.generation.max_tokens,
        temperature=config.generation.temperature,
        top_p=config.generation.top_p,
        n_samples=config.generation.n_samples,
        stop_strings=config.generation.stop_strings,
    )

    # Create workflow with IFEval reward
    workflow = RLVRWorkflow(
        reward_fn=ifeval_reward_fn,
        gconfig=gconfig,
        tokenizer=tokenizer,
        enable_thinking=config.get("enable_thinking", False),
    )

    # Create trainer
    trainer = RLTrainer(
        config=config.training,
        workflow=workflow,
        actor_config=config.actor,
    )

    # Train
    trainer.train(train_dataset)


if __name__ == "__main__":
    main()
```

---

## Step 6: Add Unit Tests

**File**: `tests/test_ifeval_reward.py`

```python
"""Unit tests for IFEval reward function."""

import pytest

from areal.reward.ifeval import ifeval_reward_fn, _remove_thinking_section


class TestIFEvalReward:
    """Test cases for ifeval_reward_fn."""

    def test_keywords_existence_correct(self):
        """Test reward when required keywords are present."""
        reward = ifeval_reward_fn(
            prompt="Write about AI",
            completions="AI is transforming the world.",
            prompt_ids=None,
            completion_ids=None,
            ground_truth='[{"instruction_id": ["keywords:existence"], "kwargs": [{"keywords": ["AI"]}]}]',
        )
        assert reward == 1.0

    def test_keywords_existence_missing(self):
        """Test reward when required keywords are missing."""
        reward = ifeval_reward_fn(
            prompt="Write about AI",
            completions="Technology is advancing rapidly.",
            prompt_ids=None,
            completion_ids=None,
            ground_truth='[{"instruction_id": ["keywords:existence"], "kwargs": [{"keywords": ["AI"]}]}]',
        )
        assert reward == 0.0

    def test_keywords_frequency_correct(self):
        """Test reward when word frequency matches requirement."""
        reward = ifeval_reward_fn(
            prompt="Write about AI",
            completions="AI is great. AI will change the world.",
            prompt_ids=None,
            completion_ids=None,
            ground_truth='[{"instruction_id": ["keywords:frequency"], "kwargs": [{"word": "AI", "frequency": 2}]}]',
        )
        assert reward == 1.0

    def test_keywords_frequency_wrong_count(self):
        """Test reward when word frequency doesn't match."""
        reward = ifeval_reward_fn(
            prompt="Write about AI",
            completions="AI is transforming everything.",
            prompt_ids=None,
            completion_ids=None,
            ground_truth='[{"instruction_id": ["keywords:frequency"], "kwargs": [{"word": "AI", "frequency": 2}]}]',
        )
        assert reward == 0.0

    def test_forbidden_words_not_present(self):
        """Test reward when forbidden words are absent."""
        reward = ifeval_reward_fn(
            prompt="Write a response",
            completions="This is a good response.",
            prompt_ids=None,
            completion_ids=None,
            ground_truth='[{"instruction_id": ["keywords:forbidden_words"], "kwargs": [{"forbidden_words": ["bad", "evil"]}]}]',
        )
        assert reward == 1.0

    def test_forbidden_words_present(self):
        """Test reward when forbidden words are present."""
        reward = ifeval_reward_fn(
            prompt="Write a response",
            completions="This is a bad response.",
            prompt_ids=None,
            completion_ids=None,
            ground_truth='[{"instruction_id": ["keywords:forbidden_words"], "kwargs": [{"forbidden_words": ["bad"]}]}]',
        )
        assert reward == 0.0

    def test_multiple_constraints_all_pass(self):
        """Test reward with multiple constraints, all passing."""
        reward = ifeval_reward_fn(
            prompt="Write about AI",
            completions="AI is great. AI is transforming the world.",
            prompt_ids=None,
            completion_ids=None,
            ground_truth='[{"instruction_id": ["keywords:existence", "keywords:frequency"], "kwargs": [{"keywords": ["AI"]}, {"word": "AI", "frequency": 2}]}]',
        )
        assert reward == 1.0

    def test_multiple_constraints_partial_pass(self):
        """Test reward with multiple constraints, partial passing."""
        reward = ifeval_reward_fn(
            prompt="Write about AI",
            completions="AI is great. Machine learning is also important.",
            prompt_ids=None,
            completion_ids=None,
            ground_truth='[{"instruction_id": ["keywords:existence", "keywords:frequency"], "kwargs": [{"keywords": ["AI"]}, {"word": "AI", "frequency": 2}]}]',
        )
        assert reward == 0.5  # Only keyword existence passes

    def test_empty_completion(self):
        """Test reward for empty completion."""
        reward = ifeval_reward_fn(
            prompt="Write something",
            completions="",
            prompt_ids=None,
            completion_ids=None,
            ground_truth='[{"instruction_id": ["keywords:existence"], "kwargs": [{"keywords": ["test"]}]}]',
        )
        assert reward == 0.0

    def test_none_ground_truth(self):
        """Test reward when ground_truth is None."""
        reward = ifeval_reward_fn(
            prompt="Write something",
            completions="Test output.",
            prompt_ids=None,
            completion_ids=None,
            ground_truth=None,
        )
        assert reward == 0.0

    def test_empty_instruction_list(self):
        """Test reward with empty instruction_id list (regression test)."""
        reward = ifeval_reward_fn(
            prompt="Write something",
            completions="Test output.",
            prompt_ids=None,
            completion_ids=None,
            ground_truth='[{"instruction_id": [], "kwargs": []}]',
        )
        assert reward == 0.0

    def test_uppercase_constraint(self):
        """Test reward for uppercase constraint."""
        reward = ifeval_reward_fn(
            prompt="Write in uppercase",
            completions="THIS IS ALL UPPERCASE",
            prompt_ids=None,
            completion_ids=None,
            ground_truth='[{"instruction_id": ["change_case:english_capital"], "kwargs": []}]',
        )
        assert reward == 1.0

    def test_lowercase_constraint(self):
        """Test reward for lowercase constraint."""
        reward = ifeval_reward_fn(
            prompt="Write in lowercase",
            completions="this is all lowercase",
            prompt_ids=None,
            completion_ids=None,
            ground_truth='[{"instruction_id": ["change_case:english_lowercase"], "kwargs": []}]',
        )
        assert reward == 1.0

    def test_json_format_constraint(self):
        """Test reward for JSON format constraint."""
        reward = ifeval_reward_fn(
            prompt="Return JSON",
            completions='{"status": "success", "value": 42}',
            prompt_ids=None,
            completion_ids=None,
            ground_truth='[{"instruction_id": ["detectable_format:json_format"], "kwargs": []}]',
        )
        assert reward == 1.0

    def test_json_format_invalid(self):
        """Test reward for invalid JSON."""
        reward = ifeval_reward_fn(
            prompt="Return JSON",
            completions='not valid json',
            prompt_ids=None,
            completion_ids=None,
            ground_truth='[{"instruction_id": ["detectable_format:json_format"], "kwargs": []}]',
        )
        assert reward == 0.0

    def test_end_phrase_constraint(self):
        """Test reward for end phrase constraint."""
        reward = ifeval_reward_fn(
            prompt="End with specific phrase",
            completions="This is my response. Thank you for asking!",
            prompt_ids=None,
            completion_ids=None,
            ground_truth='[{"instruction_id": ["startend:end_checker"], "kwargs": [{"end_phrase": "Thank you for asking!"}]}]',
        )
        assert reward == 1.0

    def test_quotation_constraint(self):
        """Test reward for quotation wrapping."""
        reward = ifeval_reward_fn(
            prompt="Wrap in quotes",
            completions='"This is quoted"',
            prompt_ids=None,
            completion_ids=None,
            ground_truth='[{"instruction_id": ["startend:quotation"], "kwargs": []}]',
        )
        assert reward == 1.0

    def test_no_comma_constraint(self):
        """Test reward for no comma constraint."""
        reward = ifeval_reward_fn(
            prompt="No commas allowed",
            completions="This sentence has no commas at all",
            prompt_ids=None,
            completion_ids=None,
            ground_truth='[{"instruction_id": ["punctuation:no_comma"], "kwargs": []}]',
        )
        assert reward == 1.0

    def test_no_comma_constraint_with_commas(self):
        """Test reward fails when commas are present."""
        reward = ifeval_reward_fn(
            prompt="No commas allowed",
            completions="This, sentence, has commas",
            prompt_ids=None,
            completion_ids=None,
            ground_truth='[{"instruction_id": ["punctuation:no_comma"], "kwargs": []}]',
        )
        assert reward == 0.0


class TestRemoveThinkingSection:
    """Test cases for _remove_thinking_section helper."""

    def test_remove_thinking_tags(self):
        """Test removing thinking tags."""
        result = _remove_thinking_section("<thinking>Let me analyze</thinking>The answer is 42")
        assert result == "The answer is 42"

    def test_remove_answer_tags(self):
        """Test removing answer tags."""
        result = _remove_thinking_section("<answer>42</answer>")
        assert result == "42"

    def test_combined_tags(self):
        """Test removing both thinking and answer tags."""
        result = _remove_thinking_section(
            "<thinking>Analysis</thinking><answer>42</answer>"
        )
        assert result == "42"

    def test_no_tags(self):
        """Test input without any tags."""
        result = _remove_thinking_section("Just plain text")
        assert result == "Just plain text"

    def test_empty_input(self):
        """Test empty input."""
        result = _remove_thinking_section("")
        assert result == ""


class TestIFEvalDataset:
    """Test cases for IFEval dataset processing."""

    def test_dataset_processing_messages_format(self):
        """Test processing dataset with messages format."""
        from datasets import Dataset
        from areal.dataset.ifeval import get_ifeval_rl_dataset

        # Create mock dataset
        mock_data = [
            {
                "messages": [{"role": "user", "content": "Test prompt"}],
                "ground_truth": '[{"instruction_id": ["keywords:existence"], "kwargs": [{"keywords": ["test"]}]}]',
            }
        ]
        dataset = Dataset.from_list(mock_data)

        # Process (without tokenizer for simplicity)
        processed = get_ifeval_rl_dataset.__wrapped__(
            path=None,
            split=None,
            tokenizer=None,
            max_length=None,
        )
        # Note: This test requires mocking load_dataset, skipped for now
```

---

## Summary

| Step | Description | Effort | Priority |
|------|-------------|--------|----------|
| 1 | Create reward function | Medium | High |
| 2 | Create/reuse IFEval registry | High (reuse: Low) | High |
| 3 | Register reward function | Low | High |
| 4 | Create dataset processing | Low | Medium |
| 5 | Create training example | Low | Low |
| 6 | Add unit tests | Medium | Medium |

### Recommended Approach

**Reuse IFEvalG from open-instruct**: Copy the `IFEvalG` directory to `areal/reward/IFEvalG/` to leverage:
- 60+ pre-built constraint checkers
- Mature, tested verification logic
- Compatibility with existing IFEval datasets like `allenai/IF_multi_constraints_upto5`

### Files to Create

```
areal/
├── reward/
│   ├── ifeval.py              # Main reward function
│   ├── ifeval_registry.py     # Registry wrapper
│   ├── IFEvalG/               # Copied from open-instruct (optional)
│   │   ├── instructions_registry.py
│   │   ├── instructions.py
│   │   ├── instructions_util.py
│   └── __init__.py            # Updated with ifeval
├── dataset/
│   └── ifeval.py              # Dataset processing
examples/
└── ifeval/
    └── ifeval_grpo.py         # Training example
tests/
└── test_ifeval_reward.py      # Unit tests
```