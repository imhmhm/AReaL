"""IFEval dataset processing for RL and SFT training."""

from datasets import load_dataset
from transformers import PreTrainedTokenizerFast

from areal.utils import logging

logger = logging.getLogger("IFEvalDataset")


def get_ifeval_rl_dataset(
    path: str,
    split: str,
    tokenizer: PreTrainedTokenizerFast,
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
        tokenizer: Tokenizer for length filtering
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
            raise ValueError(
                f"Dataset must have 'messages', 'prompt', or 'question' field. "
                f"Available fields: {list(sample.keys())}"
            )

        # Handle different ground truth field names
        ground_truth = sample.get(ground_truth_key) or sample.get("constraint")
        if ground_truth is None:
            logger.warning(f"No ground_truth found for sample, available fields: {list(sample.keys())}")

        return {
            "messages": messages,
            "ground_truth": ground_truth,
        }

    # Apply processing
    columns_to_remove = [c for c in dataset.column_names if c not in ["messages", "ground_truth"]]
    dataset = dataset.map(process, remove_columns=columns_to_remove)

    # Filter by length if max_length is provided
    if max_length is not None:
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

        dataset = dataset.filter(filter_length, desc="Filtering by max_length")

    return dataset


def get_ifeval_rl_dataset_with_verifier_field(
    path: str,
    split: str,
    tokenizer: PreTrainedTokenizerFast,
    max_length: int | None = None,
    ground_truth_key: str = "ground_truth",
    verifier_key: str = "dataset",
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
        verifier_key: Key for verifier type field

    Returns:
        Processed Dataset with 'dataset' field for verifier selection
    """
    dataset = load_dataset(path=path, split=split)

    def process(sample):
        if "messages" in sample:
            messages = sample["messages"]
        elif "prompt" in sample:
            messages = [{"role": "user", "content": sample["prompt"]}]
        else:
            raise ValueError(
                f"Dataset must have 'messages' or 'prompt' field. "
                f"Available fields: {list(sample.keys())}"
            )

        ground_truth = sample.get(ground_truth_key) or sample.get("constraint")
        verifier_type = sample.get(verifier_key, "ifeval")  # Default to ifeval

        return {
            "messages": messages,
            "ground_truth": ground_truth,
            "dataset": verifier_type,  # For multi-verifier selection
        }

    columns_to_remove = [c for c in dataset.column_names
                         if c not in ["messages", "ground_truth", "dataset"]]
    dataset = dataset.map(process, remove_columns=columns_to_remove)

    if max_length is not None:
        def filter_length(sample):
            messages = sample["messages"]
            user_content = ""
            for msg in reversed(messages):
                if msg["role"] == "user":
                    user_content = msg["content"]
                    break
            tokens = tokenizer.encode(user_content, add_special_tokens=False)
            return len(tokens) <= max_length

        dataset = dataset.filter(filter_length, desc="Filtering by max_length")

    return dataset


__all__ = [
    "get_ifeval_rl_dataset",
    "get_ifeval_rl_dataset_with_verifier_field",
]