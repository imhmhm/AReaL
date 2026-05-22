"""Instruction Following dataset processing for RL and SFT training."""

import json

from datasets import DatasetDict, load_dataset
from transformers import PreTrainedTokenizerFast

from areal.utils import logging

logger = logging.getLogger("InstructionFollowingDataset")


def _load_dataset(path: str, split: str | None):
    """Load dataset from path, supporting parquet files and auto-splitting DatasetDict."""
    if path.endswith(".parquet"):
        dataset = load_dataset("parquet", data_files=path, split=split)
    elif path.endswith(".jsonl"):
        dataset = load_dataset("json", data_files=path, split=split)
    else:
        dataset = load_dataset(path=path, split=split)
    
    # Handle DatasetDict (when split is None and dataset has multiple splits)
    if isinstance(dataset, DatasetDict):
        splits = list(dataset.keys())
        if len(splits) == 1:
            dataset = dataset[splits[0]]
        else:
            raise ValueError(
                f"Dataset has multiple splits {splits}, but no split was specified. "
                f"Please specify a split (e.g., split='train')."
            )
    
    return dataset


def get_if_rl_dataset(
    path: str,
    split: str | None,
    tokenizer: PreTrainedTokenizerFast,
    max_length: int | None = None,
    ground_truth_key: str = "ground_truth",
):
    """
    Load and process instruction following dataset for RL training.

    Supports multiple dataset formats:
    1. allenai/IF_multi_constraints_upto5 format:
       - messages: List of chat messages (OpenAI format)
       - ground_truth:
        [{
          'instruction_id': [
            'keywords:frequency',
            'length_constraints:number_paragraphs'
            ]
          'kwargs': [
            {'keyword': 'synonyms', 'frequency': 3, 'relation': 'at least'},
            {'num_paragraphs': 2}
            ]
        }]

    2. google/IFEval (nvidia/Nemotron-RL-instruction_following) format:
       - prompt: List of chat messages (string / OpenAI format)
       - instruction_id_list:
        ["keywords:frequency", "length_constraints:number_paragraphs"]
       - kwargs:
        [
          {
            "num_highlights": null,
            "relation": "at least",
            "num_words": null,
            "num_placeholders": null,
            "prompt_to_repeat": null,
            "num_bullets": null,
            "section_spliter": null,
            "num_sections": null,
            "capital_relation": null,
            "capital_frequency": null,
            "keywords": null,
            "num_paragraphs": null,
            "language": null,
            "let_relation": null,
            "letter": null,
            "let_frequency": null,
            "end_phrase": null,
            "forbidden_words": null,
            "keyword": "synonyms",
            "frequency": 3,
            "num_sentences": null,
            "postscript_marker": null,
            "first_word": null,
            "nth_paragraph": null
          },
          {
            "num_highlights": null,
            ...,
            "num_paragraphs": 2,
            ...,
            "nth_paragraph": null
          },
        ]

    Args:
        path: HuggingFace dataset path or local parquet file path
        split: Dataset split (e.g., "train", "test")
        tokenizer: Tokenizer for length filtering
        max_length: Maximum prompt length in tokens (optional)
        ground_truth_key: Key name for ground truth field (for IF_multi format)

    Returns:
        Processed HuggingFace Dataset

    """
    dataset = _load_dataset(path, split)

    def process(sample):
        # Handle different input formats for prompt/messages
        if "messages" in sample:
            messages = sample["messages"]
        elif "prompt" in sample:
            prompt_value = sample["prompt"]
            # Auto-detect format: OpenAI chat format or Alpaca text format
            if isinstance(prompt_value, list):
                # OpenAI format: list of dicts with role/content
                messages = prompt_value
            elif isinstance(prompt_value, str):
                # String format: plain text string
                messages = [{"role": "user", "content": prompt_value}]
            else:
                raise ValueError(
                    f"'prompt' field must be a list (OpenAI format) or string. "
                    f"Got type: {type(prompt_value)}"
                )
        else:
            raise ValueError(
                f"Dataset must have 'messages', 'prompt', or 'question' field. "
                f"Available fields: {list(sample.keys())}"
            )

        # Handle ground_truth in different formats
        if ground_truth_key in sample and sample[ground_truth_key] is not None:
            # allenai/IF_multi_constraints_upto5 format: use existing ground_truth field
            ground_truth = sample[ground_truth_key]
        elif "instruction_id_list" in sample and "kwargs" in sample:
            # google/IFEval format: build ground_truth from separate fields
            instruction_ids = list(sample["instruction_id_list"])
            kwargs_list = list(sample["kwargs"])
            ground_truth = json.dumps([
                {"instruction_id": instruction_ids, "kwargs": kwargs_list}
            ])
        else:
            logger.warning(
                f"No ground_truth found for sample, available fields: {list(sample.keys())}"
            )
            ground_truth = None

        return {
            "messages": messages,
            "ground_truth": ground_truth,
        }

    # Apply processing
    columns_to_remove = [c for c in dataset.column_names if c not in ["messages", "ground_truth"]]
    dataset = dataset.map(process).remove_columns(columns_to_remove)

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


__all__ = [
    "get_if_rl_dataset",
]
