"""Search QA dataset loader for SkillRL (self-contained).

AReaL's ``get_custom_dataset`` routes by dataset-path keyword (gsm8k/geometry3k/
...). Search QA isn't registered there, and registering it would require
modifying ``areal/dataset/__init__.py`` (the framework core — CLAUDE.md asks
before touching it). Instead we expose a local ``build_search_rl_dataset`` that
the ``train.py`` calls directly, keeping the example self-contained.

Each sample becomes a dict the workflow's ``arun_episode`` consumes via
``data`` / ``task_kwargs``:
    {
        "question":      str,            # the task / query
        "ground_truth":  {"target": [..]},  # EM ground truth (compute_score format)
        "data_source":   "search",
    }
"""

from __future__ import annotations

from typing import Any


def _normalize_ground_truth(gt: Any) -> dict[str, list[str]]:
    """Coerce various ground-truth shapes into compute_score's {"target": [...]}."""
    if isinstance(gt, dict) and "target" in gt:
        return gt
    if isinstance(gt, dict) and "answer" in gt:
        return {"target": gt["answer"] if isinstance(gt["answer"], list) else [gt["answer"]]}
    if isinstance(gt, (list, tuple)):
        return {"target": [str(x) for x in gt]}
    return {"target": [str(gt)]}


def build_search_rl_dataset(
    split: str | None = None,
    dataset_config: Any = None,
    **kwargs,
):
    """Build a Search RL dataset from a HuggingFace/local dataset path.

    Expects the source dataset to have ``question`` and ``answer`` (or
    ``ground_truth``) fields. Works with any HF dataset, e.g. a Search-R1 style
    QA dataset. Pass ``dataset_config.path`` (HF id or local) in the config.
    """
    from datasets import load_dataset

    path = getattr(dataset_config, "path", None) if dataset_config else None
    name = getattr(dataset_config, "name", None) if dataset_config else None
    if path is None:
        raise ValueError(
            "SkillRL Search dataset requires a `train_dataset.path` in the config "
            "(a HuggingFace dataset id or local path with `question`/`answer` fields)."
        )

    eff_split = split
    if dataset_config is not None and getattr(dataset_config, "split", None) is not None:
        eff_split = dataset_config.split

    ds = load_dataset(path, name=name) if name else load_dataset(path)
    if eff_split is not None and eff_split in ds:
        ds = ds[eff_split]
    else:
        # take the first available split if not specified
        ds = ds[list(ds.keys())[0]]

    def process(sample):
        question = sample.get("question") or sample.get("query") or sample.get("prompt")
        gt = sample.get("ground_truth") or sample.get("answer") or sample.get("answers")
        if question is None:
            raise ValueError(
                f"Search dataset sample missing `question`/`query`/`prompt`: keys={list(sample.keys())}"
            )
        return {
            "question": str(question),
            "ground_truth": _normalize_ground_truth(gt),
            "data_source": "search",
        }

    ds = ds.map(process)
    return ds


def _synthetic_rl_dataset(
    data_source: str,
    num_episodes: int,
):
    """Build a counter-style RL dataset (one row per episode).

    ALFWorld and WebShop pick their own task/goal on env reset (the game/session
    is not driven by the dataset row), so the dataset is just an episode
    counter: each row is a placeholder ``{question, ground_truth, data_source}``
    and the env provides the real task. The workflow's ``_extract_task`` reads
    the task from the env observation.
    """
    from datasets import Dataset

    rows = [
        {"question": "", "ground_truth": {}, "data_source": data_source}
        for _ in range(num_episodes)
    ]
    return Dataset.from_list(rows)


def _resolve_num_episodes(dataset_config: Any, split: str, default_train: int, default_val: int) -> int:
    """num_episodes from config (int `path` or `num_episodes` field), else default."""
    if dataset_config is None:
        return default_train if split != "test" else default_val
    n = getattr(dataset_config, "num_episodes", None)
    if n is not None:
        return int(n)
    path = getattr(dataset_config, "path", None)
    if path is not None:
        try:
            return int(path)
        except (TypeError, ValueError):
            pass
    return default_train if split != "test" else default_val


def build_alfworld_rl_dataset(
    split: str | None = None,
    dataset_config: Any = None,
    **kwargs,
):
    """ALFWorld RL dataset: a counter (env picks games on reset)."""
    n = _resolve_num_episodes(dataset_config, split, default_train=1000, default_val=128)
    return _synthetic_rl_dataset("alfworld", n)


def build_webshop_rl_dataset(
    split: str | None = None,
    dataset_config: Any = None,
    **kwargs,
):
    """WebShop RL dataset: a counter (env picks a goal session on reset)."""
    n = _resolve_num_episodes(dataset_config, split, default_train=1000, default_val=128)
    return _synthetic_rl_dataset("webshop", n)
