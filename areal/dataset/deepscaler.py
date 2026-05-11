"""Dataset loaders for DeepScaleR training data and four evaluation benchmarks.

Supported datasets
------------------
- DeepScaleR  — training split from ``deepscaler.json`` (40 k math problems)
- AIME 2024   — 30-question eval from a parquet file
- MATH-500    — 500-question eval from ``test.jsonl``
- OlympiadBench — filtered (Text-only, English) eval from ``test.parquet``
- MinervaMAth — 272-question eval from ``test.jsonl``

All loaders return a HuggingFace ``Dataset`` with exactly two columns:
``messages`` (list[dict] with ``role``/``content`` keys) and ``answer`` (str).
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from datasets import load_dataset

if TYPE_CHECKING:
    from datasets import Dataset
    from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

_BOXED_SUFFIX = "\nPlease put your final answer within \\boxed{}."


def _apply_length_filter(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizerFast | None,
    max_length: int | None,
) -> Dataset:
    if max_length is None or tokenizer is None:
        return dataset

    def _within_length(sample: dict) -> bool:
        content = sample["messages"][0]["content"]
        return len(tokenizer.encode(content)) <= max_length

    return dataset.filter(_within_length)


def _keep_only(dataset: Dataset) -> Dataset:
    columns_to_remove = [
        c for c in dataset.column_names if c not in ("messages", "answer")
    ]
    return dataset.remove_columns(columns_to_remove)


# ---------------------------------------------------------------------------
# DeepScaleR training dataset
# ---------------------------------------------------------------------------


def get_deepscaler_rl_dataset(
    path: str,
    split: str,
    tokenizer: PreTrainedTokenizerFast | None,
    max_length: int | None = None,
) -> Dataset:
    """Load the DeepScaleR training dataset for RL fine-tuning.

    Reads ``deepscaler.json`` from *path*. The *split* argument is accepted
    for API compatibility but is ignored — the file only contains training
    data (40 315 samples).

    Parameters
    ----------
    path:
        Directory that contains ``deepscaler.json``.
    split:
        Ignored; kept for API compatibility.
    tokenizer:
        Tokenizer used for optional length filtering.
    max_length:
        If set, samples whose user message exceeds this many tokens are
        dropped.
    """
    data_files = {"train": "deepscaler.json"}
    dataset = load_dataset("json", data_dir=path, data_files=data_files, split="train")

    def process(sample: dict) -> dict:
        messages = [
            {
                "role": "user",
                "content": sample["problem"] + _BOXED_SUFFIX,
            }
        ]
        return {"messages": messages, "answer": str(sample["answer"])}

    dataset = dataset.map(process)
    dataset = _keep_only(dataset)
    return _apply_length_filter(dataset, tokenizer, max_length)


# ---------------------------------------------------------------------------
# AIME 2024 evaluation dataset
# ---------------------------------------------------------------------------


def get_aime2024_rl_dataset(
    path: str,
    split: str,
    tokenizer: PreTrainedTokenizerFast | None,
    max_length: int | None = None,
) -> Dataset:
    """Load the AIME 2024 evaluation dataset (30 problems, parquet format).

    Parameters
    ----------
    path:
        Directory containing the parquet file
        ``data/train-00000-of-00001.parquet``.
    split:
        Passed through to ``load_dataset`` (typically ``"train"``).
    tokenizer:
        Tokenizer used for optional length filtering.
    max_length:
        If set, samples whose user message exceeds this many tokens are
        dropped.
    """
    parquet_file = os.path.join(path, "data", "train-00000-of-00001.parquet")
    data_files = {"train": parquet_file}
    dataset = load_dataset("parquet", data_files=data_files, split="train")

    def process(sample: dict) -> dict:
        messages = [
            {
                "role": "user",
                "content": sample["problem"] + _BOXED_SUFFIX,
            }
        ]
        return {"messages": messages, "answer": str(sample["answer"])}

    dataset = dataset.map(process)
    dataset = _keep_only(dataset)
    return _apply_length_filter(dataset, tokenizer, max_length)


# ---------------------------------------------------------------------------
# MATH-500 evaluation dataset
# ---------------------------------------------------------------------------


def get_math500_rl_dataset(
    path: str,
    split: str,
    tokenizer: PreTrainedTokenizerFast | None,
    max_length: int | None = None,
) -> Dataset:
    """Load the MATH-500 evaluation dataset (500 problems, JSONL format).

    Parameters
    ----------
    path:
        Directory containing ``test.jsonl``.
    split:
        Ignored; the file only contains a test split.
    tokenizer:
        Tokenizer used for optional length filtering.
    max_length:
        If set, samples whose user message exceeds this many tokens are
        dropped.
    """
    data_files = {"test": "test.jsonl"}
    dataset = load_dataset("json", data_dir=path, data_files=data_files, split="test")

    def process(sample: dict) -> dict:
        messages = [
            {
                "role": "user",
                "content": sample["problem"] + _BOXED_SUFFIX,
            }
        ]
        return {"messages": messages, "answer": str(sample["answer"])}

    dataset = dataset.map(process)
    dataset = _keep_only(dataset)
    return _apply_length_filter(dataset, tokenizer, max_length)


# ---------------------------------------------------------------------------
# OlympiadBench evaluation dataset
# ---------------------------------------------------------------------------


def get_olympiadbench_rl_dataset(
    path: str,
    split: str,
    tokenizer: PreTrainedTokenizerFast | None,
    max_length: int | None = None,
) -> Dataset:
    """Load the OlympiadBench evaluation dataset (Text-only, English subset).

    Reads ``test.parquet`` from *path* and filters to samples where
    ``modality == "Text-only"`` and ``language == "English"``.
    The ``final_answer`` field is a list; answers are joined with ``"; "``.

    Parameters
    ----------
    path:
        Directory containing ``test.parquet``.
    split:
        Ignored; the file only contains a test split.
    tokenizer:
        Tokenizer used for optional length filtering.
    max_length:
        If set, samples whose user message exceeds this many tokens are
        dropped.
    """
    data_files = {"test": os.path.join(path, "test.parquet")}
    dataset = load_dataset("parquet", data_files=data_files, split="test")

    dataset = dataset.filter(
        lambda s: s["modality"] == "Text-only" and s["language"] == "English"
    )

    def process(sample: dict) -> dict:
        final_answer_list = sample["final_answer"]
        answer_str = "; ".join(str(a) for a in final_answer_list)
        messages = [
            {
                "role": "user",
                "content": sample["question"] + _BOXED_SUFFIX,
            }
        ]
        return {"messages": messages, "answer": answer_str}

    dataset = dataset.map(process)
    dataset = _keep_only(dataset)
    return _apply_length_filter(dataset, tokenizer, max_length)


# ---------------------------------------------------------------------------
# MinervaMAth evaluation dataset
# ---------------------------------------------------------------------------


def get_minervamath_rl_dataset(
    path: str,
    split: str,
    tokenizer: PreTrainedTokenizerFast | None,
    max_length: int | None = None,
) -> Dataset:
    """Load the MinervaMAth evaluation dataset (272 problems, JSONL format).

    Parameters
    ----------
    path:
        Directory containing ``test.jsonl``.
    split:
        Ignored; the file only contains a test split.
    tokenizer:
        Tokenizer used for optional length filtering.
    max_length:
        If set, samples whose user message exceeds this many tokens are
        dropped.
    """
    data_files = {"test": "test.jsonl"}
    dataset = load_dataset("json", data_dir=path, data_files=data_files, split="test")

    def process(sample: dict) -> dict:
        messages = [
            {
                "role": "user",
                "content": sample["question"] + _BOXED_SUFFIX,
            }
        ]
        return {"messages": messages, "answer": str(sample["answer"])}

    dataset = dataset.map(process)
    dataset = _keep_only(dataset)
    return _apply_length_filter(dataset, tokenizer, max_length)
