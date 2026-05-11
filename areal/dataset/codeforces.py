"""Dataset loaders for Codeforces RL training and evaluation.

Data format
-----------
Both ``chunked_train_py.json`` (training) and ``test_py.json`` (evaluation)
are JSON arrays.  Each item has the following relevant fields:

.. code-block:: json

    {
      "prompt": [{"role": "user", "content": "...problem statement..."}],
      "reward_model": {
        "ground_truth": {
          "inputs":  ["test-case-1 stdin", "test-case-2 stdin", ...],
          "outputs": ["expected output 1",  "expected output 2",  ...]
        },
        "style": "rule"
      }
    }

The ``prompt`` field is already a fully-formatted chat message list, so we
pass it through directly as ``messages``.  The ``tests`` field is set to a
JSON-encoded list of ``{"input": ..., "output": ...}`` test cases in the
format expected by ``multi_agent_code_reward_fn``.  A ``test_type`` field
of ``"stdin"`` is also emitted because all Codeforces problems are
stdin/stdout style.

Loader functions
----------------
- ``get_codeforces_train_dataset`` — loads ``train_py.json`` (training).
- ``get_codeforces_test_dataset``  — loads ``test_py.json`` (evaluation).

Both accept the same signature as other AReaL dataset loaders so they can be
used transparently via ``get_custom_dataset``.

Columns emitted
---------------
``messages``   — chat-formatted prompt list, passed through from ``prompt``.
``tests``      — JSON-encoded list of ``{"input": str, "output": str}`` test cases.
                 This name matches the ``tests`` kwarg in ``multi_agent_code_reward_fn``.
``test_type``  — always ``"stdin"``; Codeforces problems are stdin/stdout style.
"""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING

from datasets import load_dataset

if TYPE_CHECKING:
    from datasets import Dataset
    from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

# Maximum number of test cases to keep per sample.  Codeforces problems can
# have many generated tests; we cap them to keep reward computation fast.
_MAX_TESTS = 20


def _apply_length_filter(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizerFast | None,
    max_length: int | None,
) -> Dataset:
    if max_length is None or tokenizer is None:
        return dataset

    def _within_length(sample: dict) -> bool:
        # messages is a list of dicts; measure the user message content.
        msgs = sample["messages"]
        user_content = next(
            (m["content"] for m in msgs if m.get("role") == "user"), ""
        )
        return len(tokenizer.encode(user_content)) <= max_length

    return dataset.filter(_within_length)


def _keep_only(dataset: Dataset) -> Dataset:
    columns_to_remove = [
        c for c in dataset.column_names if c not in ("messages", "tests", "test_type")
    ]
    return dataset.remove_columns(columns_to_remove)


def _process_sample(sample: dict) -> dict:
    """Convert a raw Codeforces JSON item into (messages, tests, test_type).

    ``messages`` is passed through as-is from the ``prompt`` field (it is
    already a chat-formatted list).  ``tests`` is a JSON-encoded list of
    ``{"input": str, "output": str}`` test cases derived from
    ``reward_model.ground_truth`` — the field name matches the ``tests``
    kwarg in ``multi_agent_code_reward_fn``.  ``test_type`` is always
    ``"stdin"`` because all Codeforces problems use stdin/stdout I/O.

    Parameters
    ----------
    sample:
        Raw dict loaded from the JSON file.

    Returns
    -------
    dict with keys ``messages``, ``tests``, and ``test_type``.
    """
    messages: list[dict] = sample["prompt"]

    ground_truth: dict = sample["reward_model"]["ground_truth"]
    inputs: list[str] = ground_truth.get("inputs", [])
    outputs: list[str] = ground_truth.get("outputs", [])

    # Zip inputs/outputs into test-case dicts; cap to _MAX_TESTS.
    test_cases = [
        {"input": inp, "output": out}
        for inp, out in zip(inputs, outputs)
    ][:_MAX_TESTS]

    return {
        "messages": messages,
        "tests": json.dumps(test_cases),
        "test_type": "stdin",
    }


# ---------------------------------------------------------------------------
# Training dataset  (chunked_train_py.json)
# ---------------------------------------------------------------------------


def get_codeforces_train_dataset(
    path: str,
    split: str,
    tokenizer: PreTrainedTokenizerFast | None,
    max_length: int | None = None,
) -> Dataset:
    """Load the Codeforces training dataset (``chunked_train_py.json``).

    Parameters
    ----------
    path:
        Directory containing ``train_py.json``
        (i.e. ``…/verifiable-prompts``).
    split:
        Accepted for API compatibility; ignored — the file contains only
        training data.
    tokenizer:
        Tokenizer used for optional prompt-length filtering.
    max_length:
        If set, samples whose user message exceeds this many tokens are
        dropped before returning the dataset.

    Returns
    -------
    HuggingFace ``Dataset`` with columns ``messages``, ``tests``, and ``test_type``.
    The ``tests`` column holds a JSON-encoded list of
    ``{"input": str, "output": str}`` test-case dicts.
    """
    json_file = os.path.join(path, "train_py.json")
    dataset = load_dataset(
        "json",
        data_files={"train": json_file},
        split="train",
    )
    dataset = dataset.map(_process_sample, remove_columns=dataset.column_names)
    return _apply_length_filter(dataset, tokenizer, max_length)


# ---------------------------------------------------------------------------
# Evaluation dataset  (test_py.json)
# ---------------------------------------------------------------------------


def get_codeforces_test_dataset(
    path: str,
    split: str,
    tokenizer: PreTrainedTokenizerFast | None,
    max_length: int | None = None,
) -> Dataset:
    """Load the Codeforces evaluation dataset (``test_py.json``).

    Parameters
    ----------
    path:
        Directory containing ``test_py.json``
        (i.e. ``…/verifiable-prompts``).
    split:
        Accepted for API compatibility; ignored — the file contains only
        test data.
    tokenizer:
        Tokenizer used for optional prompt-length filtering.
    max_length:
        If set, samples whose user message exceeds this many tokens are
        dropped before returning the dataset.

    Returns
    -------
    HuggingFace ``Dataset`` with columns ``messages``, ``tests``, and ``test_type``.
    """
    json_file = os.path.join(path, "test_py.json")
    dataset = load_dataset(
        "json",
        data_files={"test": json_file},
        split="test",
    )
    dataset = dataset.map(_process_sample, remove_columns=dataset.column_names)
    return _apply_length_filter(dataset, tokenizer, max_length)
