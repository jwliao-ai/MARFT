"""DeepCoder-Preview-Dataset loader for RL training on coding tasks.

Loads training data from three sources (24K problems total):
- 7.4K TACO Verified problems
- 16.2K PrimeIntellect SYNTHETIC-1 problems
- 599 LiveCodeBench v5 problems (May 2023 – Jul 2024)

Evaluation data (687 problems):
- 279 LiveCodeBench v5 problems (Aug 2024 – Feb 2025)
- 408 Codeforces problems from Qwen/CodeElo

Dataset: https://huggingface.co/datasets/agentica-org/DeepCoder-Preview-Dataset

Usage:
    1. Run the preprocessing step once to convert heavy parquets to JSONL::

        python -m areal.dataset.deepcoder /path/to/DeepCoder-Preview-Dataset

       This creates ``train.jsonl`` and ``test.jsonl`` under the dataset dir.

    2. Point the training config at the dataset directory.  The loader
       will pick up the JSONL files automatically.
"""

import json
import os

from areal.utils import logging

logger = logging.getLogger("DeepCoderDataset")

_SYSTEM_PROMPT = (
    "You are an expert Python programmer. You will be given a question "
    "(problem specification) and will generate a correct Python program "
    "that matches the specification and passes all tests."
)

_STDIN_FORMAT = (
    "Read the inputs from stdin solve the problem and write the answer "
    "to stdout (do not directly test on the sample inputs). Enclose your "
    "code within delimiters as follows. Ensure that when the python program "
    "runs, it reads the inputs, runs the algorithm and writes output to "
    "STDOUT.\n```python\n# YOUR CODE HERE\n```"
)

_FUNCTIONAL_FORMAT_STARTER = (
    "Complete the following starter code. Enclose your code within "
    "delimiters as follows.\n```python\n{starter_code}\n```"
)

_FUNCTIONAL_FORMAT_NAME = (
    "Implement a function called `{func_name}`. Enclose your code within "
    "delimiters as follows.\n```python\ndef {func_name}(...):\n"
    "    # YOUR CODE HERE\n```"
)

_PRIMEINTELLECT_PREFIX = (
    "Solve the following coding problem using the programming language python:\n\n"
)


# ---------------------------------------------------------------------------
# Test-case normalisation helpers
# ---------------------------------------------------------------------------


def _normalize_taco_tests(tests_json: str, max_tests: int | None):
    """Normalize taco format: ``{"inputs": [...], "outputs": [...]}`` dict.

    Some entries also have ``fn_name`` for function-call style problems.
    """
    parsed = json.loads(tests_json)
    fn_name = parsed.get("fn_name")
    test_type = "functional" if fn_name else "stdin"
    inputs = parsed.get("inputs", [])
    outputs = parsed.get("outputs", [])
    test_cases = [
        {
            "input": inp if isinstance(inp, str) else json.dumps(inp),
            "output": out if isinstance(out, str) else json.dumps(out),
        }
        for inp, out in zip(inputs, outputs)
    ]
    if max_tests and len(test_cases) > max_tests:
        test_cases = test_cases[:max_tests]
    return test_cases, test_type, fn_name


def _normalize_list_tests(tests_json: str, max_tests: int | None):
    """Normalize list-of-dicts format used by primeintellect, lcbv5, codeforces.

    Each entry has ``input`` and ``output`` keys (and sometimes ``type``
    or ``testtype``).

    When *max_tests* is set and the raw JSON is larger than 1 MB, uses
    an incremental approach to avoid fully parsing huge test-case blobs
    (lcbv5 rows can exceed 100 MB of test JSON).
    """
    if max_tests and len(tests_json) > 1_000_000:
        return _parse_list_tests_incremental(tests_json, max_tests)

    parsed = json.loads(tests_json)
    test_cases = [{"input": tc["input"], "output": tc["output"]} for tc in parsed]
    if max_tests and len(test_cases) > max_tests:
        test_cases = test_cases[:max_tests]
    return test_cases


def _parse_list_tests_incremental(tests_json: str, max_tests: int):
    """Extract first *max_tests* entries from a JSON array without parsing all.

    Uses ``json.JSONDecoder.raw_decode`` to parse one object at a time,
    stopping after *max_tests* entries.
    """
    decoder = json.JSONDecoder()
    test_cases: list[dict] = []

    pos = 0
    while pos < len(tests_json) and tests_json[pos] in " \t\n\r":
        pos += 1
    if pos >= len(tests_json) or tests_json[pos] != "[":
        return test_cases
    pos += 1

    while len(test_cases) < max_tests:
        while pos < len(tests_json) and tests_json[pos] in " \t\n\r,":
            pos += 1
        if pos >= len(tests_json) or tests_json[pos] == "]":
            break
        try:
            obj, end_pos = decoder.raw_decode(tests_json, pos)
            test_cases.append({"input": obj["input"], "output": obj["output"]})
            pos = end_pos
        except (json.JSONDecodeError, KeyError):
            break

    return test_cases


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------


def _build_user_content(
    problem: str,
    test_type: str,
    starter_code: str = "",
    func_name: str = "",
) -> str:
    if test_type == "functional" and starter_code:
        fmt = _FUNCTIONAL_FORMAT_STARTER.format(starter_code=starter_code)
    elif test_type == "functional" and func_name:
        fmt = _FUNCTIONAL_FORMAT_NAME.format(func_name=func_name)
    else:
        fmt = _STDIN_FORMAT

    return (
        f"{_SYSTEM_PROMPT}\n\n"
        f"### Question:\n{problem}\n\n"
        f"### Format:\n{fmt}\n\n"
        f"### Answer: (use the provided format with backticks)"
    )


# ---------------------------------------------------------------------------
# Parquet row iteration (for preprocessing)
# ---------------------------------------------------------------------------


def _iter_parquet_rows(base_path: str, subset: str, split: str, batch_size: int = 8):
    """Yield rows from parquet files one at a time."""
    import gc

    import pyarrow.parquet as pq

    subdir = os.path.join(base_path, subset)
    if not os.path.isdir(subdir):
        return
    files = sorted(
        os.path.join(subdir, f)
        for f in os.listdir(subdir)
        if f.startswith(f"{split}-") and f.endswith(".parquet")
    )
    for fpath in files:
        pf = pq.ParquetFile(fpath)
        for batch in pf.iter_batches(batch_size=batch_size):
            cols = batch.schema.names
            for i in range(batch.num_rows):
                yield {col: batch.column(col)[i].as_py() for col in cols}
        gc.collect()


def _process_row(row: dict, subset_name: str, max_tests: int | None) -> dict:
    """Normalize a single row into the unified sample format."""
    problem = row["problem"]
    tests_json = row["tests"]
    starter_code = row.get("starter_code", "") or ""
    metadata = row.get("metadata")

    if subset_name == "taco":
        test_cases, test_type, func_name = _normalize_taco_tests(tests_json, max_tests)
    elif subset_name == "primeintellect":
        test_cases = _normalize_list_tests(tests_json, max_tests)
        test_type = "stdin"
        func_name = None
        if problem.startswith(_PRIMEINTELLECT_PREFIX):
            problem = problem[len(_PRIMEINTELLECT_PREFIX) :]
    elif subset_name == "lcbv5":
        test_cases = _normalize_list_tests(tests_json, max_tests)
        fn = metadata.get("func_name") if metadata else None
        test_type = "functional" if fn else "stdin"
        func_name = fn
    elif subset_name == "codeforces":
        test_cases = _normalize_list_tests(tests_json, max_tests)
        test_type = "stdin"
        func_name = None
    else:
        raise ValueError(f"Unknown subset: {subset_name}")

    user_content = _build_user_content(
        problem, test_type, starter_code, func_name or ""
    )
    return {
        "messages": [{"role": "user", "content": user_content}],
        "tests": json.dumps(test_cases),
        "test_type": test_type,
        "func_name": func_name or "",
        "starter_code": starter_code,
        "source": subset_name,
    }


# ---------------------------------------------------------------------------
# Preprocessing CLI: parquet -> JSONL
# ---------------------------------------------------------------------------


def preprocess(raw_dir: str, max_tests: int = 50):
    """Convert raw parquet subsets to ``train.jsonl`` / ``test.jsonl``.

    Should be run once on a machine with sufficient memory (>16 GB).
    Large subsets (lcbv5) are streamed one row at a time.
    """
    import gc

    train_subsets = [("taco", "train"), ("primeintellect", "train"), ("lcbv5", "train")]
    test_subsets = [("lcbv5", "test"), ("codeforces", "test")]

    for out_name, subsets in [
        ("train.jsonl", train_subsets),
        ("test.jsonl", test_subsets),
    ]:
        out_path = os.path.join(raw_dir, out_name)
        total = 0
        with open(out_path, "w") as f:
            for subset_name, subset_split in subsets:
                count = 0
                for row in _iter_parquet_rows(
                    raw_dir, subset_name, subset_split, batch_size=1
                ):
                    sample = _process_row(row, subset_name, max_tests)
                    f.write(json.dumps(sample, ensure_ascii=False) + "\n")
                    count += 1
                    del row, sample
                    if count % 100 == 0:
                        gc.collect()
                total += count
                print(f"  {subset_name}/{subset_split}: {count} samples")
                gc.collect()
        print(f"Wrote {out_path} ({total} samples)")


# ---------------------------------------------------------------------------
# Dataset loader (reads preprocessed JSONL)
# ---------------------------------------------------------------------------


def get_deepcoder_rl_dataset(
    path: str,
    split: str,
    tokenizer=None,
    max_length: int | None = None,
    max_tests: int | None = 50,
):
    """Load DeepCoder-Preview-Dataset for RL training.

    Expects preprocessed ``train.jsonl`` and ``test.jsonl`` in *path*.
    Run ``python -m areal.dataset.deepcoder <path>`` first if they don't
    exist yet.

    Args:
        path: Root directory of the DeepCoder-Preview-Dataset (containing
              the preprocessed JSONL files).
        split: ``train`` or ``test``.
        tokenizer: Tokenizer for prompt-length filtering.
        max_length: Drop samples whose prompt exceeds this many tokens.
        max_tests: Ignored when loading JSONL (tests are already truncated
                   during preprocessing).  Kept for API compatibility.

    Returns:
        HuggingFace ``Dataset`` with columns: ``messages``, ``tests``,
        ``test_type``, ``func_name``, ``starter_code``, ``source``.
    """
    from datasets import load_dataset

    if split in ("test", "validation"):
        jsonl_file = "test.jsonl"
    elif split == "train":
        jsonl_file = "train.jsonl"
    else:
        raise ValueError(f"Unsupported split '{split}'. Use 'train' or 'test'.")

    jsonl_path = os.path.join(path, jsonl_file)
    if not os.path.isfile(jsonl_path):
        raise FileNotFoundError(
            f"{jsonl_path} not found. Run preprocessing first:\n"
            f"  python -m areal.dataset.deepcoder {path}"
        )

    dataset = load_dataset("json", data_files={split: jsonl_path}, split=split)
    logger.info(f"Loaded {len(dataset)} samples from {jsonl_path}")

    if max_length is not None and tokenizer is not None:

        def filter_length(sample):
            tokens = tokenizer.encode(sample["messages"][0]["content"])
            return len(tokens) <= max_length

        before = len(dataset)
        dataset = dataset.filter(filter_length)
        logger.info(
            f"Filtered by max_length={max_length}: {before} -> {len(dataset)} samples"
        )

    return dataset


# ---------------------------------------------------------------------------
# CLI entry point: python -m areal.dataset.deepcoder /path/to/dataset
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Preprocess DeepCoder-Preview-Dataset parquets into JSONL"
    )
    parser.add_argument("path", help="Root directory of DeepCoder-Preview-Dataset")
    parser.add_argument(
        "--max-tests",
        type=int,
        default=50,
        help="Max test cases per problem (default: 50)",
    )
    args = parser.parse_args()
    preprocess(args.path, max_tests=args.max_tests)
