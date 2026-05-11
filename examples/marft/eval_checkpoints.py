"""Checkpoint evaluation script for MARFT experiments.

Loads LoRA checkpoints onto a running SGLang server, generates completions
for the test dataset, and computes rewards using the same reward functions
as training (code execution for deepcoder, math verification for gsm8k/math).

Supports both single-adapter (shared LoRA) and per-agent (multi-LoRA)
evaluation with multi-agent sequential conversation.

Usage:
    python examples/marft/eval_checkpoints.py \
        --checkpoint-root ecmlp_experiments_deepcoder/checkpoints/root \
        --config-root ecmlp_experiments_deepcoder/logs/root \
        --dataset-path /path/to/DeepCoder-Preview-Dataset \
        --benchmark deepcoder \
        --output-dir ecmlp_experiments_deepcoder/eval_results \
        --base-model /path/to/Qwen2.5-3B-Instruct \
        --sglang-port 30000 \
        --resume
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import math
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import aiohttp
from tqdm import tqdm

try:
    from areal.utils import logging as _areal_logging

    logger = _areal_logging.getLogger("CheckpointEval")
except ImportError:
    import logging as _stdlib_logging

    logger = _stdlib_logging.getLogger("CheckpointEval")
    if not logger.handlers:
        _handler = _stdlib_logging.StreamHandler()
        _handler.setFormatter(
            _stdlib_logging.Formatter("[%(name)s] %(message)s")
        )
        logger.addHandler(_handler)
        logger.setLevel(_stdlib_logging.INFO)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class EvalTask:
    """A single checkpoint to evaluate."""

    experiment_name: str
    trial_name: str
    step_dir: str
    global_step: int
    checkpoint_path: str  # path to the step dir under default/
    config: dict
    is_multi_agent: bool
    adapter_paths: dict[str, str]  # {adapter_name: path}
    role_names: list[str]
    role_configs: dict  # {role_name: {system_prompt, description, ...}}
    shared_lora: bool
    benchmark: str
    n_agents: int
    lora_mode: str  # "shared" | "peragent" | "none"
    critic_mode: str  # "ctde" | "multihead" | "criticlora"
    is_anonymous: bool
    max_new_tokens: int


@dataclass
class EvalResult:
    """Result for a single sample.

    Individual seed results have ``seed >= 0``.  Aggregated records
    (mean/std over all seeds) use ``seed = -1`` and populate the
    ``reward_mean`` / ``reward_std`` fields instead of ``reward``.
    """

    experiment: str
    trial: str
    step_dir: str
    global_step: int
    sample_idx: int
    seed: int  # 0..N-1 for individual, -1 for aggregate
    reward: float
    n_agents: int
    is_anonymous: bool
    lora_mode: str
    critic_mode: str
    benchmark: str = ""
    reward_mean: float = 0.0
    reward_std: float = 0.0
    rewards: list[float] | None = None  # per-seed rewards (aggregate only)
    completion: str = ""


# ---------------------------------------------------------------------------
# Checkpoint discovery
# ---------------------------------------------------------------------------

_GLOBAL_STEP_RE = re.compile(r"globalstep(\d+)")


def _parse_global_step(step_dir: str) -> int:
    """Extract global step number from directory name."""
    m = _GLOBAL_STEP_RE.search(step_dir)
    return int(m.group(1)) if m else -1


def _parse_experiment_name(name: str) -> dict:
    """Parse experiment name into components."""
    info: dict = {
        "is_anonymous": "anonymous" in name,
        "lora_mode": "shared",
        "critic_mode": "ctde",
        "n_agents": 1,
    }

    # Extract agent count
    m = re.search(r"(\d)agent", name)
    if m:
        info["n_agents"] = int(m.group(1))

    # LoRA mode
    if "peragent" in name:
        info["lora_mode"] = "peragent"
    elif "shared" in name:
        info["lora_mode"] = "shared"

    # Critic mode
    if "criticlora" in name:
        info["critic_mode"] = "criticlora"
    elif "multihead" in name:
        info["critic_mode"] = "multihead"

    return info


def discover_checkpoints(
    checkpoint_root: str,
    config_root: str,
    benchmark: str,
    filter_pattern: str | None = None,
) -> list[EvalTask]:
    """Scan checkpoint directory structure and return evaluation tasks."""
    tasks: list[EvalTask] = []
    ckpt_root = Path(checkpoint_root)

    if not ckpt_root.exists():
        logger.warning("Checkpoint root does not exist: %s", ckpt_root)
        return tasks

    for exp_dir in sorted(ckpt_root.iterdir()):
        if not exp_dir.is_dir():
            continue
        exp_name = exp_dir.name

        # Apply filter
        if filter_pattern and filter_pattern not in exp_name:
            continue

        exp_info = _parse_experiment_name(exp_name)

        for trial_dir in sorted(exp_dir.iterdir()):
            if not trial_dir.is_dir():
                continue
            trial_name = trial_dir.name

            # Read config
            config_path = Path(config_root) / exp_name / trial_name / "config.yaml"
            config = _load_config(config_path)
            if config is None:
                logger.warning("Config not found at %s, skipping", config_path)
                continue

            # Extract multi-agent settings
            ma_config = config.get("multi_agent", {})
            role_names = ma_config.get("role_names", [])
            role_configs = ma_config.get("role_configs", {})
            shared_lora = ma_config.get("shared_lora", True)
            use_multi_lora = ma_config.get("use_multi_lora", False)
            is_multi_agent = len(role_names) > 1

            # Get max_new_tokens from gconfig
            gconfig = config.get("gconfig", {})
            max_new_tokens = gconfig.get("max_new_tokens", 13824)

            # Override lora_mode from config if available
            lora_mode = exp_info["lora_mode"]
            if not use_multi_lora:
                lora_mode = "none"
            elif shared_lora:
                lora_mode = "shared"
            elif not shared_lora:
                lora_mode = "peragent"

            # Scan step directories
            default_dir = trial_dir / "default"
            if not default_dir.exists():
                continue

            for step_dir in sorted(default_dir.iterdir()):
                if not step_dir.is_dir():
                    continue

                global_step = _parse_global_step(step_dir.name)
                if global_step < 0:
                    continue

                # Determine adapter paths
                adapter_paths = _discover_adapters(
                    step_dir, is_multi_agent, shared_lora, use_multi_lora
                )

                if not adapter_paths:
                    logger.warning(
                        "No adapters found in %s, skipping", step_dir
                    )
                    continue

                tasks.append(
                    EvalTask(
                        experiment_name=exp_name,
                        trial_name=trial_name,
                        step_dir=step_dir.name,
                        global_step=global_step,
                        checkpoint_path=str(step_dir),
                        config=config,
                        is_multi_agent=is_multi_agent,
                        adapter_paths=adapter_paths,
                        role_names=role_names,
                        role_configs=role_configs,
                        shared_lora=shared_lora,
                        benchmark=benchmark,
                        n_agents=exp_info["n_agents"],
                        lora_mode=lora_mode,
                        critic_mode=exp_info["critic_mode"],
                        is_anonymous=exp_info["is_anonymous"],
                        max_new_tokens=max_new_tokens,
                    )
                )

    return tasks


def _discover_adapters(
    step_dir: Path,
    is_multi_agent: bool,
    shared_lora: bool,
    use_multi_lora: bool,
) -> dict[str, str]:
    """Discover adapter files within a step directory.

    Returns {adapter_name: adapter_path}.
    """
    adapters: dict[str, str] = {}

    # Check for per-agent subdirectories (agent_planner, agent_solver, etc.)
    subdirs = [
        d for d in step_dir.iterdir()
        if d.is_dir() and d.name.startswith("agent_")
    ]

    if subdirs:
        # Per-agent adapters
        for subdir in sorted(subdirs):
            adapter_file = subdir / "adapter_model.safetensors"
            if adapter_file.exists():
                adapters[subdir.name] = str(subdir)
    elif (step_dir / "adapter_model.safetensors").exists():
        # Shared or single adapter
        adapters["default"] = str(step_dir)

    return adapters


def _load_config(path: Path) -> dict | None:
    """Load YAML config file."""
    if not path.exists():
        return None
    try:
        import yaml

        with open(path) as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.warning("Failed to load config %s: %s", path, e)
        return None


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------


def _load_jsonl(path: str) -> list[dict]:
    """Load items from a JSONL file."""
    items: list[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def _load_parquet(path: str) -> list[dict]:
    """Load items from a Parquet file."""
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("pandas is required for parquet datasets: pip install pandas pyarrow")
    df = pd.read_parquet(path)
    return df.to_dict(orient="records")


def _extract_gsm8k_answer(answer_str: str) -> str:
    """Extract numeric answer after '####' from GSM8K-style answer strings."""
    if "####" in answer_str:
        return answer_str.split("####", 1)[-1].strip()
    return answer_str.strip()


def _normalize_livecodebench(item: dict) -> None:
    """Normalize a LiveCodeBench item in-place.

    Builds 'messages' from question_content, and 'tests'/'test_type' from
    public_test_cases + private_test_cases (base64+zlib+pickle encoded).
    """
    import base64
    import pickle
    import zlib

    # Build messages from question_content
    content = item.get("question_content", "")
    starter = item.get("starter_code", "")
    if starter:
        content += f"\n\nStarter code:\n```python\n{starter}\n```"
    content += (
        "\n\nPlease write a Python solution. "
        "Read input from stdin and print output to stdout. "
        "Enclose your code within ```python ... ``` delimiters."
    )
    item["messages"] = [{"role": "user", "content": content}]

    # Collect test cases: public (JSON string) + private (base64+zlib+pickle)
    all_tests: list[dict] = []

    public = item.get("public_test_cases", "[]")
    if isinstance(public, str):
        try:
            all_tests.extend(json.loads(public))
        except (json.JSONDecodeError, TypeError):
            pass

    private = item.get("private_test_cases", "")
    if private and isinstance(private, str):
        try:
            decoded = base64.b64decode(private)
            decompressed = zlib.decompress(decoded)
            obj = pickle.loads(decompressed)  # noqa: S301
            if isinstance(obj, str):
                all_tests.extend(json.loads(obj))
            elif isinstance(obj, list):
                all_tests.extend(obj)
        except Exception:
            pass  # Use public tests only

    # Normalize test case keys to {"input": ..., "output": ...}
    normalized: list[dict] = []
    for tc in all_tests:
        normalized.append({
            "input": tc.get("input", ""),
            "output": tc.get("output", ""),
        })

    item["tests"] = json.dumps(normalized)
    item["test_type"] = "stdin"


_CODEFORCES_MAX_TESTS = 20


def _normalize_codeforces(item: dict) -> None:
    """Normalize a Codeforces item in-place.

    Expects the preprocessed format with 'prompt' (list of message dicts)
    and 'reward_model.ground_truth.inputs/outputs'.
    """
    # Messages from prompt field
    prompt = item.get("prompt", [])
    if isinstance(prompt, list) and len(prompt) > 0:
        item["messages"] = list(prompt)
    elif isinstance(prompt, str):
        item["messages"] = [{"role": "user", "content": prompt}]

    # Tests from reward_model.ground_truth
    rm = item.get("reward_model", {})
    if isinstance(rm, str):
        rm = json.loads(rm)
    gt = rm.get("ground_truth", {}) if isinstance(rm, dict) else {}

    if isinstance(gt, dict):
        inputs = gt.get("inputs", [])
        outputs = gt.get("outputs", [])
        test_cases = [
            {"input": inp, "output": out}
            for inp, out in zip(inputs, outputs)
        ][:_CODEFORCES_MAX_TESTS]
    else:
        test_cases = []

    item["tests"] = json.dumps(test_cases)
    item["test_type"] = "stdin"


def load_dataset(dataset_path: str, benchmark: str, max_samples: int | None = None) -> list[dict]:
    """Load test dataset from JSONL or Parquet files.

    Supports multiple formats:
    - JSONL: test.jsonl (deepcoder, math, math500, minervamath, livecodebench)
    - Parquet: test.parquet (deepscaler, aime2024, olympiadbench)
    - Parquet: main/test-*.parquet (gsm8k HuggingFace)
    - JSON: test_py.json (codeforces)
    """
    import glob as _glob

    raw_items: list[dict] = []
    source_path = ""

    # Benchmark-specific file resolution
    if benchmark == "livecodebench":
        # LiveCodeBench: code_generation_lite/test.jsonl (+ test2..test6 for more)
        lcb_dir = os.path.join(dataset_path, "code_generation_lite")
        if os.path.isdir(lcb_dir):
            jsonl_files = sorted(_glob.glob(os.path.join(lcb_dir, "test*.jsonl")))
            for jf in jsonl_files:
                raw_items.extend(_load_jsonl(jf))
            source_path = lcb_dir
        else:
            # Fallback: test.jsonl directly in dataset_path
            test_jsonl = os.path.join(dataset_path, "test.jsonl")
            if os.path.exists(test_jsonl):
                raw_items = _load_jsonl(test_jsonl)
                source_path = test_jsonl
    elif benchmark == "codeforces":
        # Codeforces: preprocessed/verifiable-prompts/test_py.json
        test_json = os.path.join(
            dataset_path, "preprocessed", "verifiable-prompts", "test_py.json"
        )
        if os.path.exists(test_json):
            with open(test_json) as f:
                raw_items = json.load(f)
            source_path = test_json
        else:
            # Fallback: test_py.json directly in dataset_path
            test_json2 = os.path.join(dataset_path, "test_py.json")
            if os.path.exists(test_json2):
                with open(test_json2) as f:
                    raw_items = json.load(f)
                source_path = test_json2
    else:
        # Generic: try test.jsonl, test.parquet, main/test-*.parquet
        test_jsonl = os.path.join(dataset_path, "test.jsonl")
        test_parquet = os.path.join(dataset_path, "test.parquet")
        main_dir = os.path.join(dataset_path, "main")

        data_dir = os.path.join(dataset_path, "data")

        if os.path.exists(test_jsonl):
            raw_items = _load_jsonl(test_jsonl)
            source_path = test_jsonl
        elif os.path.exists(test_parquet):
            raw_items = _load_parquet(test_parquet)
            source_path = test_parquet
        elif os.path.isdir(main_dir):
            parquet_files = sorted(
                _glob.glob(os.path.join(main_dir, "test-*.parquet"))
            )
            if parquet_files:
                for pf in parquet_files:
                    raw_items.extend(_load_parquet(pf))
                source_path = main_dir
        elif os.path.isdir(data_dir):
            # HuggingFace dataset layout: data/train-*.parquet or data/*.parquet
            parquet_files = sorted(
                _glob.glob(os.path.join(data_dir, "*.parquet"))
            )
            if parquet_files:
                for pf in parquet_files:
                    raw_items.extend(_load_parquet(pf))
                source_path = data_dir

    if not source_path:
        raise FileNotFoundError(
            f"No test dataset found in {dataset_path} for benchmark={benchmark}."
        )
    if not raw_items:
        raise ValueError(f"Dataset at {source_path} is empty")

    # -----------------------------------------------------------------------
    # Normalize items to a common format
    #   Code benchmarks: need 'messages', 'tests', 'test_type'
    #   Math benchmarks: need 'messages', 'answer'
    # -----------------------------------------------------------------------
    MATH_BENCHMARKS = ("gsm8k", "math", "deepscaler", "aime2024", "math500",
                       "minervamath", "olympiadbench")

    items: list[dict] = []
    for raw in raw_items:
        item = dict(raw)

        # --- Code benchmark normalization ---
        if benchmark == "livecodebench":
            _normalize_livecodebench(item)
        elif benchmark == "codeforces":
            _normalize_codeforces(item)

        # --- Math benchmark normalization ---
        # DeepScaleR / AIME2024 format: reward_model.ground_truth + prompt
        # (skip for code benchmarks which also have reward_model but use tests)
        if benchmark in MATH_BENCHMARKS and "reward_model" in item and "prompt" in item:
            rm = item["reward_model"]
            if isinstance(rm, str):
                rm = json.loads(rm)
            if isinstance(rm, dict):
                raw_gt = rm.get("ground_truth", "")
                item["answer"] = f"\\boxed{{{raw_gt}}}"
            prompt_val = item["prompt"]
            if hasattr(prompt_val, "tolist"):
                prompt_val = prompt_val.tolist()
            if isinstance(prompt_val, list) and len(prompt_val) > 0:
                item["messages"] = list(prompt_val)

        # GSM8K: answer with "#### <number>"
        if benchmark == "gsm8k" and "answer" in item:
            item["answer"] = _extract_gsm8k_answer(str(item["answer"]))

        # OlympiadBench: final_answer field (may be list/ndarray)
        if benchmark == "olympiadbench" and "final_answer" in item:
            fa = item["final_answer"]
            if hasattr(fa, "tolist"):
                fa = fa.tolist()
            if isinstance(fa, list):
                fa = fa[0] if len(fa) == 1 else str(fa)
            # Strip surrounding $ signs from LaTeX answers.
            # Do NOT wrap in \boxed{} — MathVerifyWorker.verify wraps automatically.
            item["answer"] = str(fa).strip().strip("$")

        # Ensure messages field exists for math benchmarks
        if "messages" not in item and benchmark in MATH_BENCHMARKS:
            problem_key = "question" if "question" in item else "problem"
            content = item.get(problem_key, "")
            if content and not content.endswith("\\boxed{}."):
                content += "\nPlease put your final answer within \\boxed{}."
            item["messages"] = [{"role": "user", "content": content}]

        # Ensure messages field exists for deepcoder (already has it normally)
        if "messages" not in item and benchmark == "deepcoder":
            if "prompt" in item:
                item["messages"] = [{"role": "user", "content": item["prompt"]}]

        items.append(item)
        if max_samples and len(items) >= max_samples:
            break

    # Apply max_samples if we loaded all then trim
    if max_samples and len(items) > max_samples:
        items = items[:max_samples]

    logger.info("Loaded %d test samples from %s", len(items), source_path)
    return items


# ---------------------------------------------------------------------------
# SGLang server management
# ---------------------------------------------------------------------------


def _port_is_alive(port: int, retries: int = 3, interval: float = 2.0) -> bool:
    """Return True if something is already listening on *port*.

    Retries a few times to tolerate momentary server unavailability
    (e.g. during LoRA subsystem initialization).
    """
    import urllib.request

    for attempt in range(retries):
        try:
            urllib.request.urlopen(f"http://localhost:{port}/health", timeout=5)
            return True
        except Exception:
            if attempt < retries - 1:
                time.sleep(interval)
    return False


def start_sglang_server(
    base_model: str,
    port: int,
    tp_size: int = 1,
    max_lora_rank: int = 32,
    mem_fraction: float = 0.85,
    log_dir: str = "",
) -> tuple[subprocess.Popen, str]:
    """Start SGLang server with LoRA support enabled.

    Raises RuntimeError if something is already listening on *port*.
    Returns (process, log_path).
    """
    if _port_is_alive(port):
        raise RuntimeError(
            f"Port {port} is already in use by another process. "
            f"Kill the existing server first (e.g., `kill $(lsof -t -i:{port})`) "
            f"or choose a different port with --sglang-port."
        )

    cmd = [
        sys.executable, "-m", "sglang.launch_server",
        "--model-path", base_model,
        "--port", str(port),
        "--tp-size", str(tp_size),
        "--enable-lora",
        "--max-lora-rank", str(max_lora_rank),
        "--lora-target-modules", "all",
        "--max-loaded-loras", "8",
        "--max-loras-per-batch", "8",
        "--mem-fraction-static", str(mem_fraction),
        "--trust-remote-code",
    ]
    logger.info("Starting SGLang server: %s", " ".join(cmd))

    # Redirect server output to a log file to avoid pipe-buffer deadlock.
    # (SGLang prints lots of output during startup; a subprocess.PIPE that
    # is never drained will fill the 64 KB OS buffer and block the server.)
    log_path = os.path.join(log_dir or ".", f"sglang_server_{port}.log")
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    log_fh = open(log_path, "w")
    logger.info("SGLang server log: %s", log_path)

    proc = subprocess.Popen(cmd, stdout=log_fh, stderr=subprocess.STDOUT)
    # Attach file handle so we can close it later
    proc._log_fh = log_fh  # type: ignore[attr-defined]
    return proc, log_path


def wait_for_server(port: int, timeout: int = 300) -> bool:
    """Wait for SGLang server to become healthy."""
    import urllib.request

    url = f"http://localhost:{port}/health"
    start = time.monotonic()
    while time.monotonic() - start < timeout:
        try:
            resp = urllib.request.urlopen(url, timeout=5)
            if resp.status == 200:
                logger.info("SGLang server is ready on port %d", port)
                return True
        except Exception:
            pass
        time.sleep(2)
    logger.warning("Timeout waiting for SGLang server on port %d", port)
    return False


def load_adapters(port: int, adapter_paths: dict[str, str]) -> bool:
    """Load LoRA adapters via HTTP."""
    import urllib.request

    for name, path in adapter_paths.items():
        url = f"http://localhost:{port}/load_lora_adapter"
        payload = json.dumps({"lora_name": name, "lora_path": path}).encode()
        req = urllib.request.Request(
            url, data=payload, headers={"Content-Type": "application/json"}
        )
        try:
            resp = urllib.request.urlopen(req, timeout=30)
            if resp.status != 200:
                logger.warning("Failed to load adapter %s: HTTP %d", name, resp.status)
                return False
            logger.info("Loaded adapter '%s' from %s", name, path)
        except Exception as e:
            logger.warning("Failed to load adapter %s: %s", name, e)
            return False
    return True


def unload_adapters(port: int, adapter_names: list[str]) -> None:
    """Unload LoRA adapters via HTTP."""
    import urllib.request

    for name in adapter_names:
        url = f"http://localhost:{port}/unload_lora_adapter"
        payload = json.dumps({"lora_name": name}).encode()
        req = urllib.request.Request(
            url, data=payload, headers={"Content-Type": "application/json"}
        )
        try:
            urllib.request.urlopen(req, timeout=30)
        except Exception:
            pass  # Ignore errors on unload


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------


async def _generate_one(
    session: aiohttp.ClientSession,
    port: int,
    messages: list[dict],
    lora_name: str | None,
    max_new_tokens: int,
    temperature: float,
    semaphore: asyncio.Semaphore,
) -> str:
    """Generate a single completion via SGLang /v1/chat/completions."""
    url = f"http://localhost:{port}/v1/chat/completions"
    payload: dict = {
        "model": "default",
        "messages": messages,
        "max_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": 1.0,
    }
    if lora_name:
        payload["model"] = lora_name

    async with semaphore:
        for attempt in range(3):
            try:
                async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=600)) as resp:
                    if resp.status != 200:
                        text = await resp.text()
                        if attempt < 2:
                            await asyncio.sleep(2)
                            continue
                        logger.warning("HTTP %d: %s", resp.status, text[:200])
                        return ""
                    data = await resp.json()
                    return data["choices"][0]["message"]["content"]
            except asyncio.TimeoutError:
                if attempt < 2:
                    await asyncio.sleep(2)
                    continue
                logger.warning("Request timed out")
                return ""
            except Exception as e:
                if attempt < 2:
                    await asyncio.sleep(2)
                    continue
                logger.warning("Request error: %s", e)
                return ""
    return ""


async def eval_single_agent(
    port: int,
    dataset_items: list[dict],
    lora_name: str,
    max_new_tokens: int,
    temperature: float,
    max_concurrent: int,
) -> list[str]:
    """Generate completions for single-agent evaluation."""
    semaphore = asyncio.Semaphore(max_concurrent)

    async with aiohttp.ClientSession() as session:
        tasks = []
        for item in dataset_items:
            messages = list(item["messages"])
            tasks.append(
                _generate_one(
                    session, port, messages, lora_name,
                    max_new_tokens, temperature, semaphore,
                )
            )
        completions = await asyncio.gather(*tasks)
    return list(completions)


async def eval_multi_agent(
    port: int,
    dataset_items: list[dict],
    adapter_paths: dict[str, str],
    role_names: list[str],
    role_configs: dict,
    max_new_tokens: int,
    shared_lora: bool,
    temperature: float,
    max_concurrent: int,
) -> list[str]:
    """Sequential multi-agent generation for all items.

    For each item, agents take turns generating in sequence. Each agent
    sees the system prompt + full conversation history.
    """
    semaphore = asyncio.Semaphore(max_concurrent)

    async with aiohttp.ClientSession() as session:
        tasks = []
        for item in dataset_items:
            tasks.append(
                _eval_multi_agent_one(
                    session, port, item, adapter_paths, role_names,
                    role_configs, max_new_tokens, shared_lora,
                    temperature, semaphore,
                )
            )
        completions = await asyncio.gather(*tasks)
    return list(completions)


async def _eval_multi_agent_one(
    session: aiohttp.ClientSession,
    port: int,
    item: dict,
    adapter_paths: dict[str, str],
    role_names: list[str],
    role_configs: dict,
    max_new_tokens: int,
    shared_lora: bool,
    temperature: float,
    semaphore: asyncio.Semaphore,
) -> str:
    """Run sequential multi-agent conversation for one sample."""
    # Start with user messages from dataset
    conversation: list[dict] = list(item["messages"])

    per_agent_tokens = max_new_tokens  # Already scaled by N in config

    for i, role_name in enumerate(role_names):
        role_cfg = role_configs.get(role_name, {})
        system_prompt = role_cfg.get("system_prompt", "")

        # Build messages for this agent: system + conversation so far
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.extend(conversation)

        # Determine LoRA adapter name (None = base model, no LoRA)
        if not adapter_paths:
            lora_name = None
        elif shared_lora:
            lora_name = "default"
        else:
            # Per-agent: adapter named "agent_{role_name}"
            lora_name = f"agent_{role_name}"

        # Generate
        response = await _generate_one(
            session, port, messages, lora_name,
            per_agent_tokens, temperature, semaphore,
        )

        # Add assistant response to conversation
        conversation.append({"role": "assistant", "content": response})

        # Add transition message if not the last agent
        if i < len(role_names) - 1:
            next_role = role_names[i + 1]
            next_cfg = role_configs.get(next_role, {})
            description = next_cfg.get("description", "")
            transition = f"Now it is {next_role}'s turn. {description}".strip()
            conversation.append({"role": "user", "content": transition})

    # Concatenate all assistant responses for reward computation,
    # matching the training reward which sees the full agent sequence.
    all_responses = [
        m["content"] for m in conversation if m["role"] == "assistant"
    ]
    return "\n\n".join(all_responses)


# ---------------------------------------------------------------------------
# Reward computation
# ---------------------------------------------------------------------------


def compute_reward(completion: str, item: dict, benchmark: str) -> float:
    """Compute reward using the appropriate reward function."""
    CODE_BENCHMARKS = ("deepcoder", "livecodebench", "codeforces")
    MATH_BENCHMARKS = ("gsm8k", "math", "deepscaler", "aime2024", "math500",
                       "minervamath", "olympiadbench")

    if benchmark in CODE_BENCHMARKS:
        from areal.reward.code_execution import deepcoder_reward_fn

        return deepcoder_reward_fn(
            prompt=item["messages"][0]["content"],
            completions=completion,
            prompt_ids=[],
            completion_ids=[],
            tests=item.get("tests", "[]"),
            test_type=item.get("test_type", "stdin"),
            func_name=item.get("func_name", ""),
            starter_code=item.get("starter_code", ""),
        )
    elif benchmark in MATH_BENCHMARKS:
        from areal.reward.gsm8k import gsm8k_reward_fn

        return gsm8k_reward_fn(
            prompt=item["messages"][0]["content"],
            completions=completion,
            prompt_ids=[],
            completion_ids=[],
            answer=item.get("answer", ""),
        )
    else:
        raise ValueError(f"Unsupported benchmark: {benchmark}")


# ---------------------------------------------------------------------------
# Resume support
# ---------------------------------------------------------------------------


def load_completed(output_path: str) -> set[tuple[str, int, int]]:
    """Load already-evaluated (step_dir, sample_idx, seed) triples from JSONL."""
    completed: set[tuple[str, int, int]] = set()
    if not os.path.exists(output_path):
        return completed
    with open(output_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                completed.add(
                    (record["step_dir"], record["sample_idx"], record["seed"])
                )
            except (json.JSONDecodeError, KeyError):
                continue
    return completed


# ---------------------------------------------------------------------------
# Output writing
# ---------------------------------------------------------------------------


def write_result(output_path: str, result: EvalResult) -> None:
    """Append a single result to JSONL file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    record: dict = {
        "experiment": result.experiment,
        "trial": result.trial,
        "step_dir": result.step_dir,
        "global_step": result.global_step,
        "sample_idx": result.sample_idx,
        "seed": result.seed,
        "reward": result.reward,
        "n_agents": result.n_agents,
        "is_anonymous": result.is_anonymous,
        "lora_mode": result.lora_mode,
        "critic_mode": result.critic_mode,
        "benchmark": result.benchmark,
    }
    if result.seed == -1:
        # Aggregate record
        record["reward_mean"] = result.reward_mean
        record["reward_std"] = result.reward_std
        record["rewards"] = result.rewards or []
    with open(output_path, "a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_summary(
    output_dir: str, all_results: list[EvalResult], n_seeds: int
) -> None:
    """Aggregate results and write summary CSV.

    For each (experiment, step), computes per-seed average reward across
    samples, then reports mean ± std of those per-seed averages.
    """
    if not all_results:
        logger.info("No results to summarize.")
        return

    # Only use individual seed records (seed >= 0) for the summary
    seed_results = [r for r in all_results if r.seed >= 0]
    if not seed_results:
        logger.info("No individual seed results to summarize.")
        return

    # Group by (experiment, step_dir, global_step, seed) → list of rewards
    per_seed: dict[tuple[str, str, int, int], list[float]] = {}
    for r in seed_results:
        key = (r.experiment, r.step_dir, r.global_step, r.seed)
        per_seed.setdefault(key, []).append(r.reward)

    # Collapse to (experiment, step_dir, global_step) → {seed: avg_reward}
    per_step: dict[tuple[str, str, int], dict[int, float]] = {}
    per_step_n: dict[tuple[str, str, int], int] = {}
    for (exp, sd, gs, seed), rewards in per_seed.items():
        step_key = (exp, sd, gs)
        avg_for_seed = sum(rewards) / len(rewards)
        per_step.setdefault(step_key, {})[seed] = avg_for_seed
        per_step_n[step_key] = len(rewards)

    # Determine max seed index across all results for column headers
    all_seed_ids = sorted({s for _, _, _, s in per_seed.keys()})

    # Print and write CSV (merge with existing rows from prior runs)
    summary_path = os.path.join(output_dir, "summary.csv")
    os.makedirs(output_dir, exist_ok=True)

    seed_columns = [f"seed_{s}" for s in range(n_seeds)]
    fieldnames = [
        "experiment",
        "step_dir",
        "global_step",
        "reward_mean",
        "reward_std",
        "n_seeds",
        "n_samples_per_seed",
    ] + seed_columns

    header = f"\n{'=' * 95}\n"
    header += (
        f"{'Experiment':<50} {'Step':>6}  "
        f"{'Mean':>8}  {'Std':>8}  {'Seeds':>5}  {'N/Seed':>6}\n"
    )
    header += "-" * 95
    logger.info(header)

    rows: dict[tuple[str, str, int], dict] = {}

    # Load existing rows so we don't lose results from prior filter runs
    if os.path.exists(summary_path):
        with open(summary_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = (row["experiment"], row["step_dir"], int(row["global_step"]))
                rows[key] = row

    # Overwrite with current (possibly updated) results
    for step_key in per_step:
        exp, step_dir, global_step = step_key
        seed_avgs = per_step[step_key]
        vals = list(seed_avgs.values())
        mean_val = sum(vals) / len(vals)
        std_val = (
            math.sqrt(sum((v - mean_val) ** 2 for v in vals) / len(vals))
            if len(vals) > 1
            else 0.0
        )
        n_per_seed = per_step_n[step_key]
        row = {
            "experiment": exp,
            "step_dir": step_dir,
            "global_step": global_step,
            "reward_mean": round(mean_val, 6),
            "reward_std": round(std_val, 6),
            "n_seeds": len(vals),
            "n_samples_per_seed": n_per_seed,
        }
        # Add per-seed columns
        for s in range(n_seeds):
            col = f"seed_{s}"
            if s in seed_avgs:
                row[col] = round(seed_avgs[s], 6)
            else:
                row[col] = ""
        rows[(exp, step_dir, global_step)] = row

    # Print all rows sorted by (experiment, global_step)
    for key in sorted(rows.keys(), key=lambda x: (x[0], x[2])):
        r = rows[key]
        logger.info(
            "%-50s %6s  %8s  %8s  %5s  %6s",
            r["experiment"],
            r["global_step"],
            r["reward_mean"],
            r["reward_std"],
            r["n_seeds"],
            r["n_samples_per_seed"],
        )

    logger.info("=" * 95)

    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for key in sorted(rows.keys(), key=lambda x: (x[0], x[2])):
            writer.writerow(rows[key])

    logger.info("Summary written to %s (%d rows)", summary_path, len(rows))


def _load_results_from_jsonl(
    output_dir: str, experiment: str
) -> list[EvalResult]:
    """Load all EvalResult records from an experiment's JSONL file."""
    output_path = os.path.join(output_dir, f"{experiment}.jsonl")
    results: list[EvalResult] = []
    if not os.path.exists(output_path):
        return results
    with open(output_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                results.append(
                    EvalResult(
                        experiment=record["experiment"],
                        trial=record["trial"],
                        step_dir=record["step_dir"],
                        global_step=record["global_step"],
                        sample_idx=record["sample_idx"],
                        seed=record.get("seed", 0),
                        reward=record["reward"],
                        n_agents=record["n_agents"],
                        is_anonymous=record["is_anonymous"],
                        lora_mode=record["lora_mode"],
                        critic_mode=record["critic_mode"],
                        benchmark=record.get("benchmark", ""),
                        reward_mean=record.get("reward_mean", 0.0),
                        reward_std=record.get("reward_std", 0.0),
                        rewards=record.get("rewards"),
                    )
                )
            except (json.JSONDecodeError, KeyError):
                continue
    return results


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------


async def evaluate_task(
    task: EvalTask,
    dataset_items: list[dict],
    port: int,
    output_dir: str,
    temperature: float,
    max_concurrent: int,
    resume: bool,
    save_completions: bool,
    n_seeds: int,
    reward_executor: ProcessPoolExecutor | None = None,
) -> list[EvalResult]:
    """Evaluate a single checkpoint task against the dataset for *n_seeds* runs.

    For each seed, generates completions for all samples, computes rewards,
    and writes per-seed records.  After all seeds complete, writes an
    aggregate (mean/std) record per sample.
    """
    output_path = os.path.join(output_dir, f"{task.experiment_name}.jsonl")

    # Resume support — track (step_dir, sample_idx, seed)
    completed = load_completed(output_path) if resume else set()

    # Determine which (sample_idx, seed) pairs still need work
    pending_pairs: list[tuple[int, int]] = []  # (sample_idx, seed)
    for seed in range(n_seeds):
        for idx in range(len(dataset_items)):
            if (task.step_dir, idx, seed) not in completed:
                pending_pairs.append((idx, seed))

    if not pending_pairs:
        logger.info(
            "Step %d: all %d samples x %d seeds already done, skipping",
            task.global_step, len(dataset_items), n_seeds,
        )
        return []

    # Group pending work by seed for efficient batched inference
    pending_by_seed: dict[int, list[int]] = {}
    for idx, seed in pending_pairs:
        pending_by_seed.setdefault(seed, []).append(idx)

    logger.info(
        "Step %d: %d pending pairs across %d seeds (%d samples total)",
        task.global_step,
        len(pending_pairs),
        len(pending_by_seed),
        len(dataset_items),
    )

    # Load adapters once for all seeds (skip for base model eval)
    if task.adapter_paths:
        if not load_adapters(port, task.adapter_paths):
            logger.warning("Failed to load adapters for step %d, skipping", task.global_step)
            return []

    all_results: list[EvalResult] = []
    try:
        total_pending = sum(len(v) for v in pending_by_seed.values())
        sample_pbar = tqdm(
            total=total_pending,
            desc=f"  step {task.global_step}",
            unit="sample",
            dynamic_ncols=True,
            leave=False,
        )
        for seed in range(n_seeds):
            pending_indices = pending_by_seed.get(seed)
            if not pending_indices:
                continue

            sample_pbar.set_postfix_str(f"seed {seed} generating")

            pending_items = [dataset_items[i] for i in pending_indices]

            # Generate completions
            if task.is_multi_agent:
                completions = await eval_multi_agent(
                    port=port,
                    dataset_items=pending_items,
                    adapter_paths=task.adapter_paths,
                    role_names=task.role_names,
                    role_configs=task.role_configs,
                    max_new_tokens=task.max_new_tokens,
                    shared_lora=task.shared_lora,
                    temperature=temperature,
                    max_concurrent=max_concurrent,
                )
            else:
                lora_name = (
                    next(iter(task.adapter_paths.keys()))
                    if task.adapter_paths
                    else None
                )
                completions = await eval_single_agent(
                    port=port,
                    dataset_items=pending_items,
                    lora_name=lora_name,
                    max_new_tokens=task.max_new_tokens,
                    temperature=temperature,
                    max_concurrent=max_concurrent,
                )

            # Compute rewards in parallel using process pool
            sample_pbar.set_postfix_str(f"seed {seed} scoring")
            loop = asyncio.get_running_loop()
            reward_futs = [
                loop.run_in_executor(
                    reward_executor,
                    compute_reward,
                    completion,
                    dataset_items[sample_idx],
                    task.benchmark,
                )
                for sample_idx, completion in zip(pending_indices, completions)
            ]
            rewards = await asyncio.gather(*reward_futs)

            for sample_idx, completion, reward in zip(
                pending_indices, completions, rewards
            ):
                result = EvalResult(
                    experiment=task.experiment_name,
                    trial=task.trial_name,
                    step_dir=task.step_dir,
                    global_step=task.global_step,
                    sample_idx=sample_idx,
                    seed=seed,
                    reward=reward,
                    n_agents=task.n_agents,
                    is_anonymous=task.is_anonymous,
                    lora_mode=task.lora_mode,
                    critic_mode=task.critic_mode,
                    benchmark=task.benchmark,
                    completion=completion if save_completions else "",
                )
                all_results.append(result)
                write_result(output_path, result)
                sample_pbar.update(1)

            seed_rewards = [r.reward for r in all_results if r.seed == seed]
            avg = sum(seed_rewards) / len(seed_rewards) if seed_rewards else 0.0
            n_pass = sum(1 for r in seed_rewards if r > 0)
            logger.info(
                "  Seed %d: avg_reward=%.4f, pass_rate=%d/%d",
                seed, avg, n_pass, len(seed_rewards),
            )
        sample_pbar.close()

        # ------------------------------------------------------------------
        # Write aggregate (mean/std) records per sample
        # ------------------------------------------------------------------
        # Merge newly computed results with previously completed results
        # loaded from the JSONL so that aggregates are correct on resume.
        existing_seed_results: dict[tuple[int, int], float] = {}
        if resume and os.path.exists(output_path):
            with open(output_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    if (
                        rec.get("step_dir") == task.step_dir
                        and rec.get("seed", -1) >= 0
                    ):
                        existing_seed_results[
                            (rec["sample_idx"], rec["seed"])
                        ] = rec["reward"]

        # Also include what we just wrote
        for r in all_results:
            if r.seed >= 0:
                existing_seed_results[(r.sample_idx, r.seed)] = r.reward

        # Check which samples have all seeds complete and no aggregate yet
        agg_done = {
            idx
            for (sd, idx, s) in completed
            if sd == task.step_dir and s == -1
        }

        agg_results: list[EvalResult] = []
        for sample_idx in range(len(dataset_items)):
            if sample_idx in agg_done:
                continue
            seed_rewards = [
                existing_seed_results[(sample_idx, s)]
                for s in range(n_seeds)
                if (sample_idx, s) in existing_seed_results
            ]
            if len(seed_rewards) < n_seeds:
                continue  # not all seeds done yet

            mean_r = sum(seed_rewards) / len(seed_rewards)
            std_r = math.sqrt(
                sum((r - mean_r) ** 2 for r in seed_rewards) / len(seed_rewards)
            )
            agg = EvalResult(
                experiment=task.experiment_name,
                trial=task.trial_name,
                step_dir=task.step_dir,
                global_step=task.global_step,
                sample_idx=sample_idx,
                seed=-1,
                reward=mean_r,
                n_agents=task.n_agents,
                is_anonymous=task.is_anonymous,
                lora_mode=task.lora_mode,
                critic_mode=task.critic_mode,
                benchmark=task.benchmark,
                reward_mean=mean_r,
                reward_std=std_r,
                rewards=seed_rewards,
            )
            agg_results.append(agg)
            write_result(output_path, agg)

        if agg_results:
            overall_mean = sum(a.reward_mean for a in agg_results) / len(agg_results)
            overall_std = sum(a.reward_std for a in agg_results) / len(agg_results)
            logger.info(
                "Step %d aggregate: mean_reward=%.4f, mean_std=%.4f (%d samples)",
                task.global_step, overall_mean, overall_std, len(agg_results),
            )
        all_results.extend(agg_results)
        return all_results

    finally:
        if task.adapter_paths:
            unload_adapters(port, list(task.adapter_paths.keys()))


def _build_base_tasks_from_yamls(
    yaml_paths: list[str], benchmark: str
) -> list[EvalTask]:
    """Create base-model EvalTasks from YAML config files."""
    tasks: list[EvalTask] = []
    for yaml_path in yaml_paths:
        config = _load_config(Path(yaml_path))
        if config is None:
            logger.warning("Cannot load config %s, skipping", yaml_path)
            continue

        exp_name = config.get("experiment_name", Path(yaml_path).stem)
        trial_name = config.get("trial_name", "trial1")
        ma_config = config.get("multi_agent", {})
        role_names = ma_config.get("role_names", [])
        role_configs = ma_config.get("role_configs", {})
        shared_lora = ma_config.get("shared_lora", True)
        is_multi_agent = len(role_names) > 1
        n_agents = len(role_names) if role_names else 1

        gconfig = config.get("gconfig", {})
        max_new_tokens = gconfig.get("max_new_tokens", 27648 // max(n_agents, 1))

        exp_info = _parse_experiment_name(exp_name)

        tasks.append(
            EvalTask(
                experiment_name=exp_name,
                trial_name=trial_name,
                step_dir="base",
                global_step=0,
                checkpoint_path="",
                config=config,
                is_multi_agent=is_multi_agent,
                adapter_paths={},
                role_names=role_names,
                role_configs=role_configs,
                shared_lora=shared_lora,
                benchmark=benchmark,
                n_agents=n_agents,
                lora_mode="none",
                critic_mode=exp_info["critic_mode"],
                is_anonymous=exp_info["is_anonymous"],
                max_new_tokens=max_new_tokens,
            )
        )
        logger.info(
            "Base task from %s: %s (%d agents)",
            yaml_path, exp_name, n_agents,
        )
    return tasks


async def run_evaluation(args: argparse.Namespace) -> None:
    """Main evaluation orchestrator."""

    # --eval-base-only: build tasks from YAML configs, skip checkpoint discovery
    if args.eval_base_only:
        tasks = _build_base_tasks_from_yamls(args.eval_base_only, args.benchmark)
        if not tasks:
            logger.info("No valid configs provided to --eval-base-only.")
            return
    else:
        # Normal mode: discover checkpoints
        if not args.checkpoint_root or not args.config_root:
            logger.error(
                "--checkpoint-root and --config-root are required "
                "(unless using --eval-base-only)"
            )
            return
        tasks = discover_checkpoints(
            args.checkpoint_root,
            args.config_root,
            args.benchmark,
            args.filter,
        )

        if not tasks:
            logger.info("No checkpoints found matching criteria.")
            return

        # --eval-base: inject a synthetic "base" task (global_step=0, no LoRA)
        # for each unique experiment, using its multi-agent config.
        if args.eval_base:
            seen_experiments: set[str] = set()
            base_tasks: list[EvalTask] = []
            for t in tasks:
                key = (t.experiment_name, t.trial_name)
                if key in seen_experiments:
                    continue
                seen_experiments.add(key)
                base_tasks.append(
                    EvalTask(
                        experiment_name=t.experiment_name,
                        trial_name=t.trial_name,
                        step_dir="base",
                        global_step=0,
                        checkpoint_path="",
                        config=t.config,
                        is_multi_agent=t.is_multi_agent,
                        adapter_paths={},  # empty = no LoRA
                        role_names=t.role_names,
                        role_configs=t.role_configs,
                        shared_lora=t.shared_lora,
                        benchmark=t.benchmark,
                        n_agents=t.n_agents,
                        lora_mode="none",
                        critic_mode=t.critic_mode,
                        is_anonymous=t.is_anonymous,
                        max_new_tokens=t.max_new_tokens,
                    )
                )
            tasks = base_tasks + tasks
            logger.info("--eval-base: added %d base-model tasks", len(base_tasks))

    logger.info(
        "Discovered %d checkpoint steps across experiments (x%d seeds):",
        len(tasks), args.n_seeds,
    )
    experiments = sorted(set(t.experiment_name for t in tasks))
    for exp in experiments:
        exp_tasks = [t for t in tasks if t.experiment_name == exp]
        steps = [t.global_step for t in exp_tasks]
        logger.info("  %s: steps %s", exp, steps)

    if args.dry_run:
        logger.info("--dry-run: exiting without evaluation.")
        return

    # Load dataset
    dataset_items = load_dataset(args.dataset_path, args.benchmark, args.max_samples)

    # Start SGLang server if requested
    sglang_proc = None
    if args.start_server:
        try:
            sglang_proc, sglang_log = start_sglang_server(
                args.base_model,
                args.sglang_port,
                tp_size=args.tp_size,
                max_lora_rank=args.max_lora_rank,
                mem_fraction=args.mem_fraction,
                log_dir=args.output_dir,
            )
        except RuntimeError as e:
            logger.error("%s", e)
            return
        if not wait_for_server(args.sglang_port, timeout=args.server_timeout):
            logger.warning(
                "Failed to start SGLang server. Check log: %s", sglang_log
            )
            sglang_proc.terminate()
            return
    else:
        # No --start-server: verify an existing server is reachable
        if not _port_is_alive(args.sglang_port):
            logger.error(
                "No SGLang server found on port %d. "
                "Start one first, or use --start-server.",
                args.sglang_port,
            )
            return

    # Create process pool for parallel reward computation
    n_workers = args.reward_workers
    cpu_count = os.cpu_count() or 4
    if n_workers <= 0:
        n_workers = min(cpu_count, 32)
    elif n_workers > cpu_count:
        logger.warning(
            "--reward-workers %d exceeds CPU count %d; "
            "capping to %d (code execution is CPU-bound)",
            n_workers, cpu_count, cpu_count,
        )
        n_workers = cpu_count
    reward_executor = ProcessPoolExecutor(max_workers=n_workers)
    logger.info("Reward process pool: %d workers", n_workers)

    try:
        # Group tasks by experiment for ordered evaluation
        all_results: list[EvalResult] = []

        # Flatten all tasks for the progress bar
        ordered_tasks: list[EvalTask] = []
        for exp_name in experiments:
            exp_tasks = sorted(
                [t for t in tasks if t.experiment_name == exp_name],
                key=lambda t: t.global_step,
            )
            logger.info("Experiment: %s (%d steps)", exp_name, len(exp_tasks))
            ordered_tasks.extend(exp_tasks)

        task_pbar = tqdm(
            ordered_tasks,
            desc="Steps",
            unit="step",
            dynamic_ncols=True,
        )
        for task in task_pbar:
            task_pbar.set_postfix_str(
                f"{task.experiment_name} step={task.global_step}"
            )
            results = await evaluate_task(
                task=task,
                dataset_items=dataset_items,
                port=args.sglang_port,
                output_dir=args.output_dir,
                temperature=args.temperature,
                max_concurrent=args.max_concurrent,
                resume=args.resume,
                save_completions=args.save_completions,
                n_seeds=args.n_seeds,
                reward_executor=reward_executor,
            )
            all_results.extend(results)

            # Update summary.csv after each step completes
            if results:
                step_summary = _load_results_from_jsonl(
                    args.output_dir, task.experiment_name,
                )
                write_summary(args.output_dir, step_summary, args.n_seeds)
        task_pbar.close()

        # Final summary reload across all experiments
        all_results_for_summary: list[EvalResult] = []
        for exp in experiments:
            all_results_for_summary.extend(
                _load_results_from_jsonl(args.output_dir, exp)
            )
        write_summary(args.output_dir, all_results_for_summary, args.n_seeds)

    finally:
        reward_executor.shutdown(wait=False)
        if sglang_proc is not None:
            logger.info("Shutting down SGLang server...")
            sglang_proc.terminate()
            try:
                sglang_proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                sglang_proc.kill()
            # Close the log file handle
            log_fh = getattr(sglang_proc, "_log_fh", None)
            if log_fh is not None:
                log_fh.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate MARFT LoRA checkpoints via SGLang server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint-root", default="",
        help="Root directory containing experiment checkpoint dirs "
        "(not required with --eval-base-only)",
    )
    parser.add_argument(
        "--config-root", default="",
        help="Root directory containing experiment log/config dirs "
        "(not required with --eval-base-only)",
    )
    parser.add_argument(
        "--dataset-path", required=True,
        help="Path to dataset directory (must contain test.jsonl)",
    )
    parser.add_argument(
        "--benchmark", required=True,
        choices=[
            "deepcoder", "livecodebench", "codeforces",
            "gsm8k", "math", "deepscaler",
            "aime2024", "math500", "minervamath", "olympiadbench",
        ],
        help="Benchmark type (determines reward function and dataset loading)",
    )
    parser.add_argument(
        "--output-dir", required=True,
        help="Directory for evaluation results (JSONL + summary CSV)",
    )
    parser.add_argument(
        "--base-model", default="",
        help="Path to base model (for starting SGLang server)",
    )
    parser.add_argument(
        "--sglang-port", type=int, default=30000,
        help="SGLang server port (default: 30000)",
    )
    parser.add_argument(
        "--start-server", action="store_true",
        help="Start SGLang server automatically (otherwise assume already running)",
    )
    parser.add_argument(
        "--tp-size", type=int, default=1,
        help="Tensor parallelism size for SGLang (default: 1)",
    )
    parser.add_argument(
        "--max-lora-rank", type=int, default=32,
        help="Maximum LoRA rank for SGLang server (default: 32)",
    )
    parser.add_argument(
        "--mem-fraction", type=float, default=0.85,
        help="GPU memory fraction for SGLang (default: 0.85)",
    )
    parser.add_argument(
        "--server-timeout", type=int, default=300,
        help="Timeout in seconds for SGLang server startup (default: 300)",
    )
    parser.add_argument(
        "--filter", default=None,
        help="Filter experiments by substring match (e.g. '2agent-shared')",
    )
    parser.add_argument(
        "--max-samples", type=int, default=None,
        help="Maximum number of test samples to evaluate per checkpoint",
    )
    parser.add_argument(
        "--max-concurrent", type=int, default=64,
        help="Maximum concurrent requests to SGLang (default: 64)",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.6,
        help="Sampling temperature (default: 0.6, matching eval_gconfig)",
    )
    parser.add_argument(
        "--n-seeds", type=int, default=5,
        help="Number of independent evaluation seeds per checkpoint (default: 5)",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from existing results (skip already-evaluated samples)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Discover checkpoints and print plan without evaluating",
    )
    parser.add_argument(
        "--save-completions", action="store_true",
        help="Save full completion text in results (increases file size)",
    )
    parser.add_argument(
        "--reward-workers", type=int, default=0,
        help="Number of parallel processes for reward computation. "
        "0 = auto (number of CPUs, capped at 32). "
        "Deepcoder code-execution rewards benefit greatly from parallelism.",
    )
    parser.add_argument(
        "--eval-base", action="store_true",
        help="Also evaluate the base model (no LoRA) under each experiment's "
        "multi-agent config as a pre-training baseline.",
    )
    parser.add_argument(
        "--eval-base-only", nargs="+", metavar="YAML",
        help="Evaluate the base model only (no checkpoints). "
        "Pass one or more YAML config files that define multi-agent setups. "
        "Skips checkpoint discovery entirely. "
        "Example: --eval-base-only examples/marft/deepcoder_2agent.yaml "
        "examples/marft/deepcoder_4agent.yaml",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.start_server and not args.base_model:
        logger.error("--base-model is required when using --start-server")
        sys.exit(1)

    asyncio.run(run_evaluation(args))


if __name__ == "__main__":
    main()
