"""Code execution reward function for RL training on coding tasks.

Extracts Python code from model completions, runs it against test cases
(stdin/stdout or function-call style), and returns the pass rate as the
reward signal.

Execution is done via ``subprocess.run`` with strict timeouts, memory
limits, and no network access.  This is NOT a full sandbox — it relies
on cgroup / container isolation provided by the cluster runtime.

The reward function signature matches ``areal.api.reward_api.reward_fn``
so it can be used directly with ``RLVRWorkflow`` and ``AsyncRewardWrapper``.
"""

import json
import math
import re
import subprocess
import textwrap
import time

from areal.utils import logging

logger = logging.getLogger("CodeReward")

# Execution limits
_TIMEOUT_SECONDS = 5  # per-test timeout (seconds)
_MAX_OUTPUT_BYTES = 1024 * 1024  # 1 MB
_WALL_CLOCK_BUDGET = 60  # overall budget for all tests (seconds)
_MAX_TESTS_AT_EVAL = 10  # max test cases to run during training


# ---------------------------------------------------------------------------
# Code extraction
# ---------------------------------------------------------------------------

_PYTHON_BLOCK_RE = re.compile(r"```python\s*\n?(.*?)```", re.DOTALL)
_ANY_BLOCK_RE = re.compile(r"```\w*\s*\n?(.*?)```", re.DOTALL)


def _extract_code(completion: str) -> str:
    """Extract Python code from a model completion.

    Looks for fenced code blocks (```python ... ```) first, then falls
    back to the last fenced block of any language, then the entire
    completion as a last resort.
    """
    python_blocks = _PYTHON_BLOCK_RE.findall(completion)
    if python_blocks:
        return python_blocks[-1].strip()

    any_blocks = _ANY_BLOCK_RE.findall(completion)
    if any_blocks:
        return any_blocks[-1].strip()

    return completion.strip()


# ---------------------------------------------------------------------------
# Code execution
# ---------------------------------------------------------------------------


def _run_code_stdin(code: str, stdin_input: str) -> tuple[bool, str]:
    """Run *code* as a script, feeding *stdin_input* via stdin.

    Returns ``(success, stdout_or_error)``.
    """
    try:
        result = subprocess.run(
            ["python3", "-c", code],
            input=stdin_input,
            capture_output=True,
            text=True,
            timeout=_TIMEOUT_SECONDS,
        )
        if result.returncode != 0:
            return False, result.stderr[:_MAX_OUTPUT_BYTES]
        return True, result.stdout[:_MAX_OUTPUT_BYTES]
    except subprocess.TimeoutExpired:
        return False, "TIMEOUT"
    except Exception as e:
        return False, str(e)[:_MAX_OUTPUT_BYTES]


def _run_code_functional(
    code: str,
    func_name: str,
    args_json: str,
) -> tuple[bool, str]:
    """Run *code* then call ``func_name(*args)`` and return the result.

    *args_json* is a JSON-encoded list of arguments.
    Returns ``(success, json_encoded_result_or_error)``.
    """
    wrapper = textwrap.dedent(f"""\
        import json, sys
        # --- user code ---
        {textwrap.indent(code, "        ").strip()}
        # --- end user code ---
        _args = json.loads(sys.argv[1])
        if not isinstance(_args, list):
            _args = [_args]
        _result = {func_name}(*_args)
        print(json.dumps(_result))
    """)
    try:
        result = subprocess.run(
            ["python3", "-c", wrapper, args_json],
            capture_output=True,
            text=True,
            timeout=_TIMEOUT_SECONDS,
        )
        if result.returncode != 0:
            return False, result.stderr[:_MAX_OUTPUT_BYTES]
        return True, result.stdout.strip()[:_MAX_OUTPUT_BYTES]
    except subprocess.TimeoutExpired:
        return False, "TIMEOUT"
    except Exception as e:
        return False, str(e)[:_MAX_OUTPUT_BYTES]


# ---------------------------------------------------------------------------
# Test-case evaluation
# ---------------------------------------------------------------------------

_FLOAT_RE = re.compile(r"^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$")


def _normalize_lines(text: str) -> list[str]:
    """Strip each line and drop trailing empty lines."""
    lines = [line.rstrip() for line in text.split("\n")]
    while lines and not lines[-1]:
        lines.pop()
    return lines


def _compare_output(expected: str, actual: str) -> bool:
    """Compare expected and actual output with lenient matching.

    Tolerant of:
    - Trailing whitespace per line and trailing blank lines.
    - Multiple spaces collapsed to one within a line.
    - Floating-point values that match within 1e-6 relative tolerance.
    """
    exp_lines = _normalize_lines(expected)
    act_lines = _normalize_lines(actual)

    if len(exp_lines) != len(act_lines):
        return False

    for exp_line, act_line in zip(exp_lines, act_lines):
        if exp_line == act_line:
            continue

        exp_tokens = exp_line.split()
        act_tokens = act_line.split()
        if len(exp_tokens) != len(act_tokens):
            return False

        for et, at in zip(exp_tokens, act_tokens):
            if et == at:
                continue
            if _FLOAT_RE.match(et) and _FLOAT_RE.match(at):
                try:
                    if math.isclose(float(et), float(at), rel_tol=1e-6, abs_tol=1e-9):
                        continue
                except ValueError:
                    pass
            return False

    return True


def _compare_functional(expected_json: str, actual_json: str) -> bool:
    """Compare function return values via JSON equality.

    Handles the taco convention where outputs are wrapped in a list,
    e.g. expected ``[true]`` should match actual ``true``.
    Also tolerant of floating-point differences within 1e-6.
    """
    try:
        expected = json.loads(expected_json)
        actual = json.loads(actual_json)
        if _json_close(expected, actual):
            return True
        if isinstance(expected, list) and len(expected) == 1:
            if _json_close(expected[0], actual):
                return True
        if isinstance(actual, list) and len(actual) == 1:
            if _json_close(actual[0], expected):
                return True
        return False
    except (json.JSONDecodeError, TypeError):
        return expected_json.strip() == actual_json.strip()


def _json_close(a, b) -> bool:
    """Recursive equality check with float tolerance."""
    if isinstance(a, float) and isinstance(b, (int, float)):
        return math.isclose(a, float(b), rel_tol=1e-6, abs_tol=1e-9)
    if isinstance(b, float) and isinstance(a, (int, float)):
        return math.isclose(float(a), b, rel_tol=1e-6, abs_tol=1e-9)
    if type(a) is not type(b):
        return a == b
    if isinstance(a, list):
        return len(a) == len(b) and all(_json_close(x, y) for x, y in zip(a, b))
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_json_close(a[k], b[k]) for k in a)
    return a == b


def _evaluate_tests(
    code: str,
    tests_json: str,
    test_type: str,
    func_name: str = "",
    starter_code: str = "",
    max_tests: int = _MAX_TESTS_AT_EVAL,
    wall_clock_budget: float = _WALL_CLOCK_BUDGET,
) -> float:
    """Run *code* against test cases and return the pass rate.

    For ``starter_code`` problems, the starter code is prepended to the
    user's code (the model is expected to complete/redefine the function).

    Stops early if the overall *wall_clock_budget* is exceeded or after
    *max_tests* test cases, whichever comes first.  The denominator is the
    number of tests actually evaluated, not the total, so early stopping
    does not deflate the pass rate.
    """
    test_cases = json.loads(tests_json)
    if not test_cases:
        return 0.0

    if max_tests and len(test_cases) > max_tests:
        test_cases = test_cases[:max_tests]

    if test_type == "functional" and starter_code:
        full_code = starter_code + "\n" + code
    else:
        full_code = code

    passed = 0
    evaluated = 0
    start_time = time.monotonic()
    for tc in test_cases:
        if time.monotonic() - start_time > wall_clock_budget:
            logger.debug(
                f"Wall-clock budget ({wall_clock_budget}s) exceeded after "
                f"{evaluated}/{len(test_cases)} tests"
            )
            break

        inp = tc["input"]
        expected = tc["output"]
        evaluated += 1

        if test_type == "functional" and func_name:
            ok, actual = _run_code_functional(full_code, func_name, inp)
            if ok and _compare_functional(expected, actual):
                passed += 1
        else:
            ok, actual = _run_code_stdin(full_code, inp)
            if ok and _compare_output(expected, actual):
                passed += 1

    if evaluated == 0:
        return 0.0
    return passed / evaluated


# ---------------------------------------------------------------------------
# Reward function (matches AReaL reward_fn signature)
# ---------------------------------------------------------------------------


def deepcoder_reward_fn(
    prompt,
    completions,
    prompt_ids,
    completion_ids,
    tests="[]",
    test_type="stdin",
    func_name="",
    starter_code="",
    **kwargs,
) -> float:
    """Compute reward for a code-generation completion.

    Extracts Python code from *completions*, runs it against the test
    cases in *tests*, and returns the pass rate (0.0–1.0).

    This function is designed to be wrapped with ``AsyncRewardWrapper``
    for timeout/retry handling in the rollout workflow.

    Args:
        prompt: The prompt string (unused, kept for API compat).
        completions: The model's generated text.
        prompt_ids: Token IDs of the prompt (unused).
        completion_ids: Token IDs of the completion (unused).
        tests: JSON-encoded list of ``{"input", "output"}`` test cases.
        test_type: ``"stdin"`` or ``"functional"``.
        func_name: Function name for functional problems.
        starter_code: Starter code for functional problems.
        **kwargs: Absorbs extra dataset fields.

    Returns:
        float: Fraction of test cases passed (0.0–1.0).
    """
    try:
        code = _extract_code(str(completions))
        if not code:
            return 0.0

        return _evaluate_tests(
            code,
            tests,
            test_type,
            func_name=func_name,
            starter_code=starter_code,
        )
    except Exception:
        logger.warning("Exception in deepcoder_reward_fn", exc_info=True)
        return 0.0
