# Copyright 2025 Junwei Liao, Shanghai Jiao Tong University and Shanghai Innovation Institute.
# Licensed under the Apache License, Version 2.0.

"""Multi-agent reward functions for code generation workflows.

Provides:
- ``multi_agent_code_reward_fn``: Team reward based on code execution pass rate.
  Extracts the last ```python``` block from the full conversation (typically the
  last code-producing agent's output) and runs it against test cases.
- Per-agent step reward functions for ``credit_strategy=per_step``.
"""

import re

from areal.reward.code_execution import _evaluate_tests, _extract_code
from areal.utils import logging

logger = logging.getLogger("MultiAgentCodeReward")

_CODE_BLOCK_RE = re.compile(r"```python\s*\n(.*?)```", re.DOTALL)
_NUMBERED_STEP_RE = re.compile(r"(?:^|\n)\s*(?:\d+[\.\):]|[-*])\s+\S")


def multi_agent_code_reward_fn(
    prompt: str,
    completions: str,
    prompt_ids: list[int],
    completion_ids: list[int],
    tests: str = "[]",
    test_type: str = "stdin",
    func_name: str = "",
    starter_code: str = "",
    **data,
) -> float:
    """Team reward for multi-agent code generation workflows.

    Extracts the last Python code block from the combined output of all
    agents and evaluates it against the test cases.  Returns the pass
    rate (0.0–1.0).
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
        logger.warning("Exception in multi_agent_code_reward_fn", exc_info=True)
        return 0.0


# ------------------------------------------------------------------
# Per-agent step reward functions for ``credit_strategy=per_step``
# ------------------------------------------------------------------


def planner_step_reward(step, team_reward: float, env=None, data=None) -> float:
    """Reward the planner for producing a structured algorithm plan.

    Returns a value in [0, 1]:
      - 0.5 * structure_score  (plan has >= 3 numbered/bulleted steps)
      - 0.5 * team_reward      (outcome-aligned component)
    """
    text = step.output_text
    n_steps = len(_NUMBERED_STEP_RE.findall(text))
    structure_score = min(n_steps / 3.0, 1.0)
    return 0.5 * structure_score + 0.5 * team_reward


def coder_step_reward(step, team_reward: float, env=None, data=None) -> float:
    """Reward the coder for producing a Python code block.

    Returns a value in [0, 1]:
      - team_reward  if a ```python``` block is present
      - 0.0          if no code block at all
    """
    text = step.output_text
    if _CODE_BLOCK_RE.search(text):
        return team_reward if team_reward > 0 else 0.1
    return 0.0


def debugger_step_reward(step, team_reward: float, env=None, data=None) -> float:
    """Reward the debugger based on the final team outcome.

    The debugger's job is to catch bugs or confirm correctness.
    Its reward is directly tied to whether the code passes tests.
    """
    return team_reward


# Alias: in the 2-agent (planner → solver) pipeline the solver is
# the code-writing agent, so its reward logic mirrors ``coder_step_reward``.
solver_step_reward = coder_step_reward
