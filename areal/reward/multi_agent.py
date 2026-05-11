# Copyright 2025 Junwei Liao, Shanghai Jiao Tong University and Shanghai Innovation Institute.
# Licensed under the Apache License, Version 2.0.

import re

from areal.reward import get_math_verify_worker
from areal.utils import logging

logger = logging.getLogger("MultiAgentReward")


def multi_agent_math_reward_fn(
    prompt: str,
    completions: str,
    prompt_ids: list[int],
    completion_ids: list[int],
    **data,
) -> float:
    """Reward function for multi-agent math workflows.

    Extracts the final boxed answer from the last agent's output and
    verifies it against the ground truth using ``MathVerifyWorker``.

    Compatible with standard AReaL reward API:
    ``(prompt, completions, prompt_ids, completion_ids, **data) → float``
    """
    answer = data.get("answer", "")
    if not answer:
        logger.warning("No ground-truth answer found in data.")
        return 0.0

    try:
        worker = get_math_verify_worker()
        return worker.verify(completions, str(answer))
    except BaseException:
        logger.warning(
            "Exception in multi_agent_math_reward_fn",
            exc_info=True,
        )
        return 0.0


# ------------------------------------------------------------------
# Per-agent step reward functions for ``credit_strategy=per_step``
#
# Signature: (step, team_reward, env, data) -> float
#   - step: AgentStep with agent_name, step_index, token_start,
#           token_end, output_text
#   - team_reward: scalar team reward from the main reward_fn
#   - env: SharedEnvironment (or None)
#   - data: original dataset sample dict (contains "answer", etc.)
# ------------------------------------------------------------------

_BOXED_RE = re.compile(r"\\boxed\{([^}]*)\}")
_NUMBERED_STEP_RE = re.compile(r"(?:^|\n)\s*(?:\d+[\.\):]|[-*])\s+\S")


def planner_step_reward(step, team_reward: float, env=None, data=None) -> float:
    """Reward the planner for producing a structured plan.

    Gives partial credit for structural quality (numbered / bulleted
    steps) and a bonus that scales with the team outcome.

    Returns a value in [0, 1]:
      - 0.5 * structure_score  (plan has >= 2 numbered/bulleted steps)
      - 0.5 * team_reward      (outcome-aligned component)
    """
    text = step.output_text
    n_steps = len(_NUMBERED_STEP_RE.findall(text))
    structure_score = min(n_steps / 2.0, 1.0)
    return 0.5 * structure_score + 0.5 * team_reward


def solver_step_reward(step, team_reward: float, env=None, data=None) -> float:
    """Reward the solver for producing a correct boxed answer.

    Uses ``team_reward`` (already computed by the main reward function in
    a thread-safe context) to judge correctness, and checks for the
    presence of a ``\\boxed{}`` answer as a format signal.

    Returns a value in [0, 1]:
      - team_reward  if a \\boxed{} answer is present (correct → 1.0)
      - 0.1          if \\boxed{} present but team got it wrong
      - 0.0          if no \\boxed{} at all
    """
    text = step.output_text
    matches = _BOXED_RE.findall(text)
    if not matches:
        return 0.0
    return team_reward if team_reward > 0 else 0.1


def verifier_step_reward(step, team_reward: float, env=None, data=None) -> float:
    """Reward the verifier based on the final team outcome.

    The verifier's job is to catch mistakes or confirm correctness.
    Its reward is directly tied to whether the final answer is right.

    Returns:
      - team_reward  (1.0 if final answer correct, 0.0 otherwise)
    """
    return team_reward
