# Copyright 2025 Junwei Liao, Shanghai Jiao Tong University and Shanghai Innovation Institute.
# Licensed under the Apache License, Version 2.0.

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch

from areal.workflow.multi_agent.environment import SharedEnvironment


@dataclass
class AgentStep:
    """Bookkeeping for one agent's contribution within a trajectory."""

    agent_name: str
    step_index: int
    token_start: int
    token_end: int
    output_text: str = ""


class CreditAssignment:
    """Distributes a team-level reward across agent steps.

    Strategies:
        ``equal``
            Full team reward placed only at the last token of the sequence
            (the critic + GAE handle temporal credit assignment).
        ``step_discount``
            ``team_reward * discount ** (N - 1 - i)`` placed at the last
            token of each agent step *i*, where *N* is the total number
            of steps.  Earlier agents receive a discounted share.
        ``per_step``
            Per-agent reward functions are called for each agent step to
            produce independent reward signals.  Each role can define its
            own ``reward_fn``; a global ``step_reward_fn`` serves as the
            fallback for roles that do not.
    """

    STRATEGIES = ("equal", "step_discount", "per_step")

    def __init__(
        self,
        strategy: str = "equal",
        discount: float = 1.0,
        step_reward_fn: Callable[..., float] | None = None,
        per_agent_reward_fns: dict[str, Callable[..., float]] | None = None,
    ):
        if strategy not in self.STRATEGIES:
            raise ValueError(
                f"Unknown credit strategy '{strategy}'. Choose from {self.STRATEGIES}."
            )
        if strategy == "per_step":
            has_any_fn = step_reward_fn is not None or (
                per_agent_reward_fns and len(per_agent_reward_fns) > 0
            )
            if not has_any_fn:
                raise ValueError(
                    "per_step credit strategy requires at least a global "
                    "step_reward_fn or per-agent reward_fns in role_configs."
                )
        if not (0.0 <= discount <= 1.0):
            raise ValueError(f"credit_discount must be in [0, 1], got {discount}.")
        self.strategy = strategy
        self.discount = discount
        self.step_reward_fn = step_reward_fn
        self.per_agent_reward_fns = per_agent_reward_fns or {}

    def compute_per_token_rewards(
        self,
        team_reward: float,
        agent_steps: list[AgentStep],
        seq_len: int,
        env: SharedEnvironment | None = None,
        data: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        """Return a ``[seq_len]`` float tensor with per-token rewards.

        Only the *last generated token* of each relevant agent step
        receives a non-zero reward; all other positions are zero.
        """
        rewards = torch.zeros(seq_len, dtype=torch.float32)
        if not agent_steps:
            return rewards

        if self.strategy == "equal":
            last_step = agent_steps[-1]
            idx = last_step.token_end - 1
            if 0 <= idx < seq_len:
                rewards[idx] = team_reward

        elif self.strategy == "step_discount":
            n_steps = len(agent_steps)
            for step in agent_steps:
                discounted = team_reward * (
                    self.discount ** (n_steps - 1 - step.step_index)
                )
                idx = step.token_end - 1
                if 0 <= idx < seq_len:
                    rewards[idx] = discounted

        elif self.strategy == "per_step":
            for step in agent_steps:
                fn = self.per_agent_reward_fns.get(step.agent_name, self.step_reward_fn)
                if fn is None:
                    raise ValueError(
                        f"No reward_fn for agent '{step.agent_name}' and no global "
                        f"step_reward_fn configured. Provide a reward_fn in the "
                        f"role config or set step_reward_fn as a fallback."
                    )
                step_reward = fn(
                    step=step,
                    team_reward=team_reward,
                    env=env,
                    data=data,
                )
                idx = step.token_end - 1
                if 0 <= idx < seq_len:
                    rewards[idx] = float(step_reward)

        return rewards

    def compute_per_agent_rewards(
        self,
        team_reward: float,
        agent_steps: list[AgentStep],
        seq_len: int,
        env: SharedEnvironment | None = None,
        data: dict[str, Any] | None = None,
    ) -> dict[str, float]:
        """Return a mapping of agent_name → reward for metrics logging."""
        per_agent: dict[str, float] = {}
        if not agent_steps:
            return per_agent

        if self.strategy == "equal":
            for step in agent_steps:
                per_agent[step.agent_name] = team_reward

        elif self.strategy == "step_discount":
            n_steps = len(agent_steps)
            for step in agent_steps:
                discounted = team_reward * (
                    self.discount ** (n_steps - 1 - step.step_index)
                )
                per_agent[step.agent_name] = discounted

        elif self.strategy == "per_step":
            for step in agent_steps:
                fn = self.per_agent_reward_fns.get(step.agent_name, self.step_reward_fn)
                if fn is None:
                    raise ValueError(
                        f"No reward_fn for agent '{step.agent_name}' and no global "
                        f"step_reward_fn configured."
                    )
                step_reward = fn(
                    step=step,
                    team_reward=team_reward,
                    env=env,
                    data=data,
                )
                per_agent[step.agent_name] = float(step_reward)

        return per_agent
