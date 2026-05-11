# Copyright 2025 Junwei Liao, Shanghai Jiao Tong University and Shanghai Innovation Institute.
# Licensed under the Apache License, Version 2.0.

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

from areal.workflow.multi_agent.agent_role import AgentRole


@dataclass
class SharedEnvironment:
    """Shared message history visible to all agents in a multi-agent workflow.

    The environment tracks the conversation as a flat list of chat messages
    (``{"role": ..., "content": ...}``) and provides role-aware views that
    prepend each agent's system prompt.
    """

    messages: list[dict[str, str]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Message mutation
    # ------------------------------------------------------------------

    def add_user_message(self, content: str) -> None:
        self.messages.append({"role": "user", "content": content})

    def add_assistant_message(
        self, content: str, agent_name: str | None = None
    ) -> None:
        msg: dict[str, str] = {"role": "assistant", "content": content}
        if agent_name is not None:
            msg["agent_name"] = agent_name
        self.messages.append(msg)

    # ------------------------------------------------------------------
    # Read-only views
    # ------------------------------------------------------------------

    def get_messages_for_role(self, role: AgentRole) -> list[dict[str, str]]:
        """Return the full message list prefixed with the role's system prompt."""
        system_msg = {"role": "system", "content": role.system_prompt}
        return [system_msg] + list(self.messages)

    # ------------------------------------------------------------------
    # Parallel support
    # ------------------------------------------------------------------

    def snapshot(self) -> SharedEnvironment:
        """Deep-copy the environment for a parallel branch.

        Stores the base message length at snapshot time so that
        ``merge_parallel_results`` can reliably extract new messages.
        """
        snap = SharedEnvironment(
            messages=copy.deepcopy(self.messages),
            metadata=copy.deepcopy(self.metadata),
        )
        snap._snapshot_base_len = len(self.messages)
        return snap

    _snapshot_base_len: int = -1

    def merge_parallel_results(
        self,
        branches: list[SharedEnvironment],
    ) -> None:
        """Merge messages added by parallel branches back into this environment.

        Each branch should be a snapshot taken *before* the parallel step.
        Only messages appended after the snapshot point are collected, in
        branch order.
        """
        base_len = len(self.messages)
        for branch in branches:
            branch_base = (
                branch._snapshot_base_len
                if branch._snapshot_base_len >= 0
                else base_len
            )
            new_messages = branch.messages[branch_base:]
            self.messages.extend(new_messages)

    @classmethod
    def from_data(cls, data: dict[str, Any]) -> SharedEnvironment:
        """Initialise from a dataset sample (expects ``data["messages"]``)."""
        messages = list(data.get("messages", []))
        return cls(messages=messages, metadata=dict(data))
