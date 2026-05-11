"""Orchestrator output parsing and prompt construction.

The orchestrator is an LLM agent that decides which specialist to invoke
at each step.  It communicates routing decisions via structured tags:

- ``<call>agent_name</call>`` — delegate to a specialist.
- ``<done/>`` — the task is complete.

These tags are embedded in the generated token sequence so the
orchestrator learns optimal routing through PPO.
"""

from __future__ import annotations

import re

_CALL_PATTERN = re.compile(r"<call>\s*([\w-]+)\s*</call>", re.IGNORECASE)
_DONE_PATTERN = re.compile(r"<done\s*/?>", re.IGNORECASE)


def build_orchestrator_prompt(
    role_names: list[str],
    role_descriptions: dict[str, str] | None = None,
) -> str:
    """Build a default system prompt for the orchestrator.

    Lists available agents and explains the ``<call>``/``<done/>`` protocol.
    Callers may supply a fully custom prompt instead.
    """
    if role_descriptions is None:
        role_descriptions = {}

    agent_lines = "\n".join(
        f"- {name}: {role_descriptions.get(name, 'Specialist agent.')}"
        for name in role_names
    )
    return (
        "You are an orchestrator coordinating a team of specialist agents "
        "to solve the user's task. Decide which agent should work next "
        "based on the conversation so far.\n\n"
        f"Available agents:\n{agent_lines}\n\n"
        "Instructions:\n"
        "- To delegate to an agent, write brief reasoning then: "
        "<call>agent_name</call>\n"
        "- When the task is fully complete, write: <done/>\n"
        "- You may call agents in any order and repeat calls if needed.\n"
        "- Keep your reasoning concise."
    )


def parse_orchestrator_output(
    text: str,
    valid_names: set[str],
) -> tuple[str | None, bool]:
    """Parse orchestrator output for a routing decision.

    Returns:
        ``(agent_name, is_done)`` where:

        - ``(None, True)``  — orchestrator emitted ``<done/>``.
        - ``(name, False)``  — orchestrator called a valid agent.
        - ``(None, False)`` — output is unparseable or names an
          unknown agent.
    """
    if _DONE_PATTERN.search(text):
        return None, True

    match = _CALL_PATTERN.search(text)
    if match:
        name = match.group(1).strip()
        if name in valid_names:
            return name, False

    return None, False
