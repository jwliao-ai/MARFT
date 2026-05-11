# Copyright 2025 Junwei Liao, Shanghai Jiao Tong University and Shanghai Innovation Institute.
# Licensed under the Apache License, Version 2.0.

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass


@dataclass
class GraphNode:
    """A single step in the multi-agent workflow graph.

    Attributes:
        id: Unique node identifier (e.g. ``"plan_step"``).
        role_name: Reference to an ``AgentRole.name``.
        transition_message: Optional user message injected *before* this
            agent runs, giving it context or instructions for the step.
    """

    id: str
    role_name: str
    transition_message: str | None = None


class WorkflowGraph:
    """Directed acyclic graph describing agent execution order.

    Supports sequential chains, fan-out (parallel) layers, and arbitrary
    DAG topologies.  Execution order is computed via Kahn's algorithm and
    returned as a list of *layers* — nodes in the same layer are safe to
    run concurrently.
    """

    def __init__(
        self,
        nodes: dict[str, GraphNode] | None = None,
        edges: list[tuple[str, str]] | None = None,
    ):
        self.nodes: dict[str, GraphNode] = nodes or {}
        self.edges: list[tuple[str, str]] = edges or []

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------

    @classmethod
    def sequential(
        cls,
        role_names: list[str],
        transition_messages: list[str | None] | None = None,
    ) -> WorkflowGraph:
        """Build a simple linear chain of agents.

        Args:
            role_names: Ordered list of role names, one per step.
            transition_messages: Optional per-step user messages (same
                length as *role_names*).  ``None`` entries are skipped.
        """
        if not role_names:
            raise ValueError("role_names must be non-empty.")
        if transition_messages is not None and len(transition_messages) != len(
            role_names
        ):
            raise ValueError(
                "transition_messages must have the same length as role_names."
            )

        nodes: dict[str, GraphNode] = {}
        edges: list[tuple[str, str]] = []
        for idx, name in enumerate(role_names):
            node_id = f"{name}_{idx}"
            tmsg = transition_messages[idx] if transition_messages else None
            nodes[node_id] = GraphNode(
                id=node_id, role_name=name, transition_message=tmsg
            )
            if idx > 0:
                prev_id = f"{role_names[idx - 1]}_{idx - 1}"
                edges.append((prev_id, node_id))

        graph = cls(nodes=nodes, edges=edges)
        graph.validate()
        return graph

    @classmethod
    def from_config(cls, config: dict) -> WorkflowGraph:
        """Deserialise from a dict (YAML / Hydra).

        Expected shape::

            {
                "nodes": [
                    {"id": "...", "role_name": "...", "transition_message": "..."},
                    ...
                ],
                "edges": [["src", "dst"], ...]
            }
        """
        nodes: dict[str, GraphNode] = {}
        for node_cfg in config.get("nodes", []):
            node = GraphNode(**node_cfg)
            nodes[node.id] = node
        edges = [tuple(e) for e in config.get("edges", [])]
        graph = cls(nodes=nodes, edges=edges)
        graph.validate()
        return graph

    # ------------------------------------------------------------------
    # Topology
    # ------------------------------------------------------------------

    def get_execution_order(self) -> list[list[str]]:
        """Kahn's algorithm returning layers of concurrently-runnable node IDs."""
        in_degree: dict[str, int] = {nid: 0 for nid in self.nodes}
        children: dict[str, list[str]] = defaultdict(list)

        for src, dst in self.edges:
            in_degree[dst] += 1
            children[src].append(dst)

        queue: deque[str] = deque(nid for nid, deg in in_degree.items() if deg == 0)
        layers: list[list[str]] = []

        while queue:
            layer = list(queue)
            queue.clear()
            for nid in layer:
                for child in children[nid]:
                    in_degree[child] -= 1
                    if in_degree[child] == 0:
                        queue.append(child)
            layers.append(layer)

        visited = sum(len(layer) for layer in layers)
        if visited != len(self.nodes):
            raise ValueError(
                "Cycle detected in workflow graph — "
                f"visited {visited}/{len(self.nodes)} nodes."
            )
        return layers

    def has_parallel(self) -> bool:
        """Return True if any execution layer contains more than one node."""
        return any(len(layer) > 1 for layer in self.get_execution_order())

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> None:
        """Check structural invariants and raise on error."""
        if not self.nodes:
            raise ValueError("WorkflowGraph must have at least one node.")

        node_ids = set(self.nodes.keys())
        for src, dst in self.edges:
            if src not in node_ids:
                raise ValueError(f"Edge source '{src}' is not a known node.")
            if dst not in node_ids:
                raise ValueError(f"Edge destination '{dst}' is not a known node.")

        self.get_execution_order()
