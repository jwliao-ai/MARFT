from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AgentRole:
    """Configuration for a single agent in a multi-agent workflow.

    Each role defines a persona with a system prompt and optional generation
    parameter overrides that take precedence over the workflow-level defaults.
    """

    name: str
    system_prompt: str
    description: str = ""
    max_new_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    lora_name: str | None = None
    reward_fn: str | None = None

    def __post_init__(self):
        if not self.name:
            raise ValueError("AgentRole.name must be a non-empty string.")
        if not self.system_prompt:
            raise ValueError("AgentRole.system_prompt must be a non-empty string.")
        if self.max_new_tokens is not None and self.max_new_tokens <= 0:
            raise ValueError(
                f"AgentRole.max_new_tokens must be positive, got {self.max_new_tokens}."
            )
        if self.temperature is not None and self.temperature < 0.0:
            raise ValueError(
                f"AgentRole.temperature must be non-negative, got {self.temperature}."
            )
        if self.top_p is not None and not (0.0 < self.top_p <= 1.0):
            raise ValueError(f"AgentRole.top_p must be in (0, 1], got {self.top_p}.")

    @classmethod
    def from_config(cls, config: dict) -> AgentRole:
        """Build an AgentRole from a plain dict (e.g. YAML / Hydra config)."""
        return cls(**config)

    @classmethod
    def build_roles(cls, configs: dict[str, dict]) -> dict[str, AgentRole]:
        """Build a name→AgentRole mapping from a nested config dict.

        Args:
            configs: ``{role_name: {system_prompt: ..., ...}, ...}``
        """
        roles: dict[str, AgentRole] = {}
        for name, cfg in configs.items():
            cfg_copy = dict(cfg)
            cfg_copy.setdefault("name", name)
            roles[name] = cls.from_config(cfg_copy)
        return roles
