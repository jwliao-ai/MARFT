"""Codeforces multi-agent RL training (unified single- and multi-agent).

Agents collaborate on competitive programming problems via a sequential graph.
The team reward is the code execution pass-rate on stdin/stdout test cases;
``multi_agent_code_reward_fn`` extracts the last ```python``` block from the
full conversation and evaluates it.

For single-agent mode, omit the ``multi_agent:`` section from the config (or
set ``multi_agent: null``).  The script then uses ``RLVRWorkflow`` instead of
``MultiAgentWorkflow``.

Usage::

    # Single-agent (no multi_agent: section in YAML)
    python examples/marft/codeforces_train.py \\
        --config examples/marft/codeforces_1agent.yaml \\
        scheduler.type=local

    # Multi-agent (multi_agent: section present in YAML)
    python examples/marft/codeforces_train.py \\
        --config examples/marft/codeforces_2agent.yaml \\
        scheduler.type=local
"""

from __future__ import annotations

import os
import sys

from areal import PPOTrainer
from areal.api.cli_args import PPOConfig, load_expr_config
from areal.dataset import get_custom_dataset
from areal.utils.hf_utils import load_hf_tokenizer

# ---------------------------------------------------------------------------
# Default role definitions for competitive programming
# ---------------------------------------------------------------------------

DEFAULT_ROLES: dict[str, dict] = {
    "planner": {
        "name": "planner",
        "system_prompt": (
            "You are a competitive programming problem analyst. "
            "Given a problem statement, analyze the constraints, identify the "
            "algorithm (dynamic programming, graph algorithms, math, greedy, etc.), "
            "note edge cases, and outline a clear solution strategy. "
            "Do NOT write code — only produce the analysis and plan."
        ),
        "description": "Analyses the problem and produces an algorithm plan.",
    },
    "solver": {
        "name": "solver",
        "system_prompt": (
            "You are an expert competitive programmer writing Python 3. "
            "Follow the plan provided in the conversation and implement a "
            "correct, efficient solution that handles all edge cases and "
            "satisfies the given constraints. "
            "Enclose your final solution within ```python ... ``` delimiters."
        ),
        "description": "Implements the plan as a correct Python 3 solution.",
    },
    "reviewer": {
        "name": "reviewer",
        "system_prompt": (
            "You are a competitive programming code reviewer. "
            "Examine the solution in the conversation for correctness, "
            "time/space complexity, and edge cases. "
            "If the code is correct, restate it within ```python ... ``` delimiters. "
            "If it has bugs or will TLE/MLE, provide the corrected version "
            "within ```python ... ``` delimiters."
        ),
        "description": "Reviews the solution for correctness and performance.",
    },
    "optimizer": {
        "name": "optimizer",
        "system_prompt": (
            "You are a Python performance optimizer for competitive programming. "
            "Take the solution in the conversation and optimize it for speed and "
            "memory usage so it passes within the time and memory limits. "
            "Output the final optimized solution within ```python ... ``` delimiters."
        ),
        "description": "Optimizes the solution for time and memory efficiency.",
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_multi_agent(config: PPOConfig) -> bool:
    """Return True if the config has a populated multi_agent section."""
    ma = config.multi_agent
    return ma is not None and bool(getattr(ma, "role_names", None))


def _build_multi_agent_workflow_kwargs(config: PPOConfig) -> dict:
    """Build serializable workflow_kwargs for MultiAgentWorkflow workers."""
    ma_cfg = config.multi_agent

    role_configs: dict = ma_cfg.role_configs or DEFAULT_ROLES
    role_names: list[str] = ma_cfg.role_names or list(role_configs.keys())

    # Assign per-agent LoRA names only when NOT using a single shared adapter.
    if ma_cfg.use_multi_lora and not ma_cfg.shared_lora:
        prefix = ma_cfg.lora_name_prefix or "agent"
        for name in role_names:
            cfg = role_configs[name]
            if isinstance(cfg, dict):
                if not cfg.get("lora_name"):
                    cfg["lora_name"] = f"{prefix}_{name}"
            else:
                if not getattr(cfg, "lora_name", None):
                    cfg.lora_name = f"{prefix}_{name}"

    if ma_cfg.graph_type == "sequential":
        graph_config: dict = {
            "nodes": [{"id": name, "role_name": name} for name in role_names],
            "edges": [
                [role_names[i], role_names[i + 1]]
                for i in range(len(role_names) - 1)
            ],
        }
        if ma_cfg.transition_messages:
            for i, msg in enumerate(ma_cfg.transition_messages):
                if msg is not None:
                    graph_config["nodes"][i]["transition_message"] = msg
    elif ma_cfg.graph_config is not None:
        graph_config = ma_cfg.graph_config
    else:
        raise ValueError(f"graph_type='{ma_cfg.graph_type}' requires graph_config.")

    dump_dir = None
    if ma_cfg.dump_transcripts and config.rollout.fileroot:
        dump_dir = os.path.join(config.rollout.fileroot, "multi_agent_transcripts")

    return dict(
        graph=graph_config,
        roles=role_configs,
        reward_fn="areal.reward.multi_agent_code.multi_agent_code_reward_fn",
        gconfig=config.gconfig,
        tokenizer=config.tokenizer_path,
        credit_strategy=ma_cfg.credit_strategy,
        credit_discount=ma_cfg.credit_discount,
        step_reward_fn=ma_cfg.step_reward_fn,
        enable_thinking=ma_cfg.enable_thinking,
        reward_timeout=120,
        context_length=config.sglang.context_length or 32768,
        dump_dir=dump_dir,
    )


def _build_single_agent_workflow_kwargs(config: PPOConfig) -> dict:
    """Build serializable workflow_kwargs for single-agent RLVRWorkflow."""
    return dict(
        reward_fn="areal.reward.multi_agent_code.multi_agent_code_reward_fn",
        gconfig=config.gconfig,
        tokenizer=config.tokenizer_path,
        reward_timeout=120,
        context_length=config.sglang.context_length or 32768,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(args: list[str]) -> None:
    config, _ = load_expr_config(args, PPOConfig)
    tokenizer = load_hf_tokenizer(config.tokenizer_path)

    train_dataset = get_custom_dataset(
        split="train",
        dataset_config=config.train_dataset,
        tokenizer=tokenizer,
    )
    valid_dataset = get_custom_dataset(
        split="test",
        dataset_config=config.valid_dataset,
        tokenizer=tokenizer,
    )

    if _is_multi_agent(config):
        workflow_kwargs = _build_multi_agent_workflow_kwargs(config)
        eval_workflow_kwargs = workflow_kwargs.copy()
        eval_workflow_kwargs["gconfig"] = config.gconfig.new(
            temperature=0.6, n_samples=1, lora_name=""
        )
        workflow = "areal.workflow.multi_agent.MultiAgentWorkflow"
        eval_workflow = "areal.workflow.multi_agent.MultiAgentWorkflow"
    else:
        workflow_kwargs = _build_single_agent_workflow_kwargs(config)
        eval_workflow_kwargs = workflow_kwargs.copy()
        eval_workflow_kwargs["gconfig"] = config.gconfig.new(
            temperature=0.6, n_samples=1
        )
        workflow = "areal.workflow.rlvr.RLVRWorkflow"
        eval_workflow = "areal.workflow.rlvr.RLVRWorkflow"

    with PPOTrainer(
        config,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
    ) as trainer:
        trainer.train(
            workflow=workflow,
            workflow_kwargs=workflow_kwargs,
            eval_workflow=eval_workflow,
            eval_workflow_kwargs=eval_workflow_kwargs,
        )


if __name__ == "__main__":
    main(sys.argv[1:])
