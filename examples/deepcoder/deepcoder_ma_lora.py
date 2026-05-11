"""DeepCoder multi-agent RL with per-agent LoRA adapters.

Three agents (planner → coder → debugger) collaborate on each coding
problem via a fixed sequential graph.  Each agent is backed by its own
named LoRA adapter on a shared base model.

The team reward is the code execution pass rate on the problem's test
cases.  The last agent's (debugger's) final ```python``` block is
extracted and evaluated.

Usage::

    python examples/deepcoder/deepcoder_ma_lora.py \
        --config examples/deepcoder/deepcoder_ma_lora.yaml \
        scheduler.type=local
"""

import os
import sys

from areal import PPOTrainer
from areal.api.cli_args import PPOConfig, load_expr_config
from areal.dataset import get_custom_dataset
from areal.utils.hf_utils import load_hf_tokenizer

DEFAULT_ROLES = {
    "planner": {
        "name": "planner",
        "system_prompt": (
            "You are a programming problem planner. Given a coding problem, "
            "analyze the requirements, identify edge cases, and produce a "
            "clear step-by-step algorithm plan. Do NOT write code — only "
            "produce the solution plan with data structures and approach."
        ),
        "description": "Analyzes the problem and produces an algorithm plan.",
    },
    "coder": {
        "name": "coder",
        "system_prompt": (
            "You are an expert Python programmer. Follow the plan provided in "
            "the conversation and implement a correct Python solution. "
            "Enclose your code within ```python ... ``` delimiters."
        ),
        "description": "Implements the plan as Python code.",
    },
    "debugger": {
        "name": "debugger",
        "system_prompt": (
            "You are a code reviewer and debugger. Check the code provided in "
            "the conversation for correctness, edge cases, and potential bugs. "
            "If the code is correct, restate it within ```python ... ``` "
            "delimiters. If it has bugs, provide the corrected version within "
            "```python ... ``` delimiters."
        ),
        "description": "Reviews and fixes the coder's implementation.",
    },
}


def _build_workflow_kwargs(config: PPOConfig) -> dict:
    """Build serializable workflow_kwargs for distributed workers."""
    ma_cfg = config.multi_agent

    role_configs = ma_cfg.role_configs or DEFAULT_ROLES
    role_names = ma_cfg.role_names or list(role_configs.keys())

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
        graph_config = {
            "nodes": [{"id": name, "role_name": name} for name in role_names],
            "edges": [
                [role_names[i], role_names[i + 1]] for i in range(len(role_names) - 1)
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


def main(args):
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

    workflow_kwargs = _build_workflow_kwargs(config)

    eval_workflow_kwargs = workflow_kwargs.copy()
    eval_workflow_kwargs["gconfig"] = config.gconfig.new(
        temperature=0.6, n_samples=1, lora_name=""
    )

    with PPOTrainer(
        config,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
    ) as trainer:
        trainer.train(
            workflow="areal.workflow.multi_agent.MultiAgentWorkflow",
            workflow_kwargs=workflow_kwargs,
            eval_workflow="areal.workflow.multi_agent.MultiAgentWorkflow",
            eval_workflow_kwargs=eval_workflow_kwargs,
        )


if __name__ == "__main__":
    main(sys.argv[1:])
