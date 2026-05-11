"""GSM8K multi-agent RL with per-agent LoRA adapters.

Three agents (planner → solver → verifier) collaborate on each math
problem via a fixed sequential graph.  Each agent is backed by its own
named LoRA adapter on a shared base model, enabling lightweight
specialization without full-parameter copies.

Usage::

    python examples/multi_agent_math/gsm8k_ma_lora.py \
        --config examples/multi_agent_math/gsm8k_ma_lora.yaml \
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
            "You are a math problem planner. Given a math problem, break it "
            "down into clear, numbered steps. Do NOT solve the problem — only "
            "produce the solution plan."
        ),
        "description": "Breaks the problem into a step-by-step plan.",
    },
    "solver": {
        "name": "solver",
        "system_prompt": (
            "You are a math solver. Follow the plan provided in the "
            "conversation and solve the problem step by step. "
            "Put your final numeric answer inside \\boxed{}."
        ),
        "description": "Executes the plan and produces the final answer.",
    },
    "verifier": {
        "name": "verifier",
        "system_prompt": (
            "You are a math verifier. Check the solution provided in the "
            "conversation for correctness. If the answer is correct, restate "
            "it inside \\boxed{}. If it is wrong, provide the corrected "
            "answer inside \\boxed{}."
        ),
        "description": "Verifies or corrects the solver's answer.",
    },
}


def _build_workflow_kwargs(config: PPOConfig) -> dict:
    """Build serializable workflow_kwargs for distributed workers.

    When ``multi_agent.use_multi_lora`` is enabled, each role config is
    automatically assigned a ``lora_name`` following the pattern
    ``{lora_name_prefix}_{role_name}`` (e.g. ``agent_planner``), unless
    the role already specifies an explicit ``lora_name``.
    """
    ma_cfg = config.multi_agent

    role_configs = ma_cfg.role_configs or DEFAULT_ROLES
    role_names = ma_cfg.role_names or list(role_configs.keys())

    # Auto-assign per-agent LoRA adapter names when multi-LoRA is enabled.
    # With shared_lora=False each agent trains its own adapter (Phase 2),
    # so each role needs a distinct lora_name.
    # With shared_lora=True only one physical adapter is trained; all agents
    # share the same adapter name so SGLang receives a single /load_lora_adapter
    # call and generation requests reference the same name.
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
        reward_fn="areal.reward.multi_agent.multi_agent_math_reward_fn",
        gconfig=config.gconfig,
        tokenizer=config.tokenizer_path,
        credit_strategy=ma_cfg.credit_strategy,
        credit_discount=ma_cfg.credit_discount,
        step_reward_fn=ma_cfg.step_reward_fn,
        enable_thinking=ma_cfg.enable_thinking,
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
