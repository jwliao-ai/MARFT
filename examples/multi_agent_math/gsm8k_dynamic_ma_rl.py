"""GSM8K multi-agent RL with dynamic LLM orchestration.

The orchestrator agent decides which specialist (planner, solver,
verifier) to invoke at each step via ``<call>``/``<done/>`` tags.
Routing decisions are trainable — the model learns optimal delegation
through PPO.

Usage::

    python examples/multi_agent_math/gsm8k_dynamic_ma_rl.py \
        --config examples/multi_agent_math/gsm8k_dynamic_ma_ppo.yaml \
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
    """Build serializable workflow_kwargs for distributed workers."""
    ma_cfg = config.multi_agent

    role_configs = ma_cfg.role_configs or DEFAULT_ROLES

    dump_dir = None
    if ma_cfg.dump_transcripts and config.rollout.fileroot:
        dump_dir = os.path.join(config.rollout.fileroot, "dynamic_ma_transcripts")

    return dict(
        roles=role_configs,
        reward_fn="areal.reward.multi_agent.multi_agent_math_reward_fn",
        gconfig=config.gconfig,
        tokenizer=config.tokenizer_path,
        orchestrator_prompt=ma_cfg.orchestrator_prompt,
        orchestrator_max_new_tokens=ma_cfg.orchestrator_max_new_tokens,
        max_steps=ma_cfg.max_steps,
        credit_strategy=ma_cfg.credit_strategy,
        credit_discount=ma_cfg.credit_discount,
        enable_thinking=ma_cfg.enable_thinking,
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
    eval_workflow_kwargs["gconfig"] = config.gconfig.new(temperature=0.6, n_samples=1)

    with PPOTrainer(
        config,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
    ) as trainer:
        trainer.train(
            workflow="areal.workflow.multi_agent.DynamicMultiAgentWorkflow",
            workflow_kwargs=workflow_kwargs,
            eval_workflow="areal.workflow.multi_agent.DynamicMultiAgentWorkflow",
            eval_workflow_kwargs=eval_workflow_kwargs,
        )


if __name__ == "__main__":
    main(sys.argv[1:])
