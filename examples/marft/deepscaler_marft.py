# Copyright 2025 Junwei Liao, Shanghai Jiao Tong University and Shanghai Innovation Institute.
# Licensed under the Apache License, Version 2.0.

"""DeepScaleR RL training with multi-benchmark evaluation.

Supports single-agent (RLVRWorkflow) and multi-agent (MultiAgentWorkflow)
modes from one unified script.  The active mode is determined by the config:
if ``multi_agent.role_names`` is set the script uses MultiAgentWorkflow;
otherwise it falls back to RLVRWorkflow.

Benchmarks evaluated independently with isolated metrics:
  - aime_2024/reward     (30 questions, n_samples=16, reports avg pass@1)
  - MATH-500/reward      (500 questions, n_samples=1)
  - olympiadbench/reward (674 questions, n_samples=1)
  - minervamath/reward   (272 questions, n_samples=1)

Usage::

    # Single agent
    python examples/marft/deepscaler_marft.py \\
        --config examples/marft/deepscaler_marft_2agent.yaml \\
        scheduler.type=local

    # Multi-agent (2 / 3 / 4 agents)
    python examples/marft/deepscaler_marft.py \\
        --config examples/marft/deepscaler_marft_2agent.yaml \\
        scheduler.type=local
"""

from __future__ import annotations

import functools
import os
import sys
from typing import TYPE_CHECKING

import torch.distributed as dist

from areal import PPOTrainer
from areal.api.cli_args import PPOConfig, ValidDatasetConfig, load_expr_config
from areal.dataset.deepscaler import (
    get_aime2024_rl_dataset,
    get_deepscaler_rl_dataset,
    get_math500_rl_dataset,
    get_minervamath_rl_dataset,
    get_olympiadbench_rl_dataset,
)
from areal.infra.platforms import current_platform
from areal.utils import logging
from areal.utils.hf_utils import load_hf_tokenizer

if TYPE_CHECKING:
    from datasets import Dataset

logger = logging.getLogger("DeepScaleRTrainer")

# ---------------------------------------------------------------------------
# Eval benchmark registry
# Each entry: benchmark_name -> (path, n_samples)
# Override the root via the EVAL_DATA_ROOT environment variable.
# ---------------------------------------------------------------------------
EVAL_DATA_ROOT = os.environ.get(
    "EVAL_DATA_ROOT",
    "/workspace/data",
)

EVAL_BENCHMARKS: dict[str, tuple[str, int]] = {
    "aime_2024": (f"{EVAL_DATA_ROOT}/aime_2024", 16),
    "MATH-500": (f"{EVAL_DATA_ROOT}/MATH-500", 1),
    "olympiadbench": (f"{EVAL_DATA_ROOT}/olympiadbench", 1),
    "minervamath": (f"{EVAL_DATA_ROOT}/minervamath", 1),
}

_EVAL_LOADERS = {
    "aime_2024": get_aime2024_rl_dataset,
    "MATH-500": get_math500_rl_dataset,
    "olympiadbench": get_olympiadbench_rl_dataset,
    "minervamath": get_minervamath_rl_dataset,
}


class MultiEvalPPOTrainer(PPOTrainer):
    """PPOTrainer subclass that evaluates multiple benchmarks independently.

    Each benchmark's stats are drained into a separate scope so that metrics
    like ``aime_2024/reward`` and ``MATH-500/reward`` appear as distinct keys
    in the stats logger instead of all collapsing into ``eval-rollout/reward``.

    Parameters
    ----------
    config:
        Full PPO experiment config.
    train_dataset:
        Training split dataset.
    eval_datasets:
        Ordered list of ``(name, dataset, n_samples)`` tuples.  *name* is used
        as the metric prefix; *n_samples* is the generation group size passed
        to ``eval_rollout.submit``.
    """

    def __init__(
        self,
        config: PPOConfig,
        train_dataset: Dataset,
        eval_datasets: list[tuple[str, Dataset, int]],
    ):
        # Pass valid_dataset=None so the base class does not build its own
        # eval dataloader — we manage them ourselves.
        super().__init__(config, train_dataset, valid_dataset=None)

        self._eval_datasets = eval_datasets
        self._pending_eval_stats: dict[str, float] = {}

        # Build one dataloader per benchmark, borrowing the valid_dataset
        # config for batch size / worker settings.
        valid_cfg: ValidDatasetConfig | None = config.valid_dataset
        if valid_cfg is None:
            valid_cfg = ValidDatasetConfig(
                batch_size=config.train_dataset.batch_size,
                pin_memory=config.train_dataset.pin_memory,
                num_workers=config.train_dataset.num_workers,
            )

        self._eval_dataloaders: list[tuple[str, object, int]] = []
        for bench_name, dataset, n_samp in eval_datasets:
            dataloader = self._create_dataloader(
                dataset,
                dataset_config=valid_cfg,
                rank=self.actor.data_parallel_rank,
                world_size=self.actor.data_parallel_world_size,
            )
            self._eval_dataloaders.append((bench_name, dataloader, n_samp))

    def _evaluate(
        self,
        eval_workflow,
        eval_workflow_kwargs,
        epoch: int,
        epoch_step: int,
        global_step: int,
    ):
        """Override parent to bypass the ``valid_dataloader is None`` guard.

        The base ``PPOTrainer._evaluate`` exits early when
        ``self.valid_dataloader`` is ``None``, but this subclass intentionally
        sets it to ``None`` because it manages its own per-benchmark
        dataloaders.  We replicate the dispatch logic here, skipping that
        check.
        """
        if eval_workflow is None:
            return
        self.evaluator.evaluate(
            functools.partial(
                self._evaluate_fn,
                eval_workflow=eval_workflow,
                eval_workflow_kwargs=eval_workflow_kwargs,
            ),
            epoch,
            epoch_step,
            global_step,
        )
        dist.barrier(group=self.actor.cpu_group)
        current_platform.synchronize()

    def _evaluate_fn(
        self,
        eval_workflow,
        eval_workflow_kwargs,
    ) -> None:
        """Run each benchmark independently and drain stats between them.

        Stats written to the ``"eval-rollout"`` tracker scope by the workflow
        are exported after every benchmark so they do not pollute the next
        one.  The remapped keys (``"eval-rollout/…"`` → ``"<bench>/…"``) are
        stashed in ``self._pending_eval_stats`` for later commit.
        """
        self._pending_eval_stats = {}

        if not self.actor.is_data_parallel_head():
            dist.barrier(group=self.actor.cpu_group)
            current_platform.synchronize()
            return

        for bench_name, dataloader, n_samp in self._eval_dataloaders:
            cnt = 0
            for data in dataloader:
                for item in data:
                    self.eval_rollout.submit(
                        item,
                        eval_workflow,
                        eval_workflow_kwargs,
                        group_size=n_samp,
                        is_eval=True,
                    )
                    cnt += 1

            self.eval_rollout.wait(cnt, timeout=None)

            # Drain the "eval-rollout" tracker before the next benchmark
            # can write to it.  Rename keys so metrics are distinguishable.
            raw: dict[str, float] = self.eval_rollout.export_stats()
            remapped = {
                k.replace("eval-rollout", bench_name, 1): v for k, v in raw.items()
            }
            self._pending_eval_stats.update(remapped)
            logger.info(
                "Benchmark '%s' evaluation complete: %s",
                bench_name,
                {k: f"{v:.4f}" for k, v in remapped.items()},
            )

        dist.barrier(group=self.actor.cpu_group)
        current_platform.synchronize()

    def _export_and_commit_stats(
        self, epoch: int, epoch_step: int, global_step: int
    ) -> None:
        """Commit training + per-benchmark eval stats to the stats logger.

        The eval stats were already drained from the ``eval_rollout`` tracker
        inside ``_evaluate_fn``.  Calling ``self.eval_rollout.export_stats()``
        here would return an empty dict, so we use the pre-drained
        ``_pending_eval_stats`` directly.
        """
        stats = self.actor.export_stats()
        stats.update(self.rollout.export_stats())
        # NOTE: eval_rollout stats were drained per-benchmark in _evaluate_fn;
        # do NOT call self.eval_rollout.export_stats() again here.
        stats.update(self._pending_eval_stats)
        self._pending_eval_stats = {}
        self.stats_logger.commit(epoch, epoch_step, global_step, stats)

        dist.barrier(group=self.actor.cpu_group)
        current_platform.synchronize()


# ---------------------------------------------------------------------------
# Workflow dispatch helpers
# ---------------------------------------------------------------------------

def _is_multi_agent(config: PPOConfig) -> bool:
    """Return True when the config requests a multi-agent workflow."""
    ma = config.multi_agent
    return bool(ma.role_names) or bool(ma.role_configs)


def _build_single_agent_workflow_kwargs(config: PPOConfig) -> dict:
    """Workflow kwargs for RLVRWorkflow (single-agent)."""
    return dict(
        reward_fn="areal.reward.multi_agent.multi_agent_math_reward_fn",
        gconfig=config.gconfig,
        tokenizer=config.tokenizer_path,
    )


def _build_multi_agent_workflow_kwargs(config: PPOConfig) -> dict:
    """Workflow kwargs for MultiAgentWorkflow.

    Mirrors the logic in ``examples/marft/gsm8k_ma_lora.py``:
    auto-assigns per-agent LoRA adapter names when ``use_multi_lora`` is
    enabled, then serialises the graph and role configs as plain dicts so
    distributed workers can reconstruct the workflow from string paths.
    """
    ma_cfg = config.multi_agent

    role_configs = ma_cfg.role_configs or {}
    role_names = ma_cfg.role_names or list(role_configs.keys())

    # Auto-assign per-agent LoRA adapter names only when each agent trains
    # its own independent adapter (shared_lora=False).  With shared_lora=True
    # there is one physical adapter; assigning distinct lora_names would cause
    # SGLang to load the same path twice under different names, corrupting its
    # LoRA registry and causing cudaErrorIllegalAddress on the next prefill.
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

    # ── Training dataset ─────────────────────────────────────────────────────
    train_dataset: Dataset = get_deepscaler_rl_dataset(
        path=config.train_dataset.path,
        split="train",
        tokenizer=tokenizer,
        max_length=config.train_dataset.max_length,
    )

    # ── Evaluation datasets ──────────────────────────────────────────────────
    eval_datasets: list[tuple[str, Dataset, int]] = []
    for bench_name, (bench_path, n_samp) in EVAL_BENCHMARKS.items():
        loader_fn = _EVAL_LOADERS[bench_name]
        dataset: Dataset = loader_fn(
            path=bench_path,
            split="test",
            tokenizer=tokenizer,
        )
        eval_datasets.append((bench_name, dataset, n_samp))
        logger.info(
            "Loaded eval benchmark '%s': %d samples, n_samples=%d",
            bench_name,
            len(dataset),
            n_samp,
        )

    # ── Workflow selection ───────────────────────────────────────────────────
    if _is_multi_agent(config):
        workflow_cls = "areal.workflow.multi_agent.MultiAgentWorkflow"
        workflow_kwargs = _build_multi_agent_workflow_kwargs(config)
        eval_workflow_kwargs = _build_multi_agent_workflow_kwargs(config)
        eval_workflow_kwargs["gconfig"] = config.eval_gconfig.new(lora_name="")
        logger.info(
            "Multi-agent mode: %d agents (%s)",
            len(config.multi_agent.role_names or []),
            ", ".join(config.multi_agent.role_names or []),
        )
    else:
        workflow_cls = "areal.workflow.rlvr.RLVRWorkflow"
        workflow_kwargs = _build_single_agent_workflow_kwargs(config)
        eval_workflow_kwargs = dict(
            reward_fn="areal.reward.multi_agent.multi_agent_math_reward_fn",
            gconfig=config.eval_gconfig,
            tokenizer=config.tokenizer_path,
        )
        logger.info("Single-agent mode: RLVRWorkflow")

    # ── Training ─────────────────────────────────────────────────────────────
    with MultiEvalPPOTrainer(
        config,
        train_dataset=train_dataset,
        eval_datasets=eval_datasets,
    ) as trainer:
        trainer.train(
            workflow=workflow_cls,
            workflow_kwargs=workflow_kwargs,
            eval_workflow=workflow_cls,
            eval_workflow_kwargs=eval_workflow_kwargs,
        )


if __name__ == "__main__":
    main(sys.argv[1:])
