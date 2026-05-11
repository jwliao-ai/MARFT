# Copyright 2025 Junwei Liao, Shanghai Jiao Tong University and Shanghai Innovation Institute.
# Licensed under the Apache License, Version 2.0.

from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from collections.abc import Callable
from typing import Any

import torch
from transformers import PreTrainedTokenizerFast

from areal import workflow_context
from areal.api.cli_args import GenerationHyperparameters
from areal.api.engine_api import InferenceEngine
from areal.api.io_struct import ModelRequest
from areal.api.reward_api import AsyncRewardWrapper
from areal.api.workflow_api import RolloutWorkflow
from areal.utils import logging, stats_tracker
from areal.utils.perf_tracer import (
    atrace_session_phase,
    session_context,
    trace_session,
)
from areal.workflow.multi_agent.agent_role import AgentRole
from areal.workflow.multi_agent.credit import AgentStep, CreditAssignment
from areal.workflow.multi_agent.environment import SharedEnvironment
from areal.workflow.multi_agent.graph import WorkflowGraph

logger = logging.getLogger("MultiAgentWorkflow")


class MultiAgentWorkflow(RolloutWorkflow):
    """Orchestrates a team of agents over a shared environment.

    Each agent step extends a single concatenated token sequence — the
    same accumulation strategy used by ``MultiTurnWorkflow``.  The team
    reward is distributed across agent steps via a pluggable
    ``CreditAssignment`` strategy, producing per-token rewards that feed
    into standard PPO (actor + critic + GAE).

    Parallel fan-out layers are supported: agents in the same layer run
    concurrently via ``asyncio.gather``, each generating from a snapshot
    of the environment.  Their outputs are merged back before the next
    sequential layer.

    Both ``reward_fn`` and ``tokenizer`` accept string import paths so
    that the workflow can be reconstructed from serializable kwargs on
    each distributed worker (matching the ``RLVRWorkflow`` pattern).
    """

    def __init__(
        self,
        graph: WorkflowGraph | dict,
        roles: dict[str, AgentRole] | dict[str, dict],
        reward_fn: Callable[..., Any] | str,
        gconfig: GenerationHyperparameters,
        tokenizer: PreTrainedTokenizerFast | str,
        credit_strategy: str = "equal",
        credit_discount: float = 1.0,
        step_reward_fn: Callable[..., float] | str | None = None,
        enable_thinking: bool = False,
        reward_timeout: float = 15,
        context_length: int = 32768,
        dump_dir: str | None = None,
    ):
        if isinstance(graph, dict):
            graph = WorkflowGraph.from_config(graph)
        self.graph = graph

        if roles and isinstance(next(iter(roles.values())), dict):
            roles = AgentRole.build_roles(roles)
        self.roles: dict[str, AgentRole] = roles

        self._validate_roles()

        if isinstance(tokenizer, str):
            from areal.utils.hf_utils import load_hf_tokenizer

            tokenizer = load_hf_tokenizer(tokenizer)
        self.tokenizer = tokenizer

        self.reward_fn = reward_fn
        self.reward_timeout = reward_timeout
        if not isinstance(reward_fn, str):
            self.async_reward_fn = AsyncRewardWrapper(
                reward_fn, timeout_seconds=reward_timeout
            )

        self.gconfig = gconfig.new_with_stop_and_pad_token_ids(self.tokenizer)
        self.enable_thinking = enable_thinking
        self.context_length = context_length
        self.dump_dir = dump_dir

        if isinstance(step_reward_fn, str):
            from areal.utils.dynamic_import import import_from_string

            step_reward_fn = import_from_string(step_reward_fn)

        # Resolve per-agent reward functions from role configs.
        per_agent_reward_fns: dict[str, Callable[..., float]] = {}
        for role_name, role in self.roles.items():
            if role.reward_fn is not None:
                from areal.utils.dynamic_import import import_from_string

                per_agent_reward_fns[role_name] = import_from_string(role.reward_fn)

        self.credit = CreditAssignment(
            strategy=credit_strategy,
            discount=credit_discount,
            step_reward_fn=step_reward_fn,
            per_agent_reward_fns=per_agent_reward_fns,
        )

        self._precompute_transition_tokens()

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _validate_roles(self) -> None:
        """Ensure every graph node references a known role."""
        for node in self.graph.nodes.values():
            if node.role_name not in self.roles:
                raise ValueError(
                    f"Graph node '{node.id}' references unknown role "
                    f"'{node.role_name}'.  Known roles: "
                    f"{list(self.roles.keys())}."
                )

    def _precompute_transition_tokens(self) -> None:
        """Cache token sequences for role transitions.

        Each transition consists of an EOS + user-message + generation-prompt
        boundary, mirroring the ``MultiTurnWorkflow`` multi-turn prompt
        construction.
        """
        self._transition_token_cache: dict[str, list[int]] = {}

        for node in self.graph.nodes.values():
            if node.transition_message is None:
                continue
            messages = [
                {"role": "assistant", "content": "placeholder"},
            ]
            s1 = list(self.tokenizer.apply_chat_template(messages, tokenize=True))
            messages.append({"role": "user", "content": node.transition_message})
            s2 = list(
                self.tokenizer.apply_chat_template(
                    messages, tokenize=True, add_generation_prompt=True
                )
            )
            self._transition_token_cache[node.id] = s2[len(s1) :]

    def _build_gconfig_for_role(self, role: AgentRole) -> GenerationHyperparameters:
        """Apply per-role overrides on top of the workflow-level gconfig."""
        overrides: dict[str, Any] = {}
        if role.max_new_tokens is not None:
            overrides["max_new_tokens"] = role.max_new_tokens
        if role.temperature is not None:
            overrides["temperature"] = role.temperature
        if role.top_p is not None:
            overrides["top_p"] = role.top_p
        if role.lora_name is not None:
            overrides["lora_name"] = role.lora_name
        if overrides:
            return self.gconfig.new(n_samples=1, **overrides)
        return self.gconfig.new(n_samples=1)

    def _resolve_reward_fn(self) -> None:
        """Lazily import reward_fn from string path (like RLVRWorkflow)."""
        if isinstance(self.reward_fn, str):
            from areal.utils.dynamic_import import import_from_string

            self.reward_fn = import_from_string(self.reward_fn)
            self.async_reward_fn = AsyncRewardWrapper(
                self.reward_fn, timeout_seconds=self.reward_timeout
            )

    # ------------------------------------------------------------------
    # Core episode logic
    # ------------------------------------------------------------------

    @session_context()
    async def arun_episode(
        self, engine: InferenceEngine, data: dict[str, Any]
    ) -> dict[str, torch.Tensor] | None:
        self._resolve_reward_fn()

        env = SharedEnvironment.from_data(data)

        seq: list[int] = []
        logprobs: list[float] = []
        loss_mask: list[int] = []
        versions: list[int] = []
        agent_steps: list[AgentStep] = []
        agent_gen_times: dict[str, float] = {}

        input_ids: list[int] = list(
            self.tokenizer.apply_chat_template(
                env.get_messages_for_role(self._first_role()),
                tokenize=True,
                add_generation_prompt=True,
                **({"enable_thinking": True} if self.enable_thinking else {}),
            )
        )

        initial_prompt_len = len(input_ids)
        layers = self.graph.get_execution_order()

        for layer_idx, layer in enumerate(layers):
            if len(layer) == 1:
                node_id = layer[0]
                node = self.graph.nodes[node_id]
                role = self.roles[node.role_name]

                if layer_idx > 0:
                    input_ids = self._apply_transition(input_ids, seq, node, role)

                try:
                    step, resp, gen_time = await self._run_agent(
                        engine, node, role, input_ids, len(seq)
                    )
                except Exception:
                    logger.exception(
                        "Agent '%s' (node=%s) failed at layer %d.",
                        role.name,
                        node.id,
                        layer_idx,
                    )
                    return None

                step.step_index = len(agent_steps)
                agent_steps.append(step)
                agent_gen_times[role.name] = (
                    agent_gen_times.get(role.name, 0.0) + gen_time
                )

                input_len = len(resp.input_tokens) - len(seq)
                if len(seq) > 0 and resp.input_tokens[: len(seq)] != seq:
                    logger.warning(
                        "Token prefix mismatch at step %d (agent=%s). "
                        "Expected prefix length %d but input tokens diverged.",
                        len(agent_steps) - 1,
                        role.name,
                        len(seq),
                    )
                seq += resp.input_tokens[-input_len:] + resp.output_tokens
                logprobs += [0.0] * input_len + resp.output_logprobs
                loss_mask += [0] * input_len + [1] * resp.output_len
                versions += [-1] * input_len + resp.output_versions

                output_text = self.tokenizer.decode(resp.output_tokens)
                step.output_text = output_text
                step.token_end = len(seq)
                step.token_start = step.token_end - resp.output_len
                env.add_assistant_message(output_text, agent_name=role.name)

                input_ids = list(resp.input_tokens) + list(resp.output_tokens)
            else:
                result = await self._run_parallel_layer(
                    engine,
                    layer,
                    env,
                    seq,
                    logprobs,
                    loss_mask,
                    versions,
                    input_ids,
                    agent_steps,
                    agent_gen_times,
                    layer_idx,
                )
                if result is None:
                    return None
                (
                    seq,
                    logprobs,
                    loss_mask,
                    versions,
                    input_ids,
                ) = result

        prompt_ids = seq[:initial_prompt_len]
        completion_ids = seq[initial_prompt_len:]
        prompt_str = self.tokenizer.decode(prompt_ids)
        completion_str = self.tokenizer.decode(completion_ids)

        team_reward = await self._compute_reward(
            prompt_str, completion_str, prompt_ids, completion_ids, data
        )
        team_reward = float(team_reward)

        reward_tensor = self.credit.compute_per_token_rewards(
            team_reward=team_reward,
            agent_steps=agent_steps,
            seq_len=len(seq),
            env=env,
            data=data,
        )

        self._log_metrics(
            team_reward=team_reward,
            agent_steps=agent_steps,
            agent_gen_times=agent_gen_times,
            seq_len=len(seq),
            env=env,
            data=data,
        )

        if self.dump_dir:
            await self._dump_transcript(engine, env, agent_steps, team_reward, data)

        # Build per-token agent mapping: -1 for prompt/transition tokens,
        # agent index for each agent's generated tokens.
        role_name_to_idx = {name: i for i, name in enumerate(self.roles)}
        agent_ids = [-1] * len(seq)
        for step in agent_steps:
            idx = role_name_to_idx.get(step.agent_name, -1)
            for t in range(step.token_start, step.token_end):
                agent_ids[t] = idx

        res = {
            "input_ids": torch.tensor(seq, dtype=torch.int32),
            "logprobs": torch.tensor(logprobs, dtype=torch.float32),
            "loss_mask": torch.tensor(loss_mask, dtype=torch.int32),
            "versions": torch.tensor(versions, dtype=torch.int32),
            "rewards": torch.tensor(team_reward, dtype=torch.float32),
            "token_rewards": reward_tensor,
            "agent_ids": torch.tensor(agent_ids, dtype=torch.int32),
            "attention_mask": torch.ones(len(seq), dtype=torch.bool),
        }
        res = {k: v.unsqueeze(0) for k, v in res.items()}
        # Non-tensor metadata for per-agent rollout log breakdown.
        res["agent_names"] = list(self.roles.keys())
        return res

    # ------------------------------------------------------------------
    # Single-agent step
    # ------------------------------------------------------------------

    async def _run_agent(
        self,
        engine: InferenceEngine,
        node,
        role: AgentRole,
        input_ids: list[int],
        seq_offset: int,
    ) -> tuple[AgentStep, Any, float]:
        """Generate one agent response and return the bookkeeping step."""
        gconfig = self._build_gconfig_for_role(role)

        # Dynamically cap max_new_tokens so input + output fits in context.
        # Leaves a 64-token margin for safety (special tokens, EOS, etc.).
        remaining = self.context_length - len(input_ids) - 64
        if remaining < gconfig.max_new_tokens:
            if remaining <= 0:
                raise RuntimeError(
                    f"Agent '{role.name}' input ({len(input_ids)} tokens) "
                    f"already exceeds context_length ({self.context_length}). "
                    f"Consider reducing max_new_tokens for earlier agents."
                )
            logger.info(
                "Capping max_new_tokens for agent '%s' from %d to %d "
                "(input_len=%d, context_length=%d).",
                role.name,
                gconfig.max_new_tokens,
                remaining,
                len(input_ids),
                self.context_length,
            )
            gconfig = gconfig.new(max_new_tokens=remaining)

        req = ModelRequest(
            rid=uuid.uuid4().hex,
            input_ids=input_ids,
            gconfig=gconfig,
            tokenizer=self.tokenizer,
        )
        t0 = time.perf_counter()
        async with atrace_session_phase("generate"):
            resp = await engine.agenerate(req)
        gen_time = time.perf_counter() - t0

        token_start = len(resp.input_tokens)
        token_end = token_start + resp.output_len

        step = AgentStep(
            agent_name=role.name,
            step_index=0,
            token_start=token_start,
            token_end=token_end,
        )
        return step, resp, gen_time

    # ------------------------------------------------------------------
    # Reward computation (traced)
    # ------------------------------------------------------------------

    @trace_session("reward")
    async def _compute_reward(
        self,
        prompt_str: str,
        completion_str: str,
        prompt_ids: list[int],
        completion_ids: list[int],
        data: dict[str, Any],
    ) -> float:
        """Compute team reward with session tracing."""
        reward = await self.async_reward_fn(
            prompt_str,
            completion_str,
            prompt_ids,
            completion_ids,
            **data,
        )
        return float(reward)

    # ------------------------------------------------------------------
    # Parallel layer execution
    # ------------------------------------------------------------------

    async def _run_parallel_layer(
        self,
        engine: InferenceEngine,
        layer: list[str],
        env: SharedEnvironment,
        seq: list[int],
        logprobs: list[float],
        loss_mask: list[int],
        versions: list[int],
        input_ids: list[int],
        agent_steps: list[AgentStep],
        agent_gen_times: dict[str, float],
        layer_idx: int,
    ) -> tuple[list[int], list[float], list[int], list[int], list[int]] | None:
        """Run parallel agents and concatenate results sequentially.

        Returns None if any agent in the layer fails, signalling the
        caller to abort the episode.
        """
        snapshots = [env.snapshot() for _ in layer]

        async def _run_one(node_id: str, snap: SharedEnvironment):
            node = self.graph.nodes[node_id]
            role = self.roles[node.role_name]
            branch_input_ids = self._apply_transition(
                list(input_ids), list(seq), node, role
            )
            step, resp, gen_time = await self._run_agent(
                engine, node, role, branch_input_ids, len(seq)
            )
            output_text = self.tokenizer.decode(resp.output_tokens)
            step.output_text = output_text
            snap.add_assistant_message(output_text, agent_name=role.name)
            return step, resp, snap, gen_time

        raw_results = await asyncio.gather(
            *[_run_one(nid, snap) for nid, snap in zip(layer, snapshots)],
            return_exceptions=True,
        )

        failures = [
            (layer[i], r)
            for i, r in enumerate(raw_results)
            if isinstance(r, BaseException)
        ]
        if failures:
            for node_id, exc in failures:
                logger.error(
                    "Parallel agent '%s' failed at layer %d: %s",
                    node_id,
                    layer_idx,
                    exc,
                    exc_info=exc,
                )
            return None

        for step, resp, snap, gen_time in raw_results:
            step.step_index = len(agent_steps)
            input_len = len(resp.input_tokens) - len(seq)
            seq += resp.input_tokens[-input_len:] + resp.output_tokens
            logprobs += [0.0] * input_len + resp.output_logprobs
            loss_mask += [0] * input_len + [1] * resp.output_len
            versions += [-1] * input_len + resp.output_versions
            step.token_end = len(seq)
            step.token_start = step.token_end - resp.output_len
            agent_steps.append(step)
            agent_gen_times[step.agent_name] = (
                agent_gen_times.get(step.agent_name, 0.0) + gen_time
            )

        env.merge_parallel_results([snap for _, _, snap, _ in raw_results])
        input_ids = list(seq)

        return seq, logprobs, loss_mask, versions, input_ids

    # ------------------------------------------------------------------
    # Metrics logging
    # ------------------------------------------------------------------

    def _log_metrics(
        self,
        team_reward: float,
        agent_steps: list[AgentStep],
        agent_gen_times: dict[str, float],
        seq_len: int,
        env: SharedEnvironment,
        data: dict[str, Any],
    ) -> None:
        """Emit structured metrics via stats_tracker."""
        scope = workflow_context.stat_scope()
        tracker = stats_tracker.get(scope)

        tracker.scalar(
            reward=team_reward,
            num_agents=len(agent_steps),
            seq_len=seq_len,
        )

        per_agent_rewards = self.credit.compute_per_agent_rewards(
            team_reward=team_reward,
            agent_steps=agent_steps,
            seq_len=seq_len,
            env=env,
            data=data,
        )
        for agent_name, agent_reward in per_agent_rewards.items():
            tracker.scalar(**{f"agent_{agent_name}_reward": agent_reward})

        for step in agent_steps:
            output_len = step.token_end - step.token_start
            tracker.scalar(**{f"agent_{step.agent_name}_output_len": output_len})

        for agent_name, gen_time in agent_gen_times.items():
            tracker.scalar(**{f"agent_{agent_name}_gen_time": gen_time})

    # ------------------------------------------------------------------
    # Transition helpers
    # ------------------------------------------------------------------

    def _first_role(self) -> AgentRole:
        layers = self.graph.get_execution_order()
        first_node_id = layers[0][0]
        return self.roles[self.graph.nodes[first_node_id].role_name]

    def _apply_transition(
        self,
        input_ids: list[int],
        seq: list[int],
        node,
        role: AgentRole,
    ) -> list[int]:
        """Append transition tokens (EOS + user message) and re-template."""
        if node.id in self._transition_token_cache:
            if input_ids and input_ids[-1] != self.tokenizer.eos_token_id:
                input_ids.append(self.tokenizer.eos_token_id)
            input_ids += self._transition_token_cache[node.id]
        else:
            if input_ids and input_ids[-1] != self.tokenizer.eos_token_id:
                input_ids.append(self.tokenizer.eos_token_id)
            role_transition_msg = (
                f"Now it is {role.name}'s turn. {role.description or ''}"
            ).strip()
            messages = [
                {"role": "assistant", "content": "placeholder"},
                {"role": "user", "content": role_transition_msg},
            ]
            s1 = list(self.tokenizer.apply_chat_template([messages[0]], tokenize=True))
            s2 = list(
                self.tokenizer.apply_chat_template(
                    messages, tokenize=True, add_generation_prompt=True
                )
            )
            transition_tokens = s2[len(s1) :]
            input_ids += transition_tokens
        return input_ids

    # ------------------------------------------------------------------
    # Transcript dumping (async-safe)
    # ------------------------------------------------------------------

    async def _dump_transcript(
        self,
        engine: InferenceEngine,
        env: SharedEnvironment,
        agent_steps: list[AgentStep],
        team_reward: float,
        data: dict[str, Any],
    ) -> None:
        try:
            version = (
                engine.get_version() if hasattr(engine, "get_version") else "unknown"
            )
            dump_path = os.path.join(self.dump_dir, str(version))
            os.makedirs(dump_path, exist_ok=True)

            transcript = {
                "team_reward": team_reward,
                "messages": env.messages,
                "agent_steps": [
                    {
                        "agent_name": s.agent_name,
                        "step_index": s.step_index,
                        "token_start": s.token_start,
                        "token_end": s.token_end,
                        "output_text": s.output_text,
                    }
                    for s in agent_steps
                ],
                "answer": data.get("answer", ""),
            }
            fname = os.path.join(dump_path, f"{uuid.uuid4().hex}.json")
            content = json.dumps(transcript, indent=2, ensure_ascii=False)
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, _write_file, fname, content)
        except Exception:
            logger.warning("Failed to dump multi-agent transcript.", exc_info=True)


def _write_file(path: str, content: str) -> None:
    """Write content to file in a thread-safe manner."""
    with open(path, "w") as f:
        f.write(content)
