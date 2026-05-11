# Copyright 2025 Junwei Liao, Shanghai Jiao Tong University and Shanghai Innovation Institute.
# Licensed under the Apache License, Version 2.0.

"""Dynamic multi-agent workflow driven by an LLM orchestrator.

Instead of following a fixed DAG, an orchestrator agent decides which
specialist to invoke at each step.  The orchestrator's routing decisions
(``<call>agent</call>`` / ``<done/>``) are part of the trainable token
sequence, so the model learns optimal delegation through PPO.

Token sequence structure::

    [orch system + user question]
    [orch: reasoning... <call>planner</call>]
    [transition → planner]
    [planner: Here is my plan...]
    [transition → orchestrator]
    [orch: Good. <call>solver</call>]
    [transition → solver]
    [solver: The answer is 4.]
    [transition → orchestrator]
    [orch: Verified. <done/>]
"""

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
from areal.workflow.multi_agent.orchestrator import (
    build_orchestrator_prompt,
    parse_orchestrator_output,
)

logger = logging.getLogger("DynamicMultiAgentWorkflow")


class DynamicMultiAgentWorkflow(RolloutWorkflow):
    """LLM-orchestrated dynamic multi-agent workflow.

    The orchestrator decides which specialist to invoke at each step.
    Its routing tokens are part of the trainable sequence — credit
    assignment spans both orchestrator and specialist outputs.

    Constructor accepts string import paths for ``reward_fn`` and
    ``tokenizer`` so the workflow is serializable for distributed workers.
    """

    def __init__(
        self,
        roles: dict[str, AgentRole] | dict[str, dict],
        reward_fn: Callable[..., Any] | str,
        gconfig: GenerationHyperparameters,
        tokenizer: PreTrainedTokenizerFast | str,
        orchestrator_prompt: str | None = None,
        orchestrator_max_new_tokens: int | None = None,
        max_steps: int = 10,
        credit_strategy: str = "equal",
        credit_discount: float = 1.0,
        step_reward_fn: Callable[..., float] | str | None = None,
        enable_thinking: bool = False,
        reward_timeout: float = 15,
        context_length: int = 32768,
        dump_dir: str | None = None,
    ):
        if roles and isinstance(next(iter(roles.values())), dict):
            roles = AgentRole.build_roles(roles)
        self.roles: dict[str, AgentRole] = roles

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
        self.max_steps = max_steps

        if orchestrator_prompt is None:
            role_descs = {
                name: r.description or r.system_prompt[:80]
                for name, r in self.roles.items()
            }
            orchestrator_prompt = build_orchestrator_prompt(
                role_names=list(self.roles.keys()),
                role_descriptions=role_descs,
            )
        self._orchestrator_role = AgentRole(
            name="orchestrator",
            system_prompt=orchestrator_prompt,
            max_new_tokens=orchestrator_max_new_tokens,
        )

        if isinstance(step_reward_fn, str):
            from areal.utils.dynamic_import import import_from_string

            step_reward_fn = import_from_string(step_reward_fn)

        self.credit = CreditAssignment(
            strategy=credit_strategy,
            discount=credit_discount,
            step_reward_fn=step_reward_fn,
        )

        self._build_transition_cache()

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _build_transition_cache(self) -> None:
        """Pre-compute transition token sequences for role switches."""
        self._transition_cache: dict[str, list[int]] = {}

        targets: dict[str, str] = {
            f"to_{name}": f"Now it is {name}'s turn." for name in self.roles
        }
        targets["to_orchestrator"] = "Please decide the next step."

        for key, message in targets.items():
            messages = [{"role": "assistant", "content": "placeholder"}]
            s1 = list(self.tokenizer.apply_chat_template(messages, tokenize=True))
            messages.append({"role": "user", "content": message})
            s2 = list(
                self.tokenizer.apply_chat_template(
                    messages, tokenize=True, add_generation_prompt=True
                )
            )
            self._transition_cache[key] = s2[len(s1) :]

    def _get_transition_tokens(self, target: str) -> list[int]:
        """Return cached transition tokens for switching to *target*."""
        key = f"to_{target}"
        if key in self._transition_cache:
            return self._transition_cache[key]
        message = f"Now it is {target}'s turn."
        messages = [{"role": "assistant", "content": "placeholder"}]
        s1 = list(self.tokenizer.apply_chat_template(messages, tokenize=True))
        messages.append({"role": "user", "content": message})
        s2 = list(
            self.tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True
            )
        )
        tokens = s2[len(s1) :]
        self._transition_cache[key] = tokens
        return tokens

    def _append_transition(self, input_ids: list[int], target: str) -> list[int]:
        """Append EOS + transition tokens to switch to *target* role."""
        if input_ids and input_ids[-1] != self.tokenizer.eos_token_id:
            input_ids.append(self.tokenizer.eos_token_id)
        input_ids += self._get_transition_tokens(target)
        return input_ids

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
        """Lazily import reward_fn from string path."""
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
                env.get_messages_for_role(self._orchestrator_role),
                tokenize=True,
                add_generation_prompt=True,
                **({"enable_thinking": True} if self.enable_thinking else {}),
            )
        )
        initial_prompt_len = len(input_ids)
        valid_names = set(self.roles.keys())

        num_steps = 0
        while num_steps < self.max_steps:
            # --- Orchestrator turn ---
            try:
                orch_resp, orch_time = await self._generate(
                    engine, self._orchestrator_role, input_ids
                )
            except Exception:
                logger.exception("Orchestrator failed at step %d.", num_steps)
                return None

            self._accumulate(seq, logprobs, loss_mask, versions, orch_resp)
            orch_text = self.tokenizer.decode(orch_resp.output_tokens)
            orch_step = AgentStep(
                agent_name="orchestrator",
                step_index=len(agent_steps),
                token_start=len(seq) - orch_resp.output_len,
                token_end=len(seq),
                output_text=orch_text,
            )
            agent_steps.append(orch_step)
            agent_gen_times["orchestrator"] = (
                agent_gen_times.get("orchestrator", 0.0) + orch_time
            )
            env.add_assistant_message(orch_text, agent_name="orchestrator")

            called_agent, is_done = parse_orchestrator_output(orch_text, valid_names)

            if is_done:
                break

            if called_agent is None:
                logger.warning(
                    "Orchestrator output unparseable at step %d, "
                    "terminating episode: %.200s",
                    num_steps,
                    orch_text,
                )
                break

            # --- Specialist turn ---
            role = self.roles[called_agent]
            input_ids = list(orch_resp.input_tokens) + list(orch_resp.output_tokens)
            input_ids = self._append_transition(input_ids, called_agent)

            try:
                agent_resp, agent_time = await self._generate(engine, role, input_ids)
            except Exception:
                logger.exception(
                    "Agent '%s' failed at step %d.", called_agent, num_steps
                )
                return None

            self._accumulate(seq, logprobs, loss_mask, versions, agent_resp)
            agent_text = self.tokenizer.decode(agent_resp.output_tokens)
            agent_step = AgentStep(
                agent_name=called_agent,
                step_index=len(agent_steps),
                token_start=len(seq) - agent_resp.output_len,
                token_end=len(seq),
                output_text=agent_text,
            )
            agent_steps.append(agent_step)
            agent_gen_times[called_agent] = (
                agent_gen_times.get(called_agent, 0.0) + agent_time
            )
            env.add_assistant_message(agent_text, agent_name=called_agent)

            # Transition back to orchestrator for next decision
            input_ids = list(agent_resp.input_tokens) + list(agent_resp.output_tokens)
            input_ids = self._append_transition(input_ids, "orchestrator")

            num_steps += 1

        if not seq:
            return None

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
            num_steps=num_steps,
        )

        if self.dump_dir:
            await self._dump_transcript(engine, env, agent_steps, team_reward, data)

        # Build per-token agent mapping: -1 for prompt/transition tokens,
        # specialist index for specialist tokens, len(roles) for orchestrator.
        role_name_to_idx = {name: i for i, name in enumerate(self.roles)}
        orchestrator_idx = len(self.roles)
        agent_ids = [-1] * len(seq)
        for step in agent_steps:
            if step.agent_name == "orchestrator":
                idx = orchestrator_idx
            else:
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
        res["agent_names"] = list(self.roles.keys()) + ["orchestrator"]
        return res

    # ------------------------------------------------------------------
    # Generation + accumulation helpers
    # ------------------------------------------------------------------

    async def _generate(
        self,
        engine: InferenceEngine,
        role: AgentRole,
        input_ids: list[int],
    ) -> tuple[Any, float]:
        """Run one generation step and return ``(response, gen_time)``."""
        gconfig = self._build_gconfig_for_role(role)

        # Dynamically cap max_new_tokens so input + output fits in context.
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
        return resp, gen_time

    @staticmethod
    def _accumulate(
        seq: list[int],
        logprobs: list[float],
        loss_mask: list[int],
        versions: list[int],
        resp: Any,
    ) -> None:
        """Extend running lists with a generation response."""
        input_len = len(resp.input_tokens) - len(seq)
        seq += resp.input_tokens[-input_len:] + resp.output_tokens
        logprobs += [0.0] * input_len + resp.output_logprobs
        loss_mask += [0] * input_len + [1] * resp.output_len
        versions += [-1] * input_len + resp.output_versions

    # ------------------------------------------------------------------
    # Reward computation
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
        reward = await self.async_reward_fn(
            prompt_str,
            completion_str,
            prompt_ids,
            completion_ids,
            **data,
        )
        return float(reward)

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def _log_metrics(
        self,
        team_reward: float,
        agent_steps: list[AgentStep],
        agent_gen_times: dict[str, float],
        seq_len: int,
        env: SharedEnvironment,
        data: dict[str, Any],
        num_steps: int,
    ) -> None:
        scope = workflow_context.stat_scope()
        tracker = stats_tracker.get(scope)

        orch_steps = sum(1 for s in agent_steps if s.agent_name == "orchestrator")
        specialist_steps = len(agent_steps) - orch_steps

        tracker.scalar(
            reward=team_reward,
            num_steps=num_steps,
            orchestrator_calls=orch_steps,
            specialist_calls=specialist_steps,
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

        for agent_name, gen_time in agent_gen_times.items():
            tracker.scalar(**{f"agent_{agent_name}_gen_time": gen_time})

    # ------------------------------------------------------------------
    # Transcript dumping
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
                "mode": "dynamic",
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
            logger.warning("Failed to dump dynamic workflow transcript.", exc_info=True)


def _write_file(path: str, content: str) -> None:
    with open(path, "w") as f:
        f.write(content)
