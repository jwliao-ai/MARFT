"""Unit and integration tests for the multi-agent workflow framework.

Unit tests cover the pure-logic components (AgentRole, WorkflowGraph,
SharedEnvironment, CreditAssignment). Integration tests use a mocked
InferenceEngine to test MultiAgentWorkflow.arun_episode end-to-end
without GPU or distributed infrastructure.
"""

import asyncio
import json
import os
import tempfile
from dataclasses import dataclass, field
from unittest.mock import patch

import pytest

from areal.workflow.multi_agent.agent_role import AgentRole
from areal.workflow.multi_agent.credit import AgentStep, CreditAssignment
from areal.workflow.multi_agent.environment import SharedEnvironment
from areal.workflow.multi_agent.graph import GraphNode, WorkflowGraph

# =====================================================================
# AgentRole
# =====================================================================


class TestAgentRole:
    def test_basic_creation(self):
        role = AgentRole(name="solver", system_prompt="Solve math problems.")
        assert role.name == "solver"
        assert role.system_prompt == "Solve math problems."
        assert role.max_new_tokens is None

    def test_empty_name_raises(self):
        with pytest.raises(ValueError, match="name"):
            AgentRole(name="", system_prompt="prompt")

    def test_empty_prompt_raises(self):
        with pytest.raises(ValueError, match="system_prompt"):
            AgentRole(name="solver", system_prompt="")

    def test_from_config(self):
        cfg = {
            "name": "planner",
            "system_prompt": "Plan things.",
            "max_new_tokens": 256,
        }
        role = AgentRole.from_config(cfg)
        assert role.name == "planner"
        assert role.max_new_tokens == 256

    def test_build_roles(self):
        configs = {
            "planner": {"system_prompt": "Plan."},
            "solver": {"system_prompt": "Solve.", "temperature": 0.7},
        }
        roles = AgentRole.build_roles(configs)
        assert set(roles.keys()) == {"planner", "solver"}
        assert roles["solver"].temperature == 0.7
        assert roles["planner"].name == "planner"

    def test_negative_max_new_tokens_raises(self):
        with pytest.raises(ValueError, match="max_new_tokens"):
            AgentRole(name="x", system_prompt="p", max_new_tokens=-1)

    def test_zero_max_new_tokens_raises(self):
        with pytest.raises(ValueError, match="max_new_tokens"):
            AgentRole(name="x", system_prompt="p", max_new_tokens=0)

    def test_negative_temperature_raises(self):
        with pytest.raises(ValueError, match="temperature"):
            AgentRole(name="x", system_prompt="p", temperature=-0.1)

    def test_zero_temperature_allowed(self):
        role = AgentRole(name="x", system_prompt="p", temperature=0.0)
        assert role.temperature == 0.0

    def test_top_p_out_of_range_raises(self):
        with pytest.raises(ValueError, match="top_p"):
            AgentRole(name="x", system_prompt="p", top_p=0.0)
        with pytest.raises(ValueError, match="top_p"):
            AgentRole(name="x", system_prompt="p", top_p=1.5)

    def test_top_p_valid(self):
        role = AgentRole(name="x", system_prompt="p", top_p=0.95)
        assert role.top_p == 0.95


# =====================================================================
# WorkflowGraph
# =====================================================================


class TestWorkflowGraph:
    def test_sequential_basic(self):
        graph = WorkflowGraph.sequential(["planner", "solver", "verifier"])
        layers = graph.get_execution_order()
        assert len(layers) == 3
        assert all(len(layer) == 1 for layer in layers)

    def test_sequential_single_node(self):
        graph = WorkflowGraph.sequential(["solver"])
        layers = graph.get_execution_order()
        assert len(layers) == 1

    def test_sequential_empty_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            WorkflowGraph.sequential([])

    def test_sequential_transition_messages(self):
        graph = WorkflowGraph.sequential(
            ["a", "b"], transition_messages=["msg_a", "msg_b"]
        )
        node_b = list(graph.nodes.values())[1]
        assert node_b.transition_message == "msg_b"

    def test_sequential_mismatched_messages_raises(self):
        with pytest.raises(ValueError, match="same length"):
            WorkflowGraph.sequential(["a", "b"], transition_messages=["only_one"])

    def test_parallel_detection(self):
        nodes = {
            "a": GraphNode(id="a", role_name="r1"),
            "b": GraphNode(id="b", role_name="r2"),
            "c": GraphNode(id="c", role_name="r3"),
        }
        edges = [("a", "c"), ("b", "c")]
        graph = WorkflowGraph(nodes=nodes, edges=edges)
        graph.validate()
        assert graph.has_parallel()
        layers = graph.get_execution_order()
        assert len(layers) == 2
        assert set(layers[0]) == {"a", "b"}
        assert layers[1] == ["c"]

    def test_cycle_detection(self):
        nodes = {
            "a": GraphNode(id="a", role_name="r1"),
            "b": GraphNode(id="b", role_name="r1"),
        }
        edges = [("a", "b"), ("b", "a")]
        graph = WorkflowGraph(nodes=nodes, edges=edges)
        with pytest.raises(ValueError, match="Cycle"):
            graph.validate()

    def test_unknown_edge_source(self):
        nodes = {"a": GraphNode(id="a", role_name="r1")}
        edges = [("unknown", "a")]
        graph = WorkflowGraph(nodes=nodes, edges=edges)
        with pytest.raises(ValueError, match="source"):
            graph.validate()

    def test_unknown_edge_dest(self):
        nodes = {"a": GraphNode(id="a", role_name="r1")}
        edges = [("a", "unknown")]
        graph = WorkflowGraph(nodes=nodes, edges=edges)
        with pytest.raises(ValueError, match="destination"):
            graph.validate()

    def test_empty_graph_raises(self):
        graph = WorkflowGraph(nodes={}, edges=[])
        with pytest.raises(ValueError, match="at least one node"):
            graph.validate()

    def test_from_config(self):
        config = {
            "nodes": [
                {"id": "step1", "role_name": "planner"},
                {"id": "step2", "role_name": "solver"},
            ],
            "edges": [["step1", "step2"]],
        }
        graph = WorkflowGraph.from_config(config)
        assert len(graph.nodes) == 2
        layers = graph.get_execution_order()
        assert len(layers) == 2

    def test_diamond_dag(self):
        nodes = {
            "start": GraphNode(id="start", role_name="r1"),
            "left": GraphNode(id="left", role_name="r2"),
            "right": GraphNode(id="right", role_name="r3"),
            "end": GraphNode(id="end", role_name="r4"),
        }
        edges = [
            ("start", "left"),
            ("start", "right"),
            ("left", "end"),
            ("right", "end"),
        ]
        graph = WorkflowGraph(nodes=nodes, edges=edges)
        graph.validate()
        layers = graph.get_execution_order()
        assert len(layers) == 3
        assert layers[0] == ["start"]
        assert set(layers[1]) == {"left", "right"}
        assert layers[2] == ["end"]


# =====================================================================
# SharedEnvironment
# =====================================================================


class TestSharedEnvironment:
    def test_basic_messages(self):
        env = SharedEnvironment()
        env.add_user_message("Hello")
        env.add_assistant_message("Hi", agent_name="solver")
        assert len(env.messages) == 2
        assert env.messages[1]["agent_name"] == "solver"

    def test_get_messages_for_role(self):
        role = AgentRole(name="solver", system_prompt="Be a solver.")
        env = SharedEnvironment()
        env.add_user_message("What is 2+2?")
        msgs = env.get_messages_for_role(role)
        assert len(msgs) == 2
        assert msgs[0]["role"] == "system"
        assert msgs[0]["content"] == "Be a solver."
        assert msgs[1]["role"] == "user"

    def test_snapshot_is_independent(self):
        env = SharedEnvironment()
        env.add_user_message("original")
        snap = env.snapshot()
        snap.add_user_message("branch_only")
        assert len(env.messages) == 1
        assert len(snap.messages) == 2

    def test_merge_parallel_results(self):
        env = SharedEnvironment()
        env.add_user_message("original")

        snap1 = env.snapshot()
        snap1.add_assistant_message("from branch 1", agent_name="a1")
        snap2 = env.snapshot()
        snap2.add_assistant_message("from branch 2", agent_name="a2")

        env.merge_parallel_results([snap1, snap2])
        assert len(env.messages) == 3
        assert env.messages[1]["content"] == "from branch 1"
        assert env.messages[2]["content"] == "from branch 2"

    def test_merge_uses_snapshot_base_len(self):
        env = SharedEnvironment()
        env.add_user_message("original")
        snap = env.snapshot()
        env.add_user_message("added after snapshot")
        snap.add_assistant_message("branch msg", agent_name="a1")
        env.merge_parallel_results([snap])
        assert env.messages[-1]["content"] == "branch msg"

    def test_from_data(self):
        data = {
            "messages": [{"role": "user", "content": "Solve x+1=2"}],
            "answer": "1",
        }
        env = SharedEnvironment.from_data(data)
        assert len(env.messages) == 1
        assert env.metadata["answer"] == "1"

    def test_from_data_missing_messages(self):
        env = SharedEnvironment.from_data({"answer": "42"})
        assert len(env.messages) == 0
        assert env.metadata["answer"] == "42"


# =====================================================================
# CreditAssignment
# =====================================================================


class TestCreditAssignment:
    @pytest.fixture
    def three_steps(self):
        return [
            AgentStep(agent_name="planner", step_index=0, token_start=0, token_end=10),
            AgentStep(agent_name="solver", step_index=1, token_start=10, token_end=25),
            AgentStep(
                agent_name="verifier", step_index=2, token_start=25, token_end=40
            ),
        ]

    def test_equal_strategy(self, three_steps):
        credit = CreditAssignment(strategy="equal")
        rewards = credit.compute_per_token_rewards(
            team_reward=1.0, agent_steps=three_steps, seq_len=40
        )
        assert rewards.shape == (40,)
        assert rewards[39] == 1.0
        assert rewards[9] == 0.0
        assert rewards[24] == 0.0

    def test_step_discount_strategy(self, three_steps):
        credit = CreditAssignment(strategy="step_discount", discount=0.5)
        rewards = credit.compute_per_token_rewards(
            team_reward=1.0, agent_steps=three_steps, seq_len=40
        )
        assert rewards[9] == pytest.approx(1.0 * 0.5**2)
        assert rewards[24] == pytest.approx(1.0 * 0.5**1)
        assert rewards[39] == pytest.approx(1.0 * 0.5**0)

    def test_per_step_strategy(self, three_steps):
        def step_fn(step, team_reward, env, data):
            return 0.5 if step.agent_name == "solver" else 0.1

        credit = CreditAssignment(strategy="per_step", step_reward_fn=step_fn)
        rewards = credit.compute_per_token_rewards(
            team_reward=1.0, agent_steps=three_steps, seq_len=40
        )
        assert rewards[9] == pytest.approx(0.1)
        assert rewards[24] == pytest.approx(0.5)
        assert rewards[39] == pytest.approx(0.1)

    def test_per_step_requires_fn(self):
        with pytest.raises(ValueError, match="step_reward_fn"):
            CreditAssignment(strategy="per_step")

    def test_unknown_strategy_raises(self):
        with pytest.raises(ValueError, match="Unknown"):
            CreditAssignment(strategy="unknown")

    def test_empty_steps(self):
        credit = CreditAssignment(strategy="equal")
        rewards = credit.compute_per_token_rewards(
            team_reward=1.0, agent_steps=[], seq_len=10
        )
        assert rewards.sum() == 0.0

    def test_equal_single_step(self):
        steps = [AgentStep(agent_name="solo", step_index=0, token_start=0, token_end=5)]
        credit = CreditAssignment(strategy="equal")
        rewards = credit.compute_per_token_rewards(
            team_reward=2.0, agent_steps=steps, seq_len=5
        )
        assert rewards[4] == 2.0

    def test_discount_out_of_range_raises(self):
        with pytest.raises(ValueError, match="credit_discount"):
            CreditAssignment(strategy="step_discount", discount=-0.1)
        with pytest.raises(ValueError, match="credit_discount"):
            CreditAssignment(strategy="step_discount", discount=1.5)

    def test_oob_token_end_does_not_crash(self):
        steps = [AgentStep(agent_name="a", step_index=0, token_start=0, token_end=100)]
        credit = CreditAssignment(strategy="equal")
        rewards = credit.compute_per_token_rewards(
            team_reward=1.0, agent_steps=steps, seq_len=10
        )
        assert rewards.sum() == 0.0

    def test_compute_per_agent_rewards_equal(self, three_steps):
        credit = CreditAssignment(strategy="equal")
        per_agent = credit.compute_per_agent_rewards(
            team_reward=1.0, agent_steps=three_steps, seq_len=40
        )
        assert per_agent["planner"] == 1.0
        assert per_agent["solver"] == 1.0
        assert per_agent["verifier"] == 1.0

    def test_compute_per_agent_rewards_discount(self, three_steps):
        credit = CreditAssignment(strategy="step_discount", discount=0.5)
        per_agent = credit.compute_per_agent_rewards(
            team_reward=1.0, agent_steps=three_steps, seq_len=40
        )
        assert per_agent["planner"] == pytest.approx(0.25)
        assert per_agent["solver"] == pytest.approx(0.5)
        assert per_agent["verifier"] == pytest.approx(1.0)


# =====================================================================
# Integration: Graph + Roles validation
# =====================================================================


class TestGraphRolesIntegration:
    def test_sequential_roles_match_graph(self):
        roles = AgentRole.build_roles(
            {
                "planner": {"system_prompt": "Plan."},
                "solver": {"system_prompt": "Solve."},
            }
        )
        graph = WorkflowGraph.sequential(["planner", "solver"])
        for node in graph.nodes.values():
            assert node.role_name in roles

    def test_missing_role_detected(self):
        roles = AgentRole.build_roles({"planner": {"system_prompt": "Plan."}})
        graph = WorkflowGraph.sequential(["planner", "missing_role"])
        has_missing = any(node.role_name not in roles for node in graph.nodes.values())
        assert has_missing


# =====================================================================
# Integration: MultiAgentWorkflow with mocked engine
# =====================================================================


@dataclass
class FakeGenerationHyperparameters:
    """Minimal fake gconfig for testing."""

    n_samples: int = 1
    max_new_tokens: int = 128
    temperature: float = 0.7
    top_p: float = 0.9
    lora_name: str | None = None

    def new(self, **kwargs):
        d = {
            "n_samples": self.n_samples,
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "lora_name": self.lora_name,
        }
        d.update(kwargs)
        return FakeGenerationHyperparameters(**d)

    def new_with_stop_and_pad_token_ids(self, tokenizer):
        return self


class FakeTokenizer:
    """Minimal fake tokenizer for testing."""

    eos_token_id = 2

    def apply_chat_template(
        self, messages, tokenize=True, add_generation_prompt=False, **kwargs
    ):
        tokens = [1]
        for msg in messages:
            content = msg.get("content", "")
            tokens.extend([ord(c) % 100 + 10 for c in content[:3]])
        if add_generation_prompt:
            tokens.append(99)
        return tokens

    def decode(self, token_ids):
        return f"decoded_{len(token_ids)}_tokens"


@dataclass
class FakeModelResponse:
    """Lightweight stand-in for ``areal.api.io_struct.ModelResponse``."""

    input_tokens: list[int] = field(default_factory=list)
    output_tokens: list[int] = field(default_factory=list)
    output_logprobs: list[float] = field(default_factory=list)
    output_versions: list[int] = field(default_factory=list)

    @property
    def input_len(self) -> int:
        return len(self.input_tokens)

    @property
    def output_len(self) -> int:
        return len(self.output_tokens)


class FakeEngine:
    """Mocked InferenceEngine that returns deterministic responses."""

    def __init__(self, version=1):
        self._version = version
        self._call_count = 0

    def get_version(self):
        return self._version

    async def agenerate(self, req):
        self._call_count += 1
        input_tokens = list(req.input_ids)
        output_tokens = [200 + self._call_count, 201 + self._call_count, 2]
        return FakeModelResponse(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            output_logprobs=[-0.5, -0.3, -0.1],
            output_versions=[self._version] * 3,
        )


class FailingEngine:
    """Engine that raises on the Nth call."""

    def __init__(self, fail_on_call=1, version=1):
        self._fail_on_call = fail_on_call
        self._version = version
        self._call_count = 0

    def get_version(self):
        return self._version

    async def agenerate(self, req):
        self._call_count += 1
        if self._call_count == self._fail_on_call:
            raise RuntimeError(f"Engine failure on call {self._call_count}")
        input_tokens = list(req.input_ids)
        output_tokens = [200 + self._call_count, 201 + self._call_count, 2]
        return FakeModelResponse(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            output_logprobs=[-0.5, -0.3, -0.1],
            output_versions=[self._version] * 3,
        )


def _fake_reward_fn(prompt, completions, prompt_ids, completion_ids, **data):
    """Module-level reward function (must be picklable for ProcessPoolExecutor)."""
    return 1.0 if data.get("answer") else 0.0


def _make_workflow(
    role_names=None,
    credit_strategy="equal",
    credit_discount=1.0,
    dump_dir=None,
):
    """Build a MultiAgentWorkflow for testing."""
    from areal.workflow.multi_agent.workflow import MultiAgentWorkflow

    if role_names is None:
        role_names = ["planner", "solver"]

    roles = {}
    for name in set(role_names):
        roles[name] = AgentRole(name=name, system_prompt=f"You are {name}.")

    graph = WorkflowGraph.sequential(role_names)
    tokenizer = FakeTokenizer()
    gconfig = FakeGenerationHyperparameters()

    return MultiAgentWorkflow(
        graph=graph,
        roles=roles,
        reward_fn=_fake_reward_fn,
        gconfig=gconfig,
        tokenizer=tokenizer,
        credit_strategy=credit_strategy,
        credit_discount=credit_discount,
        dump_dir=dump_dir,
    )


def _make_data():
    return {
        "messages": [{"role": "user", "content": "What is 2+2?"}],
        "answer": "4",
    }


class TestMultiAgentWorkflowIntegration:
    """Integration tests with mocked engine — no GPU required."""

    @pytest.fixture
    def workflow(self):
        return _make_workflow()

    @pytest.fixture
    def engine(self):
        return FakeEngine()

    @pytest.fixture
    def data(self):
        return _make_data()

    def test_arun_episode_returns_tensors(self, workflow, engine, data):
        with patch("areal.workflow.multi_agent.workflow.workflow_context") as mock_ctx:
            mock_ctx.stat_scope.return_value = "rollout"
            result = asyncio.get_event_loop().run_until_complete(
                workflow.arun_episode(engine, data)
            )

        assert result is not None
        assert "input_ids" in result
        assert "logprobs" in result
        assert "loss_mask" in result
        assert "versions" in result
        assert "rewards" in result
        assert "token_rewards" in result
        assert "agent_ids" in result
        assert "attention_mask" in result

        for key, tensor in result.items():
            assert tensor.dim() == 2, f"{key} should be 2D [1, seq_len]"
            assert tensor.shape[0] == 1, f"{key} batch dim should be 1"

    def test_arun_episode_seq_len_consistent(self, workflow, engine, data):
        with patch("areal.workflow.multi_agent.workflow.workflow_context") as mock_ctx:
            mock_ctx.stat_scope.return_value = "rollout"
            result = asyncio.get_event_loop().run_until_complete(
                workflow.arun_episode(engine, data)
            )

        seq_len = result["input_ids"].shape[1]
        for key, tensor in result.items():
            if key == "rewards":
                assert tensor.shape == (1, 1), "rewards should be scalar [1, 1]"
                continue
            assert tensor.shape[1] == seq_len, (
                f"{key} shape {tensor.shape} doesn't match seq_len={seq_len}"
            )

    def test_arun_episode_loss_mask_only_on_outputs(self, workflow, engine, data):
        with patch("areal.workflow.multi_agent.workflow.workflow_context") as mock_ctx:
            mock_ctx.stat_scope.return_value = "rollout"
            result = asyncio.get_event_loop().run_until_complete(
                workflow.arun_episode(engine, data)
            )

        loss_mask = result["loss_mask"][0]
        assert loss_mask.sum() > 0, "Should have some output tokens"
        assert (loss_mask >= 0).all()
        assert (loss_mask <= 1).all()

    def test_arun_episode_engine_called_per_agent(self, workflow, engine, data):
        with patch("areal.workflow.multi_agent.workflow.workflow_context") as mock_ctx:
            mock_ctx.stat_scope.return_value = "rollout"
            asyncio.get_event_loop().run_until_complete(
                workflow.arun_episode(engine, data)
            )

        assert engine._call_count == 2

    def test_arun_episode_three_agents(self, engine, data):
        workflow = _make_workflow(role_names=["planner", "solver", "verifier"])
        with patch("areal.workflow.multi_agent.workflow.workflow_context") as mock_ctx:
            mock_ctx.stat_scope.return_value = "rollout"
            result = asyncio.get_event_loop().run_until_complete(
                workflow.arun_episode(engine, data)
            )

        assert engine._call_count == 3
        assert result["input_ids"].shape[1] > 0

    def test_arun_episode_single_agent(self, engine, data):
        workflow = _make_workflow(role_names=["solver"])
        with patch("areal.workflow.multi_agent.workflow.workflow_context") as mock_ctx:
            mock_ctx.stat_scope.return_value = "rollout"
            result = asyncio.get_event_loop().run_until_complete(
                workflow.arun_episode(engine, data)
            )

        assert engine._call_count == 1
        assert result is not None

    def test_credit_strategy_step_discount(self, engine, data):
        workflow = _make_workflow(
            role_names=["planner", "solver"],
            credit_strategy="step_discount",
            credit_discount=0.5,
        )
        with patch("areal.workflow.multi_agent.workflow.workflow_context") as mock_ctx:
            mock_ctx.stat_scope.return_value = "rollout"
            result = asyncio.get_event_loop().run_until_complete(
                workflow.arun_episode(engine, data)
            )

        rewards = result["token_rewards"][0]
        nonzero = rewards[rewards != 0]
        assert len(nonzero) == 2

    def test_transcript_dumping(self, engine, data):
        with tempfile.TemporaryDirectory() as tmpdir:
            workflow = _make_workflow(dump_dir=tmpdir)
            with patch(
                "areal.workflow.multi_agent.workflow.workflow_context"
            ) as mock_ctx:
                mock_ctx.stat_scope.return_value = "rollout"
                asyncio.get_event_loop().run_until_complete(
                    workflow.arun_episode(engine, data)
                )

            version_dir = os.path.join(tmpdir, "1")
            assert os.path.isdir(version_dir)
            files = os.listdir(version_dir)
            assert len(files) == 1
            assert files[0].endswith(".json")

            with open(os.path.join(version_dir, files[0])) as f:
                transcript = json.load(f)
            assert "team_reward" in transcript
            assert "agent_steps" in transcript
            assert len(transcript["agent_steps"]) == 2


class TestMultiAgentWorkflowConstruction:
    """Test workflow construction and validation."""

    def test_unknown_role_raises(self):
        roles = {"planner": AgentRole(name="planner", system_prompt="Plan.")}
        graph = WorkflowGraph.sequential(["planner", "solver"])

        from areal.workflow.multi_agent.workflow import MultiAgentWorkflow

        with pytest.raises(ValueError, match="unknown role"):
            MultiAgentWorkflow(
                graph=graph,
                roles=roles,
                reward_fn=lambda *a, **kw: 0.0,
                gconfig=FakeGenerationHyperparameters(),
                tokenizer=FakeTokenizer(),
            )

    def test_graph_from_dict(self):
        roles = {"planner": AgentRole(name="planner", system_prompt="Plan.")}
        graph_config = {
            "nodes": [{"id": "s1", "role_name": "planner"}],
            "edges": [],
        }

        from areal.workflow.multi_agent.workflow import MultiAgentWorkflow

        wf = MultiAgentWorkflow(
            graph=graph_config,
            roles=roles,
            reward_fn=lambda *a, **kw: 0.0,
            gconfig=FakeGenerationHyperparameters(),
            tokenizer=FakeTokenizer(),
        )
        assert len(wf.graph.nodes) == 1

    def test_roles_from_dict(self):
        graph = WorkflowGraph.sequential(["solver"])
        roles_dict = {"solver": {"system_prompt": "Solve stuff."}}

        from areal.workflow.multi_agent.workflow import MultiAgentWorkflow

        wf = MultiAgentWorkflow(
            graph=graph,
            roles=roles_dict,
            reward_fn=lambda *a, **kw: 0.0,
            gconfig=FakeGenerationHyperparameters(),
            tokenizer=FakeTokenizer(),
        )
        assert isinstance(wf.roles["solver"], AgentRole)


class TestMultiAgentWorkflowGconfigOverrides:
    """Test per-role generation config overrides."""

    def test_role_overrides_applied(self):
        from areal.workflow.multi_agent.workflow import MultiAgentWorkflow

        roles = {
            "solver": AgentRole(
                name="solver",
                system_prompt="Solve.",
                max_new_tokens=256,
                temperature=0.3,
            ),
        }
        graph = WorkflowGraph.sequential(["solver"])
        wf = MultiAgentWorkflow(
            graph=graph,
            roles=roles,
            reward_fn=lambda *a, **kw: 0.0,
            gconfig=FakeGenerationHyperparameters(),
            tokenizer=FakeTokenizer(),
        )
        role = roles["solver"]
        gconfig = wf._build_gconfig_for_role(role)
        assert gconfig.max_new_tokens == 256
        assert gconfig.temperature == 0.3
        assert gconfig.n_samples == 1

    def test_no_overrides_preserves_defaults(self):
        from areal.workflow.multi_agent.workflow import MultiAgentWorkflow

        roles = {"solver": AgentRole(name="solver", system_prompt="Solve.")}
        graph = WorkflowGraph.sequential(["solver"])
        wf = MultiAgentWorkflow(
            graph=graph,
            roles=roles,
            reward_fn=lambda *a, **kw: 0.0,
            gconfig=FakeGenerationHyperparameters(temperature=0.7),
            tokenizer=FakeTokenizer(),
        )
        gconfig = wf._build_gconfig_for_role(roles["solver"])
        assert gconfig.temperature == 0.7
        assert gconfig.n_samples == 1


class TestMultiAgentWorkflowTransitions:
    """Test transition token logic."""

    def test_transition_cache_populated(self):
        from areal.workflow.multi_agent.workflow import MultiAgentWorkflow

        roles = {
            "a": AgentRole(name="a", system_prompt="A."),
            "b": AgentRole(name="b", system_prompt="B."),
        }
        graph = WorkflowGraph.sequential(
            ["a", "b"], transition_messages=[None, "Your turn solver"]
        )
        wf = MultiAgentWorkflow(
            graph=graph,
            roles=roles,
            reward_fn=lambda *a, **kw: 0.0,
            gconfig=FakeGenerationHyperparameters(),
            tokenizer=FakeTokenizer(),
        )
        assert len(wf._transition_token_cache) == 1

    def test_apply_transition_appends_eos(self):
        from areal.workflow.multi_agent.workflow import MultiAgentWorkflow

        roles = {
            "a": AgentRole(name="a", system_prompt="A.", description="does A"),
            "b": AgentRole(name="b", system_prompt="B.", description="does B"),
        }
        graph = WorkflowGraph.sequential(["a", "b"])
        wf = MultiAgentWorkflow(
            graph=graph,
            roles=roles,
            reward_fn=lambda *a, **kw: 0.0,
            gconfig=FakeGenerationHyperparameters(),
            tokenizer=FakeTokenizer(),
        )
        node = list(wf.graph.nodes.values())[1]
        role = roles["b"]
        result = wf._apply_transition([1, 50, 60], [1, 50, 60], node, role)
        assert FakeTokenizer.eos_token_id in result
        assert len(result) > 3


# =====================================================================
# Error resilience tests
# =====================================================================


class TestMultiAgentWorkflowErrorHandling:
    """Tests for graceful failure handling."""

    def test_sequential_agent_failure_returns_none(self):
        workflow = _make_workflow(role_names=["planner", "solver"])
        engine = FailingEngine(fail_on_call=1)
        data = _make_data()

        with patch("areal.workflow.multi_agent.workflow.workflow_context") as mock_ctx:
            mock_ctx.stat_scope.return_value = "rollout"
            result = asyncio.get_event_loop().run_until_complete(
                workflow.arun_episode(engine, data)
            )

        assert result is None

    def test_second_agent_failure_returns_none(self):
        workflow = _make_workflow(role_names=["planner", "solver"])
        engine = FailingEngine(fail_on_call=2)
        data = _make_data()

        with patch("areal.workflow.multi_agent.workflow.workflow_context") as mock_ctx:
            mock_ctx.stat_scope.return_value = "rollout"
            result = asyncio.get_event_loop().run_until_complete(
                workflow.arun_episode(engine, data)
            )

        assert result is None

    def test_parallel_agent_failure_returns_none(self):
        from areal.workflow.multi_agent.workflow import MultiAgentWorkflow

        roles = {
            "a": AgentRole(name="a", system_prompt="A."),
            "b": AgentRole(name="b", system_prompt="B."),
            "c": AgentRole(name="c", system_prompt="C."),
        }
        nodes = {
            "a": GraphNode(id="a", role_name="a"),
            "b": GraphNode(id="b", role_name="b"),
            "c": GraphNode(id="c", role_name="c"),
        }
        edges = [("a", "c"), ("b", "c")]
        graph = WorkflowGraph(nodes=nodes, edges=edges)

        wf = MultiAgentWorkflow(
            graph=graph,
            roles=roles,
            reward_fn=_fake_reward_fn,
            gconfig=FakeGenerationHyperparameters(),
            tokenizer=FakeTokenizer(),
        )

        engine = FailingEngine(fail_on_call=2)
        data = _make_data()

        with patch("areal.workflow.multi_agent.workflow.workflow_context") as mock_ctx:
            mock_ctx.stat_scope.return_value = "rollout"
            result = asyncio.get_event_loop().run_until_complete(
                wf.arun_episode(engine, data)
            )

        assert result is None

    def test_successful_run_not_affected_by_error_handling(self):
        workflow = _make_workflow(role_names=["planner", "solver", "verifier"])
        engine = FakeEngine()
        data = _make_data()

        with patch("areal.workflow.multi_agent.workflow.workflow_context") as mock_ctx:
            mock_ctx.stat_scope.return_value = "rollout"
            result = asyncio.get_event_loop().run_until_complete(
                workflow.arun_episode(engine, data)
            )

        assert result is not None
        assert engine._call_count == 3


# =====================================================================
# Reward computation correctness
# =====================================================================


_reward_capture: dict = {}


def _tracking_reward_fn(prompt, completions, prompt_ids, completion_ids, **data):
    """Reward fn that records what it receives into a module-level dict."""
    _reward_capture["prompt"] = prompt
    _reward_capture["completion"] = completions
    _reward_capture["prompt_ids"] = prompt_ids
    _reward_capture["completion_ids"] = completion_ids
    return 1.0


class TestMultiAgentWorkflowRewardArgs:
    """Verify that reward fn receives correct prompt/completion tokens."""

    def test_reward_receives_actual_token_ids(self):
        from areal.workflow.multi_agent.workflow import MultiAgentWorkflow

        _reward_capture.clear()

        roles = {"solver": AgentRole(name="solver", system_prompt="Solve.")}
        graph = WorkflowGraph.sequential(["solver"])
        tokenizer = FakeTokenizer()
        gconfig = FakeGenerationHyperparameters()

        wf = MultiAgentWorkflow(
            graph=graph,
            roles=roles,
            reward_fn=_tracking_reward_fn,
            gconfig=gconfig,
            tokenizer=tokenizer,
        )

        # Bypass ProcessPoolExecutor so _tracking_reward_fn runs in-process
        # and populates _reward_capture in the same memory space.
        async def _in_process_reward(*args, **kwargs):
            return _tracking_reward_fn(*args, **kwargs)

        wf.async_reward_fn = _in_process_reward

        engine = FakeEngine()
        data = _make_data()

        with patch("areal.workflow.multi_agent.workflow.workflow_context") as mock_ctx:
            mock_ctx.stat_scope.return_value = "rollout"
            result = asyncio.get_event_loop().run_until_complete(
                wf.arun_episode(engine, data)
            )

        assert result is not None
        assert isinstance(_reward_capture["prompt_ids"], list)
        assert isinstance(_reward_capture["completion_ids"], list)
        assert len(_reward_capture["prompt_ids"]) > 0
        assert len(_reward_capture["completion_ids"]) > 0
        all_ids = _reward_capture["prompt_ids"] + _reward_capture["completion_ids"]
        seq_len = result["input_ids"].shape[1]
        assert len(all_ids) == seq_len

    def test_reward_prompt_ids_match_initial_prompt(self):
        from areal.workflow.multi_agent.workflow import MultiAgentWorkflow

        _reward_capture.clear()

        roles = {"solver": AgentRole(name="solver", system_prompt="Solve.")}
        graph = WorkflowGraph.sequential(["solver"])
        tokenizer = FakeTokenizer()
        gconfig = FakeGenerationHyperparameters()

        wf = MultiAgentWorkflow(
            graph=graph,
            roles=roles,
            reward_fn=_tracking_reward_fn,
            gconfig=gconfig,
            tokenizer=tokenizer,
        )

        # Bypass ProcessPoolExecutor so _tracking_reward_fn runs in-process.
        async def _in_process_reward(*args, **kwargs):
            return _tracking_reward_fn(*args, **kwargs)

        wf.async_reward_fn = _in_process_reward

        engine = FakeEngine()
        data = _make_data()

        expected_prompt_ids = list(
            tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": "Solve."},
                    {"role": "user", "content": "What is 2+2?"},
                ],
                tokenize=True,
                add_generation_prompt=True,
            )
        )

        with patch("areal.workflow.multi_agent.workflow.workflow_context") as mock_ctx:
            mock_ctx.stat_scope.return_value = "rollout"
            asyncio.get_event_loop().run_until_complete(wf.arun_episode(engine, data))

        assert _reward_capture["prompt_ids"] == expected_prompt_ids


# =====================================================================
# Serializable kwargs construction
# =====================================================================


class TestMultiAgentWorkflowSerializable:
    """Test workflow can be constructed from serializable kwargs (dicts/strings)."""

    def test_construct_from_dicts(self):
        from areal.workflow.multi_agent.workflow import MultiAgentWorkflow

        graph_config = {
            "nodes": [
                {"id": "solver", "role_name": "solver"},
            ],
            "edges": [],
        }
        roles_config = {
            "solver": {"system_prompt": "Solve math."},
        }

        wf = MultiAgentWorkflow(
            graph=graph_config,
            roles=roles_config,
            reward_fn=_fake_reward_fn,
            gconfig=FakeGenerationHyperparameters(),
            tokenizer=FakeTokenizer(),
        )
        assert isinstance(wf.roles["solver"], AgentRole)
        assert len(wf.graph.nodes) == 1

    def test_string_reward_fn_resolved_lazily(self):
        from areal.workflow.multi_agent.workflow import MultiAgentWorkflow

        wf = MultiAgentWorkflow(
            graph=WorkflowGraph.sequential(["solver"]),
            roles={"solver": AgentRole(name="solver", system_prompt="Solve.")},
            reward_fn="areal.tests.test_multi_agent._fake_reward_fn",
            gconfig=FakeGenerationHyperparameters(),
            tokenizer=FakeTokenizer(),
        )

        assert isinstance(wf.reward_fn, str)

        engine = FakeEngine()
        data = _make_data()

        with patch("areal.workflow.multi_agent.workflow.workflow_context") as mock_ctx:
            mock_ctx.stat_scope.return_value = "rollout"
            result = asyncio.get_event_loop().run_until_complete(
                wf.arun_episode(engine, data)
            )

        assert result is not None
        assert not isinstance(wf.reward_fn, str)


# =====================================================================
# Orchestrator parsing
# =====================================================================


class TestOrchestratorParsing:
    """Tests for orchestrator output parsing utilities."""

    def test_parse_call_valid(self):
        from areal.workflow.multi_agent.orchestrator import parse_orchestrator_output

        name, done = parse_orchestrator_output(
            "Let me ask the solver. <call>solver</call>", {"solver", "planner"}
        )
        assert name == "solver"
        assert done is False

    def test_parse_call_with_whitespace(self):
        from areal.workflow.multi_agent.orchestrator import parse_orchestrator_output

        name, done = parse_orchestrator_output(
            "<call>  planner  </call>", {"solver", "planner"}
        )
        assert name == "planner"
        assert done is False

    def test_parse_done(self):
        from areal.workflow.multi_agent.orchestrator import parse_orchestrator_output

        name, done = parse_orchestrator_output("Task complete. <done/>", {"solver"})
        assert name is None
        assert done is True

    def test_parse_done_without_slash(self):
        from areal.workflow.multi_agent.orchestrator import parse_orchestrator_output

        name, done = parse_orchestrator_output("All done. <done>", {"solver"})
        assert name is None
        assert done is True

    def test_parse_done_takes_priority(self):
        from areal.workflow.multi_agent.orchestrator import parse_orchestrator_output

        name, done = parse_orchestrator_output(
            "<call>solver</call> <done/>", {"solver"}
        )
        assert name is None
        assert done is True

    def test_parse_unknown_agent(self):
        from areal.workflow.multi_agent.orchestrator import parse_orchestrator_output

        name, done = parse_orchestrator_output(
            "<call>unknown_agent</call>", {"solver", "planner"}
        )
        assert name is None
        assert done is False

    def test_parse_no_tags(self):
        from areal.workflow.multi_agent.orchestrator import parse_orchestrator_output

        name, done = parse_orchestrator_output(
            "I'm just thinking out loud.", {"solver"}
        )
        assert name is None
        assert done is False

    def test_parse_case_insensitive(self):
        from areal.workflow.multi_agent.orchestrator import parse_orchestrator_output

        name, done = parse_orchestrator_output("<CALL>solver</CALL>", {"solver"})
        assert name == "solver"

    def test_parse_hyphenated_agent_name(self):
        from areal.workflow.multi_agent.orchestrator import parse_orchestrator_output

        name, done = parse_orchestrator_output(
            "<call>code-reviewer</call>", {"code-reviewer"}
        )
        assert name == "code-reviewer"

    def test_build_prompt_includes_agents(self):
        from areal.workflow.multi_agent.orchestrator import build_orchestrator_prompt

        prompt = build_orchestrator_prompt(
            role_names=["solver", "verifier"],
            role_descriptions={"solver": "Solves math.", "verifier": "Checks answers."},
        )
        assert "solver" in prompt
        assert "verifier" in prompt
        assert "<call>" in prompt
        assert "<done/>" in prompt

    def test_build_prompt_default_descriptions(self):
        from areal.workflow.multi_agent.orchestrator import build_orchestrator_prompt

        prompt = build_orchestrator_prompt(role_names=["solver"])
        assert "solver" in prompt
        assert "Specialist agent." in prompt


# =====================================================================
# Dynamic multi-agent workflow
# =====================================================================


class ScriptedEngine:
    """Engine returning pre-scripted output texts (decoded via ScriptedTokenizer)."""

    def __init__(self, scripts: list[str], version=1):
        self._scripts = scripts
        self._version = version
        self._call_count = 0

    def get_version(self):
        return self._version

    async def agenerate(self, req):
        idx = min(self._call_count, len(self._scripts) - 1)
        text = self._scripts[idx]
        self._call_count += 1

        input_tokens = list(req.input_ids)
        output_tokens = [ord(c) for c in text] + [2]
        return FakeModelResponse(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            output_logprobs=[-0.1] * len(output_tokens),
            output_versions=[self._version] * len(output_tokens),
        )


class ScriptedTokenizer:
    """Tokenizer that round-trips ASCII via ord/chr for scripted tests."""

    eos_token_id = 2

    def apply_chat_template(
        self, messages, tokenize=True, add_generation_prompt=False, **kwargs
    ):
        tokens = [1]
        for msg in messages:
            content = msg.get("content", "")
            tokens.extend([ord(c) for c in content[:5]])
        if add_generation_prompt:
            tokens.append(99)
        return tokens

    def decode(self, token_ids):
        return "".join(chr(t) if 32 <= t < 127 else "" for t in token_ids)


def _make_dynamic_workflow(
    role_names=None,
    scripts=None,
    max_steps=10,
    orchestrator_prompt=None,
):
    from areal.workflow.multi_agent.dynamic_workflow import DynamicMultiAgentWorkflow

    if role_names is None:
        role_names = ["solver"]

    roles = {
        name: AgentRole(name=name, system_prompt=f"{name.title()} agent.")
        for name in role_names
    }

    tokenizer = ScriptedTokenizer()
    gconfig = FakeGenerationHyperparameters()

    wf = DynamicMultiAgentWorkflow(
        roles=roles,
        reward_fn=_fake_reward_fn,
        gconfig=gconfig,
        tokenizer=tokenizer,
        orchestrator_prompt=orchestrator_prompt or "You are the orchestrator.",
        max_steps=max_steps,
    )
    return wf


class TestDynamicWorkflowBasic:
    """Basic construction and configuration tests."""

    def test_construction(self):
        wf = _make_dynamic_workflow(role_names=["solver", "verifier"])
        assert "solver" in wf.roles
        assert "verifier" in wf.roles
        assert wf._orchestrator_role.name == "orchestrator"
        assert wf.max_steps == 10

    def test_auto_generated_prompt(self):
        from areal.workflow.multi_agent.dynamic_workflow import (
            DynamicMultiAgentWorkflow,
        )

        roles = {
            "solver": AgentRole(
                name="solver", system_prompt="Solve.", description="Math solver"
            ),
        }
        wf = DynamicMultiAgentWorkflow(
            roles=roles,
            reward_fn=_fake_reward_fn,
            gconfig=FakeGenerationHyperparameters(),
            tokenizer=ScriptedTokenizer(),
            max_steps=5,
        )
        assert "solver" in wf._orchestrator_role.system_prompt
        assert "Math solver" in wf._orchestrator_role.system_prompt

    def test_custom_orchestrator_prompt(self):
        wf = _make_dynamic_workflow(orchestrator_prompt="Custom orchestrator.")
        assert wf._orchestrator_role.system_prompt == "Custom orchestrator."

    def test_transition_cache_populated(self):
        wf = _make_dynamic_workflow(role_names=["solver", "verifier"])
        assert "to_solver" in wf._transition_cache
        assert "to_verifier" in wf._transition_cache
        assert "to_orchestrator" in wf._transition_cache

    def test_roles_from_dicts(self):
        from areal.workflow.multi_agent.dynamic_workflow import (
            DynamicMultiAgentWorkflow,
        )

        wf = DynamicMultiAgentWorkflow(
            roles={"solver": {"system_prompt": "Solve math."}},
            reward_fn=_fake_reward_fn,
            gconfig=FakeGenerationHyperparameters(),
            tokenizer=ScriptedTokenizer(),
            orchestrator_prompt="Orchestrate.",
        )
        assert isinstance(wf.roles["solver"], AgentRole)


class TestDynamicWorkflowExecution:
    """Integration tests for DynamicMultiAgentWorkflow.arun_episode."""

    def test_single_call_then_done(self):
        """Orchestrator calls solver once, then signals done."""
        scripts = [
            "Let me ask solver. <call>solver</call>",
            "The answer is 42.",
            "Looks correct. <done/>",
        ]
        wf = _make_dynamic_workflow(role_names=["solver"], max_steps=5)
        engine = ScriptedEngine(scripts)

        with patch(
            "areal.workflow.multi_agent.dynamic_workflow.workflow_context"
        ) as mock_ctx:
            mock_ctx.stat_scope.return_value = "rollout"
            result = asyncio.get_event_loop().run_until_complete(
                wf.arun_episode(engine, _make_data())
            )

        assert result is not None
        assert "input_ids" in result
        assert result["input_ids"].shape[0] == 1
        assert engine._call_count == 3

    def test_multiple_agents(self):
        """Orchestrator routes to planner then solver then done."""
        scripts = [
            "Planning first. <call>planner</call>",
            "Step 1: decompose. Step 2: solve.",
            "Now solve. <call>solver</call>",
            "2+2=4",
            "All done. <done/>",
        ]
        wf = _make_dynamic_workflow(role_names=["planner", "solver"], max_steps=5)
        engine = ScriptedEngine(scripts)

        with patch(
            "areal.workflow.multi_agent.dynamic_workflow.workflow_context"
        ) as mock_ctx:
            mock_ctx.stat_scope.return_value = "rollout"
            result = asyncio.get_event_loop().run_until_complete(
                wf.arun_episode(engine, _make_data())
            )

        assert result is not None
        assert engine._call_count == 5

    def test_immediate_done(self):
        """Orchestrator immediately says done without calling any agent."""
        scripts = ["This is trivial. <done/>"]
        wf = _make_dynamic_workflow(role_names=["solver"], max_steps=5)
        engine = ScriptedEngine(scripts)

        with patch(
            "areal.workflow.multi_agent.dynamic_workflow.workflow_context"
        ) as mock_ctx:
            mock_ctx.stat_scope.return_value = "rollout"
            result = asyncio.get_event_loop().run_until_complete(
                wf.arun_episode(engine, _make_data())
            )

        assert result is not None
        assert engine._call_count == 1

    def test_max_steps_terminates(self):
        """Loop terminates when max_steps is reached."""
        scripts = [
            "<call>solver</call>",
            "Working on it...",
            "<call>solver</call>",
            "Still working...",
            "<call>solver</call>",
            "More work...",
        ]
        wf = _make_dynamic_workflow(role_names=["solver"], max_steps=2)
        engine = ScriptedEngine(scripts)

        with patch(
            "areal.workflow.multi_agent.dynamic_workflow.workflow_context"
        ) as mock_ctx:
            mock_ctx.stat_scope.return_value = "rollout"
            result = asyncio.get_event_loop().run_until_complete(
                wf.arun_episode(engine, _make_data())
            )

        assert result is not None
        assert engine._call_count == 4

    def test_unparseable_output_terminates(self):
        """Unparseable orchestrator output terminates gracefully."""
        scripts = ["I have no idea what to do next."]
        wf = _make_dynamic_workflow(role_names=["solver"], max_steps=5)
        engine = ScriptedEngine(scripts)

        with patch(
            "areal.workflow.multi_agent.dynamic_workflow.workflow_context"
        ) as mock_ctx:
            mock_ctx.stat_scope.return_value = "rollout"
            result = asyncio.get_event_loop().run_until_complete(
                wf.arun_episode(engine, _make_data())
            )

        assert result is not None
        assert engine._call_count == 1

    def test_loss_mask_only_on_generated(self):
        """Loss mask should be 1 only for generated tokens."""
        scripts = ["<call>solver</call>", "Answer.", "<done/>"]
        wf = _make_dynamic_workflow(role_names=["solver"], max_steps=5)
        engine = ScriptedEngine(scripts)

        with patch(
            "areal.workflow.multi_agent.dynamic_workflow.workflow_context"
        ) as mock_ctx:
            mock_ctx.stat_scope.return_value = "rollout"
            result = asyncio.get_event_loop().run_until_complete(
                wf.arun_episode(engine, _make_data())
            )

        assert result is not None
        lm = result["loss_mask"].squeeze(0)
        assert lm.sum().item() > 0
        assert lm[0].item() == 0

    def test_seq_len_consistent(self):
        """All output tensors should have the same sequence length."""
        scripts = ["<call>solver</call>", "42", "<done/>"]
        wf = _make_dynamic_workflow(role_names=["solver"], max_steps=5)
        engine = ScriptedEngine(scripts)

        with patch(
            "areal.workflow.multi_agent.dynamic_workflow.workflow_context"
        ) as mock_ctx:
            mock_ctx.stat_scope.return_value = "rollout"
            result = asyncio.get_event_loop().run_until_complete(
                wf.arun_episode(engine, _make_data())
            )

        assert result is not None
        seq_len = result["input_ids"].shape[1]
        for key in [
            "logprobs",
            "loss_mask",
            "versions",
            "token_rewards",
            "agent_ids",
            "attention_mask",
        ]:
            assert result[key].shape[1] == seq_len, f"{key} shape mismatch"
        assert result["rewards"].shape == (1, 1), "rewards should be scalar"


class TestDynamicWorkflowErrorHandling:
    """Tests for error resilience in dynamic workflow."""

    def test_orchestrator_failure_returns_none(self):
        """Engine failure on orchestrator step returns None."""
        wf = _make_dynamic_workflow(role_names=["solver"])
        engine = FailingEngine(fail_on_call=1)

        with patch(
            "areal.workflow.multi_agent.dynamic_workflow.workflow_context"
        ) as mock_ctx:
            mock_ctx.stat_scope.return_value = "rollout"
            result = asyncio.get_event_loop().run_until_complete(
                wf.arun_episode(engine, _make_data())
            )

        assert result is None

    def test_specialist_failure_returns_none(self):
        """Engine failure on specialist step returns None."""
        wf = _make_dynamic_workflow(role_names=["solver"])

        call_count = 0
        original_scripts = ["<call>solver</call>"]
        engine = ScriptedEngine(original_scripts)

        async def _failing_agenerate(req):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise RuntimeError("Specialist engine failure")
            return await ScriptedEngine(original_scripts).agenerate(req)

        engine.agenerate = _failing_agenerate

        with patch(
            "areal.workflow.multi_agent.dynamic_workflow.workflow_context"
        ) as mock_ctx:
            mock_ctx.stat_scope.return_value = "rollout"
            result = asyncio.get_event_loop().run_until_complete(
                wf.arun_episode(engine, _make_data())
            )

        assert result is None


class TestDynamicWorkflowSerializable:
    """Test distributed-compatible construction."""

    def test_construct_from_dicts(self):
        from areal.workflow.multi_agent.dynamic_workflow import (
            DynamicMultiAgentWorkflow,
        )

        wf = DynamicMultiAgentWorkflow(
            roles={"solver": {"system_prompt": "Solve."}},
            reward_fn=_fake_reward_fn,
            gconfig=FakeGenerationHyperparameters(),
            tokenizer=ScriptedTokenizer(),
            orchestrator_prompt="You orchestrate.",
            max_steps=3,
        )
        assert isinstance(wf.roles["solver"], AgentRole)
        assert wf.max_steps == 3

    def test_string_reward_fn(self):
        from areal.workflow.multi_agent.dynamic_workflow import (
            DynamicMultiAgentWorkflow,
        )

        wf = DynamicMultiAgentWorkflow(
            roles={"solver": AgentRole(name="solver", system_prompt="Solve.")},
            reward_fn="areal.tests.test_multi_agent._fake_reward_fn",
            gconfig=FakeGenerationHyperparameters(),
            tokenizer=ScriptedTokenizer(),
            orchestrator_prompt="Orchestrate.",
        )

        assert isinstance(wf.reward_fn, str)

        scripts = ["<done/>"]
        engine = ScriptedEngine(scripts)

        with patch(
            "areal.workflow.multi_agent.dynamic_workflow.workflow_context"
        ) as mock_ctx:
            mock_ctx.stat_scope.return_value = "rollout"
            result = asyncio.get_event_loop().run_until_complete(
                wf.arun_episode(engine, _make_data())
            )

        assert result is not None
        assert not isinstance(wf.reward_fn, str)


class TestDynamicWorkflowTranscript:
    """Test transcript dumping for dynamic workflow."""

    def test_transcript_dumped(self):
        scripts = ["<call>solver</call>", "42", "<done/>"]

        with tempfile.TemporaryDirectory() as tmpdir:
            wf = _make_dynamic_workflow(role_names=["solver"], max_steps=5)
            wf.dump_dir = tmpdir
            engine = ScriptedEngine(scripts)

            with patch(
                "areal.workflow.multi_agent.dynamic_workflow.workflow_context"
            ) as mock_ctx:
                mock_ctx.stat_scope.return_value = "rollout"
                result = asyncio.get_event_loop().run_until_complete(
                    wf.arun_episode(engine, _make_data())
                )

            assert result is not None
            version_dir = os.path.join(tmpdir, "1")
            assert os.path.isdir(version_dir)
            files = os.listdir(version_dir)
            assert len(files) == 1
            with open(os.path.join(version_dir, files[0])) as f:
                transcript = json.load(f)
            assert transcript["mode"] == "dynamic"
            assert len(transcript["agent_steps"]) == 3


# =====================================================================
# Phase 7: agent_ids output + multi-adapter helpers
# =====================================================================


class TestAgentIdsStaticWorkflow:
    """Verify agent_ids tensor is present and correct in static workflow."""

    def test_agent_ids_in_result(self, workflow, engine, data):
        with patch("areal.workflow.multi_agent.workflow.workflow_context") as mock_ctx:
            mock_ctx.stat_scope.return_value = "rollout"
            result = asyncio.get_event_loop().run_until_complete(
                workflow.arun_episode(engine, data)
            )

        assert result is not None
        assert "agent_ids" in result
        assert result["agent_ids"].shape[0] == 1

    def test_agent_ids_seq_len_matches(self, workflow, engine, data):
        with patch("areal.workflow.multi_agent.workflow.workflow_context") as mock_ctx:
            mock_ctx.stat_scope.return_value = "rollout"
            result = asyncio.get_event_loop().run_until_complete(
                workflow.arun_episode(engine, data)
            )

        seq_len = result["input_ids"].shape[1]
        assert result["agent_ids"].shape[1] == seq_len

    def test_agent_ids_values(self, workflow, engine, data):
        with patch("areal.workflow.multi_agent.workflow.workflow_context") as mock_ctx:
            mock_ctx.stat_scope.return_value = "rollout"
            result = asyncio.get_event_loop().run_until_complete(
                workflow.arun_episode(engine, data)
            )

        agent_ids = result["agent_ids"][0]
        loss_mask = result["loss_mask"][0]
        assert (agent_ids >= -1).all(), "agent_ids should be >= -1"
        output_agent_ids = agent_ids[loss_mask.bool()]
        assert (output_agent_ids >= 0).all(), (
            "Output tokens should have non-negative agent_ids"
        )
        unique = output_agent_ids.unique().tolist()
        assert len(unique) >= 1, "Should have at least one agent represented"


class TestAgentIdsDynamicWorkflow:
    """Verify agent_ids tensor in dynamic workflow output."""

    def test_agent_ids_present(self):
        scripts = ["<call>solver</call>", "42", "<done/>"]
        wf = _make_dynamic_workflow(role_names=["solver"], max_steps=5)
        engine = ScriptedEngine(scripts)

        with patch(
            "areal.workflow.multi_agent.dynamic_workflow.workflow_context"
        ) as mock_ctx:
            mock_ctx.stat_scope.return_value = "rollout"
            result = asyncio.get_event_loop().run_until_complete(
                wf.arun_episode(engine, _make_data())
            )

        assert result is not None
        assert "agent_ids" in result
        seq_len = result["input_ids"].shape[1]
        assert result["agent_ids"].shape == (1, seq_len)

    def test_orchestrator_gets_distinct_index(self):
        scripts = ["<call>solver</call>", "42", "<done/>"]
        wf = _make_dynamic_workflow(role_names=["solver"], max_steps=5)
        engine = ScriptedEngine(scripts)

        with patch(
            "areal.workflow.multi_agent.dynamic_workflow.workflow_context"
        ) as mock_ctx:
            mock_ctx.stat_scope.return_value = "rollout"
            result = asyncio.get_event_loop().run_until_complete(
                wf.arun_episode(engine, _make_data())
            )

        agent_ids = result["agent_ids"][0]
        unique = agent_ids.unique().tolist()
        assert -1 in unique, "Should have prompt tokens with -1"
        non_prompt = [v for v in unique if v >= 0]
        assert len(non_prompt) >= 2, (
            "Should have at least orchestrator + solver indices"
        )


class TestMultiAdapterLossMask:
    """Test the static helper for per-adapter loss masking."""

    def test_basic_masking(self):
        import torch

        from areal.engine.multi_adapter_mixin import MultiAdapterMixin

        loss_mask = torch.tensor([[1, 1, 1, 0, 1, 1]])
        agent_ids = torch.tensor([[0, 0, 1, -1, 1, 0]])
        mask_0 = MultiAdapterMixin.get_adapter_loss_mask(loss_mask, agent_ids, 0)
        mask_1 = MultiAdapterMixin.get_adapter_loss_mask(loss_mask, agent_ids, 1)

        assert mask_0.tolist() == [[True, True, False, False, False, True]]
        assert mask_1.tolist() == [[False, False, True, False, True, False]]

    def test_no_tokens_for_adapter(self):
        import torch

        from areal.engine.multi_adapter_mixin import MultiAdapterMixin

        loss_mask = torch.tensor([[1, 1, 1]])
        agent_ids = torch.tensor([[0, 0, 0]])
        mask = MultiAdapterMixin.get_adapter_loss_mask(loss_mask, agent_ids, 1)
        assert mask.sum() == 0


class TestMultiAgentLoraConfig:
    """Test MultiAgentConfig LoRA fields."""

    def test_default_values(self):
        from areal.api.cli_args import MultiAgentConfig

        cfg = MultiAgentConfig()
        assert cfg.use_multi_lora is False
        assert cfg.lora_name_prefix == "agent"

    def test_custom_values(self):
        from areal.api.cli_args import MultiAgentConfig

        cfg = MultiAgentConfig(
            use_multi_lora=True,
            lora_name_prefix="role",
        )
        assert cfg.use_multi_lora is True
        assert cfg.lora_name_prefix == "role"


class TestCriticLoraConfig:
    """Test that critic LoRA configuration is properly supported."""

    def test_critic_config_inherits_lora_fields(self):
        from areal.api.cli_args import PPOCriticConfig

        cfg = PPOCriticConfig(
            experiment_name="test",
            trial_name="trial0",
            path="dummy",
            is_critic=True,
            use_lora=True,
            lora_rank=32,
            lora_alpha=16,
            peft_type="lora",
            target_modules=["all-linear"],
        )
        assert cfg.is_critic is True
        assert cfg.use_lora is True
        assert cfg.lora_rank == 32
        assert cfg.lora_alpha == 16
        assert cfg.peft_type == "lora"
        assert cfg.target_modules == ["all-linear"]

    def test_critic_config_lora_defaults_off(self):
        from areal.api.cli_args import PPOCriticConfig

        cfg = PPOCriticConfig(
            experiment_name="test",
            trial_name="trial0",
            path="dummy",
            is_critic=True,
        )
        assert cfg.use_lora is False

    def test_peft_config_uses_token_cls_for_critic(self):
        from peft import LoraConfig, TaskType

        from areal.api.cli_args import TrainEngineConfig

        actor_cfg = TrainEngineConfig(
            experiment_name="test",
            trial_name="trial0",
            path="dummy",
            is_critic=False,
            use_lora=True,
            lora_rank=8,
            lora_alpha=16,
            target_modules=["all-linear"],
        )
        critic_cfg = TrainEngineConfig(
            experiment_name="test",
            trial_name="trial0",
            path="dummy",
            is_critic=True,
            use_lora=True,
            lora_rank=8,
            lora_alpha=16,
            target_modules=["all-linear"],
        )

        def build_lora_config(cfg):
            target_modules = (
                "all-linear"
                if not cfg.target_modules or cfg.target_modules == ["all-linear"]
                else cfg.target_modules
            )
            task_type = TaskType.TOKEN_CLS if cfg.is_critic else TaskType.CAUSAL_LM
            kwargs = {
                "task_type": task_type,
                "r": cfg.lora_rank,
                "lora_alpha": cfg.lora_alpha,
                "target_modules": target_modules,
                "bias": "none",
            }
            if cfg.is_critic:
                kwargs["modules_to_save"] = ["score"]
            return LoraConfig(**kwargs)

        actor_lora = build_lora_config(actor_cfg)
        critic_lora = build_lora_config(critic_cfg)

        assert actor_lora.task_type == TaskType.CAUSAL_LM
        assert critic_lora.task_type == TaskType.TOKEN_CLS
        assert critic_lora.modules_to_save == ["score"]
        assert actor_lora.modules_to_save is None
        assert actor_lora.r == critic_lora.r == 8


# =====================================================================
# Independent critic configuration
# =====================================================================


class TestIndependentCriticConfig:
    """Test independent critic mode configuration and dispatch."""

    def test_multi_agent_independent_critic_default_none(self):
        from areal.api.cli_args import MultiAgentConfig

        cfg = MultiAgentConfig()
        assert cfg.independent_critic is None

    def test_multi_agent_independent_critic_lora(self):
        from areal.api.cli_args import MultiAgentConfig

        cfg = MultiAgentConfig(
            independent_critic="lora",
            use_multi_lora=True,
            role_names=["planner", "solver"],
        )
        assert cfg.independent_critic == "lora"

    def test_multi_agent_independent_critic_multi_head(self):
        from areal.api.cli_args import MultiAgentConfig

        cfg = MultiAgentConfig(
            independent_critic="multi_head",
            role_names=["planner", "solver"],
        )
        assert cfg.independent_critic == "multi_head"

    def test_critic_config_num_heads_default(self):
        from areal.api.cli_args import PPOCriticConfig

        cfg = PPOCriticConfig(
            experiment_name="test",
            trial_name="trial0",
            path="dummy",
            is_critic=True,
        )
        assert cfg.num_critic_heads == 1
        assert cfg.multi_lora_names == []

    def test_critic_config_multi_head_fields(self):
        from areal.api.cli_args import PPOCriticConfig

        cfg = PPOCriticConfig(
            experiment_name="test",
            trial_name="trial0",
            path="dummy",
            is_critic=True,
            num_critic_heads=3,
        )
        assert cfg.num_critic_heads == 3

    def test_critic_config_lora_adapter_names(self):
        from areal.api.cli_args import PPOCriticConfig

        cfg = PPOCriticConfig(
            experiment_name="test",
            trial_name="trial0",
            path="dummy",
            is_critic=True,
            multi_lora_names=["agent_planner", "agent_solver"],
        )
        assert cfg.multi_lora_names == ["agent_planner", "agent_solver"]


class TestMultiHeadCriticGather:
    """Test multi-head value gathering logic."""

    def test_gather_per_agent_ids(self):
        import torch

        num_agents = 3
        seq_len = 10
        logits = torch.randn(seq_len, num_agents)
        agent_ids = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2, 0])

        gathered = logits.gather(-1, agent_ids.unsqueeze(-1)).squeeze(-1)

        assert gathered.shape == (seq_len,)
        for i in range(seq_len):
            assert gathered[i] == logits[i, agent_ids[i]]

    def test_gather_gradient_flows_to_correct_head(self):
        import torch

        num_agents = 2
        seq_len = 4
        logits = torch.randn(seq_len, num_agents, requires_grad=True)
        agent_ids = torch.tensor([0, 1, 0, 1])

        gathered = logits.gather(-1, agent_ids.unsqueeze(-1)).squeeze(-1)
        loss = gathered.sum()
        loss.backward()

        grad = logits.grad
        assert grad is not None
        # Agent 0's head should have grad at positions 0, 2
        assert grad[0, 0] != 0
        assert grad[2, 0] != 0
        # Agent 1's head should have grad at positions 1, 3
        assert grad[1, 1] != 0
        assert grad[3, 1] != 0
        # Cross-head grads should be zero
        assert grad[0, 1] == 0
        assert grad[1, 0] == 0

    def test_gather_with_loss_mask(self):
        import torch

        num_agents = 2
        seq_len = 6
        logits = torch.randn(seq_len, num_agents, requires_grad=True)
        agent_ids = torch.tensor([0, 0, 1, 1, 0, 1])
        loss_mask = torch.tensor([0, 1, 1, 0, 1, 1], dtype=torch.float)

        gathered = logits.gather(-1, agent_ids.unsqueeze(-1)).squeeze(-1)
        loss = (gathered * loss_mask).sum()
        loss.backward()

        grad = logits.grad
        assert grad is not None
        # Masked positions should have zero grad
        assert grad[0, 0] == 0  # loss_mask[0] = 0
        assert grad[3, 1] == 0  # loss_mask[3] = 0


class TestMultiAdapterCriticSetup:
    """Test multi-adapter critic LoRA configuration."""

    def test_setup_multi_adapter_critic_creates_config(self):
        from peft import LoraConfig, TaskType

        lora_config = LoraConfig(
            task_type=TaskType.TOKEN_CLS,
            r=16,
            lora_alpha=32,
            target_modules="all-linear",
            bias="none",
            modules_to_save=["score"],
        )
        assert lora_config.task_type == TaskType.TOKEN_CLS
        assert lora_config.modules_to_save == ["score"]
        assert lora_config.r == 16

    def test_adapter_loss_mask_filtering(self):
        import torch

        from areal.engine.multi_adapter_mixin import MultiAdapterMixin

        loss_mask = torch.tensor([[1, 1, 0, 1, 1, 0]], dtype=torch.float)
        agent_ids = torch.tensor([[0, 1, 0, 1, 0, 1]])

        mask_0 = MultiAdapterMixin.get_adapter_loss_mask(loss_mask, agent_ids, 0)
        mask_1 = MultiAdapterMixin.get_adapter_loss_mask(loss_mask, agent_ids, 1)

        expected_0 = torch.tensor([[True, False, False, False, True, False]])
        expected_1 = torch.tensor([[False, True, False, True, False, False]])

        assert torch.equal(mask_0, expected_0)
        assert torch.equal(mask_1, expected_1)

    def test_adapter_loss_mask_no_overlap(self):
        import torch

        from areal.engine.multi_adapter_mixin import MultiAdapterMixin

        loss_mask = torch.ones(1, 8)
        agent_ids = torch.tensor([[0, 0, 1, 1, 2, 2, 0, 1]])

        masks = [
            MultiAdapterMixin.get_adapter_loss_mask(loss_mask, agent_ids, i)
            for i in range(3)
        ]
        combined = masks[0].long() + masks[1].long() + masks[2].long()
        assert (combined <= 1).all(), "Adapter masks should not overlap"
        assert (combined == 1).all(), "Every token should belong to exactly one adapter"


class TestTrainerCriticDispatch:
    """Test that rl_trainer dispatches to the correct critic class."""

    def test_resolve_fsdp_critic_cls_default(self):
        from areal.api.cli_args import PPOCriticConfig
        from areal.trainer.rl_trainer import PPOTrainer

        cfg = PPOCriticConfig(
            experiment_name="test",
            trial_name="trial0",
            path="dummy",
            is_critic=True,
        )
        cls = PPOTrainer._resolve_fsdp_critic_cls(cfg)
        from areal.engine.fsdp_engine import FSDPPPOCritic

        assert cls is FSDPPPOCritic

    def test_resolve_fsdp_critic_cls_multi_head(self):
        from areal.api.cli_args import PPOCriticConfig
        from areal.trainer.rl_trainer import PPOTrainer

        cfg = PPOCriticConfig(
            experiment_name="test",
            trial_name="trial0",
            path="dummy",
            is_critic=True,
            num_critic_heads=3,
        )
        cls = PPOTrainer._resolve_fsdp_critic_cls(cfg)
        from areal.engine.fsdp_engine import MultiHeadFSDPPPOCritic

        assert cls is MultiHeadFSDPPPOCritic

    def test_resolve_fsdp_critic_cls_lora(self):
        from areal.api.cli_args import PPOCriticConfig
        from areal.trainer.rl_trainer import PPOTrainer

        cfg = PPOCriticConfig(
            experiment_name="test",
            trial_name="trial0",
            path="dummy",
            is_critic=True,
            multi_lora_names=["agent_planner", "agent_solver"],
        )
        cls = PPOTrainer._resolve_fsdp_critic_cls(cfg)
        from areal.engine.fsdp_engine import MultiAdapterFSDPPPOCritic

        assert cls is MultiAdapterFSDPPPOCritic
