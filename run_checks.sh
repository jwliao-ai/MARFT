#!/usr/bin/env bash
set -e

echo "=== Ruff lint ==="
ruff check areal/workflow/multi_agent/ areal/reward/multi_agent.py areal/tests/test_multi_agent.py examples/marft/ --fix || true

echo ""
echo "=== Ruff format ==="
ruff format areal/workflow/multi_agent/ areal/reward/multi_agent.py areal/tests/test_multi_agent.py examples/marft/

echo ""
echo "=== Ruff lint (re-check) ==="
ruff check areal/workflow/multi_agent/ areal/reward/multi_agent.py areal/tests/test_multi_agent.py examples/marft/

echo ""
echo "=== Import sanity check ==="
python -c "
from areal.workflow.multi_agent import (
    AgentRole, AgentStep, CreditAssignment,
    GraphNode, MultiAgentWorkflow, SharedEnvironment, WorkflowGraph,
    DynamicMultiAgentWorkflow, build_orchestrator_prompt, parse_orchestrator_output,
)
from areal.reward.multi_agent import multi_agent_math_reward_fn
print('All imports OK')
"

echo ""
echo "=== Unit tests ==="
python -m pytest areal/tests/test_multi_agent.py -v --tb=short 2>&1

echo ""
echo "=== DONE ==="
