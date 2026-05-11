#!/usr/bin/env bash
# E2E validation for DeepCoder workflows on GPU cluster.
#
# Usage:
#   bash examples/deepcoder/run_e2e.sh single bs256      # Single-agent GRPO
#   bash examples/deepcoder/run_e2e.sh 2agent bs256      # 2-agent shared LoRA
#   bash examples/deepcoder/run_e2e.sh 2agent-peragent v2  # 2-agent per-agent LoRA
#   bash examples/deepcoder/run_e2e.sh multi-agent v2    # 3-agent shared LoRA
#   bash examples/deepcoder/run_e2e.sh all bs256         # All four sequentially
#
# The optional second argument is a trial identifier (e.g. "bs256", "v2").
# It is appended to trial_name so parallel experiments don't collide.
# If omitted, defaults to "trial0".
#
# Each run trains for a few steps with small batches to verify the full
# PPOTrainer → Workflow → Engine → CodeExecution loop works end-to-end.

set -e

MODE="${1:-all}"
TRIAL="${2:-trial0}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Hydra overrides shared by all modes
BASE_OVERRIDES="total_train_epochs=100 rollout.max_concurrent_rollouts=8 scheduler.type=local evaluator.freq_steps=100"

# Single-agent keeps smaller batch (GRPO n_samples=16 already provides exploration)
OVERRIDES_SINGLE="$BASE_OVERRIDES train_dataset.batch_size=16 valid_dataset.batch_size=16"

# Multi-agent uses larger batch to compensate for n_samples=1
OVERRIDES_MA="$BASE_OVERRIDES train_dataset.batch_size=256 valid_dataset.batch_size=256 rollout.queue_size=4096"


run_single() {
    echo "============================================="
    echo "  E2E: DeepCoder Single-Agent GRPO"
    echo "  (RLVRWorkflow + deepcoder_reward_fn)"
    echo "============================================="
    python "$SCRIPT_DIR/deepcoder_rl.py" \
        --config "$SCRIPT_DIR/deepcoder_rl.yaml" \
        $OVERRIDES_SINGLE \
        experiment_name=deepcoder-rl-solo \
        trial_name="$TRIAL" \
        2>&1 | tee "/tmp/areal_e2e_deepcoder_solo_${TRIAL}.log"
    echo ""
    echo ">>> DeepCoder single-agent E2E complete. Log: /tmp/areal_e2e_deepcoder_solo_${TRIAL}.log"
}

run_2agent() {
    echo "============================================="
    echo "  E2E: DeepCoder 2-Agent Shared LoRA"
    echo "  (planner → solver, shared adapter)"
    echo "============================================="
    python "$SCRIPT_DIR/deepcoder_ma_lora_2agent.py" \
        --config "$SCRIPT_DIR/deepcoder_ma_lora_2agent.yaml" \
        $OVERRIDES_MA \
        trial_name="$TRIAL" \
        2>&1 | tee "/tmp/areal_e2e_deepcoder_2agent_${TRIAL}.log"
    echo ""
    echo ">>> DeepCoder 2-agent shared E2E complete. Log: /tmp/areal_e2e_deepcoder_2agent_${TRIAL}.log"
}

run_2agent_peragent() {
    echo "============================================="
    echo "  E2E: DeepCoder 2-Agent Per-Agent LoRA"
    echo "  (planner → solver, independent adapters)"
    echo "============================================="
    python "$SCRIPT_DIR/deepcoder_ma_lora_2agent_peragent.py" \
        --config "$SCRIPT_DIR/deepcoder_ma_lora_2agent_peragent.yaml" \
        $OVERRIDES_MA \
        trial_name="$TRIAL" \
        2>&1 | tee "/tmp/areal_e2e_deepcoder_2agent_peragent_${TRIAL}.log"
    echo ""
    echo ">>> DeepCoder 2-agent per-agent E2E complete. Log: /tmp/areal_e2e_deepcoder_2agent_peragent_${TRIAL}.log"
}

run_multi_agent() {
    echo "============================================="
    echo "  E2E: DeepCoder Multi-Agent LoRA"
    echo "  (planner → coder → debugger)"
    echo "============================================="
    python "$SCRIPT_DIR/deepcoder_ma_lora.py" \
        --config "$SCRIPT_DIR/deepcoder_ma_lora.yaml" \
        $OVERRIDES_MA \
        trial_name="$TRIAL" \
        2>&1 | tee "/tmp/areal_e2e_deepcoder_multi_agent_${TRIAL}.log"
    echo ""
    echo ">>> DeepCoder multi-agent E2E complete. Log: /tmp/areal_e2e_deepcoder_multi_agent_${TRIAL}.log"
}

case "$MODE" in
    single)
        run_single
        ;;
    2agent)
        run_2agent
        ;;
    2agent-peragent)
        run_2agent_peragent
        ;;
    multi-agent)
        run_multi_agent
        ;;
    all)
        run_single
        echo ""
        run_2agent
        echo ""
        run_2agent_peragent
        echo ""
        run_multi_agent
        ;;
    *)
        echo "Usage: bash run_e2e.sh {single|2agent|2agent-peragent|multi-agent|all} [trial_id]"
        exit 1
        ;;
esac

echo ""
echo "============================================="
echo "  DeepCoder E2E validation complete!"
echo "============================================="
