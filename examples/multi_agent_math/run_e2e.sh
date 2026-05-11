    #!/usr/bin/env bash
# E2E validation for multi-agent workflows on GPU cluster.
#
# Usage:
#   bash examples/multi_agent_math/run_e2e.sh static        # Static DAG
#   bash examples/multi_agent_math/run_e2e.sh dynamic       # Dynamic orchestration
#   bash examples/multi_agent_math/run_e2e.sh lora          # Shared LoRA (Phase 1)
#   bash examples/multi_agent_math/run_e2e.sh lora-peragent # Per-agent LoRA (Phase 2)
#   bash examples/multi_agent_math/run_e2e.sh lora-perstep  # Per-step rewards + per-agent LoRA
#   bash examples/multi_agent_math/run_e2e.sh math-lora      # MATH dataset + shared LoRA
#   bash examples/multi_agent_math/run_e2e.sh math-lora-peragent # MATH + per-agent LoRA
#   bash examples/multi_agent_math/run_e2e.sh math-perstep  # MATH dataset + per-step rewards
#   bash examples/multi_agent_math/run_e2e.sh critic-lora   # Critic LoRA fine-tuning
#   bash examples/multi_agent_math/run_e2e.sh il-multi-head # IL: per-agent score heads
#   bash examples/multi_agent_math/run_e2e.sh il-critic-lora # IL: per-agent critic LoRA
#   bash examples/multi_agent_math/run_e2e.sh math-critic-lora # MATH + CTDE critic LoRA
#   bash examples/multi_agent_math/run_e2e.sh math-il-multi-head # MATH + IL multi-head
#   bash examples/multi_agent_math/run_e2e.sh math-il-critic-lora # MATH + IL critic LoRA
#   bash examples/multi_agent_math/run_e2e.sh single-ppo-lora # Single-agent PPO with actor+critic LoRA
#   bash examples/multi_agent_math/run_e2e.sh all           # All sequentially
#
# Each run trains for 1 epoch with small batches to verify the full
# PPOTrainer → Workflow → Engine loop works end-to-end.

set -e

MODE="${1:-all}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Hydra overrides: 1 epoch, small batches for quick validation
OVERRIDES="total_train_epochs=100 train_dataset.batch_size=256 valid_dataset.batch_size=256 rollout.queue_size=4096 rollout.max_concurrent_rollouts=8 gconfig.n_samples=1 gconfig.max_new_tokens=8192 scheduler.type=local"

run_static() {
    echo "============================================="
    echo "  E2E: Static DAG (MultiAgentWorkflow)"
    echo "============================================="
    python "$SCRIPT_DIR/gsm8k_ma_rl.py" \
        --config "$SCRIPT_DIR/gsm8k_ma_ppo.yaml" \
        $OVERRIDES \
        2>&1 | tee /tmp/areal_e2e_static.log
    echo ""
    echo ">>> Static DAG E2E complete. Log: /tmp/areal_e2e_static.log"
}

run_dynamic() {
    echo "============================================="
    echo "  E2E: Dynamic Orchestration"
    echo "  (DynamicMultiAgentWorkflow)"
    echo "============================================="
    python "$SCRIPT_DIR/gsm8k_dynamic_ma_rl.py" \
        --config "$SCRIPT_DIR/gsm8k_dynamic_ma_ppo.yaml" \
        $OVERRIDES \
        multi_agent.max_steps=3 \
        2>&1 | tee /tmp/areal_e2e_dynamic.log
    echo ""
    echo ">>> Dynamic orchestration E2E complete. Log: /tmp/areal_e2e_dynamic.log"
}

run_lora() {
    echo "============================================="
    echo "  E2E: Shared LoRA (Phase 1)"
    echo "  (shared_lora=true, same adapter all agents)"
    echo "============================================="
    python "$SCRIPT_DIR/gsm8k_ma_lora.py" \
        --config "$SCRIPT_DIR/gsm8k_ma_lora.yaml" \
        $OVERRIDES \
        2>&1 | tee /tmp/areal_e2e_lora.log
    echo ""
    echo ">>> Shared LoRA E2E complete. Log: /tmp/areal_e2e_lora.log"
}

run_lora_peragent() {
    echo "============================================="
    echo "  E2E: Per-Agent LoRA (Phase 2)"
    echo "  (shared_lora=false, independent adapters)"
    echo "============================================="
    python "$SCRIPT_DIR/gsm8k_ma_lora.py" \
        --config "$SCRIPT_DIR/gsm8k_ma_lora.yaml" \
        $OVERRIDES \
        multi_agent.shared_lora=false \
        experiment_name=gsm8k-multi-agent-lora-peragent-1438 \
        2>&1 | tee /tmp/areal_e2e_lora_peragent.log
    echo ""
    echo ">>> Per-agent LoRA E2E complete. Log: /tmp/areal_e2e_lora_peragent.log"
}

run_lora_perstep() {
    echo "============================================="
    echo "  E2E: Per-Step Rewards + Per-Agent LoRA"
    echo "  (credit_strategy=per_step, reward_sharing=false)"
    echo "============================================="
    python "$SCRIPT_DIR/gsm8k_ma_lora.py" \
        --config "$SCRIPT_DIR/gsm8k_ma_lora_perstep.yaml" \
        $OVERRIDES \
        experiment_name=gsm8k-multi-agent-lora-perstep-nccl \
        2>&1 | tee /tmp/areal_e2e_lora_perstep_nccl.log
    echo ""
    echo ">>> Per-step rewards E2E complete. Log: /tmp/areal_e2e_lora_perstep_nccl.log"
}

run_math_lora() {
    echo "============================================="
    echo "  E2E: MATH Dataset + Shared LoRA"
    echo "  (MATH dataset, shared_lora=true)"
    echo "============================================="
    python "$SCRIPT_DIR/gsm8k_ma_lora.py" \
        --config "$SCRIPT_DIR/math_ma_lora.yaml" \
        $OVERRIDES \
        experiment_name=math-multi-agent-lora-double-check \
        2>&1 | tee /tmp/areal_e2e_math_lora_trial_rollout_double_check.log
    echo ""
    echo ">>> MATH shared LoRA E2E complete. Log: /tmp/areal_e2e_math_lora_trial_rollout_double_check.log"
}

run_math_lora_peragent() {
    echo "============================================="
    echo "  E2E: MATH Dataset + Per-Agent LoRA"
    echo "  (MATH dataset, shared_lora=false)"
    echo "============================================="
    python "$SCRIPT_DIR/gsm8k_ma_lora.py" \
        --config "$SCRIPT_DIR/math_ma_lora.yaml" \
        $OVERRIDES \
        multi_agent.shared_lora=false \
        experiment_name=math-multi-agent-lora-peragent \
        2>&1 | tee /tmp/areal_e2e_math_lora_peragent.log
    echo ""
    echo ">>> MATH per-agent LoRA E2E complete. Log: /tmp/areal_e2e_math_lora_peragent.log"
}

run_math_perstep() {
    echo "============================================="
    echo "  E2E: MATH Dataset + Per-Step Rewards"
    echo "  (MATH dataset, per-agent LoRA, per_step credit)"
    echo "============================================="
    python "$SCRIPT_DIR/gsm8k_ma_lora.py" \
        --config "$SCRIPT_DIR/math_ma_lora_perstep.yaml" \
        $OVERRIDES \
        experiment_name=math-multi-agent-lora-perstep \
        2>&1 | tee /tmp/areal_e2e_math_perstep.log
    echo ""
    echo ">>> MATH per-step E2E complete. Log: /tmp/areal_e2e_math_perstep.log"
}

run_single_ppo_lora() {
    echo "============================================="
    echo "  E2E: Single-Agent PPO with Actor+Critic LoRA"
    echo "  (standard RLVR workflow, both engines LoRA)"
    echo "============================================="
    MATH_SCRIPT="$(cd "$(dirname "$0")/../math" && pwd)/gsm8k_rl.py"
    python "$MATH_SCRIPT" \
        --config "$SCRIPT_DIR/gsm8k_ppo_lora.yaml" \
        $OVERRIDES \
        2>&1 | tee /tmp/areal_e2e_single_ppo_lora.log
    echo ""
    echo ">>> Single-agent PPO LoRA E2E complete. Log: /tmp/areal_e2e_single_ppo_lora.log"
}

run_critic_lora() {
    echo "============================================="
    echo "  E2E: Critic LoRA Fine-Tuning"
    echo "  (shared LoRA for both actor and critic)"
    echo "============================================="
    python "$SCRIPT_DIR/gsm8k_ma_lora.py" \
        --config "$SCRIPT_DIR/gsm8k_ma_lora.yaml" \
        $OVERRIDES \
        critic.use_lora=true \
        experiment_name=gsm8k-multi-agent-critic-lora \
        2>&1 | tee /tmp/areal_e2e_critic_lora.log
    echo ""
    echo ">>> Critic LoRA E2E complete. Log: /tmp/areal_e2e_critic_lora.log"
}

run_il_multi_head() {
    echo "============================================="
    echo "  E2E: Independent Learning (Multi-Head)"
    echo "  (per-agent score heads, shared backbone)"
    echo "============================================="
    python "$SCRIPT_DIR/gsm8k_ma_lora.py" \
        --config "$SCRIPT_DIR/gsm8k_ma_lora.yaml" \
        $OVERRIDES \
        multi_agent.independent_critic=multi_head \
        multi_agent.shared_lora=false \
        experiment_name=gsm8k-il-multi-head \
        2>&1 | tee /tmp/areal_e2e_il_multi_head.log
    echo ""
    echo ">>> IL multi-head E2E complete. Log: /tmp/areal_e2e_il_multi_head.log"
}

run_il_critic_lora() {
    echo "============================================="
    echo "  E2E: Independent Learning (Per-Agent Critic LoRA)"
    echo "  (per-agent LoRA adapters on critic)"
    echo "============================================="
    python "$SCRIPT_DIR/gsm8k_ma_lora.py" \
        --config "$SCRIPT_DIR/gsm8k_ma_lora.yaml" \
        $OVERRIDES \
        multi_agent.independent_critic=lora \
        multi_agent.shared_lora=false \
        critic.use_lora=true \
        experiment_name=gsm8k-il-critic-lora \
        2>&1 | tee /tmp/areal_e2e_il_critic_lora.log
    echo ""
    echo ">>> IL critic LoRA E2E complete. Log: /tmp/areal_e2e_il_critic_lora.log"
}

run_math_critic_lora() {
    echo "============================================="
    echo "  E2E: MATH + CTDE no Critic LoRA"
    echo "  (MATH dataset, shared LoRA actor+critic)"
    echo "============================================="
    python "$SCRIPT_DIR/gsm8k_ma_lora.py" \
        --config "$SCRIPT_DIR/math_ma_lora.yaml" \
        $OVERRIDES \
        experiment_name=math-multi-agent-no-critic-lora \
        2>&1 | tee /tmp/areal_e2e_math_no_critic_lora.log
    echo ""
    echo ">>> MATH critic LoRA E2E complete. Log: /tmp/areal_e2e_math_no_critic_lora.log"
}

run_math_il_multi_head() {
    echo "============================================="
    echo "  E2E: MATH + Independent Learning (Multi-Head)"
    echo "  (MATH dataset, per-agent score heads)"
    echo "============================================="
    python "$SCRIPT_DIR/gsm8k_ma_lora.py" \
        --config "$SCRIPT_DIR/math_ma_lora.yaml" \
        $OVERRIDES \
        multi_agent.independent_critic=multi_head \
        multi_agent.shared_lora=false \
        experiment_name=math-il-multi-head \
        2>&1 | tee /tmp/areal_e2e_math_il_multi_head.log
    echo ""
    echo ">>> MATH IL multi-head E2E complete. Log: /tmp/areal_e2e_math_il_multi_head.log"
}

run_math_il_critic_lora() {
    echo "============================================="
    echo "  E2E: MATH + Independent Learning (Critic LoRA)"
    echo "  (MATH dataset, per-agent critic LoRA adapters)"
    echo "============================================="
    python "$SCRIPT_DIR/gsm8k_ma_lora.py" \
        --config "$SCRIPT_DIR/math_ma_lora.yaml" \
        $OVERRIDES \
        multi_agent.independent_critic=lora \
        multi_agent.shared_lora=false \
        critic.use_lora=true \
        experiment_name=math-il-critic-lora \
        2>&1 | tee /tmp/areal_e2e_math_il_critic_lora.log
    echo ""
    echo ">>> MATH IL critic LoRA E2E complete. Log: /tmp/areal_e2e_math_il_critic_lora.log"
}

case "$MODE" in
    static)
        run_static
        ;;
    dynamic)
        run_dynamic
        ;;
    lora)
        run_lora
        ;;
    lora-peragent)
        run_lora_peragent
        ;;
    lora-perstep)
        run_lora_perstep
        ;;
    math-lora)
        run_math_lora
        ;;
    math-lora-peragent)
        run_math_lora_peragent
        ;;
    math-perstep)
        run_math_perstep
        ;;
    critic-lora)
        run_critic_lora
        ;;
    il-multi-head)
        run_il_multi_head
        ;;
    il-critic-lora)
        run_il_critic_lora
        ;;
    math-critic-lora)
        run_math_critic_lora
        ;;
    math-il-multi-head)
        run_math_il_multi_head
        ;;
    math-il-critic-lora)
        run_math_il_critic_lora
        ;;
    single-ppo-lora)
        run_single_ppo_lora
        ;;
    both)
        run_static
        echo ""
        run_dynamic
        ;;
    all)
        run_static
        echo ""
        run_dynamic
        echo ""
        run_lora
        echo ""
        run_lora_peragent
        echo ""
        run_lora_perstep
        ;;
    *)
        echo "Usage: bash run_e2e.sh {static|dynamic|lora|lora-peragent|lora-perstep|math-lora|math-lora-peragent|math-perstep|critic-lora|il-multi-head|il-critic-lora|math-critic-lora|math-il-multi-head|math-il-critic-lora|single-ppo-lora|both|all}"
        exit 1
        ;;
esac

echo ""
echo "============================================="
echo "  All E2E validation complete!"
echo "============================================="
