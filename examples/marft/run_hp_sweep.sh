#!/bin/bash
# run_hp_sweep.sh — Hyperparameter exploration for MARFT
#
# Phased sweep across 4 base configurations (DeepScaleR + DeepCoder,
# 2-agent + 3-agent, Qwen2.5-3B-Instruct, shared LoRA + CTDE).
#
# Budget: ~180 experiments across 7 sub-phases (1A–3B + 4A–4D).
# All HP variations are applied via CLI overrides — no YAML files modified.
#
# Usage:
#   bash examples/marft/run_hp_sweep.sh <command> [filter]
#
# Commands:
#   phase1a [ds-2a|dc-3a|deepscaler|...]   Run Phase 1A: LR × clip grid (48)
#   phase1b [filter]                         Run Phase 1B: KL coefficient (16)
#   phase2a [filter]                         Run Phase 2A: Reward shaping (32)
#   phase2b [filter]                         Run Phase 2B: GAE parameters (16)
#   phase3a [filter]                         Run Phase 3A: Temperature (16)
#   phase3b [filter]                         Run Phase 3B: LoRA rank/alpha (24)
#   phase4a [filter]                         Run Phase 4A: Seed robustness (8)
#   phase4b [filter]                         Run Phase 4B: Cross-benchmark (8)
#   phase4c [filter]                         Run Phase 4C: Single-agent (4)
#   phase4d [filter]                         Run Phase 4D: Ablation (8)
#   list [phase]                             List experiment names
#   count [phase]                            Count experiments
#   dryrun <phase> [filter]                  Print commands without executing
#   status [phase]                           Show DONE/PENDING per experiment

set -euo pipefail

# ---------------------------------------------------------------------------
# Directory setup
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

# Source HP grid values
# shellcheck source=hp_sweep_configs.env
source "${SCRIPT_DIR}/hp_sweep_configs.env"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL_ROOT="/inspire/hdd/project/multi-agent/liaojunwei-p-liaojunwei/models"
QWEN3B_PATH="${MODEL_ROOT}/Qwen2.5-3B-Instruct"
SGLANG_MEM_FRAC="0.7"   # qwen3b needs headroom for LoRA reload

RESULTS_DIR="ecmlp_experiments/hp_sweep"
RESULTS_ENV="${RESULTS_DIR}/phase_results.env"

DRYRUN="${DRYRUN:-false}"
FORCE="${FORCE:-false}"

# ---------------------------------------------------------------------------
# Base configuration registry
#
# 4 configs: DeepScaleR 2/3-agent, DeepCoder 2/3-agent
# All use Qwen2.5-3B-Instruct, shared LoRA, CTDE critic.
# ---------------------------------------------------------------------------
BASE_CONFIGS=(ds-2a ds-3a dc-2a dc-3a)

declare -A CFG_YAML CFG_SCRIPT CFG_TRIAL CFG_BENCHMARK CFG_TB_ROOT CFG_METRIC CFG_LABEL

CFG_YAML[ds-2a]="examples/marft/deepscaler_2agent.yaml"
CFG_YAML[ds-3a]="examples/marft/deepscaler_3agent.yaml"
CFG_YAML[dc-2a]="examples/marft/deepcoder_2agent.yaml"
CFG_YAML[dc-3a]="examples/marft/deepcoder_3agent.yaml"

CFG_SCRIPT[ds-2a]="examples/marft/deepscaler_train.py"
CFG_SCRIPT[ds-3a]="examples/marft/deepscaler_train.py"
CFG_SCRIPT[dc-2a]="examples/deepcoder/deepcoder_ma_lora.py"
CFG_SCRIPT[dc-3a]="examples/deepcoder/deepcoder_ma_lora.py"

CFG_TRIAL[ds-2a]="trial2"
CFG_TRIAL[ds-3a]="trial2"
CFG_TRIAL[dc-2a]="trial1"
CFG_TRIAL[dc-3a]="trial2"

CFG_BENCHMARK[ds-2a]="deepscaler"
CFG_BENCHMARK[ds-3a]="deepscaler"
CFG_BENCHMARK[dc-2a]="deepcoder"
CFG_BENCHMARK[dc-3a]="deepcoder"

CFG_TB_ROOT[ds-2a]="ecmlp_experiments_deepscaler/tensorboard"
CFG_TB_ROOT[ds-3a]="ecmlp_experiments_deepscaler/tensorboard"
CFG_TB_ROOT[dc-2a]="ecmlp_experiments_deepcoder/tensorboard"
CFG_TB_ROOT[dc-3a]="ecmlp_experiments_deepcoder/tensorboard"

CFG_METRIC[ds-2a]="MATH-500/reward"
CFG_METRIC[ds-3a]="MATH-500/reward"
CFG_METRIC[dc-2a]="eval-rollout/reward"
CFG_METRIC[dc-3a]="eval-rollout/reward"

# Human-readable label for experiment names
CFG_LABEL[ds-2a]="deepscaler-2a"
CFG_LABEL[ds-3a]="deepscaler-3a"
CFG_LABEL[dc-2a]="deepcoder-2a"
CFG_LABEL[dc-3a]="deepcoder-3a"

# Single-agent configs for Phase 4C
declare -A SA_YAML SA_SCRIPT
SA_YAML[deepscaler]="examples/marft/deepscaler_1agent.yaml"
SA_YAML[deepcoder]="examples/marft/deepcoder_1agent.yaml"
SA_SCRIPT[deepscaler]="examples/marft/deepscaler_train.py"
SA_SCRIPT[deepcoder]="examples/deepcoder/deepcoder_rl.py"

# ---------------------------------------------------------------------------
# HP value encoding for experiment names
# ---------------------------------------------------------------------------
encode_lr() {
    # 1e-6 → lr1e6, 5e-6 → lr5e6
    echo "lr${1//-/}"
}

encode_clip() {
    case "$1" in
        0.1) echo "clip01" ;; 0.2) echo "clip02" ;; 0.3) echo "clip03" ;;
        *)   echo "clip$(echo "$1" | tr -d '.')" ;;
    esac
}

encode_kl() {
    case "$1" in
        0.0|0)  echo "kl0" ;;    0.01) echo "kl001" ;;
        0.05)   echo "kl005" ;;   0.1)  echo "kl01" ;;
        *)      echo "kl$(echo "$1" | tr -d '.')" ;;
    esac
}

encode_rs() {
    case "$1" in
        0.5) echo "rs05" ;; 1.0) echo "rs1" ;; 2.0) echo "rs2" ;; 5.0) echo "rs5" ;;
        *)   echo "rs$(echo "$1" | tr -d '.')" ;;
    esac
}

encode_rb() {
    case "$1" in
        0.0|0) echo "rb0" ;; -0.5) echo "rbn05" ;;
        *)     echo "rb$(echo "$1" | sed 's/-/n/' | tr -d '.')" ;;
    esac
}

encode_discount() {
    case "$1" in
        0.99) echo "g099" ;; 1.0) echo "g1" ;;
        *)    echo "g$(echo "$1" | tr -d '.')" ;;
    esac
}

encode_lambda() {
    case "$1" in
        0.95) echo "l095" ;; 1.0) echo "l1" ;;
        *)    echo "l$(echo "$1" | tr -d '.')" ;;
    esac
}

encode_temp() {
    case "$1" in
        0.7) echo "t07" ;; 0.8) echo "t08" ;; 1.0) echo "t1" ;; 1.2) echo "t12" ;;
        *)   echo "t$(echo "$1" | tr -d '.')" ;;
    esac
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
usage() {
    cat <<'EOF'
Usage: bash examples/marft/run_hp_sweep.sh <command> [filter]

Commands:
  phase1a [filter]   Phase 1A: LR × PPO clip grid (48 experiments)
  phase1b [filter]   Phase 1B: KL coefficient (16 experiments)
  phase2a [filter]   Phase 2A: Reward scaling/bias (32 experiments)
  phase2b [filter]   Phase 2B: GAE discount/lambda (16 experiments)
  phase3a [filter]   Phase 3A: Temperature (16 experiments)
  phase3b [filter]   Phase 3B: LoRA rank/alpha (24 experiments)
  phase4a [filter]   Phase 4A: Seed robustness (8 experiments)
  phase4b [filter]   Phase 4B: Cross-benchmark transfer (8 experiments)
  phase4c [filter]   Phase 4C: Single-agent comparison (4 experiments)
  phase4d [filter]   Phase 4D: HP group ablation (8 experiments)
  list [phase]       List experiment names
  count [phase]      Count experiments
  dryrun <phase> [filter]   Print commands without executing
  status [phase]     Show DONE/PENDING per experiment

Filters:
  ds-2a, ds-3a, dc-2a, dc-3a   Filter by base config
  deepscaler, deepcoder          Filter by benchmark

Environment variables:
  FORCE=true    Re-run completed experiments
  DRYRUN=true   Print commands without executing

Examples:
  bash examples/marft/run_hp_sweep.sh phase1a
  bash examples/marft/run_hp_sweep.sh phase1a ds-2a
  bash examples/marft/run_hp_sweep.sh phase1a deepscaler
  bash examples/marft/run_hp_sweep.sh dryrun phase1a
  bash examples/marft/run_hp_sweep.sh status phase1a
  bash examples/marft/run_hp_sweep.sh count
  FORCE=true bash examples/marft/run_hp_sweep.sh phase1a dc-3a
EOF
}

# Check if experiment completed (TensorBoard event files exist)
is_completed() {
    local exp_name=$1
    local base=$2
    local tb_root="${CFG_TB_ROOT[$base]}"
    local trial="${CFG_TRIAL[$base]}"
    local tb_dir="${tb_root}/${exp_name}/${trial}"
    [ -d "$tb_dir" ] && [ -n "$(find "$tb_dir" -maxdepth 1 -name 'events.out.tfevents.*' 2>/dev/null | head -1)" ]
}

# Filter base configs by user filter string
filter_bases() {
    local filter="${1:-}"
    if [ -z "$filter" ]; then
        echo "${BASE_CONFIGS[@]}"
        return
    fi
    local result=()
    for base in "${BASE_CONFIGS[@]}"; do
        case "$filter" in
            "$base")
                result+=("$base")
                ;;
            deepscaler)
                [[ "$base" == ds-* ]] && result+=("$base")
                ;;
            deepcoder)
                [[ "$base" == dc-* ]] && result+=("$base")
                ;;
        esac
    done
    if [ ${#result[@]} -eq 0 ]; then
        echo "ERROR: No base configs match filter '${filter}'" >&2
        echo "Valid filters: ds-2a ds-3a dc-2a dc-3a deepscaler deepcoder" >&2
        exit 1
    fi
    echo "${result[@]}"
}

# Ensure results directory exists
ensure_results_dir() {
    mkdir -p "${RESULTS_DIR}"
}

# Source phase results (best HPs from previous phases)
source_phase_results() {
    if [ -f "${RESULTS_ENV}" ]; then
        # shellcheck source=/dev/null
        source "${RESULTS_ENV}"
    fi
}

# Validate that required phase results exist for a given base config
require_phase_result() {
    local var_name=$1
    if [ -z "${!var_name:-}" ]; then
        echo "ERROR: Missing required phase result '${var_name}'." >&2
        echo "Run the prerequisite phase first, then:" >&2
        echo "  python examples/marft/extract_hp_results.py --phase <phase> --output ${RESULTS_ENV} --append" >&2
        exit 1
    fi
}

# ---------------------------------------------------------------------------
# Core dispatch: run a single HP experiment
# ---------------------------------------------------------------------------
run_hp_experiment() {
    local base=$1
    local exp_name=$2
    local overrides=$3  # space-separated Hydra CLI overrides

    # Skip completed experiments (unless FORCE=true)
    if [ "$FORCE" != "true" ] && is_completed "$exp_name" "$base"; then
        echo "SKIP (completed): ${exp_name}"
        return 0
    fi

    local yaml="${CFG_YAML[$base]}"
    local script="${CFG_SCRIPT[$base]}"

    # Base overrides: experiment name, scheduler, SGLang memory
    local base_overrides="experiment_name=${exp_name} scheduler.type=local"
    base_overrides="${base_overrides} sglang.mem_fraction_static=${SGLANG_MEM_FRAC}"

    # DeepScaleR YAMLs default to dsr-qwen1.5b; override to qwen3b.
    # Must override all four paths because run_all.sh sets them explicitly.
    if [[ "$base" == ds-* ]]; then
        base_overrides="${base_overrides} actor.path=${QWEN3B_PATH}"
        base_overrides="${base_overrides} ref.path=${QWEN3B_PATH}"
        base_overrides="${base_overrides} critic.path=${QWEN3B_PATH}"
        base_overrides="${base_overrides} sglang.model_path=${QWEN3B_PATH}"
    fi

    local all_overrides="${base_overrides} ${overrides}"

    echo "=========================================="
    echo "Running: ${exp_name}"
    echo "  Config:    ${yaml}"
    echo "  Script:    ${script}"
    echo "  Overrides: ${all_overrides}"
    echo "=========================================="

    if [ "$DRYRUN" = "true" ]; then
        echo "  [DRY RUN] python ${script} --config ${yaml} ${all_overrides}"
        return 0
    fi

    # Word-splitting on $all_overrides is intentional — each key=value is a separate arg
    # shellcheck disable=SC2086
    python "${script}" --config "${yaml}" ${all_overrides}
}

# ---------------------------------------------------------------------------
# Build the "best HP so far" overrides string for a base config.
# Called by phases 1B+ to carry forward the best HPs from earlier phases.
# ---------------------------------------------------------------------------
build_best_overrides() {
    local base=$1
    local up_to_phase=$2  # "p1a", "p1b", "p2a", "p2b", "p3a", "p3b"
    local base_upper
    base_upper=$(echo "$base" | tr '[:lower:]-' '[:upper:]_')

    source_phase_results
    local ovr=""

    # Phase 1A results: LR + clip (needed by p1b and later)
    if [[ "$up_to_phase" > "p1a" ]]; then
        local lr_var="BEST_P1A_${base_upper}_LR"
        local clip_var="BEST_P1A_${base_upper}_CLIP"
        require_phase_result "$lr_var"
        require_phase_result "$clip_var"
        ovr="actor.optimizer.lr=${!lr_var} critic.optimizer.lr=${!lr_var} actor.eps_clip=${!clip_var}"
    fi

    # Phase 1B results: KL
    if [[ "$up_to_phase" > "p1b" ]]; then
        local kl_var="BEST_P1B_${base_upper}_KL"
        require_phase_result "$kl_var"
        ovr="${ovr} actor.kl_ctl=${!kl_var}"
    fi

    # Phase 2A results: reward scaling + bias
    if [[ "$up_to_phase" > "p2a" ]]; then
        local rs_var="BEST_P2A_${base_upper}_RS"
        local rb_var="BEST_P2A_${base_upper}_RB"
        require_phase_result "$rs_var"
        require_phase_result "$rb_var"
        ovr="${ovr} actor.reward_scaling=${!rs_var} actor.reward_bias=${!rb_var}"
    fi

    # Phase 2B results: GAE
    if [[ "$up_to_phase" > "p2b" ]]; then
        local g_var="BEST_P2B_${base_upper}_DISCOUNT"
        local l_var="BEST_P2B_${base_upper}_LAMBDA"
        require_phase_result "$g_var"
        require_phase_result "$l_var"
        ovr="${ovr} actor.discount=${!g_var} actor.gae_lambda=${!l_var}"
    fi

    # Phase 3A results: temperature
    if [[ "$up_to_phase" > "p3a" ]]; then
        local t_var="BEST_P3A_${base_upper}_TEMP"
        require_phase_result "$t_var"
        ovr="${ovr} gconfig.temperature=${!t_var}"
    fi

    # Phase 3B results: LoRA rank/alpha
    if [[ "$up_to_phase" > "p3b" ]]; then
        local r_var="BEST_P3B_${base_upper}_RANK"
        local a_var="BEST_P3B_${base_upper}_ALPHA"
        require_phase_result "$r_var"
        require_phase_result "$a_var"
        ovr="${ovr} actor.lora_rank=${!r_var} actor.lora_alpha=${!a_var}"
        if [ "${!r_var}" -gt 32 ]; then
            ovr="${ovr} sglang.max_lora_rank=${!r_var}"
        fi
    fi

    echo "$ovr"
}

# Build the full "best HP" overrides for the final best config (all phases)
build_all_best_overrides() {
    local base=$1
    local base_upper
    base_upper=$(echo "$base" | tr '[:lower:]-' '[:upper:]_')

    source_phase_results
    local ovr=""

    # LR + clip (from Phase 1A)
    local lr_var="BEST_P1A_${base_upper}_LR"
    local clip_var="BEST_P1A_${base_upper}_CLIP"
    require_phase_result "$lr_var"
    require_phase_result "$clip_var"
    ovr="actor.optimizer.lr=${!lr_var} critic.optimizer.lr=${!lr_var} actor.eps_clip=${!clip_var}"

    # KL (from Phase 1B)
    local kl_var="BEST_P1B_${base_upper}_KL"
    require_phase_result "$kl_var"
    ovr="${ovr} actor.kl_ctl=${!kl_var}"

    # Reward shaping (from Phase 2A)
    local rs_var="BEST_P2A_${base_upper}_RS"
    local rb_var="BEST_P2A_${base_upper}_RB"
    require_phase_result "$rs_var"
    require_phase_result "$rb_var"
    ovr="${ovr} actor.reward_scaling=${!rs_var} actor.reward_bias=${!rb_var}"

    # GAE (from Phase 2B)
    local g_var="BEST_P2B_${base_upper}_DISCOUNT"
    local l_var="BEST_P2B_${base_upper}_LAMBDA"
    require_phase_result "$g_var"
    require_phase_result "$l_var"
    ovr="${ovr} actor.discount=${!g_var} actor.gae_lambda=${!l_var}"

    # Temperature (from Phase 3A)
    local t_var="BEST_P3A_${base_upper}_TEMP"
    require_phase_result "$t_var"
    ovr="${ovr} gconfig.temperature=${!t_var}"

    # LoRA (from Phase 3B)
    local r_var="BEST_P3B_${base_upper}_RANK"
    local a_var="BEST_P3B_${base_upper}_ALPHA"
    require_phase_result "$r_var"
    require_phase_result "$a_var"
    ovr="${ovr} actor.lora_rank=${!r_var} actor.lora_alpha=${!a_var}"
    if [ "${!r_var}" -gt 32 ]; then
        ovr="${ovr} sglang.max_lora_rank=${!r_var}"
    fi

    echo "$ovr"
}

# ---------------------------------------------------------------------------
# Phase 1A: Learning rate × PPO clip ratio (48 experiments)
# ---------------------------------------------------------------------------
run_phase1a() {
    local filter="${1:-}"
    local bases
    read -ra bases <<< "$(filter_bases "$filter")"

    echo "Phase 1A: LR × clip grid — ${#P1A_LR_VALUES[@]}×${#P1A_CLIP_VALUES[@]}×${#bases[@]} base configs"
    for base in "${bases[@]}"; do
        local label="${CFG_LABEL[$base]}"
        for lr in "${P1A_LR_VALUES[@]}"; do
            for clip in "${P1A_CLIP_VALUES[@]}"; do
                local lr_tag clip_tag
                lr_tag=$(encode_lr "$lr")
                clip_tag=$(encode_clip "$clip")
                local exp_name="hp-${label}-p1a-${lr_tag}-${clip_tag}"
                local overrides="actor.optimizer.lr=${lr} critic.optimizer.lr=${lr} actor.eps_clip=${clip}"
                run_hp_experiment "$base" "$exp_name" "$overrides"
            done
        done
    done
}

list_phase1a() {
    local filter="${1:-}"
    local bases
    read -ra bases <<< "$(filter_bases "$filter")"
    for base in "${bases[@]}"; do
        local label="${CFG_LABEL[$base]}"
        for lr in "${P1A_LR_VALUES[@]}"; do
            for clip in "${P1A_CLIP_VALUES[@]}"; do
                echo "hp-${label}-p1a-$(encode_lr "$lr")-$(encode_clip "$clip")"
            done
        done
    done
}

# ---------------------------------------------------------------------------
# Phase 1B: KL coefficient (16 experiments)
# Uses best (lr, clip) from Phase 1A per base config.
# ---------------------------------------------------------------------------
run_phase1b() {
    local filter="${1:-}"
    local bases
    read -ra bases <<< "$(filter_bases "$filter")"

    echo "Phase 1B: KL coefficient — ${#P1B_KL_VALUES[@]}×${#bases[@]} base configs"
    for base in "${bases[@]}"; do
        local label="${CFG_LABEL[$base]}"
        local best_ovr
        best_ovr=$(build_best_overrides "$base" "p1b")
        for kl in "${P1B_KL_VALUES[@]}"; do
            local kl_tag
            kl_tag=$(encode_kl "$kl")
            local exp_name="hp-${label}-p1b-${kl_tag}"
            local overrides="${best_ovr} actor.kl_ctl=${kl}"
            run_hp_experiment "$base" "$exp_name" "$overrides"
        done
    done
}

list_phase1b() {
    local filter="${1:-}"
    local bases
    read -ra bases <<< "$(filter_bases "$filter")"
    for base in "${bases[@]}"; do
        local label="${CFG_LABEL[$base]}"
        for kl in "${P1B_KL_VALUES[@]}"; do
            echo "hp-${label}-p1b-$(encode_kl "$kl")"
        done
    done
}

# ---------------------------------------------------------------------------
# Phase 2A: Reward scaling × bias (32 experiments)
# Uses best (lr, clip, kl) from Phases 1A+1B.
# ---------------------------------------------------------------------------
run_phase2a() {
    local filter="${1:-}"
    local bases
    read -ra bases <<< "$(filter_bases "$filter")"

    echo "Phase 2A: Reward scaling/bias — ${#P2A_RS_VALUES[@]}×${#P2A_RB_VALUES[@]}×${#bases[@]} base configs"
    for base in "${bases[@]}"; do
        local label="${CFG_LABEL[$base]}"
        local best_ovr
        best_ovr=$(build_best_overrides "$base" "p2a")
        for rs in "${P2A_RS_VALUES[@]}"; do
            for rb in "${P2A_RB_VALUES[@]}"; do
                local rs_tag rb_tag
                rs_tag=$(encode_rs "$rs")
                rb_tag=$(encode_rb "$rb")
                local exp_name="hp-${label}-p2a-${rs_tag}-${rb_tag}"
                local overrides="${best_ovr} actor.reward_scaling=${rs} actor.reward_bias=${rb}"
                run_hp_experiment "$base" "$exp_name" "$overrides"
            done
        done
    done
}

list_phase2a() {
    local filter="${1:-}"
    local bases
    read -ra bases <<< "$(filter_bases "$filter")"
    for base in "${bases[@]}"; do
        local label="${CFG_LABEL[$base]}"
        for rs in "${P2A_RS_VALUES[@]}"; do
            for rb in "${P2A_RB_VALUES[@]}"; do
                echo "hp-${label}-p2a-$(encode_rs "$rs")-$(encode_rb "$rb")"
            done
        done
    done
}

# ---------------------------------------------------------------------------
# Phase 2B: GAE parameters (16 experiments)
# Uses best (lr, clip, kl, reward shaping) from Phases 1+2A.
# ---------------------------------------------------------------------------
run_phase2b() {
    local filter="${1:-}"
    local bases
    read -ra bases <<< "$(filter_bases "$filter")"

    echo "Phase 2B: GAE parameters — ${#P2B_DISCOUNT_VALUES[@]}×${#P2B_LAMBDA_VALUES[@]}×${#bases[@]} base configs"
    for base in "${bases[@]}"; do
        local label="${CFG_LABEL[$base]}"
        local best_ovr
        best_ovr=$(build_best_overrides "$base" "p2b")
        for disc in "${P2B_DISCOUNT_VALUES[@]}"; do
            for lam in "${P2B_LAMBDA_VALUES[@]}"; do
                local g_tag l_tag
                g_tag=$(encode_discount "$disc")
                l_tag=$(encode_lambda "$lam")
                local exp_name="hp-${label}-p2b-${g_tag}-${l_tag}"
                local overrides="${best_ovr} actor.discount=${disc} actor.gae_lambda=${lam}"
                run_hp_experiment "$base" "$exp_name" "$overrides"
            done
        done
    done
}

list_phase2b() {
    local filter="${1:-}"
    local bases
    read -ra bases <<< "$(filter_bases "$filter")"
    for base in "${bases[@]}"; do
        local label="${CFG_LABEL[$base]}"
        for disc in "${P2B_DISCOUNT_VALUES[@]}"; do
            for lam in "${P2B_LAMBDA_VALUES[@]}"; do
                echo "hp-${label}-p2b-$(encode_discount "$disc")-$(encode_lambda "$lam")"
            done
        done
    done
}

# ---------------------------------------------------------------------------
# Phase 3A: Temperature (16 experiments)
# Uses best HPs from Phases 1+2.
# ---------------------------------------------------------------------------
run_phase3a() {
    local filter="${1:-}"
    local bases
    read -ra bases <<< "$(filter_bases "$filter")"

    echo "Phase 3A: Temperature — ${#P3A_TEMP_VALUES[@]}×${#bases[@]} base configs"
    for base in "${bases[@]}"; do
        local label="${CFG_LABEL[$base]}"
        local best_ovr
        best_ovr=$(build_best_overrides "$base" "p3a")
        for temp in "${P3A_TEMP_VALUES[@]}"; do
            local t_tag
            t_tag=$(encode_temp "$temp")
            local exp_name="hp-${label}-p3a-${t_tag}"
            local overrides="${best_ovr} gconfig.temperature=${temp}"
            run_hp_experiment "$base" "$exp_name" "$overrides"
        done
    done
}

list_phase3a() {
    local filter="${1:-}"
    local bases
    read -ra bases <<< "$(filter_bases "$filter")"
    for base in "${bases[@]}"; do
        local label="${CFG_LABEL[$base]}"
        for temp in "${P3A_TEMP_VALUES[@]}"; do
            echo "hp-${label}-p3a-$(encode_temp "$temp")"
        done
    done
}

# ---------------------------------------------------------------------------
# Phase 3B: LoRA rank/alpha (24 experiments)
# Uses best HPs from Phases 1+2+3A.
# ---------------------------------------------------------------------------
run_phase3b() {
    local filter="${1:-}"
    local bases
    read -ra bases <<< "$(filter_bases "$filter")"

    echo "Phase 3B: LoRA rank/alpha — ${#P3B_LORA_CONFIGS[@]}×${#bases[@]} base configs"
    for base in "${bases[@]}"; do
        local label="${CFG_LABEL[$base]}"
        local best_ovr
        best_ovr=$(build_best_overrides "$base" "p3b")
        for lora_cfg in "${P3B_LORA_CONFIGS[@]}"; do
            local rank alpha
            read -r rank alpha <<< "$lora_cfg"
            local exp_name="hp-${label}-p3b-r${rank}-a${alpha}"
            local overrides="${best_ovr} actor.lora_rank=${rank} actor.lora_alpha=${alpha}"
            # LoRA rank > 32 requires SGLang max_lora_rank override
            if [ "$rank" -gt 32 ]; then
                overrides="${overrides} sglang.max_lora_rank=${rank}"
            fi
            run_hp_experiment "$base" "$exp_name" "$overrides"
        done
    done
}

list_phase3b() {
    local filter="${1:-}"
    local bases
    read -ra bases <<< "$(filter_bases "$filter")"
    for base in "${bases[@]}"; do
        local label="${CFG_LABEL[$base]}"
        for lora_cfg in "${P3B_LORA_CONFIGS[@]}"; do
            local rank alpha
            read -r rank alpha <<< "$lora_cfg"
            echo "hp-${label}-p3b-r${rank}-a${alpha}"
        done
    done
}

# ---------------------------------------------------------------------------
# Phase 4A: Seed robustness (8 experiments)
# Best config per base × 2 seeds.
# ---------------------------------------------------------------------------
run_phase4a() {
    local filter="${1:-}"
    local bases
    read -ra bases <<< "$(filter_bases "$filter")"

    echo "Phase 4A: Seed robustness — ${#P4A_SEEDS[@]}×${#bases[@]} base configs"
    for base in "${bases[@]}"; do
        local label="${CFG_LABEL[$base]}"
        local best_ovr
        best_ovr=$(build_all_best_overrides "$base")
        for seed in "${P4A_SEEDS[@]}"; do
            local exp_name="hp-${label}-p4a-seed${seed}"
            local overrides="${best_ovr} seed=${seed}"
            run_hp_experiment "$base" "$exp_name" "$overrides"
        done
    done
}

list_phase4a() {
    local filter="${1:-}"
    local bases
    read -ra bases <<< "$(filter_bases "$filter")"
    for base in "${bases[@]}"; do
        local label="${CFG_LABEL[$base]}"
        for seed in "${P4A_SEEDS[@]}"; do
            echo "hp-${label}-p4a-seed${seed}"
        done
    done
}

# ---------------------------------------------------------------------------
# Phase 4B: Cross-benchmark transfer (8 experiments)
# Best DeepScaleR HPs → DeepCoder configs and vice versa.
# ---------------------------------------------------------------------------
run_phase4b() {
    local filter="${1:-}"

    echo "Phase 4B: Cross-benchmark transfer"
    # DS-2A best → DC-2A, DC-3A
    # DS-3A best → DC-2A, DC-3A
    # DC-2A best → DS-2A, DS-3A
    # DC-3A best → DS-2A, DS-3A
    local -a transfers=(
        "ds-2a:dc-2a" "ds-2a:dc-3a"
        "ds-3a:dc-2a" "ds-3a:dc-3a"
        "dc-2a:ds-2a" "dc-2a:ds-3a"
        "dc-3a:ds-2a" "dc-3a:ds-3a"
    )

    for transfer in "${transfers[@]}"; do
        local source_base="${transfer%%:*}"
        local target_base="${transfer##*:}"
        local source_label="${CFG_LABEL[$source_base]}"
        local target_label="${CFG_LABEL[$target_base]}"

        # Apply filter
        if [ -n "$filter" ]; then
            case "$filter" in
                "$source_base"|"$target_base") ;;
                deepscaler) [[ "$target_base" != ds-* ]] && [[ "$source_base" != ds-* ]] && continue ;;
                deepcoder)  [[ "$target_base" != dc-* ]] && [[ "$source_base" != dc-* ]] && continue ;;
                *) continue ;;
            esac
        fi

        # Build overrides from source base's best HPs
        local best_ovr
        best_ovr=$(build_all_best_overrides "$source_base")
        local exp_name="hp-${target_label}-p4b-from-${source_label}"
        run_hp_experiment "$target_base" "$exp_name" "$best_ovr"
    done
}

list_phase4b() {
    local -a transfers=(
        "ds-2a:dc-2a" "ds-2a:dc-3a"
        "ds-3a:dc-2a" "ds-3a:dc-3a"
        "dc-2a:ds-2a" "dc-2a:ds-3a"
        "dc-3a:ds-2a" "dc-3a:ds-3a"
    )
    for transfer in "${transfers[@]}"; do
        local source_base="${transfer%%:*}"
        local target_base="${transfer##*:}"
        echo "hp-${CFG_LABEL[$target_base]}-p4b-from-${CFG_LABEL[$source_base]}"
    done
}

# ---------------------------------------------------------------------------
# Phase 4C: Single-agent comparison (4 experiments)
# Best multi-agent HPs applied to 1-agent configs.
# ---------------------------------------------------------------------------
run_phase4c() {
    local filter="${1:-}"

    echo "Phase 4C: Single-agent comparison"
    for base in "${BASE_CONFIGS[@]}"; do
        local benchmark="${CFG_BENCHMARK[$base]}"
        local label="${CFG_LABEL[$base]}"

        if [ -n "$filter" ]; then
            case "$filter" in
                "$base"|"$benchmark") ;;
                *) continue ;;
            esac
        fi

        local yaml="${SA_YAML[$benchmark]}"
        local script="${SA_SCRIPT[$benchmark]}"
        local exp_name="hp-${benchmark}-1a-p4c-from-${label}"

        # Build best HP overrides from the multi-agent source config
        local best_ovr
        best_ovr=$(build_all_best_overrides "$base")

        # Base overrides for single-agent
        local base_overrides="experiment_name=${exp_name} scheduler.type=local"
        base_overrides="${base_overrides} sglang.mem_fraction_static=${SGLANG_MEM_FRAC}"

        if [[ "$base" == ds-* ]]; then
            base_overrides="${base_overrides} actor.path=${QWEN3B_PATH}"
            base_overrides="${base_overrides} ref.path=${QWEN3B_PATH}"
            base_overrides="${base_overrides} critic.path=${QWEN3B_PATH}"
            base_overrides="${base_overrides} sglang.model_path=${QWEN3B_PATH}"
        fi

        local all_overrides="${base_overrides} ${best_ovr}"

        # Check completion using benchmark-specific TB root
        local tb_root="${CFG_TB_ROOT[$base]}"
        local trial="${CFG_TRIAL[$base]}"
        if [ "$FORCE" != "true" ] && [ -d "${tb_root}/${exp_name}/${trial}" ]; then
            local found
            found=$(find "${tb_root}/${exp_name}/${trial}" -maxdepth 1 -name 'events.out.tfevents.*' 2>/dev/null | head -1)
            if [ -n "$found" ]; then
                echo "SKIP (completed): ${exp_name}"
                continue
            fi
        fi

        echo "=========================================="
        echo "Running: ${exp_name}"
        echo "  Config:    ${yaml}"
        echo "  Script:    ${script}"
        echo "  Overrides: ${all_overrides}"
        echo "=========================================="

        if [ "$DRYRUN" = "true" ]; then
            echo "  [DRY RUN] python ${script} --config ${yaml} ${all_overrides}"
            continue
        fi

        # shellcheck disable=SC2086
        python "${script}" --config "${yaml}" ${all_overrides}
    done
}

list_phase4c() {
    for base in "${BASE_CONFIGS[@]}"; do
        local benchmark="${CFG_BENCHMARK[$base]}"
        local label="${CFG_LABEL[$base]}"
        echo "hp-${benchmark}-1a-p4c-from-${label}"
    done
}

# ---------------------------------------------------------------------------
# Phase 4D: Ablation — reset one HP group to defaults (8 experiments)
# For each base config: reset RL group (lr+clip+kl) and reward group (rs+rb+gae).
# ---------------------------------------------------------------------------
run_phase4d() {
    local filter="${1:-}"
    local bases
    read -ra bases <<< "$(filter_bases "$filter")"

    echo "Phase 4D: HP group ablation — 2×${#bases[@]} base configs"
    for base in "${bases[@]}"; do
        local label="${CFG_LABEL[$base]}"
        local best_ovr
        best_ovr=$(build_all_best_overrides "$base")

        # Ablation 1: Reset RL group (lr, clip, kl) to defaults
        local exp_name="hp-${label}-p4d-reset-rl"
        local overrides="${best_ovr}"
        overrides="${overrides} actor.optimizer.lr=${DEFAULT_LR} critic.optimizer.lr=${DEFAULT_LR}"
        overrides="${overrides} actor.eps_clip=${DEFAULT_CLIP} actor.kl_ctl=${DEFAULT_KL}"
        run_hp_experiment "$base" "$exp_name" "$overrides"

        # Ablation 2: Reset reward group (scaling, bias, GAE) to defaults
        exp_name="hp-${label}-p4d-reset-reward"
        overrides="${best_ovr}"
        overrides="${overrides} actor.reward_scaling=${DEFAULT_RS} actor.reward_bias=${DEFAULT_RB}"
        overrides="${overrides} actor.discount=${DEFAULT_DISCOUNT} actor.gae_lambda=${DEFAULT_LAMBDA}"
        run_hp_experiment "$base" "$exp_name" "$overrides"
    done
}

list_phase4d() {
    local filter="${1:-}"
    local bases
    read -ra bases <<< "$(filter_bases "$filter")"
    for base in "${bases[@]}"; do
        local label="${CFG_LABEL[$base]}"
        echo "hp-${label}-p4d-reset-rl"
        echo "hp-${label}-p4d-reset-reward"
    done
}

# ---------------------------------------------------------------------------
# List / count / status helpers
# ---------------------------------------------------------------------------
list_all_phases() {
    list_phase1a; list_phase1b; list_phase2a; list_phase2b
    list_phase3a; list_phase3b; list_phase4a; list_phase4b
    list_phase4c; list_phase4d
}

list_phase_by_name() {
    local phase="${1:-}"
    local filter="${2:-}"
    case "$phase" in
        phase1a|p1a) list_phase1a "$filter" ;;
        phase1b|p1b) list_phase1b "$filter" ;;
        phase2a|p2a) list_phase2a "$filter" ;;
        phase2b|p2b) list_phase2b "$filter" ;;
        phase3a|p3a) list_phase3a "$filter" ;;
        phase3b|p3b) list_phase3b "$filter" ;;
        phase4a|p4a) list_phase4a "$filter" ;;
        phase4b|p4b) list_phase4b "$filter" ;;
        phase4c|p4c) list_phase4c "$filter" ;;
        phase4d|p4d) list_phase4d "$filter" ;;
        "")          list_all_phases ;;
        *)
            echo "ERROR: Unknown phase '${phase}'" >&2
            exit 1
            ;;
    esac
}

show_status() {
    local phase="${1:-}"
    local experiments
    experiments=$(list_phase_by_name "$phase")

    local done=0 pending=0
    while IFS= read -r exp_name; do
        [ -z "$exp_name" ] && continue
        # Determine which base config this experiment belongs to
        local found_base=""
        for base in "${BASE_CONFIGS[@]}"; do
            local label="${CFG_LABEL[$base]}"
            if [[ "$exp_name" == *"${label}"* ]]; then
                found_base="$base"
                break
            fi
        done

        if [ -z "$found_base" ]; then
            # Phase 4C single-agent experiments — check all TB roots
            local status="PENDING"
            for base in "${BASE_CONFIGS[@]}"; do
                if is_completed "$exp_name" "$base"; then
                    status="DONE"
                    break
                fi
            done
            printf "  %-60s %s\n" "$exp_name" "$status"
            [ "$status" = "DONE" ] && done=$((done + 1)) || pending=$((pending + 1))
            continue
        fi

        if is_completed "$exp_name" "$found_base"; then
            printf "  %-60s DONE\n" "$exp_name"
            done=$((done + 1))
        else
            printf "  %-60s PENDING\n" "$exp_name"
            pending=$((pending + 1))
        fi
    done <<< "$experiments"

    echo ""
    echo "Summary: ${done} DONE, ${pending} PENDING, $((done + pending)) total"
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
ensure_results_dir

case "${1:-}" in
    phase1a) run_phase1a "${2:-}" ;;
    phase1b) run_phase1b "${2:-}" ;;
    phase2a) run_phase2a "${2:-}" ;;
    phase2b) run_phase2b "${2:-}" ;;
    phase3a) run_phase3a "${2:-}" ;;
    phase3b) run_phase3b "${2:-}" ;;
    phase4a) run_phase4a "${2:-}" ;;
    phase4b) run_phase4b "${2:-}" ;;
    phase4c) run_phase4c "${2:-}" ;;
    phase4d) run_phase4d "${2:-}" ;;
    dryrun)
        DRYRUN=true
        case "${2:-}" in
            phase1a|phase1b|phase2a|phase2b|phase3a|phase3b|phase4a|phase4b|phase4c|phase4d)
                "run_${2}" "${3:-}"
                ;;
            "")
                echo "Usage: bash examples/marft/run_hp_sweep.sh dryrun <phase> [filter]"
                exit 1
                ;;
            *)
                echo "ERROR: Unknown phase '${2}'" >&2
                exit 1
                ;;
        esac
        ;;
    list)
        list_phase_by_name "${2:-}" "${3:-}"
        ;;
    count)
        total=$(list_phase_by_name "${2:-}" "${3:-}" | wc -l)
        echo "${total} experiments"
        ;;
    status)
        show_status "${2:-}"
        ;;
    ""|help|-h|--help)
        usage
        ;;
    *)
        echo "ERROR: Unknown command '${1}'" >&2
        usage
        exit 1
        ;;
esac
