#!/bin/bash
# run_all.sh — Comprehensive MARFT ablation experiment launcher
#
# Ablation dimensions (gsm8k / math / deepcoder / codeforces):
#   Benchmark:          gsm8k, math, deepcoder, codeforces
#   Model:              qwen1.5b, qwen3b, qwen7b, dsr-qwen1.5b, dsr-qwen7b, dsr-llama8b
#   Agent count:        1, 2, 3, 4
#   shared_lora:        shared (default), peragent          (multi-agent only)
#   independent_critic: ctde (default), multihead, criticlora  (multi-agent only)
#
# Per model:  1 single-agent + 3 counts × 2 shared × 3 critic × 2 (named+anonymous) = 37 experiments
# Per benchmark × model: 37 experiments
# Full matrix: 4 benchmarks × 6 models × 37 = 888 experiments
#
# DeepScaleR experiments (separate benchmark, same ablation matrix):
#   Train on DeepScaleR-Preview-Dataset (40 315 problems).
#   Eval is isolated per benchmark: aime_2024 (n=16), MATH-500, olympiadbench, minervamath.
#   Uses deepscaler_train.py (unified single- and multi-agent script).
#   Models: 6 (same model registry as above)
#   Per model: 1 single-agent + 3 counts × 2 shared × 3 critic = 19 experiments
#   Total deepscaler: 6 × 19 = 114 experiments
#
# Grand total: 888 + 222 = 1110 experiments
#
# Usage:
#   bash examples/marft/run_all.sh <target> [config_prefix]
#   bash examples/marft/run_all.sh all                                          # ALL 1110 experiments
#   bash examples/marft/run_all.sh gsm8k                                        # All GSM8K (6 models × 19 = 114)
#   bash examples/marft/run_all.sh gsm8k-qwen1.5b                              # All GSM8K + qwen1.5b (19)
#   bash examples/marft/run_all.sh gsm8k-qwen1.5b-2agent-shared-ctde           # One specific experiment
#   bash examples/marft/run_all.sh deepscaler                                   # All 114 DeepScaleR experiments
#   bash examples/marft/run_all.sh deepscaler-dsr-qwen1.5b                     # All 19 for one model
#   bash examples/marft/run_all.sh deepscaler-dsr-qwen1.5b-1agent              # Single-agent DeepScaleR
#   bash examples/marft/run_all.sh deepscaler-dsr-qwen1.5b-2agent-shared-ctde  # One multi-agent DeepScaleR
#   bash examples/marft/run_all.sh deepscaler-dsr-qwen1.5b-2agent-anonymous-shared-ctde  # Anonymous (no roles)
#   bash examples/marft/run_all.sh deepscaler-dsr-qwen1.5b-anonymous           # All 18 anonymous for one model
#   bash examples/marft/run_all.sh list                                         # List all experiment names
#   bash examples/marft/run_all.sh list gsm8k                                   # List GSM8K experiments
#   bash examples/marft/run_all.sh list gsm8k-qwen7b                            # List GSM8K+qwen7b experiments
#   bash examples/marft/run_all.sh list deepscaler                              # List DeepScaleR experiments
#   bash examples/marft/run_all.sh count                                         # Count total experiments
#
# Config prefix (optional second argument):
#   Prepends "{prefix}_" to the YAML config filename.
#   e.g., "retry_unconstrained" uses retry_unconstrained_deepcoder_2agent.yaml
#   bash examples/marft/run_all.sh deepcoder-qwen3b-2agent-shared-ctde retry_unconstrained

set -euo pipefail

# ---------------------------------------------------------------------------
# Directory setup
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL_ROOT="/workspace/models"

# Config prefix: when set, YAML config files are loaded as {prefix}_{benchmark}_{N}agent.yaml
# Populated from the optional second CLI argument in the Main section below.
CONFIG_PREFIX=""

BENCHMARKS=(gsm8k math deepcoder codeforces)
DEEPSCALER_MODELS=(qwen1.5b qwen3b qwen7b dsr-qwen1.5b dsr-qwen7b dsr-llama8b)
MODELS=(qwen1.5b qwen3b qwen7b dsr-qwen1.5b dsr-qwen7b dsr-llama8b)
MULTI_AGENT_COUNTS=(2 3 4)
SHARED_MODES=(shared peragent)
CRITIC_MODES=(ctde multihead criticlora)

# ---------------------------------------------------------------------------
# Model registry: short name → HuggingFace directory name
# ---------------------------------------------------------------------------
declare -A MODEL_PATHS
MODEL_PATHS=(
    [qwen1.5b]="Qwen2.5-1.5B-Instruct"
    [qwen3b]="Qwen2.5-3B-Instruct"
    [qwen7b]="Qwen2.5-7B-Instruct"
    [dsr-qwen1.5b]="DeepSeek-R1-Distill-Qwen-1.5B"
    [dsr-qwen7b]="DeepSeek-R1-Distill-Qwen-7B"
    [dsr-llama8b]="DeepSeek-R1-Distill-Llama-8B"
)

# ---------------------------------------------------------------------------
# Helper: resolve model short name to full path
# ---------------------------------------------------------------------------
resolve_model_path() {
    local model_name=$1
    local dir_name="${MODEL_PATHS[$model_name]:-}"
    if [ -z "$dir_name" ]; then
        echo "ERROR: Unknown model '${model_name}'. Valid: ${!MODEL_PATHS[*]}" >&2
        return 1
    fi
    echo "${MODEL_ROOT}/${dir_name}"
}

# ---------------------------------------------------------------------------
# Helper: SGLang memory fraction by model
#
# DSR (reasoning) models fill sequences to near-maximal context length,
# causing the KV cache to be very large during weight_update.  Reducing
# mem_fraction_static for these models leaves more headroom for weight
# loading and prevents SGLang OOM crashes.
#
# Qwen-Instruct (non-reasoning) models generate short outputs and work
# fine with the default 0.8, except for larger (3B/7B) models where LoRA
# hot-swapping requires more headroom — 0.7 prevents illegal memory access.
# ---------------------------------------------------------------------------
get_sglang_mem_fraction() {
    local model=$1
    case "$model" in
        qwen7b|qwen3b)            echo "0.7" ;;   # larger models need more headroom for LoRA reload
        dsr-qwen1.5b|dsr-qwen7b)  echo "0.6" ;;
        dsr-llama8b)               echo "0.55" ;;  # larger model needs more headroom
        *)                         echo "0.8" ;;   # qwen1.5b and other small models
    esac
}

# ---------------------------------------------------------------------------
# Helper: print usage
# ---------------------------------------------------------------------------
usage() {
    cat <<EOF
Usage: bash examples/marft/run_all.sh <target> [config_prefix]

Targets:
  all                                       Run ALL 456 experiments
  <benchmark>                               Run all models for a benchmark (114 each)
  <benchmark>-<model>                       Run all 19 experiments for benchmark+model
  <benchmark>-<model>-1agent                Run single-agent experiment
  <benchmark>-<model>-{N}agent-{s}-{c}      Run specific multi-agent experiment
  deepscaler                                Run all 114 DeepScaleR experiments
  deepscaler-<model>                        Run all 19 DeepScaleR experiments for a model
  deepscaler-<model>-1agent                 Run single-agent DeepScaleR for a model
  deepscaler-<model>-{N}agent-{s}-{c}       Run specific multi-agent DeepScaleR experiment
  deepscaler-<model>-{N}agent-anonymous-{s}-{c}  Run anonymous (no roles) multi-agent experiment
  deepscaler-<model>-anonymous              Run all 18 anonymous multi-agent experiments for a model
  list [filter]                             List experiment names (optional filter)
  count [filter]                            Count experiments (optional filter)
  help | -h | --help                        Show this message

Benchmarks:  gsm8k, math, deepcoder, codeforces, deepscaler
Models:      qwen1.5b, qwen3b, qwen7b, dsr-qwen1.5b, dsr-qwen7b, dsr-llama8b
Shared:      shared, peragent
Critic:      ctde, multihead, criticlora

Examples:
  bash examples/marft/run_all.sh gsm8k-qwen1.5b-1agent
  bash examples/marft/run_all.sh math-dsr-qwen7b-3agent-shared-ctde
  bash examples/marft/run_all.sh deepcoder-qwen7b-2agent-peragent-criticlora
  bash examples/marft/run_all.sh deepscaler-dsr-qwen1.5b
  bash examples/marft/run_all.sh deepscaler-dsr-qwen1.5b-1agent
  bash examples/marft/run_all.sh deepscaler-dsr-qwen1.5b-3agent-shared-ctde
  bash examples/marft/run_all.sh deepscaler-dsr-qwen1.5b-3agent-anonymous-shared-ctde
  bash examples/marft/run_all.sh deepscaler-dsr-qwen1.5b-anonymous
  bash examples/marft/run_all.sh deepscaler
  bash examples/marft/run_all.sh list gsm8k-qwen7b
  bash examples/marft/run_all.sh list deepscaler
  bash examples/marft/run_all.sh gsm8k-dsr-qwen1.5b
  bash examples/marft/run_all.sh deepcoder-qwen3b-2agent-shared-ctde retry_unconstrained
EOF
}

# ---------------------------------------------------------------------------
# Determine entry script: benchmark + single/multi → script path
# ---------------------------------------------------------------------------
get_script() {
    local benchmark=$1
    local is_multi=$2  # "single" or "multi"

    if [ "$is_multi" = "single" ]; then
        case "$benchmark" in
            gsm8k|math)   echo "examples/math/gsm8k_rl.py" ;;
            deepcoder)    echo "examples/deepcoder/deepcoder_rl.py" ;;
            codeforces)   echo "examples/marft/codeforces_train.py" ;;
        esac
    else
        case "$benchmark" in
            gsm8k|math)   echo "examples/multi_agent_math/gsm8k_ma_lora.py" ;;
            deepcoder)    echo "examples/deepcoder/deepcoder_ma_lora.py" ;;
            codeforces)   echo "examples/marft/codeforces_train.py" ;;
        esac
    fi
}

# ---------------------------------------------------------------------------
# Run a single-agent experiment
# ---------------------------------------------------------------------------
run_single_agent() {
    local benchmark=$1
    local model=$2

    local exp_name="${benchmark}-${model}-1agent"
    local config="examples/marft/${CONFIG_PREFIX}${benchmark}_1agent.yaml"
    local script
    script=$(get_script "$benchmark" "single")
    local model_path
    model_path=$(resolve_model_path "$model")
    local mem_frac
    mem_frac=$(get_sglang_mem_fraction "$model")

    echo "=========================================="
    echo "Running: ${exp_name}"
    echo "  Script:          ${script}"
    echo "  Config:          ${config}"
    echo "  Model:           ${model_path}"
    echo "  mem_frac_static: ${mem_frac}"
    echo "=========================================="
    python "$script" --config "$config" \
        experiment_name="${exp_name}" \
        scheduler.type=local \
        actor.path="${model_path}" \
        sglang.mem_fraction_static="${mem_frac}" \
        # recover.mode=auto
}

# ---------------------------------------------------------------------------
# Run a multi-agent experiment (named or anonymous)
# ---------------------------------------------------------------------------
run_multi_agent() {
    local benchmark=$1
    local model=$2
    local n_agents=$3
    local shared_mode=$4
    local critic_mode=$5
    local anonymous="${6:-}"  # "anonymous" or empty

    local anon_tag=""
    local config_suffix=""
    if [ "$anonymous" = "anonymous" ]; then
        anon_tag="-anonymous"
        config_suffix="_anonymous"
    fi

    local exp_name="${benchmark}-${model}-${n_agents}agent${anon_tag}-${shared_mode}-${critic_mode}"
    local config="examples/marft/${CONFIG_PREFIX}${benchmark}_${n_agents}agent${config_suffix}.yaml"
    local script
    script=$(get_script "$benchmark" "multi")
    local model_path
    model_path=$(resolve_model_path "$model")
    local mem_frac
    mem_frac=$(get_sglang_mem_fraction "$model")

    # Build CLI overrides
    local overrides="experiment_name=${exp_name} scheduler.type=local actor.path=${model_path}"
    overrides="$overrides sglang.mem_fraction_static=${mem_frac}"

    # shared_lora override (configs default to shared_lora=true)
    if [ "$shared_mode" = "peragent" ]; then
        overrides="$overrides multi_agent.shared_lora=false"
    fi

    # independent_critic override (configs default to independent_critic=null)
    case "$critic_mode" in
        ctde)        ;;  # default, no override needed
        multihead)   overrides="$overrides multi_agent.independent_critic=multi_head" ;;
        criticlora)  overrides="$overrides multi_agent.independent_critic=lora" ;;
        *)
            echo "ERROR: Unknown critic_mode '${critic_mode}'" >&2
            return 1
            ;;
    esac

    # Memory-intensive combos: 4-agent + peragent + criticlora needs gradient
    # accumulation to avoid OOM during critic ppo_update backward pass.
    if [ "$n_agents" -ge 4 ] && [ "$shared_mode" = "peragent" ] && [ "$critic_mode" = "criticlora" ]; then
        overrides="$overrides actor.ppo_n_minibatches=2 critic.ppo_n_minibatches=2"
    fi

    # Enable recovery for all experiments so crashes are resumable
    # overrides="$overrides recover.mode=auto"

    echo "=========================================="
    echo "Running: ${exp_name}"
    echo "  Script:    ${script}"
    echo "  Config:    ${config}"
    echo "  Model:     ${model_path}"
    echo "  Overrides: ${overrides}"
    echo "=========================================="
    # Word-splitting on $overrides is intentional — each key=value is a separate arg
    # shellcheck disable=SC2086
    python "$script" --config "$config" $overrides
}

# ---------------------------------------------------------------------------
# Run all 19 named + 18 anonymous = 37 experiments for one benchmark + model
# ---------------------------------------------------------------------------
run_benchmark_model() {
    local benchmark=$1
    local model=$2

    # Single agent (1 experiment)
    run_single_agent "$benchmark" "$model"

    # Named multi-agent ablations (18 experiments)
    for n_agents in "${MULTI_AGENT_COUNTS[@]}"; do
        for shared_mode in "${SHARED_MODES[@]}"; do
            for critic_mode in "${CRITIC_MODES[@]}"; do
                run_multi_agent "$benchmark" "$model" "$n_agents" "$shared_mode" "$critic_mode"
            done
        done
    done

    # Anonymous multi-agent ablations (18 experiments)
    for n_agents in "${MULTI_AGENT_COUNTS[@]}"; do
        for shared_mode in "${SHARED_MODES[@]}"; do
            for critic_mode in "${CRITIC_MODES[@]}"; do
                run_multi_agent "$benchmark" "$model" "$n_agents" "$shared_mode" "$critic_mode" "anonymous"
            done
        done
    done
}

# ---------------------------------------------------------------------------
# Run all models for one benchmark
# ---------------------------------------------------------------------------
run_benchmark() {
    local benchmark=$1
    for model in "${MODELS[@]}"; do
        run_benchmark_model "$benchmark" "$model"
    done
}

# ---------------------------------------------------------------------------
# DeepScaleR: single-agent + full multi-agent ablation, isolated multi-benchmark eval
# ---------------------------------------------------------------------------
# All DeepScaleR experiments use deepscaler_train.py which evaluates on four
# benchmarks independently (aime_2024 with n_samples=16, MATH-500, olympiadbench,
# minervamath).  The script auto-detects single- vs multi-agent mode from the YAML.
# ---------------------------------------------------------------------------
run_deepscaler() {
    local model=$1

    local exp_name="deepscaler-${model}-1agent"
    local config="examples/marft/${CONFIG_PREFIX}deepscaler_1agent.yaml"
    local script="examples/marft/deepscaler_train.py"
    local model_path
    model_path=$(resolve_model_path "$model")
    local mem_frac
    mem_frac=$(get_sglang_mem_fraction "$model")

    echo "=========================================="
    echo "Running: ${exp_name}"
    echo "  Script:          ${script}"
    echo "  Config:          ${config}"
    echo "  Model:           ${model_path}"
    echo "  mem_frac_static: ${mem_frac}"
    echo "  Eval:            aime_2024 (n=16), MATH-500, olympiadbench, minervamath"
    echo "=========================================="
    python "$script" --config "$config" \
        experiment_name="${exp_name}" \
        scheduler.type=local \
        actor.path="${model_path}" \
        ref.path="${model_path}" \
        critic.path="${model_path}" \
        sglang.model_path="${model_path}" \
        sglang.mem_fraction_static="${mem_frac}"
}

run_deepscaler_multi_agent() {
    local model=$1
    local n_agents=$2
    local shared_mode=$3
    local critic_mode=$4
    local anonymous="${5:-}"  # "anonymous" or empty

    local anon_tag=""
    local config_suffix=""
    if [ "$anonymous" = "anonymous" ]; then
        anon_tag="-anonymous"
        config_suffix="_anonymous"
    fi

    local exp_name="deepscaler-${model}-${n_agents}agent${anon_tag}-${shared_mode}-${critic_mode}"
    local config="examples/marft/${CONFIG_PREFIX}deepscaler_${n_agents}agent${config_suffix}.yaml"
    local script="examples/marft/deepscaler_train.py"
    local model_path
    model_path=$(resolve_model_path "$model")
    local mem_frac
    mem_frac=$(get_sglang_mem_fraction "$model")

    # Build CLI overrides
    local overrides="experiment_name=${exp_name} scheduler.type=local"
    overrides="$overrides actor.path=${model_path} ref.path=${model_path}"
    overrides="$overrides critic.path=${model_path} sglang.model_path=${model_path}"
    overrides="$overrides sglang.mem_fraction_static=${mem_frac}"

    # shared_lora override (configs default to shared_lora=true)
    if [ "$shared_mode" = "peragent" ]; then
        overrides="$overrides multi_agent.shared_lora=false"
    fi

    # independent_critic override (configs default to independent_critic=null)
    case "$critic_mode" in
        ctde)        ;;  # default, no override needed
        multihead)   overrides="$overrides multi_agent.independent_critic=multi_head" ;;
        criticlora)  overrides="$overrides multi_agent.independent_critic=lora" ;;
        *)
            echo "ERROR: Unknown critic_mode '${critic_mode}'" >&2
            return 1
            ;;
    esac

    # Memory-intensive combos: 4-agent + peragent + criticlora needs gradient
    # accumulation to avoid OOM during critic ppo_update backward pass.
    if [ "$n_agents" -ge 4 ] && [ "$shared_mode" = "peragent" ] && [ "$critic_mode" = "criticlora" ]; then
        overrides="$overrides actor.ppo_n_minibatches=2 critic.ppo_n_minibatches=2"
    fi

    # Enable recovery for all experiments so crashes are resumable
    # overrides="$overrides recover.mode=auto"

    echo "=========================================="
    echo "Running: ${exp_name}"
    echo "  Script:    ${script}"
    echo "  Config:    ${config}"
    echo "  Model:     ${model_path}"
    echo "  Overrides: ${overrides}"
    echo "  Eval:      aime_2024 (n=16), MATH-500, olympiadbench, minervamath"
    echo "=========================================="
    # shellcheck disable=SC2086
    python "$script" --config "$config" $overrides
}

run_all_deepscaler_model() {
    local model=$1

    # Single agent (1 experiment)
    run_deepscaler "$model"

    # Multi-agent ablations (18 experiments)
    for n_agents in "${MULTI_AGENT_COUNTS[@]}"; do
        for shared_mode in "${SHARED_MODES[@]}"; do
            for critic_mode in "${CRITIC_MODES[@]}"; do
                run_deepscaler_multi_agent "$model" "$n_agents" "$shared_mode" "$critic_mode"
            done
        done
    done
}

run_all_deepscaler() {
    local total=${#DEEPSCALER_MODELS[@]}
    total=$(( total * 19 ))
    echo "Launching all ${total} DeepScaleR experiments..."
    for model in "${DEEPSCALER_MODELS[@]}"; do
        run_all_deepscaler_model "$model"
    done
    echo "All DeepScaleR experiments completed."
}

# ---------------------------------------------------------------------------
# Run all anonymous multi-agent experiments for one DeepScaleR model (18 experiments)
# ---------------------------------------------------------------------------
run_all_deepscaler_anonymous_model() {
    local model=$1

    for n_agents in "${MULTI_AGENT_COUNTS[@]}"; do
        for shared_mode in "${SHARED_MODES[@]}"; do
            for critic_mode in "${CRITIC_MODES[@]}"; do
                run_deepscaler_multi_agent "$model" "$n_agents" "$shared_mode" "$critic_mode" "anonymous"
            done
        done
    done
}

list_deepscaler_experiments() {
    local filter="${1:-}"
    {
        for model in "${DEEPSCALER_MODELS[@]}"; do
            echo "deepscaler-${model}-1agent"
            for n_agents in "${MULTI_AGENT_COUNTS[@]}"; do
                for shared_mode in "${SHARED_MODES[@]}"; do
                    for critic_mode in "${CRITIC_MODES[@]}"; do
                        echo "deepscaler-${model}-${n_agents}agent-${shared_mode}-${critic_mode}"
                        echo "deepscaler-${model}-${n_agents}agent-anonymous-${shared_mode}-${critic_mode}"
                    done
                done
            done
        done
    } | if [ -n "$filter" ]; then grep "^${filter}" || true; else cat; fi
}

# ---------------------------------------------------------------------------
# List all experiment names (optionally filtered by prefix)
# ---------------------------------------------------------------------------
list_experiments() {
    local filter="${1:-}"

    {
        # Standard benchmarks
        for benchmark in "${BENCHMARKS[@]}"; do
            for model in "${MODELS[@]}"; do
                # Single agent
                echo "${benchmark}-${model}-1agent"

                # Named multi-agent ablations
                for n_agents in "${MULTI_AGENT_COUNTS[@]}"; do
                    for shared_mode in "${SHARED_MODES[@]}"; do
                        for critic_mode in "${CRITIC_MODES[@]}"; do
                            echo "${benchmark}-${model}-${n_agents}agent-${shared_mode}-${critic_mode}"
                        done
                    done
                done

                # Anonymous multi-agent ablations
                for n_agents in "${MULTI_AGENT_COUNTS[@]}"; do
                    for shared_mode in "${SHARED_MODES[@]}"; do
                        for critic_mode in "${CRITIC_MODES[@]}"; do
                            echo "${benchmark}-${model}-${n_agents}agent-anonymous-${shared_mode}-${critic_mode}"
                        done
                    done
                done
            done
        done

        # DeepScaleR single-agent entries
        list_deepscaler_experiments
    } | if [ -n "$filter" ]; then grep "^${filter}" || true; else cat; fi
}

# ---------------------------------------------------------------------------
# Parse and run a single experiment by name
# ---------------------------------------------------------------------------
run_experiment() {
    local name=$1

    # DeepScaleR single-agent pattern: deepscaler-{model}-1agent
    if [[ "$name" =~ ^deepscaler-(.+)-1agent$ ]]; then
        local model="${BASH_REMATCH[1]}"
        if [ -z "${MODEL_PATHS[$model]:-}" ]; then
            echo "ERROR: Unknown model '${model}' in experiment '${name}'" >&2
            echo "Valid models: ${!MODEL_PATHS[*]}" >&2
            exit 1
        fi
        run_deepscaler "$model"
        return
    fi

    # DeepScaleR anonymous multi-agent: deepscaler-{model}-{N}agent-anonymous-{shared}-{critic}
    # Must be checked before the non-anonymous pattern to prevent greedy .+ from swallowing "anonymous"
    if [[ "$name" =~ ^deepscaler-(.+)-([234])agent-anonymous-(shared|peragent)-(ctde|multihead|criticlora)$ ]]; then
        local model="${BASH_REMATCH[1]}"
        local n_agents="${BASH_REMATCH[2]}"
        local shared_mode="${BASH_REMATCH[3]}"
        local critic_mode="${BASH_REMATCH[4]}"
        if [ -z "${MODEL_PATHS[$model]:-}" ]; then
            echo "ERROR: Unknown model '${model}' in experiment '${name}'" >&2
            echo "Valid models: ${!MODEL_PATHS[*]}" >&2
            exit 1
        fi
        run_deepscaler_multi_agent "$model" "$n_agents" "$shared_mode" "$critic_mode" "anonymous"
        return
    fi

    # DeepScaleR multi-agent pattern: deepscaler-{model}-{N}agent-{shared}-{critic}
    if [[ "$name" =~ ^deepscaler-(.+)-([234])agent-(shared|peragent)-(ctde|multihead|criticlora)$ ]]; then
        local model="${BASH_REMATCH[1]}"
        local n_agents="${BASH_REMATCH[2]}"
        local shared_mode="${BASH_REMATCH[3]}"
        local critic_mode="${BASH_REMATCH[4]}"
        if [ -z "${MODEL_PATHS[$model]:-}" ]; then
            echo "ERROR: Unknown model '${model}' in experiment '${name}'" >&2
            echo "Valid models: ${!MODEL_PATHS[*]}" >&2
            exit 1
        fi
        run_deepscaler_multi_agent "$model" "$n_agents" "$shared_mode" "$critic_mode"
        return
    fi

    # DeepScaleR per-model pattern: deepscaler-{model} → run all 19
    if [[ "$name" =~ ^deepscaler-(.+)$ ]]; then
        local model="${BASH_REMATCH[1]}"
        # deepscaler-{model}-anonymous → run all 18 anonymous multi-agent
        if [[ "$model" =~ ^(.+)-anonymous$ ]]; then
            model="${BASH_REMATCH[1]}"
            if [ -n "${MODEL_PATHS[$model]:-}" ]; then
                echo "Launching all 18 anonymous deepscaler+${model} experiments..."
                run_all_deepscaler_anonymous_model "$model"
                echo "All anonymous deepscaler+${model} experiments completed."
                return
            fi
        fi
        if [ -n "${MODEL_PATHS[$model]:-}" ]; then
            local total
            total=$(list_deepscaler_experiments "deepscaler-${model}" | wc -l)
            echo "Launching all ${total} deepscaler+${model} experiments..."
            run_all_deepscaler_model "$model"
            echo "All deepscaler+${model} experiments completed."
            return
        fi
    fi

    # Single-agent pattern: {benchmark}-{model}-1agent
    # Model names can contain hyphens (e.g., dsr-qwen1.5b, dsr-llama8b),
    # so we match greedily up to the last "-1agent" suffix.
    if [[ "$name" =~ ^(gsm8k|math|deepcoder|codeforces)-(.+)-1agent$ ]]; then
        local benchmark="${BASH_REMATCH[1]}"
        local model="${BASH_REMATCH[2]}"
        if [ -z "${MODEL_PATHS[$model]:-}" ]; then
            echo "ERROR: Unknown model '${model}' in experiment '${name}'" >&2
            echo "Valid models: ${!MODEL_PATHS[*]}" >&2
            exit 1
        fi
        run_single_agent "$benchmark" "$model"
        return
    fi

    # Anonymous multi-agent pattern: {benchmark}-{model}-{N}agent-anonymous-{shared}-{critic}
    # Must be checked before the non-anonymous pattern.
    if [[ "$name" =~ ^(gsm8k|math|deepcoder|codeforces)-(.+)-([234])agent-anonymous-(shared|peragent)-(ctde|multihead|criticlora)$ ]]; then
        local benchmark="${BASH_REMATCH[1]}"
        local model="${BASH_REMATCH[2]}"
        local n_agents="${BASH_REMATCH[3]}"
        local shared_mode="${BASH_REMATCH[4]}"
        local critic_mode="${BASH_REMATCH[5]}"
        if [ -z "${MODEL_PATHS[$model]:-}" ]; then
            echo "ERROR: Unknown model '${model}' in experiment '${name}'" >&2
            echo "Valid models: ${!MODEL_PATHS[*]}" >&2
            exit 1
        fi
        run_multi_agent "$benchmark" "$model" "$n_agents" "$shared_mode" "$critic_mode" "anonymous"
        return
    fi

    # Named multi-agent pattern: {benchmark}-{model}-{N}agent-{shared}-{critic}
    # Extract from the right: last 3 segments are {N}agent, shared, critic
    if [[ "$name" =~ ^(gsm8k|math|deepcoder|codeforces)-(.+)-([234])agent-(shared|peragent)-(ctde|multihead|criticlora)$ ]]; then
        local benchmark="${BASH_REMATCH[1]}"
        local model="${BASH_REMATCH[2]}"
        local n_agents="${BASH_REMATCH[3]}"
        local shared_mode="${BASH_REMATCH[4]}"
        local critic_mode="${BASH_REMATCH[5]}"
        if [ -z "${MODEL_PATHS[$model]:-}" ]; then
            echo "ERROR: Unknown model '${model}' in experiment '${name}'" >&2
            echo "Valid models: ${!MODEL_PATHS[*]}" >&2
            exit 1
        fi
        run_multi_agent "$benchmark" "$model" "$n_agents" "$shared_mode" "$critic_mode"
        return
    fi

    # Benchmark + model pattern: {benchmark}-{model} → run all 37
    if [[ "$name" =~ ^(gsm8k|math|deepcoder|codeforces)-(.+)$ ]]; then
        local benchmark="${BASH_REMATCH[1]}"
        local model="${BASH_REMATCH[2]}"
        if [ -n "${MODEL_PATHS[$model]:-}" ]; then
            local total
            total=$(list_experiments "${benchmark}-${model}" | wc -l)
            echo "Launching all ${total} ${benchmark}+${model} experiments..."
            run_benchmark_model "$benchmark" "$model"
            echo "All ${benchmark}+${model} experiments completed."
            return
        fi
    fi

    echo "ERROR: Unrecognized experiment name '${name}'" >&2
    echo "Run 'bash examples/marft/run_all.sh list' to see valid names." >&2
    exit 1
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

# Parse optional config prefix from $2 (for run commands, not list/count)
case "${1:-}" in
    list|count|""|help|-h|--help) ;;
    *)
        if [ -n "${2:-}" ]; then
            CONFIG_PREFIX="${2}_"
            echo "Using config prefix: ${CONFIG_PREFIX}"
        fi
        ;;
esac

case "${1:-}" in
    list)
        list_experiments "${2:-}"
        ;;
    count)
        total=$(list_experiments "${2:-}" | wc -l)
        echo "${total} experiments"
        ;;
    all)
        total=$(list_experiments | wc -l)
        echo "Launching all ${total} MARFT ablation experiments..."
        for benchmark in "${BENCHMARKS[@]}"; do
            run_benchmark "$benchmark"
        done
        run_all_deepscaler
        echo "All experiments completed."
        ;;
    gsm8k|math|deepcoder|codeforces)
        total=$(list_experiments "$1" | wc -l)
        echo "Launching all ${total} ${1} experiments..."
        run_benchmark "$1"
        echo "All ${1} experiments completed."
        ;;
    deepscaler)
        run_all_deepscaler
        ;;
    ""|help|-h|--help)
        usage
        ;;
    *)
        run_experiment "$1"
        ;;
esac
