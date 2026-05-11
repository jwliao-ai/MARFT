#!/bin/bash
# run_eval_table.sh — Unified evaluation orchestrator for MARFT experiments
#
# Evaluates ALL checkpoints from two experiment groups across 8 benchmarks,
# then aggregates results into a single cross-benchmark table.
#
# Groups:
#   code  — Qwen2.5-3B-Instruct (DeepCoder training), 12 experiments
#           Benchmarks: deepcoder, livecodebench, codeforces
#   math  — Qwen2.5-1.5B-Instruct (DeepScaleR training), 16 experiments
#           Benchmarks: deepscaler, aime2024, math500, minervamath, olympiadbench
#
# Usage:
#   bash examples/marft/run_eval_table.sh all               # Run everything
#   bash examples/marft/run_eval_table.sh code               # Code group only
#   bash examples/marft/run_eval_table.sh math               # Math group only
#   bash examples/marft/run_eval_table.sh code deepcoder     # Specific benchmark
#   bash examples/marft/run_eval_table.sh math math500       # Specific benchmark
#   bash examples/marft/run_eval_table.sh baseline code deepscaler  # Baselines only
#   bash examples/marft/run_eval_table.sh table              # Aggregate only (no GPU)
#
# Environment variables:
#   N_SEEDS         Number of evaluation seeds (default: 5)
#   MAX_SAMPLES     Limit test samples per checkpoint
#   MAX_CONCURRENT  Max concurrent SGLang requests (default: 64)
#   SGLANG_PORT     SGLang server port (default: 30000)
#   TEMPERATURE     Sampling temperature (default: 0.6)
#   FILTER          Additional filter for experiments (e.g., FILTER=2agent-shared)
#   DRY_RUN         Set to 1 for dry run (default: 0)
#   REWARD_WORKERS  Parallel processes for reward computation (default: auto)
#   CUDA_DEVICES    Visible CUDA devices (e.g., CUDA_DEVICES=0,1)
#   MAX_LORA_RANK   Maximum LoRA rank for SGLang (default: 32)
#   SAVE_COMPLETIONS Set to 1 to save full completion text (default: 0)

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
DATA_ROOT="/workspace/data"

# Dataset directory names under DATA_ROOT
declare -A DATASET_MAP
DATASET_MAP=(
    [deepcoder]="DeepCoder-Preview-Dataset"
    [livecodebench]="livecodebench"
    [codeforces]="codeforces"
    [deepscaler]="DeepScaleR-Preview-Dataset"
    [aime2024]="aime_2024"
    [math500]="MATH-500"
    [minervamath]="minervamath"
    [olympiadbench]="olympiadbench"
)

# --- Code group ---
CODE_EXP_ROOT="ecmlp_experiments_deepcoder"
CODE_BASE_MODEL="${MODEL_ROOT}/Qwen2.5-3B-Instruct"
CODE_FILTER_PREFIX="deepcoder-qwen3b"
CODE_TP=1
CODE_MEM=0.80
CODE_BENCHMARKS=(deepcoder livecodebench codeforces)

# Baseline YAML configs for cross-domain code benchmarks
CODE_BASE_YAMLS=(
    examples/marft/deepcoder_2agent.yaml
    examples/marft/deepcoder_3agent.yaml
    examples/marft/deepcoder_4agent.yaml
)

# --- Math group ---
MATH_EXP_ROOT="ecmlp_experiments_deepscaler"
MATH_BASE_MODEL="${MODEL_ROOT}/Qwen2.5-1.5B-Instruct"
MATH_FILTER_PREFIX="deepscaler-qwen1.5b"
MATH_TP=1
MATH_MEM=0.85
MATH_BENCHMARKS=(deepscaler aime2024 math500 minervamath olympiadbench)

# Baseline YAML configs for cross-domain math benchmarks
MATH_BASE_YAMLS=(
    examples/marft/deepscaler_2agent.yaml
    examples/marft/deepscaler_3agent.yaml
    examples/marft/deepscaler_4agent.yaml
)

# In-domain benchmarks (where --eval-base is used with checkpoints)
CODE_INDOMAIN="deepcoder"
MATH_INDOMAIN="deepscaler"

# ---------------------------------------------------------------------------
# Defaults from environment
# ---------------------------------------------------------------------------
SGLANG_PORT="${SGLANG_PORT:-30000}"
N_SEEDS="${N_SEEDS:-5}"
MAX_CONCURRENT="${MAX_CONCURRENT:-64}"
TEMPERATURE="${TEMPERATURE:-0.6}"
MAX_LORA_RANK="${MAX_LORA_RANK:-32}"
REWARD_WORKERS="${REWARD_WORKERS:-0}"
DRY_RUN="${DRY_RUN:-0}"
SAVE_COMPLETIONS="${SAVE_COMPLETIONS:-0}"
OUTPUT_SUFFIX="${OUTPUT_SUFFIX:-}"

# Set CUDA devices if specified
if [ -n "${CUDA_DEVICES:-}" ]; then
    export CUDA_VISIBLE_DEVICES="$CUDA_DEVICES"
fi

# ---------------------------------------------------------------------------
# Server management
# ---------------------------------------------------------------------------
SGLANG_PID=""
SGLANG_LOG=""

start_server() {
    local base_model=$1 tp=$2 mem=$3
    local log_dir="${4:-/tmp}"

    # Skip server startup in dry-run mode
    if [ "${DRY_RUN}" = "1" ]; then
        echo "[run_eval_table] DRY_RUN: skipping server startup"
        return 0
    fi

    # Refuse to start if something is already listening on the port
    if curl -s "http://localhost:${SGLANG_PORT}/health" > /dev/null 2>&1; then
        echo "ERROR: Port ${SGLANG_PORT} is already in use." >&2
        echo "  Kill the existing server first:" >&2
        echo "    kill \$(lsof -t -i:${SGLANG_PORT})" >&2
        echo "  Or choose a different port: SGLANG_PORT=30001 ..." >&2
        return 1
    fi

    SGLANG_LOG="${log_dir}/sglang_server.log"
    mkdir -p "$(dirname "$SGLANG_LOG")"

    echo "[run_eval_table] Starting SGLang server..."
    echo "  Model:    ${base_model}"
    echo "  TP:       ${tp}"
    echo "  Mem:      ${mem}"
    echo "  Port:     ${SGLANG_PORT}"
    echo "  Log:      ${SGLANG_LOG}"

    python -m sglang.launch_server \
        --model-path "${base_model}" \
        --port "${SGLANG_PORT}" \
        --tp-size "${tp}" \
        --enable-lora \
        --max-lora-rank "${MAX_LORA_RANK}" \
        --lora-target-modules all \
        --max-loaded-loras 8 \
        --max-loras-per-batch 8 \
        --mem-fraction-static "${mem}" \
        --trust-remote-code \
        > "${SGLANG_LOG}" 2>&1 &
    SGLANG_PID=$!

    echo "[run_eval_table] SGLang PID: ${SGLANG_PID}"

    # Wait for server to become healthy
    local timeout=300
    local elapsed=0
    local interval=5
    while [ $elapsed -lt $timeout ]; do
        if curl -s "http://localhost:${SGLANG_PORT}/health" > /dev/null 2>&1; then
            echo "[run_eval_table] SGLang server is healthy (took ${elapsed}s)"
            return 0
        fi
        # Check if process is still alive
        if ! kill -0 "$SGLANG_PID" 2>/dev/null; then
            echo "ERROR: SGLang server died during startup. Check: ${SGLANG_LOG}" >&2
            SGLANG_PID=""
            return 1
        fi
        sleep $interval
        elapsed=$((elapsed + interval))
    done

    echo "ERROR: SGLang server did not become healthy within ${timeout}s" >&2
    echo "  Check log: ${SGLANG_LOG}" >&2
    stop_server
    return 1
}

stop_server() {
    if [ -n "$SGLANG_PID" ]; then
        echo "[run_eval_table] Stopping SGLang server (PID: ${SGLANG_PID})..."
        kill "$SGLANG_PID" 2>/dev/null || true
        wait "$SGLANG_PID" 2>/dev/null || true
        SGLANG_PID=""
        echo "[run_eval_table] Server stopped."
    fi
}

# Cleanup on exit
trap stop_server EXIT

# ---------------------------------------------------------------------------
# Evaluation functions
# ---------------------------------------------------------------------------

eval_benchmark() {
    # Evaluate all checkpoints on a specific benchmark
    local benchmark=$1
    local exp_root=$2
    local base_model=$3
    local filter_prefix=$4
    local is_indomain=$5  # "1" if this is the training benchmark

    local output_dir="${exp_root}/eval_results/${benchmark}${OUTPUT_SUFFIX}"
    mkdir -p "${output_dir}"

    echo ""
    echo "======================================================"
    echo " Evaluating: ${benchmark}"
    echo "   Output:   ${output_dir}"
    echo "   Filter:   ${filter_prefix}"
    echo "   In-domain: ${is_indomain}"
    echo "======================================================"

    # Build python command
    local cmd=(
        python examples/marft/eval_checkpoints.py
        --checkpoint-root "${exp_root}/checkpoints/root"
        --config-root "${exp_root}/logs/root"
        --dataset-path "${DATA_ROOT}/${DATASET_MAP[$benchmark]}"
        --benchmark "${benchmark}"
        --output-dir "${output_dir}"
        --base-model "${base_model}"
        --sglang-port "${SGLANG_PORT}"
        --n-seeds "${N_SEEDS}"
        --max-concurrent "${MAX_CONCURRENT}"
        --temperature "${TEMPERATURE}"
        --max-lora-rank "${MAX_LORA_RANK}"
        --reward-workers "${REWARD_WORKERS}"
        --resume
    )

    # Apply filter (combine prefix with user's FILTER if set)
    local effective_filter="${filter_prefix}"
    if [ -n "${FILTER:-}" ]; then
        effective_filter="${filter_prefix}-${FILTER}"
    fi
    cmd+=(--filter "${effective_filter}")

    # Add --eval-base for in-domain benchmark
    if [ "${is_indomain}" = "1" ]; then
        cmd+=(--eval-base)
    fi

    # Optional overrides
    if [ -n "${MAX_SAMPLES:-}" ]; then
        cmd+=(--max-samples "${MAX_SAMPLES}")
    fi
    if [ "${DRY_RUN}" = "1" ]; then
        cmd+=(--dry-run)
    fi
    if [ "${SAVE_COMPLETIONS}" = "1" ]; then
        cmd+=(--save-completions)
    fi

    echo "[run_eval_table] Running: ${cmd[*]}"
    "${cmd[@]}"
}

eval_baselines() {
    # Evaluate base model only (no checkpoints) for cross-domain benchmarks
    local benchmark=$1
    local base_model=$2
    local exp_root=$3
    local -n yaml_list=$4

    local output_dir="${exp_root}/eval_results/${benchmark}${OUTPUT_SUFFIX}"
    mkdir -p "${output_dir}"

    # Collect valid YAML paths
    local valid_yamls=()
    for yaml_path in "${yaml_list[@]}"; do
        if [ -f "${yaml_path}" ]; then
            valid_yamls+=("${yaml_path}")
        else
            echo "WARNING: YAML config not found, skipping: ${yaml_path}"
        fi
    done

    if [ ${#valid_yamls[@]} -eq 0 ]; then
        echo "[run_eval_table] No valid YAML configs for baselines on ${benchmark}, skipping."
        return 0
    fi

    echo ""
    echo "  --- Baselines for ${benchmark} (${#valid_yamls[@]} configs) ---"

    local cmd=(
        python examples/marft/eval_checkpoints.py
        --dataset-path "${DATA_ROOT}/${DATASET_MAP[$benchmark]}"
        --benchmark "${benchmark}"
        --output-dir "${output_dir}"
        --base-model "${base_model}"
        --sglang-port "${SGLANG_PORT}"
        --n-seeds "${N_SEEDS}"
        --max-concurrent "${MAX_CONCURRENT}"
        --temperature "${TEMPERATURE}"
        --max-lora-rank "${MAX_LORA_RANK}"
        --reward-workers "${REWARD_WORKERS}"
        --resume
        --eval-base-only "${valid_yamls[@]}"
    )

    if [ -n "${MAX_SAMPLES:-}" ]; then
        cmd+=(--max-samples "${MAX_SAMPLES}")
    fi
    if [ "${DRY_RUN}" = "1" ]; then
        cmd+=(--dry-run)
    fi
    if [ "${SAVE_COMPLETIONS}" = "1" ]; then
        cmd+=(--save-completions)
    fi

    echo "[run_eval_table] Running: ${cmd[*]}"
    "${cmd[@]}"
}

# ---------------------------------------------------------------------------
# Group runners
# ---------------------------------------------------------------------------

run_code_group() {
    local specific_benchmark="${1:-}"

    echo ""
    echo "########################################################"
    echo "# CODE GROUP (Qwen2.5-3B-Instruct)"
    echo "########################################################"

    start_server "${CODE_BASE_MODEL}" "${CODE_TP}" "${CODE_MEM}" \
        "${CODE_EXP_ROOT}/eval_results"

    for bm in "${CODE_BENCHMARKS[@]}"; do
        # Skip if a specific benchmark was requested and this isn't it
        if [ -n "${specific_benchmark}" ] && [ "${bm}" != "${specific_benchmark}" ]; then
            continue
        fi

        local is_indomain="0"
        if [ "${bm}" = "${CODE_INDOMAIN}" ]; then
            is_indomain="1"
        fi

        eval_benchmark "${bm}" "${CODE_EXP_ROOT}" "${CODE_BASE_MODEL}" \
            "${CODE_FILTER_PREFIX}" "${is_indomain}"

        # For cross-domain benchmarks, also evaluate baselines from YAML configs
        if [ "${is_indomain}" = "0" ]; then
            eval_baselines "${bm}" "${CODE_BASE_MODEL}" "${CODE_EXP_ROOT}" \
                CODE_BASE_YAMLS
        fi
    done

    stop_server
}

run_math_group() {
    local specific_benchmark="${1:-}"

    echo ""
    echo "########################################################"
    echo "# MATH GROUP (Qwen2.5-1.5B-Instruct)"
    echo "########################################################"

    start_server "${MATH_BASE_MODEL}" "${MATH_TP}" "${MATH_MEM}" \
        "${MATH_EXP_ROOT}/eval_results"

    for bm in "${MATH_BENCHMARKS[@]}"; do
        if [ -n "${specific_benchmark}" ] && [ "${bm}" != "${specific_benchmark}" ]; then
            continue
        fi

        local is_indomain="0"
        if [ "${bm}" = "${MATH_INDOMAIN}" ]; then
            is_indomain="1"
        fi

        eval_benchmark "${bm}" "${MATH_EXP_ROOT}" "${MATH_BASE_MODEL}" \
            "${MATH_FILTER_PREFIX}" "${is_indomain}"

        if [ "${is_indomain}" = "0" ]; then
            eval_baselines "${bm}" "${MATH_BASE_MODEL}" "${MATH_EXP_ROOT}" \
                MATH_BASE_YAMLS
        fi
    done

    stop_server
}

run_aggregate() {
    echo ""
    echo "########################################################"
    echo "# AGGREGATING RESULTS INTO CROSS-BENCHMARK TABLE"
    echo "########################################################"

    python examples/marft/aggregate_results.py \
        --code-root "${CODE_EXP_ROOT}/eval_results" \
        --math-root "${MATH_EXP_ROOT}/eval_results" \
        --output "${CODE_EXP_ROOT}/eval_results/results_table.csv"
}

run_baselines() {
    local group=$1
    local benchmark=$2

    if [ "${group}" = "code" ]; then
        local base_model="${CODE_BASE_MODEL}"
        local exp_root="${CODE_EXP_ROOT}"
        local tp="${CODE_TP}"
        local mem="${CODE_MEM}"
        local yaml_ref="CODE_BASE_YAMLS"
    elif [ "${group}" = "math" ]; then
        local base_model="${MATH_BASE_MODEL}"
        local exp_root="${MATH_EXP_ROOT}"
        local tp="${MATH_TP}"
        local mem="${MATH_MEM}"
        local yaml_ref="MATH_BASE_YAMLS"
    else
        echo "ERROR: Unknown group '${group}' for baseline. Use 'code' or 'math'." >&2
        return 1
    fi

    echo ""
    echo "########################################################"
    echo "# BASELINES ONLY: ${group} group — ${benchmark}"
    echo "########################################################"

    start_server "${base_model}" "${tp}" "${mem}" \
        "${exp_root}/eval_results"

    eval_baselines "${benchmark}" "${base_model}" "${exp_root}" "${yaml_ref}"

    stop_server
}

# ---------------------------------------------------------------------------
# Usage
# ---------------------------------------------------------------------------
usage() {
    echo "Usage: bash examples/marft/run_eval_table.sh <command> [benchmark]"
    echo ""
    echo "Commands:"
    echo "  all                       Run both groups + aggregate"
    echo "  code [benchmark]          Code group (deepcoder, livecodebench, codeforces)"
    echo "  math [benchmark]          Math group (deepscaler, aime2024, math500, minervamath, olympiadbench)"
    echo "  baseline <group> <bench>  Baselines only (no checkpoints)"
    echo "  table                     Aggregate existing results only (no GPU needed)"
    echo ""
    echo "Examples:"
    echo "  bash examples/marft/run_eval_table.sh all"
    echo "  bash examples/marft/run_eval_table.sh code"
    echo "  bash examples/marft/run_eval_table.sh code deepcoder"
    echo "  bash examples/marft/run_eval_table.sh math math500"
    echo "  bash examples/marft/run_eval_table.sh baseline code deepcoder"
    echo "  bash examples/marft/run_eval_table.sh baseline math deepscaler"
    echo "  bash examples/marft/run_eval_table.sh table"
    echo ""
    echo "Environment variables:"
    echo "  N_SEEDS, MAX_SAMPLES, MAX_CONCURRENT, SGLANG_PORT, TEMPERATURE,"
    echo "  FILTER, DRY_RUN, REWARD_WORKERS, CUDA_DEVICES, MAX_LORA_RANK,"
    echo "  SAVE_COMPLETIONS"
    exit 1
}

# ---------------------------------------------------------------------------
# Main dispatch
# ---------------------------------------------------------------------------
if [ $# -lt 1 ]; then
    usage
fi

COMMAND="$1"
BENCHMARK_ARG="${2:-}"
THIRD_ARG="${3:-}"

echo "=========================================="
echo " MARFT Evaluation Table Builder"
echo "=========================================="
echo "  Command:     ${COMMAND}"
if [ -n "${BENCHMARK_ARG}" ]; then
echo "  Benchmark:   ${BENCHMARK_ARG}"
fi
echo "  N seeds:     ${N_SEEDS}"
echo "  Port:        ${SGLANG_PORT}"
echo "  Temperature: ${TEMPERATURE}"
if [ -n "${FILTER:-}" ]; then
echo "  Filter:      ${FILTER}"
fi
if [ -n "${MAX_SAMPLES:-}" ]; then
echo "  Max samples: ${MAX_SAMPLES}"
fi
if [ -n "${OUTPUT_SUFFIX}" ]; then
echo "  Output suffix: ${OUTPUT_SUFFIX}"
fi
if [ "${DRY_RUN}" = "1" ]; then
echo "  Dry run:     YES"
fi
echo "=========================================="

case "${COMMAND}" in
    all)
        run_code_group
        run_math_group
        run_aggregate
        ;;
    code)
        run_code_group "${BENCHMARK_ARG}"
        ;;
    math)
        run_math_group "${BENCHMARK_ARG}"
        ;;
    baseline)
        if [ -z "${BENCHMARK_ARG}" ] || [ -z "${THIRD_ARG}" ]; then
            echo "ERROR: 'baseline' requires <group> and <benchmark>." >&2
            echo "  Usage: bash examples/marft/run_eval_table.sh baseline <code|math> <benchmark>" >&2
            exit 1
        fi
        run_baselines "${BENCHMARK_ARG}" "${THIRD_ARG}"
        ;;
    table)
        run_aggregate
        ;;
    *)
        echo "ERROR: Unknown command '${COMMAND}'" >&2
        usage
        ;;
esac

echo ""
echo "[run_eval_table] Done."
