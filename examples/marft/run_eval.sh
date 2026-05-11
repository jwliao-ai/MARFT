#!/bin/bash
# run_eval.sh — Entry point for MARFT checkpoint evaluation
#
# Starts an SGLang server with LoRA support, then runs eval_checkpoints.py
# to evaluate all discovered checkpoints under the given experiment root.
#
# Usage:
#   bash examples/marft/run_eval.sh <target>
#
# Targets:
#   <benchmark>-<model>          Evaluate checkpoints for given benchmark+model
#     Benchmarks with training checkpoints:
#       deepcoder, math, gsm8k, deepscaler
#     Eval-only benchmarks (use with EVAL_BASE_ONLY):
#       livecodebench, codeforces, aime2024, math500, minervamath, olympiadbench
#     Models: qwen1.5b, qwen3b, qwen7b, dsr-qwen1.5b, dsr-qwen7b, dsr-llama8b
#   custom                       Use environment variables for full control
#
# Options (environment variables):
#   FILTER          Filter experiments by substring (e.g., FILTER=2agent-shared)
#   MAX_SAMPLES     Limit test samples per checkpoint (e.g., MAX_SAMPLES=50)
#   N_SEEDS         Number of evaluation seeds (default: 5)
#   MAX_CONCURRENT  Max concurrent SGLang requests (default: 64)
#   TP_SIZE         Tensor parallelism size (default: auto from model)
#   SGLANG_PORT     SGLang server port (default: 30000)
#   TEMPERATURE     Sampling temperature (default: 0.6)
#   RESUME          Set to 1 to resume from existing results (default: 1)
#   DRY_RUN         Set to 1 for dry run (default: 0)
#   NO_SERVER       Set to 1 to skip server startup (assume running) (default: 0)
#   MEM_FRACTION    GPU memory fraction for SGLang (default: auto from model)
#   SAVE_COMPLETIONS Set to 1 to save full completion text (default: 0)
#   CUDA_DEVICES    Visible CUDA devices (e.g., CUDA_DEVICES=0,1)
#   REWARD_WORKERS  Parallel processes for reward computation (default: auto)
#   EVAL_BASE       Set to 1 to also evaluate the base model without LoRA (default: 0)
#
# Examples:
#   # Evaluate all deepcoder-qwen3b checkpoints (default: 5 seeds, resume mode)
#   bash examples/marft/run_eval.sh deepcoder-qwen3b
#
#   # Evaluate only 2-agent shared experiments with 10 samples
#   FILTER=2agent-shared MAX_SAMPLES=10 bash examples/marft/run_eval.sh deepcoder-qwen3b
#
#   # Dry run to see discovered checkpoints
#   DRY_RUN=1 bash examples/marft/run_eval.sh deepcoder-qwen3b
#
#   # Use specific GPU
#   CUDA_DEVICES=1 bash examples/marft/run_eval.sh math-qwen1.5b
#
#   # Full custom control
#   BENCHMARK=deepcoder BASE_MODEL=/path/to/model CHECKPOINT_ROOT=/path/to/ckpts \
#     CONFIG_ROOT=/path/to/logs DATASET_PATH=/path/to/data OUTPUT_DIR=/path/to/out \
#     bash examples/marft/run_eval.sh custom

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

declare -A MODEL_PATHS
MODEL_PATHS=(
    [qwen1.5b]="Qwen2.5-1.5B-Instruct"
    [qwen3b]="Qwen2.5-3B-Instruct"
    [qwen7b]="Qwen2.5-7B-Instruct"
    [dsr-qwen1.5b]="DeepSeek-R1-Distill-Qwen-1.5B"
    [dsr-qwen7b]="DeepSeek-R1-Distill-Qwen-7B"
    [dsr-llama8b]="DeepSeek-R1-Distill-Llama-8B"
)

declare -A DATASET_PATHS
DATASET_PATHS=(
    [deepcoder]="DeepCoder-Preview-Dataset"
    [math]="MATH-level-3-5"
    [gsm8k]="gsm8k"
    [deepscaler]="DeepScaleR-Preview-Dataset"
    [livecodebench]="livecodebench"
    [codeforces]="codeforces"
    [aime2024]="aime_2024"
    [math500]="MATH-500"
    [minervamath]="minervamath"
    [olympiadbench]="olympiadbench"
)

# Default TP size by model (7B+ needs TP=2 on typical GPUs)
declare -A MODEL_TP
MODEL_TP=(
    [qwen1.5b]=1
    [qwen3b]=1
    [qwen7b]=2
    [dsr-qwen1.5b]=1
    [dsr-qwen7b]=2
    [dsr-llama8b]=2
)

# Default memory fraction by model
declare -A MODEL_MEM
MODEL_MEM=(
    [qwen1.5b]=0.85
    [qwen3b]=0.80
    [qwen7b]=0.75
    [dsr-qwen1.5b]=0.85
    [dsr-qwen7b]=0.75
    [dsr-llama8b]=0.75
)

# Experiment root directory mapping
# Different experiment campaigns store checkpoints/logs in different root dirs.
# Format: experiment_root_dir (relative to PROJECT_ROOT)
declare -A EXPERIMENT_ROOTS
EXPERIMENT_ROOTS=(
    # qwen3b deepcoder experiments (dedicated root)
    [deepcoder-qwen3b]="ecmlp_experiments_deepcoder"
    # Everything else (qwen1.5b across all benchmarks)
    [deepcoder-qwen1.5b]="ecmlp_experiments"
    [math-qwen1.5b]="ecmlp_experiments"
    [gsm8k-qwen1.5b]="ecmlp_experiments"
    [gsm8k-dsr-qwen1.5b]="ecmlp_experiments"
    [math-dsr-qwen1.5b]="ecmlp_experiments"
    [deepscaler-qwen1.5b]="ecmlp_experiments_deepscaler"
    [deepscaler-dsr-qwen1.5b]="ecmlp_experiments_deepscaler"
)

# ---------------------------------------------------------------------------
# Usage
# ---------------------------------------------------------------------------
usage() {
    echo "Usage: bash examples/marft/run_eval.sh <target>"
    echo ""
    echo "Targets: <benchmark>-<model>"
    echo "  Training benchmarks (have checkpoints):"
    echo "    deepcoder-<model>       DeepCoder checkpoints"
    echo "    math-<model>            MATH checkpoints"
    echo "    gsm8k-<model>           GSM8K checkpoints"
    echo "    deepscaler-<model>      DeepScaleR checkpoints"
    echo "  Eval-only benchmarks (use with EVAL_BASE_ONLY):"
    echo "    livecodebench-<model>   LiveCodeBench (code)"
    echo "    codeforces-<model>      Codeforces (code)"
    echo "    aime2024-<model>        AIME 2024 (math)"
    echo "    math500-<model>         MATH-500 (math)"
    echo "    minervamath-<model>     MinervaMath (math)"
    echo "    olympiadbench-<model>   OlympiadBench (math)"
    echo "  Models: qwen1.5b, qwen3b, qwen7b, dsr-qwen1.5b, dsr-qwen7b, dsr-llama8b"
    echo "  custom                    Use env vars for full control"
    echo ""
    echo "Environment variables:"
    echo "  FILTER, MAX_SAMPLES, N_SEEDS, MAX_CONCURRENT, TP_SIZE, SGLANG_PORT,"
    echo "  TEMPERATURE, RESUME, DRY_RUN, NO_SERVER, MEM_FRACTION, SAVE_COMPLETIONS,"
    echo "  CUDA_DEVICES"
    echo ""
    echo "Examples:"
    echo "  bash examples/marft/run_eval.sh deepcoder-qwen3b"
    echo "  FILTER=2agent-shared DRY_RUN=1 bash examples/marft/run_eval.sh deepcoder-qwen3b"
    echo "  CUDA_DEVICES=0,1 TP_SIZE=2 bash examples/marft/run_eval.sh math-qwen1.5b"
    exit 1
}

# ---------------------------------------------------------------------------
# Parse target
# ---------------------------------------------------------------------------
if [ $# -lt 1 ]; then
    usage
fi

TARGET="$1"

# ---------------------------------------------------------------------------
# Resolve target to evaluation parameters
# ---------------------------------------------------------------------------
resolve_target() {
    local target=$1

    if [ "$target" = "custom" ]; then
        # All parameters must come from environment
        if [ -z "${BENCHMARK:-}" ] || [ -z "${BASE_MODEL:-}" ] || \
           [ -z "${CHECKPOINT_ROOT:-}" ] || [ -z "${CONFIG_ROOT:-}" ] || \
           [ -z "${DATASET_PATH:-}" ] || [ -z "${OUTPUT_DIR:-}" ]; then
            echo "ERROR: 'custom' target requires BENCHMARK, BASE_MODEL, CHECKPOINT_ROOT," >&2
            echo "       CONFIG_ROOT, DATASET_PATH, and OUTPUT_DIR environment variables." >&2
            exit 1
        fi
        return
    fi

    # Parse target: <benchmark>-<model>
    local benchmark model

    # Try to extract benchmark and model from target
    # Order matters: longer prefixes first to avoid ambiguous matches
    # (e.g. "math500" before "math", "livecodebench" before others)
    for bm in livecodebench codeforces deepcoder deepscaler olympiadbench minervamath math500 aime2024 gsm8k math; do
        if [[ "$target" == "${bm}-"* ]]; then
            benchmark="$bm"
            model="${target#${bm}-}"
            break
        fi
    done

    if [ -z "${benchmark:-}" ] || [ -z "${model:-}" ]; then
        echo "ERROR: Cannot parse target '$target'. Expected format: <benchmark>-<model>" >&2
        echo "  Benchmarks: deepcoder, livecodebench, codeforces, gsm8k, math, deepscaler," >&2
        echo "              aime2024, math500, minervamath, olympiadbench" >&2
        echo "  Models: qwen1.5b, qwen3b, qwen7b, dsr-qwen1.5b, dsr-qwen7b, dsr-llama8b" >&2
        exit 1
    fi

    # Validate model
    local model_dir="${MODEL_PATHS[$model]:-}"
    if [ -z "$model_dir" ]; then
        echo "ERROR: Unknown model '$model'. Valid: ${!MODEL_PATHS[*]}" >&2
        exit 1
    fi

    # Validate dataset
    local dataset_dir="${DATASET_PATHS[$benchmark]:-}"
    if [ -z "$dataset_dir" ]; then
        echo "ERROR: Unknown benchmark '$benchmark'." >&2
        exit 1
    fi

    # Look up experiment root
    local exp_key="${benchmark}-${model}"
    local exp_root="${EXPERIMENT_ROOTS[$exp_key]:-}"
    if [ -z "$exp_root" ] && [ -z "${EVAL_BASE_ONLY:-}" ] \
       && [ -z "${CHECKPOINT_ROOT:-}" ] && [ -z "${CONFIG_ROOT:-}" ]; then
        echo "ERROR: No experiment root configured for '$exp_key'." >&2
        echo "  Options:" >&2
        echo "    1. Set CHECKPOINT_ROOT and CONFIG_ROOT env vars to point to your checkpoints" >&2
        echo "    2. Set EVAL_BASE_ONLY to skip checkpoint discovery (base model only)" >&2
        echo "    3. Use 'custom' target for full control" >&2
        exit 1
    fi

    # Set variables (if not already set via env)
    BENCHMARK="${BENCHMARK:-$benchmark}"
    BASE_MODEL="${BASE_MODEL:-${MODEL_ROOT}/${model_dir}}"
    if [ -n "$exp_root" ]; then
        CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${exp_root}/checkpoints/root}"
        CONFIG_ROOT="${CONFIG_ROOT:-${exp_root}/logs/root}"
        OUTPUT_DIR="${OUTPUT_DIR:-${exp_root}/eval_results}"
    else
        # Eval-only: no checkpoint/config root
        CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-}"
        CONFIG_ROOT="${CONFIG_ROOT:-}"
        # Math eval-only benchmarks → ecmlp_experiments_deepscaler
        # Code eval-only benchmarks → ecmlp_experiments_deepcoder
        case "$benchmark" in
            aime2024|math500|minervamath|olympiadbench)
                OUTPUT_DIR="${OUTPUT_DIR:-ecmlp_experiments_deepscaler/eval_results}" ;;
            *)
                OUTPUT_DIR="${OUTPUT_DIR:-ecmlp_experiments_deepcoder/eval_results}" ;;
        esac
    fi
    DATASET_PATH="${DATASET_PATH:-${DATA_ROOT}/${dataset_dir}}"

    # Auto-filter by benchmark-model prefix when the experiment root contains
    # multiple benchmarks (e.g., ecmlp_experiments has gsm8k + math + deepcoder).
    # Skip auto-filter for cross-benchmark eval (no experiment root found) —
    # the user's FILTER will be used as-is to match checkpoint dir names.
    if [ -n "$exp_root" ]; then
        _AUTO_FILTER="${benchmark}-${model}"
    else
        _AUTO_FILTER=""
    fi

    # Set model-specific defaults for TP and memory fraction
    _DEFAULT_TP="${MODEL_TP[$model]:-1}"
    _DEFAULT_MEM="${MODEL_MEM[$model]:-0.85}"
}

resolve_target "$TARGET"

# ---------------------------------------------------------------------------
# Apply defaults from environment variables
# ---------------------------------------------------------------------------
SGLANG_PORT="${SGLANG_PORT:-30000}"
TP_SIZE="${TP_SIZE:-${_DEFAULT_TP:-1}}"
MEM_FRACTION="${MEM_FRACTION:-${_DEFAULT_MEM:-0.85}}"
MAX_CONCURRENT="${MAX_CONCURRENT:-64}"
N_SEEDS="${N_SEEDS:-5}"
TEMPERATURE="${TEMPERATURE:-0.6}"
RESUME="${RESUME:-1}"
DRY_RUN="${DRY_RUN:-0}"
NO_SERVER="${NO_SERVER:-0}"
SAVE_COMPLETIONS="${SAVE_COMPLETIONS:-0}"
MAX_LORA_RANK="${MAX_LORA_RANK:-32}"
REWARD_WORKERS="${REWARD_WORKERS:-0}"
EVAL_BASE="${EVAL_BASE:-0}"

# Combine auto-filter (benchmark-model) with user's FILTER.
# If user sets FILTER, use it as an additional substring within
# the auto-filtered set. If no auto-filter, use FILTER as-is.
if [ -n "${_AUTO_FILTER:-}" ]; then
    if [ -n "${FILTER:-}" ]; then
        # User wants further filtering within benchmark-model scope.
        # Combine auto-filter prefix with user's filter so both must match.
        # e.g., auto="deepcoder-qwen3b" + user="2agent-shared" → "deepcoder-qwen3b-2agent-shared"
        FILTER="${_AUTO_FILTER}-${FILTER}"
    else
        FILTER="${_AUTO_FILTER}"
    fi
fi

# Set CUDA devices if specified
if [ -n "${CUDA_DEVICES:-}" ]; then
    export CUDA_VISIBLE_DEVICES="$CUDA_DEVICES"
fi

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
echo "=========================================="
echo " MARFT Checkpoint Evaluation"
echo "=========================================="
echo "  Target:          ${TARGET}"
echo "  Benchmark:       ${BENCHMARK}"
echo "  Base model:      ${BASE_MODEL}"
echo "  Checkpoint root: ${CHECKPOINT_ROOT}"
echo "  Config root:     ${CONFIG_ROOT}"
echo "  Dataset path:    ${DATASET_PATH}"
echo "  Output dir:      ${OUTPUT_DIR}"
echo "  SGLang port:     ${SGLANG_PORT}"
echo "  TP size:         ${TP_SIZE}"
echo "  Mem fraction:    ${MEM_FRACTION}"
echo "  Max concurrent:  ${MAX_CONCURRENT}"
echo "  N seeds:         ${N_SEEDS}"
echo "  Temperature:     ${TEMPERATURE}"
echo "  Max LoRA rank:   ${MAX_LORA_RANK}"
echo "  Resume:          ${RESUME}"
echo "  Dry run:         ${DRY_RUN}"
echo "  No server:       ${NO_SERVER}"
echo "  Save completions:${SAVE_COMPLETIONS}"
if [ -n "${FILTER:-}" ]; then
echo "  Filter:          ${FILTER}"
fi
if [ -n "${MAX_SAMPLES:-}" ]; then
echo "  Max samples:     ${MAX_SAMPLES}"
fi
if [ -n "${CUDA_DEVICES:-}" ]; then
echo "  CUDA devices:    ${CUDA_DEVICES}"
fi
echo "=========================================="

# Check paths exist
if [ ! -d "${BASE_MODEL}" ]; then
    echo "ERROR: Base model not found: ${BASE_MODEL}" >&2
    exit 1
fi
if [ -z "${EVAL_BASE_ONLY:-}" ]; then
    # Checkpoint/config dirs only needed when not in --eval-base-only mode
    if [ ! -d "${CHECKPOINT_ROOT}" ]; then
        echo "ERROR: Checkpoint root not found: ${CHECKPOINT_ROOT}" >&2
        exit 1
    fi
    if [ ! -d "${CONFIG_ROOT}" ]; then
        echo "ERROR: Config root not found: ${CONFIG_ROOT}" >&2
        exit 1
    fi
fi
if [ ! -d "${DATASET_PATH}" ]; then
    echo "ERROR: Dataset path not found: ${DATASET_PATH}" >&2
    exit 1
fi

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Check eval script exists
if [ ! -f "examples/marft/eval_checkpoints.py" ]; then
    echo "ERROR: eval_checkpoints.py not found at examples/marft/eval_checkpoints.py" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Build python command
# ---------------------------------------------------------------------------
CMD=(
    python examples/marft/eval_checkpoints.py
    --checkpoint-root "${CHECKPOINT_ROOT}"
    --config-root "${CONFIG_ROOT}"
    --dataset-path "${DATASET_PATH}"
    --benchmark "${BENCHMARK}"
    --output-dir "${OUTPUT_DIR}"
    --base-model "${BASE_MODEL}"
    --sglang-port "${SGLANG_PORT}"
    --tp-size "${TP_SIZE}"
    --mem-fraction "${MEM_FRACTION}"
    --max-concurrent "${MAX_CONCURRENT}"
    --n-seeds "${N_SEEDS}"
    --temperature "${TEMPERATURE}"
    --max-lora-rank "${MAX_LORA_RANK}"
    --reward-workers "${REWARD_WORKERS}"
)

# Conditional flags
if [ "${NO_SERVER}" != "1" ]; then
    CMD+=(--start-server)
fi
if [ "${RESUME}" = "1" ]; then
    CMD+=(--resume)
fi
if [ "${DRY_RUN}" = "1" ]; then
    CMD+=(--dry-run)
fi
if [ "${SAVE_COMPLETIONS}" = "1" ]; then
    CMD+=(--save-completions)
fi
if [ -n "${FILTER:-}" ]; then
    CMD+=(--filter "${FILTER}")
fi
if [ -n "${MAX_SAMPLES:-}" ]; then
    CMD+=(--max-samples "${MAX_SAMPLES}")
fi
if [ "${EVAL_BASE}" = "1" ]; then
    CMD+=(--eval-base)
fi
if [ -n "${EVAL_BASE_ONLY:-}" ]; then
    # EVAL_BASE_ONLY is a space-separated list of YAML config paths
    # shellcheck disable=SC2086
    CMD+=(--eval-base-only ${EVAL_BASE_ONLY})
fi

# ---------------------------------------------------------------------------
# Run evaluation
# ---------------------------------------------------------------------------
echo ""
echo "Running: ${CMD[*]}"
echo ""

exec "${CMD[@]}"
