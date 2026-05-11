"""Extract best hyperparameters from TensorBoard logs for the HP sweep.

Reads TensorBoard event files for a given phase, computes 3-step moving
average of the selection metric, ranks experiments, and outputs best HP
values to an env file (for downstream phases) and a CSV summary.

Usage::

    # After Phase 1A completes:
    python examples/marft/extract_hp_results.py \\
        --phase p1a \\
        --deepscaler-tb-root ecmlp_experiments_deepscaler/tensorboard \\
        --deepcoder-tb-root ecmlp_experiments_deepcoder/tensorboard \\
        --output ecmlp_experiments/hp_sweep/phase_results.env \\
        --append

    # Generate CSV only:
    python examples/marft/extract_hp_results.py \\
        --phase p1a \\
        --csv-only
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# Base configuration registry (mirrors run_hp_sweep.sh)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BaseConfig:
    """One of the 4 base configurations being swept."""

    id: str           # ds-2a, ds-3a, dc-2a, dc-3a
    label: str        # deepscaler-2a, deepscaler-3a, etc.
    benchmark: str    # deepscaler or deepcoder
    trial: str        # trial1 or trial2
    metric: str       # TensorBoard scalar tag to optimise


BASE_CONFIGS = [
    BaseConfig("ds-2a", "deepscaler-2a", "deepscaler", "trial2", "MATH-500/reward"),
    BaseConfig("ds-3a", "deepscaler-3a", "deepscaler", "trial2", "MATH-500/reward"),
    BaseConfig("dc-2a", "deepcoder-2a", "deepcoder", "trial1", "eval-rollout/reward"),
    BaseConfig("dc-3a", "deepcoder-3a", "deepcoder", "trial2", "eval-rollout/reward"),
]

# ---------------------------------------------------------------------------
# HP extraction patterns per phase
# ---------------------------------------------------------------------------

# Maps phase → list of (hp_name, regex_group_name, env_var_suffix) tuples.
# The regex is applied to the experiment name to extract the HP value.
PHASE_HP_PATTERNS: dict[str, list[tuple[str, str, str]]] = {
    "p1a": [
        ("LR",   r"lr(?P<LR>[0-9e]+)",       "LR"),
        ("CLIP", r"clip(?P<CLIP>[0-9]+)",     "CLIP"),
    ],
    "p1b": [
        ("KL", r"kl(?P<KL>[0-9n]+)", "KL"),
    ],
    "p2a": [
        ("RS", r"rs(?P<RS>[0-9n]+)",  "RS"),
        ("RB", r"rb(?P<RB>[0-9n]+)",  "RB"),
    ],
    "p2b": [
        ("DISCOUNT", r"g(?P<DISCOUNT>[0-9]+)", "DISCOUNT"),
        ("LAMBDA",   r"l(?P<LAMBDA>[0-9]+)",   "LAMBDA"),
    ],
    "p3a": [
        ("TEMP", r"t(?P<TEMP>[0-9]+)", "TEMP"),
    ],
    "p3b": [
        ("RANK",  r"r(?P<RANK>[0-9]+)",  "RANK"),
        ("ALPHA", r"a(?P<ALPHA>[0-9]+)", "ALPHA"),
    ],
}

# Reverse-decode: encoded tag → actual HP value
DECODE_MAP: dict[str, dict[str, str]] = {
    "LR": {
        "1e6": "1e-6", "5e6": "5e-6", "1e5": "1e-5", "2e5": "2e-5",
    },
    "CLIP": {
        "01": "0.1", "02": "0.2", "03": "0.3",
    },
    "KL": {
        "0": "0.0", "001": "0.01", "005": "0.05", "01": "0.1",
    },
    "RS": {
        "05": "0.5", "1": "1.0", "2": "2.0", "5": "5.0",
    },
    "RB": {
        "0": "0.0", "n05": "-0.5",
    },
    "DISCOUNT": {
        "099": "0.99", "1": "1.0",
    },
    "LAMBDA": {
        "095": "0.95", "1": "1.0",
    },
    "TEMP": {
        "07": "0.7", "08": "0.8", "1": "1.0", "12": "1.2",
    },
    "RANK": {},   # numeric, no decoding needed
    "ALPHA": {},  # numeric, no decoding needed
}


def decode_hp(hp_name: str, encoded_value: str) -> str:
    """Convert encoded HP tag back to actual value."""
    table = DECODE_MAP.get(hp_name, {})
    if table:
        return table.get(encoded_value, encoded_value)
    return encoded_value


# ---------------------------------------------------------------------------
# TensorBoard reading
# ---------------------------------------------------------------------------

def read_scalar(logdir: str, tag: str) -> list[tuple[int, float]]:
    """Read a scalar time series from TensorBoard events.

    Returns list of (step, value) sorted by step.
    """
    try:
        from tensorboard.backend.event_processing.event_accumulator import (
            EventAccumulator,
        )
    except ImportError:
        print(
            "ERROR: tensorboard package is required. Install with: pip install tensorboard",
            file=sys.stderr,
        )
        sys.exit(1)

    ea = EventAccumulator(logdir)
    ea.Reload()

    available_tags = ea.Tags().get("scalars", [])
    if tag not in available_tags:
        return []

    events = ea.Scalars(tag)
    return [(e.step, e.value) for e in sorted(events, key=lambda e: e.step)]


def moving_average(values: list[float], window: int = 3) -> list[float]:
    """Compute simple moving average with given window size."""
    if len(values) < window:
        return values[:]
    result = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        result.append(sum(values[start : i + 1]) / (i - start + 1))
    return result


def best_metric(logdir: str, tag: str) -> float | None:
    """Extract the peak 3-step moving average of a metric."""
    series = read_scalar(logdir, tag)
    if not series:
        return None
    values = [v for _, v in series]
    ma = moving_average(values, window=3)
    return max(ma) if ma else None


# ---------------------------------------------------------------------------
# Phase result extraction
# ---------------------------------------------------------------------------

@dataclass
class ExperimentResult:
    """Result for one experiment."""

    name: str
    base_id: str
    metric_value: float | None
    hp_values: dict[str, str]  # decoded HP name → actual value


def discover_experiments(
    phase: str,
    base: BaseConfig,
    tb_root: str,
) -> list[str]:
    """Find experiment directories matching a phase + base config."""
    pattern = f"hp-{base.label}-{phase}-"
    experiments = []
    tb_path = Path(tb_root)
    if not tb_path.exists():
        return experiments
    for entry in tb_path.iterdir():
        if entry.is_dir() and entry.name.startswith(pattern):
            trial_dir = entry / base.trial
            if trial_dir.exists():
                experiments.append(entry.name)
    return sorted(experiments)


def extract_hp_values(
    exp_name: str,
    phase: str,
) -> dict[str, str]:
    """Parse HP values from experiment name."""
    patterns = PHASE_HP_PATTERNS.get(phase, [])
    result: dict[str, str] = {}
    for hp_name, regex, _ in patterns:
        match = re.search(regex, exp_name)
        if match:
            encoded = match.group(hp_name)
            result[hp_name] = decode_hp(hp_name, encoded)
    return result


def extract_phase_results(
    phase: str,
    deepscaler_tb_root: str,
    deepcoder_tb_root: str,
) -> list[ExperimentResult]:
    """Extract results for all experiments in a phase."""
    results: list[ExperimentResult] = []

    for base in BASE_CONFIGS:
        tb_root = (
            deepscaler_tb_root
            if base.benchmark == "deepscaler"
            else deepcoder_tb_root
        )

        experiments = discover_experiments(phase, base, tb_root)

        for exp_name in experiments:
            logdir = os.path.join(tb_root, exp_name, base.trial)
            metric_val = best_metric(logdir, base.metric)
            hp_values = extract_hp_values(exp_name, phase)
            results.append(
                ExperimentResult(
                    name=exp_name,
                    base_id=base.id,
                    metric_value=metric_val,
                    hp_values=hp_values,
                )
            )

    return results


def select_best_per_base(
    results: list[ExperimentResult],
) -> dict[str, ExperimentResult]:
    """Select the best experiment per base config (highest metric)."""
    best: dict[str, ExperimentResult] = {}
    for r in results:
        if r.metric_value is None:
            continue
        current = best.get(r.base_id)
        if current is None or (
            current.metric_value is not None
            and r.metric_value > current.metric_value
        ):
            best[r.base_id] = r
    return best


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def write_env_file(
    phase: str,
    best: dict[str, ExperimentResult],
    output_path: str,
    append: bool = False,
) -> None:
    """Write best HP values to env file."""
    lines: list[str] = []
    lines.append(f"# Phase {phase.upper()} best results")

    patterns = PHASE_HP_PATTERNS.get(phase, [])
    for base in BASE_CONFIGS:
        result = best.get(base.id)
        if result is None:
            lines.append(f"# WARNING: No results for {base.id}")
            continue

        base_upper = base.id.upper().replace("-", "_")
        for hp_name, _, env_suffix in patterns:
            value = result.hp_values.get(hp_name, "?")
            var_name = f"BEST_{phase.upper()}_{base_upper}_{env_suffix}"
            lines.append(f"{var_name}={value}")

        lines.append(
            f"# {base.id}: best={result.name} metric={result.metric_value:.4f}"
        )

    lines.append("")

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    mode = "a" if append else "w"
    with open(output_path, mode) as f:
        f.write("\n".join(lines) + "\n")

    print(f"{'Appended to' if append else 'Wrote'} {output_path}")


def write_csv(
    phase: str,
    results: list[ExperimentResult],
    output_dir: str,
) -> None:
    """Write per-phase CSV with all experiment results."""
    csv_path = os.path.join(output_dir, f"phase_{phase}_results.csv")
    os.makedirs(output_dir, exist_ok=True)

    patterns = PHASE_HP_PATTERNS.get(phase, [])
    hp_columns = [hp_name for hp_name, _, _ in patterns]

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["experiment", "base_config", "metric_value"] + hp_columns
        )
        for r in sorted(results, key=lambda x: (x.base_id, -(x.metric_value or 0))):
            row = [
                r.name,
                r.base_id,
                f"{r.metric_value:.4f}" if r.metric_value is not None else "N/A",
            ]
            for hp_name in hp_columns:
                row.append(r.hp_values.get(hp_name, "?"))
            writer.writerow(row)

    print(f"Wrote {csv_path} ({len(results)} experiments)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract best HPs from TensorBoard logs for the MARFT HP sweep."
    )
    parser.add_argument(
        "--phase",
        required=True,
        choices=list(PHASE_HP_PATTERNS.keys()),
        help="Phase to extract results for (p1a, p1b, p2a, p2b, p3a, p3b).",
    )
    parser.add_argument(
        "--deepscaler-tb-root",
        default="ecmlp_experiments_deepscaler/tensorboard",
        help="TensorBoard root for DeepScaleR experiments.",
    )
    parser.add_argument(
        "--deepcoder-tb-root",
        default="ecmlp_experiments_deepcoder/tensorboard",
        help="TensorBoard root for DeepCoder experiments.",
    )
    parser.add_argument(
        "--output",
        default="ecmlp_experiments/hp_sweep/phase_results.env",
        help="Path to the env file for best HP values.",
    )
    parser.add_argument(
        "--csv-dir",
        default="ecmlp_experiments/hp_sweep",
        help="Directory for per-phase CSV files.",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to existing env file instead of overwriting.",
    )
    parser.add_argument(
        "--csv-only",
        action="store_true",
        help="Only generate CSV, skip env file.",
    )
    args = parser.parse_args()

    print(f"Extracting Phase {args.phase.upper()} results...")

    results = extract_phase_results(
        phase=args.phase,
        deepscaler_tb_root=args.deepscaler_tb_root,
        deepcoder_tb_root=args.deepcoder_tb_root,
    )

    if not results:
        print("WARNING: No experiments found for this phase.", file=sys.stderr)
        sys.exit(1)

    found = sum(1 for r in results if r.metric_value is not None)
    missing = sum(1 for r in results if r.metric_value is None)
    print(f"Found {found} experiments with metrics, {missing} without.")

    # Print ranking per base config
    for base in BASE_CONFIGS:
        base_results = [r for r in results if r.base_id == base.id and r.metric_value is not None]
        if not base_results:
            print(f"\n  {base.id}: No results")
            continue
        base_results.sort(key=lambda r: -(r.metric_value or 0))
        print(f"\n  {base.id} ranking ({base.metric}):")
        for i, r in enumerate(base_results[:5], 1):
            marker = " <-- BEST" if i == 1 else ""
            print(f"    {i}. {r.name}: {r.metric_value:.4f}{marker}")

    # Write CSV
    write_csv(args.phase, results, args.csv_dir)

    # Write env file
    if not args.csv_only:
        best = select_best_per_base(results)
        write_env_file(args.phase, best, args.output, append=args.append)

    print("\nDone.")


if __name__ == "__main__":
    main()
