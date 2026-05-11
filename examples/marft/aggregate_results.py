"""Aggregate per-benchmark evaluation results into a cross-benchmark table.

Reads summary.csv files from each benchmark's eval_results directory and
joins them into a single table with one row per (method, checkpoint_step)
and columns for all 8 benchmarks.

Usage:
    python examples/marft/aggregate_results.py \
        --code-root ecmlp_experiments_deepcoder/eval_results \
        --math-root ecmlp_experiments_deepscaler/eval_results \
        --output results_table.csv

    # Filter to specific methods
    python examples/marft/aggregate_results.py \
        --code-root ecmlp_experiments_deepcoder/eval_results \
        --math-root ecmlp_experiments_deepscaler/eval_results \
        --filter "2agent-shared" \
        --output results_table.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CODE_BENCHMARKS = ["deepcoder", "livecodebench", "codeforces"]
MATH_BENCHMARKS = ["deepscaler", "aime2024", "math500", "minervamath", "olympiadbench"]
ALL_BENCHMARKS = CODE_BENCHMARKS + MATH_BENCHMARKS

BENCHMARK_ABBREV = {
    "deepcoder": "DC",
    "livecodebench": "LCB",
    "codeforces": "CF",
    "deepscaler": "DS",
    "aime2024": "AIME",
    "math500": "M500",
    "minervamath": "MM",
    "olympiadbench": "OB",
}

# Method sort order: baselines first, then by (n_agents, lora, critic)
_LORA_ORDER = {"none": 0, "shared": 1, "peragent": 2}
_CRITIC_ORDER = {"baseline": 0, "ctde": 1, "multihead": 2, "criticlora": 3}


# ---------------------------------------------------------------------------
# Method parsing
# ---------------------------------------------------------------------------


@dataclass
class MethodInfo:
    """Parsed method descriptor from an experiment name."""

    n_agents: int
    lora_mode: str  # "none" | "shared" | "peragent"
    critic_mode: str  # "baseline" | "ctde" | "multihead" | "criticlora"
    label: str  # human-readable label

    @property
    def sort_key(self) -> tuple:
        return (
            self.n_agents,
            _LORA_ORDER.get(self.lora_mode, 99),
            _CRITIC_ORDER.get(self.critic_mode, 99),
        )


def parse_method(experiment_name: str, step_dir: str) -> MethodInfo | None:
    """Parse experiment name into a method descriptor.

    Returns None for anonymous experiments (should be skipped).
    """
    if "anonymous" in experiment_name:
        return None

    # Baseline: step_dir == "base" means no training (base model eval)
    if step_dir == "base":
        m = re.search(r"(\d)agent", experiment_name)
        n_agents = int(m.group(1)) if m else 1
        label = f"{n_agents}-agent baseline"
        return MethodInfo(
            n_agents=n_agents,
            lora_mode="none",
            critic_mode="baseline",
            label=label,
        )

    # Trained checkpoint: parse lora_mode and critic_mode
    m = re.search(
        r"(\d)agent-(shared|peragent)-(ctde|criticlora|multihead)",
        experiment_name,
    )
    if not m:
        # Single-agent or unrecognized pattern
        m_single = re.search(r"(\d)agent", experiment_name)
        if m_single:
            n_agents = int(m_single.group(1))
            if n_agents == 1:
                label = "1-agent"
                return MethodInfo(
                    n_agents=1,
                    lora_mode="shared",
                    critic_mode="ctde",
                    label=label,
                )
        return None

    n_agents = int(m.group(1))
    lora_mode = m.group(2)
    critic_mode = m.group(3)
    label = f"{n_agents}-agent {lora_mode}-{critic_mode}"
    return MethodInfo(
        n_agents=n_agents,
        lora_mode=lora_mode,
        critic_mode=critic_mode,
        label=label,
    )


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkRow:
    """A single row from a benchmark's summary.csv."""

    experiment: str
    step_dir: str
    global_step: int
    reward_mean: float
    reward_std: float
    n_seeds: int
    n_samples_per_seed: int
    seed_values: dict[int, float]  # {seed_index: avg_reward}


def load_summary(summary_path: str) -> list[BenchmarkRow]:
    """Load a per-benchmark summary.csv into BenchmarkRow objects."""
    rows: list[BenchmarkRow] = []
    if not os.path.exists(summary_path):
        return rows

    with open(summary_path, newline="") as f:
        reader = csv.DictReader(f)
        for record in reader:
            # Parse per-seed columns (seed_0, seed_1, ...)
            seed_values: dict[int, float] = {}
            for key, val in record.items():
                if key.startswith("seed_") and val not in ("", None):
                    seed_idx = int(key.split("_")[1])
                    try:
                        seed_values[seed_idx] = float(val)
                    except ValueError:
                        pass

            try:
                rows.append(
                    BenchmarkRow(
                        experiment=record["experiment"],
                        step_dir=record["step_dir"],
                        global_step=int(record["global_step"]),
                        reward_mean=float(record["reward_mean"]),
                        reward_std=float(record["reward_std"]),
                        n_seeds=int(record.get("n_seeds", 0)),
                        n_samples_per_seed=int(
                            record.get("n_samples_per_seed", 0)
                        ),
                        seed_values=seed_values,
                    )
                )
            except (KeyError, ValueError) as e:
                print(f"WARNING: Skipping malformed row in {summary_path}: {e}")

    return rows


def load_all_benchmarks(
    code_root: str, math_root: str
) -> dict[str, list[BenchmarkRow]]:
    """Load summary.csv for all benchmarks from both group roots."""
    data: dict[str, list[BenchmarkRow]] = {}

    for bm in CODE_BENCHMARKS:
        path = os.path.join(code_root, bm, "summary.csv")
        rows = load_summary(path)
        if rows:
            data[bm] = rows
            print(f"  Loaded {len(rows)} rows from {path}")
        else:
            print(f"  No data found at {path}")

    for bm in MATH_BENCHMARKS:
        path = os.path.join(math_root, bm, "summary.csv")
        rows = load_summary(path)
        if rows:
            data[bm] = rows
            print(f"  Loaded {len(rows)} rows from {path}")
        else:
            print(f"  No data found at {path}")

    return data


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


@dataclass
class AggregateRow:
    """One row in the final cross-benchmark table."""

    method: MethodInfo
    experiment: str
    step_dir: str
    global_step: int
    benchmark_data: dict[str, BenchmarkRow | None]  # benchmark → data or None


def build_aggregate_table(
    all_data: dict[str, list[BenchmarkRow]],
    filter_pattern: str | None = None,
) -> list[AggregateRow]:
    """Join per-benchmark results into a single table.

    Every (experiment, step_dir, global_step) combination gets one row.
    """
    # Collect all unique (experiment, step_dir, global_step) keys
    # and their per-benchmark data
    key_benchmarks: dict[
        tuple[str, str, int], dict[str, BenchmarkRow]
    ] = defaultdict(dict)

    for bm, rows in all_data.items():
        for row in rows:
            key = (row.experiment, row.step_dir, row.global_step)
            key_benchmarks[key][bm] = row

    # Build aggregate rows
    aggregate: list[AggregateRow] = []
    for (experiment, step_dir, global_step), bm_data in key_benchmarks.items():
        # Apply filter
        if filter_pattern and filter_pattern not in experiment:
            continue

        method = parse_method(experiment, step_dir)
        if method is None:
            continue  # skip anonymous or unparseable experiments

        aggregate.append(
            AggregateRow(
                method=method,
                experiment=experiment,
                step_dir=step_dir,
                global_step=global_step,
                benchmark_data={
                    bm: bm_data.get(bm) for bm in ALL_BENCHMARKS
                },
            )
        )

    # Sort by method definition order, then by global_step
    aggregate.sort(key=lambda r: (r.method.sort_key, r.global_step))

    return aggregate


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def determine_max_seeds(all_data: dict[str, list[BenchmarkRow]]) -> int:
    """Find the maximum seed index across all benchmark data."""
    max_seed = 0
    for rows in all_data.values():
        for row in rows:
            if row.seed_values:
                max_seed = max(max_seed, max(row.seed_values.keys()))
    return max_seed + 1  # 0-indexed → count


def write_csv(
    rows: list[AggregateRow],
    output_path: str,
    n_seeds: int,
) -> None:
    """Write the aggregate table to CSV."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # Build column headers
    fieldnames = [
        "method",
        "n_agents",
        "lora_mode",
        "critic_mode",
        "experiment",
        "step_dir",
        "global_step",
    ]

    for bm in ALL_BENCHMARKS:
        fieldnames.append(f"{bm}_mean")
        fieldnames.append(f"{bm}_std")
        for s in range(n_seeds):
            fieldnames.append(f"{bm}_seed_{s}")

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()

        for row in rows:
            record: dict[str, object] = {
                "method": row.method.label,
                "n_agents": row.method.n_agents,
                "lora_mode": row.method.lora_mode,
                "critic_mode": row.method.critic_mode,
                "experiment": row.experiment,
                "step_dir": row.step_dir,
                "global_step": row.global_step,
            }

            for bm in ALL_BENCHMARKS:
                bm_row = row.benchmark_data.get(bm)
                if bm_row is not None:
                    record[f"{bm}_mean"] = round(bm_row.reward_mean, 6)
                    record[f"{bm}_std"] = round(bm_row.reward_std, 6)
                    for s in range(n_seeds):
                        val = bm_row.seed_values.get(s)
                        if val is not None:
                            record[f"{bm}_seed_{s}"] = round(val, 6)
                        else:
                            record[f"{bm}_seed_{s}"] = ""
                else:
                    record[f"{bm}_mean"] = ""
                    record[f"{bm}_std"] = ""
                    for s in range(n_seeds):
                        record[f"{bm}_seed_{s}"] = ""

            writer.writerow(record)

    print(f"\nCSV written to {output_path} ({len(rows)} rows)")


def print_markdown_table(rows: list[AggregateRow]) -> None:
    """Print a compact markdown table with mean values only."""
    abbrevs = [BENCHMARK_ABBREV[bm] for bm in ALL_BENCHMARKS]

    # Header
    header = "| Method | Step | " + " | ".join(abbrevs) + " |"
    sep = "|" + "|".join(
        ["-" * (max(len("Method"), 25) + 2)]
        + ["-" * 7]
        + ["-" * (max(len(a), 6) + 2) for a in abbrevs]
    ) + "|"

    print("\n" + header)
    print(sep)

    for row in rows:
        step_str = str(row.global_step) if row.step_dir != "base" else "base"
        cols = [f" {row.method.label:<24}", f" {step_str:>5} "]

        for bm in ALL_BENCHMARKS:
            bm_row = row.benchmark_data.get(bm)
            if bm_row is not None:
                cols.append(f" {bm_row.reward_mean:>6.4f} ")
            else:
                cols.append(f" {'':>6} ")

        print("|" + "|".join(cols) + "|")

    print("")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate MARFT eval results into a cross-benchmark table",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--code-root",
        default="ecmlp_experiments_deepcoder/eval_results",
        help="Root directory for code group eval results "
        "(default: ecmlp_experiments_deepcoder/eval_results)",
    )
    parser.add_argument(
        "--math-root",
        default="ecmlp_experiments_deepscaler/eval_results",
        help="Root directory for math group eval results "
        "(default: ecmlp_experiments_deepscaler/eval_results)",
    )
    parser.add_argument(
        "--output",
        default="results_table.csv",
        help="Output CSV file path (default: results_table.csv)",
    )
    parser.add_argument(
        "--filter",
        default=None,
        help="Filter experiments by substring (e.g., '2agent-shared')",
    )
    parser.add_argument(
        "--no-markdown",
        action="store_true",
        help="Skip printing the markdown table to stdout",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("Loading benchmark results...")
    all_data = load_all_benchmarks(args.code_root, args.math_root)

    if not all_data:
        print("No benchmark data found. Check --code-root and --math-root paths.")
        sys.exit(1)

    n_seeds = determine_max_seeds(all_data)
    print(f"\nMax seeds across benchmarks: {n_seeds}")

    print("Building aggregate table...")
    rows = build_aggregate_table(all_data, filter_pattern=args.filter)

    if not rows:
        print("No rows after filtering. Check experiment names and --filter.")
        sys.exit(1)

    print(f"Aggregate table: {len(rows)} rows")

    # Write CSV with full per-seed data
    write_csv(rows, args.output, n_seeds)

    # Print compact markdown table
    if not args.no_markdown:
        print_markdown_table(rows)


if __name__ == "__main__":
    main()
