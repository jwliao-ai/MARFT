#!/usr/bin/env python3
"""Append remaining lcbv5 rows from large parquet files to existing JSONL.

The main preprocessing (via `python -m areal.dataset.deepcoder`) may skip
large lcbv5 parquet files (>100 MB) when running under memory-constrained
environments.  This script processes those files one row at a time and
appends the results to the existing train.jsonl / test.jsonl.

Usage (on a machine with >=16 GB free memory):
    python scripts/complete_lcbv5_preprocessing.py /path/to/DeepCoder-Preview-Dataset

This is idempotent — it tracks which parquet files have already been
processed by counting existing lcbv5 rows in the JSONL.
"""

import argparse
import gc
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from areal.dataset.deepcoder import (
    _iter_parquet_rows,
    _process_row,
)

MAX_TESTS = 50


def count_lcbv5_rows(jsonl_path):
    """Count how many lcbv5 rows are already in the JSONL file."""
    if not os.path.isfile(jsonl_path):
        return 0
    count = 0
    with open(jsonl_path) as f:
        for line in f:
            obj = json.loads(line)
            if obj.get("source") == "lcbv5":
                count += 1
    return count


def append_lcbv5(raw_dir, split, jsonl_name, threshold_mb=100):
    """Append lcbv5 rows from parquet files exceeding *threshold_mb*."""
    import pyarrow.parquet as pq

    jsonl_path = os.path.join(raw_dir, jsonl_name)
    existing = count_lcbv5_rows(jsonl_path)
    print(f"{jsonl_path}: {existing} lcbv5 rows already present")

    lcbv5_dir = os.path.join(raw_dir, "lcbv5")
    files = sorted(
        os.path.join(lcbv5_dir, f)
        for f in os.listdir(lcbv5_dir)
        if f.startswith(f"{split}-") and f.endswith(".parquet")
    )

    large_files = [
        f for f in files if os.path.getsize(f) / (1024 * 1024) > threshold_mb
    ]
    if not large_files:
        print(f"  No large files to process (threshold={threshold_mb} MB)")
        return

    added = 0
    with open(jsonl_path, "a") as out:
        for fpath in large_files:
            fsize_mb = os.path.getsize(fpath) / (1024 * 1024)
            pf = pq.ParquetFile(fpath)
            nrows = pf.metadata.num_rows
            print(
                f"  Processing {os.path.basename(fpath)} "
                f"({fsize_mb:.0f} MB, {nrows} rows)..."
            )
            for batch in pf.iter_batches(batch_size=1):
                cols = batch.schema.names
                row = {col: batch.column(col)[0].as_py() for col in cols}
                sample = _process_row(row, "lcbv5", MAX_TESTS)
                out.write(json.dumps(sample, ensure_ascii=False) + "\n")
                added += 1
                del row, sample
                gc.collect()
            gc.collect()
            print(f"    done ({added} rows added so far)")

    print(f"  Appended {added} lcbv5 rows to {jsonl_path}")
    print(f"  New total lcbv5 rows: {existing + added}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", help="Root directory of DeepCoder-Preview-Dataset")
    parser.add_argument(
        "--threshold-mb",
        type=int,
        default=100,
        help="Only process parquet files larger than this (default: 100)",
    )
    args = parser.parse_args()

    print("=== Completing lcbv5 train rows ===")
    append_lcbv5(args.path, "train", "train.jsonl", args.threshold_mb)

    print("\n=== Completing lcbv5 test rows ===")
    append_lcbv5(args.path, "test", "test.jsonl", args.threshold_mb)


if __name__ == "__main__":
    main()
