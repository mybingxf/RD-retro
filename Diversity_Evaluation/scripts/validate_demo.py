#!/usr/bin/env python3
"""Validate the demo's key outputs against the included reference values."""

import argparse
import csv
import json
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_PATH = PACKAGE_ROOT / "demo" / "expected_output" / "expected_metrics.json"


def read_metric_table(path, key_column, value_column):
    values = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            values[str(row[key_column])] = row[value_column]
    return values


def close(observed, expected, tolerance):
    return abs(float(observed) - float(expected)) <= tolerance


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir")
    parser.add_argument("--tolerance", type=float, default=1e-6)
    args = parser.parse_args()

    output_root = Path(args.output_dir).resolve()
    run_dir = output_root / "roundtrip_input_top10"
    expected = json.loads(EXPECTED_PATH.read_text(encoding="utf-8"))

    required = [
        "group_acc.tsv",
        "exact_match_group_acc.tsv",
        "group_reactant_similarity_top5.tsv",
        "group_reactant_similarity_top5.summary.tsv",
        "roundtrip_detail.tsv",
        "pairs.tsv",
        "summary.tsv",
        "mapped_and_sig.jsonl",
        "global_metrics.txt",
    ]
    missing = [name for name in required if not (run_dir / name).is_file()]
    if missing:
        raise RuntimeError(f"Missing expected demo outputs: {missing}")

    group_acc = read_metric_table(run_dir / "group_acc.tsv", "K", "topk_acc")
    exact_acc = read_metric_table(
        run_dir / "exact_match_group_acc.tsv", "K", "topk_acc"
    )
    reactant_summary = read_metric_table(
        run_dir / "group_reactant_similarity_top5.summary.tsv", "metric", "value"
    )

    checks = {
        "combined_top1_accuracy": group_acc["1"],
        "combined_top5_accuracy": group_acc["5"],
        "combined_top10_accuracy": group_acc["10"],
        "exact_top1_accuracy": exact_acc["1"],
        "exact_top10_accuracy": exact_acc["10"],
        "reactant_all_mean": reactant_summary["all_mean_avg_over_groups"],
        "reactant_all_min": reactant_summary["all_min_avg_over_groups"],
        "reactant_hit_mean": reactant_summary["hit_mean_avg_over_groups"],
    }

    failures = []
    for name, observed in checks.items():
        reference = expected[name]
        status = "PASS" if close(observed, reference, args.tolerance) else "FAIL"
        print(f"{status} {name}: observed={observed}, expected={reference}")
        if status == "FAIL":
            failures.append(name)

    if failures:
        raise RuntimeError(f"Demo validation failed: {failures}")
    print("DEMO VALIDATION: PASS")


if __name__ == "__main__":
    main()
