#!/usr/bin/env python
"""Recompute route cluster statistics from saved route artifacts.

This script mirrors the clustering path used by the batch runner, but allows
overriding the clustering max_clusters parameter.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import statistics
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from route_distances.clustering import ClusteringHelper
from route_distances.route_distances import route_distances_calculator


warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"
try:
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")
except Exception:
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir-a", required=True)
    parser.add_argument("--run-dir-b")
    parser.add_argument("--jsonl-name", default="CASPyrus10k.jsonl")
    parser.add_argument("--artifact-dir-name", default="CASPyrus10k_route_artifacts")
    parser.add_argument("--max-clusters", type=int, nargs="+", required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--subset",
        choices=["all_completed", "solved_only", "common_completed", "common_solved"],
        default="common_solved",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output")
    return parser.parse_args()


def load_jsonl(path: Path) -> Dict[int, dict]:
    rows: Dict[int, dict] = {}
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            rows[int(row["index"])] = row
    return rows


def select_indices(
    rows_a: Dict[int, dict],
    rows_b: Optional[Dict[int, dict]],
    subset: str,
) -> List[int]:
    if subset == "all_completed":
        return sorted(rows_a)
    if subset == "solved_only":
        return sorted(idx for idx, row in rows_a.items() if row["stats"]["is_solved"])
    if rows_b is None:
        raise ValueError(f"subset={subset} requires --run-dir-b")

    common = sorted(set(rows_a) & set(rows_b))
    if subset == "common_completed":
        return common
    if subset == "common_solved":
        return sorted(
            idx
            for idx in common
            if rows_a[idx]["stats"]["is_solved"] and rows_b[idx]["stats"]["is_solved"]
        )
    raise ValueError(f"Unsupported subset: {subset}")


def recompute_cluster_counts(
    artifact_path: Path, max_clusters_list: List[int]
) -> Dict[int, Optional[int]]:
    with gzip.open(artifact_path, "rt") as handle:
        payload = json.load(handle)
    routes = payload.get("routes", [])
    if len(routes) < 3:
        return {max_clusters: None for max_clusters in max_clusters_list}
    calculator = route_distances_calculator("ted", content="both", timeout=None)
    distances = calculator(routes)
    results: Dict[int, Optional[int]] = {}
    for max_clusters in max_clusters_list:
        labels = ClusteringHelper.cluster(distances, 0, max_clusters=max_clusters)
        labels_list = labels.tolist() if hasattr(labels, "tolist") else list(labels)
        results[max_clusters] = (
            len({int(label) for label in labels_list}) if labels_list else None
        )
    return results


def summarize(values: Iterable[int]) -> dict:
    seq = list(values)
    if not seq:
        return {
            "n_with_cluster_count": 0,
            "avg_route_clusters": None,
            "median_route_clusters": None,
            "min_route_clusters": None,
            "max_route_clusters": None,
        }
    return {
        "n_with_cluster_count": len(seq),
        "avg_route_clusters": statistics.mean(seq),
        "median_route_clusters": statistics.median(seq),
        "min_route_clusters": min(seq),
        "max_route_clusters": max(seq),
    }


def _compute_one(
    params: Tuple[int, str, List[int]],
) -> Tuple[int, Dict[int, Optional[int]]]:
    idx, artifact_path_str, max_clusters_list = params
    return idx, recompute_cluster_counts(Path(artifact_path_str), max_clusters_list)


def collect_counts(
    rows: Dict[int, dict],
    artifact_dir: Path,
    indices: List[int],
    max_clusters_list: List[int],
    workers: int,
) -> dict:
    stored_counts: List[int] = []
    recomputed_counts: Dict[int, List[int]] = {maxc: [] for maxc in max_clusters_list}
    used_indices: Dict[int, List[int]] = {maxc: [] for maxc in max_clusters_list}

    for idx in indices:
        row = rows[idx]
        stored = row.get("route_cluster_count")
        if stored is not None:
            stored_counts.append(int(stored))

    work_items = []
    for idx in indices:
        artifact_path = artifact_dir / f"{idx:05d}.json.gz"
        if artifact_path.exists():
            work_items.append((idx, str(artifact_path), max_clusters_list))

    if workers <= 1:
        computed = [_compute_one(item) for item in work_items]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            computed = list(executor.map(_compute_one, work_items))

    for idx, counts in computed:
        for max_clusters, value in counts.items():
            if value is None:
                continue
            used_indices[max_clusters].append(idx)
            recomputed_counts[max_clusters].append(value)

    return {
        "n_indices_requested": len(indices),
        "stored_from_jsonl": summarize(stored_counts),
        "recomputed": {
            str(max_clusters): {
                "n_indices_used": len(used_indices[max_clusters]),
                "used_indices_sample": used_indices[max_clusters][:20],
                **summarize(recomputed_counts[max_clusters]),
            }
            for max_clusters in max_clusters_list
        },
    }


def main() -> None:
    args = parse_args()
    run_dir_a = Path(args.run_dir_a)
    run_dir_b = Path(args.run_dir_b) if args.run_dir_b else None

    rows_a = load_jsonl(run_dir_a / args.jsonl_name)
    rows_b = load_jsonl(run_dir_b / args.jsonl_name) if run_dir_b else None
    indices = select_indices(rows_a, rows_b, args.subset)
    if args.limit:
        indices = indices[: args.limit]

    result = {
        "run_dir_a": str(run_dir_a),
        "run_dir_b": str(run_dir_b) if run_dir_b else None,
        "subset": args.subset,
        "jsonl_name": args.jsonl_name,
        "artifact_dir_name": args.artifact_dir_name,
        "n_indices_selected": len(indices),
        "max_clusters_results": {},
    }

    artifact_dir_a = run_dir_a / args.artifact_dir_name
    artifact_dir_b = run_dir_b / args.artifact_dir_name if run_dir_b else None

    run_a = collect_counts(rows_a, artifact_dir_a, indices, args.max_clusters, args.workers)
    run_b = (
        collect_counts(rows_b, artifact_dir_b, indices, args.max_clusters, args.workers)
        if rows_b and artifact_dir_b
        else None
    )

    for max_clusters in args.max_clusters:
        entry = {
            "run_a": {
                "n_indices_requested": run_a["n_indices_requested"],
                "stored_from_jsonl": run_a["stored_from_jsonl"],
                "recomputed": run_a["recomputed"][str(max_clusters)],
            }
        }
        if run_b:
            entry["run_b"] = {
                "n_indices_requested": run_b["n_indices_requested"],
                "stored_from_jsonl": run_b["stored_from_jsonl"],
                "recomputed": run_b["recomputed"][str(max_clusters)],
            }
        result["max_clusters_results"][str(max_clusters)] = entry

    output_text = json.dumps(result, indent=2, ensure_ascii=False)
    if args.output:
        Path(args.output).write_text(output_text)
    print(output_text)


if __name__ == "__main__":
    main()
