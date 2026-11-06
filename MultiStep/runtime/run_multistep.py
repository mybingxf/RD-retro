from __future__ import annotations

import argparse
import contextlib
import csv
import gzip
import json
import multiprocessing
import os
import statistics
import sys
import time
from collections import Counter
from multiprocessing import get_context
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import yaml
from tqdm import tqdm

FINDER = None
SEARCH_STATS = None
WORKER_OPTIONS: Dict[str, Any] = {}
WORKER_PROGRESS_PATH: Optional[Path] = None


def _detect_smiles(path: Path) -> List[Tuple[int, str]]:
    if path.suffix.lower() == ".csv":
        with path.open() as fileobj:
            reader = csv.DictReader(fileobj)
            if not reader.fieldnames:
                raise ValueError(f"No CSV header found in {path}")
            field_lookup = {name.lower(): name for name in reader.fieldnames}
            smiles_key = field_lookup.get("smiles")
            if not smiles_key:
                raise ValueError(f"No smiles column found in {path}")
            return [
                (idx, row[smiles_key].strip())
                for idx, row in enumerate(reader, start=1)
                if row.get(smiles_key, "").strip()
            ]

    with path.open() as fileobj:
        lines = [line.strip() for line in fileobj if line.strip()]
    if lines and lines[0].lower() == "smiles":
        lines = lines[1:]
    return list(enumerate(lines, start=1))


def _load_config(config_path: Path, use_gpu: bool, gpu_device: str | None = None) -> Dict:
    with config_path.open() as fileobj:
        config = yaml.safe_load(fileobj)
    for _, policy_cfg in config.get("policy", {}).get("smiles-based", {}).items():
        if not isinstance(policy_cfg, dict):
            continue
        policy_cfg["use_gpu"] = use_gpu
        if gpu_device:
            policy_cfg["gpu"] = gpu_device
    return config


def _init_worker(
    config_path: str,
    use_gpu: bool,
    policy_name: str,
    stock_name: str,
    gpu_devices: List[str],
    output_dir: str,
    dataset_tag: str,
    save_route_artifacts: bool,
    cluster_routes: bool,
    route_distance_model: Optional[str],
    save_distance_matrix: bool,
    live_progress: bool,
    live_progress_interval_seconds: float,
) -> None:
    global FINDER
    global SEARCH_STATS
    global WORKER_OPTIONS
    global WORKER_PROGRESS_PATH

    from aizynthfinder.aizynthfinder import AiZynthFinder

    sys.stdout = open(os.devnull, "w")

    worker_index = max(0, multiprocessing.current_process()._identity[0] - 1)
    gpu_device = None
    if use_gpu and gpu_devices:
        gpu_device = gpu_devices[worker_index % len(gpu_devices)]

    config = _load_config(Path(config_path), use_gpu, gpu_device=gpu_device)
    FINDER = AiZynthFinder(configdict=config)
    FINDER.expansion_policy.select(policy_name)
    FINDER.stock.select(stock_name)
    SEARCH_STATS = {"use_gpu": use_gpu, "gpu_device": gpu_device}
    WORKER_OPTIONS = {
        "policy_name": policy_name,
        "output_dir": output_dir,
        "dataset_tag": dataset_tag,
        "save_route_artifacts": save_route_artifacts,
        "cluster_routes": cluster_routes,
        "route_distance_model": route_distance_model,
        "save_distance_matrix": save_distance_matrix,
        "live_progress": live_progress,
        "live_progress_interval_seconds": live_progress_interval_seconds,
    }
    if live_progress:
        progress_dir = Path(output_dir) / f"{dataset_tag}_live_progress"
        progress_dir.mkdir(parents=True, exist_ok=True)
        WORKER_PROGRESS_PATH = progress_dir / f"worker_{os.getpid()}.json"
    else:
        WORKER_PROGRESS_PATH = None


def _write_worker_progress(payload: Dict[str, Any]) -> None:
    if not WORKER_PROGRESS_PATH:
        return
    tmp_path = WORKER_PROGRESS_PATH.with_suffix(".tmp")
    with tmp_path.open("w", encoding="utf-8") as fileobj:
        json.dump(payload, fileobj, ensure_ascii=False, indent=2)
    tmp_path.replace(WORKER_PROGRESS_PATH)


def _route_artifact_path(index: int) -> Path:
    artifact_dir = (
        Path(WORKER_OPTIONS["output_dir"]) / f"{WORKER_OPTIONS['dataset_tag']}_route_artifacts"
    )
    artifact_dir.mkdir(parents=True, exist_ok=True)
    return artifact_dir / f"{index:05d}.json.gz"


def _compute_route_clusters() -> Dict[str, Any]:
    if not WORKER_OPTIONS.get("cluster_routes"):
        return {}

    kwargs: Dict[str, Any]
    if WORKER_OPTIONS.get("route_distance_model"):
        kwargs = {
            "distances_model": "lstm",
            "model_path": WORKER_OPTIONS["route_distance_model"],
        }
        matrix_kwargs = {
            "model": "lstm",
            "model_path": WORKER_OPTIONS["route_distance_model"],
        }
    else:
        kwargs = {"distances_model": "ted"}
        matrix_kwargs = {"model": "ted"}

    time0 = time.perf_counter()
    labels = FINDER.routes.cluster(n_clusters=0, **kwargs)  # type: ignore[arg-type]
    cluster_time = time.perf_counter() - time0
    labels_list = labels.tolist() if hasattr(labels, "tolist") else list(labels)
    labels_list = [int(label) for label in labels_list]
    cluster_sizes = dict(sorted(Counter(labels_list).items())) if labels_list else {}

    payload: Dict[str, Any] = {
        "route_cluster_labels": labels_list,
        "route_cluster_count": len(cluster_sizes) if labels_list else None,
        "route_cluster_sizes": cluster_sizes if labels_list else None,
        "route_cluster_time_s": cluster_time,
    }

    if WORKER_OPTIONS.get("save_distance_matrix"):
        payload["route_distance_matrix"] = FINDER.routes.distance_matrix(  # type: ignore[arg-type]
            **matrix_kwargs
        ).tolist()

    return payload


def _compute_route_step_metrics() -> Dict[str, Any]:
    reaction_trees = getattr(FINDER.routes, "reaction_trees", [])
    solved_step_counts = [
        len(list(route.reactions()))
        for route in reaction_trees
        if getattr(route, "is_solved", False)
    ]
    if not solved_step_counts:
        return {"shortest_route_steps": None}
    return {"shortest_route_steps": min(solved_step_counts)}


def _save_route_artifact(index: int, smiles: str, cluster_payload: Dict[str, Any]) -> str:
    artifact_path = _route_artifact_path(index)
    route_dicts = FINDER.routes.dict_with_scores()
    artifact = {
        "index": index,
        "smiles": smiles,
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
        "route_count": len(route_dicts),
        "routes": route_dicts,
    }
    if cluster_payload:
        artifact.update(cluster_payload)
    with gzip.open(artifact_path, "wt", encoding="utf-8") as fileobj:
        json.dump(artifact, fileobj, ensure_ascii=False)
    return str(artifact_path)


def _run_one(task: Tuple[int, str]) -> Dict:
    global FINDER
    index, smiles = task

    try:
        strategy = FINDER.expansion_policy[WORKER_OPTIONS["policy_name"]]
        policy_wrapper = getattr(strategy, "model", None)
        policy_impl = getattr(policy_wrapper, "model", None)
        if policy_impl and hasattr(policy_impl, "reset_call_telemetry"):
            policy_impl.reset_call_telemetry()

        with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
            FINDER.target_smiles = smiles
            FINDER.prepare_tree()
            started_at = time.time()
            last_progress_write = {"t": 0.0}

            _write_worker_progress(
                {
                    "status": "running",
                    "worker_pid": os.getpid(),
                    "index": index,
                    "smiles": smiles,
                    "iterations": 0,
                    "expansion_calls": 0,
                    "elapsed_time_s": 0.0,
                    "updated_at": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
                }
            )

            def _progress_callback(progress: Dict[str, Any]) -> None:
                if not WORKER_OPTIONS.get("live_progress"):
                    return
                now = time.time()
                interval = float(WORKER_OPTIONS.get("live_progress_interval_seconds", 5.0))
                if progress.get("iterations", 0) > 0 and now - last_progress_write["t"] < interval:
                    return
                last_progress_write["t"] = now
                _write_worker_progress(
                    {
                        "status": "running",
                        "worker_pid": os.getpid(),
                        "index": index,
                        "smiles": smiles,
                        "iterations": progress.get("iterations", 0),
                        "expansion_calls": progress.get("expansion_calls", 0),
                        "reactants_generations": progress.get("reactants_generations", 0),
                        "elapsed_time_s": progress.get("elapsed_time_s", now - started_at),
                        "is_solved_so_far": progress.get("is_solved", False),
                        "first_solution_iteration": progress.get("first_solution_iteration", 0),
                        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
                    }
                )

            search_time = FINDER.tree_search(
                show_progress=False, progress_callback=_progress_callback
            )
            FINDER.build_routes()
            stats = FINDER.extract_statistics()
            route_step_payload = _compute_route_step_metrics()
            cluster_payload = _compute_route_clusters()
            artifact_path = None
            if WORKER_OPTIONS.get("save_route_artifacts"):
                artifact_path = _save_route_artifact(
                    index, smiles, {**route_step_payload, **cluster_payload}
                )
        profiling = stats.get("profiling", {})
        model_calls = profiling.get("expansion_calls", FINDER.search_stats.get("iterations", 0))
        row = {
            "index": index,
            "smiles": smiles,
            "ok": True,
            "stats": stats,
            "search_time": search_time,
            "model_calls": model_calls,
            "routes_found": len(FINDER.routes),
        }
        if cluster_payload:
            row.update(cluster_payload)
        row.update(route_step_payload)
        if artifact_path:
            row["route_artifact_path"] = artifact_path
        if policy_impl and hasattr(policy_impl, "get_call_telemetry"):
            telemetry = policy_impl.get_call_telemetry()
            if telemetry:
                row.update(telemetry)
        _write_worker_progress(
            {
                "status": "complete",
                "worker_pid": os.getpid(),
                "index": index,
                "smiles": smiles,
                "iterations": FINDER.search_stats.get("iterations", 0),
                "expansion_calls": model_calls,
                "elapsed_time_s": search_time,
                "is_solved": stats.get("is_solved", False),
                "updated_at": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
            }
        )
        return row
    except Exception as err:
        _write_worker_progress(
            {
                "status": "error",
                "worker_pid": os.getpid(),
                "index": index,
                "smiles": smiles,
                "error": repr(err),
                "updated_at": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
            }
        )
        return {
            "index": index,
            "smiles": smiles,
            "ok": False,
            "error": repr(err),
        }


def _read_completed(results_path: Path) -> Dict[int, Dict]:
    completed: Dict[int, Dict] = {}
    if not results_path.exists():
        return completed
    with results_path.open() as fileobj:
        for line in fileobj:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            completed[row["index"]] = row
    return completed


def _summarize(rows: Iterable[Dict]) -> Dict:
    rows = list(rows)
    ok_rows = [row for row in rows if row.get("ok")]
    solved_rows = [row for row in ok_rows if row["stats"].get("is_solved")]
    model_calls = [row["model_calls"] for row in ok_rows]
    search_times = [row["search_time"] for row in ok_rows]
    solved_routes = [row["stats"].get("number_of_solved_routes", 0) for row in ok_rows]
    shortest_route_steps = [
        row["shortest_route_steps"]
        for row in ok_rows
        if row.get("shortest_route_steps") is not None
    ]
    route_cluster_counts = [
        row["route_cluster_count"]
        for row in ok_rows
        if row.get("route_cluster_count") is not None
    ]
    route_cluster_times = [
        row["route_cluster_time_s"]
        for row in ok_rows
        if row.get("route_cluster_time_s") is not None
    ]
    mix_unique_counts = [
        row["mix_avg_unique_candidates_per_call"]
        for row in ok_rows
        if row.get("mix_avg_unique_candidates_per_call") is not None
    ]
    mix_output_counts = [
        row["mix_avg_output_candidates_per_call"]
        for row in ok_rows
        if row.get("mix_avg_output_candidates_per_call") is not None
    ]
    mix_dup_counts = [
        row["mix_avg_duplicates_collapsed_per_call"]
        for row in ok_rows
        if row.get("mix_avg_duplicates_collapsed_per_call") is not None
    ]

    summary = {
        "completed": len(rows),
        "ok": len(ok_rows),
        "failed": len(rows) - len(ok_rows),
        "solved": len(solved_rows),
        "success_rate": (len(solved_rows) / len(ok_rows)) if ok_rows else 0.0,
        "avg_search_time_s": statistics.mean(search_times) if search_times else 0.0,
        "avg_model_calls": statistics.mean(model_calls) if model_calls else 0.0,
        "avg_solved_routes": statistics.mean(solved_routes) if solved_routes else 0.0,
    }
    if shortest_route_steps:
        summary["avg_shortest_route_steps"] = statistics.mean(shortest_route_steps)
    if route_cluster_counts:
        summary["avg_route_clusters"] = statistics.mean(route_cluster_counts)
        summary["median_route_clusters"] = statistics.median(route_cluster_counts)
    if route_cluster_times:
        summary["avg_route_cluster_time_s"] = statistics.mean(route_cluster_times)
    if mix_unique_counts:
        summary["avg_mix_unique_candidates_per_call"] = statistics.mean(mix_unique_counts)
    if mix_output_counts:
        summary["avg_mix_output_candidates_per_call"] = statistics.mean(mix_output_counts)
    if mix_dup_counts:
        summary["avg_mix_duplicates_collapsed_per_call"] = statistics.mean(mix_dup_counts)
    return summary


def _detect_gpu_devices(use_gpu: bool) -> List[str]:
    if not use_gpu:
        return []
    import torch

    if not torch.cuda.is_available():
        return []
    return [f"cuda:{idx}" for idx in range(torch.cuda.device_count())]


def _default_worker_count(use_gpu: bool, gpu_devices: List[str]) -> int:
    if use_gpu:
        return len(gpu_devices) or 1
    return max(1, min(8, os.cpu_count() or 1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--policy", default="localretro")
    parser.add_argument("--stock", default="buyables")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true", default=False)
    parser.add_argument("--use-gpu", choices=["auto", "true", "false"], default="auto")
    parser.add_argument("--save-route-artifacts", action="store_true", default=False)
    parser.add_argument("--cluster-routes", action="store_true", default=False)
    parser.add_argument("--route-distance-model")
    parser.add_argument("--save-distance-matrix", action="store_true", default=False)
    parser.add_argument("--live-progress", action="store_true", default=False)
    parser.add_argument("--live-progress-interval-seconds", type=float, default=5.0)
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    config_path = Path(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.use_gpu == "true":
        use_gpu = True
    elif args.use_gpu == "false":
        use_gpu = False
    else:
        import torch

        use_gpu = torch.cuda.is_available()

    all_tasks = _detect_smiles(dataset_path)
    if args.limit:
        all_tasks = all_tasks[: args.limit]

    dataset_tag = dataset_path.stem
    results_path = output_dir / f"{dataset_tag}.jsonl"
    summary_path = output_dir / f"{dataset_tag}_summary.json"

    completed = _read_completed(results_path) if args.resume else {}
    pending_tasks = [task for task in all_tasks if task[0] not in completed]

    gpu_devices = _detect_gpu_devices(use_gpu)
    workers = args.workers or _default_worker_count(use_gpu, gpu_devices)

    meta = {
        "dataset": str(dataset_path),
        "config": str(config_path),
        "use_gpu_requested": args.use_gpu,
        "use_gpu_effective": use_gpu,
        "gpu_devices": gpu_devices,
        "workers": workers,
        "total_targets": len(all_tasks),
        "remaining_targets": len(pending_tasks),
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
        "save_route_artifacts": args.save_route_artifacts,
        "cluster_routes": args.cluster_routes,
        "route_distance_model": args.route_distance_model,
        "save_distance_matrix": args.save_distance_matrix,
        "live_progress": args.live_progress,
        "live_progress_interval_seconds": args.live_progress_interval_seconds,
    }

    with summary_path.open("w") as fileobj:
        json.dump({"meta": meta, "aggregate": _summarize(completed.values())}, fileobj, indent=2)

    if not pending_tasks:
        print("No pending targets. Summary already up to date.")
        return

    ctx = get_context("spawn")
    file_mode = "a" if args.resume else "w"
    with results_path.open(file_mode) as results_file:
        with ctx.Pool(
            processes=workers,
            initializer=_init_worker,
            initargs=(
                str(config_path),
                use_gpu,
                args.policy,
                args.stock,
                gpu_devices,
                str(output_dir),
                dataset_tag,
                args.save_route_artifacts,
                args.cluster_routes,
                args.route_distance_model,
                args.save_distance_matrix,
                args.live_progress,
                args.live_progress_interval_seconds,
            ),
        ) as pool:
            pbar = tqdm(total=len(all_tasks), initial=len(completed), dynamic_ncols=True)
            for row in pool.imap_unordered(_run_one, pending_tasks):
                results_file.write(json.dumps(row, ensure_ascii=False) + "\n")
                results_file.flush()
                completed[row["index"]] = row
                aggregate = _summarize(completed.values())
                pbar.update(1)
                pbar.set_postfix(
                    solved=f"{aggregate['solved']}/{aggregate['ok']}",
                    sr=f"{aggregate['success_rate']:.3f}",
                    t=f"{aggregate['avg_search_time_s']:.1f}s",
                    calls=f"{aggregate['avg_model_calls']:.1f}",
                )
                with summary_path.open("w") as fileobj:
                    json.dump({"meta": meta, "aggregate": aggregate}, fileobj, indent=2)
            pbar.close()


if __name__ == "__main__":
    main()
