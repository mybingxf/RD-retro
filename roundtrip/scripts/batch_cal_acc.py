#!/usr/bin/env python3

import argparse
import csv
import os
import re
import subprocess
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import DefaultDict, Dict, Iterable, List, Optional, Tuple


DEFAULT_ROOT = Path("/root/autodl-fs/roundtripcal")
DEFAULT_CAL = DEFAULT_ROOT / "cal.py"
DEFAULT_CONDA = Path("/root/miniconda3/bin/conda")

OUTPUT_DIR_RE = re.compile(r"^\[I/O\] Output dir:\s*(.+?)\s*$", re.MULTILINE)
PARSE_STATS_RE = re.compile(
    r"^\[Parse\]\s+total=(?P<total>\d+),\s+used=(?P<used>\d+),\s+bad/empty=(?P<bad>\d+)\s*$",
    re.MULTILINE,
)
GROUP_COUNT_RE = re.compile(r"^\s*n_groups\s*=\s*(?P<n_groups>\d+)\s*$", re.MULTILINE)


def split_rxn_line(line: str) -> Optional[str]:
    line = line.strip()
    if not line or ">>" not in line:
        return None
    src, tgt = line.split(">>", 1)
    src = src.strip()
    tgt = tgt.strip()
    if not src or not tgt:
        return None
    return f"{src}>>{tgt}"


def safe_name(rel_path: str) -> str:
    return rel_path.replace("/", "__")


def count_total_and_valid(path: Path) -> Tuple[int, int]:
    total_lines = 0
    valid_lines = 0
    with path.open("r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            total_lines += 1
            if split_rxn_line(line) is not None:
                valid_lines += 1
    return total_lines, valid_lines


def discover_original_inputs(root: Path) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    inputs: List[Dict[str, object]] = []
    skipped: List[Dict[str, object]] = []

    for path in sorted(root.rglob("*10*.txt")):
        if "acc_runs" in path.parts:
            continue
        total_lines, valid_lines = count_total_and_valid(path)
        rel_path = str(path.relative_to(root))
        record = {
            "source_txt": rel_path,
            "source_path": str(path),
            "source_total_lines": total_lines,
            "source_valid_lines": valid_lines,
            "source_mod10": valid_lines % 10,
        }
        if valid_lines == 0:
            record["reason"] = "no_valid_src_tgt_lines"
            skipped.append(record)
        else:
            inputs.append(record)

    return inputs, skipped


def choose_run_dir(root: Path, label: Optional[str]) -> Path:
    parent = root / "acc_runs"
    parent.mkdir(parents=True, exist_ok=True)
    if label:
        run_dir = parent / label
        if run_dir.exists():
            raise FileExistsError(f"Run directory already exists: {run_dir}")
        run_dir.mkdir(parents=True, exist_ok=False)
        return run_dir

    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    run_dir = parent / f"batch_{stamp}"
    suffix = 1
    while run_dir.exists():
        run_dir = parent / f"batch_{stamp}_{suffix}"
        suffix += 1
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def write_tsv(path: Path, rows: Iterable[Dict[str, object]], fieldnames: List[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def empty_group_metrics() -> Dict[str, Optional[float]]:
    return {
        "top1_acc": None,
        "top3_acc": None,
        "top5_acc": None,
        "top10_acc": None,
        "rank1_acc": None,
        "rank3_acc": None,
        "rank5_acc": None,
        "rank10_acc": None,
        "avg_rankk_acc": None,
    }


def extract_metrics_from_rows(rows: Dict[int, Tuple[float, float]]) -> Dict[str, Optional[float]]:
    metrics = empty_group_metrics()
    for k, (topk, rankk) in rows.items():
        if k == 1:
            metrics["top1_acc"] = topk
            metrics["rank1_acc"] = rankk
        elif k == 3:
            metrics["top3_acc"] = topk
            metrics["rank3_acc"] = rankk
        elif k == 5:
            metrics["top5_acc"] = topk
            metrics["rank5_acc"] = rankk
        elif k == 10:
            metrics["top10_acc"] = topk
            metrics["rank10_acc"] = rankk
    if rows:
        metrics["avg_rankk_acc"] = sum(rankk for _, rankk in rows.values()) / len(rows)
    return metrics


def read_group_acc_rows(path: Path) -> Dict[int, Tuple[float, float]]:
    rows: Dict[int, Tuple[float, float]] = {}
    if not path.exists():
        return rows

    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            if row["K"] == "AVG_RANKK":
                continue
            try:
                k = int(row["K"])
                topk = float(row["topk_acc"])
                rankk = float(row["rankk_acc"])
            except ValueError:
                continue
            rows[k] = (topk, rankk)
    return rows


def build_shard_targets(valid_lines: int, group_size: int, shard_valid_lines: int) -> List[int]:
    if valid_lines <= shard_valid_lines:
        return [valid_lines]

    groups = valid_lines // group_size
    remainder = valid_lines % group_size
    groups_per_shard = max(1, shard_valid_lines // group_size)

    targets: List[int] = []
    groups_left = groups
    while groups_left > 0:
        take_groups = min(groups_per_shard, groups_left)
        targets.append(take_groups * group_size)
        groups_left -= take_groups

    if remainder:
        if targets:
            targets[-1] += remainder
        else:
            targets = [remainder]
    return targets


def stage_jobs(
    original_inputs: List[Dict[str, object]],
    run_dir: Path,
    group_size: int,
    split_threshold_valid_lines: int,
    shard_valid_lines: int,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    inputs_dir = run_dir / "inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)

    staged_jobs: List[Dict[str, object]] = []
    original_manifest: List[Dict[str, object]] = []

    for record in original_inputs:
        source_txt = str(record["source_txt"])
        source_path = Path(str(record["source_path"]))
        valid_lines = int(record["source_valid_lines"])
        shard_targets = (
            build_shard_targets(valid_lines, group_size, shard_valid_lines)
            if valid_lines > split_threshold_valid_lines
            else [valid_lines]
        )
        shard_count = len(shard_targets)

        safe_source = safe_name(source_txt)
        if shard_count == 1:
            stage_paths = [inputs_dir / f"{safe_source}.txt"]
        else:
            stage_dir = inputs_dir / safe_source
            stage_dir.mkdir(parents=True, exist_ok=True)
            stage_paths = [stage_dir / f"part_{idx:03d}.txt" for idx in range(shard_count)]

        writers = [path.open("w", encoding="utf-8") for path in stage_paths]
        try:
            shard_index = 0
            written_in_shard = 0
            current_target = shard_targets[shard_index]
            with source_path.open("r", encoding="utf-8", errors="ignore") as fh:
                for line in fh:
                    valid_line = split_rxn_line(line)
                    if valid_line is None:
                        continue
                    writers[shard_index].write(valid_line + "\n")
                    written_in_shard += 1
                    if written_in_shard == current_target and shard_index + 1 < shard_count:
                        shard_index += 1
                        written_in_shard = 0
                        current_target = shard_targets[shard_index]
        finally:
            for writer in writers:
                writer.close()

        total_staged = 0
        for idx, (stage_path, shard_lines) in enumerate(zip(stage_paths, shard_targets), start=1):
            total_staged += shard_lines
            staged_rel = str(stage_path.relative_to(run_dir))
            task_name = source_txt if shard_count == 1 else f"{source_txt}#part{idx}/{shard_count}"
            staged_jobs.append(
                {
                    "task_name": task_name,
                    "source_txt": source_txt,
                    "source_path": str(source_path),
                    "source_total_lines": int(record["source_total_lines"]),
                    "source_valid_lines": valid_lines,
                    "source_mod10": int(record["source_mod10"]),
                    "staged_input": str(stage_path),
                    "staged_input_rel": staged_rel,
                    "staged_valid_lines": shard_lines,
                    "shard_index": idx,
                    "shard_count": shard_count,
                    "is_shard": 1 if shard_count > 1 else 0,
                }
            )

        original_manifest.append(
            {
                "source_txt": source_txt,
                "source_path": str(source_path),
                "source_total_lines": int(record["source_total_lines"]),
                "source_valid_lines": valid_lines,
                "source_mod10": int(record["source_mod10"]),
                "job_count": shard_count,
                "staged_valid_lines_total": total_staged,
            }
        )

    return staged_jobs, original_manifest


def greedy_assign(jobs: List[Dict[str, object]], gpus: List[int]) -> Dict[int, List[Dict[str, object]]]:
    buckets: Dict[int, List[Dict[str, object]]] = {gpu: [] for gpu in gpus}
    loads: Dict[int, int] = {gpu: 0 for gpu in gpus}

    for job in sorted(jobs, key=lambda item: int(item["staged_valid_lines"]), reverse=True):
        gpu = min(loads, key=loads.get)
        buckets[gpu].append(job)
        loads[gpu] += int(job["staged_valid_lines"])

    return buckets


def find_output_dir(log_text: str) -> Optional[str]:
    match = OUTPUT_DIR_RE.search(log_text)
    if match:
        return match.group(1).strip()
    return None


def parse_run_stats(log_text: str) -> Dict[str, Optional[int]]:
    stats: Dict[str, Optional[int]] = {
        "parsed_total_lines": None,
        "used_lines": None,
        "bad_or_empty_lines": None,
        "n_groups": None,
    }
    parse_match = PARSE_STATS_RE.search(log_text)
    if parse_match:
        stats["parsed_total_lines"] = int(parse_match.group("total"))
        stats["used_lines"] = int(parse_match.group("used"))
        stats["bad_or_empty_lines"] = int(parse_match.group("bad"))

    group_match = GROUP_COUNT_RE.search(log_text)
    if group_match:
        stats["n_groups"] = int(group_match.group("n_groups"))
    return stats


def run_one_job(
    root: Path,
    cal_path: Path,
    conda_exe: str,
    conda_env: str,
    gpu: int,
    job: Dict[str, object],
    run_dir: Path,
    group_size: int,
    max_k: int,
    translate_chunk: int,
    batch_size: int,
    max_length: int,
) -> Dict[str, object]:
    source_txt = str(job["source_txt"])
    staged_input = Path(str(job["staged_input"]))
    task_name = str(job["task_name"])
    log_name = f"{safe_name(task_name)}.log"
    log_path = run_dir / "logs" / log_name

    cmd = [
        conda_exe,
        "run",
        "--no-capture-output",
        "-n",
        conda_env,
        "python",
        str(cal_path),
        "-i",
        str(staged_input),
        "--group-size",
        str(group_size),
        "--max-k",
        str(max_k),
        "--gpu",
        "0",
        "--translate-chunk",
        str(translate_chunk),
        "--batch-size",
        str(batch_size),
        "--max-length",
        str(max_length),
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

    started_at = datetime.utcnow().isoformat(timespec="seconds")
    start = time.time()
    with log_path.open("w", encoding="utf-8") as log_fh:
        log_fh.write(f"START_UTC\t{started_at}\n")
        log_fh.write("COMMAND\t" + " ".join(cmd) + "\n\n")
        log_fh.flush()
        proc = subprocess.run(
            cmd,
            cwd=str(root),
            env=env,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )

    finished_at = datetime.utcnow().isoformat(timespec="seconds")
    duration_sec = round(time.time() - start, 2)
    log_text = log_path.read_text(encoding="utf-8", errors="ignore")
    output_dir = find_output_dir(log_text)
    run_stats = parse_run_stats(log_text)
    output_path = Path(output_dir) if output_dir else None

    combined_rows = read_group_acc_rows(output_path / "group_acc.tsv") if output_path else {}
    exact_rows = read_group_acc_rows(output_path / "exact_match_group_acc.tsv") if output_path else {}
    combined_metrics = extract_metrics_from_rows(combined_rows)
    exact_metrics = extract_metrics_from_rows(exact_rows)

    status = "ok" if proc.returncode == 0 else "failed"
    note_parts = []
    if int(job["is_shard"]) == 1:
        note_parts.append(f"shard={job['shard_index']}/{job['shard_count']}")
    note = ", ".join(note_parts)

    result: Dict[str, object] = {
        "task_name": task_name,
        "source_txt": source_txt,
        "source_path": str(job["source_path"]),
        "gpu": gpu,
        "status": status,
        "returncode": proc.returncode,
        "source_total_lines": int(job["source_total_lines"]),
        "source_valid_lines": int(job["source_valid_lines"]),
        "source_mod10": int(job["source_mod10"]),
        "staged_input": str(staged_input),
        "staged_input_rel": str(job["staged_input_rel"]),
        "staged_valid_lines": int(job["staged_valid_lines"]),
        "shard_index": int(job["shard_index"]),
        "shard_count": int(job["shard_count"]),
        "parsed_total_lines": run_stats["parsed_total_lines"],
        "used_lines": run_stats["used_lines"],
        "bad_or_empty_lines": run_stats["bad_or_empty_lines"],
        "n_groups": run_stats["n_groups"],
        "duration_sec": duration_sec,
        "started_at_utc": started_at,
        "finished_at_utc": finished_at,
        "output_dir": str(output_path) if output_path else "",
        "log_path": str(log_path),
        "note": note,
    }
    result.update({f"combined_{key}": value for key, value in combined_metrics.items()})
    result.update({f"exact_{key}": value for key, value in exact_metrics.items()})
    return result


def worker(
    gpu: int,
    jobs: List[Dict[str, object]],
    root: Path,
    cal_path: Path,
    conda_exe: str,
    conda_env: str,
    run_dir: Path,
    group_size: int,
    max_k: int,
    translate_chunk: int,
    batch_size: int,
    max_length: int,
) -> List[Dict[str, object]]:
    results: List[Dict[str, object]] = []
    for job in jobs:
        result = run_one_job(
            root=root,
            cal_path=cal_path,
            conda_exe=conda_exe,
            conda_env=conda_env,
            gpu=gpu,
            job=job,
            run_dir=run_dir,
            group_size=group_size,
            max_k=max_k,
            translate_chunk=translate_chunk,
            batch_size=batch_size,
            max_length=max_length,
        )
        results.append(result)
    return results


def collect_job_long_rows(job_summary_rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    long_rows: List[Dict[str, object]] = []
    for row in job_summary_rows:
        if row.get("status") != "ok":
            continue
        output_dir = row.get("output_dir")
        if not output_dir:
            continue
        out_path = Path(str(output_dir))
        for metric_name, filename in (
            ("combined", "group_acc.tsv"),
            ("exact", "exact_match_group_acc.tsv"),
        ):
            acc_rows = read_group_acc_rows(out_path / filename)
            for k in sorted(acc_rows):
                topk, rankk = acc_rows[k]
                long_rows.append(
                    {
                        "task_name": row["task_name"],
                        "source_txt": row["source_txt"],
                        "metric": metric_name,
                        "K": k,
                        "topk_acc": f"{topk:.8f}",
                        "rankk_acc": f"{rankk:.8f}",
                        "n_groups": row.get("n_groups", ""),
                    }
                )
    return long_rows


def aggregate_metric_rows(job_rows: List[Dict[str, object]], filename: str) -> Tuple[Dict[int, Tuple[float, float]], int]:
    weighted_topk: DefaultDict[int, float] = defaultdict(float)
    weighted_rankk: DefaultDict[int, float] = defaultdict(float)
    total_groups = 0

    for row in job_rows:
        if row.get("status") != "ok":
            continue
        n_groups = row.get("n_groups")
        output_dir = row.get("output_dir")
        if not n_groups or not output_dir:
            continue
        n_groups_int = int(n_groups)
        total_groups += n_groups_int
        acc_rows = read_group_acc_rows(Path(str(output_dir)) / filename)
        for k, (topk, rankk) in acc_rows.items():
            weighted_topk[k] += topk * n_groups_int
            weighted_rankk[k] += rankk * n_groups_int

    merged_rows: Dict[int, Tuple[float, float]] = {}
    if total_groups == 0:
        return merged_rows, 0

    for k in sorted(weighted_topk):
        merged_rows[k] = (
            weighted_topk[k] / total_groups,
            weighted_rankk[k] / total_groups,
        )
    return merged_rows, total_groups


def build_merged_summary(
    original_manifest: List[Dict[str, object]],
    job_summary_rows: List[Dict[str, object]],
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    by_source: DefaultDict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in job_summary_rows:
        by_source[str(row["source_txt"])].append(row)

    merged_rows: List[Dict[str, object]] = []
    merged_long_rows: List[Dict[str, object]] = []

    for record in sorted(original_manifest, key=lambda item: str(item["source_txt"])):
        source_txt = str(record["source_txt"])
        rows = sorted(
            by_source.get(source_txt, []),
            key=lambda item: int(item.get("shard_index", 1)),
        )
        ok_rows = [row for row in rows if row.get("status") == "ok"]
        all_ok = len(rows) == int(record["job_count"]) and len(ok_rows) == len(rows)

        combined_rows, combined_groups = aggregate_metric_rows(rows, "group_acc.tsv")
        exact_rows, exact_groups = aggregate_metric_rows(rows, "exact_match_group_acc.tsv")
        combined_metrics = extract_metrics_from_rows(combined_rows)
        exact_metrics = extract_metrics_from_rows(exact_rows)

        for metric_name, acc_rows, total_groups in (
            ("combined", combined_rows, combined_groups),
            ("exact", exact_rows, exact_groups),
        ):
            for k in sorted(acc_rows):
                topk, rankk = acc_rows[k]
                merged_long_rows.append(
                    {
                        "input_txt": source_txt,
                        "metric": metric_name,
                        "K": k,
                        "topk_acc": f"{topk:.8f}",
                        "rankk_acc": f"{rankk:.8f}",
                        "n_groups": total_groups,
                    }
                )

        used_lines_total = sum(int(row["used_lines"]) for row in ok_rows if row.get("used_lines") is not None)
        bad_lines_total = sum(
            int(row["bad_or_empty_lines"]) for row in ok_rows if row.get("bad_or_empty_lines") is not None
        )
        duration_total = round(
            sum(float(row["duration_sec"]) for row in rows if row.get("duration_sec") not in ("", None)),
            2,
        )

        note_parts = []
        if int(record["job_count"]) > 1:
            note_parts.append(f"shards={record['job_count']}")
        if int(record["source_mod10"]) != 0:
            kept = int(record["source_valid_lines"]) - int(record["source_mod10"])
            note_parts.append(f"valid_lines={record['source_valid_lines']}, truncated_to={kept}")
        failed_count = len(rows) - len(ok_rows)
        if failed_count:
            note_parts.append(f"failed_jobs={failed_count}")

        merged_row: Dict[str, object] = {
            "input_txt": source_txt,
            "input_path": str(record["source_path"]),
            "status": "ok" if all_ok else "failed",
            "job_count": int(record["job_count"]),
            "source_total_lines": int(record["source_total_lines"]),
            "source_valid_lines": int(record["source_valid_lines"]),
            "source_mod10": int(record["source_mod10"]),
            "used_lines": used_lines_total if ok_rows else "",
            "bad_or_empty_lines": bad_lines_total if ok_rows else "",
            "n_groups": combined_groups if combined_groups else "",
            "duration_sec_total": duration_total,
            "note": ", ".join(note_parts),
        }
        merged_row.update({f"combined_{key}": value for key, value in combined_metrics.items()})
        merged_row.update({f"exact_{key}": value for key, value in exact_metrics.items()})
        merged_rows.append(merged_row)

    return merged_rows, merged_long_rows


def render_summary_markdown(summary_rows: List[Dict[str, object]]) -> str:
    headers = [
        "txt",
        "status",
        "jobs",
        "combined_top1",
        "combined_top10",
        "combined_avg",
        "exact_top1",
        "exact_top10",
        "exact_avg",
        "note",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]

    def fmt(value: object) -> str:
        if isinstance(value, float):
            return f"{value:.6f}"
        if value is None:
            return ""
        return str(value)

    for row in summary_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["input_txt"]),
                    str(row["status"]),
                    str(row["job_count"]),
                    fmt(row.get("combined_top1_acc")),
                    fmt(row.get("combined_top10_acc")),
                    fmt(row.get("combined_avg_rankk_acc")),
                    fmt(row.get("exact_top1_acc")),
                    fmt(row.get("exact_top10_acc")),
                    fmt(row.get("exact_avg_rankk_acc")),
                    str(row.get("note", "")),
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(DEFAULT_ROOT))
    ap.add_argument("--cal", default=str(DEFAULT_CAL))
    ap.add_argument("--conda-env", default="r-smiles")
    ap.add_argument("--conda-exe", default=str(DEFAULT_CONDA if DEFAULT_CONDA.exists() else "conda"))
    ap.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    ap.add_argument("--group-size", type=int, default=10)
    ap.add_argument("--max-k", type=int, default=10)
    ap.add_argument("--translate-chunk", type=int, default=5000)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-length", type=int, default=200)
    ap.add_argument("--split-threshold-valid-lines", type=int, default=200000)
    ap.add_argument("--shard-valid-lines", type=int, default=250000)
    ap.add_argument("--label", default=None)
    args = ap.parse_args()

    root = Path(args.root).resolve()
    cal_path = Path(args.cal).resolve()
    conda_exe = args.conda_exe
    gpus = [int(item.strip()) for item in args.gpus.split(",") if item.strip()]

    if not root.exists():
        raise FileNotFoundError(root)
    if not cal_path.exists():
        raise FileNotFoundError(cal_path)
    if not gpus:
        raise ValueError("No GPUs provided.")

    run_dir = choose_run_dir(root, args.label)
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)

    original_inputs, skipped_inputs = discover_original_inputs(root)
    staged_jobs, original_manifest = stage_jobs(
        original_inputs=original_inputs,
        run_dir=run_dir,
        group_size=args.group_size,
        split_threshold_valid_lines=args.split_threshold_valid_lines,
        shard_valid_lines=args.shard_valid_lines,
    )
    assignments = greedy_assign(staged_jobs, gpus)

    write_tsv(
        run_dir / "valid_inputs.tsv",
        original_manifest,
        [
            "source_txt",
            "source_path",
            "source_total_lines",
            "source_valid_lines",
            "source_mod10",
            "job_count",
            "staged_valid_lines_total",
        ],
    )
    write_tsv(
        run_dir / "skipped_inputs.tsv",
        skipped_inputs,
        ["source_txt", "source_path", "source_total_lines", "source_valid_lines", "reason"],
    )
    write_tsv(
        run_dir / "staged_jobs.tsv",
        staged_jobs,
        [
            "task_name",
            "source_txt",
            "source_path",
            "source_total_lines",
            "source_valid_lines",
            "source_mod10",
            "staged_input",
            "staged_input_rel",
            "staged_valid_lines",
            "shard_index",
            "shard_count",
            "is_shard",
        ],
    )

    assignment_rows: List[Dict[str, object]] = []
    for gpu in gpus:
        bucket = assignments[gpu]
        gpu_total_valid_lines = sum(int(job["staged_valid_lines"]) for job in bucket)
        for order, job in enumerate(bucket, start=1):
            assignment_rows.append(
                {
                    "gpu": gpu,
                    "order_on_gpu": order,
                    "task_name": job["task_name"],
                    "source_txt": job["source_txt"],
                    "staged_input_rel": job["staged_input_rel"],
                    "staged_valid_lines": int(job["staged_valid_lines"]),
                    "gpu_total_valid_lines": gpu_total_valid_lines,
                }
            )
    write_tsv(
        run_dir / "gpu_assignments.tsv",
        sorted(assignment_rows, key=lambda row: (row["gpu"], row["order_on_gpu"])),
        [
            "gpu",
            "order_on_gpu",
            "task_name",
            "source_txt",
            "staged_input_rel",
            "staged_valid_lines",
            "gpu_total_valid_lines",
        ],
    )

    print(f"[Batch] run_dir={run_dir}")
    print(
        f"[Batch] valid_inputs={len(original_manifest)} staged_jobs={len(staged_jobs)} "
        f"skipped={len(skipped_inputs)} gpus={gpus}"
    )

    job_summary_rows: List[Dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=len(gpus)) as executor:
        future_map = {
            executor.submit(
                worker,
                gpu,
                assignments[gpu],
                root,
                cal_path,
                conda_exe,
                args.conda_env,
                run_dir,
                args.group_size,
                args.max_k,
                args.translate_chunk,
                args.batch_size,
                args.max_length,
            ): gpu
            for gpu in gpus
        }
        for future in as_completed(future_map):
            gpu = future_map[future]
            try:
                gpu_rows = future.result()
            except Exception as exc:
                job_summary_rows.append(
                    {
                        "task_name": f"GPU_{gpu}_WORKER",
                        "source_txt": "",
                        "source_path": "",
                        "gpu": gpu,
                        "status": "worker_failed",
                        "returncode": "",
                        "source_total_lines": "",
                        "source_valid_lines": "",
                        "source_mod10": "",
                        "staged_input": "",
                        "staged_input_rel": "",
                        "staged_valid_lines": "",
                        "shard_index": "",
                        "shard_count": "",
                        "parsed_total_lines": "",
                        "used_lines": "",
                        "bad_or_empty_lines": "",
                        "n_groups": "",
                        "duration_sec": "",
                        "started_at_utc": "",
                        "finished_at_utc": "",
                        "output_dir": "",
                        "log_path": "",
                        "note": str(exc),
                    }
                )
            else:
                job_summary_rows.extend(gpu_rows)

    job_summary_rows = sorted(job_summary_rows, key=lambda row: (str(row["source_txt"]), str(row["task_name"])))
    write_tsv(
        run_dir / "job_summary.tsv",
        job_summary_rows,
        [
            "task_name",
            "source_txt",
            "source_path",
            "gpu",
            "status",
            "returncode",
            "source_total_lines",
            "source_valid_lines",
            "source_mod10",
            "staged_input",
            "staged_input_rel",
            "staged_valid_lines",
            "shard_index",
            "shard_count",
            "parsed_total_lines",
            "used_lines",
            "bad_or_empty_lines",
            "n_groups",
            "duration_sec",
            "started_at_utc",
            "finished_at_utc",
            "output_dir",
            "log_path",
            "combined_top1_acc",
            "combined_top3_acc",
            "combined_top5_acc",
            "combined_top10_acc",
            "combined_rank1_acc",
            "combined_rank3_acc",
            "combined_rank5_acc",
            "combined_rank10_acc",
            "combined_avg_rankk_acc",
            "exact_top1_acc",
            "exact_top3_acc",
            "exact_top5_acc",
            "exact_top10_acc",
            "exact_rank1_acc",
            "exact_rank3_acc",
            "exact_rank5_acc",
            "exact_rank10_acc",
            "exact_avg_rankk_acc",
            "note",
        ],
    )

    job_long_rows = collect_job_long_rows(job_summary_rows)
    write_tsv(
        run_dir / "job_group_acc_long.tsv",
        job_long_rows,
        ["task_name", "source_txt", "metric", "K", "topk_acc", "rankk_acc", "n_groups"],
    )

    summary_rows, merged_long_rows = build_merged_summary(original_manifest, job_summary_rows)
    write_tsv(
        run_dir / "summary.tsv",
        summary_rows,
        [
            "input_txt",
            "input_path",
            "status",
            "job_count",
            "source_total_lines",
            "source_valid_lines",
            "source_mod10",
            "used_lines",
            "bad_or_empty_lines",
            "n_groups",
            "duration_sec_total",
            "combined_top1_acc",
            "combined_top3_acc",
            "combined_top5_acc",
            "combined_top10_acc",
            "combined_rank1_acc",
            "combined_rank3_acc",
            "combined_rank5_acc",
            "combined_rank10_acc",
            "combined_avg_rankk_acc",
            "exact_top1_acc",
            "exact_top3_acc",
            "exact_top5_acc",
            "exact_top10_acc",
            "exact_rank1_acc",
            "exact_rank3_acc",
            "exact_rank5_acc",
            "exact_rank10_acc",
            "exact_avg_rankk_acc",
            "note",
        ],
    )
    write_tsv(
        run_dir / "group_acc_merged_long.tsv",
        merged_long_rows,
        ["input_txt", "metric", "K", "topk_acc", "rankk_acc", "n_groups"],
    )

    summary_md = render_summary_markdown(summary_rows)
    (run_dir / "summary.md").write_text(summary_md, encoding="utf-8")

    failed_rows = [row for row in summary_rows if row.get("status") != "ok"]
    print(f"[Batch] finished_inputs={len(summary_rows)} failed_inputs={len(failed_rows)}")
    print(f"[Batch] summary_tsv={run_dir / 'summary.tsv'}")
    print(f"[Batch] summary_md={run_dir / 'summary.md'}")
    print(f"[Batch] merged_long={run_dir / 'group_acc_merged_long.tsv'}")
    print(f"[Batch] job_summary={run_dir / 'job_summary.tsv'}")
    print(f"[Batch] skipped_inputs={run_dir / 'skipped_inputs.tsv'}")
    return 0 if not failed_rows else 1


if __name__ == "__main__":
    sys.exit(main())
