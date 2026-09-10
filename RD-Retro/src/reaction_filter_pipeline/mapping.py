"""RXNMapper and precomputed atom-mapping stages."""

from __future__ import annotations

import csv
import math
import multiprocessing as mp
import os
from pathlib import Path
from typing import Dict, List

from .utils import (
    AtomicDictWriter,
    merge_sorted_csvs,
    reset_directory,
    split_csv_round_robin,
)


MAPPING_FIELDS = [
    "candidate_id",
    "group_id",
    "rank",
    "product",
    "reactants",
    "reagent",
    "source_type",
    "predicted_product",
    "mapped_reaction_smiles",
    "mapping_confidence",
    "mapping_success",
]


def _configure_device(device: str) -> None:
    if device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        return
    if device.startswith("cuda:"):
        physical_index = device.split(":", 1)[1]
        os.environ["CUDA_VISIBLE_DEVICES"] = physical_index
        return
    if device not in {"auto", "cuda"}:
        raise ValueError(f"Unsupported RXNMapper device: {device}")


def _map_worker(
    input_path: str, output_path: str, config: Dict[str, object], device: str
) -> None:
    _configure_device(device)
    from rdkit import RDLogger
    from rxnmapper import BatchedMapper

    RDLogger.DisableLog("rdApp.*")
    mapper = BatchedMapper(
        batch_size=int(config.get("batch_size", 64)),
        canonicalize=bool(config.get("canonicalize", False)),
    )
    output = AtomicDictWriter(Path(output_path), MAPPING_FIELDS)
    batch_size = int(config.get("batch_size", 64))
    rows: List[Dict[str, str]] = []

    def flush() -> None:
        reactions = [f"{row['reactants']}>>{row['product']}" for row in rows]
        results = list(mapper.map_reactions_with_info(reactions))
        if len(results) != len(rows):
            raise ValueError("RXNMapper returned the wrong batch size")
        for row, result in zip(rows, results):
            mapped = result.get("mapped_rxn", "") if result else ""
            confidence = result.get("confidence", "") if result else ""
            try:
                valid_confidence = math.isfinite(float(confidence))
            except (TypeError, ValueError):
                valid_confidence = False
            merged = dict(row)
            merged.update(
                {
                    "mapped_reaction_smiles": mapped,
                    "mapping_confidence": confidence,
                    "mapping_success": int(
                        bool(mapped) and mapped != ">>" and valid_confidence
                    ),
                }
            )
            output.writerow(merged)

    try:
        with Path(input_path).open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                rows.append(row)
                if len(rows) >= batch_size:
                    flush()
                    rows = []
            if rows:
                flush()
    except Exception:
        output.close(commit=False)
        raise
    else:
        output.close()


def _copy_precomputed(
    forward_pass_path: Path, predictions_path: Path, output_path: Path
) -> int:
    output = AtomicDictWriter(output_path, MAPPING_FIELDS)
    count = 0
    try:
        with forward_pass_path.open("r", encoding="utf-8-sig", newline="") as candidates_handle, predictions_path.open(
            "r", encoding="utf-8-sig", newline=""
        ) as predictions_handle:
            candidates = csv.DictReader(candidates_handle)
            predictions = csv.DictReader(predictions_handle)
            required = {
                "candidate_id",
                "mapped_reaction_smiles",
                "mapping_confidence",
            }
            missing = required.difference(predictions.fieldnames or [])
            if missing:
                raise ValueError(
                    f"Precomputed mapping file is missing: {sorted(missing)}"
                )
            prediction = next(predictions, None)
            for candidate in candidates:
                candidate_id = int(candidate["candidate_id"])
                while prediction is not None and int(prediction["candidate_id"]) < candidate_id:
                    prediction = next(predictions, None)
                if prediction is None or int(prediction["candidate_id"]) != candidate_id:
                    raise ValueError(
                        f"No precomputed mapping for candidate_id={candidate_id}"
                    )
                mapped = prediction["mapped_reaction_smiles"]
                confidence = float(prediction["mapping_confidence"])
                success_value = prediction.get("mapping_success", "1")
                success = int(
                    success_value.lower() in {"1", "true", "yes"}
                    and bool(mapped)
                    and mapped != ">>"
                    and math.isfinite(confidence)
                )
                row = dict(candidate)
                row.update(
                    {
                        "mapped_reaction_smiles": mapped,
                        "mapping_confidence": confidence,
                        "mapping_success": success,
                    }
                )
                output.writerow(row)
                count += 1
                prediction = next(predictions, None)
    except Exception:
        output.close(commit=False)
        raise
    else:
        output.close()
    return count


def run_mapping(
    forward_pass_path: Path,
    output_path: Path,
    work_dir: Path,
    config: Dict[str, object],
) -> Dict[str, object]:
    backend = str(config["backend"])
    if backend == "precomputed":
        prediction_file = config.get("prediction_file")
        if not prediction_file:
            raise ValueError("precomputed mapping backend requires prediction_file")
        count = _copy_precomputed(
            forward_pass_path, Path(str(prediction_file)), output_path
        )
        return {"backend": backend, "mapped_rows": count, "devices": []}

    devices = [str(value) for value in config.get("devices", ["auto"])]
    input_dir = work_dir / "inputs"
    output_dir = work_dir / "outputs"
    reset_directory(input_dir)
    reset_directory(output_dir)
    shards = split_csv_round_robin(forward_pass_path, input_dir, len(devices))
    output_shards = [output_dir / path.name for path in shards]
    context = mp.get_context("spawn")
    processes = []
    for shard, destination, device in zip(shards, output_shards, devices):
        process = context.Process(
            target=_map_worker,
            args=(str(shard), str(destination), config, device),
        )
        process.start()
        processes.append(process)
    failures = []
    for process in processes:
        process.join()
        if process.exitcode != 0:
            failures.append(process.exitcode)
    if failures:
        raise RuntimeError(f"RXNMapper workers failed: {failures}")
    count = merge_sorted_csvs(output_shards, output_path)
    return {"backend": backend, "mapped_rows": count, "devices": devices}
