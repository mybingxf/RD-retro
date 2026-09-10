"""End-to-end orchestration and final validation."""

from __future__ import annotations

import csv
import json
import math
import time
from collections import Counter
from pathlib import Path
from typing import Callable, Dict, Tuple

from .chemistry import (
    diversity_prune,
    filter_forward_matches,
    threshold_mapping_confidence,
)
from .config import dump_config, validate_config
from .forward import run_forward
from .io import normalize_input
from .mapping import run_mapping
from .utils import count_csv_rows, package_versions, sha256sum, write_json
from . import __version__


ARTIFACT_NAMES = {
    "candidates": "00_candidates.csv",
    "groups": "00_product_groups.csv",
    "forward_predictions": "01_forward_predictions.csv",
    "forward_pass": "02_forward_pass.csv",
    "mapped": "03_rxnmapper_mapped.csv",
    "confidence_filtered": "04_confidence_filtered.csv",
    "final_long": "05_final_long.csv",
    "final_wide": "05_final_wide.csv",
    "group_stats": "05_group_stats.csv",
}


def _run_stage(
    name: str,
    primary_output: Path,
    stats_dir: Path,
    resume: bool,
    function: Callable[[], Dict[str, object]],
) -> Dict[str, object]:
    stats_path = stats_dir / f"{name}.json"
    if resume and primary_output.exists() and stats_path.exists():
        with stats_path.open("r", encoding="utf-8") as handle:
            stats = json.load(handle)
        print(f"[resume] {name}: {primary_output}", flush=True)
        return stats
    started = time.time()
    print(f"[run] {name}", flush=True)
    stats = function()
    stats["elapsed_seconds"] = time.time() - started
    write_json(stats_path, stats)
    print(f"[done] {name}: {stats['elapsed_seconds']:.3f} s", flush=True)
    return stats


def validate_outputs(
    artifacts: Dict[str, Path], threshold: float, maximum: int
) -> Dict[str, object]:
    counts = Counter()
    total = 0
    minimum_confidence = None
    previous_candidate = -1
    bad_mapped_format = 0
    with artifacts["final_long"].open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            candidate_id = int(row["candidate_id"])
            if candidate_id <= previous_candidate:
                raise ValueError("Final candidate IDs are not strictly increasing")
            previous_candidate = candidate_id
            confidence = float(row["mapping_confidence"])
            if not math.isfinite(confidence) or confidence < threshold:
                raise ValueError("Final output violates confidence threshold")
            counts[row["group_id"]] += 1
            minimum_confidence = (
                confidence
                if minimum_confidence is None
                else min(minimum_confidence, confidence)
            )
            mapped = row["mapped_reaction_smiles"]
            bad_mapped_format += mapped.count(">>") != 1 or mapped.count(">") != 2
            total += 1
    maximum_observed = max(counts.values(), default=0)
    if maximum_observed > maximum:
        raise ValueError("Final output violates per-product maximum")
    if bad_mapped_format:
        raise ValueError("Final output contains malformed mapped reactions")
    group_rows = count_csv_rows(artifacts["groups"])
    wide_rows = count_csv_rows(artifacts["final_wide"])
    if group_rows != wide_rows:
        raise ValueError("Final wide output does not preserve all product groups")
    return {
        "validated_final_candidates": total,
        "validated_nonempty_groups": len(counts),
        "validated_total_groups": group_rows,
        "validated_max_candidates_per_group": maximum_observed,
        "validated_minimum_confidence": minimum_confidence,
        "bad_mapped_reaction_formats": bad_mapped_format,
    }


def run_pipeline(
    config: Dict[str, object], resume: bool = False, overwrite: bool = False
) -> Dict[str, object]:
    validate_config(config)
    input_path = Path(str(config["input"]["path"]))
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    output_dir = Path(str(config["output_dir"]))
    known_existing = [
        output_dir / name for name in ARTIFACT_NAMES.values() if (output_dir / name).exists()
    ]
    if known_existing and not resume and not overwrite:
        raise FileExistsError(
            f"Output artifacts already exist in {output_dir}; use --resume or --overwrite"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    stats_dir = output_dir / "stage_stats"
    stats_dir.mkdir(parents=True, exist_ok=True)
    work_dir = output_dir / "work"
    work_dir.mkdir(parents=True, exist_ok=True)
    artifacts = {
        key: output_dir / filename for key, filename in ARTIFACT_NAMES.items()
    }
    dump_config(config, output_dir / "run_config.resolved.yml")
    run_started = time.time()

    normalize_stats = _run_stage(
        "00_normalize_input",
        artifacts["candidates"],
        stats_dir,
        resume,
        lambda: normalize_input(
            input_path,
            artifacts["candidates"],
            artifacts["groups"],
            config["input"],
        ),
    )
    forward_stats = _run_stage(
        "01_forward_predict",
        artifacts["forward_predictions"],
        stats_dir,
        resume,
        lambda: run_forward(
            artifacts["candidates"],
            artifacts["forward_predictions"],
            work_dir / "forward",
            config["forward"],
        ),
    )
    comparison_stats = _run_stage(
        "02_forward_compare",
        artifacts["forward_pass"],
        stats_dir,
        resume,
        lambda: filter_forward_matches(
            artifacts["candidates"],
            artifacts["forward_predictions"],
            artifacts["forward_pass"],
        ),
    )
    mapping_stats = _run_stage(
        "03_atom_mapping",
        artifacts["mapped"],
        stats_dir,
        resume,
        lambda: run_mapping(
            artifacts["forward_pass"],
            artifacts["mapped"],
            work_dir / "mapping",
            config["mapping"],
        ),
    )
    confidence_stats = _run_stage(
        "04_confidence_filter",
        artifacts["confidence_filtered"],
        stats_dir,
        resume,
        lambda: threshold_mapping_confidence(
            artifacts["mapped"],
            artifacts["confidence_filtered"],
            float(config["confidence_threshold"]),
        ),
    )
    diversity_stats = _run_stage(
        "05_diversity_prune",
        artifacts["final_long"],
        stats_dir,
        resume,
        lambda: diversity_prune(
            artifacts["confidence_filtered"],
            artifacts["groups"],
            artifacts["final_long"],
            artifacts["final_wide"],
            artifacts["group_stats"],
            config["diversity"],
        ),
    )
    validation = validate_outputs(
        artifacts,
        float(config["confidence_threshold"]),
        int(config["diversity"]["max_candidates_per_product"]),
    )
    artifact_metadata = {}
    for key, path in artifacts.items():
        artifact_metadata[key] = {
            "path": str(path),
            "rows": count_csv_rows(path),
            "sha256": sha256sum(path),
        }
    summary = {
        "software": "reaction-filter-pipeline",
        "software_version": __version__,
        "elapsed_seconds": time.time() - run_started,
        "parameters": {
            "forward_backend": config["forward"]["backend"],
            "forward_model_name_or_path": config["forward"].get(
                "model_name_or_path"
            ),
            "mapping_backend": config["mapping"]["backend"],
            "confidence_threshold_inclusive": float(
                config["confidence_threshold"]
            ),
            "max_candidates_per_product": int(
                config["diversity"]["max_candidates_per_product"]
            ),
            "morgan_radius": int(config["diversity"]["morgan_radius"]),
            "morgan_bits": int(config["diversity"]["morgan_bits"]),
            "diversity_removal_strategy": config["diversity"][
                "removal_strategy"
            ],
            "original_source_types": config["diversity"][
                "original_source_types"
            ],
            "multi_component_aggregation": "bitwise_OR",
            "similarity": "Tanimoto",
        },
        "stages": {
            "normalize": normalize_stats,
            "forward_prediction": forward_stats,
            "forward_comparison": comparison_stats,
            "mapping": mapping_stats,
            "confidence_filter": confidence_stats,
            "diversity": diversity_stats,
        },
        "validation": validation,
        "environment": package_versions(),
        "artifacts": artifact_metadata,
    }
    write_json(output_dir / "summary.json", summary)
    return summary
