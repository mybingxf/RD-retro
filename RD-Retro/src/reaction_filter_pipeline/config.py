"""Configuration loading and validation."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


DEFAULT_CONFIG: Dict[str, Any] = {
    "input": {
        "path": None,
        "format": "auto",
        "product_column": "product",
        "reactants_column": "reactants",
        "reagent_column": None,
        "group_column": "group_id",
        "rank_column": "rank",
        "source_column": "source_type",
        "default_source_type": "generated",
        "candidate_prefix": "reactants",
        "candidate_columns": None,
        "source_by_candidate_column": {},
    },
    "output_dir": "reaction_filter_output",
    "forward": {
        "backend": "reactiont5v2",
        "model_name_or_path": "sagawa/ReactionT5v2-forward",
        "revision": "933114058cb2604dc1bf536dbebdfcefbe83d4fc",
        "devices": ["auto"],
        "batch_size": 16,
        "input_max_length": 150,
        "max_new_tokens": 128,
        "num_beams": 1,
        "dtype": "auto",
        "prompt_template": "REACTANT:{reactants}REAGENT:{reagent}",
        "local_files_only": False,
        "prediction_file": None,
        "prediction_id_column": "candidate_id",
        "prediction_column": "predicted_product",
        "chemformer_root": None,
        "chemformer_checkpoint": None,
        "chemformer_vocabulary": None,
        "chemformer_max_length": 512,
    },
    "mapping": {
        "backend": "rxnmapper",
        "devices": ["auto"],
        "batch_size": 64,
        "canonicalize": False,
        "prediction_file": None,
    },
    "confidence_threshold": 0.2,
    "diversity": {
        "max_candidates_per_product": 10,
        "morgan_radius": 2,
        "morgan_bits": 2048,
        "workers": 8,
        "chunksize": 8,
        "invalid_smiles": "error",
        "removal_strategy": "confidence",
        "original_source_types": ["training", "original"],
    },
}


PATH_KEYS = (
    ("input", "path"),
    ("forward", "prediction_file"),
    ("forward", "chemformer_root"),
    ("forward", "chemformer_checkpoint"),
    ("forward", "chemformer_vocabulary"),
    ("mapping", "prediction_file"),
)


def _deep_update(target: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = value
    return target


def load_config(path: Optional[str] = None) -> Dict[str, Any]:
    config = copy.deepcopy(DEFAULT_CONFIG)
    base_dir = Path.cwd()
    if path:
        config_path = Path(path).expanduser().resolve()
        base_dir = config_path.parent
        with config_path.open("r", encoding="utf-8") as handle:
            supplied = yaml.safe_load(handle) or {}
        if not isinstance(supplied, dict):
            raise ValueError("The YAML configuration root must be a mapping")
        _deep_update(config, supplied)

    for section, key in PATH_KEYS:
        value = config[section].get(key)
        if value:
            candidate = Path(value).expanduser()
            if not candidate.is_absolute():
                candidate = base_dir / candidate
            config[section][key] = str(candidate.resolve())

    output_dir = Path(config["output_dir"]).expanduser()
    if not output_dir.is_absolute():
        output_dir = base_dir / output_dir
    config["output_dir"] = str(output_dir.resolve())
    return config


def validate_config(config: Dict[str, Any]) -> None:
    if not config["input"].get("path"):
        raise ValueError("input.path is required (or pass --input)")
    if config["input"]["format"] not in {"auto", "wide", "long"}:
        raise ValueError("input.format must be auto, wide, or long")
    if config["forward"]["backend"] not in {
        "reactiont5v2",
        "hf_seq2seq",
        "chemformer",
        "precomputed",
    }:
        raise ValueError("Unsupported forward.backend")
    if config["mapping"]["backend"] not in {"rxnmapper", "precomputed"}:
        raise ValueError("mapping.backend must be rxnmapper or precomputed")
    threshold = float(config["confidence_threshold"])
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("confidence_threshold must be in [0, 1]")
    if int(config["diversity"]["max_candidates_per_product"]) < 1:
        raise ValueError("max_candidates_per_product must be >= 1")
    if config["diversity"]["invalid_smiles"] not in {"error", "zero"}:
        raise ValueError("diversity.invalid_smiles must be error or zero")
    strategies = {
        "confidence",
        "source_then_mean_similarity",
        "mean_similarity",
    }
    if config["diversity"].get("removal_strategy") not in strategies:
        raise ValueError(
            "diversity.removal_strategy must be confidence, "
            "source_then_mean_similarity, or mean_similarity"
        )
    source_types = config["diversity"].get("original_source_types")
    if not isinstance(source_types, list) or not source_types:
        raise ValueError("diversity.original_source_types must be a non-empty list")
    config["diversity"]["original_source_types"] = [
        str(value).strip().casefold() for value in source_types if str(value).strip()
    ]
    if not config["diversity"]["original_source_types"]:
        raise ValueError("diversity.original_source_types cannot contain only blanks")
    source_mapping = config["input"].get("source_by_candidate_column", {})
    if not isinstance(source_mapping, dict):
        raise ValueError("input.source_by_candidate_column must be a mapping")
    for section in ("forward", "mapping"):
        devices = config[section].get("devices")
        if isinstance(devices, str):
            config[section]["devices"] = [x.strip() for x in devices.split(",") if x.strip()]
        if not config[section]["devices"]:
            raise ValueError(f"{section}.devices must contain at least one device")


def dump_config(config: Dict[str, Any], path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
