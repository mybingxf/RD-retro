"""Command-line interface."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from . import __version__
from .config import DEFAULT_CONFIG, load_config
from .export import append_to_training_csv
from .io import normalize_input
from .pipeline import run_pipeline


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="reaction-filter",
        description=(
            "Filter retrosynthesis candidates by forward prediction, RXNMapper "
            "confidence, and Morgan/Tanimoto diversity."
        ),
    )
    parser.add_argument("--version", action="version", version=__version__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="Run the complete filtering pipeline")
    run.add_argument("--config", help="YAML configuration file")
    run.add_argument("--input", help="Override input.path")
    run.add_argument("--output-dir", help="Override output_dir")
    run.add_argument(
        "--forward-backend",
        choices=["reactiont5v2", "hf_seq2seq", "chemformer", "precomputed"],
    )
    run.add_argument("--model-name-or-path")
    run.add_argument("--forward-devices", help="Comma-separated, e.g. cuda:0,cuda:1")
    run.add_argument("--mapping-backend", choices=["rxnmapper", "precomputed"])
    run.add_argument("--mapping-devices", help="Comma-separated, e.g. cuda:2,cuda:3")
    run.add_argument("--confidence-threshold", type=float)
    run.add_argument("--max-candidates", type=int)
    run.add_argument(
        "--diversity-strategy",
        choices=[
            "confidence",
            "source_then_mean_similarity",
            "mean_similarity",
        ],
    )
    run.add_argument("--resume", action="store_true")
    run.add_argument("--overwrite", action="store_true")

    subparsers.add_parser(
        "show-default-config", help="Print the built-in default YAML configuration"
    )
    subparsers.add_parser("list-backends", help="List available backend names")
    prepare = subparsers.add_parser(
        "prepare",
        help="Normalize input and assign candidate IDs for an external model",
    )
    prepare.add_argument("--config", help="YAML configuration file")
    prepare.add_argument("--input", help="Override input.path")
    prepare.add_argument("--output-dir", required=True)
    export = subparsers.add_parser(
        "append-training",
        help="Append final mapped candidates to an existing raw training CSV",
    )
    export.add_argument("--raw-train", required=True)
    export.add_argument("--final-long", required=True)
    export.add_argument("--output", required=True)
    export.add_argument("--id-column", default="id")
    export.add_argument("--class-column", default="class")
    export.add_argument(
        "--reaction-column", default="reactants>reagents>production"
    )
    export.add_argument("--generated-class", default="UNK")
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "show-default-config":
        print(yaml.safe_dump(DEFAULT_CONFIG, sort_keys=False), end="")
        return
    if args.command == "list-backends":
        print("Forward: reactiont5v2 (default), hf_seq2seq, chemformer, precomputed")
        print("Mapping: rxnmapper (default), precomputed")
        print(
            "Diversity: confidence (default), source_then_mean_similarity, "
            "mean_similarity"
        )
        return
    if args.command == "prepare":
        config = load_config(args.config)
        if args.input:
            config["input"]["path"] = str(Path(args.input).expanduser().resolve())
        if not config["input"].get("path"):
            raise ValueError("input.path is required (or pass --input)")
        destination = Path(args.output_dir).expanduser().resolve()
        destination.mkdir(parents=True, exist_ok=True)
        result = normalize_input(
            Path(config["input"]["path"]),
            destination / "00_candidates.csv",
            destination / "00_product_groups.csv",
            config["input"],
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    if args.command == "append-training":
        result = append_to_training_csv(
            raw_train=Path(args.raw_train).expanduser().resolve(),
            final_long=Path(args.final_long).expanduser().resolve(),
            output=Path(args.output).expanduser().resolve(),
            id_column=args.id_column,
            class_column=args.class_column,
            reaction_column=args.reaction_column,
            generated_class=args.generated_class,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    config = load_config(args.config)
    if args.input:
        config["input"]["path"] = str(Path(args.input).expanduser().resolve())
    if args.output_dir:
        config["output_dir"] = str(Path(args.output_dir).expanduser().resolve())
    if args.forward_backend:
        config["forward"]["backend"] = args.forward_backend
    if args.model_name_or_path:
        config["forward"]["model_name_or_path"] = args.model_name_or_path
    if args.forward_devices:
        config["forward"]["devices"] = [
            item.strip() for item in args.forward_devices.split(",") if item.strip()
        ]
    if args.mapping_backend:
        config["mapping"]["backend"] = args.mapping_backend
    if args.mapping_devices:
        config["mapping"]["devices"] = [
            item.strip() for item in args.mapping_devices.split(",") if item.strip()
        ]
    if args.confidence_threshold is not None:
        config["confidence_threshold"] = args.confidence_threshold
    if args.max_candidates is not None:
        config["diversity"]["max_candidates_per_product"] = args.max_candidates
    if args.diversity_strategy:
        config["diversity"]["removal_strategy"] = args.diversity_strategy
    summary = run_pipeline(config, resume=args.resume, overwrite=args.overwrite)
    compact = {
        "output_dir": config["output_dir"],
        "input_candidates": summary["stages"]["normalize"]["candidates"],
        "forward_matches": summary["stages"]["forward_comparison"][
            "forward_matches"
        ],
        "confidence_filtered": summary["stages"]["confidence_filter"][
            "confidence_filtered_rows"
        ],
        "final_candidates": summary["validation"]["validated_final_candidates"],
        "final_nonempty_groups": summary["validation"][
            "validated_nonempty_groups"
        ],
    }
    print(json.dumps(compact, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
