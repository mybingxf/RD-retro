#!/usr/bin/env python3
"""Run all round-trip evaluation stages with portable, explicit paths."""

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = PACKAGE_ROOT / "source"
DEFAULT_MT_DIR = PACKAGE_ROOT / "third_party" / "MolecularTransformer"
DEFAULT_MODEL = "STEREO_separated_augm_model_average_20.pt"


def run(command, env):
    print(f"\n[Command] {shlex.join(str(item) for item in command)}", flush=True)
    subprocess.run([str(item) for item in command], check=True, env=env)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Build grouped reactants>>product input and compute round-trip accuracy, "
            "reactant similarity, and disconnection similarity."
        )
    )
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--input-csv", help="Prediction CSV with product and ranked candidates")
    inputs.add_argument("--input-txt", help="Prebuilt reactants>>product TXT")
    parser.add_argument("--output-dir", required=True, help="New or empty output directory")
    parser.add_argument("--gt-csv", help="Aligned GT CSV for exact-match condition A")
    parser.add_argument(
        "--no-gt-csv",
        action="store_true",
        help="Skip condition A. Otherwise GT is auto-built from input CSV when possible.",
    )
    parser.add_argument("--product-col", default="")
    parser.add_argument("--candidate-prefix", default="")
    parser.add_argument("--candidate-cols", default="")
    parser.add_argument("--ground-truth-col", default="")
    parser.add_argument("--group-size", type=int, default=10)
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--mapping-batch-size", type=int, default=128)
    parser.add_argument("--translate-chunk", type=int, default=5000)
    parser.add_argument("--mt-dir", default=str(DEFAULT_MT_DIR))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--hit-fill-sim",
        type=float,
        default=0.0,
        help="Hit-similarity fill value when a group has fewer than top-n hits.",
    )
    parser.add_argument("--skip-reactant-similarity", action="store_true")
    parser.add_argument("--skip-disconnection-similarity", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.group_size <= 0:
        raise ValueError("--group-size must be > 0")
    if not (1 <= args.top_n <= args.group_size):
        raise ValueError("--top-n must be between 1 and --group-size")
    if args.gt_csv and args.no_gt_csv:
        raise ValueError("Use either --gt-csv or --no-gt-csv, not both")

    output_root = Path(args.output_dir).resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise RuntimeError(
            f"Output directory is not empty: {output_root}. Choose a new directory."
        )
    output_root.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    if args.device == "cuda":
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
        translate_gpu = 0
        # The disconnection script performs its own masking before importing
        # RXNMapper, so it receives the physical id rather than the masked id.
        mapping_gpu = args.gpu_id
    else:
        env["CUDA_VISIBLE_DEVICES"] = ""
        translate_gpu = -1
        mapping_gpu = 0

    input_txt = output_root / f"roundtrip_input_top{args.group_size}.txt"
    aligned_gt = None

    if args.input_csv:
        input_csv = Path(args.input_csv).resolve()
        build_command = [
            sys.executable,
            SOURCE_DIR / "build_roundtrip_topk_input.py",
            "--input-csv",
            input_csv,
            "--output-txt",
            input_txt,
            "--topk",
            args.group_size,
        ]
        if args.product_col:
            build_command.extend(["--product-col", args.product_col])
        if args.candidate_prefix:
            build_command.extend(["--candidate-prefix", args.candidate_prefix])
        if args.candidate_cols:
            build_command.extend(["--candidate-cols", args.candidate_cols])
        run(build_command, env)

        if not args.no_gt_csv and not args.gt_csv:
            aligned_gt = output_root / "aligned_ground_truth.csv"
            gt_command = [
                sys.executable,
                SOURCE_DIR / "build_aligned_gt_csv.py",
                "--input-csv",
                input_csv,
                "--output-csv",
                aligned_gt,
            ]
            if args.product_col:
                gt_command.extend(["--product-col", args.product_col])
            if args.ground_truth_col:
                gt_command.extend(["--ground-truth-col", args.ground_truth_col])
            run(gt_command, env)
    else:
        shutil.copyfile(Path(args.input_txt).resolve(), input_txt)

    if args.gt_csv:
        aligned_gt = Path(args.gt_csv).resolve()

    run_dir = output_root / input_txt.stem
    cal_command = [
        sys.executable,
        SOURCE_DIR / "cal.py",
        "--input",
        input_txt,
        "--group-size",
        args.group_size,
        "--mt-dir",
        Path(args.mt_dir).resolve(),
        "--model",
        args.model,
        "--gpu",
        translate_gpu,
        "--batch-size",
        args.batch_size,
        "--translate-chunk",
        args.translate_chunk,
    ]
    if aligned_gt is None or args.no_gt_csv:
        cal_command.append("--no-gt-csv")
    else:
        cal_command.extend(["--gt-csv", aligned_gt])
    run(cal_command, env)

    if not run_dir.exists():
        raise RuntimeError(f"Expected round-trip output directory was not created: {run_dir}")

    if not args.skip_reactant_similarity:
        run(
            [
                sys.executable,
                SOURCE_DIR / "group_reactant_similarity_posthoc.py",
                "--run-dir",
                run_dir,
                "--top-n",
                args.top_n,
                "--group-size",
                args.group_size,
                "--hit-fill-sim",
                args.hit_fill_sim,
            ],
            env,
        )

    if not args.skip_disconnection_similarity:
        run(
            [
                sys.executable,
                SOURCE_DIR / "disconnection_similarity_nomap.py",
                "--input",
                input_txt,
                "--outdir",
                run_dir,
                "--device",
                args.device,
                "--gpu-id",
                mapping_gpu,
                "--batch",
                args.mapping_batch_size,
                "--group-size",
                args.group_size,
                "--top-k",
                args.top_n,
            ],
            env,
        )

    summary = {
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "input": str(Path(args.input_csv or args.input_txt).resolve()),
        "roundtrip_input": str(input_txt),
        "aligned_gt": str(aligned_gt) if aligned_gt else None,
        "run_dir": str(run_dir),
        "group_size": args.group_size,
        "top_n": args.top_n,
        "device": args.device,
        "reactant_similarity_completed": not args.skip_reactant_similarity,
        "disconnection_similarity_completed": not args.skip_disconnection_similarity,
    }
    (output_root / "pipeline_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(f"\n[Pipeline complete] Results: {run_dir}")


if __name__ == "__main__":
    main()
