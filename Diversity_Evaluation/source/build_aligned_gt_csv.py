#!/usr/bin/env python3
"""Build a product-aligned ground-truth CSV for condition-A scoring."""

import argparse
import csv
from pathlib import Path


PRODUCT_CANDIDATES = ["product", "source_smiles", "src", "target", "tgt", "products"]
GROUND_TRUTH_CANDIDATES = [
    "ground_truth",
    "reactants",
    "ground_truth_reactants",
    "target_reactants",
]


def detect(fieldnames, requested, candidates, kind):
    names = set(fieldnames)
    if requested:
        if requested not in names:
            raise RuntimeError(f"CSV missing requested {kind} column: {requested}")
        return requested
    for candidate in candidates:
        if candidate in names:
            return candidate
    raise RuntimeError(
        f"Could not auto-detect {kind} column. Available columns: {fieldnames}"
    )


def clean(value):
    return "".join(str(value or "").split())


def reactants_only(value):
    value = clean(value)
    if ">>" in value:
        return value.split(">>", 1)[0]
    if value.count(">") >= 2:
        return value.split(">", 1)[0]
    return value


def main():
    parser = argparse.ArgumentParser(
        description="Build aligned id,reactants>reagents>production CSV from predictions."
    )
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--product-col", default="")
    parser.add_argument("--ground-truth-col", default="")
    args = parser.parse_args()

    input_csv = Path(args.input_csv).resolve()
    output_csv = Path(args.output_csv).resolve()
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with input_csv.open("r", encoding="utf-8", errors="ignore", newline="") as fin:
        reader = csv.DictReader(fin)
        if reader.fieldnames is None:
            raise RuntimeError(f"CSV has no header: {input_csv}")
        product_col = detect(
            reader.fieldnames, args.product_col, PRODUCT_CANDIDATES, "product"
        )
        gt_col = detect(
            reader.fieldnames,
            args.ground_truth_col,
            GROUND_TRUTH_CANDIDATES,
            "ground-truth reactants",
        )

        rows = 0
        with output_csv.open("w", encoding="utf-8", newline="") as fout:
            writer = csv.writer(fout)
            writer.writerow(["id", "reactants>reagents>production"])
            for line_number, row in enumerate(reader, start=2):
                product = clean(row.get(product_col))
                reactants = reactants_only(row.get(gt_col))
                if not product or not reactants:
                    raise RuntimeError(
                        f"Empty product or ground truth at CSV row {line_number}"
                    )
                writer.writerow([rows, f"{reactants}>>{product}"])
                rows += 1

    print(f"input_csv={input_csv}")
    print(f"output_csv={output_csv}")
    print(f"product_col={product_col}")
    print(f"ground_truth_col={gt_col}")
    print(f"rows={rows}")


if __name__ == "__main__":
    main()
