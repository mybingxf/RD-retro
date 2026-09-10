#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

MISSING_VALUES = {"", "nan", "none", "null", "na"}
PRODUCT_CANDIDATES = ["product", "source_smiles", "src", "target", "tgt", "products"]
CANDIDATE_SCHEMES = [
    "top_{}",
    "prediction_{}",
    "pred_{}",
    "reactant_{}",
    "reactants_{}",
    "rank_{}",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build roundtripcal reactants>>product TXT from ranked prediction CSV."
    )
    parser.add_argument("--input-csv", required=True, help="Prediction CSV path")
    parser.add_argument("--output-txt", required=True, help="Output TXT path")
    parser.add_argument("--topk", type=int, default=10, help="Candidates per product")
    parser.add_argument(
        "--product-col",
        default="",
        help="Product column. Auto-detects product/source_smiles when omitted.",
    )
    parser.add_argument(
        "--candidate-prefix",
        default="",
        help="Candidate column prefix, e.g. top_ for top_1..top_K.",
    )
    parser.add_argument(
        "--candidate-cols",
        default="",
        help="Comma-separated explicit candidate columns. Overrides prefix auto-detection.",
    )
    parser.add_argument(
        "--missing-value",
        default="9999",
        help="Placeholder reactant for blank/NaN candidate cells.",
    )
    return parser.parse_args()


def clean_cell(value):
    if value is None:
        return ""
    return str(value).strip()


def clean_product(value):
    return clean_cell(value)


def is_missing(value):
    return clean_cell(value).lower() in MISSING_VALUES


def detect_product_col(fieldnames, requested):
    names = set(fieldnames)
    if requested:
        if requested not in names:
            raise RuntimeError(f"CSV missing requested product column: {requested}")
        return requested
    for col in PRODUCT_CANDIDATES:
        if col in names:
            return col
    raise RuntimeError(
        "Could not auto-detect product column. Pass --product-col. "
        f"Available columns: {fieldnames}"
    )


def detect_candidate_cols(fieldnames, topk, prefix, explicit):
    names = set(fieldnames)
    if explicit:
        cols = [c.strip() for c in explicit.split(",") if c.strip()]
        if len(cols) != topk:
            raise RuntimeError(f"--candidate-cols must contain exactly {topk} columns")
        missing = [c for c in cols if c not in names]
        if missing:
            raise RuntimeError(f"CSV missing requested candidate columns: {missing}")
        return cols

    schemes = [prefix + "{}"] if prefix else CANDIDATE_SCHEMES
    for scheme in schemes:
        cols = [scheme.format(i) for i in range(1, topk + 1)]
        if all(c in names for c in cols):
            return cols

    tried = []
    for scheme in schemes:
        tried.append([scheme.format(i) for i in range(1, min(topk, 3) + 1)])
    raise RuntimeError(
        "Could not auto-detect candidate columns. Pass --candidate-prefix or --candidate-cols. "
        f"Tried examples: {tried}. Available columns: {fieldnames}"
    )


def main():
    args = parse_args()
    if args.topk <= 0:
        raise ValueError("--topk must be > 0")

    input_csv = Path(args.input_csv).resolve()
    output_txt = Path(args.output_txt).resolve()
    output_txt.parent.mkdir(parents=True, exist_ok=True)

    with input_csv.open("r", encoding="utf-8", errors="ignore", newline="") as fin:
        reader = csv.DictReader(fin)
        if reader.fieldnames is None:
            raise RuntimeError(f"CSV has no header: {input_csv}")

        product_col = detect_product_col(reader.fieldnames, args.product_col)
        candidate_cols = detect_candidate_cols(
            reader.fieldnames, args.topk, args.candidate_prefix, args.candidate_cols
        )

        rows = 0
        lines = 0
        with output_txt.open("w", encoding="utf-8", newline="") as fout:
            for row in reader:
                product = clean_product(row.get(product_col))
                if not product:
                    raise RuntimeError(f"Empty product at CSV row {rows + 2}")
                for col in candidate_cols:
                    reactants = args.missing_value if is_missing(row.get(col)) else clean_cell(row.get(col))
                    fout.write(f"{reactants}>>{product}\n")
                    lines += 1
                rows += 1

    expected = rows * args.topk
    if lines != expected:
        raise RuntimeError(f"Internal error: wrote {lines} lines, expected {expected}")

    print(f"input_csv={input_csv}")
    print(f"output_txt={output_txt}")
    print(f"product_col={product_col}")
    print(f"candidate_cols={','.join(candidate_cols)}")
    print(f"rows={rows}")
    print(f"topk={args.topk}")
    print(f"lines={lines}")


if __name__ == "__main__":
    main()
