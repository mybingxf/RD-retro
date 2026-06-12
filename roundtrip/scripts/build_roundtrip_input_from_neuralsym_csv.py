#!/usr/bin/env python3

import argparse
import csv
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", required=True, help="NeuralSym proposals CSV")
    parser.add_argument("--output-txt", required=True, help="Output TXT path")
    parser.add_argument("--topk", type=int, default=10, help="How many candidates per product")
    return parser.parse_args()


def normalize_precursor(value: str) -> str:
    value = (value or "").strip()
    if not value or value.lower() == "nan":
        return "9999"
    return value


def main():
    args = parse_args()
    input_csv = Path(args.input_csv).resolve()
    output_txt = Path(args.output_txt).resolve()

    with input_csv.open("r", encoding="utf-8", newline="") as fin, \
         output_txt.open("w", encoding="utf-8", newline="") as fout:
        reader = csv.DictReader(fin)
        if reader.fieldnames is None:
            raise RuntimeError(f"CSV has no header: {input_csv}")

        candidate_cols = [f"cand_precursor_{i}" for i in range(1, args.topk + 1)]
        missing_cols = [col for col in candidate_cols if col not in reader.fieldnames]
        if missing_cols:
            raise RuntimeError(f"Missing candidate columns: {missing_cols}")
        if "prod_smi" not in reader.fieldnames:
            raise RuntimeError("CSV missing prod_smi column")

        line_count = 0
        row_count = 0
        for row in reader:
            product = row["prod_smi"].strip()
            if not product:
                raise RuntimeError(f"Empty prod_smi at row {row_count}")
            for col in candidate_cols:
                precursor = normalize_precursor(row[col])
                fout.write(f"{precursor}>>{product}\n")
                line_count += 1
            row_count += 1

    print(f"rows={row_count}")
    print(f"lines={line_count}")
    print(f"output={output_txt}")


if __name__ == "__main__":
    main()
