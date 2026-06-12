#!/usr/bin/env python3
"""Build a forward-filter-only Retroformer raw_train.csv.

Input is the post-processed ReactionT5/RXNMapper CSV containing mapped
`reaction`, `reactants`, and `product` columns. The script keeps unique
(reactants, product) pairs in input order, writes them as USPTO-style
`id,class,reactants>reagents>production`, and optionally appends an original
USPTO-50K raw_train.csv.
"""

import argparse
import csv
import shutil
from pathlib import Path


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--filtered-csv", required=True, help="CSV with reaction/reactants/product columns")
    ap.add_argument("--out-dir", required=True, help="Output dataset directory")
    ap.add_argument("--append-train", default=None, help="Optional original raw_train.csv to append")
    ap.add_argument("--copy-val", default=None, help="Optional raw_val.csv to copy into out-dir")
    ap.add_argument("--copy-test", default=None, help="Optional raw_test.csv to copy into out-dir")
    return ap.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "raw_train.csv"
    manifest = out_dir / "dataset_manifest.txt"

    if args.copy_val:
        shutil.copy2(args.copy_val, out_dir / "raw_val.csv")
    if args.copy_test:
        shutil.copy2(args.copy_test, out_dir / "raw_test.csv")

    seen = set()
    input_rows = 0
    kept = 0
    bad = 0
    appended = 0

    with Path(args.filtered_csv).open("r", encoding="utf-8", newline="") as fin, \
            out_csv.open("w", encoding="utf-8", newline="") as fout:
        reader = csv.DictReader(fin)
        required = {"reaction", "reactants", "product"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise SystemExit(f"missing required columns: {sorted(missing)}")

        writer = csv.writer(fout)
        writer.writerow(["id", "class", "reactants>reagents>production"])

        for row in reader:
            input_rows += 1
            rxn = (row.get("reaction") or "").strip()
            reactants = (row.get("reactants") or "").strip()
            product = (row.get("product") or "").strip()
            if not rxn or ">>" not in rxn or not reactants or not product:
                bad += 1
                continue
            key = (reactants, product)
            if key in seen:
                continue
            seen.add(key)
            writer.writerow(["UNK", "UNK", rxn])
            kept += 1

        if args.append_train:
            with Path(args.append_train).open("r", encoding="utf-8", newline="") as fapp:
                next(fapp)
                for line in fapp:
                    fout.write(line)
                    appended += 1

    manifest.write_text(
        "\n".join([
            "dataset=forward_filter_only",
            f"filtered_csv={args.filtered_csv}",
            "rule=deduplicate by raw (reactants, product); no fingerprint diversity pruning",
            f"input_rows={input_rows}",
            f"unique_forward_rows={kept}",
            f"bad_rows={bad}",
            f"appended_original_train_rows={appended}",
            f"raw_train_rows={kept + appended}",
            f"raw_train={out_csv}",
        ]) + "\n",
        encoding="utf-8",
    )
    print(manifest.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
