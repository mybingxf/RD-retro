#!/usr/bin/env python3
"""Build a no-filter random-sample-10 reaction TXT/CSV from input_data.csv."""

import argparse
import csv
import random
import shutil
from collections import defaultdict
from pathlib import Path


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-csv", required=True, help="CSV with REACTANT and PRODUCT columns")
    ap.add_argument("--out-dir", required=True, help="Final dataset directory")
    ap.add_argument("--intermediate-dir", required=True, help="Directory for unmapped CSV and TXT")
    ap.add_argument("--seed", type=int, default=20260531)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--copy-val", default=None)
    ap.add_argument("--copy-test", default=None)
    return ap.parse_args()


def main():
    args = parse_args()
    rng = random.Random(args.seed)
    out_dir = Path(args.out_dir)
    inter_dir = Path(args.intermediate_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    inter_dir.mkdir(parents=True, exist_ok=True)

    if args.copy_val:
        shutil.copy2(args.copy_val, out_dir / "raw_val.csv")
    if args.copy_test:
        shutil.copy2(args.copy_test, out_dir / "raw_test.csv")

    prod_to_reactants = defaultdict(list)
    prod_seen = defaultdict(set)
    input_rows = valid_rows = duplicate_rows = bad_rows = 0

    with Path(args.input_csv).open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        missing = {"REACTANT", "PRODUCT"}.difference(reader.fieldnames or [])
        if missing:
            raise SystemExit(f"missing columns: {sorted(missing)}")
        for row in reader:
            input_rows += 1
            react = (row.get("REACTANT") or "").strip()
            prod = (row.get("PRODUCT") or "").strip()
            if not react or not prod:
                bad_rows += 1
                continue
            if react in prod_seen[prod]:
                duplicate_rows += 1
                continue
            prod_seen[prod].add(react)
            prod_to_reactants[prod].append(react)
            valid_rows += 1

    unmapped_csv = inter_dir / "raw_train_unmapped.csv"
    txt = inter_dir / "random_sample10.txt"
    selected_rows = products_lt_k = products_eq_k = products_gt_k = 0

    with unmapped_csv.open("w", encoding="utf-8", newline="") as fcsv, \
            txt.open("w", encoding="utf-8") as ftxt:
        writer = csv.writer(fcsv)
        writer.writerow(["id", "class", "reactants>reagents>production"])
        for prod_idx, prod in enumerate(sorted(prod_to_reactants.keys())):
            reacts = prod_to_reactants[prod]
            if len(reacts) < args.k:
                chosen = list(reacts)
                products_lt_k += 1
            elif len(reacts) == args.k:
                chosen = list(reacts)
                products_eq_k += 1
            else:
                chosen = rng.sample(reacts, args.k)
                products_gt_k += 1
            for rank, react in enumerate(chosen):
                rxn = f"{react}>>{prod}"
                rid = f"NOFILTER_RANDOM{args.k}_{prod_idx:05d}__sample{rank:02d}"
                writer.writerow([rid, "UNK", rxn])
                ftxt.write(rxn + "\n")
                selected_rows += 1

    manifest = inter_dir / "sample_manifest.txt"
    manifest.write_text(
        "\n".join([
            "dataset=no_filter_random_sample10",
            f"input_csv={args.input_csv}",
            f"random_seed={args.seed}",
            f"k={args.k}",
            f"input_rows={input_rows}",
            f"valid_unique_product_reactant_rows={valid_rows}",
            f"duplicate_rows_skipped={duplicate_rows}",
            f"bad_rows_skipped={bad_rows}",
            f"products_with_candidates={len(prod_to_reactants)}",
            f"products_lt_k={products_lt_k}",
            f"products_eq_k={products_eq_k}",
            f"products_gt_k={products_gt_k}",
            f"selected_rows={selected_rows}",
            f"unmapped_csv={unmapped_csv}",
            f"reaction_txt={txt}",
        ]) + "\n",
        encoding="utf-8",
    )
    print(manifest.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
