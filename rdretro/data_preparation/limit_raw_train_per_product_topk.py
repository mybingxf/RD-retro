#!/usr/bin/env python3
"""Keep at most K unique reactions per product from a raw_train.csv."""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")


def norm_multi(smiles: str):
    parts = []
    for part in (smiles or "").split("."):
        part = part.strip()
        if not part:
            continue
        mol = Chem.MolFromSmiles(part)
        if mol is None:
            return None
        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(0)
        parts.append(Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True))
    return ".".join(sorted(parts)) if parts else None


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-csv", required=True)
    ap.add_argument("--output-csv", required=True)
    ap.add_argument("--k", type=int, default=10)
    return ap.parse_args()


def main():
    args = parse_args()
    kept_per_product = defaultdict(int)
    seen_rxn_per_product = defaultdict(set)
    product_cache = {}
    reactant_cache = {}
    input_rows = kept_rows = dropped_over_k = dropped_duplicate = bad_rxn = 0

    def cached(cache, value):
        if value not in cache:
            cache[value] = norm_multi(value)
        return cache[value]

    with Path(args.input_csv).open("r", encoding="utf-8", newline="") as fin, \
            Path(args.output_csv).open("w", encoding="utf-8", newline="") as fout:
        reader = csv.DictReader(fin)
        if not reader.fieldnames or "reactants>reagents>production" not in reader.fieldnames:
            raise SystemExit("missing reactants>reagents>production column")
        writer = csv.DictWriter(fout, fieldnames=reader.fieldnames)
        writer.writeheader()

        for row in reader:
            input_rows += 1
            rxn = row["reactants>reagents>production"]
            if ">>" not in rxn:
                bad_rxn += 1
                continue
            reactants, product = rxn.split(">>", 1)
            product_key = cached(product_cache, product.strip()) or product.strip()
            if kept_per_product[product_key] >= args.k:
                dropped_over_k += 1
                continue
            react_key = cached(reactant_cache, reactants.strip()) or reactants.strip()
            rxn_key = f"{react_key}>>{product_key}"
            if rxn_key in seen_rxn_per_product[product_key]:
                dropped_duplicate += 1
                continue
            seen_rxn_per_product[product_key].add(rxn_key)
            kept_per_product[product_key] += 1
            writer.writerow(row)
            kept_rows += 1

    stats = Path(args.output_csv).with_suffix(".stats.txt")
    stats.write_text(
        "\n".join([
            f"input_csv={args.input_csv}",
            f"output_csv={args.output_csv}",
            f"k={args.k}",
            f"input_rows={input_rows}",
            f"kept_rows={kept_rows}",
            f"dropped_over_k={dropped_over_k}",
            f"dropped_duplicate={dropped_duplicate}",
            f"bad_rxn_rows={bad_rxn}",
            f"num_products={len(kept_per_product)}",
            f"max_kept_per_product={max(kept_per_product.values()) if kept_per_product else 0}",
        ]) + "\n",
        encoding="utf-8",
    )
    print(stats.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
