#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
from pathlib import Path
from tqdm import tqdm


def pick_mapped_rxn(item: dict) -> str:
    # rxnmapper 常见字段名，做个鲁棒提取
    for k in ["mapped_rxn", "mapped_rxn_smiles", "mapped_reaction", "mapped_smiles"]:
        v = item.get(k)
        if isinstance(v, str) and ">" in v:
            return v
    for v in item.values():
        if isinstance(v, str) and ">" in v:
            return v
    raise ValueError(f"cannot find mapped reaction in keys={list(item.keys())}")


def flush_batch(rxn_mapper, batch, writer, failed_writer=None):
    if not batch:
        return
    try:
        results = rxn_mapper.get_attention_guided_atom_maps(batch)
    except Exception:
        # 整批失败：原样写回（不中断，保证行数对齐）
        for rxn in batch:
            writer.writerow(["UNK", "UNK", rxn])
            if failed_writer:
                failed_writer.writerow([rxn])
        return

    for rxn, item in zip(batch, results):
        try:
            mapped = pick_mapped_rxn(item)
            writer.writerow(["UNK", "UNK", mapped])
        except Exception:
            writer.writerow(["UNK", "UNK", rxn])
            if failed_writer:
                failed_writer.writerow([rxn])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--in_txt", required=True, help="input txt, each line: reactants>>products")
    ap.add_argument("-o", "--out_csv", required=True, help="output csv: id,class,reactants>reagents>production")
    ap.add_argument("--failed_txt", default=None, help="optional: save failed lines")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--no_pbar", action="store_true")
    args = ap.parse_args()

    in_path = Path(args.in_txt)
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    failed_path = Path(args.failed_txt) if args.failed_txt else None
    if failed_path:
        failed_path.parent.mkdir(parents=True, exist_ok=True)

    from rxnmapper import RXNMapper
    rxn_mapper = RXNMapper()

    total = None
    try:
        total = sum(1 for _ in open(in_path, "r", encoding="utf-8"))
    except Exception:
        pass

    batch = []

    with open(in_path, "r", encoding="utf-8") as fin, \
         open(out_path, "w", encoding="utf-8", newline="") as fout, \
         (open(failed_path, "w", encoding="utf-8", newline="") if failed_path else open("/dev/null", "w")) as fbad:

        writer = csv.writer(fout)
        writer.writerow(["id", "class", "reactants>reagents>production"])

        failed_writer = csv.writer(fbad) if failed_path else None

        iterator = fin
        if args.no_pbar:
            for line in iterator:
                rxn = line.strip()
                if not rxn:
                    continue
                batch.append(rxn)
                if len(batch) >= args.batch_size:
                    flush_batch(rxn_mapper, batch, writer, failed_writer)
                    batch = []
        else:
            for line in tqdm(iterator, total=total, desc="mapping", unit="rxn"):
                rxn = line.strip()
                if not rxn:
                    continue
                batch.append(rxn)
                if len(batch) >= args.batch_size:
                    flush_batch(rxn_mapper, batch, writer, failed_writer)
                    batch = []

        if batch:
            flush_batch(rxn_mapper, batch, writer, failed_writer)

    print(f"[OK] wrote: {out_path}")
    if failed_path:
        print(f"[OK] failed lines saved to: {failed_path}")


if __name__ == "__main__":
    main()
