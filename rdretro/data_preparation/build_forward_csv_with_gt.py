#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import re
from pathlib import Path
from collections import defaultdict, deque

import pandas as pd
from rdkit import Chem
from tqdm import tqdm


# 匹配 txt 中的候选：('SMILES', 0.605)
TUPLE_PAT = re.compile(
    r"\(\s*'([^']*)'\s*,\s*([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)\s*\)"
)


def sanitize_smiles(smiles: str, cache: dict) -> str:
    """
    去原子映射 + RDKit canonical 标准化。
    若 RDKit 解析失败，返回原始字符串（避免破坏数据）。
    """
    if smiles is None:
        return ""
    s = str(smiles).strip()
    if s == "":
        return ""

    if s in cache:
        return cache[s]

    mol = Chem.MolFromSmiles(s)
    if mol is None:
        cache[s] = s
        return s

    for atom in mol.GetAtoms():
        if atom.GetAtomMapNum():
            atom.SetAtomMapNum(0)

    out = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    cache[s] = out
    return out


def parse_candidates_from_line(line: str):
    """
    输入行：
      0    ('smi1', 0.6) ('smi2', 0.3) ...
    返回：
      idx(int), [(smi, score), ...]
    """
    line = line.strip()
    if not line:
        return None

    parts = line.split(maxsplit=1)
    if not parts:
        return None

    idx = int(parts[0])
    rest = parts[1] if len(parts) > 1 else ""

    cands = []
    for m in TUPLE_PAT.finditer(rest):
        smi = m.group(1).strip()
        score = float(m.group(2))
        if smi:
            cands.append((smi, score))
    return idx, cands


def count_lines(path: Path) -> int:
    """Count lines for tqdm total."""
    n = 0
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for _ in f:
            n += 1
    return n


def main():
    ap = argparse.ArgumentParser(
        description="Build final forward-prediction CSV from candidate txt + test csv, and insert GT rows before first matching PRODUCT."
    )
    ap.add_argument("-i_txt", required=True, help="Input candidate txt (LocalRetro_USPTO_50K.txt-like).")
    ap.add_argument("-i_csv", required=True, help="Input test csv (preprocessed_test.csv-like).")
    ap.add_argument("-o_csv", required=True, help="Output final CSV path.")

    ap.add_argument("--reactant-col", default="Reactants", help="Reactant column name in test csv (default: Reactants).")
    ap.add_argument("--product-col", default="Products", help="Product column name in test csv (default: Products).")
    ap.add_argument("--one-based-index", action="store_true", help="Treat txt leading index as 1-based (1->row0).")
    ap.add_argument("--topk", type=int, default=None, help="Only keep top-k candidates per txt line (default: keep all).")
    ap.add_argument("--reagent-blank", default=" ", help="REAGENT placeholder when absent (default: single space).")
    ap.add_argument("--sanitize-reactant", action="store_true", help="Also sanitize candidate REACTANT SMILES (recommended).")
    args = ap.parse_args()

    txt_path = Path(args.i_txt)
    test_path = Path(args.i_csv)
    out_path = Path(args.o_csv)

    if not txt_path.exists():
        raise FileNotFoundError(f"txt not found: {txt_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"csv not found: {test_path}")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ---- 1) 读取 test csv，构建 product->reactants 队列（GT 插入用），以及按行号取 product 的数组 ----
    cache = {}
    df_test = pd.read_csv(test_path, dtype=str, keep_default_na=False)

    if args.reactant_col not in df_test.columns:
        raise KeyError(f"'{args.reactant_col}' not in {test_path}. Columns={list(df_test.columns)}")
    if args.product_col not in df_test.columns:
        raise KeyError(f"'{args.product_col}' not in {test_path}. Columns={list(df_test.columns)}")

    prod_list = []
    gt_map = defaultdict(deque)  # product -> deque(reactant)

    n_test = len(df_test)
    reactants_series = df_test[args.reactant_col]
    products_series = df_test[args.product_col]

    for r_raw, p_raw in tqdm(
        zip(reactants_series, products_series),
        total=n_test,
        desc="Sanitize test Reactants/Products",
        unit="rxn",
    ):
        p = sanitize_smiles(p_raw, cache)
        r = sanitize_smiles(r_raw, cache)
        prod_list.append(p)
        if p.strip() != "" and r.strip() != "":
            gt_map[p].append(r)

    # ---- 2) 逐行读 txt，写输出；遇到某个 product 第一次出现时，先插入 GT 行 ----
    header = ["REACTANT", "REAGENT", "PRODUCT"]
    seen_products = set()

    # 为 txt 加 total（需要先数一遍行数）
    txt_total = count_lines(txt_path)

    total_candidate_rows = 0
    inserted_gt_rows = 0
    bad_lines = 0
    oob_idx = 0

    with txt_path.open("r", encoding="utf-8", errors="ignore") as fin, out_path.open(
        "w", encoding="utf-8", newline=""
    ) as fout:
        writer = csv.DictWriter(fout, fieldnames=header)
        writer.writeheader()

        for line in tqdm(fin, total=txt_total, desc="Parse candidate txt", unit="line"):
            parsed = parse_candidates_from_line(line)
            if parsed is None:
                bad_lines += 1
                continue

            idx, cands = parsed
            if args.one_based_index:
                idx -= 1

            if idx < 0 or idx >= len(prod_list):
                oob_idx += 1
                continue

            product = prod_list[idx]
            if product.strip() == "":
                continue

            # 第一次遇到该 product：插入所有 GT 行
            if product not in seen_products:
                seen_products.add(product)
                if product in gt_map and len(gt_map[product]) > 0:
                    while gt_map[product]:
                        gt_react = gt_map[product].popleft()
                        writer.writerow({
                            "REACTANT": gt_react,
                            "REAGENT": args.reagent_blank,
                            "PRODUCT": product
                        })
                        inserted_gt_rows += 1

            if args.topk is not None:
                cands = cands[:args.topk]

            for smi, _score in cands:
                reactant = smi
                if args.sanitize_reactant:
                    reactant = sanitize_smiles(reactant, cache)

                writer.writerow({
                    "REACTANT": reactant,
                    "REAGENT": args.reagent_blank,
                    "PRODUCT": product
                })
                total_candidate_rows += 1

    # ---- 3) 若某些 GT 的 product 在 txt 中从未出现：剩余 GT 追加到末尾（带进度条） ----
    remaining_gt = sum(len(dq) for dq in gt_map.values())
    if remaining_gt > 0:
        with out_path.open("a", encoding="utf-8", newline="") as fout:
            writer = csv.DictWriter(fout, fieldnames=header)
            pbar = tqdm(total=remaining_gt, desc="Append unmatched GT", unit="row")
            for p, dq in gt_map.items():
                while dq:
                    gt_react = dq.popleft()
                    writer.writerow({
                        "REACTANT": gt_react,
                        "REAGENT": args.reagent_blank,
                        "PRODUCT": p
                    })
                    inserted_gt_rows += 1
                    pbar.update(1)
            pbar.close()

    print(
        f"[OK] Wrote: {out_path}\n"
        f"     test_rows={n_test}\n"
        f"     txt_lines={txt_total}, bad_lines={bad_lines}, oob_idx={oob_idx}\n"
        f"     candidate_rows_written={total_candidate_rows}\n"
        f"     gt_rows_inserted={inserted_gt_rows}, unmatched_gt_appended={remaining_gt}"
    )


if __name__ == "__main__":
    main()
