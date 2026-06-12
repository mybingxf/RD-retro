#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
from pathlib import Path
from collections import defaultdict, deque

import pandas as pd
from rdkit import Chem


def sanitize_smiles(smiles: str, cache: dict) -> str:
    """Remove atom-mapping and canonicalize SMILES. Uses cache to speed up repeats."""
    if smiles is None:
        return ""
    s = str(smiles).strip()
    if s == "":
        return ""

    if s in cache:
        return cache[s]

    mol = Chem.MolFromSmiles(s)
    if mol is None:
        cache[s] = s  # 解析失败就原样返回
        return s

    for atom in mol.GetAtoms():
        if atom.GetAtomMapNum():
            atom.SetAtomMapNum(0)

    out = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    cache[s] = out
    return out


def main():
    ap = argparse.ArgumentParser(
        description="Insert GT (Reactants, Products) rows from preprocessed_test.csv before first matching PRODUCT row in candidate CSV."
    )
    ap.add_argument(
        "--candidate-csv",
        default="LocalRetro_USPTO_50K_forward_input.csv",
        help="Existing candidate CSV (default: LocalRetro_USPTO_50K_forward_input.csv).",
    )
    ap.add_argument(
        "--test-csv",
        default="/root/autodl-fs/localretro/data/USPTO_50K/preprocessed_test.csv",
        help="USPTO_50K preprocessed_test.csv path.",
    )
    ap.add_argument(
        "--test-reactant-col",
        default="Reactants",
        help="Column name for reactants in test csv (default: Reactants).",
    )
    ap.add_argument(
        "--test-product-col",
        default="Products",
        help="Column name for products in test csv (default: Products).",
    )
    ap.add_argument(
        "--output-csv",
        default=None,
        help="Output path. If not set, overwrite candidate-csv (in-place).",
    )
    ap.add_argument(
        "--reagent-blank",
        default=" ",
        help="Fill value for REAGENT when absent (default: single space).",
    )
    args = ap.parse_args()

    candidate_path = Path(args.candidate_csv)
    test_path = Path(args.test_csv)
    out_path = Path(args.output_csv) if args.output_csv else candidate_path

    if not candidate_path.exists():
        raise FileNotFoundError(f"candidate csv not found: {candidate_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"test csv not found: {test_path}")

    # ---------- 1) 读取 test csv 并构建 product -> deque(reactant) ----------
    # cache 用于加速重复 SMILES 的标准化
    cache = {}

    df_test = pd.read_csv(test_path, dtype=str, keep_default_na=False)

    if args.test_reactant_col not in df_test.columns:
        raise KeyError(f"'{args.test_reactant_col}' not in {test_path}. Columns={list(df_test.columns)}")
    if args.test_product_col not in df_test.columns:
        raise KeyError(f"'{args.test_product_col}' not in {test_path}. Columns={list(df_test.columns)}")

    gt_map = defaultdict(deque)  # product_smi -> deque([reactant_smi1, reactant_smi2, ...])

    # 先对 test 的两列做标准化（按行）
    for r_smi, p_smi in zip(df_test[args.test_reactant_col], df_test[args.test_product_col]):
        p = sanitize_smiles(p_smi, cache)
        r = sanitize_smiles(r_smi, cache)
        if p.strip() == "" or r.strip() == "":
            continue
        gt_map[p].append(r)

    # ---------- 2) 流式读取 candidate csv，遇到某 product 第一次出现就插 GT ----------
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")

    seen_products = set()
    inserted_count = 0

    with candidate_path.open("r", encoding="utf-8", newline="") as fin, tmp_path.open(
        "w", encoding="utf-8", newline=""
    ) as fout:
        reader = csv.DictReader(fin)
        if reader.fieldnames is None:
            raise RuntimeError("candidate csv has no header")

        # 要求至少有这三列
        required = {"REACTANT", "REAGENT", "PRODUCT"}
        missing = required - set(reader.fieldnames)
        if missing:
            raise KeyError(f"candidate csv missing columns: {missing}. Found columns={reader.fieldnames}")

        writer = csv.DictWriter(fout, fieldnames=reader.fieldnames)
        writer.writeheader()

        for row in reader:
            # 标准化 candidate 的 PRODUCT，确保和 gt_map 对齐
            prod_raw = row.get("PRODUCT", "")
            prod = sanitize_smiles(prod_raw, cache)
            row["PRODUCT"] = prod

            # 规范 REAGENT：空就填单空格
            reagent = row.get("REAGENT", "")
            if reagent is None or str(reagent).strip() == "":
                row["REAGENT"] = args.reagent_blank

            # 第一次遇到这个 product：先插入 gt 行
            if prod not in seen_products:
                seen_products.add(prod)
                if prod in gt_map and len(gt_map[prod]) > 0:
                    # 可能有重复 product（一般 test 集不会），按原始顺序全部插
                    while gt_map[prod]:
                        gt_react = gt_map[prod].popleft()
                        gt_row = dict(row)  # 复制字段结构，保证列一致
                        gt_row["REACTANT"] = gt_react
                        gt_row["REAGENT"] = args.reagent_blank
                        gt_row["PRODUCT"] = prod
                        writer.writerow(gt_row)
                        inserted_count += 1

            # 再写当前候选行
            writer.writerow(row)

    # ---------- 3) 把未插入的 GT（candidate 中没找到对应 product）追加到文件末尾 ----------
    # 这一步在流式里不好直接追加，所以再打开 tmp 追加写
    remaining = 0
    # 统计剩余
    for p, dq in gt_map.items():
        remaining += len(dq)

    if remaining > 0:
        # 追加写末尾：保持列一致
        with tmp_path.open("a", encoding="utf-8", newline="") as fout:
            # 重新读取 header
            with candidate_path.open("r", encoding="utf-8", newline="") as fin:
                reader = csv.DictReader(fin)
                fieldnames = reader.fieldnames
            writer = csv.DictWriter(fout, fieldnames=fieldnames)

            # 追加的行里，候选 REACTANT 只能填 gt reactant；其他列按要求
            for p, dq in gt_map.items():
                while dq:
                    gt_react = dq.popleft()
                    append_row = {k: "" for k in fieldnames}
                    append_row["REACTANT"] = gt_react
                    append_row["REAGENT"] = args.reagent_blank
                    append_row["PRODUCT"] = p
                    writer.writerow(append_row)
                    inserted_count += 1

    # 原子替换（防止中断损坏原文件）
    tmp_path.replace(out_path)

    print(
        f"[OK] Wrote: {out_path}\n"
        f"     inserted_gt_rows={inserted_count}\n"
        f"     remaining_gt_appended={remaining}  (only if product not found in candidate CSV)"
    )


if __name__ == "__main__":
    main()
