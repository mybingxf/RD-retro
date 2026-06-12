#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import pandas as pd
from rdkit import Chem


def canon_no_atommap(smiles: str) -> str:
    """
    Remove atom-mapping numbers and canonicalize SMILES with RDKit.
    If parsing fails, return original string.
    """
    s = (smiles or "").strip()
    if not s:
        return s

    mol = Chem.MolFromSmiles(s)
    if mol is None:
        return smiles  # 保留原样，避免破坏数据

    # 去除 atom-map
    for atom in mol.GetAtoms():
        if atom.GetAtomMapNum() != 0:
            atom.SetAtomMapNum(0)

    # 标准化 / canonical
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-i", "--input",
        default="LocalRetro_USPTO_50K_forward_input.csv",
        help="Input CSV path (default: LocalRetro_USPTO_50K_forward_input.csv)"
    )
    ap.add_argument(
        "-o", "--output",
        default=None,
        help="Output CSV path. If not set, overwrite input (in-place)."
    )
    ap.add_argument(
        "--product-col",
        default="PRODUCT",
        help="Product column name (default: PRODUCT)"
    )
    args = ap.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output) if args.output else in_path

    if not in_path.exists():
        raise FileNotFoundError(f"Not found: {in_path}")

    # keep_default_na=False 防止空白被读成 NaN
    df = pd.read_csv(in_path, dtype=str, keep_default_na=False)

    if args.product_col not in df.columns:
        raise KeyError(f"Column '{args.product_col}' not in CSV. Columns={list(df.columns)}")

    # 处理 PRODUCT
    df[args.product_col] = df[args.product_col].apply(canon_no_atommap)

    # 可选：确保 REAGENT 为空时填单空格（与你模型要求一致）
    if "REAGENT" in df.columns:
        df["REAGENT"] = df["REAGENT"].apply(lambda x: " " if (x is None or str(x).strip() == "") else x)

    # 安全写入：先写临时文件再替换（防止写到一半中断损坏原文件）
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    df.to_csv(tmp_path, index=False)
    tmp_path.replace(out_path)

    print(f"[OK] Wrote: {out_path}  (rows={len(df)})")


if __name__ == "__main__":
    main()
