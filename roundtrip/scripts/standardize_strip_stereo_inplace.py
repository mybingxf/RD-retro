#!/usr/bin/env python3
# standardize_strip_stereo_inplace.py

import os
import sys
import argparse
from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

def _strip_trailing_annotation(s: str) -> str:
    # e.g. MIT style: "...>>... 1-2;2-13" 只取空格前第一段
    return s.strip().split(" ")[0].strip()

def _std_one_smiles(smiles: str) -> (str, bool):
    """
    返回: (标准化后smiles, 是否成功)
    - 去立体：MolToSmiles(isomericSmiles=False) => 不输出 @ / \
    - 保留 atom-map（如果原本就有），RDKit 会继续输出 :num
    """
    smi = smiles.strip()
    if smi == "":
        return smi, True

    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return smiles, False

    # 去立体 + canonical
    out = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
    return out, True

def _process_side(side: str):
    # side 可能是 "A.B.C"；逐个分子标准化
    side = side.strip()
    if side == "":
        return side, 0, 0

    parts = side.split(".")
    new_parts = []
    ok = 0
    bad = 0
    for p in parts:
        if p == "":
            continue
        new_p, success = _std_one_smiles(p)
        new_parts.append(new_p)
        if success:
            ok += 1
        else:
            bad += 1
    return ".".join(new_parts), ok, bad

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_path", required=True, help="输入txt，每行 reactants>>product")
    ap.add_argument("--backup", action="store_true", help="是否保留 .bak 备份（强烈建议开）")
    ap.add_argument("--keep_empty_lines", action="store_true", help="是否保留空行（默认丢弃空行）")
    args = ap.parse_args()

    in_path = args.in_path
    if not os.path.isfile(in_path):
        print(f"[ERROR] not found: {in_path}", file=sys.stderr)
        sys.exit(1)

    tmp_path = in_path + ".tmp"
    bak_path = in_path + ".bak"

    total = 0
    kept = 0
    bad_mol = 0
    bad_lines = 0

    with open(in_path, "r", encoding="utf-8") as fin, open(tmp_path, "w", encoding="utf-8") as fout:
        for line in fin:
            total += 1
            raw = line.rstrip("\n")
            s = raw.strip()
            if s == "":
                if args.keep_empty_lines:
                    fout.write("\n")
                    kept += 1
                continue

            if ">>" not in s:
                # 非反应行：原样写回（也可选择跳过）
                fout.write(raw + "\n")
                kept += 1
                bad_lines += 1
                continue

            left, right = s.split(">>", 1)
            left = _strip_trailing_annotation(left)
            right = _strip_trailing_annotation(right)

            new_left, okL, badL = _process_side(left)
            new_right, okR, badR = _process_side(right)

            bad_mol += (badL + badR)

            fout.write(f"{new_left}>>{new_right}\n")
            kept += 1

    # 覆盖原文件（可选备份）
    if args.backup:
        if os.path.exists(bak_path):
            os.remove(bak_path)
        os.replace(in_path, bak_path)  # 原文件 -> .bak
        os.replace(tmp_path, in_path)  # tmp -> 原文件名
        print(f"[OK] replaced in-place with backup: {bak_path}")
    else:
        os.replace(tmp_path, in_path)
        print("[OK] replaced in-place (no backup)")

    print("=== stats ===")
    print("in_path:", in_path)
    print("total_lines:", total)
    print("written_lines:", kept)
    print("bad_lines(no >>):", bad_lines)
    print("bad_molecules(rdkit parse failed, kept original):", bad_mol)

if __name__ == "__main__":
    main()
