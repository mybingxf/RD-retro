# json_to_filter_reactiontxt.py
# Usage:
#   python json_to_filter_reactiontxt.py --in matched_predicted_reactants.json --out filter_reactiontxt.txt

import argparse
import json
from typing import Optional
from rdkit import Chem


def canon_smiles(smi: str) -> Optional[str]:
    smi = (smi or "").strip()
    if not smi:
        return None
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None
        for a in mol.GetAtoms():
            a.SetAtomMapNum(0)
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    except Exception:
        return None


def canon_mixture(mix: str) -> Optional[str]:
    """对 A.B.C 这种 mixture：分别 canonical 后再用 '.' 拼回去"""
    mix = (mix or "").strip()
    if not mix:
        return None
    frags = [f.strip() for f in mix.split(".") if f.strip()]
    if not frags:
        return None
    can_frags = []
    for f in frags:
        c = canon_smiles(f)
        if c is None:
            return None
        can_frags.append(c)
    return ".".join(can_frags)


def main(in_json: str, out_txt: str, do_canon: bool):
    with open(in_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    n_products = 0
    n_lines = 0
    n_skipped = 0

    with open(out_txt, "w", encoding="utf-8") as fout:
        for rec in data:
            prod = (rec.get("product") or "").strip()
            preds = rec.get("predicted_reactants_pruned") or []
            if not prod or not preds:
                continue

            n_products += 1

            if do_canon:
                prod_c = canon_smiles(prod)
                if prod_c is None:
                    n_skipped += len(preds)
                    continue
            else:
                prod_c = prod

            for r in preds:
                r = (r or "").strip()
                if not r:
                    n_skipped += 1
                    continue

                if do_canon:
                    r_c = canon_mixture(r)
                    if r_c is None:
                        n_skipped += 1
                        continue
                else:
                    r_c = r

                fout.write(f"{r_c}>>{prod_c}\n")
                n_lines += 1

    print("Done.")
    print(f"Products with >=1 predicted_reactants: {n_products}")
    print(f"Lines written: {n_lines}")
    print(f"Lines skipped: {n_skipped}")
    print(f"Output: {out_txt}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_json", required=True, help="matched_predicted_reactants.json")
    ap.add_argument("--out", dest="out_txt", required=True, help="output reaction txt path")
    ap.add_argument("--canon", type=int, default=0, help="1 to RDKit-canonicalize reactants/product; 0 to keep as-is")
    args = ap.parse_args()

    main(args.in_json, args.out_txt, do_canon=bool(args.canon))
