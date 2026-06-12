# collect_matched_reactants.py
# Usage example:
#   python collect_matched_reactants.py \
#       --reactants reactants.txt \
#       --products product.txt \
#       --predicted predicted_product_tokenized.txt \
#       --predicted_is_tokenized 1 \
#       --out out.json

import argparse
import json
from functools import lru_cache
from typing import Optional, List

from rdkit import Chem


def detok_if_needed(s: str, is_tokenized: bool) -> str:
    s = s.strip()
    return s.replace(" ", "") if is_tokenized else s


@lru_cache(maxsize=200000)
def canon_smiles(smi: str) -> Optional[str]:
    """RDKit canonicalize; return None if invalid/empty."""
    if smi is None:
        return None
    smi = smi.strip()
    if not smi:
        return None
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None
        # 保险：去 atom map（pred 通常没有，但不影响）
        for a in mol.GetAtoms():
            a.SetAtomMapNum(0)
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    except Exception:
        return None


def match_pred_to_prod(pred: str, prod: str) -> bool:
    """
    True if:
      (1) canonical(pred) == canonical(prod)
      (2) or prod equals any fragment of pred split by '.'
    """
    prod_c = canon_smiles(prod)
    if prod_c is None:
        return False

    pred_c = canon_smiles(pred)
    if pred_c is not None and pred_c == prod_c:
        return True

    # containment check: product equals one fragment in predicted mixture
    # (pred may contain '.' => multiple molecules)
    for frag in pred.split("."):
        frag = frag.strip()
        if not frag:
            continue
        frag_c = canon_smiles(frag)
        if frag_c is not None and frag_c == prod_c:
            return True

    # fallback: raw string check (rarely useful, but cheap)
    if pred == prod:
        return True
    if prod in pred.split("."):
        return True

    return False


def main(
    reactants_path: str,
    products_path: str,
    predicted_path: str,
    predicted_is_tokenized: bool,
    out_json_path: str,
):
    # product -> record
    # 为了避免 JSON 过大，这里对 reactants / predicted_reactants 都做“去重”
    records = {}

    total_lines = 0
    matched_lines = 0

    with open(reactants_path, "r", encoding="utf-8") as f_reac, \
         open(products_path, "r", encoding="utf-8") as f_prod, \
         open(predicted_path, "r", encoding="utf-8") as f_pred:

        for i, (reac, prod, pred) in enumerate(zip(f_reac, f_prod, f_pred)):
            total_lines += 1
            reac = reac.strip()
            prod = prod.strip()
            pred = detok_if_needed(pred, predicted_is_tokenized)

            if not prod:
                continue

            rec = records.get(prod)
            if rec is None:
                rec = {
                    "product": prod,
                    "reactants": [],
                    "predicted_reactants": []
                }
                # 用 set 做去重（内部维护，不直接进 JSON）
                rec["_reactants_set"] = set()
                rec["_pred_reactants_set"] = set()
                records[prod] = rec

            # collect all reactants (unique)
            if reac and reac not in rec["_reactants_set"]:
                rec["_reactants_set"].add(reac)
                rec["reactants"].append(reac)

            # match
            if reac and pred and match_pred_to_prod(pred, prod):
                matched_lines += 1
                if reac not in rec["_pred_reactants_set"]:
                    rec["_pred_reactants_set"].add(reac)
                    rec["predicted_reactants"].append(reac)

            if total_lines % 200000 == 0:
                print(f"[progress] lines={total_lines}, matched_lines={matched_lines}, unique_products={len(records)}")

    # compute stats
    products_list = []
    sum_keep = 0
    for prod, rec in records.items():
        keep_n = len(rec["predicted_reactants"])
        sum_keep += keep_n

        # clean internal sets
        rec.pop("_reactants_set", None)
        rec.pop("_pred_reactants_set", None)

        products_list.append(rec)

    n_products = len(products_list)
    avg_keep = (sum_keep / n_products) if n_products > 0 else 0.0

    with open(out_json_path, "w", encoding="utf-8") as fout:
        json.dump(products_list, fout, ensure_ascii=False, indent=2)

    print("==== Summary ====")
    print(f"Total aligned lines processed: {total_lines}")
    print(f"Matched lines (pred matches/contains product): {matched_lines}")
    print(f"Unique products: {n_products}")
    print(f"Average predicted_reactants per product: {avg_keep:.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reactants", required=True, help="reactants.txt (detok, one per line)")
    ap.add_argument("--products", required=True, help="product.txt (detok, one per line)")
    ap.add_argument("--predicted", required=True, help="predicted product txt (tokenized or detok)")
    ap.add_argument("--predicted_is_tokenized", type=int, default=1, help="1 if predicted file is tokenized with spaces; 0 otherwise")
    ap.add_argument("--out", required=True, help="output json path")
    args = ap.parse_args()

    main(
        reactants_path=args.reactants,
        products_path=args.products,
        predicted_path=args.predicted,
        predicted_is_tokenized=bool(args.predicted_is_tokenized),
        out_json_path=args.out,
    )
