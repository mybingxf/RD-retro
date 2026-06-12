# filter_invalid_predicted_reactants_with_tqdm.py
# Usage:
#   python filter_invalid_predicted_reactants_with_tqdm.py --in in.json --out out.filtered.json

import argparse
import json
from rdkit import Chem
from tqdm import tqdm


def is_valid_smiles(smi: str) -> bool:
    smi = (smi or "").strip()
    if not smi:
        return False
    try:
        mol = Chem.MolFromSmiles(smi)  # sanitize=True by default
        return mol is not None
    except Exception:
        return False


def is_valid_mixture(mix: str) -> bool:
    """
    reactants 允许用 '.' 表示多个分子；要求每个 fragment 都合法
    """
    mix = (mix or "").strip()
    if not mix:
        return False
    frags = [f.strip() for f in mix.split(".") if f.strip()]
    if not frags:
        return False
    return all(is_valid_smiles(f) for f in frags)


def main(in_json: str, out_json: str):
    with open(in_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    total_pred = 0
    valid_pred = 0
    per_prod_before = []
    per_prod_after = []

    pbar = tqdm(data, desc="Filtering predicted_reactants", unit="product")

    for rec in pbar:
        preds = rec.get("predicted_reactants", [])
        total_pred += len(preds)
        per_prod_before.append(len(preds))

        filtered = []
        for r in preds:
            if is_valid_mixture(r):
                filtered.append(r)

        valid_pred += len(filtered)
        per_prod_after.append(len(filtered))
        rec["predicted_reactants"] = filtered

        # 动态显示有效率
        ratio = (valid_pred / total_pred) if total_pred else 0.0
        pbar.set_postfix(
            total_pred=total_pred,
            valid_pred=valid_pred,
            valid_ratio=f"{ratio:.3f}"
        )

    avg_before = sum(per_prod_before) / len(per_prod_before) if per_prod_before else 0.0
    avg_after = sum(per_prod_after) / len(per_prod_after) if per_prod_after else 0.0
    ratio = (valid_pred / total_pred) if total_pred else 0.0

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print("==== Validity Check ====")
    print(f"Products (unique): {len(per_prod_before)}")
    print(f"Total predicted_reactants: {total_pred}")
    print(f"Valid predicted_reactants: {valid_pred}")
    print(f"Validity ratio: {ratio:.4f}")
    print(f"Avg predicted_reactants per product: {avg_before:.4f}  ->  {avg_after:.4f}")
    print(f"Filtered JSON written to: {out_json}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_json", required=True)
    ap.add_argument("--out", dest="out_json", required=True)
    args = ap.parse_args()
    main(args.in_json, args.out_json)
