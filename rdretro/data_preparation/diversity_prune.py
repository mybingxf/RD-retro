#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Diversity pruning for multi-reactant-set candidates per product.

Input JSON format: List[{
  "product": "...",
  "reactants": [...],
  "predicted_reactants": [...]
}, ...]

Behavior:
- For each record: prune `predicted_reactants` if present, else `reactants` (or use --prefer-field).
- Greedy keep: keep a candidate if its max similarity to kept set < threshold.
- Similarity methods:
    - "or": OR-combined Morgan bit fingerprint + Tanimoto (fast, recommended)
    - "assign": set-to-set similarity via greedy matching of molecule fingerprints

Requires: RDKit
  conda install -c conda-forge rdkit

Optional: tqdm for progress bar
  pip install tqdm
"""

import argparse
import json
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    from rdkit import Chem, DataStructs
    from rdkit.Chem import AllChem
except Exception as e:
    raise RuntimeError(
        "RDKit is required. Install via: conda install -c conda-forge rdkit\n"
        f"Original import error: {e}"
    )

# tqdm progress bar (optional)
try:
    from tqdm import tqdm
except Exception:
    tqdm = None  # fallback: no progress bar

ION_LIKE = {
    "[Na+]", "[K+]", "[Li+]", "[Mg+2]", "[Ca+2]", "[Zn+2]", "[Cu+2]",
    "[Cl-]", "[Br-]", "[I-]", "[F-]", "[OH-]", "[NH4+]"
}

def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(obj: Any, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def split_set_smiles(set_smiles: str) -> List[str]:
    return [p.strip() for p in set_smiles.split(".") if p.strip()]

def is_single_atom_ion(mol: Chem.Mol) -> bool:
    if mol is None:
        return False
    if mol.GetNumAtoms() != 1:
        return False
    a = mol.GetAtomWithIdx(0)
    return a.GetFormalCharge() != 0

def normalize_reactant_set(
    set_smiles: str,
    drop_ions: bool = True,
    isomeric: bool = True,
) -> Tuple[Optional[str], List[str]]:
    """
    Returns:
      - normalized_set_smiles (sorted canonical SMILES joined by '.')
      - list of canonical molecule smiles (sorted)
    If any molecule cannot be parsed, returns (None, []).
    """
    parts = split_set_smiles(set_smiles)
    cans: List[str] = []
    for p in parts:
        mol = Chem.MolFromSmiles(p)
        if mol is None:
            return None, []
        p_can = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=isomeric)

        if drop_ions:
            if p_can in ION_LIKE or is_single_atom_ion(mol):
                continue

        cans.append(p_can)

    if not cans:
        return None, []
    cans.sort()
    return ".".join(cans), cans

def morgan_fp_bit(mol: Chem.Mol, radius: int, nbits: int, use_chirality: bool = True):
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits, useChirality=use_chirality)

def set_representation(
    set_smiles: str,
    method: str,
    radius: int,
    nbits: int,
    drop_ions: bool,
    isomeric: bool,
) -> Tuple[Optional[str], Optional[Union[Any, List[Any]]]]:
    """
    Returns (normalized_set_smiles, representation)
    method:
      - "or": ExplicitBitVect (OR-combined)
      - "assign": List[ExplicitBitVect] per molecule
    """
    norm, mol_smis = normalize_reactant_set(set_smiles, drop_ions=drop_ions, isomeric=isomeric)
    if norm is None:
        return None, None

    mols = []
    for ms in mol_smis:
        m = Chem.MolFromSmiles(ms)
        if m is None:
            return None, None
        mols.append(m)

    if method == "or":
        fp = None
        for m in mols:
            f = morgan_fp_bit(m, radius, nbits, use_chirality=True)
            fp = f if fp is None else (fp | f)
        return norm, fp

    if method == "assign":
        fps = [morgan_fp_bit(m, radius, nbits, use_chirality=True) for m in mols]
        return norm, fps

    raise ValueError(f"Unknown method: {method}")

def sim_or(fp_a, fp_b) -> float:
    return float(DataStructs.TanimotoSimilarity(fp_a, fp_b))

def sim_assign(fps_a: List[Any], fps_b: List[Any]) -> float:
    """
    Greedy assignment:
      - compute pairwise sims
      - repeatedly match the max pair, remove matched indices
      - score = sum(matched_sims) / max(lenA, lenB)  (unmatched count as 0)
    """
    if not fps_a or not fps_b:
        return 0.0
    la, lb = len(fps_a), len(fps_b)

    sims = []
    for i in range(la):
        for j in range(lb):
            sims.append((float(DataStructs.TanimotoSimilarity(fps_a[i], fps_b[j])), i, j))
    sims.sort(reverse=True, key=lambda x: x[0])

    used_a = set()
    used_b = set()
    total = 0.0
    for s, i, j in sims:
        if i in used_a or j in used_b:
            continue
        used_a.add(i)
        used_b.add(j)
        total += s
        if len(used_a) == la or len(used_b) == lb:
            break
    return total / float(max(la, lb))

def prune_candidates(
    candidates: List[str],
    method: str,
    threshold: float,
    max_keep: Optional[int],
    radius: int,
    nbits: int,
    drop_ions: bool,
    isomeric: bool,
    dedup: bool,
) -> Dict[str, Any]:
    reps = []
    norms = []
    origs = []

    seen_norm = set()
    invalid = 0

    for c in candidates:
        norm, rep = set_representation(
            c, method=method, radius=radius, nbits=nbits,
            drop_ions=drop_ions, isomeric=isomeric
        )
        if norm is None or rep is None:
            invalid += 1
            continue
        if dedup and norm in seen_norm:
            continue
        seen_norm.add(norm)
        origs.append(c)
        norms.append(norm)
        reps.append(rep)

    kept_idx: List[int] = []
    removed_idx: List[int] = []

    def similarity(rep_x, rep_y) -> float:
        return sim_or(rep_x, rep_y) if method == "or" else sim_assign(rep_x, rep_y)

    for i in range(len(origs)):
        if max_keep is not None and len(kept_idx) >= max_keep:
            removed_idx.append(i)
            continue
        if not kept_idx:
            kept_idx.append(i)
            continue

        max_sim = 0.0
        for k in kept_idx:
            s = similarity(reps[i], reps[k])
            if s > max_sim:
                max_sim = s
            if max_sim >= threshold:
                break

        if max_sim < threshold:
            kept_idx.append(i)
        else:
            removed_idx.append(i)

    kept = [origs[i] for i in kept_idx]
    kept_norm = [norms[i] for i in kept_idx]
    removed = [origs[i] for i in removed_idx]

    internal_div = None
    if len(kept_idx) >= 2:
        sims = []
        for a in range(len(kept_idx)):
            for b in range(a + 1, len(kept_idx)):
                sa = kept_idx[a]
                sb = kept_idx[b]
                sims.append(similarity(reps[sa], reps[sb]))
        avg_sim = sum(sims) / len(sims) if sims else 0.0
        internal_div = 1.0 - avg_sim

    return {
        "kept": kept,
        "kept_norm": kept_norm,
        "removed": removed,
        "stats": {
            "input_n": len(candidates),
            "valid_n": len(origs),
            "invalid_n": invalid,
            "kept_n": len(kept),
            "removed_n": len(removed),
            "internal_diversity": internal_div,
            "method": method,
            "threshold": threshold,
            "max_keep": max_keep,
        }
    }

def pick_field(d: Dict[str, Any], prefer: str) -> Optional[str]:
    if prefer in d and isinstance(d[prefer], list):
        return prefer
    if "predicted_reactants" in d and isinstance(d["predicted_reactants"], list):
        return "predicted_reactants"
    if "reactants" in d and isinstance(d["reactants"], list):
        return "reactants"
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True, help="input json path")
    ap.add_argument("-o", "--output", required=True, help="output json path")

    ap.add_argument("--prefer-field", default="predicted_reactants",
                    help="field to prune if exists; fallback to predicted_reactants/reactants")
    ap.add_argument("--out-field", default=None,
                    help="output field name (default: <used_field>_pruned)")

    ap.add_argument("--method", choices=["or", "assign"], default="or",
                    help="similarity method")
    ap.add_argument("--threshold", type=float, default=0.75,
                    help="max similarity threshold; keep if max_sim < threshold")
    ap.add_argument("--max-keep", type=int, default=10,
                    help="max kept candidates per product (0 means keep all that pass threshold)")
    ap.add_argument("--radius", type=int, default=2, help="Morgan radius")
    ap.add_argument("--nbits", type=int, default=2048, help="Morgan nBits")

    ap.add_argument("--no-drop-ions", action="store_true", help="do not drop simple ions/counterions")
    ap.add_argument("--no-isomeric", action="store_true", help="do not keep stereochemistry in canonical smiles")
    ap.add_argument("--no-dedup", action="store_true", help="do not deduplicate by normalized set smiles")

    # NEW: optional summary output
    ap.add_argument("--summary-path", default=None,
                    help="optional path to write a global summary json")

    args = ap.parse_args()

    data = load_json(args.input)
    if not isinstance(data, list):
        raise ValueError("Input JSON must be a list of records.")

    drop_ions = not args.no_drop_ions
    isomeric = not args.no_isomeric
    dedup = not args.no_dedup
    max_keep = None if args.max_keep == 0 else args.max_keep

    kept_counts: List[int] = []
    divs: List[float] = []
    processed_products = 0

    iterator = data
    if tqdm is not None:
        iterator = tqdm(data, desc="Pruning per product", unit="prod")

    for rec in iterator:
        if not isinstance(rec, dict):
            continue
        field = pick_field(rec, args.prefer_field)
        if field is None:
            continue
        cand_list = rec.get(field, [])
        if not isinstance(cand_list, list) or not cand_list:
            continue

        result = prune_candidates(
            candidates=cand_list,
            method=args.method,
            threshold=args.threshold,
            max_keep=max_keep,
            radius=args.radius,
            nbits=args.nbits,
            drop_ions=drop_ions,
            isomeric=isomeric,
            dedup=dedup,
        )

        out_field = args.out_field or f"{field}_pruned"
        rec[out_field] = result["kept"]
        rec[out_field + "_canonical"] = result["kept_norm"]
        rec[out_field + "_removed"] = result["removed"]
        rec[out_field + "_stats"] = result["stats"]

        kept_counts.append(result["stats"]["kept_n"])
        if result["stats"]["internal_diversity"] is not None:
            divs.append(result["stats"]["internal_diversity"])
        processed_products += 1

    save_json(data, args.output)

    # NEW: global stats printed clearly
    if processed_products > 0:
        avg_kept = sum(kept_counts) / processed_products
        min_kept = min(kept_counts) if kept_counts else 0
        max_kept = max(kept_counts) if kept_counts else 0
        avg_div = (sum(divs) / len(divs)) if divs else None

        print(f"\n[Done] Products processed: {processed_products}")
        print(f"  Avg kept per product: {avg_kept:.3f}")
        print(f"  Min/Max kept: {min_kept} / {max_kept}")
        if avg_div is not None:
            print(f"  Avg internal diversity (kept): {avg_div:.3f}")

        if args.summary_path:
            summary = {
                "products_processed": processed_products,
                "avg_kept_per_product": avg_kept,
                "min_kept": min_kept,
                "max_kept": max_kept,
                "avg_internal_diversity": avg_div,
                "method": args.method,
                "threshold": args.threshold,
                "max_keep": max_keep,
                "radius": args.radius,
                "nbits": args.nbits,
                "drop_ions": drop_ions,
                "isomeric": isomeric,
                "dedup": dedup,
            }
            save_json(summary, args.summary_path)
            print(f"  Summary saved to: {args.summary_path}")
    else:
        print("\n[Done] No products were processed (no suitable field found or empty lists).")

if __name__ == "__main__":
    main()
