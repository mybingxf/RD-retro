#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
import logging

from tqdm import tqdm
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

logging.basicConfig(format='[%(levelname)s] %(message)s', level=logging.INFO)


def canon_smiles(smiles: str) -> Optional[str]:
    """Canonicalize SMILES (no atom-map in input); return None if invalid."""
    if smiles is None:
        return None
    s = "".join(smiles.split())
    if not s:
        return None
    m = Chem.MolFromSmiles(s)
    if m is None:
        return None
    return Chem.MolToSmiles(m, canonical=True, isomericSmiles=True)


def parse_rxn_line(line: str) -> Optional[Tuple[str, str]]:
    line = line.strip()
    if not line or ">>" not in line:
        return None
    r, p = line.split(">>", 1)
    r = "".join(r.split())
    p = "".join(p.split())
    if not r or not p:
        return None
    return r, p


def bond_sig(mol: Chem.Mol, use_bond_type: bool = True) -> Set[Tuple]:
    """
    Return set of bonds keyed by atom-map nums.
    If use_bond_type: (m1,m2,bt) else: (m1,m2)
    """
    out = set()
    for b in mol.GetBonds():
        a1, a2 = b.GetBeginAtom(), b.GetEndAtom()
        m1, m2 = a1.GetAtomMapNum(), a2.GetAtomMapNum()
        if m1 == 0 or m2 == 0:
            continue
        i, j = (m1, m2) if m1 < m2 else (m2, m1)
        if use_bond_type:
            if b.GetIsAromatic():
                bt = "AR"
            else:
                bt = b.GetBondType()
                if bt == Chem.BondType.SINGLE:
                    bt = "S"
                elif bt == Chem.BondType.DOUBLE:
                    bt = "D"
                elif bt == Chem.BondType.TRIPLE:
                    bt = "T"
                else:
                    bt = str(bt)
            out.add((i, j, bt))
        else:
            out.add((i, j))
    return out


def disconnection_signature(mapped_rxn: str, use_bond_type: bool = True) -> Optional[Set[Tuple]]:
    """
    For mapped reaction R>>P:
      formed_bonds = bonds(P) - bonds(R)
    This is the "disconnection" in retrosynthesis view.
    """
    if not mapped_rxn or ">>" not in mapped_rxn:
        return None
    r, p = mapped_rxn.split(">>", 1)
    rm = Chem.MolFromSmiles(r)
    pm = Chem.MolFromSmiles(p)
    if rm is None or pm is None:
        return None
    rb = bond_sig(rm, use_bond_type=use_bond_type)
    pb = bond_sig(pm, use_bond_type=use_bond_type)
    formed = pb - rb
    return formed


def jaccard(a: Set[Tuple], b: Set[Tuple]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def chunked(lst, n):
    for i in range(0, len(lst), n):
        yield i, lst[i:i+n]


def build_rxnmapper(device: str):
    # import must be AFTER CUDA_VISIBLE_DEVICES set
    from rxnmapper import RXNMapper
    try:
        return RXNMapper(device=device)
    except TypeError:
        return RXNMapper()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True, help="reactions txt, each line: reactants>>product (NO atom mapping)")
    ap.add_argument("-o", "--outdir", required=True, help="output directory")
    ap.add_argument("--batch", type=int, default=128, help="mapping batch size (default 128)")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="rxnmapper device (default cuda)")
    ap.add_argument("--gpu-id", type=int, default=0, help="which GPU to use when device=cuda (default 0)")
    ap.add_argument("--use-bond-type", action="store_true", help="use bond type in signature (default: False)")
    ap.add_argument("--max-lines", type=int, default=0, help="debug: only process first N lines (0=all)")
    ap.add_argument("--pairs-limit-per-product", type=int, default=0,
                    help="if >0, keep only top-N most similar pairs per product to reduce output size")

    # ===== NEW: block grouping mode =====
    ap.add_argument("--group-size", type=int, default=0,
                    help="if >0, every group-size lines are treated as ONE product-group (do NOT merge across groups)")
    ap.add_argument("--top-k", type=int, default=0,
                    help="if >0 with group-size, only use first top-k lines in each group for similarity/mapping (others ignored)")
    args = ap.parse_args()

    # --- GPU selection (robust): restrict visible GPUs to one card ---
    if args.device == "cuda":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
        device_for_mapper = "cuda"
    else:
        device_for_mapper = "cpu"

    inp = Path(args.input).resolve()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    pairs_tsv = outdir / "pairs.tsv"
    summary_tsv = outdir / "summary.tsv"
    mapped_jsonl = outdir / "mapped_and_sig.jsonl"
    global_txt = outdir / "global_metrics.txt"

    # 1) read reactions
    print(f"[Read] Reading lines from {inp} ...")
    lines = inp.read_text(encoding="utf-8", errors="ignore").splitlines()
    if args.max_lines and args.max_lines > 0:
        lines = lines[: args.max_lines]

    rxns_raw: List[str] = []
    products_can: List[str] = []

    # ===== NEW: groups by fixed blocks =====
    use_block_grouping = args.group_size and args.group_size > 0
    group_size = int(args.group_size) if use_block_grouping else 0
    top_k = int(args.top_k) if (use_block_grouping and args.top_k and args.top_k > 0) else (group_size if use_block_grouping else 0)

    # group_id -> list of indices (in rxns_raw)
    groups_by_gid: Dict[int, List[int]] = defaultdict(list)
    group_product: Dict[int, str] = {}  # group_id -> product (canonical) for display

    bad = 0
    if use_block_grouping:
        print(f"[Group] block mode ON: group_size={group_size}, top_k={top_k}")
    else:
        print(f"[Group] block mode OFF: fallback to auto product grouping (original behavior)")

    for line_no, line in enumerate(lines):
        # if block mode: only keep first top_k lines in each group
        if use_block_grouping:
            gid = line_no // group_size
            within = line_no % group_size
            if within >= top_k:
                continue  # ignore non-topk lines in this block
        else:
            gid = None  # not used

        sp = parse_rxn_line(line)
        if sp is None:
            bad += 1
            continue
        r, p = sp
        p_can = canon_smiles(p)
        if not p_can:
            bad += 1
            continue

        rxns_raw.append(f"{r}>>{p_can}")
        products_can.append(p_can)

        if use_block_grouping:
            idx = len(rxns_raw) - 1
            groups_by_gid[gid].append(idx)
            if gid not in group_product:
                group_product[gid] = p_can
            elif group_product[gid] != p_can:
                logging.warning(f"Group {gid}: product mismatch detected (expected {group_product[gid]}, got {p_can})")

    n = len(rxns_raw)
    print(f"[Parse] input_lines={len(lines)}, used={n}, bad/skip={bad}")
    if n == 0:
        raise RuntimeError("No valid reactions parsed from input.")

    # 2) map reactions (RXNMapper)
    print(f"[Map] init RXNMapper on device={device_for_mapper} (CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES','')}) ...")
    rxn_mapper = build_rxnmapper(device_for_mapper)

    mapped_rxn: List[Optional[str]] = [None] * n
    sigs: List[Optional[Set[Tuple]]] = [None] * n
    skipped_map_count = 0

    for start, batch_rxns in tqdm(list(chunked(rxns_raw, args.batch)), desc="Atom-mapping", unit="batch"):
        results = []
        try:
            results = rxn_mapper.get_attention_guided_atom_maps(batch_rxns)
        except Exception:
            # fallback to single
            for single_rxn in batch_rxns:
                try:
                    res_list = rxn_mapper.get_attention_guided_atom_maps([single_rxn])
                    results.append(res_list[0])
                except Exception:
                    results.append({})

        for k, d in enumerate(results):
            i = start + k
            mr = d.get("mapped_rxn") or d.get("mapped_reaction") or d.get("mapped_rxn_smiles")
            mapped_rxn[i] = mr
            if mr:
                try:
                    sigs[i] = disconnection_signature(mr, use_bond_type=args.use_bond_type)
                except Exception:
                    sigs[i] = None
            if mapped_rxn[i] is None:
                skipped_map_count += 1

    print(f"[Map] Finished. Skipped/Failed mapping count: {skipped_map_count}")

    # dump mapped + signatures
    with mapped_jsonl.open("w", encoding="utf-8") as f:
        for i in range(n):
            sig_list = None
            if sigs[i] is not None:
                sig_list = [list(x) for x in sorted(sigs[i])]
            f.write(json.dumps({
                "i": i,
                "product": products_can[i],
                "rxn_input": rxns_raw[i],
                "mapped_rxn": mapped_rxn[i],
                "sig": sig_list,
            }, ensure_ascii=False) + "\n")

    # 3) build groups (changed only here)
    if use_block_grouping:
        # groups are blocks; DO NOT merge groups with same product
        max_gid = max(groups_by_gid.keys()) if groups_by_gid else -1
        group_list: List[Tuple[str, List[int]]] = []
        for gid in range(max_gid + 1):
            idxs = groups_by_gid.get(gid, [])
            prod = group_product.get(gid, f"GROUP_{gid}")
            group_list.append((prod, idxs))
        total_products = len(group_list)
    else:
        # original behavior: group by canonical product
        groups: Dict[str, List[int]] = defaultdict(list)
        for i, p in enumerate(products_can):
            groups[p].append(i)
        group_list = list(groups.items())
        total_products = len(groups)

    # global metrics accumulators
    global_sum_avg_sim_include_all = 0.0
    global_sum_avg_sim_only_pairs = 0.0
    n_products_with_pairs = 0

    global_sum_pair_sims = 0.0
    global_n_pairs = 0

    # 4) compute similarities within each group
    with pairs_tsv.open("w", encoding="utf-8") as fpairs, \
         summary_tsv.open("w", encoding="utf-8") as fsum:

        fpairs.write("product\tn_sig_valid\ti\tj\tsim_jaccard\tsig_i\tsig_j\n")
        fsum.write("product\tn_total\tn_sig_valid\tn_unique_sig\tavg_sim\tmin_sim\tmax_sim\n")

        for product, idxs in tqdm(group_list, desc="Per-product similarity", unit="prod"):
            valid = [i for i in idxs if sigs[i] is not None]
            m = len(valid)

            if m <= 1:
                fsum.write(f"{product}\t{len(idxs)}\t{m}\t{m}\t0.000000\t0.000000\t0.000000\n")
                continue

            uniq = set(frozenset(sigs[i]) for i in valid)
            n_uniq = len(uniq)

            pair_rows = []
            sims_all = []

            for a in range(m):
                ia = valid[a]
                sa = sigs[ia]
                for b in range(a + 1, m):
                    ib = valid[b]
                    sb = sigs[ib]
                    sim = jaccard(sa, sb)
                    sims_all.append(sim)
                    pair_rows.append((sim, ia, ib))

            avg_sim = sum(sims_all) / len(sims_all) if sims_all else 0.0
            min_sim = min(sims_all) if sims_all else 0.0
            max_sim = max(sims_all) if sims_all else 0.0

            global_sum_avg_sim_include_all += avg_sim
            global_sum_avg_sim_only_pairs += avg_sim
            n_products_with_pairs += 1

            global_sum_pair_sims += sum(sims_all)
            global_n_pairs += len(sims_all)

            fsum.write(f"{product}\t{len(idxs)}\t{m}\t{n_uniq}\t{avg_sim:.6f}\t{min_sim:.6f}\t{max_sim:.6f}\n")

            if args.pairs_limit_per_product and args.pairs_limit_per_product > 0:
                pair_rows.sort(reverse=True, key=lambda x: x[0])
                pair_rows = pair_rows[: args.pairs_limit_per_product]

            for sim, ia, ib in pair_rows:
                sig_i = ";".join([",".join(map(str, t)) for t in sorted(sigs[ia])])
                sig_j = ";".join([",".join(map(str, t)) for t in sorted(sigs[ib])])
                fpairs.write(f"{product}\t{m}\t{ia}\t{ib}\t{sim:.6f}\t{sig_i}\t{sig_j}\n")

    global_avg = global_sum_avg_sim_include_all / max(total_products, 1)
    global_avg_pairs_only = (global_sum_avg_sim_only_pairs / n_products_with_pairs) if n_products_with_pairs > 0 else 0.0
    global_pair_weighted = (global_sum_pair_sims / global_n_pairs) if global_n_pairs > 0 else 0.0

    metrics_text = (
        f"total_products: {total_products}\n"
        f"products_with_pairs(m>=2): {n_products_with_pairs}\n"
        f"GLOBAL_AVG_sim_per_product_include_all(m<=1 as 0): {global_avg:.8f}\n"
        f"GLOBAL_AVG_sim_per_product_only_pairs(m>=2): {global_avg_pairs_only:.8f}\n"
        f"GLOBAL_AVG_sim_weighted_by_pairs: {global_pair_weighted:.8f}\n"
        f"global_total_pairs: {global_n_pairs}\n"
        f"device_for_mapper: {device_for_mapper}\n"
        f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES','')}\n"
        f"skipped_map_count: {skipped_map_count}\n"
        f"group_mode: {'block' if use_block_grouping else 'auto_product'}\n"
        f"group_size: {group_size}\n"
        f"top_k: {top_k}\n"
    )
    global_txt.write_text(metrics_text, encoding="utf-8")

    print("\n[Done]")
    print(f"  outdir : {outdir}")
    print(f"  summary: {summary_tsv}")
    print(f"  pairs  : {pairs_tsv}")
    print(f"  mapped : {mapped_jsonl}")
    print(f"  global : {global_txt}")
    print("\n" + metrics_text)


if __name__ == "__main__":
    main()
