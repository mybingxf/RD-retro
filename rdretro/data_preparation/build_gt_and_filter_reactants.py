#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

from rdkit import Chem, RDLogger
from rdkit import DataStructs
from rdkit.Chem import AllChem

RDLogger.DisableLog("rdApp.*")

ATOMMAP_IN_BRACKET = re.compile(r":\d+(?=\])")  # [C:12] -> [C]


def clean_ws(s: str) -> str:
    return "".join(str(s).split())


def strip_atommap_text(s: str) -> str:
    return ATOMMAP_IN_BRACKET.sub("", s)


def canon_smiles_key(smiles: str, ignore_stereo: bool = False) -> Optional[str]:
    """
    用于 product 匹配的 key：
      - 去空格
      - 去 atom-map（如果有）
      - RDKit parse
      - canonical smiles
    ignore_stereo=True 时忽略立体信息（isomericSmiles=False）
    """
    s = clean_ws(smiles)
    if not s:
        return None
    s = strip_atommap_text(s)
    m = Chem.MolFromSmiles(s)
    if m is None:
        return None
    for a in m.GetAtoms():
        a.SetAtomMapNum(0)
    return Chem.MolToSmiles(m, canonical=True, isomericSmiles=(not ignore_stereo))


def parse_rxn_line(line: str) -> Optional[Tuple[str, str]]:
    line = clean_ws(line)
    if not line:
        return None
    if ">>" in line:
        src, prod = line.split(">>", 1)
        if not src or not prod:
            return None
        return src, prod
    if ">" in line:
        segs = line.split(">")
        if len(segs) < 3:
            return None
        src = segs[0]
        prod = segs[-1]
        if not src or not prod:
            return None
        return src, prod
    return None


def dedup_keep_order(xs: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in xs:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def normalize_group_smiles(group: str, strip_atommap: bool = True) -> Tuple[str, bool]:
    s = clean_ws(group)
    if strip_atommap:
        s = strip_atommap_text(s)
    if not s:
        return "", True

    parts = [p for p in s.split(".") if p]
    can_parts = []
    for p in parts:
        m = Chem.MolFromSmiles(p)
        if m is None:
            return s, False
        for a in m.GetAtoms():
            a.SetAtomMapNum(0)
        can_parts.append(Chem.MolToSmiles(m, canonical=True, isomericSmiles=True))

    can_parts.sort()
    return ".".join(can_parts), True


def group_fp(group_norm: str, radius: int, nbits: int,
             fp_cache: Dict[Tuple[str, int, int], Optional[DataStructs.ExplicitBitVect]]
             ) -> Optional[DataStructs.ExplicitBitVect]:
    key = (group_norm, radius, nbits)
    if key in fp_cache:
        return fp_cache[key]

    if not group_norm:
        fp_cache[key] = None
        return None

    parts = [p for p in group_norm.split(".") if p]
    fp_all = None
    for p in parts:
        m = Chem.MolFromSmiles(p)
        if m is None:
            fp_cache[key] = None
            return None
        fp = AllChem.GetMorganFingerprintAsBitVect(m, radius, nBits=nbits)
        fp_all = fp if fp_all is None else (fp_all | fp)

    fp_cache[key] = fp_all
    return fp_all


def tanimoto(fp1, fp2) -> float:
    if fp1 is None or fp2 is None:
        return 0.0
    return float(DataStructs.TanimotoSimilarity(fp1, fp2))


def build_product_to_gt(train_txt: Path, ignore_stereo_key: bool,
                        max_gt_per_product: Optional[int] = None) -> Dict[str, List[str]]:
    """
    product_key(canonical) -> [reactants...]
    """
    mp = defaultdict(list)
    bad_prod = 0

    with train_txt.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            sp = parse_rxn_line(line)
            if sp is None:
                continue
            src, prod = sp
            src = clean_ws(src)
            prod = clean_ws(prod)
            if not src or not prod:
                continue

            key = canon_smiles_key(prod, ignore_stereo=ignore_stereo_key)
            if key is None:
                bad_prod += 1
                continue
            mp[key].append(src)

    out = {}
    for k, srcs in mp.items():
        srcs = dedup_keep_order(srcs)
        if max_gt_per_product is not None and len(srcs) > max_gt_per_product:
            srcs = srcs[:max_gt_per_product]
        out[k] = srcs

    return out


def to_list(x: Any) -> List[str]:
    if x is None:
        return []
    if isinstance(x, list):
        return [str(i) for i in x]
    return [str(x)]


def choose_remove(a: str, b: str, candidates: List[str], is_gt: Dict[str, bool],
                  fp_cache: Dict[Tuple[str, int, int], Optional[DataStructs.ExplicitBitVect]],
                  radius: int, nbits: int) -> str:
    ag = is_gt.get(a, False)
    bg = is_gt.get(b, False)
    if ag and not bg:
        return b
    if bg and not ag:
        return a

    fpa = group_fp(a, radius, nbits, fp_cache)
    fpb = group_fp(b, radius, nbits, fp_cache)

    def mean_sim(x: str, fpx) -> float:
        s = 0.0
        cnt = 0
        for y in candidates:
            if y == x:
                continue
            fpy = group_fp(y, radius, nbits, fp_cache)
            s += tanimoto(fpx, fpy)
            cnt += 1
        return s / cnt if cnt > 0 else 0.0

    ma = mean_sim(a, fpa)
    mb = mean_sim(b, fpb)
    return a if ma >= mb else b


def filter_candidates(
    gt_list: List[str],
    pred_list: List[str],
    min_keep: int,
    keep_ratio: float,
    max_keep: Optional[int],
    sim_threshold: float,
    force_target: bool,
    radius: int,
    nbits: int,
) -> Tuple[List[str], Dict[str, int]]:
    is_gt = {}
    for g in gt_list:
        is_gt[g] = True
    for p in pred_list:
        is_gt.setdefault(p, False)

    candidates = dedup_keep_order(gt_list + pred_list)

    stats = {
        "n_gt": len(gt_list),
        "n_pred": len(pred_list),
        "n_cand_init": len(candidates),
        "n_removed": 0,
        "n_removed_gt": 0,
        "n_removed_pred": 0,
    }

    if not candidates:
        return [], stats

    target_keep = math.ceil(len(candidates) * keep_ratio)
    target_keep = max(target_keep, min_keep, len(gt_list))
    if max_keep is not None:
        target_keep = min(target_keep, max_keep)

    if target_keep >= len(candidates):
        return candidates, stats

    fp_cache: Dict[Tuple[str, int, int], Optional[DataStructs.ExplicitBitVect]] = {}

    def max_sim_pair(cands: List[str]) -> Tuple[float, int, int]:
        max_sim = -1.0
        mi = -1
        mj = -1
        fps = [group_fp(x, radius, nbits, fp_cache) for x in cands]
        m = len(cands)
        for i in range(m):
            for j in range(i + 1, m):
                s = tanimoto(fps[i], fps[j])
                if s > max_sim:
                    max_sim = s
                    mi, mj = i, j
        return max_sim, mi, mj

    while len(candidates) > target_keep:
        smax, i, j = max_sim_pair(candidates)
        if i < 0 or j < 0:
            break
        if smax < sim_threshold:
            break

        a, b = candidates[i], candidates[j]
        rm = choose_remove(a, b, candidates, is_gt, fp_cache, radius, nbits)
        candidates.remove(rm)
        stats["n_removed"] += 1
        if is_gt.get(rm, False):
            stats["n_removed_gt"] += 1
        else:
            stats["n_removed_pred"] += 1

    if force_target:
        while len(candidates) > target_keep:
            smax, i, j = max_sim_pair(candidates)
            if i < 0 or j < 0:
                break
            a, b = candidates[i], candidates[j]
            rm = choose_remove(a, b, candidates, is_gt, fp_cache, radius, nbits)
            candidates.remove(rm)
            stats["n_removed"] += 1
            if is_gt.get(rm, False):
                stats["n_removed_gt"] += 1
            else:
                stats["n_removed_pred"] += 1

    return candidates, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-json", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--train-txt",
                    default="/root/autodl-fs/roundtripcal/USPTO_50K_raw_train_canon_nomap.txt")
    ap.add_argument("--max-gt-per-product", type=int, default=None)

    # ✅ 新增：product 匹配 key 是否忽略立体
    ap.add_argument("--ignore-stereo-key", action="store_true",
                    help="product 匹配时忽略立体（isomericSmiles=False）")

    # filter params
    ap.add_argument("--min-keep", type=int, default=5)
    ap.add_argument("--keep-ratio", type=float, default=1.0)
    ap.add_argument("--max-keep", type=int, default=None)
    ap.add_argument("--sim-threshold", type=float, default=0.9)
    ap.add_argument("--force-target", action="store_true")
    ap.add_argument("--fp-radius", type=int, default=2)
    ap.add_argument("--fp-nbits", type=int, default=2048)
    ap.add_argument("--no-progress", action="store_true")
    args = ap.parse_args()

    in_json = Path(args.in_json).resolve()
    out_json = Path(args.out_json).resolve()
    train_txt = Path(args.train_txt).resolve()
    out_json.parent.mkdir(parents=True, exist_ok=True)

    print(f"[Load] building canonical product->gt from: {train_txt}")
    prod2gt = build_product_to_gt(train_txt, ignore_stereo_key=args.ignore_stereo_key,
                                  max_gt_per_product=args.max_gt_per_product)
    print(f"[Load] unique canonical products in train txt: {len(prod2gt)}")

    data = json.loads(in_json.read_text(encoding="utf-8", errors="ignore"))
    items = data if isinstance(data, list) else ([data] if isinstance(data, dict) and "product" in data else list(data.values()))

    it = items
    if tqdm is not None and (not args.no_progress):
        it = tqdm(items, desc="Processing items", unit="item")

    out_items = []
    missing_gt = 0
    bad_product_key = 0
    total_removed = 0

    for rec in it:
        if not isinstance(rec, dict):
            continue
        product_raw = clean_ws(rec.get("product", ""))
        if not product_raw:
            continue

        pkey = canon_smiles_key(product_raw, ignore_stereo=args.ignore_stereo_key)
        if pkey is None:
            bad_product_key += 1
            gt_raw = []
        else:
            gt_raw = prod2gt.get(pkey, [])
            if not gt_raw:
                missing_gt += 1

        pred_raw = to_list(rec.get("predicted_reactants", []))

        gt_norm = dedup_keep_order([normalize_group_smiles(x, True)[0] for x in gt_raw if normalize_group_smiles(x, True)[0]])
        pred_norm = dedup_keep_order([normalize_group_smiles(x, True)[0] for x in pred_raw if normalize_group_smiles(x, True)[0]])

        filt, st = filter_candidates(
            gt_list=gt_norm,
            pred_list=pred_norm,
            min_keep=args.min_keep,
            keep_ratio=args.keep_ratio,
            max_keep=args.max_keep,
            sim_threshold=args.sim_threshold,
            force_target=args.force_target,
            radius=args.fp_radius,
            nbits=args.fp_nbits,
        )
        total_removed += st["n_removed"]

        out_items.append({
            "product": product_raw,       # 输出保持原样
            "gt_reactants": gt_norm,
            "filter_reactants": filt,
        })

    out_json.write_text(json.dumps(out_items, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n[Done]")
    print(f"  in_json       : {in_json}")
    print(f"  out_json      : {out_json}")
    print(f"  n_items       : {len(out_items)}")
    print(f"  bad_product_key (RDKit parse fail) : {bad_product_key}")
    print(f"  missing_gt    : {missing_gt}  (canonical product key 不在 train 映射里)")
    print(f"  removed_total : {total_removed}")


if __name__ == "__main__":
    main()
#python build_gt_and_filter_reactants.py   --in-json /root/autodl-fs/try/train/localretro/matched_predicted_reactants.json   --out-json /root/autodl-fs/roundtripcal/new_filtered.json   --sim-threshold 0.90   --min-keep 5