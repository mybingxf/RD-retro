#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

from rdkit import Chem, RDLogger, DataStructs
from rdkit.Chem import AllChem

RDLogger.DisableLog("rdApp.*")

ATOMMAP_IN_BRACKET = re.compile(r":\d+(?=\])")  # [C:12] -> [C]


def clean_ws(s: str) -> str:
    return "".join(str(s).split())


def strip_atommap_text(s: str) -> str:
    return ATOMMAP_IN_BRACKET.sub("", s)


def dedup_keep_order(xs: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in xs:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def normalize_group_smiles(group: str, strip_atommap: bool = True) -> Tuple[str, bool]:
    """
    把一个“反应物组”（可能含多个分子，用 . 分隔）标准化成可比较 key：
      - 去空格
      - 可选去 atom-map
      - 每个分子 RDKit canonicalize（保留立体）
      - 组内分子排序后再用 '.' 拼回
    返回: (norm, ok)
    """
    s = clean_ws(group)
    if strip_atommap:
        s = strip_atommap_text(s)
    if not s:
        return "", False

    parts = [p for p in s.split(".") if p]
    can_parts = []
    for p in parts:
        m = Chem.MolFromSmiles(p)
        if m is None:
            return "", False
        for a in m.GetAtoms():
            a.SetAtomMapNum(0)
        can_parts.append(Chem.MolToSmiles(m, canonical=True, isomericSmiles=True))

    can_parts.sort()
    return ".".join(can_parts), True


def group_fp(
    group_norm: str,
    radius: int,
    nbits: int,
    fp_cache: Dict[Tuple[str, int, int], Optional[DataStructs.ExplicitBitVect]],
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


def build_sim_matrix(fps: List[DataStructs.ExplicitBitVect]):
    """
    返回 sim 矩阵（diag = -1，方便找最大相似对时跳过自相似）
    sim[i][j] = Tanimoto(fp_i, fp_j)
    """
    import numpy as np

    n = len(fps)
    sim = np.zeros((n, n), dtype=float)
    for i in range(n):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps)  # list[float], self=1
        sim[i, :] = sims
    # diag 设置成 -1，避免 max 取到 self
    for i in range(n):
        sim[i, i] = -1.0
    return sim


def prune_to_k(
    candidates: List[str],
    is_gt: List[bool],
    k: int,
    fp_radius: int,
    fp_nbits: int,
) -> List[str]:
    """
    按“上面删除逻辑”剪到 k：
      1) 每次找最相似的一对
      2) 优先删非 GT
      3) 否则删“平均相似度更高”的那个（更冗余）
    """
    if len(candidates) <= k:
        return candidates

    import numpy as np

    fp_cache: Dict[Tuple[str, int, int], Optional[DataStructs.ExplicitBitVect]] = {}
    fps = []
    keep_idx = []
    for idx, c in enumerate(candidates):
        fp = group_fp(c, fp_radius, fp_nbits, fp_cache)
        if fp is None:
            continue
        fps.append(fp)
        keep_idx.append(idx)

    # 如果都算不出指纹：退化为“保留前 k 个”
    if len(fps) == 0:
        return candidates[:k]

    # 注意：我们只对“能算出 fp 的候选”做相似度剪枝
    cand = [candidates[i] for i in keep_idx]
    gt = [is_gt[i] for i in keep_idx]

    if len(cand) <= k:
        return cand

    sim = build_sim_matrix(fps)  # diag=-1
    n = len(cand)

    while n > k:
        # 找全局最大相似的一对
        flat_idx = int(np.argmax(sim))
        i = flat_idx // n
        j = flat_idx % n
        if i == j:
            # 极端情况下 diag 没处理好时兜底
            sim[i, i] = -1.0
            continue

        # decide remove
        ag = gt[i]
        bg = gt[j]
        if ag and not bg:
            rm = j
        elif bg and not ag:
            rm = i
        else:
            # 平均相似度（对其它候选），用 sim.sum + 1 抵消 diag=-1
            # sum_others = row_sum + 1
            row_sum_i = float(sim[i, :].sum() + 1.0)
            row_sum_j = float(sim[j, :].sum() + 1.0)
            mean_i = row_sum_i / (n - 1)
            mean_j = row_sum_j / (n - 1)
            rm = i if mean_i >= mean_j else j

        # 删除 rm：删掉 rm 行/列
        cand.pop(rm)
        gt.pop(rm)
        sim = np.delete(sim, rm, axis=0)
        sim = np.delete(sim, rm, axis=1)
        n -= 1

    return cand


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-json", required=True)
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--out-txt", default=None)

    ap.add_argument("--k", type=int, default=10, help="每个 product 最多保留多少个 reactants")
    ap.add_argument("--fp-radius", type=int, default=2)
    ap.add_argument("--fp-nbits", type=int, default=2048)
    ap.add_argument("--no-progress", action="store_true")

    # 字段名可改（你这份文件默认就是 reactants / gt_reactants）
    ap.add_argument("--reactants-key", default="reactants")
    ap.add_argument("--gt-key", default="gt_reactants")
    args = ap.parse_args()

    in_path = Path(args.in_json)
    if args.out_json is None:
        out_json = in_path.with_suffix("").as_posix() + f"_filtered{args.k}.json"
        out_json = Path(out_json)
    else:
        out_json = Path(args.out_json)

    if args.out_txt is None:
        out_txt = in_path.with_suffix("").as_posix() + f"_filtered{args.k}.txt"
        out_txt = Path(out_txt)
    else:
        out_txt = Path(args.out_txt)

    data = json.loads(in_path.read_text(encoding="utf-8", errors="ignore"))
    if not isinstance(data, list):
        raise ValueError("输入 JSON 需要是 list[dict] 结构。")

    it = data
    if tqdm is not None and (not args.no_progress):
        it = tqdm(data, desc="Processing", unit="item")

    dist = {i: 0 for i in range(1, args.k + 1)}
    n_zero = 0
    n_items = 0
    rxn_lines: List[str] = []
    out_items = []

    for rec in it:
        if not isinstance(rec, dict):
            continue

        product = clean_ws(rec.get("product", ""))
        product = strip_atommap_text(product)
        if not product:
            continue

        n_items += 1

        raw_reacts = rec.get(args.reactants_key, []) or []
        raw_gt = rec.get(args.gt_key, []) or []

        if not isinstance(raw_reacts, list):
            raw_reacts = [raw_reacts]
        if not isinstance(raw_gt, list):
            raw_gt = [raw_gt]

        # 规范化 + 过滤掉 RDKit 解析失败的
        gt_norm = []
        for x in raw_gt:
            nx, ok = normalize_group_smiles(x, True)
            if ok and nx:
                gt_norm.append(nx)

        react_norm = []
        for x in raw_reacts:
            nx, ok = normalize_group_smiles(x, True)
            if ok and nx:
                react_norm.append(nx)

        gt_norm = dedup_keep_order(gt_norm)
        react_norm = dedup_keep_order(react_norm)

        # 合并候选：GT 放前面，保证同款“优先保留 GT”的语义
        candidates = dedup_keep_order(gt_norm + react_norm)
        is_gt = [c in set(gt_norm) for c in candidates]

        # ≤k 不删；>k 剪到 k
        if len(candidates) <= args.k:
            kept = candidates
        else:
            kept = prune_to_k(
                candidates=candidates,
                is_gt=is_gt,
                k=args.k,
                fp_radius=args.fp_radius,
                fp_nbits=args.fp_nbits,
            )

        n_kept = len(kept)
        if n_kept == 0:
            n_zero += 1
        else:
            if n_kept > args.k:
                # 理论上不会发生；兜底
                n_kept = args.k
                kept = kept[:args.k]
            dist[n_kept] += 1

        # 写反应式：每个 kept reactant 一行
        for r in kept:
            rxn_lines.append(f"{r}>>{product}")

        out_items.append({
            "product": product,
            "gt_reactants_norm": gt_norm,
            "reactants_norm": react_norm,
            f"kept_reactants_top{args.k}": kept,
        })

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_txt.parent.mkdir(parents=True, exist_ok=True)

    out_json.write_text(json.dumps(out_items, ensure_ascii=False, indent=2), encoding="utf-8")
    out_txt.write_text("\n".join(rxn_lines) + ("\n" if rxn_lines else ""), encoding="utf-8")

    # 打印统计
    denom = max(1, (n_items - n_zero))
    print("\n[Done]")
    print(f"  in_json   : {in_path}")
    print(f"  out_json  : {out_json}")
    print(f"  out_txt   : {out_txt}")
    print(f"  n_items   : {n_items}")
    print(f"  n_zero    : {n_zero}  (该条所有 reactants/gt 都解析失败导致为0，已不计入1~k占比)")
    print(f"  n_lines   : {len(rxn_lines)}  (写入 txt 的反应式行数)")

    print("\n[Distribution of kept reactants count (1~k)]")
    for i in range(1, args.k + 1):
        ratio = dist[i] / denom
        print(f"  {i:2d}: {dist[i]:8d}  ({ratio:.4%})")


if __name__ == "__main__":
    main()
