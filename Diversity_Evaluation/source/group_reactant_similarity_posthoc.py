#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import re
from pathlib import Path
from typing import List, Optional, Dict, Tuple

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


ATOMMAP_IN_BRACKET = re.compile(r":\d+(?=\])")  # [C:12] -> [C]


def _require_rdkit():
    try:
        from rdkit import Chem, DataStructs
        from rdkit.Chem import rdFingerprintGenerator
        return Chem, DataStructs, rdFingerprintGenerator
    except Exception as e:
        raise RuntimeError(
            "RDKit is required for Morgan-fingerprint Tanimoto similarity.\n"
            "Install it with: conda install -c conda-forge rdkit"
        ) from e


def clean(s: str) -> str:
    return "".join(s.split())


def strip_atommap_smiles(s: str) -> str:
    return ATOMMAP_IN_BRACKET.sub("", s)


def read_lines(p: Path) -> List[str]:
    return p.read_text(encoding="utf-8", errors="ignore").splitlines()


def load_correct_flags(
    run_dir: Path,
    n_lines: int,
    group_size_arg: Optional[int],
    prefer_detail: bool = True,
) -> Tuple[List[int], int]:
    """
    Returns: (correct_flags aligned to src_raw line indices, group_size)

    Priority:
      1) roundtrip_detail.tsv (has idx + correct + rank_in_group)
      2) compute from tgt_raw + pred_clean
    """
    detail = run_dir / "roundtrip_detail.tsv"
    tgt_raw = run_dir / "tgt_raw.txt"
    pred_clean = run_dir / "pred_clean.txt"

    # (1) from detail
    if prefer_detail and detail.exists():
        flags = [0] * n_lines
        max_rank = 0
        with detail.open(
            "r", encoding="utf-8", errors="ignore", newline=""
        ) as f:
            reader = csv.DictReader(f, delimiter="\t")
            required = {"idx", "rank_in_group", "correct"}
            missing = required.difference(reader.fieldnames or [])
            if missing:
                raise RuntimeError(
                    f"{detail} is missing required columns: {sorted(missing)}"
                )
            for row in reader:
                try:
                    idx = int(row["idx"])
                    rank_in_group = int(row["rank_in_group"])
                    ok = int(row["correct"])
                except Exception:
                    continue
                if 0 <= idx < n_lines:
                    flags[idx] = 1 if ok == 1 else 0
                if rank_in_group > max_rank:
                    max_rank = rank_in_group

        group_size = group_size_arg if group_size_arg is not None else max_rank
        if group_size is None or group_size <= 0:
            raise RuntimeError(
                "Could not infer group_size from roundtrip_detail.tsv; pass --group-size."
            )
        return flags, group_size

    # (2) from tgt_raw + pred_clean
    if not tgt_raw.exists() or not pred_clean.exists():
        raise FileNotFoundError(
            "roundtrip_detail.tsv is absent and tgt_raw.txt / pred_clean.txt are "
            "incomplete; correct flags cannot be calculated."
        )

    tgts = read_lines(tgt_raw)
    preds = read_lines(pred_clean)
    n = min(n_lines, len(tgts), len(preds))
    flags = [0] * n
    for i in range(n):
        tgt = clean(tgts[i])
        pred = clean(preds[i])
        parts = pred.split(".") if pred else []
        flags[i] = 1 if (tgt and tgt in parts) else 0

    group_size = group_size_arg
    if group_size is None or group_size <= 0:
        raise RuntimeError("Pass --group-size when roundtrip_detail.tsv is absent.")

    if n < n_lines:
        flags.extend([0] * (n_lines - n))
    return flags, group_size


def reactant_set_fp(
    reactants: str,
    radius: int,
    nbits: int,
    strip_atommap: bool,
    cache: Dict[tuple, object],
):
    """
    reactants: "A.B.C"
    Return a reactant-set fingerprint formed by bitwise OR across molecules.
    """
    key = (reactants, radius, nbits, strip_atommap)
    if key in cache:
        return cache[key]

    Chem, DataStructs, rdFingerprintGenerator = _require_rdkit()
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=radius, fpSize=nbits
    )

    s = clean(reactants)
    if strip_atommap:
        s = strip_atommap_smiles(s)
    if not s:
        cache[key] = None
        return None

    parts = [p for p in s.split(".") if p]
    fp_all = None
    for p in parts:
        mol = Chem.MolFromSmiles(p)
        if mol is None:
            cache[key] = None
            return None
        fp = generator.GetFingerprint(mol)
        fp_all = fp if fp_all is None else (fp_all | fp)

    cache[key] = fp_all
    return fp_all


def pairwise_sims(fps: List[object]) -> List[float]:
    Chem, DataStructs, rdFingerprintGenerator = _require_rdkit()
    sims: List[float] = []
    m = len(fps)
    for i in range(m):
        for j in range(i + 1, m):
            if fps[i] is None or fps[j] is None:
                sims.append(0.0)
            else:
                sims.append(float(DataStructs.TanimotoSimilarity(fps[i], fps[j])))
    return sims


def agg_mean_min(sims: List[float]) -> Tuple[float, float]:
    if not sims:
        return 0.0, 0.0
    return sum(sims) / len(sims), min(sims)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default=".", help="cal.py output directory; default: current directory")
    ap.add_argument("--top-n", default=5, type=int, help="Use the first N candidates per group")
    ap.add_argument("--group-size", type=int, default=None,
                    help="Candidates per group; inferred from roundtrip_detail.tsv when omitted")

    ap.add_argument("--radius", type=int, default=2, help="Morgan radius (default=2)")
    ap.add_argument("--nbits", type=int, default=2048, help="Morgan nBits (default=2048)")
    ap.add_argument("--keep-atommap", action="store_true", help="Retain atom mapping (removed by default)")

    # Scalar used when a group contains fewer than top-n combined hits.
    ap.add_argument("--hit-fill-sim", type=float, default=0.0,
                    help="Similarity assigned when hit count < top-n (default: 0.0)")

    ap.add_argument("--no-progress", action="store_true")
    ap.add_argument("--output", default=None,
                    help="Output TSV name (default: group_reactant_similarity_topN.tsv)")
    args = ap.parse_args()

    if not (0.0 <= args.hit_fill_sim <= 1.0):
        raise ValueError("--hit-fill-sim must be in [0,1]")

    run_dir = Path(args.run_dir).resolve()
    src_raw = run_dir / "src_raw.txt"
    if not src_raw.exists():
        raise FileNotFoundError(f"Missing {src_raw}; check that --run-dir is a cal.py output directory.")

    top_n = args.top_n
    if top_n <= 0:
        raise ValueError("--top-n must be > 0")

    src_lines = read_lines(src_raw)
    n_lines = len(src_lines)

    correct_flags, group_size = load_correct_flags(
        run_dir=run_dir,
        n_lines=n_lines,
        group_size_arg=args.group_size,
        prefer_detail=True,
    )

    if top_n > group_size:
        raise ValueError(f"--top-n ({top_n}) cannot exceed group_size ({group_size})")

    n_groups = n_lines // group_size
    n_use = n_groups * group_size
    if n_use == 0:
        raise RuntimeError("Not enough data to form one complete group")
    if n_use != n_lines:
        raise RuntimeError(
            f"src_raw.txt has {n_lines} lines, which is not divisible by "
            f"group_size ({group_size})"
        )

    src_lines = src_lines[:n_use]
    correct_flags = correct_flags[:n_use]

    strip_atommap = (not args.keep_atommap)
    out_name = args.output or f"group_reactant_similarity_top{top_n}.tsv"
    out_path = run_dir / out_name
    summary_path = run_dir / f"{out_path.stem}.summary.tsv"

    fp_cache: Dict[tuple, object] = {}

    fill_groups = 0

    # Accumulators for averages over all groups.
    sum_all_mean = 0.0
    sum_all_min = 0.0
    sum_hit_mean = 0.0
    sum_hit_min = 0.0

    it = range(n_groups)
    if tqdm is not None and (not args.no_progress):
        it = tqdm(it, desc="Group similarity", unit="group")

    with out_path.open("w", encoding="utf-8") as fout:
        fout.write(
            "group\t"
            "all_used\tall_mean\tall_min\t"
            "hit_used\thit_mean\thit_min\t"
            "hit_filled\n"
        )

        for g in it:
            start = g * group_size
            grp_src = src_lines[start:start + group_size]
            grp_ok = correct_flags[start:start + group_size]

            # All-TopN: use the first N ranked candidates.
            sel_all = grp_src[:top_n]
            fps_all = [reactant_set_fp(s, args.radius, args.nbits, strip_atommap, fp_cache) for s in sel_all]
            sims_all = pairwise_sims(fps_all)
            all_mean, all_min = agg_mean_min(sims_all)

            # Hit-TopN: first N combined hits; fill groups with too few hits.
            hit_list = [s for s, ok in zip(grp_src, grp_ok) if ok == 1]
            if len(hit_list) < top_n:
                hit_used = len(hit_list)
                hit_mean, hit_min = args.hit_fill_sim, args.hit_fill_sim
                hit_filled = 1
                fill_groups += 1
            else:
                sel_hit = hit_list[:top_n]
                fps_hit = [reactant_set_fp(s, args.radius, args.nbits, strip_atommap, fp_cache) for s in sel_hit]
                sims_hit = pairwise_sims(fps_hit)
                hit_mean, hit_min = agg_mean_min(sims_hit)
                hit_used = top_n
                hit_filled = 0

            # Accumulate for averages over all groups.
            sum_all_mean += all_mean
            sum_all_min += all_min
            sum_hit_mean += hit_mean
            sum_hit_min += hit_min

            fout.write(
                f"{g}\t"
                f"{top_n}\t{all_mean:.8f}\t{all_min:.8f}\t"
                f"{hit_used}\t{hit_mean:.8f}\t{hit_min:.8f}\t"
                f"{hit_filled}\n"
            )

    # Global averages over all groups.
    all_mean_avg = sum_all_mean / n_groups
    all_min_avg = sum_all_min / n_groups
    hit_mean_avg = sum_hit_mean / n_groups
    hit_min_avg = sum_hit_min / n_groups

    with summary_path.open("w", encoding="utf-8") as f:
        f.write("metric\tvalue\n")
        f.write(f"groups_used\t{n_groups}\n")
        f.write(f"group_size\t{group_size}\n")
        f.write(f"top_n\t{top_n}\n")
        f.write(f"hit_fill_sim\t{args.hit_fill_sim}\n")
        f.write(f"hit_filled_groups\t{fill_groups}\n")
        f.write(f"all_mean_avg_over_groups\t{all_mean_avg:.10f}\n")
        f.write(f"all_min_avg_over_groups\t{all_min_avg:.10f}\n")
        f.write(f"hit_mean_avg_over_groups\t{hit_mean_avg:.10f}\n")
        f.write(f"hit_min_avg_over_groups\t{hit_min_avg:.10f}\n")

    print("\n[Done] Reactant similarity (post-hoc)")
    print(f"  run_dir            = {run_dir}")
    print(f"  group_size         = {group_size}")
    print(f"  top_n              = {top_n}")
    print(f"  groups_used        = {n_groups}")
    print(f"  hit_fill_sim       = {args.hit_fill_sim}")
    print(f"  hit_filled_groups  = {fill_groups}  (groups with hit<{top_n} use {args.hit_fill_sim})")
    print(f"  saved              = {out_path}")
    print(f"  summary            = {summary_path}")

    print("\n[Average over ALL groups]")
    print(f"  all_mean_avg = {all_mean_avg:.6f}")
    print(f"  all_min_avg  = {all_min_avg:.6f}")
    print(f"  hit_mean_avg = {hit_mean_avg:.6f}")
    print(f"  hit_min_avg  = {hit_min_avg:.6f}")


if __name__ == "__main__":
    main()
