#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import math
import re
import subprocess
import csv
import sys
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Set

from tqdm import tqdm

SMI_REGEX = re.compile(
    r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
)

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MT_DIR = PACKAGE_ROOT / "third_party" / "MolecularTransformer"

def tokenize_smiles(s: str) -> str:
    s = "".join(s.split())
    if not s:
        return ""
    toks = SMI_REGEX.findall(s)
    if not toks or "".join(toks) != s:
        return " ".join(list(s))
    return " ".join(toks)

def split_rxn_line(line: str) -> Optional[Tuple[str, str]]:
    line = line.strip()
    if not line or ">>" not in line:
        return None
    src, tgt = line.split(">>", 1)
    src, tgt = src.strip(), tgt.strip()
    if not src or not tgt:
        return None
    return src, tgt

def safe_outdir_for_input(input_path: Path) -> Path:
    base_dir = input_path.parent
    stem = input_path.stem
    out = base_dir / stem
    if not out.exists():
        out.mkdir(parents=True, exist_ok=False)
        return out
    for k in range(1, 1000):
        cand = base_dir / f"{stem}_{k}"
        if not cand.exists():
            cand.mkdir(parents=True, exist_ok=False)
            return cand
    raise RuntimeError("Too many existing output dirs.")

def run_translate_chunk(mt_dir: Path, model_name: str, src_tokenized: Path, out_pred_tokenized: Path,
                        batch_size: int, max_length: int, gpu: int) -> None:
    translate_py = mt_dir / "translate.py"
    model_path = mt_dir / "experiments" / "models" / model_name
    if not translate_py.exists():
        raise FileNotFoundError(f"translate.py not found: {translate_py}")
    if not model_path.exists():
        raise FileNotFoundError(f"model not found: {model_path}")

    cmd = [
        sys.executable, "translate.py",
        "-model", str(model_path),
        "-src", str(src_tokenized),
        "-output", str(out_pred_tokenized),
        "-batch_size", str(batch_size),
        "-replace_unk",
        "-max_length", str(max_length),
        "-fast",
        "-gpu", str(gpu),
    ]
    subprocess.run(cmd, cwd=str(mt_dir), check=True)

def clean_smiles(s: str) -> str:
    return "".join(s.split())

def compute_group_acc(correct_flags: List[int], group_size: int, max_k: int):
    n = len(correct_flags)
    g = n // group_size
    if g == 0:
        raise RuntimeError("Not enough lines to form a group.")
    correct_flags = correct_flags[: g * group_size]

    rows = []
    for k in range(1, max_k + 1):
        topk_hit = 0
        rankk_hit = 0
        for gi in range(g):
            start = gi * group_size
            grp = correct_flags[start:start + group_size]
            if any(grp[:k]):
                topk_hit += 1
            if grp[k - 1] == 1:
                rankk_hit += 1
        rows.append((k, topk_hit / g, rankk_hit / g))
    return g, rows

def mean_rank_acc(rows: List[Tuple[int, float, float]]) -> float:
    """Mean of rank-k accuracy over all reported k values."""
    if not rows:
        return 0.0
    return sum(rankk for _, _, rankk in rows) / len(rows)

def write_group_acc(path: Path, rows: List[Tuple[int, float, float]], desc: str) -> float:
    """
    Write K/top-k/rank-k accuracy and append one summary row for mean rank-k acc.
    The summary row keeps the same TSV schema:
      K = AVG_RANKK, topk_acc = NA, rankk_acc = mean(rank1_acc..rankK_acc)
    """
    avg_rankk = mean_rank_acc(rows)
    with path.open("w", encoding="utf-8") as fg:
        fg.write("K\ttopk_acc\trankk_acc\n")
        for (k, topk, rankk) in tqdm(rows, desc=desc, unit="K"):
            fg.write(f"{k}\t{topk:.8f}\t{rankk:.8f}\n")
        fg.write(f"AVG_RANKK\tNA\t{avg_rankk:.8f}\n")
    return avg_rankk

def count_lines(path: Path) -> int:
    n = 0
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for _ in f:
            n += 1
    return n

# =========================
# Condition A utilities
# =========================
def _require_rdkit():
    try:
        from rdkit import Chem  # noqa
        return True
    except Exception:
        return False

def normalize_smiles_single(smiles: str) -> Optional[str]:
    """
    RDKit parse -> remove atom map -> canonicalize
    Return canonical SMILES (string) or None if invalid.
    """
    smiles = clean_smiles(smiles)
    if not smiles:
        return None
    from rdkit import Chem
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    for a in mol.GetAtoms():
        a.SetAtomMapNum(0)
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)

def normalize_smiles_multi(smiles: str) -> Optional[Tuple[str, ...]]:
    """
    Split by '.', normalize each component, then sort -> tuple.
    Treat as (multi)set by canonical sorted tuple.
    """
    smiles = clean_smiles(smiles)
    if not smiles:
        return None
    parts = [p for p in smiles.split(".") if p]
    if not parts:
        return None
    normed = []
    for p in parts:
        n = normalize_smiles_single(p)
        if n is None:
            return None
        normed.append(n)
    normed.sort()
    return tuple(normed)

def parse_rxn_field(rxn: str) -> Optional[Tuple[str, str]]:
    """
    rxn string: reactants>reagents>production
    return (reactants, production)
    """
    if not isinstance(rxn, str):
        return None
    rxn = rxn.strip()
    if not rxn or ">" not in rxn:
        return None
    parts = rxn.split(">")
    if len(parts) < 3:
        return None
    reactants = parts[0].strip()
    product = parts[2].strip()
    if not reactants or not product:
        return None
    return reactants, product

def load_gt_map(csv_path: Path) -> Dict[Tuple[str, ...], Set[Tuple[str, ...]]]:
    """
    Build mapping:
      product_key(tuple of canonical mol smiles) -> set of reactants_key(tuple sorted canonical)
    """
    if not _require_rdkit():
        raise RuntimeError("RDKit is required for condition A, but cannot import rdkit. Please install RDKit.")

    gt_map: Dict[Tuple[str, ...], Set[Tuple[str, ...]]] = {}
    total = 0
    kept = 0
    bad = 0

    with csv_path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise RuntimeError(f"CSV has no header: {csv_path}")

        rxn_col = "reactants>reagents>production"
        if rxn_col not in reader.fieldnames:
            rxn_col = reader.fieldnames[-1]

        for row in tqdm(reader, desc=f"Loading GT CSV ({csv_path.name})", unit="row"):
            total += 1
            rxn = row.get(rxn_col, "")
            pr = parse_rxn_field(rxn)
            if pr is None:
                bad += 1
                continue
            reactants_raw, product_raw = pr

            prod_key = normalize_smiles_multi(product_raw)
            reac_key = normalize_smiles_multi(reactants_raw)
            if prod_key is None or reac_key is None:
                bad += 1
                continue

            gt_map.setdefault(prod_key, set()).add(reac_key)
            kept += 1

    print(f"[GT] total_rows={total}, kept_pairs={kept}, bad_rows={bad}, unique_products={len(gt_map)}")
    return gt_map

def condition_b_hit(target_raw: str, mt_pred_raw: str) -> bool:
    """
    Condition B:
      target product == product predicted by MT using predicted precursors.
    More robust than raw string compare: canonicalize (RDKit) when possible.

    Keep compatibility with your old logic:
      - if target is single molecule and MT predicts multi ('.'), accept if target is among components
      - else accept if the whole (multi)set matches
    """
    target_raw = clean_smiles(target_raw)
    mt_pred_raw = clean_smiles(mt_pred_raw)

    if _require_rdkit():
        tgt_key = normalize_smiles_multi(target_raw)
        pred_key = normalize_smiles_multi(mt_pred_raw)
        if tgt_key is None or pred_key is None:
            return False
        if tgt_key == pred_key:
            return True
        if len(tgt_key) == 1 and tgt_key[0] in pred_key:
            return True
        return False

    parts = mt_pred_raw.split(".") if mt_pred_raw else []
    return True if (target_raw in parts) else False

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True, help="input txt, each line: SRC>>TGT")
    ap.add_argument("--group-size", type=int, required=True, help="n lines per group (rank1..rankn)")
    ap.add_argument("--max-k", type=int, default=None, help="compute K up to this value (default: group-size)")

    ap.add_argument(
        "--mt-dir",
        default=str(DEFAULT_MT_DIR),
        help=f"Molecular Transformer directory (default: {DEFAULT_MT_DIR})",
    )
    ap.add_argument("--dataset", default="STEREO_separated_augm")
    ap.add_argument("--model", default=None, help="model filename under experiments/models/")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-length", type=int, default=200)

    ap.add_argument("--translate-chunk", type=int, default=5000,
                    help="lines per translate chunk to show progress (0 = run once without chunking)")

    ap.add_argument("--gt-csv", default=None,
                    help=(
                        "CSV path with columns including reactants>reagents>production; "
                        "used for condition A (GT match). Required unless --no-gt-csv is used."
                    ))
    ap.add_argument("--no-gt-csv", action="store_true",
                    help="Disable loading GT CSV and skip condition A; only use condition B (MT roundtrip).")

    args = ap.parse_args()

    input_path = Path(args.input).resolve()
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    group_size = args.group_size
    if group_size <= 0:
        raise ValueError("--group-size must be > 0")
    max_k = args.max_k if args.max_k is not None else group_size
    max_k = min(max_k, group_size)

    mt_dir = Path(args.mt_dir).resolve()
    model_name = args.model or f"{args.dataset}_model_average_20.pt"

    out_dir = safe_outdir_for_input(input_path)
    print(f"[I/O] Output dir: {out_dir}")

    # Files
    src_raw = out_dir / "src_raw.txt"
    tgt_raw = out_dir / "tgt_raw.txt"
    src_tok = out_dir / "src_tokenized.txt"
    pred_tok = out_dir / "pred_tokenized.txt"
    pred_clean = out_dir / "pred_clean.txt"
    detail = out_dir / "roundtrip_detail.tsv"
    group_acc = out_dir / "group_acc.tsv"
    exact_match_group_acc = out_dir / "exact_match_group_acc.tsv"

    # Load GT map for condition A by default. Use --no-gt-csv to skip it explicitly.
    gt_map = None
    if args.no_gt_csv:
        print("[GT] --no-gt-csv provided, condition A (GT match) will be skipped.")
    else:
        if not args.gt_csv:
            raise ValueError("Provide --gt-csv /path/to/file.csv, or use --no-gt-csv.")
        csv_path = Path(args.gt_csv).resolve()
        if not csv_path.exists():
            raise FileNotFoundError(
                f"GT CSV not found: {csv_path}. "
                "Provide --gt-csv /path/to/file.csv, or use --no-gt-csv to skip condition A."
            )
        gt_map = load_gt_map(csv_path)

    # Step 1: Parse
    total_lines = count_lines(input_path)
    used = 0
    bad = 0
    with input_path.open("r", encoding="utf-8", errors="ignore") as fin, \
         src_raw.open("w", encoding="utf-8") as fsrc, \
         tgt_raw.open("w", encoding="utf-8") as ftgt:
        for line in tqdm(fin, total=total_lines, desc="Parsing SRC>>TGT", unit="line"):
            sp = split_rxn_line(line)
            if sp is None:
                bad += 1
                continue
            src, tgt = sp
            fsrc.write(src + "\n")
            ftgt.write(tgt + "\n")
            used += 1
    print(f"[Parse] total={total_lines}, used={used}, bad/empty={bad}")
    if used == 0:
        raise RuntimeError("No valid lines like SRC>>TGT found.")
    if bad:
        raise RuntimeError(
            f"Found {bad} invalid/empty input lines. Every line must be reactants>>product "
            "so rank groups remain aligned."
        )
    if used % group_size != 0:
        raise RuntimeError(
            f"Input has {used} valid lines, which is not divisible by --group-size "
            f"({group_size}). Fix the input rather than truncating an incomplete group."
        )

    # Step 2: Tokenize
    with src_raw.open("r", encoding="utf-8", errors="ignore") as fin, \
         src_tok.open("w", encoding="utf-8") as fout:
        for line in tqdm(fin, total=used, desc="Tokenizing SRC", unit="rxn"):
            fout.write(tokenize_smiles(line.strip()) + "\n")

    # Step 3: Translate (with chunked progress)
    if args.translate_chunk == 0 or args.translate_chunk >= used:
        print("[MT] Translating in one shot (no chunk progress inside translate)...")
        run_translate_chunk(mt_dir, model_name, src_tok, pred_tok,
                            args.batch_size, args.max_length, args.gpu)
    else:
        chunk = args.translate_chunk
        n_chunks = math.ceil(used / chunk)
        print(f"[MT] Translating with chunking: chunk={chunk}, chunks={n_chunks}")

        if pred_tok.exists():
            pred_tok.unlink()

        with pred_tok.open("a", encoding="utf-8") as fout_pred:
            for ci in tqdm(range(n_chunks), desc="MT translate chunks", unit="chunk"):
                start = ci * chunk
                end = min((ci + 1) * chunk, used)

                chunk_src = out_dir / f"src_tokenized.chunk_{ci:04d}.txt"
                chunk_out = out_dir / f"pred_tokenized.chunk_{ci:04d}.txt"

                with src_tok.open("r", encoding="utf-8", errors="ignore") as fin:
                    for _ in range(start):
                        fin.readline()
                    lines = [fin.readline() for _ in range(end - start)]
                chunk_src.write_text("".join(lines), encoding="utf-8")

                # Translate this chunk.
                run_translate_chunk(mt_dir, model_name, chunk_src, chunk_out,
                                    args.batch_size, args.max_length, args.gpu)

                # Append to the complete prediction file.
                fout_pred.write(chunk_out.read_text(encoding="utf-8", errors="ignore"))

                # Remove transient chunk files after successful append.
                if chunk_src.exists():
                    chunk_src.unlink()
                if chunk_out.exists():
                    chunk_out.unlink()

    pred_count = count_lines(pred_tok)
    if pred_count != used:
        raise RuntimeError(
            f"Molecular Transformer produced {pred_count} lines for {used} inputs. "
            "Scoring was stopped to avoid silently misaligned results."
        )

    # Step 4: Clean + Score per-line (streaming, add condition A)
    correct_flags: List[int] = []      # Existing metric: condition A OR condition B
    exact_match_flags: List[int] = []  # Exactly-match metric: condition A only
    n_scored = 0

    with src_raw.open("r", encoding="utf-8", errors="ignore") as fsrc, \
         tgt_raw.open("r", encoding="utf-8", errors="ignore") as ftgt, \
         pred_tok.open("r", encoding="utf-8", errors="ignore") as fpred, \
         pred_clean.open("w", encoding="utf-8") as fp, \
         detail.open("w", encoding="utf-8") as fd:

        fd.write("idx\tgroup\trank_in_group\tsrc\tgt_reactants_hit\tcondA_gt_match\tcondB_mt_roundtrip\ttgt\tpred_clean\tcorrect\n")

        for i, (src_line, tgt_line, pred_line) in enumerate(
            tqdm(zip(fsrc, ftgt, fpred), total=used, desc="Cleaning+Scoring", unit="rxn")
        ):
            src = clean_smiles(src_line.strip())
            tgt = clean_smiles(tgt_line.strip())
            pred_c = clean_smiles(pred_line.strip())
            fp.write(pred_c + "\n")

            # ----- Condition A: predicted precursors == ground truth reactants (from CSV by product) -----
            condA = 0
            gt_hit = 0
            if gt_map is not None:
                prod_key = normalize_smiles_multi(tgt)
                pred_reac_key = normalize_smiles_multi(src)
                if prod_key is not None and pred_reac_key is not None:
                    gt_sets = gt_map.get(prod_key)
                    if gt_sets:
                        gt_hit = 1
                        if pred_reac_key in gt_sets:
                            condA = 1

            # ----- Condition B: MT(pred precursors) -> target product -----
            condB = 1 if condition_b_hit(tgt, pred_c) else 0

            ok = 1 if (condA or condB) else 0
            correct_flags.append(ok)
            exact_match_flags.append(condA)

            g = i // group_size
            r = (i % group_size) + 1
            fd.write(f"{i}\t{g}\t{r}\t{src}\t{gt_hit}\t{condA}\t{condB}\t{tgt}\t{pred_c}\t{ok}\n")
            n_scored += 1

    if n_scored != used:
        print(f"[WARN] scored_lines={n_scored} != used_lines={used} (pred/tgt/src line count mismatch?)")

    # Step 5: Group metrics
    # Existing metric: correct = condition A OR condition B
    n_groups, rows = compute_group_acc(correct_flags, group_size=group_size, max_k=max_k)
    avg_rankk = write_group_acc(group_acc, rows, desc="Computing group acc")

    # Exactly-match metric: condition A only
    exact_rows = None
    exact_avg_rankk = None
    if gt_map is not None:
        exact_n_groups, exact_rows = compute_group_acc(exact_match_flags, group_size=group_size, max_k=max_k)
        if exact_n_groups != n_groups:
            print(f"[WARN] exact_n_groups={exact_n_groups} != n_groups={n_groups}")
        exact_avg_rankk = write_group_acc(
            exact_match_group_acc,
            exact_rows,
            desc="Computing exact-match group acc",
        )
    else:
        print("[ExactMatch] GT CSV was skipped, so condition-A-only exact-match metrics were not written.")

    print("\n[Done]")
    print(f"  used_lines                = {n_scored}")
    print(f"  group_size                = {group_size}")
    print(f"  n_groups                  = {n_groups}")
    print(f"  outputs dir               = {out_dir}")
    print(f"  group_acc.tsv             = {group_acc}")
    if gt_map is not None:
        print(f"  exact_match_group_acc.tsv = {exact_match_group_acc}")
    print(f"  detail.tsv                = {detail}")

    k, topk, rankk = rows[-1]
    print(f"\n[Combined: condA OR condB] K={k}  topK_acc={topk:.6f}  rankK_acc={rankk:.6f}  avg_rankK_acc={avg_rankk:.6f}")
    if exact_rows is not None and exact_avg_rankk is not None:
        ek, etopk, erankk = exact_rows[-1]
        print(f"[Exact match: condA only] K={ek}  topK_acc={etopk:.6f}  rankK_acc={erankk:.6f}  avg_rankK_acc={exact_avg_rankk:.6f}")

if __name__ == "__main__":
    main()
