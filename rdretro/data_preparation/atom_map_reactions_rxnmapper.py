#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Atom-map reaction SMILES (reactants>>products) using RXNMapper.

Why this script:
- Your localretro/train_isomeric.txt is UNMAPPED. True R-SMILES root alignment needs atom-maps.
- RXNMapper provides attention-guided atom mapping from a transformer model.  citeturn2view0

Install (recommended in a separate conda env to avoid breaking your OpenNMT env):
    pip install "rxnmapper[rdkit]"   citeturn2view0turn1view0

Input format (one per line):
    reactants>>products
(reactants can be multi-fragment with '.', products can also be '.' but usually single)

Outputs:
- out_path: mapped reactions in the same line order (one per input line)
- optional conf_path: confidence per line (float or blank)

Notes:
- We set CUDA_VISIBLE_DEVICES *before* importing rxnmapper if you pass --cuda.
- For invalid reactions, BatchedMapper returns ">>" (per README). citeturn2view0
"""

import os
import sys
import argparse
from typing import List, Tuple, Optional

from tqdm import tqdm


def strip_anno(s: str) -> str:
    # Remove trailing annotations after a space, e.g. " ... 1-2;2-13"
    return s.strip().split(" ")[0]


def normalize_line(line: str) -> Optional[str]:
    line = line.strip()
    if not line:
        return None
    if ">>" not in line:
        return None
    left, right = line.split(">>", 1)
    left = strip_anno(left)
    right = strip_anno(right)
    if not left or not right:
        return None
    return f"{left}>>{right}"


def batched(iterable, batch_size: int):
    buf = []
    for x in iterable:
        buf.append(x)
        if len(buf) >= batch_size:
            yield buf
            buf = []
    if buf:
        yield buf


def count_lines(path: str) -> int:
    # Optional helper; not used by default because it scans the whole file.
    n = 0
    with open(path, "r", encoding="utf-8") as f:
        for _ in f:
            n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_path", type=str, required=True, help="Input TXT (reactants>>products per line)")
    ap.add_argument("--out_path", type=str, required=True, help="Output TXT for mapped reactions")
    ap.add_argument("--batch_size", type=int, default=64, help="RXNMapper batch size")
    ap.add_argument("--cuda", type=str, default=None,
                    help="GPU id(s) for CUDA_VISIBLE_DEVICES, e.g. '0' or '0,1'. "
                         "Omit for default; set to '' or '-1' to force CPU.")
    ap.add_argument("--resume", action="store_true",
                    help="If out_path exists, skip already-written lines (simple line-count resume).")
    ap.add_argument("--on_fail", choices=["keep", "drop"], default="keep",
                    help="When mapping fails for a line: keep original normalized line or drop it.")
    ap.add_argument("--conf_path", type=str, default=None,
                    help="Optional: write confidence per line (one float per mapped line).")
    ap.add_argument("--min_conf", type=float, default=None,
                    help="Optional: if confidence < min_conf, treat as fail (applies only when conf_path is used).")
    ap.add_argument("--progress_total", action="store_true",
                    help="If set, pre-count input lines to show a tqdm total (slower startup).")
    args = ap.parse_args()

    # Set CUDA visibility BEFORE importing rxnmapper/torch/transformers
    if args.cuda is not None:
        if args.cuda.strip() in ("", "-1", "cpu", "CPU"):
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda.strip()

    try:
        from rxnmapper import BatchedMapper
    except Exception as e:
        print("[ERROR] Failed to import rxnmapper.", file=sys.stderr)
        print("Install with: pip install \"rxnmapper[rdkit]\"", file=sys.stderr)
        raise

    mapper = BatchedMapper(batch_size=int(args.batch_size))  # handles batching + errors  citeturn2view0

    # Resume support (line-count based)
    already = 0
    if args.resume and os.path.exists(args.out_path):
        with open(args.out_path, "r", encoding="utf-8") as f:
            already = sum(1 for _ in f)
        print(f"[RESUME] Detected {already} lines already in {args.out_path}. Will skip same # input lines.")

    # Prepare output dir
    os.makedirs(os.path.dirname(args.out_path) or ".", exist_ok=True)
    if args.conf_path:
        os.makedirs(os.path.dirname(args.conf_path) or ".", exist_ok=True)

    total = count_lines(args.in_path) if args.progress_total else None

    in_kept = 0
    out_written = 0
    dropped = 0
    failed = 0

    # Open outputs (append if resume)
    out_mode = "a" if (args.resume and os.path.exists(args.out_path)) else "w"
    conf_mode = "a" if (args.resume and args.conf_path and os.path.exists(args.conf_path)) else "w"

    with open(args.in_path, "r", encoding="utf-8") as fin, \
         open(args.out_path, out_mode, encoding="utf-8") as fout:

        fconf = open(args.conf_path, conf_mode, encoding="utf-8") if args.conf_path else None

        # Skip already-processed input lines (regardless of validity) for simplicity.
        for _ in range(already):
            _ = fin.readline()

        pbar = tqdm(total=(total - already) if total is not None else None, desc="atom-mapping", unit="rxn")

        def write_one(mapped: str, conf: Optional[float], original_norm: str):
            nonlocal out_written, failed, dropped
            ok = True

            if not mapped or mapped.strip() in ("", ">>"):
                ok = False

            # Optional confidence filtering
            if ok and (args.min_conf is not None) and (conf is not None) and (conf < args.min_conf):
                ok = False

            if not ok:
                failed += 1
                if args.on_fail == "drop":
                    dropped += 1
                    return
                mapped = original_norm  # keep original normalized line
                conf = conf if conf is not None else None

            fout.write(mapped.strip() + "\n")
            if fconf is not None:
                if conf is None:
                    fconf.write("\n")
                else:
                    fconf.write(f"{conf:.6f}\n")
            out_written += 1

        buf_norm: List[str] = []

        for raw in fin:
            pbar.update(1)
            norm = normalize_line(raw)
            if norm is None:
                # ignore empty/invalid lines entirely (do not write)
                continue
            buf_norm.append(norm)
            in_kept += 1

            if len(buf_norm) >= args.batch_size:
                if args.conf_path:
                    infos = list(mapper.map_reactions_with_info(buf_norm))
                    for orig, info in zip(buf_norm, infos):
                        # info may be {} on failure
                        mapped = info.get("mapped_rxn", "") if isinstance(info, dict) else ""
                        conf = info.get("confidence", None) if isinstance(info, dict) else None
                        write_one(mapped, conf, orig)
                else:
                    mapped_list = list(mapper.map_reactions(buf_norm))
                    for orig, mapped in zip(buf_norm, mapped_list):
                        write_one(mapped, None, orig)
                buf_norm = []

        # flush last batch
        if buf_norm:
            if args.conf_path:
                infos = list(mapper.map_reactions_with_info(buf_norm))
                for orig, info in zip(buf_norm, infos):
                    mapped = info.get("mapped_rxn", "") if isinstance(info, dict) else ""
                    conf = info.get("confidence", None) if isinstance(info, dict) else None
                    write_one(mapped, conf, orig)
            else:
                mapped_list = list(mapper.map_reactions(buf_norm))
                for orig, mapped in zip(buf_norm, mapped_list):
                    write_one(mapped, None, orig)

        pbar.close()
        if fconf is not None:
            fconf.close()

    print("[DONE]")
    print("  in_path     :", os.path.abspath(args.in_path))
    print("  out_path    :", os.path.abspath(args.out_path))
    if args.conf_path:
        print("  conf_path   :", os.path.abspath(args.conf_path))
    print("  kept_valid  :", in_kept)
    print("  written     :", out_written)
    print("  failed_map  :", failed)
    print("  dropped     :", dropped)
    print("  batch_size  :", args.batch_size)
    if args.cuda is not None:
        print("  CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES", ""))


if __name__ == "__main__":
    main()
