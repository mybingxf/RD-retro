#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import re
import subprocess
from pathlib import Path
from typing import Optional, Tuple

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


# ===== MT 官方 tokenizer（与 MolecularTransformer README 一致）=====
# https://github.com/pschwllr/MolecularTransformer
MT_PATTERN = "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
SMI_REGEX = re.compile(MT_PATTERN)

# 常见不可见/格式字符（会导致“看起来是空串”但实际上有字符）
INVISIBLE_TRANSLATION = str.maketrans({
    "\ufeff": None,  # BOM
    "\u200b": None,  # zero width space
    "\u200c": None,  # ZWNJ
    "\u200d": None,  # ZWJ
    "\u2060": None,  # word joiner
})


def sanitize_smiles_text(s: str) -> str:
    """
    清洗 SMILES：
    - 去掉 NUL/control/不可见字符
    - 去掉所有空白
    """
    if s is None:
        return ""
    if not isinstance(s, str):
        s = str(s)

    # 去 NUL
    s = s.replace("\x00", "")

    # 去 BOM/零宽
    s = s.translate(INVISIBLE_TRANSLATION)

    # 去 ASCII control（<32）与 DEL(127)
    # 注意：这里不会影响正常 SMILES 字符
    s = "".join(ch for ch in s if (ord(ch) >= 32 and ord(ch) != 127))

    # 去空白（space/tab 等）
    s = "".join(s.split())

    return s


def tokenize_smiles_mt(smi: str) -> str:
    """MT 的 smi_tokenizer：要求可逆，无法完整 tokenization 就报错。"""
    smi = sanitize_smiles_text(smi)
    if not smi:
        raise ValueError("Empty SMILES after sanitize (should have been filled).")

    tokens = [tok for tok in SMI_REGEX.findall(smi)]
    if smi != "".join(tokens):
        raise ValueError(
            "MT tokenizer failed (string cannot be fully tokenized).\n"
            f"SMILES: {smi}\n"
            f"SMILES_REPR: {repr(smi)}\n"
            f"SMILES_LEN: {len(smi)}\n"
            f"TOKENS_JOINED: {''.join(tokens)}\n"
            f"TOKENS: {tokens[:50]}{' ...' if len(tokens) > 50 else ''}"
        )
    return " ".join(tokens)


def split_rxn_line(line: str) -> Optional[Tuple[str, str]]:
    """
    input line: SRC>>TGT
    - 允许 SRC 为空（>>PRODUCT），后面会补 fill-empty-src
    - TGT 为空则判为 bad
    """
    line = line.strip()
    if not line or ">>" not in line:
        return None
    src, tgt = line.split(">>", 1)
    src, tgt = src.strip(), tgt.strip()
    if not tgt:
        return None
    return src, tgt  # src 可以为空


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


def run_translate(mt_dir: Path, model_name: str, src_tokenized: Path, pred_tokenized: Path,
                  batch_size: int, max_length: int, gpu: int) -> None:
    translate_py = mt_dir / "translate.py"
    if not translate_py.exists():
        raise FileNotFoundError(f"translate.py not found in {mt_dir}")

    model_path = mt_dir / "experiments" / "models" / model_name
    if not model_path.exists():
        raise FileNotFoundError(f"model not found: {model_path}")

    cmd = [
        "python", "translate.py",
        "-model", str(model_path),
        "-src", str(src_tokenized),
        "-output", str(pred_tokenized),
        "-batch_size", str(batch_size),
        "-replace_unk",
        "-max_length", str(max_length),
        "-fast",
        "-gpu", str(gpu),
    ]
    print("\n[MT] Running translate:")
    print(" ", " ".join(cmd))
    subprocess.run(cmd, cwd=str(mt_dir), check=True)


def clean_smiles(s: str) -> str:
    return sanitize_smiles_text(s)


def compute_group_acc(correct_flags, group_size: int, max_k: int):
    n = len(correct_flags)
    g = n // group_size
    if g == 0:
        raise RuntimeError("Not enough lines to form a single group.")
    correct_flags = correct_flags[: g * group_size]

    out = []
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
        out.append((k, topk_hit / g, rankk_hit / g))
    return g, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True, help="input txt, each line: SRC>>TGT")
    ap.add_argument("--group-size", type=int, required=True, help="n lines per group (rank1..rankn).")
    ap.add_argument("--max-k", type=int, default=None, help="compute K up to this value (default: group-size)")

    ap.add_argument("--mt-dir", default="/root/autodl-fs/MolecularTransformer")
    ap.add_argument("--dataset", default="STEREO_mixed_augm")
    ap.add_argument("--model", default=None,
                    help="model filename under experiments/models/. Default: <dataset>_model_average_20.pt")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-length", type=int, default=200)
    ap.add_argument("--no-progress", action="store_true")

    # 仍然保留：空 SRC 补 C
    ap.add_argument("--fill-empty-src", default="C")

    args = ap.parse_args()

    input_path = Path(args.input).resolve()
    mt_dir = Path(args.mt_dir).resolve()
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    group_size = args.group_size
    if group_size <= 0:
        raise ValueError("--group-size must be > 0")
    max_k = args.max_k if args.max_k is not None else group_size
    max_k = min(max_k, group_size)
    if max_k <= 0:
        raise ValueError("--max-k must be > 0")

    model_name = args.model or f"{args.dataset}_model_average_20.pt"

    out_dir = safe_outdir_for_input(input_path)
    print(f"[I/O] Output dir: {out_dir}")

    src_raw = out_dir / "src_raw.txt"
    tgt_raw = out_dir / "tgt_raw.txt"
    src_tok = out_dir / "src_tokenized.txt"
    pred_tok = out_dir / "pred_tokenized.txt"
    pred_clean = out_dir / "pred_clean.txt"
    detail = out_dir / "roundtrip_detail.tsv"
    group_acc = out_dir / "group_acc.tsv"

    # 1) parse + sanitize + fill empty
    total = 0
    used = 0
    bad = 0
    empty_src_filled = 0
    empty_src_filled_after_sanitize = 0
    changed_by_sanitize = 0

    with input_path.open("r", encoding="utf-8", errors="ignore") as fin, \
         src_raw.open("w", encoding="utf-8") as fsrc, \
         tgt_raw.open("w", encoding="utf-8") as ftgt:
        for line in fin:
            total += 1
            sp = split_rxn_line(line)
            if sp is None:
                bad += 1
                continue
            src, tgt = sp

            # 原始是否空
            if src == "":
                empty_src_filled += 1

            src_s = sanitize_smiles_text(src)
            tgt_s = sanitize_smiles_text(tgt)

            # 统计 sanitize 是否改变
            if src_s != sanitize_smiles_text(src) or tgt_s != sanitize_smiles_text(tgt):
                changed_by_sanitize += 1

            if not tgt_s:
                bad += 1
                continue

            if not src_s:
                empty_src_filled_after_sanitize += 1
                src_s = args.fill_empty_src

            fsrc.write(src_s + "\n")
            ftgt.write(tgt_s + "\n")
            used += 1

    print(f"[Parse] total={total}, used={used}, bad/empty={bad}, "
          f"empty_src_filled={empty_src_filled}, empty_src_filled_after_sanitize={empty_src_filled_after_sanitize}")

    if used == 0:
        raise RuntimeError("No valid lines like SRC>>TGT found.")

    # 2) tokenize src
    with src_raw.open("r", encoding="utf-8", errors="ignore") as fin, \
         src_tok.open("w", encoding="utf-8") as fout:
        for line in fin:
            s = line.rstrip("\n")
            s = sanitize_smiles_text(s)
            if not s:
                s = args.fill_empty_src
            fout.write(tokenize_smiles_mt(s) + "\n")

    # 3) translate
    run_translate(mt_dir, model_name, src_tok, pred_tok, args.batch_size, args.max_length, args.gpu)

    # 4) score per-line
    tgts = tgt_raw.read_text(encoding="utf-8", errors="ignore").splitlines()
    preds = pred_tok.read_text(encoding="utf-8", errors="ignore").splitlines()
    n = min(len(tgts), len(preds))
    if len(tgts) != len(preds):
        print(f"[WARN] tgt lines={len(tgts)} != pred lines={len(preds)}; align by min={n}")

    it = range(n)
    if tqdm is not None and (not args.no_progress):
        it = tqdm(it, desc="Scoring per-line", unit="rxn")

    correct_flags = [0] * n
    with pred_clean.open("w", encoding="utf-8") as fp, \
         detail.open("w", encoding="utf-8") as fd:
        fd.write("idx\tgroup\trank_in_group\ttgt\tpred_clean\tcorrect\n")
        for i in it:
            tgt = clean_smiles(tgts[i])
            pred_c = clean_smiles(preds[i])
            fp.write(pred_c + "\n")
            parts = pred_c.split(".") if pred_c else []
            ok = 1 if (tgt in parts) else 0
            correct_flags[i] = ok

            g = i // group_size
            r = (i % group_size) + 1
            fd.write(f"{i}\t{g}\t{r}\t{tgt}\t{pred_c}\t{ok}\n")

    # 5) group-level topK / rankK
    n_groups, rows = compute_group_acc(correct_flags, group_size=group_size, max_k=max_k)

    with group_acc.open("w", encoding="utf-8") as fg:
        fg.write("K\ttopk_acc\trankk_acc\n")
        for k, topk, rankk in rows:
            fg.write(f"{k}\t{topk:.8f}\t{rankk:.8f}\n")

    print("\n[Group Result]")
    print(f"  used_lines = {n} (truncated to {n_groups * group_size} for full groups)")
    print(f"  group_size = {group_size}")
    print(f"  n_groups   = {n_groups}")
    print(f"  saved: {group_acc}")

    k, topk, rankk = rows[-1]
    print(f"\nK={k}  topK_acc={topk:.6f}  rankK_acc={rankk:.6f}")


if __name__ == "__main__":
    main()
