---
name: retroformer-roundtrip-cal
description: Build and evaluate roundtripcal inputs from Retroformer or retrosynthesis prediction CSVs. Use when asked to convert top-k predicted reactants plus products into reaction TXT files for roundtripcal, run /root/autodl-fs/roundtripcal/cal.py with group-size 10, copy and run group_reactant_similarity_posthoc.py, and run disconnection_similarity_nomap.py for top-10/top-5 roundtrip or ablation experiments.
---

# Retroformer Roundtrip Cal

## Purpose

Use this workflow for roundtrip posthoc evaluation of Retroformer-like prediction CSVs. The default root is `/root/autodl-fs/roundtripcal`; the default candidate group size is 10; the default posthoc similarity top-n is 5.

## Inputs

- Use a prediction CSV with one product per row and ranked reactant candidates, usually `product` plus `top_1` ... `top_10` from Retroformer merged output.
- The bundled builder also supports `source_smiles` plus `prediction_1` ... `prediction_10` and other common candidate prefixes.
- Put generated TXT files under `/root/autodl-fs/roundtripcal` unless the user asks otherwise.
- Name TXT files with the model name first, then concrete experiment details, ending with `_top10.txt`, for example `Retroformer_ablation_random10_origvocab_model378100_rebuilt_test_top10.txt`.

## Build The TXT

Create one line per candidate in rank order:

```text
reactants>>product
```

Every CSV row becomes exactly 10 TXT lines, so `line_count == row_count * 10`. Missing or blank candidate cells must be written as `9999>>product` to preserve group alignment.

For Retroformer merged/rebuilt outputs, always use the prediction CSV's own `product` column for the right-hand side. Do not use `row_index` to fetch products from a `raw_test.csv`; in ablation/rebuilt outputs that index may not be the same row order as the merged predictions, which silently corrupts `reactants>>product` pairs and makes condition B roundtrip accuracy near zero. If products contain the leading vocabulary artifact `<UNK>`, remove uppercase `<UNK>` before writing the TXT.

Use the bundled builder:

```bash
python /root/.codex/skills/retroformer-roundtrip-cal/scripts/build_roundtrip_topk_input.py \
  --input-csv /path/to/predictions.csv \
  --output-txt /root/autodl-fs/roundtripcal/MODEL_EXPERIMENT_top10.txt \
  --topk 10
```

For Retroformer CSVs, auto-detection should find `product` and `top_1` ... `top_10`. If it does not, pass `--product-col` and `--candidate-prefix`, for example `--product-col product --candidate-prefix top_`.

Before running GPU work, verify the builder output printed `lines = rows * 10`. If not, stop and fix the input conversion.

## Run Roundtrip Accuracy

Run `cal.py` with group size 10 and otherwise default settings:

```bash
python /root/autodl-fs/roundtripcal/cal.py \
  -i /root/autodl-fs/roundtripcal/MODEL_EXPERIMENT_top10.txt \
  --group-size 10
```

If condition A / exact-match numbers are needed and the default `cal.py` GT CSV is not the same aligned test set, create an aligned GT CSV from the same prediction CSV rows using `ground_truth>>clean_product`, then pass it with `--gt-csv /path/to/aligned.gt.csv`. A wrong default GT mostly breaks condition A, while a wrong TXT product breaks condition B as well.

Record the output directory printed as `[I/O] Output dir: ...`. If the log is unavailable, infer it from the TXT stem: `/root/autodl-fs/roundtripcal/<stem>`; if that already existed, `cal.py` creates `/root/autodl-fs/roundtripcal/<stem>_1`, then `_2`, etc.

## Run Posthoc Similarity

After `cal.py` finishes, copy the posthoc script into the just-created output directory and execute it with top-n 5 and group-size 10:

```bash
cp /root/autodl-fs/roundtripcal/megan_newfilter_top10/group_reactant_similarity_posthoc.py <out_dir>/
python <out_dir>/group_reactant_similarity_posthoc.py \
  --run-dir <out_dir> \
  --top-n 5 \
  --group-size 10
```

Then run disconnection similarity. At minimum provide `-i` and `-o`; keep grouped top-5 behavior unless the user explicitly asks for script defaults only:

```bash
python /root/autodl-fs/roundtripcal/disconnection_similarity_nomap.py \
  -i /root/autodl-fs/roundtripcal/MODEL_EXPERIMENT_top10.txt \
  -o <out_dir> \
  --group-size 10 \
  --top-k 5
```

## Multiple Jobs And GPUs

When running several evaluations at once, assign different physical GPUs with `CUDA_VISIBLE_DEVICES` and pass GPU id `0` inside the masked process:

```bash
CUDA_VISIBLE_DEVICES=1 python /root/autodl-fs/roundtripcal/cal.py \
  -i <txt> \
  --group-size 10 \
  --gpu 0

CUDA_VISIBLE_DEVICES=1 python /root/autodl-fs/roundtripcal/disconnection_similarity_nomap.py \
  -i <txt> \
  -o <out_dir> \
  --gpu-id 0 \
  --group-size 10 \
  --top-k 5
```

Use `CUDA_VISIBLE_DEVICES`, not `CUDA_VISIABLE`. If not masking GPUs, pass the physical id directly to `--gpu` and `--gpu-id`.

## Reporting

Report the TXT path, `cal.py` output directory, and the key generated files:

- `group_acc.tsv`
- `exact_match_group_acc.tsv` if present
- `group_reactant_similarity_top5.tsv` and `.summary.tsv`
- disconnection outputs written by `disconnection_similarity_nomap.py`

Summarize final metrics only after all three stages finish, and mention any stage that could not be run.
