# Round-trip evaluation for retrosynthesis predictions

Version 1.0.0 (9 September 2026)

This archive contains source code, instructions for obtaining the required
pretrained forward-reaction model, and a small real-data demo for reproducing
the round-trip evaluation used in the accompanying study. It computes:

1. grouped Top-k round-trip accuracy;
2. grouped reactant-set similarity; and
3. grouped disconnection similarity.

The preferred entry point is `source/run_roundtrip_pipeline.py`. It runs all
three stages and keeps each product's ranked candidates aligned as fixed-size
blocks (Top-10 by default).

## Archive contents

```text
README.md                              this file
environment.yml / requirements.txt    reproducible Python environment
source/                                evaluation source code
scripts/run_demo.sh                    one-command demo
scripts/check_install.py               dependency and checkpoint check
scripts/validate_demo.py               compares demo results with references
demo/                                  two-product Top-10 demo and references
third_party/MolecularTransformer/      forward-model source and model destination
LICENSES/                              third-party licence text
THIRD_PARTY_NOTICES.md                 provenance and local modifications
MANIFEST.sha256                        per-file integrity checks (in the zip)
```

## 1. System requirements

### Supported and tested platform

The submitted version was tested on:

- Ubuntu 22.04.3 LTS, Linux 6.5, x86-64;
- Python 3.10.19;
- NumPy 2.2.6, pandas 2.3.3, RDKit 2025.09.3, SciPy 1.15.3;
- PyTorch 2.8.0, torchtext 0.3.1, tqdm 4.67.1;
- RXNMapper 0.4.2, Transformers 4.57.3, tokenizers 0.22.1; and
- NVIDIA GeForce RTX 4090 (24 GB), driver 535.129.03, for GPU testing.

All direct dependencies and exact tested versions are in `requirements.txt`.
The code is expected to work on other recent x86-64 Linux distributions but
these have not been tested. macOS and Windows have not been tested.

### Hardware

A GPU is **not required** for the included two-product demo. A typical desktop
with a 64-bit x86 processor, 8 GB RAM, and approximately 3 GB free disk space
is sufficient after accounting for the Python environment and the separately
downloaded Molecular Transformer checkpoint (approximately 157 MB).

For the full USPTO-50K-size calculation (5,007 products x 10 candidates), an
NVIDIA CUDA-capable GPU with at least 8 GB VRAM is strongly recommended. CPU
execution is supported but will be substantially slower. More RAM or unusual
hardware is not required.

## 2. Installation guide

The installation requires Internet access to obtain Python packages and to
download the Molecular Transformer checkpoint. The checkpoint is not
redistributed in this archive and is not downloaded automatically by the
software. The RXNMapper PyPI package carries its own pretrained atom-mapping
model.

### Download the Molecular Transformer checkpoint

1. Open the upstream model distribution page:
   <https://ibm.ent.box.com/v/MolecularTransformerModels>
2. Download the file named exactly:
   `STEREO_separated_augm_model_average_20.pt`
3. Place it at this path relative to the extracted archive root:

   ```text
   third_party/MolecularTransformer/experiments/models/STEREO_separated_augm_model_average_20.pt
   ```

Do not rename the file. The default pipeline configuration looks for that
exact name. `scripts/check_install.py` verifies both its location and its
expected SHA-256 checksum:

```text
456d1027bf186945d394de17ab325046e3034e71c7ca93c4c735ad9a8b0dc8c3
```

### Conda installation (recommended)

From the extracted archive directory:

```bash
conda env create -f environment.yml
conda activate roundtripcal
python scripts/check_install.py
```

Expected last line:

```text
INSTALLATION CHECK: PASS
```

### venv/pip alternative

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python scripts/check_install.py
```

On a normal desktop with a broadband connection, environment creation usually
takes 10-30 minutes. Most of that time is downloading and installing PyTorch,
RDKit, and RXNMapper. Installation from a warm package cache is typically under
5 minutes.

The legacy Molecular Transformer depends on the `torchtext.data.Field` API,
which is why `torchtext==0.3.1` is intentionally pinned. Do not replace it with
a recent torchtext release without also porting the bundled OpenNMT code.

The installation check will fail with a download-and-placement message if the
external checkpoint has not yet been copied to the expected directory.

## 3. Demo

Run all stages on CPU from the archive root:

```bash
bash scripts/run_demo.sh roundtrip_demo_run cpu
```

The separately downloaded checkpoint must be present at the path documented
above before running the demo.

To use the first visible GPU instead:

```bash
bash scripts/run_demo.sh roundtrip_demo_run cuda 0
```

The output directory must be new or empty. The script runs the pipeline and
then invokes the validator. A successful run ends with:

```text
DEMO VALIDATION: PASS
```

The demo took 7.7 seconds on the tested system in forced CPU mode. Allow
approximately 30 seconds to 3 minutes on a normal desktop CPU, depending on
processor and PyTorch startup speed. GPU startup overhead makes this tiny demo
similar in duration; GPUs are useful for full datasets.

### Expected demo results

The reference values are in `demo/expected_output/expected_metrics.json`:

| Metric | Expected value |
|---|---:|
| Combined Top-1 accuracy | 0.50000000 |
| Combined Top-5 accuracy | 1.00000000 |
| Combined Top-10 accuracy | 1.00000000 |
| Exact-match Top-1 accuracy | 0.00000000 |
| Exact-match Top-10 accuracy | 1.00000000 |
| All-Top-5 reactant mean similarity | 0.4642679373 |
| All-Top-5 reactant minimum similarity | 0.3228385807 |
| Hit-Top-5 reactant mean similarity | 0.5146364276 |
| Disconnection similarity, per-product mean including all | 0.15000000 |

The main output is `roundtrip_demo_run/roundtrip_input_top10/`. It contains
`group_acc.tsv`, `exact_match_group_acc.tsv`, `roundtrip_detail.tsv`,
`group_reactant_similarity_top5.tsv` and its summary, `summary.tsv`,
`pairs.tsv`, `mapped_and_sig.jsonl`, and `global_metrics.txt`.

Accuracy and fingerprint values should match exactly at the shown precision.
Atom mapping can have small floating-point or implementation differences on a
different PyTorch/Transformers stack; the demo validator therefore checks that
the disconnection files exist but does not require their numeric values to be
bit-identical.

## 4. Instructions for use on your data

### Input CSV format

The default Top-10 CSV schema is:

```text
product,ground_truth,top_1,top_2,...,top_10
```

There must be one product per row and candidates must be ordered by rank.
`ground_truth` contains dot-separated true reactants. Blank candidate cells
are converted to `9999` so that every product still occupies exactly ten lines.

Common alternatives such as `source_smiles` plus `prediction_1` through
`prediction_10` are detected automatically. For other names, use
`--product-col`, `--candidate-prefix`, or a comma-separated `--candidate-cols`.

Run the full default Top-10/Top-5 evaluation:

```bash
python source/run_roundtrip_pipeline.py \
  --input-csv /path/to/predictions.csv \
  --output-dir /path/to/new_results_directory \
  --device cuda \
  --gpu-id 0
```

If the CSV contains `product` and `ground_truth`, an aligned ground-truth CSV
is built automatically. To supply one explicitly:

```bash
python source/run_roundtrip_pipeline.py \
  --input-csv /path/to/predictions.csv \
  --gt-csv /path/to/aligned_ground_truth.csv \
  --output-dir /path/to/new_results_directory \
  --device cuda \
  --gpu-id 0
```

If no ground truth is available, skip condition A explicitly:

```bash
python source/run_roundtrip_pipeline.py \
  --input-csv /path/to/predictions.csv \
  --no-gt-csv \
  --output-dir /path/to/new_results_directory \
  --device cpu
```

This still computes forward-model round-trip accuracy (condition B), reactant
similarity, and disconnection similarity, but does not write
`exact_match_group_acc.tsv`.

### Prebuilt TXT input

The pipeline also accepts a TXT file with one ranked candidate per line:

```text
reactants>>product
```

For Top-10 evaluation, every product must occupy exactly ten consecutive lines
and the total line count must be divisible by ten:

```bash
python source/run_roundtrip_pipeline.py \
  --input-txt /path/to/model_test_top10.txt \
  --gt-csv /path/to/aligned_ground_truth.csv \
  --output-dir /path/to/new_results_directory \
  --device cuda \
  --gpu-id 0
```

Do not reconstruct the product from an unrelated row index. Always use the
product aligned with that prediction row; otherwise candidate/product pairs
can be silently corrupted.

### Running individual stages

Build the grouped input and aligned GT:

```bash
python source/build_roundtrip_topk_input.py \
  --input-csv predictions.csv \
  --output-txt model_test_top10.txt \
  --topk 10

python source/build_aligned_gt_csv.py \
  --input-csv predictions.csv \
  --output-csv model_test_top10.aligned_gt.csv
```

Compute round-trip accuracy (use `--gpu -1` for CPU):

```bash
python source/cal.py \
  --input model_test_top10.txt \
  --group-size 10 \
  --gt-csv model_test_top10.aligned_gt.csv \
  --gpu 0
```

`cal.py` prints its newly created run directory. Use that directory for the
remaining stages:

```bash
python source/group_reactant_similarity_posthoc.py \
  --run-dir /path/to/cal_run_directory \
  --top-n 5 \
  --group-size 10

python source/disconnection_similarity_nomap.py \
  --input model_test_top10.txt \
  --outdir /path/to/cal_run_directory \
  --device cuda \
  --gpu-id 0 \
  --group-size 10 \
  --top-k 5
```

Use `python source/<script>.py --help` for all options. For multiple GPU jobs,
assign a different physical GPU to each process with `--gpu-id`; the unified
pipeline handles `CUDA_VISIBLE_DEVICES` masking internally.

## Output and metric definitions

- **Condition A:** the predicted reactant multiset exactly matches the aligned
  ground truth after RDKit canonicalisation and atom-map removal.
- **Condition B:** the Molecular Transformer forward prediction reproduces the
  target product after RDKit canonicalisation. For a single target molecule, a
  matching component in a multi-product forward prediction is accepted.
- **Combined round-trip hit:** condition A OR condition B.
- **Reactant similarity:** Morgan radius-2, 2,048-bit fingerprints are OR-ed
  across reactant components; pairwise Tanimoto mean and minimum are reported
  for All-Top-N and the first N combined hits in rank order. When fewer than N
  hits exist, the group receives `--hit-fill-sim` (default 0.0).
- **Disconnection similarity:** RXNMapper atom maps each of the first N
  candidates per fixed product block. The formed-bond signature is
  `bonds(product) - bonds(reactants)`, and pairwise Jaccard similarity is
  reported. Bond type is ignored unless `--use-bond-type` is supplied when the
  stage is run directly.

Rows marked `AVG_RANKK` in accuracy TSVs are summaries: `rankk_acc` is the mean
of rank-specific accuracies over all reported ranks; `topk_acc` is `NA`.

## Verification and reproducibility notes

Run lightweight unit tests without invoking either neural model:

```bash
python -m unittest discover -s tests -v
```

After extraction, verify archive files with:

```bash
sha256sum -c MANIFEST.sha256
```

`MANIFEST.sha256` covers files redistributed in this archive. The external
checkpoint is intentionally excluded; `scripts/check_install.py` verifies it
separately after download.

The model checksum is also checked by `scripts/check_install.py`. The pipeline
refuses invalid input lines, incomplete candidate blocks, mismatched forward
prediction counts, and non-empty output directories to prevent silent
misalignment or accidental overwrites.

## Third-party software and citation

See `THIRD_PARTY_NOTICES.md` and `LICENSES/`. The bundled forward model is the
Molecular Transformer of Schwaller et al., *ACS Central Science* 5, 1572-1583
(2019). Atom mapping uses RXNMapper from Schwaller et al., *Science Advances*
7, eabe4166 (2021). Please cite these works together with the accompanying
study when appropriate.
