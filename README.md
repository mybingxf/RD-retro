# RD-Retro

RD-Retro is a data-level bootstrapping framework for retrosynthesis models. It
constructs a candidate reaction pool, removes candidates that fail independent
forward validation, reduces redundant precursor sets with fingerprint-based
diversity pruning, and evaluates predictions with round-trip and diversity
metrics.

This repository snapshot is organized for paper reproducibility and later GitHub
release. It contains the processing code, the round-trip evaluation workflow, and
the local model weights used by the pipeline. Large generated datasets,
prediction outputs, and round-trip result folders are intentionally not included.
This README intentionally describes the workflow without fixed dataset-size
counts; run-specific counts should be reported in experiment logs or manuscript
tables.

## Repository Layout

```text
RD-retro/
  rdretro/data_preparation/
    Scripts for candidate-pool construction, forward-filtered dataset creation,
    random-sampling ablations, diversity pruning, atom mapping, and raw CSV
    generation.

  roundtrip/
    RoundTrip evaluation scripts and the original workflow notes from the
    retroformer-roundtrip-cal skill.

  external/ReactionT5v2/
    Minimal ReactionT5v2 code required for forward reaction validation.

  external/MolecularTransformer/
    Minimal MolecularTransformer runtime used as the post hoc forward oracle in
    round-trip evaluation.

  weights/
    Local model weights used by ReactionT5v2, MolecularTransformer, and
    RXNMapper.

  docs/
    Source mapping and generated file manifest for this repository snapshot.
```

## Environment

The pipeline expects a Python environment with common cheminformatics and deep
learning packages:

- RDKit
- PyTorch
- Transformers
- pandas
- tqdm
- rxnmapper
- lmdb

A minimal dependency list is provided in `requirements.txt`. Depending on your
CUDA and RDKit setup, it may be safer to install RDKit through conda and the
remaining Python packages through pip.

## Input Data

The repository does not include raw training data or generated prediction files.
You should provide these externally:

- USPTO-style raw CSV files with columns:
  `id`, `class`, `reactants>reagents>production`
- A retrosynthesis candidate file, usually with product/candidate columns or
  `REACTANT` and `PRODUCT` columns
- Optional aligned ground-truth CSV files for exact-match round-trip scoring

All scripts are written to operate on explicit input and output paths. The code
does not require input data to be placed inside the repository.

## RD-Retro Data Construction

The full RD-Retro data construction workflow has the following stages.

### Candidate Pool Construction

Candidate precursor sets are collected for each product from a retrosynthesis
model or a precomputed candidate file. The candidate pool may contain duplicate
reactant sets, invalid SMILES, or candidates that do not reproduce the target
product under a forward model.

Relevant scripts:

- `rdretro/data_preparation/collect_matched_reactants.py`
- `rdretro/data_preparation/filter_invalid_predicted_reactants.py`
- `rdretro/data_preparation/json_to_filter_reactiontxt.py`
- `rdretro/data_preparation/build_gt_and_filter_reactants.py`

### Forward Feasibility Filtering

ReactionT5v2 is used as an independent forward reaction model. Each candidate
reactant set is passed through the forward model, and candidates are retained
when the predicted product matches the target product after canonicalization.

Relevant code:

- `external/ReactionT5v2/task_forward/prediction.py`
- `external/ReactionT5v2/task_forward/filter_retro_candidates.py`
- the multi-GPU prediction shell script under
  `external/ReactionT5v2/task_forward/`

The local forward-model weights are stored under:

```text
weights/reactiont5_forward/ReactionT5v2-forward/
```

### Atom Mapping and Mapping-Score Filtering

RXNMapper is used to atom-map candidate reactions and provide a mapping
confidence signal. The mapped reactions can then be filtered by a confidence
threshold before further processing.

Relevant scripts:

- `rdretro/data_preparation/atom_map_reactions_rxnmapper.py`
- `rdretro/data_preparation/map_txt_to_gtcsv_worker.py`
- the multi-GPU mapping and merge shell script under
  `rdretro/data_preparation/`

The local RXNMapper transformer weights are stored under the RXNMapper weights
directory:

```text
weights/rxnmapper/
```

### Deduplication

After forward validation and atom mapping, candidates are deduplicated by their
canonical `(reactants, product)` identity. This removes repeated candidates
without applying a diversity objective.

The helper script for building the forward-filter-only ablation dataset is:

```text
rdretro/data_preparation/build_forward_only_raw_train.py
```

### Diversity-Aware Pruning

For each product-specific candidate group, RD-Retro reduces redundant reactant
sets using Morgan fingerprints and Tanimoto similarity. The pruning procedure
repeatedly identifies highly similar candidate pairs and removes the more
redundant candidate while preferentially preserving training-supported
candidates when applicable.

The default paper setting keeps at most a fixed number of candidates per product
group. This cap can be changed through script arguments.

Relevant script:

```text
rdretro/data_preparation/prune_to10_make_rxn_txt.py
```

### Raw Training CSV Assembly

After candidate selection, retained reactions are converted into the USPTO raw
CSV format:

```text
id,class,reactants>reagents>production
```

For the full RD-Retro dataset, the selected bootstrapped reactions can be
combined with the original training split. The helper scripts also support
constructing ablation datasets without one or more filtering stages.

## Ablation Dataset Helpers

The repository includes helper scripts for common ablations.

### Forward-Only Dataset

This keeps candidates that pass forward validation and mapping-score filtering,
deduplicates them, and skips fingerprint-based diversity pruning.

Example for a forward-only dataset:

```bash
python rdretro/data_preparation/build_forward_only_raw_train.py \
  --filtered-csv /path/to/filtered_mapped_reactions.csv \
  --out-dir /path/to/forward_only_dataset \
  --append-train /path/to/original/raw_train.csv \
  --copy-val /path/to/original/raw_val.csv \
  --copy-test /path/to/original/raw_test.csv
```

### No-Filter Random-Sample Dataset

This samples up to a fixed number of reactant candidates per product directly
from an input candidate CSV. It does not apply ReactionT5v2 forward validation
or fingerprint-based diversity pruning. Atom mapping should be run afterward if
the target retrosynthesis model requires mapped training reactions.

Example for a random-sample dataset:

```bash
python rdretro/data_preparation/build_no_filter_random_sample10.py \
  --input-csv /path/to/input_data.csv \
  --out-dir /path/to/no_filter_random_dataset \
  --intermediate-dir /path/to/no_filter_random_intermediates \
  --seed <RANDOM_SEED> \
  --k <MAX_CANDIDATES_PER_PRODUCT> \
  --copy-val /path/to/original/raw_val.csv \
  --copy-test /path/to/original/raw_test.csv
```

### Per-Product Top-K Limiting

This trims an existing raw training CSV so that each canonical product has at
most a fixed number of unique reactions.

Example for limiting an existing raw training CSV:

```bash
python rdretro/data_preparation/limit_raw_train_per_product_topk.py \
  --input-csv /path/to/raw_train.csv \
  --output-csv /path/to/raw_train_topk_per_product.csv \
  --k <MAX_CANDIDATES_PER_PRODUCT>
```

## RoundTrip Evaluation

The round-trip workflow is the repository version of the
`retroformer-roundtrip-cal` skill. It evaluates ranked retrosynthesis outputs
with exact-match and forward-consistency metrics, then computes reactant-level
and disconnection-level diversity.

### Step 1: Build RoundTrip Input TXT

Prediction CSVs are converted into grouped `reactants>>product` text files. Each
product row is expanded into one line per ranked candidate.

Example for building a round-trip input file:

```bash
python roundtrip/scripts/build_roundtrip_topk_input.py \
  --input-csv /path/to/predictions.csv \
  --output-txt /path/to/model_experiment_topk.txt \
  --topk <TOP_K>
```

For Retroformer merged or rebuilt prediction CSVs, use the CSV's own `product`
column as the right-hand product. Do not recover products from a separate
`raw_test.csv` with `row_index`, because row order can differ between files and
will corrupt the round-trip pairing.

### Step 2: Run RoundTrip Accuracy

`cal.py` runs MolecularTransformer on each predicted reactant set and compares
the predicted product against the target product. It also supports exact-match
condition A when an aligned ground-truth CSV is provided.

Example for running round-trip accuracy:

```bash
python roundtrip/scripts/cal.py \
  -i /path/to/model_experiment_topk.txt \
  --group-size <GROUP_SIZE> \
  --gt-csv /path/to/aligned_raw_test.csv \
  --gpu <GPU_ID>
```

The repository copy of `cal.py` defaults to the MolecularTransformer runtime
stored in:

```text
external/MolecularTransformer/
```

If your MolecularTransformer installation is elsewhere, pass `--mt-dir`.

### Step 3: Compute Reactant-Set Similarity

This post hoc script computes fingerprint-based reactant similarity within each
product group.

Example for reactant-set similarity:

```bash
python roundtrip/scripts/group_reactant_similarity_posthoc.py \
  --run-dir /path/to/cal_output_dir \
  --top-n <TOP_N> \
  --group-size <GROUP_SIZE>
```

### Step 4: Compute Disconnection Similarity

This script atom-maps prediction reactions and computes Jaccard similarity over
bond-change signatures. It is used to measure transformation-level diversity.

Example for disconnection similarity:

```bash
python roundtrip/scripts/disconnection_similarity_nomap.py \
  -i /path/to/model_experiment_topk.txt \
  -o /path/to/cal_output_dir \
  --group-size <GROUP_SIZE> \
  --top-k <TOP_K> \
  --gpu-id <GPU_ID>
```

## Model Weights

The repository snapshot includes local copies of the model weights needed for
the pipeline:

- ReactionT5v2 forward model:
  `weights/reactiont5_forward/ReactionT5v2-forward/`
- RXNMapper transformer model:
  `weights/rxnmapper/`
- MolecularTransformer models:
  `weights/molecular_transformer/`
- MolecularTransformer runtime model directory:
  `external/MolecularTransformer/experiments/models/`

These files are large. If pushing the project to GitHub, use Git LFS or replace
the weight directories with download instructions.

## What Is Not Included

The repository does not include:

- raw USPTO-style train/validation/test files
- generated bootstrapped `raw_train.csv` files
- large candidate pools such as `input_data.csv`
- model prediction CSVs
- round-trip output folders
- historical logs and intermediate experiment artifacts

These files should be regenerated or provided separately.

## Provenance

See `docs/SOURCES.md` for a path-by-path record of where each copied module came
from in the original local workspace. See `docs/file_manifest.tsv` for the
current repository file manifest.
