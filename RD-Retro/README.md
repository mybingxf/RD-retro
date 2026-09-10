# Reaction Filter Pipeline

Version 1.1.0

This archive contains the source code and demonstration data for filtering
retrosynthesis candidates by:

1. top-1 forward-reaction agreement with the proposed product;
2. RXNMapper atom mapping and mapping-confidence thresholding; and
3. within-product Morgan-fingerprint/Tanimoto diversity selection.

The default forward model is **ReactionT5v2**
(`sagawa/ReactionT5v2-forward`). The default confidence threshold is **0.2**
(inclusive), and the default maximum is **10 candidates per product**. The
member-removal rule used during diversity pruning is selectable. Version 1.1.0
keeps the earlier confidence-based rule as the backward-compatible default and
also implements the source-aware mean-similarity rule in Methods 4.4,
Eqs. (8)-(10). All of these values are configurable.

The pipeline is:

```text
top-k retrosynthesis candidates
  -> forward top-1 product prediction
  -> RDKit-canonical exact product agreement
  -> RXNMapper atom mapping + confidence
  -> confidence >= threshold
  -> Morgan(radius=2, 2048-bit) OR fingerprint per precursor set
  -> selectable iterative maximum-Tanimoto pruning to N candidates per product
```

Model weights are not redistributed in this ZIP. ReactionT5v2 is downloaded
from its official Hugging Face repository on first use; RXNMapper installs its
model with the package. See `THIRD_PARTY_NOTICES.md`.

## Nature Research code checklist

This archive includes:

- complete Python source code under `src/`;
- a prebuilt, platform-independent Python wheel under `dist/`;
- install metadata (`pyproject.toml`, `environment.yml`, and
  `requirements-tested.txt`);
- a small demonstration dataset and fixed expected results under `examples/`;
- executable offline, RXNMapper, and ReactionT5v2 demonstration configurations;
- a `configs/manuscript_method.yml` template for the Methods 4.4 diversity rule;
- automated tests under `tests/`;
- SHA-256 checksums for the wheel and demo fixtures in `CHECKSUMS.sha256`; and
- this README, covering system requirements, installation, demo, expected
  output/runtime, and use on new data.

Python is interpreted source software, so no platform-specific compiled
standalone executable is required. Binary dependencies such as RDKit and
PyTorch are installed by `pip`/Conda.

## 1. System requirements

### Supported and tested systems

The supplied code requires:

- Python 3.9–3.11 (Python 3.10 recommended);
- Linux, macOS, or Windows for the offline/precomputed workflow;
- RDKit for SMILES canonicalization and Morgan fingerprints;
- PyTorch, Transformers, and RXNMapper for model-backed runs.

Full integration was tested on:

- Ubuntu 22.04-compatible Linux, x86-64;
- Python 3.10.19;
- NumPy 1.26.4;
- PyYAML 6.0.2;
- tqdm 4.66.5;
- RDKit 2022.09.5;
- PyTorch 2.7.1+cu118;
- Transformers 4.42.3 and tokenizers 0.19.1;
- RXNMapper 0.4.3;
- NVIDIA driver 535.129.03; and
- NVIDIA GeForce RTX 4090 GPUs with 24 GB memory.

The exact tested Python package set is in `requirements-tested.txt`.

### Hardware

No non-standard hardware is required for the offline demo or small CPU runs.
A conventional 4-core CPU and 16 GB RAM are sufficient for the included demo.

An NVIDIA CUDA GPU is strongly recommended for large ReactionT5v2/RXNMapper
runs. The pipeline can use one or more GPUs by assigning one model worker per
device. Each worker loads its own model copy. For the default ReactionT5v2
checkpoint, allow at least 4 GB GPU memory per forward worker; longer inputs or
larger batches may require more. CPU inference is supported but slower.

The pinned ReactionT5v2 model occupies approximately 758 MiB in the local
Hugging Face cache. Large result CSVs can require several times the input file
size because every stage is retained for auditability.

## 2. Installation guide

### Minimal installation for the deterministic offline demo

Unzip the archive, open a terminal in its root directory, and run:

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e .
```

On a normal desktop with broadband internet, this normally takes 2–5 minutes.
It installs RDKit but not the neural-model dependencies.

The included wheel is an alternative to editable source installation:

```bash
python -m pip install dist/reaction_filter_pipeline-1.1.0-py3-none-any.whl
```

Keep the extracted `configs/` and `examples/` directories beside the wheel when
following the demo commands.

### Full installation for ReactionT5v2 and RXNMapper

```bash
python -m pip install -e ".[models]"
```

Alternatively, with Conda/Miniconda:

```bash
conda env create -f environment.yml
conda activate reaction-filter-pipeline
```

A normal desktop installation typically takes 5–15 minutes, depending on the
PyTorch wheel and network speed. The first ReactionT5v2 run additionally
downloads about 758 MiB. For a specific CUDA version, install the appropriate
PyTorch build from <https://pytorch.org/get-started/locally/> before installing
this package.

Verify the installation:

```bash
reaction-filter --version
reaction-filter list-backends
python -m unittest discover -s tests -v
```

Verify the supplied wheel and demo fixtures on Linux/macOS with:

```bash
sha256sum -c CHECKSUMS.sha256
```

The command may also be invoked as:

```bash
python -m reaction_filter_pipeline --version
```

## 3. Demo

### A. Deterministic offline demo (recommended first check)

This demo uses fixed forward predictions and fixed RXNMapper results generated
for the included 18 candidates. It executes every comparison, thresholding,
and diversity-selection operation without downloading a model.

```bash
reaction-filter run --config configs/demo_offline.yml --overwrite
```

Expected console summary:

```json
{
  "input_candidates": 18,
  "forward_matches": 15,
  "confidence_filtered": 12,
  "final_candidates": 9,
  "final_nonempty_groups": 3
}
```

The three final product groups each contain three candidates. Compare the
principal result with the supplied expected file:

```bash
diff demo_output_offline/05_final_wide.csv examples/expected/demo_final_wide.csv
diff demo_output_offline/05_final_long.csv examples/expected/demo_final_long.csv
```

Both commands should produce no output. This demo took approximately 0.5 s on
the tested system and should finish in under 10 s on a normal desktop after
installation.

### B. Demo with a live RXNMapper model

Install the `models` extra, then run:

```bash
reaction-filter run --config configs/demo_rxnmapper.yml --overwrite
```

This uses the fixed forward predictions but recomputes atom mappings and
confidence values with RXNMapper on CPU. It took approximately 4.5 s on the
tested system; allow 15–60 s on a normal desktop. Small numerical differences
in confidence can occur with different dependency versions.

### C. Demo with the default ReactionT5v2 and RXNMapper models

```bash
reaction-filter run \
  --config configs/demo_reactiont5v2.yml \
  --forward-devices cpu \
  --mapping-devices cpu \
  --overwrite
```

The included reaction is the example from the official ReactionT5v2 model
card. The expected top-1 prediction is:

```text
CN1CCC=C(CO)C1
```

It appears in `demo_output_reactiont5v2/01_forward_predictions.csv` and passes
the canonical exact-match filter. On the tested system, the first CPU run took
about 210 s including the 758 MiB model download; a cached CPU run took about
7 s for the full forward-plus-mapping pipeline. Network and CPU performance
will affect these times.

To use a GPU instead:

```bash
reaction-filter run \
  --config configs/demo_reactiont5v2.yml \
  --forward-devices cuda:0 \
  --mapping-devices cuda:0 \
  --overwrite
```

The official model and source are available at:

- <https://huggingface.co/sagawa/ReactionT5v2-forward>
- <https://github.com/sagawatatsuya/ReactionT5v2>

For reproducibility, the default configuration pins Hugging Face revision
`933114058cb2604dc1bf536dbebdfcefbe83d4fc` rather than a moving `main` branch.

### D. Deterministic demo of the manuscript diversity rule

The demo input labels the first candidate in every group as `training`. Run:

```bash
reaction-filter run \
  --config configs/demo_manuscript_method.yml \
  --overwrite
```

The run retains 9 candidates and reports three
`removals_by_source_priority` in `summary.json`/`05_group_stats.csv`. Compare
the result with:

```bash
diff demo_output_manuscript_method/05_final_wide.csv \
  examples/expected/demo_manuscript_final_wide.csv
```

This output intentionally differs from the confidence-based demo for the ester
and amide groups, demonstrating that strategy selection is consequential.

## 4. Instructions for use on your data

### Input format A: wide top-k CSV

This is the default format used by LocalRetro-style prediction files:

```csv
product,reactants1,reactants2,reactants3
CCOC(C)=O,CCO.CC(=O)O,CCO.CC(=O)Cl,
CCNC(C)=O,CCN.CC(=O)O,CCN.CC(=O)Cl,CCN.CC(=O)Br
```

Candidate columns are selected by the `reactants` prefix, or may be listed
explicitly using `input.candidate_columns`. Blank candidate cells are ignored.
Every source row remains a distinct product group, even if two rows contain
the same product SMILES. This preserves exact correspondence with the input.

### Input format B: long CSV

```csv
group_id,rank,product,reactants,reagent,source_type
reaction_1,1,CCOC(C)=O,CCO.CC(=O)O,,training
reaction_1,2,CCOC(C)=O,CCO.CC(=O)Cl,,generated
reaction_2,1,CCNC(C)=O,CCN.CC(=O)Cl,,generated
```

Rows for each `group_id` must be contiguous. A reagent column is optional.
For wide inputs, reagent is blank because the precursor set is supplied as a
single dot-separated reactant string. `source_type` is optional and defaults
to `generated`; it is required in practice when the source-aware manuscript
rule must distinguish original training reactions from model predictions.

### Default complete run

Copy `configs/default.yml`, edit `input.path` and `output_dir`, then run:

```bash
reaction-filter run --config my_config.yml
```

Or override common settings directly:

```bash
reaction-filter run \
  --config configs/default.yml \
  --input /path/to/top100.csv \
  --output-dir /path/to/results \
  --confidence-threshold 0.2 \
  --max-candidates 10 \
  --diversity-strategy source_then_mean_similarity \
  --forward-devices cuda:0,cuda:1 \
  --mapping-devices cuda:0,cuda:1
```

The two GPU stages run sequentially, so the same devices may be reused. A
round-robin shard and one process are created for each listed device.

### Key adjustable parameters

| YAML key | Default | Meaning |
|---|---:|---|
| `forward.backend` | `reactiont5v2` | Forward-prediction implementation |
| `forward.model_name_or_path` | `sagawa/ReactionT5v2-forward` | Hugging Face ID or local checkpoint |
| `forward.batch_size` | 16 | Forward inference batch size per device |
| `forward.devices` | `[auto]` | CPU/GPU workers |
| `mapping.backend` | `rxnmapper` | Atom-mapping implementation |
| `mapping.batch_size` | 64 | RXNMapper batch size per device |
| `mapping.devices` | `[auto]` | CPU/GPU mapping workers |
| `confidence_threshold` | 0.2 | Inclusive RXNMapper confidence cutoff |
| `diversity.max_candidates_per_product` | 10 | Final maximum for each source product row |
| `diversity.removal_strategy` | `confidence` | Which member of the most-similar pair to delete |
| `diversity.original_source_types` | `[training, original]` | Source labels treated as original data |
| `diversity.morgan_radius` | 2 | Morgan fingerprint radius |
| `diversity.morgan_bits` | 2048 | Fingerprint bit dimension |
| `diversity.workers` | 8 | CPU processes for diversity selection |

Print every available setting with:

```bash
reaction-filter show-default-config
```

### Forward-model backends

#### ReactionT5v2 (default)

`reactiont5v2` uses Hugging Face `AutoTokenizer` and
`AutoModelForSeq2SeqLM`. The default prompt follows the official model:

```text
REACTANT:{reactants}REAGENT:{reagent}
```

Top-1 generation is used for round-trip validation.

#### Other Hugging Face sequence-to-sequence models

Select `hf_seq2seq`, set `model_name_or_path`, and adjust `prompt_template` to
the model's required input syntax:

```yaml
forward:
  backend: hf_seq2seq
  model_name_or_path: /path/to/model-or-hub-id
  revision: null
  prompt_template: "{reactants}"
  devices: [cuda:0]
```

#### Chemformer

The direct Chemformer adapter expects an external MolecularAI/Chemformer
checkout, a forward checkpoint, its vocabulary, and a compatible Chemformer
Python environment:

```yaml
forward:
  backend: chemformer
  devices: [cuda:0]
  batch_size: 64
  chemformer_root: /path/to/Chemformer
  chemformer_checkpoint: /path/to/forward_model.ckpt
  chemformer_vocabulary: /path/to/bart_vocab_downstream.json
  chemformer_max_length: 512
```

Chemformer has older, checkpoint-specific dependencies. If they conflict with
the main environment, run Chemformer in its official environment, export the
top-1 products, and use the `precomputed` backend below. This also provides a
stable interface for any other forward model.

#### Precomputed predictions from any model

The prediction CSV must be sorted by integer `candidate_id` and contain:

```csv
candidate_id,predicted_product
0,CCOC(C)=O
1,CCOC(C)=O
```

Configure:

```yaml
forward:
  backend: precomputed
  prediction_file: /path/to/predictions.csv
  prediction_id_column: candidate_id
  prediction_column: predicted_product
```

Generate the normalized IDs before calling an external model:

```bash
reaction-filter prepare \
  --config my_config.yml \
  --input /path/to/candidates.csv \
  --output-dir /path/to/prepared
```

Use `/path/to/prepared/00_candidates.csv` as the external model input, save the
sorted prediction CSV, then set that file as `forward.prediction_file` for the
complete run. Normalization is deterministic for unchanged input/configuration.

### Forward-product comparison

Both target and predicted products are parsed with RDKit, atom-map labels are
removed, and the molecules are serialized as canonical isomeric SMILES. A
candidate passes only if the two canonical strings are exactly equal. Invalid
predictions are rejected and counted in the stage statistics.

### RXNMapper confidence filter

For every forward-consistent candidate, RXNMapper receives:

```text
reactants>>product
```

The mapped reaction and overall RXNMapper `confidence` are retained. The
threshold is inclusive: with the default, `confidence >= 0.2` passes and values
below 0.2 are removed. Mapping failures are removed separately and counted.

For a completely offline rerun, `mapping.backend: precomputed` accepts a
sorted CSV containing `candidate_id`, `mapped_reaction_smiles`,
`mapping_confidence`, and optionally `mapping_success`.

### Diversity selection algorithms

For each candidate precursor set:

1. split the reactant SMILES on `.`;
2. calculate a Morgan bit fingerprint for every component;
3. combine component fingerprints with bitwise OR into the same bit dimension;
4. calculate all pairwise Tanimoto similarities within one product group;
5. choose the currently most similar live pair;
6. choose one member for removal using `diversity.removal_strategy`; and
7. repeat, recalculating any required current-pool statistics, until at most
   the configured number remains.

Three removal strategies are available:

- `confidence` (backward-compatible default): remove the member with lower
  RXNMapper mapping confidence. On a confidence tie, retain the lower original
  rank, followed by the lower normalized candidate ID.
- `source_then_mean_similarity` (Methods 4.4, Eqs. (8)-(10)): if exactly one
  member has a `source_type` listed in `original_source_types`, retain that
  original-training member. Otherwise, calculate each member's mean Tanimoto
  similarity to every other currently live candidate and remove the member
  with the higher mean (the more redundant candidate).
- `mean_similarity`: always apply the current-pool mean-similarity rule and
  ignore source labels. This is useful as an ablation.

For both mean-similarity strategies, an exactly equal mean is resolved by
retaining the lower original rank and then the lower candidate ID. Mapping
confidence is deliberately not used in this tie-break. Pair-selection ties are
deterministic by normalized input order.

To select the manuscript rule, either start from
`configs/manuscript_method.yml` or set:

```yaml
input:
  source_column: source_type
  default_source_type: generated

diversity:
  removal_strategy: source_then_mean_similarity
  original_source_types: [training, original]
```

For a wide CSV, all candidates default to `generated`. Individual columns can
be assigned another source type as follows:

```yaml
input:
  default_source_type: generated
  source_by_candidate_column:
    reactants1: training
    reactants2: training
```

The source label affects the diversity-removal decision only; every input row
still goes through the configured forward and mapping stages. If the input pool
contains only generated candidates, the source-priority branch cannot occur
and `source_then_mean_similarity` uses its mean-similarity branch throughout.

Important reproducibility note: changing `removal_strategy` can change the
retained reactions even with identical fingerprints and mapping confidence.
Version 1.1.0 therefore does not silently change the default. Before assigning
a strategy to previously reported results, identify which rule produced those
result files and rerun affected experiments if the rule is changed.

The procedure stops as soon as the requested count is reached, so a final group
can still contain an exact fingerprint collision if both candidates fall
within that limit.

The default behavior is to stop with an error on invalid reactant SMILES. Set
`diversity.invalid_smiles: zero` to ignore invalid components and use the union
of remaining valid components; such events are counted in `05_group_stats.csv`.

### Output files

| File | Content |
|---|---|
| `00_candidates.csv` | Normalized long-form candidates and stable IDs |
| `00_product_groups.csv` | Every input product group, including empty groups |
| `01_forward_predictions.csv` | Top-1 product from the selected forward backend |
| `02_forward_pass.csv` | Candidates whose canonical predicted product matches |
| `03_rxnmapper_mapped.csv` | Atom-mapped reactions and mapping confidence |
| `04_confidence_filtered.csv` | Successful mappings at or above the threshold |
| `05_final_long.csv` | Final candidates with mapping and confidence metadata |
| `05_final_wide.csv` | `product,reactants1,...,reactantsN` for downstream tools |
| `05_group_stats.csv` | Before/after counts, similarities, and removal-rule counts per group |
| `summary.json` | Parameters, counts, versions, file hashes, and validation |
| `run_config.resolved.yml` | Exact resolved configuration used for the run |

Intermediate stages are intentionally retained for audit and reproduction.
Use `--resume` only with the same resolved configuration to skip completed
stages. Use `--overwrite` to recompute known artifacts in an existing output
directory.

### Optional export into an existing training CSV

To reproduce the manuscript workflow that appends mapped generated reactions
to a raw training set while leaving generated IDs blank:

```bash
reaction-filter append-training \
  --raw-train /path/to/raw_train.csv \
  --final-long /path/to/results/05_final_long.csv \
  --output /path/to/chemformer_filterdiversity_50K.csv \
  --id-column id \
  --class-column class \
  --reaction-column 'reactants>reagents>production' \
  --generated-class UNK
```

Original rows are copied first without modification. Generated rows have blank
IDs and use the atom-mapped `reactants>>product` string from the final long
table.

## Reproducibility and tests

Run:

```bash
python -m unittest discover -s tests -v
```

Every completed run writes SHA-256 hashes, row counts, package versions, the
resolved configuration, selected diversity strategy, removal-rule counts, and
per-stage timings into `summary.json`. Neural predictions are deterministic
because top-1 beam generation is used without sampling. Diversity selection
has explicit deterministic tie rules. Automated tests include a fixed
counterexample proving that the confidence and manuscript strategies may
retain different candidates.

The offline demo provides byte-stable expected final CSVs. RXNMapper confidence
can vary slightly across PyTorch/Transformers versions; use the pinned tested
environment when exact numeric reproduction is required.

## Known limitations

- RXNMapper confidence is a model-derived mapping score, not a calibrated
  probability of reaction feasibility.
- Bitwise OR intentionally ignores component multiplicity and component order.
- Forward exact matching depends on RDKit stereochemical canonicalization.
- The package does not redistribute third-party neural-network weights.
- Chemformer checkpoint compatibility depends on the corresponding upstream
  Chemformer code and vocabulary versions.
- Source-aware pruning is only meaningful when candidate provenance is
  correctly supplied in `source_type` (or its configured equivalent).

## References

If these models are used in a publication, cite their original articles and
software:

- Sagawa, T. and Kojima, R. *ReactionT5: a pre-trained transformer model for
  accurate chemical reaction prediction with limited data*. Journal of
  Cheminformatics **17**, 126 (2025). DOI: 10.1186/s13321-025-01075-4.
- Schwaller, P. et al. *Extraction of organic chemistry grammar from unsupervised
  learning of chemical reactions*. Science Advances **7**, eabe4166 (2021)
  (RXNMapper).
- Irwin, R. et al. *Chemformer: a pre-trained transformer for computational
  chemistry*. Machine Learning: Science and Technology **3**, 015022 (2022).

## License

The code in this archive is released under the MIT License. Third-party
software and models retain their original licenses; see
`THIRD_PARTY_NOTICES.md`.
