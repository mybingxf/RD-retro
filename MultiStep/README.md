# Multi-step retrosynthesis

This directory contains the multi-step planning implementation used for the
CASPyrus10k experiments in the RD-Retro manuscript. It is a vendored snapshot
of the Models Matter/AiZynthFinder search stack, with the Chemformer
single-step wrapper and the Chemformer fusion policy used by the experiments.

The release contains source code and configuration templates only. It does not
contain training data, stock files, neural-network checkpoints, or previously
generated route files. Those files must be supplied by the user.

## Components

- aizynthfinder/: the AiZynthFinder search and route-extraction code used by
  the recovered planner.
- external/modelsmatter_modelzoo/: the Models Matter single-step model
  interface, including Chemformer and the chemformer_mix policy.
- runtime/run_multistep.py: resumable batch runner for target SMILES.
- runtime/recompute_route_cluster_stats.py: optional route-clustering
  reanalysis.
- configs/source.yml, configs/rd_retro.yml, and configs/mixed.yml:
  path-portable templates for the three manuscript policies.

The chemformer_mix policy runs both checkpoints at each expansion, canonicalizes
precursor sets, sums probabilities for duplicate sets, keeps the highest-scoring
merged candidates, and renormalizes their probabilities.

## Installation

The vendored search code follows the original Models Matter environment
(Python 3.7--3.9). A Conda environment can be created from
environment.yml; the exact dependency specification is also retained in
pyproject.toml.

Install the Chemformer implementation separately, then expose this directory
and the model-zoo directory:

~~~bash
cd MultiStep
conda env create -f environment.yml -n rdretro-multistep
conda activate rdretro-multistep

# Install the Chemformer package required by the selected checkpoint.
# The location is the module_path value in the YAML configuration.
export PYTHONPATH="$PWD:$PWD/external/modelsmatter_modelzoo:$PYTHONPATH"
~~~

The public repository deliberately does not redistribute model weights. Set
the checkpoint, Chemformer source, vocabulary, and stock paths in the selected
configuration before running.

## Running the three policies

The templates use the same search limits as the manuscript multi-step
evaluation: Retro* search, beam width 10, at most 10 actions per expansion,
200 search iterations, an 8-hour wall-time limit, maximum transform depth 7,
target exclusion from stock, and all_routes: true.

Replace the placeholder paths in a configuration, then run:

~~~bash
python runtime/run_multistep.py \
  --dataset /path/to/CASPyrus10k.csv \
  --config configs/source.yml \
  --output-dir runs/source \
  --policy chemformer_orig \
  --stock buyables \
  --workers 1 \
  --use-gpu true \
  --save-route-artifacts \
  --resume
~~~

Use the following policy/configuration pairs:

| Policy | Configuration | Runner policy |
|---|---|---|
| Source | configs/source.yml | chemformer_orig |
| RD-Retro | configs/rd_retro.yml | chemformer_new |
| Mixed | configs/mixed.yml | chemformer_mix |

resume is appropriate when continuing an existing run. For a new run,
omit it. Multiple workers each load a model copy; choose the worker count
according to available CPU/GPU memory.

Each JSONL row records the target, search statistics, route count and (when
requested) the route-artifact path. The route count is the number of solved
reaction trees extracted from the search tree after the implementation's
recursive tree-hash deduplication. A target is solved when all terminal
precursors belong to the selected stock. These computational definitions do
not imply experimental validation.

## Reproducibility notes

The templates are intentionally free of machine-specific paths. For exact
reproduction, record the checkpoint identifiers, Chemformer version, stock
file checksum, dataset checksum, CUDA/PyTorch versions, worker count and
whether a run was resumed. Mixed evaluates two single-step checkpoints per
expansion and therefore does not have the same single-step inference cost as
the single-model policies.

The route-clustering script operates on saved route artifacts:

~~~bash
python runtime/recompute_route_cluster_stats.py \
  --run-dir-a runs/source \
  --jsonl-name CASPyrus10k.jsonl \
  --artifact-dir-name CASPyrus10k_route_artifacts \
  --max-clusters 5 10 \
  --subset solved_only \
  --workers 1
~~~

## Provenance and licenses

The search implementation is derived from AiZynthFinder and Models Matter.
Their MIT license text is included in licenses/AIZYNTHFINDER-MIT.txt; the
upstream projects are linked in licenses/NOTICE.md. The surrounding RD-Retro
code remains under the repository's existing license and notices.
