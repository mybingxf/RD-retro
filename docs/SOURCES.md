# Source Mapping

This repository snapshot was assembled from the local experimental workspace.
Only code, workflow notes, and model weights needed for RD-Retro data
construction and roundtrip evaluation were copied. Historical outputs were not
copied.

## Data Preparation

Copied from `/root/autodl-fs/nature/`:

- `sanitize_product_in_csv.py`
- `insert_gt_rows_before_candidates.py`
- `build_forward_csv_with_gt.py`

Copied from `/root/autodl-fs/nature/all_data/`:

- `prune_to10_make_rxn_txt.py`
- `map_txt_to_gtcsv_worker.py`
- `run_map_txt_and_merge_gt_8gpu.sh`

Copied from `/root/autodl-fs/try/train/localretro/`:

- `collect_matched_reactants.py`
- `filter_invalid_predicted_reactants.py`
- `json_to_filter_reactiontxt.py`
- `atom_map_reactions_rxnmapper.py`
- `build_gt_and_filter_reactants.py`
- `diversity_prune.py`

New repository helper scripts:

- `build_forward_only_raw_train.py`
- `build_no_filter_random_sample10.py`
- `limit_raw_train_per_product_topk.py`

## ReactionT5 Forward Filtering

Copied minimal code from `/root/autodl-fs/ReactionT5v2/`:

- `generation_utils.py`
- `utils.py`
- `models.py`
- `task_forward/*.py`
- `task_forward/run_9gpu_predict.sh`

Copied local forward model weights from:

- `/root/autodl-fs/ReactionT5v2/task_forward/weights/ReactionT5v2-forward`

## RoundTrip Evaluation

Copied skill workflow from:

- `/root/.codex/skills/retroformer-roundtrip-cal/SKILL.md`
- `/root/.codex/skills/retroformer-roundtrip-cal/scripts/build_roundtrip_topk_input.py`

Copied calculation scripts from `/root/autodl-fs/roundtripcal/`:

- `cal.py`
- `disconnection_similarity_nomap.py`
- `batch_cal_acc.py`
- `roundtrip_mt_grouped_acc.py`
- `rt_withgt.py`
- `build_roundtrip_input_from_prediction_csv.py`
- `build_roundtrip_input_from_neuralsym_csv.py`
- `standardize_strip_stereo_inplace.py`
- `megan_newfilter_top10/group_reactant_similarity_posthoc.py`

The repository copy of `cal.py` was adjusted to default to
`external/MolecularTransformer` inside this repository.

## MolecularTransformer RoundTrip Oracle

Copied minimal runtime from `/root/autodl-fs/MolecularTransformer/`:

- `translate.py`
- `preprocess.py`
- `train.py`
- `setup.py`
- `requirements.txt`
- `onmt/`
- `tools/`
- upstream README and license

Copied model weights from:

- `/root/autodl-fs/MolecularTransformer/experiments/models/`

## RXNMapper

Copied the local RXNMapper transformer model from:

- `/root/miniconda3/lib/python3.10/site-packages/rxnmapper/models/transformers/albert_heads_8_uspto_all_1310k`

RXNMapper Python code is not vendored as a source package; install it with
`pip install rxnmapper`.

