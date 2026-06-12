# RoundTrip Evaluation Module

This folder contains the roundtrip calculation code copied from the local
`retroformer-roundtrip-cal` skill and the supporting scripts from
`/root/autodl-fs/roundtripcal`.

## Files

- `skill/retroformer-roundtrip-cal.SKILL.md`: original workflow instructions.
- `scripts/build_roundtrip_topk_input.py`: convert prediction CSVs into
  grouped `reactants>>product` TXT inputs.
- `scripts/cal.py`: compute exact-match and MolecularTransformer round-trip
  accuracy.
- `scripts/group_reactant_similarity_posthoc.py`: compute reactant-set
  fingerprint similarity for grouped predictions.
- `scripts/disconnection_similarity_nomap.py`: compute disconnection-level
  similarity using RXNMapper atom mapping.

## Notes

The repository copy of `cal.py` defaults to
`external/MolecularTransformer` under the repository root. If your forward-model
runtime is elsewhere, pass `--mt-dir /path/to/MolecularTransformer`.

For exact-match condition A, pass an aligned `--gt-csv`. For ablation or merged
Retroformer outputs, create the aligned GT CSV from the same prediction rows,
using `ground_truth>>clean_product`.

