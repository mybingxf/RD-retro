# Third-party notices

## Molecular Transformer / OpenNMT-py

The directory `third_party/MolecularTransformer` contains the inference subset
of the Molecular Transformer repository. The pretrained checkpoint is not
redistributed in this archive. Users must obtain it from the upstream model
distribution page and place it in the local model directory before use.

- Upstream repository: `https://github.com/pschwllr/MolecularTransformer`
- Upstream commit: `aeb339daf0a029b391f8307fb3f467f461605dd2`
- Model distribution: `https://ibm.ent.box.com/v/MolecularTransformerModels`
- Required filename: `STEREO_separated_augm_model_average_20.pt`
- Required destination:
  `third_party/MolecularTransformer/experiments/models/STEREO_separated_augm_model_average_20.pt`
- Checkpoint SHA-256:
  `456d1027bf186945d394de17ab325046e3034e71c7ca93c4c735ad9a8b0dc8c3`
- Licence: MIT and BSD-3-Clause; see
  `LICENSES/MolecularTransformer-LICENSE.md`.

Only source files required for forward inference are included; model weights,
training data, tests, cached bytecode, Git metadata, and other pretrained
checkpoints are omitted.

Two compatibility edits are present relative to the upstream commit:

1. `onmt/model_builder.py` passes `weights_only=False` to `torch.load`, required
   for legacy checkpoint objects under PyTorch 2.6 and later.
2. `onmt/translate/translator.py` converts beam-search selection indices to
   integer-long tensors, required by current PyTorch.

No model architecture or learned parameters were changed.

## RXNMapper

Disconnection similarity depends on `rxnmapper==0.4.2`, installed from PyPI by
`requirements.txt`. RXNMapper includes its pretrained ALBERT atom-mapping model
inside the Python package, so it does not download a model at demo runtime.
RXNMapper is MIT-licensed. Its upstream repository is
`https://github.com/rxn4chemistry/rxnmapper`.

## Other Python dependencies

PyTorch, torchtext, RDKit, NumPy, pandas, SciPy, Transformers, tokenizers, tqdm,
and the remaining packages are installed from their standard distributions and
are not copied into this archive. Their exact tested versions are recorded in
`requirements.txt`; each remains governed by its own licence.
