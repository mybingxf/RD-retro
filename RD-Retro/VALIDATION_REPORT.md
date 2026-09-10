# Validation report

Date: 2026-09-09

## Release checks performed

- Built `reaction_filter_pipeline-1.1.0-py3-none-any.whl` from the supplied
  source.
- Installed the source distribution into a newly created Python 3.10 virtual
  environment with no inherited site packages.
- Clean installation time: 112.4 s on the test host; `pip check` reported no
  broken requirements.
- Installed the v1.1.0 wheel into an isolated target directory and loaded the
  package from that directory rather than from the source tree.
- Ran ten automated unit/integration tests from the isolated wheel: all passed
  in 0.61 s.
- Verified the source-aware rule against Methods 4.4 on physical PDF page 20,
  Eqs. (8)-(10). Tests cover source priority, current-pool mean similarity,
  the confidence-based legacy rule, and a deterministic counterexample where
  the two rules retain different candidate IDs.
- Ran the deterministic offline demo: 18 input candidates, 15 forward matches,
  12 confidence-filtered candidates, and 9 final candidates, identical to the
  supplied expected CSV.
- Ran a second deterministic offline demo using the manuscript rule: 9 final
  candidates, with three removals attributed to source priority and output
  identical to its supplied expected CSV.
- Ran the live RXNMapper CPU demo after the v1.1.0 field/schema change: atom
  mapping completed successfully in 3.9 s for the full pipeline.
- Ran the pinned ReactionT5v2 CPU demo. It predicted the official model-card
  answer `CN1CCC=C(CO)C1`, passed the forward exact-match check, and RXNMapper
  returned confidence 0.9246448292449849. The cached v1.1.0 full run completed
  in 6.6 s.
- ReactionT5v2 first CPU run: approximately 210 s including the 758 MiB model
  download. Cached full demo: approximately 7 s.
- Installed the wheel before final ZIP assembly into an isolated target
  directory and reran both offline demos: passed.
- Tested the ZIP with `unzip -t`: no archive errors.
- Verified all entries in `CHECKSUMS.sha256`.

## Tested environment

- Ubuntu 22.04-compatible Linux, x86-64
- Python 3.10.19
- RDKit 2022.09.5
- PyTorch 2.7.1+cu118
- Transformers 4.42.3 / tokenizers 0.19.1
- RXNMapper 0.4.3
- NVIDIA driver 535.129.03; GeForce RTX 4090, 24 GB

The live ReactionT5v2 release test was run on CPU because the available GPUs
were occupied by unrelated training jobs. The package's GPU selection uses the
same PyTorch model and was syntax/integration checked, while RXNMapper had also
been exercised on CUDA during the originating full-data workflow.

The generic Hugging Face and precomputed forward backends are covered by the
tested ReactionT5v2/precomputed paths. The direct Chemformer adapter requires a
checkpoint-compatible external Chemformer dependency stack and is not part of
the clean default environment; the README documents the precomputed interface
as the portable fallback.
