# Demo data

`predictions_demo.csv` is a two-product, Top-10 subset prepared solely to
exercise the software. Each row contains a product, its ground-truth
reactants, and ten ranked retrosynthesis candidates. The reactions are drawn
from the public USPTO-50K benchmark; the ranked candidates are example model
outputs.

`aligned_gt_demo.csv` contains the same two reactions in the schema consumed
by condition-A exact-match scoring. The pipeline can reproduce this file from
the `product` and `ground_truth` columns of `predictions_demo.csv`.

No demo record is intended for model training or scientific benchmarking.
