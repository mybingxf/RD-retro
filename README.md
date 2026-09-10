# Software package for peer review

This package accompanies the manuscript:

"Reliability-constrained training-distribution refinement broadens
retrosynthetic route exploration"

The package contains two components:

1. `RD-Retro/`
   Implementation of the RD-Retro training-distribution refinement framework,
   including forward-consistency filtering and redundancy-aware candidate
   selection.

2. `Diversity_Evaluation/`
   Code for evaluating retrosynthetic reliability and diversity, including
   RoundTrip accuracy, reactant-set similarity, and reaction-center similarity.

Neural-network weights are not redistributed in this package. The
`Diversity_Evaluation` README explains how to download the required Molecular
Transformer checkpoint from its upstream distribution page.

Detailed installation and usage instructions are provided in the README file
within each subdirectory.
