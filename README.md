# Software package for peer review

This package accompanies the manuscript:

"Reliability-constrained training-distribution refinement broadens
retrosynthetic route exploration"

The package contains three components:

1. `RD-Retro/`
   Implementation of the RD-Retro training-distribution refinement framework,
   including forward-consistency filtering and redundancy-aware candidate
   selection.

2. `Diversity_Evaluation/`
   Code for evaluating retrosynthetic reliability and diversity, including
   RoundTrip accuracy, reactant-set similarity, and reaction-center similarity.

3. `MultiStep/`
   The recovered multi-step retrosynthesis implementation used for the
   manuscript's CASPyrus10k planning experiments. It is based on the
   Models Matter/AiZynthFinder search code and includes Source, RD-Retro and
   Mixed Chemformer configuration templates.

Neural-network weights are not redistributed in this package. The
`Diversity_Evaluation` README explains how to download the required Molecular
Transformer checkpoint from its upstream distribution page.

Detailed installation and usage instructions are provided in the README file
within each subdirectory.
