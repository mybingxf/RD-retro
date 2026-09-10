import csv
import json
import tempfile
import unittest
from pathlib import Path

from rdkit import DataStructs

from reaction_filter_pipeline.chemistry import (
    _combined_fingerprint,
    _choose_removal,
    _init_diversity_worker,
    _prune_group,
    canonicalize_smiles,
)
from reaction_filter_pipeline.config import DEFAULT_CONFIG, load_config
from reaction_filter_pipeline.pipeline import run_pipeline


ROOT = Path(__file__).resolve().parents[1]


class ChemistryTests(unittest.TestCase):
    def test_canonical_comparison_is_order_independent_for_fragments(self):
        self.assertEqual(
            canonicalize_smiles("CCO.O"), canonicalize_smiles("O.CCO")
        )

    def test_canonical_comparison_ignores_atom_map_labels(self):
        self.assertEqual(
            canonicalize_smiles("[CH3:1][CH2:2][OH:3]"),
            canonicalize_smiles("CCO"),
        )

    def test_multicomponent_fingerprint_is_bitwise_or_and_order_independent(self):
        _init_diversity_worker(radius=2, bits=2048, invalid_policy="error")
        first, _, _ = _combined_fingerprint("CCO.CC(=O)O")
        second, _, _ = _combined_fingerprint("CC(=O)O.CCO")
        self.assertEqual(DataStructs.TanimotoSimilarity(first, second), 1.0)

    def test_lower_confidence_member_of_most_similar_pair_is_removed(self):
        _init_diversity_worker(radius=2, bits=2048, invalid_policy="error")
        result = _prune_group(
            (
                "g",
                "CCOC(C)=O",
                [
                    (0, 1, 0.3, "CCO.CC(=O)Cl"),
                    (1, 2, 0.9, "CCO.CC(=O)Br"),
                    (2, 3, 0.8, "CCCC.CC(=O)O"),
                ],
                2,
            )
        )
        self.assertNotIn(0, result["selected_ids"])
        self.assertEqual(len(result["selected_ids"]), 2)

    def test_manuscript_mean_similarity_counterexample(self):
        # The maximum-similarity pair is (0, 2). Candidate 0 has the higher
        # mean similarity and is removed by Methods 4.4, despite its higher
        # mapping confidence. The two strategies therefore retain different IDs.
        candidates = [
            (0, 1, 0.9, "C", "generated"),
            (1, 2, 0.8, "CC", "generated"),
            (2, 3, 0.1, "CCC", "generated"),
        ]
        similarities = [
            [0.0, 0.90, 0.95],
            [0.90, 0.0, 0.10],
            [0.95, 0.10, 0.0],
        ]
        alive = [True, True, True]
        confidence_removed, confidence_rule = _choose_removal(
            0,
            2,
            candidates,
            alive,
            similarities,
            "confidence",
            ["training", "original"],
        )
        manuscript_removed, manuscript_rule = _choose_removal(
            0,
            2,
            candidates,
            alive,
            similarities,
            "source_then_mean_similarity",
            ["training", "original"],
        )
        self.assertEqual((confidence_removed, confidence_rule), (2, "confidence"))
        self.assertEqual(
            (manuscript_removed, manuscript_rule), (0, "mean_similarity")
        )
        self.assertEqual({0, 1, 2} - {confidence_removed}, {0, 1})
        self.assertEqual({0, 1, 2} - {manuscript_removed}, {1, 2})

    def test_manuscript_rule_prioritizes_training_candidate(self):
        candidates = [
            (0, 1, 0.01, "C", "training"),
            (1, 2, 0.99, "CC", "generated"),
        ]
        similarities = [[0.0, 1.0], [1.0, 0.0]]
        removed, rule = _choose_removal(
            0,
            1,
            candidates,
            [True, True],
            similarities,
            "source_then_mean_similarity",
            ["training", "original"],
        )
        self.assertEqual((removed, rule), (1, "source_priority"))

    def test_mean_similarity_strategy_does_not_use_source_priority(self):
        candidates = [
            (0, 1, 0.01, "C", "training"),
            (1, 2, 0.99, "CC", "generated"),
            (2, 3, 0.50, "CCC", "generated"),
        ]
        similarities = [
            [0.0, 0.95, 0.90],
            [0.95, 0.0, 0.10],
            [0.90, 0.10, 0.0],
        ]
        removed, rule = _choose_removal(
            0,
            1,
            candidates,
            [True, True, True],
            similarities,
            "mean_similarity",
            ["training", "original"],
        )
        self.assertEqual((removed, rule), (0, "mean_similarity"))


class OfflineIntegrationTest(unittest.TestCase):
    def test_publication_demo(self):
        config = load_config(str(ROOT / "configs" / "demo_offline.yml"))
        with tempfile.TemporaryDirectory() as temporary:
            config["output_dir"] = temporary
            config["diversity"]["workers"] = 1
            summary = run_pipeline(config, overwrite=True)
            self.assertEqual(summary["stages"]["normalize"]["candidates"], 18)
            self.assertEqual(
                summary["stages"]["forward_comparison"]["forward_matches"], 15
            )
            self.assertEqual(
                summary["stages"]["confidence_filter"][
                    "confidence_filtered_rows"
                ],
                12,
            )
            self.assertEqual(summary["validation"]["validated_final_candidates"], 9)

            with open(
                Path(temporary) / "05_final_wide.csv",
                "r",
                encoding="utf-8-sig",
                newline="",
            ) as actual_handle, open(
                ROOT / "examples" / "expected" / "demo_final_wide.csv",
                "r",
                encoding="utf-8-sig",
                newline="",
            ) as expected_handle:
                self.assertEqual(
                    list(csv.DictReader(actual_handle)),
                    list(csv.DictReader(expected_handle)),
                )

    def test_manuscript_method_demo(self):
        config = load_config(
            str(ROOT / "configs" / "demo_manuscript_method.yml")
        )
        with tempfile.TemporaryDirectory() as temporary:
            config["output_dir"] = temporary
            config["diversity"]["workers"] = 1
            summary = run_pipeline(config, overwrite=True)
            diversity = summary["stages"]["diversity"]
            self.assertEqual(
                diversity["removal_strategy"],
                "source_then_mean_similarity",
            )
            self.assertEqual(diversity["removals_by_source_priority"], 3)
            self.assertEqual(diversity["removals_by_confidence"], 0)
            with open(
                Path(temporary) / "05_final_wide.csv",
                "r",
                encoding="utf-8-sig",
                newline="",
            ) as actual_handle, open(
                ROOT / "examples" / "expected" / "demo_manuscript_final_wide.csv",
                "r",
                encoding="utf-8-sig",
                newline="",
            ) as expected_handle:
                self.assertEqual(
                    list(csv.DictReader(actual_handle)),
                    list(csv.DictReader(expected_handle)),
                )


class DefaultsTest(unittest.TestCase):
    def test_publication_defaults(self):
        self.assertEqual(DEFAULT_CONFIG["forward"]["backend"], "reactiont5v2")
        self.assertEqual(
            DEFAULT_CONFIG["forward"]["model_name_or_path"],
            "sagawa/ReactionT5v2-forward",
        )
        self.assertEqual(DEFAULT_CONFIG["confidence_threshold"], 0.2)
        self.assertEqual(
            DEFAULT_CONFIG["diversity"]["max_candidates_per_product"], 10
        )
        self.assertEqual(
            DEFAULT_CONFIG["diversity"]["removal_strategy"], "confidence"
        )


if __name__ == "__main__":
    unittest.main()
