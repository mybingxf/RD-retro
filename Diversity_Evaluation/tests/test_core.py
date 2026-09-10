import csv
import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CoreTests(unittest.TestCase):
    def test_group_accuracy(self):
        cal = load_module("roundtrip_cal", PACKAGE_ROOT / "source" / "cal.py")
        groups, rows = cal.compute_group_acc([0, 1, 1, 0], group_size=2, max_k=2)
        self.assertEqual(groups, 2)
        self.assertEqual(rows[0], (1, 0.5, 0.5))
        self.assertEqual(rows[1], (2, 1.0, 0.5))

    def test_detail_reader_uses_named_correct_column(self):
        similarity = load_module(
            "reactant_similarity",
            PACKAGE_ROOT / "source" / "group_reactant_similarity_posthoc.py",
        )
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            with (run_dir / "roundtrip_detail.tsv").open(
                "w", encoding="utf-8", newline=""
            ) as handle:
                writer = csv.writer(handle, delimiter="\t")
                writer.writerow(
                    [
                        "idx",
                        "group",
                        "rank_in_group",
                        "src",
                        "gt_reactants_hit",
                        "condA_gt_match",
                        "condB_mt_roundtrip",
                        "tgt",
                        "pred_clean",
                        "correct",
                    ]
                )
                writer.writerow([0, 0, 1, "CC", 1, 0, 1, "CC", "CC", 1])
                writer.writerow([1, 0, 2, "CO", 1, 1, 0, "CO", "CN", 1])
            flags, group_size = similarity.load_correct_flags(run_dir, 2, 2)
            self.assertEqual(flags, [1, 1])
            self.assertEqual(group_size, 2)

    def test_builder_preserves_missing_rank(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            input_csv = tmpdir / "predictions.csv"
            output_txt = tmpdir / "roundtrip.txt"
            input_csv.write_text(
                "product,top_1,top_2\nCCO,CC,\n", encoding="utf-8"
            )
            subprocess.run(
                [
                    sys.executable,
                    str(PACKAGE_ROOT / "source" / "build_roundtrip_topk_input.py"),
                    "--input-csv",
                    str(input_csv),
                    "--output-txt",
                    str(output_txt),
                    "--topk",
                    "2",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertEqual(
                output_txt.read_text(encoding="utf-8").splitlines(),
                ["CC>>CCO", "9999>>CCO"],
            )


if __name__ == "__main__":
    unittest.main()
