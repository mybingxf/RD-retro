#!/usr/bin/env python3
"""Verify imports and the externally downloaded forward-model checkpoint."""

import hashlib
import importlib
import sys
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
MODEL = (
    PACKAGE_ROOT
    / "third_party"
    / "MolecularTransformer"
    / "experiments"
    / "models"
    / "STEREO_separated_augm_model_average_20.pt"
)
EXPECTED_MODEL_SHA256 = "456d1027bf186945d394de17ab325046e3034e71c7ca93c4c735ad9a8b0dc8c3"
MODEL_DOWNLOAD_URL = "https://ibm.ent.box.com/v/MolecularTransformerModels"
MODULES = [
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("rdkit", "rdkit"),
    ("rxnmapper", "rxnmapper"),
    ("scipy", "scipy"),
    ("tokenizers", "tokenizers"),
    ("torch", "torch"),
    ("torchtext", "torchtext"),
    ("tqdm", "tqdm"),
    ("transformers", "transformers"),
]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    print(f"Python={sys.version.split()[0]}")
    for label, module_name in MODULES:
        module = importlib.import_module(module_name)
        print(f"{label}={getattr(module, '__version__', 'installed')}")

    from rdkit import Chem
    from torchtext.data import Field

    if Chem.MolFromSmiles("CCO") is None:
        raise RuntimeError("RDKit SMILES parsing failed")
    if Field is None:
        raise RuntimeError("Legacy torchtext Field API is unavailable")
    if not MODEL.is_file():
        raise FileNotFoundError(
            "Molecular Transformer checkpoint not found. "
            f"Download STEREO_separated_augm_model_average_20.pt from "
            f"{MODEL_DOWNLOAD_URL} and place it at: {MODEL}"
        )
    observed = sha256(MODEL)
    print(f"model_sha256={observed}")
    if observed != EXPECTED_MODEL_SHA256:
        raise RuntimeError("Bundled Molecular Transformer checkpoint checksum mismatch")
    print("INSTALLATION CHECK: PASS")


if __name__ == "__main__":
    main()
