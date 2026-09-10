"""Optional export of final mapped candidates into a raw training CSV."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict

from .utils import AtomicDictWriter, sha256sum


def append_to_training_csv(
    raw_train: Path,
    final_long: Path,
    output: Path,
    id_column: str,
    class_column: str,
    reaction_column: str,
    generated_class: str,
) -> Dict[str, object]:
    with raw_train.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
    if not fieldnames:
        raise ValueError("Raw training CSV has no header")
    for required in (id_column, class_column, reaction_column):
        if required not in fieldnames:
            raise ValueError(f"Raw training CSV is missing {required!r}")

    writer = AtomicDictWriter(output, fieldnames)
    original_rows = generated_rows = 0
    try:
        with raw_train.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                writer.writerow(row)
                original_rows += 1
        with final_long.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if "mapped_reaction_smiles" not in (reader.fieldnames or []):
                raise ValueError("Final long CSV lacks mapped_reaction_smiles")
            for candidate in reader:
                mapped = candidate["mapped_reaction_smiles"]
                if mapped.count(">>") != 1 or mapped.count(">") != 2:
                    raise ValueError("Malformed mapped reaction in final long CSV")
                row = {field: "" for field in fieldnames}
                row[id_column] = ""
                row[class_column] = generated_class
                row[reaction_column] = mapped
                writer.writerow(row)
                generated_rows += 1
    except Exception:
        writer.close(commit=False)
        raise
    else:
        writer.close()
    return {
        "output": str(output),
        "raw_training_rows": original_rows,
        "generated_rows": generated_rows,
        "total_rows": original_rows + generated_rows,
        "generated_id": "",
        "generated_class": generated_class,
        "sha256": sha256sum(output),
    }

