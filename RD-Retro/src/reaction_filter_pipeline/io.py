"""Input normalization for wide and long retrosynthesis candidate CSV files."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Tuple

from .utils import AtomicDictWriter, sha256sum


CANDIDATE_FIELDS = [
    "candidate_id",
    "group_id",
    "rank",
    "product",
    "reactants",
    "reagent",
    "source_type",
]
GROUP_FIELDS = ["group_id", "product"]


def normalize_input(
    input_path: Path,
    candidate_output: Path,
    group_output: Path,
    input_config: Dict[str, object],
) -> Dict[str, object]:
    with input_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError("Input CSV has no header")
        supplied_format = str(input_config.get("format", "auto"))
        reactants_column = str(input_config.get("reactants_column", "reactants"))
        if supplied_format == "auto":
            input_format = "long" if reactants_column in reader.fieldnames else "wide"
        else:
            input_format = supplied_format

    if input_format == "wide":
        stats = _normalize_wide(
            input_path, candidate_output, group_output, input_config
        )
    else:
        stats = _normalize_long(
            input_path, candidate_output, group_output, input_config
        )
    stats.update(
        {
            "input_path": str(input_path),
            "input_sha256": sha256sum(input_path),
            "input_format": input_format,
        }
    )
    return stats


def _normalize_wide(
    input_path: Path,
    candidate_output: Path,
    group_output: Path,
    config: Dict[str, object],
) -> Dict[str, object]:
    product_column = str(config.get("product_column", "product"))
    prefix = str(config.get("candidate_prefix", "reactants"))
    configured_columns = config.get("candidate_columns")
    default_source = str(config.get("default_source_type", "generated")).strip()
    source_by_column = {
        str(key): str(value).strip()
        for key, value in dict(config.get("source_by_candidate_column", {})).items()
    }
    if not default_source:
        raise ValueError("input.default_source_type cannot be blank")
    candidates_writer = AtomicDictWriter(candidate_output, CANDIDATE_FIELDS)
    groups_writer = AtomicDictWriter(group_output, GROUP_FIELDS)
    candidate_count = 0
    group_count = 0
    empty_groups = 0
    try:
        with input_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if product_column not in (reader.fieldnames or []):
                raise ValueError(f"Missing product column: {product_column}")
            if configured_columns:
                candidate_columns = [str(x) for x in configured_columns]
            else:
                candidate_columns = [
                    name
                    for name in (reader.fieldnames or [])
                    if name != product_column and name.startswith(prefix)
                ]
            if not candidate_columns:
                raise ValueError(
                    "No candidate columns found; set input.candidate_columns or "
                    "input.candidate_prefix"
                )
            for missing in set(candidate_columns).difference(reader.fieldnames or []):
                raise ValueError(f"Missing candidate column: {missing}")

            for row_index, row in enumerate(reader):
                product = (row.get(product_column) or "").strip()
                if not product:
                    raise ValueError(f"Blank product at CSV data row {row_index + 1}")
                group_id = str(row_index)
                groups_writer.writerow({"group_id": group_id, "product": product})
                in_group = 0
                for rank, column in enumerate(candidate_columns, start=1):
                    reactants = (row.get(column) or "").strip()
                    if not reactants:
                        continue
                    candidates_writer.writerow(
                        {
                            "candidate_id": candidate_count,
                            "group_id": group_id,
                            "rank": rank,
                            "product": product,
                            "reactants": reactants,
                            "reagent": "",
                            "source_type": source_by_column.get(
                                column, default_source
                            ),
                        }
                    )
                    candidate_count += 1
                    in_group += 1
                empty_groups += in_group == 0
                group_count += 1
    except Exception:
        candidates_writer.close(commit=False)
        groups_writer.close(commit=False)
        raise
    else:
        candidates_writer.close()
        groups_writer.close()
    return {
        "groups": group_count,
        "empty_input_groups": empty_groups,
        "candidates": candidate_count,
        "candidate_columns": candidate_columns,
    }


def _normalize_long(
    input_path: Path,
    candidate_output: Path,
    group_output: Path,
    config: Dict[str, object],
) -> Dict[str, object]:
    product_column = str(config.get("product_column", "product"))
    reactants_column = str(config.get("reactants_column", "reactants"))
    reagent_column = config.get("reagent_column")
    group_column = str(config.get("group_column", "group_id"))
    rank_column = str(config.get("rank_column", "rank"))
    source_column = config.get("source_column", "source_type")
    default_source = str(config.get("default_source_type", "generated")).strip()
    if not default_source:
        raise ValueError("input.default_source_type cannot be blank")
    candidates_writer = AtomicDictWriter(candidate_output, CANDIDATE_FIELDS)
    groups_writer = AtomicDictWriter(group_output, GROUP_FIELDS)
    candidate_count = 0
    group_count = 0
    closed_groups = set()
    current_group = None
    current_product = None
    rank_in_group = 0
    try:
        with input_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            required = {product_column, reactants_column, group_column}
            missing = required.difference(reader.fieldnames or [])
            if missing:
                raise ValueError(f"Missing long-input columns: {sorted(missing)}")
            for row_index, row in enumerate(reader):
                group_id = (row.get(group_column) or "").strip()
                product = (row.get(product_column) or "").strip()
                reactants = (row.get(reactants_column) or "").strip()
                if not group_id or not product:
                    raise ValueError(
                        f"Blank group/product at CSV data row {row_index + 1}"
                    )
                if group_id != current_group:
                    if current_group is not None:
                        closed_groups.add(current_group)
                    if group_id in closed_groups:
                        raise ValueError(
                            "Long input must be contiguous by group; group "
                            f"{group_id!r} reappeared at data row {row_index + 1}"
                        )
                    current_group = group_id
                    current_product = product
                    rank_in_group = 0
                    groups_writer.writerow({"group_id": group_id, "product": product})
                    group_count += 1
                elif product != current_product:
                    raise ValueError(f"Product changed inside group {group_id!r}")
                if not reactants:
                    continue
                rank_in_group += 1
                supplied_rank = (row.get(rank_column) or "").strip()
                rank = int(supplied_rank) if supplied_rank else rank_in_group
                reagent = (
                    (row.get(str(reagent_column)) or "").strip()
                    if reagent_column
                    else ""
                )
                source_type = (
                    (row.get(str(source_column)) or "").strip()
                    if source_column
                    else ""
                ) or default_source
                candidates_writer.writerow(
                    {
                        "candidate_id": candidate_count,
                        "group_id": group_id,
                        "rank": rank,
                        "product": product,
                        "reactants": reactants,
                        "reagent": reagent,
                        "source_type": source_type,
                    }
                )
                candidate_count += 1
    except Exception:
        candidates_writer.close(commit=False)
        groups_writer.close(commit=False)
        raise
    else:
        candidates_writer.close()
        groups_writer.close()
    return {
        "groups": group_count,
        "empty_input_groups": 0,
        "candidates": candidate_count,
    }
