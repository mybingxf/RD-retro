"""Chemical comparison, confidence filtering, and diversity selection."""

from __future__ import annotations

import csv
import heapq
import math
import multiprocessing as mp
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .io import CANDIDATE_FIELDS
from .mapping import MAPPING_FIELDS
from .utils import AtomicDictWriter


FORWARD_PASS_FIELDS = CANDIDATE_FIELDS + ["predicted_product"]


@lru_cache(maxsize=200_000)
def canonicalize_smiles(smiles: str) -> Optional[str]:
    from rdkit import Chem

    if not smiles:
        return None
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return None
    for atom in molecule.GetAtoms():
        atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)


def filter_forward_matches(
    candidates_path: Path, predictions_path: Path, output_path: Path
) -> Dict[str, object]:
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")
    output = AtomicDictWriter(output_path, FORWARD_PASS_FIELDS)
    total = 0
    matches = 0
    invalid_predictions = 0
    invalid_target_groups = set()
    try:
        with candidates_path.open("r", encoding="utf-8-sig", newline="") as candidates_handle, predictions_path.open(
            "r", encoding="utf-8-sig", newline=""
        ) as predictions_handle:
            candidates = csv.DictReader(candidates_handle)
            predictions = csv.DictReader(predictions_handle)
            for candidate, prediction in zip(candidates, predictions):
                candidate_id = int(candidate["candidate_id"])
                if int(prediction["candidate_id"]) != candidate_id:
                    raise ValueError("Candidate/prediction IDs are not aligned")
                target = canonicalize_smiles(candidate["product"])
                predicted_text = prediction["predicted_product"]
                predicted = canonicalize_smiles(predicted_text)
                if target is None:
                    invalid_target_groups.add(candidate["group_id"])
                if predicted is None:
                    invalid_predictions += 1
                if target is not None and predicted == target:
                    row = dict(candidate)
                    row["predicted_product"] = predicted_text
                    output.writerow(row)
                    matches += 1
                total += 1
            if next(candidates, None) is not None or next(predictions, None) is not None:
                raise ValueError("Candidate and prediction row counts differ")
    except Exception:
        output.close(commit=False)
        raise
    else:
        output.close()
    return {
        "predictions_compared": total,
        "forward_matches": matches,
        "forward_match_rate": matches / total if total else None,
        "invalid_forward_predictions": invalid_predictions,
        "invalid_target_groups": len(invalid_target_groups),
    }


def threshold_mapping_confidence(
    mapped_path: Path, output_path: Path, threshold: float
) -> Dict[str, object]:
    output = AtomicDictWriter(output_path, MAPPING_FIELDS)
    total = 0
    kept = 0
    mapping_failed = 0
    below_threshold = 0
    groups_before = set()
    groups_after = set()
    try:
        with mapped_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                total += 1
                group_id = row["group_id"]
                groups_before.add(group_id)
                try:
                    confidence = float(row["mapping_confidence"])
                except (TypeError, ValueError):
                    confidence = float("nan")
                success = row["mapping_success"].lower() in {"1", "true", "yes"}
                if not success or not math.isfinite(confidence):
                    mapping_failed += 1
                    continue
                if confidence < threshold:
                    below_threshold += 1
                    continue
                output.writerow(row)
                kept += 1
                groups_after.add(group_id)
    except Exception:
        output.close(commit=False)
        raise
    else:
        output.close()
    return {
        "mapped_rows": total,
        "mapping_failed": mapping_failed,
        "below_confidence_threshold": below_threshold,
        "confidence_threshold_inclusive": threshold,
        "confidence_filtered_rows": kept,
        "groups_before_confidence_filter": len(groups_before),
        "groups_after_confidence_filter": len(groups_after),
        "groups_emptied_by_confidence_filter": len(groups_before - groups_after),
    }


_MORGAN_GENERATOR = None
_FP_BITS = 2048
_INVALID_POLICY = "error"
_REMOVAL_STRATEGY = "confidence"
_ORIGINAL_SOURCE_TYPES = frozenset({"training", "original"})


def _init_diversity_worker(
    radius: int,
    bits: int,
    invalid_policy: str,
    removal_strategy: str = "confidence",
    original_source_types: Sequence[str] = ("training", "original"),
) -> None:
    global _MORGAN_GENERATOR, _FP_BITS, _INVALID_POLICY
    global _REMOVAL_STRATEGY, _ORIGINAL_SOURCE_TYPES
    from rdkit import RDLogger
    from rdkit.Chem import rdFingerprintGenerator

    RDLogger.DisableLog("rdApp.*")
    _FP_BITS = bits
    _INVALID_POLICY = invalid_policy
    _REMOVAL_STRATEGY = removal_strategy
    _ORIGINAL_SOURCE_TYPES = frozenset(
        str(value).strip().casefold() for value in original_source_types
    )
    _MORGAN_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(
        radius=radius, fpSize=bits
    )


def _combined_fingerprint(reactants: str):
    from rdkit import Chem, DataStructs

    fingerprint = DataStructs.ExplicitBitVect(_FP_BITS)
    valid_components = 0
    invalid_components = 0
    for text in reactants.split("."):
        text = text.strip()
        if not text:
            continue
        molecule = Chem.MolFromSmiles(text)
        if molecule is None:
            invalid_components += 1
            if _INVALID_POLICY == "error":
                raise ValueError(f"Invalid reactant component: {text}")
            continue
        for atom in molecule.GetAtoms():
            atom.SetAtomMapNum(0)
        fingerprint |= _MORGAN_GENERATOR.GetFingerprint(molecule)
        valid_components += 1
    if valid_components == 0 and _INVALID_POLICY == "error":
        raise ValueError(f"No valid reactant component: {reactants}")
    return fingerprint, valid_components, invalid_components


def _mean_similarity(
    candidate_index: int,
    alive: Sequence[bool],
    similarity_matrix: Sequence[Sequence[float]],
) -> float:
    peers = [
        index
        for index, is_alive in enumerate(alive)
        if is_alive and index != candidate_index
    ]
    if not peers:
        return 0.0
    return sum(similarity_matrix[candidate_index][index] for index in peers) / len(
        peers
    )


def _choose_removal(
    left: int,
    right: int,
    candidates: Sequence[Tuple[int, int, float, str, str]],
    alive: Sequence[bool],
    similarity_matrix: Sequence[Sequence[float]],
    strategy: str,
    original_source_types: Sequence[str],
) -> Tuple[int, str]:
    """Choose one member of the current maximum-similarity pair to remove."""

    left_candidate = candidates[left]
    right_candidate = candidates[right]
    original_types = {
        str(value).strip().casefold() for value in original_source_types
    }

    if strategy == "source_then_mean_similarity":
        left_is_original = left_candidate[4].strip().casefold() in original_types
        right_is_original = right_candidate[4].strip().casefold() in original_types
        if left_is_original != right_is_original:
            return (right if left_is_original else left), "source_priority"

    if strategy in {"source_then_mean_similarity", "mean_similarity"}:
        left_mean = _mean_similarity(left, alive, similarity_matrix)
        right_mean = _mean_similarity(right, alive, similarity_matrix)
        if left_mean != right_mean:
            return (left if left_mean > right_mean else right), "mean_similarity"
        # The manuscript does not define exact ties. Keep the earlier original
        # rank, then lower candidate ID, without introducing mapping confidence.
        left_tie_key = (left_candidate[1], left_candidate[0])
        right_tie_key = (right_candidate[1], right_candidate[0])
        return (left if left_tie_key > right_tie_key else right), "mean_similarity"

    if strategy != "confidence":
        raise ValueError(f"Unsupported diversity removal strategy: {strategy}")
    left_key = (left_candidate[2], -left_candidate[1], -left_candidate[0])
    right_key = (right_candidate[2], -right_candidate[1], -right_candidate[0])
    return (right if left_key >= right_key else left), "confidence"


def _prune_group(task):
    from rdkit import DataStructs

    group_id, product, candidates, maximum = task
    fingerprints = []
    valid_components = 0
    invalid_components = 0
    empty_fingerprints = 0
    for candidate in candidates:
        fingerprint, valid, invalid = _combined_fingerprint(candidate[3])
        fingerprints.append(fingerprint)
        valid_components += valid
        invalid_components += invalid
        empty_fingerprints += valid == 0

    pairs = []
    similarity_matrix = [
        [0.0 for _ in candidates] for _ in candidates
    ]
    for left in range(len(candidates) - 1):
        similarities = DataStructs.BulkTanimotoSimilarity(
            fingerprints[left], fingerprints[left + 1 :]
        )
        for offset, similarity in enumerate(similarities):
            right = left + offset + 1
            value = float(similarity)
            similarity_matrix[left][right] = value
            similarity_matrix[right][left] = value
            pairs.append((-value, left, right))
    heapq.heapify(pairs)
    initial_max = -pairs[0][0] if pairs else None
    alive = [True] * len(candidates)
    alive_count = len(candidates)
    removals_by_rule = {
        "confidence": 0,
        "source_priority": 0,
        "mean_similarity": 0,
    }
    while alive_count > maximum:
        while pairs:
            _, left, right = heapq.heappop(pairs)
            if alive[left] and alive[right]:
                break
        else:
            raise RuntimeError(f"No live pair remains for group {group_id}")
        # tuple: candidate_id, rank, confidence, reactants, source_type
        removed, rule = _choose_removal(
            left,
            right,
            candidates,
            alive,
            similarity_matrix,
            _REMOVAL_STRATEGY,
            _ORIGINAL_SOURCE_TYPES,
        )
        removals_by_rule[rule] += 1
        alive[removed] = False
        alive_count -= 1

    final_max = None
    while pairs:
        negative_similarity, left, right = pairs[0]
        if alive[left] and alive[right]:
            final_max = -negative_similarity
            break
        heapq.heappop(pairs)
    selected = [
        candidate[0]
        for index, candidate in enumerate(candidates)
        if alive[index]
    ]
    return {
        "group_id": group_id,
        "product": product,
        "candidates_before_diversity": len(candidates),
        "candidates_after_diversity": len(selected),
        "candidates_removed_diversity": len(candidates) - len(selected),
        "initial_max_pair_similarity": initial_max,
        "final_max_pair_similarity": final_max,
        "valid_reactant_components": valid_components,
        "invalid_reactant_components": invalid_components,
        "empty_fingerprints": empty_fingerprints,
        "removals_by_confidence": removals_by_rule["confidence"],
        "removals_by_source_priority": removals_by_rule["source_priority"],
        "removals_by_mean_similarity": removals_by_rule["mean_similarity"],
        "selected_ids": selected,
    }


def _iter_group_tasks(path: Path, maximum: int):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        group_id = None
        product = None
        candidates = []
        closed = set()
        for row in reader:
            next_group = row["group_id"]
            if group_id is not None and next_group != group_id:
                closed.add(group_id)
                yield group_id, product, candidates, maximum
                candidates = []
            if next_group in closed:
                raise ValueError(f"Non-contiguous group in filtered data: {next_group}")
            group_id = next_group
            product = row["product"]
            candidates.append(
                (
                    int(row["candidate_id"]),
                    int(row["rank"]),
                    float(row["mapping_confidence"]),
                    row["reactants"],
                    row.get("source_type", "generated"),
                )
            )
        if group_id is not None:
            yield group_id, product, candidates, maximum


GROUP_STATS_FIELDS = [
    "group_id",
    "product",
    "candidates_before_diversity",
    "candidates_after_diversity",
    "candidates_removed_diversity",
    "initial_max_pair_similarity",
    "final_max_pair_similarity",
    "valid_reactant_components",
    "invalid_reactant_components",
    "empty_fingerprints",
    "removals_by_confidence",
    "removals_by_source_priority",
    "removals_by_mean_similarity",
]


def diversity_prune(
    confidence_path: Path,
    groups_path: Path,
    final_long_path: Path,
    final_wide_path: Path,
    group_stats_path: Path,
    config: Dict[str, object],
) -> Dict[str, object]:
    maximum = int(config["max_candidates_per_product"])
    workers = max(1, int(config.get("workers", 1)))
    chunksize = max(1, int(config.get("chunksize", 8)))
    radius = int(config.get("morgan_radius", 2))
    bits = int(config.get("morgan_bits", 2048))
    invalid_policy = str(config.get("invalid_smiles", "error"))
    removal_strategy = str(config.get("removal_strategy", "confidence"))
    original_source_types = [
        str(value).strip().casefold()
        for value in config.get("original_source_types", ["training", "original"])
    ]
    group_output = AtomicDictWriter(group_stats_path, GROUP_STATS_FIELDS)
    selected_ids = set()
    before = after = removed = group_count = pruned_groups = 0
    invalid_components = empty_fingerprints = 0
    initial_sum = final_sum = 0.0
    initial_count = final_count = 0
    confidence_removals = source_priority_removals = mean_similarity_removals = 0

    context = mp.get_context("spawn")
    pool = context.Pool(
        processes=workers,
        initializer=_init_diversity_worker,
        initargs=(
            radius,
            bits,
            invalid_policy,
            removal_strategy,
            original_source_types,
        ),
    )
    try:
        results = pool.imap(
            _prune_group,
            _iter_group_tasks(confidence_path, maximum),
            chunksize=chunksize,
        )
        for result in results:
            selected_ids.update(result.pop("selected_ids"))
            group_output.writerow(result)
            group_count += 1
            before += result["candidates_before_diversity"]
            after += result["candidates_after_diversity"]
            removed += result["candidates_removed_diversity"]
            pruned_groups += result["candidates_removed_diversity"] > 0
            invalid_components += result["invalid_reactant_components"]
            empty_fingerprints += result["empty_fingerprints"]
            confidence_removals += result["removals_by_confidence"]
            source_priority_removals += result["removals_by_source_priority"]
            mean_similarity_removals += result["removals_by_mean_similarity"]
            if result["initial_max_pair_similarity"] is not None:
                initial_sum += result["initial_max_pair_similarity"]
                initial_count += 1
            if result["final_max_pair_similarity"] is not None:
                final_sum += result["final_max_pair_similarity"]
                final_count += 1
    except Exception:
        pool.terminate()
        group_output.close(commit=False)
        raise
    else:
        pool.close()
        group_output.close()
    finally:
        pool.join()

    if len(selected_ids) != after:
        raise ValueError("Selected candidate IDs are not unique")

    final_output = AtomicDictWriter(final_long_path, MAPPING_FIELDS)
    selected_by_group: Dict[str, List[Tuple[int, str]]] = {}
    written = 0
    try:
        with confidence_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                if int(row["candidate_id"]) not in selected_ids:
                    continue
                final_output.writerow(row)
                selected_by_group.setdefault(row["group_id"], []).append(
                    (int(row["rank"]), row["reactants"])
                )
                written += 1
    except Exception:
        final_output.close(commit=False)
        raise
    else:
        final_output.close()
    if written != after:
        raise ValueError(f"Final long output has {written} rows, expected {after}")

    wide_fields = ["product"] + [
        f"reactants{i}" for i in range(1, maximum + 1)
    ]
    wide_output = AtomicDictWriter(final_wide_path, wide_fields)
    total_groups = nonempty_groups = wide_candidates = 0
    try:
        with groups_path.open("r", encoding="utf-8-sig", newline="") as handle:
            for group in csv.DictReader(handle):
                candidates = sorted(selected_by_group.get(group["group_id"], []))
                if len(candidates) > maximum:
                    raise ValueError(f"Group {group['group_id']} exceeds final maximum")
                row = {"product": group["product"]}
                for index, (_, reactants) in enumerate(candidates, start=1):
                    row[f"reactants{index}"] = reactants
                wide_output.writerow(row)
                total_groups += 1
                nonempty_groups += bool(candidates)
                wide_candidates += len(candidates)
    except Exception:
        wide_output.close(commit=False)
        raise
    else:
        wide_output.close()
    if wide_candidates != after:
        raise ValueError("Final long/wide candidate counts differ")

    return {
        "groups_after_confidence_filter": group_count,
        "groups_pruned_for_diversity": pruned_groups,
        "candidates_before_diversity": before,
        "candidates_removed_for_diversity": removed,
        "final_candidates": after,
        "total_product_groups": total_groups,
        "final_nonempty_product_groups": nonempty_groups,
        "final_empty_product_groups": total_groups - nonempty_groups,
        "mean_final_candidates_per_all_product_groups": (
            after / total_groups if total_groups else None
        ),
        "mean_final_candidates_per_nonempty_product_group": (
            after / nonempty_groups if nonempty_groups else None
        ),
        "mean_initial_group_max_similarity": (
            initial_sum / initial_count if initial_count else None
        ),
        "mean_final_group_max_similarity": (
            final_sum / final_count if final_count else None
        ),
        "invalid_reactant_components": invalid_components,
        "empty_fingerprints": empty_fingerprints,
        "morgan_radius": radius,
        "morgan_bits": bits,
        "multi_component_aggregation": "bitwise_OR",
        "similarity": "Tanimoto",
        "max_candidates_per_product": maximum,
        "removal_strategy": removal_strategy,
        "original_source_types": original_source_types,
        "removals_by_confidence": confidence_removals,
        "removals_by_source_priority": source_priority_removals,
        "removals_by_mean_similarity": mean_similarity_removals,
        "tie_break": (
            "higher confidence, then lower original rank, then lower candidate_id"
            if removal_strategy == "confidence"
            else "lower original rank, then lower candidate_id when mean similarity ties"
        ),
    }
