"""Small file and provenance utilities."""

from __future__ import annotations

import csv
import hashlib
import heapq
import importlib.metadata
import json
import os
import platform
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


class AtomicDictWriter:
    def __init__(self, path: Path, fieldnames: Sequence[str]):
        self.path = path
        self.tmp_path = Path(str(path) + ".tmp")
        ensure_parent(path)
        self.handle = self.tmp_path.open("w", encoding="utf-8", newline="")
        self.writer = csv.DictWriter(
            self.handle,
            fieldnames=list(fieldnames),
            extrasaction="ignore",
            lineterminator="\n",
        )
        self.writer.writeheader()

    def writerow(self, row: Dict[str, object]) -> None:
        self.writer.writerow(row)

    def writerows(self, rows: Iterable[Dict[str, object]]) -> None:
        self.writer.writerows(rows)

    def close(self, commit: bool = True) -> None:
        if self.handle.closed:
            return
        self.handle.close()
        if commit:
            os.replace(self.tmp_path, self.path)
        elif self.tmp_path.exists():
            self.tmp_path.unlink()


def write_json(path: Path, value: object) -> None:
    ensure_parent(path)
    tmp_path = Path(str(path) + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(tmp_path, path)


def sha256sum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def count_csv_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        next(reader, None)
        return sum(1 for _ in reader)


def split_csv_round_robin(input_path: Path, output_dir: Path, n_shards: int) -> List[Path]:
    """Stream a CSV into sorted-by-candidate-ID round-robin shards."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = [output_dir / f"shard_{i:03d}.csv" for i in range(n_shards)]
    writers = []
    handles = []
    tmp_paths = [Path(str(path) + ".tmp") for path in paths]
    try:
        with input_path.open("r", encoding="utf-8-sig", newline="") as source:
            reader = csv.DictReader(source)
            if not reader.fieldnames:
                raise ValueError(f"No header in {input_path}")
            for tmp_path in tmp_paths:
                handle = tmp_path.open("w", encoding="utf-8", newline="")
                writer = csv.DictWriter(
                    handle, fieldnames=reader.fieldnames, lineterminator="\n"
                )
                writer.writeheader()
                handles.append(handle)
                writers.append(writer)
            for row_index, row in enumerate(reader):
                writers[row_index % n_shards].writerow(row)
    finally:
        for handle in handles:
            handle.close()
    for tmp_path, path in zip(tmp_paths, paths):
        os.replace(tmp_path, path)
    return paths


def merge_sorted_csvs(paths: Sequence[Path], output_path: Path, key: str = "candidate_id") -> int:
    """K-way merge CSV shards whose integer key is increasing within each shard."""
    readers = []
    handles = []
    heap = []
    fieldnames = None
    try:
        for shard_index, path in enumerate(paths):
            handle = path.open("r", encoding="utf-8-sig", newline="")
            reader = csv.DictReader(handle)
            if not reader.fieldnames:
                handle.close()
                continue
            if fieldnames is None:
                fieldnames = reader.fieldnames
            elif reader.fieldnames != fieldnames:
                raise ValueError("Mapped/prediction shard schemas do not match")
            local_index = len(readers)
            readers.append(reader)
            handles.append(handle)
            row = next(reader, None)
            if row is not None:
                heapq.heappush(heap, (int(row[key]), local_index, row))
        if fieldnames is None:
            raise ValueError("No shard headers found")
        output = AtomicDictWriter(output_path, fieldnames)
        count = 0
        previous = -1
        try:
            while heap:
                numeric_key, local_index, row = heapq.heappop(heap)
                if numeric_key <= previous:
                    raise ValueError(f"Merged {key} is not strictly increasing")
                previous = numeric_key
                output.writerow(row)
                count += 1
                next_row = next(readers[local_index], None)
                if next_row is not None:
                    heapq.heappush(
                        heap, (int(next_row[key]), local_index, next_row)
                    )
        except Exception:
            output.close(commit=False)
            raise
        else:
            output.close()
        return count
    finally:
        for handle in handles:
            handle.close()


def reset_directory(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def package_versions() -> Dict[str, str]:
    result = {
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    for name in (
        "numpy",
        "PyYAML",
        "tqdm",
        "rdkit-pypi",
        "torch",
        "transformers",
        "tokenizers",
        "sentencepiece",
        "rxnmapper",
    ):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = "not installed"
    try:
        from rdkit import rdBase

        result["rdkit_runtime"] = rdBase.rdkitVersion
    except ImportError:
        result["rdkit_runtime"] = "not installed"
    return result
