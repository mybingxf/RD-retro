"""Pluggable forward-reaction prediction backends."""

from __future__ import annotations

import csv
import multiprocessing as mp
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from .utils import (
    AtomicDictWriter,
    merge_sorted_csvs,
    reset_directory,
    split_csv_round_robin,
)


PREDICTION_FIELDS = ["candidate_id", "predicted_product"]


def _clean_prediction(value: str) -> str:
    return value.replace(" ", "").rstrip(".")


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    import torch

    return "cuda:0" if torch.cuda.is_available() else "cpu"


class HuggingFaceSeq2SeqForward:
    def __init__(self, config: Dict[str, object], device: str):
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self.torch = torch
        self.device = _resolve_device(device)
        model_name = str(config["model_name_or_path"])
        revision = config.get("revision") or None
        local_only = bool(config.get("local_files_only", False))
        tokenizer_kwargs = {
            "revision": revision,
            "local_files_only": local_only,
        }
        model_kwargs = dict(tokenizer_kwargs)
        dtype_name = str(config.get("dtype", "auto"))
        if dtype_name != "auto":
            try:
                model_kwargs["torch_dtype"] = getattr(torch, dtype_name)
            except AttributeError as error:
                raise ValueError(f"Unsupported torch dtype: {dtype_name}") from error
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name, **model_kwargs)
        self.model.to(self.device)
        self.model.eval()
        self.input_max_length = int(config.get("input_max_length", 150))
        self.max_new_tokens = int(config.get("max_new_tokens", 128))
        self.num_beams = int(config.get("num_beams", 1))
        self.prompt_template = str(
            config.get("prompt_template", "REACTANT:{reactants}REAGENT:{reagent}")
        )

    def predict(self, rows: Sequence[Dict[str, str]]) -> List[str]:
        prompts = [
            self.prompt_template.format(
                reactants=row["reactants"], reagent=row.get("reagent", "")
            )
            for row in rows
        ]
        encoded = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=self.input_max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        with self.torch.inference_mode():
            sequences = self.model.generate(
                **encoded,
                num_beams=self.num_beams,
                num_return_sequences=1,
                max_new_tokens=self.max_new_tokens,
            )
        return [
            _clean_prediction(value)
            for value in self.tokenizer.batch_decode(
                sequences, skip_special_tokens=True
            )
        ]


class ChemformerForward:
    """Adapter for an external MolecularAI/Chemformer checkout and checkpoint."""

    def __init__(self, config: Dict[str, object], device: str):
        root = config.get("chemformer_root")
        checkpoint = config.get("chemformer_checkpoint")
        vocabulary = config.get("chemformer_vocabulary")
        if not root or not checkpoint or not vocabulary:
            raise ValueError(
                "Chemformer requires forward.chemformer_root, "
                "chemformer_checkpoint, and chemformer_vocabulary"
            )
        root_path = str(Path(str(root)).resolve())
        if root_path not in sys.path:
            sys.path.insert(0, root_path)

        import numpy as np
        import torch

        import molbart.utils.data_utils  # noqa: F401
        from molbart.data.util import BatchEncoder
        from molbart.models import BARTModel
        from molbart.utils.samplers import BeamSearchSampler
        from molbart.utils.scores import ScoreCollection
        from molbart.utils.tokenizers import ChemformerTokenizer

        self.np = np
        self.torch = torch
        self.device = _resolve_device(device)
        if not self.device.startswith("cuda"):
            raise ValueError("This Chemformer adapter currently requires a CUDA device")
        tokenizer = ChemformerTokenizer(filename=str(Path(str(vocabulary)).resolve()))
        sampler = BeamSearchSampler(
            tokenizer,
            ScoreCollection(),
            int(config.get("chemformer_max_length", 512)),
            device=self.device,
            data_device=self.device,
            sample_unique=False,
        )
        self.model = BARTModel.load_from_checkpoint(
            str(Path(str(checkpoint)).resolve()), decode_sampler=sampler
        )
        self.model.num_beams = 1
        self.model.eval()
        self.model.to(self.device)
        self.encoder = BatchEncoder(
            tokenizer=tokenizer,
            masker=None,
            max_seq_len=int(config.get("chemformer_max_length", 512)),
        )

    def predict(self, rows: Sequence[Dict[str, str]]) -> List[str]:
        try:
            encoder_ids, encoder_mask = self.encoder(
                [row["reactants"] for row in rows]
            )
            batch = {
                "encoder_input": encoder_ids.to(self.device, non_blocking=True),
                "encoder_pad_mask": encoder_mask.to(self.device, non_blocking=True),
            }
            with self.torch.inference_mode():
                sampled, _ = self.model.sample_molecules(batch, sampling_alg="beam")
            sampled = self.np.asarray(sampled)
            values = sampled.tolist() if sampled.ndim == 1 else sampled[:, 0].tolist()
            return [_clean_prediction(str(value)) for value in values]
        except RuntimeError as error:
            if "out of memory" not in str(error).lower() or len(rows) <= 1:
                raise
            self.torch.cuda.empty_cache()
            middle = len(rows) // 2
            return self.predict(rows[:middle]) + self.predict(rows[middle:])


def _make_backend(config: Dict[str, object], device: str):
    backend = str(config["backend"])
    if backend in {"reactiont5v2", "hf_seq2seq"}:
        return HuggingFaceSeq2SeqForward(config, device)
    if backend == "chemformer":
        return ChemformerForward(config, device)
    raise ValueError(f"Backend {backend!r} cannot run as a model worker")


def _forward_worker(
    input_path: str, output_path: str, config: Dict[str, object], device: str
) -> None:
    backend = _make_backend(config, device)
    output = AtomicDictWriter(Path(output_path), PREDICTION_FIELDS)
    batch_size = int(config.get("batch_size", 16))
    batch: List[Dict[str, str]] = []
    try:
        with Path(input_path).open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                batch.append(row)
                if len(batch) >= batch_size:
                    predictions = backend.predict(batch)
                    if len(predictions) != len(batch):
                        raise ValueError("Forward backend returned the wrong batch size")
                    output.writerows(
                        {
                            "candidate_id": item["candidate_id"],
                            "predicted_product": prediction,
                        }
                        for item, prediction in zip(batch, predictions)
                    )
                    batch = []
            if batch:
                predictions = backend.predict(batch)
                if len(predictions) != len(batch):
                    raise ValueError("Forward backend returned the wrong batch size")
                output.writerows(
                    {
                        "candidate_id": item["candidate_id"],
                        "predicted_product": prediction,
                    }
                    for item, prediction in zip(batch, predictions)
                )
    except Exception:
        output.close(commit=False)
        raise
    else:
        output.close()


def _copy_precomputed(
    candidates_path: Path, predictions_path: Path, output_path: Path, config: Dict[str, object]
) -> int:
    id_column = str(config.get("prediction_id_column", "candidate_id"))
    prediction_column = str(config.get("prediction_column", "predicted_product"))
    output = AtomicDictWriter(output_path, PREDICTION_FIELDS)
    count = 0
    try:
        with candidates_path.open("r", encoding="utf-8-sig", newline="") as candidates_handle, predictions_path.open(
            "r", encoding="utf-8-sig", newline=""
        ) as predictions_handle:
            candidates = csv.DictReader(candidates_handle)
            predictions = csv.DictReader(predictions_handle)
            if id_column not in (predictions.fieldnames or []) or prediction_column not in (
                predictions.fieldnames or []
            ):
                raise ValueError("Precomputed forward file is missing configured columns")
            prediction = next(predictions, None)
            for candidate in candidates:
                candidate_id = int(candidate["candidate_id"])
                while prediction is not None and int(prediction[id_column]) < candidate_id:
                    prediction = next(predictions, None)
                if prediction is None or int(prediction[id_column]) != candidate_id:
                    raise ValueError(
                        f"No precomputed forward prediction for candidate_id={candidate_id}"
                    )
                output.writerow(
                    {
                        "candidate_id": candidate_id,
                        "predicted_product": prediction[prediction_column],
                    }
                )
                count += 1
                prediction = next(predictions, None)
    except Exception:
        output.close(commit=False)
        raise
    else:
        output.close()
    return count


def run_forward(
    candidates_path: Path,
    output_path: Path,
    work_dir: Path,
    config: Dict[str, object],
) -> Dict[str, object]:
    backend = str(config["backend"])
    if backend == "precomputed":
        prediction_file = config.get("prediction_file")
        if not prediction_file:
            raise ValueError("precomputed forward backend requires prediction_file")
        count = _copy_precomputed(
            candidates_path, Path(str(prediction_file)), output_path, config
        )
        return {"backend": backend, "predictions": count, "devices": []}

    devices = [str(value) for value in config.get("devices", ["auto"])]
    input_dir = work_dir / "inputs"
    output_dir = work_dir / "outputs"
    reset_directory(input_dir)
    reset_directory(output_dir)
    shards = split_csv_round_robin(candidates_path, input_dir, len(devices))
    output_shards = [output_dir / path.name for path in shards]
    context = mp.get_context("spawn")
    processes = []
    for shard, destination, device in zip(shards, output_shards, devices):
        process = context.Process(
            target=_forward_worker,
            args=(str(shard), str(destination), config, device),
        )
        process.start()
        processes.append(process)
    failures = []
    for process in processes:
        process.join()
        if process.exitcode != 0:
            failures.append(process.exitcode)
    if failures:
        raise RuntimeError(f"Forward prediction workers failed: {failures}")
    count = merge_sorted_csvs(output_shards, output_path)
    return {"backend": backend, "predictions": count, "devices": devices}

