"""
Wrapper for SingleStepModelZoo for Chemformer (https://github.com/MolecularAI/Chemformer).

This wrapper supports both the legacy Models Matter Chemformer bundle and the
newer Chemformer codebase used in this workspace.
"""

import argparse
import os
import sys
import uuid
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from rdkit import Chem
from ssbenchmark.ssmodels.base_model import Registries, SSMethod
from ssbenchmark.utils import canonicalize_smiles, split_reactions


def _canonicalize_reactants_smiles(reactants_str: str) -> str:
    parts = [part for part in str(reactants_str).split(".") if part]
    canonical_parts: List[str] = []
    for part in parts:
        mol = Chem.MolFromSmiles(part)
        if mol is None:
            canonical_parts.append(part)
            continue
        canonical_parts.append(Chem.MolToSmiles(mol))
    canonical_parts.sort()
    return ".".join(canonical_parts)


@Registries.ModelChoice.register(key="chemformer")
@Registries.ModelChoice.register(key="chemformer_new")
@Registries.ModelChoice.register(key="chemformer_orig")
class model_chemformer(SSMethod):
    def __init__(self, module_path=None):
        self.model_name = "Chemformer"
        self._module_path = module_path
        self._backend = "legacy"
        self._device = "cpu"
        self._batch_size = 64
        self._num_beams = 50
        self._num_workers = 0
        self._model_type = "bart"
        self._max_seq_len = None
        if module_path is not None:
            sys.path.insert(0, os.path.abspath(module_path))

    def preprocess(self, data, reaction_col):
        data["reactants"], data["spectators"], data["products"] = split_reactions(
            data[reaction_col].tolist()
        )
        data["products"] = data.products.apply(canonicalize_smiles)
        data["reactants"] = data.reactants.apply(canonicalize_smiles)
        data["products_mol"] = [
            Chem.MolFromSmiles(m) if m is not None else None for m in data["products"]
        ]
        data["reactants_mol"] = [
            Chem.MolFromSmiles(m) if m is not None else None for m in data["reactants"]
        ]
        data["products_mol"] = [
            Chem.MolToSmiles(m) if m is not None else None for m in data["products_mol"]
        ]
        data["reactants_mol"] = [
            Chem.MolToSmiles(m) if m is not None else None
            for m in data["reactants_mol"]
        ]
        data = data.dropna()
        data = data.rename(columns={"class": "reaction_type", "split": "set"})
        if "reaction_type" not in data.columns:
            data["reaction_type"] = np.nan
        return data[["reactants_mol", "products_mol", "reaction_type", "set"]]

    def process_input(self, data, reaction_col):
        print("preprocess input")
        return self.preprocess(data, reaction_col)

    def preprocess_store(self, data, preprocess_root, instance_name):
        print("preprocess store")
        oroot = os.path.abspath(preprocess_root)
        if not self.check_path(oroot):
            self.make_path(oroot)
        data = data.reset_index(drop=True)
        data.to_pickle(os.path.join(oroot, f"{instance_name}.pickle"))
        data = data[data["set"] == "test"]
        data.to_csv(
            os.path.join(oroot, f"test_smiles_{instance_name}.txt"),
            index=False,
            header=False,
        )

    def process_output(self, data_root, instance_name, k):
        opath = data_root
        if not self.check_path(opath):
            raise FileNotFoundError(f"File not found at {opath}")
        data = pd.read_pickle(opath)
        columns = [f"prediction_{i}" for i in range(k)]
        data = data[columns]
        data = data.values.tolist()
        return [[canonicalize_smiles(p) for p in pred] for pred in data]

    def model_setup(self, use_gpu=False, **kwargs):
        self._device = kwargs.get("gpu", "cuda:0") if use_gpu else "cpu"
        self._batch_size = int(kwargs.get("batch_size", 64))
        self._num_beams = int(kwargs.get("num_beams", kwargs.get("n_beams", 50)))
        self._num_workers = int(kwargs.get("num_workers", 0))
        self._model_type = kwargs.get("model_type", "bart")

        if self._has_current_backend():
            self._setup_current_backend(use_gpu, **kwargs)
            return
        self._setup_legacy_backend(use_gpu, **kwargs)

    def _has_current_backend(self) -> bool:
        try:
            import molbart.utils.data_utils as util  # noqa: F401
            from molbart.models import Chemformer  # noqa: F401
            from molbart.data.datamodules import SynthesisDataModule  # noqa: F401
            from omegaconf import OmegaConf  # noqa: F401
        except ModuleNotFoundError:
            return False
        except ImportError:
            return False
        return True

    def _setup_current_backend(self, use_gpu=False, **kwargs):
        from omegaconf import OmegaConf
        import molbart.utils.data_utils as util
        from molbart.models import Chemformer

        self._backend = "current"
        self._current_util = util
        self._max_seq_len = kwargs.get("max_seq_len", util.DEFAULT_MAX_SEQ_LEN)

        data_device = kwargs.get("data_device", self._device if use_gpu else "cpu")
        if use_gpu and str(data_device).startswith("cuda"):
            data_device = self._device
        cfg = OmegaConf.create(
            {
                "seed": kwargs.get("seed", 1),
                "n_gpus": 1 if use_gpu else 0,
                "data_path": kwargs.get("data_path", ""),
                "dataset_part": kwargs.get("dataset_part", "full"),
                "vocabulary_path": kwargs["vocab_path"],
                "data_device": data_device,
                "task": kwargs.get("task", "backward_prediction"),
                "batch_size": self._batch_size,
                "model_path": kwargs["model_path"],
                "model_type": self._model_type,
                "n_beams": self._num_beams,
                "n_unique_beams": kwargs.get("n_unique_beams"),
                "train_mode": kwargs.get("train_mode", "eval"),
                "device": self._device,
                "datamodule": None,
                "max_seq_len": self._max_seq_len,
            }
        )
        self.model = Chemformer(cfg)

    def _setup_legacy_backend(self, use_gpu=False, **kwargs):
        import molbart.util as util
        from molbart.decoder import DecodeSampler

        self._backend = "legacy"
        parser = argparse.ArgumentParser()
        parser.add_argument("--reactants_path", type=str)
        parser.add_argument("--model_path", type=str)
        parser.add_argument("--products_path", type=str)
        parser.add_argument("--vocab_path", type=str, default=util.DEFAULT_VOCAB_PATH)
        parser.add_argument(
            "--chem_token_start_idx", type=int, default=util.DEFAULT_CHEM_TOKEN_START
        )
        parser.add_argument("--batch_size", type=int, default=self._batch_size)
        parser.add_argument("--num_beams", type=int, default=self._num_beams)

        args = parser.parse_args(
            [
                f"--reactants_path={kwargs.get('reactants_path', None)}",
                f"--model_path={kwargs.get('model_path', None)}",
                f"--products_path={kwargs.get('products_path', None)}",
                f"--vocab_path={kwargs.get('vocab_path', util.DEFAULT_VOCAB_PATH)}",
                f"--chem_token_start_idx={kwargs.get('chem_token_start_idx', util.DEFAULT_CHEM_TOKEN_START)}",
                f"--batch_size={self._batch_size}",
                f"--num_beams={self._num_beams}",
            ]
        )

        self.tokeniser = util.load_tokeniser(args.vocab_path, args.chem_token_start_idx)
        sampler = DecodeSampler(self.tokeniser, util.DEFAULT_MAX_SEQ_LEN)
        self.model = util.load_bart(args, sampler)
        self.model.num_beams = args.num_beams
        sampler.max_seq_len = self.model.max_seq_len
        self.model_args = args
        self._max_seq_len = self.model.max_seq_len

    def build_datamodule(self, args, dataset, tokeniser, max_seq_len):
        from molbart.data.datamodules import FineTuneReactionDataModule

        test_idxs = range(len(dataset))
        dm = FineTuneReactionDataModule(
            dataset,
            tokeniser,
            args.batch_size,
            max_seq_len,
            val_idxs=[],
            test_idxs=test_idxs,
        )
        return dm

    def predict(self, model, test_loader):
        model = model.to(self._device)
        model.eval()

        smiles = []
        log_lhs = []
        original_smiles = []

        for batch in test_loader:
            device_batch = {
                key: val.to(self._device) if isinstance(val, torch.Tensor) else val
                for key, val in batch.items()
            }
            with torch.no_grad():
                smiles_batch, log_lhs_batch = model.sample_molecules(
                    device_batch, sampling_alg="beam"
                )

            smiles.extend(smiles_batch)
            log_lhs.extend(log_lhs_batch)
            original_smiles.extend(batch["target_smiles"])

        return smiles, log_lhs, original_smiles

    def _current_model_call(self, X):
        import molbart.utils.data_utils as util  # Ensure current package import order.
        from molbart.models import Chemformer  # noqa: F401
        from molbart.data.datamodules import SynthesisDataModule

        dm = SynthesisDataModule(
            reactants=list(X),
            products=list(X),
            dataset_path="",
            tokenizer=self.model.tokenizer,
            batch_size=self._batch_size,
            max_seq_len=self._max_seq_len,
            augment_prob=0.0,
            reverse=True,
            unified_model=self._model_type == "unified",
            i_chunk=0,
            n_chunks=1,
        )
        self.model.set_datamodule(datamodule=dm)
        self.model.datamodule._num_workers = self._num_workers
        smiles, log_lhs, _ = self.model.predict(dataset="full")

        predictions = [
            [_canonicalize_reactants_smiles(pred) for pred in pred_row]
            for pred_row in smiles
        ]
        probabilities = [
            F.softmax(torch.tensor(row, dtype=torch.float32), dim=0).tolist()
            for row in log_lhs
        ]
        return predictions, probabilities

    def _legacy_model_call(self, X):
        from molbart.data.datasets import ReactionDataset

        dataset = ReactionDataset(X, X)
        dm = self.build_datamodule(
            self.model_args, dataset, self.tokeniser, self.model.max_seq_len
        )
        dm.setup()
        test_loader = dm.test_dataloader()
        smiles, log_lhs, _ = self.predict(self.model, test_loader)
        output = F.softmax(torch.tensor(log_lhs, dtype=torch.float32), dim=1)
        predictions = [
            [_canonicalize_reactants_smiles(pred) for pred in pred_row]
            for pred_row in smiles
        ]
        return predictions, output.tolist()

    def model_call(self, X):
        if self._backend == "current":
            return self._current_model_call(X)
        return self._legacy_model_call(X)


@Registries.ModelChoice.register(key="chemformer_mix")
class model_chemformer_mix(SSMethod):
    def __init__(self, module_path=None):
        self.model_name = "ChemformerMix"
        self._module_path = module_path
        self.reset_call_telemetry()

    def process_input(self, data, reaction_col):
        return model_chemformer(self._module_path).process_input(data, reaction_col)

    def preprocess_store(self, data, preprocess_root, instance_name):
        return model_chemformer(self._module_path).preprocess_store(
            data, preprocess_root, instance_name
        )

    def process_output(self, *args, **kwargs):
        return None

    def model_setup(self, use_gpu=False, **kwargs):
        model_a_cfg = dict(kwargs.pop("model_a"))
        model_b_cfg = dict(kwargs.pop("model_b"))
        self.merge_top_k = int(kwargs.pop("merge_top_k", 50))
        self.output_top_k = int(kwargs.pop("output_top_k", 0))
        self.renormalize_scores = bool(kwargs.pop("renormalize_scores", True))
        assigned_gpu = kwargs.get("gpu")

        if assigned_gpu:
            model_a_cfg.setdefault("gpu", assigned_gpu)
            model_b_cfg.setdefault("gpu", assigned_gpu)

        self.model_a = model_chemformer(self._module_path)
        self.model_b = model_chemformer(self._module_path)
        self.model_a.model_setup(use_gpu, **model_a_cfg)
        self.model_b.model_setup(use_gpu, **model_b_cfg)
        self.reset_call_telemetry()

    def reset_call_telemetry(self):
        self._merge_calls = 0
        self._total_input_candidates = 0
        self._total_unique_candidates = 0
        self._total_output_candidates = 0
        self._total_duplicates_collapsed = 0

    def get_call_telemetry(self):
        if self._merge_calls == 0:
            return {}
        return {
            "mix_merge_calls": self._merge_calls,
            "mix_avg_input_candidates_per_call": self._total_input_candidates / self._merge_calls,
            "mix_avg_unique_candidates_per_call": self._total_unique_candidates / self._merge_calls,
            "mix_avg_output_candidates_per_call": self._total_output_candidates / self._merge_calls,
            "mix_avg_duplicates_collapsed_per_call": self._total_duplicates_collapsed / self._merge_calls,
        }

    def _merge_single(
        self,
        reactants_a: List[str],
        priors_a: List[float],
        reactants_b: List[str],
        priors_b: List[float],
    ) -> Tuple[List[str], List[float]]:
        merged: Dict[str, float] = {}

        for reactants, priors in (
            (reactants_a[: self.merge_top_k], priors_a[: self.merge_top_k]),
            (reactants_b[: self.merge_top_k], priors_b[: self.merge_top_k]),
        ):
            for reactants_str, score in zip(reactants, priors):
                key = _canonicalize_reactants_smiles(reactants_str)
                merged[key] = merged.get(key, 0.0) + float(score)

        ranked = sorted(merged.items(), key=lambda item: item[1], reverse=True)
        if not ranked:
            return [], []

        input_count = len(reactants_a[: self.merge_top_k]) + len(reactants_b[: self.merge_top_k])
        unique_count = len(ranked)

        reactants_out = [item[0] for item in ranked]
        scores_out = [item[1] for item in ranked]

        if self.output_top_k > 0:
            reactants_out = reactants_out[: self.output_top_k]
            scores_out = scores_out[: self.output_top_k]

        if self.renormalize_scores:
            total = float(sum(scores_out))
            if total > 0:
                scores_out = [score / total for score in scores_out]

        output_count = len(reactants_out)
        self._merge_calls += 1
        self._total_input_candidates += input_count
        self._total_unique_candidates += unique_count
        self._total_output_candidates += output_count
        self._total_duplicates_collapsed += max(0, input_count - unique_count)

        return reactants_out, scores_out

    def model_call(self, X):
        reactants_a, priors_a = self.model_a.model_call(X)
        reactants_b, priors_b = self.model_b.model_call(X)

        merged_reactants = []
        merged_priors = []
        for ra, pa, rb, pb in zip(reactants_a, priors_a, reactants_b, priors_b):
            reactants, priors = self._merge_single(list(ra), list(pa), list(rb), list(pb))
            merged_reactants.append(reactants)
            merged_priors.append(priors)
        return merged_reactants, merged_priors


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--module_path", type=str, default=None)
    parser.add_argument("--preprocess_batch", type=str, default=None)
    parser.add_argument("--preprocess_join", type=str, default=None)
    parser.add_argument("--reaction_col", type=str, default="reaction_smiles")
    parser.add_argument("--preprocess_output", type=str, default=None)
    parser.add_argument("--instance_name", type=str, default=None)

    args = parser.parse_args()

    model_instance = model_chemformer(module_path=args.module_path)
    if args.preprocess_batch is not None:
        df = model_instance.read_csv(args.preprocess_batch)
        df = model_instance.preprocess(df, args.reaction_col)
        model_instance.write_pickle(df, args.preprocess_output, str(uuid.uuid4()))

    if args.preprocess_join is not None:
        df = model_instance.gather_batches(args.preprocess_join, "pickle")
        model_instance.preprocess_store(df, args.preprocess_output, args.instance_name)
