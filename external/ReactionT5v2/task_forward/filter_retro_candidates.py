import argparse
import os
import sys
from collections import defaultdict

import pandas as pd
import torch
from rdkit import Chem
from rdkit import RDLogger
from torch.utils.data import DataLoader
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

RDLogger.DisableLog("rdApp.*")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from generation_utils import ReactionT5Dataset
from utils import canonicalize, seed_everything, space_clean


def clean_product(product: str) -> str:
    product = str(product)
    if product.startswith("<UNK>"):
        product = product[len("<UNK>") :]
    return product.strip()


def is_valid(smiles: str):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None or mol.GetNumAtoms() == 0:
            return None
        canonical = Chem.MolToSmiles(mol, canonical=True)
        return canonical if canonical else None
    except Exception:
        return None


def preprocess_df(df, drop_duplicates=True):
    for col in ["REACTANT", "PRODUCT", "CATALYST", "REAGENT", "SOLVENT"]:
        if col not in df.columns:
            df[col] = None
        df[col] = df[col].fillna(" ")
    if drop_duplicates:
        df = (
            df[["REACTANT", "PRODUCT", "CATALYST", "REAGENT", "SOLVENT"]]
            .drop_duplicates()
            .reset_index(drop=True)
        )
    df["REAGENT"] = df["CATALYST"] + "." + df["REAGENT"] + "." + df["SOLVENT"]
    df["REAGENT"] = df["REAGENT"].apply(lambda x: space_clean(x))
    df["REAGENT"] = df["REAGENT"].apply(lambda x: canonicalize(x) if x != " " else " ")
    df["input"] = "REACTANT:" + df["REACTANT"] + "REAGENT:" + df["REAGENT"]
    return df


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--end_index", type=int, default=-1)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--input_max_length", type=int, default=400)
    parser.add_argument("--output_max_length", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(args.seed)

    top50_df = pd.read_csv(args.input_csv)
    if args.end_index > 0:
        top50_df = top50_df.iloc[args.start_index : args.end_index].copy()
    else:
        top50_df = top50_df.iloc[args.start_index :].copy()
    top50_df.reset_index(drop=True, inplace=True)

    top_columns = [col for col in top50_df.columns if col.startswith("top_")]
    flat_rows = []
    for row in top50_df.itertuples(index=False):
        product = is_valid(clean_product(row.product))
        if not product:
            continue
        seen = set()
        for col in top_columns:
            reactants = is_valid(str(getattr(row, col)).strip())
            if not reactants or reactants in seen:
                continue
            seen.add(reactants)
            flat_rows.append(
                {
                    "row_index": int(row.row_index),
                    "sample_id": str(row.sample_id),
                    "PRODUCT": product,
                    "REACTANT": reactants,
                    "REAGENT": " ",
                    "CATALYST": " ",
                    "SOLVENT": " ",
                }
            )

    flat_df = pd.DataFrame(flat_rows)
    if flat_df.empty:
        pd.DataFrame(columns=["row_index", "sample_id", "accepted_reactants", "accepted_count"]).to_csv(
            args.output_csv, index=False
        )
        print("no valid candidates")
        return

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, return_tensors="pt")
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path).to(args.device)
    model.eval()

    class Cfg:
        pass

    cfg = Cfg()
    cfg.tokenizer = tokenizer
    cfg.input_max_length = args.input_max_length

    pred_input_df = preprocess_df(flat_df, drop_duplicates=False)
    dataset = ReactionT5Dataset(cfg, pred_input_df)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
    )

    predictions = []
    for inputs in dataloader:
        inputs = {k: v.to(args.device) for k, v in inputs.items()}
        with torch.no_grad():
            output = model.generate(
                **inputs,
                min_length=1,
                max_length=args.output_max_length,
                num_beams=1,
                num_return_sequences=1,
            )
        decoded = tokenizer.batch_decode(output, skip_special_tokens=True)
        predictions.extend([canonicalize(pred.replace(" ", "").rstrip(".")) for pred in decoded])

    accepted = defaultdict(list)
    for meta, predicted in zip(flat_df.itertuples(index=False), predictions):
        if predicted and predicted == meta.PRODUCT:
            accepted[(int(meta.row_index), str(meta.sample_id))].append(meta.REACTANT)

    out_rows = []
    for row in top50_df.itertuples(index=False):
        key = (int(row.row_index), str(row.sample_id))
        reactants = accepted.get(key, [])
        out_rows.append(
            {
                "row_index": key[0],
                "sample_id": key[1],
                "accepted_reactants": "|||".join(reactants),
                "accepted_count": len(reactants),
            }
        )

    out_df = pd.DataFrame(out_rows)
    out_df.to_csv(args.output_csv, index=False)
    print(f"wrote {len(out_df)} rows to {args.output_csv}")
    print(f"accepted_total={int(out_df['accepted_count'].sum())}")


if __name__ == "__main__":
    main()
