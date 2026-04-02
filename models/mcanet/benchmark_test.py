# -*- coding:utf-8 -*-

import os
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pandas as pd
from tqdm import tqdm

from config import hyperparameter
from model import mcanet
from utils.DataSetsFunction import CustomDataSet, collate_fn_outer
from utils.TestModel import test_model_outer
from config import hyperparameter




def load_test_data(test_file_path):
    """
    Expects CSV with: ID, Protein_sequence, SMILES, label.
    Returns:
        data_list: list of (ID, SMILES, Protein_sequence, label) rows.
    """
    print(f"Loading test data: {test_file_path}")
    data_list = []

    df = pd.read_csv(test_file_path)
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Loading test data"):
        data_list.append([row['ID'], row['SMILES'], row['Protein'], row['Y'],row['SMILES'],row['Protein']])

    print(f"Data loaded, {len(data_list)} records")


    return data_list



def test_outer_dataset(
    test_file_path,
    model_path,
    output_dir,
    output_filename,
    DEVICE,
):

    os.makedirs(output_dir, exist_ok=True)

    test_data_list = load_test_data(test_file_path)

    test_dataset = CustomDataSet(test_data_list)
    test_dataset_loader = DataLoader(
        test_dataset,
        batch_size=128,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn_outer,
        drop_last=False
    )
    

    hp = hyperparameter()
    print(f"Loading model: {model_path}")
    model = mcanet(hp).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))

    print("✓ Model loaded successfully")
    print("\nStarting test...")
    test_results = test_model_outer(
        model,
        test_file_path,
        test_dataset_loader,
        DEVICE,
        FOLD_NUM=1
    )
    

    results_df = pd.DataFrame(
        test_results,
        columns=['ID', 'SMILES', 'Protein', 'Y', 'prediction_mcanet']
    )
    results_df.to_csv(os.path.join(output_dir, output_filename), index=False)
    print(f'✓ Predictions saved to: {os.path.join(output_dir, output_filename)}')

    
    return None


def main():
    """Main entry; configure via CLI args or by editing below."""
    import argparse
    base_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(
        description='mcanet test benchmarks'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
    )
    parser.add_argument(
        '--model-path',
        type=str,
        default=os.path.join(base_dir, "weights", "mcanet_chembl36.pth"),
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda')
    
    
    args = parser.parse_args()
    dataset_name = os.path.splitext(os.path.basename(args.dataset))[0]
    DEVICE = torch.device(args.device)
    output_dir = os.path.join(base_dir, "..", "..", "test_results", dataset_name)
    test_outer_dataset(
        test_file_path=args.dataset,
        model_path=args.model_path,
        output_dir=output_dir,
        output_filename=f"{dataset_name}_mcanet.csv",
        DEVICE=DEVICE,
    )


if __name__ == "__main__":
    main()

