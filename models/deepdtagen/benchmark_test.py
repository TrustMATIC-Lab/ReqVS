import torch
import torch.nn as nn
import time
import pickle
from tqdm import tqdm
import pandas as pd
import os
from rdkit import Chem
from DEMO.required_files_for_demo.demo_utils import process_single_sample
import argparse
from DEMO.required_files_for_demo.model_aff import Model3
import hashlib
import csv
from torch_geometric.data import Batch

def predict_affinity(smiles, protein_sequence, model, tokenizer, device):
    """Predict affinity for a single compound; simplified, processes one sample directly."""
    data = process_single_sample(smiles, protein_sequence)

    batch = Batch.from_data_list([data])
    batch = batch.to(device)

    model.eval()
    with torch.no_grad():
        predictions = model(batch)
    
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.item()
    else:
        predictions = predictions[0].item()
    return predictions



def parse_args():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, type=str)
    parser.add_argument('--model_path', type=str, default=os.path.join(base_dir, 'weights', 'deepdtagen_chembl36.pth'))
    parser.add_argument('--tokenizer_path', type=str, default=os.path.join(base_dir, 'data', 'chembl36_aug_reg_tokenizer.pkl'))
    parser.add_argument('--device', type=str, default='cuda:0')
    return parser.parse_args()
def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    args = parse_args()


    input_path = args.dataset
    model_path = args.model_path
    tokenizer_path = args.tokenizer_path
    device = torch.device(args.device)
    # 只使用文件名（不含路径和扩展名）
    test_name = os.path.splitext(os.path.basename(args.dataset))[0]
    # base_dir 是 models/deepdtagen，需要往上一级到根目录
    output_path = os.path.join(base_dir, '..', '..', 'test_results', test_name, f'{test_name}_deepdtagen.csv')

    print(f"Using device: {device}")
    # Load tokenizer
    with open(tokenizer_path, 'rb') as f:
        tokenizer = pickle.load(f)

    # Load model
    model = Model3(tokenizer)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    with open(input_path,'r') as fin,open(output_path,'w') as fout:
        reader = csv.DictReader(fin)
        fieldnames = reader.fieldnames + ['prediction_deepdtagen']
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()

        for row in tqdm(reader):

            smiles = row['SMILES']
            protein_sequence = row['Protein']
            predicted_affinity = predict_affinity(smiles, protein_sequence, model, tokenizer, device)
            row['prediction_deepdtagen'] = predicted_affinity
            writer.writerow(row)


if __name__ == "__main__":
    main()