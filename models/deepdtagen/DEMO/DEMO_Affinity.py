import torch
import torch.nn as nn
import pickle
from tqdm import tqdm
import pandas as pd
from torch.utils.data import DataLoader
from required_files_for_demo.demo_utils import *

from required_files_for_demo.model_aff import DeepDTAGen

def demo():
    dataset_name = 'bindingdb'

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #device = torch.device('cpu')

    # Paths
    model_path = f'deepdtagen_model_{dataset_name}.pth'
    tokenizer_path = f'{dataset_name}_tokenizer.pkl'

    smiles = "[H][C@@]1(O[C@@H]2[C@@H](O)[C@@H](O)[C@@H](CO)O[C@@]2([H])O[C@@H]2[C@@H](O)[C@H](O)[C@H](O[C@@]2([H])O[C@H]2CC[C@@]3(C)[C@@]([H])(CC[C@]4(C)[C@]3([H])CC=C3[C@]5([H])CC(C)(C)C[C@@H](O)[C@]5(C)CC[C@@]43C)[C@@]2(C)CO)C(O)=O)O[C@@H](C)[C@H](O)[C@@H](O)[C@H]1O"
    protein_sequence = "MYNGSCCRIEGDTISQVMPPLLIVAFVLGALGNGVALCGFCFHMKTWKPSTVYLFNLAVADFLLMICLPFRTDYYLRRRHWAFFGDIPCRVGLFTLAMNRAGSIVFLTVVAADRYKVVHPHHAVNTISTRVAAGIVCTLWALVILGTVYLLLENHLCVQETAVSCESFIMESANGWHDIMFQLEFFMPLGIILFCSFKIVWSLRRRQQLARQARMKKATRFIMVVAIVFITCYLPSVSARLYFLWTVPSSACDPSVHGALHITLSFTVMNSMLDPLVYYFSSPSFPKFYNKLKICSLKPKQPGHSKTQRPEEMPISNLGRRSCISVANSFQSQSDGQWDPHIVEWH"

    # Load tokenizer
    with open(tokenizer_path, 'rb') as f:
        tokenizer = pickle.load(f)

    # Load model
    model = DeepDTAGen(tokenizer)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)

    # Load test data
    processed_data = f'./data/processed/{smiles}.pt'
    if not os.path.isfile(processed_data):
        test_data = process_latent_a(smiles, protein_sequence)
    else:
        test_data = torch.load(processed_data)
    print(test_data)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False, collate_fn=collate)

    # Evaluate the model
    model.eval()
    with torch.no_grad():
        for data in tqdm(test_loader, desc='Testing'):
            predictions = model(data.to(device))
        print("Predicted Affinity :", predictions)

if __name__ == "__main__":
    demo()