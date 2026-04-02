import torch
import torch.nn as nn
import pickle
from tqdm import tqdm
import pandas as pd
from torch.utils.data import DataLoader
import argparse
from sklearn.metrics import average_precision_score
from rdkit import RDLogger, Chem
from utils import *
from model import DeepDTAGen

# --- Utility Functions for SMILES Evaluation ---
def is_valid_smiles(smiles):
    """Check if a SMILES string represents a valid molecule."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol is not None
    except Exception:
        return False
    
def evaluate_model(model, test_loader, auc_thresholds, aupr_threshold, device):
    """Evaluate the model on the test dataset and compute various metrics."""
    model.eval()
    total_true = torch.Tensor().to(device)
    total_predict = torch.Tensor().to(device)

    with torch.no_grad():
        for data in tqdm(test_loader, desc="Testing"):
            predictions, _, lm_loss, kl_loss = model(data.to(device))
            total_true = torch.cat((total_true, data.y.view(-1, 1)), dim=0)
            total_predict = torch.cat((total_predict, predictions), dim=0)

    ground_truth = total_true.cpu().numpy().flatten()
    predicted = total_predict.cpu().numpy().flatten()

    # Calculate evaluation metrics
    mse_loss = mse(ground_truth, predicted)
    concordance_index = get_cindex(ground_truth, predicted)
    rm2_value = get_rm2(ground_truth, predicted)
    rms_error = rmse(ground_truth, predicted)
    pearson_corr = pearson(ground_truth, predicted)
    spearman_corr = spearman(ground_truth, predicted)

    # Calculate AUPR for a single threshold
    aupr_value = get_aupr(predicted, ground_truth, aupr_threshold)

    
    auc_values = []
    aupr_values = []
    for threshold in auc_thresholds:
        binary_pred = (predicted > threshold).astype(int)
        binary_true = (ground_truth > threshold).astype(int)
        auc = get_auc(binary_pred, binary_true)
        auc_values.append(auc)
    return {
        f"MSE: {mse_loss:.4f}",
        f"CI: {concordance_index:.4f}",
        f"RM2: {rm2_value:.4f}",
        f"AUPR: {aupr_value:.4f}",
        f'AUC (ROC): {auc_values}',
    }

def evaluate_smiles(df):
    """Evaluate the validity, uniqueness, and novelty of generated SMILES."""
    # Extract SMILES strings from both columns
    generated_smiles = df['Generated_SMILES'].dropna().tolist()
    input_smiles = df['target_smiles'].dropna().tolist()

    # Validity: Check how many generated molecules are valid
    valid_smiles = [smiles for smiles in generated_smiles if is_valid_smiles(smiles)]
    validity_ratio = len(valid_smiles) / len(generated_smiles) if generated_smiles else 0

    # Uniqueness: Check how many unique valid SMILES strings are there
    unique_smiles = set(valid_smiles)
    uniqueness_ratio = len(unique_smiles) / len(valid_smiles) if valid_smiles else 0

    # Novelty: Check how many valid unique SMILES are novel (not in the input dataset)
    input_smiles_set = set(input_smiles)
    novel_unique_smiles = [smiles for smiles in unique_smiles if smiles not in input_smiles_set]
    novelty_ratio = len(novel_unique_smiles) / len(unique_smiles) if unique_smiles else 0

    return {
        "Validity Ratio": validity_ratio,
        "Uniqueness Ratio": uniqueness_ratio,
        "Novelty Ratio": novelty_ratio,
    }


def load_model_and_data(dataset_name, device):
    """Load the model, tokenizer, and test data."""
    model_path = f'models/deepdtagen_model_{dataset_name}.pth'
    tokenizer_path = f'data/{dataset_name}_tokenizer.pkl'
    test_batch_size = 128

    # Define thresholds based on dataset
    if dataset_name == 'kiba':
        auc_thresholds = [10.0, 10.50, 11.0, 11.50, 12.0, 12.50]  # Multiple thresholds for AUC
        aupr_threshold = 12.1  # Single threshold for AUPR
    elif dataset_name == 'davis' or dataset_name == 'bindingdb':
        auc_thresholds = [5.0, 5.50, 6.0, 6.50, 7.0, 7.50, 8.0, 8.50]  # Multiple thresholds for AUC
        aupr_threshold = 7.0  # Single threshold for AUPR
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # Load tokenizer
    with open(tokenizer_path, 'rb') as f:
        tokenizer = pickle.load(f)

    # Load model
    model = DeepDTAGen(tokenizer)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)

    # Load test data
    test_data = TestbedDataset(root='data', dataset=f'{dataset_name}_test')
    test_loader = DataLoader(test_data, batch_size=test_batch_size, shuffle=False)

    df = pd.read_csv(f'data/{dataset_name}_generated.csv')

    return model, test_loader, auc_thresholds, aupr_threshold, df


def main(dataset_name):
    """Main function to evaluate both the model and generated molecules."""
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model and data
    model, test_loader, auc_thresholds, aupr_threshold, df = load_model_and_data(dataset_name, device)

    # Model Evaluation
    print("\nEvaluating the model...")
    model_metrics = evaluate_model(model, test_loader, auc_thresholds, aupr_threshold, device)

    # Print model evaluation results
    print("\nModel Evaluation Results:")
    for value in model_metrics:
        print(f"{value}")


    # SMILES Evaluation
    print("\nEvaluating the generated SMILES...")
    smiles_metrics = evaluate_smiles(df)

    # Print SMILES evaluation results
    print("\nSMILES Evaluation Results:")
    for key, value in smiles_metrics.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate DeepDTAGen model and generated molecules.")
    parser.add_argument("--dataset", type=str, required=True, help="Name of the dataset (e.g., davis, bindingdb, kiba)")
    args = parser.parse_args()

    main(args.dataset)
