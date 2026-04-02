import torch
import torch.nn as nn
import time
import pickle
from tqdm import tqdm
import pandas as pd
import os
from rdkit import Chem
from required_files_for_demo.demo_utils import *
from required_files_for_demo.model_aff import DeepDTAGen
import hashlib


def predict_affinity(smiles, protein_sequence, model, tokenizer, device):
    """预测单个化合物的亲和力"""
    # 生成唯一的文件名
    filename = hashlib.md5(f"{smiles}_{protein_sequence[:50]}".encode()).hexdigest() + ".pt"
    processed_data = os.path.join('data/processed', filename)

    # 检查是否已有处理好的数据
    if os.path.isfile(processed_data):
        test_data = torch.load(processed_data)
    else:
        # 处理数据
        test_data = process_latent_a(smiles, protein_sequence)
        # 确保目录存在
        os.makedirs('data/processed', exist_ok=True)
        # 保存处理后的数据
        torch.save(test_data, processed_data)

    # 创建DataLoader
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False, collate_fn=collate)

    # 预测亲和力
    model.eval()
    with torch.no_grad():
        for data in tqdm(test_loader, desc='Testing'):
            predictions = model(data.to(device))
            return predictions.item()

    return None


def main():
    # 竞赛蛋白质序列
    TARGET_PROTEIN = "MYNGSCCRIEGDTISQVMPPLLIVAFVLGALGNGVALCGFCFHMKTWKPSTVYLFNLAVADFLLMICLPFRTDYYLRRRHWAFFGDIPCRVGLFTLAMNRAGSIVFLTVVAADRYKVVHPHHAVNTISTRVAAGIVCTLWALVILGTVYLLLENHLCVQETAVSCESFIMESANGWHDIMFQLEFFMPLGIILFCSFKIVWSLRRRQQLARQARMKKATRFIMVVAIVFITCYLPSVSARLYFLWTVPSSACDPSVHGALHITLSFTVMNSMLDPLVYYFSSPSFPKFYNKLKICSLKPKQPGHSKTQRPEEMPISNLGRRSCISVANSFQSQSDGQWDPHIVEWH"

    # 1. 加载化合物库
    compounds_df = pd.read_csv("filtered_active2.csv")
    print(f"Loaded {len(compounds_df)} compounds")

    # 2. 设置设备和模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 使用BindingDB预训练模型
    model_path = 'deepdtagen_model_bindingdb.pth'
    tokenizer_path = 'bindingdb_tokenizer.pkl'

    # Load tokenizer
    with open(tokenizer_path, 'rb') as f:
        tokenizer = pickle.load(f)

    # Load model
    model = DeepDTAGen(tokenizer)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # 3. 为每个化合物预测亲和力
    results = []
    failed_compounds = []

    # 创建进度条
    progress_bar = tqdm(compounds_df.iterrows(), total=len(compounds_df), desc="Predicting affinity")

    for idx, row in progress_bar:
        compound_id = row['compound_id']
        smiles = row['smiles']

        # 更新进度条描述
        progress_bar.set_description(f"Processing {compound_id[:10]}...")

        try:
            # 预测亲和力
            affinity = predict_affinity(smiles, TARGET_PROTEIN, model, tokenizer, device)

            if affinity is not None:
                results.append({
                    'compound_id': compound_id,
                    'smiles': smiles,
                    'predicted_affinity': affinity
                })
                #time.sleep(2)
            else:
                failed_compounds.append({
                    'compound_id': compound_id,
                    'smiles': smiles,
                    'reason': 'Prediction failed'
                })
        except Exception as e:
            failed_compounds.append({
                'compound_id': compound_id,
                'smiles': smiles,
                'reason': str(e)
            })

    # 4. 按亲和力排序（高到低）
    results_df = pd.DataFrame(results)
    if not results_df.empty:
        results_df = results_df.sort_values('predicted_affinity', ascending=False)

        # 保存top 1000化合物
        top_results = results_df.head(1000)
        top_results.to_csv("competition_top_results.csv", index=False)
        print(f"Saved top {len(top_results)} results to competition_top_results.csv")

        # 保存完整结果
        results_df.to_csv("competition_full_results.csv", index=False)
        print(f"Saved full results to competition_full_results.csv")
    else:
        print("No valid results to save!")

    # 保存失败记录
    if failed_compounds:
        failed_df = pd.DataFrame(failed_compounds)
        failed_df.to_csv("failed_predictions.csv", index=False)
        print(f"Saved {len(failed_df)} failed compounds to failed_predictions.csv")

    # 打印统计信息
    print("\nProcessing Summary:")
    print(f"Total compounds: {len(compounds_df)}")
    print(f"Successfully predicted: {len(results)}")
    print(f"Failed predictions: {len(failed_compounds)}")


if __name__ == "__main__":
    main()