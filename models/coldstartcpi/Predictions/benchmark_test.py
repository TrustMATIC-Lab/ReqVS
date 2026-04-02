# -*- coding: utf-8 -*-
"""
适配 ColdstartCPI 的推理脚本 (分块版)
用于对超大 CSV 数据集（如 DUD-E, 1G+）进行低内存推理
"""
import warnings
warnings.filterwarnings("ignore")
import os
import sys
import re
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from rdkit import Chem
from gensim.models import word2vec
from transformers import BertModel, BertTokenizer
import gc
import argparse
# 确保可以导入项目中的模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Train.ColdstartCPI.model import ColdstartCPI
from Predictions.Mol2Vec.mol2vec.features import mol2alt_sentence, MolSentence, Atom2Substructure
basedir = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--model_file',type=str, default=os.path.join(basedir,'..','weights/coldstarcpi_chembl36.pth'))
parser.add_argument('--device',type=str, default='cuda:0')

args = parser.parse_args()
test_name = os.path.splitext(os.path.basename(args.dataset))[0]
# basedir 是 models/coldstartcpi/Predictions，需要往上三级到根目录
OUTPUT_CSV_PATH = os.path.join(basedir, '..', '..', '..', 'test_results', test_name, f'{test_name}_coldstartcpi.csv')



# === 配置参数 ===
BATCH_SIZE = 256 # 批处理大小
CHUNK_SIZE = 40000 # 每次读取 CSV 的行数，防止内存爆炸
DEVICE = torch.device(args.device)
if DEVICE.type == 'cuda':
    torch.backends.cudnn.benchmark = True

# 路径配置 (请根据实际情况修改)
MODEL_WEIGHT_PATH = args.model_file
MOL2VEC_MODEL_PATH = os.path.join(basedir,'..','Feature_generation','Mol2Vec','model_300dim.pkl')
INPUT_CSV_PATH = args.dataset

class InferenceDataset(Dataset):
    def __init__(self, data_list, drug_descriptor, drug_matrix, protein_embed_dict):
        """
        data_list: list of [row_id, protein_seq, smiles, label]
        """
        self.data_list = data_list
        self.drug_descriptor = drug_descriptor
        self.drug_matrix = drug_matrix
        self.protein_embed_dict = protein_embed_dict

    def __getitem__(self, item):
        # data_list stored: [row_id, protein_seq, smiles, label]
        return self.data_list[item]

    def __len__(self):
        return len(self.data_list)

class CollateInference:
    def __init__(self, drug_descriptor, drug_matrix, protein_embed_dict, d_max=100, p_max=1000):
        self.drug_descriptor = drug_descriptor
        self.drug_matrix = drug_matrix
        self.protein_embed_dict = protein_embed_dict
        self.d_max = d_max
        self.p_max = p_max

    def __call__(self, batch_data):
        # batch_data: list of [row_id, protein_seq, smiles, label]
        
        row_ids = []
        proteins = []
        smiles_list = []
        labels = []
        
        d_g_list = []
        d_m_list = []
        d_masks = []
        
        p_g_list = []
        p_m_list = []
        p_masks = []

        for sample in batch_data:
            row_id, p_seq, smiles, label = sample
            row_ids.append(row_id)
            proteins.append(p_seq)
            smiles_list.append(smiles)
            labels.append(label)

            # --- Drug Features ---
            if smiles in self.drug_descriptor:
                d_g_list.append(self.drug_descriptor[smiles])
                d_mat_origin = self.drug_matrix[smiles]
            else:
                # Fallback
                d_g_list.append(np.zeros(300))
                d_mat_origin = np.zeros((1, 300))

            d_matrix = np.zeros([self.d_max, 300]) 
            d_mask = np.zeros([self.d_max])
            d_dim = min(d_mat_origin.shape[0], self.d_max)
            d_matrix[:d_dim] = d_mat_origin[:d_dim]
            d_mask[d_dim:] = 1 
            d_m_list.append(d_matrix)
            d_masks.append(d_mask == 1)

            # --- Protein Features ---
            if p_seq in self.protein_embed_dict:
                p_vec, p_mat_origin = self.protein_embed_dict[p_seq]
                p_g_list.append(p_vec)
            else:
                 # Fallback
                p_g_list.append(np.zeros(1024))
                p_mat_origin = np.zeros((1, 1024))

            p_matrix = np.zeros([self.p_max, 1024])
            p_mask = np.zeros([self.p_max])
            p_dim = min(p_mat_origin.shape[0], self.p_max)
            p_matrix[:p_dim] = p_mat_origin[:p_dim]
            p_mask[p_dim:] = 1
            p_m_list.append(p_matrix)
            p_masks.append(p_mask == 1)

        # Convert to Tensor
        d_g_tensor = torch.from_numpy(np.array(d_g_list)).float()
        d_m_tensor = torch.from_numpy(np.array(d_m_list)).float()
        p_g_tensor = torch.from_numpy(np.array(p_g_list)).float()
        p_m_tensor = torch.from_numpy(np.array(p_m_list)).float()
        d_masks = torch.from_numpy(np.array(d_masks))
        p_masks = torch.from_numpy(np.array(p_masks))
        
        return row_ids, proteins, smiles_list, labels, [d_g_tensor, d_m_tensor, p_g_tensor, p_m_tensor, d_masks, p_masks]

def extract_drug_features(smiles_list, drug_model):
    """
    针对当前 chunk 的 smiles 列表提取特征
    """
    keys = set(drug_model.wv.vocab.keys())
    unseen = 'UNK'
    unseen_vec = drug_model.wv.word_vec(unseen)
    
    drug_descriptor = {}
    drug_matrix = {}
    
    # print(f"Extracting features for {len(smiles_list)} unique compounds...")
    # 这里的 tqdm 可能会刷屏，如果是 chunk 模式可以关掉或简化
    for smiles in smiles_list:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                sentence = MolSentence(mol2alt_sentence(mol, 1))
                matrix = Atom2Substructure(mol, 1, drug_model, keys, unseen_vec)
                
                if matrix.shape[0] > 1:
                    vector = sum([drug_model.wv.word_vec(y) for y in sentence if y in set(sentence) & keys])
                else:
                    vector = matrix[0]
                
                if isinstance(vector, int): 
                    pass
                else:
                    drug_descriptor[smiles] = vector
                    drug_matrix[smiles] = matrix
        except Exception:
            pass
            
    return drug_descriptor, drug_matrix

def extract_all_protein_features(csv_path, col_name):
    """
    预先读取整个 CSV 中的唯一 Protein 序列并提取特征
    """
    print("Scanning CSV for unique proteins...")
    # 只读取 Protein 列，去重
    try:
        df_prot = pd.read_csv(csv_path, usecols=[col_name])
        unique_proteins = df_prot[col_name].unique().tolist()
        print(f"Found {len(unique_proteins)} unique proteins.")
    except Exception as e:
        print(f"Error reading proteins: {e}")
        return {}

    # 使用本地 Transformers 模型替代 bio_embeddings
    print("Loading ProtTrans (Transformers) model from local files...")
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    local_model_path = os.path.join(repo_root, "Feature_generation", "ProtTrans", "prot_bert_bfd")
    try:
        tokenizer = BertTokenizer.from_pretrained(local_model_path, do_lower_case=False)
        prot_model = BertModel.from_pretrained(local_model_path)
    except Exception as e:
        print(f"Error loading local ProtTrans model: {e}")
        return {}
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    prot_model = prot_model.to(device)
    prot_model.eval()
    
    protein_embed_dict = {}
    print(f"Extracting features for {len(unique_proteins)} proteins...")
    
    def get_prottrans_embedding(sequence):
        # 替换非常见氨基酸并加入空格分隔
        sequence = " ".join(list(re.sub(r"[UZOB]", "X", sequence)))
        with torch.no_grad():
            encoded = tokenizer(sequence, return_tensors='pt').to(device)
            output = prot_model(**encoded)
            embedding = output.last_hidden_state.detach().cpu().numpy()[0]
        seq_embed = embedding[1:-1]
        global_embed = np.mean(seq_embed, axis=0)
        return global_embed, seq_embed

    for seq in tqdm(unique_proteins):
        try:
            seq_processed = seq[:2000]
            vector, matrix = get_prottrans_embedding(seq_processed)
            protein_embed_dict[seq] = [vector, matrix]
        except Exception as e:
            print(f"Error processing Protein length {len(seq)}: {e}")
            
    return protein_embed_dict

def main():
    print(f"Using device: {DEVICE}")
    
    if not os.path.exists(INPUT_CSV_PATH):
        print(f"Error: Input file {INPUT_CSV_PATH} not found.")
        return

    # 0. 确定列名映射
    print("Checking CSV columns...")
    df_head = pd.read_csv(INPUT_CSV_PATH, nrows=5)
    col_map = {}
    for col in df_head.columns:
        clean_col = col.strip()
        if 'Protein' in clean_col: col_map['Protein'] = col
        if 'SMILES' in clean_col: col_map['SMILES'] = col
        if 'Y' in clean_col or 'Label' in clean_col: col_map['Y'] = col
        if 'ID' in clean_col: col_map['ID'] = col
    
    print(f"Mapped columns: {col_map}")
    if 'Protein' not in col_map or 'SMILES' not in col_map:
        print("Error: Missing required columns (Protein, SMILES)")
        return

    # 1. 预先加载并提取 Protein 特征 (因为很少，全局一份)
    prot_embed_dict = extract_all_protein_features(INPUT_CSV_PATH, col_map['Protein'])
    if not prot_embed_dict:
        print("Warning: No protein features extracted!")

    # 2. 加载 Mol2Vec 模型 (常驻内存)
    print("Loading Mol2Vec model...")
    mol2vec_model = word2vec.Word2Vec.load(MOL2VEC_MODEL_PATH)

    # 3. 加载 PyTorch 模型
    print("Loading ColdstartCPI model...")
    model = ColdstartCPI(unify_num=512, head_num=4, dataset="ChEMBL36")
    if os.path.exists(MODEL_WEIGHT_PATH):
        print(f"Loading weights from {MODEL_WEIGHT_PATH}")
        state_dict = torch.load(MODEL_WEIGHT_PATH, map_location=DEVICE)
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        model.load_state_dict(new_state_dict)
    else:
        print(f"Warning: Model weight file not found at {MODEL_WEIGHT_PATH}. Using random weights!")
    
    model.to(DEVICE)
    model.eval()

    # 4. 分块处理 CSV
    print(f"Starting chunked inference (Chunk size: {CHUNK_SIZE})...")
    
    # 确保输出目录存在
    output_dir = os.path.dirname(OUTPUT_CSV_PATH)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"Created output directory: {output_dir}")
    
    # 如果输出文件存在，先删除，避免重复追加
    if os.path.exists(OUTPUT_CSV_PATH):
        os.remove(OUTPUT_CSV_PATH)
        print("Removed existing output file.")

    chunk_iterator = pd.read_csv(INPUT_CSV_PATH, chunksize=CHUNK_SIZE)
    
    total_processed = 0
    
    for i, chunk in enumerate(chunk_iterator):
        print(f"Processing Chunk {i+1}...")
        
        # 补充 ID 列
        if 'ID' not in col_map:
            chunk['ID'] = range(total_processed, total_processed + len(chunk))
            col_id = 'ID'
        else:
            col_id = col_map['ID']

        # 提取当前 chunk 的 unique SMILES
        chunk_smiles = chunk[col_map['SMILES']].unique().tolist()
        
        # 提取 Drug 特征
        drug_desc, drug_mat = extract_drug_features(chunk_smiles, mol2vec_model)
        
        # 准备 Dataset（保持原逻辑：仅当两者均有特征时加入）
        valid_data = []
        for idx, row in chunk.iterrows():
            smiles = row[col_map['SMILES']]
            prot = row[col_map['Protein']]
            label = row[col_map['Y']] if 'Y' in col_map else 0
            row_id = row[col_id]
            
            if smiles in drug_desc and prot in prot_embed_dict:
                valid_data.append([row_id, prot, smiles, label])
        
        if not valid_data:
            print(f"Chunk {i+1} has no valid data (features missing). Skipping.")
            total_processed += len(chunk)
            continue
            
        # 构造 DataLoader
        dataset = InferenceDataset(valid_data, drug_desc, drug_mat, prot_embed_dict)
        collate = CollateInference(drug_desc, drug_mat, prot_embed_dict)
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=(DEVICE.type=='cuda'), collate_fn=collate)
        
        # 推理
        chunk_results = []
        with torch.no_grad():
            autocast_enabled = (DEVICE.type == 'cuda')
            for batch in loader:
                row_ids, proteins, smiles_list, labels, input_tensors = batch
                input_tensors = [t.to(DEVICE, non_blocking=True) for t in input_tensors]
                with torch.cuda.amp.autocast(enabled=autocast_enabled):
                    logits = model(input_tensors)
                    probs = F.softmax(logits, dim=1)[:, 1].float().detach().cpu().numpy()
                
                for k in range(len(row_ids)):
                    chunk_results.append({
                        "ID": row_ids[k],
                        "Protein": proteins[k],
                        "SMILES": smiles_list[k],
                        "Y": labels[k],
                        "modelColdstartCPI_prediction": probs[k]
                    })
        
        # 保存当前 Chunk 结果
        result_df = pd.DataFrame(chunk_results)
        # 如果是第一个块，写入 header；后续块不写入 header
        write_header = (i == 0)
        result_df.to_csv(OUTPUT_CSV_PATH, mode='a', index=False, header=write_header)
        
        print(f"Chunk {i+1} saved. ({len(result_df)} records)")
        total_processed += len(chunk)
        
        # 清理内存
        del dataset, loader, drug_desc, drug_mat, chunk_results, result_df, chunk
        gc.collect()

    print(f"All done! Total processed: {total_processed}")
    print(f"Results saved to {OUTPUT_CSV_PATH}")

if __name__ == "__main__":
    main()
