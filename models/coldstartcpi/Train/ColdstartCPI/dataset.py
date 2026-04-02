import torch
from torch.utils.data import Dataset, DataLoader
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import StratifiedKFold
class CustomDataSet(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs

    def __getitem__(self, item):
        return self.pairs[item]

    def __len__(self):
        return len(self.pairs)

class collater_embeding():
    """"""
    def __init__(self,drug_f,drug_m,protein_m,d_max = 100,p_max = 1000):
        self.drug_f = drug_f
        self.drug_m = drug_m
        self.protein_m = protein_m
        self.d_max = d_max
        self.p_max = p_max

    def __call__(self, batch_data):
        # 1. 解析 batch_data (list of [d_id, p_id, label])
        # 转为 numpy array 以便批量索引
        batch_np = np.array(batch_data)
        d_ids = batch_np[:, 0]
        p_ids = batch_np[:, 1]
        labels = batch_np[:, 2].astype(float)

        batch_size = len(batch_data)

        # 2. 获取特征维度 (假设同一batch内维度一致，取第一个样本)
        # 不再强制转int，直接使用 d_ids[0] 作为 key
        d_feat_dim = self.drug_m[d_ids[0]].shape[1]
        p_feat_dim = self.protein_m[p_ids[0]][1].shape[1]

        # 3. 预分配大数组
        # Drug Matrix
        d_m_tensor = np.zeros((batch_size, self.d_max, d_feat_dim), dtype=np.float32)
        d_masks = np.zeros((batch_size, self.d_max), dtype=bool)
        
        # Protein Matrix
        p_m_tensor = np.zeros((batch_size, self.p_max, p_feat_dim), dtype=np.float32)
        p_masks = np.zeros((batch_size, self.p_max), dtype=bool)
        
        d_g_list = []
        p_g_list = []

        # 4. 填充数据
        for i in range(batch_size):
            d_id = d_ids[i]
            p_id = p_ids[i]
            
            # --- Drug ---
            d_g_list.append(self.drug_f[d_id])
            
            d_mat = self.drug_m[d_id]
            d_len = min(d_mat.shape[0], self.d_max)
            # 直接切片赋值，避免 if-else
            d_m_tensor[i, :d_len, :] = d_mat[:d_len]
            d_masks[i, d_len:] = 1 # mask 1 表示 padding

            # --- Protein ---
            p_data = self.protein_m[p_id]
            p_g_list.append(p_data[0])
            
            p_mat = p_data[1]
            p_len = min(p_mat.shape[0], self.p_max)
            p_m_tensor[i, :p_len, :] = p_mat[:p_len]
            p_masks[i, p_len:] = 1

        # 转 Tensor
        d_g_tensor = torch.from_numpy(np.array(d_g_list)).float()
        d_m_tensor = torch.from_numpy(d_m_tensor).float()
        p_g_tensor = torch.from_numpy(np.array(p_g_list)).float()
        p_m_tensor = torch.from_numpy(p_m_tensor).float()
        d_masks = torch.from_numpy(d_masks)
        p_masks = torch.from_numpy(p_masks)
        labels_tensor = torch.from_numpy(labels).long()

        return [d_g_tensor, d_m_tensor, p_g_tensor, p_m_tensor, d_masks, p_masks], labels_tensor

import os,pickle
def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def split_train_valid(data_df, fold, val_ratio=0.1):
    cv_split = StratifiedShuffleSplit(n_splits=2, test_size=val_ratio, random_state=fold)
    train_index, val_index = next(iter(cv_split.split(X=range(len(data_df)), y=data_df['label'])))

    train_df = data_df.iloc[train_index]
    val_df = data_df.iloc[val_index]

    return train_df, val_df

def load_scenario_dataset(DATASET,setting,i, batch_size):
    columns = ['head', 'tail', 'label']
    # 修正相对路径，使其基于当前运行脚本的相对位置，或者使用更稳健的相对路径
    # 假设从项目根目录运行，即 d:\pyproject\ColdstartCPI
    # 数据集在 Datasets/ChEMBL36/warm_start/...
    train_df = pd.read_csv("Datasets/{}/{}/train_set{}.csv".format(DATASET,setting,i))[columns]
    valid_df = pd.read_csv("Datasets/{}/{}/valid_set{}.csv".format(DATASET,setting,i))[columns]
    test_df = pd.read_csv("Datasets/{}/{}/test_set{}.csv".format(DATASET,setting,i))[columns]
    train_set = CustomDataSet(train_df.values)
    val_set = CustomDataSet(valid_df.values)
    test_set = CustomDataSet(test_df.values)
    try:
        drug_features = load_pickle("Datasets/{}/feature/compound_Mol2Vec300.pkl".format(DATASET))
        drug_pretrain = load_pickle("Datasets/{}/feature/compound_Atom2Vec300.pkl".format(DATASET))
        protein_pretrain = load_pickle("Datasets/{}/feature/aas_ProtTransBertBFD1024.pkl".format(DATASET))
    except:
        print("Pre-training features for compounds and proteins are not found in the {}/feature folder, \n\
        please check the file naming or run Mol2Vec.py and generator.py first.".format(DATASET))
        raise
    collate_fn = collater_embeding(drug_features, drug_pretrain, protein_pretrain)

    # 彻底回退到 num_workers=0，这是最稳定的设置，避免任何多进程报错
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0,
                                    collate_fn=collate_fn,pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0,pin_memory=True,collate_fn=collate_fn)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0,pin_memory=True,collate_fn=collate_fn)

    print("Number of samples in the train set: ", len(train_set))
    print("Number of samples in the validation set: ", len(val_set))
    print("Number of samples in the test set: ", len(test_set))

    return train_loader, val_loader, test_loader

def load_Miss_dataset(DATASET,miss_rate, batch_size, fold=0):
    columns = ['head', 'tail', 'label']
    full_df = pd.read_csv("Datasets/{}/full_pair.csv".format(DATASET))[columns]
    train_df, valid_test_df = split_train_valid(full_df, fold=fold,val_ratio=miss_rate/100)
    test_df, val_df = split_train_valid(valid_test_df, fold=fold, val_ratio=0.1)
    train_set = CustomDataSet(train_df.values)
    val_set = CustomDataSet(val_df.values)
    test_set = CustomDataSet(test_df.values)
    try:
        drug_features = load_pickle("Datasets/{}/feature/compound_Mol2Vec300.pkl".format(DATASET))
        drug_pretrain = load_pickle("Datasets/{}/feature/compound_Atom2Vec300.pkl".format(DATASET))
        protein_pretrain = load_pickle("Datasets/{}/feature/aas_ProtTransBertBFD1024.pkl".format(DATASET))
    except:
        print("Pre-training features for compounds and proteins are not found in the {}/feature folder, \n\
        please check the file naming or run Mol2Vec.py and generator.py first.".format(DATASET))
        raise
    collate_fn = collater_embeding(drug_features, drug_pretrain, protein_pretrain)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0,
                                    collate_fn=collate_fn)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0,collate_fn=collate_fn)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0,collate_fn=collate_fn)
    print("Number of samples in the train set: ", len(train_set))
    print("Number of samples in the validation set: ", len(val_set))
    print("Number of samples in the test set: ", len(test_set))

    return train_loader, val_loader, test_loader

"""load BindingDB of AIBind datasets"""
def load_BindingDB_AIBind_dataset(DATASET,scenarios, batch_size, fold=0):
    # 兼容旧接口
    return load_scenario_dataset(DATASET, scenarios, fold, batch_size)
