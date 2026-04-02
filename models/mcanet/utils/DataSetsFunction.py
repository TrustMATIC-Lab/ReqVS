import torch
from torch.utils.data import Dataset
import numpy as np
CHARISOSMISET = {"#": 29, "%": 30, ")": 31, "(": 1, "+": 32, "-": 33, "/": 34, ".": 2,
                 "1": 35, "0": 3, "3": 36, "2": 4, "5": 37, "4": 5, "7": 38, "6": 6,
                 "9": 39, "8": 7, "=": 40, "A": 41, "@": 8, "C": 42, "B": 9, "E": 43,
                 "D": 10, "G": 44, "F": 11, "I": 45, "H": 12, "K": 46, "M": 47, "L": 13,
                 "O": 48, "N": 14, "P": 15, "S": 49, "R": 16, "U": 50, "T": 17, "W": 51,
                 "V": 18, "Y": 52, "[": 53, "Z": 19, "]": 54, "\\": 20, "a": 55, "c": 56,
                 "b": 21, "e": 57, "d": 22, "g": 58, "f": 23, "i": 59, "h": 24, "m": 60,
                 "l": 25, "o": 61, "n": 26, "s": 62, "r": 27, "u": 63, "t": 28, "y": 64}

CHARISOSMILEN = 64

CHARPROTSET = {"A": 1, "C": 2, "B": 3, "E": 4, "D": 5, "G": 6,
               "F": 7, "I": 8, "H": 9, "K": 10, "M": 11, "L": 12,
               "O": 13, "N": 14, "Q": 15, "P": 16, "S": 17, "R": 18,
               "U": 19, "T": 20, "W": 21, "V": 22, "Y": 23, "X": 24, "Z": 25}

CHARPROTLEN = 25


def label_smiles(line, smi_ch_ind, MAX_SMI_LEN=100):
    """将SMILES字符串转换为固定长度的整数数组

    参数:
        line (str): SMILES字符串
        smi_ch_ind (dict): 字符到索引的映射
        MAX_SMI_LEN (int): 最大长度，默认为100

    返回:
        np.array: 固定长度的整数数组
    """
    # 创建固定长度的零数组
    X = np.zeros(MAX_SMI_LEN, dtype=np.int64)

    # 清理输入字符串
    line = line.strip().replace(' ', '')

    # 处理字符直到达到最大长度
    for i, ch in enumerate(line[:MAX_SMI_LEN]):
        if ch in smi_ch_ind:
            X[i] = smi_ch_ind[ch]
        else:
            # 遇到未知字符，跳过或记录警告
            print(f"警告: 在SMILES字符串中遇到未知字符 '{ch}'，位置 {i}")
            # 可以选择跳过（保持为0）或使用特殊标记

    return X


def label_sequence(line, smi_ch_ind, MAX_SEQ_LEN=1000):
    X = np.zeros(MAX_SEQ_LEN, np.int64())
    for i, ch in enumerate(line[:MAX_SEQ_LEN]):
        X[i] = smi_ch_ind[ch]
    return X


class CustomDataSet(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs

    def __getitem__(self, item):
        return self.pairs[item]

    def __len__(self):
        return len(self.pairs)

'''
def collate_fn(batch_data):
    N = len(batch_data)
    drug_ids, protein_ids = [], []
    compound_max = 100
    protein_max = 1000
    compound_new = torch.zeros((N, compound_max), dtype=torch.long)
    protein_new = torch.zeros((N, protein_max), dtype=torch.long)
    labels_new = torch.zeros(N, dtype=torch.long)
    for i, pair in enumerate(batch_data):
        pair = pair.strip().split()
        drug_id, protein_id, compoundstr, proteinstr, label = pair[-5], pair[-4], pair[-3], pair[-2], pair[-1]
        drug_ids.append(drug_id)
        protein_ids.append(protein_id)
        compoundint = torch.from_numpy(label_smiles(
            compoundstr, CHARISOSMISET, compound_max))
        compound_new[i] = compoundint
        proteinint = torch.from_numpy(label_sequence(
            proteinstr, CHARPROTSET, protein_max))
        protein_new[i] = proteinint
        label = float(label)
        labels_new[i] = np.int(label)
    return (compound_new, protein_new, labels_new)
'''
#训练用

def collate_fn(batch_data):
    compound_max = 100
    protein_max = 1000
    batch_data = [data for data in batch_data if data is not None]  # 过滤掉None
    N = len(batch_data)
    drug_ids = []
    protein_ids = []
    compound_new = torch.zeros((N, compound_max), dtype=torch.long)
    protein_new = torch.zeros((N, protein_max), dtype=torch.long)
    labels_new = torch.zeros(N, dtype=torch.long)

    for i, pair in enumerate(batch_data):
        pair = pair.strip().split()

        # 根据字段数量自动判断格式
        if len(pair) == 5:
            # 旧格式: drug_id, protein_id, compoundstr, proteinstr, label
            drug_id, protein_id, compoundstr, proteinstr, label = pair
        elif len(pair) == 3:
            # 新格式: compoundstr (smiles), proteinstr (sequence), label
            compoundstr, proteinstr, label = pair
            drug_id, protein_id = "N/A", "N/A"  # 添加占位符ID
        else:
            raise ValueError(f"无效的数据行格式: {pair}")

        drug_ids.append(drug_id)
        protein_ids.append(protein_id)
        compoundint = torch.from_numpy(label_smiles(
            compoundstr, CHARISOSMISET, compound_max))
        compound_new[i] = compoundint
        proteinint = torch.from_numpy(label_sequence(
            proteinstr, CHARPROTSET, protein_max))
        protein_new[i] = proteinint
        label = float(label)
        labels_new[i] = int(label)
    return (compound_new, protein_new, labels_new)

def collate_fn_VSdataset(batch_data):
    """
    batch_data 中每一项是：
        (compoundstr, proteinstr, label)
    """
    compound_max = 100
    protein_max = 1000

    batch_data = [d for d in batch_data if d is not None]
    N = len(batch_data)

    compound_new = torch.zeros((N, compound_max), dtype=torch.long)
    protein_new = torch.zeros((N, protein_max), dtype=torch.long)
    labels_new = torch.zeros(N, dtype=torch.long)
    
    for i, pair in enumerate(batch_data):
        # 直接解包（你必须保证 Dataset.__getitem__ 返回这四个）
        compoundstr, proteinstr, label = pair


        # SMILES → tensor
        compound_new[i] = torch.from_numpy(
            label_smiles(compoundstr, CHARISOSMISET, compound_max)
        )

        # Protein → tensor
        protein_new[i] = torch.from_numpy(
            label_sequence(proteinstr, CHARPROTSET, protein_max)
        )

        labels_new[i] = int(label)

    return (compound_new, protein_new, labels_new)

def collate_fn_outer(batch_data):
    """
    batch_data 中每一项是：
        (ID, compoundstr, proteinstr, label)
    """
    compound_max = 100
    protein_max = 1000

    batch_data = [d for d in batch_data if d is not None]
    N = len(batch_data)

    IDs = []
    compound_new = torch.zeros((N, compound_max), dtype=torch.long)
    protein_new = torch.zeros((N, protein_max), dtype=torch.long)
    labels_new = torch.zeros(N, dtype=torch.long)
    orig_smiles_list = []
    orig_protein_list = []
    for i, pair in enumerate(batch_data):
        # 直接解包（你必须保证 Dataset.__getitem__ 返回这四个）
        ID, compoundstr, proteinstr, label,orig_smiles,orig_protein = pair

        IDs.append(ID)
        orig_smiles_list.append(orig_smiles)
        orig_protein_list.append(orig_protein)
        # SMILES → tensor
        compound_new[i] = torch.from_numpy(
            label_smiles(compoundstr, CHARISOSMISET, compound_max)
        )

        # Protein → tensor
        protein_new[i] = torch.from_numpy(
            label_sequence(proteinstr, CHARPROTSET, protein_max)
        )

        labels_new[i] = int(label)

    return IDs, compound_new, protein_new, labels_new, orig_smiles_list, orig_protein_list

#推理用
# def collate_fn(batch_data):
#     N = len(batch_data)
#     compound_max = 100
#     protein_max = 1000
#     compound_new = torch.zeros((N, compound_max), dtype=torch.long)
#     protein_new = torch.zeros((N, protein_max), dtype=torch.long)
#     labels_new = torch.zeros(N, dtype=torch.long)
#     compound_ids = []  # 新增：存储化合物ID

#     for i, pair in enumerate(batch_data):
#         if len(pair) == 3:
#             compound_id, compoundstr, proteinstr = pair
#         else:
#             raise ValueError(f"Unexpected length of data pair: {len(pair)}")

#         # 保存化合物ID
#         compound_ids.append(compound_id)

#         compoundint = torch.from_numpy(label_smiles(
#             compoundstr, CHARISOSMISET, compound_max))
#         compound_new[i] = compoundint
#         proteinint = torch.from_numpy(label_sequence(
#             proteinstr, CHARPROTSET, protein_max))
#         protein_new[i] = proteinint
#         labels_new[i] = 0  # 预测时使用占位符标签

#     # 返回四个值：化合物序列、蛋白质序列、标签、化合物ID列表
#     return (compound_new, protein_new, labels_new, compound_ids)