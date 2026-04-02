import os
import torch
import numpy as np
import pickle
from tqdm import tqdm
from transformers import BertModel, BertTokenizer
import argparse
import re

# 设置使用的 GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def get_prottrans_embedding(sequence, tokenizer, model, device):
    """
    使用 Transformers 提取特征 (替代 bio_embeddings)
    """
    # 预处理序列：添加空格分隔 (ProtTrans 需要: "M K ...")
    sequence = " ".join(list(re.sub(r"[UZOB]", "X", sequence)))
    
    # Tokenize
    # max_length=1024 并不是硬限制，ProtTrans 可以处理更长，但为了显存安全建议截断或分段
    # 这里我们暂不截断，让显卡尽力跑
    encoded_input = tokenizer(sequence, return_tensors='pt').to(device)
    
    # 推理
    with torch.no_grad():
        output = model(**encoded_input)
    
    # 获取最后一层隐藏状态 [1, seq_len+2, 1024]
    # output.last_hidden_state shape: (batch, seq_len+2, hidden_size)
    embedding = output.last_hidden_state.detach().cpu().numpy()[0] 
    
    # 去掉 [CLS] (index 0) 和 [SEP] (index -1)
    seq_embedding = embedding[1:-1]
    
    # 计算全局特征 (Mean Pooling)
    global_embedding = np.mean(seq_embedding, axis=0)
    
    return global_embedding, seq_embedding

if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    # 允许任意数据集名称
    parse.add_argument('--dataset', type=str, default="BindingDB_AIBind",
                       help='the scenario of experiment setting')
    opt = parse.parse_args()
    Dataset = opt.dataset
    
    print(f"=== 开始处理数据集: {Dataset} ===")
    
    # 路径配置
    base_dir = os.path.dirname(os.path.abspath(__file__))
    # 本地模型文件夹路径
    local_model_path = os.path.join(base_dir, "prot_bert_bfd")
    
    # 数据集路径
    # 假设 Dataset 目录在 ../../Datasets/{Dataset}/feature/
    feature_dir = os.path.abspath(os.path.join(base_dir, "../../Datasets", Dataset, "feature"))
    protein_list_path = os.path.join(feature_dir, "protein_list.txt")
    output_path = os.path.join(feature_dir, "aas_ProtTransBertBFD1024.pkl")

    if not os.path.exists(protein_list_path):
        print(f"[Error] 找不到输入文件: {protein_list_path}")
        exit(1)
        
    if not os.path.exists(local_model_path):
        print(f"[Error] 找不到本地模型文件夹: {local_model_path}")
        print("请确保你已上传 pytorch_model.bin 等文件到该目录")
        exit(1)

    # 加载模型
    print(f"正在从本地加载模型: {local_model_path} ...")
    try:
        tokenizer = BertTokenizer.from_pretrained(local_model_path, do_lower_case=False)
        model = BertModel.from_pretrained(local_model_path)
    except Exception as e:
        print(f"[Fatal] 模型加载失败: {e}")
        print("请检查 pytorch_model.bin 是否完整 (大小应约 1.56GB)")
        exit(1)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    print(f"模型加载成功！使用设备: {device}")

    protein_embed_dict = {}
    
    # 读取数据
    with open(protein_list_path, "r") as file:
        lines = file.readlines()
        
    print(f"待处理序列总数: {len(lines)}")
    
    # 批处理循环
    error_count = 0
    for line in tqdm(lines, total=len(lines), desc="Extracting Features"):
        parts = line.strip().split()
        if len(parts) < 2:
            continue
        
        pid = parts[0]
        aas = parts[1]
        
        try:
            vector, matrix = get_prottrans_embedding(aas, tokenizer, model, device)
            protein_embed_dict[pid] = [vector, matrix]
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"\n[Warning] 显存不足，跳过蛋白 {pid} (长度: {len(aas)})")
                torch.cuda.empty_cache()
            else:
                print(f"\n[Error] 处理 {pid} 时发生 Runtime 错误: {e}")
            error_count += 1
        except Exception as e:
            print(f"\n[Error] 处理 {pid} 未知错误: {e}")
            error_count += 1

    # 保存结果
    print(f"保存结果到: {output_path}")
    with open(output_path, 'wb') as f:
        pickle.dump(protein_embed_dict, f)
    
    print(f"=== 完成！(成功: {len(protein_embed_dict)}, 失败: {error_count}) ===")
