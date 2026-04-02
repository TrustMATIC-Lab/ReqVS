import pandas as pd
import sys

# 默认路径，如果在云端运行，需要用户确认
file_path = "/root/autodl-tmp/DUD-E.csv"

# 如果命令行传入了路径，则使用命令行路径
if len(sys.argv) > 1:
    file_path = sys.argv[1]

try:
    print(f"Reading file from: {file_path}")
    # 尝试读取前几行
    df = pd.read_csv(file_path, nrows=5)
    print("\n=== Columns ===")
    print(df.columns.tolist())
    
    print("\n=== First 5 rows ===")
    print(df)
    
    print("\n=== Data Info ===")
    print(df.info())
    
except FileNotFoundError:
    print(f"Error: File not found at {file_path}")
    print("Please make sure the file exists and the path is correct.")
except Exception as e:
    print(f"Error reading file: {e}")
