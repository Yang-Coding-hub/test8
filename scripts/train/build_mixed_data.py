import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

import json
import pandas as pd
import random
import os
from src.core.paths import FITD_TRAIN_DATA, MMLU_DIR, MIXED_TRAIN_DATA

def build_mixed_dataset():
    # Build balanced CIDT training data by mixing harmful FITD and benign MMLU samples.
    print(">>> 正在加载恶意越狱数据 (FITD)...")
    with open(FITD_TRAIN_DATA, 'r', encoding='utf-8') as f:
        fitd_data = json.load(f)
        
    print(f"已加载 {len(fitd_data)} 条恶意样本 (Label 1)")

    print(">>> 正在从 MMLU 提取高难度良性数据...")
    mmlu_dirs = ["college_computer_science", "high_school_mathematics", "professional_law"]
    benign_data = []

    for subj in mmlu_dirs:
        # 我们用验证集 (dev) 来做训练，避免污染 test 评测集
        path = str(MMLU_DIR / subj / "dev-00000-of-00001.parquet")
        if os.path.exists(path):
            df = pd.read_parquet(path)
            # 为了数据均衡，每个学科抽取一部分题目作为良性对话
            for _, row in df.iterrows():
                q = f"Question: {row['question']}\nA. {row['choices'][0]}\nB. {row['choices'][1]}\nC. {row['choices'][2]}\nD. {row['choices'][3]}"
                
                # 构建格式与 FITD 相同的良性数据，标签全是 0
                benign_data.append({
                    "turns": [q],
                    "labels": [0.0]  # 绝对安全
                })
                
    print(f"已加载 {len(benign_data)} 条高难度良性样本 (Label 0)")

    # 混合并打乱数据
    mixed_data = fitd_data + benign_data
    random.shuffle(mixed_data)

    save_path = MIXED_TRAIN_DATA
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(mixed_data, f, indent=4)

    print("=" * 60)
    print(f"✅ 混合数据集构建完成！总样本数: {len(mixed_data)}")
    print(f"文件已保存至: {save_path}")
    print("=" * 60)

if __name__ == '__main__':
    build_mixed_dataset()