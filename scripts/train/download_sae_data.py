import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

import os
import json
import pandas as pd
import urllib.request
from datasets import load_dataset
from src.core.paths import DATA_DIR, SAE_MALICIOUS_PROMPTS_JSON, SAE_BENIGN_PROMPTS_JSON

def setup_data_dir():
    # 🌟 核心：确保你指定的目标文件夹存在
    base_dir = str(DATA_DIR)
    os.makedirs(base_dir, exist_ok=True)
    return base_dir

def download_and_process_data(base_dir):
    print(f"📁 目标文件夹已确认: {base_dir}")

    # ==========================================
    # 1. 下载 AdvBench (恶意越狱数据)
    # ==========================================
    print("\n>>> 正在下载 AdvBench 恶意意图数据集...")
    advbench_url = "https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/data/advbench/harmful_behaviors.csv"
    csv_path = os.path.join(base_dir, "harmful_behaviors.csv")
    
    try:
        urllib.request.urlretrieve(advbench_url, csv_path)
        df_adv = pd.read_csv(csv_path)
        # 提取 'goal' 列，也就是具体的恶意提问
        malicious_prompts = df_adv['goal'].tolist()
        print(f"✅ 成功提取 {len(malicious_prompts)} 条恶意 Prompt。")
        
        # 保存为 JSON
        mal_json_path = str(SAE_MALICIOUS_PROMPTS_JSON)
        with open(mal_json_path, "w", encoding="utf-8") as f:
            json.dump(malicious_prompts, f, indent=4)
        print(f"💾 恶意数据已保存至: {mal_json_path}")
        
    except Exception as e:
        print(f"❌ 下载 AdvBench 失败: {e}")
        malicious_prompts = []

    # ==========================================
    # 2. 下载 Alpaca (日常良性对话数据)
    # ==========================================
    print("\n>>> 正在下载 Alpaca 良性指令数据集...")
    try:
        # 从 HuggingFace 加载 alpaca 训练集
        dataset = load_dataset("tatsu-lab/alpaca", split="train")
        
        # 为了让正负样本平衡，我们截取与恶意数据相同数量的良性数据 (约 520 条)
        target_length = len(malicious_prompts) if len(malicious_prompts) > 0 else 520
        benign_prompts = [ex["instruction"] for ex in dataset][:target_length]
        print(f"✅ 成功提取 {len(benign_prompts)} 条良性 Prompt。")
        
        # 保存为 JSON
        ben_json_path = str(SAE_BENIGN_PROMPTS_JSON)
        with open(ben_json_path, "w", encoding="utf-8") as f:
            json.dump(benign_prompts, f, indent=4)
        print(f"💾 良性数据已保存至: {ben_json_path}")
        
    except Exception as e:
        print(f"❌ 下载 Alpaca 失败: {e}")

    print("\n" + "="*60)
    print("🎉 所有 SAE 提纯所需数据已成功下载并格式化！")
    print("="*60)

if __name__ == "__main__":
    base_dir = setup_data_dir()
    download_and_process_data(base_dir)