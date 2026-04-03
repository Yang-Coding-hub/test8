import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

import torch
import json
import pandas as pd
import os
from tqdm import tqdm
from src.core.configs import Config
from src.core.paths import SAE_BENIGN_PROMPTS_JSON, MMLU_DIR, HB_BENIGN
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_diverse_benign_prompts(num_samples=2000):
    # Collect diverse benign prompts to estimate benign activation subspace Hb.
    prompts = []
    
    # 1. 加载日常对话良性数据 (Alpaca)
    print(">>> 正在加载 Alpaca 日常对话数据...")
    alpaca_path = SAE_BENIGN_PROMPTS_JSON
    if os.path.exists(alpaca_path):
        with open(alpaca_path, "r", encoding="utf-8") as f:
            alpaca_data = json.load(f)
            # 提取一半数量用于构建 Hb
            prompts.extend(alpaca_data[:num_samples // 2])
            
    # 2. 加载学术/逻辑良性数据 (MMLU)
    print(">>> 正在加载 MMLU 复杂逻辑与数学数据...")
    hardcore_subjects = [
        "high_school_mathematics", 
        "college_computer_science", 
        "high_school_physics",
        "professional_law",
        "college_medicine"
    ]
    
    mmlu_prompts = []
    for subj in hardcore_subjects:
        path = str(MMLU_DIR / subj / "test-00000-of-00001.parquet")
        if os.path.exists(path):
            df = pd.read_parquet(path)
            for _, row in df.iterrows():
                q = f"Question: {row['question']}\nA. {row['choices'][0]}\nB. {row['choices'][1]}\nC. {row['choices'][2]}\nD. {row['choices'][3]}"
                mmlu_prompts.append(q)
                
    # 提取另一半用于构建 Hb
    prompts.extend(mmlu_prompts[:num_samples // 2])
    
    print(f"✅ 成功融合良性数据集，共计 {len(prompts)} 条 (日常对话 + 硬核学术)。")
    return prompts

def extract_and_save_Hb():
    print("\n=== 开始加载模型以提取良性激活 (双卡 auto 模式) ===")
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_PATH)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        Config.MODEL_PATH,
        device_map="auto",
        torch_dtype=Config.DTYPE,
        output_hidden_states=True
    )
    model.eval()

    prompts = load_diverse_benign_prompts(num_samples=2000)
    
    # 钩子函数：捕获第 15 层最后一个 Token 的激活
    layer_15_activations = []
    def capture_activation_hook(module, input, output):
        hidden_states = output[0] if isinstance(output, tuple) else output
        # 将激活拉回 cuda:0 以备后续拼接
        layer_15_activations.append(hidden_states[:, -1, :].detach().to(Config.DEVICE).to(torch.float32))

    target_layer = model.model.layers[15]
    hook_handle = target_layer.register_forward_hook(capture_activation_hook)

    print("\n=== 正在进行前向传播，收集良性空间特征 ===")
    collected_tensors = []
    
    for prompt in tqdm(prompts):
        layer_15_activations.clear()
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            _ = model(**inputs)
            
        # 收集形状为 [1, 4096] 的激活向量
        if len(layer_15_activations) > 0:
            collected_tensors.append(layer_15_activations[0].squeeze(0))
            
    hook_handle.remove()

    # ==========================================
    # 核心数学计算：构建 Hb 矩阵 [hidden_dim, num_samples]
    # ==========================================
    print("\n=== 正在组合 Hb 矩阵 ===")
    # 形状转换为 [4096, N]
    H_b_matrix = torch.stack(collected_tensors, dim=1) 
    
    print(f"✅ Hb 矩阵构建完成，形状: {tuple(H_b_matrix.shape)}")
    
    # 覆盖保存你之前的旧 Hb 矩阵
    save_path = HB_BENIGN
    torch.save(H_b_matrix, save_path)
    print(f"💾 全新完备的良性激活矩阵已保存至: {save_path}")

if __name__ == "__main__":
    extract_and_save_Hb()