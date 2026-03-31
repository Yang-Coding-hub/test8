import torch
import os
from configs import Config
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm

def extract_activations():
    print("=== 开始加载模型与分词器 ===")
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_PATH)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    model = AutoModelForCausalLM.from_pretrained(
        Config.MODEL_PATH,
        device_map="auto",
        torch_dtype=Config.DTYPE
    )
    model.eval()
    print(f"模型加载完成")
    
    # 存储激活值的列表（现在每个元素是 [Seq, Dim]，消除Batch维度）
    all_activations = []
    
    # [修复] 改进的Hook函数：处理变长序列，直接收集Token级激活
    def activation_capture_hook(module, input, output):
        hidden_states = output[0] if isinstance(output, tuple) else output
        # hidden_states shape: [1, Seq, Dim] (因为batch_size=1)
        # 直接 squeeze 掉 batch 维度，变成 [Seq, Dim]，加入列表
        all_activations.append(hidden_states.squeeze(0).detach().cpu().to(torch.float32))
    
    target_layer_idx = 15
    if hasattr(model, "model"):
        target_layer = model.model.layers[target_layer_idx]
    else:
        target_layer = model.layers[target_layer_idx]
    
    hook_handle = target_layer.register_forward_hook(activation_capture_hook)
    print(f"已注册Hook到第{target_layer_idx}层")
    
    local_data_path = "/home/asus/ycy/test8/data/mmlu"
    print(f"=== 加载本地MMLU数据集 ===")
    
    dataset = load_dataset(
        local_data_path,
        name="all",
        split="validation",
        cache_dir="/home/asus/ycy/test8/data"
    )
    
    print("=== 开始提取激活值 ===")
    max_samples = 300
    count = 0
    
    for sample in tqdm(dataset, total=max_samples):
        if count >= max_samples:
            break
        
        question = sample["question"]
        choices = sample["choices"]
        prompt = f"Question: {question}\nChoices: {choices}\nAnswer:"
        
        # [修复] 不强制padding到max_length，让模型自然处理
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256).to(Config.DEVICE)
        
        with torch.no_grad():
            _ = model(**inputs)
        
        count += 1
    
    hook_handle.remove()
    print("激活提取完成")
    
    print("=== 处理生成H_b矩阵 ===")
    # [修复] 先把所有激活在Seq维度拼接，变成 [Total_tokens, Dim]
    all_acts = torch.cat(all_activations, dim=0)
    
    # 随机采样 2000 个 token 向量用于 SVD（足够计算稳定的零空间）
    num_samples_for_svd = min(2000, all_acts.shape[0])
    indices = torch.randperm(all_acts.shape[0])[:num_samples_for_svd]
    
    # H_b shape: [Dim, N_Samples]，符合零空间计算要求
    H_b = all_acts[indices].T
    
    save_path = "/home/asus/ycy/test8/Hb_benign_layer15.pt"
    torch.save(H_b, save_path)
    print(f"已保存到: {save_path}, Shape: {tuple(H_b.shape)}")

if __name__ == "__main__":
    extract_activations()