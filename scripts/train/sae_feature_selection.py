import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

import argparse
import torch
import json
import pandas as pd
import os
from src.core.configs import Config
from transformers import AutoModelForCausalLM, AutoTokenizer
from safetensors.torch import load_file
from tqdm import tqdm
from src.core.paths import SAE_MALICIOUS_PROMPTS_JSON, SAE_BENIGN_PROMPTS_JSON, MMLU_DIR, VERIFIED_REFUSAL_FEATURES_PT

def load_prompts():
    # Prepare harmful vs benign prompt sets for SAE feature discrimination.
    print(">>> 正在加载数据集...")
    # 1. 加载恶意数据 (AdvBench)
    malicious_path = SAE_MALICIOUS_PROMPTS_JSON
    with open(malicious_path, "r", encoding="utf-8") as f:
        refusal_prompts = json.load(f)
        
    # 2. 加载良性数据 1：日常对话 (Alpaca)
    benign_path = SAE_BENIGN_PROMPTS_JSON
    with open(benign_path, "r", encoding="utf-8") as f:
        benign_prompts = json.load(f)
        
    # 3. 加载良性数据 2：硬核学术数据 (MMLU)
    print(">>> 正在将 MMLU 复杂数据合并入良性对照组...")
    hardcore_subjects = [
        "high_school_mathematics", 
        "college_computer_science", 
        "high_school_physics",
        "professional_law"
    ]
    
    mmlu_prompts = []
    for subj in hardcore_subjects:
        # 🌟 核心修复：使用 test 数据集路径
        path = str(MMLU_DIR / subj / "test-00000-of-00001.parquet")
        if os.path.exists(path):
            try:
                df = pd.read_parquet(path)
                for _, row in df.iterrows():
                    q = f"Question: {row['question']}\nA. {row['choices'][0]}\nB. {row['choices'][1]}\nC. {row['choices'][2]}\nD. {row['choices'][3]}"
                    mmlu_prompts.append(q)
                print(f"  ✅ 成功读取 MMLU 学科: {subj}")
            except Exception as e:
                print(f"  ❌ 读取 {subj} 失败: {e}")
        else:
            print(f"  ⚠️ 找不到文件: {path}")
                
    # 取与恶意数据等量的 MMLU 数据加入良性组
    target_len = len(refusal_prompts) if len(refusal_prompts) > 0 else 520
    benign_prompts.extend(mmlu_prompts[:target_len])
    
    return refusal_prompts, benign_prompts

def _extract_feature_matrix(prompts, tokenizer, model, encoder_weight, encoder_bias):
    layer_15_activations = []

    def capture_activation_hook(module, input, output):
        hidden_states = output[0] if isinstance(output, tuple) else output
        layer_15_activations.append(hidden_states[:, -1, :].detach().to(Config.DEVICE).to(torch.float32))

    target_layer = model.model.layers[15]
    hook_handle = target_layer.register_forward_hook(capture_activation_hook)

    feats = []
    for prompt in tqdm(prompts):
        layer_15_activations.clear()
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            _ = model(**inputs)

        if len(layer_15_activations) > 0:
            act = layer_15_activations[0]
            feature_act = torch.relu(torch.matmul(act, encoder_weight.T) + encoder_bias)
            feats.append(feature_act.squeeze(0).detach().cpu())

    hook_handle.remove()
    return torch.stack(feats, dim=0) if feats else torch.empty(0, encoder_weight.shape[0])


def _stable_topk_indices(refusal_matrix, benign_matrix, top_k=64, iters=7, sample_frac=0.8, seed=42):
    if refusal_matrix.size(0) == 0 or benign_matrix.size(0) == 0:
        return [], torch.zeros(refusal_matrix.size(1) if refusal_matrix.ndim == 2 else 0), {}

    g = torch.Generator(device="cpu")
    g.manual_seed(seed)

    n_r = refusal_matrix.size(0)
    n_b = benign_matrix.size(0)
    k_r = max(1, int(n_r * sample_frac))
    k_b = max(1, int(n_b * sample_frac))

    feat_dim = refusal_matrix.size(1)
    freq = torch.zeros(feat_dim, dtype=torch.float32)
    score_sum = torch.zeros(feat_dim, dtype=torch.float32)
    per_iter_sets = []

    for _ in range(iters):
        idx_r = torch.randint(low=0, high=n_r, size=(k_r,), generator=g)
        idx_b = torch.randint(low=0, high=n_b, size=(k_b,), generator=g)

        score = refusal_matrix[idx_r].mean(dim=0) - benign_matrix[idx_b].mean(dim=0)
        top = torch.topk(score, k=top_k).indices
        per_iter_sets.append(set(top.tolist()))
        freq[top] += 1.0
        score_sum += score

    mean_score = score_sum / float(iters)
    stable = sorted(
        range(feat_dim),
        key=lambda i: (freq[i].item(), mean_score[i].item()),
        reverse=True,
    )[:top_k]

    overlaps = []
    for i in range(len(per_iter_sets)):
        for j in range(i + 1, len(per_iter_sets)):
            inter = len(per_iter_sets[i] & per_iter_sets[j])
            union = len(per_iter_sets[i] | per_iter_sets[j])
            overlaps.append(inter / max(1, union))

    stats = {
        "bootstrap_iters": iters,
        "sample_frac": sample_frac,
        "avg_jaccard": float(sum(overlaps) / max(1, len(overlaps))),
    }
    return stable, mean_score, stats


def select_refusal_features(top_k=64, bootstrap_iters=7, sample_frac=0.8, seed=42):
    print("=== 开始加载模型与SAE权重 (双卡 auto 模式) ===")
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_PATH)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        Config.MODEL_PATH,
        device_map="auto",
        torch_dtype=Config.DTYPE,
        output_hidden_states=True
    )
    model.eval()
    
    sae_weights = load_file(Config.SAE_PATH)
    encoder_weight = sae_weights["encoder.weight"].to(Config.DEVICE).to(torch.float32)
    encoder_bias = sae_weights["encoder.bias"].to(Config.DEVICE).to(torch.float32)
    print(f"SAE编码器加载完成，特征总数：{encoder_weight.shape[0]}")
    
    refusal_prompts, benign_prompts = load_prompts()
    print(f"✅ 将使用 {len(refusal_prompts)} 条恶意数据 vs {len(benign_prompts)} 条良性混合数据 进行特征对抗筛选。")
    
    print("\n=== 正在提取大规模恶意场景的 SAE 特征激活 ===")
    refusal_matrix = _extract_feature_matrix(
        refusal_prompts, tokenizer, model, encoder_weight, encoder_bias
    )

    print("\n=== 正在提取大规模良性混合场景的 SAE 特征激活 ===")
    benign_matrix = _extract_feature_matrix(
        benign_prompts, tokenizer, model, encoder_weight, encoder_bias
    )

    print("\n=== 计算特异性分数并筛选稳定 Top-K 核心拒绝特征 ===")
    top_indices, refusal_score, stable_stats = _stable_topk_indices(
        refusal_matrix,
        benign_matrix,
        top_k=top_k,
        iters=bootstrap_iters,
        sample_frac=sample_frac,
        seed=seed,
    )
    
    print("=" * 60)
    print(f"✅ 语义级特征提纯完成！Top {top_k} 拒绝特征索引：")
    print(f"VERIFIED_REFUSAL_INDICES = {top_indices}")
    print("=" * 60)
    
    torch.save({
        "top_indices": top_indices,
        "refusal_score": refusal_score,
        "stable_stats": stable_stats,
        "malicious_count": int(refusal_matrix.size(0)),
        "benign_count": int(benign_matrix.size(0)),
    }, VERIFIED_REFUSAL_FEATURES_PT)
    print(f"特征权重已保存到 {VERIFIED_REFUSAL_FEATURES_PT}")
    print(f"稳定性统计: {stable_stats}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stable SAE refusal feature selection")
    parser.add_argument("--top-k", type=int, default=64)
    parser.add_argument("--bootstrap-iters", type=int, default=7)
    parser.add_argument("--sample-frac", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    select_refusal_features(
        top_k=args.top_k,
        bootstrap_iters=args.bootstrap_iters,
        sample_frac=args.sample_frac,
        seed=args.seed,
    )