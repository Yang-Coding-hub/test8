import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

import torch
import pandas as pd
from tqdm import tqdm
from src.core.configs import Config
from src.core.paths import HB_BENIGN, CIDT_CHECKPOINT, MMLU_DIR
from src.core.models import AlphaSteerProjector, NullSpaceProjectedSAE, CIDTRouter, SafeLLMSystem
from transformers import AutoModelForCausalLM, AutoTokenizer
import re

def load_defense_system():
    # Build full defended inference stack for utility evaluation.
    print(">>> 正在加载 Llama-3-8B-Instruct (双卡 auto 模式)...")
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_PATH)
    tokenizer.pad_token, tokenizer.padding_side = tokenizer.eos_token, "left"
    
    base_model = AutoModelForCausalLM.from_pretrained(
        Config.MODEL_PATH, 
        device_map="auto",  
        torch_dtype=Config.DTYPE,
        attn_implementation="eager"
    )
    
    Hb = torch.load(HB_BENIGN, weights_only=True).to(Config.DEVICE).to(Config.DTYPE)
    P_matrix = AlphaSteerProjector().fit_null_space(Hb)
    sae_module = NullSpaceProjectedSAE(P_matrix).to(Config.DEVICE)
    
    cidt_router = CIDTRouter().to(Config.DEVICE)
    cidt_router.load_state_dict(torch.load(CIDT_CHECKPOINT, weights_only=True))
    cidt_router.eval()
    
    class LLMBasedIntentEncoder(torch.nn.Module):
        def __init__(self, base_model):
            super().__init__()
            self.base_model = base_model
        def forward(self, input_ids, attention_mask):
            with torch.no_grad():
                outputs = self.base_model(
                    input_ids=input_ids.to(self.base_model.device),
                    attention_mask=attention_mask.to(self.base_model.device),
                    output_hidden_states=True
                )
                return outputs.hidden_states[15].to(Config.DEVICE)
                
    intent_encoder = LLMBasedIntentEncoder(base_model)
    safe_system = SafeLLMSystem(base_model, tokenizer, intent_encoder, cidt_router, sae_module)
    return safe_system, base_model, tokenizer

def format_mmlu_prompt(row):
    # 🌟 核心修复：用最严厉的指令限制模型的输出，只让它吐出一个字母
    question = row['question']
    choices = row['choices']
    prompt = f"Question: {question}\nA. {choices[0]}\nB. {choices[1]}\nC. {choices[2]}\nD. {choices[3]}\n\nTask: Output ONLY the single letter of the correct answer (A, B, C, or D). DO NOT explain. DO NOT output any other words."
    return prompt

if __name__ == '__main__':
    safe_system, base_model, tokenizer = load_defense_system()
    safe_system.eval()
    
    subjects = [
        "college_computer_science", 
        "high_school_mathematics", 
        "professional_law"
    ]
    
    total_correct = 0
    total_questions = 0
    
    print("\n>>> 开始进行 MMLU 基础能力测试 (强制单字输出模式)...")
    for subject in subjects:
        file_path = str(MMLU_DIR / subject / "test-00000-of-00001.parquet")
        try:
            df = pd.read_parquet(file_path)
            # 取前 20 题进行测试
            df = df.head(20) 
        except Exception as e:
            print(f"读取 {subject} 失败: {e}")
            continue
            
        subject_correct = 0
        print(f"\n评估学科: {subject}")
        
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            prompt = format_mmlu_prompt(row)
            gt_ans = chr(ord('A') + row['answer']) 
            
            chat_str = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], 
                tokenize=False, add_generation_prompt=True
            )
            
            inputs = tokenizer(chat_str, return_tensors="pt")
            session_ids = inputs["input_ids"].unsqueeze(0) 
            session_masks = inputs["attention_mask"].unsqueeze(0)
            lengths = torch.tensor([1])
            
            # 使用系统的标准推理
            resps = safe_system.multi_turn_generate(
                session_ids.to(Config.DEVICE), 
                session_masks.to(Config.DEVICE), 
                lengths
            )
            
            pred_text = resps[0].strip().upper()
            
            # 🌟 修复判卷逻辑：利用正则提取第一个出现的 A/B/C/D
            match = re.search(r'\b([A-D])\b', pred_text.replace('.', ' '))
            if match:
                extracted_ans = match.group(1)
            else:
                # 兜底：如果模型还是不听话，强制看前两个字符
                extracted_ans = pred_text[:2]
                
            if gt_ans in extracted_ans:
                subject_correct += 1
                
        total_correct += subject_correct
        total_questions += len(df)
        print(f"{subject} 准确率: {subject_correct / len(df) * 100:.2f}%")
        
    print("=" * 60)
    print("🎯 MMLU 基础能力评测结果 (开启防御后):")
    print(f"总体准确率: {total_correct / total_questions * 100:.2f}% ({total_correct}/{total_questions})")
    print("=" * 60)