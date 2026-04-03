import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

import torch
import torch.nn as nn
import json
from tqdm import tqdm
from src.core.configs import Config
from src.core.paths import MIXED_TRAIN_DATA, CIDT_CHECKPOINT
from src.core.models import CIDTRouter
from transformers import AutoModelForCausalLM, AutoTokenizer

def train_cidt():
    device = Config.DEVICE
    epochs = 6
    lr = 1e-4
    entropy_lambda = Config.ROUTER_ENTROPY_LAMBDA
    align_lambda = Config.ROUTER_ALIGN_LAMBDA

    # 🌟 修复 1：加载真正的混合数据集 (确保数量远大于 200)
    data_path = MIXED_TRAIN_DATA
    with open(data_path, "r", encoding="utf-8") as f:
        train_data = json.load(f)

    print("=" * 60)
    print(f"训练数据条数: {len(train_data)}")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_PATH)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        Config.MODEL_PATH,
        torch_dtype=Config.DTYPE,
        device_map="auto"
    )
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    cidt = CIDTRouter().to(device)
    optimizer = torch.optim.AdamW(cidt.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    print("开始训练 CIDT...\n")

    for epoch in range(epochs):
        total_loss = 0
        pbar = tqdm(train_data, desc=f"Epoch {epoch+1}/{epochs}")

        for sample in pbar:
            turns = sample["turns"]
            labels = sample["labels"]

            token_embeds_list = []
            mask_list = []  # 🌟 修复 2：创建一个列表来收集真实的 Attention Mask

            for turn in turns:
                inputs = tokenizer(
                    turn,
                    return_tensors="pt",
                    truncation=True,
                    max_length=Config.MAX_LENGTH,
                    padding="max_length"
                )
                inputs = {k: v.to(model.device) for k, v in inputs.items()}

                with torch.no_grad():
                    out = model(**inputs, output_hidden_states=True)
                    emb = out.hidden_states[15].to(device).to(Config.DTYPE)
                
                token_embeds_list.append(emb)
                # 提取真实的 mask 并转移到 CIDT 所在的设备
                mask_list.append(inputs["attention_mask"].to(device).to(Config.DTYPE))

            token_embeds = torch.stack(token_embeds_list, dim=1)
            # 🌟 修复 3：将收集到的真实 mask 叠合，替换掉之前的 torch.ones
            mask = torch.stack(mask_list, dim=1) 

            target_risk = torch.tensor([labels[-1]], dtype=Config.DTYPE, device=device)

            # 现在的 mask 会告诉 CIDT 忽略 padding token，只关注真实文字！
            route_w, pred_risk_logits, h_next = cidt(token_embeds, mask)

            risk_loss = loss_fn(pred_risk_logits.to(torch.float32), target_risk.unsqueeze(0).to(torch.float32))

            # 归一化路由下用熵正则鼓励“尖锐”选择，避免平均分配
            route_entropy = -(route_w * torch.log(route_w + 1e-8)).sum(dim=2).mean().to(torch.float32)

            # 将路由整体激活强度与风险标签对齐，避免路由头无监督漂移
            route_strength = torch.sigmoid(route_w.mean(dim=(1, 2))).to(torch.float32)
            align_loss = nn.functional.mse_loss(route_strength, target_risk.to(torch.float32))

            loss = risk_loss + entropy_lambda * route_entropy + align_lambda * align_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                risk=f"{risk_loss.item():.4f}",
                entropy=f"{route_entropy.item():.4f}",
                align=f"{align_loss.item():.4f}"
            )

        avg_loss = total_loss / len(train_data)
        print(f"Epoch {epoch+1} 平均损失: {avg_loss:.4f}\n")

    torch.save(cidt.state_dict(), CIDT_CHECKPOINT)
    print("="*60)
    print("🎉 CIDT 修复并训练完成！模型已保存！")
    print("="*60)

if __name__ == "__main__":
    train_cidt()