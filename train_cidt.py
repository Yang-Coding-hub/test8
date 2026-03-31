import torch
import torch.nn as nn
import json
from tqdm import tqdm
from configs import Config
from models import CIDTRouter
from transformers import AutoModelForCausalLM, AutoTokenizer

def train_cidt():
    device = Config.DEVICE
    epochs = 6
    lr = 1e-4

    # 加载真实FITD数据
    with open("/home/asus/ycy/test8/fitd_train_data.json", "r", encoding="utf-8") as f:
        train_data = json.load(f)

    print("=" * 60)
    print(f"训练数据条数: {len(train_data)}")
    print("=" * 60)

    # 加载LLM
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_PATH)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        Config.MODEL_PATH,
        dtype=Config.DTYPE,
        device_map="cuda:0"
    )
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    # ✅ 使用你自己的 CIDTRouter
    cidt = CIDTRouter().to(device)
    optimizer = torch.optim.AdamW(cidt.parameters(), lr=lr)
    loss_fn = nn.BCELoss()

    print("开始训练 CIDT...\n")

    for epoch in range(epochs):
        total_loss = 0
        pbar = tqdm(train_data, desc=f"Epoch {epoch+1}/{epochs}")

        for sample in pbar:
            turns = sample["turns"]
            labels = sample["labels"]
            num_turns = len(turns)

            # 存储每一轮的 token_embeds (shape: [1, seq_len, 4096])
            token_embeds_list = []
            for turn in turns:
                inputs = tokenizer(
                    turn,
                    return_tensors="pt",
                    truncation=True,
                    max_length=Config.MAX_LENGTH,
                    padding="max_length"
                ).to(device)

                with torch.no_grad():
                    out = model(**inputs, output_hidden_states=True)
                    # 取第15层隐层
                    emb = out.hidden_states[15].to(Config.DTYPE)
                token_embeds_list.append(emb)

            # shape: [1, num_turns, seq_len, 4096]
            token_embeds = torch.stack(token_embeds_list, dim=1)
            B, T, S, D = token_embeds.shape

            # mask shape: [1, num_turns, seq_len]
            mask = torch.ones(1, T, S, device=device, dtype=Config.DTYPE)

            # label
            target_risk = torch.tensor(
                [labels[-1]],  # 最后一轮是风险
                dtype=Config.DTYPE,
                device=device
            )

            # ✅ 完全匹配你的 forward 接口
            route_w, pred_risk, h_next = cidt(token_embeds, mask)
            loss = loss_fn(pred_risk, target_risk.unsqueeze(0))

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = total_loss / len(train_data)
        print(f"Epoch {epoch+1} 平均损失: {avg_loss:.4f}\n")

    torch.save(cidt.state_dict(), "/home/asus/ycy/test8/cidt_final.pt")
    print("="*60)
    print("🎉 CIDT 训练完成！模型已保存！")
    print("路径: /home/asus/ycy/test8/cidt_final.pt")
    print("="*60)

if __name__ == "__main__":
    train_cidt()