import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

import json
import re
import os
from src.core.paths import FITD_TRAIN_DATA

def extract_from_real_logs():
    # Parse external FITD logs into local multi-turn training json.
    log_root = os.getenv(
        "FITD_LOG_ROOT",
        "/home/asus/ycy/Foot-in-the-door-Jailbreak/logs/Llama-3.1-8B-Instruct",
    )
    LOG_PATHS = [
        os.path.join(log_root, "0", "0.log"),
        os.path.join(log_root, "0", "1.log"),
        os.path.join(log_root, "1", "0.log"),
    ]

    train_data = []

    print("="*60)
    print("🚀 从真实 FITD 攻击日志提取多轮训练数据（终极版）")
    print("="*60)

    for log_path in LOG_PATHS:
        try:
            with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()
        except:
            continue

        current_prompts = []
        for line in lines:
            # 匹配用户输入的 prompt
            prompt_match = re.search(r'Processing index \d+ with prompt: (.*)', line)
            if prompt_match:
                prompt = prompt_match.group(1).strip()
                current_prompts.append(prompt)

            # 遇到回复表示一轮结束
            if "**Response:**" in line and len(current_prompts) >= 2:
                # 构造多轮：前面是良性/引导，最后一轮是有害
                turns = current_prompts
                labels = [0] * (len(turns)-1) + [1]

                train_data.append({
                    "turns": turns,
                    "labels": labels
                })
                current_prompts = []

        print(f"✅ 从 {log_path} 提取完成")

    # 去重
    unique_data = []
    seen = set()
    for d in train_data:
        key = "|||".join(d["turns"])
        if key not in seen:
            seen.add(key)
            unique_data.append(d)

    # 保存
    save_path = FITD_TRAIN_DATA
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(unique_data, f, ensure_ascii=False, indent=2)

    print("\n" + "="*60)
    print(f"🔥 最终提取完成！")
    print(f"📊 训练数据总数：{len(unique_data)} 条")
    print(f"📁 已保存到：{save_path}")
    print("="*60)
    print("✅ 现在可以运行：python train_cidt.py")
    print("="*60)

if __name__ == "__main__":
    extract_from_real_logs()